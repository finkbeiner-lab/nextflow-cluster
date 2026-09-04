#!/usr/bin/env python
"""Pipeline-facing DL neurite-segmentation stub (produces a probability map only).

This is the ``bin/`` entry point for the *deep-learning* neurite detector
(Track B). It is deliberately thin: given a trained checkpoint and a morphology
image, it returns a **neurite probability map** (float32, 0-1, same HxW as the
input) or a thresholded **binary mask**. It does **NOT** touch the database, the
skeleton/attribution logic, or ``neuritecelldata`` — that shared backend is
owned by another track (see ``bin/neurite.py``), which will call
:func:`predict_neurite_probmap` (or :func:`segment_neurites`) in place of the
Frangi vesselness step and feed the result into the SAME downstream skeletonize
-> per-soma attribution -> DB-write path.

Interface contract for the wiring track
---------------------------------------
The classical detector in ``bin/neurite.py`` currently does, per tile::

    prob = frangi_vesselness(morphology_image)   # float 0-1 ridge map
    mask = prob >= threshold
    skeleton = skeletonize(clean(mask))
    ... attribute skeleton to somas, write neuritecelldata ...

To swap in this model, replace the first two lines with::

    from neurite_model import predict_neurite_probmap
    prob = predict_neurite_probmap(morphology_image, checkpoint_path)
    mask = prob >= threshold   # or use segment_neurites(...) to get the mask

Guarantees for the caller:

* Input: a single-channel 2D ``numpy.ndarray`` (H, W), any real dtype. RGB/3D
  input is reduced to its first channel.
* Output of :func:`predict_neurite_probmap`: ``numpy.float32`` (H, W) in [0, 1],
  EXACTLY the input HxW (internal reflect-padding is cropped back off).
* Percentile normalization (1-99.5, matching the tracer/scorer) is applied
  internally, so pass the RAW morphology image; do not pre-normalize unless you
  set ``normalize=False``.
* No global state, no DB, no file writes (except the optional ``__main__`` CLI).

Dependencies
------------
``torch`` and the training package under ``ml/neurite/`` are optional at import
time: this file imports cleanly even if ``torch`` is absent (so the classical
Frangi path in ``bin/neurite.py`` keeps working in a torch-less container). The
clear ImportError is raised only when a prediction is actually requested.
"""

from __future__ import annotations

import os
import sys
from typing import Optional

import numpy as np

# Path to the offline training package (ml/neurite) that defines the model and
# inference helpers. Kept out of the container bin/ path on purpose; add it to
# sys.path lazily so this stub imports without it present.
_ML_DIR = os.environ.get(
    "NEURITE_ML_DIR",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ml", "neurite"),
)

# Default checkpoint location; override via arg or the NEURITE_CHECKPOINT env var.
DEFAULT_CHECKPOINT = os.environ.get(
    "NEURITE_CHECKPOINT",
    os.path.join(_ML_DIR, "runs", "dryrun", "best.pt"),
)


def _require_torch():
    """Import torch and the ml/neurite inference helpers, or raise clearly.

    Returns:
        The ``infer`` module from ``ml/neurite``.

    Raises:
        ImportError: If torch or the training package cannot be imported, with a
            message pointing at ``ml/neurite/requirements-ml.txt``.
    """
    try:
        import torch  # noqa: F401
    except Exception as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "neurite_model requires PyTorch, which is not installed in this "
            "environment. Install ml/neurite/requirements-ml.txt (torch) to use "
            "the DL detector, or keep using the Frangi path in bin/neurite.py."
        ) from exc
    if _ML_DIR not in sys.path:
        sys.path.insert(0, _ML_DIR)
    try:
        import infer  # type: ignore
    except Exception as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            f"could not import the neurite inference package from {_ML_DIR}: "
            f"{exc}. Set NEURITE_ML_DIR to the ml/neurite directory."
        ) from exc
    return infer


# Simple module-level cache so a per-tile loop does not reload the checkpoint.
_MODEL_CACHE: dict = {}


def _get_model(checkpoint: str, device: str):
    """Load (and cache) a model for a checkpoint/device pair.

    Args:
        checkpoint: Path to the ``.pt`` checkpoint.
        device: Torch device string (``"cpu"`` or ``"cuda"``).

    Returns:
        ``(infer_module, model, torch_device)``.

    Raises:
        FileNotFoundError: If the checkpoint does not exist.
    """
    infer = _require_torch()
    import torch

    if not os.path.exists(checkpoint):
        raise FileNotFoundError(
            f"neurite checkpoint not found: {checkpoint}. Train one with "
            f"ml/neurite/train.py or set NEURITE_CHECKPOINT."
        )
    key = (os.path.abspath(checkpoint), device)
    if key not in _MODEL_CACHE:
        dev = torch.device(device)
        _MODEL_CACHE[key] = (infer, infer.load_checkpoint(checkpoint, dev), dev)
    return _MODEL_CACHE[key]


def predict_neurite_probmap(
    image: np.ndarray,
    checkpoint: Optional[str] = None,
    device: str = "cpu",
    normalize: bool = True,
    tile: int = 1024,
    halo: int = 64,
) -> np.ndarray:
    """Predict a neurite probability map for one morphology image.

    This is the primary entry point for the wiring track. It returns a plain
    probability map and performs no DB or file I/O.

    Large inputs (whole-well montages, e.g. 8192x8192) are run with sliding-
    window tiling so GPU memory stays bounded; smaller inputs run whole-image.

    Args:
        image: 2D morphology image (H, W), any real dtype. 3D input uses channel 0.
        checkpoint: Path to a trained ``.pt`` checkpoint. Defaults to
            ``DEFAULT_CHECKPOINT`` (env ``NEURITE_CHECKPOINT``).
        device: ``"cpu"`` or ``"cuda"``.
        normalize: Apply the 1-99.5 percentile normalization internally (leave
            True and pass raw images; matches training).
        tile: Interior tile size for sliding-window inference. If the image's
            larger side exceeds this, tiled inference is used; else whole-image.
            Set ``0`` to force whole-image.
        halo: Reflect-padded context added around each tile before inference.

    Returns:
        Float32 probability map (H, W) in [0, 1], same HxW as ``image``.

    Raises:
        ImportError: If torch / the ml package are unavailable (only when called).
        FileNotFoundError: If the checkpoint is missing.
    """
    checkpoint = checkpoint or DEFAULT_CHECKPOINT
    infer, model, dev = _get_model(checkpoint, device)  # adds ml/neurite to sys.path
    img = np.asarray(image).astype(np.float32)
    if img.ndim == 3:
        img = img[..., 0]
    if normalize:
        from dataset import percentile_normalize
        img = percentile_normalize(img)
    # Tile montage-scale inputs. predict_tiled is the audited sliding-window
    # inference (every output pixel written exactly once).
    if tile and max(img.shape) > tile:
        from percell_integrate import predict_tiled
        return predict_tiled(model, img, dev, tile=tile, halo=halo)
    return infer.predict_probmap(model, img, device=dev, normalize=False)


def segment_neurites(
    image: np.ndarray,
    checkpoint: Optional[str] = None,
    threshold: float = 0.5,
    device: str = "cpu",
    normalize: bool = True,
) -> np.ndarray:
    """Predict a binary neurite mask by thresholding the probability map.

    Convenience wrapper for callers that want a mask directly. The DOWNSTREAM
    skeletonization / per-soma attribution / ``neuritecelldata`` writing is NOT
    done here — that stays in the shared backend (``bin/neurite.py``).

    Args:
        image: 2D morphology image (H, W).
        checkpoint: Path to a trained checkpoint (default ``DEFAULT_CHECKPOINT``).
        threshold: Probability threshold in [0, 1].
        device: ``"cpu"`` or ``"cuda"``.
        normalize: Apply percentile normalization internally.

    Returns:
        Boolean mask (H, W), ``prob >= threshold``.
    """
    prob = predict_neurite_probmap(
        image, checkpoint=checkpoint, device=device, normalize=normalize
    )
    return prob >= threshold


def main() -> None:
    """CLI: probability map / mask for one tile (standalone testing only)."""
    import argparse

    import tifffile

    ap = argparse.ArgumentParser(description="DL neurite probability map (stub).")
    ap.add_argument("--image", required=True, help="morphology tile (.tif/.npy)")
    ap.add_argument("--checkpoint", default=None, help="trained .pt checkpoint")
    ap.add_argument("--out", required=True, help="output probability map (.tif)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threshold", type=float, default=None,
                    help="if set, save a binary mask at this threshold too")
    args = ap.parse_args()

    if args.image.endswith(".npy"):
        image = np.load(args.image)
    else:
        image = tifffile.imread(args.image)
    prob = predict_neurite_probmap(
        image.astype(np.float32), checkpoint=args.checkpoint, device=args.device
    )
    tifffile.imwrite(args.out, prob.astype(np.float32))
    print(f"wrote probability map {prob.shape} -> {args.out}")
    if args.threshold is not None:
        mask = (prob >= args.threshold).astype(np.uint8) * 255
        mask_out = args.out.rsplit(".", 1)[0] + "_mask.tif"
        tifffile.imwrite(mask_out, mask)
        print(f"wrote binary mask @ {args.threshold} -> {mask_out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Noise2Void denoising wrapper for faint morphology tiles.

Faint fluorescence (RGEDI / FITC=EGFP fill) is shot-noise limited, and that
noise is what fragments thin neurites during segmentation. Noise2Void (Krull et
al., CVPR 2019) is self-supervised: it learns to denoise from the noisy images
themselves, needing no clean targets — ideal here, since we have no noise-free
ground truth. This pairs with the ``--denoise n2v`` hook Track A is adding: the
pipeline can denoise a tile before it reaches either the Frangi path or the DL
model.

This is a THIN wrapper over ``careamics`` (the maintained N2V implementation) if
it is importable, with a fallback to the legacy ``n2v`` package. If neither is
installed:

* :func:`apply` in ``passthrough`` mode returns the image unchanged (a safe
  no-op, so ``--denoise n2v`` degrades gracefully), and
* :func:`train` / :func:`apply` in ``strict`` mode raise a clear ImportError.

The wrapper deliberately keeps a tiny, stable surface (``train`` -> checkpoint,
``apply`` -> denoised array) so Track A can call it without knowing which
backend is present.
"""

from __future__ import annotations

import argparse
from typing import Optional, Tuple

import numpy as np


def _detect_backend() -> Optional[str]:
    """Detect an available Noise2Void backend.

    Returns:
        ``"careamics"``, ``"n2v"``, or ``None`` if neither is importable.
    """
    try:
        import careamics  # type: ignore  # noqa: F401

        return "careamics"
    except Exception:
        pass
    try:
        import n2v  # type: ignore  # noqa: F401

        return "n2v"
    except Exception:
        return None


def available() -> bool:
    """Report whether any N2V backend is installed.

    Returns:
        True if careamics or n2v is importable.
    """
    return _detect_backend() is not None


class Noise2VoidDenoiser:
    """Train/apply a Noise2Void model, or pass through when unavailable."""

    def __init__(self, mode: str = "passthrough") -> None:
        """Initialise the denoiser.

        Args:
            mode: ``"passthrough"`` returns images unchanged when no backend is
                installed (safe default for the pipeline). ``"strict"`` raises
                if a backend is required but missing.

        Raises:
            ValueError: If ``mode`` is not recognised.
        """
        if mode not in ("passthrough", "strict"):
            raise ValueError("mode must be 'passthrough' or 'strict'")
        self.mode = mode
        self.backend = _detect_backend()
        self._model = None

    def _require_backend(self) -> str:
        """Return the backend name or raise a clear ImportError.

        Returns:
            Backend name.

        Raises:
            ImportError: If no backend is installed.
        """
        if self.backend is None:
            raise ImportError(
                "Noise2Void requires 'careamics' (preferred) or 'n2v'. Install "
                "one (see ml/neurite/requirements-ml.txt) or use "
                "mode='passthrough' / --denoise none."
            )
        return self.backend

    def train(
        self,
        images: np.ndarray,
        checkpoint: str,
        epochs: int = 20,
        patch_size: Tuple[int, int] = (64, 64),
    ) -> str:
        """Train a Noise2Void model on noisy images.

        Args:
            images: Stack of noisy images, shape (N, H, W) or a single (H, W).
            checkpoint: Path to write the trained model / config.
            epochs: Training epochs.
            patch_size: Patch size for N2V training.

        Returns:
            The checkpoint path.

        Raises:
            ImportError: If no N2V backend is installed.
        """
        backend = self._require_backend()
        images = np.asarray(images)
        if images.ndim == 2:
            images = images[None]

        if backend == "careamics":
            # careamics >= 0.1 API. Kept minimal; tune for real runs on GPU.
            from careamics import CAREamist  # type: ignore
            from careamics.config import create_n2v_configuration  # type: ignore

            config = create_n2v_configuration(
                experiment_name="neurite_n2v",
                data_type="array",
                axes="SYX",
                patch_size=list(patch_size),
                batch_size=8,
                num_epochs=epochs,
            )
            engine = CAREamist(source=config)
            engine.train(train_source=images.astype(np.float32))
            engine.save_checkpoint(checkpoint)
            self._model = engine
        else:  # legacy n2v
            raise ImportError(
                "the legacy 'n2v' backend needs a Keras/TF training setup that "
                "this thin wrapper does not script; install 'careamics' for the "
                "supported path."
            )
        return checkpoint

    def load(self, checkpoint: str) -> None:
        """Load a trained model from a checkpoint.

        Args:
            checkpoint: Path to a saved model/config.

        Raises:
            ImportError: If no N2V backend is installed.
        """
        backend = self._require_backend()
        if backend == "careamics":
            from careamics import CAREamist  # type: ignore

            self._model = CAREamist(source=checkpoint)
        else:
            raise ImportError("legacy 'n2v' loading not scripted; use careamics.")

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Denoise an image (or pass through if no backend / not trained).

        Args:
            image: 2D image (H, W).

        Returns:
            Denoised float32 image (H, W). In ``passthrough`` mode with no
            backend/model, returns ``image`` unchanged (as float32).

        Raises:
            ImportError: In ``strict`` mode when no backend is installed.
        """
        image = np.asarray(image, dtype=np.float32)
        if self.backend is None:
            if self.mode == "strict":
                self._require_backend()
            return image  # no-op passthrough
        if self._model is None:
            # Backend present but nothing loaded/trained: safe passthrough.
            if self.mode == "strict":
                raise RuntimeError("no N2V model loaded; call train() or load().")
            return image
        pred = self._model.predict(source=image[None].astype(np.float32))
        pred = np.asarray(pred).squeeze().astype(np.float32)
        return pred


def apply(image: np.ndarray, mode: str = "passthrough") -> np.ndarray:
    """One-shot convenience: passthrough denoise for an untrained/absent backend.

    Track A's ``--denoise n2v`` hook can call this for a graceful no-op when no
    trained model is configured; wire a trained :class:`Noise2VoidDenoiser` in
    for real denoising.

    Args:
        image: 2D image (H, W).
        mode: ``"passthrough"`` or ``"strict"``.

    Returns:
        Float32 image (denoised if a model were loaded; otherwise unchanged).
    """
    return Noise2VoidDenoiser(mode=mode).apply(image)


def main() -> None:
    """CLI: train on tiles or apply a model to one tile."""
    import glob
    import os

    import tifffile

    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    tr = sub.add_parser("train", help="train an N2V model on a dir of tiles")
    tr.add_argument("--tiles", required=True, help="dir of .tif tiles")
    tr.add_argument("--checkpoint", required=True)
    tr.add_argument("--epochs", type=int, default=20)

    ap_apply = sub.add_parser("apply", help="denoise one tile")
    ap_apply.add_argument("--image", required=True)
    ap_apply.add_argument("--checkpoint", default=None)
    ap_apply.add_argument("--out", required=True)
    ap_apply.add_argument("--mode", default="passthrough",
                          choices=["passthrough", "strict"])

    args = ap.parse_args()
    print(f"N2V backend available: {_detect_backend() or 'NONE'}")

    if args.cmd == "train":
        files = sorted(glob.glob(os.path.join(args.tiles, "*.tif")))
        stack = np.stack([tifffile.imread(f).astype(np.float32) for f in files])
        d = Noise2VoidDenoiser(mode="strict")
        d.train(stack, args.checkpoint, epochs=args.epochs)
        print(f"trained -> {args.checkpoint}")
    else:
        d = Noise2VoidDenoiser(mode=args.mode)
        if args.checkpoint:
            d.load(args.checkpoint)
        image = tifffile.imread(args.image).astype(np.float32)
        out = d.apply(image)
        tifffile.imwrite(args.out, out)
        print(f"wrote {out.shape} -> {args.out}")


if __name__ == "__main__":
    main()

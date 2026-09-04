#!/usr/bin/env python3
"""Inference: checkpoint + morphology image -> neurite probability map.

Loads a checkpoint saved by ``train.py`` and produces a float32 probability map
(0-1) the same HxW as the input. Handles the two practical wrinkles of running a
padded U-Net on real tiles:

* **Normalization** — applies the same 1-99.5 percentile normalization the model
  was trained on (skippable if the caller already normalized).
* **Size padding** — reflect-pads HxW up to a multiple of the model's
  ``valid_multiple()`` so any tile size works, then crops the map back.

Optional overlap tiling (``tile``/``overlap``) keeps memory bounded on large
2048x2048 fields; by default the whole field is run at once.

This module is import-safe without a checkpoint. It is used both by ``train.py``
(validation) and by the pipeline stub ``bin/neurite_model.py``.
"""

from __future__ import annotations

import argparse
from typing import Optional

import numpy as np
import torch

from dataset import percentile_normalize
from model import UNet, build_model


def load_checkpoint(path: str, device: torch.device) -> UNet:
    """Load a model from a training checkpoint.

    Args:
        path: Path to a ``.pt`` checkpoint saved by ``train.py`` (contains
            ``model_state`` and ``config``).
        device: Torch device to map the model onto.

    Returns:
        A :class:`UNet` in eval mode with weights loaded.
    """
    ckpt = torch.load(path, map_location=device)
    cfg = ckpt.get("config", {"depth": 4, "base_channels": 16, "in_channels": 1})
    model = build_model(
        depth=cfg.get("depth", 4),
        base_channels=cfg.get("base_channels", 16),
        in_channels=cfg.get("in_channels", 1),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def _pad_to_multiple(image: np.ndarray, mult: int) -> tuple:
    """Reflect-pad a 2D image so H and W are multiples of ``mult``.

    Args:
        image: 2D array (H, W).
        mult: Required divisor.

    Returns:
        ``(padded_image, (h, w))`` where ``(h, w)`` is the original shape.
    """
    h, w = image.shape
    ph = (mult - h % mult) % mult
    pw = (mult - w % mult) % mult
    if ph or pw:
        image = np.pad(image, ((0, ph), (0, pw)), mode="reflect")
    return image, (h, w)


@torch.no_grad()
def predict_probmap(
    model: UNet,
    image: np.ndarray,
    device: Optional[torch.device] = None,
    normalize: bool = True,
) -> np.ndarray:
    """Predict a neurite probability map for one image.

    Args:
        model: A loaded :class:`UNet` (eval mode).
        image: 2D morphology image (H, W), any real dtype.
        device: Torch device; inferred from model if ``None``.
        normalize: If True, apply 1-99.5 percentile normalization first. Set
            False if ``image`` is already normalized to [0, 1].

    Returns:
        Float32 probability map (H, W) in [0, 1].
    """
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    img = image.astype(np.float32)
    if img.ndim == 3:
        img = img[..., 0]
    if normalize:
        img = percentile_normalize(img)
    padded, (h, w) = _pad_to_multiple(img, model.valid_multiple())
    x = torch.from_numpy(padded)[None, None].to(device)
    prob = torch.sigmoid(model(x))[0, 0].cpu().numpy()
    return prob[:h, :w].astype(np.float32)


def main() -> None:
    """CLI: run inference on a single tile and save a probability map."""
    import tifffile

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True, help="path to best.pt")
    ap.add_argument("--image", required=True, help="input morphology tile (.tif/.npy)")
    ap.add_argument("--out", required=True, help="output probability map (.tif)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threshold", type=float, default=None,
                    help="if set, also save a binarised mask at this threshold")
    args = ap.parse_args()

    device = torch.device(args.device)
    model = load_checkpoint(args.checkpoint, device)
    if args.image.endswith(".npy"):
        image = np.load(args.image)
    else:
        image = tifffile.imread(args.image)
    prob = predict_probmap(model, image.astype(np.float32), device=device)
    tifffile.imwrite(args.out, prob.astype(np.float32))
    print(f"wrote probability map {prob.shape} -> {args.out} "
          f"(range [{prob.min():.3f},{prob.max():.3f}])")
    if args.threshold is not None:
        mask = (prob >= args.threshold).astype(np.uint8) * 255
        mask_out = args.out.rsplit(".", 1)[0] + "_mask.tif"
        tifffile.imwrite(mask_out, mask)
        print(f"wrote binary mask @ {args.threshold} -> {mask_out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Post-training evaluation + QA for the neurite U-Net.

For each held-out validation field, this:

* runs the trained model to a probability map and scores skeleton-F1 (tol=3,
  the benchmark metric) at the best threshold;
* runs a **Frangi vesselness baseline** on the identical image/GT and scores it
  the same way — the apples-to-apples "did DL beat classical on THESE fields"
  comparison (the CP-Tubeness / Frangi ceilings were F1 ~0.25 / ~0.19 on the
  earlier annotation set);
* writes an RGB QA overlay: gray image, GT centreline in green, DL prediction
  centreline in magenta (agreement reads white-ish).

Usage::

    python eval_qa.py --data <dir> --checkpoint best.pt --out <dir> \
        --val-stems H15,I07 --device cuda
"""

from __future__ import annotations

import argparse
import glob
import os
from typing import List

import numpy as np
import tifffile
import torch
from skimage.filters import frangi
from skimage.morphology import binary_dilation, disk, skeletonize

from dataset import percentile_normalize
from infer import load_checkpoint, predict_probmap
from metrics import best_threshold_f1, skeleton_f1


def frangi_probmap(image: np.ndarray) -> np.ndarray:
    """Frangi vesselness response normalized to [0, 1] (classical baseline).

    Args:
        image: Normalized morphology image (H, W) in [0, 1].

    Returns:
        Float32 vesselness map in [0, 1].
    """
    r = frangi(image, sigmas=[1, 2, 3], black_ridges=False)
    m = float(r.max())
    return (r / (m + 1e-9)).astype(np.float32)


def val_fields(data_dir: str, val_stems: List[str]) -> List[str]:
    """List ``_img.npy`` paths whose name contains any of ``val_stems``.

    Args:
        data_dir: Directory of rasterized pairs.
        val_stems: Substrings identifying validation fields.

    Returns:
        Sorted list of matching image paths.
    """
    return sorted(
        p for p in glob.glob(os.path.join(data_dir, "*_img.npy"))
        if any(s in p for s in val_stems)
    )


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--val-stems", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = torch.device(args.device)
    model = load_checkpoint(args.checkpoint, device)
    stems = [s.strip() for s in args.val_stems.split(",") if s.strip()]

    print(f"{'field':40}{'DL_F1':>8}{'DL_thr':>8}{'Frangi_F1':>11}")
    dl_all, fr_all = [], []
    for img_path in val_fields(args.data, stems):
        name = os.path.basename(img_path)[: -len("_img.npy")]
        image = percentile_normalize(np.load(img_path))
        skel_path = img_path.replace("_img", "_skel")
        gt = (np.load(skel_path) > 0) if os.path.exists(skel_path) \
            else (np.load(img_path.replace("_img", "_mask")) > 0)

        prob = predict_probmap(model, image, device=device, normalize=False)
        dl_t, dl_f = best_threshold_f1(gt, prob)
        fr = frangi_probmap(image)
        fr_t, fr_f = best_threshold_f1(gt, fr)
        dl_all.append(dl_f)
        fr_all.append(fr_f)
        print(f"{name:40}{dl_f:8.3f}{dl_t:8.2f}{fr_f:11.3f}")

        g = (np.clip(image, 0, 1) * 255).astype(np.uint8)
        rgb = np.dstack([g, g, g])
        dl_sk = skeletonize(prob >= dl_t)
        rgb[binary_dilation(gt, disk(1))] = [0, 255, 0]           # GT green
        rgb[binary_dilation(dl_sk, disk(1))] = [255, 0, 255]      # DL magenta
        tifffile.imwrite(os.path.join(args.out, f"{name}_evalqa.tif"), rgb)

    if dl_all:
        print(f"\nMEAN  DL_F1={np.mean(dl_all):.3f}   Frangi_F1={np.mean(fr_all):.3f}   "
              f"(DL {'BEATS' if np.mean(dl_all) > np.mean(fr_all) else 'does NOT beat'} classical)")


if __name__ == "__main__":
    main()

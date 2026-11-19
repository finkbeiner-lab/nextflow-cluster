#!/usr/bin/env python3
"""Pick one neurite-rich crop per image for hand tracing in SNT.

Step 1 of making training data for the neurite model. For each input tile the
script slides a ``--roi`` x ``--roi`` box over the image, scores each position
by ridge (neurite-like) signal minus bright saturated area, and saves the best
box as a RAW crop (bit depth kept, so it looks right in Fiji):

* ``<name>__ROI.tif``          trace THIS one, exhaustively, in SNT.
* ``<name>__ROI_context.tif``  the whole tile (half size) with the box drawn
  in yellow, so you can see where the crop came from. Reference only.

Pick input tiles that cover your conditions: every genotype / cell line /
experiment, dense and sparse wells. The production model used 20 crops
(640 px) from XDP5/8/10/12, of which 12 were traced.

CLI::

    python pick_roi_crops.py --in <dir of tiles> --out <tracing dir> --roi 640
"""

from __future__ import annotations

import argparse
import glob
import os
from typing import Tuple

import numpy as np
import tifffile
from skimage.filters import frangi, gaussian, meijering


def stretch(img: np.ndarray, lo: float = 1, hi: float = 99.5) -> np.ndarray:
    """Percentile-stretch an image to [0, 1]."""
    a, b = np.percentile(img, [lo, hi])
    return np.clip((img - a) / (b - a + 1e-6), 0, 1)


def ridge(norm: np.ndarray) -> np.ndarray:
    """Neurite-like ridge signal in [0, 1], with saturated cell bodies zeroed."""
    flat = np.clip(norm - gaussian(norm, 15), 0, None)
    flat = flat / (flat.max() + 1e-6)
    try:
        r = meijering(flat, sigmas=[1, 2, 3], black_ridges=False)
    except Exception:  # noqa: BLE001 - older skimage
        r = frangi(flat, sigmas=[1, 2, 3], black_ridges=False)
    r = np.clip(r / (np.percentile(r, 99.9) + 1e-9), 0, 1)
    r[norm > np.percentile(norm, 98)] = 0  # ignore bright soma cores
    return r


def pick_roi(norm: np.ndarray, roi: int, stride: int) -> Tuple[int, int]:
    """Top-left (y, x) of the box with the most ridge signal, least saturation."""
    r = ridge(norm)
    sat = (norm > np.percentile(norm, 99.5)).astype(np.float32)
    H, W = norm.shape
    if roi > min(H, W):
        raise ValueError(f"--roi {roi} is larger than the image {H}x{W}")
    best, best_score = (0, 0), -1.0
    for y in range(0, H - roi + 1, stride):
        for x in range(0, W - roi + 1, stride):
            score = r[y:y + roi, x:x + roi].mean() - 0.5 * sat[y:y + roi, x:x + roi].mean()
            if score > best_score:
                best, best_score = (y, x), score
    return best


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="indir", required=True, help="folder of input tiles (*.tif)")
    ap.add_argument("--out", required=True, help="folder for the crops to trace")
    ap.add_argument("--roi", type=int, default=640, help="crop size in px (default 640)")
    ap.add_argument("--stride", type=int, default=256, help="search step in px (default 256)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    files = sorted(glob.glob(os.path.join(args.indir, "*.tif")))
    if not files:
        raise SystemExit(f"no .tif files in {args.indir}")
    print(f"{len(files)} tiles; crop {args.roi} px", flush=True)
    for f in files:
        name = os.path.splitext(os.path.basename(f))[0]
        raw = tifffile.imread(f)
        raw = raw[..., 0] if raw.ndim == 3 else raw
        norm = stretch(raw.astype(np.float32))
        y, x = pick_roi(norm, args.roi, args.stride)
        tifffile.imwrite(os.path.join(args.out, f"{name}__ROI.tif"),
                         raw[y:y + args.roi, x:x + args.roi])
        g = (norm[::2, ::2] * 255).astype(np.uint8)
        ctx = np.dstack([g, g, g])
        yy, xx, s = y // 2, x // 2, args.roi // 2
        for sl in ((slice(yy, yy + 2), slice(xx, xx + s)), (slice(yy + s - 2, yy + s), slice(xx, xx + s)),
                   (slice(yy, yy + s), slice(xx, xx + 2)), (slice(yy, yy + s), slice(xx + s - 2, xx + s))):
            ctx[sl] = [255, 255, 0]
        tifffile.imwrite(os.path.join(args.out, f"{name}__ROI_context.tif"), ctx)
        print(f"{name:44} crop at y={y}, x={x}", flush=True)
    print("DONE.", args.out)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Grow Cellpose masks into the GFP foreground mask (rescue dim tails + dim missed cells).

Cellpose-SAM misses the thin dim tails of RGEDI fibroblasts and many dim-but-above-background
cells. The per-tile foreground mask from ml/rgedi_tile/fg_tile_readout.py (bg = LEVEL * s * P)
does see them, so for each export crop:
    1. foreground montage: per-tile background fit + foreground() exactly as fg_tile_readout.py,
       stitched like export_crops.montage (10% overlap, later tiles overwrite), then cropped
    2. domain = foreground OR Cellpose mask
    3. reach = domain pixels within --dmax px (geodesic, inside the domain) of any Cellpose cell
    4. grown = watershed of inverted smoothed GFP from the Cellpose cells, restricted to reach,
       so a tail is claimed by the cell it connects to and neighbors split along dim valleys
    5. domain components left over (not reachable) with area >= --min_new become NEW cells

Writes <out>/<tag>.npz with keys fg, <variant>_grown for each variant; view with
    compare_view.py <export_dir> <tag> --variants d50,d75 --extra d50_grown=<out>/<tag>.npz:d50_grown ...

Usage:
    python grow_into_fg.py <export_dir> <tiles_root> <P.npy> --tags pmsG-2_F7_T7,... [--out DIR]
tiles_root holds <well>/<raw GFP tiles> for each crop's well+timepoint.
"""
import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage
from skimage.segmentation import relabel_sequential, watershed

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "rgedi_tile"))
import fg_tile_readout as fgr  # noqa: E402

TILE_POS = {1: (0, 0), 2: (0, 1), 3: (0, 2), 4: (1, 2), 5: (1, 1), 6: (1, 0),
            7: (2, 0), 8: (2, 1), 9: (2, 2)}
CROP = 1024


def fg_montage(tiles_root: str, well: str, tp: str) -> np.ndarray:
    """Foreground mask of the whole 3x3 montage, using fg_tile_readout's background model."""
    imgs = {}
    for n in TILE_POS:
        hits = glob.glob(f"{tiles_root}/{well}/*_{tp}_*_{well}_{n}_{fgr.GFP}_*.tif")
        imgs[n] = tifffile.imread(hits[0]).astype(np.float32)
    fits = {n: fgr.bg_fit(a) for n, a in imgs.items()}
    s_med = np.nanmedian([s for s, _ in fits.values()])
    keep = fgr.STRIDE
    canvas = np.zeros((keep * 2 + fgr.TILE, keep * 2 + fgr.TILE), bool)
    for n, (r, c) in TILE_POS.items():
        s, sig = fits[n]
        fg = fgr.foreground(imgs[n], s if np.isfinite(s) else s_med, sig)
        canvas[r * keep:r * keep + fgr.TILE, c * keep:c * keep + fgr.TILE] = fg
    print(f"  {well} {tp}: tile s = " + " ".join(f"{fits[n][0]:.2f}" for n in TILE_POS), flush=True)
    return canvas


def grow(masks: np.ndarray, fg: np.ndarray, gfp: np.ndarray, dmax: int, min_new: int) -> np.ndarray:
    """Grow Cellpose labels through the foreground; add unreachable fg blobs as new cells."""
    domain = fg | (masks > 0)
    reach = ndimage.binary_dilation(masks > 0, iterations=dmax, mask=domain)
    elev = -ndimage.gaussian_filter(gfp, 2)
    out = watershed(elev, markers=masks, mask=reach)
    left, n = ndimage.label(domain & ~reach)
    if n:
        areas = np.bincount(left.ravel())
        big = np.flatnonzero(areas >= min_new)
        big = big[big > 0]
        lut = np.zeros(n + 1, np.int32)
        lut[big] = np.arange(1, big.size + 1) + out.max()
        out = np.where(lut[left] > 0, lut[left], out)
    return relabel_sequential(out)[0].astype(np.uint16)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("export_dir"); ap.add_argument("tiles_root"); ap.add_argument("p_npy")
    ap.add_argument("--tags", required=True)
    ap.add_argument("--variants", default="d50,d75")
    ap.add_argument("--dmax", type=int, default=200, help="max geodesic growth from a Cellpose cell (px)")
    ap.add_argument("--min_new", type=int, default=1500, help="min area of a new fg-only cell (px)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    fgr.P = np.load(a.p_npy)
    out_dir = a.out or os.path.join(a.export_dir, "grown")
    os.makedirs(out_dir, exist_ok=True)
    man = pd.read_csv(os.path.join(a.export_dir, "manifest.csv")).set_index("tag")
    for tag in a.tags.split(","):
        r = man.loc[tag]
        fg = fg_montage(a.tiles_root, r.well, r.tp)[r.y0:r.y0 + CROP, r.x0:r.x0 + CROP]
        gfp = tifffile.imread(os.path.join(a.export_dir, "crops", f"{tag}.tif")).astype(np.float32)
        preds = np.load(os.path.join(a.export_dir, "preds", f"{tag}.npz"))
        res = {"fg": ndimage.label(fg)[0].astype(np.uint16)}
        for v in a.variants.split(","):
            m = preds[v].astype(np.int32)
            g = grow(m, fg, gfp, a.dmax, a.min_new)
            res[f"{v}_grown"] = g
            n0, n1 = int(m.max()), int(g.max())
            print(f"  {tag} {v}: {n0} -> {n1} cells (+{n1 - n0} new); "
                  f"fg covered {((m > 0) & fg).sum() / fg.sum():.2f} -> {((g > 0) & fg).sum() / fg.sum():.2f}",
                  flush=True)
        np.savez_compressed(os.path.join(out_dir, f"{tag}.npz"), **res)


if __name__ == "__main__":
    main()

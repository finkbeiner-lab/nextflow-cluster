#!/usr/bin/env python
"""Convert Fiji-edited ROI sets into Cellpose training masks + a what-changed report.

For each <tag>_edited.zip in the labeling folder: rasterize every ROI (polygon, freehand,
traced, brush-edited composite, rectangle, oval) into a uint16 label image. Where ROIs
overlap, the SMALLER ROI wins (keeps compact cells on top of the spread cells they sit on).
Compares against <tag>_prefill.zip: unchanged / edited / deleted / added cells.

Outputs (in <label_dir>/masks/):
    <tag>_masks.tif      uint16 labels (0 = background), same shape as <tag>.tif
    <tag>_review.png     GFP + model outline (magenta) vs yours (green)
    labels_summary.csv   per crop: n_cells, unchanged, edited, added, deleted, area_gain
Usage:
    python rois_to_masks.py <label_dir>
"""
import argparse
import os

import numpy as np
import pandas as pd
import roifile
import tifffile
from PIL import Image
from scipy import ndimage
from skimage.draw import ellipse, polygon2mask
from skimage.segmentation import find_boundaries


def roi_to_binary(roi: roifile.ImagejRoi, shape: tuple) -> np.ndarray:
    t = roi.roitype
    if t == roifile.ROI_TYPE.RECT and not roi.composite:
        m = np.zeros(shape, bool)
        m[max(roi.top, 0):roi.bottom, max(roi.left, 0):roi.right] = True
        return m
    if t == roifile.ROI_TYPE.OVAL:
        m = np.zeros(shape, bool)
        cy, cx = (roi.top + roi.bottom) / 2, (roi.left + roi.right) / 2
        rr, cc = ellipse(cy, cx, (roi.bottom - roi.top) / 2, (roi.right - roi.left) / 2, shape=shape)
        m[rr, cc] = True
        return m
    paths = roi.coordinates(multi=True)
    m = np.zeros(shape, bool)
    for p in paths:
        if p is None or len(p) < 3:
            continue
        m ^= polygon2mask(shape, np.asarray(p)[:, ::-1])  # (x,y) -> (row,col); XOR = holes
    return m


def rasterize(path: str, shape: tuple) -> np.ndarray:
    """ROI zip -> label image; smaller ROIs painted last so they win overlaps."""
    bins = [roi_to_binary(r, shape) for r in roifile.roiread(path)]
    bins = [b for b in bins if b.sum() >= 20]
    order = np.argsort([-b.sum() for b in bins])
    lab = np.zeros(shape, np.uint16)
    for new_id, i in enumerate(order, start=1):
        lab[bins[i]] = new_id
    # drop labels fully covered by smaller ones, renumber
    ids = np.unique(lab); ids = ids[ids > 0]
    remap = np.zeros(int(lab.max()) + 1, np.uint16); remap[ids] = np.arange(1, len(ids) + 1)
    return remap[lab]


def match(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU matrix (labels of a) x (labels of b)."""
    na, nb = int(a.max()), int(b.max())
    inter = np.zeros((na + 1, nb + 1))
    np.add.at(inter, (a.ravel(), b.ravel()), 1)
    area_a = inter.sum(1, keepdims=True); area_b = inter.sum(0, keepdims=True)
    iou = inter / np.maximum(area_a + area_b - inter, 1)
    return iou[1:, 1:]


def review_png(raw: np.ndarray, pre: np.ndarray, fin: np.ndarray, path: str) -> None:
    lo, hi = np.percentile(raw, 1), np.percentile(raw, 99.5)
    g = (np.clip((raw - lo) / (hi - lo + 1e-9), 0, 1) ** 0.5 * 255).astype(np.uint8)
    rgb = np.repeat(g[..., None], 3, 2)
    bp = ndimage.binary_dilation(find_boundaries(pre, mode="inner"))
    bf = ndimage.binary_dilation(find_boundaries(fin, mode="inner"))
    rgb[bp] = (255, 0, 255); rgb[bf] = (0, 255, 0)
    Image.fromarray(rgb).save(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("label_dir")
    a = ap.parse_args()
    d = a.label_dir
    od = os.path.join(d, "masks"); os.makedirs(od, exist_ok=True)
    man = pd.read_csv(os.path.join(d, "manifest.csv"))
    rows = []
    for r in man.itertuples():
        ed = os.path.join(d, f"{r.tag}_edited.zip")
        empty = os.path.join(d, f"{r.tag}_edited_EMPTY.txt")
        if not (os.path.exists(ed) or os.path.exists(empty)):
            print(f"{r.tag}: not labeled yet"); continue
        raw = tifffile.imread(os.path.join(d, f"{r.tag}.tif")).astype(np.float32)
        fin = rasterize(ed, raw.shape) if os.path.exists(ed) else np.zeros(raw.shape, np.uint16)
        pre = rasterize(os.path.join(d, f"{r.tag}_prefill.zip"), raw.shape)
        tifffile.imwrite(os.path.join(od, f"{r.tag}_masks.tif"), fin)
        review_png(raw, pre, fin, os.path.join(od, f"{r.tag}_review.png"))
        iou = match(pre, fin) if pre.max() and fin.max() else np.zeros((int(pre.max()), int(fin.max())))
        best_pre = iou.max(1) if iou.size else np.zeros(int(pre.max()))
        best_fin = iou.max(0) if iou.size else np.zeros(int(fin.max()))
        rows.append(dict(tag=r.tag, kind=r.kind, split=r.split, n_prefill=int(pre.max()), n_cells=int(fin.max()),
                         unchanged=int((best_fin >= 0.95).sum()),
                         edited=int(((best_fin >= 0.3) & (best_fin < 0.95)).sum()),
                         added=int((best_fin < 0.3).sum()), deleted=int((best_pre < 0.3).sum()),
                         area_gain=round(float((fin > 0).sum() / max(1, (pre > 0).sum())), 3)))
        print(f"{r.tag}: {rows[-1]}")
    if rows:
        pd.DataFrame(rows).to_csv(os.path.join(od, "labels_summary.csv"), index=False)
    print(f"-> {od}")


if __name__ == "__main__":
    main()

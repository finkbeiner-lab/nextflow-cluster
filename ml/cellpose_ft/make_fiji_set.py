#!/usr/bin/env python
"""Build a Fiji labeling folder from a round-1 export (SNT-style ground truth for cells).

For every crop writes:
    <tag>.tif          raw 16-bit GFP16 crop (what you look at; Fiji auto-contrast)
    <tag>_prefill.zip  ImageJ ROI set = the model's outlines from the chosen variant
plus label_cells.ijm (the Fiji macro that walks you through the crops) and manifest.csv.
The macro saves your edits as <tag>_edited.zip; rois_to_masks.py turns those into
Cellpose training masks.

Usage:
    python make_fiji_set.py <export_dir> <label_dir> [--variant d50]
                            [--variant-for TAG=VAR ...]
"""
import argparse
import os
import shutil

import numpy as np
import pandas as pd
import roifile
from scipy import ndimage
from skimage.measure import approximate_polygon, find_contours

# sparse compact crops where 0.6x scale lost dim cells (round-1 panels) -> start from d25
DEFAULT_OVERRIDES = {"pmsG-1_C6_T10": "d25", "pmsG-1_E3_T2": "d25"}
# labeling order: a complete round-1 set first (2 compact, 2 spread, 1 dense, 1 compact-sparse
# training crops + the 3 held-out), then the rest -- stop after 9 and the tuned model prefills the rest
PRIORITY = ["pmsG-1_E6_T6", "pmsG-2_C8_T0", "pmsG-1_C6_T10", "pmsG-1_F4_T1", "pmsG-2_E8_T6",
            "pmsG-1_F5_T14", "pmsG-1_D4_T3", "pmsG-1_E5_T12", "pmsG-2_F6_T6"]


def mask_to_polygon(binary: np.ndarray) -> np.ndarray:
    """Outline (x, y) of the largest connected piece of one cell mask."""
    lab, n = ndimage.label(binary)
    if n > 1:
        sizes = ndimage.sum(binary, lab, range(1, n + 1))
        binary = lab == (1 + int(np.argmax(sizes)))
    padded = np.pad(binary, 1).astype(np.float32)
    contours = find_contours(padded, 0.5)
    c = max(contours, key=len) - 1.0          # undo pad; (row, col)
    c = approximate_polygon(c, tolerance=0.75)  # fewer vertices, same shape
    return c[:, ::-1]                          # -> (x, y) for ImageJ


def write_roiset(masks: np.ndarray, path: str) -> int:
    rois = []
    objs = ndimage.find_objects(masks)
    for lab, sl in enumerate(objs, start=1):
        if sl is None:
            continue
        y0, x0 = sl[0].start, sl[1].start
        sub = masks[sl] == lab
        if sub.sum() < 20:
            continue
        xy = mask_to_polygon(sub) + [x0, y0]
        roi = roifile.ImagejRoi.frompoints(xy.astype(np.float32))
        roi.roitype = roifile.ROI_TYPE.POLYGON
        roi.name = f"cell_{len(rois) + 1:03d}"
        rois.append(roi)
    if os.path.exists(path):
        os.remove(path)
    roifile.roiwrite(path, rois)
    return len(rois)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("export_dir")
    ap.add_argument("label_dir")
    ap.add_argument("--variant", default="d50")
    ap.add_argument("--variant-for", action="append", default=[])
    a = ap.parse_args()
    override = dict(DEFAULT_OVERRIDES)
    override.update(dict(s.split("=", 1) for s in a.variant_for))

    man = pd.read_csv(os.path.join(a.export_dir, "manifest.csv"))
    rank = {t: i for i, t in enumerate(PRIORITY)}
    man = man.assign(order=[rank.get(t, len(PRIORITY) + i) for i, t in enumerate(man.tag)])
    man = man.sort_values("order").drop(columns="order")
    man["round"] = [1 if t in rank else 2 for t in man.tag]
    os.makedirs(a.label_dir, exist_ok=True)
    rows = []
    for r in man.itertuples():
        v = override.get(r.tag, a.variant)
        shutil.copy(os.path.join(a.export_dir, "crops", f"{r.tag}.tif"),
                    os.path.join(a.label_dir, f"{r.tag}.tif"))
        masks = np.load(os.path.join(a.export_dir, "preds", f"{r.tag}.npz"))[v]
        n = write_roiset(masks, os.path.join(a.label_dir, f"{r.tag}_prefill.zip"))
        rows.append(dict(r._asdict(), prefill_variant=v, prefill_n=n))
        print(f"[round {r.round}] {r.tag}: {n} prefill ROIs from {v} ({r.kind}, {r.split})")
    pd.DataFrame(rows).drop(columns="Index").to_csv(os.path.join(a.label_dir, "manifest.csv"), index=False)
    shutil.copy(os.path.join(os.path.dirname(os.path.abspath(__file__)), "label_cells.ijm"),
                os.path.join(a.label_dir, "label_cells.ijm"))
    print(f"-> {a.label_dir}  (open Fiji, Plugins > Macros > Run... > label_cells.ijm)")


if __name__ == "__main__":
    main()

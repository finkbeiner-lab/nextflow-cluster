#!/usr/bin/env python
"""Build a Cellpose-GUI correction folder from a round-1 export.

Writes <tag>.tif + <tag>_seg.npy (masks pre-filled from the chosen variant) for every
crop, so the GUI opens each image with the model's masks already drawn: fix them, then
Ctrl+S to save. Only the keys the 4.0.7 GUI loader needs are written (outlines, masks,
ismanual, filename, diameter).

Usage:
    python make_gui_set.py <export_dir> <out_dir> --variant d50 [--variant-for TAG=VAR ...]
"""
import argparse
import os
import shutil

import numpy as np
import pandas as pd
from skimage.segmentation import find_boundaries


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("export_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--variant", required=True, help="default variant: prod | d25 | d50 | d75")
    ap.add_argument("--variant-for", action="append", default=[],
                    help="per-crop override TAG=VARIANT (repeatable)")
    a = ap.parse_args()
    override = dict(s.split("=", 1) for s in a.variant_for)
    man = pd.read_csv(os.path.join(a.export_dir, "manifest.csv"))
    os.makedirs(a.out_dir, exist_ok=True)
    for tag in man["tag"]:
        v = override.get(tag, a.variant)
        dst_img = os.path.join(a.out_dir, f"{tag}.tif")
        seg = os.path.join(a.out_dir, f"{tag}_seg.npy")
        if os.path.exists(seg):
            print(f"{tag}: _seg.npy exists, not overwriting (may hold your corrections)")
            continue
        shutil.copy(os.path.join(a.export_dir, "crops", f"{tag}.tif"), dst_img)
        masks = np.load(os.path.join(a.export_dir, "preds", f"{tag}.npz"))[v].astype(np.uint16)
        outlines = (masks * find_boundaries(masks, mode="inner")).astype(np.uint16)
        np.save(seg, {"outlines": outlines, "masks": masks,
                      "ismanual": np.zeros(int(masks.max()), bool),
                      "filename": os.path.basename(dst_img), "diameter": np.nan})
        print(f"{tag}: {v} ({int(masks.max())} masks)")
    man.to_csv(os.path.join(a.out_dir, "manifest.csv"), index=False)


if __name__ == "__main__":
    main()

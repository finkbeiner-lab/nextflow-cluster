#!/usr/bin/env python
"""Per-cell GFP / RFP16 from the hevo-pmsG-1 reference Cellpose-SAM montage masks (no tracking).

Masks: <analysisdir>/CellMasksMontage/<well>/*_ENCODED.tif (segmentation_montage.py cellpose run,
labels keep pre-filter numbering). Images: <analysisdir>/MontagedImages/<well>/*_MONTAGE.tif
(raw robo4_serpentine stitch, 1080 px stride). Red = Epi-RFP16, the first (pre-stim) read.
Each cell is assigned to its source tile by centroid (last-tile-wins ownership).

Output: OUT/cells.csv  (well, tp, label, tile, area, cy, cx, gfp_mean, gfp_median, rfp_mean, rfp_median)
No DB writes.
"""
import argparse
import glob
import os
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage

ADIR = "/gladstone/finkbeiner/home/aholub/GXYTMPS/GXYTMP-hevo-pmsG-1"
GFP, RFP = "Epi-GFP16", "Epi-RFP16"
STRIDE, SIDE = 1080, 3
GRID = np.array([[1, 2, 3], [6, 5, 4], [7, 8, 9]], dtype=np.uint8)


def one(path_glob):
    hits = glob.glob(path_glob)
    return hits[0] if len(hits) == 1 else None


def job(args):
    well, tp = args
    mp = one(f"{ADIR}/CellMasksMontage/{well}/*_{tp}_0-0_{well}_*_{GFP}_*ENCODED.tif")
    gp = one(f"{ADIR}/MontagedImages/{well}/*_{tp}_0-0_{well}_*_{GFP}_*MONTAGE.tif")
    rp = one(f"{ADIR}/MontagedImages/{well}/*_{tp}_0-0_{well}_*_{RFP}_*MONTAGE.tif")
    if not (mp and gp and rp):
        return pd.DataFrame(), f"missing {well} {tp}: mask={bool(mp)} gfp={bool(gp)} rfp={bool(rp)}"
    m = tifffile.imread(mp).astype(np.int32)
    g = tifffile.imread(gp).astype(np.float32)
    r = tifffile.imread(rp).astype(np.float32)
    idx = np.unique(m); idx = idx[idx > 0]
    if idx.size == 0:
        return pd.DataFrame(), f"{well} {tp}: 0 cells"
    area = ndimage.sum(np.ones_like(m, np.float32), m, idx)
    cy, cx = np.array(ndimage.center_of_mass(np.ones_like(m, np.float32), m, idx)).T
    tile = GRID[np.minimum(cy // STRIDE, SIDE - 1).astype(int), np.minimum(cx // STRIDE, SIDE - 1).astype(int)]
    d = pd.DataFrame(dict(well=well, tp=tp, label=idx, tile=tile, area=area.astype(int),
                          cy=cy.round(1), cx=cx.round(1),
                          gfp_mean=ndimage.mean(g, m, idx), gfp_median=ndimage.median(g, m, idx),
                          rfp_mean=ndimage.mean(r, m, idx), rfp_median=ndimage.median(r, m, idx)))
    return d, f"{well} {tp}: {len(d)} cells"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/gladstone/finkbeiner/home/aholub/rgedi_tile/cp_cells/hevo-pmsG-1")
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    wells = sorted(os.listdir(f"{ADIR}/CellMasksMontage"))
    jobs = [(w, f"T{i}") for w in wells for i in range(19)]
    parts = []
    with Pool(a.workers) as pool:
        for k, (d, msg) in enumerate(pool.imap_unordered(job, jobs)):
            parts.append(d)
            if k % 20 == 0 or not msg.endswith("cells") or msg.endswith(": 0 cells"):
                print(f"[{time.time() - t0:.0f}s] {k + 1}/{len(jobs)} {msg}", flush=True)
    d = pd.concat(parts, ignore_index=True)
    d.to_csv(f"{a.out}/cells.csv", index=False)
    print(f"wrote {a.out}/cells.csv: {len(d)} cells from {d.groupby(['well', 'tp']).ngroups} montages "
          f"({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Round-1 crop export for fine-tuning Cellpose-SAM on miniSOG-RGEDI fibroblasts.

For each (experiment, well, timepoint) in CROPS: build the full-res GFP16 montage from RAW
tiles (robo4_serpentine 1 2 3/6 5 4/7 8 9, 10% overlap), run Cellpose-SAM on the WHOLE
montage exactly as production does (so predictions match the pipeline), pick one 1024^2
window, and save the raw crop + the cropped predictions of several variants:

    prod  = locked RGEDI recipe (diameter=25, flow 0.6, cellprob -1.0) + MAD/size-floor filter
    d25   = same Cellpose call, NO post-filter        (does the filter drop tail fragments?)
    d50   = diameter=50  -> image scaled 0.6x         (whole cell fits in a 256 px tile?)
    d75   = diameter=75  -> image scaled 0.4x

Cellpose 4 rescales by 30/diameter internally and returns full-res masks. The best
variant becomes the starting point for manual correction in the Cellpose GUI
(make_gui_set.py) -- corrections, not drawing from scratch.

Outputs (OUT):
    crops/<tag>.tif        raw uint16 GFP crop (training image)
    preds/<tag>.npz        uint16 masks per variant
    qc/<tag>.png           GFP (gamma 0.5) + outlines of each variant, full res
    manifest.csv           crop geometry, montage stretch (lo/hi = p1/p99.5), split
    metrics.csv            per crop x variant: n_masks, median_area, fg_coverage, spill, n_frag
No DB writes.
"""
import glob
import os
import re
import time

import numpy as np
import pandas as pd
import tifffile
from cellpose import models
from PIL import Image, ImageDraw
from scipy import ndimage
from skimage.filters import gaussian, threshold_triangle
from skimage.segmentation import find_boundaries, relabel_sequential

OUT = "/gladstone/finkbeiner/steve/Austin/Robo5-Hevolution/hevo-pmsG-1_GXYTMP/cp_finetune/round1_export"
RAW = {
    "hevo-pmsG-1": "/gladstone/finkbeiner/robodata/Robo4Images/hevo-pmsG-1",
    "hevo-pmsG-2": "/gladstone/finkbeiner/robodata/Robo4Images/hevo-pmsG-2",
}
LINE = {
    "hevo-pmsG-1": {"C": "TP0357", "D": "TP0359", "E": "TP0388", "F": "TP0398"},
    "hevo-pmsG-2": {"C": "TP057", "D": "TP057", "E": "TP059", "F": "TP059"},
}
DOSE = {
    "hevo-pmsG-1": {3: "500ms", 4: "1s", 5: "5s", 6: "10s"},
    "hevo-pmsG-2": {3: "250ms", 4: "500ms", 5: "1s", 6: "2s", 7: "5s", 8: "0ms"},
}
GFP = "Epi-GFP16"

# (exp, well, tp, kind, split). kind: spread = early flat cells with dim tails;
# compact = late / high-dose retracting cells; dense = crowded parallel streams.
CROPS = [
    ("hevo-pmsG-1", "C4", "T0", "spread", "train"),
    ("hevo-pmsG-1", "D5", "T0", "spread", "train"),
    ("hevo-pmsG-1", "E3", "T2", "spread", "train"),
    ("hevo-pmsG-1", "F4", "T1", "spread", "train"),
    ("hevo-pmsG-1", "E6", "T6", "compact", "train"),
    ("hevo-pmsG-1", "C6", "T10", "compact", "train"),
    ("hevo-pmsG-1", "F5", "T14", "compact", "train"),
    ("hevo-pmsG-1", "D6", "T16", "dense", "train"),
    ("hevo-pmsG-2", "C8", "T0", "spread", "train"),
    ("hevo-pmsG-2", "D7", "T7", "compact", "train"),  # D row has no T8 (scope died mid-T8)
    ("hevo-pmsG-2", "E8", "T6", "dense", "train"),
    ("hevo-pmsG-2", "F7", "T7", "compact", "train"),
    ("hevo-pmsG-1", "D4", "T3", "spread", "heldout"),
    ("hevo-pmsG-1", "E5", "T12", "compact", "heldout"),
    ("hevo-pmsG-2", "F6", "T6", "dense", "heldout"),
]

TILE_POS = {1: (0, 0), 2: (0, 1), 3: (0, 2), 4: (1, 2), 5: (1, 1), 6: (1, 0),
            7: (2, 0), 8: (2, 1), 9: (2, 2)}
OVERLAP = 0.10
CROP = 1024
FLOW, CELLPROB, CLEAN_K = 0.6, -1.0, 2.0
VARIANTS = ["prod", "d25", "d50", "d75"]
DIAMS = {"d25": 25, "d50": 50, "d75": 75}


def tiles_for(raw, well, tp, ch):
    """Map tile number -> file for one well/timepoint/channel (warns on duplicates)."""
    files = glob.glob(f"{raw}/{well}/*_{tp}_*_{well}_*_{ch}_*.tif")
    if not files:
        files = glob.glob(f"{raw}/**/*_{tp}_*_{well}_*_{ch}_*.tif", recursive=True)
    out = {}
    for f in sorted(files):
        m = re.search(rf"_{well}_(\d+)_{re.escape(ch)}_", os.path.basename(f))
        if not m:
            continue
        t = int(m.group(1))
        if t in out:
            print(f"  WARN duplicate tile {t}: keeping {os.path.basename(out[t])}", flush=True)
            continue
        out[t] = f
    return out


def montage(raw, well, tp, ch):
    ts = tiles_for(raw, well, tp, ch)
    if len(ts) < 9:
        print(f"  only {len(ts)} tiles found for {well} {tp} {ch}", flush=True)
        return None
    t0 = tifffile.imread(ts[1])
    H = W = t0.shape[0]
    ov = int(round(H * OVERLAP)); keep = H - ov
    canvas = np.zeros((keep * 3 + ov, keep * 3 + ov), dtype=np.float32)
    for tnum, (r, c) in TILE_POS.items():
        a = tifffile.imread(ts[tnum]).astype(np.float32)
        if a.ndim == 3:
            a = a[..., 0]
        canvas[r * keep:r * keep + H, c * keep:c * keep + W] = a
    return canvas


def norm255(raw):
    lo, hi = np.percentile(raw, 1), np.percentile(raw, 99.5)
    if hi <= lo:
        return np.zeros_like(raw, dtype=np.float32), lo, hi
    return (np.clip((raw - lo) / (hi - lo), 0, 1) * 255.0).astype(np.float32), lo, hi


def post_filter(masks, raw):
    """Production MAD intensity cleanup + size floor (bin/cellpose_montage_rgedi.py)."""
    nlab = int(masks.max())
    if nlab == 0:
        return masks
    counts = np.bincount(masks.ravel(), minlength=nlab + 1); counts[0] = 0
    bg = raw[masks == 0]; bm = np.median(bg); mad = 1.4826 * np.median(np.abs(bg - bm))
    meds = ndimage.labeled_comprehension(raw, masks, np.arange(1, nlab + 1), np.median, float, 0.0)
    bright = np.zeros(nlab + 1, bool); bright[1:] = meds >= bm + CLEAN_K * mad
    inten_areas = counts[1:][bright[1:]]
    floor = max(300.0, 0.6 * float(np.median(inten_areas))) if inten_areas.size else 300.0
    keep = bright & (counts >= floor)
    return np.where(keep[masks], masks, 0)


def pick_window(raw, kind):
    """Choose the 1024^2 window: densest for 'dense', else ~60th pct of foreground."""
    ds = 4
    small = gaussian(raw[::ds, ::ds], 2, preserve_range=True)
    fg = small > threshold_triangle(small)
    ii = np.pad(fg.astype(np.float64).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    w = CROP // ds; step = 128 // ds
    cands = []
    for y in range(0, fg.shape[0] - w + 1, step):
        for x in range(0, fg.shape[1] - w + 1, step):
            s = ii[y + w, x + w] - ii[y, x + w] - ii[y + w, x] + ii[y, x]
            cands.append((s / (w * w), y * ds, x * ds))
    cands = np.array(cands)
    if kind == "dense":
        i = int(np.argmax(cands[:, 0]))
    else:
        ok = cands[cands[:, 0] >= 0.10]
        if len(ok) == 0:
            ok = cands
        target = np.percentile(ok[:, 0], 60)
        i = int(np.argmin(np.abs(cands[:, 0] - target) + (cands[:, 0] < 0.10) * 10))
    return int(cands[i, 1]), int(cands[i, 2]), float(cands[i, 0])


def crop_masks(m, y0, x0):
    c = m[y0:y0 + CROP, x0:x0 + CROP].copy()
    # drop slivers created by the crop edge (< 150 px touching the border) -- not worth labeling
    border = np.unique(np.concatenate([c[0], c[-1], c[:, 0], c[:, -1]]))
    areas = np.bincount(c.ravel())
    for l in border:
        if l and areas[l] < 150:
            c[c == l] = 0
    return relabel_sequential(c)[0].astype(np.uint16)


def metrics(m, fg):
    n = int(m.max())
    areas = np.bincount(m.ravel())[1:]
    areas = areas[areas > 0]
    med = float(np.median(areas)) if areas.size else 0.0
    cov = float(((m > 0) & fg).sum() / max(1, fg.sum()))
    spill = float(((m > 0) & ~fg).sum() / max(1, (m > 0).sum()))
    return dict(n_masks=n, median_area=med, fg_coverage=round(cov, 4), spill=round(spill, 4),
                n_frag=int((areas < 0.3 * med).sum()) if areas.size else 0)


def color_lut(n, seed=0):
    """Saturated random hues so outlines stand out on bright cells."""
    import colorsys
    rng = np.random.RandomState(seed)
    lut = np.array([colorsys.hsv_to_rgb(h, 1.0, 1.0) for h in rng.rand(n + 1)]) * 255
    lut = lut.astype(np.uint8); lut[0] = 0
    return lut


def qc_panel(raw_crop, lo, hi, preds, mets, title):
    g = np.clip((raw_crop - lo) / (hi - lo + 1e-9), 0, 1) ** 0.5
    base = np.repeat((g * 255).astype(np.uint8)[..., None], 3, 2)
    tiles = [base]
    for v in VARIANTS:
        m = preds[v]
        rgb = base.copy()
        b = ndimage.binary_dilation(find_boundaries(m, mode="inner"))
        rgb[b] = color_lut(int(m.max()))[m][b]
        tiles.append(rgb)
    labels = ["GFP16 (gamma 0.5)"] + [
        f"{v}: n={mets[v]['n_masks']} area~{mets[v]['median_area']:.0f} "
        f"cov={mets[v]['fg_coverage']:.2f} frag={mets[v]['n_frag']}" for v in VARIANTS]
    blank = np.zeros_like(base)
    grid = np.concatenate([np.concatenate(tiles[:3], 1),
                           np.concatenate(tiles[3:] + [blank], 1)], 0)
    im = Image.fromarray(grid); d = ImageDraw.Draw(im)
    for k, lab in enumerate(labels):
        x, y = (k % 3) * CROP, (k // 3) * CROP
        d.rectangle([x, y, x + CROP, y + 22], fill=(0, 0, 0))
        d.text((x + 6, y + 5), lab, fill=(255, 255, 255))
    d.text((2 * CROP + 20, CROP + 40), title, fill=(255, 255, 255))
    d.text((2 * CROP + 20, CROP + 70), "cov = fraction of GFP foreground (triangle thr)", fill=(200, 200, 200))
    d.text((2 * CROP + 20, CROP + 90), "inside masks; frag = masks < 0.3x median area", fill=(200, 200, 200))
    return im


def main():
    import torch
    for sub in ("crops", "preds", "qc"):
        os.makedirs(f"{OUT}/{sub}", exist_ok=True)
    model = models.CellposeModel(gpu=True, use_bfloat16=False)  # bf16 is ~10x slower on V100
    print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU", flush=True)
    man, mets_rows = [], []
    if os.path.exists(f"{OUT}/manifest.csv"):  # resume: keep finished crops, run only new ones
        man = pd.read_csv(f"{OUT}/manifest.csv").to_dict("records")
        mets_rows = pd.read_csv(f"{OUT}/metrics.csv").to_dict("records")
    done = {r["tag"] for r in man}
    for exp, well, tp, kind, split in CROPS:
        tag = f"{exp.replace('hevo-', '')}_{well}_{tp}"
        if tag in done and os.path.exists(f"{OUT}/preds/{tag}.npz"):
            print(f"{tag}: already exported, skip", flush=True); continue
        t_start = time.time()
        raw = montage(RAW[exp], well, tp, GFP)
        if raw is None:
            print(f"{tag}: SKIP (montage incomplete)", flush=True); continue
        img, lo, hi = norm255(raw)
        full = {}
        for v, d in DIAMS.items():
            full[v] = model.eval(img, diameter=d, flow_threshold=FLOW,
                                 cellprob_threshold=CELLPROB)[0].astype(np.int32)
        full["prod"] = post_filter(full["d25"], raw)
        y0, x0, fgfrac = pick_window(raw, kind)
        rc = raw[y0:y0 + CROP, x0:x0 + CROP]
        preds = {v: crop_masks(full[v], y0, x0) for v in VARIANTS}
        sm = gaussian(rc, 2, preserve_range=True)
        fg = sm > threshold_triangle(gaussian(raw[::4, ::4], 2, preserve_range=True))
        mets = {v: metrics(preds[v], fg) for v in VARIANTS}
        tifffile.imwrite(f"{OUT}/crops/{tag}.tif", np.clip(rc, 0, 65535).astype(np.uint16))
        np.savez_compressed(f"{OUT}/preds/{tag}.npz", **preds)
        line = LINE[exp][well[0]]; dose = DOSE[exp][int(well[1:])]
        qc_panel(rc, lo, hi, preds, mets,
                 f"{tag}  {line} {dose}  kind={kind} split={split}  win=({y0},{x0}) fg={fgfrac:.2f}"
                 ).save(f"{OUT}/qc/{tag}.png")
        man.append(dict(tag=tag, exp=exp, well=well, tp=tp, line=line, dose=dose, kind=kind,
                        split=split, y0=y0, x0=x0, size=CROP, lo=lo, hi=hi,
                        montage_h=raw.shape[0], montage_w=raw.shape[1], fg_frac=round(fgfrac, 3)))
        for v in VARIANTS:
            mets_rows.append(dict(tag=tag, kind=kind, split=split, variant=v, **mets[v]))
        print(f"{tag}: {line} {dose} {kind}/{split} win=({y0},{x0}) "
              + " ".join(f"{v}:n={mets[v]['n_masks']},cov={mets[v]['fg_coverage']:.2f}" for v in VARIANTS)
              + f"  [{time.time() - t_start:.0f}s]", flush=True)
        pd.DataFrame(man).to_csv(f"{OUT}/manifest.csv", index=False)
        pd.DataFrame(mets_rows).to_csv(f"{OUT}/metrics.csv", index=False)
    if mets_rows:
        df = pd.DataFrame(mets_rows)
        print("\nMEAN BY VARIANT x KIND\n",
              df.groupby(["variant", "kind"])[["n_masks", "median_area", "fg_coverage", "spill", "n_frag"]]
              .mean().round(3).to_string(), flush=True)
    print(f"DONE -> {OUT} ({len(man)} crops in manifest, {len(CROPS)} requested)", flush=True)


if __name__ == "__main__":
    main()

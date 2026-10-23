#!/usr/bin/env python
"""Cellpose-SAM vs the lab's threshold segmentation on miniSOG-RGEDI montages, plus the
per-cell / per-tile intensity tables needed for a tracking-free per-tile RGEDI readout.

Methods (all on the full-res Epi-GFP16 montage, robo4_serpentine 3x3, 1080 px stride):
    CP    Cellpose-SAM, the recipe the hevo-pmsG-1 reference run used
          (bin/segmentation_montage.py --segmentation_method cellpose, script defaults:
          p1/p99.5 stretch + CLAHE 0.03, diameter 70, flow 0.6, cellprob -1.5, then MAD k=3
          intensity cleanup + debris filter tiny<800 / small-round<3000 & ecc<0.8).
          hevo-pmsG-1: read from the existing CellMasksMontage/*_ENCODED.tif (the reference).
          hevo-pmsG-2: run here on the GPU; the unfiltered output is also measured as CPraw.
    OG35  lab threshold segmentation as written in segmentation_montage.py: threshold =
          int(mean + 3.5*std) of the montage, connected components, keep
          300 < area < 36000 (lab lower bound, script upper bound; the config's 300-1000 is
          nucleus-sized and would delete every 20X fibroblast).
    OG20  same, sd_scale_factor 2.0 (fibroblast-tuned sensitivity check).

Each cell is assigned to its source tile by centroid using the montage's last-tile-wins
ownership (owner row = min(y // 1080, 2), same for columns; grid 1 2 3 / 6 5 4 / 7 8 9).

Outputs (OUT/<exp>/):
    parts/<well>_<tp>_{cells,tiles,compare}.csv   per-montage pieces (resumable)
    cells.csv.gz     method, well, tp, tile, label, area, cy, cx, eccentricity, solidity,
                     gfp, rfp[, rfp_pre]  (per-cell mean intensities, raw counts)
    tiles.csv        method, well, tp, tile, n_px, mask_frac, bg_gfp, bg_rfp[, bg_rfp_pre],
                     tile_gfp, tile_rfp[, tile_rfp_pre]  (bg = median of non-mask pixels)
    compare.csv      per montage x OG variant vs CP: counts, areas, IoU matching, merges,
                     OG-only objects, brightness of CP cells OG missed
    masks/<well>_<tp>.npz   OG masks (+ CP filtered/raw for pmsG-2)
    overlays/*.jpg   GFP with CP (green) / OG (amber) outlines, crop + whole-montage views
No DB writes.
"""
import argparse
import glob
import os
import re
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
import tifffile
from PIL import Image
from scipy import ndimage
from skimage import exposure, measure
from skimage.segmentation import find_boundaries

STRIDE, TILE, SIDE = 1080, 1200, 3
GRID = [[1, 2, 3], [6, 5, 4], [7, 8, 9]]
RAW = {
    "hevo-pmsG-1": "/gladstone/finkbeiner/robodata/Robo4Images/hevo-pmsG-1",
    "hevo-pmsG-2": "/gladstone/finkbeiner/robodata/Robo4Images/hevo-pmsG-2",
}
ANALYSIS = {"hevo-pmsG-1": "/gladstone/finkbeiner/steve/Austin/Robo5-Hevolution/hevo-pmsG-1_GXYTMP"}
GFP = "Epi-GFP16"
# primary red first; pmsG-1 primary = post-stim read (-2), pre-stim kept for comparison
REDS = {
    "hevo-pmsG-1": {"rfp": "Epi-RFP16-2", "rfp_pre": "Epi-RFP16"},
    "hevo-pmsG-2": {"rfp": "Epi-RFP16"},
}
OG_VARIANTS = {"OG35": 3.5, "OG20": 2.0}
OG_AREA = (300, 36000)
CP = dict(diameter=70, flow=0.6, cellprob=-1.5, clahe=0.03, batch=16,
          clean_k=3.0, tiny=800, small=3000, ecc=0.8)
OVERLAYS = {
    "hevo-pmsG-1": [("C3", "T0"), ("E3", "T2"), ("E6", "T6"), ("F5", "T14"), ("D6", "T16")],
    "hevo-pmsG-2": [("C8", "T0"), ("C7", "T8"), ("E8", "T6"), ("F6", "T6"), ("D3", "T7")],
}

_owner_cache = {}


def owner_map(shape):
    """Tile number owning each montage pixel (last-tile-wins raster paste)."""
    if shape not in _owner_cache:
        rows = np.minimum(np.arange(shape[0]) // STRIDE, SIDE - 1)
        cols = np.minimum(np.arange(shape[1]) // STRIDE, SIDE - 1)
        _owner_cache[shape] = np.array(GRID, dtype=np.uint8)[rows[:, None], cols[None, :]]
    return _owner_cache[shape]


# ---------------------------------------------------------------- image access

def raw_tiles(exp, well, tp, ch):
    files = glob.glob(f"{RAW[exp]}/{well}/*_{tp}_*_{well}_*_{ch}_*.tif")
    out = {}
    for f in sorted(files):
        m = re.search(rf"_{tp}_.*_{well}_(\d+)_{re.escape(ch)}_", os.path.basename(f))
        if m and int(m.group(1)) not in out:
            out[int(m.group(1))] = f
    return out


def build_montage(exp, well, tp, ch):
    """Raw stitch exactly like bin/montage.py robo4_serpentine (raster paste, last wins)."""
    ts = raw_tiles(exp, well, tp, ch)
    if len(ts) < SIDE * SIDE:
        return None
    mont = np.zeros((STRIDE * (SIDE - 1) + TILE,) * 2, dtype=np.uint16)
    for i in range(SIDE):
        for j in range(SIDE):
            a = tifffile.imread(ts[GRID[i][j]])
            mont[i * STRIDE:i * STRIDE + TILE, j * STRIDE:j * STRIDE + TILE] = a
    return mont


def pmsg1_montage(well, tp, ch):
    hits = glob.glob(f"{ANALYSIS['hevo-pmsG-1']}/MontagedImages/{well}/*_{tp}_0-0_{well}_*_{ch}_*MONTAGE.tif")
    return tifffile.imread(hits[0]) if len(hits) == 1 else None


def pmsg1_cp_mask(well, tp):
    hits = glob.glob(f"{ANALYSIS['hevo-pmsG-1']}/CellMasksMontage/{well}/*_{tp}_0-0_{well}_*_{GFP}_*ENCODED.tif")
    # saved labels keep Cellpose's pre-filter numbering (gaps where cells were dropped)
    return _relabel(tifffile.imread(hits[0]).astype(np.int32)) if len(hits) == 1 else None


def timepoints(exp, well):
    tps = set()
    for f in glob.glob(f"{RAW[exp]}/{well}/*_{GFP}_*.tif"):
        m = re.search(r"_(T\d+)_", os.path.basename(f))
        if m:
            tps.add(m.group(1))
    return sorted(tps, key=lambda t: int(t[1:]))


# ---------------------------------------------------------------- segmentation

def og_segment(img, sd_scale):
    """bin/segmentation_montage.py threshold path (sd_from_mean + CC + area filter)."""
    thresh = int(np.mean(img) + np.std(img) * sd_scale)
    masks = measure.label((img > thresh).astype(np.uint8))
    areas = np.bincount(masks.ravel())
    keep = (areas > OG_AREA[0]) & (areas < OG_AREA[1])
    keep[0] = False
    return _relabel(np.where(keep[masks], masks, 0)), thresh


def _relabel(masks):
    ids = np.unique(masks)
    lut = np.zeros(int(masks.max()) + 1, dtype=np.int32)
    lut[ids[ids > 0]] = np.arange(1, (ids > 0).sum() + 1)
    return lut[masks]


def cp_filter(img, masks):
    """Copy of Segmentation._filter_masks_fast (bin/segmentation_montage.py) at CP defaults."""
    if masks.max() == 0:
        return masks
    bg = img[masks == 0]
    bg_med = np.median(bg)
    thr = bg_med + CP["clean_k"] * 1.4826 * np.median(np.abs(bg - bg_med))
    props = measure.regionprops(masks)
    present = np.array([rp.label for rp in props])
    meds = ndimage.labeled_comprehension(img, masks, present, np.median, float, 0.0)
    keep = np.zeros(int(masks.max()) + 1, dtype=bool)
    for rp, med in zip(props, meds):
        small_round = rp.area < CP["small"] and rp.eccentricity < CP["ecc"]
        if med >= thr and rp.area >= CP["tiny"] and not small_round:
            keep[rp.label] = True
    return np.where(keep[masks], masks, 0)


def cp_segment(model, img):
    raw = img.astype(np.float32)
    lo, hi = np.percentile(raw, 1), np.percentile(raw, 99.5)
    st = np.clip((raw - lo) / (hi - lo + 1e-9), 0, 1)
    st = exposure.equalize_adapthist(st, clip_limit=CP["clahe"])
    masks = model.eval((st * 255.0).astype(np.float32), batch_size=CP["batch"],
                       diameter=CP["diameter"], flow_threshold=CP["flow"],
                       cellprob_threshold=CP["cellprob"])[0].astype(np.int32)
    return masks, _relabel(cp_filter(raw, masks))


# ---------------------------------------------------------------- measurement

def measure_cells(masks, chans, owner):
    """Per-cell table: geometry + mean intensity per channel + source tile."""
    if masks.max() == 0:
        return pd.DataFrame()
    p = pd.DataFrame(measure.regionprops_table(
        masks, properties=("label", "area", "centroid", "eccentricity", "solidity")))
    p = p.rename(columns={"centroid-0": "cy", "centroid-1": "cx"})
    idx = p["label"].values
    for name, img in chans.items():
        p[name] = ndimage.mean(img.astype(np.float32), masks, idx)
    cy = np.clip(p["cy"].round().astype(int), 0, masks.shape[0] - 1)
    cx = np.clip(p["cx"].round().astype(int), 0, masks.shape[1] - 1)
    p.insert(1, "tile", owner[cy, cx])
    return p


def measure_tiles(masks, chans, owner):
    rows = []
    fg = masks > 0
    for t in range(1, 10):
        sel = owner == t
        r = dict(tile=t, n_px=int(sel.sum()), mask_frac=float(fg[sel].mean()))
        bgsel = sel & ~fg
        for name, img in chans.items():
            r[f"bg_{name}"] = float(np.median(img[bgsel][::4])) if bgsel.any() else np.nan
            r[f"tile_{name}"] = float(img[sel].mean())
        rows.append(r)
    return pd.DataFrame(rows)


def compare(cp, og, gfp):
    """Instance agreement of an OG mask against the CP mask."""
    n_cp, n_og = int(cp.max()), int(og.max())
    a_cp = np.bincount(cp.ravel(), minlength=n_cp + 1)
    a_og = np.bincount(og.ravel(), minlength=n_og + 1)
    both = (cp > 0) & (og > 0)
    pair = cp[both].astype(np.int64) * (n_og + 1) + og[both]
    u, inter = np.unique(pair, return_counts=True)
    pc, po = u // (n_og + 1), u % (n_og + 1)
    iou = inter / (a_cp[pc] + a_og[po] - inter)
    # CP cell is "inside" an OG object when >=50% of its area falls in that object
    inside = inter >= 0.5 * a_cp[pc]
    cp_in_og = np.zeros(n_cp + 1, bool); cp_in_og[pc[inside]] = True
    cp_matched = np.zeros(n_cp + 1, bool); cp_matched[pc[iou >= 0.5]] = True
    merges = np.bincount(po[inside], minlength=n_og + 1)
    cp_fg_in_og = np.bincount(po, weights=inter, minlength=n_og + 1)
    og_only = (cp_fg_in_og[1:] < 0.2 * a_og[1:]).sum() if n_og else 0
    cp_gfp = ndimage.mean(gfp.astype(np.float32), cp, np.arange(1, n_cp + 1)) if n_cp else np.array([])
    hit, miss = cp_gfp[cp_in_og[1:]], cp_gfp[~cp_in_og[1:]]
    fcp, fog = cp > 0, og > 0
    return dict(
        n_cp=n_cp, n_og=n_og,
        med_area_cp=float(np.median(a_cp[1:])) if n_cp else np.nan,
        med_area_og=float(np.median(a_og[1:])) if n_og else np.nan,
        fg_cp=float(fcp.mean()), fg_og=float(fog.mean()),
        fg_jaccard=float((fcp & fog).sum() / max(1, (fcp | fog).sum())),
        cp_detected_frac=float(cp_in_og[1:].mean()) if n_cp else np.nan,
        cp_matched_iou50=float(cp_matched[1:].mean()) if n_cp else np.nan,
        og_merged_frac=float((merges[1:] >= 2).mean()) if n_og else np.nan,
        og_cells_per_obj=float(merges[1:][merges[1:] > 0].mean()) if (merges[1:] > 0).any() else np.nan,
        og_only_frac=float(og_only / n_og) if n_og else np.nan,
        gfp_cp_detected=float(np.median(hit)) if hit.size else np.nan,
        gfp_cp_missed=float(np.median(miss)) if miss.size else np.nan,
    )


# ---------------------------------------------------------------- overlays

def gray8(img):
    lo, hi = np.percentile(img, 1), np.percentile(img, 99.5)
    return (np.clip((img.astype(np.float32) - lo) / (hi - lo + 1e-9), 0, 1) ** 0.6 * 255).astype(np.uint8)


def outline(base, masks, color):
    out = base.copy()
    out[find_boundaries(masks, mode="inner")] = color
    return out


def write_overlays(path_stub, gfp, method_masks):
    colors = {"CP": (40, 255, 40), "CPraw": (0, 200, 255), "OG35": (255, 176, 0), "OG20": (255, 90, 200)}
    g = gray8(gfp)
    base = np.repeat(g[..., None], 3, 2)
    y0 = x0 = (gfp.shape[0] - 1024) // 2
    crops, fulls = [base[y0:y0 + 1024, x0:x0 + 1024]], [base[::3, ::3]]
    for name, m in method_masks.items():
        c = np.array(colors.get(name, (0, 200, 255)), np.uint8)
        crops.append(outline(base[y0:y0 + 1024, x0:x0 + 1024], m[y0:y0 + 1024, x0:x0 + 1024], c))
        tint = base.copy().astype(np.float32)
        tint[m > 0] = 0.55 * tint[m > 0] + 0.45 * c
        fulls.append(outline(tint.astype(np.uint8)[::3, ::3], m[::3, ::3], c))
    Image.fromarray(np.concatenate(crops, 1)).save(f"{path_stub}_crop.jpg", quality=88)
    Image.fromarray(np.concatenate(fulls, 1)).save(f"{path_stub}_full.jpg", quality=85)


# ---------------------------------------------------------------- per montage

def process(exp, well, tp, out, model=None):
    stub = f"{out}/parts/{well}_{tp}"
    if os.path.exists(f"{stub}_compare.csv"):
        return "skip"
    t0 = time.time()
    get = (lambda ch: pmsg1_montage(well, tp, ch)) if exp == "hevo-pmsG-1" else \
          (lambda ch: build_montage(exp, well, tp, ch))
    gfp = get(GFP)
    reds = {k: get(ch) for k, ch in REDS[exp].items()}
    if gfp is None or any(v is None for v in reds.values()):
        return f"missing images {well} {tp}"
    chans = {"gfp": gfp, **reds}
    owner = owner_map(gfp.shape)

    save = {}
    if exp == "hevo-pmsG-1":
        cp = pmsg1_cp_mask(well, tp)
        if cp is None:
            return f"missing CP mask {well} {tp}"
    else:
        cp_raw, cp = cp_segment(model, gfp)
        save.update(cp=cp.astype(np.uint16), cp_raw=cp_raw.astype(np.uint16))
    method_masks = {"CP": cp}
    if exp != "hevo-pmsG-1":
        # unfiltered Cellpose: shows what the MAD/debris cleanup removes
        method_masks["CPraw"] = _relabel(cp_raw)
    thr = {}
    for name, sd in OG_VARIANTS.items():
        method_masks[name], thr[name] = og_segment(gfp, sd)
        save[name.lower()] = method_masks[name].astype(np.uint16)
    np.savez_compressed(f"{out}/masks/{well}_{tp}.npz", **save)

    cells, tiles, comp = [], [], []
    for name, m in method_masks.items():
        c = measure_cells(m, chans, owner)
        if not c.empty:
            c.insert(0, "method", name)
            cells.append(c)
        t = measure_tiles(m, chans, owner)
        t.insert(0, "method", name)
        tiles.append(t)
        if name != "CP":
            comp.append(dict(og=name, threshold=thr.get(name, np.nan), **compare(cp, m, gfp)))
    for df, kind in ((cells, "cells"), (tiles, "tiles")):
        d = pd.concat(df, ignore_index=True) if df else pd.DataFrame()
        d.insert(0, "tp", tp); d.insert(0, "well", well)
        d.to_csv(f"{stub}_{kind}.csv", index=False)
    if (well, tp) in OVERLAYS[exp]:
        write_overlays(f"{out}/overlays/{exp}_{well}_{tp}", gfp, method_masks)
    comp = pd.DataFrame(comp); comp.insert(0, "tp", tp); comp.insert(0, "well", well)
    comp.to_csv(f"{stub}_compare.csv", index=False)  # written last = completion marker
    return f"{well} {tp}: CP {int(cp.max())} OG35 {int(method_masks['OG35'].max())} " \
           f"OG20 {int(method_masks['OG20'].max())} ({time.time() - t0:.0f}s)"


def _worker(args):
    try:
        return process(*args)
    except Exception as e:  # keep the pool alive; report and move on
        return f"ERROR {args[1]} {args[2]}: {e!r}"


def collect(out):
    for kind in ("cells", "tiles", "compare"):
        parts = sorted(glob.glob(f"{out}/parts/*_{kind}.csv"))
        d = pd.concat([pd.read_csv(p) for p in parts if os.path.getsize(p) > 1], ignore_index=True)
        d.to_csv(f"{out}/{kind}.csv.gz" if kind == "cells" else f"{out}/{kind}.csv", index=False)
        print(f"{kind}: {len(d)} rows from {len(parts)} montages", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True, choices=list(RAW))
    ap.add_argument("--out", default="/gladstone/finkbeiner/home/aholub/rgedi_tile")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--wells", default="all")
    ap.add_argument("--tps", default="all")
    a = ap.parse_args()
    out = f"{a.out}/{a.exp}"
    for d in ("parts", "masks", "overlays"):
        os.makedirs(f"{out}/{d}", exist_ok=True)
    wells = sorted(os.listdir(RAW[a.exp])) if a.wells == "all" else a.wells.split(",")
    wells = [w for w in wells if re.fullmatch(r"[A-P]\d+", w)]
    jobs = [(a.exp, w, tp, out) for w in wells for tp in timepoints(a.exp, w)
            if a.tps == "all" or tp in a.tps.split(",")]
    print(f"{a.exp}: {len(jobs)} montages, {len(wells)} wells", flush=True)

    if a.exp == "hevo-pmsG-1":
        with Pool(a.workers) as pool:
            for msg in pool.imap_unordered(_worker, jobs):
                print(msg, flush=True)
    else:
        from cellpose import models
        model = models.CellposeModel(gpu=True, use_bfloat16=False)
        for j in jobs:
            try:
                print(process(*j, model=model), flush=True)
            except Exception as e:
                print(f"ERROR {j[1]} {j[2]}: {e!r}", flush=True)
    collect(out)


if __name__ == "__main__":
    main()

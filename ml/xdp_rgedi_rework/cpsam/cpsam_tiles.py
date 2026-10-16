#!/usr/bin/env python3
"""Neurite-recipe somas on RAW tiles, measured on Galaxy BACKGROUND-CORRECTED tiles.

For each (well, timepoint) in a manifest:

1. **Masks** come from the neurite pipeline's soma recipe, run per raw FITC tile
   (``percell_integrate.segment_somas``: Cellpose-SAM diameter 25 / flow 0.6 /
   cellprob -1.0, then the MAD intensity clean k=2.0 and the size floor
   ``max(300, 0.6*median area)``). If the neurite run already cached that
   (well, tp) (XDP10: T7), its ``soma_labels`` are reused unchanged.
2. **Intensities** are read from the matching Galaxy ``BackgroundCorrected``
   tiles (``..._<site>_<FITC|RFP>_0_1_0_BGs.tif``). Raw and BG tiles share one
   pixel grid (BG = raw - smooth background), so each tile mask applies 1:1.

Labels are made unique per well/timepoint with the same per-tile offset and the
same 4x4 row-major placement as ``montage_percell.build_montages``, so the saved
8192x8192 label montage has the layout of the neurite cache.

Outputs under ``--out-dir``:

* ``masks/<well>_<tp>.npz``: ``soma_labels`` (uint16/uint32) + ``meta`` (JSON).
* ``percell/<well>_<tp>.csv.gz``: one row per cell (see ``PERCELL_COLS``).
* ``qc/timing_<task>.csv``; with ``--compare-cache``, ``qc/cache_compare_<task>.csv``;
  with ``--overlay-tps``, ``qc/overlays/<well>_<tp>_{full,zoom}.jpg``.

Re-running skips (well, tp) pairs whose mask and percell files already exist.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage

GRID = 4
TILE_SITES = list(range(1, GRID * GRID + 1))
# The neurite run's locked soma recipe (XDP10 cache meta + provenance.json).
RECIPE = dict(diameter=25, flow=0.6, cellprob=-1.0, clean_k=2.0)
PERCELL_COLS = [
    "experiment", "well", "timepoint", "tp", "cell_line", "condition", "label", "site",
    "area", "cy", "cx",
    "fitc_mean", "fitc_median", "fitc_min", "fitc_max", "fitc_std",
    "rfp_mean", "rfp_median", "rfp_min", "rfp_max", "rfp_std",
    "raw_fitc_mean", "fitc_zero_frac",
]


# ----------------------------- tile discovery -------------------------------
def site_paths(folder: str, well: str, tp: str, channel: str,
               suffix: str) -> Dict[int, str]:
    """Map site -> tile path for one well/timepoint/channel.

    Args:
        folder: Directory holding the well's tiles.
        well: Well id, e.g. ``"D08"``.
        tp: Timepoint token, e.g. ``"T7"``.
        channel: ``"FITC"`` or ``"RFP"``.
        suffix: ``""`` for raw tiles, ``"_BGs"`` for Galaxy BG-corrected tiles.

    Returns:
        ``{site: path}``.
    """
    pat = re.compile(rf"_{tp}_0-1_{well}_(\d+)_{channel}_0_1_0{suffix}\.tif$")
    out: Dict[int, str] = {}
    for name in os.listdir(folder):
        m = pat.search(name)
        if m:
            out[int(m.group(1))] = os.path.join(folder, name)
    return out


def read_tile(path: str) -> np.ndarray:
    """Read one tile as float32 (drop any trailing channel axis)."""
    a = tifffile.imread(path).astype(np.float32)
    return a[..., 0] if a.ndim == 3 else a


def tile_slice(site: int, th: int, tw: int) -> Tuple[slice, slice]:
    """Montage slice of a site in the 4x4 row-major grid (montage_percell.place)."""
    r, c = (site - 1) // GRID, (site - 1) % GRID
    return slice(r * th, (r + 1) * th), slice(c * tw, (c + 1) * tw)


# ----------------------------- masks ----------------------------------------
def segment_well(model, raw_tiles: Dict[int, np.ndarray]) -> np.ndarray:
    """Neurite soma recipe per raw tile, labels offset and placed as in build_montages.

    Args:
        model: Cellpose-SAM model.
        raw_tiles: ``{site: raw FITC tile}`` for sites 1..16.

    Returns:
        Soma label montage (GRID*th, GRID*tw), int32, labels unique per well.
    """
    from percell_integrate import segment_somas  # needs torch/cellpose (GPU path only)
    th, tw = raw_tiles[1].shape
    soma = np.zeros((GRID * th, GRID * tw), np.int32)
    offset = 0
    for s in TILE_SITES:
        labels = segment_somas(model, raw_tiles[s], RECIPE["diameter"], RECIPE["flow"],
                               RECIPE["cellprob"], RECIPE["clean_k"])
        if labels.max() > 0:
            lab = labels.copy()
            lab[labels > 0] += offset
            offset = int(lab.max())
            ys, xs = tile_slice(s, th, tw)
            soma[ys, xs] = lab
    return soma


def load_cache(cache_path: str) -> np.ndarray:
    """Load neurite-cache soma labels, refusing a cache built with another recipe."""
    z = np.load(cache_path)
    meta = json.loads(str(z["meta"]))
    for k, v in RECIPE.items():
        if float(meta.get(k, np.nan)) != float(v):
            raise ValueError(f"{cache_path}: cache {k}={meta.get(k)} != recipe {v}")
    return z["soma_labels"].astype(np.int32)


def match_stats(ref: np.ndarray, new: np.ndarray, iou_min: float = 0.5) -> Dict[str, float]:
    """Object agreement between two label images (ref = neurite cache).

    Args:
        ref: Reference label image.
        new: Freshly computed label image.
        iou_min: IoU needed to call two objects the same cell.

    Returns:
        Counts and the fraction of ref / new objects matched at ``iou_min``.
    """
    fg = (ref > 0) & (new > 0)
    pairs, inter = np.unique(np.stack([ref[fg], new[fg]]), axis=1, return_counts=True)
    a_ref = np.bincount(ref.ravel())
    a_new = np.bincount(new.ravel())
    iou = inter / (a_ref[pairs[0]] + a_new[pairs[1]] - inter)
    good = iou >= iou_min
    n_ref = int((a_ref[1:] > 0).sum())
    n_new = int((a_new[1:] > 0).sum())
    return dict(n_cache=n_ref, n_fresh=n_new,
                frac_cache_matched=len(np.unique(pairs[0][good])) / max(n_ref, 1),
                frac_fresh_matched=len(np.unique(pairs[1][good])) / max(n_new, 1),
                median_iou_matched=float(np.median(iou[good])) if good.any() else float("nan"))


# ----------------------------- measurement ----------------------------------
def _label_stats(img: np.ndarray, lab: np.ndarray, idx: np.ndarray) -> Dict[str, np.ndarray]:
    """Mean/median/min/max/std of ``img`` per label in ``idx``."""
    return dict(
        mean=np.asarray(ndimage.mean(img, lab, idx)),
        median=np.asarray(ndimage.median(img, lab, idx)),
        min=np.asarray(ndimage.minimum(img, lab, idx)),
        max=np.asarray(ndimage.maximum(img, lab, idx)),
        std=np.asarray(ndimage.standard_deviation(img, lab, idx)),
    )


def measure_tile(lab: np.ndarray, bg_fitc: np.ndarray, bg_rfp: np.ndarray,
                 raw_fitc: np.ndarray, y0: int, x0: int) -> pd.DataFrame:
    """Per-cell stats of one tile's labels on the BG-corrected FITC/RFP tiles.

    Args:
        lab: Tile label image (global labels; 0 = background).
        bg_fitc: Galaxy BG-corrected FITC tile.
        bg_rfp: Galaxy BG-corrected RFP tile.
        raw_fitc: Raw FITC tile (QC column only).
        y0: Tile row offset in the montage.
        x0: Tile column offset in the montage.

    Returns:
        One row per label present in the tile.
    """
    # Compact the tile's global labels to 1..n (ndimage stats scale with max label).
    ids, inv = np.unique(lab, return_inverse=True)
    compact = inv.reshape(lab.shape).astype(np.int32)
    if ids[0] != 0:
        compact += 1
    labels = ids[ids > 0]
    if labels.size == 0:
        return pd.DataFrame()
    lab = compact
    idx = np.arange(1, labels.size + 1)
    area = np.asarray(ndimage.sum_labels(np.ones_like(lab, np.float32), lab, idx))
    com = np.asarray(ndimage.center_of_mass(np.ones_like(lab, np.float32), lab, idx))
    f = _label_stats(bg_fitc, lab, idx)
    r = _label_stats(bg_rfp, lab, idx)
    zero = np.asarray(ndimage.sum_labels((bg_fitc == 0).astype(np.float32), lab, idx))
    df = pd.DataFrame(dict(label=labels, area=area.astype(int),
                           cy=com[:, 0] + y0, cx=com[:, 1] + x0))
    for k, v in f.items():
        df[f"fitc_{k}"] = v
    for k, v in r.items():
        df[f"rfp_{k}"] = v
    df["raw_fitc_mean"] = np.asarray(ndimage.mean(raw_fitc, lab, idx))
    df["fitc_zero_frac"] = zero / area
    return df


# ----------------------------- overlays -------------------------------------
def write_overlay(path_stem: str, bg_fitc_montage: np.ndarray, soma: np.ndarray) -> None:
    """Soma outlines (green) on the BG-corrected FITC montage: 4x-down full + 1024 crop."""
    from PIL import Image
    from skimage.segmentation import find_boundaries
    lo, hi = np.percentile(bg_fitc_montage, [1, 99.5])
    gray = (np.clip((bg_fitc_montage - lo) / max(hi - lo, 1e-9), 0, 1) * 255).astype(np.uint8)
    edge = find_boundaries(soma, mode="inner")
    rgb = np.stack([gray] * 3, -1)
    rgb[edge] = (0, 255, 0)
    H, W = gray.shape
    small = rgb.reshape(H // 4, 4, W // 4, 4, 3).max(axis=(1, 3))
    Image.fromarray(small).save(path_stem + "_full.jpg", quality=90)
    cy, cx, h = H // 2, W // 2, min(512, H // 4)  # centre crop spans the 4-tile corner
    Image.fromarray(rgb[cy - h:cy + h, cx - h:cx + h]).save(path_stem + "_zoom.jpg", quality=92)


# ----------------------------- driver ---------------------------------------
def atomic_save_npz(path: str, **arrays) -> None:
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def process(row: Dict[str, str], args: argparse.Namespace, model_box: dict,
            cmp_rows: List[dict]) -> Optional[dict]:
    """Segment (or reuse) and measure one (well, timepoint). Returns a timing row."""
    exp, well, tp = row["experiment"], row["well"], row["timepoint"]
    mask_path = os.path.join(args.out_dir, "masks", f"{well}_{tp}.npz")
    pc_path = os.path.join(args.out_dir, "percell", f"{well}_{tp}.csv.gz")
    if os.path.exists(mask_path) and os.path.exists(pc_path):
        return None
    t0 = time.time()
    raw_f = site_paths(row["raw_dir"], well, tp, "FITC", "")
    bg_f = site_paths(row["bg_dir"], well, tp, "FITC", "_BGs")
    bg_r = site_paths(row["bg_dir"], well, tp, "RFP", "_BGs")
    for name, d in (("raw FITC", raw_f), ("BG FITC", bg_f), ("BG RFP", bg_r)):
        if sorted(d) != TILE_SITES:
            raise FileNotFoundError(f"{well} {tp}: {name} sites {sorted(d)} != 1..16")
    raw_tiles = {s: read_tile(raw_f[s]) for s in TILE_SITES}
    th, tw = raw_tiles[1].shape

    cache_path = os.path.join(args.cache_dir, f"{exp}_{well}_{tp}.npz") if args.cache_dir else ""
    use_cache = bool(cache_path) and os.path.exists(cache_path)
    fresh = None
    if not use_cache or args.compare_cache:
        if "cp" not in model_box:
            from cellpose import models
            # use_bfloat16=False: bf16 is ~10x slower on the galaxy V100s (cellpose 4.x default True)
            model_box["cp"] = models.CellposeModel(gpu=True, pretrained_model="cpsam",
                                                   use_bfloat16=False)
        fresh = segment_well(model_box["cp"], raw_tiles)
    t_seg = time.time() - t0
    if use_cache:
        soma = load_cache(cache_path)
        if soma.shape != (GRID * th, GRID * tw):
            raise ValueError(f"{cache_path}: shape {soma.shape} != montage {(GRID * th, GRID * tw)}")
        if fresh is not None:
            cmp_rows.append(dict(well=well, timepoint=tp, **match_stats(soma, fresh)))
        source = "neurite_cache"
    else:
        soma = fresh
        source = "fresh"

    parts = []
    bg_fitc_montage = np.zeros(soma.shape, np.float32) if tp in args.overlay_tps else None
    for s in TILE_SITES:
        ys, xs = tile_slice(s, th, tw)
        bgf, bgr = read_tile(bg_f[s]), read_tile(bg_r[s])
        if bgf.shape != (th, tw) or bgr.shape != (th, tw):
            raise ValueError(f"{well} {tp} site {s}: BG tile shape != raw {th}x{tw}")
        df = measure_tile(soma[ys, xs], bgf, bgr, raw_tiles[s], ys.start, xs.start)
        if len(df):
            df.insert(0, "site", s)
            parts.append(df)
        if bg_fitc_montage is not None:
            bg_fitc_montage[ys, xs] = bgf
    pc = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=PERCELL_COLS)
    pc["experiment"], pc["well"], pc["timepoint"] = exp, well, tp
    pc["tp"] = int(tp.lstrip("T"))
    pc["cell_line"], pc["condition"] = row.get("cell_line", ""), row.get("condition", "")
    pc = pc[PERCELL_COLS]

    meta = dict(experiment=exp, well=well, timepoint=tp, mask_source=source, grid=GRID,
                cache_path=cache_path if use_cache else "", **RECIPE,
                raw_dir=row["raw_dir"], bg_dir=row["bg_dir"], n_cells=int(len(pc)))
    dtype = np.uint16 if soma.max() < 65536 else np.uint32
    atomic_save_npz(mask_path, soma_labels=soma.astype(dtype), meta=np.array(json.dumps(meta)))
    tmp = pc_path + ".tmp.csv.gz"
    pc.to_csv(tmp, index=False, compression="gzip")
    os.replace(tmp, pc_path)
    if bg_fitc_montage is not None:
        write_overlay(os.path.join(args.out_dir, "qc", "overlays", f"{well}_{tp}"),
                      bg_fitc_montage, soma)
    return dict(well=well, timepoint=tp, source=source, n_cells=len(pc),
                seg_s=round(t_seg, 1), total_s=round(time.time() - t0, 1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--manifest", required=True,
                    help="CSV: experiment,well,timepoint,cell_line,condition,raw_dir,bg_dir")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--cache-dir", default="", help="neurite-run cache dir (reused when present)")
    ap.add_argument("--task-index", type=int, default=0, help="this task's share: rows[i::n]")
    ap.add_argument("--n-tasks", type=int, default=1)
    ap.add_argument("--compare-cache", action="store_true",
                    help="also segment cached (well,tp) fresh and record agreement (pilot)")
    ap.add_argument("--overlay-tps", default="", help="comma list of timepoints to overlay, e.g. T1,T7")
    ap.add_argument("--preflight", action="store_true",
                    help="only count raw/BG tiles for every manifest row -> qc/preflight.csv")
    args = ap.parse_args()
    args.overlay_tps = {t for t in args.overlay_tps.split(",") if t}
    for sub in ("masks", "percell", "qc", "qc/overlays"):
        os.makedirs(os.path.join(args.out_dir, sub), exist_ok=True)

    if args.preflight:
        pf = []
        for row in csv.DictReader(open(args.manifest)):
            w, tp = row["well"], row["timepoint"]
            n = {}
            for key, folder, ch, suf in (("raw_fitc", "raw_dir", "FITC", ""),
                                         ("bg_fitc", "bg_dir", "FITC", "_BGs"),
                                         ("bg_rfp", "bg_dir", "RFP", "_BGs")):
                n[key] = (len(site_paths(row[folder], w, tp, ch, suf))
                          if os.path.isdir(row[folder]) else 0)
            pf.append(dict(well=w, timepoint=tp, **n, ok=all(v == 16 for v in n.values())))
        pf = pd.DataFrame(pf)
        pf.to_csv(os.path.join(args.out_dir, "qc", "preflight.csv"), index=False)
        print(f"preflight: {int(pf.ok.sum())}/{len(pf)} (well,tp) items have all 16 raw FITC + "
              f"BG FITC + BG RFP tiles")
        if not pf.ok.all():
            print(pf[~pf.ok].to_string(index=False))
        return

    rows = list(csv.DictReader(open(args.manifest)))[args.task_index::args.n_tasks]
    print(f"task {args.task_index}/{args.n_tasks}: {len(rows)} (well,tp) items", flush=True)
    timing, cmp_rows, failed = [], [], []
    model_box: dict = {}
    for i, row in enumerate(rows, 1):
        tag = f"{row['well']} {row['timepoint']}"
        try:
            t = process(row, args, model_box, cmp_rows)
        except Exception as e:  # one bad montage must not stop the task
            failed.append(dict(well=row["well"], timepoint=row["timepoint"], error=repr(e)))
            print(f"[{i}/{len(rows)}] {tag} FAILED: {e!r}", flush=True)
            continue
        if t is None:
            print(f"[{i}/{len(rows)}] {tag} done already, skipped", flush=True)
            continue
        timing.append(t)
        print(f"[{i}/{len(rows)}] {tag} {t['source']} cells={t['n_cells']} "
              f"seg={t['seg_s']}s total={t['total_s']}s", flush=True)
    qc = os.path.join(args.out_dir, "qc")
    suffix = f"{args.task_index:03d}"
    if timing:
        pd.DataFrame(timing).to_csv(os.path.join(qc, f"timing_{suffix}.csv"), index=False)
    if cmp_rows:
        pd.DataFrame(cmp_rows).to_csv(os.path.join(qc, f"cache_compare_{suffix}.csv"), index=False)
    if failed:
        pd.DataFrame(failed).to_csv(os.path.join(qc, f"failed_{suffix}.csv"), index=False)
        raise SystemExit(f"{len(failed)} item(s) failed; see qc/failed_{suffix}.csv")
    print("task complete", flush=True)


if __name__ == "__main__":
    main()

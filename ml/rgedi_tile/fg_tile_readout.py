#!/usr/bin/env python
"""Per-tile foreground readout for miniSOG-RGEDI plates (no cell segmentation, no tracking).

For every raw GFP tile:
    1. empty-dish map P = per-pixel 1st percentile of raw GFP tiles (T0/T6/T12/T18, all wells
       of --p_source), computed at 2x downsample and upsampled back to 1200^2
    2. per-tile background scale s = lowest mode of smooth(GFP, 1)/P (fallback: median s of
       the other tiles of that well+timepoint, then of that timepoint plate-wide)
    3. background = gaussian(GFP, SMOOTH) <= LEVEL * s * P   (+ k * noise, k = 0 here)
    4. foreground = NOT background, minus 4-connected specks < MIN_SPECK px (no hole filling)
Settings were tuned by eye on the tuning page (ml/rgedi_tile/make_tuning_page.py).

Outputs (OUT/<exp>/):
    tiles.csv   one row per well x timepoint x tile: background fit, fg fraction, mean/median of
                GFP and each red over foreground pixels, median of each channel over background
    qc/<well>_<tp>.jpg   montage (1/4 scale) of GFP with background tinted blue
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
from skimage import measure
from skimage.transform import resize

RAW = {
    "hevo-pmsG-1": "/gladstone/finkbeiner/robodata/Robo4Images/hevo-pmsG-1",
    "hevo-pmsG-2": "/gladstone/finkbeiner/robodata/Robo4Images/hevo-pmsG-2",
}
GFP = "Epi-GFP16"
# the first (pre-stim) RFP16 read of each timepoint; pmsG-1's post-stim -2 read is skipped because
# death changes don't show up within seconds of the stim, and pmsG-2 only has this read
REDS = {
    "hevo-pmsG-1": {"rfp": "Epi-RFP16"},
    "hevo-pmsG-2": {"rfp": "Epi-RFP16"},
}
LEVEL, K, SMOOTH, MIN_SPECK = 1.5, 0.0, 6.0, 1000
TILE, STRIDE = 1200, 1080
GRID = [[1, 2, 3], [6, 5, 4], [7, 8, 9]]

P = None  # set in each worker by _init


def tile_path(exp, well, tp, n, ch):
    hits = glob.glob(f"{RAW[exp]}/{well}/*_{tp}_*_{well}_{n}_{ch}_*.tif")
    return hits[0] if len(hits) == 1 else None


def build_p(exp):
    files = []
    for w in sorted(os.listdir(RAW[exp])):
        for tp in ("T0", "T6", "T12", "T18"):
            files += glob.glob(f"{RAW[exp]}/{w}/*_{tp}_0-0_{w}_*_{GFP}_*.tif")
    st = np.empty((len(files), TILE // 2, TILE // 2), np.uint16)
    for i, f in enumerate(files):
        st[i] = tifffile.imread(f).reshape(TILE // 2, 2, TILE // 2, 2).mean((1, 3))
    k = max(1, int(round(0.01 * len(files))))
    p2 = np.partition(st, k, axis=0)[k].astype(np.float32)
    print(f"empty-dish map from {len(files)} tiles", flush=True)
    return resize(p2, (TILE, TILE), order=1, preserve_range=True).astype(np.float32)


def bg_fit(img):
    """Background scale s (background = s * P) and noise, from the lowest mode of img/P."""
    r = ndimage.gaussian_filter(img, 1) / P
    h, e = np.histogram(r, bins=400, range=(0.5, 4.0))
    hs = ndimage.gaussian_filter1d(h.astype(float), 2)
    c = 0.5 * (e[1:] + e[:-1])
    for i in range(1, len(hs) - 1):
        if hs[i] >= hs[i - 1] and hs[i] >= hs[i + 1] and h[max(0, i - 6):i + 7].sum() > 0.02 * r.size:
            s = c[i]
            left = (img - s * P)[(r < s) & (r > s - 0.5)]
            return float(s), (float(np.sqrt(np.mean(left ** 2))) if left.size > 500 else np.nan)
    return np.nan, np.nan


def foreground(img, s, sigma):
    sm = ndimage.gaussian_filter(img, SMOOTH, mode="nearest", truncate=3.0)
    fg = sm > LEVEL * s * P + (K * sigma if K else 0.0)
    lab = measure.label(fg, connectivity=1)
    keep = np.bincount(lab.ravel()) >= MIN_SPECK
    keep[0] = False
    return keep[lab]


def _init(p):
    global P
    P = p


def fit_job(args):
    exp, well, tp, n = args
    f = tile_path(exp, well, tp, n, GFP)
    if f is None:
        return (well, tp, n, np.nan, np.nan, False)
    s, sig = bg_fit(tifffile.imread(f).astype(np.float32))
    return (well, tp, n, s, sig, True)


def measure_job(args):
    exp, well, tp, fits, qc_dir = args  # fits: {tile: (s, sigma, source)}
    rows, mont_g, mont_m = [], np.zeros((840, 840), np.float32), np.zeros((840, 840), bool)
    for i in range(3):
        for j in range(3):
            n = GRID[i][j]
            paths = {"gfp": tile_path(exp, well, tp, n, GFP),
                     **{k: tile_path(exp, well, tp, n, ch) for k, ch in REDS[exp].items()}}
            if any(p is None for p in paths.values()):
                continue
            img = {k: tifffile.imread(p).astype(np.float32) for k, p in paths.items()}
            s, sig, src = fits[n]
            fg = foreground(img["gfp"], s, sig)
            r = dict(well=well, tp=tp, tile=n, s=s, sigma=sig, s_source=src,
                     fg_frac=float(fg.mean()), n_fg_px=int(fg.sum()))
            bg = ~fg
            for k, a in img.items():
                v = a[fg]
                r[f"{k}_mean"] = float(v.mean()) if v.size else np.nan
                r[f"{k}_median"] = float(np.median(v)) if v.size else np.nan
                r[f"bg_{k}_median"] = float(np.median(a[bg])) if bg.any() else np.nan
            rows.append(r)
            q = STRIDE // 4
            mont_g[i * q:i * q + TILE // 4, j * q:j * q + TILE // 4] = img["gfp"][::4, ::4]
            mont_m[i * q:i * q + TILE // 4, j * q:j * q + TILE // 4] = fg[::4, ::4]
    if rows:
        lo, hi = np.percentile(mont_g, 0.5), np.percentile(mont_g, 99.7)
        g = (np.clip((mont_g - lo) / (hi - lo + 1e-9), 0, 1) ** 0.6 * 255)
        rgb = np.repeat(g[..., None], 3, 2)
        rgb[~mont_m] = 0.55 * rgb[~mont_m] + 0.45 * np.array([40, 90, 255])
        Image.fromarray(rgb.astype(np.uint8)).save(f"{qc_dir}/{well}_{tp}.jpg", quality=85)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True, choices=list(RAW))
    ap.add_argument("--p_source", default="hevo-pmsG-1")
    ap.add_argument("--out", default="/gladstone/finkbeiner/home/aholub/rgedi_tile/fgbg")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--wells", default="all")
    ap.add_argument("--tps", default="all")
    a = ap.parse_args()
    t0 = time.time()
    out = f"{a.out}/{a.exp}"
    os.makedirs(f"{out}/qc", exist_ok=True)
    p = build_p(a.p_source)
    np.save(f"{out}/empty_dish_P.npy", p)

    wells = sorted(w for w in os.listdir(RAW[a.exp]) if re.fullmatch(r"[A-P]\d+", w))
    if a.wells != "all":
        wells = [w for w in wells if w in a.wells.split(",")]
    wt = []
    for w in wells:
        tps = {re.search(r"_(T\d+)_", os.path.basename(f)).group(1)
               for f in glob.glob(f"{RAW[a.exp]}/{w}/*_{GFP}_*.tif")}
        wt += [(w, tp) for tp in sorted(tps, key=lambda t: int(t[1:]))
               if a.tps == "all" or tp in a.tps.split(",")]
    print(f"{a.exp}: {len(wt)} well-timepoints", flush=True)

    with Pool(a.workers, initializer=_init, initargs=(p,)) as pool:
        fits = pool.map(fit_job, [(a.exp, w, tp, n) for w, tp in wt for n in range(1, 10)])
        fd = pd.DataFrame(fits, columns=["well", "tp", "tile", "s", "sigma", "found"])
        tp_med = fd.groupby("tp")[["s", "sigma"]].median()
        jobs = []
        for (w, tp), g in fd.groupby(["well", "tp"]):
            ws, wsig = g.s.median(), g.sigma.median()
            fits_wt = {}
            for r in g.itertuples():
                if np.isfinite(r.s):
                    fits_wt[r.tile] = (r.s, r.sigma if np.isfinite(r.sigma) else wsig, "tile")
                elif np.isfinite(ws):
                    fits_wt[r.tile] = (ws, wsig, "well_tp_median")
                else:
                    fits_wt[r.tile] = (tp_med.loc[tp, "s"], tp_med.loc[tp, "sigma"], "tp_median")
            jobs.append((a.exp, w, tp, fits_wt, f"{out}/qc"))
        print(f"background fits done ({time.time() - t0:.0f}s); "
              f"{int((~np.isfinite(fd.s)).sum())} tiles used a fallback", flush=True)
        rows = []
        for k, res in enumerate(pool.imap_unordered(measure_job, jobs)):
            rows += res
            if k % 25 == 0:
                print(f"  measured {k + 1}/{len(jobs)} well-timepoints ({time.time() - t0:.0f}s)", flush=True)
    d = pd.DataFrame(rows).sort_values(["well", "tp", "tile"],
                                       key=lambda c: c.str[1:].astype(int) if c.name == "tp" else c)
    for k in REDS[a.exp]:
        d[f"{k}_gfp_ratio_mean"] = d[f"{k}_mean"] / d["gfp_mean"]
        d[f"{k}_gfp_ratio_median"] = d[f"{k}_median"] / d["gfp_median"]
    d.to_csv(f"{out}/tiles.csv", index=False)
    print(f"wrote {out}/tiles.csv: {len(d)} tiles ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()

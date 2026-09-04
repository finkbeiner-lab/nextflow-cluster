#!/usr/bin/env python
"""RGEDI segmentation: ungated Cellpose-SAM instance detection on aligned montages.

Detection only -- the EGFP transfection gate is applied post-hoc per experiment x
timepoint by egfp_gate.py (so gates re-derive without re-running the GPU). For each
(experiment, well, timepoint) in the manifest: run Cellpose-SAM on the FITC (EGFP
morphology) aligned montage, measure per-cell morphology + FITC and (projected) RFP
intensity, save the label mask (+ per-cell FITC + meta) and append per-cell rows.

Manifest CSV columns: exp, gxytmp, well, tp   (tp like 'T6'; gxytmp = that experiment's
GXYTMP root holding AlignedImages/). Locked Cellpose soma recipe.

Usage: python cellpose_montage_rgedi.py <manifest.csv> --out-dir DIR
Outputs: <out>/percell_all.csv (all cells, ungated) + <out>/masks/<tag>_masks.npz
"""
import argparse
import glob
import os
import numpy as np
import pandas as pd
import tifffile
from cellpose import models
from scipy import ndimage
from skimage.measure import regionprops_table

DIAM, FLOW, CELLPROB, CLEAN_K = 25, 0.6, -1.0, 2.0
PROPS = ('label', 'area', 'centroid', 'eccentricity', 'solidity',
         'perimeter', 'intensity_mean', 'intensity_max')


def norm255(raw):
    lo, hi = np.percentile(raw, 1), np.percentile(raw, 99.5)
    if hi <= lo:
        return np.zeros_like(raw, dtype=np.float32)
    return (np.clip((raw - lo) / (hi - lo), 0, 1) * 255.0).astype(np.float32)


def cellpose_seg(model, raw):
    """Cellpose-SAM + MAD intensity cleanup + size floor. Returns int32 label image."""
    masks = model.eval(norm255(raw), diameter=DIAM, flow_threshold=FLOW,
                       cellprob_threshold=CELLPROB)[0]
    nlab = int(masks.max())
    if nlab == 0:
        return masks.astype(np.int32)
    counts = np.bincount(masks.ravel(), minlength=nlab + 1); counts[0] = 0
    bg = raw[masks == 0]; bm = np.median(bg); mad = 1.4826 * np.median(np.abs(bg - bm))
    thr = bm + CLEAN_K * mad
    meds = ndimage.labeled_comprehension(raw, masks, np.arange(1, nlab + 1), np.median, float, 0.0)
    bright = np.zeros(nlab + 1, bool); bright[1:] = meds >= thr
    inten_areas = counts[1:][bright[1:]]
    floor = max(300.0, 0.6 * float(np.median(inten_areas))) if inten_areas.size else 300.0
    keep = np.zeros(nlab + 1, bool)
    for l in range(1, nlab + 1):
        if counts[l] >= floor and bright[l]:
            keep[l] = True
    return np.where(keep[masks], masks, 0).astype(np.int32)


def measure(lab, fitc, rfp):
    if int(lab.max()) == 0:
        return pd.DataFrame()
    t = regionprops_table(lab, intensity_image=fitc, properties=PROPS)
    df = pd.DataFrame(t).rename(columns={'intensity_mean': 'fitc_mean', 'intensity_max': 'fitc_max',
                                         'centroid-0': 'cy', 'centroid-1': 'cx'})
    tr = regionprops_table(lab, intensity_image=rfp, properties=('label', 'intensity_mean', 'intensity_max'))
    df = df.merge(pd.DataFrame(tr).rename(columns={'intensity_mean': 'rfp_mean', 'intensity_max': 'rfp_max'}),
                  on='label', how='left')
    df['circularity'] = 4.0 * np.pi * df['area'] / (df['perimeter'] ** 2 + 1e-9)
    return df


def find_one(pat):
    h = sorted(glob.glob(pat))
    return h[0] if h else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('manifest')
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--fitc', default='FITC'); ap.add_argument('--rfp', default='RFP')
    a = ap.parse_args()
    mdir = os.path.join(a.out_dir, 'masks'); os.makedirs(mdir, exist_ok=True)
    pdir = os.path.join(a.out_dir, 'percell'); os.makedirs(pdir, exist_ok=True)
    man = pd.read_csv(a.manifest)
    model = None   # lazy-load Cellpose only if there is work to do (resume-friendly)
    for r in man.itertuples():
        tag = f"{r.exp}_{r.well}_{r.tp}"
        mnpz = os.path.join(mdir, f"{tag}_masks.npz"); pcsv = os.path.join(pdir, f"{tag}.csv")
        if os.path.exists(mnpz) and os.path.exists(pcsv):
            print(f"{tag}: skip (already done)", flush=True); continue   # RESUME
        try:
            fp = find_one(f"{r.gxytmp}/AlignedImages/{r.well}/*_{r.tp}_0-1_{r.well}_0_{a.fitc}_*_ALIGNED.tif")
            rp = find_one(f"{r.gxytmp}/AlignedImages/{r.well}/*_{r.tp}_0-1_{r.well}_0_{a.rfp}_*_ALIGNED.tif")
            if not (fp and rp):
                print(f"{tag}: MISSING fitc={bool(fp)} rfp={bool(rp)}", flush=True); continue
            fitc = tifffile.imread(fp).astype(np.float32); rfp = tifffile.imread(rp).astype(np.float32)
            if model is None:
                # use_bfloat16=False: bf16 is ~10x slower on the galaxy V100s (cellpose 4.x default is True)
                model = models.CellposeModel(gpu=True, use_bfloat16=False)
            cp = cellpose_seg(model, fitc)
            d = measure(cp, fitc, rfp)
            np.savez_compressed(mnpz, cp=cp,
                                cp_label=(d['label'].to_numpy() if len(d) else np.array([], int)),
                                cp_fitc=(d['fitc_mean'].to_numpy() if len(d) else np.array([], float)),
                                meta=np.array([r.exp, r.well, r.tp]))
            if len(d):
                d.insert(0, 'method', 'CP')
                for c, v in (('exp', r.exp), ('well', r.well), ('tp', r.tp),
                             ('line', getattr(r, 'line', '')), ('genotype', getattr(r, 'genotype', ''))):
                    d[c] = v
            d.to_csv(pcsv, index=False)   # per-montage (empty ok -> marks done, resume-safe)
            print(f"{tag}: {len(d)} cells", flush=True)
        except Exception as e:
            import traceback; print(f"{tag}: FAILED {type(e).__name__}: {e}", flush=True); traceback.print_exc()
    # rebuild percell_all.csv from every per-montage file (correct after resume / multiple jobs)
    parts = [p for p in (pd.read_csv(f) for f in sorted(glob.glob(os.path.join(pdir, '*.csv')))) if len(p)]
    if parts:
        pd.concat(parts, ignore_index=True).to_csv(os.path.join(a.out_dir, 'percell_all.csv'), index=False)
    print(f"DONE -> {a.out_dir}/percell_all.csv + masks/ ({len(parts)} montages with cells)", flush=True)


if __name__ == '__main__':
    main()

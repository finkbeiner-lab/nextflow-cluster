#!/usr/bin/env python
"""Live/dead ratio threshold calibration for RGEDI MRID (auto + manual).

The RFP/GFP ratio separates live cells (low ratio) from dead cells (high ratio,
RFP spikes on death). This module sets the live/dead cutoff per (experiment,
timepoint), anchored to the LIVE (lower-ratio) population -- the same negative-
anchored GMM logic used for the EGFP transfection gate (see calibrate_gate.py).

Why per exp x timepoint, live-anchored (not whole-distribution antimode):
  - GFP maturation (early) and differential photobleaching (late) drift the live
    baseline over time -> per-timepoint tracks that technical drift.
  - Death is an RFP SPIKE (the biology); anchoring to the live mode detects it as
    cells jumping a fixed distance above the live baseline, instead of normalizing
    it away, and stays valid when a frame is unimodal (all-live early / all-dead late).

Methods:
  auto (default): fit 2-comp GMM on log10(ratio) per group; lower mode = live.
    live_threshold = 10^(mu_live + k * sd_live); dead_threshold = live_threshold
    shifted up by `ambiguous_margin` dex (MRID drops cells between the two).
    k is anchored so a reference reproduces a target (default the classic 0.07).
    BIC + reliability flag; unreliable groups fall back to manual (never silent).
  manual: use fixed live/dead thresholds from config for every group.
"""
import argparse
import numpy as np
import pandas as pd

import gmm1d  # dependency-free 2-component GMM (no sklearn in the pipeline container)

MIN_RATIO = 1e-6


def fit_live_mode(ratio):
    """2-comp GMM on log10(ratio); return LIVE (lower) mode stats + BIC + reliability."""
    x = np.log10(np.clip(ratio.astype(float), MIN_RATIO, None))
    f = gmm1d.fit2(x)
    mu, sd = float(f['mu'][0]), float(f['sigma'][0])   # lower mean = live
    mu_hi, w = float(f['mu'][1]), float(f['w'][0])
    bimodal = f['bic2'] < f['bic1']
    reliable = bimodal and (0.10 <= w <= 0.90) and abs(mu_hi - mu) >= 0.25
    return dict(mu=mu, sd=sd, mu_dead=mu_hi, w_live=w, reliable=reliable)


def calibrate(df, method, anchor_exp, anchor_value, live_manual, dead_manual,
              ambiguous_margin):
    """Return {(exp,tp): (live_thr, dead_thr, reliable)} + the anchor constant k."""
    k = None
    if method == 'auto':
        ref = df[df.exp == anchor_exp] if (df.exp == anchor_exp).any() else df
        a = fit_live_mode(ref.ratio)
        k = (np.log10(anchor_value) - a['mu']) / a['sd']      # constant fixed by anchor
    out = {}
    for (exp, tp), g in df.groupby(['exp', 'tp']):
        if method == 'manual':
            out[(exp, tp)] = (live_manual, dead_manual, True)
            continue
        f = fit_live_mode(g.ratio)
        if not f['reliable']:
            if live_manual is not None and dead_manual is not None:
                out[(exp, tp)] = (live_manual, dead_manual, False)
                print(f"  !! {exp} {tp}: live-mode fit UNRELIABLE (w_live={f['w_live']:.2f}) "
                      f"-> using manual {live_manual}/{dead_manual}", flush=True)
            else:
                # unreliable (e.g. an early, all-live/unimodal frame): still gate from this
                # group's own live mode -- keeps ~all cells live, correct when few have died.
                lt = 10 ** (f['mu'] + k * f['sd'])
                out[(exp, tp)] = (float(lt), float(10 ** (f['mu'] + k * f['sd'] + ambiguous_margin)), False)
                print(f"  !! {exp} {tp}: live/dead fit UNRELIABLE (w_live={f['w_live']:.2f}) -> "
                      f"live-anchored gate {lt:.4f} (mostly live); set --live/--dead to override", flush=True)
            continue
        live_thr = 10 ** (f['mu'] + k * f['sd'])
        dead_thr = 10 ** (f['mu'] + k * f['sd'] + ambiguous_margin)
        out[(exp, tp)] = (float(live_thr), float(dead_thr), True)
    return out, k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('percell_ratio_csv', help='per-cell CSV with columns exp, tp, ratio')
    ap.add_argument('--method', default='auto', choices=['auto', 'manual'])
    ap.add_argument('--anchor-exp', default='XDP5')
    ap.add_argument('--anchor', type=float, default=0.07, help='live cutoff the anchor exp reproduces')
    ap.add_argument('--live', type=float, default=None, help='manual live threshold / fallback')
    ap.add_argument('--dead', type=float, default=None, help='manual dead threshold / fallback')
    ap.add_argument('--ambiguous-margin', type=float, default=0.0, help='dex gap live->dead (drop-zone)')
    ap.add_argument('--out', default='livedead_thresholds.csv')
    a = ap.parse_args()
    df = pd.read_csv(a.percell_ratio_csv)
    gates, k = calibrate(df, a.method, a.anchor_exp, a.anchor, a.live, a.dead, a.ambiguous_margin)
    rows = [dict(exp=e, tp=t, live_threshold=round(lv, 6), dead_threshold=round(dd, 6), reliable=rel)
            for (e, t), (lv, dd, rel) in gates.items()]
    pd.DataFrame(rows).sort_values(['exp', 'tp']).to_csv(a.out, index=False)
    print(f"method={a.method}  anchor={a.anchor_exp}={a.anchor}  k={k}")
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"wrote {a.out}")


if __name__ == '__main__':
    main()

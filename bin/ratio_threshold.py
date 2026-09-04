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
from sklearn.mixture import GaussianMixture

MIN_RATIO = 1e-6


def fit_live_mode(ratio):
    """2-comp GMM on log10(ratio); return LIVE (lower) mode stats + BIC + reliability."""
    x = np.log10(np.clip(ratio.astype(float), MIN_RATIO, None)).reshape(-1, 1)
    g2 = GaussianMixture(2, random_state=0, n_init=5).fit(x)
    g1 = GaussianMixture(1, random_state=0).fit(x)
    i = int(np.argmin(g2.means_.ravel()))          # lower mean = live
    mu = float(g2.means_.ravel()[i]); sd = float(np.sqrt(g2.covariances_.ravel()[i]))
    mu_hi = float(g2.means_.ravel()[1 - i]); w = float(g2.weights_.ravel()[i])
    bimodal = g2.bic(x) < g1.bic(x)
    reliable = bimodal and (0.10 <= w <= 0.90) and abs(mu_hi - mu) >= 0.25
    return dict(mu=mu, sd=sd, mu_dead=mu_hi, w_live=w, reliable=reliable)


def calibrate(df, method, anchor_exp, anchor_value, live_manual, dead_manual,
              ambiguous_margin):
    """Return {(exp,tp): (live_thr, dead_thr, reliable)} + the anchor constant k."""
    k = None
    if method == 'auto':
        a = fit_live_mode(df[df.exp == anchor_exp].ratio)
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
                raise ValueError(
                    f"{exp} {tp}: ratio not cleanly bimodal and no manual thresholds set. "
                    f"Set mrid_live_threshold/mrid_dead_threshold for this dataset.")
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

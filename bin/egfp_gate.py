#!/usr/bin/env python
"""Per experiment x timepoint EGFP+ gate calibration (formula-driven, reproducible).

Method: in every image the UNTRANSFECTED cells are a built-in negative control -
they form the dim mode of the Cellpose FITC-mean distribution. We fit a 2-component
Gaussian mixture on log10(FITC mean) per (experiment, timepoint) group (wells pooled,
so cell-line/density differences do NOT move the gate), identify the negative (lower)
component (mu0, sd0), and place the gate a fixed statistical distance above it.

The distance is a SINGLE universal constant fixed by one anchor choice
(ANCHOR_EXP = 400). Two equivalent-in-spirit forms are computed:
  - percentile (default): gate = empirical P-th percentile of the negative cells,
      where P = percentile that ANCHOR_VALUE occupies in the anchor experiment's
      negative population (non-parametric, robust).
  - kSD: gate = 10^(mu0 + k*sd0), where k = (log10(ANCHOR_VALUE) - mu0_anchor)/sd0_anchor
      (parametric; assumes the negative mode is log-normal).

No per-experiment human judgement: each group supplies its own mu0/sd0; the anchor
constant is fixed once. BIC (2- vs 1-component) flags groups that are not cleanly
bimodal, where the fit should not be trusted.

Usage: python calibrate_gate.py <ungated_percell.csv> [--anchor-exp XDP5] [--anchor 400]
CSV must have columns: method, exp, well, tp, genotype, fitc_mean.
"""
import argparse
import numpy as np
import pandas as pd
from scipy.stats import percentileofscore, norm

import gmm1d  # dependency-free 2-component GMM (no sklearn in the pipeline container)

pd.set_option('display.width', 200, 'display.max_columns', 40)


def fit_neg(fitc_mean):
    """Fit 2-comp GMM on log10(fitc_mean); return negative-mode stats + BIC + neg cells."""
    x = np.log10(np.clip(fitc_mean.values.astype(float), 1, None))
    f = gmm1d.fit2(x)
    neg = x[f['assign'] == 0]                         # cells assigned to the negative (lower) mode
    med = float(np.median(neg))
    mad = float(1.4826 * np.median(np.abs(neg - med)))   # robust scale (log space)
    return dict(mu0=float(f['mu'][0]), sd0=float(f['sigma'][0]), mu1=float(f['mu'][1]),
                w0=float(f['w'][0]), neg=neg, med=med, mad=mad, bic2=f['bic2'], bic1=f['bic1'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('csv')
    ap.add_argument('--anchor-exp', default='XDP5')
    ap.add_argument('--anchor', type=float, default=None,
                    help='opt-in: anchor the gate so this value is reproduced on --anchor-exp; '
                         'if unset, --k (default 1.5 SD above background) is used')
    ap.add_argument('--out', default='gate_calibration.csv')
    ap.add_argument('--percell-out', default='percell_gated.csv')
    ap.add_argument('--manual-fitc', type=float, default=None,
                    help='fixed FITC gate used for any exp*tp whose automatic fit is REVIEW-flagged')
    ap.add_argument('--k', type=float, default=1.5,
                    help='gate = k SD above each group background mode (default 1.5); '
                         'ignored when --anchor is given')
    a = ap.parse_args()

    df = pd.read_csv(a.csv)
    cp = df[df.method == 'CP'].copy()
    la = np.log10(a.anchor) if a.anchor is not None else None

    # ---- gate strength: default k SD above background; --anchor opts into anchored mode ----
    if a.anchor is None:
        k = kr = a.k
        Pp = float(norm.cdf(k) * 100)
        print(f"GATE STRENGTH: k = {k:.2f} SD above each group's background mode "
              f"(= {Pp:.1f}th parametric pctile of background)\n")
    else:
        af = fit_neg(cp[cp.exp == a.anchor_exp].fitc_mean)
        k = (la - af['mu0']) / af['sd0']
        kr = (la - af['med']) / af['mad']
        Pp = float(norm.cdf(k) * 100)
        Pe = float(percentileofscore(af['neg'], la, kind='mean'))
        print(f"ANCHOR: {a.anchor_exp} pooled -> negative mode={10**af['mu0']:.0f} "
              f"(sd0={af['sd0']:.3f}dex, median={10**af['med']:.0f}). {a.anchor:.0f} sits at:")
        print(f"   kSD  : k  = {k:.3f} SD above negative mean  (= {Pp:.1f}th parametric pctile of negative)")
        print(f"   MAD  : kr = {kr:.3f} MAD above negative median")
        print(f"   empirical pctile-of-negative = {Pe:.1f}  "
              f"{'<-- SATURATED (anchor beyond the negative cells)' if Pe>=99.5 else ''}")
        print(f"   BIC 2-comp={af['bic2']:.0f} vs 1-comp={af['bic1']:.0f} -> "
              f"{'bimodal OK' if af['bic2'] < af['bic1'] else 'NOT clearly bimodal!'}\n")

    rows = []
    gate_map = {}
    hdr = (f"{'exp':7}{'tp':4}{'n_cells':>8}{'neg_mode':>9}{'pos_mode':>9}{'fit':>8}"
           f"{'gate_kSD':>9}{'gate_MAD':>9}{'keep_kSD':>9}{'keep_MAD':>9}")
    print(hdr); print('-' * len(hdr))
    for (exp, tp), g in cp.groupby(['exp', 'tp']):
        f = fit_neg(g.fitc_mean)
        bimodal = f['bic2'] < f['bic1']
        # reliability: need a clear 2-mode split AND a real negative population to anchor to
        reliable = bimodal and (0.10 <= f['w0'] <= 0.90) and abs(f['mu1'] - f['mu0']) >= 0.25
        if not reliable and a.manual_fitc is not None:
            gate_ksd = gate_mad = float(a.manual_fitc)   # manual override for the flagged group
            flag = 'MANUAL'
        else:
            gate_ksd = 10 ** (f['mu0'] + k * f['sd0'])
            gate_mad = 10 ** (f['med'] + kr * f['mad'])
            flag = 'ok' if reliable else 'REVIEW'
        gate_map[(exp, tp)] = (gate_ksd, gate_mad)
        keep_ksd = int((g.fitc_mean >= gate_ksd).sum())
        keep_mad = int((g.fitc_mean >= gate_mad).sum())
        if flag == 'REVIEW':
            print(f"  !! {exp} {tp}: gate fit UNRELIABLE (bimodal={bimodal}, neg_wt={f['w0']:.2f}, "
                  f"mode_sep={abs(f['mu1']-f['mu0']):.2f}dex) -- inspect, or set --manual-fitc", flush=True)
        print(f"{exp:7}{tp:4}{len(g):8d}{10**f['mu0']:9.0f}{10**f['mu1']:9.0f}{flag:>8}"
              f"{gate_ksd:9.0f}{gate_mad:9.0f}{keep_ksd:9d}{keep_mad:9d}")
        rows.append(dict(exp=exp, tp=tp, n_cells=len(g),
                         neg_mode=round(10**f['mu0'], 1), pos_mode=round(10**f['mu1'], 1),
                         neg_weight=round(f['w0'], 3), reliable=reliable,
                         gate_ksd=round(gate_ksd, 1), gate_mad=round(gate_mad, 1),
                         keep_ksd=keep_ksd, keep_mad=keep_mad))
    pd.DataFrame(rows).to_csv(a.out, index=False)

    # ---- ANALYSIS TABLE: per-cell EGFP+ Cellpose cells (both channels + gate flags) ----
    pc = cp.copy()
    pc['gate_ksd'] = [gate_map.get((e, t), (np.nan, np.nan))[0] for e, t in zip(pc.exp, pc.tp)]
    pc['gate_mad'] = [gate_map.get((e, t), (np.nan, np.nan))[1] for e, t in zip(pc.exp, pc.tp)]
    pc['egfp_pos_ksd'] = pc.fitc_mean >= pc.gate_ksd
    pc['egfp_pos_mad'] = pc.fitc_mean >= pc.gate_mad
    pc.to_csv(a.percell_out, index=False)
    npos = int(pc.egfp_pos_ksd.sum())
    if a.anchor is None:
        print(f"\ngate = k={k:.2f} SD above background (parametric pctile={Pp:.1f})")
    else:
        print(f"\nanchor (fixed by {a.anchor_exp}={a.anchor:.0f}):  k={k:.3f}SD  kr={kr:.3f}MAD  "
              f"(parametric pctile={Pp:.1f})")
    print(f"wrote {a.out}  (gate table)")
    print(f"wrote {a.percell_out}  (ANALYSIS table: {len(pc)} Cellpose cells, "
          f"{npos} EGFP+ by kSD; filter egfp_pos_ksd==True)")


if __name__ == '__main__':
    main()

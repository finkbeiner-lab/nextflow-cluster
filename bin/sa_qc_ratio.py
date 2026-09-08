#!/usr/bin/env python
"""Sodium-arsenite (S.A.) positive-control QC for the Cellpose+EGFP-gate pipeline.

S.A. dosed near the end of a run kills cells, so at the death timepoint the per-cell
GEDI ratio (RFP/GFP) splits into a LIVE (low-ratio) and a DEAD (high-ratio) population.
The antimode of that split is the optimized live/dead threshold for the experiment.
This QC shows the new pipeline recovers that separation, and checks that the EGFP gate
does not silently remove the dead population (dying cells dim in GFP).

Input = percell_gated.csv from egfp_gate.py (every Cellpose cell, with fitc_mean,
rfp_mean, well, line, genotype, and the egfp_pos_ksd gate flag). Condition (S.A. vs
untreated) per well comes from the run manifest.

Outputs (to --out-dir):
  <exp>_sa_ratio_by_line.png   S.A. vs untreated per-cell ratio, per line + pooled, threshold marked
  <exp>_sa_gate_effect.png     S.A. cells before/after the EGFP gate (does the gate eat the dead pop?)
  <exp>_sa_threshold.csv       threshold + per-line/condition %dead + gate-retention stats
"""
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import gmm1d  # dependency-free 2-component GMM (no sklearn in the container)

LIVE_C, DEAD_C, THR_C = '#2E8B6B', '#D2453D', '#222222'


def antimode(mu0, s0, w0, mu1, s1, w1):
    """Crossing point of two weighted Gaussians, in (mu0, mu1) -- the live/dead boundary."""
    a = 1.0 / (2 * s1 * s1) - 1.0 / (2 * s0 * s0)
    b = mu0 / (s0 * s0) - mu1 / (s1 * s1)
    c = (mu1 * mu1) / (2 * s1 * s1) - (mu0 * mu0) / (2 * s0 * s0) \
        + np.log((w1 / s1) / (w0 / s0))
    if abs(a) < 1e-12:
        x = -c / b if abs(b) > 1e-12 else 0.5 * (mu0 + mu1)
        return x
    disc = b * b - 4 * a * c
    if disc < 0:
        return 0.5 * (mu0 + mu1)
    for r in sorted((-b + s * np.sqrt(disc)) / (2 * a) for s in (1, -1)):
        if mu0 <= r <= mu1:
            return r
    return 0.5 * (mu0 + mu1)


def fit_threshold(logr):
    """2-comp GMM on log10(ratio); return (thr_log, live_mode, dead_mode, bimodal)."""
    f = gmm1d.fit2(np.asarray(logr, float))
    order = np.argsort(f['mu'])
    mu = np.array(f['mu'])[order]; sd = np.array(f['sigma'])[order]; w = np.array(f['w'])[order]
    thr = antimode(mu[0], sd[0], w[0], mu[1], sd[1], w[1])
    return thr, mu[0], mu[1], f['bic2'] < f['bic1']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gated', required=True, help='percell_gated.csv from egfp_gate.py')
    ap.add_argument('--manifest', required=True, help='run manifest with well,drug,line columns')
    ap.add_argument('--exp', default='EXP')
    ap.add_argument('--out-dir', default='.')
    ap.add_argument('--gate-col', default='egfp_pos_ksd')
    ap.add_argument('--sa-values', default='SA', help='comma-sep drug labels counted as S.A.')
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    df = pd.read_csv(a.gated)
    man = pd.read_csv(a.manifest)
    sa_set = set(v.strip() for v in a.sa_values.split(','))
    well_drug = {w: ('S.A.' if str(d).strip() in sa_set else 'untreated')
                 for w, d in zip(man['well'], man['drug'])}
    df['cond'] = df['well'].map(well_drug)
    df = df[df['cond'].notna()].copy()
    df['ratio'] = df['rfp_mean'] / (df['fitc_mean'] + 1e-9)
    df['logr'] = np.log10(np.clip(df['ratio'], 1e-4, None))
    egfp = df[df[a.gate_col] == True].copy()          # noqa: E712  (EGFP+ kept cells)

    # ---- threshold from the pooled EGFP+ cells (live+dead across both conditions) ----
    thr_log, live_m, dead_m, bimodal = fit_threshold(egfp['logr'])
    thr = 10 ** thr_log
    # SA-only threshold as a cross-check (dead vs surviving-live within the S.A. wells)
    sa_egfp = egfp[egfp.cond == 'S.A.']
    thr_sa_log, _, _, sa_bimodal = fit_threshold(sa_egfp['logr'])

    lines = sorted(egfp['line'].dropna().unique())
    xlo, xhi = np.percentile(egfp['logr'], [0.5, 99.5])
    bins = np.linspace(xlo, xhi, 60)

    def dead_frac(sub):
        return float((sub['logr'] > thr_log).mean()) if len(sub) else np.nan

    # ---- Figure 1: per-line S.A. vs untreated, shared threshold ----
    panels = lines + ['All lines pooled']
    ncol = 3; nrow = int(np.ceil(len(panels) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 4.1, nrow * 3.1), squeeze=False)
    rows_csv = []
    for i, name in enumerate(panels):
        ax = axes[i // ncol][i % ncol]
        sub = egfp if name == 'All lines pooled' else egfp[egfp.line == name]
        for cond, col in (('untreated', LIVE_C), ('S.A.', DEAD_C)):
            s = sub[sub.cond == cond]
            if len(s):
                ax.hist(s['logr'], bins=bins, density=True, color=col, alpha=0.55,
                        label=f'{cond} (n={len(s):,})')
        ax.axvline(thr_log, color=THR_C, ls='--', lw=1.4)
        ax.set_title(name, fontsize=10, fontweight='bold')
        ax.set_yticks([]); ax.set_xlabel('log₁₀(RFP/GFP)', fontsize=8)
        ax.legend(fontsize=7, frameon=False)
        du, dd = dead_frac(sub[sub.cond == 'untreated']), dead_frac(sub[sub.cond == 'S.A.'])
        ax.text(0.03, 0.97, f'% > thr\nS.A. {dd*100:.0f}%\nuntr {du*100:.0f}%',
                transform=ax.transAxes, va='top', fontsize=7.5, color=THR_C)
        if name != 'All lines pooled':
            g = sub['genotype'].iloc[0] if len(sub) else ''
            rows_csv.append(dict(line=name, genotype=g,
                                 n_sa=int((sub.cond == 'S.A.').sum()),
                                 n_untr=int((sub.cond == 'untreated').sum()),
                                 pct_dead_sa=round(dd * 100, 1), pct_dead_untr=round(du * 100, 1)))
    for j in range(len(panels), nrow * ncol):
        axes[j // ncol][j % ncol].axis('off')
    fig.suptitle(f'{a.exp}  S.A. death control -- per-cell GEDI ratio (Cellpose + EGFP gate, T13)\n'
                 f'live/dead threshold = {thr:.3f}  (log₁₀ = {thr_log:.3f}); '
                 f"{'bimodal OK' if bimodal else 'NOT clearly bimodal -- inspect'}",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    p1 = os.path.join(a.out_dir, f'{a.exp}_sa_ratio_by_line.png')
    fig.savefig(p1, dpi=150); plt.close(fig)

    # ---- Figure 2: does the EGFP gate remove the dead population? ----
    fig2, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    sa_all = df[df.cond == 'S.A.']
    sa_kept = sa_all[sa_all[a.gate_col] == True]     # noqa: E712
    sa_drop = sa_all[sa_all[a.gate_col] != True]
    for s, col, lab in ((sa_all, '#888888', f'all detections (n={len(sa_all):,})'),
                        (sa_kept, LIVE_C, f'EGFP+ kept (n={len(sa_kept):,})')):
        ax[0].hist(s['logr'], bins=bins, density=True, color=col, alpha=0.6, label=lab)
    ax[0].axvline(thr_log, color=THR_C, ls='--', lw=1.4)
    ax[0].set_title('S.A. cells: before vs after EGFP gate', fontsize=10, fontweight='bold')
    ax[0].set_xlabel('log₁₀(RFP/GFP)', fontsize=8); ax[0].set_yticks([])
    ax[0].legend(fontsize=8, frameon=False)
    # retention of the DEAD population specifically
    dead_all = int((sa_all['logr'] > thr_log).sum())
    dead_kept = int((sa_kept['logr'] > thr_log).sum())
    ret = 100 * dead_kept / dead_all if dead_all else float('nan')
    ax[1].axis('off')
    ax[1].text(0.0, 0.95,
               f'S.A. cells detected      {len(sa_all):,}\n'
               f'kept by EGFP gate       {len(sa_kept):,}  ({100*len(sa_kept)/max(len(sa_all),1):.0f}%)\n'
               f'dropped (dim GFP)       {len(sa_drop):,}\n\n'
               f'DEAD cells (ratio > thr)\n'
               f'  among all detections  {dead_all:,}\n'
               f'  retained after gate   {dead_kept:,}  ({ret:.0f}%)\n\n'
               f'-> gate {"PRESERVES" if ret >= 80 else "MAY CLIP"} the dead population',
               transform=ax[1].transAxes, va='top', family='monospace', fontsize=9.5)
    fig2.suptitle(f'{a.exp}  EGFP gate effect on the S.A. dead population', fontsize=11)
    fig2.tight_layout(rect=[0, 0, 1, 0.93])
    p2 = os.path.join(a.out_dir, f'{a.exp}_sa_gate_effect.png')
    fig2.savefig(p2, dpi=150); plt.close(fig2)

    # ---- threshold CSV ----
    out = pd.DataFrame(rows_csv)
    meta = pd.DataFrame([dict(line='__EXPERIMENT__', genotype='',
                              n_sa=int((egfp.cond == 'S.A.').sum()),
                              n_untr=int((egfp.cond == 'untreated').sum()),
                              pct_dead_sa=round(dead_frac(sa_egfp) * 100, 1),
                              pct_dead_untr=round(dead_frac(egfp[egfp.cond == 'untreated']) * 100, 1))])
    out = pd.concat([meta, out], ignore_index=True)
    out['threshold_ratio'] = round(thr, 4)
    out['threshold_log10'] = round(thr_log, 4)
    out['threshold_sa_only'] = round(10 ** thr_sa_log, 4)
    out['bimodal'] = bimodal
    out['dead_retained_pct'] = round(ret, 1)
    pcsv = os.path.join(a.out_dir, f'{a.exp}_sa_threshold.csv')
    out.to_csv(pcsv, index=False)

    print(f"[{a.exp}] EGFP+ cells: {len(egfp):,}  (S.A. {int((egfp.cond=='S.A.').sum()):,} / "
          f"untreated {int((egfp.cond=='untreated').sum()):,})")
    print(f"  live mode 10^{live_m:.3f}={10**live_m:.3f}  dead mode 10^{dead_m:.3f}={10**dead_m:.3f}")
    print(f"  THRESHOLD (RFP/GFP) = {thr:.4f}   [S.A.-only cross-check {10**thr_sa_log:.4f}]  "
          f"bimodal={bimodal}")
    print(f"  dead-population retention through EGFP gate = {ret:.0f}%")
    print(f"  wrote {p1}\n         {p2}\n         {pcsv}")


if __name__ == '__main__':
    main()

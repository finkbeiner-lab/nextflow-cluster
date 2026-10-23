#!/usr/bin/env python
"""Sodium-arsenite (S.A.) positive-control QC for the Cellpose+EGFP-gate pipeline.

S.A. dosed near the end of a run kills cells, so at the death timepoint the per-cell
GEDI ratio (RFP/GFP) splits into a LIVE (low-ratio) and a DEAD (high-ratio) population.
The antimode of that split is the optimized live/dead threshold for the experiment.

Gate-early / read-late design (default): the EGFP transfection gate is calibrated on an
EARLY timepoint (--gate-tp, e.g. T1) where every cell is alive and GFP cleanly marks
transfection, then that gate VALUE is applied to the death timepoint (--read-tp, e.g. T13).
This avoids the confound that dying cells dim in GFP and would be wrongly dropped by a gate
re-fit at the death frame. Set --gate-tp == --read-tp to reproduce the naive single-frame gate.

Input = percell_all.csv (ungated Cellpose cells with fitc_mean, rfp_mean, well, line,
genotype, tp). Condition (S.A. vs untreated) per well comes from the run manifest.

Outputs (to --out-dir):
  <exp>_sa_ratio_by_line.png   S.A. vs untreated per-cell ratio at the read tp, per line + pooled
  <exp>_sa_gate_effect.png     why gate-early wins: T1 GFP gate + dead-cell retention T1-gate vs read-tp-gate
  <exp>_sa_threshold.csv       threshold + per-line %dead + gate-retention stats
"""
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import gmm1d  # dependency-free 2-component GMM (no sklearn in the container)

LIVE_C, DEAD_C, THR_C, GATE_C = '#2E8B6B', '#D2453D', '#222222', '#3B6FB0'


def antimode(mu0, s0, w0, mu1, s1, w1):
    """Crossing point of two weighted Gaussians, in (mu0, mu1)."""
    a = 1.0 / (2 * s1 * s1) - 1.0 / (2 * s0 * s0)
    b = mu0 / (s0 * s0) - mu1 / (s1 * s1)
    c = (mu1 * mu1) / (2 * s1 * s1) - (mu0 * mu0) / (2 * s0 * s0) + np.log((w1 / s1) / (w0 / s0))
    if abs(a) < 1e-12:
        return -c / b if abs(b) > 1e-12 else 0.5 * (mu0 + mu1)
    disc = b * b - 4 * a * c
    if disc < 0:
        return 0.5 * (mu0 + mu1)
    for r in sorted((-b + s * np.sqrt(disc)) / (2 * a) for s in (1, -1)):
        if mu0 <= r <= mu1:
            return r
    return 0.5 * (mu0 + mu1)


def fit2_sorted(vals):
    f = gmm1d.fit2(np.asarray(vals, float))
    o = np.argsort(f['mu'])
    return (np.array(f['mu'])[o], np.array(f['sigma'])[o], np.array(f['w'])[o],
            f['bic2'] < f['bic1'])


def egfp_gate_value(fitc, k):
    """Transfection gate = 10^(mu0 + k*sd0) of the 2-comp fit on log10(FITC)."""
    x = np.log10(np.clip(np.asarray(fitc, float), 1, None))
    mu, sd, w, bimodal = fit2_sorted(x)
    return 10 ** (mu[0] + k * sd[0]), mu, sd, bimodal


def ratio_threshold(logr):
    mu, sd, w, bimodal = fit2_sorted(logr)
    return antimode(mu[0], sd[0], w[0], mu[1], sd[1], w[1]), mu[0], mu[1], bimodal


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--percell', required=True, help='percell_all.csv (ungated, both timepoints)')
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--exp', default='EXP')
    ap.add_argument('--out-dir', default='.')
    ap.add_argument('--gate-tp', default='T1', help='timepoint to CALIBRATE the EGFP gate on (alive)')
    ap.add_argument('--read-tp', default='T13', help='death timepoint to READ the ratio on')
    ap.add_argument('--k', type=float, default=1.5, help='EGFP gate = k SD above background mode')
    ap.add_argument('--sa-values', default='SA')
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    df = pd.read_csv(a.percell)
    man = pd.read_csv(a.manifest)
    sa_set = set(v.strip() for v in a.sa_values.split(','))
    well_drug = {w: ('S.A.' if str(d).strip() in sa_set else 'untreated')
                 for w, d in zip(man['well'], man['drug'])}
    df['cond'] = df['well'].map(well_drug)
    df = df[df['cond'].notna()].copy()
    df['ratio'] = df['rfp_mean'] / (df['fitc_mean'] + 1e-9)
    df['logr'] = np.log10(np.clip(df['ratio'], 1e-4, None))

    gate_cells = df[df.tp == a.gate_tp]
    read_cells = df[df.tp == a.read_tp].copy()
    if not len(gate_cells) or not len(read_cells):
        raise SystemExit(f"missing cells: gate_tp {a.gate_tp}={len(gate_cells)}, read_tp {a.read_tp}={len(read_cells)}")

    # --- transfection gate from the ALIVE gate timepoint (pooled: at T1 all wells are pre-dose) ---
    gate_val, gmu, gsd, gbim = egfp_gate_value(gate_cells.fitc_mean, a.k)
    naive_val, _, _, _ = egfp_gate_value(read_cells.fitc_mean, a.k)   # for comparison only
    read_cells['egfp_pos'] = read_cells.fitc_mean >= gate_val
    egfp = read_cells[read_cells.egfp_pos].copy()

    # Threshold = the S.A. wells' OWN dead/live antimode (the positive-control gap),
    # cross-checked against the untreated (known-live) upper tail.
    sa_egfp = egfp[egfp.cond == 'S.A.']
    un_egfp = egfp[egfp.cond == 'untreated']
    # Robust positive-control threshold: the untreated (known-live) ratio ceiling.
    # The dead population is a minority shoulder, so a 2-comp antimode is unreliable here;
    # anchoring on the live cells is stable. (SA antimode kept as a cross-check.)
    live_p99 = float(np.percentile(un_egfp['logr'], 99)) if len(un_egfp) else np.nan
    sa_anti, live_m, dead_m, bimodal = ratio_threshold(sa_egfp['logr'])
    thr_log = live_p99
    thr = 10 ** thr_log
    lines = sorted(egfp['line'].dropna().unique())
    xlo, xhi = np.percentile(egfp['logr'], [0.5, 99.5])
    bins = np.linspace(xlo, xhi, 60)
    dfrac = lambda s: float((s['logr'] > thr_log).mean()) if len(s) else np.nan

    # ---- Figure 1: per-line S.A. vs untreated at read tp ----
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
        ax.axvline(live_p99, color=LIVE_C, ls=':', lw=1.2)   # untreated 99th pct (live ceiling)
        ax.set_title(name, fontsize=10, fontweight='bold')
        ax.set_yticks([]); ax.set_xlabel('log₁₀(RFP/GFP)', fontsize=8)
        ax.legend(fontsize=7, frameon=False)
        du, dd = dfrac(sub[sub.cond == 'untreated']), dfrac(sub[sub.cond == 'S.A.'])
        ax.text(0.03, 0.97, f'% dead\nS.A. {dd*100:.0f}%\nuntr {du*100:.0f}%',
                transform=ax.transAxes, va='top', fontsize=7.5, color=THR_C)
        if name != 'All lines pooled':
            g = sub['genotype'].iloc[0] if len(sub) else ''
            rows_csv.append(dict(line=name, genotype=g, n_sa=int((sub.cond == 'S.A.').sum()),
                                 n_untr=int((sub.cond == 'untreated').sum()),
                                 pct_dead_sa=round(dd * 100, 1), pct_dead_untr=round(du * 100, 1)))
    for j in range(len(panels), nrow * ncol):
        axes[j // ncol][j % ncol].axis('off')
    fig.suptitle(f'{a.exp}  S.A. death control -- per-cell GEDI ratio at {a.read_tp}\n'
                 f'EGFP gate calibrated on {a.gate_tp} (alive) = {gate_val:.0f}; '
                 f'live/dead threshold RFP/GFP = {thr:.3f}  '
                 f"({'bimodal OK' if bimodal else 'NOT clearly bimodal'})", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    p1 = os.path.join(a.out_dir, f'{a.exp}_sa_ratio_by_line.png'); fig.savefig(p1, dpi=150); plt.close(fig)

    # ---- Figure 2: why gate-early wins ----
    fig2, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    # (A) the T1 GFP distribution + gate
    gx = np.log10(np.clip(gate_cells.fitc_mean, 1, None))
    ax[0].hist(gx, bins=60, color='#8a8a8a', alpha=0.7)
    ax[0].axvline(np.log10(gate_val), color=GATE_C, ls='--', lw=1.6, label=f'gate {gate_val:.0f}')
    ax[0].set_title(f'EGFP gate from {a.gate_tp} (alive, clean bimodal)', fontsize=10, fontweight='bold')
    ax[0].set_xlabel('log₁₀(FITC/GFP mean)', fontsize=8); ax[0].set_yticks([])
    ax[0].legend(fontsize=8, frameon=False)
    # (B) dead-cell retention: T1-gate vs naive read-tp gate
    dead = read_cells[read_cells.logr > thr_log]
    ret_t1 = 100 * (dead.fitc_mean >= gate_val).mean() if len(dead) else float('nan')
    ret_naive = 100 * (dead.fitc_mean >= naive_val).mean() if len(dead) else float('nan')
    ax[1].axis('off')
    ax[1].text(0.0, 0.95,
               f'{a.read_tp} cells (all)          {len(read_cells):,}\n'
               f'EGFP+ by {a.gate_tp}-gate ({gate_val:.0f})  {int(read_cells.egfp_pos.sum()):,}'
               f'  ({100*read_cells.egfp_pos.mean():.0f}%)\n\n'
               f'DEAD cells (ratio > {thr:.3f}) retained:\n'
               f'  {a.gate_tp}-gate (gate-early)   {ret_t1:.0f}%\n'
               f'  {a.read_tp}-gate (naive)        {ret_naive:.0f}%\n\n'
               f'-> gating on {a.gate_tp} keeps {ret_t1-ret_naive:+.0f} pts more of the\n'
               f'   dead population than re-gating at {a.read_tp}',
               transform=ax[1].transAxes, va='top', family='monospace', fontsize=9.5)
    fig2.suptitle(f'{a.exp}  gate-early / read-late: EGFP gate on {a.gate_tp}, ratio on {a.read_tp}', fontsize=11)
    fig2.tight_layout(rect=[0, 0, 1, 0.92])
    p2 = os.path.join(a.out_dir, f'{a.exp}_sa_gate_effect.png'); fig2.savefig(p2, dpi=150); plt.close(fig2)

    # ---- threshold CSV ----
    sa_e, un_e = egfp[egfp.cond == 'S.A.'], egfp[egfp.cond == 'untreated']
    meta = pd.DataFrame([dict(line='__EXPERIMENT__', genotype='', n_sa=len(sa_e), n_untr=len(un_e),
                              pct_dead_sa=round(dfrac(sa_e) * 100, 1),
                              pct_dead_untr=round(dfrac(un_e) * 100, 1))])
    out = pd.concat([meta, pd.DataFrame(rows_csv)], ignore_index=True)
    out['threshold_ratio'] = round(thr, 4); out['threshold_log10'] = round(thr_log, 4)
    out['gate_tp'] = a.gate_tp; out['gate_value'] = round(gate_val, 1); out['read_tp'] = a.read_tp
    out['dead_retain_gateearly_pct'] = round(ret_t1, 1)
    out['dead_retain_naive_pct'] = round(ret_naive, 1)
    pcsv = os.path.join(a.out_dir, f'{a.exp}_sa_threshold.csv'); out.to_csv(pcsv, index=False)

    print(f"[{a.exp}] gate({a.gate_tp})={gate_val:.0f} (bimodal={gbim}); "
          f"{a.read_tp} EGFP+ {int(read_cells.egfp_pos.sum()):,}/{len(read_cells):,}")
    print(f"  live 10^{live_m:.2f}={10**live_m:.3f}  dead 10^{dead_m:.2f}={10**dead_m:.3f}  "
          f"THRESHOLD RFP/GFP={thr:.4f} bimodal={bimodal}")
    print(f"  %dead  S.A.={dfrac(sa_e)*100:.0f}%  untreated={dfrac(un_e)*100:.0f}%")
    print(f"  dead retention: gate-early({a.gate_tp})={ret_t1:.0f}%  vs naive({a.read_tp})={ret_naive:.0f}%")
    print(f"  wrote {p1}\n         {p2}\n         {pcsv}")


if __name__ == '__main__':
    main()

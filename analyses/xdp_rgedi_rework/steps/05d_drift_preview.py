"""Step 05d: how each candidate cutoff rule behaves over time in untreated wells (evidence for D20).

For every case whose S.A. dead population is separated (step 05c), apply that case's rule B, C and D
cutoffs (one fixed set per line x experiment, as MRID uses them) to the untreated wells at EVERY
timepoint and report the death fraction dead/(live+dead). Also report the untreated live spread
above and below the peak at T1 and at the fit timepoint, to show how the live population changes.
No S.A. cells are involved; any rise over time is either real death or an upward drift of live cells.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from lib import cfg, data, figs  # noqa: E402
from lib import mixture as mx  # noqa: E402
from lib.paths import LOCAL_OUT  # noqa: E402
from lib.provenance import StepRecorder  # noqa: E402

STEP = "05d_drift_preview"
plt = figs.plt


def spread(x):
    m = mx.kde_mode(x)
    lo, hi = x[x < m], x[x > m]
    return m, (m - np.median(lo)) / 0.674, (np.median(hi) - m) / 0.674


def main() -> None:
    P = cfg.params()
    rec = StepRecorder(STEP, LOCAL_OUT / STEP)
    pv = LOCAL_OUT / "05c_cutoff_preview"
    pop = pd.read_csv(pv / "populations.csv")
    rules = pd.read_csv(pv / "rule_consequences.csv")
    pop = pop[pop.dead_found.astype(bool) & (pop.ashman_ut_dead >= P["usability"]["min_ashman_d"])]
    rows, sp_rows = [], []
    for exp in [e for e in figs.EXP_ORDER if e in set(pop.exp)]:
        d = data.load(exp, columns=["line", "condition", "tp", "log10_ratio", "in_scope"])
        ref = d if exp == "XDP1" else d[d.condition == "untreated"]
        fit_tps = [int(t) for t in str(pop[pop.exp == exp].fit_tps.iloc[0]).split(",")]
        for line in pop[pop.exp == exp].line:
            r = rules[(rules.exp == exp) & (rules.line == line)].set_index("rule")
            for tp, g in ref[ref.line == line].groupby("tp"):
                if exp == "XDP1" and tp > 10:
                    continue          # whole plate dosed after T10
                x = g.log10_ratio.dropna().to_numpy()
                for rule in ("B", "C", "D"):
                    if rule not in r.index:
                        continue
                    lo, hi = np.log10(r.loc[rule, "live_thr"]), np.log10(r.loc[rule, "dead_thr"])
                    live, dead = np.mean(x < lo), np.mean(x > hi)
                    rows.append({"exp": exp, "line": line, "tp": tp, "rule": rule, "death_frac": dead / (live + dead),
                                 "ambiguous": 1 - live - dead})
            for label, tps in (("T1", [1]), ("fit", fit_tps if exp != "XDP1" else [10])):
                x = ref[(ref.line == line) & ref.tp.isin(tps)].log10_ratio.dropna().to_numpy()
                m, below, above = spread(x)
                sp_rows.append({"exp": exp, "line": line, "when": label, "mode_ratio": 10 ** m, "sd_below": below,
                                "sd_above": above})
    df = pd.DataFrame(rows)
    sp = pd.DataFrame(sp_rows)
    df.to_csv(rec.out_dir / "untreated_death_by_tp.csv", index=False)
    sp.to_csv(rec.out_dir / "live_spread.csv", index=False)
    t1 = df[df.tp == 1].groupby("rule").death_frac
    last = df.loc[df.groupby(["exp", "line", "rule"]).tp.transform("max") == df.tp].groupby("rule").death_frac
    rec.summary = {"cases": int(df.groupby(["exp", "line"]).ngroups),
                   "death_T1": {k: [float(v.min()), float(v.max())] for k, v in t1},
                   "death_last_untreated_tp": {k: [float(v.min()), float(v.max())] for k, v in last},
                   "sd_below_T1": [float(sp[sp.when == "T1"].sd_below.min()), float(sp[sp.when == "T1"].sd_below.max())],
                   "sd_above_T1": [float(sp[sp.when == "T1"].sd_above.min()), float(sp[sp.when == "T1"].sd_above.max())],
                   "sd_below_fit": [float(sp[sp.when == "fit"].sd_below.min()), float(sp[sp.when == "fit"].sd_below.max())],
                   "sd_above_fit": [float(sp[sp.when == "fit"].sd_above.min()), float(sp[sp.when == "fit"].sd_above.max())],
                   "mode_T1": [float(sp[sp.when == "T1"].mode_ratio.min()), float(sp[sp.when == "T1"].mode_ratio.max())],
                   "mode_fit": [float(sp[sp.when == "fit"].mode_ratio.min()), float(sp[sp.when == "fit"].mode_ratio.max())]}
    # Figure: untreated death fraction over time per rule, one panel per experiment.
    exps = [e for e in figs.EXP_ORDER if e in set(df.exp)]
    fig, axes = plt.subplots(1, len(exps), figsize=(3.3 * len(exps), 3.0), sharey=True, squeeze=False)
    style = {"B": ("--", 1.2), "C": ("-", 2.0), "D": (":", 1.6)}
    for ax, exp in zip(axes[0], exps):
        hours = cfg.read_timepoints(exp).set_index("tp").hours
        for (line, rule), g in df[df.exp == exp].groupby(["line", "rule"]):
            g = g.sort_values("tp")
            ax.plot(g.tp.map(hours).to_numpy(), 100 * g.death_frac.to_numpy(), ls=style[rule][0], lw=style[rule][1],
                    color=figs.LINE_COLOR[line], label=f"{figs.LINE_SHORT[line]} rule {rule}")
        ax.set_title(exp, fontsize=9)
        ax.set_xlabel("hours")
    axes[0][0].set_ylabel("untreated cells called dead (%)")
    h, l = axes[0][-1].get_legend_handles_labels()
    axes[0][-1].legend(h, l, fontsize=6, loc="upper left", ncol=2)
    fig.suptitle("Untreated wells only: death fraction over time under each candidate rule (solid C, dotted D, "
                 "dashed B; one fixed cutoff set per line)", x=0.01, ha="left", fontsize=10)
    path = figs.save(fig, rec.out_dir / "F_drift_rules.png")
    rec.figure("Fdrift", path, "Untreated wells only (XDP1: all wells up to T10, before dosing): share of cells called "
               "dead at each timepoint under rules B (dashed), C (solid) and D (dotted), using each line's fixed cutoffs. "
               "A rise over time is either real death or an upward drift of living cells.")
    rec.write()
    print(rec.summary)
    piv = df.pivot_table(index=["exp", "line", "tp"], columns="rule", values="death_frac")
    print((100 * piv).round(1).to_string())


if __name__ == "__main__":
    main()

"""Step 03: cell-count QC and plate-wide anomaly flags (no thresholds involved).

Per experiment (in-scope lines; siRNA wells reported but not plotted):
  * cells per well per timepoint; wells with no cells at some timepoint
  * totals per line x condition x timepoint, normalised to T1 and to the last pre-dose timepoint
  * plate-wide high-ratio fraction in UNTREATED wells per timepoint (share of cells above the
    plate's own T1 99th percentile) - flags timepoint-wide artefacts such as XDP5/XDP1 T7
Figure F3: count trajectories, S.A. vs untreated, normalised to the last pre-dose timepoint.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from lib import cfg, data, figs  # noqa: E402
from lib.paths import LOCAL_OUT  # noqa: E402
from lib.provenance import StepRecorder  # noqa: E402

STEP = "03_qc_counts"
plt = figs.plt


def pre_tp(exp: str):
    s = cfg.experiments()[exp]["sa"]
    return s["pre_tp"] if isinstance(s["pre_tp"], int) else None


def main() -> None:
    rec = StepRecorder(STEP, LOCAL_OUT / STEP)
    exps = [e for e in figs.EXP_ORDER if data.parquet_path(e).exists()]
    rec.summary["experiments"] = exps
    for exp in exps:
        d = data.load(exp, columns=["well", "line", "condition", "tp", "hours", "log10_ratio", "in_scope"])
        roles = cfg.well_roles(exp)
        roles = roles[roles.in_scope]
        tps = cfg.read_timepoints(exp)
        tps = tps[tps.tp.isin(d.tp.unique())]          # timepoints actually present (XDP8 has no T2)
        well_tp = d.groupby(["well", "tp"]).size().rename("n").reset_index()
        grid = pd.MultiIndex.from_product([roles.well, tps.tp], names=["well", "tp"]).to_frame(index=False)
        grid = grid.merge(well_tp, on=["well", "tp"], how="left").merge(
            roles[["well", "line", "condition", "sirna_name"]], on="well")
        grid.to_csv(rec.out_dir / f"cells_per_well_{exp}.csv", index=False)
        empty = grid[grid.n.isna()]
        rec.check(f"{exp}.no_empty_well_tps", f"{exp}: every in-scope well has cells at every timepoint",
                  None if len(empty) else True,
                  {"n_empty": len(empty), "examples": empty[["well", "tp"]].astype(str).values.tolist()[:15]})

        tot = grid.groupby(["line", "condition", "tp"]).n.sum().rename("cells").reset_index()
        tot["norm_T1"] = tot.cells / tot.groupby(["line", "condition"]).cells.transform(
            lambda s: s.iloc[0])
        p = pre_tp(exp)
        if p is not None:
            base = tot[tot.tp == p].set_index(["line", "condition"]).cells
            tot["norm_pre"] = tot.cells / tot.set_index(["line", "condition"]).index.map(base)
        tot.insert(0, "exp", exp)
        tot.to_csv(rec.out_dir / f"totals_{exp}.csv", index=False)

        # Plate-wide high-ratio fraction in untreated wells (artefact screen).
        ref = d[d.condition == ("untreated" if exp != "XDP1" else "sa")]   # XDP1: whole plate dosed
        thr = np.nanpercentile(ref[ref.tp == ref.tp.min()].log10_ratio, 99)
        hi = (ref.assign(hi=ref.log10_ratio > thr).groupby("tp").hi.mean().rename("frac_above_T1_p99")
              .reset_index())
        early = hi.frac_above_T1_p99.iloc[:3].mean()
        hi["fold_vs_T1_T3"] = hi.frac_above_T1_p99 / max(early, 1e-9)
        hi["spike"] = (hi.fold_vs_T1_T3 > 4) & (hi.frac_above_T1_p99 > 0.03)
        hi.insert(0, "exp", exp)
        hi.to_csv(rec.out_dir / f"plate_highratio_{exp}.csv", index=False)
        spikes = hi[hi.spike].tp.tolist()
        rec.check(f"{exp}.plate_spikes", f"{exp}: timepoints where the share of high-ratio cells in reference "
                  "wells jumps plate-wide (>4x the T1-T3 average and >3%)", None if spikes else True,
                  {"spike_tps": spikes, "threshold_ratio": float(10 ** thr)})
        rec.summary[exp] = {"pre_tp": p, "n_empty_well_tps": int(len(empty)), "spike_tps": spikes,
                            "highratio_threshold": float(10 ** thr)}
        plot_counts(exp, grid, tot, p, rec)
    rec.write()
    for c in rec.checks:
        if c["status"] != "PASS":
            print(c["status"], c["id"], str(c["value"])[:200])


def plot_counts(exp: str, grid: pd.DataFrame, tot: pd.DataFrame, p, rec: StepRecorder) -> None:
    """F3: per-well counts normalised to the reference timepoint; S.A. violet, untreated grey."""
    lines = [l for l in figs.LINE_ORDER if l in set(grid.line)]
    ref_tp = p if p is not None else grid.tp.min()
    hours = cfg.read_timepoints(exp).set_index("tp").hours
    fig, axes = plt.subplots(1, len(lines), figsize=(3.1 * len(lines), 2.9), sharey=True, squeeze=False)
    g = grid[grid.condition.isin(["sa", "untreated"])].copy()
    base = g[g.tp == ref_tp].set_index("well").n
    g["rel"] = g.n / g.well.map(base)
    g["h"] = g.tp.map(hours)
    for ax, line in zip(axes[0], lines):
        for cond, color, lw in (("untreated", figs.UT, 0.5), ("sa", figs.SA, 0.7)):
            sub = g[(g.line == line) & (g.condition == cond)]
            for _, w in sub.groupby("well"):
                ax.plot(w.h.to_numpy(), np.log2(w.rel.to_numpy()), color=color, lw=lw, alpha=0.35)
            m = sub.groupby("h").rel.median()
            if len(m):
                ax.plot(m.index.to_numpy(), np.log2(m.to_numpy()), color=color, lw=2.2,
                        label=("S.A. wells" if cond == "sa" else "untreated wells") + " (median)")
        s = cfg.experiments()[exp]["sa"]
        for dose in s.get("doses", []):
            if dose.get("after_tp") and dose.get("before_tp"):
                x = (hours[dose["after_tp"]] + hours[dose["before_tp"]]) / 2
                ax.axvline(x, color=figs.INK2, ls="--", lw=1)
        ax.axhline(0, color=figs.AXIS, lw=0.8)
        ax.set_title(figs.LINE_SHORT[line] + f"  ({line})", fontsize=9)
        ax.set_xlabel("hours")
        ax.set_ylim(-1.5, 1.0)
    axes[0][0].set_ylabel(f"cells / cells at T{ref_tp} (log2)")
    axes[0][-1].legend(loc="lower left", fontsize=7)
    fig.suptitle(f"{exp}: detected cells per well relative to T{ref_tp}"
                 + ("" if p is not None else " (no S.A. readout)"), x=0.01, ha="left", fontsize=10)
    path = figs.save(fig, rec.out_dir / f"F3_counts_{exp}.png")
    rec.figure(f"F3_{exp}", path, f"{exp}: detected cells per well (thin) and median (thick) relative to "
               f"T{ref_tp}; dashed line = S.A. dose.", exp=exp)


if __name__ == "__main__":
    main()

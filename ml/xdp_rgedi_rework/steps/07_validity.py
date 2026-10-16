"""Step 07: do the cutoffs hold? (rule R5)

For all 21 line x experiment cases, using the final cutoffs (direct from step 06 where usable, otherwise
transferred in step 08):
  1. drift - the reference live peak at every timepoint, in units of the reference live SD; tiered on the
     drift reached at two or more timepoints (STABLE <= 0.5, MODERATE <= 1, MAJOR > 1) so that a single
     imaging-day spike does not decide the tier; the largest single value and its timepoint are reported; and the share of reference cells called
     dead at every timepoint under rules B, C and D
  2. edge column - XDP10: S.A. wells (column 22) vs untreated column 21, before dosing (T1-T8): difference
     in the live peak in reference-SD units (flag > 0.25)
  3. naturally dying vs S.A.-killed cells - median ratio of reference cells called dead vs S.A. cells called
     dead (rule D), at the S.A. readout
  4. S.A. survivors vs untreated live cells - peak of S.A. cells below the rule-D cutoff vs the reference peak
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

STEP = "07_validity"
plt = figs.plt


def final_cutoffs() -> pd.DataFrame:
    d = pd.read_csv(LOCAL_OUT / "06_fit_thresholds" / "thresholds_direct.csv")
    t = pd.read_csv(LOCAL_OUT / "08_transfer" / "thresholds_transfer.csv")
    d = d[d.usable].assign(source="direct S.A. fit")
    t = t.assign(source="transfer")
    cols = ["exp", "line", "source", "mode_log", "sd_inner", "B_live", "B_dead", "C_live", "D_live"]
    cols += [c for c in ("W90_live", "W90_dead", "W95_live", "W95_dead") if c in d.columns and c in t.columns]
    return pd.concat([d[cols], t[cols]], ignore_index=True)


def main() -> None:
    P = cfg.params()
    V = P["validity"]
    rec = StepRecorder(STEP, LOCAL_OUT / STEP, V)
    cut = final_cutoffs()
    drift_rows, death_rows, nat_rows = [], [], []
    for exp in figs.EXP_ORDER:
        e = cfg.experiments()[exp]
        s = e["sa"]
        d = data.load(exp, columns=["well", "col", "line", "condition", "tp", "log10_ratio", "in_scope"])
        d = d[d.log10_ratio.notna()]
        last_ref_tp = 10 if exp == "XDP1" else int(d.tp.max())
        for r in cut[cut.exp == exp].itertuples():
            ref = d[d.line == r.line]
            ref = ref[ref.tp <= 10] if exp == "XDP1" else ref[ref.condition == "untreated"]
            for tp, g in ref.groupby("tp"):
                x = g.log10_ratio.to_numpy()
                mode_t = mx.kde_mode(x)
                drift_rows.append({"exp": exp, "line": r.line, "tp": tp, "mode_ratio": 10 ** mode_t,
                                   "drift_sd": (mode_t - r.mode_log) / r.sd_inner})
                rules = [("B", r.B_live, r.B_dead), ("C", r.C_live, r.C_live), ("D", r.D_live, r.D_live)]
                rules += [(w, getattr(r, f"{w}_live"), getattr(r, f"{w}_dead")) for w in ("W90", "W95") if hasattr(r, f"{w}_live")]
                for rule, lo, hi in rules:
                    live, dead = np.mean(x < np.log10(lo)), np.mean(x > np.log10(hi))
                    death_rows.append({"exp": exp, "line": r.line, "tp": tp, "rule": rule,
                                       "death_frac": dead / (live + dead), "ambiguous": 1 - live - dead})
            if s["design"] in ("wells", "whole_plate") and s["fit_tps"]:
                fit = s["fit_tps"]
                sa = d[(d.line == r.line) & (d.condition == "sa") & d.tp.isin(fit)].log10_ratio.to_numpy()
                rf = (d[(d.line == r.line) & (d.tp == s["pre_tp"])] if exp == "XDP1" else
                      d[(d.line == r.line) & (d.condition == "untreated") & d.tp.isin(fit)]).log10_ratio.to_numpy()
                late = d[(d.line == r.line) & (d.condition == "untreated") & (d.tp == last_ref_tp)].log10_ratio.to_numpy() \
                    if exp != "XDP1" else rf
                c = np.log10(r.D_live)
                surv = sa[sa < c]
                nat_rows.append({"exp": exp, "line": r.line, "source": r.source,
                                 "sa_dead_median": 10 ** np.median(sa[sa > c]) if (sa > c).any() else np.nan,
                                 "ref_dead_median": 10 ** np.median(late[late > c]) if (late > c).sum() > 20 else np.nan,
                                 "n_ref_dead": int((late > c).sum()),
                                 "survivor_peak": 10 ** mx.kde_mode(surv) if len(surv) > 50 else np.nan,
                                 "ref_peak": 10 ** r.mode_log,
                                 "survivor_shift_sd": (mx.kde_mode(surv) - r.mode_log) / r.sd_inner if len(surv) > 50 else np.nan})
        print(exp, "done", flush=True)
    drift = pd.DataFrame(drift_rows)
    death = pd.DataFrame(death_rows)
    nat = pd.DataFrame(nat_rows)
    drift.to_csv(rec.out_dir / "drift.csv", index=False)
    death.to_csv(rec.out_dir / "reference_death_by_tp.csv", index=False)
    nat.to_csv(rec.out_dir / "natural_vs_sa.csv", index=False)
    def tier_stats(g):
        a = g.assign(ad=g.drift_sd.abs()).sort_values("ad", ascending=False)
        return pd.Series({"max_abs_drift_sd": a.ad.iloc[0], "max_at_tp": int(a.tp.iloc[0]),
                          "sustained_drift_sd": a.ad.iloc[1] if len(a) > 1 else a.ad.iloc[0]})
    dmax = drift.groupby(["exp", "line"]).apply(tier_stats).reset_index()
    # Tier on drift reached at >= 2 timepoints (second-largest), so a single imaging-day spike does not decide it.
    dmax["tier"] = np.where(dmax.sustained_drift_sd <= V["drift_stable_sd"], "STABLE",
                            np.where(dmax.sustained_drift_sd <= V["drift_major_sd"], "MODERATE", "MAJOR"))
    dmax.to_csv(rec.out_dir / "drift_tiers.csv", index=False)
    for r in dmax.itertuples():
        rec.check(f"{r.exp}.{figs.LINE_SHORT[r.line]}.drift", f"{r.exp} {figs.LINE_SHORT[r.line]}: reference live peak "
                  "stays within 0.5 live SD of its value at the reference timepoint (at >= 2 timepoints)", None if r.tier != "STABLE" else True,
                  {"sustained_drift_sd": round(r.sustained_drift_sd, 2), "max_abs_drift_sd": round(r.max_abs_drift_sd, 2),
                   "max_at_tp": int(r.max_at_tp), "tier": r.tier})

    # Edge column: XDP10 S.A. column 22 vs untreated column 21, T1-T8.
    d10 = data.load("XDP10", columns=["col", "line", "condition", "tp", "log10_ratio", "in_scope"])
    edge = []
    for line in figs.LINE_ORDER:
        g = d10[(d10.line == line) & (d10.tp <= 8)]
        if not len(g):
            continue
        c22 = g[g["col"] == 22].log10_ratio.dropna().to_numpy()
        c21 = g[g["col"] == 21].log10_ratio.dropna().to_numpy()
        if len(c22) < 100 or len(c21) < 100:
            continue
        sd = cut[(cut.exp == "XDP10") & (cut.line == line)].sd_inner.iloc[0]
        diff = (mx.kde_mode(c22) - mx.kde_mode(c21)) / sd
        edge.append({"line": line, "peak_col22": 10 ** mx.kde_mode(c22), "peak_col21": 10 ** mx.kde_mode(c21),
                     "diff_sd": diff, "flag": abs(diff) > V["edge_flag_sd"]})
        rec.check(f"XDP10.{figs.LINE_SHORT[line]}.edge", f"XDP10 {figs.LINE_SHORT[line]}: before dosing, the S.A. "
                  f"column (22) has the same live peak as column 21 (within {V['edge_flag_sd']} SD)",
                  None if abs(diff) > V["edge_flag_sd"] else True, round(diff, 2))
    pd.DataFrame(edge).to_csv(rec.out_dir / "edge_xdp10.csv", index=False)

    tiers = dmax.tier.value_counts().to_dict()
    rec.summary = {"drift_tiers": tiers, "major_drift": [f"{e} {figs.LINE_SHORT[l]}" for e, l in
                                                         zip(dmax[dmax.tier == "MAJOR"].exp, dmax[dmax.tier == "MAJOR"].line)],
                   "edge": pd.DataFrame(edge).round(3).to_dict("records"),
                   "survivor_shift_sd_range": [float(nat.survivor_shift_sd.min()), float(nat.survivor_shift_sd.max())]}
    plot(death, drift, rec)
    rec.write()
    print(rec.summary)


def plot(death, drift, rec):
    exps = [e for e in figs.EXP_ORDER if e in set(death.exp)]
    fig, axes = plt.subplots(2, len(exps), figsize=(3.0 * len(exps), 5.6), sharey="row", squeeze=False)
    for j, exp in enumerate(exps):
        hours = cfg.read_timepoints(exp).set_index("tp").hours
        for line, g in drift[drift.exp == exp].groupby("line"):
            g = g.sort_values("tp")
            axes[0][j].plot(g.tp.map(hours).to_numpy(), g.drift_sd.to_numpy(), color=figs.LINE_COLOR[line],
                            marker="o", ms=3, label=figs.LINE_SHORT[line])
        axes[0][j].axhspan(-0.5, 0.5, color=figs.AMBIG, alpha=0.5, lw=0)
        axes[0][j].set_title(exp, fontsize=9)
        for (line, rule), g in death[(death.exp == exp) & (death.rule == "D")].groupby(["line", "rule"]):
            g = g.sort_values("tp")
            axes[1][j].plot(g.tp.map(hours).to_numpy(), 100 * g.death_frac.to_numpy(), color=figs.LINE_COLOR[line],
                            marker="o", ms=3)
        axes[1][j].set_xlabel("hours")
    axes[0][0].set_ylabel("live peak shift\n(reference live SDs)")
    axes[1][0].set_ylabel("reference cells called\ndead, rule D (%)")
    axes[0][-1].legend(fontsize=7)
    fig.suptitle("Untreated (reference) wells over time: drift of the live peak (grey band = ±0.5 SD) and the share "
                 "called dead under rule D", x=0.01, ha="left", fontsize=10)
    path = figs.save(fig, rec.out_dir / "F8_drift.png")
    rec.figure("F8", path, "Top: how far the live peak of untreated wells (XDP1: all wells before dosing) moves from "
               "its position at the S.A. readout, in units of the live SD. Bottom: share of those cells called dead "
               "under the final rule-D cutoffs at each timepoint.")


if __name__ == "__main__":
    main()

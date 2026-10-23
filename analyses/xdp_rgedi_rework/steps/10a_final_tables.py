"""Step 10a: final threshold tables for both tracks (decision D20) and MRID-format threshold files.

Track 1 = rule D (single cutoff), with rule C as its sensitivity; track 2 = rule B (two cutoffs, cells in
between dropped). Every line x experiment gets a direct cutoff (step 06, usable cases) or a transferred
one (step 08). Writes:
  thresholds/thresholds_all.csv                        one row per line x experiment, all rules + context
  thresholds/<track>/<EXP>-RGEDI_thr_df.csv           MRID format: cell_line, live_threshold, dead_threshold
and a comparison figure of the cutoffs and of T1 death in untreated wells under each rule.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from lib import cfg, figs  # noqa: E402
from lib.paths import LOCAL_OUT, RAW_CACHE  # noqa: E402
from lib.provenance import StepRecorder  # noqa: E402

STEP = "10a_final_tables"
plt = figs.plt
TRACKS = {"window90_primary": ("W90_live", "W90_dead"), "window95_sensitivity": ("W95_live", "W95_dead"),   # D28
          "track1_ruleD": ("D_live", "D_live"), "track1_ruleC_sensitivity": ("C_live", "C_live"),
          "track2_ruleB": ("B_live", "B_dead")}


def main() -> None:
    rec = StepRecorder(STEP, LOCAL_OUT / STEP)
    d = pd.read_csv(LOCAL_OUT / "06_fit_thresholds" / "thresholds_direct.csv")
    t = pd.read_csv(LOCAL_OUT / "08_transfer" / "thresholds_transfer.csv")
    death = pd.read_csv(LOCAL_OUT / "07_validity" / "reference_death_by_tp.csv")
    tiers = pd.read_csv(LOCAL_OUT / "07_validity" / "drift_tiers.csv")
    audit = pd.read_csv(RAW_CACHE / "MRID-rework/audit/threshold_audit_ALL.csv")
    cols = ["B_live", "B_dead", "C_live", "D_live", "W90_live", "W90_dead", "W95_live", "W95_dead"]
    du = d[d.usable].copy()
    du["source"] = "direct S.A. fit"
    t = t.copy()
    t["source"] = "transfer"
    keep = ["exp", "line", "source", "mode_ratio", "sd_inner"] + cols + [f"{c}_{q}" for c in cols for q in ("lo", "hi")]
    allr = pd.concat([du.reindex(columns=keep + ["few_wells", "dead_mu_ratio", "dead_w"]),
                      t.reindex(columns=keep + ["reference", "sources"])], ignore_index=True)
    allr["order"] = allr.exp.map({e: i for i, e in enumerate(figs.EXP_ORDER)}) * 10 + allr.line.map(
        {l: i for i, l in enumerate(figs.LINE_ORDER)})
    allr = allr.sort_values("order").drop(columns="order")
    aud = audit.rename(columns={"experiment": "exp", "cell_line": "line"})[
        ["exp", "line", "live_threshold", "dead_threshold", "firstrun_live_global", "firstrun_dead_global",
         "baseline_death_pct"]].rename(columns={"live_threshold": "sept_live", "dead_threshold": "sept_dead",
                                                "firstrun_live_global": "firstrun_live", "firstrun_dead_global": "firstrun_dead",
                                                "baseline_death_pct": "sept_T1_death_pct"})
    allr = allr.merge(aud, on=["exp", "line"], how="left")
    t1 = death[death.tp == 1].pivot_table(index=["exp", "line"], columns="rule", values="death_frac")
    t1.columns = [f"T1_death_{c}" for c in t1.columns]
    lastd = death.loc[death.groupby(["exp", "line", "rule"]).tp.transform("max") == death.tp].pivot_table(
        index=["exp", "line"], columns="rule", values="death_frac")
    lastd.columns = [f"last_death_{c}" for c in lastd.columns]
    allr = allr.merge(t1.reset_index(), on=["exp", "line"], how="left").merge(lastd.reset_index(), on=["exp", "line"], how="left")
    allr = allr.merge(tiers, on=["exp", "line"], how="left")
    out = rec.out_dir / "thresholds"
    out.mkdir(exist_ok=True)
    allr.to_csv(out / "thresholds_all.csv", index=False)
    bad_files = []
    for track, (lo, hi) in TRACKS.items():
        (out / track).mkdir(exist_ok=True)
        for exp, g in allr.groupby("exp"):
            thr = pd.DataFrame({"cell_line": g.line.to_numpy(), "live_threshold": g[lo].to_numpy(),
                                "dead_threshold": g[hi].to_numpy()})
            path = out / track / f"{cfg.experiments()[exp]['expt_name']}_thr_df.csv"
            thr.to_csv(path, index=False)
            back = pd.read_csv(path)
            in_scope = [l for l in cfg.experiments()[exp]["lines"] if l in cfg.lines_in_scope()]
            if (list(back.columns) != ["cell_line", "live_threshold", "dead_threshold"]
                    or sorted(back.cell_line) != sorted(in_scope) or (back.live_threshold > back.dead_threshold).any()
                    or back.isna().any().any()):
                bad_files.append(str(path))
    rec.check("final.thr_df_files", "Every MRID threshold file has exactly cell_line, live_threshold, dead_threshold, "
              "one row per in-scope line, no missing values, live <= dead", not bad_files, bad_files)
    rec.summary = {
        "cases": len(allr), "direct": int((allr.source == "direct S.A. fit").sum()), "transfer": int((allr.source == "transfer").sum()),
        "D_range": [float(allr.D_live.min()), float(allr.D_live.max())],
        "C_range": [float(allr.C_live.min()), float(allr.C_live.max())],
        "B_live_range": [float(allr.B_live.min()), float(allr.B_live.max())],
        "B_dead_range": [float(allr.B_dead.min()), float(allr.B_dead.max())],
        "T1_death_range": {r: [float(allr[f"T1_death_{r}"].min()), float(allr[f"T1_death_{r}"].max())]
                           for r in ["B", "C", "D", "W90", "W95"] if f"T1_death_{r}" in allr},
        "W90_range": [float(allr.W90_live.min()), float(allr.W90_dead.max())],
        "W90_fold_width": [float((allr.W90_dead / allr.W90_live).min()), float((allr.W90_dead / allr.W90_live).max())],
        "sept_T1_death_values": sorted(set(allr.sept_T1_death_pct.dropna().round(1).tolist())),
    }
    rec.check("final.one_row_per_case", "Final table has exactly one row per in-scope line x experiment (21)",
              len(allr) == 21 and not allr.duplicated(["exp", "line"]).any(), len(allr))
    rec.check("final.live_le_dead", "Every track's live cutoff <= dead cutoff", bool((allr.B_live <= allr.B_dead).all()), None)
    plot(allr, rec)
    rec.write()
    print(allr[["exp", "line", "source", "D_live", "C_live", "B_live", "B_dead", "firstrun_live", "firstrun_dead",
                "sept_live", "sept_dead", "T1_death_D", "T1_death_C", "T1_death_B"]].round(3).to_string())


def plot(a, rec):
    a = a.copy()
    a["label"] = a.exp + " " + a.line.map(figs.LINE_SHORT)
    y = np.arange(len(a))[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(11, 0.33 * len(a) + 1.4), gridspec_kw={"width_ratios": [1.6, 1]})
    ax = axes[0]
    for i, r in enumerate(a.itertuples()):
        yy = y[i]
        ax.plot([r.B_live, r.B_dead], [yy, yy], color=figs.AMBIG, lw=6, solid_capstyle="butt")
        ax.plot(r.D_live, yy, "o", color=figs.LIVE, ms=7, mfc=figs.LIVE if r.source != "transfer" else "none", mew=1.6)
        ax.plot(r.C_live, yy, "s", color=figs.DEAD, ms=5, mfc=figs.DEAD if r.source != "transfer" else "none", mew=1.4)
        if not np.isnan(r.sept_dead):
            ax.plot(r.sept_dead, yy, "|", color=figs.INK2, ms=10, mew=1.4)
        if not np.isnan(r.firstrun_dead):
            ax.plot(r.firstrun_dead, yy, "x", color=figs.MUTED, ms=6)
    ax.set_xscale("log")
    ax.set_yticks(y)
    ax.set_yticklabels(a.label.to_numpy(), fontsize=8)
    ax.set_xlabel("GEDI ratio cutoff")
    ax.grid(axis="y", visible=False)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=figs.LIVE, marker="o", ls="", label="track 1: rule D"),
         Line2D([], [], color=figs.DEAD, marker="s", ls="", label="rule C (sensitivity)"),
         Line2D([], [], color=figs.AMBIG, lw=6, label="track 2: rule B live-dead band"),
         Line2D([], [], color=figs.INK2, marker="|", ls="", ms=10, label="September dead cutoff"),
         Line2D([], [], color=figs.MUTED, marker="x", ls="", label="first-run dead cutoff"),
         Line2D([], [], color=figs.INK2, marker="o", mfc="none", ls="", label="open = transferred")]
    ax.legend(handles=h, fontsize=7, loc="lower right")
    ax = axes[1]
    w = 0.25
    for k, (rule, color) in enumerate((("B", figs.MUTED), ("C", figs.DEAD), ("D", figs.LIVE))):
        ax.barh(y + (k - 1) * w, 100 * a[f"T1_death_{rule}"].to_numpy(), height=w, color=color, label=f"rule {rule}")
    ax.axvline(9.1, color=figs.INK2, ls="--", lw=1.2, label="September (9.1% by construction)")
    ax.set_yticks(y)
    ax.set_yticklabels([])
    ax.set_xlabel("untreated cells called dead at T1 (%)")
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(axis="y", visible=False)
    fig.suptitle("Final cutoffs per line x experiment, both tracks, with the earlier runs for comparison", x=0.01,
                 ha="left", fontsize=10)
    path = figs.save(fig, rec.out_dir / "F11_final.png")
    rec.figure("F11", path, "Left: final cutoffs (filled = fitted directly on S.A.-killed cells, open = transferred). "
               "Right: share of untreated cells (XDP1: all wells) called dead at T1 under each rule, against the 9.1% "
               "that the September rule produced by construction.")


if __name__ == "__main__":
    main()

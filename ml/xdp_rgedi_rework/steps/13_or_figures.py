"""Step 13: odds-ratio figures and tables (decision D05 / D23), from the MRID runs of step 12.

  (a) every line vs each control line (CTR-33114.C and CTR-33113.2I) over time, 95% CI - per threshold set
  (b) pairwise line x line odds ratios at each harmonised timepoint (primary set)
  (c) per-experiment time-averaged odds ratios with the pooled estimate (forest) - per threshold set
  (d) track comparison: XDP lines vs each control under every threshold set
Odds ratio > 1 = higher odds of death in the comparison line than in the reference line.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from lib import figs, sets  # noqa: E402
from lib.stats import dl_meta  # noqa: E402
from lib.paths import LOCAL_OUT  # noqa: E402
from lib.provenance import StepRecorder  # noqa: E402

STEP = "13_or_figures"
plt = figs.plt
SETS = ["W90", "W95", "D_sd", "D_abs", "C_sd", "C_abs", "B", "D_sd_drift", "W90_shared", "D_sd_shared"]
SET_LABEL = {**{k: k for k in SETS}, **sets.LABELS,
             "D_sd_shared": "rule D single cutoff, experiments with all four lines (XDP5, XDP10, XDP11)"}
SET_COLOR = {"W90": "#4a3aa7", "W95": "#9a8fd6", "D_sd": "#2a78d6", "D_abs": "#1baf7a", "C_sd": "#eb6834",
             "C_abs": "#eda100", "B": "#52514e", "D_sd_drift": "#898781", "W90_shared": "#4a3aa7", "D_sd_shared": "#2a78d6"}
MAIN = (sets.PRIMARY, "D_sd", "B")             # drawn solid and thick in the track comparison
REFS = ["CTR-33114.C", "CTR-33113.2I"]
S = figs.LINE_SHORT


SHARED = ["XDP5-RGEDI", "XDP10-RGEDI", "XDP11-RGEDI"]       # the experiments that contain all four lines


def cutoff_sources() -> dict:
    t = pd.read_csv(LOCAL_OUT / "10a_final_tables" / "thresholds" / "thresholds_all.csv")
    return {(cfg_name(e), l): ("direct" if s_.startswith("direct") else "transfer") for e, l, s_ in zip(t.exp, t.line, t.source)}


def cfg_name(exp: str) -> str:
    return f"{exp}-RGEDI"


def load(set_name: str):
    o = LOCAL_OUT / "12_logodds" / set_name
    if not (o / "OR_all_pairs.csv").exists():
        return None
    orr = pd.read_csv(o / "OR_all_pairs.csv").merge(pd.read_csv(o / "time_index_hours.csv"), on="Time")
    fix = lambda df: df.rename(columns=lambda c: c.split(".")[0])  # noqa: E731  (R writes OR.Estimate etc.)
    return {"or": orr, "exp": fix(pd.read_csv(o / "OR_time_averaged_per_experiment.csv")),
            "pooled": fix(pd.read_csv(o / "OR_time_averaged_pooled.csv")),
            "re": fix(pd.read_csv(o / "OR_rederived_per_timepoint.csv"))}


def fig_a(name, R, rec):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, ref in zip(axes, REFS):
        d = R["or"][R["or"].Reference == ref]
        for cmp, g in d.groupby("Compare"):
            g = g.sort_values("Hour")
            c = figs.LINE_COLOR[cmp]
            ax.fill_between(g.Hour.to_numpy(), g.lower.to_numpy(), g.upper.to_numpy(), color=c, alpha=0.15, lw=0)
            ax.plot(g.Hour.to_numpy(), g.OR.to_numpy(), color=c, marker="o", ms=4, lw=2)
            ax.annotate(S[cmp], (g.Hour.iloc[-1], g.OR.iloc[-1]), xytext=(4, 0), textcoords="offset points",
                        fontsize=8, color=figs.INK2, va="center")
        ax.axhline(1, color=figs.INK2, ls="--", lw=1)
        ax.set_yscale("log")
        figs.or_axis(ax, "y")
        ax.set_title(f"vs {S[ref]} ({ref})", fontsize=9)
        ax.set_xlabel("hours (harmonised across experiments)")
    axes[0].set_ylabel("odds ratio of death (95% CI)")
    fig.suptitle(f"{SET_LABEL[name]}: odds of death relative to each control line, all experiments combined (MRID)",
                 x=0.01, ha="left", fontsize=10)
    path = figs.save(fig, rec.out_dir / f"Fa_or_vs_controls_{name}.png")
    rec.figure(f"Fa_{name}", path, f"{SET_LABEL[name]}: MRID odds ratio of death for each line relative to CTR-33114.C "
               "(left) and CTR-33113.2I (right) at each harmonised timepoint, with 95% CI. Above 1 = more death than the "
               "control line.", set=name)


def fig_b(name, R, rec):
    d = R["or"]
    lines = figs.LINE_ORDER
    hours = sorted(d.Hour.unique())
    n = len(hours)
    cols = 6
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.0 * cols, 2.0 * rows + 0.6), squeeze=False)
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f0efec", "#e34948"])
    vmax = 1.0
    for k, h in enumerate(hours):
        ax = axes[k // cols][k % cols]
        M = np.full((4, 4), np.nan)
        for i, a in enumerate(lines):          # row = comparison
            for j, b in enumerate(lines):      # column = reference
                r = d[(d.Hour == h) & (d.Compare == a) & (d.Reference == b)]
                if len(r):
                    M[i, j] = np.log2(r.OR.iloc[0])
        ax.imshow(M, cmap=cmap, vmin=-vmax, vmax=vmax)
        for i in range(4):
            for j in range(4):
                if not np.isnan(M[i, j]):
                    ax.text(j, i, f"{2 ** M[i, j]:.2f}", ha="center", va="center", fontsize=6.5, color=figs.INK)
        ax.set_xticks(range(4))
        ax.set_yticks(range(4))
        ax.set_xticklabels([S[l] for l in lines], fontsize=7)
        ax.set_yticklabels([S[l] for l in lines], fontsize=7)
        ax.set_title(f"{int(h)} h", fontsize=8)
        ax.grid(False)
    for k in range(n, rows * cols):
        axes[k // cols][k % cols].set_axis_off()
    fig.suptitle(f"{SET_LABEL[name]}: pairwise odds ratio of death, row line vs column line (blue < 1 < red; colour "
                 "scale capped at 0.5-2)", x=0.01, ha="left", fontsize=10)
    path = figs.save(fig, rec.out_dir / f"Fb_pairwise_{name}.png")
    scope = ("XDP5, XDP10 and XDP11 combined" if name.endswith("_shared") else "all experiments combined")
    rec.figure(f"Fb_{name}", path, f"{SET_LABEL[name]}: odds ratio of death of the row line relative to the column line "
               f"at each harmonised timepoint ({int(min(hours))}-{int(max(hours))} h; MRID, {scope}).", set=name)


def fig_c(name, R, rec):
    e, p = R["exp"], R["pooled"][R["pooled"].model == "line x experiment"]
    pm = R["pooled"][R["pooled"].model == "MRID structure"]
    src = cutoff_sources()
    est = [c for c in e.columns if c.startswith("OR")][0]
    lo = [c for c in e.columns if c.startswith("lower")][0]
    hi = [c for c in e.columns if c.startswith("upper")][0]
    comps = [l for l in figs.LINE_ORDER]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.0), sharex=True)
    for ax, ref in zip(axes, REFS):
        y, labels = 0, []
        for cmp in comps:
            if cmp == ref:
                continue
            g = e[(e.Reference == ref) & (e.Compare == cmp)].copy()
            g["o"] = g.Experiment.str.replace("-RGEDI", "").map({x: i for i, x in enumerate(figs.EXP_ORDER)})
            for r in g.sort_values("o").itertuples():
                v, l_, h_ = getattr(r, est), getattr(r, lo), getattr(r, hi)
                mixed = src.get((r.Experiment, cmp)) != src.get((r.Experiment, ref))
                ax.plot([l_, h_], [y, y], color=figs.LINE_COLOR[cmp], lw=1.6)
                ax.plot(v, y, "o", color=figs.LINE_COLOR[cmp], ms=5, mfc="none" if mixed else figs.LINE_COLOR[cmp], mew=1.5)
                labels.append(f"{S[cmp]}  {r.Experiment.replace('-RGEDI', '')}" + ("  *" if mixed else ""))
                y -= 1
            q = p[(p.Reference == ref) & (p.Compare == cmp)]
            q2 = pm[(pm.Reference == ref) & (pm.Compare == cmp)]
            if len(q):
                v, l_, h_ = q[est].iloc[0], q[lo].iloc[0], q[hi].iloc[0]
                ax.fill([l_, v, h_, v], [y, y + 0.35, y, y - 0.35], color=figs.LINE_COLOR[cmp])
                if len(q2):
                    v2, l2, h2 = q2[est].iloc[0], q2[lo].iloc[0], q2[hi].iloc[0]
                    ax.plot([l2, v2, h2, v2, l2], [y - 0.55, y - 0.2, y - 0.55, y - 0.9, y - 0.55], color=figs.LINE_COLOR[cmp], lw=1)
                labels.append(f"{S[cmp]}  pooled")
                y -= 1.6
        ax.axvline(1, color=figs.INK2, ls="--", lw=1)
        ax.set_xscale("log")
        figs.or_axis(ax, "x")
        ys = []
        yy = 0
        for lab in labels:
            ys.append(yy)
            yy -= 1.6 if lab.endswith("pooled") else 1
        ax.set_yticks(ys)
        ax.set_yticklabels(labels, fontsize=7.5)
        ax.grid(axis="y", visible=False)
        ax.set_title(f"vs {S[ref]}", fontsize=9)
        ax.set_xlabel("time-averaged odds ratio of death (95% CI)")
    fig.suptitle(f"{SET_LABEL[name]}: per-experiment odds ratios (dots; open/* = the two lines' cutoffs come from different "
                 "sources) and pooled estimates (filled = allows line x experiment variation; outline = MRID structure)",
                 x=0.01, ha="left", fontsize=9)
    path = figs.save(fig, rec.out_dir / f"Fc_forest_{name}.png")
    rec.figure(f"Fc_{name}", path, f"{SET_LABEL[name]}: time-averaged odds ratio of death of each line relative to each "
               "control line, per experiment (well random effect), and pooled over experiments with experiment and well random "
               "effects (outline diamond, MRID structure) and with an added line x experiment random effect (filled diamond). "
               "Reproducible differences sit on the same side of 1 in every experiment.", set=name)


def fig_d(all_R, rec):
    comps = ["XDP-33109.2B", "XDP-33363.D"]
    fig, axes = plt.subplots(len(comps), 2, figsize=(11, 6.6), sharex=True, sharey=True)
    for i, cmp in enumerate(comps):
        for j, ref in enumerate(REFS):
            ax = axes[i][j]
            for name, R in all_R.items():
                g = R["or"][(R["or"].Reference == ref) & (R["or"].Compare == cmp)].sort_values("Hour")
                ax.plot(g.Hour.to_numpy(), g.OR.to_numpy(), color=SET_COLOR[name], lw=2.2 if name in MAIN else 1.3,
                        ls="-" if name in MAIN else "--", label=SET_LABEL[name])
            ax.axhline(1, color=figs.INK2, ls=":", lw=1)
            ax.set_yscale("log")
            figs.or_axis(ax, "y")
            ax.set_title(f"{S[cmp]} vs {S[ref]}", fontsize=9)
            if i == len(comps) - 1:
                ax.set_xlabel("hours")
            if j == 0:
                ax.set_ylabel("odds ratio of death")
    axes[0][1].legend(fontsize=7, loc="upper left")
    fig.suptitle("Do the conclusions depend on the threshold track? XDP lines vs each control line under every "
                 "threshold set", x=0.01, ha="left", fontsize=10)
    path = figs.save(fig, rec.out_dir / "Fd_track_comparison.png")
    rec.figure("Fd", path, "Odds ratio of death of each XDP line relative to each control line over time, under every "
               "threshold set (point estimates). Lines that track each other mean the conclusion does not depend on the "
               "threshold choice.")


def main() -> None:
    rec = StepRecorder(STEP, LOCAL_OUT / STEP)
    all_R = {n: R for n in SETS if (R := load(n)) is not None}
    src = cutoff_sources()
    rows = []
    for name, R in all_R.items():
        fig_a(name, R, rec)
        fig_c(name, R, rec)
        if name == sets.PRIMARY:
            fig_b(name, R, rec)
        x = R["re"].merge(R["or"], on=["Reference", "Compare", "Time"], suffixes=("_re", ""))
        dev = float(np.abs(np.log(x["OR_re"]) - np.log(x.OR)).max()) if len(x) else np.nan
        rec.check(f"{name}.mrid_rederived", f"{name}: MRID per-timepoint odds ratios equal an independent re-derivation "
                  "from named coefficients for every ordered line pair (max |log ratio| < 0.01)", bool(dev < 0.01),
                  {"max_abs_log_ratio": dev, "n": len(x), "n_mrid": len(R["or"])})
        e = R["exp"]
        for (ref, cmp), ge in e.groupby(["Reference", "Compare"]):
            pq = R["pooled"][(R["pooled"].Reference == ref) & (R["pooled"].Compare == cmp)].set_index("model")
            same = ge[[src.get((x_, cmp)) == src.get((x_, ref)) for x_ in ge.Experiment]]
            row = {"set": name, "Reference": ref, "Compare": cmp, "n_experiments": len(ge),
                   "experiments_ci_above_1": int((ge.lower > 1).sum()), "experiments_ci_below_1": int((ge.upper < 1).sum())}
            for spec, tag in (("MRID structure", "mridpool"), ("line x experiment", "lxe")):
                if spec in pq.index:
                    row.update({f"{tag}_OR": pq.loc[spec, "OR"], f"{tag}_lo": pq.loc[spec, "lower"], f"{tag}_hi": pq.loc[spec, "upper"]})
            for tag, sub in (("meta_all", ge), ("meta_shared", ge[ge.Experiment.isin(SHARED)]),
                             ("meta_noXDP1", ge[ge.Experiment != "XDP1-RGEDI"]), ("meta_samesource", same)):
                m = dl_meta(sub.OR, sub.lower, sub.upper)
                row.update({f"{tag}_OR": m["OR"], f"{tag}_lo": m["lo"], f"{tag}_hi": m["hi"], f"{tag}_k": m["k"],
                            f"{tag}_I2": m["I2"]})
            rows.append(row)
    comp = pd.DataFrame(rows)
    comp.to_csv(rec.out_dir / "pooled_or_by_set.csv", index=False)
    shared = f"{sets.PRIMARY}_shared"
    if shared in all_R:
        fig_b(shared, all_R[shared], rec)
    fig_d({k: v for k, v in all_R.items() if not k.endswith("_shared")}, rec)
    rec.summary = {"sets": list(all_R)}
    rec.write()
    cols = ["set", "Reference", "Compare", "mridpool_OR", "lxe_OR", "lxe_lo", "lxe_hi", "meta_all_OR", "meta_all_lo",
            "meta_all_hi", "meta_all_I2", "meta_shared_OR", "meta_noXDP1_OR", "meta_samesource_OR", "meta_samesource_k"]
    print(comp[[c for c in cols if c in comp]].round(2).to_string())


if __name__ == "__main__":
    main()

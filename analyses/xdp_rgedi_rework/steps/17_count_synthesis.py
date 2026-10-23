"""Step 17: combine the S.A. capture fractions, the MRID odds ratios and the live-cell retention models (D26, D27).

  1. S.A.-corrected odds ratio per experiment: log OR_corr = log OR + log(c_ref / c_cmp). Its variance adds the
     bootstrap variance of log c for both lines (percentile-based SE). Experiments or lines without a capture
     fraction take the ratio pooled over experiments (DerSimonian-Laird): its between-experiment variance is added
     per experiment, and its estimation variance once to the pooled result (shared input, not independent).
     Versions: "D26 as frozen" (capture usable by the D26 rule), "D27 gated" (only capture fractions that pass the
     validity gate of Step 15), and two readout sensitivities of the gated version (XDP10 read at T9 instead of
     T10; XDP11 with baseline T11). Uncorrected and corrected per-experiment odds ratios are pooled with
     DerSimonian-Laird tau^2 and Hartung-Knapp intervals (k = 3-6 experiments).
  2. Decision outcomes: D26 as frozen, and the D27 proposal.
  3. Agreement between odds ratios and live-cell loss, per pair x experiment, on matched windows (Step 16b).
  4. Labelled sensitivity: visible dead share divided by the line's capture fraction (gated cases only).
Figures 17.1 (forest) and 17.2 (visible dead share vs dropout-corrected cumulative death, where defensible).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

from lib import figs, sets  # noqa: E402
from lib.sets import PRIMARY  # noqa: E402
from lib.paths import LOCAL_OUT  # noqa: E402
from lib.provenance import StepRecorder  # noqa: E402
from lib.stats import dl_pool  # noqa: E402

STEP = "17_count_synthesis"
plt = figs.plt
S = figs.LINE_SHORT
PAIRS = [("XDP-33109.2B", "CTR-33114.C"), ("XDP-33363.D", "CTR-33114.C"), ("CTR-33113.2I", "CTR-33114.C"),
         ("XDP-33109.2B", "CTR-33113.2I"), ("XDP-33363.D", "CTR-33113.2I")]
SETS = list(dict.fromkeys([PRIMARY, "D_sd", "B"]))
EXP_OF = {f"{e}-RGEDI": e for e in figs.EXP_ORDER}
# version -> (basis column in capture.csv, readout variant per experiment; default "primary")
VERSIONS = {"D26 as frozen": ("usable", {}), "D27 gated": ("valid", {}),
            "D27 gated, XDP10 read at T9": ("valid", {"XDP10": "t9"}),
            "D27 gated, XDP11 baseline T11": ("valid", {"XDP11": "pre11"})}


def capture_tables(set_: str, basis: str, readout: dict) -> pd.DataFrame:
    """Capture fraction per experiment x line for one threshold set, readout choice and usability basis."""
    cap = pd.read_csv(LOCAL_OUT / "15_sa_capture" / "capture.csv")
    cap = cap[cap.set == set_]
    cap = cap[[v == readout.get(e, "primary") for e, v in zip(cap.exp, cap.variant)]].copy()
    cap["use"] = cap[basis].astype(bool)
    cap["var_log_c"] = cap.log_c_se ** 2
    return cap.set_index(["exp", "line"])


def pooled_ratio(cap: pd.DataFrame, cmp_: str, ref: str, tau2_fallback: float = None) -> dict:
    """log(c_ref/c_cmp) pooled over experiments where both lines have a capture fraction in use.

    With a single contributing experiment the between-experiment spread cannot be estimated; it is then borrowed
    from the gated primary version (``tau2_fallback``), or the ratio is marked not estimable.
    """
    ys, ses = [], []
    for exp in sorted({e for e, _ in cap.index}):
        if (exp, cmp_) in cap.index and (exp, ref) in cap.index:
            a, b = cap.loc[(exp, cmp_)], cap.loc[(exp, ref)]
            if a.use and b.use:
                ys.append(np.log(b.c / a.c))
                ses.append(np.sqrt(a.var_log_c + b.var_log_c))
    m = dl_pool(ys, ses)
    src = f"pooled over {m['k']} experiments"
    if m["k"] >= 2:
        tau2 = m["tau2"]
    elif m["k"] == 1 and tau2_fallback is not None and np.isfinite(tau2_fallback):
        tau2, src = tau2_fallback, "pooled over 1 experiment (between-experiment spread borrowed from the gated version)"
    else:
        return {"log_ratio": np.nan, "var": np.nan, "shared_var": 0.0, "k": m["k"], "shared": True,
                "source": "not estimable"}
    return {"log_ratio": m["mu"], "var": tau2, "shared_var": m["se_mu"] ** 2, "k": m["k"], "shared": True,
            "source": src}


def correction(cap: pd.DataFrame, exp: str, cmp_: str, ref: str, pooled: dict) -> dict:
    """log(c_ref / c_cmp) and its variance for one experiment, or the pooled value when not measurable there."""
    if (exp, cmp_) in cap.index and (exp, ref) in cap.index:
        a, b = cap.loc[(exp, cmp_)], cap.loc[(exp, ref)]
        if a.use and b.use:
            return {"log_ratio": float(np.log(b.c / a.c)), "var": float(a.var_log_c + b.var_log_c), "shared": False,
                    "shared_var": 0.0, "source": "own S.A. wells"}
    return pooled


def corrected_or(log_or: float, se_or: float, corr: dict):
    """Corrected log OR, its SE for pooling (shared input excluded; added once in the pool), and its full SE."""
    est = log_or + corr["log_ratio"]
    se = np.sqrt(se_or ** 2 + corr["var"])
    return est, se, np.sqrt(se ** 2 + corr.get("shared_var", 0.0))


def pool_row(g: pd.DataFrame, ycol: str, scol: str, shared: pd.Series = None, shared_var: float = 0.0,
             scol_full: str = None) -> dict:
    """Random-effects pool with HK interval, the largest weight, and a leave-one-experiment-out range."""
    g = g[np.isfinite(g[ycol]) & np.isfinite(g[scol])]
    sh = None if shared is None else shared.loc[g.index].to_numpy()
    m = dl_pool(g[ycol], g[scol], sh, shared_var)
    sf = g[scol_full or scol]
    out = {"k": m["k"], "OR": np.exp(m["mu"]), "lo": np.exp(m["lo"]), "hi": np.exp(m["hi"]), "I2": m["I2"],
           "pred_lo": np.exp(m["pred_lo"]), "pred_hi": np.exp(m["pred_hi"]),
           "exp_ci_above_1": int((g[ycol] - 1.96 * sf > 0).sum()), "exp_ci_below_1": int((g[ycol] + 1.96 * sf < 0).sum()),
           "k_own": int((~g.shared.astype(bool)).sum()) if shared is not None else int(m["k"])}
    if m["k"] >= 2:
        w = 1 / (g[scol].to_numpy() ** 2 + (m["tau2"] if np.isfinite(m["tau2"]) else 0.0))
        i = int(np.argmax(w))
        out.update({"max_weight": float(w[i] / w.sum()), "max_weight_exp": g.exp.iloc[i]})
        loo = []
        for j in range(len(g)):
            keep = np.arange(len(g)) != j
            mj = dl_pool(g[ycol].to_numpy()[keep], g[scol].to_numpy()[keep], None if sh is None else sh[keep], shared_var)
            loo.append((g.exp.iloc[j], np.exp(mj["mu"]), np.exp(mj["lo"]), np.exp(mj["hi"])))
        lo_ = min(loo, key=lambda t: t[1])
        hi_ = max(loo, key=lambda t: t[1])
        out.update({"loo_min": lo_[1], "loo_min_without": lo_[0], "loo_max": hi_[1], "loo_max_without": hi_[0],
                    "loo_excl1": int(sum((t[2] > 1) or (t[3] < 1) for t in loo)), "loo": "; ".join(
                        f"without {e}: {o:.2f} [{a:.2f}, {b:.2f}]" for e, o, a, b in loo)})
    return out


def decision_outcome(inv: pd.DataFrame, set_: str, basis: str) -> dict:
    iv = inv[(inv.set == set_) & (inv.variant == "primary") & (inv.basis == basis)]
    eq = {r.exp: (None if pd.isna(r.equal_visibility) else str(r.equal_visibility) == "True") for r in iv.itertuples()}
    failing = [e for e in figs.EXP_ORDER if eq.get(e) is False]
    return {"equal": [e for e in figs.EXP_ORDER if eq.get(e) is True], "failing": failing,
            "no_readout": [e for e in figs.EXP_ORDER if e in eq and eq[e] is None],
            "equal_visibility_all": not failing}


def main() -> None:
    rec = StepRecorder(STEP, LOCAL_OUT / STEP)
    inv = pd.read_csv(LOCAL_OUT / "15_sa_capture" / "capture_invariance.csv")
    c2 = pd.read_csv(LOCAL_OUT / "16b_retention_fe" / "c2_per_experiment.csv")
    c2p = pd.read_csv(LOCAL_OUT / "16b_retention_fe" / "c2_pooled.csv")
    c1p = LOCAL_OUT / "16_retention" / "c1_pairs.csv"
    c1 = pd.read_csv(c1p) if c1p.exists() else None
    j1 = pd.read_csv(LOCAL_OUT / "16b_retention_fe" / "j1.csv")
    per_exp, pooled_rows, agree_rows = [], [], []
    # unit check of the correction: a capture ratio of exactly 1 with no variance leaves the odds ratio unchanged
    e0, s0, _ = corrected_or(np.log(1.7), 0.2, {"log_ratio": 0.0, "var": 0.0})
    rec.check("correction_identity", "Corrected odds ratio equals the uncorrected one when the capture ratio is 1",
              bool(np.isclose(e0, np.log(1.7)) and np.isclose(s0, 0.2)))
    for set_ in SETS:
        orx = pd.read_csv(LOCAL_OUT / "12_logodds" / set_ / "OR_time_averaged_per_experiment.csv").rename(
            columns=lambda c: c.split(".")[0])
        cap_g = capture_tables(set_, "valid", {})
        tau_fb = {(c, r): pooled_ratio(cap_g, c, r)["var"] for c, r in PAIRS}
        for version, (basis, readout) in VERSIONS.items():
            cap = capture_tables(set_, basis, readout)
            for cmp_, ref in PAIRS:
                pr = pooled_ratio(cap, cmp_, ref, tau_fb[(cmp_, ref)])
                for r in orx[(orx.Compare == cmp_) & (orx.Reference == ref)].itertuples():
                    exp = EXP_OF[r.Experiment]
                    lo_, se_ = np.log(r.OR), (np.log(r.upper) - np.log(r.lower)) / (2 * 1.96)
                    cr = correction(cap, exp, cmp_, ref, pr)
                    est, se, se_full = corrected_or(lo_, se_, cr)
                    per_exp.append({"set": set_, "version": version, "exp": exp, "Compare": cmp_, "Reference": ref,
                                    "log_OR": lo_, "se_OR": se_, "OR": r.OR, "OR_lo": r.lower, "OR_hi": r.upper,
                                    "capture_ratio_ref_over_cmp": float(np.exp(cr["log_ratio"])),
                                    "capture_ratio_source": cr["source"], "shared": cr["shared"],
                                    "shared_var": cr["shared_var"], "log_OR_corr": est, "se_corr": se,
                                    "se_corr_full": se_full, "OR_corr": np.exp(est),
                                    "OR_corr_lo": np.exp(est - 1.96 * se_full), "OR_corr_hi": np.exp(est + 1.96 * se_full)})
    pe = pd.DataFrame(per_exp)
    for (set_, cmp_, ref), g0 in pe.groupby(["set", "Compare", "Reference"]):
        g = g0[g0.version == "D26 as frozen"]
        pooled_rows.append({"set": set_, "Compare": cmp_, "Reference": ref, "kind": "MRID (uncorrected)",
                            **pool_row(g, "log_OR", "se_OR")})
        for version, g in g0.groupby("version"):
            sv = float(g[g.shared].shared_var.iloc[0]) if g.shared.any() else 0.0
            pooled_rows.append({"set": set_, "Compare": cmp_, "Reference": ref, "kind": f"S.A.-corrected, {version}",
                                **pool_row(g, "log_OR_corr", "se_corr", g.shared, sv, scol_full="se_corr_full")})
    po = pd.DataFrame(pooled_rows)
    s13 = pd.read_csv(LOCAL_OUT / "13_or_figures" / "pooled_or_by_set.csv")
    mm = po[po.kind == "MRID (uncorrected)"].merge(s13, on=["set", "Compare", "Reference"])
    dmax = float((mm.OR - mm.meta_all_OR).abs().max())
    rec.check("uncorrected_matches_step13", "Uncorrected pooled odds ratios (point estimates) equal Step 13's "
              "random-effects meta-analysis", bool(len(mm) == len(po[po.kind == "MRID (uncorrected)"]) and dmax < 1e-9),
              {"pairs": int(len(mm)), "max_abs_diff": dmax})

    # ---- agreement between odds ratios and live-cell retention -------------------------------------------------
    # primary: the odds ratio on exactly the retention window (same wells and timepoints, Step 16b);
    # secondary: MRID's time-averaged odds ratio over its harmonised hours (whole run). The 21 pair x experiment
    # cases are not independent (pairs share lines), so no p-value is attached to the agreement.
    orw = pd.read_csv(LOCAL_OUT / "16b_retention_fe" / "or_window.csv")
    pg = pe[pe.version == "D27 gated"]
    for set_ in SETS:
        for cmp_, ref in PAIRS:
            for r in c2[(c2.fit == set_) & (c2.Compare == cmp_) & (c2.Reference == ref)].itertuples():
                w = orw[(orw.fit == set_) & (orw.exp == r.exp) & (orw.Compare == cmp_) & (orw.Reference == ref)]
                m = pg[(pg.set == set_) & (pg.exp == r.exp) & (pg.Compare == cmp_) & (pg.Reference == ref)]
                if not len(w):
                    continue
                w = w.iloc[0]
                row = {"set": set_, "exp": r.exp, "Compare": cmp_, "Reference": ref, "loss_per_day": -r.est,
                       "loss_se": r.se, "log_OR_window": w.est, "se_OR_window": w.se,
                       "sign_agree_window": bool(np.sign(w.est) == np.sign(-r.est)),
                       "both_clear_window": bool(abs(w.est) > 1.96 * w.se and abs(r.est) > r.q * r.se)}
                if len(m):
                    m = m.iloc[0]
                    row.update({"log_OR_mrid": m.log_OR, "log_OR_corr": m.log_OR_corr,
                                "sign_agree_mrid": bool(np.sign(m.log_OR) == np.sign(-r.est)),
                                "sign_agree_corr": bool(np.sign(m.log_OR_corr) == np.sign(-r.est))})
                agree_rows.append(row)
    ag = pd.DataFrame(agree_rows)
    agree = {}
    for set_ in SETS:
        a_ = ag[ag.set == set_]
        clear = a_[a_.both_clear_window]
        mm = a_.dropna(subset=["log_OR_mrid"])
        agree[set_] = {"n": int(len(a_)), "sign_agree": int(a_.sign_agree_window.sum()),
                       "spearman": float(stats.spearmanr(a_.log_OR_window, a_.loss_per_day)[0]),
                       "n_both_clear": int(len(clear)), "sign_agree_both_clear": int(clear.sign_agree_window.sum()),
                       "n_mrid": int(len(mm)), "sign_agree_mrid": int(mm.sign_agree_mrid.sum()),
                       "spearman_mrid": float(stats.spearmanr(mm.log_OR_mrid, mm.loss_per_day)[0]),
                       "sign_agree_corr": int(mm.sign_agree_corr.sum()),
                       "spearman_corr": float(stats.spearmanr(mm.log_OR_corr, mm.loss_per_day)[0]),
                       "agree_exps": sorted({e for e, ok in zip(clear.exp, clear.sign_agree_window) if ok})}

    # ---- C1 (pooled glmer) vs C2 (per-experiment, pooled) -------------------------------------------------
    if c1 is not None:
        for cmp_, ref in PAIRS:
            a = c1[(c1.fit == PRIMARY) & (c1.Compare == cmp_) & (c1.Reference == ref)]
            b = c2p[(c2p.fit == PRIMARY) & (c2p.Compare == cmp_) & (c2p.Reference == ref)]
            if len(a) and len(b):
                same = bool(np.sign(np.log(a.ratio_per_day.iloc[0])) == np.sign(np.log(b.ratio_per_day.iloc[0])))
                rec.check(f"c1_c2_sign.{S[cmp_]}_{S[ref]}", f"Pooled retention {S[cmp_]} vs {S[ref]}: the mixed model "
                          "and the per-experiment pooling agree in direction", same or None,
                          {"c1": round(float(a.ratio_per_day.iloc[0]), 4), "c2": round(float(b.ratio_per_day.iloc[0]), 4)})

    # ---- decision outcomes ------------------------------------------------------------------------------------
    decision = {set_: {"D26 as frozen": decision_outcome(inv, set_, "usable"), "D27 gated": decision_outcome(inv, set_, "valid")}
                for set_ in SETS}

    # ---- labelled sensitivity: visible dead share in untreated wells divided by the line's capture fraction ------
    cap_all = pd.read_csv(LOCAL_OUT / "15_sa_capture" / "capture.csv")
    vc = cap_all[(cap_all.variant == "primary") & cap_all.set.isin(SETS) & cap_all.valid].copy()
    vc["visible_dead_share_ut"] = vc.D1_ut / (vc.L1_ut + vc.D1_ut)
    vc["implied_if_seen_like_SA"] = np.minimum(vc.visible_dead_share_ut / vc.c, 1.0)
    vc[["set", "exp", "line", "t", "visible_dead_share_ut", "c", "implied_if_seen_like_SA"]].to_csv(
        rec.out_dir / "visible_vs_capture.csv", index=False)
    pe.to_csv(rec.out_dir / "or_corrected_per_experiment.csv", index=False)
    po.to_csv(rec.out_dir / "or_pooled_uncorrected_vs_corrected.csv", index=False)
    ag.to_csv(rec.out_dir / "agreement_or_vs_retention.csv", index=False)
    summary = {"decision": decision, "agreement": agree,
               "pooled": {set_: {f"{S[c]} vs {S[r]}": {x.kind: [round(float(x.OR), 3), round(float(x.lo), 3),
                                                                 round(float(x.hi), 3), int(x.exp_ci_above_1),
                                                                 int(x.exp_ci_below_1), int(x.k)] for x in g.itertuples()}
                                  for (c, r), g in po[po.set == set_].groupby(["Compare", "Reference"])} for set_ in SETS},
               "j1_passing": {s: sorted(j1[(j1.set == s) & j1.passes].exp) for s in SETS}}
    rec.summary = summary
    plot_forest(pe, po, c2, c2p, c1, rec)
    plot_cumdeath(rec)
    rec.write()
    print(po[po.set == PRIMARY].round(3).to_string())
    print(summary["decision"], summary["agreement"])
    for c in rec.checks:
        if c["status"] != "PASS":
            print(c["status"], c["id"], str(c["value"])[:300])


def plot_forest(pe, po, c2, c2p, c1, rec) -> None:
    """Figure 17.1: per pair, odds ratios (uncorrected vs S.A.-corrected) and relative live-cell loss, per experiment."""
    fig, axes = plt.subplots(2, len(PAIRS), figsize=(17, 6.4), sharey="row")
    for j, (cmp_, ref) in enumerate(PAIRS):
        ax = axes[0, j]
        g = pe[(pe.set == PRIMARY) & (pe.version == "D27 gated") & (pe.Compare == cmp_) & (pe.Reference == ref)]
        ys = {e: len(figs.EXP_ORDER) - i for i, e in enumerate(figs.EXP_ORDER)}
        for r in g.itertuples():
            y = ys[r.exp]
            ax.errorbar(r.OR, y + 0.15, xerr=[[r.OR - r.OR_lo], [r.OR_hi - r.OR]], fmt="o", color=figs.DEAD, ms=4, capsize=2,
                        label="MRID odds ratio" if r.exp == g.exp.iloc[0] else None)
            ax.errorbar(r.OR_corr, y - 0.15, xerr=[[r.OR_corr - r.OR_corr_lo], [r.OR_corr_hi - r.OR_corr]], fmt="s",
                        mfc="white" if r.capture_ratio_source != "own S.A. wells" else figs.SA, color=figs.SA, ms=4,
                        capsize=2, label="S.A.-corrected" if r.exp == g.exp.iloc[0] else None)
        for k, (kind, col, mk) in enumerate((("MRID (uncorrected)", figs.DEAD, "D"),
                                              ("S.A.-corrected, D27 gated", figs.SA, "D"))):
            p = po[(po.set == PRIMARY) & (po.Compare == cmp_) & (po.Reference == ref) & (po.kind == kind)].iloc[0]
            ax.errorbar(p.OR, 0.15 - 0.3 * k, xerr=[[p.OR - p.lo], [p.hi - p.OR]], fmt=mk, color=col, ms=6, capsize=3)
        alt = po[(po.set == PRIMARY) & (po.Compare == cmp_) & (po.Reference == ref)
                 & po.kind.str.startswith("S.A.-corrected") & (po.kind != "S.A.-corrected, D27 gated")]
        ax.scatter(alt.OR, np.full(len(alt), -0.3), marker="|", s=60, color=figs.SA, zorder=3,
                   label="corrected, other versions" if j == 0 else None)
        ax.set_yticks(list(ys.values()) + [0])
        ax.set_yticklabels(list(ys.keys()) + ["pooled"], fontsize=7.5)
        ax.set_xscale("log")
        figs.or_axis(ax, "x")
        ax.axvline(1, color=figs.INK2, ls=":", lw=1)
        ax.set_title(f"{S[cmp_]} vs {S[ref]}")
        ax2 = axes[1, j]
        g2 = c2[(c2.fit == PRIMARY) & (c2.Compare == cmp_) & (c2.Reference == ref)]
        for r in g2.itertuples():
            if r.exp not in ys:
                continue
            v, lo, hi = np.exp(-7 * r.est), np.exp(-7 * (r.est + 1.96 * r.se)), np.exp(-7 * (r.est - 1.96 * r.se))
            ax2.errorbar(v, ys[r.exp], xerr=[[v - lo], [hi - v]], fmt="o", color=figs.LIVE, ms=4, capsize=2)
        if c1 is not None:
            p = c1[(c1.fit == PRIMARY) & (c1.Compare == cmp_) & (c1.Reference == ref)]
            if len(p):
                p = p.iloc[0]
                v, lo, hi = p.ratio_per_day ** -7, p.upper ** -7, p.lower ** -7
                ax2.errorbar(v, 0, xerr=[[v - lo], [hi - v]], fmt="D", color=figs.LIVE, ms=6, capsize=3)
        ax2.set_yticks(list(ys.values()) + [0])
        ax2.set_yticklabels(list(ys.keys()) + ["pooled"], fontsize=7.5)
        ax2.set_xscale("log")
        from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
        ax2.xaxis.set_major_locator(FixedLocator([0.5, 0.67, 0.8, 1, 1.25, 1.5, 2]))
        ax2.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax2.xaxis.set_minor_formatter(NullFormatter())
        ax2.axvline(1, color=figs.INK2, ls=":", lw=1)
    axes[0, 0].set_ylabel("odds ratio of visible death")
    axes[1, 0].set_ylabel("control ÷ line retention per week")
    for ax in axes[1]:
        ax.set_xlabel("> 1: first line worse")
    fig.suptitle(f"Figure 17.1  Each line vs a control line, {sets.label(PRIMARY)}: MRID odds ratio, S.A.-corrected odds ratio (D27 gated; "
                 "filled: own S.A. wells, open: pooled capture), and live-cell loss", x=0.01, ha="left", fontsize=10)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.legend(h, l, loc="upper right", ncol=3, fontsize=7.5, bbox_to_anchor=(0.995, 0.965), frameon=False)
    path = figs.save(fig, rec.out_dir / "F17_forest.png")
    rec.figure("F17_1", path, "Top: MRID time-averaged odds ratio per experiment (orange) and after correcting for the "
               "lines' S.A. capture fractions (violet, D27 gated version; open = capture ratio pooled over experiments "
               "because this experiment has no valid S.A. readout for the pair). Pooled values: random-effects with "
               "Hartung-Knapp intervals; ticks below the pooled row = the other corrected versions (D26 as frozen, XDP10 "
               "read at T9, XDP11 baseline T11). Bottom: control-to-line retention ratio per week, per experiment (slope "
               "through the timepoint contrasts, lack-of-fit inflated) and pooled (mixed model). Top row covers MRID's "
               "whole run, bottom row the counting window from k0; the matched-window comparison is in 17.2. Every axis: "
               "above 1 means the first line does worse.")


def plot_cumdeath(rec) -> None:
    """Figure 17.2: visible dead share vs dropout-corrected cumulative death, where the calibration test passed."""
    p = LOCAL_OUT / "16b_retention_fe" / "cum_death.csv"
    f = pd.read_csv(p) if p.exists() and p.stat().st_size > 5 else pd.DataFrame()
    f = f[f.set == PRIMARY] if len(f) else f
    if not len(f):
        return
    exps = [e for e in figs.EXP_ORDER if e in set(f.exp)]
    fig, axes = plt.subplots(1, len(exps), figsize=(max(9.0, 4.2 * len(exps)), 3.2), squeeze=False, sharey=True)
    for ax, exp in zip(axes[0], exps):
        g = f[f.exp == exp]
        for line in figs.LINE_ORDER:
            s = g[g.line == line].sort_values("hours")
            if not len(s):
                continue
            col = figs.LINE_COLOR[line]
            ax.plot(s.hours.to_numpy(), 100 * s.visible_dead_share.to_numpy(), color=col, lw=1.2, ls=":")
            ax.plot(s.hours.to_numpy(), 100 * s.cum_death_since_k0.to_numpy(), color=col, lw=2, label=S[line])
            ax.annotate(S[line], (s.hours.iloc[-1], 100 * s.cum_death_since_k0.iloc[-1]), xytext=(3, 0),
                        textcoords="offset points", fontsize=7, color=col, va="center")
            ax.fill_between(s.hours.to_numpy(), 100 * s.cum_death_lo.to_numpy(), 100 * s.cum_death_hi.to_numpy(),
                            color=col, alpha=0.12, lw=0)
        ax.set_title(exp)
        ax.set_xlabel("hours")
    axes[0][0].set_ylabel("% of cells")
    axes[0][0].legend(fontsize=7)
    fig.suptitle(f"Figure 17.2  {', '.join(exps)}: visible dead share (dotted) vs cumulative death implied by V (solid, 95% "
                 f"band); {sets.label(PRIMARY)}", x=0.01, ha="left", fontsize=10)
    path = figs.save(fig, rec.out_dir / "F17_cumdeath.png")
    rec.figure("F17_2", path, "Dotted: share of detected cells called dead at each timepoint (what MRID models). Solid: "
               "share of cells that have died since k0, from the cumulative dead/live odds divided by the visibility "
               "window V estimated in the calibration test. Shown only where that test is not rejected and Step 15 does "
               "not find unequal capture between lines; XDP11 passes the test but is left out because its S.A. wells show "
               "unequal capture, and XDP1 has no S.A. readout to check this. V is estimated from the live-count slopes, so "
               "the solid curves are not independent of the counts.")


if __name__ == "__main__":
    main()

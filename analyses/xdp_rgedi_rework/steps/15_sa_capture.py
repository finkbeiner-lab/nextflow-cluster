"""Step 15: how much S.A.-induced death GEDI sees - capture fraction per line x experiment (decision D26).

For one line in one experiment, S.A. and untreated wells share the plate, imaging, maturation and detection, so
the untreated wells give the counts the S.A. wells would have had without dosing (difference in differences):
    Lhat = L_SA(t0) * L_UT(t) / L_UT(t0)                       expected live cells without the dose
    K    = Lhat - L_SA(t)                                      live cells removed by S.A.
    Dhat = Lhat * O_SA(t0) * O_UT(t) / O_UT(t0)                expected dead cells without the dose (O = D/L)
    E    = D_SA(t) - Dhat                                      excess cells visible as GEDI-dead
    c    = E / K                                               capture fraction: share of removed cells seen dead
c < 1 means part of the S.A. death has already left detection (GFP lost). Equal c across lines is consistent with
equal visibility of S.A.-killed cells at this readout; it depends on readout lag and on when each line's cells died, so
it is a proxy, not a measurement, of how long dead cells stay visible. Validity gate (D27, proposed after review): a
capture fraction is used only if the S.A. wells also lost detected cells against the same counterfactual
(N_SA(t) / Nhat with 95% upper bound < 1) and fewer than 2.5% of bootstrap replicates have c <= 0; under a single
cutoff (track 1) c > 1 is otherwise only possible when the counterfactual for detected cells failed. Sums are pooled
over wells; uncertainty is a well bootstrap within line x condition (B = 2000);
XDP10 (2 S.A. wells per line) draws its S.A. counts as Poisson (flag FEW_WELLS). Placebo: the same estimator
from T1 to the pre-dose timepoint (no dose yet; expect K ~ 0).
Figures 15.1 (removed fraction and capture fraction) and 15.2 (capture vs GFP brightness).
"""

import sys
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import statsmodels.api as sm  # noqa: E402

from lib import cfg, data, figs, livedead, sets  # noqa: E402
from lib.sets import PRIMARY  # noqa: E402
from lib.paths import LOCAL_OUT  # noqa: E402
from lib.provenance import StepRecorder  # noqa: E402
from lib.counts import capture_estimate  # noqa: E402
from lib.stats import dl_pool  # noqa: E402

STEP = "15_sa_capture"
plt = figs.plt
MIN_WELLS_BOOT = 3
YMAX15 = 2.5


def variants(exp: str, P: dict) -> list:
    """(variant, t0, t) comparisons for one experiment: primary, sensitivity timepoints, alternative baseline, placebo."""
    s = cfg.experiments()[exp]["sa"]
    t0, t = s["pre_tp"], P["capture_tps"][exp]
    out = [("primary", t0, t)]
    out += [(f"t{tt}", t0, tt) for tt in P.get("capture_sensitivity_tps", {}).get(exp, [])]
    if s.get("alt_pre_tp"):
        out.append((f"pre{s['alt_pre_tp']}", s["alt_pre_tp"], t))
    out.append(("placebo", 1, t0))
    return out


COLS = ["L0", "L1", "D0", "D1", "N0", "N1"]


def well_table(cnt: pd.DataFrame, line: str, t0: int, t: int, min_live: int) -> pd.DataFrame:
    """Per-well L/D/N at t0 and t for one line (S.A. and untreated), after the D26 QC rules."""
    c = cnt[(cnt.line == line) & cnt.tp.isin([t0, t])]
    w = c.pivot_table(index=["well", "condition"], columns="tp", values=["L", "D", "N", "missing"], aggfunc="first")
    w.columns = [f"{a}{b}" for a, b in w.columns]
    w = w.reset_index()
    keep = (~w[f"missing{t0}"].astype(bool)) & (~w[f"missing{t}"].astype(bool)) & (w[f"L{t0}"] >= min_live)
    w = w[keep].rename(columns={f"L{t0}": "L0", f"L{t}": "L1", f"D{t0}": "D0", f"D{t}": "D1", f"N{t0}": "N0", f"N{t}": "N1"})
    return w[["well", "condition"] + COLS]


def detected_ratio(s: np.ndarray, u: np.ndarray) -> float:
    """Detected cells in S.A. wells after dosing relative to the no-dose counterfactual (difference in differences)."""
    return s[5] / (s[4] * u[5] / u[4])


def boot(sa_w: pd.DataFrame, ut_w: pd.DataFrame, B: int, seed: int) -> pd.DataFrame:
    """Well bootstrap (Poisson draws of the S.A. counts when fewer than 3 S.A. wells)."""
    rng = np.random.default_rng(seed)
    S, U = sa_w[COLS].to_numpy(float), ut_w[COLS].to_numpy(float)
    few = len(S) < MIN_WELLS_BOOT
    out = np.empty((B, 4))
    for b in range(B):
        u = U[rng.integers(0, len(U), len(U))].sum(0)
        s = rng.poisson(S).sum(0).astype(float) if few else S[rng.integers(0, len(S), len(S))].sum(0)
        r = capture_estimate(s[:4], u[:4])
        out[b] = (r["killed_frac"], r["E"] / r["Lhat"], r["c"], detected_ratio(s, u))
    return pd.DataFrame(out, columns=["killed_frac", "excess_dead_frac", "c", "det_ratio"])


def seed_for(*parts) -> int:
    return (cfg.params()["seed"] + zlib.crc32("|".join(map(str, parts)).encode())) % (2 ** 32)


def main() -> None:
    P = cfg.params()["counts"]
    rec = StepRecorder(STEP, LOCAL_OUT / STEP, params=P)
    cnt_all = pd.read_csv(LOCAL_OUT / "14_count_tables" / "counts_long.csv")
    rows, reps, pairs, inv = [], [], [], []
    for exp in P["capture_tps"]:
        for name in livedead.SETS:
            cnt = cnt_all[(cnt_all.exp == exp) & (cnt_all.set == name)]
            for var, t0, t in variants(exp, P):
                for line in figs.LINE_ORDER:
                    if line not in set(cnt[cnt.condition == "sa"].line):
                        continue
                    w = well_table(cnt, line, t0, t, P["min_baseline_live"])
                    sa_w, ut_w = w[w.condition == "sa"], w[w.condition == "untreated"]
                    if len(sa_w) == 0 or len(ut_w) < 2:
                        continue
                    ss, us = sa_w[COLS].sum().to_numpy(float), ut_w[COLS].sum().to_numpy(float)
                    pt = capture_estimate(ss[:4], us[:4])
                    bt = boot(sa_w, ut_w, P["B_capture"], seed_for(exp, name, var, line))
                    pos = bt.c[bt.c > 0]
                    lq = np.log(pos).quantile([0.025, 0.975]) if len(pos) > 10 else None
                    r = {"exp": exp, "set": name, "variant": var, "line": line, "t0": t0, "t": t,
                         "n_wells_sa": len(sa_w), "n_wells_ut": len(ut_w), "few_wells": len(sa_w) < MIN_WELLS_BOOT,
                         **{f"{k}_{c}": int(sa_w[k].sum()) if c == "sa" else int(ut_w[k].sum())
                            for c in ("sa", "ut") for k in ("L0", "L1", "D0", "D1")},
                         **{k: float(v) for k, v in pt.items()},
                         "excess_dead_frac": float(pt["E"] / pt["Lhat"]),
                         "killed_lo": float(bt.killed_frac.quantile(0.025)), "killed_hi": float(bt.killed_frac.quantile(0.975)),
                         "excess_lo": float(bt.excess_dead_frac.quantile(0.025)),
                         "excess_hi": float(bt.excess_dead_frac.quantile(0.975)),
                         "c_lo": float(bt.c.quantile(0.025)), "c_hi": float(bt.c.quantile(0.975)),
                         "frac_c_nonpos": float((bt.c <= 0).mean()),
                         "log_c_sd": float(np.log(pos).std(ddof=1)) if len(pos) > 10 else np.nan,
                         "log_c_se": float((lq.iloc[1] - lq.iloc[0]) / (2 * 1.96)) if lq is not None else np.nan,
                         "det_ratio": float(detected_ratio(ss, us)), "det_lo": float(bt.det_ratio.quantile(0.025)),
                         "det_hi": float(bt.det_ratio.quantile(0.975))}
                    r["usable"] = bool(var != "placebo" and r["killed_frac"] >= P["min_killed_frac"] and r["killed_lo"] > 0
                                       and r["c"] > 0)
                    r["valid"] = bool(r["usable"] and r["det_hi"] < 1 and r["frac_c_nonpos"] < 0.025)
                    rows.append(r)
                    if var != "placebo":
                        reps.append(bt.assign(exp=exp, set=name, variant=var, line=line, rep=np.arange(len(bt))))
    res = pd.DataFrame(rows)
    rep = pd.concat(reps, ignore_index=True)
    res.to_csv(rec.out_dir / "capture.csv", index=False)
    rep.to_csv(rec.out_dir / "capture_boot.csv", index=False)
    stale = rec.out_dir / "capture_boot_primary.csv"
    if stale.exists():
        stale.unlink()                                    # replaced by capture_boot.csv (all readout variants)

    # ---- equal visibility across lines within an experiment ------------------------------------------------
    # basis "usable" = D26 as frozen; basis "valid" = with the D27 validity gate
    alpha = P["invariance_alpha"]
    for basis in ("usable", "valid"):
        for (exp, name, var), g in res[res.variant != "placebo"].groupby(["exp", "set", "variant"]):
            u = g[g[basis]]
            lines = [l for l in figs.LINE_ORDER if l in set(u.line)]
            pair_ok = []
            rr = rep[(rep.exp == exp) & (rep.set == name) & (rep.variant == var)]
            for i, a_ in enumerate(lines):
                for b_ in lines[i + 1:]:
                    ra, rb = rr[rr.line == a_].c.to_numpy(), rr[rr.line == b_].c.to_numpy()
                    ok_ = (ra > 0) & (rb > 0)
                    lo, hi = np.percentile(np.log(ra[ok_]) - np.log(rb[ok_]), [2.5, 97.5])
                    est = float(np.log(u.set_index("line").loc[a_].c / u.set_index("line").loc[b_].c))
                    inc = bool(lo <= 0 <= hi)
                    pair_ok.append(inc)
                    pairs.append({"basis": basis, "exp": exp, "set": name, "variant": var, "line_A": a_, "line_B": b_,
                                  "log_ratio": est, "ratio": float(np.exp(est)), "lo": float(np.exp(lo)),
                                  "hi": float(np.exp(hi)), "ci_includes_1": inc})
            q = dl_pool(np.log(u.c.to_numpy()), u.log_c_se.to_numpy()) if len(u) >= 2 else {"Q": np.nan, "Q_p": np.nan}
            inv.append({"basis": basis, "exp": exp, "set": name, "variant": var, "n_lines": len(u),
                        "lines_excluded": ", ".join(figs.LINE_SHORT[l] for l in g[~g[basis]].line),
                        "Q": q["Q"], "Q_p": q["Q_p"], "all_pairs_include_1": bool(all(pair_ok)) if pair_ok else np.nan,
                        "equal_visibility": bool(all(pair_ok) and q["Q_p"] >= alpha) if len(u) >= 2 else np.nan})
    pairs, inv = pd.DataFrame(pairs), pd.DataFrame(inv)
    pairs.to_csv(rec.out_dir / "capture_pairs.csv", index=False)
    inv.to_csv(rec.out_dir / "capture_invariance.csv", index=False)

    # ---- does GFP brightness predict capture? -----------------------------------------------------------------
    gfp = []
    for exp in P["capture_tps"]:
        d = data.load(exp, columns=["line", "condition", "tp", "gfp", "in_scope"])
        t0 = cfg.experiments()[exp]["sa"]["pre_tp"]
        m = d[(d.condition == "untreated") & (d.tp == t0)].groupby("line").gfp.median()
        gfp += [{"exp": exp, "line": l, "median_gfp_ut_t0": float(v)} for l, v in m.items()]
    gfp = pd.DataFrame(gfp)
    mod = []
    for name in livedead.SETS:
        u = res[(res.set == name) & (res.variant == "primary") & res.valid].merge(gfp, on=["exp", "line"])
        if len(u) < 6:
            continue
        X = pd.get_dummies(u.exp, drop_first=True).astype(float)
        X["log_gfp"] = np.log(u.median_gfp_ut_t0)
        X = sm.add_constant(X)
        fit = sm.WLS(np.log(u.c), X, weights=1 / u.log_c_se ** 2).fit()
        lo, hi = fit.conf_int().loc["log_gfp"]
        mod.append({"set": name, "n": len(u), "slope_log_c_per_log_gfp": float(fit.params["log_gfp"]),
                    "lo": float(lo), "hi": float(hi), "p": float(fit.pvalues["log_gfp"])})
    mod = pd.DataFrame(mod)
    gfp.to_csv(rec.out_dir / "median_gfp.csv", index=False)
    mod.to_csv(rec.out_dir / "gfp_moderator.csv", index=False)

    # ---- checks, summary, figures --------------------------------------------------------------------------
    plc = res[res.variant == "placebo"]
    for name in livedead.SETS:
        p_ = plc[plc.set == name]
        rec.check(f"{name}.placebo_within_noise", f"{name}: placebo (T1 to the pre-dose timepoint, no dose) removed-cell "
                  "fraction has a 95% CI including 0 for every line x experiment",
                  bool(((p_.killed_lo <= 0) & (p_.killed_hi >= 0)).all()) or None,
                  {"outside": p_[(p_.killed_lo > 0) | (p_.killed_hi < 0)][["exp", "line", "killed_frac", "killed_lo",
                                                                           "killed_hi"]].round(3).values.tolist()})
    prim = res[(res.variant == "primary")]
    rec.check("boot_c_positive", "Capture fraction bootstrap replicates are positive for usable cases (<2.5% non-positive)",
              bool((prim[prim.usable].frac_c_nonpos < 0.025).all()),
              prim[prim.usable][["exp", "set", "line", "frac_c_nonpos"]].query("frac_c_nonpos >= 0.025").values.tolist())
    rec.check("gate_excludes", "Validity gate (D27): cases usable under D26 but excluded because detected cells did not "
              "fall or c is unstable", None, prim[prim.usable & ~prim.valid][["exp", "set", "line", "c", "det_ratio", "det_hi",
                                                                             "frac_c_nonpos"]].round(3).values.tolist())
    summary = {"sets": {}}
    for name in livedead.SETS:
        pr = prim[prim.set == name]
        summary["sets"][name] = {
            "usable_cases": int(pr.usable.sum()), "valid_cases": int(pr.valid.sum()), "cases": int(len(pr)),
            "c_by_exp": {e: {figs.LINE_SHORT[l]: [round(c, 3), round(lo, 3), round(hi, 3)]
                             for l, c, lo, hi in zip(g.line, g.c, g.c_lo, g.c_hi)} for e, g in pr.groupby("exp")},
            "killed_by_exp": {e: {figs.LINE_SHORT[l]: round(k, 3) for l, k in zip(g.line, g.killed_frac)}
                              for e, g in pr.groupby("exp")},
        }
        for basis in ("usable", "valid"):
            iv = inv[(inv.set == name) & (inv.variant == "primary") & (inv.basis == basis)]
            summary["sets"][name][basis] = {
                "equal_visibility": {r.exp: r.equal_visibility for r in iv.itertuples()},
                "all_experiments_equal_visibility": bool(iv.equal_visibility.dropna().astype(bool).all()),
                "experiments_failing": [r.exp for r in iv.itertuples()
                                        if pd.notna(r.equal_visibility) and not bool(r.equal_visibility)]}
    rec.summary = summary
    plot_capture(res, rec)
    plot_gfp(res, gfp, mod, rec)
    rec.write()
    print(inv[inv.variant == "primary"].round(3).to_string())
    print(prim[prim.set.isin([PRIMARY, "D_sd", "B"])][["exp", "set", "line", "killed_frac", "c", "c_lo", "c_hi", "det_ratio", "det_hi",
                                               "usable", "valid"]].round(3).to_string())
    for c in rec.checks:
        if c["status"] != "PASS":
            print(c["status"], c["id"], str(c["value"])[:300])


def plot_capture(res: pd.DataFrame, rec: StepRecorder) -> None:
    """Figure 15.1: removed-cell fraction (S.A. vs placebo) and capture fraction per line x experiment, track 1."""
    exps = list(cfg.params()["counts"]["capture_tps"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    xt, xl = [], []
    x = 0
    for exp in exps:
        for line in figs.LINE_ORDER:
            g = res[(res.exp == exp) & (res.set == PRIMARY) & (res.line == line)]
            p, pl = g[g.variant == "primary"], g[g.variant == "placebo"]
            if not len(p):
                continue
            col = figs.LINE_COLOR[line]
            p, pl = p.iloc[0], pl.iloc[0] if len(pl) else None
            axes[0].errorbar(x, p.killed_frac, [[p.killed_frac - p.killed_lo], [p.killed_hi - p.killed_frac]], fmt="o",
                             color=col, ms=5, capsize=2)
            if pl is not None:
                axes[0].errorbar(x + 0.3, pl.killed_frac, [[pl.killed_frac - pl.killed_lo], [pl.killed_hi - pl.killed_frac]],
                                 fmt="o", mfc="white", color=col, ms=4, capsize=2)
            mk = "o" if p.valid else ("s" if p.usable else "x")
            top = min(p.c_hi, YMAX15)
            axes[1].errorbar(x, min(p.c, YMAX15), [[max(min(p.c, YMAX15) - p.c_lo, 0)], [max(top - min(p.c, YMAX15), 0)]],
                             fmt=mk, color=col, mfc=col if p.valid else "white", ms=5, capsize=2)
            if p.c_hi > YMAX15:
                axes[1].annotate(f"↑ {p.c_hi:.1f}", (x, YMAX15), xytext=(0, -2), textcoords="offset points", ha="center",
                                 va="top", fontsize=6.5, color=col)
            xt.append(x)
            xl.append(f"{figs.LINE_SHORT[line]}{' *' if p.few_wells else ''}")
            x += 1
        x += 0.8
    for ax in axes:
        ax.set_xticks(xt)
        ax.set_xticklabels(xl, fontsize=7)
        ax.axhline(0, color=figs.AXIS, lw=0.8)
    pos = 0
    for exp in exps:
        n = len(res[(res.exp == exp) & (res.set == PRIMARY) & (res.variant == "primary")])
        for ax in axes:
            ax.text(pos + (n - 1) / 2, 1.01, exp, transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                    fontsize=8, color=figs.INK2, fontweight="bold")
        pos += n + 0.8
    axes[0].set_ylabel("live cells removed by S.A. (share)")
    axes[0].set_title("A. Share of live cells removed (filled: S.A.; open: placebo, no dose)", loc="left", fontsize=9,
                      pad=16)
    axes[1].axhline(1, color=figs.INK2, ls=":", lw=1)
    axes[1].axhspan(1, YMAX15, color=figs.AMBIG, alpha=0.5, lw=0, zorder=0)
    axes[1].text(0.99, 0.97, "above 1: not a capture fraction (detected cells did not fall)", transform=axes[1].transAxes,
                 ha="right", va="top", fontsize=7, color=figs.INK2)
    axes[1].set_ylabel("capture fraction c")
    axes[1].set_title("B. Share of removed cells seen as GEDI-dead (filled: passes the validity gate)", loc="left",
                      fontsize=9, pad=16)
    axes[1].set_ylim(-0.2, YMAX15)
    fig.suptitle(f"Figure 15.1  S.A. as a calibration: how much of the death GEDI sees ({sets.label(PRIMARY)}; "
                 "* = 2 S.A. wells, Poisson bootstrap)", x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    path = figs.save(fig, rec.out_dir / "F15_capture.png")
    rec.figure("F15_1", path, "A: share of live cells removed by S.A. at the S.A. readout (XDP3 T10, XDP5 T9, XDP10 T10, "
               "XDP11 T13; filled) and in the placebo comparison from T1 to the pre-dose timepoint (open). B: capture "
               "fraction, the share of removed cells still visible as GEDI-dead; 1 = all seen, below 1 = the rest has "
               "dropped out of detection. Filled points pass the validity gate (D27: detected cells fell); open squares are "
               "usable under D26 but fail the gate; crosses are not usable. Intervals cut at the top are labelled with "
               "their upper end. 95% bootstrap intervals; " + sets.label(PRIMARY) + " cutoffs.")


def plot_gfp(res: pd.DataFrame, gfp: pd.DataFrame, mod: pd.DataFrame, rec: StepRecorder) -> None:
    """Figure 15.2: capture fraction vs median GFP of untreated cells, per line x experiment."""
    u = res[(res.set == PRIMARY) & (res.variant == "primary") & res.valid].merge(gfp, on=["exp", "line"])
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    marker = {"XDP3": "o", "XDP5": "D", "XDP10": "s", "XDP11": "^"}
    for exp, g in u.groupby("exp"):
        g = g.sort_values("median_gfp_ut_t0")
        ax.plot(g.median_gfp_ut_t0.to_numpy(), g.c.to_numpy(), color=figs.AXIS, lw=1, zorder=1)
        for r in g.itertuples():
            ax.errorbar(r.median_gfp_ut_t0, r.c, [[r.c - r.c_lo], [r.c_hi - r.c]], fmt=marker.get(exp, "o"),
                        color=figs.LINE_COLOR[r.line], ms=5, capsize=2, zorder=2, lw=1)
        last = g.iloc[-1]
        ax.annotate(exp, (last.median_gfp_ut_t0, last.c), xytext=(6, 0), textcoords="offset points", fontsize=7,
                    color=figs.INK2, va="center")
    for l in figs.LINE_ORDER:
        ax.plot([], [], "o", color=figs.LINE_COLOR[l], label=figs.LINE_SHORT[l])
    ax.legend(fontsize=7, loc="lower right", ncol=4)
    ax.set_xscale("log")
    ax.set_yscale("log")
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
    ax.xaxis.set_major_locator(FixedLocator([2000, 3000, 5000, 10000, 15000]))
    ax.yaxis.set_major_locator(FixedLocator([0.25, 0.5, 1, 2]))
    for a_ in (ax.xaxis, ax.yaxis):
        a_.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        a_.set_minor_formatter(NullFormatter())
    ax.set_xlabel("median GFP of untreated cells before dosing (log)")
    ax.set_ylabel("capture fraction c (log)")
    m = mod[mod.set == PRIMARY]
    if len(m):
        m = m.iloc[0]
        ax.set_title(f"Within-experiment slope {m.slope_log_c_per_log_gfp:.2f} [{m.lo:.2f}, {m.hi:.2f}], p = {m.p:.2g}",
                     loc="left", fontsize=8.5)
    fig.suptitle("Figure 15.2  Is capture related to how bright a line's GFP is? (cases passing the validity gate)",
                 x=0.01, ha="left", fontsize=10)
    path = figs.save(fig, rec.out_dir / "F15_gfp.png")
    rec.figure("F15_2", path, "Capture fraction (95% interval) against the line's median GFP before dosing, for cases "
               "passing the validity gate; grey lines join the lines of one experiment (marker shape = experiment). The slope is from a weighted regression of log capture on log GFP with experiment "
               "effects (only differences within an experiment count).")


if __name__ == "__main__":
    main()

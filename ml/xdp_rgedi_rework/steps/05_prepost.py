"""Step 05: Julia's pre- vs post-S.A. comparison (rule R6) - computed before any threshold.

Per experiment x line, S.A. wells at the last pre-dose timepoint vs each post-dose timepoint,
with untreated (UT) wells over the same interval as the time control. The well is the unit:
  * per-well median and 90th percentile of log10 ratio, and per-well median log10 GFP and RFP
    (to show whether a ratio rise comes from RFP gain or GFP loss); change pre->post per well
  * Delta_SA, Delta_UT (mean over wells), DiD = Delta_SA - Delta_UT, reported as a fold change
    10**DiD with a well-bootstrap 95% CI. With fewer than 3 S.A. wells the bootstrap CI is not
    estimable and is left blank (flag FEW_WELLS)
  * permutation p-value over S.A./UT well labels: exact (all labellings enumerated) when there are
    <= 50,000 labellings, otherwise Monte Carlo
  * shift function: change in each decile, S.A. and UT, paired well bootstrap (same wells pre/post)
  * cell totals: per-well log(N_post/N_pre); DiD reported as an S.A./UT count ratio
  * 'baseline' = primary pre-dose timepoint, or the alternative (alt_pre_tp in the config) used as a
    sensitivity where the primary baseline is a flagged artefact day (XDP5 T7, XDP11 T12)
XDP1 (no UT wells): plate-mean change compared with same-length changes inside T1-T10 that do not
touch the T7 artefact ('pre-dose comparison intervals'). XDP8: no post-dose data.
"""

import itertools
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from lib import cfg, data, figs  # noqa: E402
from lib.paths import LOCAL_OUT  # noqa: E402
from lib.provenance import StepRecorder  # noqa: E402

STEP = "05_prepost"
plt = figs.plt
DECILES = np.round(np.arange(0.1, 1.0, 0.1), 1)
STATS = {"med": "median log10 ratio", "q90": "90th pct log10 ratio", "gfp": "median log10 GFP",
         "rfp": "median log10 RFP"}
EXACT_MAX = 50_000
MIN_WELLS_CI = 3


def timing(exp: str):
    """Return (pre_tp, post_tps, alt_pre_tp, source) for an experiment."""
    s = cfg.experiments()[exp]["sa"]
    if exp == "XDP3":
        dec = {d["id"]: d for d in cfg.decisions()}["D12"]
        src = "inferred onset T10 (D12 " + ("approved" if dec["status"].startswith("approved") else "proposed") + ")"
        return 9, [10], None, src
    if s["design"] == "none_imaged":
        return None, [], None, "no S.A. readout"
    return s["pre_tp"], s["post_tps"], s.get("alt_pre_tp"), "dosing record"


def well_stats(d: pd.DataFrame) -> pd.DataFrame:
    g = d.groupby(["well", "condition", "tp"])
    out = pd.DataFrame({"med": g.log10_ratio.median(), "q90": g.log10_ratio.quantile(0.9),
                        "gfp": g.lg.median(), "rfp": g.lr.median(), "n": g.size()}).reset_index()
    return out


def perm_test(sa: np.ndarray, ut: np.ndarray, n_perm: int, rng):
    allc = np.concatenate([sa, ut])
    n, k = len(allc), len(sa)
    obs = abs(sa.mean() - ut.mean())
    total = math.comb(n, k)
    if total <= EXACT_MAX:
        s_all = allc.sum()
        hits = 0
        for idx in itertools.combinations(range(n), k):
            m1 = allc[list(idx)].mean()
            m2 = (s_all - allc[list(idx)].sum()) / (n - k)
            hits += abs(m1 - m2) >= obs - 1e-12
        return hits / total, True, total
    hits = sum(abs((p := rng.permutation(allc))[:k].mean() - p[k:].mean()) >= obs - 1e-12 for _ in range(n_perm))
    return (hits + 1) / (n_perm + 1), False, total


def did(ws: pd.DataFrame, pre: int, post: int, stat: str, B: int, n_perm: int, rng):
    piv = ws.pivot_table(index=["well", "condition"], columns="tp", values=stat)
    if pre not in piv.columns or post not in piv.columns:
        return None
    ch = (piv[post] - piv[pre]).dropna().reset_index(name="chg")
    sa = ch[ch.condition == "sa"].chg.to_numpy()
    ut = ch[ch.condition == "untreated"].chg.to_numpy()
    if len(sa) == 0 or len(ut) == 0:
        return None
    out = {"n_wells_sa": len(sa), "n_wells_ut": len(ut), "delta_sa": float(sa.mean()), "delta_ut": float(ut.mean())}
    out["did"] = out["delta_sa"] - out["delta_ut"]
    out["fold"] = 10 ** out["did"]
    if len(sa) >= MIN_WELLS_CI:
        bs = np.array([rng.choice(sa, len(sa)).mean() - rng.choice(ut, len(ut)).mean() for _ in range(B)])
        out["did_lo"], out["did_hi"] = np.percentile(bs, [2.5, 97.5]).tolist()
        out["fold_lo"], out["fold_hi"] = 10 ** out["did_lo"], 10 ** out["did_hi"]
    out["perm_p"], out["perm_exact"], out["n_labellings"] = perm_test(sa, ut, n_perm, rng)
    out["few_wells"] = len(sa) < MIN_WELLS_CI
    return out


def count_did(ws: pd.DataFrame, pre: int, post: int, B: int, n_perm: int, rng):
    piv = ws.pivot_table(index=["well", "condition"], columns="tp", values="n")
    if pre not in piv.columns or post not in piv.columns:
        return None
    ch = np.log(piv[post] / piv[pre]).replace([np.inf, -np.inf], np.nan).dropna().reset_index(name="lc")
    sa = ch[ch.condition == "sa"].lc.to_numpy()
    ut = ch[ch.condition == "untreated"].lc.to_numpy()
    if not len(sa) or not len(ut):
        return None
    est = sa.mean() - ut.mean()
    out = {"n_wells_sa": len(sa), "n_wells_ut": len(ut), "count_ratio": float(np.exp(est)),
           "sa_post_over_pre": float(np.exp(sa.mean())), "ut_post_over_pre": float(np.exp(ut.mean())),
           "few_wells": len(sa) < MIN_WELLS_CI}
    if len(sa) >= MIN_WELLS_CI:
        dd = np.array([rng.choice(sa, len(sa)).mean() - rng.choice(ut, len(ut)).mean() for _ in range(B)])
        lo, hi = np.percentile(dd, [2.5, 97.5])
        out["count_ratio_lo"], out["count_ratio_hi"] = float(np.exp(lo)), float(np.exp(hi))
    out["perm_p"], out["perm_exact"], out["n_labellings"] = perm_test(sa, ut, n_perm, rng)
    return out


def shift_function(d: pd.DataFrame, pre: int, post: int, B: int, rng):
    """Change in each decile; the bootstrap draws wells once and uses the same wells at pre and post."""
    rows = []
    for cond in ("sa", "untreated"):
        a = d[(d.condition == cond) & (d.tp == pre)]
        b = d[(d.condition == cond) & (d.tp == post)]
        wells = sorted(set(a.well) & set(b.well))
        if not wells:
            continue
        wa = {w: g.log10_ratio.dropna().to_numpy() for w, g in a.groupby("well")}
        wb = {w: g.log10_ratio.dropna().to_numpy() for w, g in b.groupby("well")}
        qa = np.quantile(np.concatenate([wa[w] for w in wells]), DECILES)
        qb = np.quantile(np.concatenate([wb[w] for w in wells]), DECILES)
        bs = []
        for _ in range(min(B, 400)):
            pick = [wells[i] for i in rng.integers(0, len(wells), len(wells))]
            bs.append(np.quantile(np.concatenate([wb[w] for w in pick]), DECILES)
                      - np.quantile(np.concatenate([wa[w] for w in pick]), DECILES))
        lo, hi = np.percentile(np.array(bs), [2.5, 97.5], axis=0)
        for k, dec in enumerate(DECILES):
            rows.append({"condition": cond, "decile": dec, "shift": qb[k] - qa[k], "lo": lo[k], "hi": hi[k],
                         "n_wells": len(wells)})
    return rows


def xdp1_comparison(ws: pd.DataFrame, pre: int, post: int, stat: str, excl):
    """Plate-mean change pre->post vs every same-length change inside T1-T10 that avoids excluded intervals."""
    piv = ws.pivot_table(index="well", columns="tp", values=stat)
    lag = post - pre
    obs = float((piv[post] - piv[pre]).mean())
    bad = {tuple(x) for x in excl}
    null = []
    for a in range(1, 11 - lag):
        b = a + lag
        steps = {(t, t + 1) for t in range(a, b)}
        if b > 10 or steps & bad:
            continue
        null.append(float((piv[b] - piv[a]).mean()))
    null = np.array(null)
    mx = float(np.abs(null).max()) if len(null) else np.nan
    return {"delta_sa": obs, "comparison_n": len(null), "comparison_max_abs": mx,
            "comparison_mean": float(null.mean()) if len(null) else np.nan,
            "ratio_to_comparison_max": obs / mx if len(null) else np.nan}


def main() -> None:
    P = cfg.params()
    B, n_perm = P["bootstrap"]["B_prepost"], P["bootstrap"]["n_perm"]
    if len(sys.argv) > 1:
        B, n_perm = int(sys.argv[1]), int(sys.argv[1])
    rec = StepRecorder(STEP, LOCAL_OUT / STEP, {"B": B, "n_perm": n_perm, "exact_perm_max": EXACT_MAX,
                                                 "min_wells_for_ci": MIN_WELLS_CI})
    rng = np.random.default_rng(P["seed"])
    excl = P["dose_timing"]["xdp1_exclude_intervals"]
    rows, shifts, counts = [], [], []
    exps = [e for e in figs.EXP_ORDER if data.parquet_path(e).exists()]
    for exp in exps:
        pre, posts, alt, src = timing(exp)
        rec.summary[exp] = {"pre_tp": pre, "post_tps": posts, "alt_pre_tp": alt, "timing_source": src}
        d = data.load(exp, columns=["well", "line", "condition", "tp", "log10_ratio", "gfp", "rfp", "in_scope"])
        d = d[d.condition.isin(["sa", "untreated"])].copy()
        d["lg"] = np.log10(d.gfp.where(d.gfp > 0))
        d["lr"] = np.log10(d.rfp.where(d.rfp > 0))
        tot = d.groupby(["line", "condition", "tp"]).size().rename("cells").reset_index()
        tot.insert(0, "exp", exp)
        counts.append(tot)
        if pre is None:
            continue
        baselines = [("primary", pre)] + ([("alt", alt)] if alt else [])
        for line in [l for l in figs.LINE_ORDER if l in set(d.line)]:
            dl = d[d.line == line]
            ws = well_stats(dl)
            for bname, b_tp in baselines:
                for post in posts:
                    base = {"exp": exp, "line": line, "baseline": bname, "pre_tp": b_tp, "post_tp": post,
                            "timing_source": src}
                    for stat in STATS:
                        if exp == "XDP1":
                            if stat in ("med", "q90"):
                                rows.append({**base, "stat": stat, **xdp1_comparison(ws, b_tp, post, stat, excl)})
                            continue
                        r = did(ws, b_tp, post, stat, B, n_perm, rng)
                        if r:
                            rows.append({**base, "stat": stat, **r})
                    if exp != "XDP1":
                        c = count_did(ws, b_tp, post, B, n_perm, rng)
                        if c:
                            rows.append({**base, "stat": "count", **c})
                    if bname == "primary":
                        for s in shift_function(dl, b_tp, post, B, rng):
                            shifts.append({**base, **s})
        plot_distributions(exp, d, pre, posts, rec)
        plot_ridgeline(exp, d, rec)
        print(exp, "done", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(rec.out_dir / "prepost_results.csv", index=False)
    sh = pd.DataFrame(shifts)
    sh.to_csv(rec.out_dir / "shift_function.csv", index=False)
    pd.concat(counts).to_csv(rec.out_dir / "cell_totals.csv", index=False)
    plot_forest(res, rec)
    plot_shift(sh, rec)

    # Per-case pattern of the shift (computed, so the document never generalises by hand).
    pat = []
    for (exp, line), g in sh[sh.condition == "sa"].groupby(["exp", "line"]):
        g = g[g.post_tp == g.post_tp.max()]
        u = sh[(sh.exp == exp) & (sh.line == line) & (sh.condition == "untreated") & (sh.post_tp == g.post_tp.max())]
        net = g.set_index("decile")["shift"] - (u.set_index("decile")["shift"] if len(u) else 0)
        top, mid, bottom = net.loc[0.9], net.loc[[0.4, 0.5, 0.6]].mean(), net.loc[0.1]
        kind = ("upper tail rises most" if top > mid + 0.03 else
                "near-uniform shift" if abs(top - mid) <= 0.03 else "upper tail rises least")
        pat.append({"exp": exp, "line": line, "net_d10": bottom, "net_mid": mid, "net_d90": top, "pattern": kind})
    pd.DataFrame(pat).to_csv(rec.out_dir / "shift_pattern.csv", index=False)

    for (exp, line), g in res[(res.stat == "med") & (res.baseline == "primary")].groupby(["exp", "line"]):
        last = g.sort_values("post_tp").iloc[-1]
        sl = figs.LINE_SHORT[line]
        if exp == "XDP1":
            ok = last["ratio_to_comparison_max"] > 3
            rec.check(f"{exp}.{sl}.median_shift", f"{exp} {sl}: plate-mean median change T{int(last.pre_tp)}->"
                      f"T{int(last.post_tp)} exceeds 3x the largest same-length change before dosing", bool(ok),
                      {"ratio": last["ratio_to_comparison_max"], "comparison_n": last["comparison_n"]})
            continue
        if last["few_wells"]:
            ok, how = last["perm_p"] < 0.05, f"exact permutation p ({int(last['n_labellings'])} labellings)"
        else:
            ok, how = last["did_lo"] > 0, "bootstrap 95% CI"
        rec.check(f"{exp}.{sl}.median_shift", f"{exp} {sl}: S.A. raised the median ratio more than untreated by "
                  f"T{int(last['post_tp'])} ({how})", bool(ok),
                  {"fold": last["fold"], "ci": [last.get("fold_lo"), last.get("fold_hi")], "perm_p": last["perm_p"]})
    rec.write()
    for c in rec.checks:
        print(c["status"], c["id"], str(c["value"])[:160])


def plot_distributions(exp, d, pre, posts, rec):
    lines = [l for l in figs.LINE_ORDER if l in set(d.line)]
    bins = np.linspace(-2.4, 0.2, 90)
    has_ut = (d.condition == "untreated").any()
    fig, axes = plt.subplots(1, len(lines), figsize=(3.2 * len(lines), 2.8), sharey=True, squeeze=False)
    post = posts[-1]
    for ax, line in zip(axes[0], lines):
        dl = d[d.line == line]

        def hist(cond, tp, **kw):
            x = dl[(dl.condition == cond) & (dl.tp == tp)].log10_ratio.dropna().to_numpy()
            if len(x):
                ax.hist(x, bins=bins, density=True, histtype="step", **kw)
        hist("sa", pre, color=figs.SA, lw=1.4, ls="--", label=f"S.A. wells T{pre} (pre)")
        hist("sa", post, color=figs.SA, lw=2.0, label=f"S.A. wells T{post} (post)")
        if has_ut:
            hist("untreated", post, color=figs.UT, lw=1.6, label=f"untreated T{post}")
        figs.log_ratio_axis(ax)
        ax.set_xlim(-2.2, 0.1)
        ax.set_title(f"{figs.LINE_SHORT[line]}", fontsize=9)
        ax.set_xlabel("GEDI ratio (RFP/GFP, log scale)")
    axes[0][0].set_ylabel("density")
    axes[0][-1].legend(fontsize=7, loc="upper right")
    fig.suptitle(f"{exp}: per-cell GEDI ratio before vs after S.A." + ("" if has_ut else
                 " (whole plate dosed - no untreated wells)"), x=0.01, ha="left", fontsize=10)
    path = figs.save(fig, rec.out_dir / f"F4_distributions_{exp}.png")
    rec.figure(f"F4_{exp}", path, f"{exp}: distribution of the per-cell GEDI ratio in S.A. wells at the last "
               f"pre-dose timepoint (T{pre}, dashed) and after dosing (T{post}, solid)"
               + (", with untreated wells at the same timepoint (grey)." if has_ut else "."), exp=exp)


def plot_ridgeline(exp, d, rec):
    lines = [l for l in figs.LINE_ORDER if l in set(d.line)]
    tps = sorted(d.tp.unique())
    bins = np.linspace(-2.4, 0.2, 80)
    centers = (bins[:-1] + bins[1:]) / 2
    fig, axes = plt.subplots(1, len(lines), figsize=(3.0 * len(lines), 0.42 * len(tps) + 1.2), sharey=True,
                             squeeze=False)
    for ax, line in zip(axes[0], lines):
        for k, tp in enumerate(tps):
            for cond, color in (("untreated", figs.UT), ("sa", figs.SA)):
                x = d[(d.line == line) & (d.condition == cond) & (d.tp == tp)].log10_ratio.dropna().to_numpy()
                if len(x) < 30:
                    continue
                h, _ = np.histogram(x, bins=bins, density=True)
                ax.plot(centers, k + h / 4.5, color=color, lw=1.1)
            ax.axhline(k, color=figs.GRID, lw=0.5)
        figs.log_ratio_axis(ax)
        ax.set_xlim(-2.2, 0.1)
        ax.set_yticks(range(len(tps)))
        ax.set_yticklabels([f"T{t}" for t in tps])
        ax.set_title(figs.LINE_SHORT[line], fontsize=9)
        ax.grid(False)
    fig.suptitle(f"{exp}: ratio distribution at every timepoint - S.A. wells (violet), untreated (grey)",
                 x=0.01, ha="left", fontsize=10)
    path = figs.save(fig, rec.out_dir / f"F4r_ridgeline_{exp}.png")
    rec.figure(f"F4r_{exp}", path, f"{exp}: ratio distributions at every timepoint (ridgeline). An S.A. effect "
               "should appear only after the dose; a change shared by both colours is plate-wide.", exp=exp)


def plot_forest(res, rec):
    r = res[(res.stat == "med") & (res.baseline == "primary") & (res.exp != "XDP1")].copy()
    r = r.sort_values(["exp", "line", "post_tp"]).groupby(["exp", "line"]).tail(1)
    if not len(r):
        return
    r["order"] = r.exp.map({e: i for i, e in enumerate(figs.EXP_ORDER)}) * 10 + r.line.map(
        {l: i for i, l in enumerate(figs.LINE_ORDER)})
    r = r.sort_values("order", ascending=False)
    fig, ax = plt.subplots(figsize=(6.5, 0.32 * len(r) + 1.0))
    y = np.arange(len(r))
    ax.axvline(1, color=figs.AXIS, lw=1)
    has = r.fold_lo.notna().to_numpy()
    ax.errorbar(r.fold.to_numpy()[has], y[has],
                xerr=[(r.fold - r.fold_lo).to_numpy()[has], (r.fold_hi - r.fold).to_numpy()[has]],
                fmt="o", color=figs.SA, ms=6, lw=1.6, capsize=0, label="bootstrap 95% CI")
    ax.plot(r.fold.to_numpy()[~has], y[~has], "o", mfc="none", mec=figs.SA, ms=7, mew=1.6,
            label="2 S.A. wells: no CI (exact permutation p in table)")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{e} {figs.LINE_SHORT[l]}  (T{int(a)}->T{int(b)})" for e, l, a, b in
                        zip(r.exp, r.line, r.pre_tp, r.post_tp)], fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("fold change in median GEDI ratio, S.A. relative to untreated")
    ax.grid(axis="y", visible=False)
    ax.legend(fontsize=7, loc="lower right")
    fig.suptitle("Change in median ratio in S.A. wells beyond the change in untreated wells", x=0.01, ha="left",
                 fontsize=10)
    path = figs.save(fig, rec.out_dir / "F5_did_forest.png")
    rec.figure("F5", path, "Median-shift difference-in-differences per line x experiment: change in S.A. wells "
               "minus change in untreated wells between the last pre-dose and the last post-dose timepoint, shown as "
               "a fold change in the ratio (1 = no S.A. effect). Open circles: XDP10, 2 S.A. wells per line.")


def plot_shift(sh, rec):
    if not len(sh):
        return
    last = sh.groupby(["exp", "line"]).post_tp.transform("max")
    sh = sh[sh.post_tp == last]
    exps = [e for e in figs.EXP_ORDER if e in set(sh.exp)]
    lines = figs.LINE_ORDER
    fig, axes = plt.subplots(len(exps), len(lines), figsize=(2.5 * len(lines), 1.9 * len(exps)), sharex=True,
                             sharey=True, squeeze=False)
    for i, e in enumerate(exps):
        for j, l in enumerate(lines):
            ax = axes[i][j]
            s = sh[(sh.exp == e) & (sh.line == l)]
            if not len(s):
                ax.set_axis_off()
                continue
            for cond, color in (("untreated", figs.UT), ("sa", figs.SA)):
                c = s[s.condition == cond]
                ax.fill_between(c.decile.to_numpy(), c.lo.to_numpy(), c.hi.to_numpy(), color=color, alpha=0.18, lw=0)
                ax.plot(c.decile.to_numpy(), c["shift"].to_numpy(), color=color, marker="o", ms=3, lw=1.4)
            ax.axhline(0, color=figs.AXIS, lw=0.8)
            if i == 0:
                ax.set_title(figs.LINE_SHORT[l], fontsize=9)
            if j == 0:
                ax.set_ylabel(e, fontsize=9)
    fig.suptitle("Shift function: change in each decile of log10 ratio, pre -> last post timepoint "
                 "(S.A. violet, untreated grey; x = decile 0.1-0.9)", x=0.01, ha="left", fontsize=10)
    path = figs.save(fig, rec.out_dir / "F5s_shift_function.png")
    rec.figure("F5s", path, "Shift function: how far each decile of the ratio distribution moved between the "
               "pre-dose and last post-dose timepoint (95% CI, same wells resampled at both times). A dead "
               "population appearing raises the upper deciles most; a rise in every decile is a shift of the whole "
               "population (e.g. stressed but living cells).")


if __name__ == "__main__":
    main()

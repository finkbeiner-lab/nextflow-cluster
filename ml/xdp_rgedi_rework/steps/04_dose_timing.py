"""Step 04: when does the S.A. effect appear? (rule R2; needed for XDP3, validated on XDP5/10/11)

Well-level outcomes per timepoint (S.A. and untreated wells of in-scope lines):
  q90_log    - 90th percentile of log10 ratio (primary; death acts on the upper tail)
  median_log - median log10 ratio
  log_count  - log(cells / cells at T1)
Event study (per line):  y_wt = a_w + g_t + sum_{s>1} d_s * SA_w * 1[t=s]   (OLS, SE clustered by well)
Change point (plate):    y_wt = a_w + g_{t,line} + d_line * SA_w * 1[t >= tau]; tau_hat = argmin RSS
Bootstrap: resample wells within line x condition (B from params) -> distribution of tau_hat.
Accept tau_hat (q90 outcome) only if: bootstrap mode in >= 90% of replicates, d_line(tau_hat) > 0 in
every line, and the event-study effect at tau_hat exceeds 3x the largest pre-tau |d_s| in every line.
XDP1 (whole plate, no untreated): within-well first differences vs a null band of T1-T10 intervals.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import statsmodels.api as sm  # noqa: E402

from lib import cfg, data, figs  # noqa: E402
from lib.paths import LOCAL_OUT  # noqa: E402
from lib.provenance import StepRecorder  # noqa: E402

STEP = "04_dose_timing"
plt = figs.plt
OUTCOMES = {"q90_log": "90th pct of log10 ratio", "median_log": "median log10 ratio",
            "log_count": "log(cells / cells at T1)"}


def well_outcomes(exp: str) -> pd.DataFrame:
    d = data.load(exp, columns=["well", "line", "condition", "tp", "log10_ratio", "in_scope"])
    d = d[d.condition.isin(["sa", "untreated"])]
    g = d.groupby(["well", "line", "condition", "tp"]).log10_ratio
    w = pd.DataFrame({"median_log": g.median(), "q90_log": g.quantile(0.9), "n": g.size()}).reset_index()
    t1 = w[w.tp == w.tp.min()].set_index("well").n
    w["log_count"] = np.log(w.n / w.well.map(t1))
    w["sa"] = (w.condition == "sa").astype(int)
    return w.dropna(subset=["median_log", "q90_log", "log_count"])


def event_study(w: pd.DataFrame, outcome: str, line: str) -> pd.DataFrame:
    sub = w[w.line == line].copy()
    tps = sorted(sub.tp.unique())
    X = pd.get_dummies(sub.well, prefix="w", dtype=float)
    X = X.join(pd.get_dummies(sub.tp, prefix="t", drop_first=True, dtype=float))
    for s in tps[1:]:
        X[f"d_{s}"] = ((sub.tp == s) & (sub.sa == 1)).astype(float)
    fit = sm.OLS(sub[outcome].to_numpy(), X.to_numpy()).fit(cov_type="cluster",
                                                              cov_kwds={"groups": pd.factorize(sub.well)[0]})
    idx = {c: i for i, c in enumerate(X.columns)}
    rows = [{"line": line, "outcome": outcome, "tp": tps[0], "delta": 0.0, "se": 0.0}]
    for s in tps[1:]:
        i = idx[f"d_{s}"]
        rows.append({"line": line, "outcome": outcome, "tp": s, "delta": float(fit.params[i]),
                     "se": float(fit.bse[i])})
    return pd.DataFrame(rows)


def _residualizer(well_code: np.ndarray, lt_code: np.ndarray):
    """Return a function projecting vectors off the span of well + line x timepoint dummies."""
    n = len(well_code)
    base = np.zeros((n, well_code.max() + 1 + lt_code.max() + 1))
    base[np.arange(n), well_code] = 1.0
    base[np.arange(n), well_code.max() + 1 + lt_code] = 1.0
    u, sv, _ = np.linalg.svd(base, full_matrices=False)
    u = u[:, sv > sv.max() * 1e-10]
    return lambda v: v - u @ (u.T @ v)


def _cp_fit(y, tp, sa, line_code, n_lines, resid, taus):
    ry = resid(y)
    rss, deltas = {}, {}
    for tau in taus:
        post = (tp >= tau) & (sa == 1)
        cols = np.zeros((len(y), n_lines))
        cols[post, line_code[post]] = 1.0
        rc = resid(cols)
        beta, *_ = np.linalg.lstsq(rc, ry, rcond=None)
        r = ry - rc @ beta
        rss[tau] = float(r @ r)
        deltas[tau] = beta
    return rss, deltas


def _codes(w: pd.DataFrame, lines):
    line_code = w.line.map({l: i for i, l in enumerate(lines)}).to_numpy()
    lt = pd.factorize(w.line + "_" + w.tp.astype(str))[0]
    return line_code, lt


def changepoint(w: pd.DataFrame, outcome: str, taus, lines):
    """Plate-level change-point model: RSS per tau and per-line deltas (fixed effects partialled out)."""
    line_code, lt = _codes(w, lines)
    resid = _residualizer(pd.factorize(w.wid)[0], lt)
    rss, deltas = _cp_fit(w[outcome].to_numpy(float), w.tp.to_numpy(), w.sa.to_numpy(), line_code,
                          len(lines), resid, taus)
    return rss, {t: dict(zip(lines, d.tolist())) for t, d in deltas.items()}


def boot_tau(w: pd.DataFrame, outcome: str, taus, lines, B: int, seed: int):
    """Well bootstrap (within line x condition) of the change-point estimate."""
    rng = np.random.default_rng(seed)
    rows_of = {wl: np.flatnonzero(w.well.to_numpy() == wl) for wl in w.well.unique()}
    strata = [g.well.unique() for _, g in w.groupby(["line", "condition"])]
    y_all, tp_all, sa_all = w[outcome].to_numpy(float), w.tp.to_numpy(), w.sa.to_numpy()
    line_all, lt_all = _codes(w, lines)
    out = []
    for _ in range(B):
        idx, inst = [], []
        k = 0
        for ws in strata:
            for wl in rng.choice(ws, len(ws), replace=True):
                r = rows_of[wl]
                idx.append(r)
                inst.append(np.full(len(r), k))
                k += 1
        idx = np.concatenate(idx)
        inst = np.concatenate(inst)
        resid = _residualizer(inst, pd.factorize(lt_all[idx])[0])
        rss, _ = _cp_fit(y_all[idx], tp_all[idx], sa_all[idx], line_all[idx], len(lines), resid, taus)
        out.append(min(rss, key=rss.get))
    return out


def xdp1_differences(w: pd.DataFrame, params: dict) -> pd.DataFrame:
    excl = {tuple(x) for x in params["dose_timing"]["xdp1_exclude_intervals"]}
    rows = []
    for outcome in OUTCOMES:
        piv = w.pivot_table(index=["well", "line"], columns="tp", values=outcome)
        tps = sorted(piv.columns)
        for line, g in piv.groupby(level="line"):
            for a, b in zip(tps[:-1], tps[1:]):
                dlt = (g[b] - g[a]).dropna()
                rows.append({"outcome": outcome, "line": line, "interval": f"T{a}->T{b}", "t_from": a, "t_to": b,
                             "mean_change": float(dlt.mean()), "se": float(dlt.std(ddof=1) / np.sqrt(len(dlt))),
                             "null_band": bool(b <= 10 and (a, b) not in excl)})
    df = pd.DataFrame(rows)
    for (outcome, line), g in df.groupby(["outcome", "line"]):
        null = g[g.null_band].mean_change.abs().max()
        df.loc[g.index, "ratio_to_null_max"] = g.mean_change.abs() / null
    return df


def plot_event(exp, es, hours, tau, rec):
    lines = [l for l in figs.LINE_ORDER if l in set(es.line)]
    fig, axes = plt.subplots(len(OUTCOMES), len(lines), figsize=(2.9 * len(lines), 2.3 * len(OUTCOMES)),
                             sharex=True, squeeze=False)
    for i, (oc, label) in enumerate(OUTCOMES.items()):
        for j, line in enumerate(lines):
            ax = axes[i][j]
            s = es[(es.outcome == oc) & (es.line == line)].sort_values("tp")
            h = s.tp.map(hours).to_numpy()
            ax.axhline(0, color=figs.AXIS, lw=0.8)
            ax.fill_between(h, (s.delta - 1.96 * s.se).to_numpy(), (s.delta + 1.96 * s.se).to_numpy(),
                            color=figs.SA, alpha=0.15, lw=0)
            ax.plot(h, s.delta.to_numpy(), color=figs.SA, marker="o", ms=4)
            if tau is not None:
                ax.axvspan(hours[tau] - 6, hours[tau] + 6, color=figs.DEAD, alpha=0.15, lw=0)
            if i == 0:
                ax.set_title(f"{figs.LINE_SHORT[line]}", fontsize=9)
            if j == 0:
                ax.set_ylabel(label, fontsize=8)
            if i == len(OUTCOMES) - 1:
                ax.set_xlabel("hours")
    fig.suptitle(f"{exp}: S.A. minus untreated, per timepoint (relative to T1; 95% CI, SE clustered by well)"
                 + (f"; shaded = estimated onset T{tau}" if tau else ""), x=0.01, ha="left", fontsize=10)
    path = figs.save(fig, rec.out_dir / f"F_event_{exp}.png")
    rec.figure(f"Fevent_{exp}", path, f"{exp}: event study - difference between S.A. and untreated wells at "
               "each timepoint, net of the T1 difference. Before dosing the points should sit on zero.", exp=exp)


def plot_xdp1(df, hours, rec):
    lines = [l for l in figs.LINE_ORDER if l in set(df.line)]
    fig, axes = plt.subplots(len(OUTCOMES), len(lines), figsize=(3.0 * len(lines), 2.3 * len(OUTCOMES)),
                             sharex=True, squeeze=False)
    for i, oc in enumerate(OUTCOMES):
        for j, line in enumerate(lines):
            ax = axes[i][j]
            s = df[(df.outcome == oc) & (df.line == line)].sort_values("t_to")
            h = s.t_to.map(hours).to_numpy()
            null = s[s.null_band]
            band = null.mean_change.abs().max()
            ax.axhspan(-band, band, color=figs.AMBIG, alpha=0.7, lw=0, label="null band (T1-T10 changes)")
            ax.errorbar(h, s.mean_change.to_numpy(), yerr=1.96 * s.se.to_numpy(), fmt="o-", color=figs.SA, ms=4,
                        lw=1.2)
            for x in ((hours[10] + hours[11]) / 2, (hours[12] + hours[13]) / 2):
                ax.axvline(x, color=figs.INK2, ls="--", lw=1)
            if i == 0:
                ax.set_title(figs.LINE_SHORT[line], fontsize=9)
            if j == 0:
                ax.set_ylabel(OUTCOMES[oc] + "\nchange vs previous tp", fontsize=8)
            if i == len(OUTCOMES) - 1:
                ax.set_xlabel("hours")
    fig.suptitle("XDP1 (whole plate dosed, no untreated wells): mean within-well change between consecutive "
                 "timepoints; dashed = S.A. doses", x=0.01, ha="left", fontsize=10)
    path = figs.save(fig, rec.out_dir / "F_event_XDP1.png")
    rec.figure("Fevent_XDP1", path, "XDP1: 12-h changes per well (mean, 95% CI). Grey band = largest change "
               "seen between T1 and T10 (excluding intervals touching the T7 artefact).", exp="XDP1")


# Pre-dose windows for the negative control: the onset scan run where no S.A. effect can exist.
NULL_WINDOWS = {"XDP3": 9, "XDP5": 6, "XDP10": 8, "XDP11": 12}
EXPECTED = {"XDP5": [8, 9], "XDP10": [9], "XDP11": [13], "XDP3": [10]}


def onset_analysis(w, lines, taus, B, seed):
    res = {}
    for oc in OUTCOMES:
        rss, deltas = changepoint(w, oc, taus, lines)
        tau = min(rss, key=rss.get)
        boots = boot_tau(w, oc, taus, lines, B, seed)
        res[oc] = {"tau_hat": int(tau), "deltas": deltas[tau],
                   "boot_distribution": {int(k): int(v) for k, v in pd.Series(boots).value_counts().sort_index().items()},
                   "boot_support": float((np.array(boots) == tau).mean())}
    return res


def main() -> None:
    P = cfg.params()
    rec = StepRecorder(STEP, LOCAL_OUT / STEP, P["dose_timing"])
    B = int(sys.argv[1]) if len(sys.argv) > 1 else P["bootstrap"]["B_tau"]
    B_null = min(B, 200)
    thr = P["dose_timing"]["accept_boot_mode_frac"]
    exps = [e for e in ("XDP1", "XDP3", "XDP5", "XDP10", "XDP11") if data.parquet_path(e).exists()]
    for exp in exps:
        hours = cfg.read_timepoints(exp).set_index("tp").hours
        w = well_outcomes(exp)
        w["wid"] = w.well
        w.to_csv(rec.out_dir / f"well_outcomes_{exp}.csv", index=False)
        if exp == "XDP1":
            df = xdp1_differences(w, P)
            df.to_csv(rec.out_dir / "xdp1_differences.csv", index=False)
            summ = {"documented_doses": "T10->T11 and T12->T13"}
            for oc in ("q90_log", "median_log"):
                q = df[df.outcome == oc]
                first = {}
                for line, g in q.groupby("line"):
                    hit = g[(g.t_from >= 10) & (g.ratio_to_null_max > P["dose_timing"]["accept_effect_ratio"])]
                    first[line] = int(hit.t_to.min()) if len(hit) else None
                summ[f"first_interval_over_3x_{oc}"] = first
                summ[f"t10_t11_ratio_{oc}"] = q[q.t_to == 11].set_index("line").ratio_to_null_max.to_dict()
                summ[f"null_max_interval_{oc}"] = (q[q.null_band].sort_values("mean_change", key=abs)
                                                   .groupby("line").tail(1).set_index("line").interval.to_dict())
            t7 = df[(df.outcome == "median_log") & (df.t_to == 7)].set_index("line").mean_change.to_dict()
            summ["t6_t7_median_change"] = t7
            rec.summary[exp] = summ
            first = summ["first_interval_over_3x_q90_log"]
            rec.check("XDP1.effect_after_dose1", "XDP1: the first T10+ interval whose 90th-percentile change exceeds "
                      "3x the largest T1-T10 change is T10->T11 in every line",
                      all(v == 11 for v in first.values()), first)
            plot_xdp1(df, hours, rec)
            continue
        lines = [l for l in figs.LINE_ORDER if l in set(w.line)]
        es = pd.concat([event_study(w, oc, l) for oc in OUTCOMES for l in lines])
        es.to_csv(rec.out_dir / f"event_study_{exp}.csv", index=False)
        taus = sorted(w.tp.unique())[1:]
        res = onset_analysis(w, lines, taus, B, P["seed"])
        tau = res["q90_log"]["tau_hat"]
        q = es[es.outcome == "q90_log"]
        ratio = {}
        for l in lines:
            s = q[q.line == l].set_index("tp")
            pre = s.loc[[t for t in s.index if 1 < t < tau], "delta"].abs()
            ratio[l] = float(s.loc[tau, "delta"] / pre.max()) if len(pre) else float("inf")
        acc = {"boot_mode_frac": res["q90_log"]["boot_support"] >= thr,
               "sign_all_lines": all(v > 0 for v in res["q90_log"]["deltas"].values()),
               "effect_vs_pre": all(v > P["dose_timing"]["accept_effect_ratio"] for v in ratio.values())}
        d19 = acc["boot_mode_frac"] and acc["sign_all_lines"]
        # Negative control: the same scan on a window that ends before the dose.
        wn = w[w.tp <= NULL_WINDOWS[exp]].copy()
        if exp == "XDP5":
            wn = wn[wn.tp != 7]
        null = onset_analysis(wn, lines, sorted(wn.tp.unique())[1:], B_null, P["seed"] + 1)
        null_pass = {oc: (r["boot_support"] >= thr and all(v > 0 for v in r["deltas"].values()))
                     for oc, r in null.items()}
        rec.summary[exp] = {
            "tau_hat": {oc: r["tau_hat"] for oc, r in res.items()},
            "boot_support": {oc: r["boot_support"] for oc, r in res.items()},
            "boot_distribution": {oc: r["boot_distribution"] for oc, r in res.items()},
            "tau_hat_q90": int(tau), "tau_hours": float(hours[tau]), "boot_B": B,
            "delta_at_tau_by_line": res["q90_log"]["deltas"], "effect_to_pre_ratio": ratio,
            "acceptance_preregistered": acc, "accepted_preregistered": all(acc.values()),
            "d19_rule": {"rule": "90th-pct onset; bootstrap support >= 0.90; positive effect in every line",
                         "pass": bool(d19)},
            "matches_record": tau in EXPECTED[exp], "expected": EXPECTED[exp],
            "last_timepoint": int(max(w.tp)), "onset_is_last_timepoint": int(tau) == int(max(w.tp)),
            "null_window": {"last_tp": NULL_WINDOWS[exp], "excluded": [7] if exp == "XDP5" else [],
                            "tau_hat": {oc: r["tau_hat"] for oc, r in null.items()},
                            "boot_support": {oc: r["boot_support"] for oc, r in null.items()},
                            "d19_rule_would_accept": null_pass, "B": B_null},
        }
        rec.check(f"{exp}.tau_accepted_preregistered", f"{exp}: onset T{tau} passes all three pre-registered "
                  "acceptance rules", all(acc.values()), acc)
        rec.check(f"{exp}.tau_matches_record", f"{exp}: 90th-pct onset T{tau} is consistent with the dosing record "
                  f"(expected {EXPECTED[exp]})" + (" [XDP3: no record]" if exp == "XDP3" else ""),
                  tau in EXPECTED[exp], {"tau": int(tau), "expected": EXPECTED[exp]})
        rec.check(f"{exp}.null_window_rejected", f"{exp}: on a pre-dose-only window the proposed D19 rule finds "
                  "no onset for the 90th-pct outcome", not null_pass["q90_log"], rec.summary[exp]["null_window"])
        plot_event(exp, es, hours, tau, rec)
        print(exp, rec.summary[exp]["tau_hat"], rec.summary[exp]["boot_support"], "null", null_pass, flush=True)
    rec.write()
    for c in rec.checks:
        print(c["status"], c["id"], str(c["value"])[:200])


if __name__ == "__main__":
    main()

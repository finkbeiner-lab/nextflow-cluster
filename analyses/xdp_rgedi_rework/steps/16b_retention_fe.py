"""Step 16b: per-experiment live-cell retention, the GEDI calibration test, and a simulation check (decision D26).

C2  per experiment: Poisson GLM of live cells per well x timepoint with well and timepoint fixed effects and
    line x timepoint terms (relative to 14C), well-clustered SEs (the curves); the slope of each pair is a GLS
    line through its timepoint contrasts with the SE inflated by the lack of fit (line-level timepoint shocks,
    shared by all wells of a line, are not absorbed by well clustering). Pooled over experiments with
    DerSimonian-Laird tau^2 and Hartung-Knapp intervals. exp(slope_A - slope_B) = daily live-cell retention of A
    relative to B. The single-slope Poisson fit (well-clustered SEs only) is kept as est_simple/se_simple.
J1  calibration test (one likelihood joining counts and GEDI): replace the line slopes by the line's cumulative
    dead/live odds (CumO, odds x hours since k0). If visible death explains live-cell loss, the coefficient is
    -1/V (V = hours a dead cell stays visible) and no line differences remain (robust Wald test of the slopes
    added back). Passing (D26): bootstrap 95% CI of the coefficient below 0 and Wald p >= 0.05; then the
    dropout-corrected cumulative death since k0 is F(t) = 1 - exp(-CumO(t) / V).
SIM simulation from the two-equation model with known death rates, visibility and non-death changes, to show
    what each estimator recovers and when it is biased.
Figures 16.1 (retention curves), 16.2 (calibration), 16.3 (simulation).
"""

import os
import sys
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")   # one BLAS thread per worker: 10 workers x full thread pools made fits ~50x slower

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import statsmodels.api as sm  # noqa: E402
from scipy import stats  # noqa: E402

from lib import cfg, figs, sets  # noqa: E402
from lib.sets import PRIMARY  # noqa: E402
from lib.counts import FEFit, capture_estimate, cum_odds, slope_columns, tp_columns, window, window_or  # noqa: E402
from lib.paths import LOCAL_OUT  # noqa: E402
from lib.provenance import StepRecorder  # noqa: E402
from lib.stats import dl_pool  # noqa: E402

STEP = "16b_retention_fe"
plt = figs.plt
REF = "CTR-33114.C"
FITS = {PRIMARY: (PRIMARY, "L"), "D_sd": ("D_sd", "L"), "B": ("B", "L"), "N": (PRIMARY, "N")}
J1_SETS = tuple(dict.fromkeys([PRIMARY, "D_sd", "B"]))
EXP_COLOR = dict(zip(figs.EXP_ORDER, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#4a3aa7", "#898781"]))
S = figs.LINE_SHORT


def seed_for(*parts) -> int:
    return (cfg.params()["seed"] + zlib.crc32("|".join(map(str, parts)).encode())) % (2 ** 32)


def pair_rows(f: FEFit, lines, prefix: str, extra: dict, suffix: str = "") -> list:
    rows = []
    for ref in lines:
        for cmp_ in lines:
            if cmp_ == ref:
                continue
            w = {}
            if cmp_ != REF:
                w[f"{prefix}{cmp_}{suffix}"] = 1.0
            if ref != REF:
                w[f"{prefix}{ref}{suffix}"] = -1.0
            if any(k not in f.names for k in w):
                continue
            est, se = f.contrast(w)
            rows.append({**extra, "Reference": ref, "Compare": cmp_, "est": est, "se": se})
    return rows


def c2_fit(d: pd.DataFrame) -> FEFit:
    return FEFit(d, slope_columns(d, REF))


def c2_gls(fc: FEFit, d: pd.DataFrame, start_tp: int, lines) -> list:
    """Pair slopes from the line x timepoint contrasts: GLS line through 0, SE inflated by lack of fit (phi >= 1)."""
    days = d.groupby("tp").days.first()
    tps = [t for t in sorted(days.index) if t != start_tp]
    x = days.loc[tps].to_numpy(float)
    rows = []
    for ref in lines:
        for cmp_ in lines:
            if cmp_ == ref:
                continue
            w = np.zeros((len(tps), len(fc.names)))
            for k, t in enumerate(tps):
                if cmp_ != REF:
                    w[k, fc.names.index(f"t:{cmp_}:{t}")] = 1.0
                if ref != REF:
                    w[k, fc.names.index(f"t:{ref}:{t}")] = -1.0
            dv = w @ fc.b.to_numpy()
            S_ = w @ fc.V.to_numpy() @ w.T
            Si = np.linalg.pinv(S_)
            xs = float(x @ Si @ x)
            beta = float(x @ Si @ dv) / xs
            r = dv - beta * x
            phi = float(r @ Si @ r) / max(len(tps) - 1, 1)
            q = float(stats.t.ppf(0.975, len(tps) - 1)) if phi > 1 and len(tps) > 1 else 1.96
            rows.append({"Reference": ref, "Compare": cmp_, "est": beta, "se": float(np.sqrt(max(phi, 1.0) / xs)),
                         "phi": phi, "n_tp": len(tps), "q": q})
    return rows


def partial_r2(dd: pd.DataFrame) -> float:
    """R^2 of cumulative odds on the line x days slopes after removing well and timepoint effects (collinearity)."""
    wells, tps = pd.Categorical(dd.well), pd.Categorical(dd.tp)
    F = np.hstack([np.eye(len(wells.categories))[wells.codes], np.eye(len(tps.categories))[tps.codes][:, 1:]])
    res = lambda v: v - F @ np.linalg.lstsq(F, v, rcond=None)[0]  # noqa: E731
    y = res(dd.CumO.to_numpy(float))
    S_ = slope_columns(dd, REF)
    X = np.column_stack([res(S_[c].to_numpy(float)) for c in S_.columns])
    fit = X @ np.linalg.lstsq(X, y, rcond=None)[0]
    return float(1 - np.sum((y - fit) ** 2) / np.sum(y ** 2))


def j1_fit(d: pd.DataFrame, with_slopes: bool = False) -> FEFit:
    dd = d.merge(cum_odds(d), on=["line", "tp"], how="left")
    X = dd[["CumO"]]
    if with_slopes:
        X = pd.concat([X, slope_columns(dd, REF)], axis=1)
    return FEFit(dd, X)


def j1_boot(args):
    """Well bootstrap of the J1 coefficient (CumO recomputed from the resampled wells each time)."""
    d, B, seed, label = args
    rng = np.random.default_rng(seed)
    by_line = {l: g.well.unique() for l, g in d.groupby("line")}
    parts = {w: g for w, g in d.groupby("well")}
    out = []
    for _ in range(B):
        draws = []
        for l, ws in by_line.items():
            for i, w in enumerate(rng.choice(ws, len(ws), replace=True)):
                draws.append(parts[w].assign(well=f"{w}#{i}"))
        bd = pd.concat(draws, ignore_index=True)
        try:
            out.append(float(j1_fit(bd).b["CumO"]))
        except Exception:  # noqa: BLE001 - a failed refit is recorded as missing
            out.append(np.nan)
        if len(out) % 100 == 0:
            print(f"J1 bootstrap {label}: {len(out)}/{B}", flush=True)
    return np.array(out)


# ---------------------------------------------------------------------------------------------------------
# simulation
# ---------------------------------------------------------------------------------------------------------
SIM_LINES = ["CTR-33114.C", "XDP-33109.2B", "CTR-33113.2I", "XDP-33363.D"]
SIM_BASE = {"h": [0.02, 0.04, 0.02, 0.03], "pi": [0.6] * 4, "kappa": [1 / 24] * 4, "rho": [0.0] * 4}
SCENARIOS = {
    "S1 all assumptions hold": {},
    "S2 09B gains cells for non-death reasons (+3% per day)": {"rho": [0.0, 0.03, 0.0, 0.0]},
    "S3 09B dead cells seen 1.5x as often": {"pi": [0.6, 0.9, 0.6, 0.6]},
    "S4 09B dead cells stay visible 1.5x longer (readout 1 image after dosing)": {"kappa": [1 / 24, 1 / 36, 1 / 24, 1 / 24]},
    "S4b as S4, readout 2 images after dosing": {"kappa": [1 / 24, 1 / 36, 1 / 24, 1 / 24], "readout": 2},
    "S5 09B killed earlier after dosing (equal visibility)": {"kill_early": [False, True, False, False]},
}


def sim_par(over: dict) -> dict:
    par = {k: list(v) for k, v in SIM_BASE.items()}
    par.update({k: list(v) for k, v in over.items() if k in SIM_BASE})
    par["readout"] = over.get("readout", 1)
    par["kill_early"] = over.get("kill_early", [False] * 4)
    return par
N_UT, N_SA, TPS, DT = 24, 6, 10, 0.25
PRE, POST, H_SA = 8, 9, 0.03          # dose between T8 and T9 (hours 168-192), S.A. hazard per hour


def sim_truth(par: dict) -> dict:
    """Noise-free capture fraction per line (same ODE as the simulation)."""
    out = {}
    for i, l in enumerate(SIM_LINES):
        res = {}
        for sa in (False, True):
            L, D = _ode(np.array([1.0]), par, i, sa)
            res[sa] = (L[0], D[0])
        Lu, Du, Ls, Ds = res[False][0], res[False][1], res[True][0], res[True][1]
        t1 = POST - 1 + par["readout"] - 1
        r = capture_estimate(np.array([Ls[PRE - 1], Ls[t1], Ds[PRE - 1], Ds[t1]]),
                             np.array([Lu[PRE - 1], Lu[t1], Du[PRE - 1], Du[t1]]))
        out[l] = r["c"]
    return out


def _ode(hmult: np.ndarray, par: dict, i: int, sa: bool):
    """Euler integration of live and visible-dead per unit initial live cell; returns values at each timepoint."""
    h = par["h"][i] / 24 * hmult
    pi, kap, rho = par["pi"][i], par["kappa"][i], par["rho"][i] / 24
    q = pi * h / (kap - (h - rho))
    L, D = np.ones_like(h), q.copy()
    outL, outD = np.zeros((len(h), TPS)), np.zeros((len(h), TPS))
    outL[:, 0], outD[:, 0] = L, D
    steps = int(24 / DT)
    for k in range(1, TPS):
        for s in range(steps):
            t = (k - 1) * 24 + s * DT
            if par["kill_early"][i]:                # same total kill, all in the first 12 h after dosing
                hs = 2 * H_SA if (sa and (PRE - 1) * 24 <= t < (PRE - 1) * 24 + 12) else 0.0
            else:
                hs = H_SA if (sa and (PRE - 1) * 24 <= t < (POST - 1) * 24) else 0.0
            dL = -(h + hs - rho) * L
            dD = pi * (h + hs) * L - kap * D
            L, D = L + DT * dL, D + DT * dD
        outL[:, k], outD[:, k] = L, D
    return outL, outD


def simulate(par: dict, rng) -> pd.DataFrame:
    rows = []
    a = np.ones(TPS)
    a[4], a[7] = 0.85, 1.08                       # imaging-day detection changes common to all lines
    for i, l in enumerate(SIM_LINES):
        for sa in (False, True):
            n = N_SA if sa else N_UT
            l0 = rng.gamma(20, 400 / 20, n)
            hm = rng.lognormal(-0.02, 0.2, n)
            Lm, Dm = _ode(hm, par, i, sa)
            eps = rng.lognormal(0, 0.05, (n, TPS))
            L = rng.poisson(l0[:, None] * Lm * a * eps)
            D = rng.poisson(l0[:, None] * Dm * a * eps)
            for w in range(n):
                for t in range(TPS):
                    rows.append((f"{l}:{'sa' if sa else 'ut'}{w}", l, "sa" if sa else "untreated", t + 1, 24.0 * t,
                                 L[w, t], D[w, t]))
    d = pd.DataFrame(rows, columns=["well", "line", "condition", "tp", "hours", "L", "D"])
    return d


def or_fit(d: pd.DataFrame) -> pd.Series:
    """Time-averaged dead-vs-live odds ratio of each line vs 14C (binomial GLM, timepoint effects, well-clustered)."""
    X = pd.get_dummies(d.tp.astype(str), prefix="tp", drop_first=True).astype(float)
    for l in SIM_LINES[1:]:
        X[f"o:{l}"] = (d.line == l).astype(float)
    X = sm.add_constant(X)
    y = np.column_stack([d.D.to_numpy(float), d.L.to_numpy(float)])
    r = sm.GLM(y, X, family=sm.families.Binomial()).fit(cov_type="cluster",
                                                        cov_kwds={"groups": pd.Categorical(d.well).codes})
    return pd.Series({l: r.params[f"o:{l}"] for l in SIM_LINES[1:]})


def sim_rep(args):
    name, over, seed = args
    par = sim_par(over)
    rng = np.random.default_rng(seed)
    d = simulate(par, rng)
    ut = d[d.condition == "untreated"].assign(reference=True, missing=False)
    w = ut.assign(days=(ut.hours - 0) / 24, y=ut.L)
    out = {"scenario": name}
    gl = {r["Compare"]: r for r in c2_gls(FEFit(w, tp_columns(w, REF, 1)), w, 1, SIM_LINES) if r["Reference"] == REF}
    for l in SIM_LINES[1:]:
        est, se = gl[l]["est"], gl[l]["se"]
        truth = -(par["h"][SIM_LINES.index(l)] - par["h"][0])        # the death-rate difference (per day)
        out[f"c2_{S[l]}"] = est
        q = gl[l]["q"]
        out[f"c2_cover_{S[l]}"] = (est - q * se <= truth <= est + q * se)
        out[f"c2_truth_{S[l]}"] = truth
    j = j1_fit(w)
    j2 = j1_fit(w, with_slopes=True)
    out["j1_b"] = float(j.b["CumO"])
    out["j1_V"] = -1 / out["j1_b"] if out["j1_b"] < 0 else np.nan
    out["j1_wald_p"] = j2.wald([c for c in j2.names if c.startswith("s:")])["p"]
    lor = or_fit(ut)
    cs = {}
    for l in SIM_LINES:
        g = d[d.line == l]
        t1 = POST + par["readout"] - 1
        sums = lambda c: np.array([g[(g.condition == c) & (g.tp == PRE)].L.sum(), g[(g.condition == c) & (g.tp == t1)].L.sum(),  # noqa: E731
                                   g[(g.condition == c) & (g.tp == PRE)].D.sum(), g[(g.condition == c) & (g.tp == t1)].D.sum()], float)
        cs[l] = capture_estimate(sums("sa"), sums("untreated"))["c"]
    for l in SIM_LINES[1:]:
        i = SIM_LINES.index(l)
        out[f"logHR_truth_{S[l]}"] = np.log(par["h"][i] / par["h"][0])
        out[f"logOR_{S[l]}"] = lor[l]
        out[f"logOR_corr_{S[l]}"] = lor[l] + np.log(cs[SIM_LINES[0]] / cs[l]) if cs[l] > 0 and cs[SIM_LINES[0]] > 0 else np.nan
        out[f"c_ratio_{S[l]}"] = cs[l] / cs[SIM_LINES[0]]
    return out


def run_simulation(n_sim: int):
    jobs = [(name, over, seed_for("sim", name, r)) for name, over in SCENARIOS.items() for r in range(n_sim)]
    with ProcessPoolExecutor(10) as ex:
        res = pd.DataFrame(list(ex.map(sim_rep, jobs, chunksize=10)))
    truth = []
    for name, over in SCENARIOS.items():
        par = sim_par(over)
        tc = sim_truth(par)
        V = [par["pi"][i] / par["kappa"][i] for i in range(4)]
        truth.append({"scenario": name, "V_hours": V[0], **{f"V_{S[l]}": V[i] for i, l in enumerate(SIM_LINES)},
                      **{f"c_true_{S[l]}": tc[l] for l in SIM_LINES},
                      **{f"c_ratio_true_{S[l]}": tc[l] / tc[SIM_LINES[0]] for l in SIM_LINES[1:]}})
    return res, pd.DataFrame(truth)


def summarise_sim(res: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, g in res.groupby("scenario", sort=False):
        t = truth.set_index("scenario").loc[name]
        r = {"scenario": name, "n": len(g)}
        for l in ("09B", "13I", "63D"):
            r[f"c2_bias_{l}"] = float((g[f"c2_{l}"] - g[f"c2_truth_{l}"]).mean())
            r[f"c2_coverage_{l}"] = float(g[f"c2_cover_{l}"].mean())
            r[f"OR_bias_{l}"] = float((g[f"logOR_{l}"] - g[f"logHR_truth_{l}"]).mean())
            r[f"ORcorr_bias_{l}"] = float((g[f"logOR_corr_{l}"] - g[f"logHR_truth_{l}"]).mean())
            r[f"c_ratio_median_{l}"] = float(g[f"c_ratio_{l}"].median())
            r[f"c_ratio_true_{l}"] = float(t[f"c_ratio_true_{l}"])
        r["V_true_14C"] = float(t["V_14C"])
        r["V_true_09B"] = float(t["V_09B"])
        r["j1_V_median"] = float(g.j1_V.median())
        r["j1_reject_rate"] = float((g.j1_wald_p < 0.05).mean())
        rows.append(r)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------------------
def main() -> None:
    P = cfg.params()["counts"]
    rec = StepRecorder(STEP, LOCAL_OUT / STEP, params=P)
    cnt = pd.read_csv(LOCAL_OUT / "14_count_tables" / "counts_long.csv")
    k0 = pd.read_csv(LOCAL_OUT / "14_count_tables" / "k0.csv").set_index("exp").k0
    c2_rows, curve_rows, j1_rows, f_rows, calib_rows, jobs, orw_rows = [], [], [], [], [], [], []
    data = {}
    for exp in figs.EXP_ORDER:
        for label, (set_, outcome) in FITS.items():
            d = window(cnt, exp, set_, int(k0[exp]), outcome)
            data[(exp, label)] = d
            lines = [l for l in figs.LINE_ORDER if l in set(d.line)]
            f = c2_fit(d)
            simple = {(r["Reference"], r["Compare"]): r for r in pair_rows(f, lines, "s:", {})}
            fc = FEFit(d, tp_columns(d, REF, int(k0[exp])))
            for r in c2_gls(fc, d, int(k0[exp]), lines):
                sp = simple[(r["Reference"], r["Compare"])]
                c2_rows.append({"exp": exp, "fit": label, "n_wells": d.well.nunique(), "days": float(d.days.max()), **r,
                                "est_simple": sp["est"], "se_simple": sp["se"]})
            for t in sorted(set(d.tp) - {int(k0[exp])}):
                for r in pair_rows(fc, lines, "t:", {"exp": exp, "fit": label, "tp": t,
                                                     "days": float(d[d.tp == t].days.iloc[0])}, suffix=f":{t}"):
                    curve_rows.append(r)
            if label in J1_SETS:
                orw_rows.append(window_or(d, REF).assign(exp=exp, fit=label))
                jobs.append(((exp, label), (d, P["B_j1"], seed_for("j1", exp, label), f"{exp} {label}")))
        print(exp, "C2 done", flush=True)
    with ProcessPoolExecutor(10) as ex:
        boots = dict(zip([k for k, _ in jobs], ex.map(j1_boot, [a for _, a in jobs])))
    iv = pd.read_csv(LOCAL_OUT / "15_sa_capture" / "capture_invariance.csv")
    iv = iv[(iv.basis == "valid") & (iv.variant == "primary")]
    unequal_by_set = {s_: set(iv[(iv.set == s_) & (iv.equal_visibility.astype(str) == "False")].exp) for s_ in J1_SETS}
    for (exp, label), bs in boots.items():
        d = data[(exp, label)]
        j, j2 = j1_fit(d), j1_fit(d, with_slopes=True)
        b, se = float(j.b["CumO"]), float(np.sqrt(j.V.loc["CumO", "CumO"]))
        bs_ok = bs[np.isfinite(bs)]
        blo, bhi = np.percentile(bs_ok, [2.5, 97.5])
        wd = j2.wald([c for c in j2.names if c.startswith("s:")])
        passed = bool(bhi < 0 and wd["p"] >= P["j1_alpha"])
        status = ("not rejected" if passed else
                  "line differences remain" if wd["p"] < P["j1_alpha"] and bhi < 0 else
                  "no relation and line differences remain" if wd["p"] < P["j1_alpha"] else
                  "no relation to visible death")
        r2 = partial_r2(d.merge(cum_odds(d), on=["line", "tp"], how="left"))
        j1_rows.append({"exp": exp, "set": label, "b": b, "b_se_cluster": se, "b_lo_boot": blo, "b_hi_boot": bhi,
                        "frac_boot_b_nonneg": float((bs_ok >= 0).mean()),
                        "V_hours": -1 / b if b < 0 else np.nan,
                        "V_lo": -1 / blo if bhi < 0 else np.nan, "V_hi": -1 / bhi if bhi < 0 else np.nan,
                        "wald_W": wd["W"], "wald_df": wd["df"], "wald_p": wd["p"], "passes": passed, "status": status,
                        "n_lines": int(d.line.nunique()), "collinearity_r2": r2})
        co = cum_odds(d)
        lines = [l for l in figs.LINE_ORDER if l in set(d.line)]
        sl = {l: (0.0 if l == REF else float(c2_fit(d).b[f"s:{l}"])) for l in lines}
        T = float((d.hours.max() - d.hours.min()))
        for l in lines:
            cl = co[co.line == l]
            calib_rows.append({"exp": exp, "set": label, "line": l, "mean_odds": float(cl.CumO.max() / T),
                               "slope_vs_14C_per_day": sl[l]})
        if passed and exp not in unequal_by_set[label]:   # one V for all lines needs equal visibility (Step 15)
            g = d.groupby(["line", "tp"]).agg(L=("L", "sum"), D=("D", "sum"), hours=("hours", "first")).reset_index()
            g = g.merge(co, on=["line", "tp"])
            V = -1 / b
            for r in g.itertuples():
                f_rows.append({"exp": exp, "set": label, "line": r.line, "tp": r.tp, "hours": r.hours,
                               "visible_dead_share": r.D / (r.L + r.D), "CumO": r.CumO,
                               "cum_death_since_k0": 1 - np.exp(-r.CumO / V),
                               "cum_death_lo": 1 - np.exp(-r.CumO * (-bhi)), "cum_death_hi": 1 - np.exp(-r.CumO * (-blo))})
    c2 = pd.DataFrame(c2_rows)
    c2["ratio_per_day"] = np.exp(c2.est)
    c2["lower"], c2["upper"] = np.exp(c2.est - c2.q * c2.se), np.exp(c2.est + c2.q * c2.se)
    pooled = []
    for (fit, ref, cmp_), g in c2.groupby(["fit", "Reference", "Compare"]):
        m = dl_pool(g.est, g.se)
        pooled.append({"fit": fit, "Reference": ref, "Compare": cmp_, "k": m["k"], "ratio_per_day": np.exp(m["mu"]),
                       "lower": np.exp(m["lo"]), "upper": np.exp(m["hi"]),
                       "pred_lower": np.exp(m["pred_lo"]), "pred_upper": np.exp(m["pred_hi"]),
                       "I2": m["I2"], "tau2": m["tau2"], "Q_p": m["Q_p"],
                       "exp_ci_below_1": int((g.est + g.q * g.se < 0).sum()),
                       "exp_ci_above_1": int((g.est - g.q * g.se > 0).sum())})
    pooled = pd.DataFrame(pooled)
    curves = pd.DataFrame(curve_rows)
    curves["ratio"], curves["lower"], curves["upper"] = (np.exp(curves.est), np.exp(curves.est - 1.96 * curves.se),
                                                         np.exp(curves.est + 1.96 * curves.se))
    j1 = pd.DataFrame(j1_rows)
    fcum = pd.DataFrame(f_rows)
    calib = pd.DataFrame(calib_rows)
    orw = pd.concat(orw_rows, ignore_index=True)
    orw["OR"], orw["lower"], orw["upper"] = np.exp(orw.est), np.exp(orw.est - 1.96 * orw.se), np.exp(orw.est + 1.96 * orw.se)
    for name, df in (("c2_per_experiment", c2), ("c2_pooled", pooled), ("curves", curves), ("j1", j1), ("or_window", orw),
                     ("cum_death", fcum), ("calibration_points", calib)):
        df.to_csv(rec.out_dir / f"{name}.csv", index=False)

    print("simulation ...", flush=True)
    sim, truth = run_simulation(P["n_sim"])
    sim.to_csv(rec.out_dir / "sim_reps.csv", index=False)
    simsum = summarise_sim(sim, truth)
    simsum.to_csv(rec.out_dir / "sim_summary.csv", index=False)
    truth.to_csv(rec.out_dir / "sim_truth.csv", index=False)

    s1 = simsum[simsum.scenario.str.startswith("S1")].iloc[0]
    cov = min(s1[f"c2_coverage_{l}"] for l in ("09B", "13I", "63D"))
    rec.check("sim.c2_coverage", "Simulation S1: retention-slope 95% CIs cover the true death-rate difference in at least "
              "90% of replicates", cov >= 0.90, {"min_coverage": round(cov, 3)})
    s2 = simsum[simsum.scenario.str.startswith("S2")].iloc[0]
    rec.check("sim.j1_power", "Simulation S2 (09B gains cells for non-death reasons): calibration test rejects in at least "
              "80% of replicates", s2.j1_reject_rate >= 0.8 or None, {"reject_rate": round(s2.j1_reject_rate, 3)})
    rec.check("sim.j1_size", "Simulation S1: calibration test rejects in at most 10% of replicates",
              s1.j1_reject_rate <= 0.10 or None, {"reject_rate": round(s1.j1_reject_rate, 3)})
    s3 = simsum[simsum.scenario.str.startswith("S3")].iloc[0]
    rec.check("sim.correction_S3", "Simulation S3 (09B dead cells seen more often): corrected odds ratio unbiased for the "
              "death-rate ratio (|mean log bias| < 0.05) while the uncorrected one is biased",
              bool(abs(s3.ORcorr_bias_09B) < 0.05) or None,
              {"uncorrected_bias": round(s3.OR_bias_09B, 3), "corrected_bias": round(s3.ORcorr_bias_09B, 3)})
    s5 = simsum[simsum.scenario.str.startswith("S5")].iloc[0]
    rec.check("sim.timing_bias_S5", "Simulation S5 (kill timing differs, equal visibility): the S.A. correction creates a "
              "bias that the uncorrected odds ratio does not have (informational)", None,
              {"uncorrected_bias": round(s5.OR_bias_09B, 3), "corrected_bias": round(s5.ORcorr_bias_09B, 3),
               "c_ratio_true": round(s5.c_ratio_true_09B, 3)})
    rec.check("j1.boot_ok", "J1 bootstrap refits succeeded", bool(all(np.isfinite(b).mean() > 0.98 for b in boots.values())),
              {f"{k[0]}:{k[1]}": float(np.isfinite(b).mean()) for k, b in boots.items()})
    rec.summary = {
        "j1": {f"{r.exp}:{r.set}": {"V_hours": None if pd.isna(r.V_hours) else round(r.V_hours, 1),
                                    "wald_p": round(r.wald_p, 4), "passes": bool(r.passes)} for r in j1.itertuples()},
        "j1_passing": {s: sorted(j1[(j1.set == s) & j1.passes].exp.tolist()) for s in J1_SETS},
        "c2_pooled_primary": {f"{S[r.Compare]} vs {S[r.Reference]}": [round(r.ratio_per_day, 4), round(r.lower, 4),
                                                                   round(r.upper, 4), round(r.I2, 2)]
                           for r in pooled[pooled.fit == PRIMARY].itertuples()},
        "sim": simsum.round(4).to_dict(orient="records"),
    }
    plot_curves(curves, rec)
    plot_calibration(calib, j1, rec)
    plot_sim(sim, simsum, rec)
    rec.write()
    print(j1.round(4).to_string())
    print(pooled[(pooled.fit == PRIMARY) & pooled.Reference.isin([REF, "CTR-33113.2I"])].round(4).to_string())
    print(simsum.round(3).T.to_string())
    for c in rec.checks:
        if c["status"] != "PASS":
            print(c["status"], c["id"], str(c["value"])[:300])


PAIRS = [("XDP-33109.2B", "CTR-33114.C"), ("XDP-33363.D", "CTR-33114.C"), ("CTR-33113.2I", "CTR-33114.C"),
         ("XDP-33109.2B", "CTR-33113.2I"), ("XDP-33363.D", "CTR-33113.2I")]


def plot_curves(curves: pd.DataFrame, rec: StepRecorder) -> None:
    """Figure 16.1: live-cell retention of each line relative to a control line, per experiment, track 1."""
    c1p = LOCAL_OUT / "16_retention" / "c1_pairs.csv"
    c1 = pd.read_csv(c1p) if c1p.exists() else None
    fig, axes = plt.subplots(1, len(PAIRS), figsize=(17, 3.4), sharey=True)
    for ax, (cmp_, ref) in zip(axes, PAIRS):
        g = curves[(curves.fit == PRIMARY) & (curves.Compare == cmp_) & (curves.Reference == ref)]
        xmax = 0
        for exp in figs.EXP_ORDER:
            e = g[g.exp == exp].sort_values("days")
            if not len(e):
                continue
            x = np.r_[0, e.days.to_numpy()]
            ax.plot(x, -np.log2(np.r_[1, e.ratio.to_numpy()]), color=EXP_COLOR[exp], lw=1.6, marker="o", ms=2.5)
            ax.fill_between(e.days.to_numpy(), -np.log2(e.upper.to_numpy()), -np.log2(e.lower.to_numpy()),
                            color=EXP_COLOR[exp], alpha=0.10, lw=0)
            ax.annotate(exp, (x[-1], -np.log2(e.ratio.iloc[-1])), xytext=(3, 0), textcoords="offset points",
                        fontsize=6.5, color=EXP_COLOR[exp], va="center")
            xmax = max(xmax, x[-1])
        if c1 is not None:
            r = c1[(c1.fit == PRIMARY) & (c1.Compare == cmp_) & (c1.Reference == ref)]
            if len(r):
                r = r.iloc[0]
                xx = np.linspace(0, xmax, 20)
                ax.plot(xx, -xx * np.log2(r.ratio_per_day), color=figs.INK, ls="--", lw=1.4, label="pooled (mixed model)")
                ax.fill_between(xx, -xx * np.log2(r.upper), -xx * np.log2(r.lower), color=figs.INK, alpha=0.08, lw=0)
        ax.axhline(0, color=figs.AXIS, lw=0.8)
        ax.set_title(f"{S[cmp_]} vs {S[ref]}")
        ax.set_xlabel("days since k0")
    axes[0].set_ylabel("control ÷ line retention (log2)")
    axes[0].legend(loc="lower left", fontsize=7)
    fig.suptitle("Figure 16.1  Live cells kept by the control line relative to the first line (above 0: the first line keeps "
                 f"fewer of its live cells; {sets.label(PRIMARY)})", x=0.01, ha="left", fontsize=10)
    path = figs.save(fig, rec.out_dir / "F16_retention.png")
    rec.figure("F16_1", path, "Control-to-line retention = (live cells of the control at t / at k0) divided by the same for "
               "the first line, log2 (above 0: the first line keeps fewer live cells), from a Poisson model with well and "
               "timepoint effects per experiment (coloured, 95% band). Dashed: pooled trend from the mixed model.")


def plot_calibration(calib: pd.DataFrame, j1: pd.DataFrame, rec: StepRecorder) -> None:
    """Figure 16.2: does visible death explain live-cell loss? Per experiment, line points relative to the mean."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, set_ in zip(axes, (PRIMARY, "D_sd")):
        g = calib[calib.set == set_]
        for exp, e in g.groupby("exp"):
            x = (e.mean_odds - e.mean_odds.mean()).to_numpy()
            y = -(e.slope_vs_14C_per_day - e.slope_vs_14C_per_day.mean()).to_numpy()
            ax.scatter(x, 100 * y, color=EXP_COLOR[exp], s=28, zorder=3)
            for xi, yi, l in zip(x, y, e.line):
                ax.annotate(f"{exp} {S[l]}", (xi, 100 * yi), xytext=(3, 2), textcoords="offset points", fontsize=6,
                            color=figs.INK2)
            jr = j1[(j1.exp == exp) & (j1.set == set_)]
            if len(jr):
                b = float(jr.b.iloc[0])
                xx = np.array([x.min(), x.max()])
                ax.plot(xx, -100 * 24 * b * xx, color=EXP_COLOR[exp], lw=1, ls="-" if jr.passes.iloc[0] else ":")
        lim = ax.get_xlim()
        ax.axhline(0, color=figs.AXIS, lw=0.8)
        ax.axvline(0, color=figs.AXIS, lw=0.8)
        ax.set_xlim(lim)
        ax.set_xlabel("dead/live odds, difference from the experiment's mean")
        ax.set_title(sets.label(set_), loc="left")
    axes[0].set_ylabel("extra live-cell loss, % per day\n(difference from the experiment's mean)")
    fig.suptitle("Figure 16.2  Do lines with more visible death also lose live cells faster? (lines: fitted relation per "
                 "experiment; solid = calibration test passed)", x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    path = figs.save(fig, rec.out_dir / "F16_calibration.png")
    rec.figure("F16_2", path, "Each point is a line in one experiment: how much more of its live cells it loses per day "
               "than the experiment's average line, against how much higher its dead/live odds are. If visible death "
               "explained the loss, points would fall on a rising line through 0 with slope 2400/V (y in % per day, V in "
               "hours a dead cell stays visible). Solid lines: experiments where the calibration test is not rejected.")


def plot_sim(sim: pd.DataFrame, simsum: pd.DataFrame, rec: StepRecorder) -> None:
    """Figure 16.3: what each estimator recovers in the simulated scenarios (09B vs 14C)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    names = list(simsum.scenario)
    ys = np.arange(len(names))[::-1]
    for y, name in zip(ys, names):
        g = sim[sim.scenario == name]
        t = g["logHR_truth_09B"].iloc[0]
        for k, (col, color, lab) in enumerate((("logOR_09B", figs.DEAD, "odds ratio (MRID-type)"),
                                                ("logOR_corr_09B", figs.SA, "S.A.-corrected odds ratio"))):
            v = np.exp(g[col] - t)
            axes[0].errorbar(np.median(v), y + 0.15 - 0.3 * k, xerr=[[np.median(v) - np.percentile(v, 5)],
                                                                    [np.percentile(v, 95) - np.median(v)]],
                             fmt="o", color=color, ms=5, capsize=2, label=lab if y == ys[0] else None)
        tt = g["c2_truth_09B"].iloc[0]
        v = 100 * (g["c2_09B"] - tt)
        axes[1].errorbar(np.median(v), y, xerr=[[np.median(v) - np.percentile(v, 5)], [np.percentile(v, 95) - np.median(v)]],
                         fmt="o", color=figs.LIVE, ms=5, capsize=2)
    axes[0].axvline(1, color=figs.INK2, ls=":", lw=1)
    axes[0].set_xscale("log")
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
    axes[0].xaxis.set_major_locator(FixedLocator([0.5, 0.75, 1, 1.5, 2]))
    axes[0].xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    axes[0].xaxis.set_minor_formatter(NullFormatter())
    axes[0].set_xlabel("estimate / true death-rate ratio (median, 5-95%)")
    axes[0].set_yticks(ys)
    axes[0].set_yticklabels(names, fontsize=7.5)
    axes[0].legend(fontsize=7, loc="upper right")
    axes[1].axvline(0, color=figs.INK2, ls=":", lw=1)
    axes[1].set_yticks(ys)
    axes[1].set_yticklabels([])
    axes[1].set_xlabel("live-cell retention slope minus true death-rate difference (% per day)")
    fig.suptitle("Figure 16.3  Simulation (09B vs 14C, 200 runs per scenario): what each estimate recovers in each scenario",
                 x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    path = figs.save(fig, rec.out_dir / "F16_simulation.png")
    rec.figure("F16_3", path, "Left: odds ratio of visible death (as MRID) and the S.A.-corrected odds ratio, divided by "
               "the true death-rate ratio (1 = unbiased; median and 5-95% of runs). Right: live-cell retention slope minus "
               "the true death-rate difference (0 = unbiased). S2: 09B gains cells for non-death reasons - the retention "
               "slope is biased, the odds ratio is not. S3: 09B dead cells are seen more often - the odds ratio is biased "
               "and the S.A. correction removes it. S4/S4b: 09B dead cells stay visible longer - the correction removes "
               "only part of the bias at a one-image readout and over-corrects at a two-image readout. S5: 09B's S.A. "
               "kill happens earlier after dosing with equal visibility - the odds ratio is unbiased and the correction "
               "creates a bias.")


if __name__ == "__main__":
    main()

"""Count-model helpers for the GFP-dropout analysis (Steps 15-17, decision D26).

Notation per line in one experiment, at imaging hours t:
    d log E[L]/dt = -h(t) + r(t)        L live cells, h death hazard, r net non-death change
    E[D] / E[L]  ~  h(t) * V            D visible dead cells, V hours a dead cell stays visible
"""

from typing import Dict, Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

MIN_LIVE = 50


def capture_estimate(sa: np.ndarray, ut: np.ndarray) -> Dict[str, float]:
    """S.A. capture fraction from pooled counts (difference in differences against untreated wells).

    Args:
        sa: summed (L0, L1, D0, D1) of the S.A. wells at the pre-dose (0) and post-dose (1) timepoints.
        ut: the same sums for the untreated wells of the same line.

    Returns:
        Lhat (expected live without dose), K (live removed), killed_frac, Dhat, E (excess dead), c = E / K.
    """
    L0s, L1s, D0s, D1s = sa
    L0u, L1u, D0u, D1u = ut
    lhat = L0s * L1u / L0u
    K = lhat - L1s
    dhat = lhat * (D0s / L0s) * (D1u / L1u) / (D0u / L0u)
    E = D1s - dhat
    return {"Lhat": lhat, "K": K, "killed_frac": K / lhat, "Dhat": dhat, "E": E, "c": E / K if K != 0 else np.nan}


def window(cnt: pd.DataFrame, exp: str, set_: str, start_tp: int, outcome: str = "L") -> pd.DataFrame:
    """Reference wells of one experiment from ``start_tp`` on, QC applied (D26), with days since the start."""
    d = cnt[(cnt.exp == exp) & (cnt.set == set_) & cnt.reference & ~cnt.missing & (cnt.tp >= start_tp)].copy()
    base = d[d.tp == start_tp].set_index("well").L
    d = d[d.well.isin(base[base >= MIN_LIVE].index)]
    h0 = d[d.tp == start_tp].hours.iloc[0]
    d["days"] = (d.hours - h0) / 24
    d["y"] = d[outcome]
    return d.sort_values(["well", "tp"]).reset_index(drop=True)


def cum_odds(d: pd.DataFrame) -> pd.DataFrame:
    """Line-level dead/live odds per timepoint (pooled over wells) and their trapezoid integral in hours."""
    g = d.groupby(["line", "tp"]).agg(L=("L", "sum"), D=("D", "sum"), hours=("hours", "first")).reset_index()
    g["O"] = g.D / g.L
    g = g.sort_values(["line", "tp"])
    dt = g.groupby("line").hours.diff().fillna(0)
    mid = (g.O + g.groupby("line").O.shift(1)) / 2
    g["CumO"] = (mid.fillna(0) * dt).groupby(g.line).cumsum()
    return g[["line", "tp", "O", "CumO"]]


class FEFit:
    """Poisson GLM with well and timepoint fixed effects and well-clustered covariance, reporting extra terms."""

    def __init__(self, d: pd.DataFrame, extra: pd.DataFrame, cluster: Optional[np.ndarray] = None):
        wells = pd.Categorical(d.well)
        tps = pd.Categorical(d.tp)
        Xw = np.eye(len(wells.categories))[wells.codes]
        Xt = np.eye(len(tps.categories))[tps.codes][:, 1:]
        X = np.hstack([Xw, Xt, extra.to_numpy(float)])
        groups = wells.codes if cluster is None else cluster
        self.names = list(extra.columns)
        self.res = sm.GLM(d.y.to_numpy(float), X, family=sm.families.Poisson()).fit(
            cov_type="cluster", cov_kwds={"groups": groups}, maxiter=100)
        k = len(self.names)
        self.b = pd.Series(self.res.params[-k:], index=self.names)
        self.V = pd.DataFrame(self.res.cov_params()[-k:, -k:], index=self.names, columns=self.names)

    def contrast(self, w: Dict[str, float]):
        """Estimate and SE of a linear combination of the extra terms."""
        v = pd.Series(0.0, index=self.names)
        for k, x in w.items():
            v[k] = x
        return float(v @ self.b), float(np.sqrt(v @ self.V @ v))

    def wald(self, names) -> Dict[str, float]:
        """Joint robust Wald test that the named terms are all zero."""
        b = self.b[names].to_numpy()
        V = self.V.loc[names, names].to_numpy()
        W = float(b @ np.linalg.solve(V, b))
        return {"W": W, "df": len(names), "p": float(stats.chi2.sf(W, len(names)))}


def slope_columns(d: pd.DataFrame, ref: str) -> pd.DataFrame:
    """days x line indicator for every non-reference line present (slopes relative to ``ref``)."""
    return pd.DataFrame({f"s:{l}": d.days * (d.line == l) for l in sorted(set(d.line)) if l != ref}, index=d.index)


def tp_columns(d: pd.DataFrame, ref: str, start_tp: int) -> pd.DataFrame:
    """line x timepoint indicators (non-reference lines, timepoints after the start)."""
    cols = {}
    for l in sorted(set(d.line)):
        if l == ref:
            continue
        for t in sorted(set(d.tp)):
            if t != start_tp:
                cols[f"t:{l}:{t}"] = ((d.line == l) & (d.tp == t)).astype(float)
    return pd.DataFrame(cols, index=d.index)


def window_or(d: pd.DataFrame, ref: str) -> pd.DataFrame:
    """Time-averaged dead-vs-live odds ratio of each line vs ``ref`` on exactly the rows of a retention window.

    Binomial GLM of (D, L) per well x timepoint with timepoint effects and line terms, well-clustered SEs, so the
    odds ratio covers the same wells and timepoints as the retention and calibration models.

    Returns:
        One row per ordered line pair: Reference, Compare, est (log odds ratio), se.
    """
    lines = sorted(set(d.line))
    X = pd.get_dummies(d.tp.astype(str), prefix="tp", drop_first=True).astype(float)
    for l in lines:
        if l != ref:
            X[f"o:{l}"] = (d.line == l).astype(float)
    X = sm.add_constant(X)
    y = np.column_stack([d.D.to_numpy(float), d.L.to_numpy(float)])
    r = sm.GLM(y, X, family=sm.families.Binomial()).fit(cov_type="cluster",
                                                        cov_kwds={"groups": pd.Categorical(d.well).codes})
    b, V = r.params, r.cov_params()
    rows = []
    for a in lines:
        for c in lines:
            if a == c:
                continue
            w = pd.Series(0.0, index=b.index)
            if c != ref:
                w[f"o:{c}"] += 1
            if a != ref:
                w[f"o:{a}"] -= 1
            rows.append({"Reference": a, "Compare": c, "est": float(w @ b), "se": float(np.sqrt(w @ V @ w))})
    return pd.DataFrame(rows)

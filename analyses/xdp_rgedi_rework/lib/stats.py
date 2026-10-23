"""Small statistics helpers shared by several steps."""

from typing import Optional

import numpy as np


def dl_meta(est, lo, hi):
    """DerSimonian-Laird random-effects pooling of per-experiment odds ratios (SE from the 95% CI)."""
    y = np.log(np.asarray(est, float))
    se = (np.log(np.asarray(hi, float)) - np.log(np.asarray(lo, float))) / (2 * 1.96)
    k = len(y)
    if k == 0:
        return dict(OR=np.nan, lo=np.nan, hi=np.nan, I2=np.nan, k=0)
    if k == 1:
        return dict(OR=float(np.exp(y[0])), lo=float(np.exp(y[0] - 1.96 * se[0])), hi=float(np.exp(y[0] + 1.96 * se[0])),
                    I2=np.nan, k=1)
    w = 1 / se ** 2
    ybar = np.sum(w * y) / np.sum(w)
    Q = float(np.sum(w * (y - ybar) ** 2))
    tau2 = max(0.0, (Q - (k - 1)) / (np.sum(w) - np.sum(w ** 2) / np.sum(w)))
    ws = 1 / (se ** 2 + tau2)
    mu = np.sum(ws * y) / np.sum(ws)
    sm = np.sqrt(1 / np.sum(ws))
    return dict(OR=float(np.exp(mu)), lo=float(np.exp(mu - 1.96 * sm)), hi=float(np.exp(mu + 1.96 * sm)),
                I2=float(max(0.0, (Q - (k - 1)) / Q)) if Q > 0 else 0.0, k=k)


def dl_pool(y, se, shared: Optional[np.ndarray] = None, shared_var: float = 0.0) -> dict:
    """Random-effects pooling of estimates on their own (log) scale, with small-k intervals.

    DerSimonian-Laird tau^2; the confidence interval uses the Hartung-Knapp-Sidik-Jonkman variance (truncated so
    it is never narrower than the DL one) with t(k-1) quantiles, and the prediction interval for a new study uses
    t(k-2) with tau^2 + se^2 (Higgins-Thompson-Spiegelhalter). ``shared`` flags studies that carry one common
    estimated input (e.g. the same pooled capture ratio); its variance ``shared_var`` is added once to the pooled
    estimate with the summed weights of those studies, instead of being treated as independent per study.

    Args:
        y: per-study estimates (e.g. log ratios).
        se: their standard errors (excluding any shared component).
        shared: optional boolean mask of studies carrying the shared input.
        shared_var: variance of the shared input.

    Returns:
        mu, se_mu (DL), tau2, Q, Q_p (chi-square, k-1 df), I2, k, and lo/hi (HKSJ CI), pred_lo/pred_hi (log scale).
    """
    from scipy import stats
    y, se = np.asarray(y, float), np.asarray(se, float)
    sh = np.zeros(len(y), bool) if shared is None else np.asarray(shared, bool)
    ok = np.isfinite(y) & np.isfinite(se) & (se > 0)
    y, se, sh = y[ok], se[ok], sh[ok]
    k = len(y)
    nan = dict(mu=np.nan, se_mu=np.nan, tau2=np.nan, Q=np.nan, Q_p=np.nan, I2=np.nan, k=k, lo=np.nan, hi=np.nan,
               pred_lo=np.nan, pred_hi=np.nan)
    if k == 0:
        return nan
    if k == 1:
        v = se[0] ** 2 + (shared_var if sh[0] else 0.0)
        return {**nan, "mu": float(y[0]), "se_mu": float(np.sqrt(v)), "lo": float(y[0] - 1.96 * np.sqrt(v)),
                "hi": float(y[0] + 1.96 * np.sqrt(v))}
    w = 1 / se ** 2
    ybar = np.sum(w * y) / np.sum(w)
    Q = float(np.sum(w * (y - ybar) ** 2))
    tau2 = max(0.0, (Q - (k - 1)) / (np.sum(w) - np.sum(w ** 2) / np.sum(w)))
    ws = 1 / (se ** 2 + tau2)
    mu = float(np.sum(ws * y) / np.sum(ws))
    shared_term = (np.sum(ws[sh]) / np.sum(ws)) ** 2 * shared_var if sh.any() else 0.0
    var_dl = 1 / np.sum(ws) + shared_term
    q = float(np.sum(ws * (y - mu) ** 2) / (k - 1))
    var_hk = max(q, 1.0) / np.sum(ws) + shared_term
    t1 = stats.t.ppf(0.975, k - 1)
    out = dict(mu=mu, se_mu=float(np.sqrt(var_dl)), tau2=float(tau2), Q=Q, Q_p=float(stats.chi2.sf(Q, k - 1)),
               I2=float(max(0.0, (Q - (k - 1)) / Q)) if Q > 0 else 0.0, k=k,
               lo=float(mu - t1 * np.sqrt(var_hk)), hi=float(mu + t1 * np.sqrt(var_hk)), pred_lo=np.nan, pred_hi=np.nan)
    if k >= 3:
        t2 = stats.t.ppf(0.975, k - 2)
        out["pred_lo"], out["pred_hi"] = mu - t2 * np.sqrt(tau2 + var_dl), mu + t2 * np.sqrt(tau2 + var_dl)
    return out

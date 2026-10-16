"""Mixture fits on log10(GEDI ratio) and the D02 cutoff rule.

D02 (Julia's SD convention, S.A.-informed):
    a = mu_L + k*sd_L        (dead cutoff: <= 1-Phi(k) of live cells exceed it)
    b = mu_D - k*sd_D        (live cutoff: <= 1-Phi(k) of dead cells fall below it)
    live_thr = 10**min(a, b),  dead_thr = 10**max(a, b),  crossed = b > a

Component identification has two pre-specified variants:
    'k2_sorted' - 2 components, L = lower mean, D = upper mean (plan R3 primary)
    'k3_mapped' - 3 components; L = component with the highest responsibility at
                  the untreated (reference) mode; D = highest-mean component above
                  L with weight >= 0.05; any component below L is a 'low tail'
                  and plays no part in the cutoffs (plan R3 S1 / decision D18)
"""

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

import numpy as np
from scipy.stats import norm
from sklearn.mixture import GaussianMixture

PHI = norm.cdf


@dataclass
class Fit:
    k: int
    mu: np.ndarray
    sd: np.ndarray
    w: np.ndarray
    bic: float
    loglik: float
    n: int
    converged: bool

    def pdf(self, x: np.ndarray) -> np.ndarray:
        return sum(w * norm.pdf(x, m, s) for m, s, w in zip(self.mu, self.sd, self.w))

    def cdf(self, x: np.ndarray) -> np.ndarray:
        return sum(w * PHI((x - m) / s) for m, s, w in zip(self.mu, self.sd, self.w))

    def posterior(self, x: np.ndarray) -> np.ndarray:
        dens = np.vstack([w * norm.pdf(x, m, s) for m, s, w in zip(self.mu, self.sd, self.w)])
        return dens / np.maximum(dens.sum(axis=0), 1e-300)


def fit_gmm(x: np.ndarray, k: int, mp: dict, seed: int, init: Optional[Fit] = None) -> Fit:
    """Fit a k-component 1-D Gaussian mixture; components returned sorted by mean."""
    x = np.asarray(x, float).reshape(-1, 1)
    kw = dict(n_components=k, reg_covar=mp["reg_covar"], tol=mp["tol"], max_iter=mp["max_iter"],
              random_state=seed)
    if init is not None and init.k == k:
        g = GaussianMixture(**kw, n_init=1, means_init=init.mu.reshape(-1, 1),
                            precisions_init=(1.0 / init.sd ** 2).reshape(-1, 1, 1),
                            weights_init=init.w / init.w.sum())
    else:
        g = GaussianMixture(**kw, n_init=mp["n_init"], init_params=mp["init_params"])
    g.fit(x)
    o = np.argsort(g.means_.ravel())
    return Fit(k=k, mu=g.means_.ravel()[o], sd=np.sqrt(g.covariances_.ravel()[o]), w=g.weights_[o],
               bic=float(g.bic(x)), loglik=float(g.score(x) * len(x)), n=len(x),
               converged=bool(g.converged_))


def identify(fit: Fit, variant: str, ref_mode: Optional[float]) -> Dict[str, Optional[int]]:
    """Return indices of the live (L) and dead (D) components (D may be None)."""
    if variant == "k2_sorted":
        return {"L": 0, "D": 1 if fit.k > 1 else None}
    if variant == "k3_mapped":
        if ref_mode is None:
            raise ValueError("k3_mapped needs the reference (untreated) mode")
        live = int(np.argmax(fit.posterior(np.array([ref_mode]))[:, 0]))
        above = [i for i in range(fit.k) if fit.mu[i] > fit.mu[live] and fit.w[i] >= 0.05]
        dead = int(max(above, key=lambda i: fit.mu[i])) if above else None
        return {"L": live, "D": dead}
    raise ValueError(variant)


def cutoffs(mu_l: float, sd_l: float, mu_d: float, sd_d: float, k_sd: float) -> Dict[str, float]:
    a = mu_l + k_sd * sd_l
    b = mu_d - k_sd * sd_d
    lo, hi = min(a, b), max(a, b)
    return {"a_log": a, "b_log": b, "live_thr_log": lo, "dead_thr_log": hi,
            "live_thr": 10 ** lo, "dead_thr": 10 ** hi, "crossed": bool(b > a)}


def ashman_d(mu_l: float, sd_l: float, mu_d: float, sd_d: float) -> float:
    return float(np.sqrt(2) * abs(mu_d - mu_l) / np.sqrt(sd_l ** 2 + sd_d ** 2))


def ks_distance(x: np.ndarray, fit: Fit) -> float:
    xs = np.sort(x)
    emp = np.arange(1, len(xs) + 1) / len(xs)
    return float(np.max(np.abs(emp - fit.cdf(xs))))


def refit_weights(x: np.ndarray, fit: Fit, iters: int = 200) -> np.ndarray:
    """EM on mixing weights only (components fixed) - model-based composition of another sample."""
    w = np.full(fit.k, 1.0 / fit.k)
    dens = np.vstack([norm.pdf(x, m, s) for m, s in zip(fit.mu, fit.sd)])
    for _ in range(iters):
        r = w[:, None] * dens
        r /= np.maximum(r.sum(axis=0), 1e-300)
        w_new = r.mean(axis=1)
        if np.max(np.abs(w_new - w)) < 1e-9:
            break
        w = w_new
    return w


@dataclass
class ThresholdResult:
    variant: str
    k: int
    n_fit: int
    n_trim_lo: int
    n_trim_hi: int
    mu_L: float
    sd_L: float
    w_L: float
    mu_D: Optional[float]
    sd_D: Optional[float]
    w_D: Optional[float]
    ok: bool
    reason: str = ""
    cut: Dict[str, float] = field(default_factory=dict)
    diag: Dict[str, float] = field(default_factory=dict)
    fit: Optional[Fit] = None

    def row(self) -> dict:
        d = {k: v for k, v in asdict(self).items() if k not in ("cut", "diag", "fit")}
        d.update(self.cut)
        d.update(self.diag)
        if self.fit is not None:
            d.update({f"comp{i}_{p}": float(getattr(self.fit, p)[i]) for i in range(self.fit.k)
                      for p in ("mu", "sd", "w")})
        return d


def fit_thresholds(x: np.ndarray, variant: str, mp: dict, k_sd: float, seed: int,
                   ref_mode: Optional[float] = None, init: Optional[Fit] = None,
                   diagnostics: bool = True) -> ThresholdResult:
    """Full R3 procedure on one sample of log10 ratios: fit, trim, refit, identify, cut."""
    k = 2 if variant == "k2_sorted" else 3
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    f1 = fit_gmm(x, k, mp, seed, init)
    ids = identify(f1, variant, ref_mode)
    if ids["D"] is None:
        return ThresholdResult(variant, k, len(x), 0, 0, float(f1.mu[ids["L"]]), float(f1.sd[ids["L"]]),
                               float(f1.w[ids["L"]]), None, None, None, False,
                               "no dead component above live", fit=f1)
    L, D = ids["L"], ids["D"]
    lo = f1.mu[L] - mp["trim_sd"] * f1.sd[L]       # trim relative to the live and dead components
    hi = f1.mu[D] + mp["trim_sd"] * f1.sd[D]
    keep = (x >= lo) & (x <= hi)
    f = fit_gmm(x[keep], k, mp, seed, f1)
    ids = identify(f, variant, ref_mode)
    if ids["D"] is None:
        return ThresholdResult(variant, k, int(keep.sum()), int((x < lo).sum()), int((x > hi).sum()),
                               float(f.mu[ids["L"]]), float(f.sd[ids["L"]]), float(f.w[ids["L"]]),
                               None, None, None, False, "no dead component above live after trim", fit=f)
    L, D = ids["L"], ids["D"]
    cut = cutoffs(f.mu[L], f.sd[L], f.mu[D], f.sd[D], k_sd)
    res = ThresholdResult(variant, k, int(keep.sum()), int((x < lo).sum()), int((x > hi).sum()),
                          float(f.mu[L]), float(f.sd[L]), float(f.w[L]), float(f.mu[D]), float(f.sd[D]),
                          float(f.w[D]), True, "", cut, {}, f)
    if diagnostics:
        xx = x[keep]
        post = f.posterior(xx)
        live_assigned = xx[np.argmax(post, axis=0) == L]
        iqr_sd = (np.subtract(*np.percentile(live_assigned, [75, 25])) / 1.349
                  if len(live_assigned) > 10 else np.nan)
        res.diag = {
            "ashman_d": ashman_d(f.mu[L], f.sd[L], f.mu[D], f.sd[D]),
            "ks": ks_distance(xx, f),
            "sd_L_over_iqr": float(f.sd[L] / iqr_sd) if iqr_sd and np.isfinite(iqr_sd) else np.nan,
            "frac_ambiguous": float(np.mean((xx > cut["live_thr_log"]) & (xx < cut["dead_thr_log"]))),
            "frac_called_live": float(np.mean(xx < cut["live_thr_log"])),
            "frac_called_dead": float(np.mean(xx > cut["dead_thr_log"])),
            "retention_live": float(PHI((cut["live_thr_log"] - f.mu[L]) / f.sd[L])),
            "retention_dead": float(1 - PHI((cut["dead_thr_log"] - f.mu[D]) / f.sd[D])),
            "frac_maxpost_lt_0.9": float(np.mean(post.max(axis=0) < 0.9)),
        }
    return res


def mixture_bic_table(x: np.ndarray, mp: dict, seed: int, ks=(1, 2, 3)) -> Dict[int, float]:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return {k: fit_gmm(x, k, mp, seed).bic for k in ks}


def kde_mode(x: np.ndarray, grid: Optional[np.ndarray] = None) -> float:
    """Model-free mode of a 1-D sample (Gaussian KDE, Scott bandwidth)."""
    from scipy.stats import gaussian_kde
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) > 20000:
        x = np.random.default_rng(0).choice(x, 20000, replace=False)
    grid = grid if grid is not None else np.linspace(np.percentile(x, 1), np.percentile(x, 99), 600)
    return float(grid[np.argmax(gaussian_kde(x)(grid))])


def well_bootstrap(cells_by_well: List[np.ndarray], variant: str, mp: dict, k_sd: float, seed: int,
                   ref_mode: Optional[float], base: ThresholdResult, B: int,
                   stratified_cells: bool = False) -> List[dict]:
    """Cluster bootstrap over wells (or cells within wells when there are too few wells)."""
    rng = np.random.default_rng(seed)
    out = []
    nw = len(cells_by_well)
    for b in range(B):
        if stratified_cells:
            xb = np.concatenate([rng.choice(c, len(c), replace=True) for c in cells_by_well])
        else:
            idx = rng.integers(0, nw, nw)
            xb = np.concatenate([cells_by_well[i] for i in idx])
        r = fit_thresholds(xb, variant, mp, k_sd, seed + b + 1, ref_mode, init=base.fit, diagnostics=False)
        row = {"b": b, "ok": r.ok, "mu_L": r.mu_L, "sd_L": r.sd_L, "mu_D": r.mu_D, "sd_D": r.sd_D,
               "w_D": r.w_D}
        row.update(r.cut)
        out.append(row)
    return out


# ---------------------------------------------------------------------------------------------------------
# Live population of reference (untreated) cells and the D20 cutoff rules.

def live_reference(x: np.ndarray):
    """Centre = KDE mode; spread from the side below the mode, which dead cells cannot reach.

    Returns (mode, sd_inner, sd_half): sd_inner = (mode - median of cells below the mode) / 0.674,
    sd_half = mode - 31.73rd percentile of cells below the mode.
    """
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    m = kde_mode(x)
    left = np.sort(x[x < m])
    return m, (m - np.median(left)) / 0.674, m - left[int(0.3173 * len(left))]


def equal_density(m: float, s: float, mu_d: float, s_d: float) -> float:
    """Ratio (log10) between the two centres where the live and dead Gaussian densities are equal."""
    from scipy.optimize import brentq
    f = lambda v: norm.logpdf(v, m, s) - norm.logpdf(v, mu_d, s_d)  # noqa: E731
    try:
        return float(brentq(f, m, mu_d))
    except ValueError:
        return float("nan")


def rule_cutoffs(m: float, s_live: float, mu_d: float, s_d: float, k_sd: float) -> Dict[str, tuple]:
    """D20 rules, all in log10: B = (min(a,b), max(a,b)) with a = m + k s_live, b = mu_d - k s_d;
    C = single cutoff b; D = single cutoff at equal density."""
    a, b = m + k_sd * s_live, mu_d - k_sd * s_d
    d = equal_density(m, s_live, mu_d, s_d)
    return {"B": (min(a, b), max(a, b)), "C": (b, b), "D": (d, d)}


def window_edges(m: float, s: float, mu_d: float, s_d: float, odds: float) -> tuple:
    """Confidence window around the equal-density cutoff (D28), in log10.

    Live edge: the first value below the centre where the live density is ``odds`` times the dead density;
    dead edge: the first value above the centre where the dead density is ``odds`` times the live density
    (equal priors, so the centre is rule D). Returns (live_edge, dead_edge); an edge is NaN when that
    confidence is never reached on its side (heavily overlapping populations).
    """
    from scipy.optimize import brentq
    c = equal_density(m, s, mu_d, s_d)
    if not np.isfinite(c):
        return float("nan"), float("nan")
    k = np.log(odds)
    llr = lambda v: norm.logpdf(v, mu_d, s_d) - norm.logpdf(v, m, s)  # noqa: E731
    out = []
    for target, grid in ((-k, np.linspace(c, m - 8 * s, 4000)), (k, np.linspace(c, mu_d + 8 * s_d, 4000))):
        f = llr(grid) - target
        hit = np.nonzero(np.sign(f[1:]) != np.sign(f[:-1]))[0]
        out.append(float(brentq(lambda v: llr(v) - target, grid[hit[0]], grid[hit[0] + 1])) if len(hit) else float("nan"))
    return out[0], out[1]

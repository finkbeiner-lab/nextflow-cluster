#!/usr/bin/env python
"""Tiny dependency-free 1-D two-component Gaussian mixture (EM).

Replaces scikit-learn's GaussianMixture so the RGEDI gate/threshold modules run in
the pipeline container (numpy only, no sklearn). Used to find the background/live
(lower) mode of a log-intensity or log-ratio distribution.
"""
import numpy as np


def _npdf(x, mu, var):
    return np.exp(-0.5 * (x - mu) ** 2 / var) / np.sqrt(2 * np.pi * var)


def fit2(x, iters=200, tol=1e-7):
    """2-component 1-D GMM: k-means init (finds the two bulks) + soft-EM refine.

    k-means init anchors the two components on the background and expressing bulks;
    the soft-EM refine from there gives proper likelihoods/BIC without the max-
    likelihood drift onto the dim tail that plagues random-init EM here.
    Keys: mu[2] (ascending), sigma[2], w[2], bic2, bic1, assign (0 = lower bulk).
    """
    x = np.asarray(x, float).ravel()
    n = len(x)
    # --- k-means (1-D, 2 clusters) init ---
    c = np.array([np.percentile(x, 25), np.percentile(x, 75)])
    for _ in range(60):
        a = (np.abs(x - c[1]) < np.abs(x - c[0])).astype(int)
        nc = np.array([x[a == 0].mean() if (a == 0).any() else c[0],
                       x[a == 1].mean() if (a == 1).any() else c[1]])
        if np.allclose(nc, c):
            break
        c = nc
    mu = np.sort(c)
    var = np.array([max(x[a == 0].var(), 1e-4) if (a == 0).any() else x.var(),
                    max(x[a == 1].var(), 1e-4) if (a == 1).any() else x.var()])
    w = np.array([0.5, 0.5])
    # --- soft-EM refine ---
    ll = -np.inf; r = np.full((2, n), 0.5)
    for _ in range(iters):
        p = np.stack([w[k] * _npdf(x, mu[k], var[k]) for k in range(2)])
        s = p.sum(0) + 1e-300
        r = p / s
        ll_new = float(np.sum(np.log(s)))
        nk = r.sum(1) + 1e-9
        w = nk / n
        mu = (r * x).sum(1) / nk
        var = (r * (x - mu[:, None]) ** 2).sum(1) / nk + 1e-4
        if abs(ll_new - ll) < tol:
            ll = ll_new; break
        ll = ll_new
    order = np.argsort(mu)
    mu, var, w, r = mu[order], var[order], w[order], r[order]
    bic2 = -2 * ll + 5 * np.log(n)
    ll1 = float(np.sum(np.log(_npdf(x, x.mean(), x.var() + 1e-6) + 1e-300)))
    bic1 = -2 * ll1 + 2 * np.log(n)
    return dict(mu=mu, sigma=np.sqrt(var), w=w, bic2=bic2, bic1=bic1,
                assign=np.argmax(r, 0))

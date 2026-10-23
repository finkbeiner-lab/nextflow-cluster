"""Step 05b: model adequacy - does a 2-component mixture separate live from dead S.A. cells?

For each S.A. experiment x line (fit set = post-dose S.A. wells at the fit timepoints; XDP3 uses the
provisional onset T10 until D12 is decided) fit 1, 2 and 3 Gaussian components to log10 ratio and
show what each component captures. No cutoffs are computed here (that is step 06). This is the
evidence for decision D18.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import norm  # noqa: E402

from lib import cfg, data, figs  # noqa: E402
from lib import mixture as mx  # noqa: E402
from lib.paths import LOCAL_OUT  # noqa: E402
from lib.provenance import StepRecorder  # noqa: E402

STEP = "05b_model_check"
plt = figs.plt


def fit_set(exp: str):
    s = cfg.experiments()[exp]["sa"]
    if exp == "XDP3":
        return [10], 9           # provisional (D12)
    return s["fit_tps"], s["pre_tp"]


def reference_cells(d: pd.DataFrame, exp: str, line: str, fit_tps, pre):
    """Untreated cells at the fit timepoints (XDP1: all wells at the pre-dose timepoint)."""
    if exp == "XDP1":
        return d[(d.line == line) & (d.tp == pre)].log10_ratio.dropna().to_numpy()
    return d[(d.line == line) & (d.condition == "untreated") & d.tp.isin(fit_tps)].log10_ratio.dropna().to_numpy()


def main() -> None:
    P = cfg.params()
    mp, seed = P["mixture"], P["seed"]
    rec = StepRecorder(STEP, LOCAL_OUT / STEP, mp)
    rows = []
    exps = ["XDP1", "XDP3", "XDP5", "XDP10", "XDP11"]
    for exp in exps:
        d = data.load(exp, columns=["well", "line", "condition", "tp", "log10_ratio", "in_scope"])
        tps, pre = fit_set(exp)
        lines = [l for l in figs.LINE_ORDER if l in set(d.line)]
        fig, axes = plt.subplots(1, len(lines), figsize=(3.3 * len(lines), 3.0), sharey=True, squeeze=False)
        bins = np.linspace(-2.6, 0.3, 110)
        for ax, line in zip(axes[0], lines):
            x = d[(d.line == line) & (d.condition == "sa") & d.tp.isin(tps)].log10_ratio.dropna().to_numpy()
            ref = reference_cells(d, exp, line, tps, pre)
            mode = mx.kde_mode(ref)
            fits = {k: mx.fit_gmm(x, k, mp, seed) for k in (1, 2, 3)}
            k3 = fits[3]
            ids3 = mx.identify(k3, "k3_mapped", mode)
            low = [i for i in range(3) if k3.mu[i] < k3.mu[ids3["L"]]]
            row = {"exp": exp, "line": line, "fit_tps": ",".join(map(str, tps)), "n_cells": len(x),
                   "n_wells": int(d[(d.line == line) & (d.condition == "sa")].well.nunique()),
                   "ref_mode_log": mode, "ref_mode_ratio": 10 ** mode,
                   "bic1": fits[1].bic, "bic2": fits[2].bic, "bic3": fits[3].bic,
                   "k2_lower_mu": fits[2].mu[0], "k2_lower_sd": fits[2].sd[0], "k2_lower_w": fits[2].w[0],
                   "k2_upper_mu": fits[2].mu[1], "k2_upper_sd": fits[2].sd[1], "k2_upper_w": fits[2].w[1],
                   "k2_lower_is_live": bool(np.argmax(fits[2].posterior(np.array([mode]))[:, 0]) == 0),
                   "k3_live_mu": k3.mu[ids3["L"]], "k3_live_sd": k3.sd[ids3["L"]], "k3_live_w": k3.w[ids3["L"]],
                   "k3_dead_mu": k3.mu[ids3["D"]] if ids3["D"] is not None else np.nan,
                   "k3_dead_sd": k3.sd[ids3["D"]] if ids3["D"] is not None else np.nan,
                   "k3_dead_w": k3.w[ids3["D"]] if ids3["D"] is not None else np.nan,
                   "k3_lowtail_w": float(sum(k3.w[i] for i in low)),
                   "k3_lowtail_mu": float(k3.mu[low[0]]) if low else np.nan}
            rows.append(row)
            ax.hist(x, bins=bins, density=True, color=figs.SA, alpha=0.18, lw=0, label="post-dose S.A. cells")
            if len(ref):
                ax.hist(ref, bins=bins, density=True, histtype="step", color=figs.UT, lw=1.2,
                        label="reference (untreated)" if exp != "XDP1" else f"same wells, T{pre}")
            gx = np.linspace(-2.6, 0.3, 400)
            f2 = fits[2]
            for i in range(2):
                ax.plot(gx, f2.w[i] * norm.pdf(gx, f2.mu[i], f2.sd[i]), color=figs.INK2, lw=1.1, ls="--",
                        label="2-component fit" if i == 0 else None)
            role = {ids3["L"]: ("live", figs.LIVE)}
            if ids3["D"] is not None:
                role[ids3["D"]] = ("dead", figs.DEAD)
            for i in range(3):
                name, color = role.get(i, ("low tail", figs.LOWTAIL))
                ax.plot(gx, k3.w[i] * norm.pdf(gx, k3.mu[i], k3.sd[i]), color=color, lw=2.0,
                        label=f"3-comp: {name}")
            figs.log_ratio_axis(ax)
            ax.set_xlim(-2.5, 0.2)
            ax.set_title(f"{figs.LINE_SHORT[line]}  (n = {len(x):,})", fontsize=9)
            ax.set_xlabel("GEDI ratio (log scale)")
        axes[0][0].set_ylabel("density")
        axes[0][-1].legend(fontsize=6.5, loc="upper left")
        fig.suptitle(f"{exp}: post-dose S.A. cells (T{','.join(map(str, tps))}) - 2-component fit (dashed) vs "
                     "3-component fit (live / dead / low tail)", x=0.01, ha="left", fontsize=10)
        path = figs.save(fig, rec.out_dir / f"F_model_check_{exp}.png")
        rec.figure(f"Fmodel_{exp}", path, f"{exp}: what a 2-component and a 3-component mixture capture in the "
                   "post-dose S.A. cells. Grey outline = untreated (reference) cells at the same timepoint.", exp=exp)
    df = pd.DataFrame(rows)
    df.to_csv(rec.out_dir / "model_check.csv", index=False)
    rec.summary = {
        "n_cases": len(df),
        "k2_lower_component_is_live": int(df.k2_lower_is_live.sum()),
        "k2_lower_weight_median": float(df.k2_lower_w.median()),
        "k3_lowtail_weight_range": [float(df.k3_lowtail_w.min()), float(df.k3_lowtail_w.max())],
        "bic_prefers_3_over_2": int((df.bic3 < df.bic2).sum()),
        "rows": df.round(4).to_dict("records"),
    }
    rec.check("model.k2_identifies_live", "2-component fit: the lower component holds the untreated (live) mode "
              "in every case", bool(df.k2_lower_is_live.all()),
              {"cases_where_lower_is_live": int(df.k2_lower_is_live.sum()), "of": len(df)})
    rec.write()
    print(df[["exp", "line", "n_cells", "k2_lower_is_live", "k2_lower_w", "k3_live_mu", "k3_dead_mu", "k3_dead_w",
              "k3_lowtail_w"]].round(3).to_string())


if __name__ == "__main__":
    main()

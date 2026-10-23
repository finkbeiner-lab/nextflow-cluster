"""Step 05c: cutoff preview - what each candidate cutoff rule would do, case by case (evidence for D20).

Populations (log10 ratio), per experiment x line at the S.A. fit timepoints:
  untreated live  - untreated wells (XDP1: the same wells at pre-dose T10). Centre = mode (kernel density);
                    spread from the side BELOW the mode, which dead cells cannot reach:
                      sd_half  = mode - 31.73rd pct of cells below the mode   (whole left half)
                      sd_inner = (mode - median of cells below the mode)/0.674 (inner left half, less
                                 sensitive to the far-left shoulder)
  S.A. dead       - the dead component of a 3-component mixture fitted to post-dose S.A. cells
  S.A. survivors  - the live component of that same fit
Candidate rules (k = 2.5 SD throughout):
  A  D02 as first implemented: dead = survivor mean + k SD; live = dead mean - k SD (all from S.A. wells)
  B  D02 with the live side from untreated wells: dead = mode + k sd_inner; live = dead mean - k SD
  C  dead-side single cutoff: live = dead = dead mean - k SD (a cell is 'dead' when it lies within the
     S.A.-dead population's range or above)
  D  equal-density boundary between untreated live (mode, sd_inner) and S.A. dead (single cutoff)
Usability (R4, U4): dead-component weight in S.A. cells minus the weight of the SAME component in the reference
cells (weights re-estimated with the components held fixed).
For each rule: % of the S.A. dead population called dead, % of untreated cells called live / ambiguous /
dead, and the MRID-style death fraction dead/(live+dead) in untreated and S.A. wells.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.optimize import brentq  # noqa: E402
from scipy.stats import norm  # noqa: E402

from lib import cfg, data, figs  # noqa: E402
from lib import mixture as mx  # noqa: E402
from lib.paths import LOCAL_OUT  # noqa: E402
from lib.provenance import StepRecorder  # noqa: E402

STEP = "05c_cutoff_preview"
plt = figs.plt
# Fit sets as proposed for review point A (XDP3 onset T10 = D12; XDP1 second dose = D14).
FIT = {"XDP1": ([13, 14], 10), "XDP3": ([10], 9), "XDP5": ([9], 7), "XDP10": ([9, 10], 8), "XDP11": ([13], 12)}
RULE_NAMES = {"A": "both sides from S.A. wells", "B": "live side from untreated wells",
              "C": "single cutoff from the S.A. dead population", "D": "equal-density boundary"}


def live_reference(x: np.ndarray):
    m = mx.kde_mode(x)
    left = np.sort(x[x < m])
    sd_half = m - left[int(0.3173 * len(left))]
    sd_inner = (m - np.median(left)) / 0.674
    return m, sd_half, sd_inner


def equal_density(m, s, mu_d, s_d):
    f = lambda v: norm.logpdf(v, m, s) - norm.logpdf(v, mu_d, s_d)  # noqa: E731
    try:
        return brentq(f, m, mu_d)
    except ValueError:
        return np.nan


def consequences(lo, hi, ut, sa, mu_d, s_d):
    def frac(x):
        live, dead = np.mean(x < lo), np.mean(x > hi)
        return live, 1 - live - dead, dead, dead / (live + dead) if live + dead else np.nan
    ul, ua, ud, udf = frac(ut)
    sl, sa_, sd, sdf = frac(sa)
    return {"live_thr": 10 ** lo, "dead_thr": 10 ** hi, "dead_pop_called_dead": 1 - norm.cdf((hi - mu_d) / s_d),
            "ut_live": ul, "ut_ambiguous": ua, "ut_dead": ud, "ut_death_frac": udf, "sa_death_frac": sdf,
            "sa_ambiguous": sa_}


def summarize(pop: pd.DataFrame, rules: pd.DataFrame, P: dict) -> dict:
    """Ranges and rule medians over USABLE cases only (dead population found, Ashman's D >= the usability limit)."""
    pop = pop.copy()
    pop["usable_sep"] = pop.dead_found.astype(bool) & (pop.ashman_ut_dead >= P["usability"]["min_ashman_d"])
    u = pop[pop.usable_sep]
    keys = set(zip(u.exp, u.line))
    ru = rules[[k in keys for k in zip(rules.exp, rules.line)]]
    rB = ru[ru.rule == "B"]
    return {
        "cases": len(pop), "cases_with_dead_population": int(pop.dead_found.sum()), "cases_separated": len(u),
        "not_separated": [f"{e} {l}" for e, l in zip(pop[pop.dead_found.astype(bool) & ~pop.usable_sep].exp,
                                                       pop[pop.dead_found.astype(bool) & ~pop.usable_sep].line)],
        "ut_sd_inner_range": [float(pop.ut_sd_inner.min()), float(pop.ut_sd_inner.max())],
        "ut_sd_half_range": [float(pop.ut_sd_half.min()), float(pop.ut_sd_half.max())],
        "dead_mu_range": [float(u.dead_mu.min()), float(u.dead_mu.max())],
        "dead_sd_range": [float(u.dead_sd.min()), float(u.dead_sd.max())],
        "sep_dex_range": [float(u.sep_dex.min()), float(u.sep_dex.max())],
        "needed_dex_range": [float(u.needed_for_two_cutoffs_dex.min()), float(u.needed_for_two_cutoffs_dex.max())],
        "two_cutoff_bands_overlap_cases": int((u.sep_dex < u.needed_for_two_cutoffs_dex).sum()),
        "rule_B_dead_cutoff_above_dead_centre": int((rB.dead_pop_called_dead < 0.5).sum()),
        "rule_medians": ru.groupby("rule")[["dead_pop_called_dead", "ut_ambiguous", "ut_death_frac", "sa_death_frac"]]
        .median().round(3).to_dict("index"),
        "rule_C_cutoff_range": [float(ru[ru.rule == "C"].live_thr.min()), float(ru[ru.rule == "C"].live_thr.max())],
        "rule_C_ut_death_range": [float(ru[ru.rule == "C"].ut_death_frac.min()), float(ru[ru.rule == "C"].ut_death_frac.max())],
    }


def main() -> None:
    P = cfg.params()
    mp, seed, k = P["mixture"], P["seed"], P["cutoff_rule"]["k_sd"]
    rec = StepRecorder(STEP, LOCAL_OUT / STEP, {"k_sd": k, "fit_sets": FIT})
    rows, rule_rows = [], []
    for exp, (tps, pre) in FIT.items():
        d = data.load(exp, columns=["line", "condition", "tp", "log10_ratio", "in_scope"])
        lines = [l for l in figs.LINE_ORDER if l in set(d.line)]
        fig, axes = plt.subplots(1, len(lines), figsize=(3.4 * len(lines), 3.1), sharey=True, squeeze=False)
        bins = np.linspace(-2.6, 0.3, 110)
        gx = np.linspace(-2.6, 0.3, 500)
        for ax, line in zip(axes[0], lines):
            sa = d[(d.line == line) & (d.condition == "sa") & d.tp.isin(tps)].log10_ratio.dropna().to_numpy()
            ut = (d[(d.line == line) & (d.tp == pre)] if exp == "XDP1" else
                  d[(d.line == line) & (d.condition == "untreated") & d.tp.isin(tps)]).log10_ratio.dropna().to_numpy()
            m, sd_half, sd_inner = live_reference(ut)
            f3 = mx.fit_gmm(sa, 3, mp, seed)
            ids = mx.identify(f3, "k3_mapped", m)
            row = {"exp": exp, "line": line, "fit_tps": ",".join(map(str, tps)), "ref": f"T{pre} same wells" if exp == "XDP1"
                   else "untreated, same tps", "n_sa": len(sa), "n_ut": len(ut), "ut_mode": 10 ** m,
                   "ut_sd_half": sd_half, "ut_sd_inner": sd_inner, "sa_live_mu": 10 ** f3.mu[ids["L"]],
                   "sa_live_sd": f3.sd[ids["L"]]}
            ax.hist(ut, bins=bins, density=True, histtype="step", color=figs.UT, lw=1.3,
                    label="untreated" if exp != "XDP1" else f"same wells T{pre}")
            ax.hist(sa, bins=bins, density=True, color=figs.SA, alpha=0.18, lw=0, label="post-dose S.A.")
            ax.plot(gx, norm.pdf(gx, m, sd_inner) * 0.9 * np.max(np.histogram(ut, bins=bins, density=True)[0])
                    / norm.pdf(m, m, sd_inner), color=figs.LIVE, lw=1.6, label="untreated live (mode, inner SD)")
            if ids["D"] is None:
                row["dead_found"] = False
                rows.append(row)
                ax.set_title(f"{figs.LINE_SHORT[line]} - no dead population", fontsize=9)
            else:
                D, L = ids["D"], ids["L"]
                mu_d, s_d, w_d = f3.mu[D], f3.sd[D], f3.w[D]
                w_ref = mx.refit_weights(ut, f3)          # same components, weights re-estimated on reference cells
                row.update({"dead_found": True, "dead_mu": 10 ** mu_d, "dead_sd": s_d, "dead_w": w_d,
                            "ref_dead_w": float(w_ref[D]), "dead_excess": float(w_d - w_ref[D]),
                            "ashman_ut_dead": mx.ashman_d(m, sd_inner, mu_d, s_d),
                            "sep_dex": mu_d - m, "needed_for_two_cutoffs_dex": k * (sd_inner + s_d)})
                rows.append(row)
                ax.plot(gx, w_d * norm.pdf(gx, mu_d, s_d), color=figs.DEAD, lw=2, label="S.A. dead component")
                cand = {}
                a, b = f3.mu[L] + k * f3.sd[L], mu_d - k * s_d
                cand["A"] = (min(a, b), max(a, b))
                a = m + k * sd_inner
                cand["B"] = (min(a, b), max(a, b))
                cand["C"] = (b, b)
                e = equal_density(m, sd_inner, mu_d, s_d)
                cand["D"] = (e, e)
                for rule, (lo, hi) in cand.items():
                    if np.isnan(lo):
                        continue
                    rule_rows.append({"exp": exp, "line": line, "rule": rule, **consequences(lo, hi, ut, sa, mu_d, s_d)})
                lo, hi = cand["B"]
                ax.axvspan(lo, hi, color=figs.AMBIG, alpha=0.6, lw=0, label="rule B ambiguous band")
                ax.axvline(cand["C"][0], color=figs.DEAD, lw=1.2, ls="--", label="rule C cutoff")
                if not np.isnan(cand["D"][0]):
                    ax.axvline(cand["D"][0], color=figs.INK2, lw=1.2, ls=":", label="rule D cutoff")
                ax.set_title(f"{figs.LINE_SHORT[line]}", fontsize=9)
            figs.log_ratio_axis(ax)
            ax.set_xlim(-2.4, 0.2)
            ax.set_xlabel("GEDI ratio (log scale)")
        axes[0][0].set_ylabel("density")
        axes[0][-1].legend(fontsize=6.3, loc="upper left")
        fig.suptitle(f"{exp}: untreated live population vs S.A.-killed population at T{','.join(map(str, tps))}, "
                     "with candidate cutoffs", x=0.01, ha="left", fontsize=10)
        path = figs.save(fig, rec.out_dir / f"F_preview_{exp}.png")
        rec.figure(f"Fpreview_{exp}", path, f"{exp}: grey = untreated cells, violet = post-dose S.A. cells, blue = "
                   "untreated live population (mode and spread below the mode), orange = S.A. dead component. Grey band = "
                   "ambiguous zone under rule B; dashed orange = rule C; dotted = rule D.", exp=exp)
        print(exp, "done", flush=True)
    pop = pd.DataFrame(rows)
    rules = pd.DataFrame(rule_rows)
    pop.to_csv(rec.out_dir / "populations.csv", index=False)
    rules.to_csv(rec.out_dir / "rule_consequences.csv", index=False)
    rec.summary = summarize(pop, rules, P)
    rec.write()
    print(pop.round(3).to_string())
    print(rules.round(3).to_string())


if __name__ == "__main__":
    if "--summary-only" in sys.argv:
        out = LOCAL_OUT / STEP
        P = cfg.params()
        smry = summarize(pd.read_csv(out / "populations.csv"), pd.read_csv(out / "rule_consequences.csv"), P)
        (out / "summary.json").write_text(__import__("json").dumps(smry, indent=2))
        print(smry)
    else:
        main()

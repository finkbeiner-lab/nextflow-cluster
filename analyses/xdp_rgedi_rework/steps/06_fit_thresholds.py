"""Step 06: final S.A.-informed cutoffs per line x experiment for rules B, C and D (decision D20).

Per S.A. case (experiment x line, fit timepoints from the config: XDP1 T13-T14, XDP3 T10, XDP5 T9,
XDP10 T9-T10, XDP11 T13):
  reference live population  untreated wells at the fit timepoints (XDP1: the same wells at T10);
                             centre = mode, spread = sd_inner (below the mode)          [lib.mixture.live_reference]
  S.A. dead population       dead component of a 3-component mixture on post-dose S.A. cells (D18)
  cutoffs (log10)            B: two cutoffs, 2.5 SD on each side; C: dead mean - 2.5 SD; D: equal density
Usability (R4): >= 500 cells, >= 2 S.A. wells, >= 50 cells per component, BIC gain 1->2 components > 10,
Ashman's D (reference live vs S.A. dead) >= 2, dead share >= 10 points above its share in the reference
cells (components held fixed). Usable cases keep their own cutoffs; the rest go to transfer (step 08).
Uncertainty: wells resampled with replacement (S.A. and reference wells separately; XDP1 jointly;
XDP10 has 2 S.A. wells, so its cells are resampled within those wells and the case is flagged FEW_WELLS).
Each replicate refits the mixture (warm start + one fresh k-means start, better BIC kept).

Usage: python -I steps/06_fit_thresholds.py [B]
"""

import sys
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from lib import cfg, data, figs  # noqa: E402
from lib import mixture as mx  # noqa: E402
from lib.paths import LOCAL_OUT  # noqa: E402
from lib.provenance import StepRecorder  # noqa: E402

STEP = "06_fit_thresholds"
plt = figs.plt
RULES = ("B", "C", "D")


def case_cells(exp: str, line: str):
    e = cfg.experiments()[exp]["sa"]
    fit, pre = e["fit_tps"], e["pre_tp"]
    d = data.load(exp, columns=["well", "line", "condition", "tp", "log10_ratio", "in_scope"])
    d = d[(d.line == line) & d.log10_ratio.notna()]
    sa = d[(d.condition == "sa") & d.tp.isin(fit)]
    ref = d[d.tp == pre] if exp == "XDP1" else d[(d.condition == "untreated") & d.tp.isin(fit)]
    by = lambda t: {w: g.log10_ratio.to_numpy() for w, g in t.groupby("well")}  # noqa: E731
    return by(sa), by(ref), fit, pre


def fit_case(sa: np.ndarray, ref: np.ndarray, mp: dict, k_sd: float, seed: int, init=None):
    m, sd_in, sd_half = mx.live_reference(ref)
    f = mx.fit_gmm(sa, 3, mp, seed, init)
    if init is not None:                       # also try a fresh start; keep the better fit
        f2 = mx.fit_gmm(sa, 3, dict(mp, n_init=1), seed)
        if f2.bic < f.bic:
            f = f2
    ids = mx.identify(f, "k3_mapped", m)
    out = {"mode": m, "sd_inner": sd_in, "sd_half": sd_half, "fit": f, "D_idx": ids["D"]}
    if ids["D"] is None:
        return out
    D = ids["D"]
    out.update({"mu_d": f.mu[D], "sd_d": f.sd[D], "w_d": f.w[D],
                "ref_w_d": float(mx.refit_weights(ref, f)[D]),
                "ashman": mx.ashman_d(m, sd_in, f.mu[D], f.sd[D])})
    out["cuts"] = mx.rule_cutoffs(m, sd_in, f.mu[D], f.sd[D], k_sd)
    return out


def run_case(args):
    exp, line, B, P = args
    mp, k_sd, U = P["mixture"], P["cutoff_rule"]["k_sd"], P["usability"]
    sa_w, ref_w, fit, pre = case_cells(exp, line)
    sa = np.concatenate(list(sa_w.values()))
    ref = np.concatenate(list(ref_w.values()))
    base = fit_case(sa, ref, mp, k_sd, P["seed"])
    bic = mx.mixture_bic_table(sa, mp, P["seed"], ks=(1, 2))
    row = {"exp": exp, "line": line, "fit_tps": ",".join(map(str, fit)),
           "reference": f"same wells T{pre}" if exp == "XDP1" else "untreated, same timepoints",
           "n_sa_cells": len(sa), "n_sa_wells": len(sa_w), "n_ref_cells": len(ref), "n_ref_wells": len(ref_w),
           "mode_log": base["mode"], "mode_ratio": 10 ** base["mode"], "sd_inner": base["sd_inner"],
           "sd_half": base["sd_half"], "dbic_12": bic[1] - bic[2], "dead_found": base["D_idx"] is not None}
    if base["D_idx"] is not None:
        row.update({"dead_mu_log": base["mu_d"], "dead_mu_ratio": 10 ** base["mu_d"], "dead_sd": base["sd_d"],
                    "dead_w": base["w_d"], "ref_dead_w": base["ref_w_d"], "dead_excess": base["w_d"] - base["ref_w_d"],
                    "ashman": base["ashman"], "min_comp_cells": float(np.min(base["fit"].w) * len(sa))})
        for r, (lo, hi) in base["cuts"].items():
            row[f"{r}_live"], row[f"{r}_dead"] = 10 ** lo, 10 ** hi
    tests = {
        "U_cells": len(sa) >= U["min_cells"], "U_wells": len(sa_w) >= U["min_wells"],
        "U_dead_found": base["D_idx"] is not None,
        "U_comp_cells": bool(row.get("min_comp_cells", 0) >= U["min_cells_per_component"]),
        "U_bic": row["dbic_12"] > U["min_dbic_12"],
        "U_separation": bool(row.get("ashman", 0) >= U["min_ashman_d"]),
        "U_dead_excess": bool(row.get("dead_excess", 0) >= U["min_sa_dead_excess"]),
    }
    row.update(tests)
    row["usable"] = all(tests.values())
    row["few_wells"] = len(sa_w) < 3
    # Bootstrap.
    boots = []
    if base["D_idx"] is not None and B > 0:
        rng = np.random.default_rng(P["seed"] + zlib.crc32(f"{exp}|{line}".encode()) % 10_000)  # stable across runs
        sa_keys, ref_keys = list(sa_w), list(ref_w)
        for b in range(B):
            if exp == "XDP1":                         # same wells provide S.A. (T13-14) and reference (T10) cells
                pick = rng.choice(sa_keys, len(sa_keys))
                sab = np.concatenate([sa_w[w] for w in pick])
                refb = np.concatenate([ref_w[w] for w in pick if w in ref_w])
            else:
                if row["few_wells"]:
                    sab = np.concatenate([rng.choice(v, len(v)) for v in sa_w.values()])
                else:
                    sab = np.concatenate([sa_w[w] for w in rng.choice(sa_keys, len(sa_keys))])
                refb = np.concatenate([ref_w[w] for w in rng.choice(ref_keys, len(ref_keys))])
            r = fit_case(sab, refb, dict(mp, n_init=1), k_sd, P["seed"] + b + 1, base["fit"])
            rb = {"exp": exp, "line": line, "b": b, "ok": r["D_idx"] is not None}
            if rb["ok"]:
                rb.update({"mode_log": r["mode"], "sd_inner": r["sd_inner"], "dead_mu_log": r["mu_d"],
                           "dead_sd": r["sd_d"], "dead_w": r["w_d"]})
                for k, (lo, hi) in r["cuts"].items():
                    rb[f"{k}_live"], rb[f"{k}_dead"] = 10 ** lo, 10 ** hi
            boots.append(rb)
        bt = pd.DataFrame(boots)
        ok = bt[bt.ok]
        row["boot_B"], row["boot_ok_frac"] = B, len(ok) / B
        # Share of replicates whose dead component differs markedly from the base fit (a second mixture solution).
        row["boot_alt_solution_frac"] = float((np.abs(ok.dead_w - base["w_d"]) > 0.08).mean()) if len(ok) else np.nan
        for c in [f"{r}_{s}" for r in RULES for s in ("live", "dead")] + ["dead_w"]:
            if c in ok and ok[c].notna().any():
                row[f"{c}_lo"], row[f"{c}_hi"] = np.nanpercentile(ok[c], [2.5, 97.5])
    return row, boots


def plot_case_panels(res: pd.DataFrame, rec: StepRecorder):
    """One figure per experiment: reference and S.A. distributions with the B, C, D cutoffs."""
    for exp in [e for e in figs.EXP_ORDER if e in set(res.exp)]:
        rr = res[res.exp == exp]
        lines = [l for l in figs.LINE_ORDER if l in set(rr.line)]
        fig, axes = plt.subplots(1, len(lines), figsize=(3.4 * len(lines), 3.0), sharey=True, squeeze=False)
        bins = np.linspace(-2.6, 0.3, 110)
        for ax, line in zip(axes[0], lines):
            r = rr[rr.line == line].iloc[0]
            sa_w, ref_w, fit, pre = case_cells(exp, line)
            ax.hist(np.concatenate(list(ref_w.values())), bins=bins, density=True, histtype="step", color=figs.UT,
                    lw=1.3, label="reference (untreated)" if exp != "XDP1" else f"same wells T{pre}")
            ax.hist(np.concatenate(list(sa_w.values())), bins=bins, density=True, color=figs.SA, alpha=0.18, lw=0,
                    label="post-dose S.A.")
            if r.dead_found:
                ax.axvspan(np.log10(r.B_live), np.log10(r.B_dead), color=figs.AMBIG, alpha=0.6, lw=0,
                           label="rule B: dropped band")
                ax.axvline(np.log10(r.D_live), color=figs.LIVE, lw=2, label=f"rule D {r.D_live:.3f}")
                ax.axvline(np.log10(r.C_live), color=figs.DEAD, lw=1.4, ls="--", label=f"rule C {r.C_live:.3f}")
            status = "usable" if r.usable else "→ transfer"
            ax.set_title(f"{figs.LINE_SHORT[line]} ({status})", fontsize=9)
            figs.log_ratio_axis(ax)
            ax.set_xlim(-2.4, 0.2)
            ax.set_xlabel("GEDI ratio (log scale)")
            ax.legend(fontsize=6, loc="upper left")
        axes[0][0].set_ylabel("density")
        fig.suptitle(f"{exp}: direct S.A.-informed cutoffs at T{','.join(map(str, fit))}", x=0.01, ha="left", fontsize=10)
        path = figs.save(fig, rec.out_dir / f"F6_cutoffs_{exp}.png")
        rec.figure(f"F6_{exp}", path, f"{exp}: reference cells (grey outline) and post-dose S.A. cells (violet) with the "
                   "rule D cutoff (blue), rule C cutoff (dashed orange) and rule B's dropped band (grey).", exp=exp)


def main() -> None:
    P = cfg.params()
    B = int(sys.argv[1]) if len(sys.argv) > 1 else P["bootstrap"]["B_fit"]
    rec = StepRecorder(STEP, LOCAL_OUT / STEP, {"B": B, "k_sd": P["cutoff_rule"]["k_sd"], "usability": P["usability"]})
    cases = []
    for exp in ("XDP1", "XDP3", "XDP5", "XDP10", "XDP11"):
        for line in figs.LINE_ORDER:
            if line in cfg.experiments()[exp]["lines"]:
                cases.append((exp, line, B, P))
    rows, boots = [], []
    with ProcessPoolExecutor(max_workers=10) as ex:
        for row, bt in ex.map(run_case, cases):
            rows.append(row)
            boots += bt
            print(row["exp"], row["line"], "usable" if row["usable"] else "transfer",
                  {k: round(row.get(f"{k}_live", np.nan), 3) for k in RULES}, flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(rec.out_dir / "thresholds_direct.csv", index=False)
    pd.DataFrame(boots).to_csv(rec.out_dir / "bootstrap_replicates.csv", index=False)
    for r in res.itertuples():
        rec.check(f"{r.exp}.{figs.LINE_SHORT[r.line]}.usable", f"{r.exp} {figs.LINE_SHORT[r.line]}: S.A. dead population "
                  "passes all usability tests (else: transfer)", None if not r.usable else True,
                  {t: bool(getattr(r, t)) for t in ("U_cells", "U_wells", "U_dead_found", "U_comp_cells", "U_bic",
                                                     "U_separation", "U_dead_excess")})
    plot_case_panels(res, rec)
    u = res[res.usable]
    rec.summary = {"cases": len(res), "usable": int(res.usable.sum()),
                   "transfer": [f"{e} {figs.LINE_SHORT[l]}" for e, l in zip(res[~res.usable].exp, res[~res.usable].line)],
                   "D_range": [float(u.D_live.min()), float(u.D_live.max())],
                   "C_range": [float(u.C_live.min()), float(u.C_live.max())],
                   "B_live_range": [float(u.B_live.min()), float(u.B_live.max())],
                   "B_dead_range": [float(u.B_dead.min()), float(u.B_dead.max())], "B": B,
                   "seed_scheme": "crc32(exp|line) + params.seed (reproducible)"}
    rec.write()
    print(rec.summary)


if __name__ == "__main__":
    main()

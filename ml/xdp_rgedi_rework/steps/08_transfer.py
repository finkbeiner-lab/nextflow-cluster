"""Step 08: transfer cutoffs to cases without a usable S.A. dead population (rule R7 / decision D03).

Source cases = usable cases from step 06. For each, every cutoff is written as an offset from that case's
own reference live population in live-SD units:  kappa = (log10 cutoff - mode) / sd_inner.
For a line, the offset is the median over its usable source experiments. A target case gets
   cutoff = 10 ** (mode* + median kappa * sd_inner*)
where (mode*, sd_inner*) come from the target's own reference cells:
   XDP8 untreated wells at T5 (its last image), XDP3 untreated at T10, XDP5 13I untreated at T9,
   XDP10 09B untreated at T9-T10, XDP1 09B / 63D the same wells at T10 (before dosing).
Uncertainty: source experiments and target wells resampled together (hierarchical bootstrap). This covers
resampling only - not the choice of offset form (D21) or the leave-one-out error - so the intervals are labelled so.
Validation: leave one experiment out - predict each usable case from the other experiments of its line
and compare with its direct cutoff (tolerance 0.10 log10 units, params.yaml).
Sensitivities: offset pooled across lines; absolute offset in log units instead of live-SD units.
"""

import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from lib import cfg, data, figs  # noqa: E402
from lib import mixture as mx  # noqa: E402
from lib.paths import LOCAL_OUT  # noqa: E402
from lib.provenance import StepRecorder  # noqa: E402

STEP = "08_transfer"
plt = figs.plt
CUTS = ["B_live", "B_dead", "C_live", "D_live",       # C and D are single cutoffs (live == dead)
        "W90_live", "W90_dead", "W95_live", "W95_dead"]   # D28 window edges (run 2), transferred like any cutoff


def target_reference(exp: str, line: str):
    e = cfg.experiments()[exp]["sa"]
    d = data.load(exp, columns=["well", "line", "condition", "tp", "log10_ratio", "in_scope"])
    d = d[(d.line == line) & d.log10_ratio.notna()]
    if exp == "XDP8":
        sel, desc = d[(d.condition == "untreated") & (d.tp == e["transfer_target_tp"])], f"untreated T{e['transfer_target_tp']}"
    elif exp == "XDP1":
        sel, desc = d[d.tp == e["pre_tp"]], f"same wells T{e['pre_tp']}"
    else:
        sel, desc = d[(d.condition == "untreated") & d.tp.isin(e["fit_tps"])], "untreated T" + ",".join(map(str, e["fit_tps"]))
    return {w: g.log10_ratio.to_numpy() for w, g in sel.groupby("well")}, desc


def transfer_one(args):
    exp, line, kappas, B, seed = args
    wells, desc = target_reference(exp, line)
    x = np.concatenate(list(wells.values()))
    m, sd_in, _ = mx.live_reference(x)
    out = {"exp": exp, "line": line, "reference": desc, "n_ref_cells": len(x), "n_ref_wells": len(wells),
           "mode_log": m, "mode_ratio": 10 ** m, "sd_inner": sd_in,
           "n_sources": int(kappas[CUTS[0]].notna().sum()), "sources": ",".join(kappas.exp)}
    for c in CUTS:
        out[c] = 10 ** (m + kappas[c].median() * sd_in)
        out[f"{c}_absolute"] = 10 ** (m + kappas[f"abs_{c}"].median())     # D21 alternative
    rng = np.random.default_rng(seed)
    keys = list(wells)
    bt = {c: [] for c in CUTS}
    for _ in range(B):
        xb = np.concatenate([wells[w] for w in rng.choice(keys, len(keys))])
        mb, sb, _ = mx.live_reference(xb)
        kb = kappas.iloc[rng.integers(0, len(kappas), len(kappas))]
        for c in CUTS:
            bt[c].append(10 ** (mb + kb[c].median() * sb))
    for c in CUTS:
        out[f"{c}_lo"], out[f"{c}_hi"] = np.percentile(bt[c], [2.5, 97.5])
    return out


def main() -> None:
    P = cfg.params()
    B = int(sys.argv[1]) if len(sys.argv) > 1 else P["bootstrap"]["B_transfer"]
    tol = P["transfer"]["loeo_tolerance_log10"]
    rec = StepRecorder(STEP, LOCAL_OUT / STEP, {"B": B, "loeo_tolerance_log10": tol})
    direct = pd.read_csv(LOCAL_OUT / "06_fit_thresholds" / "thresholds_direct.csv")
    src = direct[direct.usable].copy()
    for c in CUTS:
        src[f"k_{c}"] = (np.log10(src[c]) - src.mode_log) / src.sd_inner
        src[f"abs_{c}"] = np.log10(src[c]) - src.mode_log
    src.to_csv(rec.out_dir / "source_offsets.csv", index=False)

    # Leave one experiment out.
    loeo = []
    for r in src.itertuples():
        others = src[(src.line == r.line) & (src.exp != r.exp)]
        if not len(others):
            continue
        row = {"exp": r.exp, "line": r.line, "n_other_sources": len(others)}
        for c in CUTS:
            pred = r.mode_log + others[f"k_{c}"].median() * r.sd_inner
            row[f"{c}_direct"], row[f"{c}_pred"] = getattr(r, c), 10 ** pred
            row[f"{c}_err_log10"] = pred - np.log10(getattr(r, c))
            pred_pool = r.mode_log + src[src.exp != r.exp][f"k_{c}"].median() * r.sd_inner
            row[f"{c}_err_pooled"] = pred_pool - np.log10(getattr(r, c))
            pred_abs = r.mode_log + others[f"abs_{c}"].median()
            row[f"{c}_err_absolute"] = pred_abs - np.log10(getattr(r, c))
        loeo.append(row)
    loeo = pd.DataFrame(loeo)
    loeo.to_csv(rec.out_dir / "loeo_validation.csv", index=False)

    # Targets.
    targets = [(r.exp, r.line) for r in direct[~direct.usable].itertuples()]
    targets += [("XDP8", l) for l in figs.LINE_ORDER if l in cfg.experiments()["XDP8"]["lines"]]
    jobs = []
    for i, (exp, line) in enumerate(targets):
        k = src[src.line == line][["exp"] + [f"k_{c}" for c in CUTS] + [f"abs_{c}" for c in CUTS]].rename(
            columns={f"k_{c}": c for c in CUTS})
        jobs.append((exp, line, k.reset_index(drop=True), B, P["seed"] + i))
    with ProcessPoolExecutor(max_workers=10) as ex:
        tr = pd.DataFrame(list(ex.map(transfer_one, jobs)))
    tr.to_csv(rec.out_dir / "thresholds_transfer.csv", index=False)

    rec.summary = {
        "targets": len(tr), "sources_by_line": src.groupby("line").exp.apply(list).to_dict(),
        "offset_ranges": {c: [float(src[f"k_{c}"].min()), float(src[f"k_{c}"].max())] for c in CUTS},
        "loeo_abs_err_median": {c: float(loeo[f"{c}_err_log10"].abs().median()) for c in CUTS},
        "loeo_abs_err_max": {c: float(loeo[f"{c}_err_log10"].abs().max()) for c in CUTS},
        "loeo_within_tol": {c: int((loeo[f"{c}_err_log10"].abs() <= tol).sum()) for c in CUTS},
        "loeo_n": len(loeo),
        "loeo_abs_err_median_pooled": {c: float(loeo[f"{c}_err_pooled"].abs().median()) for c in CUTS},
        "loeo_abs_err_median_absolute": {c: float(loeo[f"{c}_err_absolute"].abs().median()) for c in CUTS},
    }
    for c in CUTS:
        n_ok = rec.summary["loeo_within_tol"][c]
        rec.check(f"transfer.loeo.{c}", f"Leave-one-experiment-out: {c} predicted within {tol} log10 units",
                  n_ok == len(loeo), {"within": n_ok, "of": len(loeo),
                                      "median_abs_err": rec.summary["loeo_abs_err_median"][c]})
    plot(src, loeo, tol, rec)
    rec.write()
    print(rec.summary)
    print(tr[["exp", "line", "reference", "mode_ratio", "sd_inner"] + CUTS].round(3).to_string())


def plot(src, loeo, tol, rec):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    ax = axes[0]
    for j, c in enumerate(["D_live", "C_live", "B_live", "B_dead"]):
        for line in figs.LINE_ORDER:
            s = src[src.line == line]
            ax.scatter(np.full(len(s), j) + 0.12 * (figs.LINE_ORDER.index(line) - 1.5), s[f"k_{c}"].to_numpy(),
                       color=figs.LINE_COLOR[line], s=26, label=figs.LINE_SHORT[line] if j == 0 else None)
    ax.set_xticks(range(4))
    ax.set_xticklabels(["rule D", "rule C", "rule B live", "rule B dead"])
    ax.set_ylabel("cutoff offset (live SDs above the live peak)")
    ax.legend(fontsize=7)
    ax.set_title("Offsets in the experiments with a usable S.A. dead population", fontsize=9)
    ax = axes[1]
    lim = [0.08, 0.45]
    ax.plot(lim, lim, color=figs.AXIS, lw=1)
    for c, mk in (("D_live", "o"), ("C_live", "s"), ("B_dead", "^")):
        ax.scatter(loeo[f"{c}_direct"].to_numpy(), loeo[f"{c}_pred"].to_numpy(), marker=mk, s=28,
                   label=c.replace("_live", "").replace("_", " "), color={"D_live": figs.LIVE, "C_live": figs.DEAD,
                                                                          "B_dead": figs.INK2}[c])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("cutoff fitted directly (ratio)")
    ax.set_ylabel("cutoff predicted from other experiments")
    ax.legend(fontsize=7)
    ax.set_title(f"Leave one experiment out (tolerance ±{tol} log10)", fontsize=9)
    path = figs.save(fig, rec.out_dir / "F10_transfer.png")
    rec.figure("F10", path, "Left: each cutoff expressed in live-SD units above its line's live peak, one dot per "
               "experiment with a usable dead population. Right: each usable case's cutoff predicted from the other "
               "experiments of the same line vs the cutoff fitted directly.")


if __name__ == "__main__":
    main()

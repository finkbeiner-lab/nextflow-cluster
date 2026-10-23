"""Step 09: sensitivity of the direct cutoffs (rule R8). Point estimates, usable cases only.

Variants (each compared with the primary cutoffs of step 06):
  k2.0 / k3.0     SD multiplier 2.0 and 3.0 instead of 2.5 (rules B and C; D does not use it)
  fit_tp          single fit timepoints: XDP10 T9 or T10, XDP1 T13 or T14; XDP5 T8+T9
  no_edge         XDP11 without the column-22 S.A. wells
  pooled_dead     the S.A. dead population pooled over all lines of the experiment (live side per line)
Reported: cutoff change in log10 units and the change in the share of reference cells called dead at the
last reference timepoint.
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

STEP = "09_sensitivity"
plt = figs.plt


def cells(exp, line, tps, drop_col=None, all_lines=False):
    s = cfg.experiments()[exp]["sa"]
    d = data.load(exp, columns=["line", "col", "condition", "tp", "log10_ratio", "in_scope"])
    d = d[d.log10_ratio.notna()]
    sa = d[(d.condition == "sa") & d.tp.isin(tps) & ((d.line == line) | all_lines)]
    if drop_col is not None:
        sa = sa[sa["col"] != drop_col]
    dl = d[d.line == line]
    ref = dl[dl.tp == s["pre_tp"]] if exp == "XDP1" else dl[(dl.condition == "untreated") & dl.tp.isin(tps)]
    last = dl[dl.tp <= 10] if exp == "XDP1" else dl[dl.condition == "untreated"]
    last = last[last.tp == last.tp.max()]
    return sa.log10_ratio.to_numpy(), ref.log10_ratio.to_numpy(), last.log10_ratio.to_numpy()


def cut_from(sa, ref, mp, k, seed):
    m, sd, _ = mx.live_reference(ref)
    f = mx.fit_gmm(sa, 3, mp, seed)
    ids = mx.identify(f, "k3_mapped", m)
    if ids["D"] is None:
        return None
    D = ids["D"]
    return mx.rule_cutoffs(m, sd, f.mu[D], f.sd[D], k)


def death(x, lo, hi):
    live, dead = np.mean(x < lo), np.mean(x > hi)
    return dead / (live + dead)


def run(args):
    exp, line, P, prim = args
    mp = dict(P["mixture"], n_init=8)
    k0 = P["cutoff_rule"]["k_sd"]
    fit = cfg.experiments()[exp]["sa"]["fit_tps"]
    variants = [("k2.0", dict(tps=fit), 2.0), ("k3.0", dict(tps=fit), 3.0)]
    if exp == "XDP10":
        variants += [("fit T9", dict(tps=[9]), k0), ("fit T10", dict(tps=[10]), k0)]
    if exp == "XDP1":
        variants += [("fit T13", dict(tps=[13]), k0), ("fit T14", dict(tps=[14]), k0)]
    if exp == "XDP5":
        variants += [("fit T8+T9", dict(tps=[8, 9]), k0)]
    if exp == "XDP11":
        variants += [("no column 22", dict(tps=fit, drop_col=22), k0)]
    if exp != "XDP1":
        variants += [("dead pooled over lines", dict(tps=fit, all_lines=True), k0)]
    rows = []
    _, _, last = cells(exp, line, fit)
    for name, kw, k in variants:
        sa, ref, _ = cells(exp, line, **kw)
        c = cut_from(sa, ref, mp, k, P["seed"])
        if c is None:
            rows.append({"exp": exp, "line": line, "variant": name, "dead_found": False})
            continue
        row = {"exp": exp, "line": line, "variant": name, "dead_found": True}
        for rule, (lo, hi) in c.items():
            if rule == "D" and name.startswith("k"):
                continue
            p_lo, p_hi = np.log10(prim[f"{rule}_live"]), np.log10(prim[f"{rule}_dead"])
            row[f"{rule}_dlog_live"], row[f"{rule}_dlog_dead"] = lo - p_lo, hi - p_hi
            row[f"{rule}_ddeath"] = death(last, lo, hi) - death(last, p_lo, p_hi)
        rows.append(row)
    return rows


def main() -> None:
    P = cfg.params()
    rec = StepRecorder(STEP, LOCAL_OUT / STEP)
    direct = pd.read_csv(LOCAL_OUT / "06_fit_thresholds" / "thresholds_direct.csv")
    u = direct[direct.usable]
    jobs = [(r.exp, r.line, P, r._asdict()) for r in u.itertuples()]
    out = []
    with ProcessPoolExecutor(max_workers=10) as ex:
        for rows in ex.map(run, jobs):
            out += rows
    df = pd.DataFrame(out)
    df.to_csv(rec.out_dir / "sensitivity.csv", index=False)
    summ = {}
    for v, g in df.groupby("variant"):
        summ[v] = {c: float(g[c].abs().max()) for c in g.columns if c.endswith(("dlog_live", "dlog_dead", "ddeath"))
                   and g[c].notna().any()}
        summ[v]["n"] = len(g)
        summ[v]["dead_lost"] = int((~g.dead_found.astype(bool)).sum())
    rec.summary = summ
    rec.write()
    print(df.round(3).to_string())


if __name__ == "__main__":
    main()

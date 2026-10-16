"""Step 14: live, dead and detected cell counts per well x timepoint for every threshold set (decision D26).

Per experiment (in-scope lines; untreated and S.A. wells; siRNA wells never used):
  N  all detected cells (threshold-free)
  L  cells called live, D cells called dead, A cells dropped between rule B's two cutoffs (and non-finite ratios)
The live/dead call is Step 11's (lib/livedead.py), so L and D in untreated wells equal the MRID inputs.
QC (D26, on N only, so identical for every set): a well x timepoint with no cells, or whose count falls below
half of both neighbours and then recovers, is missing (death cannot be undone).
k0 (start of the retention window, D26): the timepoint in the first half of the run where the mean detected
count per untreated well (all in-scope lines pooled) peaks; XDP1 (whole plate dosed after T10) uses T1-T10.
Figure 14.1: live cells per untreated well relative to k0, per line, with k0 and the dose marked.
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from lib import cfg, data, figs, livedead, sets  # noqa: E402
from lib.sets import PRIMARY  # noqa: E402
from lib.paths import LOCAL_OUT  # noqa: E402
from lib.provenance import StepRecorder  # noqa: E402

STEP = "14_count_tables"
plt = figs.plt
XDP1_LAST_REF_TP = 10           # D23: XDP1 wells serve as the untreated reference up to T10


def is_reference(df: pd.DataFrame, exp: str) -> pd.Series:
    """Rows usable as the untreated reference: untreated wells, or XDP1 (whole plate dosed) up to T10."""
    if exp == "XDP1":
        return df.tp <= XDP1_LAST_REF_TP
    return df.condition == "untreated"


def qc_missing(grid: pd.DataFrame, dip_fraction: float) -> pd.DataFrame:
    """Flag missing well x timepoints: no cells, or a dip below ``dip_fraction`` of both neighbours."""
    g = grid.sort_values(["well", "tp"]).copy()
    n = g.N.where(g.N > 0)
    prev_n = n.groupby(g.well).shift(1)
    next_n = n.groupby(g.well).shift(-1)
    g["qc_zero"] = g.N == 0
    g["qc_dip"] = (n < dip_fraction * np.fmin(prev_n, next_n)) & prev_n.notna() & next_n.notna()
    g["missing"] = g.qc_zero | g.qc_dip
    return g


def choose_k0(grid: pd.DataFrame, exp: str) -> dict:
    """D26 k0 rule on the threshold-free detected count (line-blind)."""
    ref = grid[grid.reference & ~grid.missing]
    mean_n = ref.groupby("tp").N.mean()
    tps = sorted(mean_n.index)
    first_half = tps[:math.ceil(len(tps) / 2)]
    k0 = int(mean_n.loc[first_half].idxmax())
    return {"k0": k0, "first_half": first_half, "mean_detected_per_well": {int(k): float(v) for k, v in mean_n.items()}}


def main() -> None:
    P = cfg.params()["counts"]
    rec = StepRecorder(STEP, LOCAL_OUT / STEP, params=P)
    ctx = livedead.drift_context()
    cuts = {form: livedead.cutoff_table(form) for form in ("sd", "abs")}
    long_rows, wells_rows, k0_rows = [], [], []
    summary = {"k0": {}, "qc": {}}
    for exp in figs.EXP_ORDER:
        e = cfg.experiments()[exp]
        d = data.load(exp, columns=["well", "line", "condition", "tp", "obj", "ratio", "in_scope"])
        d = d[d.condition.isin(["untreated", "sa"])]
        roles = cfg.well_roles(exp)
        roles = roles[roles.in_scope & roles.condition.isin(["untreated", "sa"])]
        tps = cfg.read_timepoints(exp)
        tps = tps[tps.tp.isin(d.tp.unique())]
        grid = pd.MultiIndex.from_product([roles.well, tps.tp], names=["well", "tp"]).to_frame(index=False)
        grid = grid.merge(roles[["well", "row", "col", "line", "condition"]], on="well").merge(tps, on="tp")
        grid = grid.merge(d.groupby(["well", "tp"]).size().rename("N").reset_index(), on=["well", "tp"], how="left")
        grid["N"] = grid.N.fillna(0).astype(int)
        grid["reference"] = is_reference(grid, exp)
        grid = qc_missing(grid, P["dip_fraction"])
        k = choose_k0(grid, exp)
        summary["k0"][exp] = k["k0"]
        k0_rows.append({"exp": exp, "k0": k["k0"], "first_half": " ".join(f"T{t}" for t in k["first_half"]),
                        **{f"mean_N_T{t}": v for t, v in k["mean_detected_per_well"].items()}})
        summary["qc"][exp] = {"zero": int(grid.qc_zero.sum()), "dip": int((grid.qc_dip & ~grid.qc_zero).sum()),
                              "examples": grid[grid.missing][["well", "tp"]].astype(str).values.tolist()[:12]}
        t0 = e["sa"]["pre_tp"]

        for name, (_, _, form) in livedead.SETS.items():
            c = livedead.classify(d, exp, name, cuts[form], ctx)
            agg = c.assign(L=c.live_guesses == 1, D=c.live_guesses == 0, A=c.live_guesses.isna()).groupby(
                ["well", "tp"])[["L", "D", "A"]].sum().astype(int).reset_index()
            g = grid.merge(agg, on=["well", "tp"], how="left")
            g[["L", "D", "A"]] = g[["L", "D", "A"]].fillna(0).astype(int)
            g.insert(0, "set", name)
            g.insert(0, "exp", exp)
            long_rows.append(g)
            ok = bool(((g.L + g.D + g.A) == g.N).all())
            rec.check(f"{exp}.{name}.LDA_sum", f"{exp} {name}: live + dead + dropped = detected in every well x timepoint", ok)
            # wells: live count at k0 and at the pre-dose timepoint (baseline QC, D26)
            for base, col in ((k["k0"], "L_k0"), (t0, "L_t0")):
                if base is None:
                    continue
                b = g[(g.tp == base)].set_index("well")
                g[col] = g.well.map(b.L.where(~b.missing))
            w = g.groupby(["exp", "set", "well", "line", "condition"]).agg(
                **{c_: (c_, "first") for c_ in ("L_k0", "L_t0") if c_ in g.columns}).reset_index()
            for c_ in ("L_k0", "L_t0"):
                if c_ in w.columns:
                    w[f"low_{c_[2:]}"] = ~(w[c_] >= P["min_baseline_live"])
            wells_rows.append(w)
        print(exp, "k0 =", k["k0"], summary["qc"][exp]["zero"], "zero,", summary["qc"][exp]["dip"], "dip", flush=True)

    long = pd.concat(long_rows, ignore_index=True)
    wells = pd.concat(wells_rows, ignore_index=True)
    k0_tab = pd.DataFrame(k0_rows)
    long.to_csv(rec.out_dir / "counts_long.csv", index=False)
    wells.to_csv(rec.out_dir / "wells_baseline.csv", index=False)
    k0_tab.to_csv(rec.out_dir / "k0.csv", index=False)
    for f in ("counts_long.csv", "wells_baseline.csv", "k0.csv"):
        rec.add_output(rec.out_dir / f)

    # ---- consistency with Step 11 (MRID inputs) and Step 3 (detected counts) ---------------------------------
    for name in livedead.SETS:
        s11 = pd.read_csv(LOCAL_OUT / "11_mrid_inputs" / f"counts_{name}.csv")
        mine = long[(long.set == name) & long.reference].groupby(["exp", "line", "tp"])[["L", "D"]].sum().reset_index()
        m = s11.merge(mine, left_on=["exp", "Sci_SampleID", "Timepoint"], right_on=["exp", "line", "tp"], how="outer")
        ok = bool(len(m) and (m.live == m.L).all() and (m.dead == m.D).all())
        rec.check(f"{name}.matches_step11", f"{name}: live and dead counts in reference wells equal the Step 11 MRID "
                  "inputs in every line x timepoint", ok, {"groups": int(len(m)),
                                                          "mismatches": int(((m.live != m.L) | (m.dead != m.D)).sum())})
    n_bad = 0
    n_groups = 0
    for exp in figs.EXP_ORDER:
        s3 = pd.read_csv(LOCAL_OUT / "03_qc_counts" / f"cells_per_well_{exp}.csv")
        mine = long[(long.exp == exp) & (long.set == PRIMARY)][["well", "tp", "N"]]
        m = mine.merge(s3, on=["well", "tp"], how="left")
        n_groups += len(m)
        n_bad += int((m.N != m.n.fillna(0)).sum())
    rec.check("N_matches_step3", "Detected cells per well x timepoint equal Step 3's counts", n_bad == 0,
              {"groups": n_groups, "mismatches": n_bad})

    summary["n_rows_per_set"] = int(len(long) / len(livedead.SETS))
    summary["n_wells"] = {exp: {c: int(long[(long.exp == exp) & (long.set == PRIMARY) & (long.condition == c)].well.nunique())
                                for c in ("untreated", "sa")} for exp in figs.EXP_ORDER}
    summary["low_baseline_wells"] = {
        name: {exp: int(wells[(wells.set == name) & (wells.exp == exp) & wells.low_k0 &
                              ((wells.condition == "untreated") | (wells.exp == "XDP1"))].well.nunique())
               for exp in figs.EXP_ORDER} for name in livedead.SETS}
    rec.summary = summary
    plot_live(long, k0_tab, rec)
    rec.write()
    for c in rec.checks:
        if c["status"] != "PASS":
            print(c["status"], c["id"], str(c["value"])[:200])


def plot_live(long: pd.DataFrame, k0_tab: pd.DataFrame, rec: StepRecorder) -> None:
    """Figure 14.1: live cells per reference well relative to k0 (track 1), per line, all experiments."""
    k0 = k0_tab.set_index("exp").k0
    fig, axes = plt.subplots(1, len(figs.EXP_ORDER), figsize=(17, 3.2), sharey=True)
    for ax, exp in zip(axes, figs.EXP_ORDER):
        g = long[(long.exp == exp) & (long.set == PRIMARY) & long.reference & ~long.missing].copy()
        base = g[g.tp == k0[exp]].set_index("well").L
        base = base[base >= cfg.params()["counts"]["min_baseline_live"]]        # D26: wells < 50 live cells at k0 dropped
        g = g[g.well.isin(base.index)]
        g["rel"] = g.L / g.well.map(base)
        g = g[np.isfinite(g.rel) & (g.rel > 0)]
        for line in figs.LINE_ORDER:
            s = g[g.line == line]
            if not len(s):
                continue
            m = s.groupby("hours").rel.apply(lambda x: np.exp(np.mean(np.log(x))))
            ax.plot(m.index.to_numpy(), np.log2(m.to_numpy()), color=figs.LINE_COLOR[line], lw=2,
                    label=figs.LINE_SHORT[line])
            ax.annotate(figs.LINE_SHORT[line], (m.index[-1], np.log2(m.iloc[-1])), xytext=(3, 0),
                        textcoords="offset points", fontsize=7, color=figs.LINE_COLOR[line], va="center")
        hk = g[g.tp == k0[exp]].hours.iloc[0]
        ax.axvline(hk, color=figs.INK2, ls=":", lw=1)
        ax.text(hk, 0.95, f" k0 = T{k0[exp]}", transform=ax.get_xaxis_transform(), fontsize=7, color=figs.INK2)
        if exp == "XDP1":
            ax.text(0.98, 0.04, "T1-T10 (before dosing)", transform=ax.transAxes, ha="right", fontsize=7, color=figs.MUTED)
        ax.axhline(0, color=figs.AXIS, lw=0.8)
        ax.set_title(exp)
        ax.set_xlabel("hours")
    axes[0].set_ylabel("live cells per well / at k0 (log2)")
    axes[0].legend(loc="lower left", fontsize=7)
    fig.suptitle(f"Figure 14.1  Live cells in untreated wells relative to k0 ({sets.label(PRIMARY)}; geometric mean over wells)",
                 x=0.01, ha="left", fontsize=10)
    path = figs.save(fig, rec.out_dir / "F14_live_counts.png")
    rec.figure("F14_1", path, "Live cells per reference well relative to the start of the counting window (k0), per "
               "line, " + sets.label(PRIMARY) + " cutoffs; geometric mean over wells (Table 14.2 uses summed counts), missing well x "
               "timepoints and wells with fewer than 50 live cells at k0 excluded. XDP1 uses its wells up to T10 "
               "(before dosing). XDP8's T3 is an imaging-failure day.")


if __name__ == "__main__":
    main()

"""Step 06b: confidence window around the rule D cutoff (decision D28, run 2).

For every case with a usable S.A. fit, the live and dead populations of Step 06 (untreated live peak and inner
spread; S.A.-dead mixture component) give the rule D centre, where a cell is equally likely to be live or dead.
The window extends on each side until a cell is N times more likely to belong to one population than the other:
    live  if  dead density / live density <= 1/N
    dead  if  dead density / live density >= N        cells in between are dropped (as MRID does)
N = 9 (90% sure, primary set W90) and N = 19 (95% sure, sensitivity set W95), with equal priors so the centre
stays at rule D. When the dead edge is never reached (heavily overlapping populations), it is capped at the centre
of the S.A.-dead population and the case is flagged. Uncertainty comes from the Step 06 bootstrap replicates
(each replicate's refitted populations), so no refitting is needed. Adds W90_live, W90_dead, W95_live, W95_dead
(+ _lo/_hi) to 06_fit_thresholds/thresholds_direct.csv; transfer (Step 08) then treats them like any cutoff.
Figure F6W: the window per case.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from lib import figs  # noqa: E402
from lib import mixture as mx  # noqa: E402
from lib.paths import LOCAL_OUT  # noqa: E402
from lib.provenance import StepRecorder  # noqa: E402

STEP = "06b_window_cutoffs"
plt = figs.plt
WINDOWS = {"W90": 9.0, "W95": 19.0}


def edges(m, s, mu_d, s_d, odds):
    lo, hi = mx.window_edges(m, s, mu_d, s_d, odds)
    capped = not np.isfinite(hi)
    if capped:
        hi = mu_d
    return lo, hi, capped


def main() -> None:
    rec = StepRecorder(STEP, LOCAL_OUT / STEP, {"odds": WINDOWS, "priors": "equal (centre = rule D)"})
    path = LOCAL_OUT / "06_fit_thresholds" / "thresholds_direct.csv"
    d = pd.read_csv(path)
    d = d.drop(columns=[c for c in d.columns if c.startswith(("W90_", "W95_"))])
    reps = pd.read_csv(LOCAL_OUT / "06_fit_thresholds" / "bootstrap_replicates.csv")
    rows, flags = [], []
    for r in d.itertuples():
        out = {"exp": r.exp, "line": r.line}
        if r.usable:
            c = mx.equal_density(r.mode_log, r.sd_inner, r.dead_mu_log, r.dead_sd)
            rec.check(f"{r.exp}.{r.line}.centre", f"{r.exp} {r.line}: window centre equals the rule D cutoff",
                      bool(np.isclose(10 ** c, r.D_live, rtol=1e-6)), {"centre": 10 ** c, "D_live": r.D_live})
            b = reps[(reps.exp == r.exp) & (reps.line == r.line) & reps.ok.astype(bool)]
            for name, odds in WINDOWS.items():
                lo, hi, capped = edges(r.mode_log, r.sd_inner, r.dead_mu_log, r.dead_sd, odds)
                out[f"{name}_live"], out[f"{name}_dead"] = 10 ** lo, 10 ** hi
                out[f"{name}_dead_capped"] = capped
                if capped:
                    flags.append(f"{r.exp} {figs.LINE_SHORT[r.line]} {name}")
                bl, bh = [], []
                for q in b.itertuples():
                    l_, h_, _ = edges(q.mode_log, q.sd_inner, q.dead_mu_log, q.dead_sd, odds)
                    bl.append(l_)
                    bh.append(h_)
                bl, bh = np.array(bl), np.array(bh)
                out[f"{name}_live_lo"], out[f"{name}_live_hi"] = 10 ** np.nanpercentile(bl, [2.5, 97.5])
                out[f"{name}_dead_lo"], out[f"{name}_dead_hi"] = 10 ** np.nanpercentile(bh, [2.5, 97.5])
                out[f"{name}_boot_ok"] = float(np.isfinite(bl).mean())
            rows.append(out)
        else:
            rows.append(out)
    w = pd.DataFrame(rows)
    d = d.merge(w, on=["exp", "line"], how="left", validate="one_to_one")
    d.to_csv(path, index=False)
    rec.add_output(path, note="window columns added in place (run 2)")
    u = d[d.usable]
    for name in WINDOWS:
        rec.check(f"{name}.ordered", f"{name}: live edge < centre < dead edge in every usable case",
                  bool(((u[f"{name}_live"] < u.D_live) & (u.D_live < u[f"{name}_dead"])).all()))
    rec.check("dead_edge_capped", "Cases where the dead edge never reaches the target confidence (capped at the S.A.-dead "
              "centre)", None if flags else True, flags)
    rec.summary = {name: {"live_range": [float(u[f"{name}_live"].min()), float(u[f"{name}_live"].max())],
                          "dead_range": [float(u[f"{name}_dead"].min()), float(u[f"{name}_dead"].max())],
                          "fold_width_median": float(np.median(u[f"{name}_dead"] / u[f"{name}_live"])),
                          "fold_width_range": [float((u[f"{name}_dead"] / u[f"{name}_live"]).min()),
                                               float((u[f"{name}_dead"] / u[f"{name}_live"]).max())]}
                   for name in WINDOWS}
    rec.summary["capped"] = flags
    plot(u, rec)
    rec.write()
    print(u[["exp", "line", "W90_live", "D_live", "W90_dead", "W95_live", "W95_dead"]].round(3).to_string())
    for c in rec.checks:
        if c["status"] != "PASS":
            print(c["status"], c["id"], c["value"])


def plot(u: pd.DataFrame, rec: StepRecorder) -> None:
    """F6W: rule D centre with the 90% and 95% windows, per case with its own S.A. fit."""
    u = u.iloc[::-1].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    for i, r in u.iterrows():
        col = figs.LINE_COLOR[r.line]
        ax.plot(np.log10([r.W95_live, r.W95_dead]), [i, i], color=figs.GRID, lw=9, solid_capstyle="butt", zorder=1)
        ax.plot(np.log10([r.W90_live, r.W90_dead]), [i, i], color=col, lw=9, alpha=0.45, solid_capstyle="butt", zorder=2)
        ax.plot(np.log10(r.D_live), i, "o", color=col, ms=7, zorder=3)
    ax.set_yticks(range(len(u)))
    ax.set_yticklabels(u.exp + " " + u.line.map(figs.LINE_SHORT), fontsize=8)
    figs.log_ratio_axis(ax, "x", [0.08, 0.1, 0.12, 0.15, 0.2, 0.25, 0.3])
    ax.set_xlabel("GEDI ratio (log scale)")
    ax.plot([], [], color=figs.INK2, lw=6, alpha=0.45, label="90% window (dropped zone)")
    ax.plot([], [], color=figs.GRID, lw=6, label="95% window")
    ax.plot([], [], "o", color=figs.INK2, label="rule D centre")
    ax.legend(fontsize=7.5, loc="lower right")
    fig.suptitle("Figure 6W  Buffer zone around the rule D cutoff: cells inside are neither live nor dead",
                 x=0.01, ha="left", fontsize=10)
    path = figs.save(fig, rec.out_dir / "F6W_windows.png")
    rec.figure("F6W", path, "Per line x experiment with its own S.A. fit: rule D cutoff (dot) and the zone where a cell is "
               "less than 90% (coloured) or 95% (grey) likely to belong to one population. Cells below the zone are "
               "called live, above it dead, and cells inside it are dropped.")


if __name__ == "__main__":
    main()

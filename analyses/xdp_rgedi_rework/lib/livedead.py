"""Live/dead calls per threshold set, shared by Step 11 (MRID inputs) and Step 14 (count tables).

Sets (cutoffs per line x experiment; direct where usable, transferred otherwise):
  D_sd        rule D, live-SD transfer                       (track 1, primary)
  D_abs       rule D, absolute-offset transfer               (D21 sensitivity)
  C_sd        rule C, live-SD transfer                       (rule C sensitivity)
  C_abs       rule C, absolute-offset transfer
  B           rule B (two cutoffs; cells between dropped)   (track 2)
  D_sd_drift  D_sd with the cutoff moved with the untreated live peak at each timepoint, for the cases
              whose live population drifts (MODERATE or MAJOR in step 07)   (D22 sensitivity)
Classification follows MRID's mrid_build_combined_livedead exactly: live if ratio < live cutoff, dead if
ratio > dead cutoff, otherwise dropped.
"""

from typing import List, Set, Tuple

import numpy as np
import pandas as pd

from lib.paths import LOCAL_OUT

SETS = {"W90": ("W90_live", "W90_dead", "sd"), "W95": ("W95_live", "W95_dead", "sd"),    # D28 (run 2)
        "D_sd": ("D_live", "D_live", "sd"), "D_abs": ("D_live", "D_live", "abs"), "C_sd": ("C_live", "C_live", "sd"),
        "C_abs": ("C_live", "C_live", "abs"), "B": ("B_live", "B_dead", "sd"), "D_sd_drift": ("D_live", "D_live", "sd")}

WINDOW_COLS = ["W90_live", "W90_dead", "W95_live", "W95_dead"]


def cutoff_table(form: str) -> pd.DataFrame:
    """Final cutoffs per experiment x line: direct S.A. fits where usable, transferred otherwise.

    Args:
        form: ``sd`` (live-SD transfer) or ``abs`` (absolute-offset transfer, D21).

    Returns:
        One row per experiment x line with B_live, B_dead, C_live, D_live and source.
    """
    d = pd.read_csv(LOCAL_OUT / "06_fit_thresholds" / "thresholds_direct.csv")
    t = pd.read_csv(LOCAL_OUT / "08_transfer" / "thresholds_transfer.csv")
    cols = ["exp", "line", "B_live", "B_dead", "C_live", "D_live"] + [c for c in WINDOW_COLS if c in d.columns]
    d = d[d.usable][cols].assign(source="direct")
    if form == "abs":
        t = t.assign(C_live=t.C_live_absolute, D_live=t.D_live_absolute, B_live=t.C_live_absolute)
    t = t[cols].assign(source="transfer")
    return pd.concat([d, t], ignore_index=True)


def drift_context() -> Tuple[pd.DataFrame, Set[Tuple[str, str]], pd.DataFrame]:
    """Inputs of the time-anchored set: per-timepoint live peaks, drifting cases, and the reference peaks.

    Returns:
        (drift table from step 07, set of drifting (exp, line), reference live mode per exp x line).
    """
    drift = pd.read_csv(LOCAL_OUT / "07_validity" / "drift.csv")
    tiers = pd.read_csv(LOCAL_OUT / "07_validity" / "drift_tiers.csv")
    drifting = set(zip(tiers[tiers.tier != "STABLE"].exp, tiers[tiers.tier != "STABLE"].line))
    dir_ = pd.read_csv(LOCAL_OUT / "06_fit_thresholds" / "thresholds_direct.csv")
    ref_mode = pd.concat([dir_[dir_.usable][["exp", "line", "mode_log"]],         # same rows as the final cutoffs
                          pd.read_csv(LOCAL_OUT / "08_transfer" / "thresholds_transfer.csv")[["exp", "line", "mode_log"]]])
    assert not ref_mode.duplicated(["exp", "line"]).any()
    return drift, drifting, ref_mode


def classify(cells: pd.DataFrame, exp: str, name: str, cut: pd.DataFrame, ctx) -> pd.DataFrame:
    """Attach the cutoffs of one threshold set and the live/dead call to a per-cell frame.

    Args:
        cells: per-cell rows of one experiment (needs line, tp, ratio).
        exp: experiment key.
        name: threshold-set name (a key of ``SETS``).
        cut: ``cutoff_table`` for the set's transfer form.
        ctx: ``drift_context()`` output (used by D_sd_drift only).

    Returns:
        Cells of lines with a cutoff, with columns lo, hi and live_guesses (1 live, 0 dead, NaN dropped).
    """
    lo_col, hi_col, _ = SETS[name]
    d = cells.merge(cut[cut.exp == exp][["line", lo_col, hi_col]].rename(columns={lo_col: "lo", hi_col: "hi"})
                    if lo_col != hi_col else
                    cut[cut.exp == exp][["line", lo_col]].rename(columns={lo_col: "lo"}).assign(
                        hi=lambda x: x.lo), on="line", how="inner")
    if name == "D_sd_drift":
        drift, drifting, ref_mode = ctx
        dr = drift[drift.exp == exp].merge(ref_mode[ref_mode.exp == exp], on=["exp", "line"])
        dr["shift"] = np.where([(exp, l) in drifting for l in dr.line], dr.mode_ratio / 10 ** dr.mode_log, 1.0)
        d = d.merge(dr[["line", "tp", "shift"]], on=["line", "tp"], how="left")
        d["shift"] = d["shift"].fillna(1.0)
        d["lo"], d["hi"] = d.lo * d["shift"], d.hi * d["shift"]
    live = np.where(d.ratio < d.lo, 1.0, np.where(d.ratio > d.hi, 0.0, np.nan))
    return d.assign(live_guesses=live)


def drifting_cases(ctx) -> List[str]:
    """Short labels of the drifting cases, for summaries."""
    from lib import figs
    return sorted(f"{e} {figs.LINE_SHORT[l]}" for e, l in ctx[1])

#!/usr/bin/env python
"""Stable cell tracking filter.

Python port of ``nextflow-stably-track_montage_info.R``.

Identifies stably tracked cells across timepoints in a montage experiment.
A cell (well + tracked_id) is considered "stably tracked" only if it meets
ALL of these criteria:

    1. Present at every timepoint (no missing observations)
    2. Centroid displacement between consecutive timepoints < threshold (px)
    3. Cell area fold-change between consecutive timepoints < area fold threshold
    4. Mean pixel intensity fold-change between consecutive timepoints
       < intensity fold threshold

Criteria 3 and 4 catch tracking ID swaps -- cases where the tracker
reassigns the same ID to a different cell. Even if the centroids happen
to be close, the area and intensity will usually jump abruptly.

Outputs:
    1. The montage CSV is rewritten (in place by default) with the new
       columns ``stably_tracked`` (TRUE/FALSE) and ``max_displacement``.
       NOTE: as in the R script, only rows of the selected channel are kept.
    2. A separate CSV with only the stable cell rows
       (well, tracked_id, timepoint).

Example:
    python nextflow-stably-track_montage_info.py \\
        --input_csv  /path/EXP_tracked_montage_summary.csv \\
        --stable_ids_csv /path/EXP_tracked_montage_summary_stable_ids.csv
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = [
    "well", "tracked_id", "timepoint", "centroid_x", "centroid_y",
    "area", "PixelIntensityMean",
]
CELL_KEYS = ["well", "tracked_id"]


def max_consecutive_fold_change(df: pd.DataFrame, col: str, first: pd.Series,
                                groups: pd.core.groupby.DataFrameGroupBy) -> pd.Series:
    """Largest fold-change of ``col`` between consecutive rows of each cell.

    The ratio is max(a/b, b/a) so it is always >= 1 regardless of direction
    (e.g. 200 -> 100 gives 2.0, not 0.5). NaN ratios are ignored, matching
    R's ``max(..., na.rm = TRUE)``.

    Args:
        df: Data sorted by well, tracked_id, timepoint.
        col: Column to compute fold-change on.
        first: Boolean mask of the first row of each cell.
        groups: ``df`` grouped by well, tracked_id.

    Returns:
        Series indexed by (well, tracked_id) with the max fold-change.
    """
    cur = df[col]
    prev = groups[col].shift()
    ratios = np.maximum(cur / prev, prev / cur).where(~first)
    return ratios.groupby([df["well"], df["tracked_id"]], sort=False).max()


def compute_cell_metrics(dt: pd.DataFrame) -> pd.DataFrame:
    """Compute per-cell quality metrics.

    For each cell (well + tracked_id):
        - n_tp: number of unique timepoints the cell appears in
        - max_displacement: largest centroid jump between consecutive timepoints
        - max_area_fc: largest area fold-change between consecutive timepoints
        - max_intensity_fc: largest intensity fold-change between consecutive
          timepoints

    Metrics are NaN for cells observed only once.

    Args:
        dt: Single-channel tracked montage data.

    Returns:
        DataFrame with one row per cell.
    """
    df = dt.sort_values(CELL_KEYS + ["timepoint"], kind="mergesort").reset_index(drop=True)
    groups = df.groupby(CELL_KEYS, sort=False)
    first = groups.cumcount() == 0

    n_rows = groups.size()
    n_tp = groups["timepoint"].nunique()

    # Euclidean distance between consecutive centroids. Like R's max() without
    # na.rm, any missing centroid makes the cell's max_displacement NaN.
    disp = np.sqrt(groups["centroid_x"].diff() ** 2 + groups["centroid_y"].diff() ** 2)
    disp_has_nan = (disp.isna() & ~first).groupby([df["well"], df["tracked_id"]], sort=False).any()
    max_disp = disp.groupby([df["well"], df["tracked_id"]], sort=False).max()
    max_disp[disp_has_nan] = np.nan

    max_area_fc = max_consecutive_fold_change(df, "area", first, groups)
    max_int_fc = max_consecutive_fold_change(df, "PixelIntensityMean", first, groups)

    metrics = pd.DataFrame({
        "n_tp": n_tp,
        "max_displacement": max_disp,
        "max_area_fc": max_area_fc,
        "max_intensity_fc": max_int_fc,
    })
    single = n_rows <= 1
    metrics.loc[single, ["max_displacement", "max_area_fc", "max_intensity_fc"]] = np.nan
    return metrics.reset_index()


def print_filter_summary(m: pd.DataFrame, n_timepoints: int, threshold: float,
                         area_fold_threshold: float, intensity_fold_threshold: float) -> None:
    """Print how many cells each criterion removes, to help tune thresholds."""
    complete = m["n_tp"] == n_timepoints
    print("--- Filter summary ---")
    print(f"Total cells: {len(m)}")
    print(f"  Missing timepoints:          {(~complete).sum()} removed")
    print(f"  Displacement >= {threshold:g}px:        "
          f"{(complete & (m['max_displacement'] >= threshold)).sum()} removed")
    print(f"  Area fold-change >= {area_fold_threshold:.1f}:     "
          f"{(complete & (m['max_area_fc'] >= area_fold_threshold)).sum()} removed")
    print(f"  Intensity fold-change >= {intensity_fold_threshold:.1f}: "
          f"{(complete & (m['max_intensity_fc'] >= intensity_fold_threshold)).sum()} removed")
    print(f"  Stably tracked:              {m['stably_tracked'].sum()} remain")


def run(input_csv: str, stable_ids_csv: str, output_csv: Optional[str], channel: str,
        threshold: float, area_fold_threshold: float, intensity_fold_threshold: float) -> None:
    """Run the stable cell tracking filter.

    Args:
        input_csv: Tracked montage summary CSV.
        stable_ids_csv: Output path for the stable (well, tracked_id, timepoint) rows.
        output_csv: Output path for the annotated CSV; defaults to ``input_csv``.
        channel: MeasurementTag value to keep (morphology channel).
        threshold: Max allowed centroid jump (px) between consecutive timepoints.
        area_fold_threshold: Max allowed area fold-change.
        intensity_fold_threshold: Max allowed mean intensity fold-change.
    """
    output_csv = output_csv or input_csv

    # ---- Step 1: Load data ----
    dt = pd.read_csv(input_csv)

    # Filter to morphology channel only -- avoid mixing channels, which
    # creates artificial intensity jumps between consecutive rows
    dt = dt[dt["MeasurementTag"] == channel]
    print(dt.head())
    print(REQUIRED_COLUMNS)
    missing = [c for c in REQUIRED_COLUMNS if c not in dt.columns]
    if missing:
        sys.exit(f"Missing required columns: {missing}")

    # ---- Step 2: Count total timepoints ----
    n_timepoints = dt["timepoint"].nunique()
    
    print(n_timepoints)

    # ---- Step 3: Compute per-cell quality metrics ----
    m = compute_cell_metrics(dt)
    
    print(m.head())
    
    print(m["n_tp"].unique())

    # ---- Step 4: Determine stability ----
    # NaN comparisons are False, so cells with missing metrics are unstable
    m["stably_tracked"] = (
        (m["n_tp"] == n_timepoints)
        & (m["max_displacement"] < threshold)
        & (m["max_area_fc"] < area_fold_threshold)
        & (m["max_intensity_fc"] < intensity_fold_threshold)
    )
    m.to_csv("no_filter.csv", index=False)
    # ---- Step 5: Print filter summary ----
    print_filter_summary(m, n_timepoints, threshold, area_fold_threshold, intensity_fold_threshold)

    # ---- Step 6: Merge stability flag back onto full data ----
    # Remove leftover columns from previous runs to avoid _x/_y duplicates
    stale = ["stably_tracked", "max_displacement", "stably_tracked.x", "stably_tracked.y",
             "max_displacement.x", "max_displacement.y"]
    dt = dt.drop(columns=[c for c in stale if c in dt.columns])

    dt = dt.merge(m[CELL_KEYS + ["stably_tracked", "max_displacement"]], on=CELL_KEYS, how="left")
    dt = dt.sort_values(CELL_KEYS, kind="mergesort").reset_index(drop=True)

    # ---- Step 7: Print tracking summary ----
    total_cells = len(m)
    stable_cells = int(m["stably_tracked"].sum())
    unstable_cells = total_cells - stable_cells
    pct = (lambda n: 100 * n / total_cells) if total_cells else (lambda n: float("nan"))

    print("\n========== TRACKING SUMMARY ==========")
    print(f"  Total cells:          {total_cells}")
    print(f"  Stably tracked:       {stable_cells}  ({pct(stable_cells):.1f}%)")
    print(f"  Unstable / removed:   {unstable_cells}  ({pct(unstable_cells):.1f}%)")
    print(f"  Total timepoints:     {n_timepoints}")
    print("======================================")

    # ---- Step 8: Save outputs ----
    # 8a. Annotated montage CSV (TRUE/FALSE to match R's fwrite output)
    stable_mask = dt["stably_tracked"].astype(bool)
    out = dt.copy()
    out["stably_tracked"] = np.where(stable_mask, "TRUE", "FALSE")
    out.to_csv(output_csv, index=False)
    print(f"\nUpdated {output_csv} with stability flags")

    # 8b. Stable cell rows only (for downstream analysis)
    stable_rows = dt.loc[stable_mask, ["well", "tracked_id", "timepoint"]]
    Path(stable_ids_csv).parent.mkdir(parents=True, exist_ok=True)
    stable_rows.to_csv(stable_ids_csv, index=False)
    n_stable = len(stable_rows.drop_duplicates(CELL_KEYS))
    print(f"Saved {len(stable_rows)} stable rows ({n_stable} cells) to {stable_ids_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Flag stably tracked cells in a tracked montage summary CSV.")
    parser.add_argument("--input_csv", required=True, help="Tracked montage summary CSV.")
    parser.add_argument("--stable_ids_csv", default=None,
                        help="Output CSV of stable rows. Default: <input>_stable_ids.csv")
    parser.add_argument("--output_csv", default=None,
                        help="Annotated output CSV. Default: overwrite --input_csv (as the R script does).")
    parser.add_argument("--channel", default="RFP", help="MeasurementTag to analyze (default: RFP).")
    parser.add_argument("--displacement_threshold", type=float, default=100,
                        help="Max centroid jump in px between consecutive timepoints (default: 100).")
    parser.add_argument("--area_fold_threshold", type=float, default=1.5,
                        help="Max area fold-change between consecutive timepoints (default: 1.5).")
    parser.add_argument("--intensity_fold_threshold", type=float, default=1.5,
                        help="Max mean intensity fold-change between consecutive timepoints (default: 1.5).")
    args = parser.parse_args()

    stable_ids_csv = args.stable_ids_csv
    if stable_ids_csv is None:
        p = Path(args.input_csv)
        stable_ids_csv = str(p.with_name(f"{p.stem}_stable_ids.csv"))

    run(args.input_csv, stable_ids_csv, args.output_csv, args.channel,
        args.displacement_threshold, args.area_fold_threshold, args.intensity_fold_threshold)


if __name__ == "__main__":
    main()

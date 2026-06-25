#!/usr/bin/env python
"""Stable-cell tracking filter + reporter trajectory extractor.

Two-channel analysis for EOS / photoconvert-style experiments. Ported
from ``stable_cell_filter.R`` (data.table) to Python + pandas so it
fits the existing bin/ + nextflow-cluster.sif toolchain.

Logic
-----
Stability is judged on the **morphology channel** (dense + stable, e.g.
FITC). A cell is *stably tracked* only if:

    1. Present at every timepoint
    2. Centroid displacement between consecutive timepoints < `threshold` (px)
    3. Area fold-change between consecutive timepoints < `area_fold_threshold`
    4. Mean pixel intensity fold-change < `intensity_fold_threshold`
       (catches tracker ID swaps where a cell jumps in brightness; FITC
       stays roughly steady so the check is meaningful)

The **reporter channel** (e.g. RFP) is intentionally NOT used for
stability — its intensity is supposed to decay over the experiment, so
applying an intensity-fold gate would discard every real cell. Instead,
for cells judged stable on the morphology channel, we extract the
reporter trajectory (decay curve over timepoints).

Outputs (the input CSV is never modified)
-----------------------------------------
* ``<input>_annotated.csv``             — morphology rows + stability flags
* ``<input>_stable_ids_1.csv``          — stable cell rows (well, tracked_id, timepoint)
* ``<input>_reporter_trajectories.csv`` — reporter rows for the stable cells

CLI
---
Required:

    --input_csv  <path>      tracked-cell summary CSV (typically
                             ``<analysisdir>/<experiment>_tracked_montage_summary.csv``)

Optional:

    --experiment            informational; recorded in logs
    --morphology_channel    default FITC
    --reporter_channel      default RFP
    --displacement_threshold  default 100.0 (px)
    --area_fold_threshold   default 1.5
    --intensity_fold_threshold default 1.5
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


logger = logging.getLogger("stable_cell_filter")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _setup_logging() -> None:
    """Mirror the bin/ convention of writing to ./finkbeiner_logs/.

    All console output goes to **stderr** rather than stdout so the
    downstream Nextflow process can capture the final stable-IDs CSV
    path from stdout cleanly (it's the only thing this script writes
    to stdout — see ``run()``).
    """
    log_dir = Path("./finkbeiner_logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_dir / "stable_cell_filter.log")
    handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    )
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.addHandler(logging.StreamHandler(sys.stderr))


def _resolve_input_csv(opts: argparse.Namespace) -> str | None:
    """Resolve ``--input_csv``, auto-deriving from PG if not supplied.

    When ``--input_csv`` is empty, this looks up ``analysisdir`` for the
    given ``--experiment`` via :class:`db_util.Ops` and returns the
    canonical tracked-cell summary path that ``bin/tracking_montage.py``
    writes (see ``tracking_montage.py:1211``):

        <analysisdir>/<experiment>_tracked_montage_summary.csv

    Args:
        opts: Parsed CLI namespace.

    Returns:
        The resolved CSV path, or ``None`` if no path could be derived
        (caller should log + exit 1).
    """
    supplied = (opts.input_csv or "").strip()
    if supplied:
        return supplied

    if not opts.experiment:
        logger.error(
            "--input_csv is empty and --experiment was not provided; "
            "cannot auto-derive the tracked-cell summary CSV path."
        )
        return None

    # Lazy import — db_util pulls in SQLAlchemy + psycopg, which we'd
    # rather not load when the user supplied an explicit --input_csv.
    try:
        from db_util import Ops  # noqa: PLC0415
    except ImportError as exc:
        logger.error(
            "could not import db_util to auto-derive --input_csv: %s. "
            "Pass --input_csv explicitly.",
            exc,
        )
        return None

    try:
        ops = Ops(opts)
        _imagedir, analysisdir = ops.get_imagedir_and_analysisdir()
    except AttributeError:
        # Ops doesn't expose get_imagedir_and_analysisdir; fall back to
        # the direct experimentdata lookup via Database.
        try:
            from sql import Database  # noqa: PLC0415

            db = Database()
            rows = db.get_table_value(
                tablename="experimentdata",
                column="analysisdir",
                kwargs=dict(experiment=opts.experiment),
            )
            if not rows:
                logger.error(
                    "no experimentdata row found for experiment %r; "
                    "cannot auto-derive --input_csv.",
                    opts.experiment,
                )
                return None
            analysisdir = rows[0][0]
        except Exception as exc:  # noqa: BLE001
            logger.error("PG lookup for analysisdir failed: %s", exc)
            return None
    except Exception as exc:  # noqa: BLE001
        logger.error("PG lookup for analysisdir failed: %s", exc)
        return None

    if not analysisdir:
        logger.error(
            "analysisdir for experiment %r is empty in PG.", opts.experiment
        )
        return None

    derived = str(
        Path(analysisdir) / f"{opts.experiment}_tracked_montage_summary.csv"
    )
    logger.info("auto-derived --input_csv from PG: %s", derived)
    return derived


def _max_consecutive_displacement(group: pd.DataFrame) -> float:
    """Largest centroid jump between consecutive timepoints for one cell.

    Returns NaN for cells with fewer than two timepoints.
    """
    if len(group) < 2:
        return float("nan")
    dx = group["centroid_x"].diff().to_numpy()[1:]
    dy = group["centroid_y"].diff().to_numpy()[1:]
    return float(np.sqrt(dx * dx + dy * dy).max())


def _max_fold_change(series: pd.Series) -> float:
    """Max consecutive ratio, made symmetric so direction doesn't matter.

    ``pmax(a/b, b/a)`` matches the R script.
    """
    if len(series) < 2:
        return float("nan")
    arr = series.to_numpy(dtype=float)
    if np.any(arr == 0):
        # Avoid divide-by-zero; treat zero-intensity / zero-area frames as
        # an explicit fail signal. Returning +inf forces the fold-change
        # filter to drop this cell.
        return float("inf")
    ratios = np.maximum(arr[1:] / arr[:-1], arr[:-1] / arr[1:])
    return float(np.nanmax(ratios))


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------
def run(opts: argparse.Namespace) -> int:
    """Execute the stability filter end-to-end.

    Args:
        opts: Parsed CLI namespace.

    Returns:
        Process exit code (0 on success, non-zero on validation failure).
    """
    resolved_csv = _resolve_input_csv(opts)
    if not resolved_csv:
        return 1
    in_path = Path(resolved_csv)
    if not in_path.is_file():
        logger.error("input_csv does not exist: %s", in_path)
        return 1
    opts.input_csv = resolved_csv  # downstream code reads opts.input_csv

    # Output paths derived from the input — never overwrite the source.
    base = str(in_path.with_suffix(""))
    annotated_csv = Path(base + "_annotated.csv")
    stable_ids_csv = Path(base + "_stable_ids_1.csv")
    reporter_traj_csv = Path(base + "_reporter_trajectories.csv")
    for out in (annotated_csv, stable_ids_csv, reporter_traj_csv):
        if out.resolve() == in_path.resolve():
            logger.error("refusing to overwrite the input file: %s", out)
            return 1

    logger.info("reading %s", in_path)
    try:
        dt_full = pd.read_csv(in_path)
    except pd.errors.EmptyDataError:
        logger.error("input CSV is empty: %s", in_path)
        return 1
    if dt_full.empty:
        logger.error("input CSV has 0 rows: %s", in_path)
        return 1
    logger.info("loaded %d rows", len(dt_full))

    required_cols = {
        "well",
        "tracked_id",
        "timepoint",
        "centroid_x",
        "centroid_y",
        "area",
        "PixelIntensityMean",
        "MeasurementTag",
    }
    missing = required_cols - set(dt_full.columns)
    if missing:
        logger.error(
            "missing required columns: %s. available: %s",
            sorted(missing),
            sorted(dt_full.columns),
        )
        return 1

    available_channels = sorted(dt_full["MeasurementTag"].dropna().unique())
    logger.info("available channels: %s", available_channels)
    if opts.morphology_channel not in available_channels:
        logger.error(
            "morphology_channel %r not found. available: %s",
            opts.morphology_channel,
            available_channels,
        )
        return 1
    if opts.reporter_channel not in available_channels:
        logger.error(
            "reporter_channel %r not found. available: %s",
            opts.reporter_channel,
            available_channels,
        )
        return 1

    # ---- Filter to morphology channel for stability analysis ---------------
    dt = dt_full[dt_full["MeasurementTag"] == opts.morphology_channel].copy()
    if dt.empty:
        logger.error(
            "0 rows after filtering MeasurementTag == %r",
            opts.morphology_channel,
        )
        return 1
    logger.info(
        "stability channel %r: %d rows", opts.morphology_channel, len(dt)
    )

    n_timepoints = int(dt["timepoint"].nunique())
    logger.info("total unique timepoints: %d", n_timepoints)

    # ---- Per-cell quality metrics ------------------------------------------
    dt = dt.sort_values(["well", "tracked_id", "timepoint"]).reset_index(
        drop=True
    )

    def _per_cell(group: pd.DataFrame) -> pd.Series:
        return pd.Series(
            {
                "n_tp": int(group["timepoint"].nunique()),
                "max_displacement": _max_consecutive_displacement(group),
                "max_area_fc": _max_fold_change(group["area"]),
                "max_intensity_fc": _max_fold_change(
                    group["PixelIntensityMean"]
                ),
            }
        )

    # ``include_groups=False`` silences a pandas 2.2+ deprecation: by
    # default ``apply`` re-passes the grouping columns to the callback,
    # but we use ``well`` and ``tracked_id`` only as the group key — never
    # inside ``_per_cell`` — so excluding them is both correct and
    # future-proof.
    metrics = (
        dt.groupby(["well", "tracked_id"], sort=False)
        .apply(_per_cell, include_groups=False)
        .reset_index()
    )

    # ---- Stability decision ------------------------------------------------
    metrics["stably_tracked"] = (
        (metrics["n_tp"] == n_timepoints)
        & metrics["max_displacement"].notna()
        & (metrics["max_displacement"] < opts.displacement_threshold)
        & metrics["max_area_fc"].notna()
        & (metrics["max_area_fc"] < opts.area_fold_threshold)
        & metrics["max_intensity_fc"].notna()
        & (metrics["max_intensity_fc"] < opts.intensity_fold_threshold)
    )

    # ---- Filter breakdown (parity with the R reporter) ---------------------
    n_total = len(metrics)
    n_missing_tp = int((metrics["n_tp"] != n_timepoints).sum())
    n_disp = int(
        (
            (metrics["n_tp"] == n_timepoints)
            & metrics["max_displacement"].notna()
            & (metrics["max_displacement"] >= opts.displacement_threshold)
        ).sum()
    )
    n_area = int(
        (
            (metrics["n_tp"] == n_timepoints)
            & metrics["max_area_fc"].notna()
            & (metrics["max_area_fc"] >= opts.area_fold_threshold)
        ).sum()
    )
    n_int = int(
        (
            (metrics["n_tp"] == n_timepoints)
            & metrics["max_intensity_fc"].notna()
            & (metrics["max_intensity_fc"] >= opts.intensity_fold_threshold)
        ).sum()
    )
    n_stable = int(metrics["stably_tracked"].sum())
    logger.info("--- filter breakdown ---")
    logger.info("total cells (%s): %d", opts.morphology_channel, n_total)
    logger.info("  missing timepoints: %d removed", n_missing_tp)
    logger.info(
        "  displacement >= %.0fpx: %d removed",
        opts.displacement_threshold,
        n_disp,
    )
    logger.info(
        "  area fold-change >= %.2f: %d removed",
        opts.area_fold_threshold,
        n_area,
    )
    logger.info(
        "  intensity fold-change >= %.2f: %d removed",
        opts.intensity_fold_threshold,
        n_int,
    )
    logger.info("  stably tracked: %d remain", n_stable)

    # ---- Merge stability flag back onto the morphology channel data --------
    for col in ("stably_tracked", "max_displacement"):
        if col in dt.columns:
            dt = dt.drop(columns=[col])
    dt = dt.merge(
        metrics[["well", "tracked_id", "stably_tracked", "max_displacement"]],
        on=["well", "tracked_id"],
        how="left",
    )

    # ---- Summary -----------------------------------------------------------
    unique_cells = dt[["well", "tracked_id"]].drop_duplicates()
    stable_unique = (
        dt[dt["stably_tracked"] == True][  # noqa: E712 (pandas idiom)
            ["well", "tracked_id"]
        ]
        .drop_duplicates()
    )
    total_cells = len(unique_cells)
    n_stable_cells = len(stable_unique)
    n_unstable = total_cells - n_stable_cells
    pct_stable = (
        100.0 * n_stable_cells / total_cells if total_cells > 0 else 0.0
    )
    pct_unstable = 100.0 * n_unstable / total_cells if total_cells > 0 else 0.0
    logger.info("========== TRACKING SUMMARY ==========")
    logger.info("  stability channel:    %s", opts.morphology_channel)
    logger.info("  reporter channel:     %s", opts.reporter_channel)
    logger.info("  total cells:          %d", total_cells)
    logger.info(
        "  stably tracked:       %d  (%.1f%%)", n_stable_cells, pct_stable
    )
    logger.info(
        "  unstable / removed:   %d  (%.1f%%)", n_unstable, pct_unstable
    )
    logger.info("  total timepoints:     %d", n_timepoints)
    logger.info("======================================")

    # ---- Write outputs -----------------------------------------------------
    dt.to_csv(annotated_csv, index=False)
    logger.info("wrote annotated CSV (%d rows) to %s", len(dt), annotated_csv)

    stable_rows = dt.loc[
        dt["stably_tracked"] == True,  # noqa: E712
        ["well", "tracked_id", "timepoint"],
    ]
    stable_rows.to_csv(stable_ids_csv, index=False)
    logger.info(
        "wrote stable IDs (%d rows, %d cells) to %s",
        len(stable_rows),
        len(stable_rows[["well", "tracked_id"]].drop_duplicates()),
        stable_ids_csv,
    )

    # ---- Reporter trajectories for stable cells ----------------------------
    if stable_unique.empty:
        logger.info("no stable cells; skipping %s", reporter_traj_csv)
    else:
        reporter = dt_full[dt_full["MeasurementTag"] == opts.reporter_channel]
        reporter = reporter.merge(
            stable_unique, on=["well", "tracked_id"], how="inner"
        )
        reporter = reporter.sort_values(["well", "tracked_id", "timepoint"])
        reporter.to_csv(reporter_traj_csv, index=False)
        logger.info(
            "wrote reporter (%s) trajectories: %d rows, %d cells to %s",
            opts.reporter_channel,
            len(reporter),
            len(reporter[["well", "tracked_id"]].drop_duplicates()),
            reporter_traj_csv,
        )

    # ---- Emit the stable-IDs path on stdout (last line, no decoration) ----
    # The Nextflow STABLE_CELL_FILTER process captures stdout as a value
    # channel and forwards the path to OVERLAY_MONTAGE's --cell_ids. All
    # other output went to stderr (see _setup_logging). Keep this print
    # as the LAST thing the script does so the channel emission is clean.
    print(str(stable_ids_csv.resolve()), flush=True)
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stable-cell tracking filter + reporter trajectory extractor."
        )
    )
    parser.add_argument(
        "--input_csv",
        default="",
        help=(
            "Path to the tracked-cell summary CSV. Optional — when empty, "
            "the path is auto-derived from PG as "
            "<analysisdir>/<experiment>_tracked_montage_summary.csv "
            "(same convention as bin/tracking_montage.py:1211)."
        ),
    )
    parser.add_argument(
        "--experiment",
        default="",
        help="Experiment name (informational; recorded in logs).",
    )
    parser.add_argument(
        "--morphology_channel",
        default="FITC",
        help="Channel used to decide which cells are stably tracked.",
    )
    parser.add_argument(
        "--reporter_channel",
        default="RFP",
        help="Channel whose trajectory we extract for stable cells.",
    )
    parser.add_argument(
        "--displacement_threshold",
        type=float,
        default=100.0,
        help="Max centroid displacement (px) between consecutive timepoints.",
    )
    parser.add_argument(
        "--area_fold_threshold",
        type=float,
        default=1.5,
        help="Max area fold-change between consecutive timepoints.",
    )
    parser.add_argument(
        "--intensity_fold_threshold",
        type=float,
        default=1.5,
        help="Max intensity fold-change on the morphology channel.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    _setup_logging()
    opts = _parse_args()
    if opts.experiment:
        logger.info("experiment context: %s", opts.experiment)
    sys.exit(run(opts))

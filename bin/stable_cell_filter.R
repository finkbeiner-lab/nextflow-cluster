#!/usr/bin/env Rscript
# =============================================================================
# STABLE CELL TRACKING FILTER + REPORTER TRAJECTORIES
# =============================================================================
# Two-channel analysis for EOS / photoconvert-style experiments:
#
#   * Stability is assessed on the morphology channel (e.g. FITC), which is
#     dense and stable over time.  A cell is "stably tracked" only if:
#       1. Present at every timepoint
#       2. Centroid displacement between consecutive timepoints < `threshold` px
#       3. Cell area fold-change between consecutive timepoints < `area_fold_threshold`
#       4. Mean pixel intensity fold-change < `intensity_fold_threshold`
#     The intensity check catches tracker ID-swaps where a cell suddenly
#     becomes much brighter/dimmer.  FITC works for this because the cell
#     keeps producing protein, so its FITC signal is roughly steady.
#
#   * The reporter channel (e.g. RFP) is NOT used for stability — its
#     intensity is supposed to decay over the experiment, so applying an
#     intensity-stability filter would throw out every real cell.  Instead,
#     for cells judged stable by the morphology channel, we extract the
#     reporter trajectory (the RFP decay curve over timepoints).
#
# OUTPUTS (input file is never modified):
#   1. <input>_annotated.csv             - morphology rows + stability flags
#   2. <input>_stable_ids.csv            - stable cell rows (well, tracked_id, timepoint)
#   3. <input>_reporter_trajectories.csv - reporter rows for the stable cells
# =============================================================================

library(data.table)

# ---- File paths --------------------------------------------------------------
input_csv <- "/Volumes/Finkbeiner-Kaye/SheridanMoore/GXYTMP/Nextflow/ALS-Set55-04172025TDP43EOS-JAK/ALS-Set55-04172025TDP43EOS-JAK_tracked_montage_summary.csv"

# Derive output names from the input — never overwrite the source
base <- sub("\\.csv$", "", input_csv)
annotated_csv             <- paste0(base, "_annotated.csv")
stable_ids_csv            <- paste0(base, "_stable_ids.csv")
reporter_trajectories_csv <- paste0(base, "_reporter_trajectories.csv")

# ---- Filter settings ---------------------------------------------------------
morphology_channel       <- "FITC"   # Used to decide which cells are stably tracked
reporter_channel         <- "RFP"    # Channel whose decay trajectory we want
threshold                <- 100      # Max centroid jump (px) between consecutive timepoints
area_fold_threshold      <- 1.5      # Max area fold-change (e.g. 1.5 = ±50%)
intensity_fold_threshold <- 1.5      # Max intensity fold-change (applied to morphology channel only)

# ---- Helpers -----------------------------------------------------------------
stop_with <- function(msg) {
  cat(sprintf("\nERROR: %s\n", msg), file = stderr())
  quit(status = 1)
}

# ---- Step 1: Load source CSV -------------------------------------------------
if (!file.exists(input_csv)) {
  stop_with(sprintf("Input file not found: %s", input_csv))
}
cat(sprintf("Reading %s\n", input_csv))
dt_full <- as.data.table(fread(input_csv))

if (nrow(dt_full) == 0) {
  stop_with(sprintf("Input CSV has 0 data rows (header only?): %s", input_csv))
}
cat(sprintf("Loaded %d rows.\n", nrow(dt_full)))

# ---- Step 2: Validate required columns ---------------------------------------
required_cols <- c("well", "tracked_id", "timepoint",
                   "centroid_x", "centroid_y",
                   "area", "PixelIntensityMean", "MeasurementTag")
missing <- setdiff(required_cols, names(dt_full))
if (length(missing) > 0) {
  stop_with(sprintf("Missing required columns: %s\nAvailable columns: %s",
                    paste(missing, collapse = ", "),
                    paste(names(dt_full), collapse = ", ")))
}

available_channels <- sort(unique(dt_full$MeasurementTag))
cat(sprintf("Available channels: %s\n",
            paste(available_channels, collapse = ", ")))

if (!(morphology_channel %in% available_channels)) {
  stop_with(sprintf("morphology_channel '%s' not found. Available: %s",
                    morphology_channel,
                    paste(available_channels, collapse = ", ")))
}
if (!(reporter_channel %in% available_channels)) {
  stop_with(sprintf("reporter_channel '%s' not found. Available: %s",
                    reporter_channel,
                    paste(available_channels, collapse = ", ")))
}

# ---- Step 3: Filter to the morphology channel for stability analysis ---------
# Keep dt_full untouched so we can extract reporter rows later.
dt <- dt_full[MeasurementTag == morphology_channel]
if (nrow(dt) == 0) {
  stop_with(sprintf("0 rows after filtering MeasurementTag == '%s'", morphology_channel))
}
cat(sprintf("Stability channel '%s': %d rows.\n", morphology_channel, nrow(dt)))

# ---- Step 4: Count total timepoints -----------------------------------------
n_timepoints <- uniqueN(dt$timepoint)
cat(sprintf("Total unique timepoints: %d\n", n_timepoints))

# ---- Step 5: Compute per-cell quality metrics --------------------------------
# For each (well, tracked_id):
#   - n_tp             : how many timepoints the cell appears in
#   - max_displacement : largest centroid jump between consecutive timepoints
#   - max_area_fc      : largest area fold-change between consecutive timepoints
#   - max_intensity_fc : largest intensity fold-change between consecutive timepoints
# pmax(a/b, b/a) makes the ratio always >= 1 regardless of direction.
displacement_dt <- dt[order(well, tracked_id, timepoint),
                      .(
                        n_tp = uniqueN(timepoint),

                        max_displacement = if (.N > 1)
                          max(sqrt(diff(centroid_x)^2 + diff(centroid_y)^2))
                        else NA_real_,

                        max_area_fc = if (.N > 1) {
                          a <- area
                          ratios <- pmax(a[-1] / a[-.N], a[-.N] / a[-1])
                          max(ratios, na.rm = TRUE)
                        } else NA_real_,

                        max_intensity_fc = if (.N > 1) {
                          m <- PixelIntensityMean
                          ratios <- pmax(m[-1] / m[-.N], m[-.N] / m[-1])
                          max(ratios, na.rm = TRUE)
                        } else NA_real_
                      ),
                      by = .(well, tracked_id)
]

# ---- Step 6: Determine stability ---------------------------------------------
displacement_dt[, stably_tracked :=
                  n_tp == n_timepoints &
                  !is.na(max_displacement) &
                  max_displacement < threshold &
                  !is.na(max_area_fc) &
                  max_area_fc < area_fold_threshold &
                  !is.na(max_intensity_fc) &
                  max_intensity_fc < intensity_fold_threshold]

# ---- Step 7: Print filter breakdown ------------------------------------------
cat("\n--- Filter breakdown ---\n")
cat(sprintf("Total cells (channel '%s'):    %d\n",
            morphology_channel, nrow(displacement_dt)))
cat(sprintf("  Missing timepoints:           %d removed\n",
            nrow(displacement_dt[n_tp != n_timepoints])))
cat(sprintf("  Displacement >= %dpx:         %d removed\n", threshold,
            nrow(displacement_dt[n_tp == n_timepoints &
                                 !is.na(max_displacement) &
                                 max_displacement >= threshold])))
cat(sprintf("  Area fold-change >= %.2f:    %d removed\n", area_fold_threshold,
            nrow(displacement_dt[n_tp == n_timepoints &
                                 !is.na(max_area_fc) &
                                 max_area_fc >= area_fold_threshold])))
cat(sprintf("  Intensity fold-change >= %.2f: %d removed\n", intensity_fold_threshold,
            nrow(displacement_dt[n_tp == n_timepoints &
                                 !is.na(max_intensity_fc) &
                                 max_intensity_fc >= intensity_fold_threshold])))
cat(sprintf("  Stably tracked:               %d remain\n",
            nrow(displacement_dt[stably_tracked == TRUE])))

# ---- Step 8: Merge stability flag back onto morphology data ------------------
# Drop any prior columns to avoid .x/.y duplicates from re-runs
for (col in c("stably_tracked", "max_displacement",
              "stably_tracked.x", "stably_tracked.y",
              "max_displacement.x", "max_displacement.y")) {
  if (col %in% names(dt)) dt[, (col) := NULL]
}

dt <- merge(dt,
            displacement_dt[, .(well, tracked_id, stably_tracked, max_displacement)],
            by = c("well", "tracked_id"),
            all.x = TRUE)

# ---- Step 9: Print tracking summary ------------------------------------------
total_cells    <- uniqueN(dt, by = c("well", "tracked_id"))
stable_cells   <- uniqueN(dt[stably_tracked == TRUE], by = c("well", "tracked_id"))
unstable_cells <- total_cells - stable_cells

cat("\n========== TRACKING SUMMARY ==========\n")
cat(sprintf("  Stability channel:    %s\n", morphology_channel))
cat(sprintf("  Reporter channel:     %s\n", reporter_channel))
cat(sprintf("  Total cells:          %d\n", total_cells))
cat(sprintf("  Stably tracked:       %d  (%.1f%%)\n",
            stable_cells, 100 * stable_cells / total_cells))
cat(sprintf("  Unstable / removed:   %d  (%.1f%%)\n",
            unstable_cells, 100 * unstable_cells / total_cells))
cat(sprintf("  Total timepoints:     %d\n", n_timepoints))
cat("======================================\n")

# ---- Step 10: Write annotated + stable IDs (NEVER overwrite the input) -------
if (normalizePath(annotated_csv, mustWork = FALSE) == normalizePath(input_csv, mustWork = FALSE)) {
  stop_with("Refusing to overwrite the input file. Adjust output paths.")
}

fwrite(dt, annotated_csv)
cat(sprintf("\nWrote annotated CSV (%d rows) to %s\n",
            nrow(dt), annotated_csv))

stable_rows <- dt[stably_tracked == TRUE, .(well, tracked_id, timepoint)]
fwrite(stable_rows, stable_ids_csv)
cat(sprintf("Wrote stable IDs (%d rows, %d cells) to %s\n",
            nrow(stable_rows),
            uniqueN(stable_rows, by = c("well", "tracked_id")),
            stable_ids_csv))

# ---- Step 11: Extract reporter (RFP) trajectories for stable cells -----------
# Pull the rows from the original full table where:
#   - MeasurementTag == reporter_channel (e.g. RFP)
#   - (well, tracked_id) is in the set of stably-tracked cells
# These are the decay curves the experiment is actually trying to measure.
stable_cell_keys <- unique(displacement_dt[stably_tracked == TRUE, .(well, tracked_id)])

if (nrow(stable_cell_keys) == 0) {
  cat(sprintf("\nNo stable cells found - skipping %s.\n", reporter_trajectories_csv))
} else {
  reporter_dt <- dt_full[MeasurementTag == reporter_channel]
  reporter_dt <- merge(reporter_dt, stable_cell_keys,
                       by = c("well", "tracked_id"),
                       all.x = FALSE)   # inner join - only stable cells
  reporter_dt <- reporter_dt[order(well, tracked_id, timepoint)]

  fwrite(reporter_dt, reporter_trajectories_csv)
  cat(sprintf("Wrote reporter (%s) trajectories: %d rows, %d cells to %s\n",
              reporter_channel,
              nrow(reporter_dt),
              uniqueN(reporter_dt, by = c("well", "tracked_id")),
              reporter_trajectories_csv))
}

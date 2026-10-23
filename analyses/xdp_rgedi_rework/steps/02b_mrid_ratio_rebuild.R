#!/usr/bin/env Rscript
# Step 02b: compute per-cell GEDI ratios with MRID's OWN mrid_ratio() (unchanged), exactly as
# the Sept rework did (run_phase1_ratio.R): fread -> as.data.frame, read.csv(platemap, UTF-8-BOM).
#
# Usage: Rscript --vanilla 02b_mrid_ratio_rebuild.R <cell_data.csv> <platemap.csv> <out_dir> \
#            <timepoint_removal e.g. "2" or ""> <mrid_ratio_functions.R>
suppressWarnings(suppressMessages({
  library(data.table); library(ggplot2); library(plyr); library(dplyr); library(gridExtra)
}))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 5) stop("usage: <cell_data> <platemap> <out_dir> <tp_removal> <MRID_Ratio_Functions.R>")
cell_csv <- normalizePath(args[1]); pm_csv <- normalizePath(args[2]); out_dir <- args[3]
tp_removal <- if (nzchar(args[4])) as.integer(strsplit(args[4], ",")[[1]]) else c()
source(normalizePath(args[5]))

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
setwd(out_dir)
t0 <- Sys.time()
cell_data <- as.data.frame(data.table::fread(cell_csv, showProgress = FALSE))
plate_layout <- read.csv(pm_csv, fileEncoding = "UTF-8-BOM")
plate_layout <- plate_layout[, c("Sci_WellID", "Sci_SampleID", "Drug")]
cat(sprintf("cell_data rows=%d cols=%d; platemap rows=%d unique wells=%d\n", nrow(cell_data),
            ncol(cell_data), nrow(plate_layout), length(unique(plate_layout$Sci_WellID))))
ratio_df <- mrid_ratio(cell_data, tp_removal, c(), plate_layout, out_dir)
cat(sprintf("ratio rows=%d ; timepoints=%s ; elapsed %.1f min\n", nrow(ratio_df),
            paste(sort(unique(ratio_df$Timepoint)), collapse = ","),
            as.numeric(difftime(Sys.time(), t0, units = "mins"))))
writeLines(capture.output(sessionInfo()), file.path(out_dir, "sessionInfo.txt"))

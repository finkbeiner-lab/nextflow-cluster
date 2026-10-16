#!/usr/bin/env Rscript
# Step 12: MRID log-odds for one threshold set - MRID's own functions, unchanged statistics.
# Combined across experiments (multiple_experiments = TRUE), cell-line comparison, nonlinear time
# (timepoint as a factor, TimepointIsNumeric = FALSE), every line used as a reference.
# Functions: repo bin/mrid_R/MRID_Logodd_Functions_June2026.R (with the seq_len / file.path fixes).
# Usage: Rscript --vanilla 12_logodds.R <input_set_dir> <output_dir> <repo_bin_dir>
# plyr must be loaded BEFORE dplyr (or not at all): MRID's finalize_df() calls group_by() %>% summarise() unqualified,
# and plyr::summarise masking dplyr::summarise silently disables its "drop hours seen in one experiment" filter.
suppressPackageStartupMessages({
  library(plyr); library(dplyr); library(tidyr); library(stringr); library(readr); library(rlang)
  library(ggplot2); library(xtable); library(knitr); library(lme4)
})
stopifnot(identical(environment(summarise), asNamespace("dplyr")))
args <- commandArgs(trailingOnly = TRUE)
input_path <- normalizePath(args[1]); output_path <- args[2]; BIN <- normalizePath(args[3])
dir.create(output_path, recursive = TRUE, showWarnings = FALSE)
output_path <- normalizePath(output_path)
source(file.path(BIN, "mrid_R", "MRID_Ratio_Functions.R"))
source(file.path(BIN, "mrid_R", "MRID_Logodd_Functions_June2026.R"))
set.seed(1)
setwd(output_path)

# Globals the MRID functions read.
drug <- FALSE; keep_drug <- c(); keep_line <- c(); keep_line_drug <- c()
multiple_experiments <- TRUE; multiple_plates <- FALSE
compare_cellline <- TRUE; compare_class <- FALSE; compare_drug <- FALSE
classes <- c("CTR", "XDP"); colorpalette <- c(); TimepointIsNumeric <- FALSE

sink(file.path(output_path, "mrid_log.txt"), split = FALSE)
ratio_files <- list.files(path = input_path, pattern = "ratio_output[0-9_]*?.csv", recursive = TRUE)
timepoint_files <- list.files(path = input_path, pattern = "timepoint[s_hours]*?.csv", recursive = TRUE)
stopifnot(length(ratio_files) > 0, length(timepoint_files) == length(ratio_files))
logodd_data <- do.call(rbind, lapply(ratio_files, function(f) mrid_get_logodd(input_path, f, timepoint_files,
                                                                               multiple_experiments, multiple_plates)))
logodd_data <- finalize_df(logodd_data, drug, keep_drug, keep_line, keep_line_drug, c(), compare_class,
                           multiple_experiments, classes)
write.csv(logodd_data, file.path(output_path, "Data.csv"))
ref_condition_list <- levels(factor(logodd_data$CellLine))
warn_log <- character(0)
withCallingHandlers(
  mrid_lmer_condition_comparisons(logodd_data, ref_condition_list, output_path, compare_cellline, compare_class,
                                  compare_drug, multiple_experiments, multiple_plates, classes,
                                  TimepointIsNumeric = FALSE),
  warning = function(w) { warn_log <<- c(warn_log, conditionMessage(w)); invokeRestart("muffleWarning") })
sink()
write.csv(data.frame(warning = warn_log), file.path(output_path, "warnings.csv"), row.names = FALSE)
# Which experiments contribute at each harmonised hour, per line (the time axis mixes experiments).
contrib <- unique(logodd_data[, c("Hour", "CellLine", "Experiment")])
write.csv(contrib, file.path(output_path, "experiments_per_hour.csv"), row.names = FALSE)
or_files <- list.files(path = output_path, pattern = "^OR_", full.names = TRUE)
odd <- do.call(rbind, lapply(or_files, read.csv))
write.csv(odd, file.path(output_path, "OR_all_pairs.csv"), row.names = FALSE)
file.remove(or_files)
hours <- sort(as.numeric(as.character(unique(logodd_data$Hour))))
write.csv(data.frame(Time = seq_along(hours) - 1, Hour = hours), file.path(output_path, "time_index_hours.csv"),
          row.names = FALSE)
writeLines(capture.output(sessionInfo()), file.path(output_path, "sessionInfo.txt"))
cat("done", output_path, "\n")

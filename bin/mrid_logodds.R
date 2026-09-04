#!/usr/bin/env Rscript
# MRID logodds survival stage -- headless, param-driven port of get_MRID_Logodd_*.Rmd.
#
# Sources MRID_Logodd_Functions_June2026.R (the vcov/Plate-fixed version) and runs the
# Rmd's pipeline chunks non-interactively, supplying as GLOBAL variables the inputs the
# functions read from the knit environment (the functions were written to read globals;
# we set them here rather than refactor ~1000 lines). Statistics are UNCHANGED.
#
# Usage:
#   Rscript mrid_logodds.R --input_path DIR --output_path DIR [options]
# Options (config-driven): --drug TRUE/FALSE --keep_drug a,b --keep_line a,b
#   --compare class|cellline|drug --classes CTR,XDP --ref_conditions a,b
#   --multiple_experiments TRUE --multiple_plates FALSE --time_model linear|nonlinear
#
# NOTE (staged): needs the container R env (lme4, dplyr, tidyr, stringr, ggplot2, plyr,
# gridExtra, gghalves, xtable, knitr, scales, rlang) -- validate on the cluster, not locally.
# TODO before first run: apply the two known bug fixes flagged in docs/MRID_PORT_PLAN.md
#   (for(i in 1:range(n_exp)) -> seq_len(n_exp);  '//' join in convert_old_cnnoutput).

suppressPackageStartupMessages({
  library(tidyverse); library(rlang); library(ggplot2)
  library(xtable); library(knitr); library(lme4); library(dplyr)
})

# ---- arg parsing (simple --key value) ----
args <- commandArgs(trailingOnly = TRUE)
getarg <- function(k, default = NULL) {
  i <- which(args == paste0("--", k)); if (length(i)) args[i + 1] else default
}
asbool <- function(x) isTRUE(as.logical(x))
splitc <- function(x) if (is.null(x) || x == "") c() else strsplit(x, ",")[[1]]

BIN <- dirname(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE)))
source(file.path(BIN, "mrid_R", "MRID_Ratio_Functions.R"))
source(file.path(BIN, "mrid_R", "MRID_Logodd_Functions_June2026.R"))
set.seed(1)  # deterministic sample(colors()) etc.

# ---- inputs (as GLOBALS the functions expect) ----
input_path  <- getarg("input_path")
output_path <- getarg("output_path", input_path)
drug        <- asbool(getarg("drug", "FALSE"))
keep_drug   <- splitc(getarg("keep_drug", ""))
keep_line   <- splitc(getarg("keep_line", ""))
keep_line_drug <- splitc(getarg("keep_line_drug", ""))
multiple_experiments <- asbool(getarg("multiple_experiments", "TRUE"))
multiple_plates      <- asbool(getarg("multiple_plates", "FALSE"))
cmp <- getarg("compare", "class")
compare_cellline <- cmp == "cellline"
compare_class    <- cmp == "class"
compare_drug     <- cmp == "drug"
classes            <- splitc(getarg("classes", "CTR,XDP"))
ref_condition_list <- splitc(getarg("ref_conditions", ""))
colorpalette       <- c()
TimepointIsNumeric <- getarg("time_model", "linear") == "nonlinear"  # numeric-time=linear model

create_output_dir(output_path)
convert_old_cnnoutput(input_path)
ratio_files     <- list.files(path = input_path, pattern = "ratio_output[0-9_]*?.csv", recursive = TRUE)
timepoint_files <- list.files(path = input_path, pattern = "timepoint[_hours]*?.csv", recursive = TRUE)
stopifnot(length(ratio_files) > 0, length(timepoint_files) > 0)

logodd_data <- do.call(rbind, lapply(ratio_files, mrid_get_logodd,
  input_path = input_path, timepoint_hours = timepoint_files,
  multi_exp = multiple_experiments, multi_plates = multiple_plates))
logodd_data <- finalize_df(logodd_data, drug, keep_drug, keep_line, keep_line_drug,
  ref_condition_list, compare_class, multiple_experiments, classes)
write.csv(logodd_data, file.path(output_path, "Data.csv"))

# pipeline chunks (headless; plots + OR_*.csv written to output_path)
mrid_death_plot(logodd_data, drug, multiple_experiments, multiple_plates, colorpalette, compare_class)
mrid_residuals_plot(logodd_data, drug, multiple_experiments, multiple_plates, colorpalette,
  compare_class, compare_cellline, compare_drug, TimepointIsNumeric)
mrid_lmer_condition_comparisons(logodd_data, ref_condition_list, output_path,
  compare_cellline, compare_class, compare_drug, multiple_experiments, multiple_plates, classes)

# combine odds-ratio files
or_files <- list.files(path = output_path, pattern = "^OR_", full.names = TRUE)
if (length(or_files)) {
  odd_ratio_data <- do.call(rbind, lapply(or_files, read.csv))
  file.remove(or_files)
  odd_ratio_data$Reference <- factor(odd_ratio_data$Reference)
  for (ref in levels(odd_ratio_data$Reference))
    mrid_logodd_combine_plot(odd_ratio_data, ref, output_path, colorpalette,
      confident_int = 0.95, compare_class)
}
cat("MRID logodds done ->", output_path, "\n")

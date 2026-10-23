#!/usr/bin/env Rscript
# Step 12b: odds-ratio models built on MRID's well x timepoint counts (Data.csv from step 12).
#  (1) Independent re-derivation of MRID's per-timepoint odds ratios: the same pairwise glmer
#      (1|Experiment) + (1|Experiment:Sci_WellID) + CellLine * Timepoint(factor), with the OR at each
#      timepoint taken from named coefficients and the variance-covariance matrix (not by position).
#  (2) Time-averaged odds ratio per experiment (for the forest plot): per experiment and line pair,
#      glmer cbind(dead, live) ~ CellLine + Timepoint(factor) + (1|Sci_WellID).
#  (3) The same time-averaged odds ratio pooled over experiments, two ways: MRID's random-effect structure
#      (1|Experiment) + (1|Experiment:Sci_WellID), and with an added (1|Experiment:CellLine) term so that the
#      line difference may vary between experiments (its interval then includes that variation).
#  (1) runs for every ordered line pair; (2) and (3) for the two control lines as reference.
# References: both control lines. Usage: Rscript --vanilla 12b_or_models.R <step12_output_dir>
suppressPackageStartupMessages({ library(lme4); library(dplyr) })
out <- normalizePath(commandArgs(trailingOnly = TRUE)[1])
d <- read.csv(file.path(out, "Data.csv"))
d$dead <- d$classifier.score.dead; d$live <- d$live_guesses
d$Timepoint <- factor(d$Timepoint)
refs <- c("CTR-33114.C", "CTR-33113.2I")
lines <- sort(unique(d$CellLine))
ctrl <- glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))

wald <- function(est, se) c(OR = exp(est), lower = exp(est - 1.96 * se), upper = exp(est + 1.96 * se),
                            p = 2 * pnorm(-abs(est / se)))

rows1 <- list(); rows2 <- list(); rows3 <- list(); warn_log <- list()
catch <- function(expr, what) withCallingHandlers(expr, warning = function(w) {
  warn_log[[length(warn_log) + 1]] <<- data.frame(model = what, warning = conditionMessage(w))
  invokeRestart("muffleWarning") })
for (ref in lines) for (cmp in setdiff(lines, ref)) {
  pr <- d[d$CellLine %in% c(ref, cmp), ]
  pr$CellLine <- relevel(factor(pr$CellLine), ref = ref)
  pr$Timepoint <- droplevels(pr$Timepoint)
  # (1) per-timepoint OR, nonlinear, combined
  f <- catch(glmer(cbind(dead, live) ~ (1 | Experiment) + (1 | Experiment:Sci_WellID) + CellLine + Timepoint +
               Timepoint:CellLine, data = pr, family = binomial, control = ctrl), paste("per-timepoint", ref, cmp))
  b <- fixef(f); V <- as.matrix(vcov(f)); cl <- paste0("CellLine", cmp)
  tps <- levels(pr$Timepoint)
  for (k in seq_along(tps)) {
    w <- setNames(rep(0, length(b)), names(b)); w[cl] <- 1
    if (k > 1) {
      nm <- intersect(c(paste0("Timepoint", tps[k], ":", cl), paste0(cl, ":Timepoint", tps[k])), names(b))
      if (length(nm) == 1) w[nm] <- 1 else next
    }
    est <- sum(w * b); se <- sqrt(as.numeric(t(w) %*% V %*% w))
    rows1[[length(rows1) + 1]] <- data.frame(Reference = ref, Compare = cmp, Time = k - 1, t(wald(est, se)))
  }
  if (!(ref %in% refs)) next
  # (3) pooled time-averaged OR: as MRID's structure, and with a line-by-experiment random effect, which lets the
  #     line difference vary between experiments (heterogeneity) so the interval reflects it
  for (spec in c("MRID structure", "line x experiment")) {
    fm <- if (spec == "MRID structure") cbind(dead, live) ~ CellLine + Timepoint + (1 | Experiment) + (1 | Experiment:Sci_WellID)
          else cbind(dead, live) ~ CellLine + Timepoint + (1 | Experiment) + (1 | Experiment:CellLine) + (1 | Experiment:Sci_WellID)
    f3 <- catch(glmer(fm, data = pr, family = binomial, control = ctrl), paste("pooled", spec, ref, cmp))
    s3 <- coef(summary(f3))[cl, ]
    rows3[[length(rows3) + 1]] <- data.frame(Reference = ref, Compare = cmp, Experiment = "all (pooled)", model = spec,
                                            n_wells = length(unique(paste(pr$Experiment, pr$Sci_WellID))),
                                            t(wald(s3[1], s3[2])), singular = isSingular(f3))
  }
  # (2) per-experiment time-averaged OR
  for (ex in unique(pr$Experiment)) {
    pe <- pr[pr$Experiment == ex, ]
    if (length(unique(pe$CellLine)) < 2) next
    pe$Timepoint <- droplevels(pe$Timepoint)
    f2 <- tryCatch(catch(glmer(cbind(dead, live) ~ CellLine + Timepoint + (1 | Sci_WellID), data = pe,
                               family = binomial, control = ctrl), paste("per-experiment", ex, ref, cmp)),
                   error = function(e) NULL)
    if (is.null(f2)) next
    s2 <- coef(summary(f2))[cl, ]
    rows2[[length(rows2) + 1]] <- data.frame(Reference = ref, Compare = cmp, Experiment = ex,
                                            n_wells = length(unique(pe$Sci_WellID)), t(wald(s2[1], s2[2])),
                                            singular = isSingular(f2))
  }
}
write.csv(do.call(rbind, rows1), file.path(out, "OR_rederived_per_timepoint.csv"), row.names = FALSE)
write.csv(do.call(rbind, rows2), file.path(out, "OR_time_averaged_per_experiment.csv"), row.names = FALSE)
write.csv(do.call(rbind, rows3), file.path(out, "OR_time_averaged_pooled.csv"), row.names = FALSE)
w <- if (length(warn_log)) do.call(rbind, warn_log) else data.frame(model = character(0), warning = character(0))
write.csv(w, file.path(out, "warnings_12b.csv"), row.names = FALSE)
cat("done\n")

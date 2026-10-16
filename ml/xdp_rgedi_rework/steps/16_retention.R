#!/usr/bin/env Rscript
# Step 16 (pooled part): live-cell retention model across experiments (decision D26, model C1).
#   L_wt ~ Poisson, log E[L] = ET + LE + slope_line * days + well level/slope + observation-level noise
#   ET    experiment x timepoint effects: plate-wide imaging-day jumps, shared maturation and loss
#   LE    line x experiment intercepts: plating density
#   days  (hours - hours at k0) / 24, so weekend gaps keep their true length
#   (0 + days | LE) line x experiment variation in the slope (the D25 heterogeneity term); (1 + days | Well);
#   (1 | Obs) observation-level effect (Poisson-lognormal overdispersion).
# Slopes are identified relative to one line (14C); every pair contrast comes from the coefficients and their
# covariance. exp(slope_A - slope_B) = ratio of the daily live-cell retention of A to B (< 1: A loses live cells
# faster). It is a death-rate difference only if non-death count changes are the same in both lines.
# Fits: the primary set (env XDP_RGEDI_PRIMARY, default W90 = rule D centre with 90% window, D28) with and without
# the heterogeneity term, the rule D single cutoff (D_sd, run-1 primary), rule B, detected cells N (threshold-free),
# the primary set with the window starting at T1 and at k0 + 1, and without XDP10 (13I gains cells there).
# Fitting: nAGQ = 0 (fixed effects estimated inside the penalised IRLS step; with ~90 fixed effects the default
# second stage optimises them by derivative-free search and does not finish in reasonable time). Counts are in the
# hundreds, where this is accurate; the primary fit is repeated with the Laplace approximation (nAGQ = 1) as a check.
# Usage: Rscript --vanilla 16_retention.R <out_root> [laplace]   (reads 14_count_tables, writes 16_retention)
suppressPackageStartupMessages({ library(lme4); library(DHARMa) })
args <- commandArgs(trailingOnly = TRUE)
root <- normalizePath(args[1])
LAPLACE <- length(args) > 1 && args[2] == "laplace"
P <- Sys.getenv("XDP_RGEDI_PRIMARY", "W90")
src <- file.path(root, "14_count_tables"); out <- file.path(root, "16_retention")
dir.create(out, showWarnings = FALSE)
cnt <- read.csv(file.path(src, "counts_long.csv"), stringsAsFactors = FALSE)
cnt$reference <- cnt$reference %in% c("True", "TRUE", TRUE)
cnt$missing <- cnt$missing %in% c("True", "TRUE", TRUE)
k0 <- read.csv(file.path(src, "k0.csv"), stringsAsFactors = FALSE)
wells <- read.csv(file.path(src, "wells_baseline.csv"), stringsAsFactors = FALSE)
REF <- "CTR-33114.C"
LINES <- c("CTR-33114.C", "CTR-33113.2I", "XDP-33109.2B", "XDP-33363.D")
SHORT <- c("CTR-33114.C" = "14C", "CTR-33113.2I" = "13I", "XDP-33109.2B" = "09B", "XDP-33363.D" = "63D")
ctrl <- glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))
MIN_LIVE <- 50

warn_log <- list()
catch <- function(expr, what) withCallingHandlers(expr, warning = function(w) {
  warn_log[[length(warn_log) + 1]] <<- data.frame(model = what, warning = conditionMessage(w))
  invokeRestart("muffleWarning") }, message = function(m) {
  warn_log[[length(warn_log) + 1]] <<- data.frame(model = what, warning = trimws(conditionMessage(m)))
  invokeRestart("muffleMessage") })

window_data <- function(set, start = c("k0", "T1", "k0+1"), outcome = "L", drop_exp = NULL) {
  start <- match.arg(start)
  d <- cnt[cnt$set == set & cnt$reference & !cnt$missing & !(cnt$exp %in% drop_exp), ]
  st <- setNames(k0$k0, k0$exp)
  if (start == "T1") st[] <- 1
  if (start == "k0+1") st <- st + 1
  d <- d[d$tp >= st[d$exp], ]
  # baseline: live count at the window start (missing baseline or < 50 live cells -> well dropped)
  base <- d[d$tp == st[d$exp], c("exp", "well", "L")]
  ok <- paste(base$exp, base$well)[base$L >= MIN_LIVE]
  d <- d[paste(d$exp, d$well) %in% ok, ]
  h0 <- tapply(d$hours[d$tp == st[d$exp]], d$exp[d$tp == st[d$exp]], function(x) x[1])
  d$days <- (d$hours - h0[d$exp]) / 24
  d$y <- d[[outcome]]
  d$ET <- factor(paste(d$exp, d$tp, sep = ":T")); d$LE <- factor(paste(d$exp, d$line, sep = ":"))
  d$Well <- factor(paste(d$exp, d$well, sep = ":")); d$Obs <- factor(seq_len(nrow(d)))
  d$LET <- factor(paste(d$exp, d$line, d$tp, sep = ":"))
  for (l in setdiff(LINES, REF)) d[[paste0("s_", SHORT[l])]] <- d$days * (d$line == l)
  d
}

fit_c1 <- function(d, hetero = TRUE, what = "", nAGQ = 0) {
  slopes <- paste0("s_", SHORT[setdiff(LINES, REF)])
  fx <- paste("y ~ 0 + ET + LE +", paste(slopes, collapse = " + "))
  # (1 | LET): line x experiment x timepoint shocks shared by all wells of a line (added after review: per-line
  # timepoint deviations from a straight trend are large in XDP3, XDP8, XDP10, XDP11)
  re <- if (hetero) "+ (0 + days | LE) + (1 | LET) + (1 + days | Well) + (1 | Obs)" else "+ (1 | LET) + (1 + days | Well) + (1 | Obs)"
  catch(glmer(as.formula(paste(fx, re)), data = d, family = poisson, control = ctrl, nAGQ = nAGQ), what)
}

pairs_of <- function(f, label) {
  b <- fixef(f); V <- as.matrix(vcov(f))
  idx <- setNames(paste0("s_", SHORT[setdiff(LINES, REF)]), setdiff(LINES, REF))
  stopifnot(all(idx %in% names(b)))   # rank-deficient columns dropped by lme4 must be intercepts, never slopes
  vec <- function(l) { w <- setNames(rep(0, length(b)), names(b)); if (l != REF) w[idx[[l]]] <- 1; w }
  rows <- list()
  for (ref in LINES) for (cmp in setdiff(LINES, ref)) {
    w <- vec(cmp) - vec(ref)
    est <- sum(w * b); se <- sqrt(as.numeric(t(w) %*% V %*% w))
    rows[[length(rows) + 1]] <- data.frame(fit = label, Reference = ref, Compare = cmp, slope_diff = est, se = se,
                                           ratio_per_day = exp(est), lower = exp(est - 1.96 * se),
                                           upper = exp(est + 1.96 * se), p = 2 * pnorm(-abs(est / se)))
  }
  do.call(rbind, rows)
}

specs <- list(
  list(label = P, set = P, start = "k0", outcome = "L", hetero = TRUE),
  list(label = paste0(P, "_no_hetero"), set = P, start = "k0", outcome = "L", hetero = FALSE),
  list(label = "D_sd", set = "D_sd", start = "k0", outcome = "L", hetero = TRUE),
  list(label = "B", set = "B", start = "k0", outcome = "L", hetero = TRUE),
  list(label = "N", set = P, start = "k0", outcome = "N", hetero = TRUE),
  list(label = paste0(P, "_T1"), set = P, start = "T1", outcome = "L", hetero = TRUE),
  list(label = paste0(P, "_k0p1"), set = P, start = "k0+1", outcome = "L", hetero = TRUE),
  list(label = paste0(P, "_noXDP10"), set = P, start = "k0", outcome = "L", hetero = TRUE, drop = "XDP10"))
if (LAPLACE) {
  d <- window_data(P, "k0", "L")
  t0 <- Sys.time()
  f <- fit_c1(d, TRUE, paste(P, "laplace"), nAGQ = 1)
  p <- pairs_of(f, paste0(P, "_laplace"))
  p$minutes <- round(as.numeric(difftime(Sys.time(), t0, units = "mins")), 1)
  write.csv(p, file.path(out, "c1_pairs_laplace.csv"), row.names = FALSE)
  wl <- if (length(warn_log)) do.call(rbind, warn_log) else data.frame(model = character(), warning = character())
  write.csv(wl, file.path(out, "c1_warnings_laplace.csv"), row.names = FALSE)
  cat("LAPLACE ALLDONE\n"); quit(save = "no")
}
pairs <- list(); info <- list(); dharma <- list()
for (s in specs) {
  t0 <- Sys.time()
  d <- window_data(s$set, s$start, s$outcome, s$drop)
  f <- fit_c1(d, s$hetero, s$label)
  pairs[[s$label]] <- pairs_of(f, s$label)
  vc <- as.data.frame(VarCorr(f))
  info[[s$label]] <- data.frame(fit = s$label, n_obs = nrow(d), n_wells = nlevels(d$Well), n_LE = nlevels(d$LE),
                                sd_LE_slope = if (s$hetero) vc$sdcor[vc$grp == "LE" & vc$var1 == "days" & is.na(vc$var2)] else NA,
                                sd_well_slope = vc$sdcor[vc$grp == "Well" & vc$var1 == "days" & is.na(vc$var2)],
                                sd_obs = vc$sdcor[vc$grp == "Obs"], sd_LET = vc$sdcor[vc$grp == "LET"], singular = isSingular(f),
                                minutes = round(as.numeric(difftime(Sys.time(), t0, units = "mins")), 2))
  if (s$label == P) {
    set.seed(20261014)
    r <- simulateResiduals(f, n = 250)
    disp <- testDispersion(r, plot = FALSE); uni <- testUniformity(r, plot = FALSE)
    d$res <- residuals(r)
    d <- d[order(d$Well, d$tp), ]
    lag1 <- unlist(lapply(split(d$res, d$Well), function(x) if (length(x) > 2) cor(x[-1], x[-length(x)]) else NA))
    dharma[[1]] <- data.frame(fit = s$label, dispersion_ratio = unname(disp$statistic), dispersion_p = disp$p.value,
                              ks_D = unname(uni$statistic), ks_p = uni$p.value,
                              median_within_well_lag1 = median(lag1, na.rm = TRUE),
                              share_wells_lag1_pos = mean(lag1 > 0, na.rm = TRUE))
  }
  cat(s$label, "done", format(Sys.time(), "%H:%M:%S"), "\n")
}
write.csv(do.call(rbind, pairs), file.path(out, "c1_pairs.csv"), row.names = FALSE)
write.csv(do.call(rbind, info), file.path(out, "c1_fits.csv"), row.names = FALSE)
write.csv(do.call(rbind, dharma), file.path(out, "c1_dharma.csv"), row.names = FALSE)
wl <- if (length(warn_log)) do.call(rbind, warn_log) else data.frame(model = character(), warning = character())
write.csv(wl, file.path(out, "c1_warnings.csv"), row.names = FALSE)
writeLines(capture.output(sessionInfo()), file.path(out, "c1_sessionInfo.txt"))
cat("ALLDONE\n")

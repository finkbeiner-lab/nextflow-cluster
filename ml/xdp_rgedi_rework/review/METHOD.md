# How the live/dead cutoffs were made (run 2, "W90")

This page describes the cutoff rule in enough detail to write it again from scratch. All the values it refers to
are in `2_live_dead_calls/cutoffs_all_parameters.csv`, so each stage can be checked on its own.

Everything is done on **log10(GEDI ratio)**, with GEDI ratio = RFP mean / GFP mean of a cell. "Case" means one
cell line in one experiment.

## Overview

1. For each case with S.A.-killed wells, describe two populations:
   - **live**: the untreated cells of that line (centre and spread);
   - **dead**: the dead component of a mixture fitted to the S.A. cells after the dose.
2. The **centre** of the buffer zone is where the two population densities are equal.
3. The **edges** are where one population is 9 times as dense as the other (90% sure, equal priors).
4. A case with no usable S.A. population gets its edges by **transfer** from the same line in other experiments.
5. Every cell is then called live (below the live edge), dead (above the dead edge), or dropped (in between).

## Step 1 - The cells used for each fit

Per case, the code uses only that line's cells, at the S.A. fit timepoints of the experiment
(`3_experiment_info/experiment_design.csv`, column `sa_fit_tps`):

| Experiment | S.A. cells | Live reference cells |
|---|---|---|
| XDP1 | all wells at T13-T14 (whole plate dosed after T10 and T12) | the same wells at T10, before the first dose |
| XDP3 | S.A. wells at T10 | untreated wells of the line at T10 |
| XDP5 | S.A. wells at T9 | untreated wells of the line at T9 |
| XDP10 | S.A. wells at T9-T10 | untreated wells of the line at T9-T10 |
| XDP11 | S.A. wells at T13 | untreated wells of the line at T13 |
| XDP8 | none (S.A. plate was not imaged) | (transfer only, see step 5) |

In `all_cells/`, S.A. cells have `condition = sa` and reference cells have `condition = untreated`.

## Step 2 - The live population (reference cells)

- **Centre m** = the mode of the reference cells: Gaussian kernel density (Scott bandwidth), evaluated on 600
  points between the 1st and 99th percentile. If there are more than 20,000 cells, the code uses a random
  subsample of 20,000 (numpy `default_rng(0)`).
- **Spread s** = (m - median of the reference cells below m) / 0.674.

  The spread comes from the low side only, because dead cells (high ratio) cannot reach it. For a normal
  distribution this is the SD.

Columns: `mode_ratio` (10^m) and `sd_inner` (s, in log10 units).

## Step 3 - The dead population (S.A. cells)

- Fit a **3-component 1-D Gaussian mixture** to log10(ratio) of the S.A. cells: scikit-learn `GaussianMixture`,
  20 k-means starts, `reg_covar = 1e-6`, `tol = 1e-8`, `max_iter = 2000`, `random_state = 20261013`.
- **Live component**: the component with the highest posterior probability at m (from step 2).
- **Dead component**: of the components whose mean is above the live component's mean and whose weight is at
  least 0.05, the one with the highest mean. Its mean and SD are mu_d and s_d.
- Any component below the live component is a low tail and is not used.

Columns: `dead_mu_ratio` (10^mu_d), `dead_w` (its weight). The SD s_d is in the step 06 output
(`dead_sd` in `thresholds_direct.csv` of a rerun).

**Usability.** A case keeps its own fit only if it passes all of these tests. If not, it goes to transfer
(step 5). It is never dropped.
- at least 500 S.A. cells and at least 2 S.A. wells;
- a dead component is found, and every component holds at least 50 cells (weight x cells);
- BIC improves by more than 10 from 1 to 2 components;
- Ashman's D between the live population (m, s) and the dead component is at least 2;
- the dead component's weight in the S.A. cells is at least 0.10 higher than in the reference cells. (The
  weight in the reference cells comes from refitting only the weights, with the components held fixed.)

## Step 4 - Centre and edges of the buffer zone

With L = Normal(m, s) and D = Normal(mu_d, s_d), both unweighted (equal priors):

- **Centre c**: the value between m and mu_d where the two densities are equal: log L(c) = log D(c).
  Column `D_live` ("rule D").
- **Live edge**: the value below c where log D(x) - log L(x) = -ln 9, so the cell is 9 times more likely to be
  live. Column `W90_live`.
- **Dead edge**: the value above c where log D(x) - log L(x) = +ln 9. Column `W90_dead`.
- If the dead edge is never reached (populations overlap heavily), it is set to mu_d and the case is flagged.
  This happens only for XDP11 63D in the 95% set. It does not happen in W90.

All three are reported as ratios (10^value).

### Worked example: XDP11, 14C (direct S.A. fit, T13)

| Quantity | Value (log10) | Ratio |
|---|---|---|
| live centre m | -1.1507 | 0.0707 |
| live spread s | 0.2718 | |
| dead mean mu_d | -0.5828 | 0.2613 |
| dead SD s_d | 0.1302 | |
| centre c (equal density) | -0.8099 | **0.1549** |
| live edge (D/L = 1/9) | -0.9173 | **0.1210** |
| dead edge (D/L = 9) | -0.6593 | **0.2191** |

These numbers reproduce the XDP11 14C row of `cutoffs_all_parameters.csv`.

## Step 5 - Transfer (cases without a usable S.A. fit)

Cases that use transfer: XDP1 09B and 63D, all of XDP3 and XDP8, XDP5 13I, XDP10 09B
(`cutoff_source = transfer`).

1. For every usable case of the **same line**, write each edge as an offset from that case's own live
   population, in live-SD units: kappa = (edge - m) / s.
2. Take the **median kappa** over those source experiments.
3. Measure the target's own live population (step 2) on its reference cells:
   - XDP8: untreated wells at T5 (its last image);
   - XDP3: untreated wells at T10;
   - XDP5 13I: untreated wells at T9;
   - XDP10 09B: untreated wells at T9-T10;
   - XDP1 09B and 63D: the same wells at T10, before dosing.
4. Edge of the target = 10^(m_target + median kappa x s_target).

Source experiments per line: 14C: XDP1, 5, 10, 11. 13I: XDP10, 11. 09B: XDP5, 11. 63D: XDP5, 10, 11.

### Worked example: XDP3, 14C (transfer)

- kappa of the 14C live edge in XDP1, 5, 10, 11: 2.086, 1.254, 1.345, 0.859. The median is 1.299.
- kappa of the 14C dead edge: 2.669, 2.254, 2.108, 1.808. The median is 2.181.
- XDP3 14C reference (untreated, T10): m = -1.2996, s = 0.1276.
- Live edge = 10^(-1.2996 + 1.299 x 0.1276) = **0.0735**. Dead edge = 10^(-1.2996 + 2.181 x 0.1276) = **0.0952**.

## Step 6 - Calls and MRID input

- Each case has one pair of edges, used at every timepoint.
- A cell is **live** if ratio < live edge and **dead** if ratio > dead edge. Otherwise it is **dropped** (not in the
  MRID input).
- MRID input = untreated wells only (XDP1: all wells, T1-T10). In MRID's format, `live_guesses` = 1 for live
  and 0 for dead.
- `classifier.score.live` = `live_guesses` and `classifier.score.dead` = 1 - `live_guesses` for every cell in
  the files. MRID counts dead cells per well and timepoint from `classifier.score.dead` and live cells from
  `live_guesses`.

## Uncertainty (intervals in `cutoffs_all_parameters.csv`, columns `*_lo`, `*_hi`)

- **Direct fits**: well-cluster bootstrap, 1,000 replicates. S.A. wells and reference wells are resampled
  separately. In XDP1 they are resampled together, because the same wells give both. XDP10 has 2 S.A. wells per
  line, so its cells are resampled within those wells. Each replicate refits the mixture (warm start plus one
  fresh start, the better BIC is kept).
- **Transfer**: source experiments and target wells are resampled together, 2,000 replicates.
- These intervals cover resampling only, not the choice of rule.

## Where each stage is in the code

| Stage | File |
|---|---|
| cells per case, mixture, usability, bootstrap | `steps/06_fit_thresholds.py` |
| live centre and spread, mixture, equal density, window edges | `lib/mixture.py` (`live_reference`, `fit_gmm`, `identify`, `equal_density`, `window_edges`) |
| buffer-zone edges | `steps/06b_window_cutoffs.py` |
| transfer | `steps/08_transfer.py` |
| calls and MRID input files | `lib/livedead.py`, `steps/11_mrid_inputs.py` |
| rules and settings | `config/params.yaml`, `config/decisions.yaml` (D20, D21, D28), `config/experiments.yaml` |

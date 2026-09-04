# MRID → Nextflow Port Plan (finalized 2026-09-03)

Fold the Gladstone Bioinformatics-Core **MRID** RGEDI survival analysis (authors Kaye,
Reuben, Lam) into the Nextflow pipeline so an RGEDI run goes end-to-end:

```
montage → align → Cellpose + EGFP-gate segmentation → track → ratio → MRID GLMM survival stats → tables + plots
```

all from `finkbeiner.config`, no R/Rmd hand-steps. **Keep the statistics faithful in R; port the plumbing.**

## What MRID is (two stages)

- **Ratio stage** (`get_MRID_Ratio.R` + `MRID_Ratio_Functions.R`): per experiment. Pairs each
  cell's FITC+RFP on `timepoint_well_neuron`, computes `ratio = RFP_PixelIntensityMean /
  GFP_PixelIntensityMean`, splits **live/dead by two hand-picked thresholds** (NOT a trained model
  in the current path — the original CNN is legacy, see `convert_old_cnnoutput`), writes
  `<expt>_ratio_output.csv` + cell counts + QC plots.
- **Logodds stage** (`MRID_Logodd_Functions_June2026.R`, driven by an `.Rmd`): across experiments.
  Reads `ratio_output.csv` + timepoint→hours, aggregates per well×timepoint dead/live counts, fits a
  **binomial GLMM** and derives odds ratios over time:
  ```r
  glmer(cbind(classifier.score.dead, live_guesses) ~ Condition + Timepoint + Condition:Timepoint
        + (1|Experiment) + (1|Experiment:Sci_WellID) + (1|Experiment:Plate) + (1|CellLine),
        family = binomial)
  ```
  OR = exp(estimate); Wald 95/99% CIs; LRT p-value vs a Timepoint-only null. Linear (numeric time,
  single slope) vs nonlinear (factor time, per-tp coef) variants. **Use June2026** — it fixes a
  vcov/Plate bug in the older `RT_July1` that corrupts SEs/CIs when Plate is a fixed effect.

## Key decisions

1. **Keep the GLMM in R** (wrapped), not reimplemented in Python — reproducing `glmer` binomial mixed
   models in statsmodels is risk for no benefit. Adapters + ratio in Python `bin/` like the rest.
2. **Live/dead threshold = auto (default) + manual override** (Austin, 2026-09-03), scoped **per
   experiment × timepoint, anchored to the LIVE (lower-ratio) mode** — the same negative-anchored GMM
   logic as the EGFP gate, with "live" as the reference population. Rationale (biology):
   - **Per-timepoint** because two *technical* drifts move the live baseline over time: GFP maturation
     early (GFP brightens → live ratio drifts down) and differential photobleaching late. A fixed
     per-experiment threshold would miscall early/late frames.
   - **Anchored to the live mode, NOT the whole-distribution antimode**, because death is an **RFP
     spike** (the biology we measure): re-fitting against the whole per-tp distribution would
     partially normalize the death signal away, and would break when a frame is unimodal (early
     all-live / late all-dead). Anchoring to the live mode tracks the technical drift while detecting
     death as cells jumping a fixed distance above the live baseline, and stays valid when unimodal.
   - **Manual override** for datasets where the live mode isn't cleanly separable (BIC/reliability
     fail) — auto flags `REVIEW` and falls back to config thresholds (never emits a silent bad gate).
   - Reuses `calibrate_gate.py`'s GMM+BIC+reliability machinery (shared module).
3. **Port all plots** (ratio: SD/violin/condition/cell-count heatmaps; logodds: death-rate/residual/
   OR-over-time), folded into each process.
4. **Input = our gated per-cell output** (`percell_gated.csv`), with legacy `cell_data.csv` as a
   back-compat path.

## Three PRs

### PR1 — `bin/ratio.py` + `bin/ratio_threshold.py` (Python; replaces MRID's ratio stage)
- `ratio.py`: read gated per-cell data (or legacy `cell_data.csv`), pair FITC+RFP, `ratio=RFP/GFP`,
  apply live/dead thresholds → write MRID's exact `<expt>_ratio_output.csv` schema
  (`filenames, Sci_WellID, Sci_SampleID, Drug, Timepoint, ratio, live_guesses,
  classifier.score.live, classifier.score.dead`) + `cell_count.csv` / `cell_count_well.csv`.
  Faithful details from the deep read: dedup duplicate objects by largest `BlobArea`; MRID drops cells
  in the ambiguous band between live and dead thresholds; `classifier.score.*` are hard 0/1 on
  `live_threshold`; back-fill empty wells as NA in `cell_count_well`.
- `ratio_threshold.py`: per exp×tp GMM on `log10(ratio)`, lower mode = live, threshold = fixed
  distance above it (anchored); BIC+reliability flag; `--method auto|manual`, `--live/--dead`,
  `--ambiguous-margin`. Writes `livedead_thresholds.csv` (auditable, like `gate_calibration.csv`).
- **Validate:** reproduce MRID `Example Output/final_outputs/UC-PCMI7_ratio_output.csv` in manual mode
  (thresholds 0.07/0.1) byte-for-byte, then exercise auto mode.

### PR2 — `bin/mrid_logodds.R` + Nextflow process `MRID_LOGODDS` (wrapped R stats)
- Headless, param-driven `Rscript` that sources `MRID_Logodd_Functions_June2026.R` (+ the ratio
  functions it needs), **sets the globals the Rmd used to provide** (`multiple_experiments`,
  `multiple_plates`, `logodd_data`, `fit_data`, `ratio_df`, …) so the functions run unchanged, and
  executes the pipeline chunks headless: `mrid_get_logodd` → `finalize_df` → `mrid_death_plot` →
  `mrid_residuals_plot` → `mrid_lmer_condition_comparisons` → `mrid_logodd_combine_plot`.
- Writes `Data.csv`, `OR_*.csv`, `Combined_OR_*.csv`, and all plots to `output_path`.
- Minimal-refactor de-interactive-ification (no stats touched): params from args, `set.seed()`,
  drop `Sys.Date()` from filenames, redirect CWD writes to `output_path`, fix the two real bugs
  (`for(i in 1:range(n_exp))`; `//` join in `convert_old_cnnoutput`).
- Container R deps to confirm/add: `lme4, dplyr, tidyr, stringr, ggplot2, plyr, gridExtra, gghalves,
  xtable, knitr, scales, rlang` (several are used-but-undeclared).
- **Validate** OR/CI outputs against MRID `Example Output` on the same input.

### PR3 — config, plate layout, pipeline wiring
- `bin/plate_layout.py`: generate `Sci_WellID, Sci_SampleID, Drug` from the experiment config /
  platemaps (we already have XDP line maps) → `mrid_plate_layout = 'auto'`.
- `finkbeiner.config` additions (see `nextflow/config_additions.txt`).
- `modules.nf` + `pipeline.nf`: `MRID_RATIO` (per experiment) → `MRID_LOGODDS` (across experiments),
  gated by `DO_MRID_RATIO` / `DO_MRID_LOGODDS`, downstream of segmentation/tracking.

## Automation blockers handled (from the deep read)
Human-in-the-loop thresholds → config + auto method; global-variable coupling → driver sets them;
CWD writes → `output_path`; `Sys.Date()` filenames → fixed names; knit-only `print/cat/kable` results →
structured CSVs; `sample(colors())` → seeded; two logic bugs → fixed.

## Build/validate order
PR1 (pure Python, testable now vs MRID example) → PR2 (needs container R to validate) → PR3 (wiring).
Each validates against MRID's own `Example Input`/`Example Output` before wiring in.
</content>

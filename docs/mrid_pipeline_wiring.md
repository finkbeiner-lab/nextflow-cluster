# RGEDI → MRID pipeline: modules, run recipe, wiring

## Runnable now (2-stage self-contained sbatch, validated Python chain)

The full RGEDI analysis reuses the existing background-corrected + aligned montages and runs:
**Cellpose (ungated) → EGFP gate (k=1.5) → track → ratio → MRID logodds.** Two staged sbatches in
`~/Downloads/sbatch-claude/` bind-mount the committed `bin/` over `/app` (no SIF rebuild):

- `rgedi_segment.sbatch` (GPU): `cellpose_montage_rgedi.py` on a well×timepoint manifest →
  `<OUTDIR>/percell_all.csv` + `masks/`. Edit the embedded manifest for the full well set. The
  default manifest (D03 CTR + J03 XDP × T1–T9) is a ready **test**.
- `rgedi_analyze.sbatch` (CPU): `egfp_gate.py` (k=1.5) → `track_cells.py` → `ratio.py` →
  `mrid_logodds.R` → `<OUTDIR>/mrid/` (ratio_output, OR_*.csv, Data.csv, plots).

**One-time cluster prereq:** `cd ~/nextflow-cluster && git fetch origin && git checkout austin/RGEDI && git pull`
(so the bind-mounted `bin/` has these modules + `bin/mrid_R/`).

Module deps: all Python is numpy/scipy/skimage/pandas/tifffile/cellpose only — **no sklearn**
(`gmm1d.py` replaces it). `mrid_logodds.R` needs the container R stack (lme4, tidyverse, …).
Not yet validated on cluster: the R stage + first full multi-timepoint run — the first run shakes those out.

## EGFP gate default: k = 1.5
`egfp_gate.py` defaults to **k = 1.5 SD above each experiment×timepoint background mode** (Austin,
2026-09-03), with `--manual-fitc` overriding any REVIEW-flagged group (e.g. XDP10, whose split isn't
cleanly bimodal). `--anchor VALUE` opts into the old anchored mode.

---

# MRID pipeline wiring (PR3) — Nextflow form, reviewable patch (sbatch above is the validated path)

New files staged (safe additions): `bin/ratio.py`, `bin/ratio_threshold.py`, `bin/mrid_logodds.R`,
`bin/plate_layout.py`, `bin/mrid_R/*.R`. The changes below to `finkbeiner.config`, `modules.nf`,
and `pipeline.nf` are held here as a patch so the live pipeline isn't disturbed until we commit.

## `finkbeiner.config` additions
```groovy
// ---- MRID RGEDI survival analysis ----
DO_MRID_RATIO         = true
DO_MRID_LOGODDS       = true
mrid_plate_layout     = 'auto'          // 'auto' (generate from platemap) or /path/to/layout.csv
mrid_platemap_source  = ''              // platemap csv when mrid_plate_layout='auto'
mrid_livedead_method  = 'auto'          // 'auto' | 'manual'
mrid_live_threshold   = ''              // manual live cutoff / fallback for unreliable auto fits
mrid_dead_threshold   = ''              // manual dead cutoff / fallback
mrid_ambiguous_margin = 0.0             // dex gap live->dead drop-zone (auto)
mrid_anchor_exp       = ''              // reference experiment for the auto threshold constant
mrid_anchor_value     = 0.07            // live cutoff the anchor experiment reproduces
mrid_compare          = 'class'         // 'class' | 'cellline' | 'drug'
mrid_classes          = 'CTR,XDP'       // for compare=class; first token = class 0
mrid_ref_conditions   = ''              // lmer reference condition(s)
mrid_time_model       = 'linear'        // 'linear' (numeric time) | 'nonlinear' (factor time)
mrid_multiple_experiments = true
mrid_multiple_plates      = false
```

## `modules.nf` — new processes
```groovy
process MRID_RATIO {
    input:  val ready
    output: val true, emit: ready
    when:   params.DO_MRID_RATIO
    script:
    def layout = params.mrid_plate_layout == 'auto' ? "${params.output_path}/plate_layout.csv"
                                                     : params.mrid_plate_layout
    def livearg = params.mrid_live_threshold  ? "--live ${params.mrid_live_threshold}"  : ''
    def deadarg = params.mrid_dead_threshold  ? "--dead ${params.mrid_dead_threshold}"  : ''
    """
    ${params.mrid_plate_layout == 'auto' ?
      "python /app/plate_layout.py ${params.mrid_platemap_source} --out ${layout}" : "true"}
    python /app/ratio.py --input ${params.output_path}/percell_gated.csv \\
        --plate-layout ${layout} --expt-name ${params.experiment} \\
        --output-path ${params.output_path}/mrid \\
        --method ${params.mrid_livedead_method} --anchor-exp ${params.mrid_anchor_exp} \\
        --anchor ${params.mrid_anchor_value} --ambiguous-margin ${params.mrid_ambiguous_margin} \\
        ${livearg} ${deadarg}
    """
}

process MRID_LOGODDS {
    input:  val ready
    output: val true, emit: ready
    when:   params.DO_MRID_LOGODDS
    script:
    """
    Rscript /app/mrid_logodds.R --input_path ${params.output_path}/mrid \\
        --output_path ${params.output_path}/mrid \\
        --compare ${params.mrid_compare} --classes ${params.mrid_classes} \\
        --ref_conditions ${params.mrid_ref_conditions} --time_model ${params.mrid_time_model} \\
        --multiple_experiments ${params.mrid_multiple_experiments} \\
        --multiple_plates ${params.mrid_multiple_plates}
    """
}
```

## `pipeline.nf` — wire after segmentation/tracking
```groovy
if (params.DO_MRID_RATIO)   { ratio_ready   = MRID_RATIO(tracking_ready) }
if (params.DO_MRID_LOGODDS) { logodds_ready = MRID_LOGODDS(ratio_ready) }
```

## Container check
Confirm `bin/mrid_R` R deps are in `nextflow-cluster.def` (R already present): `lme4, dplyr, tidyr,
stringr, ggplot2, plyr, gridExtra, gghalves, xtable, knitr, scales, rlang`. Add any missing, rebuild SIF.

## Note on ratio_output filenames / timepoint hours
`mrid_logodds.R` expects `<expt>_ratio_output.csv` + a `timepoint.csv` (Timepoint,Hour) in `input_path`.
`ratio.py` writes the ratio_output; add a tiny timepoint.csv writer (from `ElapsedHours`) to `ratio.py`
or a helper — flagged as the one remaining glue piece for PR2's input.
</content>

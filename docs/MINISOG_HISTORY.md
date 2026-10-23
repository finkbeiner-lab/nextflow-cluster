# MINISOG — Development History & Decisions Log

A running record of **what we tried, what worked, and what didn't** while building the
miniSOG-RGEDI death-quantification module (`bin/minisog.py`) and analyzing its first
experiment. Written so that a future run — especially the planned **secondary run with
shorter timepoint intervals and more stimulation conditions** — can build on what worked
and *not re-walk the dead ends*. If the new data misbehaves, the "What didn't work and
why" sections below are the map back.

Branch: `austin/minisog-rgedi` (isolated worktree; does not touch `austin/neurite-module`).
Author: Austin Holub, Finkbeiner Lab. Last updated 2026-09-02.

---

## 1. What MINISOG does

miniSOG-RGEDI is a two-part biosensor. **miniSOG** is blue-light-stimulated → produces ROS
→ eventually toxic → death. **RGEDI** is the ratiometric death reporter: a **red** channel
that flares on the fast calcium influx of death, over a **green (GFP)** morphology/expression
channel. Death is a **ratio**, not a raw intensity:

```
GEDI ratio = red (RGEDI death) / GFP (morphology)      [per tracked cell, per timepoint]
```

Normalizing by GFP removes cell-size / expression variation. MINISOG consumes tracked
per-cell intensities, computes this ratio series, calls per-cell **time-of-death**, and
emits Kaplan-Meier survival + dose-response + a per-cell-line comparison.

**GEDI method reference:** Linsley et al., *Nat. Commun.* 2021, DOI
`10.1038/s41467-021-25549-9`.

---

## 2. Experiment 1 — `hevo-pmsG-1` (the pilot)

Patient fibroblasts (GESTALT collection, Hevolution project). Galaxy DB exp_id
`8b244215-31b5-410b-9d00-8837ad9e1d71`.

| | |
|---|---|
| Wells | 16 (C3–F6) |
| Timepoints | T0–T18 (19 tp, 4 h spacing, 0–72 h) — *finished at T18; T17/T18 arrived later* |
| Tiles | 9 (3×3 montage) |
| Cell lines (rows) | C=TP0357 (93 F), D=TP0359 (35 F), E=TP0388 (26 M), F=TP0398 (84 M) |
| Dose (columns) | col3=500 ms, col4=1 s, col5=5 s, col6=10 s blue light. **All wells stimulated — no unstim control.** |
| Channels | `Epi-GFP16` (morphology), `Epi-Blue` (stim), `Epi-RFP16`/`Epi-RFP16-2` (pre/post red), `Epi-NarrowRFP`/`Epi-NarrowRFP-2` (pre/post, dropped) |

**Readout convention:** death accumulates during the inter-timepoint incubation, so the
**post-stim (`-2`) red series** is the death trajectory, with **T0 as the baseline**. The
`-2` suffix = post-stim.

**Single-GFP design (this run):** only **one** `Epi-GFP16` per timepoint (no `Epi-GFP16-2`),
so the single GFP is the shared ratio denominator for *both* reds: pre = `Epi-RFP16/Epi-GFP16`,
post = `Epi-RFP16-2/Epi-GFP16`. MINISOG handles this via `--gfp_channel Epi-GFP16`.

---

## 3. The pipeline that works (end-to-end)

Per well: **MONTAGE → Cellpose SEGMENTATION (GFP) → TRACKING → MINISOG**.

1. **Montage** (`montage.py`): stitch the 9 tiles. Pattern `robo4_serpentine`, overlap 0.1.
2. **Segment** (`segmentation_montage.py --segmentation_method cellpose`): Cellpose-SAM v4 on
   the **GFP** montage, with CLAHE (clip 0.03) + MAD (k=3) + shape-debris cleanup.
3. **Track** (`tracking_montage.py`): `--track_type proximity --max_dist 450 --motion`,
   **`--target_channel "Epi-GFP16,Epi-RFP16,Epi-RFP16-2"`** (GFP *must* be a target so the
   per-cell GFP intensity — the ratio denominator — lands in the CSV).
4. **MINISOG** (`minisog.py --intensity_source csv`): reads
   `<analysisdir>/<exp>_tracked_montage_summary.csv`, builds the ratio series, thresholds,
   calls death, writes `minisogtrackdata` + `minisogcomparisondata`, emits plots/CSVs.

We run this as **self-contained `.sbatch` scripts** (code embedded as base64) staged in
`~/Downloads/sbatch-claude/` and submitted by Austin over ScaleFT — see
`cluster-batch-workflow.md`. Claude never submits cluster jobs.

---

## 4. Decision log — what we tried, what worked, what didn't

### 4.1 Segmentation & tracking flow

- ❌ **Tile-level flow** (Cellpose per tile → `tracking.py` → `intensity.py`). Dead end:
  `tracking.py` imports `gurobipy`, which is **not in any SIF**. Also tile `intensity.py` is
  incompatible with montage masks.
- ✅ **Montage flow** (`*_montage.py`, scipy `linear_sum_assignment`, gurobi-free). This is
  the lab's production path and what we use.
- ❌ **Overlap tracking** on these motile fibroblasts: 62% dropout, median track 7/17. Cells
  move too far between 4 h frames.
- ✅ **Proximity + motion tracking** (`--track_type proximity --max_dist 450 --motion`):
  dropout ~34%, median 14/17. Plateaus at 450 (600 no better).
- ✅ **Cellpose-SAM v4 + CLAHE** for segmentation. Default threshold seg (area 300–1000 px)
  is nucleus-sized — far too small for 20× spread fibroblasts. CLAHE 0.03 lifts dim
  cytoplasm (coverage 26%→42%); MAD k=3 + shape filter (area<3000 & ecc<0.8, or <800 px)
  removes round dead-cell debris.

### 4.2 Sensor choice: RFP16 vs NarrowRFP

- ✅ **RFP16 wins decisively** in all 4 lines: dynamic range ~1.4–2.6× vs NarrowRFP flat
  ~1.05–1.15× (≈9× dimmer, no usable death signal). **NarrowRFP dropped** from all
  downstream analysis.

### 4.3 Death metric: raw red vs GEDI ratio

- ❌ **Raw-RFP dynamic-range metric** reported a "saturated dose-response" (500 ms ≈ 10 s).
  This was an **artifact** of the raw metric + survivorship bias, not biology.
- ✅ **GEDI ratio (red/GFP) + T0-percentile threshold** recovered a **clean, monotonic
  dose-response**. This is the method in the module.

### 4.4 Montage tile ordering (Austin caught this)

- ❌ **`legacy` pattern** (reverse EVEN rows, `3 2 1 / 4 5 6 / 9 8 7`) — wrong for this data.
  Misplaced columns so vertical seams never matched; no overlap value could fix it.
- ✅ **`robo4_serpentine`** (reverse ODD rows, **`1 2 3 / 6 5 4 / 7 8 9`**), solved by
  object-to-object cross-correlation across all 12 internal seams, confirmed on 2 wells
  (C6, F6). Overlap = **120 px = 10%** (matches template). **Naming note:** captured on
  **Robo4**, not the IXM — earlier called `ixm`, now `robo4_serpentine` (aliases
  `serpentine_odd`/`serpentine_lr`/`ixm`). `montage.py` + configs updated.

### 4.5 Montage refinements — the "is it scientifically sound?" study

Two refinements *looked* better; we tested whether they change the **measurement** by
re-deriving death three ways on one well (F6, 10 s) through the identical deterministic
pipeline (the three-way measurement study).

- ✅ **Per-tile registration → ADOPT.** Purely geometric (relocates tiles, never alters
  intensity), fixes genuine seam-cell mismeasurement. Shifted death fraction 28→32% and
  time-of-death one frame earlier — principled. **Not yet wired into `montage.py`** (still a
  scratch prototype); the remaining geometry refinement.
- ❌ **Self-estimated flat-field → DISPLAY-ONLY.** Flipped **15% of death calls** on identical
  footprints with no ground truth. It's per-channel and self-estimated, so it (a) breaks the
  ratio's natural vignetting cancellation and (b) leaks biology into the "illumination"
  estimate (visible as the threshold drifting). Keep out of the quantitative path; a *measured*
  reference flat-field would be required before reconsidering.

### 4.6 Death threshold — what it is, and the 0.25 question

- The threshold is **data-driven**: `--death_threshold_pct 99` = the 99th percentile of the
  **T0** (all-live) ratio distribution (the "live ceiling"). For Exp 1 this = **0.535**.
- ❌ **A borrowed absolute 0.25** (from other GEDI pipelines) is **too low in our units** —
  our live-cell median ratio is 0.26, so 0.25 sits on the live peak and calls ~50% of the
  harmless 500 ms cells "dead," washing out the dose-response. **Lesson: the absolute ratio
  value is pipeline-specific** (depends on exposure/gain/background/ratio definition) and is
  **not portable**. Always re-derive the threshold from the new experiment's own T0 — do not
  transplant a number.

### 4.7 Trying to get a bimodal live/dead readout

Motivation: GEDI death is a fast switch, so one *hopes* for a bimodal ratio (clean live/dead
clusters). It isn't, on this data. We tried hard:

- ❌ Raw ratio (BC 0.386), ratio/baseline fold-change (0.255), RFP/baseline (0.194),
  background-subtracted ratio (0.242), late+high-dose snapshot (0.383) — all **unimodal**
  (bimodality coefficient < 0.555).
- ~ **Per-cell peak fold-change** was the best (BC **0.527**, borderline) — a live spike with
  a dead tail, not two peaks.
- ❌ **2D multivariate** (ratio-rise × morphology/area) + 2-component GMM — still **one
  continuous cloud**; the GMM just bisects it. Morphology (area) doesn't cleanly separate
  because dying cells that round up are largely *lost* rather than measured as small.

**Root cause (important):** the dead mode is **erased upstream**, not by the math —
(1) **tracking dropout** deletes dying cells from their tracks right as they'd become
confidently dead, and (2) **mean-intensity-over-mask** dilutes the bright/punctate dead RFP.
No feature engineering on the summary CSV can recover a bimodality that the measurement threw
away. See §6 for the real fixes.

### 4.8 Changepoint (switch) death detector

- ~ **Per-cell**, the switch detector works beautifully: dead cells show sharp
  `mean(after)/mean(before)` steps (10–37×), live cells stay flat. Great as a *confidence*
  signal.
- ❌ **As a standalone population death-call it underperforms**: it over-calls low dose
  (fold-change amplifies low-baseline noise → 500 ms/1 s 5–12% vs 1% for value-threshold) and
  under-calls high dose (the sustained-step requirement + dropout drop the fastest deaths →
  10 s 28% vs 55%). It **flattens the dose-response and scrambles the line ranking**. Do not
  use switch-only.

### 4.9 ✅ Combined value-threshold + switch-confirmation (SHIPPED)

The winner and what's in the module. **Death = value-threshold crossing (sustained) AND a
confirmed switch** (`switch_mag = mean(after)/mean(before) ≥ switch_fc`, default 1.5). Death
time = the value crossing.

- **Preserves** the value-threshold's clean dose-response and line ranking (unlike
  switch-only).
- **Rejects** always-high / drifting cells that clear the absolute threshold without a real
  low→high death *event* (the "confidently dead" filter Austin wanted).
- More conservative: removes ~15–25% of value-threshold deaths (the switch-unconfirmed ones);
  75–89% of value deaths *are* switch-confirmed at every dose. Sparse low-dose deaths are
  validated as real (median switch ~2.6).
- **Tradeoff:** slightly under-counts cells that were already elevated at track start (we
  can't observe their rise). Sensitivity vs confidence is a choice: `--switch_confirm` off =
  max sensitivity, on = max confidence.

Built as: `minisog.py --switch_confirm --switch_fc 1.5`. `switch_mag` + `switch_confirmed`
are recorded on **every** track regardless of the flag, so you can post-hoc filter without
re-running.

---

## 5. Final results — Experiment 1 (complete T0–T18, corrected geometry)

Primary sensor **RFP16post** (`Epi-RFP16-2/Epi-GFP16`), plain value-threshold (T0 p99 = 0.535):

| Line | Donor | % dead @72 h | median t-death | dose ρ |
|---|---|---|---|---|
| TP0388 | 26 M | 59% (most vulnerable) | 16 h | 1.00 |
| TP0357 | 93 F | 30% | 36 h | 0.80 |
| TP0398 | 84 M | 21% | 36 h | 1.00 |
| TP0359 | 35 F | 9% (most resistant) | 44 h | 0.95 |

KM by dose: 500 ms/1 s ≈ 99%, 5 s ≈ 73%, 10 s ≈ 38% survive @72 h. **Pre ≈ post** (within
~1%), validating the single-GFP normalization. **No age→death correlation** (youngest donor
is most vulnerable; n=4, underpowered). Combined call (`--switch_confirm`) gives dose
1/1/18/43% and the same ranking, more conservatively.

Full write-up: `~/Library/CloudStorage/Dropbox-Gladstone/Austin Holub/hevolution/
MINISOG_hevo-pmsG-1_report.html`.

---

## 6. Known limitations & the real fixes (do these before concluding "graded biology")

The two upstream limiters below flatten whatever bimodality/contrast exists. Fix them before
deciding the death is genuinely graded rather than switch-like.

1. **Tracking dropout at death (highest leverage).** Dying fibroblasts round up / detach and
   leave their track exactly when they'd be confidently dead — biasing survival optimistic and
   deleting the dead mode. Fix: track on a dilated/union mask, allow mask discontinuity through
   the transition, or keep measuring RFP at the last-known centroid after the morphology mask
   is lost. Report `dropout` as a first-class metric.
2. **Mean-intensity-over-mask dilutes the punctate dead RFP.** Fix: measure **integrated** or
   **high-percentile** (e.g. 90th-pixel) RFP within the mask, or in a fixed disk at the
   centroid. (Requires re-measuring from images+masks — a tracking/intensity change.)
3. Per-tile **registration** (adopted, §4.5) still needs wiring into `montage.py`.
4. Proper **measured** flat-field if illumination correction is ever wanted quantitatively.

---

## 7. Guidance for the SECONDARY run (shorter intervals + more stim conditions)

**Why the new design should help, and what to watch:**

- ✅ **Shorter timepoint intervals directly attack limiter #1.** Finer sampling catches the
  fast death switch *before* the cell drops out — more confirmed switches, less truncation of
  the dead population, and a better chance at the bimodal readout. This is the single most
  promising change.
- ✅ **More stimulation conditions** → a finer dose axis → a stronger dose-response test
  (the `compare_sensors` dose ρ becomes better-powered).
- ⚠️ **Re-derive the threshold — never transplant 0.535 or 0.25.** The T0-percentile is
  scale-dependent; a new experiment (new exposures/gains) has its own ratio scale. Keep
  `--death_threshold_pct 99` (data-driven) and let it recompute. Sanity-check: the resulting
  threshold should sit *above* the live-cell median, in the dose-separating window.
- ⚠️ **GFP must be in the tracking `--target_channel`** (with the reds) or MINISOG has no
  ratio denominator and fails ("no data for sensor / no per-track metrics"). This bit us once.
- ⚠️ **Single vs pre/post GFP:** if the new run images GFP once per timepoint (like Exp 1),
  keep `--gfp_channel Epi-GFP16` (shared denominator). If it images GFP pre *and* post, decide
  which GFP pairs with which red.
- ⚠️ **Schema change:** `minisogtrackdata` gained `switch_mag` + `switch_confirmed` (2026-09-02).
  **Drop the two minisog tables before the first run after this change** so they recreate with
  the new columns (`minisogtrackdata`, `minisogcomparisondata`); rows are per-experiment and
  re-added each run.
- ✅ **Turn on `--switch_confirm`** for the confident-death readout, but also keep a
  value-only run for comparison (they answer sensitivity vs confidence).
- 🧭 **If it doesn't work / back-to-the-drawing-board:** the ranked culprits are almost always
  (1) wrong montage ordering → check seams, (2) GFP missing from tracking targets, (3)
  threshold transplanted instead of re-derived, (4) dropout truncating fast deaths (shorter
  intervals should mitigate; if not, implement limiter #1's fix). The §4 "❌" entries are
  approaches already ruled out — don't re-attempt raw-RFP metric, legacy ordering,
  overlap-only tracking, switch-only death calls, or self-estimated flat-field in the
  quantitative path.

---

## 8. Module reference

**Key MINISOG flags** (`bin/minisog.py`):

| flag | default | meaning |
|---|---|---|
| `--sensors` | `RFP16:Epi-RFP16-2` | `name:post_channel[,…]`; Exp 1 used `RFP16post:Epi-RFP16-2,RFP16pre:Epi-RFP16` |
| `--gfp_channel` | `Epi-GFP16` | ratio denominator (single-GFP design) |
| `--death_metric` | `ratio` | `ratio` (red/GFP) or `raw` |
| `--death_threshold_pct` | `99` | T0 percentile for the death threshold (data-driven; re-derives per run) |
| `--death_persist` | `2` | sustained frames above threshold to call death (also censors unconfirmable last-frame crossings) |
| `--switch_confirm` | off | require a confirmed low→high switch (confident-death) |
| `--switch_fc` | `1.5` | min `mean(after)/mean(before)` for the switch |
| `--baseline_timepoint` | `0` | timepoint defining the T0 all-live baseline |
| `--min_track_len` | `4` | min observed timepoints to score a track |
| `--intensity_source` | `auto` | `csv` (montage summary) / `db` / `auto` |

**DB tables:** `minisogtrackdata` (per track × sensor: metrics, `died`, `death_timepoint`,
`time_to_death`, `switch_mag`, `switch_confirmed`), `minisogcomparisondata` (per line ×
sensor: dose-response, `pct_dead`, `median_time_to_death`). UUID PKs, cascade on delete.

**Nextflow:** `DO_MINISOG` + `minisog_*` params (incl. `minisog_switch_confirm`,
`minisog_switch_fc`) in `finkbeiner.config` → `pipeline.nf` → `modules.nf` `process MINISOG`.
Or run `minisog.py` directly (how the self-contained sbatch does it).

---

*This log is intentionally candid about dead ends. If you add a run, append what you tried and
what happened — future-you will thank present-you.*

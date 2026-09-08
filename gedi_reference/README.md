# GEDI reference — canonical v3 (source of truth)

These three files are **Jeremy Linsley's canonical GEDI2 survival analysis**, added verbatim
(2026-09-08) as the reference we implement against. **Do not edit them** — they are the spec,
not our pipeline code. Validated over hundreds of runs (Hal3, Hal4).

| file | what it is |
|---|---|
| `GEDI_ANALYSIS_RULES.md` | the rules (measurement → GEDI2 two-pass background → 0.025 death → entry/censoring → KM/stats) |
| `gedi_v3_final.py` | reference implementation of the analysis: consumes `gedi_cells.csv` (per-cell long table: `well,tile,timepoint,track_id,area,rfp_mean,gfp_mean`) + `platemap.csv`, writes `gedi_survival_v2.csv` + `gedi_traces_v2.csv` |
| `resilience_final.py` | KM stats: Greenwood 95% CI, log-rank vs vehicle, Benjamini-Hochberg FDR, per-arm protection |

## How this maps into our pipeline (in progress, 2026-09-08)

The reference is written for Jeremy's **single-tile** MASKTRACKED environment. Two separable pieces:

1. **Analysis** (`gedi_v3_final.py` + `resilience_final.py`) — portable and self-contained; it
   consumes a `gedi_cells.csv`. We port this faithfully as the new death-quant, **replacing** the
   montage-based `bin/minisog.py` (T0-percentile threshold + persistence/switch-confirmation are
   dropped — see `MINISOG_HISTORY.md`).
2. **Measurement front-end** (produces `gedi_cells.csv`) — the part that needs re-architecting to
   **single tiles**: per-tile Cellpose segmentation → per-tile persistent-label tracking →
   MASKTRACKED → largest-connected-component measurement per (well, tile, timepoint, track_id).
   Blocked historically by `gurobipy` (absent from the SIF); the gurobi-free path is to run the
   montage tracker's scipy machinery per single tile.

**hevo-pmsG-1 adaptations** (our data ≠ Hal3/Hal4): stim is **whole-well dose by column**
(500/1000/5000/10000 ms), not per-tile DMD — so `stim` comes from `welldata`/`dosagedata`, not
the `TIER_IDX`/`tiers()` DMD map. Conditions are **cell-line × dose**, not drug-vs-DMSO — the
`km_ci`/`lr`/`bh` machinery is reused but the comparison structure is adapted. Threshold 0.025 is
Hal-specific and **must be re-derived** for our imaging conditions (§3 of the rules; the first run
stops at Plot 1 for user eyeball).

# XDP10 Cellpose-SAM cell_data (neurite soma recipe, Galaxy BG intensities)

The neurite analysis found many more somas than the original Galaxy segmentation. That segmentation was
Cellpose-SAM plus an intensity clean gate, but the neurite run saved masks for only one timepoint per experiment.
MRID needs every timepoint. This folder holds XDP10 segmented at **all 10 timepoints** with the same recipe, written
as a Galaxy-format `cell_data.csv`. It is meant to be run through the S.A.-threshold MRID pipeline
(`analyses/xdp_rgedi_rework`, run 3).

## Method

1. **Masks.** For each raw FITC tile (2048x2048, 16 sites per well), run `percell_integrate.segment_somas` as
   committed at HEAD. This is the exact version (sha256 `bf13c9f6...`) recorded in the XDP10 neurite run's
   `provenance.json`. It does:
   - Cellpose-SAM: diameter 25, flow 0.6, cellprob -1.0, bf16 off;
   - the MAD intensity clean gate, k = 2.0;
   - the size floor max(300, 0.6 x median area).
   At T7 the neurite run's cached masks (`neurite-rework/XDP10/cache`) are reused unchanged.
2. **Intensities.** Measured on the matching Galaxy background-corrected tiles
   (`XDP10/GXYTMP/BackgroundCorrected/<well>/..._BGs.tif`). A raw tile and its BG tile share one pixel grid, so
   each tile mask applies 1:1. There is no montage or alignment step and no tracking.
3. **`cell_data.csv`.** Galaxy legacy long format, 2 rows per cell (MeasurementTag FITC / RFP). The rows share
   `ObjectLabelsFound` within a well and timepoint. The ratio is computed downstream by MRID's own `mrid_ratio()`
   (RFP mean / FITC mean). Cells with a BG FITC mean of 0 are dropped.

## Scope

- **Wells (120).** Rows D-G and J-M (lines 33114.C, 33113.2I, 33109.2B, 33363.D):
  - untreated: cols 8-21;
  - S.A.: col 22, dosed T8 to T9.
- **Excluded.** siRNA cols 2-7, and 33362.C (rows H-I, out of MRID scope).
- **Timepoints:** T1-T10.

## Layout

| Path | Contents |
|---|---|
| `manifest.csv` | (well, timepoint) items with raw and BG tile folders |
| `masks/<well>_<tp>.npz` | `soma_labels`: 8192x8192 montage, 4x4 row-major, same layout as the neurite cache; plus `meta` |
| `percell/<well>_<tp>.csv.gz` | per cell: area, centroid, BG FITC/RFP mean/median/min/max/std, raw FITC mean, zero-pixel fraction |
| `cell_data.csv` | Galaxy-format input for MRID |
| `qc/preflight.csv` | tile counts per item |
| `qc/timing_*.csv` | per-montage timing |
| `qc/cache_compare_*.csv` | fresh vs neurite-cache agreement at T7 |
| `qc/counts_well_tp.csv` | cells per well x tp, Cellpose-SAM vs Galaxy |
| `qc/dropped_zero_fitc.csv` | cells dropped for BG FITC mean 0 |
| `qc/overlays/` | outlines on BG FITC (pilot wells) |
| `pilot/` | pilot (D08, D22) cell_data and QC |
| `provenance/` | one JSON per job: git commit, code sha256, container versions |

Code: `analyses/xdp_rgedi_rework/cpsam/` (`cpsam_tiles.py`, `build_celldata.py`, `build_sbatch.py`).

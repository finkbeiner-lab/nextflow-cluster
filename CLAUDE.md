# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Nextflow-based image analysis pipeline for high-content microscopy data at Gladstone/Finkbeiner Lab. Runs on an HPC cluster with Slurm scheduling and Singularity (Apptainer) containers. All modules read/write to a PostgreSQL database (`galaxy` on `fb-postgres01.gladstone.internal`).

## Running the Pipeline

```bash
# Submit to Slurm (the standard way to run)
sbatch run.sh

# Or run directly (still uses Slurm as the executor via nextflow.config)
nextflow run pipeline.nf -with-apptainer nextflow-cluster.sif -c finkbeiner.config
```

Before running, edit `finkbeiner.config` to set:
- Which modules to enable (`DO_*` flags)
- Experiment name, wells, timepoints, channels
- Module-specific parameters

The `run.sh` script validates the config via `bin/validate_config.py` before launching Nextflow.

## Architecture

### Nextflow Layer (DSL 2)
- **`pipeline.nf`** - Main workflow orchestrator. Reads params, creates channels, and conditionally runs processes based on `DO_*` flags in config. Handles dependency ordering between steps.
- **`modules.nf`** - Process definitions. Each process invokes a Python script from `bin/` inside the Singularity container. Processes communicate through the database, not through Nextflow file channels.
- **`finkbeiner.config`** - User-editable parameters (experiment-specific). This is the primary file users modify.
- **`nextflow.config`** - Cluster/executor configuration (Slurm, container path, resource defaults). Rarely edited.

### Key Workflow Patterns
- Individual modules can be toggled independently via `DO_*` boolean flags
- Two bundled workflows exist for common use cases:
  - `DO_STD_WORKFLOW` / `DO_BUNDLED_STD_WORKFLOW`: MONTAGE → ALIGN_MONTAGE_DFT → SEGMENTATION → TRACKING → OVERLAY (per well)
  - `DO_STD_WORKFLOW_IXM`: MONTAGE → SEGMENTATION → TRACKING → OVERLAY (per well, no alignment)
- Wells are processed in parallel (one Slurm job per well in bundled workflows)
- `INTENSITY` uses `each` keyword to process target channels in parallel
- Process dependencies are managed via `ready` flag channels (not file-based)

### Python Modules (`bin/`)
All executable Python scripts live in `bin/`. The bundled workflows (IXM and STD) use only the **core 11 files** listed below. All other scripts in `bin/` are for standalone modules or legacy use.

#### Core files (used by bundled workflows)
| File | Purpose |
|---|---|
| **`sql.py`** | Low-level PostgreSQL interface (SQLAlchemy). Imported as `from sql import Database`. Manages all table creation, CRUD operations, and schema reflection. |
| **`db_util.py`** | Database operations wrapper. `Ops` class queries tiledata and filters by wells/channels/timepoints using the toggle system. |
| **`utils.py`** | Shared utilities: natural sort, file listing, directory creation, well/timepoint range expansion (`get_iter_from_user`). |
| **`validate_config.py`** | Config validation (called by `run.sh` before Nextflow launch). Checks for whitespace issues, backslashes, malformed assignments. |
| **`montage.py`** | Stitches tiles into well montages. Supports standard (left-to-right) and legacy/snake tile patterns. Batch DB updates for performance. |
| **`align_montage_dft.py`** | Two-pass DFT-based montage alignment: computes shifts on morphology channel via phase cross-correlation (with DFT fallback), then applies shifts to all other channels. |
| **`segmentation_montage.py`** | Threshold-based cell segmentation on montaged images (sd_from_mean, triangle, minimum, yen, etc.). Parallel tile processing with ThreadPoolExecutor. |
| **`segmentation_helper_montage.py`** | Helper functions for segmentation: mask I/O (`save_mask`), numpy type conversion, and batch celldata/intensitycelldata DB operations. |
| **`tracking_montage.py`** | Cell tracking on montaged masks. Supports overlap, proximity, and motion (Hungarian algorithm + linear prediction) tracking methods. Parallel contour processing. |
| **`overlay_montage.py`** | Overlays tracked cell IDs as text with leader lines on montaged images. Supports filtering by cell ID set or stable CSV. Parallel timepoint processing. |
| **`__init__.py`** | Package init (empty). |

#### Other modules (standalone / not used by bundled workflows)
- **`segmentation.py`** / **`tracking.py`** / **`overlay.py`** - Tile-level (non-montage) equivalents
- **`align_tiles_dft.py`** - Tile-level DFT alignment
- **`cellpose_segmentation.py`** - ML-based cell segmentation using Cellpose (GPU)
- **`intensity.py`** - Projects morphology masks onto other channels, calculates intensity stats
- **`register_experiment.py`** - Registers new experiments from IXM/Robo raw data
- **`crop.py`** / **`crop_mask.py`** - Object cropping from raw/mask images
- **`normalization.py`** - Background correction (subtraction/division)
- **`cnn.py`** - CNN training/inference
- **`get_csvs.py`** - CSV export from database
- **`plate_montage.py`** - Plate-level montage (all wells)
- **`update_path.py`** - Database path migration
- **`mask_to_masktracked.py`** - Copies T0 masks to tracked mask paths

#### Code style conventions
All core Python files follow these conventions:
- **Docstrings**: Google style (`Args:`, `Returns:`, `Raises:`)
- **Type hints**: On all function/method signatures and return types
- **Imports**: Grouped as standard library → third-party → local, no unused imports
- **Logging**: Module-level logger with file handler writing to `./finkbeiner_logs/`

### Container
- **`nextflow-cluster.def`** - Singularity definition file (Ubuntu 22.04, Python 3.9, OpenCV, Cellpose, Gurobi, R, Java 17).
- **`Dockerfile`** - Docker equivalent (PyTorch base, used for local dev).
- The built container is at `nextflow-cluster.sif` in the repo root.

### Database Schema
PostgreSQL tables form a hierarchy: `experimentdata` → `welldata` → `tiledata` → `celldata` → `intensitycelldata`. Also: `channeldata`, `dosagedata`, `punctadata`, `intensitypunctadata`, `cropdata`, `modeldata`, `modelcropdata`. All tables use UUID primary keys. Foreign keys cascade on delete.

## Well/Timepoint/Channel Selection

Parameters use a toggle system:
- `chosen_wells` = `'all'` or comma-separated (e.g., `'A1,A2,B3'`)
- `wells_toggle` = `'include'` or `'exclude'`
- Same pattern for `chosen_timepoints` (supports ranges like `'T0-T7'`) and `chosen_channels`

## Container Bind Mounts

The Singularity container binds:
- `$HOME/nextflow-cluster/bin:/app` - Python scripts
- `/gladstone/finkbeiner:/gladstone/finkbeiner:rw` - Data filesystem
- `--nv` flag for GPU passthrough

## Slurm Configuration

- Executor: Slurm, queue: `galaxy`
- Default: 1 CPU, 15GB RAM, 600h per task
- Bundled workflows: 4-17 CPUs, 20GB RAM, 8h per well
- `maxForks = 22` concurrent jobs

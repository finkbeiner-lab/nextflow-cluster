# Neurite DL segmentation (Track B — "DL, built now")

Offline deep-learning training harness for **2D neurite segmentation on faint
fluorescence** (RGEDI / FITC = EGFP fill), plus a pipeline-facing inference
stub. It replaces the CellProfiler / Frangi-vesselness detector that fragments
thin, low-contrast processes.

This directory lives **outside** the container `bin/` path on purpose: it is a
standalone training package that runs on a workstation/GPU, not inside the
Nextflow pipeline. The only pipeline-facing file is `bin/neurite_model.py` (the
inference stub another track wires in).

> **Status.** This is scaffolding proven end-to-end on CPU with the current
> **two** annotated fields (`C03_t1`, `I03_t1`). It is not a trained production
> model. Labels come from **Track 0's** SNT annotation set; extend beyond the
> two fields and run full training on GPU before using in anger.

---

## Why these choices

* **soft-clDice loss** (`losses.py`; Shit et al., CVPR 2021). Plain Dice/BCE
  reward pixel overlap, so a network can score well while leaving neurites
  broken — the exact failure of the baseline. clDice compares each mask against
  the *soft skeleton* of the other (differentiable skeletonization via iterated
  min/max-pooling), so a one-pixel gap in a process is penalised heavily. That
  targets **topology/connectivity**, which is what per-cell neurite length and
  the F1 metric care about. We combine it with Dice (region overlap under heavy
  class imbalance) and BCE (per-pixel calibration).
* **tol=3 skeleton F1** (`metrics.py`). The eval metric is a **verbatim reuse**
  of the distance-transform precision/recall/F1 at pixel tolerance 3 from the
  benchmark scorer (`scratchpad/score_all.py`), so DL numbers are directly
  comparable to the Frangi and CellProfiler benchmarks.
* **Percentile normalization (1–99.5)** everywhere (`dataset.py`,
  inference) — the same window the tracer/scorer use, so the model sees the
  intensities the benchmark was built on.
* **Split by field** in training to avoid crop-level train/val leakage.

## Files

| File | Purpose |
|---|---|
| `rasterize_traces.py` | Parse SNT `.traces` (gzipped XML) → rasterize polylines → skeleton, dilate to a neurite-width target, save `(img, mask, skel)` `.npy`. Reuses the scorer's exact parsing. |
| `losses.py` | Self-contained PyTorch soft-clDice + Dice + BCE (`SoftDiceClDiceBCELoss`). Optional `monai` path. `python losses.py` runs a sanity check. |
| `model.py` | Compact configurable 2D U-Net, 1-channel in, 1-logit out. CPU-runnable. |
| `dataset.py` | `NeuriteDataset`: percentile-normalized, random foreground-biased crops, flips/rotations/intensity jitter. Built for very few fields. |
| `metrics.py` | tol=3 distance-transform skeleton F1 (identical to the benchmark) + threshold sweep. |
| `train.py` | Training loop: U-Net + combined loss + Adam, per-epoch loss & val-F1, best checkpoint. `--dry-run` proves the loop on CPU. |
| `infer.py` | Checkpoint + image → probability map (handles normalization + size padding). |
| `denoise_n2v.py` | Thin Noise2Void wrapper (careamics/n2v) with graceful passthrough no-op when uninstalled. Pairs with Track A's `--denoise n2v`. |
| `requirements-ml.txt` | Dependencies. |
| `../../bin/neurite_model.py` | **Pipeline-facing** inference stub (probability map only; no DB). |

## Workflow

```bash
# 0. (once) install deps into a dedicated env
python -m venv .venv-neurite && . .venv-neurite/bin/activate
pip install -r ml/neurite/requirements-ml.txt

# 1. rasterize SNT traces into (image, mask) training pairs
python ml/neurite/rasterize_traces.py \
    --traces /Users/aholub/Desktop/neurite-annotation/traces \
    --tiles  /Users/aholub/Desktop/neurite-annotation/tiles \
    --out    ml/neurite/data --dilation-radius 2

# 2a. CPU dry run (proves the harness on the 2 existing fields)
python ml/neurite/train.py --dry-run --data ml/neurite/data --out ml/neurite/runs/dryrun

# 2b. real training (later, GPU, once Track 0 expands the annotation set)
python ml/neurite/train.py --data <big_data_dir> --out runs/full \
    --device cuda --epochs 200 --crop 512 --depth 4 --base 32

# 3. inference → probability map
python ml/neurite/infer.py --checkpoint runs/full/best.pt \
    --image tile.tif --out prob.tif --threshold 0.5
```

### Optional: Noise2Void denoising

```bash
python ml/neurite/denoise_n2v.py train --tiles <dir> --checkpoint n2v.ckpt
python ml/neurite/denoise_n2v.py apply --image tile.tif --checkpoint n2v.ckpt --out den.tif
```
With no N2V backend installed, `apply` is a safe passthrough (no-op) so Track A's
`--denoise n2v` hook degrades gracefully.

## Dry-run result (CPU, 2 fields)

`train.py --dry-run` trains on `C03_t1` and validates on the held-out `I03_t1`
(a genuine cross-field split, not overfit-on-self). Depth-3 / base-8 U-Net
(29,481 params), 8 epochs, ~4 min on CPU:

```
epoch  train_loss   val_F1   thr
    1      1.3634    0.000  0.30
    3      1.3291    0.043  0.50
    6      1.2698    0.103  0.60
    8      1.2322    0.237  0.60   *best
```

Training loss decreases monotonically and held-out F1 climbs to 0.237 — the
point is only to prove the harness runs end to end and produces a real loss
curve + F1 number, **not** a usable model. A production model needs the larger
SNT annotation set + longer GPU training and should be benchmarked against the
Frangi baseline with the same tol=3 F1.

## `bin/neurite_model.py` interface (for the wiring track)

`bin/neurite_model.py` produces a **neurite probability map only**. It does NOT
touch the database, skeletonization, per-soma attribution, or `neuritecelldata`
— that shared backend stays in `bin/neurite.py`. To swap the DL detector in for
the Frangi step there:

```python
from neurite_model import predict_neurite_probmap   # or segment_neurites

prob = predict_neurite_probmap(morphology_image, checkpoint_path)  # float32 (H,W) 0-1
mask = prob >= threshold
# ... existing skeletonize → attribute-to-soma → write neuritecelldata unchanged ...
```

Guarantees:

* Input: single-channel 2D `numpy.ndarray` (H, W), any real dtype (3D → channel 0).
* `predict_neurite_probmap` → `float32` (H, W) in [0, 1], **exactly** the input
  HxW (internal reflect-padding cropped back off).
* 1–99.5 percentile normalization applied internally — pass the RAW image.
* No DB, no global state, no file writes; checkpoint is cached across tiles.
* `torch` is imported lazily: the module imports cleanly even where torch is
  absent (so the Frangi path in a torch-less container is unaffected); a clear
  `ImportError` is raised only when a prediction is actually requested.
* Checkpoint resolves from the `checkpoint=` arg, else `NEURITE_CHECKPOINT`, else
  `ml/neurite/runs/dryrun/best.pt`. `NEURITE_ML_DIR` overrides the package path.

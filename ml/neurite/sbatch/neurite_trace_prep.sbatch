#!/bin/bash
#SBATCH --partition=galaxy
#SBATCH --job-name=nrt_trace_prep
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=nrt_trace_prep_%j.out
#
# Neurite model training data, CPU only. Two steps, run one at a time:
#
#   STEP=crops      pick one neurite-rich crop per tile for hand tracing
#                   reads  TILES/*.tif   writes TRACE_DIR/<name>__ROI.tif (+ _context.tif)
#   (then trace every __ROI.tif in Fiji > SNT and save <name>.traces in TRACE_DIR)
#   STEP=rasterize  turn the .traces into training pairs
#                   reads  TRACE_DIR/*.traces + *__ROI.tif   writes DATA/<name>_{img,mask,skel}.npy + _qa.tif
#
# Submit from a folder in your cluster home:
#   sbatch --export=ALL,STEP=crops,TILES=/path/tiles,TRACE_DIR=/gladstone/finkbeiner/<lab>/<project>/tracing \
#          ~/nextflow-cluster/ml/neurite/sbatch/neurite_trace_prep.sbatch
#   sbatch --export=ALL,STEP=rasterize,TRACE_DIR=...,DATA=/gladstone/finkbeiner/<lab>/<project>/training-data \
#          ~/nextflow-cluster/ml/neurite/sbatch/neurite_trace_prep.sbatch
# Optional: ROI=640 (crop size, px).
set -euo pipefail

REPO=${REPO:-$HOME/nextflow-cluster}
SIF=${SIF:-/gladstone/finkbeiner/kaye/XDP/dev-nextflow-modules/nextflow-cluster.sif}
: "${STEP:?Set STEP=crops or STEP=rasterize}"
: "${TRACE_DIR:?Set TRACE_DIR=<folder for crops and .traces files>}"
[ -r "$SIF" ] || { echo "ERROR: cannot read the container: $SIF (pass SIF=/path/to.sif)"; exit 1; }
echo "code: $(git -C "$REPO" rev-parse --abbrev-ref HEAD) @ $(git -C "$REPO" rev-parse --short HEAD)"
run() { apptainer exec -B /gladstone/finkbeiner:/gladstone/finkbeiner "$SIF" python "$@"; }

case "$STEP" in
  crops)
    : "${TILES:?Set TILES=<folder of whole tiles (*.tif) to crop from>}"
    run "$REPO/ml/neurite/pick_roi_crops.py" --in "$TILES" --out "$TRACE_DIR" --roi "${ROI:-640}" ;;
  rasterize)
    : "${DATA:?Set DATA=<folder for the training pairs>}"
    run "$REPO/ml/neurite/rasterize_traces.py" --traces "$TRACE_DIR" --tiles "$TRACE_DIR" \
        --out "$DATA" --dilation-radius 2 ;;
  *) echo "ERROR: STEP must be crops or rasterize, got '$STEP'"; exit 1 ;;
esac
echo DONE

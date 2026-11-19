#!/bin/bash
#SBATCH --partition=galaxy
#SBATCH --job-name=nrt_train
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=nrt_train_%j.out
#
# Train the clDice U-Net neurite model on hand-traced crops, then score it
# (and a classical Frangi filter) on the held-out crops and write QA images.
# Settings = the production model (run1, held-out F1 0.645).
#
# DATA     training pairs from neurite_trace_prep.sbatch STEP=rasterize
# OUT      new folder for the model (best.pt, *_evalqa.tif)
# VAL      held-out crops: parts of their names joined by ':', e.g. H15:I07
#          (':' because sbatch --export splits on commas). Pick 2 or more crops
#          from different wells/lines; they are not used for training.
# INIT     optional: start from an existing checkpoint (fine-tune) instead of from zero,
#          e.g. INIT=/gladstone/finkbeiner/kaye/XDP/dev-nextflow-modules/neurite-model-runs/run1/best.pt
#
#   sbatch --export=ALL,DATA=...,OUT=/gladstone/finkbeiner/<lab>/<project>/neurite-model/run2,VAL=H15:I07 \
#          ~/nextflow-cluster/ml/neurite/sbatch/neurite_train.sbatch
set -euo pipefail

REPO=${REPO:-$HOME/nextflow-cluster}
SIF=${SIF:-/gladstone/finkbeiner/kaye/XDP/dev-nextflow-modules/nextflow-cluster.sif}
EPOCHS=${EPOCHS:-200}
: "${DATA:?Set DATA=<training pairs folder>}"
: "${OUT:?Set OUT=<new folder for the model>}"
: "${VAL:?Set VAL=<held-out crop names, e.g. H15:I07>}"
VAL=${VAL//:/,}
[ -r "$SIF" ] || { echo "ERROR: cannot read the container: $SIF (pass SIF=/path/to.sif)"; exit 1; }
[ ! -e "$OUT/best.pt" ] || { echo "ERROR: $OUT/best.pt exists; use a new OUT"; exit 1; }
echo "code: $(git -C "$REPO" rev-parse --abbrev-ref HEAD) @ $(git -C "$REPO" rev-parse --short HEAD)"
mkdir -p "$OUT"
INIT_ARG=()
[ -n "${INIT:-}" ] && INIT_ARG=(--init "$INIT")

cd "$REPO/ml/neurite"
apptainer exec --nv -B /gladstone/finkbeiner:/gladstone/finkbeiner "$SIF" \
  python train.py --data "$DATA" --out "$OUT" --device cuda --epochs "$EPOCHS" --batch 8 \
    --crop 256 --iters-per-epoch 200 --lr 1e-3 --depth 4 --base 32 \
    --pos-weight 5 --cldice-iters 10 --val-stems "$VAL" --seed 0 ${INIT_ARG[@]+"${INIT_ARG[@]}"}
apptainer exec --nv -B /gladstone/finkbeiner:/gladstone/finkbeiner "$SIF" \
  python eval_qa.py --data "$DATA" --checkpoint "$OUT/best.pt" --out "$OUT" \
    --val-stems "$VAL" --device cuda
echo "DONE: model and QA images in $OUT"

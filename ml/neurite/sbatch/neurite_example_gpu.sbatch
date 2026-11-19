#!/bin/bash
#SBATCH --partition=galaxy
#SBATCH --job-name=nrt_example_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=nrt_example_gpu_%j.out
#
# Neurite module TEST 2 of 2: run the example from the RAW images on a GPU.
# This is the full path you use for new data: montage -> Cellpose-SAM somas ->
# clDice neurites -> per-cell measurements. About 4-5 minutes per well after
# the job starts (2 wells). The GPU queue can add hours of waiting.
#
# The numbers should be very close to the expected output (the check allows
# 5% per column). The job also writes QC overlay images that you can open in Fiji.
#
# Submit from a folder in your cluster home (the log is written there):
#   sbatch --export=ALL,OUT=/gladstone/finkbeiner/<your-lab-folder>/neurite-example-test \
#          ~/nextflow-cluster/ml/neurite/sbatch/neurite_example_gpu.sbatch
set -euo pipefail

REPO=${REPO:-$HOME/nextflow-cluster}
SIF=${SIF:-/gladstone/finkbeiner/kaye/XDP/dev-nextflow-modules/nextflow-cluster.sif}
CKPT=${CKPT:-/gladstone/finkbeiner/kaye/XDP/dev-nextflow-modules/neurite-model-runs/run1/best.pt}
KIT=${KIT:-/gladstone/finkbeiner/kaye/XDP/neurite-rework/example-test}
RAW_ROOT=${RAW_ROOT:-/gladstone/finkbeiner/robodata/IXM4Galaxy/Austin}
: "${OUT:?Set OUT=<a folder in your lab project space>, e.g. sbatch --export=ALL,OUT=... }"
RUN_OUT="$OUT/gpu_from_raw"

if [ ! -r "$SIF" ]; then
  echo "ERROR: cannot read the container: $SIF"
  echo "  The lab-shared container for this module is"
  echo "  /gladstone/finkbeiner/kaye/XDP/dev-nextflow-modules/nextflow-cluster.sif"
  echo "  If it exists, ask IT to add you to the finkbeinerlab group. Or pass SIF=/path/to.sif."
  exit 1
fi
echo "container: $SIF"
for f in "$REPO/ml/neurite/montage_percell.py" "$SIF" "$CKPT" "$KIT/manifest.csv"; do
  [ -e "$f" ] || { echo "ERROR: not found: $f"; exit 1; }
done
echo "code: $(git -C "$REPO" rev-parse --abbrev-ref HEAD) @ $(git -C "$REPO" rev-parse --short HEAD)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# A fresh, empty cache: this test must compute everything from the raw tiles.
mkdir -p "$RUN_OUT/cache"

apptainer exec --nv -B /gladstone/finkbeiner:/gladstone/finkbeiner "$SIF" \
  python "$REPO/ml/neurite/montage_percell.py" \
    --manifest "$KIT/manifest.csv" --raw-root "$RAW_ROOT" \
    --out-dir "$RUN_OUT" --cache-dir "$RUN_OUT/cache" \
    --checkpoint "$CKPT" --device cuda \
    --threshold 0.5 --diameter 25 --flow 0.6 --cellprob -1.0 --clean-k 2.0 \
    --flatten-size 128 --seam-band 6 \
    --root-px 6 --min-radial 0.8 --cross-cos -0.7 \
    --run-label example_gpu

echo
apptainer exec -B /gladstone/finkbeiner:/gladstone/finkbeiner "$SIF" \
  python "$REPO/ml/neurite/compare_runs.py" \
    --expected "$KIT/expected" --got "$RUN_OUT/latest" --rtol 0.05

#!/bin/bash
#SBATCH --partition=galaxy
#SBATCH --job-name=nrt_example_cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=nrt_example_cpu_%j.out
#
# Neurite module TEST 1 of 2: recompute the example from cached segmentation.
# CPU only (no GPU), about 5 minutes. The result must be IDENTICAL to the
# expected output. If it is, the code checkout, container and model are correct.
#
# Submit from a folder in your cluster home (the log is written there):
#   sbatch --export=ALL,OUT=/gladstone/finkbeiner/<your-lab-folder>/neurite-example-test \
#          ~/nextflow-cluster/ml/neurite/sbatch/neurite_example_cpu.sbatch
#
# OUT must be in a lab project folder on the HIVE, not in your cluster home.
set -euo pipefail

REPO=${REPO:-$HOME/nextflow-cluster}
SIF=${SIF:-/gladstone/finkbeiner/kaye/XDP/dev-nextflow-modules/nextflow-cluster.sif}
CKPT=${CKPT:-/gladstone/finkbeiner/kaye/XDP/dev-nextflow-modules/neurite-model-runs/run1/best.pt}
KIT=${KIT:-/gladstone/finkbeiner/kaye/XDP/neurite-rework/example-test}
RAW_ROOT=${RAW_ROOT:-/gladstone/finkbeiner/robodata/IXM4Galaxy/Austin}
: "${OUT:?Set OUT=<a folder in your lab project space>, e.g. sbatch --export=ALL,OUT=... }"
RUN_OUT="$OUT/cpu_from_cache"

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

# Copy the example cache, so this test reads only your own copy.
mkdir -p "$RUN_OUT/cache"
cp -n "$KIT"/cache/*.npz "$RUN_OUT/cache/"

apptainer exec -B /gladstone/finkbeiner:/gladstone/finkbeiner "$SIF" \
  python "$REPO/ml/neurite/montage_percell.py" \
    --manifest "$KIT/manifest.csv" --raw-root "$RAW_ROOT" \
    --out-dir "$RUN_OUT" --cache-dir "$RUN_OUT/cache" \
    --checkpoint "$CKPT" --device cpu \
    --threshold 0.5 --diameter 25 --flow 0.6 --cellprob -1.0 --clean-k 2.0 \
    --flatten-size 128 --seam-band 6 \
    --root-px 6 --min-radial 0.8 --cross-cos -0.7 \
    --run-label example_cpu

echo
apptainer exec -B /gladstone/finkbeiner:/gladstone/finkbeiner "$SIF" \
  python "$REPO/ml/neurite/compare_runs.py" \
    --expected "$KIT/expected" --got "$RUN_OUT/latest" --exact

#!/bin/bash
#SBATCH --partition=galaxy
#SBATCH --job-name=nrt_run
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=05:00:00
#SBATCH --output=nrt_run_%j.out
#
# Neurite module: run your own wells. Template.
#
# 1. Make a manifest CSV (one row per well), for example:
#      experiment,well,timepoint,cell_line,genotype,raw_folder
#      MYEXP,B02,T5,lineA,CTR,MYEXP-RGEDI
#    The raw tiles must be at  RAW_ROOT/<raw_folder>/<well>/  with IXM names
#    like  ..._<timepoint>_..._<well>_<site>_FITC_....tif  (16 sites, 4x4 grid).
# 2. Set --time: about 5 minutes per well on the GPU, plus margin.
#    A job that runs out of time can be submitted again: wells already in
#    the cache are skipped quickly.
# 3. Submit from a folder in your cluster home:
#      sbatch --export=ALL,MANIFEST=/path/manifest.csv,RAW_ROOT=/gladstone/finkbeiner/robodata/...,OUT=/gladstone/finkbeiner/<lab-folder>/<project>/neurite,LABEL=first \
#             ~/nextflow-cluster/ml/neurite/sbatch/neurite_run.sbatch
#
# Outputs: OUT/runs/<timestamp>_<LABEL>/{percell.csv,perwell.csv,params.json,*_percell.tif}
#          OUT/latest -> the newest run;  OUT/cache/ = saved segmentation (re-use it).
# To change only post-segmentation settings later (threshold, attribution), run
# again with the same OUT and a new LABEL. Use DEVICE=cpu and remove the --gres line:
# every well comes from the cache, so no GPU is needed.
set -euo pipefail

REPO=${REPO:-$HOME/nextflow-cluster}
SIF=${SIF:-/gladstone/finkbeiner/kaye/XDP/dev-nextflow-modules/nextflow-cluster.sif}
CKPT=${CKPT:-/gladstone/finkbeiner/kaye/XDP/dev-nextflow-modules/neurite-model-runs/run1/best.pt}
DEVICE=${DEVICE:-cuda}
LABEL=${LABEL:-run}
: "${MANIFEST:?Set MANIFEST=<your manifest.csv>}"
: "${RAW_ROOT:?Set RAW_ROOT=<folder that holds raw_folder/well/ tiles>}"
: "${OUT:?Set OUT=<a folder in your lab project space>}"

for f in "$REPO/ml/neurite/montage_percell.py" "$SIF" "$CKPT" "$MANIFEST"; do
  [ -e "$f" ] || { echo "ERROR: not found: $f"; exit 1; }
done
echo "code: $(git -C "$REPO" rev-parse --abbrev-ref HEAD) @ $(git -C "$REPO" rev-parse --short HEAD)"
mkdir -p "$OUT/cache"

# Settings below = the locked recipe used for the XDP results (Oct 2026).
apptainer exec --nv -B /gladstone/finkbeiner:/gladstone/finkbeiner "$SIF" \
  python "$REPO/ml/neurite/montage_percell.py" \
    --manifest "$MANIFEST" --raw-root "$RAW_ROOT" \
    --out-dir "$OUT" --cache-dir "$OUT/cache" \
    --checkpoint "$CKPT" --device "$DEVICE" \
    --threshold 0.5 --diameter 25 --flow 0.6 --cellprob -1.0 --clean-k 2.0 \
    --flatten-size 128 --seam-band 6 \
    --root-px 6 --min-radial 0.8 --cross-cos -0.7 \
    --run-label "$LABEL"

echo "DONE: $(readlink -f "$OUT/latest")"

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
#    Use the timepoint text exactly as it appears in your file names.
# 2. Describe your images (any microscope). The defaults are the ImageXpress (IXM)
#    layout:  RAW_ROOT/<raw_folder>/<well>/..._<timepoint>_..._<well>_<site>_FITC_....tif,
#    16 sites in a 4x4 grid, read left to right, no overlap. For other data set:
#      WELL_DIR      folder of one well under RAW_ROOT   (default {raw_folder}/{well})
#      TILE_PATTERN  file names of the cell-fill channel (default *_{tp}_*_{well}_{site}_FITC_*.tif)
#                    {well} {tp} {experiment} come from the manifest, {site} = tile number,
#                    * = anything. No {site} + GRID=1x1 = one stitched image per well.
#      GRID          tiles per well, ROWSxCOLUMNS        (default 4x4)
#      TILE_ORDER    standard | serpentine | legacy      (default standard)
#      OVERLAP_PX    px shared by neighbouring tiles     (default 0)
#    Check the pattern first on the dev node:  ls <one well folder>  and compare.
# 3. Set --time: about 5 minutes per well on the GPU, plus margin.
#    A job that runs out of time can be submitted again: wells already in
#    the cache are skipped quickly.
# 4. Submit from a folder in your cluster home:
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
WELL_DIR=${WELL_DIR:-'{raw_folder}/{well}'}
TILE_PATTERN=${TILE_PATTERN:-'*_{tp}_*_{well}_{site}_FITC_*.tif'}
GRID=${GRID:-4x4}
TILE_ORDER=${TILE_ORDER:-standard}
OVERLAP_PX=${OVERLAP_PX:-0}
: "${MANIFEST:?Set MANIFEST=<your manifest.csv>}"
: "${RAW_ROOT:?Set RAW_ROOT=<folder that holds raw_folder/well/ tiles>}"
: "${OUT:?Set OUT=<a folder in your lab project space>}"

if [ ! -r "$SIF" ]; then
  echo "ERROR: cannot read the container: $SIF"
  echo "  The lab-shared container for this module is"
  echo "  /gladstone/finkbeiner/kaye/XDP/dev-nextflow-modules/nextflow-cluster.sif"
  echo "  If it exists, ask IT to add you to the finkbeinerlab group. Or pass SIF=/path/to.sif."
  exit 1
fi
echo "container: $SIF"
for f in "$REPO/ml/neurite/montage_percell.py" "$SIF" "$CKPT" "$MANIFEST"; do
  [ -e "$f" ] || { echo "ERROR: not found: $f"; exit 1; }
done
echo "code: $(git -C "$REPO" rev-parse --abbrev-ref HEAD) @ $(git -C "$REPO" rev-parse --short HEAD)"
echo "input: $RAW_ROOT/$WELL_DIR/$TILE_PATTERN  grid $GRID $TILE_ORDER, overlap $OVERLAP_PX px"
mkdir -p "$OUT/cache"

# Settings below = the locked recipe used for the XDP results (Oct 2026).
apptainer exec --nv -B /gladstone/finkbeiner:/gladstone/finkbeiner "$SIF" \
  python "$REPO/ml/neurite/montage_percell.py" \
    --manifest "$MANIFEST" --raw-root "$RAW_ROOT" \
    --well-dir "$WELL_DIR" --tile-pattern "$TILE_PATTERN" \
    --grid "$GRID" --tile-order "$TILE_ORDER" --overlap-px "$OVERLAP_PX" \
    --out-dir "$OUT" --cache-dir "$OUT/cache" \
    --checkpoint "$CKPT" --device "$DEVICE" \
    --threshold 0.5 --diameter 25 --flow 0.6 --cellprob -1.0 --clean-k 2.0 \
    --flatten-size 128 --seam-band 6 \
    --root-px 6 --min-radial 0.8 --cross-cos -0.7 \
    --run-label "$LABEL"

echo "DONE: $(readlink -f "$OUT/latest")"

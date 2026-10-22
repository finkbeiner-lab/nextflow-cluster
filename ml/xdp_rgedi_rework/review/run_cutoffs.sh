#!/usr/bin/env bash
# Rerun the live/dead cutoffs and MRID (steps 06-12) from the review package, then compare with the package.
# Usage: bash review/run_cutoffs.sh <package_dir> <new_cache_dir> [B]
#   B = bootstrap replicates for the cutoff intervals: a number (default 50, about 15 min in all), or "full" for
#       the settings of Austin's run (params.yaml: 1000 in step 06, 2000 in step 08; about 2.5 h).
#       The cutoffs themselves do not depend on B.
# Python and R: set XDP_RGEDI_PYTHON / XDP_RGEDI_RSCRIPT if python3 / Rscript on PATH are not the right ones.
set -euo pipefail
PKG=$(cd "${1:?package dir}" && pwd)
mkdir -p "${2:?new cache dir}"; CACHE=$(cd "$2" && pwd)
B=${3:-50}
if [ "$B" = full ]; then B=""; fi               # empty = the B of config/params.yaml
HERE=$(cd "$(dirname "$0")/.." && pwd)          # .../ml/xdp_rgedi_rework
BIN=$(cd "$HERE/../../bin" && pwd)              # .../bin (MRID R functions)
export XDP_RGEDI_PYTHON=${XDP_RGEDI_PYTHON:-$(command -v python3)}
export XDP_RGEDI_RSCRIPT=${XDP_RGEDI_RSCRIPT:-$(command -v Rscript)}
export XDP_RGEDI_CACHE=$CACHE XDP_RGEDI_RUN=rerun XDP_RGEDI_PARQUET=$CACHE/parquet
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
PY="$XDP_RGEDI_PYTHON -I -B"                # -B: no __pycache__ in the package (-I ignores PYTHONDONTWRITEBYTECODE)
OUT=$CACHE/out/rerun
cd "$HERE"
step() { echo; echo "=== $* ($(date +%H:%M:%S))"; }
step "build cache from package";  $PY review/setup_from_package.py "$PKG" "$CACHE"
step "06 S.A. fits";               $PY steps/06_fit_thresholds.py $B
step "06b buffer zones";           $PY steps/06b_window_cutoffs.py
step "08 transfer";                $PY steps/08_transfer.py $B
step "07 validity";                $PY steps/07_validity.py
step "11 MRID inputs";             $PY steps/11_mrid_inputs.py
step "12 MRID (lab R code)";       "$XDP_RGEDI_RSCRIPT" --vanilla steps/12_logodds.R "$OUT/11_mrid_inputs/input/W90" "$OUT/12_logodds/W90" "$BIN"
step "compare with package";       $PY review/compare_to_package.py "$PKG" "$OUT"

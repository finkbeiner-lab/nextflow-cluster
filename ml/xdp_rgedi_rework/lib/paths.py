"""Fixed locations for the XDP RGEDI S.A.-threshold rework.

Importing this module also puts the repo ``bin/`` folder on ``sys.path`` so the
shared helpers (``gmm1d``, ``sa_qc_ratio``) import cleanly under ``python -I``.
"""

import os
import sys
from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parents[1]          # ml/xdp_rgedi_rework
REPO_ROOT = PKG_ROOT.parents[1]                          # nextflow-cluster-master
REPO_BIN = REPO_ROOT / "bin"

CONFIG_DIR = PKG_ROOT / "config"
SHARE = Path("/Volumes/Finkbeiner-Kaye/XDP")
MRID_REWORK = SHARE / "MRID-rework"
OUT_ROOT = SHARE / "RGEDI-full-rework"

CACHE = Path("/Users/aholub/xdp_rgedi_cache")
RAW_CACHE = CACHE / "raw"
# Shared per-cell extracts. XDP_RGEDI_PARQUET points a run at its own folder (run 3 swaps XDP10's input
# and must not overwrite the parquet runs 1-2 read); unset = the shared folder.
PARQUET_CACHE = Path(os.environ.get("XDP_RGEDI_PARQUET", str(CACHE / "parquet")))
# Each analysis run has its own output folder (see out/RUNS.md). Select one with XDP_RGEDI_RUN; run 1 is read-only.
RUN = os.environ.get("XDP_RGEDI_RUN", "run2_ruleD-window_20261015")
LOCAL_OUT = CACHE / "out" / RUN    # steps write here first, then sync to OUT_ROOT / RUN
LOGS = CACHE / "logs" / RUN

PYTHON = "/Users/aholub/anaconda3/bin/python3"
RSCRIPT = "/usr/local/bin/Rscript"

for _p in (str(REPO_BIN), str(PKG_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

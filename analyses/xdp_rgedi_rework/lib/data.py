"""Per-cell data access: slim parquet extracts of the MRID per-cell ratio caches.

The cache CSVs (one RFP row per cell, GFP mean merged in) are read once from the
local raw copy, joined to the config well roles and timepoint hours, and saved
as ``parquet/<EXP>.parquet`` with only the columns the analysis needs.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from lib import cfg
from lib.paths import PARQUET_CACHE, RAW_CACHE

CACHE_COLS = {
    "Sci_WellID": "well_raw", "Sci_SampleID": "line_cache", "Drug": "drug_cache",
    "ObjectLabelsFound": "obj", "BlobArea": "area", "BlobCentroidX": "cx", "BlobCentroidY": "cy",
    "GFPPixelIntensityMean": "gfp", "PixelIntensityMean": "rfp", "Timepoint": "tp", "ratio": "ratio",
}


def staged_path(key: str) -> Path:
    staging = cfg.load_yaml("staging.yaml")
    return RAW_CACHE / staging["large"][key]


def attach_roles(df: pd.DataFrame, exp: str) -> pd.DataFrame:
    """Join config well roles (line, condition) and timepoint hours onto a per-cell frame."""
    roles = cfg.well_roles(exp)[["well", "row", "col", "line", "condition", "sirna_name", "in_scope"]]
    tps = cfg.read_timepoints(exp)
    out = df.merge(roles, on="well", how="left", validate="many_to_one")
    out = out.merge(tps, on="tp", how="left", validate="many_to_one")
    out.insert(0, "exp", exp)
    with np.errstate(divide="ignore", invalid="ignore"):
        out["log10_ratio"] = np.where(out["ratio"] > 0, np.log10(out["ratio"]), np.nan)
    return out


def extract_cache(exp: str, csv_path: Optional[Path] = None) -> pd.DataFrame:
    """Read an MRID per-cell cache CSV into the slim schema (roles attached)."""
    src = csv_path or staged_path(cfg.experiments()[exp]["source"]["key"])
    raw = pd.read_csv(src, usecols=list(CACHE_COLS), dtype={"Sci_WellID": str, "Sci_SampleID": str,
                                                            "Drug": str},
                      keep_default_na=False, na_values=["NA", "NaN", ""])
    raw = raw.rename(columns=CACHE_COLS)
    raw["well"] = raw["well_raw"].map(cfg.pad_well)
    raw["tp"] = raw["tp"].astype(int)
    for c in ("gfp", "rfp", "ratio", "area", "cx", "cy"):
        raw[c] = pd.to_numeric(raw[c], errors="coerce").astype("float64")
    raw["obj"] = raw["obj"].astype("int64")
    return attach_roles(raw.drop(columns=["well_raw"]), exp)


def parquet_path(exp: str) -> Path:
    return PARQUET_CACHE / f"{exp}.parquet"


def load(exp: str, columns=None, in_scope_only: bool = True) -> pd.DataFrame:
    """Load the slim per-cell table for one experiment (in-scope lines by default)."""
    df = pd.read_parquet(parquet_path(exp), columns=columns)
    if in_scope_only and "in_scope" in df.columns:
        df = df[df["in_scope"]].copy()
    return df

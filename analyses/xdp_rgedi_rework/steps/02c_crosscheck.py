"""Step 02c: prove the per-cell ratio step is reproducible and understood.

1. R reproduction: MRID mrid_ratio() re-run today on XDP8's raw cell_data.csv with the Sept
   arguments (timepoint 2 removed, Sept platemap) must equal the Sept XDP8 cache.
2. Independent Python port of mrid_ratio() (dedupe by largest BlobArea per plate/well/object/
   timepoint/channel; FITC-RFP inner join on timepoint_well_object; ratio = RFP mean / GFP mean;
   drop wells without cells at the first timepoint) must reproduce the R output for XDP8 and XDP10.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from lib import cfg, data  # noqa: E402
from lib.paths import LOCAL_OUT, RAW_CACHE  # noqa: E402
from lib.provenance import StepRecorder, sha256_file  # noqa: E402

STEP = "02c_crosscheck"
KEY = ["Timepoint", "Sci_WellID", "ObjectLabelsFound"]


def mrid_ratio_py(raw: pd.DataFrame, platemap: pd.DataFrame, tp_removal=()) -> pd.DataFrame:
    d = raw[~raw.Timepoint.isin(tp_removal)].copy()
    d = d.sort_values("BlobArea", ascending=False, kind="mergesort")
    d = d.drop_duplicates(["Sci_PlateID", "Sci_WellID", "ObjectLabelsFound", "Timepoint", "MeasurementTag"])
    fitc = d[d.MeasurementTag.str.contains("FITC|GFP")]
    rfp = d[d.MeasurementTag.str.contains("RFP")].merge(platemap, on="Sci_WellID")
    first = rfp.Timepoint.min()
    wells_t1 = set(rfp.loc[rfp.Timepoint == first, "Sci_WellID"])
    m = rfp.merge(fitc[KEY + ["PixelIntensityMean"]].rename(columns={"PixelIntensityMean": "GFPPixelIntensityMean"}),
                  on=KEY)
    m["ratio"] = m.PixelIntensityMean / m.GFPPixelIntensityMean
    return m[m.Sci_WellID.isin(wells_t1)]


def compare(a: pd.DataFrame, b: pd.DataFrame, cols) -> dict:
    a = a[KEY + cols].copy()
    b = b[KEY + cols].copy()
    for df in (a, b):
        df["Sci_WellID"] = df["Sci_WellID"].astype(str)
    m = a.merge(b, on=KEY, how="outer", suffixes=("_a", "_b"), indicator=True)
    both = m[m._merge == "both"]
    out = {"rows_a": len(a), "rows_b": len(b), "only_a": int((m._merge == "left_only").sum()),
           "only_b": int((m._merge == "right_only").sum())}
    for c in cols:
        x, y = both[f"{c}_a"].to_numpy(float), both[f"{c}_b"].to_numpy(float)
        out[f"max_rel_diff_{c}"] = float(np.nanmax(np.abs(x - y) / np.maximum(np.abs(y), 1e-12))) if len(x) else None
    return out


def main() -> None:
    rec = StepRecorder(STEP, LOCAL_OUT / STEP)
    staging = json.loads((RAW_CACHE / "staging_manifest.json").read_text())
    # 1. R reproduction of the Sept XDP8 cache.
    r_today = LOCAL_OUT / "02b_rebuild" / "XDP8_repro" / "mrid_celldata_ratio.csv"
    sept = data.staged_path("XDP8_cache")
    rec.add_input(r_today, sha256_file(r_today))
    rec.add_input(sept, staging["XDP8_cache"]["sha256"])
    same_bytes = sha256_file(r_today) == staging["XDP8_cache"]["sha256"]
    a = pd.read_csv(r_today, low_memory=False)
    b = pd.read_csv(sept, low_memory=False)
    cmp1 = compare(a, b, ["ratio", "PixelIntensityMean", "GFPPixelIntensityMean"])
    cmp1["byte_identical"] = same_bytes
    rec.summary["r_reproduction_xdp8"] = cmp1
    rec.check("XDP8.r_reproduces_sept", "MRID mrid_ratio() re-run today on XDP8 raw data reproduces the Sept "
              "cache (same cells, identical ratios)", same_bytes or (cmp1["only_a"] == 0 and cmp1["only_b"] == 0
              and cmp1["max_rel_diff_ratio"] < 1e-12), cmp1)

    # 2. Python port vs R output.
    for exp, raw_key, r_out, pm_path, tp_rm in (
            ("XDP8", "XDP8_raw", r_today, RAW_CACHE / cfg.experiments()["XDP8"]["platemap"], (2,)),
            ("XDP10", "XDP10_rerun_raw", LOCAL_OUT / "02b_rebuild" / "XDP10" / "mrid_celldata_ratio.csv",
             LOCAL_OUT / "01_metadata" / "platemap_mrid_XDP10.csv", ())):
        raw = pd.read_csv(data.staged_path(raw_key), low_memory=False)
        pm = pd.read_csv(pm_path, encoding="utf-8-sig", dtype=str, keep_default_na=False)[
            ["Sci_WellID", "Sci_SampleID", "Drug"]]
        py = mrid_ratio_py(raw, pm, tp_rm)
        r = pd.read_csv(r_out, low_memory=False)
        c = compare(py, r, ["ratio"])
        rec.summary[f"python_port_{exp}"] = c
        rec.check(f"{exp}.python_port_matches_r", f"{exp}: independent Python port of mrid_ratio() gives the same "
                  "cells and ratios as MRID's R function", c["only_a"] == 0 and c["only_b"] == 0
                  and c["max_rel_diff_ratio"] < 1e-12, c)
    rec.write()
    for ck in rec.checks:
        print(ck["status"], ck["id"], ck["value"])


if __name__ == "__main__":
    main()

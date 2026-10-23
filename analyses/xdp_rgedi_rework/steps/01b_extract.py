"""Step 01b: convert staged MRID per-cell caches into slim parquet with well roles attached.

Checks per experiment: one row per (timepoint, well, object); cache line label ==
config line for every cell; ratio == RFP/GFP; every cell maps to a configured well
and a known timepoint. XDP10 is read from the step-02b rebuild (MRID mrid_ratio on the verified re-extract).

Usage: python -I steps/01b_extract.py [EXP ...]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from lib import cfg, data  # noqa: E402
from lib.paths import LOCAL_OUT, RAW_CACHE  # noqa: E402
from lib.provenance import StepRecorder, sha256_file  # noqa: E402

STEP = "01b_extract"


def extract_one(exp: str, rec: StepRecorder) -> dict:
    source = cfg.experiments()[exp]["source"]
    if source["kind"] == "rebuild":
        # Per-cell ratios recomputed with MRID's own mrid_ratio() in step 02b.
        src = LOCAL_OUT / "02b_rebuild" / exp / "mrid_celldata_ratio.csv"
        rec.add_input(src, sha256_file(src), produced_by="02b_mrid_ratio_rebuild.R")
    else:
        src = data.staged_path(source["key"])
        staging = json.loads((RAW_CACHE / "staging_manifest.json").read_text())
        rec.add_input(src, staging[source["key"]]["sha256"])
    df = data.extract_cache(exp, src)

    key_dup = df.duplicated(["tp", "well", "obj"]).sum()
    rec.check(f"{exp}.unique_cells", f"{exp}: one row per (timepoint, well, object)", key_dup == 0,
              int(key_dup))
    unmapped = df.line.isna().sum()
    rec.check(f"{exp}.wells_mapped", f"{exp}: every cell's well is in the config", unmapped == 0,
              int(unmapped))
    no_hours = df.hours.isna().sum()
    rec.check(f"{exp}.tps_known", f"{exp}: every timepoint is in the timepoint file", no_hours == 0,
              sorted(df.loc[df.hours.isna(), "tp"].unique().tolist()))
    line_bad = (df.line_cache != df.line).sum()
    rec.check(f"{exp}.line_labels", f"{exp}: cache line label == config line for every cell",
              line_bad == 0, int(line_bad))
    with np.errstate(divide="ignore", invalid="ignore"):
        recomputed = df.rfp / df.gfp
    rel = np.abs(recomputed - df.ratio) / np.maximum(np.abs(df.ratio), 1e-12)
    rec.check(f"{exp}.ratio_definition", f"{exp}: ratio == RFP mean / GFP mean (rel. tol 1e-9)",
              bool((rel[np.isfinite(rel)] < 1e-9).all()), float(np.nanmax(rel)))
    nonpos = int((df.ratio <= 0).sum())
    nonfinite = int((~np.isfinite(df.ratio)).sum())
    rec.check(f"{exp}.ratio_nonpositive", f"{exp}: cells with ratio <= 0 or non-finite (excluded from log fits)",
              None, {"ratio_le_0": nonpos, "non_finite": nonfinite, "n": len(df)})

    df = df.drop(columns=["line_cache", "drug_cache"])
    path = data.parquet_path(exp)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, compression="zstd", index=False)
    rec.add_output(path, rows=len(df))
    counts = (df.groupby(["line", "condition", "tp"]).size().rename("n_cells").reset_index())
    counts.insert(0, "exp", exp)
    counts.to_csv(rec.out_dir / f"cell_counts_{exp}.csv", index=False)
    return {"rows": len(df), "rows_in_scope": int(df.in_scope.sum()), "wells": int(df.well.nunique()),
            "timepoints": sorted(df.tp.unique().tolist()), "ratio_le_0": nonpos}


def main() -> None:
    exps = sys.argv[1:] or list(cfg.experiments())
    rec = StepRecorder(STEP, LOCAL_OUT / STEP)
    prev = rec.out_dir / "summary.json"
    rec.summary = json.loads(prev.read_text()) if prev.exists() else {}
    if (rec.out_dir / "checks.json").exists():
        rec.checks = [c for c in json.loads((rec.out_dir / "checks.json").read_text())
                      if c["id"].split(".")[0] not in exps]
    for exp in exps:
        print(f"extract {exp} ...", flush=True)
        rec.summary[exp] = extract_one(exp, rec)
        print(f"  {exp}: {rec.summary[exp]}", flush=True)
    rec.write()
    for c in rec.checks:
        if c["status"] != "PASS":
            print("  ", c["status"], c["id"], c["value"])


if __name__ == "__main__":
    main()

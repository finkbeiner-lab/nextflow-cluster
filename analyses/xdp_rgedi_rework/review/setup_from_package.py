"""Build a data cache from the review package, so the cutoff steps (06-12) run on another computer.

The steps read per-cell parquet files and the legacy timepoint files from the cache (lib/paths.py).
This script writes them from the package:
  <cache>/parquet/<EXP>.parquet      from 2_live_dead_calls/all_cells/<EXP>_cells.csv.gz (same schema as Austin's)
  <cache>/raw/<timepoints path>      from 3_experiment_info/original_timepoints/ (path as in config/experiments.yaml)
  <cache>/raw/<platemap path>        from 3_experiment_info/original_platemaps/
Usage: python -I review/setup_from_package.py <package_dir> <cache_dir>
"""

import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PKG_ROOT = Path(__file__).resolve().parents[1]
BACK = {"ObjectLabelsFound": "obj", "GFP_mean": "gfp", "RFP_mean": "rfp"}
SCHEMA = ["exp", "gfp", "obj", "area", "cx", "cy", "rfp", "tp", "ratio", "well", "row", "col", "line", "condition",
          "sirna_name", "in_scope", "hours", "log10_ratio"]


def main() -> None:
    package, cache = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
    exps = yaml.safe_load((PKG_ROOT / "config" / "experiments.yaml").read_text())["experiments"]
    info = package / "3_experiment_info"
    (cache / "parquet").mkdir(parents=True, exist_ok=True)
    for exp, e in exps.items():
        for key, folder in (("timepoints", "original_timepoints"), ("platemap", "original_platemaps")):
            dst = cache / "raw" / e[key]
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(info / folder / Path(e[key]).name, dst)
        d = pd.read_csv(package / "2_live_dead_calls" / "all_cells" / f"{exp}_cells.csv.gz",
                        dtype={"well": str, "row": str, "line": str, "condition": str, "sirna_name": str})
        d = d.rename(columns=BACK)
        d["in_scope"] = True
        with np.errstate(divide="ignore", invalid="ignore"):
            d["log10_ratio"] = np.where(d.ratio > 0, np.log10(d.ratio), np.nan)
        d = d[SCHEMA].astype({"obj": "int64", "tp": "int64", "col": "int64"})
        d.to_parquet(cache / "parquet" / f"{exp}.parquet", index=False)
        print(f"{exp}: {len(d):,} cells -> {cache / 'parquet' / (exp + '.parquet')}", flush=True)


if __name__ == "__main__":
    main()

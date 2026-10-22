"""Compare a rerun of steps 06-12 with the results in the review package.

Checks, for the primary set W90:
  1. cutoffs     every line x experiment live and dead cutoff (relative difference)
  2. MRID input  the same cells, with the same live/dead call, in every experiment
  3. MRID output every odds ratio and 95% interval in OR_all_pairs.csv
Point estimates must match. Bootstrap intervals of the cutoffs match only when the rerun used the same B.
Writes <rerun_out>/compare_report.csv and prints PASS or FAIL for each check.
Usage: python -I review/compare_to_package.py <package_dir> <rerun_out_dir>   (XDP_RGEDI_CACHE / _RUN set as for the rerun)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from lib import livedead  # noqa: E402

TOL = 1e-9


def main() -> None:
    package, out = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
    rows = []
    mine = livedead.cutoff_table("sd")[["exp", "line", "W90_live", "W90_dead"]]
    ref = pd.read_csv(package / "2_live_dead_calls" / "cutoffs_per_line.csv")
    m = ref.merge(mine, on=["exp", "line"], how="outer", indicator=True)
    for side, a, b in (("live", "live_if_ratio_below", "W90_live"), ("dead", "dead_if_ratio_above", "W90_dead")):
        rel = np.abs(m[b] / m[a] - 1)
        rows.append({"check": f"cutoffs ({side})", "n": len(m), "max_rel_diff": float(rel.max()),
                     "pass": bool((m._merge == "both").all() and rel.max() < TOL)})
    calls = package / "2_live_dead_calls" / "MRID_input"
    for f in sorted(calls.glob("*/*_ratio_output.csv")):
        a = pd.read_csv(f, usecols=["filenames", "live_guesses"])
        b = pd.read_csv(out / "11_mrid_inputs" / "input" / "W90" / f.parent.name / f.name,
                        usecols=["filenames", "live_guesses"])
        j = a.merge(b, on="filenames", how="outer", suffixes=("_pkg", "_rerun"), indicator=True)
        same = (j._merge == "both").all() and (j.live_guesses_pkg == j.live_guesses_rerun).all()
        rows.append({"check": f"MRID input {f.parent.name}", "n": len(a), "n_rerun": len(b),
                     "cells_differing": int((j._merge != "both").sum() + (j.live_guesses_pkg != j.live_guesses_rerun).sum()),
                     "pass": bool(same)})
    a = pd.read_csv(package / "1_results" / "MRID_output" / "OR_all_pairs.csv")
    b = pd.read_csv(out / "12_logodds" / "W90" / "OR_all_pairs.csv")
    j = a.merge(b, on=["Reference", "Compare", "Time"], suffixes=("_pkg", "_rerun"))
    d = max(float(np.abs(np.log(j[f"{c}_pkg"] / j[f"{c}_rerun"])).max()) for c in ("OR", "lower", "upper"))
    rows.append({"check": "MRID odds ratios (OR_all_pairs)", "n": len(a), "n_rerun": len(b), "max_abs_log_diff": d,
                 "pass": bool(len(j) == len(a) == len(b) and d < 1e-6)})
    rep = pd.DataFrame(rows)
    rep.to_csv(out / "compare_report.csv", index=False)
    for r in rep.to_dict("records"):
        print(f"{'PASS' if r['pass'] else 'FAIL'}  {r['check']}")
    print("ALL PASS" if rep["pass"].all() else "SOME CHECKS FAILED - see compare_report.csv")


if __name__ == "__main__":
    main()

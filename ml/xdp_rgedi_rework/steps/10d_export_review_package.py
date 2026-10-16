"""Step 10d: export a self-contained review package (for an independent check of the run-2 W90 result).

Writes into DEST (default: Austin's Dropbox data-for-reuben folder):
  all_cells/<EXP>_cells.csv.gz          every in-scope cell, with the W90 cutoffs and its call
  our_MRID_output/                      MRID's own outputs for set W90 (step 12)
  extra_models/                         the models added on top of MRID (step 12b, step 13 pooling)
  plate_maps_and_timepoints/            original plate maps and timepoint files, corrected well roles, S.A. design
  source_files.csv                      the per-cell source file of each experiment (share + cluster path, sha256)
  thresholds_full.csv                   every cutoff and fit parameter from step 10a
  code/                                 snapshot of ml/xdp_rgedi_rework and the MRID R functions, environment
  RGEDI_SA_thresholds_evidence_*.html   the evidence document
The check at the end requires all_cells to reproduce the MRID input files exactly.
Usage: python -I steps/10d_export_review_package.py [DEST]
"""

import hashlib
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from lib import cfg, data, figs, livedead  # noqa: E402
from lib.paths import LOCAL_OUT, PKG_ROOT, RAW_CACHE, REPO_BIN, REPO_ROOT, RSCRIPT  # noqa: E402

SET = "W90"
DEST = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
    "/Users/aholub/Library/CloudStorage/Dropbox-Gladstone/Austin Holub/other/Austin/data-for-reuben")
SHARE_MAC, SHARE_CLUSTER = "/Volumes/Finkbeiner-Kaye", "/gladstone/finkbeiner/kaye"
RUN1 = LOCAL_OUT.parent / "run1_ruleD-single-cutoff_20261014"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def reason_not_used(d: pd.DataFrame, exp: str) -> pd.Series:
    """Why a cell is not an MRID input (empty string = used), following step 11 cells_for_mrid."""
    if exp == "XDP1":
        return pd.Series(np.where(d.tp > 10, "XDP1 after T10 (after S.A. dose)", ""), index=d.index)
    return pd.Series(np.select([d.condition == "sa", d.condition == "sirna"], ["S.A. well", "siRNA well"],
                               default=""), index=d.index)


def export_cells(out: Path) -> list:
    cut = livedead.cutoff_table(livedead.SETS[SET][2])
    checks = []
    out.mkdir(parents=True, exist_ok=True)
    for exp in figs.EXP_ORDER:
        d = data.load(exp)
        d = livedead.classify(d, exp, SET, cut, None)
        d["not_used_reason"] = reason_not_used(d, exp)
        call = np.where(d.live_guesses == 1, "live", np.where(d.live_guesses == 0, "dead", "dropped (buffer zone)"))
        d["call_by_cutoff"] = call
        d["in_MRID_input"] = (d.not_used_reason == "") & d.live_guesses.notna()
        d["filenames"] = exp + "_T" + d.tp.astype(str) + "_" + d.well + "_" + d.obj.astype(str)
        cols = ["exp", "filenames", "well", "row", "col", "line", "condition", "sirna_name", "tp", "hours", "obj",
                "area", "cx", "cy", "gfp", "rfp", "ratio", "lo", "hi", "call_by_cutoff", "not_used_reason",
                "in_MRID_input"]
        d = d[cols].rename(columns={"lo": "live_cutoff", "hi": "dead_cutoff", "obj": "ObjectLabelsFound",
                                    "gfp": "GFP_mean", "rfp": "RFP_mean"})
        d.to_csv(out / f"{exp}_cells.csv.gz", index=False, compression={"method": "gzip", "compresslevel": 6})
        # The used, classified cells must be exactly the MRID input of this set.
        e = cfg.experiments()[exp]["expt_name"]
        m = pd.read_csv(LOCAL_OUT / "11_mrid_inputs" / "input" / SET / e / f"{e}_ratio_output.csv",
                        usecols=["filenames", "live_guesses"])
        u = d[d.in_MRID_input]
        same = (len(u) == len(m) and set(u.filenames) == set(m.filenames)
                and int((u.call_by_cutoff == "live").sum()) == int(m.live_guesses.sum()))
        checks.append({"exp": exp, "cells": len(d), "in_MRID_input": len(u), "mrid_rows": len(m),
                       "dropped_buffer": int((d.not_used_reason.eq("") & d.call_by_cutoff.str.startswith("dropped")).sum()),
                       "matches_MRID_input": bool(same)})
        print(checks[-1], flush=True)
    return checks


def export_tables(dest: Path) -> None:
    lo = LOCAL_OUT / "12_logodds" / SET
    a = dest / "our_MRID_output"
    a.mkdir(parents=True, exist_ok=True)
    for f in ["Data.csv", "OR_all_pairs.csv", "experiments_per_hour.csv", "time_index_hours.csv", "warnings.csv",
              "mrid_log.txt", "sessionInfo.txt"]:
        shutil.copy2(lo / f, a / f)
    b = dest / "extra_models"
    b.mkdir(parents=True, exist_ok=True)
    for f in ["OR_rederived_per_timepoint.csv", "OR_time_averaged_per_experiment.csv", "OR_time_averaged_pooled.csv",
              "warnings_12b.csv"]:
        shutil.copy2(lo / f, b / f)
    p = pd.read_csv(LOCAL_OUT / "13_or_figures" / "pooled_or_by_set.csv")
    p[p.set == SET].to_csv(b / "pooled_or_summary.csv", index=False)
    shutil.copy2(LOCAL_OUT / "10a_final_tables" / "thresholds" / "thresholds_all.csv", dest / "thresholds_full.csv")
    docs = sorted((LOCAL_OUT / "docs").glob("RGEDI_SA_thresholds_evidence_*.html"))
    shutil.copy2(docs[-1], dest / docs[-1].name)


def export_layout(dest: Path) -> None:
    out = dest / "plate_maps_and_timepoints"
    for sub in ("original_platemaps", "original_timepoints", "corrected_well_roles"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    design = []
    for exp in figs.EXP_ORDER:
        e = cfg.experiments()[exp]
        shutil.copy2(RAW_CACHE / e["platemap"], out / "original_platemaps" / Path(e["platemap"]).name)
        shutil.copy2(RAW_CACHE / e["timepoints"], out / "original_timepoints" / Path(e["timepoints"]).name)
        cfg.well_roles(exp).to_csv(out / "corrected_well_roles" / f"{exp}_well_roles.csv", index=False)
        s = e["sa"]
        tps = cfg.read_timepoints(exp)
        design.append({"exp": exp, "timepoints": f"T{tps.tp.min()}-T{tps.tp.max()}",
                       "hours": ", ".join(f"{h:g}" for h in tps.hours), "sa_design": s.get("design"),
                       "sa_doses": " | ".join(f"after T{x.get('after_tp')}: {x.get('note')}" for x in s.get("doses", [])),
                       "sa_pre_tp": s.get("pre_tp"), "sa_post_tps": s.get("post_tps"), "sa_fit_tps": s.get("fit_tps"),
                       "sa_provenance": s.get("provenance"), "sa_evidence": s.get("evidence"),
                       "flagged_tps": json.dumps(e.get("flagged_tps", {})), "notes": e.get("notes", "")})
    shutil.copy2(RAW_CACHE / "XDP10" / "XDP10-RGEDI-platemap.csv", out / "original_platemaps" / "XDP10-RGEDI-platemap.csv")
    shutil.copy2(RAW_CACHE / "XDP-experiments-info.xlsx", out / "XDP-experiments-info.xlsx")
    pd.DataFrame(design).to_csv(out / "experiment_design.csv", index=False)


def export_sources(dest: Path) -> None:
    """One row per source file: the per-cell file each experiment used, and the raw files behind the rebuilds."""
    man = json.loads((RAW_CACHE / "staging_manifest.json").read_text())

    def row(exp: str, role: str, mac: str, nbytes: int, digest: str) -> dict:
        return {"exp": exp, "role": role, "mac_path": mac, "cluster_path": mac.replace(SHARE_MAC, SHARE_CLUSTER),
                "bytes": nbytes, "sha256": digest}

    rows = []
    for exp in figs.EXP_ORDER:
        src = cfg.experiments()[exp]["source"]
        if src["kind"] == "cache":
            m = man[src["key"]]
            rows.append(row(exp, "USED: per-cell file (MRID mrid_ratio output, Sept 2026 rework)", m["source"],
                            m["bytes"], m["sha256"]))
            continue
        reb = RUN1 / "02b_rebuild" / exp / "mrid_celldata_ratio.csv"
        share = (f"{SHARE_MAC}/XDP/RGEDI-full-rework/run1_ruleD-single-cutoff_20261014/02b_rebuild/{exp}/"
                 "mrid_celldata_ratio.csv")
        rows.append(row(exp, "USED: per-cell file (MRID mrid_ratio() rerun on the raw file below, step 02b)", share,
                        reb.stat().st_size, sha256(reb)))
        m = man[src["raw_key"]]
        rows.append(row(exp, "raw cell_data.csv, input of the rebuild", m["source"], m["bytes"], m["sha256"]))
        if src.get("verify_key"):
            m = man[src["verify_key"]]
            rows.append(row(exp, "NOT USED: truncated original cell_data.csv (the first run used this one)",
                            m["source"], m["bytes"], m["sha256"]))
    pd.DataFrame(rows).to_csv(dest / "source_files.csv", index=False)


def export_code(dest: Path) -> None:
    out = dest / "code"
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(PKG_ROOT, out / "xdp_rgedi_rework",
                    ignore=shutil.ignore_patterns("__pycache__", ".DS_Store", "*.pyc"))
    shutil.copytree(REPO_BIN / "mrid_R", out / "mrid_R", ignore=shutil.ignore_patterns(".DS_Store"))
    for f in ("gmm1d.py", "sa_qc_ratio.py"):
        if (REPO_BIN / f).exists():
            shutil.copy2(REPO_BIN / f, out / f)
    import matplotlib, scipy, sklearn, statsmodels, yaml  # noqa: E401
    head = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    status = subprocess.run(["git", "-C", str(REPO_ROOT), "status", "--short", "--", str(PKG_ROOT)],
                            capture_output=True, text=True).stdout.strip()
    r = subprocess.run([RSCRIPT, "--version"], capture_output=True, text=True)
    lines = [f"python {platform.python_version()}", f"numpy {np.__version__}", f"pandas {pd.__version__}",
             f"scipy {scipy.__version__}", f"scikit-learn {sklearn.__version__}", f"statsmodels {statsmodels.__version__}",
             f"matplotlib {matplotlib.__version__}", f"pyyaml {yaml.__version__}", (r.stdout or r.stderr).strip(),
             "R packages: see our_MRID_output/sessionInfo.txt",
             f"repo nextflow-cluster HEAD {head} (branch austin/master)",
             f"ml/xdp_rgedi_rework git status: {status or 'clean'}"]
    (out / "environment.txt").write_text("\n".join(lines) + "\n")


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    checks = export_cells(DEST / "all_cells")
    export_tables(DEST)
    export_layout(DEST)
    export_sources(DEST)
    export_code(DEST)
    (DEST / "export_checks.json").write_text(json.dumps(checks, indent=2))
    bad = [c["exp"] for c in checks if not c["matches_MRID_input"]]
    print("ALL MATCH" if not bad else f"MISMATCH: {bad}")


if __name__ == "__main__":
    main()

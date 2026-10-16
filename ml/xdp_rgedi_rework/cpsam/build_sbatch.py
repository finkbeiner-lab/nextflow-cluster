#!/usr/bin/env python3
"""Write the self-contained XDP10 Cellpose-SAM sbatch scripts (Austin submits them).

Each script embeds, as base64, the code it runs plus the manifest, so nothing has
to be copied to the cluster first:

* ``cpsam_tiles.py`` and ``build_celldata.py`` (this folder);
* ``percell_integrate.py`` and ``model.py`` **as committed at HEAD**. That is the
  exact version (sha256 bf13c9f6...) recorded in the XDP10 neurite run's
  provenance.json, not the uncommitted working copy.

Scripts written to ``--out`` (default ``~/Downloads/sbatch-claude``):

* ``xdp10_cpsam_pilot.sbatch``: GPU. Tile preflight for all items, then 2 wells x
  10 tps (D08 untreated + D22 S.A., both CTR-33114.C), fresh-vs-cache T7 check,
  overlays at T1/T7/T10, and a pilot cell_data.csv.
* ``xdp10_cpsam_seg.sbatch``: GPU job array over all items (resumable).
* ``xdp10_cpsam_post.sbatch``: CPU. Full cell_data.csv + count QC vs Galaxy.
"""

from __future__ import annotations

import argparse
import base64
import datetime
import hashlib
import io
import os
import subprocess
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
EXP, PLATE_ID = "XDP10", "PID20250129_XDP10-RGEDI"
RAW_ROOT = "/gladstone/finkbeiner/robodata/IXM4Galaxy/Austin/XDP10-RGEDI"
BG_ROOT = "/gladstone/finkbeiner/kaye/XDP/XDP10/GXYTMP/BackgroundCorrected"
CACHE_DIR = "/gladstone/finkbeiner/kaye/XDP/neurite-rework/XDP10/cache"
GALAXY_CELLDATA = "/gladstone/finkbeiner/kaye/XDP/XDP10/GXYTMP/rerun-ExtraCellInfo/cell_data.csv"
OUT_DIR = "/gladstone/finkbeiner/kaye/XDP/RGEDI-full-rework/cpsam-seg/XDP10"
SIF = "$HOME/nextflow-cluster-master/nextflow-cluster.sif"
# In-scope lines by row pair (experiments.yaml XDP10; 33362.C rows H-I out of scope).
LINES = {"D": "CTR-33114.C", "E": "CTR-33114.C", "F": "CTR-33113.2I", "G": "CTR-33113.2I",
         "J": "XDP-33109.2B", "K": "XDP-33109.2B", "L": "XDP-33363.D", "M": "XDP-33363.D"}
UNTREATED_COLS, SA_COL = range(8, 22), 22
TIMEPOINTS = [f"T{t}" for t in range(1, 11)]
PILOT_WELLS = ["D08", "D22"]


def manifest(wells=None) -> pd.DataFrame:
    rows = []
    for r, line in LINES.items():
        for c in list(UNTREATED_COLS) + [SA_COL]:
            w = f"{r}{c:02d}"
            if wells and w not in wells:
                continue
            for tp in TIMEPOINTS:
                rows.append(dict(experiment=EXP, well=w, timepoint=tp, cell_line=line,
                                 condition="sa" if c == SA_COL else "untreated",
                                 raw_dir=f"{RAW_ROOT}/{w}", bg_dir=f"{BG_ROOT}/{w}"))
    return pd.DataFrame(rows)


def git(*a: str) -> str:
    return subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, text=True).stdout.strip()


def git_show(path: str) -> bytes:
    """File content at HEAD, byte-exact (no strip: the sha256 must match provenance)."""
    return subprocess.run(["git", "-C", str(REPO), "show", f"HEAD:{path}"], capture_output=True,
                          check=True).stdout


def embedded_files(man: pd.DataFrame) -> dict:
    files = {
        "cpsam_tiles.py": (HERE / "cpsam_tiles.py").read_bytes(),
        "build_celldata.py": (HERE / "build_celldata.py").read_bytes(),
        "percell_integrate.py": git_show("ml/neurite/percell_integrate.py"),
        "model.py": git_show("ml/neurite/model.py"),
        "README.md": (HERE / "README.md").read_bytes(),
    }
    buf = io.StringIO()
    man.to_csv(buf, index=False)
    files["manifest.csv"] = buf.getvalue().encode()
    return files


def embed_block(files: dict) -> str:
    out = []
    for name, data in files.items():
        tag = "B64_" + name.replace(".", "_")
        b64 = base64.encodebytes(data).decode()
        out.append(f"base64 -d > \"$WORK/{name}\" <<'{tag}'\n{b64}{tag}\n")
    return "\n".join(out)


def provenance_py(files: dict, role: str) -> str:
    sha = {k: hashlib.sha256(v).hexdigest() for k, v in files.items()}
    info = dict(role=role, built=datetime.datetime.now().isoformat(timespec="seconds"),
                git_commit=git("rev-parse", "HEAD"), git_branch=git("rev-parse", "--abbrev-ref", "HEAD"),
                cpsam_code_committed=not git("status", "--porcelain", "--", str(HERE)),
                code_sha256=sha, sif=SIF.replace("$HOME", "~"))
    return f"""python - <<'PROV' || echo "WARN: provenance step failed (continuing)"
import json, os, platform, datetime
info = {info!r}
info.update(run_utc=datetime.datetime.utcnow().isoformat(), host=platform.node(),
            slurm_job_id=os.environ.get("SLURM_JOB_ID"), array_task=os.environ.get("SLURM_ARRAY_TASK_ID"))
v = {{}}
for m in ("numpy", "scipy", "skimage", "pandas", "tifffile", "torch", "cellpose"):
    try:
        v[m] = __import__(m).__version__
    except Exception as e:
        v[m] = repr(e)
info["versions"] = v
p = os.path.join("{OUT_DIR}", "provenance", info["role"] + "_" + str(info["slurm_job_id"]) + "_" + str(info["array_task"]) + ".json")
os.makedirs(os.path.dirname(p), exist_ok=True)
json.dump(info, open(p, "w"), indent=2)
print("provenance ->", p)
PROV"""


HEADER = """#!/bin/bash
#SBATCH --partition=galaxy
{gres}#SBATCH --job-name={name}
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem}
#SBATCH --time={time}
#SBATCH --output={name}_%{jid}.out
{extra}#
# {desc}
# Somas: neurite recipe (Cellpose-SAM d25/flow0.6/cellprob-1, MAD clean k2.0, size floor) on RAW
# FITC tiles ({raw}); per-cell FITC/RFP measured on Galaxy BG-corrected tiles ({bg}).
# Output: {out}
# Built {built} by ml/xdp_rgedi_rework/cpsam/build_sbatch.py (git {sha}).
set -euo pipefail
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
OUT={out}
SIF={sif}
mkdir -p "$OUT"
"""


def write(path: Path, text: str) -> None:
    path.write_text(text)
    os.chmod(path, 0o755)
    print("wrote", path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=str(Path.home() / "Downloads" / "sbatch-claude"))
    ap.add_argument("--items-per-task", type=int, default=50,
                    help="array sizing (~4.5 min/montage fresh, ~1 min cached -> ~3.7 h per task)")
    args = ap.parse_args()
    out = Path(args.out)
    full = manifest()
    pilot = manifest(PILOT_WELLS)
    n_tasks = -(-len(full) // args.items_per_task)
    common = dict(raw=RAW_ROOT, bg=BG_ROOT, out=OUT_DIR, sif=SIF, sha=git("rev-parse", "--short", "HEAD"),
                  built=datetime.date.today().isoformat())
    run = 'apptainer exec --nv -B /gladstone/finkbeiner:/gladstone/finkbeiner "$SIF"'

    # ---- pilot -----------------------------------------------------------------
    files = embedded_files(full)
    files["pilot_manifest.csv"] = pilot.to_csv(index=False).encode()
    body = f"""
{embed_block(files)}
cp "$WORK/manifest.csv" "$OUT/manifest.csv"; cp "$WORK/README.md" "$OUT/README.md"
cd "$WORK"
nvidia-smi --query-gpu=name,memory.total --format=csv
{run} bash -s <<'IN_SIF'
set -euo pipefail
{provenance_py(files, "pilot")}
python cpsam_tiles.py --manifest manifest.csv --out-dir "{OUT_DIR}" --preflight
python cpsam_tiles.py --manifest pilot_manifest.csv --out-dir "{OUT_DIR}" --cache-dir "{CACHE_DIR}" \\
    --compare-cache --overlay-tps T1,T7,T10
mkdir -p "{OUT_DIR}/pilot"
python build_celldata.py --seg-dir "{OUT_DIR}" --manifest pilot_manifest.csv --plate-id {PLATE_ID} \\
    --out "{OUT_DIR}/pilot/cell_data.csv" --galaxy-celldata "{GALAXY_CELLDATA}" --allow-incomplete
for f in counts_well_tp.csv timing_000.csv cache_compare_000.csv preflight.csv; do
    cp "{OUT_DIR}/qc/$f" "{OUT_DIR}/pilot/$f" 2>/dev/null || true
done
echo "--- fresh vs neurite-cache T7"; cat "{OUT_DIR}"/qc/cache_compare_000.csv
echo "--- timing"; cat "{OUT_DIR}"/qc/timing_000.csv
IN_SIF
echo "PILOT DONE -> $OUT (pilot/, qc/, qc/overlays/)"
"""
    write(out / "xdp10_cpsam_pilot.sbatch", HEADER.format(
        gres="#SBATCH --gres=gpu:1\n", name="xdp10_cpsam_pilot", cpus=6, mem="32G", time="05:00:00",
        jid="j", extra="",
        desc=f"XDP10 Cellpose-SAM PILOT: preflight {len(full)} items; segment {PILOT_WELLS} x T1-T10.",
        **common) + body)

    # ---- array -----------------------------------------------------------------
    files = embedded_files(full)
    body = f"""
{embed_block(files)}
cd "$WORK"
nvidia-smi --query-gpu=name,memory.total --format=csv
{run} bash -s <<'IN_SIF'
set -euo pipefail
{provenance_py(files, "seg")}
python cpsam_tiles.py --manifest manifest.csv --out-dir "{OUT_DIR}" --cache-dir "{CACHE_DIR}" \\
    --task-index $SLURM_ARRAY_TASK_ID --n-tasks {n_tasks}
IN_SIF
echo "TASK $SLURM_ARRAY_TASK_ID DONE"
"""
    write(out / "xdp10_cpsam_seg.sbatch", HEADER.format(
        gres="#SBATCH --gres=gpu:1\n", name="xdp10_cpsam_seg", cpus=6, mem="32G", time="05:00:00",
        jid="A_%a", extra=f"#SBATCH --array=0-{n_tasks - 1}%2\n",
        desc=(f"XDP10 Cellpose-SAM SEGMENTATION: {len(full)} (well,tp) items, {n_tasks} tasks x "
              f"~{args.items_per_task}, max 2 GPUs at once. Resumable: resubmit to finish leftovers."),
        **common) + body)

    # ---- post ------------------------------------------------------------------
    files = {k: v for k, v in embedded_files(full).items()
             if k in ("build_celldata.py", "manifest.csv", "README.md")}
    body = f"""
{embed_block(files)}
cd "$WORK"
cp "$WORK/README.md" "$OUT/README.md"
apptainer exec -B /gladstone/finkbeiner:/gladstone/finkbeiner "$SIF" bash -s <<'IN_SIF'
set -euo pipefail
{provenance_py(files, "post")}
cat "{OUT_DIR}"/qc/failed_*.csv 2>/dev/null && echo "WARNING: failed items above" || true
python build_celldata.py --seg-dir "{OUT_DIR}" --manifest manifest.csv --plate-id {PLATE_ID} \\
    --galaxy-celldata "{GALAXY_CELLDATA}"
python - <<'PY'
import glob, pandas as pd
t = pd.concat([pd.read_csv(f) for f in glob.glob("{OUT_DIR}/qc/timing_*.csv")])
print("montages timed", len(t), "| median s/montage", t.total_s.median(), "| by source:")
print(t.groupby("source").total_s.describe().round(1).to_string())
PY
IN_SIF
ls -la "$OUT/cell_data.csv"
echo "POST DONE -> $OUT/cell_data.csv (+ qc/counts_well_tp.csv)"
"""
    write(out / "xdp10_cpsam_post.sbatch", HEADER.format(
        gres="", name="xdp10_cpsam_post", cpus=2, mem="32G", time="03:00:00", jid="j", extra="",
        desc="XDP10 Cellpose-SAM POST: Galaxy-format cell_data.csv from all per-cell files.",
        **common) + body)

    pilot.to_csv(HERE / "xdp10_pilot_manifest.csv", index=False)
    full.to_csv(HERE / "xdp10_manifest.csv", index=False)
    print(f"items: full {len(full)} ({full.well.nunique()} wells), pilot {len(pilot)}, array tasks {n_tasks}")


if __name__ == "__main__":
    main()

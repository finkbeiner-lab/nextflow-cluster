"""Load and resolve the experiment ground truth (config/*.yaml).

Turns the well-block notation of ``experiments.yaml`` into explicit per-well
tables (line + condition) and normalises the legacy platemap / timepoint files
(BOM, CR line endings, NA vs blank Drug, unpadded well IDs, ``ElaspedHours``).
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import yaml

from lib.paths import CONFIG_DIR, RAW_CACHE

ROWS = "ABCDEFGHIJKLMNOP"


def load_yaml(name: str) -> dict:
    return yaml.safe_load((CONFIG_DIR / name).read_text())


def _deep_merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        out[k] = _deep_merge(base[k], v) if isinstance(v, dict) and isinstance(base.get(k), dict) else v
    return out


def experiments() -> Dict[str, dict]:
    """Experiment config; ``XDP_RGEDI_EXP_OVERRIDE`` (a yaml in config/) deep-merges per-run changes.

    A mapping value in the override replaces only the keys it names, except ``source``,
    which is replaced whole (so a ``cache`` source can become a ``rebuild`` source).
    """
    exps = load_yaml("experiments.yaml")["experiments"]
    name = os.environ.get("XDP_RGEDI_EXP_OVERRIDE")
    if name:
        for exp, over in load_yaml(name)["experiments"].items():
            if exp not in exps:
                raise KeyError(f"{name}: unknown experiment {exp}")
            src = over.pop("source", None)
            exps[exp] = _deep_merge(exps[exp], over)
            if src is not None:
                exps[exp]["source"] = src
    return exps


def lines_in_scope() -> Dict[str, dict]:
    return load_yaml("experiments.yaml")["lines_in_scope"]


def params() -> dict:
    return load_yaml("params.yaml")


def decisions() -> List[dict]:
    return load_yaml("decisions.yaml")


def pad_well(well: str) -> str:
    """'C2' -> 'C02'; 'C02' unchanged."""
    m = re.fullmatch(r"\s*([A-Pa-p])\s*0*(\d{1,2})\s*", str(well))
    if not m:
        raise ValueError(f"not a well id: {well!r}")
    return f"{m.group(1).upper()}{int(m.group(2)):02d}"


def _expand_rows(spec: str) -> List[str]:
    out: List[str] = []
    for part in str(spec).split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out += list(ROWS[ROWS.index(a.strip()):ROWS.index(b.strip()) + 1])
        else:
            out.append(part)
    return out


def _expand_cols(spec) -> List[int]:
    out: List[int] = []
    for part in str(spec).split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def expand_block(block: dict) -> List[str]:
    """{rows: 'C-N', cols: '2,4,5'} -> ['C02', 'C04', ...] (zero-padded)."""
    return [f"{r}{c:02d}" for r in _expand_rows(block["rows"]) for c in _expand_cols(block["cols"])]


def expand_blocks(blocks) -> List[str]:
    wells: List[str] = []
    for b in blocks or []:
        wells += expand_block(b)
    return wells


def well_roles(exp: str) -> pd.DataFrame:
    """Return one row per configured well: well, row, col, line, condition, sirna_name, in_scope.

    ``condition`` is one of ``sa``, ``untreated``, ``sirna``. Raises if a well is
    assigned two conditions or two lines (the config must be unambiguous).
    """
    e = experiments()[exp]
    scope = lines_in_scope()
    line_of: Dict[str, str] = {}
    for line, block in e["lines"].items():
        for w in expand_block(block):
            if w in line_of:
                raise ValueError(f"{exp}: well {w} assigned to {line_of[w]} and {line}")
            line_of[w] = line
    cond_of: Dict[str, str] = {}
    sirna_of: Dict[str, Optional[str]] = {}

    def assign(wells: List[str], cond: str, name: Optional[str] = None) -> None:
        for w in wells:
            if w in cond_of:
                raise ValueError(f"{exp}: well {w} assigned to {cond_of[w]} and {cond}")
            cond_of[w] = cond
            sirna_of[w] = name

    c = e["conditions"]
    assign(expand_blocks(c.get("sa")), "sa")
    assign(expand_blocks(c.get("untreated")), "untreated")
    for name, blocks in (c.get("sirna") or {}).items():
        assign(expand_blocks(blocks), "sirna", name)

    rows = []
    for w, line in line_of.items():
        rows.append({"well": w, "row": w[0], "col": int(w[1:]), "line": line,
                     "condition": cond_of.get(w), "sirna_name": sirna_of.get(w),
                     "in_scope": line in scope})
    df = pd.DataFrame(rows).sort_values(["row", "col"]).reset_index(drop=True)
    missing = df[df.condition.isna()]
    if len(missing):
        raise ValueError(f"{exp}: wells with a line but no condition: {missing.well.tolist()[:10]}")
    return df


def read_platemap(exp: str, apply_patches: bool = True) -> pd.DataFrame:
    """Read the legacy platemap, normalised (padded wells, stripped strings, Drug '' for untreated)."""
    e = experiments()[exp]
    raw = pd.read_csv(RAW_CACHE / e["platemap"], encoding="utf-8-sig", dtype=str,
                      keep_default_na=False)
    raw.columns = [c.strip() for c in raw.columns]
    pm = raw[["Sci_WellID", "Sci_SampleID", "Drug"]].apply(lambda s: s.str.strip())
    pm["Drug"] = pm["Drug"].replace({"NA": ""})
    pm["well"] = pm["Sci_WellID"].map(pad_well)
    if apply_patches:
        for patch in e.get("platemap_patches", []):
            if patch["op"] == "keep_rows":
                pm = pm.iloc[patch["first"] - 1:patch["last"]].copy()
            elif patch["op"] == "set_drug":
                wells = set(expand_block(patch["block"]))
                pm.loc[pm.well.isin(wells), "Drug"] = patch["value"]
            else:
                raise ValueError(f"unknown platemap patch op {patch['op']}")
    return pm.reset_index(drop=True)


def read_timepoints(exp: str) -> pd.DataFrame:
    """Return tp (int) and hours (float) from the legacy timepoint file (BOM / 'T' prefix tolerant)."""
    e = experiments()[exp]
    raw = pd.read_csv(RAW_CACHE / e["timepoints"], encoding="utf-8-sig", dtype=str)
    tp = raw.iloc[:, 0].str.strip().str.lstrip("Tt").astype(int)
    hours = raw.iloc[:, 1].astype(float)
    return pd.DataFrame({"tp": tp, "hours": hours})

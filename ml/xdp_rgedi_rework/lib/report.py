"""Helpers for the evidence document: load step outputs, embed figures, format tables and numbers."""

import base64
import html
import json
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd

from lib.paths import LOCAL_OUT

STATUS_CLASS = {"PASS": "ok", "WARN": "warn", "FAIL": "fail"}
DECISION_CLASS = {"approved_austin": "ok", "approved_julia": "ok", "proposed": "warn", "rejected": "fail",
                  "superseded": "muted"}
DECISION_LABEL = {"approved_austin": "approved (Austin)", "approved_julia": "approved (Julia)",
                  "proposed": "needs decision", "rejected": "rejected", "superseded": "superseded"}


def step_json(step: str, name: str):
    p = LOCAL_OUT / step / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else ({} if name in ("summary", "manifest") else [])


def step_csv(step: str, name: str) -> pd.DataFrame:
    p = LOCAL_OUT / step / name
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def _resolve(path) -> Path:
    """Figure paths are stored absolute; map one written before the run folders existed (or by another run) into
    this run's folder, so a copied step finds its own figures."""
    p = Path(path)
    if p.exists():
        return p
    parts = p.parts
    if "out" in parts:
        i = parts.index("out")
        rest = parts[i + 1:]
        if rest and rest[0].startswith("run"):
            rest = rest[1:]
        q = LOCAL_OUT.joinpath(*rest)
        if q.exists():
            return q
    return p


def img(path, alt: str, cls: str = "plate") -> str:
    p = _resolve(path)
    if not p.exists():
        return f'<p class="missing">Figure not yet produced: {html.escape(p.name)}</p>'
    b64 = base64.b64encode(p.read_bytes()).decode()
    return f'<img class="{cls}" alt="{html.escape(alt)}" src="data:image/png;base64,{b64}">'


def figure(fig: dict, number: str = "") -> str:
    cap = html.escape(fig.get("caption", ""))
    lab = f'<span class="fig-no">{html.escape(number)}</span> ' if number else ""
    return f'<figure>{img(fig["path"], fig.get("caption", ""))}<figcaption>{lab}{cap}</figcaption></figure>'


def figs_of(step: str) -> dict:
    return {f["id"]: f for f in step_json(step, "figures")}


def fmt(x, nd: int = 2, pct: bool = False) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "–"
    if pct:
        return f"{100 * x:.{nd}f}%"
    if isinstance(x, (int,)) and not isinstance(x, bool):
        return f"{x:,}"
    return f"{x:.{nd}f}"


def badge(status: str, label: Optional[str] = None) -> str:
    cls = STATUS_CLASS.get(status, DECISION_CLASS.get(status, "muted"))
    text = label or DECISION_LABEL.get(status, status)
    return f'<span class="badge {cls}">{html.escape(text)}</span>'


def table(df: pd.DataFrame, cols: List[str], headers: List[str], num_cols: Iterable[str] = (),
          raw_cols: Iterable[str] = (), cls: str = "") -> str:
    num_cols, raw_cols = set(num_cols), set(raw_cols)
    th = "".join(f'<th class="{"num" if c in num_cols else ""}">{html.escape(h)}</th>' for c, h in zip(cols, headers))
    rows = []
    for _, r in df.iterrows():
        tds = []
        for c in cols:
            v = r[c]
            if c in raw_cols:
                cell = "" if v is None or (isinstance(v, float) and pd.isna(v)) else str(v)
            else:
                cell = html.escape("–" if v is None or (isinstance(v, float) and pd.isna(v)) else str(v))
            tds.append(f'<td class="{"num" if c in num_cols else ""}">{cell}</td>')
        rows.append("<tr>" + "".join(tds) + "</tr>")
    return (f'<div class="table-wrap"><table class="{cls}"><thead><tr>{th}</tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table></div>')


def all_checks(steps: Iterable[str]) -> pd.DataFrame:
    rows = []
    for s in steps:
        for c in step_json(s, "checks"):
            rows.append({"step": s, **c})
    return pd.DataFrame(rows)

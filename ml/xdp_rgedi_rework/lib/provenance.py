"""Provenance helpers: checksums, git state, package versions, step manifests."""

import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Optional

from lib.paths import PKG_ROOT, REPO_ROOT

CHUNK = 8 * 1024 * 1024


def sha256_file(path: Path) -> str:
    """Return the sha256 hex digest of a file, streamed in 8 MB chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def git_state() -> Dict[str, Any]:
    """Return HEAD, branch, and a hash of the uncommitted diff of this package."""
    def run(*args: str) -> str:
        try:
            return subprocess.run(["git", "-C", str(REPO_ROOT), *args], capture_output=True,
                                  text=True, check=True).stdout.strip()
        except (subprocess.CalledProcessError, OSError):   # no git, or not a repository (review package)
            return ""

    rel = str(PKG_ROOT.relative_to(REPO_ROOT))
    status = run("status", "--porcelain", "--", rel)
    diff = run("diff", "HEAD", "--", rel)
    untracked = sorted(p for p in PKG_ROOT.rglob("*")
                       if p.is_file() and "__pycache__" not in p.parts)
    h = hashlib.sha256(diff.encode())
    for p in untracked:
        h.update(str(p.relative_to(PKG_ROOT)).encode())
        h.update(p.read_bytes())
    return {
        "head": run("rev-parse", "HEAD"),
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "package_dirty": bool(status),
        "package_content_sha256": h.hexdigest(),
    }


def versions() -> Dict[str, str]:
    """Return Python and key package versions."""
    out = {"python": platform.python_version()}
    for mod in ("numpy", "pandas", "scipy", "sklearn", "statsmodels", "matplotlib", "pyarrow", "jinja2"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:  # noqa: BLE001 - version capture must never fail a step
            out[mod] = "unavailable"
    return out


class StepRecorder:
    """Collects manifest / summary / checks / figures for one pipeline step.

    Every step writes the same four JSON files into its output folder so the
    report builder and the tests can read any step the same way.
    """

    def __init__(self, step: str, out_dir: Path, params: Optional[Dict[str, Any]] = None):
        self.step = step
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.t0 = time.time()
        self.manifest: Dict[str, Any] = {"step": step, "params": params or {}, "inputs": [],
                                         "outputs": [], "git": git_state(), "versions": versions()}
        self.summary: Dict[str, Any] = {}
        self.checks: list = []
        self.figures: list = []

    def add_input(self, path: Path, sha256: Optional[str] = None, **extra: Any) -> None:
        p = Path(path)
        entry = {"path": str(p), "bytes": p.stat().st_size if p.exists() else None, "sha256": sha256}
        entry.update(extra)
        self.manifest["inputs"].append(entry)

    def add_output(self, path: Path, **extra: Any) -> None:
        entry = {"path": str(path)}
        entry.update(extra)
        self.manifest["outputs"].append(entry)

    def check(self, cid: str, description: str, ok: Optional[bool], value: Any = None,
              warn: bool = False) -> bool:
        """Record a check. ``ok=None`` records a WARN-only informational check."""
        status = "PASS" if ok else ("WARN" if (warn or ok is None) else "FAIL")
        self.checks.append({"id": cid, "description": description, "status": status,
                            "value": value})
        return bool(ok)

    def figure(self, fid: str, path: Path, caption: str, **extra: Any) -> None:
        entry = {"id": fid, "path": str(path), "caption": caption}
        entry.update(extra)
        self.figures.append(entry)

    def write(self) -> None:
        self.manifest["elapsed_s"] = round(time.time() - self.t0, 1)
        self.manifest["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
        for name, obj in (("manifest", self.manifest), ("summary", self.summary),
                          ("checks", self.checks), ("figures", self.figures)):
            with open(self.out_dir / f"{name}.json", "w") as fh:
                json.dump(obj, fh, indent=2, default=_json_default)

    @property
    def failed(self) -> list:
        return [c for c in self.checks if c["status"] == "FAIL"]


def _json_default(o: Any) -> Any:
    try:
        import numpy as np
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
    except ImportError:
        pass
    if isinstance(o, Path):
        return str(o)
    return str(o)

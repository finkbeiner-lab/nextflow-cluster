"""Step 00a: copy input files from the SMB share to the local cache, once.

The share reads at ~0.5 MB/s, so every file is copied exactly once, one at a
time, in 8 MB chunks, with the sha256 computed during the same read. A copy goes
to ``<name>.part`` and is renamed only when complete; a rerun skips files whose
size and recorded sha256 already match. Progress (MB/s, ETA) goes to stdout.

Usage:
    python -I steps/00_stage.py [--only KEY ...] [--small-only]
"""

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.paths import CONFIG_DIR, RAW_CACHE  # noqa: E402

import yaml  # noqa: E402

CHUNK = 8 * 1024 * 1024
MANIFEST = RAW_CACHE / "staging_manifest.json"


def load_manifest() -> dict:
    if MANIFEST.exists():
        return json.loads(MANIFEST.read_text())
    return {}


def save_manifest(man: dict) -> None:
    tmp = MANIFEST.with_suffix(".tmp")
    tmp.write_text(json.dumps(man, indent=2))
    tmp.replace(MANIFEST)


def copy_with_hash(src: Path, dst: Path, label: str) -> dict:
    """Stream-copy ``src`` to ``dst`` computing sha256; return a manifest entry."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    part = dst.with_name(dst.name + ".part")
    total = src.stat().st_size
    h = hashlib.sha256()
    done, t0, last = 0, time.time(), 0.0
    with open(src, "rb") as fin, open(part, "wb") as fout:
        while True:
            block = fin.read(CHUNK)
            if not block:
                break
            fout.write(block)
            h.update(block)
            done += len(block)
            now = time.time()
            if now - last > 30:
                rate = done / max(now - t0, 1e-6) / 1e6
                eta = (total - done) / max(rate * 1e6, 1) / 60
                print(f"  [{label}] {done/1e6:8.1f}/{total/1e6:.1f} MB  {rate:.2f} MB/s  ETA {eta:.1f} min",
                      flush=True)
                last = now
    if part.stat().st_size != total:
        raise IOError(f"{label}: copied {part.stat().st_size} B but source has {total} B")
    part.replace(dst)
    secs = time.time() - t0
    return {"source": str(src), "local": str(dst), "bytes": total, "sha256": h.hexdigest(),
            "source_mtime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(src.stat().st_mtime)),
            "copied_at": time.strftime("%Y-%m-%d %H:%M:%S"), "seconds": round(secs, 1),
            "mb_per_s": round(total / max(secs, 1e-6) / 1e6, 3)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--only", nargs="*", help="large-file keys to copy (default: all)")
    ap.add_argument("--small-only", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load((CONFIG_DIR / "staging.yaml").read_text())
    root = Path(cfg["share_root"])
    man = load_manifest()

    jobs = [(rel, rel) for rel in cfg["small"]]
    if not args.small_only:
        keys = args.only or list(cfg["large"])
        jobs += [(key, cfg["large"][key]) for key in keys]

    for key, rel in jobs:
        src = root / rel
        dst = RAW_CACHE / rel
        prev = man.get(key)
        if prev and dst.exists() and dst.stat().st_size == prev["bytes"] == src.stat().st_size:
            print(f"skip {key} (already staged, sha256 {prev['sha256'][:12]})", flush=True)
            continue
        print(f"copy {key}: {src} ({src.stat().st_size/1e6:.1f} MB)", flush=True)
        man[key] = copy_with_hash(src, dst, key)
        save_manifest(man)
        print(f"  done {key}: {man[key]['mb_per_s']} MB/s, sha256 {man[key]['sha256'][:12]}", flush=True)
    print("STAGING COMPLETE", flush=True)


if __name__ == "__main__":
    shutil.disk_usage(str(RAW_CACHE))  # fail fast if the cache folder is missing
    main()

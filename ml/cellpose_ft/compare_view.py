#!/usr/bin/env python
"""Side-by-side view of one crop under each segmentation variant, with a zoomed inset.

Top row: whole 1024^2 crop per variant (GFP gamma 0.5, per-cell colored outlines).
Bottom row: the same zoom window for every variant, auto-placed where the variants
disagree most (or set with --zoom y,x). Reusable for base-vs-finetuned comparisons:
any npz of label images keyed by variant name works.

Usage:
    python compare_view.py <export_dir> <tag> [--variants prod,d25,d50,d75] [--zoom Y,X]
                           [--zsize 320] [--out PNG] [--extra NAME=path.npz:key ...]
"""
import argparse
import colorsys
import os

import numpy as np
import pandas as pd
import tifffile
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage
from skimage.segmentation import find_boundaries

PANEL = 640  # display size of each full-crop panel


def lut(n: int, seed: int = 0) -> np.ndarray:
    """Saturated random hue per label (0 = black)."""
    rng = np.random.RandomState(seed)
    c = np.array([colorsys.hsv_to_rgb(h, 1.0, 1.0) for h in rng.rand(n + 1)]) * 255
    c = c.astype(np.uint8); c[0] = 0
    return c


def render(gray: np.ndarray, masks: np.ndarray, thick: int) -> np.ndarray:
    """Gray image with colored mask outlines, drawn at native resolution."""
    rgb = np.repeat(gray[..., None], 3, 2).copy()
    b = find_boundaries(masks, mode="inner")
    if thick > 1:
        b = ndimage.binary_dilation(b, iterations=thick - 1)
    rgb[b] = lut(int(masks.max()))[ndimage.grey_dilation(masks, size=thick)][b] if thick > 1 \
        else lut(int(masks.max()))[masks][b]
    return rgb


def disagreement_window(preds: dict, zsize: int) -> tuple:
    """Top-left of the zsize window where variant foregrounds disagree most."""
    fgs = np.stack([(m > 0) for m in preds.values()]).astype(np.float32)
    dis = fgs.max(0) - fgs.min(0)
    # also count boundary disagreement (splits/merges) via edge maps
    edges = np.stack([find_boundaries(m, mode="inner") for m in preds.values()]).astype(np.float32)
    dis += 0.5 * (edges.max(0) - edges.min(0))
    score = ndimage.uniform_filter(dis, zsize, mode="constant")
    h = zsize // 2
    score[:h] = score[-h:] = 0; score[:, :h] = score[:, -h:] = 0
    cy, cx = np.unravel_index(np.argmax(score), score.shape)
    return int(cy - h), int(cx - h)


def label(im: Image.Image, text: str, size: int = 18) -> None:
    d = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size)
    except OSError:
        font = ImageFont.load_default()
    d.rectangle([0, 0, im.width, size + 10], fill=(0, 0, 0))
    d.text((8, 4), text, fill=(255, 255, 255), font=font)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("export_dir"); ap.add_argument("tag")
    ap.add_argument("--variants", default="prod,d25,d50,d75")
    ap.add_argument("--zoom", default=None, help="Y,X top-left of zoom window (crop coords)")
    ap.add_argument("--zsize", type=int, default=320)
    ap.add_argument("--extra", action="append", default=[], help="NAME=path.npz:key")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    man = pd.read_csv(os.path.join(a.export_dir, "manifest.csv")).set_index("tag")
    r = man.loc[a.tag]
    raw = tifffile.imread(os.path.join(a.export_dir, "crops", f"{a.tag}.tif")).astype(np.float32)
    gray = (np.clip((raw - r.lo) / (r.hi - r.lo + 1e-9), 0, 1) ** 0.5 * 255).astype(np.uint8)
    npz = np.load(os.path.join(a.export_dir, "preds", f"{a.tag}.npz"))
    names = a.variants.split(",")
    preds = {v: npz[v] for v in names}
    for e in a.extra:
        name, rest = e.split("=", 1); path, key = rest.rsplit(":", 1)
        preds[name] = np.load(path)[key]; names.append(name)

    mets = pd.read_csv(os.path.join(a.export_dir, "metrics.csv"))
    mets = mets[mets.tag == a.tag].set_index("variant")
    desc = {"prod": "current recipe (d25 + filter)", "d25": "d25, no filter",
            "d50": "0.6x scale", "d75": "0.4x scale"}

    zs = a.zsize
    zy, zx = (map(int, a.zoom.split(","))) if a.zoom else disagreement_window(preds, zs)
    zy = min(max(zy, 0), raw.shape[0] - zs); zx = min(max(zx, 0), raw.shape[1] - zs)

    cols = []
    for v in names:
        m = preds[v]
        full = Image.fromarray(render(gray, m, thick=3)).resize((PANEL, PANEL), Image.LANCZOS)
        d = ImageDraw.Draw(full); s = PANEL / raw.shape[0]
        d.rectangle([zx * s, zy * s, (zx + zs) * s, (zy + zs) * s], outline=(255, 255, 255), width=2)
        n = int(len(np.unique(m)) - 1)
        extra = f"  cov {mets.loc[v, 'fg_coverage']:.2f}" if v in mets.index else ""
        label(full, f"{v}: {desc.get(v, v)} | n={n}{extra}")
        zoom = Image.fromarray(render(gray[zy:zy + zs, zx:zx + zs], m[zy:zy + zs, zx:zx + zs], thick=2)
                               ).resize((PANEL, PANEL), Image.NEAREST)
        label(zoom, f"{v}: zoom {zs}px at ({zy},{zx})")
        col = Image.new("RGB", (PANEL, 2 * PANEL + 6), (40, 40, 40))
        col.paste(full, (0, 0)); col.paste(zoom, (0, PANEL + 6))
        cols.append(col)
    raw_col = Image.new("RGB", (PANEL, 2 * PANEL + 6), (40, 40, 40))
    g_full = Image.fromarray(gray).resize((PANEL, PANEL), Image.LANCZOS).convert("RGB")
    label(g_full, f"GFP16 raw (gamma 0.5) {r.line} {r.dose} {r.tp}")
    g_zoom = Image.fromarray(gray[zy:zy + zs, zx:zx + zs]).resize((PANEL, PANEL), Image.NEAREST).convert("RGB")
    label(g_zoom, "GFP16 raw, zoom")
    raw_col.paste(g_full, (0, 0)); raw_col.paste(g_zoom, (0, PANEL + 6))
    cols.insert(0, raw_col)

    out = Image.new("RGB", (len(cols) * (PANEL + 6) - 6, 2 * PANEL + 6 + 36), (40, 40, 40))
    for i, c in enumerate(cols):
        out.paste(c, (i * (PANEL + 6), 36))
    label(out, f"{a.tag}  ({r.kind}, {r.split})  — outlines = one color per Cellpose mask; "
               f"white box = zoom region (where variants disagree most)", size=20)
    dst = a.out or os.path.join(a.export_dir, "compare", f"{a.tag}_variants.png")
    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    out.save(dst)
    print(dst)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""End-to-end per-cell neurite quantification: somas x neurites -> length/cell.

This is the integration step that turns the two trained halves of the pipeline
into the actual biological readout:

1. **Somas** — Cellpose-SAM on the FITC morphology image with the locked recipe
   (diameter=25, cellprob=-1.0, flow=0.6), intensity cleanup (drop faint false
   positives), and a size-dependent debris floor ``max(300, 0.6*median area)``.
2. **Neurites** — the trained clDice U-Net probability map, thresholded.
3. **Attribution (soma-rooted, geodesic ownership)** — each neurite pixel is
   assigned to the soma reachable by the shortest path *through the neurite mask*
   (a watershed seeded from the somas, constrained to the neurite foreground).
   Neurites in a component that contains no soma get owner 0 and are dropped:
   this enforces Austin's "a neurite must root in a labelled soma" rule and
   rejects background texture / unlabelled-cell ghosting for free.
4. **Length** — geometric skeleton length (orthogonal + sqrt(2)*diagonal steps)
   of each soma's attributed neurite skeleton, excluding pixels inside soma
   bodies.

Outputs, per input tile ``<name>``:

* one row per soma appended to ``percell.csv`` (experiment, well, line, genotype,
  soma_id, area_px, cy, cx, neurite_len_px, skel_px);
* ``<name>_percell.tif`` — gray image + soma outlines (green) + neurite skeleton
  colour-coded by owning soma.

The model/checkpoint is loaded via the sibling ``model.py`` (embedded alongside
this file at run time); no dependency on the DB or the Nextflow layer.
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
from typing import Dict, List, Tuple

import numpy as np
import tifffile
import torch
from cellpose import models
from scipy import ndimage
from scipy.ndimage import convolve
from skimage.measure import regionprops
from skimage.morphology import (binary_dilation, disk, remove_small_objects,
                                skeletonize)
from skimage.segmentation import find_boundaries, watershed

from model import build_model

CTR_LINES = {"33114.C", "33113.2I", "33362.C", "32517.I"}
XDP_LINES = {"33109.2B", "33363.D"}


# ----------------------------- somas ---------------------------------------
def clean_masks(raw: np.ndarray, masks: np.ndarray, k: float) -> np.ndarray:
    """Drop Cellpose cells at background intensity (dim false positives).

    Args:
        raw: Raw morphology image (H, W).
        masks: Cellpose integer label image (H, W).
        k: Keep a cell only if its median >= bg median + k*MAD.

    Returns:
        Filtered label image.
    """
    bg = raw[masks == 0]
    if bg.size == 0 or masks.max() == 0:
        return masks
    bg_med = np.median(bg)
    bg_mad = 1.4826 * np.median(np.abs(bg - bg_med))
    thr = bg_med + k * bg_mad
    nlab = int(masks.max())
    # Per-label median intensity in one C-level pass (no Python per-label loop).
    meds = ndimage.labeled_comprehension(
        raw, masks, np.arange(1, nlab + 1), np.median, np.float64, 0.0)
    keep = np.zeros(nlab + 1, dtype=bool)
    keep[1:] = meds >= thr
    return np.where(keep[masks], masks, 0)


def segment_somas(model: "models.CellposeModel", raw: np.ndarray,
                  diameter: int, flow: float, cellprob: float,
                  clean_k: float) -> np.ndarray:
    """Cellpose-SAM somas with cleanup + size-dependent debris floor.

    Args:
        model: A ``CellposeModel``.
        raw: Raw morphology image (H, W), float32.
        diameter: Cellpose cell diameter.
        flow: Flow threshold.
        cellprob: Cell-probability threshold.
        clean_k: Intensity cleanup strength (see :func:`clean_masks`).

    Returns:
        Relabeled soma image (1..N), debris removed.
    """
    lo, hi = np.percentile(raw, 1), np.percentile(raw, 99.5)
    norm = np.clip((raw - lo) / (hi - lo + 1e-9), 0, 1) * 255.0 if hi > lo \
        else np.zeros_like(raw)
    masks = model.eval(norm, diameter=diameter, flow_threshold=flow,
                       cellprob_threshold=cellprob)[0]
    masks = clean_masks(raw, masks, clean_k)
    if masks.max() == 0:
        return masks.astype(np.int32)
    # Areas via bincount; size-dependent debris floor; relabel with a LUT
    # (all O(image), no per-label Python loop).
    counts = np.bincount(masks.ravel())
    counts[0] = 0
    nonzero = counts[counts > 0]
    floor = max(300.0, 0.6 * float(np.median(nonzero))) if nonzero.size else 300.0
    keep = counts >= floor
    remap = np.zeros(counts.size, dtype=np.int32)
    remap[keep] = np.arange(1, int(keep.sum()) + 1, dtype=np.int32)
    return remap[masks]


# ----------------------------- neurites ------------------------------------
def percentile_normalize(image: np.ndarray) -> np.ndarray:
    """1-99.5 percentile normalization to [0, 1] (matches training)."""
    x = np.clip(image.astype(np.float32), 0, None)
    lo, hi = np.percentile(x, 1), np.percentile(x, 99.5)
    return np.clip((x - lo) / (hi - lo), 0, 1) if hi > lo else np.zeros_like(x)


def load_model(checkpoint: str, device: torch.device):
    """Load a trained U-Net checkpoint (config-driven)."""
    ckpt = torch.load(checkpoint, map_location=device)
    cfg = ckpt.get("config", {"depth": 4, "base_channels": 32, "in_channels": 1})
    model = build_model(depth=cfg.get("depth", 4),
                        base_channels=cfg.get("base_channels", 32),
                        in_channels=cfg.get("in_channels", 1)).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


@torch.no_grad()
def predict_tiled(model, image01: np.ndarray, device: torch.device,
                  tile: int = 512, halo: int = 64) -> np.ndarray:
    """Sliding-window sigmoid probability map (bounds GPU memory on big tiles).

    Args:
        model: Loaded U-Net (eval mode).
        image01: Normalized image (H, W) in [0, 1].
        device: Torch device.
        tile: Interior tile size.
        halo: Reflect-padded context added on each side before inference.

    Returns:
        Float32 probability map (H, W).
    """
    H, W = image01.shape
    mult = model.valid_multiple()
    out = np.zeros((H, W), np.float32)
    step = tile
    for y0 in range(0, H, step):
        for x0 in range(0, W, step):
            y1, x1 = min(y0 + tile, H), min(x0 + tile, W)
            ry0, rx0 = max(0, y0 - halo), max(0, x0 - halo)
            ry1, rx1 = min(H, y1 + halo), min(W, x1 + halo)
            patch = image01[ry0:ry1, rx0:rx1]
            ph = (mult - patch.shape[0] % mult) % mult
            pw = (mult - patch.shape[1] % mult) % mult
            pad = np.pad(patch, ((0, ph), (0, pw)), mode="reflect")
            t = torch.from_numpy(pad)[None, None].to(device)
            pr = torch.sigmoid(model(t))[0, 0].cpu().numpy()
            pr = pr[:patch.shape[0], :patch.shape[1]]
            out[y0:y1, x0:x1] = pr[y0 - ry0:y1 - ry0, x0 - rx0:x1 - rx0]
    return out


# ----------------------------- attribution ---------------------------------
def diaglen(sk: np.ndarray) -> float:
    """Geometric length of a 1-px skeleton (ortho + sqrt(2)*diagonal steps)."""
    o = convolve(sk.astype(np.uint8),
                 np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]),
                 mode="constant") * sk
    d = convolve(sk.astype(np.uint8),
                 np.array([[1, 0, 1], [0, 0, 0], [1, 0, 1]]),
                 mode="constant") * sk
    return float(o.sum() / 2 + (d.sum() / 2) * np.sqrt(2))


def attribute(soma_labels: np.ndarray, neurite_mask: np.ndarray
              ) -> Tuple[np.ndarray, np.ndarray]:
    """Assign each neurite skeleton pixel to the soma it connects to.

    Geodesic ownership: a watershed seeded from soma labels, constrained to
    travel only through the neurite foreground (union with somas so processes
    reach their cell body). Skeleton pixels inside soma bodies are dropped.

    Args:
        soma_labels: Integer soma image (H, W), 0=background.
        neurite_mask: Boolean neurite mask (H, W).

    Returns:
        ``(owner, skel_neurite)``: ``owner`` is the soma id owning each fg pixel
        (0 where unreachable from any soma), ``skel_neurite`` is the attributed
        neurite skeleton (bool, soma interiors removed).
    """
    fg = neurite_mask | (soma_labels > 0)
    # Flat landscape -> watershed distributes fg by geodesic distance to the
    # nearest soma marker WITHIN the connected mask; unreached fg stays 0.
    owner = watershed(np.zeros_like(soma_labels, dtype=np.uint8),
                      markers=soma_labels, mask=fg)
    skel = skeletonize(neurite_mask)
    skel_neurite = skel & (soma_labels == 0)
    return owner, skel_neurite


def per_soma_lengths(owner: np.ndarray, skel_neurite: np.ndarray,
                     soma_labels: np.ndarray) -> Dict[int, Tuple[float, int]]:
    """Total neurite length + skeleton px per soma (vectorized).

    Computes each skeleton pixel's local length contribution ONCE for the whole
    image (two convolutions), then sums by owner with ``bincount``. This is
    O(image) rather than O(image x n_somas) -- essential at montage scale where
    a per-soma full-image ``diaglen`` is pathologically slow. The total across
    somas equals the plain :func:`diaglen` of the full skeleton.

    Args:
        owner: Geodesic owner id per pixel (from :func:`attribute`).
        skel_neurite: Attributed neurite skeleton (bool).
        soma_labels: Integer soma image.

    Returns:
        ``{soma_id: (length_px, skel_px)}`` for every soma (0 if no neurites).
    """
    sk = skel_neurite.astype(np.uint8)
    ortho = convolve(sk, np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]),
                     mode="constant") * sk
    diag = convolve(sk, np.array([[1, 0, 1], [0, 0, 0], [1, 0, 1]]),
                    mode="constant") * sk
    # per-pixel length contribution: half of each incident ortho/diag edge.
    weight = 0.5 * ortho + 0.5 * np.sqrt(2) * diag
    ids = (owner * skel_neurite).ravel()
    maxid = int(soma_labels.max())
    length = np.bincount(ids, weights=weight.ravel(), minlength=maxid + 1)
    skpx = np.bincount(ids, minlength=maxid + 1)
    return {lab: (float(length[lab]), int(skpx[lab]))
            for lab in range(1, maxid + 1)}


# ----------------------------- io / naming ---------------------------------
def parse_name(name: str) -> Tuple[str, str, str, str]:
    """Parse experiment/well/line/genotype from a tile filename.

    Example: ``XDP10-RGEDI_F02_T6_t2__33113.2I`` ->
    ``("XDP10", "F02", "33113.2I", "CTR")``.

    Args:
        name: Tile stem (no extension).

    Returns:
        ``(experiment, well, cell_line, genotype)``. Unknown fields -> "?".
    """
    exp, well, line = "?", "?", "?"
    head = name.split("__")[0]
    parts = head.split("_")
    if parts:
        exp = parts[0].split("-")[0]
    if len(parts) > 1:
        well = parts[1]
    if "__" in name:
        line = name.split("__")[1].split("_")[0]
    geno = "CTR" if line in CTR_LINES else ("XDP" if line in XDP_LINES else "?")
    return exp, well, line, geno


def colorize(image01: np.ndarray, soma_labels: np.ndarray,
             owner: np.ndarray, skel_neurite: np.ndarray) -> np.ndarray:
    """RGB overlay: gray image, green soma outlines, per-owner neurite colours."""
    g = (np.clip(image01, 0, 1) * 255).astype(np.uint8)
    rgb = np.dstack([g, g, g])
    ow = (owner * skel_neurite).astype(np.int32)
    maxid = int(soma_labels.max())
    # Per-owner colour lookup table; thicken labels with ONE grey dilation
    # (no per-soma full-image dilation loop).
    rng = np.random.default_rng(0)
    lut = rng.integers(80, 256, size=(maxid + 1, 3)).astype(np.uint8)
    lut[0] = 0
    thick = ndimage.grey_dilation(ow, size=(3, 3))
    m = thick > 0
    rgb[m] = lut[thick[m]]
    rgb[find_boundaries(soma_labels, mode="inner")] = [0, 255, 0]
    return rgb


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in-dir", required=True, help="dir of morphology tiles (.tif)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--diameter", type=int, default=25)
    ap.add_argument("--flow", type=float, default=0.6)
    ap.add_argument("--cellprob", type=float, default=-1.0)
    ap.add_argument("--clean-k", type=float, default=2.0)
    ap.add_argument("--min-neurite", type=int, default=25,
                    help="drop neurite mask components smaller than this (px)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)
    cp_model = models.CellposeModel(gpu=(args.device == "cuda"))
    unet = load_model(args.checkpoint, device)

    csv_path = os.path.join(args.out_dir, "percell.csv")
    fields = ["experiment", "well", "cell_line", "genotype", "tile", "soma_id",
              "area_px", "cy", "cx", "neurite_len_px", "skel_px"]
    fh = open(csv_path, "w", newline="")
    writer = csv.DictWriter(fh, fieldnames=fields)
    writer.writeheader()

    tiles = sorted(glob.glob(os.path.join(args.in_dir, "*.tif")))
    print(f"found {len(tiles)} tiles")
    print(f"{'tile':44}{'somas':>7}{'len/soma':>10}{'geno':>6}")
    for f in tiles:
        name = os.path.splitext(os.path.basename(f))[0]
        exp, well, line, geno = parse_name(name)
        raw = tifffile.imread(f).astype(np.float32)
        if raw.ndim == 3:
            raw = raw[..., 0]
        soma_labels = segment_somas(cp_model, raw, args.diameter, args.flow,
                                    args.cellprob, args.clean_k)
        image01 = percentile_normalize(raw)
        prob = predict_tiled(unet, image01, device)
        neurite_mask = remove_small_objects(prob >= args.threshold, args.min_neurite)
        owner, skel_neurite = attribute(soma_labels, neurite_mask)
        lengths = per_soma_lengths(owner, skel_neurite, soma_labels)

        props = {p.label: p for p in regionprops(soma_labels)}
        total_len = 0.0
        for lab, (length, skpx) in lengths.items():
            p = props.get(lab)
            if p is None:
                continue
            cy, cx = p.centroid
            writer.writerow(dict(experiment=exp, well=well, cell_line=line,
                                 genotype=geno, tile=name, soma_id=lab,
                                 area_px=int(p.area), cy=round(cy, 1), cx=round(cx, 1),
                                 neurite_len_px=round(length, 1), skel_px=skpx))
            total_len += length
        n = int(soma_labels.max())
        print(f"{name:44}{n:7d}{(total_len / max(n, 1)):10.1f}{geno:>6}")
        tifffile.imwrite(os.path.join(args.out_dir, f"{name}_percell.tif"),
                         colorize(image01, soma_labels, owner, skel_neurite)[::2, ::2])
    fh.close()
    print(f"\nDONE. per-cell table -> {csv_path}")


if __name__ == "__main__":
    main()

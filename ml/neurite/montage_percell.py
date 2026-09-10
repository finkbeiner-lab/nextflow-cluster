#!/usr/bin/env python3
"""Whole-well montage per-cell neurite quantification (CP tool-to-tool batch).

For each well in a manifest, this rebuilds a whole-well montage DIRECTLY FROM
RAW IXM tiles (CellProfiler used individually background-subtracted tiles, which
we deliberately do not use), then runs the trained pipeline on it:

* **Montage geometry** — 16 sites in a 4x4 ``standard`` raster grid, edge-to-edge
  (overlap measured ~0: IXM fields are contiguous, so edge-to-edge keeps neurites
  continuous across seams; verified against CP's own BGs_MN montage).
* **Per-tile normalization** — each raw tile is percentile-normalized (1-99.5)
  BEFORE stitching. This removes tile-to-tile vignetting seams and matches exactly
  how the clDice model was trained (on per-tile percentile-normalized crops).
* **Somas** — Cellpose-SAM per tile (locked recipe), labels offset to be globally
  unique, placed into the montage canvas (per-tile keeps Cellpose at the 2048px
  scale it was tuned at).
* **Neurites** — clDice U-Net over the full montage (tiled inference), so
  processes crossing tile seams are detected continuously.
* **Attribution** — geodesic soma-rooted ownership (see percell_integrate).

Outputs: ``percell.csv`` (one row per soma, whole-well) + ``perwell.csv``
(aggregates) + per-well downsampled overlays. These lay directly against CP's
``Neurite.csv`` per-object skeleton lengths for the head-to-head.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
from typing import Dict, List, Tuple

import numpy as np
import tifffile
import torch
from cellpose import models
from scipy import ndimage
from skimage.measure import regionprops
from skimage.morphology import remove_small_objects

from percell_integrate import (attribute, colorize, load_model, parse_name,
                               percentile_normalize, per_soma_lengths,
                               predict_tiled, segment_somas)

GRID = 4  # 4x4 = 16 tiles


def well_tiles(welldir: str, well: str, tp: str) -> Tuple[List[str], List[int]]:
    """List a well's FITC tiles at one timepoint, ordered by site 1..16.

    Args:
        welldir: Directory of raw tiles for the well.
        well: Well id (e.g. ``"J03"``).
        tp: Timepoint token (e.g. ``"T9"``).

    Returns:
        ``(paths, sites)`` sorted ascending by site index.
    """
    pat = re.compile(rf'_{well}_(\d+)_FITC_')
    fs = [f for f in glob.glob(f"{welldir}/*_{tp}_*_{well}_*_FITC_*.tif") if pat.search(f)]
    fs.sort(key=lambda f: int(pat.search(f).group(1)))
    return fs, [int(pat.search(f).group(1)) for f in fs]


def flatten_tile(raw: np.ndarray, size: int) -> np.ndarray:
    """Data-driven rolling-background subtraction (pseudo flat-field).

    Estimates the smooth low-frequency illumination (vignetting + background
    pedestal) with a morphological opening on a downsampled copy -- which ignores
    small bright objects (somas, thin neurites) so they are NOT subtracted -- then
    upsamples and subtracts it. No measured flat-field reference required. Applied
    to the neurite-image path only; Cellpose somas run on the untouched raw so
    this isolates the neurite-segmentation variable.

    Args:
        raw: Raw tile (H, W), float32.
        size: Approximate full-resolution footprint (px) of the illumination
            scale to remove; must exceed the largest object (soma) diameter so
            objects survive. Typical ~128.

    Returns:
        Background-subtracted tile (H, W), float32, clipped at 0.
    """
    down = 4
    small = raw[::down, ::down]
    fp = max(3, size // down)
    bg = ndimage.grey_opening(small, size=(fp, fp))
    bg = ndimage.gaussian_filter(bg, sigma=fp / 2.0)
    bg = ndimage.zoom(bg, (raw.shape[0] / bg.shape[0], raw.shape[1] / bg.shape[1]),
                      order=1)
    return np.clip(raw - bg, 0, None).astype(np.float32)


def place(canvas: np.ndarray, tile: np.ndarray, site: int, th: int, tw: int) -> None:
    """Place a tile into the montage canvas at its standard-raster grid cell.

    Args:
        canvas: Montage array to write into (modified in place).
        tile: Tile image (th, tw).
        site: 1-based site index (1..16).
        th: Tile height.
        tw: Tile width.
    """
    r, c = (site - 1) // GRID, (site - 1) % GRID
    canvas[r * th:(r + 1) * th, c * tw:(c + 1) * tw] = tile


def build_montages(cp_model, paths: List[str], sites: List[int],
                   diameter: int, flow: float, cellprob: float, clean_k: float,
                   flatten_size: int = 0
                   ) -> Tuple[np.ndarray, np.ndarray]:
    """Build the (normalized image, soma-label) montages for one well.

    Args:
        cp_model: Cellpose model.
        paths: Tile paths ordered by site.
        sites: Site indices matching ``paths``.
        diameter/flow/cellprob/clean_k: Cellpose soma-recipe params.

    Returns:
        ``(image01, soma_labels)`` montages, both (GRID*th, GRID*tw).
    """
    # Read all 16 tiles once (16 x 2048^2 float32 ~= 256MB, fine).
    tiles = []
    for p in paths:
        raw = tifffile.imread(p).astype(np.float32)
        if raw.ndim == 3:
            raw = raw[..., 0]
        tiles.append(raw)
    th, tw = tiles[0].shape
    H, W = GRID * th, GRID * tw

    # Neurite-image source: optional data-driven rolling-background flattening
    # (vignetting / illumination correction) applied per tile BEFORE percentile
    # normalization. Cellpose somas still run on the untouched raw below, so this
    # only affects the neurite-segmentation input.
    img_src = ([flatten_tile(t, flatten_size) for t in tiles]
               if flatten_size > 0 else tiles)

    # Empty-tile guard: percentile-normalizing a near-signal-free tile amplifies
    # its noise into a fake bright band. Compare each tile's dynamic range to the
    # montage-wide median range; if a tile has < EMPTY_FRAC of it, it carries no
    # real signal -> emit zeros instead of stretching noise. Computed on the same
    # domain (flattened or raw) that feeds the image montage.
    EMPTY_FRAC = 0.15
    ranges = np.array([np.percentile(t, 99.5) - np.percentile(t, 1) for t in img_src])
    global_range = float(np.median(ranges))

    image01 = np.zeros((H, W), np.float32)
    soma = np.zeros((H, W), np.int32)
    offset = 0
    for raw, imgt, s in zip(tiles, img_src, sites):
        lo, hi = np.percentile(imgt, 1), np.percentile(imgt, 99.5)
        if global_range <= 0 or (hi - lo) < EMPTY_FRAC * global_range:
            norm = np.zeros_like(imgt)  # empty/low-signal tile: do not amplify
        else:
            norm = np.clip((imgt - lo) / (hi - lo), 0, 1).astype(np.float32)
        place(image01, norm, s, th, tw)
        labels = segment_somas(cp_model, raw, diameter, flow, cellprob, clean_k)
        if labels.max() > 0:
            lab = labels.copy()
            lab[labels > 0] += offset
            offset = int(lab.max())
            place(soma, lab, s, th, tw)
    return image01, soma


# --------------------------- arborization ----------------------------------
# Per-cell branching metrics on the attributed neurite skeleton (NumPy/scipy
# only -- no skan). Branch points are counted as NODES (adjacent degree>=3
# skeleton pixels clustered into one), and tips exclude the soma-attachment
# roots, so the numbers read as anatomy: a straight process = 0 branch points /
# 1 tip, a single Y = 1 branch point / 2 tips.
_K8 = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
_S8 = np.ones((3, 3))


def _arbor_metrics(cell_skel: np.ndarray, soma: np.ndarray,
                   dilation: int) -> Tuple[int, int, int, float]:
    """Arborization of one cell's attributed neurite skeleton.

    Args:
        cell_skel: Boolean skeleton for this cell (soma interior already removed).
        soma: Boolean mask of this cell's soma body.
        dilation: Soma-dilation radius used to build the emergence ring.

    Returns:
        ``(n_branch_points, n_tips, n_primary_neurites, max_branch_length)``:
        bifurcation nodes, terminal tips (excluding soma roots), branches
        leaving the soma, and the longest path from the soma (px).
    """
    from scipy import ndimage as ndi
    from skimage.morphology import binary_dilation, disk
    if cell_skel.sum() == 0:
        return 0, 0, 0, 0.0
    nb = ndi.convolve(cell_skel.astype(np.uint8), _K8, mode="constant") * cell_skel
    # branch NODES: cluster adjacent degree>=3 pixels so one junction counts once
    n_branch = int(ndi.label(nb >= 3, structure=_S8)[1])
    ring = binary_dilation(soma, disk(max(1, dilation) + 1)) & ~soma
    # tips: degree-1 endpoints that are NOT the soma-attachment roots
    n_tips = int(np.sum((nb == 1) & ~ring))
    # primary neurites: skeleton components crossing the soma ring
    emerging = cell_skel & ring
    n_primary = int(ndi.label(emerging, structure=_S8)[1]) if emerging.any() else 0
    return n_branch, n_tips, n_primary, _max_path_from_soma(cell_skel, soma)


def _max_path_from_soma(cell_skel: np.ndarray, soma: np.ndarray) -> float:
    """Longest geodesic distance along the skeleton from the soma (px).

    Args:
        cell_skel: Boolean skeleton for this cell.
        soma: Boolean mask of this cell's soma body.

    Returns:
        Longest shortest-path length (px) reachable along the skeleton.
    """
    from collections import deque
    from scipy import ndimage as ndi
    if cell_skel.sum() == 0:
        return 0.0
    seed = cell_skel & ndi.binary_dilation(soma, structure=np.ones((3, 3)))
    if not seed.any():
        return 0.0
    dist = np.full(cell_skel.shape, -1, dtype=np.int32)
    q = deque(zip(*np.where(seed)))
    for y, x in list(q):
        dist[y, x] = 0
    maxd, H, W = 0, cell_skel.shape[0], cell_skel.shape[1]
    while q:
        y, x = q.popleft()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                ny, nx = y + dy, x + dx
                if 0 <= ny < H and 0 <= nx < W and cell_skel[ny, nx] and dist[ny, nx] < 0:
                    dist[ny, nx] = dist[y, x] + 1
                    maxd = max(maxd, dist[ny, nx])
                    q.append((ny, nx))
    return float(maxd)


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True,
                    help="CSV: experiment,well,timepoint,cell_line,genotype")
    ap.add_argument("--raw-root", required=True,
                    help="root holding <EXP>-RGEDI/<WELL>/ raw tiles")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--diameter", type=int, default=25)
    ap.add_argument("--flow", type=float, default=0.6)
    ap.add_argument("--cellprob", type=float, default=-1.0)
    ap.add_argument("--clean-k", type=float, default=2.0)
    ap.add_argument("--min-neurite", type=int, default=25)
    ap.add_argument("--seam-band", type=int, default=6,
                    help="zero neurite detections within this many px of each "
                         "interior tile seam (kills per-tile-normalization edge "
                         "artifacts); 0 disables")
    ap.add_argument("--flatten-size", type=int, default=0,
                    help="data-driven rolling-background (pseudo flat-field) "
                         "footprint in px applied to the neurite image before "
                         "normalization; ~128 corrects vignetting, 0 disables")
    ap.add_argument("--soma-dilation", type=int, default=3,
                    help="soma-dilation radius (px) used to count primary "
                         "neurites emerging from each soma (arborization)")
    ap.add_argument("--cache-dir", default="",
                    help="if set, save per-well {soma_labels(uint16), "
                         "neurite_prob(uint8), meta} .npz here and reuse it on "
                         "re-run so metrics recompute on CPU with NO Cellpose / "
                         "U-Net -- enables resume and later re-thresholding "
                         "without the GPU")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    if args.cache_dir:
        os.makedirs(args.cache_dir, exist_ok=True)
    device = torch.device(args.device)

    # Lazy model init: Cellpose + the U-Net are built only when a well actually
    # needs the GPU (a cache miss). A re-run whose wells are all cached touches
    # no GPU and runs on a CPU node -- that is the "recompute from cache" path
    # (re-threshold, new metrics) with no cluster GPU.
    _models = {}
    def ensure_models():
        if "cp" not in _models:
            cp_model = models.CellposeModel(gpu=(args.device == "cuda"), pretrained_model="cpsam", use_bfloat16=False)
            _models["cp"] = cp_model
            _models["unet"] = load_model(args.checkpoint, device)
        return _models["cp"], _models["unet"]

    manifest = list(csv.DictReader(open(args.manifest)))
    print(f"manifest: {len(manifest)} wells")

    cell_fh = open(os.path.join(args.out_dir, "percell.csv"), "w", newline="")
    cfields = ["experiment", "well", "cell_line", "genotype", "soma_id",
               "area_px", "cy", "cx", "neurite_len_px", "skel_px",
               "n_branch_points", "n_end_points", "n_primary_neurites",
               "max_branch_length"]
    cwriter = csv.DictWriter(cell_fh, fieldnames=cfields)
    cwriter.writeheader()
    well_fh = open(os.path.join(args.out_dir, "perwell.csv"), "w", newline="")
    wfields = ["experiment", "well", "cell_line", "genotype", "n_somas",
               "n_somas_with_neurite", "total_len_px", "mean_len_per_soma",
               "median_len_per_soma", "mean_branch_points", "mean_tips",
               "mean_primary_neurites", "mean_max_branch_len", "frac_branched"]
    wwriter = csv.DictWriter(well_fh, fieldnames=wfields)
    wwriter.writeheader()

    print(f"{'experiment/well':18}{'line':10}{'geno':>5}{'somas':>7}{'len/soma':>10}")
    for row in manifest:
        exp, well, tp = row["experiment"], row["well"], row["timepoint"]
        welldir = os.path.join(args.raw_root, (row.get("raw_folder") or f"{exp}-RGEDI"), well)
        paths, sites = well_tiles(welldir, well, tp)
        if len(paths) != 16:
            print(f"{exp}/{well:14}  SKIP: found {len(paths)} FITC tiles (need 16)")
            continue
        # Per-well try/except: one bad well (corrupt tile, transient CUDA error,
        # etc.) must NOT abort the remaining wells. Completed wells are already
        # flushed to disk; a failed well is logged and skipped.
        try:
            cache_path = (os.path.join(args.cache_dir, f"{exp}_{well}_{tp}.npz")
                          if args.cache_dir else None)
            if cache_path and os.path.exists(cache_path):
                # Reuse saved intermediates -> recompute metrics on CPU (no
                # Cellpose / U-Net). Overlays are not regenerated (image not cached).
                z = np.load(cache_path)
                soma = z["soma_labels"].astype(np.int32)
                prob = z["neurite_prob"].astype(np.float32) / 255.0
                image01, from_cache = None, True
            else:
                cp_model, unet = ensure_models()
                image01, soma = build_montages(cp_model, paths, sites, args.diameter,
                                               args.flow, args.cellprob, args.clean_k,
                                               flatten_size=args.flatten_size)
                prob = predict_tiled(unet, image01, device)
                from_cache = False
                if cache_path:  # write once, atomically (temp then rename)
                    tmp = cache_path + ".tmp.npz"
                    np.savez_compressed(
                        tmp, soma_labels=soma.astype(np.uint16),
                        neurite_prob=(np.clip(prob, 0, 1) * 255).round().astype(np.uint8),
                        meta=np.array(json.dumps(dict(
                            experiment=exp, well=well, timepoint=tp,
                            cell_line=row.get("cell_line", "?"),
                            genotype=row.get("genotype", "?"), grid=GRID,
                            threshold=args.threshold, seam_band=args.seam_band,
                            min_neurite=args.min_neurite, flatten_size=args.flatten_size,
                            diameter=args.diameter, flow=args.flow,
                            cellprob=args.cellprob, clean_k=args.clean_k))))
                    os.replace(tmp, cache_path)
            raw_mask = prob >= args.threshold
            # Seam suppression: the per-tile normalization step at each interior
            # tile boundary reads as a straight edge that the ridge-sensitive
            # U-Net fires on (false long straight neurites). Zero a thin band at
            # each interior seam; real neurites crossing a seam lose only ~2*band
            # px (negligible for length). Outer edges are untouched.
            th_m, tw_m = soma.shape[0] // GRID, soma.shape[1] // GRID
            band = args.seam_band
            if band > 0:
                for kk in range(1, GRID):
                    raw_mask[kk * th_m - band:kk * th_m + band, :] = False
                    raw_mask[:, kk * tw_m - band:kk * tw_m + band] = False
            neurite_mask = remove_small_objects(raw_mask, args.min_neurite)
            owner, skel = attribute(soma, neurite_mask)
            lengths = per_soma_lengths(owner, skel, soma)
            props = {p.label: p for p in regionprops(soma)}

            # Per-cell arborization runs on a CROP, not the whole montage:
            # find_objects gives each owner label's bbox (soma + its attributed
            # neurites) in one pass, so the graph metrics cost O(sum of cell
            # bboxes), not O(n_cells * montage). The crop bounds every
            # owner==lab pixel plus a soma-ring pad, so metrics are identical to
            # a whole-image computation, just translated into a smaller array.
            owner_slices = ndimage.find_objects(owner)
            pad = int(args.soma_dilation) + 2
            Hh, Ww = owner.shape

            def cell_arbor(lab: int, skpx: int):
                sl = owner_slices[lab - 1] if lab - 1 < len(owner_slices) else None
                if skpx <= 0 or sl is None:
                    return 0, 0, 0, 0.0
                ys, xs = sl
                ysl = slice(max(0, ys.start - pad), min(Hh, ys.stop + pad))
                xsl = slice(max(0, xs.start - pad), min(Ww, xs.stop + pad))
                cell_skel = (owner[ysl, xsl] == lab) & skel[ysl, xsl]
                soma_c = soma[ysl, xsl] == lab
                return _arbor_metrics(cell_skel, soma_c, int(args.soma_dilation))

            line, geno = row.get("cell_line", "?"), row.get("genotype", "?")
            percell_lens = []
            a_branch, a_tips, a_prim, a_max = [], [], [], []
            for lab, (length, skpx) in lengths.items():
                p = props.get(lab)
                if p is None:
                    continue
                cy, cx = p.centroid
                nb, ne, npri, mx = cell_arbor(lab, skpx)
                cwriter.writerow(dict(experiment=exp, well=well, cell_line=line,
                                      genotype=geno, soma_id=lab, area_px=int(p.area),
                                      cy=round(cy, 1), cx=round(cx, 1),
                                      neurite_len_px=round(length, 1), skel_px=skpx,
                                      n_branch_points=nb, n_end_points=ne,
                                      n_primary_neurites=npri,
                                      max_branch_length=round(mx, 1)))
                percell_lens.append(length)
                if skpx > 0:  # arborization aggregates over neurite-bearing somas
                    a_branch.append(nb); a_tips.append(ne)
                    a_prim.append(npri); a_max.append(mx)
            n = int(soma.max())
            nwith = int(np.sum(np.array(percell_lens) > 0))
            total = float(np.sum(percell_lens))
            amean = lambda v: round(float(np.mean(v)), 2) if v else 0.0
            wwriter.writerow(dict(experiment=exp, well=well, cell_line=line, genotype=geno,
                                  n_somas=n, n_somas_with_neurite=nwith,
                                  total_len_px=round(total, 1),
                                  mean_len_per_soma=round(total / max(n, 1), 1),
                                  median_len_per_soma=round(float(np.median(percell_lens)) if percell_lens else 0.0, 1),
                                  mean_branch_points=amean(a_branch),
                                  mean_tips=amean(a_tips),
                                  mean_primary_neurites=amean(a_prim),
                                  mean_max_branch_len=amean(a_max),
                                  frac_branched=round(float(np.mean([b >= 1 for b in a_branch])), 3) if a_branch else 0.0))
            cell_fh.flush(); well_fh.flush()
            tag = " (cache)" if from_cache else ""
            print(f"{exp+'/'+well:18}{line:10}{geno:>5}{n:7d}{(total/max(n,1)):10.1f}{tag}")
            if image01 is not None:  # overlay only on a fresh compute (image not cached)
                tifffile.imwrite(os.path.join(args.out_dir, f"{exp}_{well}_{tp}_percell.tif"),
                                 colorize(image01, soma, owner, skel)[::4, ::4])
        except Exception as exc:  # noqa: BLE001 - keep the batch alive
            print(f"{exp}/{well}  FAILED: {type(exc).__name__}: {exc}", flush=True)
            continue
    cell_fh.close(); well_fh.close()
    print(f"\nDONE. per-cell + per-well tables in {args.out_dir}")


if __name__ == "__main__":
    main()

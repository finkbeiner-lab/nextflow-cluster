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
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)
    cp_model = models.CellposeModel(gpu=(args.device == "cuda"))
    unet = load_model(args.checkpoint, device)

    manifest = list(csv.DictReader(open(args.manifest)))
    print(f"manifest: {len(manifest)} wells")

    cell_fh = open(os.path.join(args.out_dir, "percell.csv"), "w", newline="")
    cfields = ["experiment", "well", "cell_line", "genotype", "soma_id",
               "area_px", "cy", "cx", "neurite_len_px", "skel_px"]
    cwriter = csv.DictWriter(cell_fh, fieldnames=cfields)
    cwriter.writeheader()
    well_fh = open(os.path.join(args.out_dir, "perwell.csv"), "w", newline="")
    wfields = ["experiment", "well", "cell_line", "genotype", "n_somas",
               "n_somas_with_neurite", "total_len_px", "mean_len_per_soma",
               "median_len_per_soma"]
    wwriter = csv.DictWriter(well_fh, fieldnames=wfields)
    wwriter.writeheader()

    print(f"{'experiment/well':18}{'line':10}{'geno':>5}{'somas':>7}{'len/soma':>10}")
    for row in manifest:
        exp, well, tp = row["experiment"], row["well"], row["timepoint"]
        welldir = os.path.join(args.raw_root, f"{exp}-RGEDI", well)
        paths, sites = well_tiles(welldir, well, tp)
        if len(paths) != 16:
            print(f"{exp}/{well:14}  SKIP: found {len(paths)} FITC tiles (need 16)")
            continue
        # Per-well try/except: one bad well (corrupt tile, transient CUDA error,
        # etc.) must NOT abort the remaining wells. Completed wells are already
        # flushed to disk; a failed well is logged and skipped.
        try:
            image01, soma = build_montages(cp_model, paths, sites, args.diameter,
                                           args.flow, args.cellprob, args.clean_k,
                                           flatten_size=args.flatten_size)
            prob = predict_tiled(unet, image01, device)
            raw_mask = prob >= args.threshold
            # Seam suppression: the per-tile normalization step at each interior
            # tile boundary reads as a straight edge that the ridge-sensitive
            # U-Net fires on (false long straight neurites). Zero a thin band at
            # each interior seam; real neurites crossing a seam lose only ~2*band
            # px (negligible for length). Outer edges are untouched.
            th_m, tw_m = image01.shape[0] // GRID, image01.shape[1] // GRID
            band = args.seam_band
            if band > 0:
                for kk in range(1, GRID):
                    raw_mask[kk * th_m - band:kk * th_m + band, :] = False
                    raw_mask[:, kk * tw_m - band:kk * tw_m + band] = False
            neurite_mask = remove_small_objects(raw_mask, args.min_neurite)
            owner, skel = attribute(soma, neurite_mask)
            lengths = per_soma_lengths(owner, skel, soma)
            props = {p.label: p for p in regionprops(soma)}

            line, geno = row.get("cell_line", "?"), row.get("genotype", "?")
            percell_lens = []
            for lab, (length, skpx) in lengths.items():
                p = props.get(lab)
                if p is None:
                    continue
                cy, cx = p.centroid
                cwriter.writerow(dict(experiment=exp, well=well, cell_line=line,
                                      genotype=geno, soma_id=lab, area_px=int(p.area),
                                      cy=round(cy, 1), cx=round(cx, 1),
                                      neurite_len_px=round(length, 1), skel_px=skpx))
                percell_lens.append(length)
            n = int(soma.max())
            nwith = int(np.sum(np.array(percell_lens) > 0))
            total = float(np.sum(percell_lens))
            wwriter.writerow(dict(experiment=exp, well=well, cell_line=line, genotype=geno,
                                  n_somas=n, n_somas_with_neurite=nwith,
                                  total_len_px=round(total, 1),
                                  mean_len_per_soma=round(total / max(n, 1), 1),
                                  median_len_per_soma=round(float(np.median(percell_lens)) if percell_lens else 0.0, 1)))
            cell_fh.flush(); well_fh.flush()
            print(f"{exp+'/'+well:18}{line:10}{geno:>5}{n:7d}{(total/max(n,1)):10.1f}")
            tifffile.imwrite(os.path.join(args.out_dir, f"{exp}_{well}_{tp}_percell.tif"),
                             colorize(image01, soma, owner, skel)[::4, ::4])
        except Exception as exc:  # noqa: BLE001 - keep the batch alive
            print(f"{exp}/{well}  FAILED: {type(exc).__name__}: {exc}", flush=True)
            continue
    cell_fh.close(); well_fh.close()
    print(f"\nDONE. per-cell + per-well tables in {args.out_dir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Per-cell soma brightness vs neurite length.

Tests whether dim cells are worth keeping for the neurite readout. For every
kept cell it records a background-normalized soma brightness (SNR above local
background, in MAD units) alongside its neurite skeleton length, plus genotype.
Two questions drive it:

1. Do dim cells have ~zero neurites? -> soma_snr vs skel_px.
2. Is the brightness distribution the same for CTR and XDP? -> confound check.

Reuses the exact pipeline (Cellpose somas + clDice neurites + geodesic
attribution). Adds only the per-cell intensity readout. Output: one CSV row per
cell for downstream plotting.
"""

from __future__ import annotations

import argparse
import csv
import glob
import os

import numpy as np
import tifffile
import torch
from cellpose import models
from skimage.measure import regionprops
from skimage.morphology import remove_small_objects

from percell_integrate import (attribute, load_model, parse_name,
                               percentile_normalize, per_soma_lengths,
                               predict_tiled, segment_somas)


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--diameter", type=int, default=25)
    ap.add_argument("--flow", type=float, default=0.6)
    ap.add_argument("--cellprob", type=float, default=-1.0)
    ap.add_argument("--clean-k", type=float, default=2.0)
    ap.add_argument("--min-neurite", type=int, default=25)
    args = ap.parse_args()

    device = torch.device(args.device)
    cp_model = models.CellposeModel(gpu=(args.device == "cuda"))
    unet = load_model(args.checkpoint, device)

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    fh = open(args.out_csv, "w", newline="")
    fields = ["experiment", "well", "cell_line", "genotype", "soma_id",
              "area_px", "soma_mean", "soma_snr", "skel_px", "has_neurite"]
    w = csv.DictWriter(fh, fieldnames=fields)
    w.writeheader()

    tiles = sorted(glob.glob(os.path.join(args.in_dir, "*.tif")))
    print(f"{len(tiles)} tiles")
    print(f"{'tile':40}{'cells':>6}{'w/neurite':>10}{'medSNR':>8}")
    for f in tiles:
        name = os.path.splitext(os.path.basename(f))[0]
        try:
            exp, well, line, geno = parse_name(name)
            raw = tifffile.imread(f).astype(np.float32)
            if raw.ndim == 3:
                raw = raw[..., 0]
            somas = segment_somas(cp_model, raw, args.diameter, args.flow,
                                  args.cellprob, args.clean_k)
            image01 = percentile_normalize(raw)
            prob = predict_tiled(unet, image01, device)
            nmask = remove_small_objects(prob >= args.threshold, args.min_neurite)
            owner, skel = attribute(somas, nmask)
            lengths = per_soma_lengths(owner, skel, somas)

            bg = raw[somas == 0]
            bg_med = float(np.median(bg))
            bg_mad = float(1.4826 * np.median(np.abs(bg - bg_med))) + 1e-9
            snrs = []
            for p in regionprops(somas, intensity_image=raw):
                snr = (float(p.mean_intensity) - bg_med) / bg_mad
                _, skpx = lengths.get(p.label, (0.0, 0))
                snrs.append(snr)
                w.writerow(dict(experiment=exp, well=well, cell_line=line,
                                genotype=geno, soma_id=p.label, area_px=int(p.area),
                                soma_mean=round(float(p.mean_intensity), 1),
                                soma_snr=round(snr, 2), skel_px=int(skpx),
                                has_neurite=int(skpx > 0)))
            fh.flush()
            nwith = sum(1 for p in regionprops(somas) if lengths.get(p.label, (0, 0))[1] > 0)
            med = float(np.median(snrs)) if snrs else 0.0
            print(f"{name:40}{int(somas.max()):6d}{nwith:10d}{med:8.1f}", flush=True)
        except Exception as e:
            print(f"{name:40}  FAILED: {type(e).__name__}: {e}", flush=True)
    fh.close()
    print(f"\nDONE. per-cell table -> {args.out_csv}")


if __name__ == "__main__":
    main()

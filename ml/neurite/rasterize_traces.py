#!/usr/bin/env python3
"""Rasterize SNT ``.traces`` hand-annotations into training targets.

This turns the ground-truth SNT traces used by the F1 scorer
(``scratchpad/score_all.py`` / ``score_traces.py``) into ``(image, mask)``
pairs a segmentation network can train on.

The SNT ``.traces`` format is gzipped XML::

    <tracings>
      <imagesize width="2048" height="2048" .../>
      <path id="0" reallength="123.4" ...>
        <point x="10" y="20" .../>
        <point x="11" y="21" .../>
        ...
      </path>
      ...
    </tracings>

Each ``<path>`` is a polyline through the labelled neurite centreline. We
rasterize every polyline segment with the SAME integer Bresenham-style walk the
scorer uses (so the centreline pixels are identical to the benchmark), then
optionally DILATE the 1-px centreline out to a realistic neurite width to make a
dense training target. The undilated skeleton is what the tol=3 F1 metric
compares against at eval time; the dilated version is the pixel-wise target the
network regresses, because a 1-px target is far too sparse/ill-posed for a CNN
to learn from faint fluorescence.

Outputs, per field ``<name>``, into ``--out``:

* ``<name>_img.npy``   — raw morphology image, float32, original HxW.
* ``<name>_mask.npy``  — dilated binary neurite target, uint8 {0,1}.
* ``<name>_skel.npy``  — 1-px skeleton centreline, uint8 {0,1} (for metric/QA).

CLI::

    python rasterize_traces.py \
        --traces /Users/aholub/Desktop/neurite-annotation/traces \
        --tiles  /Users/aholub/Desktop/neurite-annotation/tiles \
        --out    ml/neurite/data \
        --dilation-radius 2

Labels currently cover only the two annotated fields (``C03_t1``, ``I03_t1``).
Track 0 owns extending the SNT annotation set; point ``--traces`` at the larger
set once it exists and everything downstream scales without code changes.
"""

from __future__ import annotations

import argparse
import glob
import gzip
import os
import xml.etree.ElementTree as ET
from typing import List, Tuple

import numpy as np
import tifffile
from skimage.morphology import binary_dilation, disk, skeletonize


def parse_traces(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Parse an SNT ``.traces`` file into a rasterized centreline mask.

    This reproduces the exact parsing/rasterization used by the benchmark
    scorer (``scratchpad/score_all.py``): integer polyline walk followed by
    ``skimage.morphology.skeletonize`` to thin to a 1-px centreline.

    Args:
        path: Path to a gzipped SNT ``.traces`` XML file.

    Returns:
        A tuple ``(raster, skeleton)`` where ``raster`` is the boolean HxW
        polyline rasterization and ``skeleton`` is its 1-px skeleton. Both are
        boolean arrays of shape (H, W) matching the ``<imagesize>`` header.
    """
    with gzip.open(path) as f:
        root = ET.parse(f).getroot()
    width = int(root.find("imagesize").get("width"))
    height = int(root.find("imagesize").get("height"))
    raster = np.zeros((height, width), dtype=bool)
    for p in root.findall("path"):
        pts = [(int(pt.get("x")), int(pt.get("y"))) for pt in p.findall("point")]
        for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
            n = max(abs(x1 - x0), abs(y1 - y0), 1)
            for i in range(n + 1):
                x = int(round(x0 + (x1 - x0) * i / n))
                y = int(round(y0 + (y1 - y0) * i / n))
                if 0 <= y < height and 0 <= x < width:
                    raster[y, x] = True
    return raster, skeletonize(raster)


def find_raw_tile(tiles_dir: str, name: str) -> str:
    """Locate the raw morphology tile for a field name.

    Args:
        tiles_dir: Directory holding ``<name>_raw.tif`` files.
        name: Field stem, e.g. ``"C03_t1"``.

    Returns:
        Absolute path to the raw tile.

    Raises:
        FileNotFoundError: If no matching raw tile is found.
    """
    candidates = sorted(glob.glob(os.path.join(tiles_dir, f"{name}_raw*.tif")))
    if not candidates:
        raise FileNotFoundError(f"no raw tile for '{name}' in {tiles_dir}")
    return candidates[0]


def field_names(traces_dir: str) -> List[str]:
    """List field stems that have a ``.traces`` file.

    Args:
        traces_dir: Directory of ``.traces`` files.

    Returns:
        Sorted list of field stems (filenames without the ``.traces`` suffix).
    """
    return sorted(
        os.path.splitext(os.path.basename(p))[0]
        for p in glob.glob(os.path.join(traces_dir, "*.traces"))
    )


def rasterize_field(
    traces_dir: str,
    tiles_dir: str,
    name: str,
    dilation_radius: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the ``(image, mask, skeleton)`` triple for one field.

    Args:
        traces_dir: Directory of ``.traces`` files.
        tiles_dir: Directory of raw morphology tiles.
        name: Field stem.
        dilation_radius: Radius (px) of the disk used to dilate the centreline
            into a training target. ``0`` keeps the 1-px skeleton.

    Returns:
        ``(image, mask, skeleton)``: float32 image (H, W); uint8 dilated mask;
        uint8 1-px skeleton.
    """
    _, skel = parse_traces(os.path.join(traces_dir, f"{name}.traces"))
    image = tifffile.imread(find_raw_tile(tiles_dir, name)).astype(np.float32)
    if image.ndim == 3:
        image = image[..., 0]
    if dilation_radius > 0:
        mask = binary_dilation(skel, disk(dilation_radius))
    else:
        mask = skel
    return image, mask.astype(np.uint8), skel.astype(np.uint8)


def main() -> None:
    """CLI entry point: rasterize every annotated field into ``--out``."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--traces", required=True, help="dir of SNT .traces files")
    ap.add_argument("--tiles", required=True, help="dir of <name>_raw.tif tiles")
    ap.add_argument("--out", required=True, help="output dir for .npy pairs")
    ap.add_argument(
        "--dilation-radius",
        type=int,
        default=2,
        help="disk radius (px) to dilate centreline into a training target",
    )
    ap.add_argument(
        "--save-tif",
        action="store_true",
        help="also save uint8 mask/skeleton TIFFs for visual QA",
    )
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    names = field_names(args.traces)
    if not names:
        raise SystemExit(f"no .traces files found in {args.traces}")

    for name in names:
        image, mask, skel = rasterize_field(
            args.traces, args.tiles, name, args.dilation_radius
        )
        np.save(os.path.join(args.out, f"{name}_img.npy"), image)
        np.save(os.path.join(args.out, f"{name}_mask.npy"), mask)
        np.save(os.path.join(args.out, f"{name}_skel.npy"), skel)
        if args.save_tif:
            tifffile.imwrite(
                os.path.join(args.out, f"{name}_mask.tif"), mask * 255
            )
            tifffile.imwrite(
                os.path.join(args.out, f"{name}_skel.tif"), skel * 255
            )
        print(
            f"{name}: img {image.shape} range[{image.min():.0f},{image.max():.0f}]  "
            f"skel_px={int(skel.sum())}  mask_px={int(mask.sum())}"
        )
    print(f"wrote {len(names)} field(s) to {args.out}")


if __name__ == "__main__":
    main()

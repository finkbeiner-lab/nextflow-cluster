#!/usr/bin/env python3
"""Whole-well montage per-cell neurite quantification (CP tool-to-tool batch).

For each well in a manifest, this rebuilds a whole-well montage DIRECTLY FROM
RAW tiles (CellProfiler used individually background-subtracted tiles, which
we deliberately do not use), then runs the trained pipeline on it:

* **Input** — any microscope. ``--tile-pattern`` describes the file names
  (``{well}``, ``{tp}``, ``{site}``, ``*`` placeholders), ``--well-dir`` the
  folder, ``--grid`` the tile grid (``1x1`` = one already-stitched image per
  well), ``--tile-order`` the scan order and ``--overlap-px`` the tile overlap.
  Defaults = the ImageXpress (IXM) layout used for the XDP results.
* **Montage geometry** — default 16 sites in a 4x4 ``standard`` grid, edge-to-edge
  (overlap measured ~0: IXM fields are contiguous, so edge-to-edge keeps neurites
  continuous across seams; verified against CP's own BGs_MN montage). With
  ``--overlap-px N`` each shared strip is split down the middle, so no region
  appears twice.
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
import datetime
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
from skimage.filters import apply_hysteresis_threshold
from skimage.morphology import remove_small_objects
from skimage.segmentation import relabel_sequential

from percell_integrate import (attribute, colorize, load_model, parse_name,
                               percentile_normalize, per_soma_lengths,
                               predict_tiled, segment_somas)

# Defaults = ImageXpress (IXM) layout used for the XDP results.
DEFAULT_TILE_PATTERN = "*_{tp}_*_{well}_{site}_FITC_*.tif"
DEFAULT_WELL_DIR = "{raw_folder}/{well}"
TILE_ORDERS = ("standard", "serpentine", "legacy")  # same names as bin/montage.py


class Layout:
    """Where each acquired tile goes in the whole-well montage.

    Tiles are ranked by their site number (lowest first); rank ``k`` is the
    ``k``-th tile of the microscope's scan. ``order`` gives the scan path:

    * ``standard``:   1 2 3 / 4 5 6 / 7 8 9 (left to right on every row; IXM)
    * ``serpentine``: 1 2 3 / 6 5 4 / 7 8 9 (odd rows reversed; e.g. Robo4)
    * ``legacy``:     3 2 1 / 4 5 6 / 9 8 7 (even rows reversed)

    With ``overlap`` px shared between neighbours, each shared strip is split
    down the middle (half cropped from each tile), so nothing appears twice.
    """

    def __init__(self, rows: int, cols: int, order: str = "standard",
                 overlap: int = 0) -> None:
        if rows < 1 or cols < 1:
            raise ValueError(f"grid must be at least 1x1, got {rows}x{cols}")
        if order not in TILE_ORDERS:
            raise ValueError(f"tile order must be one of {TILE_ORDERS}, got {order!r}")
        if overlap < 0:
            raise ValueError(f"overlap must be >= 0 px, got {overlap}")
        self.rows, self.cols, self.order, self.overlap = rows, cols, order, overlap

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "Layout":
        """Build from ``--grid ROWSxCOLS``, ``--tile-order``, ``--overlap-px``."""
        m = re.fullmatch(r"\s*(\d+)\s*[xX]\s*(\d+)\s*", args.grid)
        if not m:
            raise SystemExit(f"--grid must look like 4x4 (rows x columns), got {args.grid!r}")
        try:
            return cls(int(m.group(1)), int(m.group(2)), args.tile_order, args.overlap_px)
        except ValueError as exc:
            raise SystemExit(str(exc))

    @property
    def n(self) -> int:
        """Number of tiles per well."""
        return self.rows * self.cols

    def describe(self) -> str:
        """Short text for cache meta and logs, e.g. ``4x4 standard overlap 0``."""
        return f"{self.rows}x{self.cols} {self.order} overlap {self.overlap}"

    def cell(self, k: int) -> Tuple[int, int]:
        """Grid (row, col) of the ``k``-th tile (0-based scan rank)."""
        r, j = divmod(k, self.cols)
        rev = ((self.order == "serpentine" and r % 2 == 1)
               or (self.order == "legacy" and r % 2 == 0))
        return r, (self.cols - 1 - j if rev else j)

    def _crop(self, i: int, n: int) -> Tuple[int, int]:
        """(start, end) px to crop from a tile at grid index ``i`` of ``n``."""
        before = self.overlap - self.overlap // 2 if i > 0 else 0
        after = self.overlap // 2 if i < n - 1 else 0
        return before, after

    def geometry(self, th: int, tw: int) -> Tuple[int, int, List[int], List[int]]:
        """Montage size and row/column start offsets for tiles of size (th, tw).

        Returns:
            ``(H, W, row_starts, col_starts)``; the interior seams are at
            ``row_starts[1:]`` and ``col_starts[1:]``.
        """
        if 2 * self.overlap >= min(th, tw):
            raise ValueError(f"overlap {self.overlap} px is too large for {th}x{tw} tiles")
        def starts(n: int, size: int) -> Tuple[int, List[int]]:
            out, pos = [], 0
            for i in range(n):
                out.append(pos)
                b, a = self._crop(i, n)
                pos += size - b - a
            return pos, out
        H, rs = starts(self.rows, th)
        W, cs = starts(self.cols, tw)
        return H, W, rs, cs

    def place(self, canvas: np.ndarray, tile: np.ndarray, k: int) -> None:
        """Write the ``k``-th tile (overlap cropped) into the montage, in place."""
        th, tw = tile.shape
        _, _, rs, cs = self.geometry(th, tw)
        r, c = self.cell(k)
        t0, t1 = self._crop(r, self.rows)
        l0, l1 = self._crop(c, self.cols)
        part = tile[t0:th - t1, l0:tw - l1]
        canvas[rs[r]:rs[r] + part.shape[0], cs[c]:cs[c] + part.shape[1]] = part


def _pattern_parts(pattern: str) -> List[str]:
    """Split a tile pattern into literal text and placeholder/wildcard tokens."""
    return [p for p in re.split(r"(\{site\}|\{well\}|\{tp\}|\{experiment\}|\*|\?)",
                                pattern) if p]


def well_dir(raw_root: str, template: str, row: Dict[str, str]) -> str:
    """The folder that holds one well's raw tiles.

    Args:
        raw_root: ``--raw-root``.
        template: ``--well-dir``; placeholders ``{experiment}``, ``{well}``,
            ``{tp}``, ``{raw_folder}`` (the manifest column, or
            ``<experiment>-RGEDI`` when the column is absent/empty).
        row: Manifest row.
    """
    exp = row["experiment"]
    return os.path.join(raw_root, template.format(
        experiment=exp, well=row["well"], tp=row["timepoint"],
        raw_folder=(row.get("raw_folder") or f"{exp}-RGEDI")))


def well_tiles(welldir: str, pattern: str, exp: str, well: str,
               tp: str) -> Tuple[List[str], List[int]]:
    """List a well's tiles at one timepoint, ordered by site number.

    Args:
        welldir: Directory of raw tiles for the well.
        pattern: File-name pattern. ``{well}``, ``{tp}``, ``{experiment}`` are
            filled from the manifest; ``{site}`` matches the site number; ``*``
            and ``?`` are wildcards. Without ``{site}`` there must be exactly one
            matching image (one already-stitched image per well, ``--grid 1x1``).
        exp: Experiment name.
        well: Well id (e.g. ``"J03"``).
        tp: Timepoint token as it appears in the file names (e.g. ``"T9"``).

    Returns:
        ``(paths, sites)`` sorted ascending by site number.
    """
    fill = {"{well}": well, "{tp}": tp, "{experiment}": exp}
    rx, gl = "", ""
    for p in _pattern_parts(pattern):
        if p == "{site}":
            rx, gl = rx + r"(\d+)", gl + "*"
        elif p in fill:
            rx, gl = rx + re.escape(fill[p]), gl + glob.escape(fill[p])
        elif p == "*":
            rx, gl = rx + ".*", gl + "*"
        elif p == "?":
            rx, gl = rx + ".", gl + "?"
        else:
            rx, gl = rx + re.escape(p), gl + glob.escape(p)
    pat = re.compile(rx)
    has_site = "{site}" in pattern
    hits = []
    for f in glob.glob(os.path.join(glob.escape(welldir), gl)):
        m = pat.fullmatch(os.path.basename(f))
        if m:
            hits.append((int(m.group(1)) if has_site else 1, f))
    hits.sort()
    return [f for _, f in hits], [s for s, _ in hits]


def find_well_tiles(args: argparse.Namespace, row: Dict[str, str],
                    layout: Layout) -> Tuple[List[str], str]:
    """Find one manifest row's tiles and check there is one per grid cell.

    Returns:
        ``(paths, problem)``: paths in scan order, and ``""`` if they fit the
        grid, else a one-line reason to skip the well.
    """
    exp, well, tp = row["experiment"], row["well"], row["timepoint"]
    wdir = well_dir(args.raw_root, args.well_dir, row)
    paths, sites = well_tiles(wdir, args.tile_pattern, exp, well, tp)
    if len(paths) != layout.n:
        return paths, (f"found {len(paths)} tiles (need {layout.n} for a "
                       f"{layout.rows}x{layout.cols} grid) in {wdir}")
    if len(set(sites)) != len(sites):
        return paths, f"site numbers repeat in {wdir}; make --tile-pattern more specific"
    return paths, ""


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


def neurite_mask_from_prob(prob: np.ndarray, threshold: float, thr_low: float,
                           seam_band: int, min_neurite: int,
                           seams: Tuple[List[int], List[int]]) -> np.ndarray:
    """Binary neurite mask from the U-Net probability montage.

    Args:
        prob: Neurite probability montage (H, W) in [0, 1].
        threshold: Seed threshold; a component must reach this to be kept.
        thr_low: Hysteresis low threshold. Pixels >= ``thr_low`` join the mask
            only when connected to a seed, so a trace whose probability dips
            below ``threshold`` stays one piece instead of fragmenting. Values
            >= ``threshold`` disable hysteresis (plain ``prob >= threshold``).
        seam_band: Zero this many px either side of each interior tile seam.
        min_neurite: Drop connected components smaller than this (px).
        seams: ``(row_seams, col_seams)`` px positions of the interior seams
            (see :func:`montage_seams`).

    Returns:
        Boolean neurite mask (H, W).
    """
    if thr_low < threshold:
        mask = apply_hysteresis_threshold(prob, thr_low, threshold)
    else:
        mask = prob >= threshold
    # Seam suppression: the per-tile normalization step at each interior tile
    # boundary reads as a straight edge that the ridge-sensitive U-Net fires on
    # (false long straight neurites). Zero a thin band at each interior seam;
    # real neurites crossing a seam lose only ~2*band px (negligible for length).
    # Applied after hysteresis so a seam line can never seed a real trace.
    if seam_band > 0:
        for y in seams[0]:
            mask[max(0, y - seam_band):y + seam_band, :] = False
        for x in seams[1]:
            mask[:, max(0, x - seam_band):x + seam_band] = False
    return remove_small_objects(mask, min_neurite)


def montage_seams(layout: Layout,
                  montage_shape: Tuple[int, int]) -> Tuple[List[int], List[int]]:
    """Interior seam positions for a montage built with ``layout``.

    The tile size follows from the montage size (``H = rows*th - (rows-1)*overlap``),
    so a cache built with another grid/overlap is caught here.

    Raises:
        ValueError: If ``layout`` cannot give ``montage_shape``.
    """
    ov = layout.overlap
    th = (montage_shape[0] + (layout.rows - 1) * ov) // layout.rows
    tw = (montage_shape[1] + (layout.cols - 1) * ov) // layout.cols
    H, W, rs, cs = layout.geometry(th, tw)
    if (H, W) != tuple(montage_shape):
        raise ValueError(f"montage is {montage_shape[0]}x{montage_shape[1]} but --grid/"
                         f"--overlap-px ({layout.describe()}) with {th}x{tw} tiles gives "
                         f"{H}x{W}; use the layout the cache was built with")
    return rs[1:], cs[1:]


def cache_params(args: argparse.Namespace) -> Dict[str, object]:
    """Params that determine the cached soma labels + neurite prob.

    The cache file name is only ``<exp>_<well>_<tp>``, so these are recorded in
    each cache's meta and checked on reuse (:func:`check_cache`).
    """
    return dict(flatten_size=args.flatten_size, diameter=args.diameter, flow=args.flow,
                cellprob=args.cellprob, clean_k=args.clean_k,
                soma_flatten=args.soma_flatten, soma_floor_px=args.soma_floor_px,
                soma_min_width=args.soma_min_width,
                layout=Layout.from_args(args).describe(),
                checkpoint=os.path.basename(os.path.dirname(os.path.abspath(args.checkpoint)))
                + "/" + os.path.basename(args.checkpoint))


_WARNED_CKPT = []


def check_cache(z, args: argparse.Namespace, cache_path: str) -> None:
    """Raise if a cache entry was built with different segmentation params.

    Guards the silent-reuse trap: pointing a run with a new checkpoint or soma
    recipe at an old ``--cache-dir`` would otherwise reuse the old soma/prob.
    Caches predating a param are treated as having its default (soma_* = 0);
    caches predating ``checkpoint`` can't be verified and only warn.

    Raises:
        ValueError: On any mismatch (the per-well handler logs and skips the well).
    """
    meta = json.loads(str(z["meta"])) if "meta" in z.files else {}
    bad = []
    for k, v in cache_params(args).items():
        if k in meta:
            if meta[k] != v:
                bad.append(f"{k}: cache={meta[k]} run={v}")
        elif k.startswith("soma_") and v != 0:
            bad.append(f"{k}: cache=0 (pre-dates option) run={v}")
        elif k == "checkpoint" and not _WARNED_CKPT:
            _WARNED_CKPT.append(1)
            print(f"WARN: cache {os.path.basename(cache_path)} predates checkpoint tracking; "
                  f"cannot verify it was built with {v}", flush=True)
    if bad:
        raise ValueError(f"cache {cache_path} built with different params ({'; '.join(bad)}); "
                         "use a fresh --cache-dir")


def _read_tiles(paths: List[str]) -> List[np.ndarray]:
    """Read tiles once as float32 (drop any trailing channel axis)."""
    tiles = []
    for p in paths:
        raw = tifffile.imread(p).astype(np.float32)
        if raw.ndim == 3:
            raw = raw[..., 0]
        tiles.append(raw)
    return tiles


def _check_tiles(tiles: List[np.ndarray], layout: Layout) -> Tuple[int, int, int, int]:
    """Check tile count/sizes against the layout; return ``(th, tw, H, W)``."""
    if len(tiles) != layout.n:
        raise ValueError(f"{len(tiles)} tiles for a {layout.rows}x{layout.cols} grid")
    th, tw = tiles[0].shape
    if any(t.shape != (th, tw) for t in tiles):
        raise ValueError("tiles in a well differ in size")
    H, W, _, _ = layout.geometry(th, tw)
    return th, tw, H, W


def _montage_image01(tiles: List[np.ndarray], layout: Layout,
                     flatten_size: int = 0) -> np.ndarray:
    """Assemble the normalized [0,1] grayscale montage from raw tiles (no GPU).

    Split out of :func:`build_montages` so the montage image can be rebuilt on a
    cache hit (soma+prob restored from disk) to regenerate overlays without ever
    touching Cellpose / the U-Net.
    """
    _, _, H, W = _check_tiles(tiles, layout)
    # Neurite-image source: optional data-driven rolling-background flattening
    # (vignetting / illumination correction) applied per tile BEFORE percentile
    # normalization.
    img_src = ([flatten_tile(t, flatten_size) for t in tiles]
               if flatten_size > 0 else tiles)
    # Empty-tile guard: percentile-normalizing a near-signal-free tile amplifies
    # its noise into a fake bright band. Compare each tile's dynamic range to the
    # montage-wide median range; if a tile has < EMPTY_FRAC of it, emit zeros.
    EMPTY_FRAC = 0.15
    ranges = np.array([np.percentile(t, 99.5) - np.percentile(t, 1) for t in img_src])
    global_range = float(np.median(ranges))
    image01 = np.zeros((H, W), np.float32)
    for k, imgt in enumerate(img_src):
        lo, hi = np.percentile(imgt, 1), np.percentile(imgt, 99.5)
        if global_range <= 0 or (hi - lo) < EMPTY_FRAC * global_range:
            norm = np.zeros_like(imgt)  # empty/low-signal tile: do not amplify
        else:
            norm = np.clip((imgt - lo) / (hi - lo), 0, 1).astype(np.float32)
        layout.place(image01, norm, k)
    return image01


def build_image01(paths: List[str], layout: Layout,
                  flatten_size: int = 0) -> np.ndarray:
    """CPU-only montage grayscale rebuild from raw tiles (for overlay regen)."""
    return _montage_image01(_read_tiles(paths), layout, flatten_size)


def build_montages(cp_model, paths: List[str], layout: Layout,
                   diameter: int, flow: float, cellprob: float, clean_k: float,
                   flatten_size: int = 0, soma_flatten: int = 0,
                   soma_floor_px: float = 0.0, soma_min_width: float = 0.0
                   ) -> Tuple[np.ndarray, np.ndarray]:
    """Build the (normalized image, soma-label) montages for one well.

    Args:
        cp_model: Cellpose model.
        paths: Tile paths ordered by site number (scan order).
        layout: Grid, scan order and overlap of the tiles.
        diameter/flow/cellprob/clean_k: Cellpose soma-recipe params.
        flatten_size: Rolling-background footprint for the NEURITE image.
        soma_flatten: Rolling-background footprint for the SOMA image (Cellpose
            input + intensity cleanup); ``0`` = untouched raw tiles.
        soma_floor_px/soma_min_width: see :func:`segment_somas`.

    Returns:
        ``(image01, soma_labels)`` montages, both the montage size (H, W).
    """
    # Read all tiles once (16 x 2048^2 float32 ~= 256MB for the IXM default).
    tiles = _read_tiles(paths)
    _, _, H, W = _check_tiles(tiles, layout)
    image01 = _montage_image01(tiles, layout, flatten_size)

    # Somas run on the raw tiles unless --soma-flatten is set, in which case
    # Cellpose and its intensity cleanup see a flat-fielded copy (vignetting
    # otherwise hides dim/clumped cells). Independent of the neurite flatten.
    soma = np.zeros((H, W), np.int32)
    offset = 0
    for k, raw in enumerate(tiles):
        seg = flatten_tile(raw, soma_flatten) if soma_flatten > 0 else None
        labels = segment_somas(cp_model, raw, diameter, flow, cellprob, clean_k,
                               seg_image=seg, floor_px=soma_floor_px,
                               min_width=soma_min_width)
        if labels.max() > 0:
            lab = labels.copy()
            lab[labels > 0] += offset
            offset = int(lab.max())
            layout.place(soma, lab, k)
    if layout.overlap > 0:
        # Cropping the overlap drops some labels; renumber 1..n so that
        # n_somas (= soma.max()) stays a count.
        soma = relabel_sequential(soma)[0]
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


def _resolve_run_dir(out_dir: str, run_id: str, run_label: str,
                     overwrite: bool) -> Tuple[str, str]:
    """Create a fresh, non-clobbering per-run output dir under ``out_dir/runs``.

    Each analyze pass writes to its own ``runs/<id>/`` so re-tuning
    post-segmentation params never overwrites an earlier run's CSVs/overlays
    (runs accumulate for comparison). The shared ``--cache-dir`` is separate and
    is deliberately NOT versioned -- the expensive Cellpose/U-Net segmentations
    are reused across runs. A convenience ``latest`` symlink points at the newest
    run.

    Args:
        out_dir: Base output dir (holds ``runs/`` and ``latest``).
        run_id: Explicit run id; if empty, a timestamp (+ label) is generated.
        run_label: Optional human tag appended to a generated run id.
        overwrite: Allow reusing a run dir that already has outputs.

    Returns:
        ``(run_dir, run_id)``.

    Raises:
        SystemExit: If the target run dir already holds a ``percell.csv`` and
            ``overwrite`` is False (refuses to clobber a prior run).
    """
    runs_root = os.path.join(out_dir, "runs")
    os.makedirs(runs_root, exist_ok=True)
    if not run_id:
        ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        run_id = f"{ts}_{run_label}" if run_label else ts
    run_dir = os.path.join(runs_root, run_id)
    if os.path.exists(os.path.join(run_dir, "percell.csv")) and not overwrite:
        raise SystemExit(
            f"run dir already has outputs: {run_dir}\n"
            "refusing to overwrite a prior run -- pass a new --run-label/--run-id, "
            "or --overwrite to reuse this one.")
    os.makedirs(run_dir, exist_ok=True)
    latest = os.path.join(out_dir, "latest")
    try:  # 'latest' is a convenience; non-fatal if the FS refuses symlinks
        if os.path.islink(latest) or os.path.exists(latest):
            if os.path.islink(latest):
                os.unlink(latest)
        os.symlink(os.path.join("runs", run_id), latest)
    except OSError:
        pass
    return run_dir, run_id


def _write_params(run_dir: str, args: argparse.Namespace, run_id: str,
                  n_wells: int) -> None:
    """Write ``params.json`` so each run is self-describing for comparison."""
    keys = ["manifest", "raw_root", "out_dir", "cache_dir", "checkpoint",
            "device", "threshold", "diameter", "flow", "cellprob", "clean_k",
            "min_neurite", "seam_band", "flatten_size", "soma_dilation",
            "thr_low", "attach_px", "root_px", "min_radial", "cross_cos", "soma_flatten", "soma_floor_px", "soma_min_width",
            "tile_pattern", "well_dir", "grid", "tile_order", "overlap_px"]
    meta = dict(run_id=run_id,
                created=datetime.datetime.now().isoformat(timespec="seconds"),
                n_wells=n_wells,
                **{k: getattr(args, k) for k in keys})
    with open(os.path.join(run_dir, "params.json"), "w") as fh:
        json.dump(meta, fh, indent=2)


def run_overlay_only(args: argparse.Namespace) -> None:
    """(Re)build per-well overlay TIFs for an existing run -- CPU only, no GPU.

    Restores soma+prob from ``--cache-dir`` and rebuilds the montage grayscale
    from the raw tiles, then redraws the overlay. Thresholds are read from the
    target run's ``params.json`` (falling back to CLI args) so overlays match
    that run exactly. Wells with no cache entry are skipped (they need a GPU
    analyze pass first).
    """
    if not args.cache_dir:
        raise SystemExit("--overlay-only needs --cache-dir (overlays rebuild "
                         "from cached soma+prob; no GPU).")
    run_dir = (os.path.join(args.out_dir, "runs", args.run_id)
               if args.run_id else os.path.join(args.out_dir, "latest"))
    if not os.path.isdir(run_dir):
        raise SystemExit(f"--overlay-only: run dir not found: {run_dir} "
                         "(pass --run-id of an existing run).")
    pf = os.path.join(run_dir, "params.json")
    p = json.load(open(pf)) if os.path.exists(pf) else {}
    threshold = p.get("threshold", args.threshold)
    seam_band = p.get("seam_band", args.seam_band)
    min_neurite = p.get("min_neurite", args.min_neurite)
    flatten_size = p.get("flatten_size", args.flatten_size)
    # Runs predating these knobs had neither: default = off (their exact behaviour).
    thr_low = p.get("thr_low", threshold if "threshold" in p else args.thr_low)
    attach_px = p.get("attach_px", 0 if p else args.attach_px)
    root_px = p.get("root_px", 0 if p else args.root_px)
    min_radial = p.get("min_radial", args.min_radial)
    cross_cos = p.get("cross_cos", None if p else args.cross_cos)
    # Input layout: from the run's params.json when recorded (runs before the
    # layout options were all IXM 4x4 = the CLI defaults).
    for k in ("tile_pattern", "well_dir", "grid", "tile_order", "overlap_px"):
        if k in p:
            setattr(args, k, p[k])
    layout = Layout.from_args(args)

    manifest = list(csv.DictReader(open(args.manifest)))
    n = 0
    for row in manifest:
        exp, well, tp = row["experiment"], row["well"], row["timepoint"]
        cache_path = os.path.join(args.cache_dir, f"{exp}_{well}_{tp}.npz")
        if not os.path.exists(cache_path):
            print(f"{exp}/{well}: no cache -> skip (run a GPU analyze pass first)")
            continue
        paths, problem = find_well_tiles(args, row, layout)
        if problem:
            print(f"{exp}/{well}: {problem} -> skip")
            continue
        z = np.load(cache_path)
        soma = z["soma_labels"].astype(np.int32)
        prob = z["neurite_prob"].astype(np.float32) / 255.0
        image01 = build_image01(paths, layout, flatten_size)
        neurite_mask = neurite_mask_from_prob(prob, threshold, thr_low, seam_band,
                                              min_neurite, montage_seams(layout, prob.shape))
        owner, skel = attribute(soma, neurite_mask, attach_px, root_px, min_radial,
                                cross_cos)
        tifffile.imwrite(
            os.path.join(run_dir, f"{exp}_{well}_{tp}_percell.tif"),
            colorize(image01, soma, owner, skel)[::4, ::4])
        n += 1
        print(f"{exp}/{well}: overlay rebuilt")
    print(f"\nDONE. {n} overlays (re)built in {run_dir} (no GPU).")


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True,
                    help="CSV: experiment,well,timepoint,cell_line,genotype[,raw_folder]")
    ap.add_argument("--raw-root", required=True,
                    help="root folder of the raw images; each well's folder is "
                         "<raw-root>/<--well-dir>")
    ap.add_argument("--well-dir", default=DEFAULT_WELL_DIR,
                    help="folder of one well's images under --raw-root. Placeholders: "
                         "{experiment} {well} {tp} {raw_folder} (manifest column; "
                         "<experiment>-RGEDI if absent). Use '.' if all images are in "
                         "--raw-root itself. Default: %(default)s")
    ap.add_argument("--tile-pattern", default=DEFAULT_TILE_PATTERN,
                    help="file-name pattern of the morphology-channel tiles. {well} {tp} "
                         "{experiment} are filled from the manifest, {site} matches the "
                         "tile/site number, * and ? are wildcards. Leave out {site} for "
                         "one already-stitched image per well (with --grid 1x1). "
                         "Default (ImageXpress, FITC): %(default)s")
    ap.add_argument("--grid", default="4x4",
                    help="tiles per well as ROWSxCOLUMNS, e.g. 4x4, 3x5, 1x1. The well "
                         "must have exactly ROWS*COLUMNS matching tiles. Default: %(default)s")
    ap.add_argument("--tile-order", default="standard", choices=TILE_ORDERS,
                    help="scan order of the tiles by site number: standard = 1 2 3/4 5 6 "
                         "(IXM); serpentine = 1 2 3/6 5 4 (e.g. Robo4); legacy = 3 2 1/4 5 6. "
                         "Same names as bin/montage.py. Default: %(default)s")
    ap.add_argument("--overlap-px", type=int, default=0,
                    help="px shared by neighbouring tiles (0 = edge to edge, IXM). Each "
                         "shared strip is split down the middle. Default: %(default)s")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--thr-low", type=float, default=None,
                    help="hysteresis low threshold: pixels >= this join a trace "
                         "connected to a >= --threshold seed, so traces whose "
                         "probability dips don't fragment (default: = --threshold, "
                         "i.e. off). Post-segmentation knob: safe to retune off cache.")
    ap.add_argument("--attach-px", type=int, default=0,
                    help="grow somas by this many px for neurite attribution only, "
                         "so traces stopping just short of the cell body still count "
                         "(default 0 = off). Post-segmentation knob: safe off cache.")
    ap.add_argument("--root-px", type=int, default=0,
                    help="root-contact attribution: credit a trace to a soma only if it "
                         "leaves the soma outward within this many px (a neighbour's "
                         "neurite passing by is not credited). 0 = off (seed from whole "
                         "soma bodies). Overrides --attach-px. Post-segmentation knob: safe off cache.")
    ap.add_argument("--min-radial", type=float, default=0.8,
                    help="minimum contact radiality (0..~1.4) to root, with --root-px. "
                         "Tune against a pseudo-soma null (real soma masks moved to random "
                         "spots): pick the value where contacts at real somas start to "
                         "outnumber contacts at pseudo-somas beyond the density baseline.")
    ap.add_argument("--cross-cos", type=float, default=None,
                    help="with --root-px: drop pairs of root contacts that point in "
                         "opposite directions (cosine < this) and are joined by the "
                         "neurite mask through the soma -- one neurite crossing over "
                         "the cell body, not two roots. Default off; ~-0.7 is a "
                         "reasonable starting value.")
    ap.add_argument("--diameter", type=int, default=25)
    ap.add_argument("--flow", type=float, default=0.6)
    ap.add_argument("--cellprob", type=float, default=-1.0)
    ap.add_argument("--clean-k", type=float, default=2.0)
    ap.add_argument("--soma-flatten", type=int, default=0,
                    help="rolling-background footprint (px) for the SOMA image: Cellpose and "
                         "its intensity cleanup run on a flat-fielded copy of each tile "
                         "(0 = untouched raw, the original behaviour; ~128 for vignetted data)")
    ap.add_argument("--soma-floor-px", type=float, default=0,
                    help="fixed minimum soma area in px (0 = size-dependent floor "
                         "max(300, 0.6*median), the original behaviour)")
    ap.add_argument("--soma-min-width", type=float, default=0,
                    help="drop somas whose minor axis is below this many px, e.g. slices "
                         "along thick neurites (0 = off)")
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
                         "without the GPU. SHARED across runs (not versioned).")
    ap.add_argument("--run-label", default="",
                    help="human tag appended to the auto-timestamped run id; "
                         "outputs land in <out-dir>/runs/<timestamp>[_label]/")
    ap.add_argument("--run-id", default="",
                    help="explicit run id (overrides the timestamp); also "
                         "selects the target run for --overlay-only")
    ap.add_argument("--overwrite", action="store_true",
                    help="allow writing into a run dir that already has outputs "
                         "(default: refuse, to protect earlier runs)")
    ap.add_argument("--overlay-only", action="store_true",
                    help="do not analyze; (re)build overlay TIFs for an existing "
                         "run from --cache-dir + raw tiles (CPU only, no GPU). "
                         "Targets --run-id, else <out-dir>/latest.")
    args = ap.parse_args()
    if args.thr_low is None:
        args.thr_low = args.threshold
    layout = Layout.from_args(args)  # exits with a clear message on a bad --grid

    if args.overlay_only:
        run_overlay_only(args)
        return

    os.makedirs(args.out_dir, exist_ok=True)
    if args.cache_dir:
        os.makedirs(args.cache_dir, exist_ok=True)
    # Per-run output dir (never clobbers an earlier run); cache stays shared.
    run_dir, run_id = _resolve_run_dir(args.out_dir, args.run_id, args.run_label,
                                       args.overwrite)
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
    _write_params(run_dir, args, run_id, len(manifest))
    print(f"run: {run_id} -> {run_dir}")

    cell_fh = open(os.path.join(run_dir, "percell.csv"), "w", newline="")
    cfields = ["experiment", "well", "timepoint", "cell_line", "genotype", "soma_id",
               "area_px", "cy", "cx", "neurite_len_px", "skel_px",
               "n_branch_points", "n_end_points", "n_primary_neurites",
               "max_branch_length"]
    cwriter = csv.DictWriter(cell_fh, fieldnames=cfields)
    cwriter.writeheader()
    well_fh = open(os.path.join(run_dir, "perwell.csv"), "w", newline="")
    wfields = ["experiment", "well", "timepoint", "cell_line", "genotype", "n_somas",
               "n_somas_with_neurite", "total_len_px", "mean_len_per_soma",
               "median_len_per_soma", "mean_branch_points", "mean_tips",
               "mean_primary_neurites", "mean_max_branch_len", "frac_branched"]
    wwriter = csv.DictWriter(well_fh, fieldnames=wfields)
    wwriter.writeheader()

    print(f"{'experiment/well':18}{'line':10}{'geno':>5}{'somas':>7}{'len/soma':>10}")
    for row in manifest:
        exp, well, tp = row["experiment"], row["well"], row["timepoint"]
        paths, problem = find_well_tiles(args, row, layout)
        if problem:
            print(f"{exp}/{well:14}  SKIP: {problem}")
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
                check_cache(z, args, cache_path)  # refuse a cache built with other seg params
                soma = z["soma_labels"].astype(np.int32)
                prob = z["neurite_prob"].astype(np.float32) / 255.0
                image01, from_cache = None, True
            else:
                cp_model, unet = ensure_models()
                image01, soma = build_montages(cp_model, paths, layout, args.diameter,
                                               args.flow, args.cellprob, args.clean_k,
                                               flatten_size=args.flatten_size,
                                               soma_flatten=args.soma_flatten,
                                               soma_floor_px=args.soma_floor_px,
                                               soma_min_width=args.soma_min_width)
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
                            genotype=row.get("genotype", "?"),
                            threshold=args.threshold, seam_band=args.seam_band,
                            min_neurite=args.min_neurite, **cache_params(args)))))
                    os.replace(tmp, cache_path)
            neurite_mask = neurite_mask_from_prob(prob, args.threshold, args.thr_low,
                                                  args.seam_band, args.min_neurite,
                                                  montage_seams(layout, prob.shape))
            owner, skel = attribute(soma, neurite_mask, args.attach_px,
                                    args.root_px, args.min_radial, args.cross_cos)
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
                cwriter.writerow(dict(experiment=exp, well=well, timepoint=tp, cell_line=line,
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
            wwriter.writerow(dict(experiment=exp, well=well, timepoint=tp, cell_line=line, genotype=geno,
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
                tifffile.imwrite(os.path.join(run_dir, f"{exp}_{well}_{tp}_percell.tif"),
                                 colorize(image01, soma, owner, skel)[::4, ::4])
        except Exception as exc:  # noqa: BLE001 - keep the batch alive
            print(f"{exp}/{well}  FAILED: {type(exc).__name__}: {exc}", flush=True)
            continue
    cell_fh.close(); well_fh.close()
    print(f"\nDONE. per-cell + per-well tables in {run_dir}")
    print(f"      (latest -> runs/{run_id}; cache shared at {args.cache_dir or 'none'})")


if __name__ == "__main__":
    main()

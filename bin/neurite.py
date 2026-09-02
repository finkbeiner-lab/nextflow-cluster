#!/usr/bin/env python
"""NEURITE — per-cell neurite length and arborization for the Finkbeiner pipeline.

This is the ``bin/`` script backing the ``NEURITE`` catalog module
(``deepcell/backend/catalog/neurite.json``). It lives alongside
``segmentation.py``, ``intensity.py``, ``puncta.py`` and is invoked per tile by
the ``NEURITE`` process in ``modules.nf``.

Why this module exists
----------------------
The first-pass neurite analysis fed raw morphology images straight into a
CellProfiler adaptive-threshold + skeletonize pipeline. On dense MSN cultures
that segmentation is weak: thin, low-contrast processes are lost and skeletons
fragment. This module fixes the root cause with three changes:

1.  **Multiscale vesselness enhancement** (Frangi/Sato) on the morphology
    channel *before* thresholding, so thin curvilinear processes are boosted
    relative to background and blobs.
2.  **Per-soma attribution**: skeleton segments are assigned to the soma they
    are physically *connected to along the skeleton* (not merely the nearest
    centroid), giving genuine *per-cell* measurements (total neurite length,
    branch points, primary neurites, max path length) instead of whole-field
    aggregates. ``max_soma_distance`` is kept only as a near-miss fallback for
    small skeleton gaps. Per-cell rows are what the downstream mixed-effects
    model needs (see README).
3.  **Spur pruning** at ``min_branch_length`` px to suppress segmentation noise.

Design notes
------------
*   The morphology channel is both the image we trace neurites on and the
    channel whose upstream ``SEGMENTATION`` / ``CELLPOSE`` mask (``maskpath``,
    labelled by ``randomcellid``) gives the somas. One row per cell is written
    to the ``neuritecelldata`` table (created in ``sql.py``), keyed the same way
    as ``intensitycelldata`` (experimentdata/welldata/tiledata/celldata/
    channeldata ids) so it joins cleanly with intensity, tracking, and survival
    outputs.
*   DB / image access goes through the same helpers ``intensity.py`` uses:
    ``Normalize`` (``get_df_for_training``, background correction), ``Ops``, and
    ``Database`` (``get_df_from_query`` / ``get_table_uuid`` / ``add_row`` /
    ``delete_based_on_duplicate_name``). All of that coupling lives in the
    ``Neurite`` class below; the science in ``measure_cell_neurites`` is
    API-independent and unit-tested on plain arrays
    (``neurite-module/tests/test_neurite_core.py``).
*   The heavy lifting uses scikit-image (``frangi``, ``skeletonize``) and, when
    available, ``skan`` for skeleton length; a NumPy fallback computes length
    and branch/endpoint counts from the skeleton if ``skan`` is absent.

CLI (argument order matches the ``NEURITE`` process ``input:`` block in
modules.nf and the ``cli_flags`` order in the manifest).
"""

from __future__ import annotations

import argparse
import datetime
import logging
import os
import sys
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger("Neurite")
now = datetime.datetime.now()
TIMESTAMP = '%d%02d%02d%02d%02d' % (now.year, now.month, now.day, now.hour, now.minute)
_fink_log_dir = './finkbeiner_logs'
if not os.path.exists(_fink_log_dir):
    os.makedirs(_fink_log_dir, exist_ok=True)
_fh = logging.FileHandler(os.path.join(_fink_log_dir, f'Neurite-log_{TIMESTAMP}.log'))
_fh.setLevel(20)
logger.addHandler(_fh)


# --------------------------------------------------------------------------- #
#  CLI
# --------------------------------------------------------------------------- #
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the pipeline-standard CLI for the NEURITE module."""
    p = argparse.ArgumentParser(description="Per-cell neurite quantification.")
    # context
    p.add_argument("--experiment", required=True)
    p.add_argument("--img_norm_name", default="subtraction",
                   choices=["division", "subtraction", "identity"])
    p.add_argument("--chosen_wells", default="all")
    p.add_argument("--chosen_timepoints", default="all")
    p.add_argument("--wells_toggle", default="include")
    p.add_argument("--timepoints_toggle", default="include")
    # Normalize/Ops read these even though the NEURITE process does not pass
    # them; keep the defaults so the shared DB helpers don't KeyError.
    p.add_argument("--chosen_channels", default="all")
    p.add_argument("--channels_toggle", default="include")
    # scalars
    p.add_argument("--morphology_channel", required=True)
    p.add_argument("--vesselness_sigma_min", type=float, default=1.0)
    p.add_argument("--vesselness_sigma_max", type=float, default=8.0)
    p.add_argument("--vesselness_sigma_steps", type=int, default=5)
    p.add_argument("--neurite_threshold", type=float, default=0.15)
    # Adaptive thresholding of the vesselness response. ``global`` reproduces the
    # original single-threshold behaviour; ``hysteresis`` (default) recovers
    # faint distal neurites that are connected to a confident ridge; ``otsu``
    # picks a data-driven cut on the nonzero response.
    p.add_argument("--threshold_method", default="hysteresis",
                   choices=["global", "hysteresis", "otsu"])
    # Hysteresis band. When unset, low defaults to ``neurite_threshold`` and high
    # to ``2 * neurite_threshold`` (clamped to [low, 1.0]).
    p.add_argument("--hysteresis_low", type=float, default=None)
    p.add_argument("--hysteresis_high", type=float, default=None)
    p.add_argument("--min_branch_length", type=int, default=10)
    p.add_argument("--max_soma_distance", type=int, default=150)
    p.add_argument("--soma_dilation", type=int, default=3)
    # Optional denoise front-end. ``none`` (default) is a pure no-op so existing
    # behaviour is unchanged. ``n2v`` attempts to load a Noise2Void model from
    # ``--denoise_model`` and apply it before enhancement; if the package or model
    # is unavailable it logs a warning and falls back to the raw image.
    p.add_argument("--denoise", default="none", choices=["none", "n2v"])
    p.add_argument("--denoise_model", default=None)
    # Neurite detector. ``frangi`` (default) is the classical multiscale
    # vesselness above -- unchanged behaviour. ``cldice`` runs the trained 2D
    # U-Net (soft-clDice) from ml/neurite via bin/neurite_model.py, which clears
    # the ~0.25 F1 ceiling that classical fiber filters plateau at on faint
    # RGEDI neurites (held-out F1 0.645 vs 0.135 Frangi). Requires --checkpoint
    # and torch in the container; a GPU is strongly recommended (montage-scale
    # inference is slow on CPU).
    p.add_argument("--detector", default="frangi", choices=["frangi", "cldice"])
    p.add_argument("--checkpoint", default=None,
                   help="Path to the trained clDice .pt checkpoint (detector=cldice).")
    p.add_argument("--neurite_prob_threshold", type=float, default=0.5,
                   help="Probability cutoff [0-1] for the clDice neurite map.")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"],
                   help="Torch device for the clDice detector.")
    p.add_argument("--tile", type=int, default=0)
    return p.parse_args(argv)


# --------------------------------------------------------------------------- #
#  Core science — pure array in, measurements out (unit-testable, no DB/IO)
# --------------------------------------------------------------------------- #
@dataclass
class NeuriteMeasurement:
    """Per-cell neurite metrics."""

    cellid: int
    total_neurite_length: float          # summed skeleton length (px)
    n_branch_points: int                 # arborization: skeleton nodes deg>=3
    n_end_points: int                    # neurite tips (deg==1)
    n_primary_neurites: int              # branches leaving the soma
    max_branch_length: float             # longest single path from soma (px)
    n_skeleton_px: int                   # raw skeleton pixel count (QC)


def enhance_vesselness(
    img: np.ndarray, sigma_min: float, sigma_max: float, steps: int
) -> np.ndarray:
    """Multiscale tubeness enhancement, normalized to [0, 1].

    Uses skimage Frangi across a linear sigma range tuned to neurite radii.
    This is the step that rescues thin processes the global threshold was
    dropping.
    """
    from skimage.filters import frangi

    img = img.astype(np.float32)
    # Robust contrast normalization. Min-max normalization let a few bright
    # outliers (bright somata, debris, saturated pixels) set the top of the
    # range and compress the faint neurite signal into a sliver, so the fixed
    # threshold caught almost nothing. Clip to [p1, p99.5] instead. Negatives
    # (from background subtraction) are floored at 0 first.
    img = np.clip(img, 0.0, None)
    lo, hi = np.percentile(img, 1.0), np.percentile(img, 99.5)
    if hi > lo:
        img = np.clip((img - lo) / (hi - lo), 0.0, 1.0)
    else:
        img = np.zeros_like(img)
    sigmas = np.linspace(sigma_min, sigma_max, max(1, steps))
    resp = frangi(img, sigmas=sigmas, black_ridges=False)
    # Normalize by a high percentile, not the max: a single very strong ridge
    # response would otherwise scale every real neurite below the threshold.
    p = float(np.percentile(resp, 99.9))
    if p > 0:
        resp = np.clip(resp / p, 0.0, 1.0)
    return resp


def _threshold_vesselness(resp: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    """Binarize the vesselness response with the selected adaptive method.

    Replaces the original single global cut. Three methods are supported via
    ``args.threshold_method``:

    * ``global``: ``resp >= neurite_threshold`` (backward-compatible behaviour).
    * ``hysteresis`` (default): double-threshold via
      :func:`skimage.filters.apply_hysteresis_threshold`. Pixels above ``high``
      seed the mask and any pixel above ``low`` that is connected to a seed is
      kept. This recovers faint distal neurites that are continuous with a
      confident ridge -- the key RGEDI failure mode -- without flooding the
      background with isolated low-response speckle. ``low`` defaults to
      ``neurite_threshold`` and ``high`` to ``2 * neurite_threshold`` (clamped),
      overridable via ``--hysteresis_low`` / ``--hysteresis_high``.
    * ``otsu``: :func:`skimage.filters.threshold_otsu` on the nonzero response.

    Args:
        resp: vesselness response normalized to [0, 1].
        args: parsed CLI namespace (reads ``threshold_method``,
            ``neurite_threshold``, ``hysteresis_low``, ``hysteresis_high``).

    Returns:
        Boolean neurite foreground mask, same shape as ``resp``.
    """
    method = getattr(args, "threshold_method", "hysteresis")
    thr = float(args.neurite_threshold)

    if method == "global":
        return resp >= thr

    if method == "otsu":
        from skimage.filters import threshold_otsu
        nz = resp[resp > 0]
        if nz.size == 0:
            return np.zeros(resp.shape, dtype=bool)
        try:
            t = float(threshold_otsu(nz))
        except Exception:  # noqa: BLE001 - degenerate response -> fall back to global
            t = thr
        return resp >= t

    # hysteresis (default)
    from skimage.filters import apply_hysteresis_threshold
    low = getattr(args, "hysteresis_low", None)
    high = getattr(args, "hysteresis_high", None)
    low = thr if low is None else float(low)
    high = 2.0 * thr if high is None else float(high)
    # Keep a valid, in-range band: low <= high <= 1.0.
    high = min(1.0, max(high, low))
    if not (resp > 0).any():
        return np.zeros(resp.shape, dtype=bool)
    return apply_hysteresis_threshold(resp, low, high).astype(bool)


def _maybe_denoise(img: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    """Optionally denoise the morphology image before enhancement.

    Controlled by ``args.denoise``. ``none`` (default) returns the input
    unchanged (pure no-op). ``n2v`` attempts to load a Noise2Void model from
    ``args.denoise_model`` and apply it; the real model is provided by Track B.
    Any failure (missing package, missing/invalid model, prediction error) is
    logged and the raw image is returned -- this hook must never crash the run.

    Args:
        img: 2D morphology image.
        args: parsed CLI namespace (reads ``denoise``, ``denoise_model``).

    Returns:
        The denoised image, or ``img`` unchanged on the ``none`` path or on any
        failure.
    """
    method = getattr(args, "denoise", "none")
    if method in (None, "none"):
        return img
    if method == "n2v":
        model_path = getattr(args, "denoise_model", None)
        try:
            if not model_path or not os.path.isdir(model_path):
                raise FileNotFoundError(f"denoise_model not found: {model_path!r}")
            from n2v.models import N2V  # type: ignore
            basedir, name = os.path.split(os.path.normpath(model_path))
            model = N2V(config=None, name=name or ".", basedir=basedir or ".")
            den = model.predict(img.astype(np.float32), axes="YX")
            return np.asarray(den)
        except Exception as e:  # noqa: BLE001 - denoise is best-effort; never crash
            logger.warning("denoise=n2v unavailable (%s); using raw image", e)
            return img
    return img


def _skeleton_graph_metrics(skel: np.ndarray) -> tuple[int, int]:
    """Return (n_branch_points, n_end_points) from a boolean skeleton.

    Uses a 3x3 neighbor count on skeleton pixels: degree-1 = endpoint,
    degree>=3 = branch point. NumPy-only so it works without ``skan``.
    """
    from scipy.ndimage import convolve

    k = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
    nb = convolve(skel.astype(np.uint8), k, mode="constant", cval=0)
    nb = nb * skel  # only count on-skeleton pixels
    n_end = int(np.sum(nb == 1))
    n_branch = int(np.sum(nb >= 3))
    return n_branch, n_end


def measure_cell_neurites(
    morphology: np.ndarray,
    soma_labels: np.ndarray,
    args: argparse.Namespace,
) -> list[NeuriteMeasurement]:
    """Measure per-cell neurites given a morphology image and labeled somas.

    Args:
        morphology: 2D grayscale morphology image (the traced channel).
        soma_labels: 2D integer label image; each soma/cell has a unique id
            (0 = background), as produced by SEGMENTATION/CELLPOSE. In the
            pipeline these labels are the ``randomcellid`` values.
        args: parsed CLI namespace with the tuning parameters.

    Returns:
        One :class:`NeuriteMeasurement` per soma label present. ``cellid``
        holds the soma label (``randomcellid``).
    """
    from skimage.morphology import (
        skeletonize, binary_dilation, binary_closing, disk, remove_small_objects)

    # 0. Optional denoise front-end (no-op unless --denoise n2v is requested).
    morphology = _maybe_denoise(morphology, args)

    # 1. Detect the neurite foreground mask. ``frangi`` (default) is the
    #    classical multiscale vesselness + adaptive threshold. ``cldice`` swaps
    #    in the trained U-Net probability map (drop-in via bin/neurite_model.py);
    #    everything downstream (skeletonize -> attribution -> skan metrics -> DB)
    #    is identical for both detectors.
    if args.detector == "cldice":
        from neurite_model import predict_neurite_probmap
        prob = predict_neurite_probmap(
            morphology, checkpoint=args.checkpoint, device=args.device,
            normalize=True)
        neurite_mask = prob >= args.neurite_prob_threshold
    else:
        vness = enhance_vesselness(
            morphology,
            args.vesselness_sigma_min,
            args.vesselness_sigma_max,
            args.vesselness_sigma_steps,
        )
        neurite_mask = _threshold_vesselness(vness, args)
    # Clean the neurite mask before skeletonizing. The robust vesselness comes
    # through fragmented (real thin processes break into sub-10px pieces), so a
    # bare small-object filter would delete real-but-broken neurites along with
    # texture. Close first (bridge the gaps) so genuine neurites become large
    # connected components, THEN drop isolated speckle by size -- background
    # texture blobs go, real processes stay. Spur-pruning later only trims ends.
    neurite_mask = binary_closing(neurite_mask, disk(2))
    neurite_mask = remove_small_objects(
        neurite_mask, min_size=max(20, 4 * int(args.min_branch_length)))

    # 2. Union somas into the mask so skeletons connect to their cell body.
    somas_bin = soma_labels > 0
    if args.soma_dilation > 0:
        somas_bin = binary_dilation(somas_bin, disk(args.soma_dilation))
    fg = neurite_mask | somas_bin

    # 3. Skeletonize the whole field once.
    skel = skeletonize(fg)

    # 4. Prune spurs shorter than min_branch_length (iterative endpoint erosion).
    skel = _prune_spurs(skel, args.min_branch_length)

    # 5. Attribute skeleton pixels to the soma they are physically connected to
    #    ALONG the skeleton (connectivity), not merely the nearest centroid.
    owner = _attribute_skeleton_to_somas(
        skel, soma_labels, int(args.soma_dilation), int(args.max_soma_distance))

    results: list[NeuriteMeasurement] = []
    for cid in np.unique(soma_labels):
        if cid == 0:
            continue
        cell_skel = owner == cid
        n_px = int(cell_skel.sum())
        if n_px == 0:
            results.append(NeuriteMeasurement(int(cid), 0.0, 0, 0, 0, 0.0, 0))
            continue
        length = _skeleton_length(cell_skel)
        n_branch, n_end = _skeleton_graph_metrics(cell_skel)
        n_primary = _count_primary_neurites(cell_skel, soma_labels == cid, args.soma_dilation)
        max_len = _max_path_from_soma(cell_skel, soma_labels == cid)
        results.append(
            NeuriteMeasurement(
                cellid=int(cid),
                total_neurite_length=round(length, 3),
                n_branch_points=n_branch,
                n_end_points=n_end,
                n_primary_neurites=n_primary,
                max_branch_length=round(max_len, 3),
                n_skeleton_px=n_px,
            )
        )
    return results


def _skeleton_length(skel: np.ndarray) -> float:
    """Geometric skeleton length: straight steps=1, diagonal steps=sqrt(2)."""
    try:
        from skan import Skeleton
        if skel.sum() < 2:
            return 0.0
        return float(Skeleton(skel).path_lengths().sum())
    except Exception:
        # NumPy fallback: count edges weighted by distance.
        from scipy.ndimage import convolve
        ortho = convolve(skel.astype(np.uint8),
                         np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]),
                         mode="constant") * skel
        diag = convolve(skel.astype(np.uint8),
                        np.array([[1, 0, 1], [0, 0, 0], [1, 0, 1]]),
                        mode="constant") * skel
        return float(ortho.sum() / 2.0 + (diag.sum() / 2.0) * np.sqrt(2))


def _prune_spurs(skel: np.ndarray, min_len: int) -> np.ndarray:
    """Remove terminal branches shorter than ``min_len`` px by endpoint erosion."""
    from scipy.ndimage import convolve

    skel = skel.copy()
    k = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
    for _ in range(max(1, min_len)):
        nb = convolve(skel.astype(np.uint8), k, mode="constant") * skel
        endpoints = (nb == 1)
        if not endpoints.any():
            break
        skel[endpoints] = False
    return skel


def _count_primary_neurites(cell_skel: np.ndarray, soma: np.ndarray, dilation: int) -> int:
    """Count distinct skeleton branches touching the soma perimeter."""
    from scipy import ndimage as ndi
    from skimage.morphology import binary_dilation, disk

    ring = binary_dilation(soma, disk(max(1, dilation) + 1)) & ~soma
    emerging = cell_skel & ring
    if not emerging.any():
        return 0
    _, n = ndi.label(emerging, structure=np.ones((3, 3)))
    return int(n)


def _max_path_from_soma(cell_skel: np.ndarray, soma: np.ndarray) -> float:
    """Longest geodesic distance along the skeleton from the soma (px)."""
    from scipy import ndimage as ndi
    if cell_skel.sum() == 0:
        return 0.0
    seed = cell_skel & ndi.binary_dilation(soma, structure=np.ones((3, 3)))
    if not seed.any():
        return 0.0
    # BFS geodesic distance restricted to the skeleton.
    dist = np.full(cell_skel.shape, -1, dtype=np.int32)
    from collections import deque
    q = deque(zip(*np.where(seed)))
    for y, x in list(q):
        dist[y, x] = 0
    maxd = 0
    H, W = cell_skel.shape
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


def _attribute_skeleton_to_somas(
    skel: np.ndarray,
    soma_labels: np.ndarray,
    soma_dilation: int,
    max_soma_distance: int,
) -> np.ndarray:
    """Assign each skeleton pixel to the soma it is connected to along the skeleton.

    This replaces the previous Euclidean/Voronoi assignment (a whole-field
    ``distance_transform_edt`` that gave every skeleton pixel to its nearest soma
    *centroid* regardless of whether a physical process actually joined them).
    A skeleton branch that runs close to soma B but is only continuous with soma
    A is now correctly credited to A.

    Algorithm:

    1.  *Seed* the skeleton pixels that touch a soma (within a small Euclidean
        radius derived from ``soma_dilation``) with that soma's label. The
        Euclidean distance transform is used only to plant these sources.
    2.  Multi-source breadth-first search *along the 8-connected skeleton graph*
        propagates labels outward. Because BFS expands in order of graph
        distance, every pixel receives the label of the nearest seed measured
        along the skeleton. A component that seeds from two somas is therefore
        split at the geodesic midpoint (nearest-along-skeleton), and a component
        connected to a single soma is fully credited to it.
    3.  *Orphan* components that never reach a soma are dropped, except for a
        near-miss fallback: if a component's closest pixel is within
        ``max_soma_distance`` px of a soma, the whole component is attached to
        that soma. ``max_soma_distance`` is thus only a fallback for small gaps,
        not the primary attribution rule.

    Args:
        skel: 2D boolean pruned skeleton of the neurite+soma foreground.
        soma_labels: 2D integer soma-label image (0 = background); labels are the
            ``randomcellid`` values.
        soma_dilation: soma dilation (px) used upstream; sets the seed radius.
        max_soma_distance: fallback attachment radius (px) for orphan components.

    Returns:
        Integer owner-label image the same shape as ``skel``: each skeleton pixel
        holds the soma label it was attributed to, 0 where unattributed.
    """
    from collections import deque
    from scipy import ndimage as ndi

    owner = np.zeros_like(soma_labels)
    if not skel.any() or not (soma_labels > 0).any():
        return owner

    H, W = skel.shape
    # Euclidean nearest-soma label + distance, used only to seed sources (step 1)
    # and for the orphan fallback (step 3).
    inv = soma_labels == 0
    dist, (iy, ix) = ndi.distance_transform_edt(inv, return_indices=True)
    nearest_label = soma_labels[iy, ix]

    seed_radius = max(int(soma_dilation) + 1, 2)
    seed_mask = skel & (dist <= seed_radius)

    # Step 2: multi-source BFS along the skeleton.
    dist_along = np.full(skel.shape, -1, dtype=np.int32)
    q: deque[tuple[int, int]] = deque()
    ys, xs = np.where(seed_mask)
    for y, x in zip(ys.tolist(), xs.tolist()):
        owner[y, x] = nearest_label[y, x]
        dist_along[y, x] = 0
        q.append((y, x))
    while q:
        y, x = q.popleft()
        cur = owner[y, x]
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                ny, nx = y + dy, x + dx
                if 0 <= ny < H and 0 <= nx < W and skel[ny, nx] and dist_along[ny, nx] < 0:
                    dist_along[ny, nx] = dist_along[y, x] + 1
                    owner[ny, nx] = cur
                    q.append((ny, nx))

    # Step 3: near-miss fallback for components that never touched a soma.
    orphan = skel & (dist_along < 0)
    if orphan.any() and max_soma_distance > 0:
        lbl, n = ndi.label(orphan, structure=np.ones((3, 3)))
        for comp in range(1, n + 1):
            comp_mask = lbl == comp
            cy, cx = np.where(comp_mask)
            comp_dist = dist[cy, cx]
            k = int(np.argmin(comp_dist))
            if comp_dist[k] <= max_soma_distance:
                owner[comp_mask] = nearest_label[cy[k], cx[k]]
    return owner


# --------------------------------------------------------------------------- #
#  Pipeline orchestration — mirrors intensity.py's Intensity.run()
# --------------------------------------------------------------------------- #
class Neurite:
    """Runs neurite quantification over an experiment, one tile at a time.

    Deliberately parallels ``intensity.py``'s ``Intensity`` class so the two
    read the same way: build a tiledata DataFrame for the morphology channel,
    group by (well, timepoint), and for each morphology tile load the soma mask
    + image, measure per cell, and upsert rows into ``neuritecelldata``.
    """

    def __init__(self, opt):
        # Imports are inside the class so the pure-array core above can be
        # imported (and unit-tested) on machines without the pipeline DB stack.
        from normalization import Normalize
        from db_util import Ops

        self.opt = opt
        self.Norm = Normalize(self.opt)
        self.Op = Ops(self.opt)

    def run(self) -> None:
        import imageio
        import pandas as pd
        from sql import Database

        Db = Database()
        tiledata_df = self.Norm.get_df_for_training(['channeldata'])
        morph_df = tiledata_df[tiledata_df.channel == self.opt.morphology_channel]
        if self.opt.tile:  # 0 == all tiles
            morph_df = morph_df[morph_df.tile == self.opt.tile]
        if morph_df.empty:
            logger.warning('No tiles for morphology channel %s', self.opt.morphology_channel)
            print('Done.')
            return

        total = 0
        for (well, timepoint), df in morph_df.groupby(['well', 'timepoint']):
            if df.maskpath.iloc[0] is None:
                print(f'{well} T{timepoint} has null maskpath. Skipping. Check morphology channel.')
                continue
            welldata_id = df.welldata_id.iloc[0]
            morph_channel_uuid = Db.get_table_uuid(
                'channeldata', dict(channel=self.opt.morphology_channel, welldata_id=welldata_id))

            # Background image (per well/timepoint) for the morphology channel,
            # used only when the tile is not an already-normalized aligned TIFF.
            built_bg = False
            try:
                self.Norm.get_background_image(df, well, timepoint)
                built_bg = True
            except Exception as e:  # noqa: BLE001 - bg is best-effort; tracing tolerates raw images
                logger.warning('background image failed for %s T%s: %s', well, timepoint, e)

            for _, row in df.iterrows():
                if row.maskpath is None:
                    continue
                try:
                    labelled_mask = imageio.v3.imread(row.maskpath)  # labels == randomcellid
                except Exception as e:  # noqa: BLE001 - a corrupt tile must not kill the run
                    logger.warning('well=%s tp=%s tile=%s: cannot read mask %s: %s; '
                                   'skipping tile', well, timepoint,
                                   getattr(row, 'tile', '?'), row.maskpath, e)
                    continue
                # A soma mask must be a non-empty 2D label image. Some tiles have
                # a corrupt/empty mask (e.g. a TIFF with no pages -> shape (0,)),
                # which would otherwise crash downstream morphology ops.
                if getattr(labelled_mask, 'ndim', 0) != 2 or labelled_mask.size == 0:
                    logger.warning('well=%s tp=%s tile=%s: mask is not a non-empty 2D '
                                   'label image (shape=%s from %s); skipping tile',
                                   well, timepoint, getattr(row, 'tile', '?'),
                                   getattr(labelled_mask, 'shape', None), row.maskpath)
                    continue

                aligned = getattr(row, 'alignedtilepath', None)
                img_path = aligned if pd.notna(aligned) else row.filename
                if img_path is None or not os.path.exists(img_path):
                    logger.warning('missing image for %s T%s tile %s: %s',
                                   well, timepoint, getattr(row, 'tile', '?'), img_path)
                    continue
                img = imageio.v3.imread(img_path)

                # Skip background subtraction on aligned TIFFs (already normalized).
                if built_bg and not pd.notna(aligned):
                    try:
                        img = self.Norm.image_bg_correction[self.opt.img_norm_name](img, well, timepoint)
                    except Exception as e:  # noqa: BLE001
                        logger.warning('bg correction failed for %s T%s: %s', well, timepoint, e)

                celldata_df = Db.get_df_from_query('celldata', dict(tiledata_id=row.tiledata_id))
                if celldata_df.empty:
                    continue
                by_rcid = {int(c.randomcellid): c for _, c in celldata_df.iterrows()}

                measurements = measure_cell_neurites(img, labelled_mask, self.opt)
                dcts = []
                for m in measurements:
                    c = by_rcid.get(int(m.cellid))
                    if c is None:  # mask label with no celldata row (e.g. edge object)
                        continue
                    dcts.append(dict(
                        experimentdata_id=c.experimentdata_id,
                        welldata_id=c.welldata_id,
                        tiledata_id=c.tiledata_id,
                        celldata_id=c.id,
                        channeldata_id=morph_channel_uuid,
                        total_neurite_length=m.total_neurite_length,
                        n_branch_points=m.n_branch_points,
                        n_end_points=m.n_end_points,
                        n_primary_neurites=m.n_primary_neurites,
                        max_branch_length=m.max_branch_length,
                        n_skeleton_px=m.n_skeleton_px,
                    ))

                # Idempotent upsert: clear this tile+channel's rows, then insert.
                Db.delete_based_on_duplicate_name(
                    tablename='neuritecelldata',
                    kwargs=dict(tiledata_id=row.tiledata_id, channeldata_id=morph_channel_uuid))
                if dcts:
                    Db.add_row(tablename='neuritecelldata', dct=dcts)
                total += len(dcts)
                logger.info('well=%s tp=%s tile=%s cells=%d', well, timepoint,
                            getattr(row, 'tile', '?'), len(dcts))

            if built_bg:
                try:
                    del self.Norm.backgrounds[well][timepoint]
                except Exception:  # noqa: BLE001
                    pass

        logger.info('NEURITE done: %d cell-rows written', total)
        print('Done.')


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args(argv)
    print(args)
    logger.warning('Running Neurite from Database.')
    Neurite(args).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Skeleton F1 metric — identical to the benchmark scorer.

This is a verbatim reuse of the distance-transform precision/recall/F1 at pixel
tolerance 3 from ``scratchpad/score_all.py`` / ``score_traces.py``, so training
numbers are directly comparable to the CellProfiler / Frangi benchmarks. Both
prediction and ground truth are thinned to 1-px skeletons before scoring;
recall counts GT skeleton pixels within ``tol`` of a predicted pixel, precision
counts predicted skeleton pixels within ``tol`` of a GT pixel.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize

TOL = 3


def skeleton_f1(
    gt: np.ndarray, det: np.ndarray, tol: int = TOL
) -> Tuple[float, float, float]:
    """Distance-transform precision/recall/F1 between two masks.

    Args:
        gt: Ground-truth mask (bool or {0,1}), (H, W). Skeletonized internally.
        det: Detected/predicted mask (bool or {0,1}), (H, W). Skeletonized
            internally.
        tol: Pixel tolerance for a match (default 3).

    Returns:
        ``(precision, recall, f1)`` as floats. Returns zeros if either skeleton
        is empty.
    """
    gt = skeletonize(np.asarray(gt) > 0)
    det = skeletonize(np.asarray(det) > 0)
    if gt.sum() == 0 or det.sum() == 0:
        return 0.0, 0.0, 0.0
    dg = distance_transform_edt(~gt)
    dd = distance_transform_edt(~det)
    rec = float((dd[gt] <= tol).mean())
    pre = float((dg[det] <= tol).mean())
    f1 = 2 * pre * rec / (pre + rec) if (pre + rec) > 0 else 0.0
    return pre, rec, f1


def best_threshold_f1(
    gt: np.ndarray,
    prob: np.ndarray,
    thresholds: Tuple[float, ...] = (0.3, 0.4, 0.5, 0.6, 0.7),
    tol: int = TOL,
) -> Tuple[float, float]:
    """Sweep probability thresholds and return the best (threshold, F1).

    Args:
        gt: Ground-truth mask, (H, W).
        prob: Predicted probability map in [0, 1], (H, W).
        thresholds: Candidate thresholds to binarise ``prob``.
        tol: Pixel tolerance passed to :func:`skeleton_f1`.

    Returns:
        ``(best_threshold, best_f1)``.
    """
    best_t, best_f = thresholds[0], -1.0
    for t in thresholds:
        _, _, f = skeleton_f1(gt, prob >= t, tol=tol)
        if f > best_f:
            best_t, best_f = t, f
    return best_t, best_f

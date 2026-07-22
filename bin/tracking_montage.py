#!/usr/bin/env python
"""Cell tracking on montaged microscopy images via minimum-cost flow optimization.

Reads encoded segmentation masks from the database, matches cell identities
across timepoints using overlap, proximity, or motion-prediction strategies
(the last leveraging the Hungarian algorithm), and writes tracked masks plus
a summary CSV with per-cell intensity measurements.
"""

import sys
import cv2
import numpy as np
import logging
import argparse
import os
import fcntl
import signal
from typing import Any, Dict, List, Optional, Tuple

from db_util import Ops
from sql import Database
import pandas as pd
import collections
import tifffile
import datetime
from time import time
from multiprocessing import Pool, cpu_count
from multiprocessing import shared_memory
from scipy.optimize import linear_sum_assignment

logger = logging.getLogger("TrackingDB")
logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

now = datetime.datetime.now()
TIMESTAMP = '%d%02d%02d%02d%02d' % (now.year, now.month, now.day, now.hour, now.minute)
print(f'Starting tracking processing at {now.strftime("%Y-%m-%d %H:%M:%S")}')

fink_log_dir = './finkbeiner_logs'
if not os.path.exists(fink_log_dir):
    os.makedirs(fink_log_dir)
logname = os.path.join(fink_log_dir, f'Tracking-log_{TIMESTAMP}.log')
fh = logging.FileHandler(logname)
logger.addHandler(fh)
# Also mirror this logger to the experiment-scoped debug log
# (<params.output_path>/pipeline_debug.log) shared across all parallel wells.
from experiment_logger import attach_experiment_log  # noqa: E402
attach_experiment_log(logger, os.environ.get('NEXTFLOW_OUTPUT_PATH', ''), 'TRACK_MONTAGE')
logger.warning('Running OPTIMIZED Tracking from Database.')

def read_tiff_safe(file_path: str) -> Optional[np.ndarray]:
    """Read a TIFF image with a fast primary reader and OpenCV fallback.

    Args:
        file_path: Absolute path to the TIFF file.

    Returns:
        The image as a NumPy array, or ``None`` if both readers fail.
    """
    try:
        img = tifffile.imread(file_path)
        return img
    except Exception:
        try:
            img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
            return img if img is not None else None
        except Exception:
            return None

def process_label_contour(args: Tuple[np.ndarray, int]) -> Optional["Cell"]:
    """Extract the largest external contour for a single label in a mask.

    Designed to be called via ``multiprocessing.Pool.map``.

    Args:
        args: A tuple of (mask_array, label_value).

    Returns:
        A ``Cell`` wrapping the contour, or ``None`` if no contour is found.
    """
    mask, label = args
    try:
        bin_mask = (mask == label).astype(np.uint8) * 255
        cnts, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            cnt = max(cnts, key=cv2.contourArea)
            return Cell(cnt, randomcellid_montage=label)
        return None
    except Exception:
        return None

# Per-worker cache of attached SharedMemory blocks by name. The parent writes
# each timepoint's mask into a reusable shared block; workers attach lazily on
# first sight of a new name and reuse the numpy view across every subsequent
# label task. This lets a single Pool (created once per tracker.run()) service
# every timepoint without re-pickling the mask (previously the mask was pinned
# via a Pool initializer, but that forced a fresh Pool per timepoint).
_WORKER_SHM_CACHE: Dict[str, Tuple["shared_memory.SharedMemory", np.ndarray]] = {}


def _init_label_worker() -> None:
    """Pool initializer.

    No-op today (shared memory is attached lazily inside the worker on the
    first task that references a given block name), but kept as an explicit
    hook so future setup — e.g. limiting OpenCV thread counts per worker —
    has an obvious place to live.
    """
    return None


def _process_label_shm(
    args: Tuple[str, Tuple[int, ...], str, int],
) -> Optional["Cell"]:
    """Pool worker: attach shared mask by name (once), extract contour for label.

    The parent broadcasts ``(shm_name, shape, dtype_str, label)`` per label.
    The first call for a new ``shm_name`` attaches to the block; every
    subsequent call reuses the cached numpy view.  When the parent
    reallocates (new block name — e.g. the mask grew), the worker evicts and
    closes its old cache entry so at most one shared block is held.

    Args:
        args: ``(shm_name, shape, dtype_str, label)``.

    Returns:
        A ``Cell`` for ``label`` or ``None`` if no contour was found.
    """
    shm_name, shape, dtype_str, label = args
    cache = _WORKER_SHM_CACHE.get(shm_name)
    if cache is None:
        # A new block name means the parent replaced the shared buffer.
        # Release any old attachments so we don't accumulate them.
        for old_name in list(_WORKER_SHM_CACHE.keys()):
            old_shm, _ = _WORKER_SHM_CACHE.pop(old_name)
            try:
                old_shm.close()
            except Exception:
                pass
        shm = shared_memory.SharedMemory(name=shm_name)
        arr = np.ndarray(shape, dtype=np.dtype(dtype_str), buffer=shm.buf)
        _WORKER_SHM_CACHE[shm_name] = (shm, arr)
        cache = _WORKER_SHM_CACHE[shm_name]
    _, mask = cache
    return process_label_contour((mask, label))

class MotionTracker:
    """Linear motion tracker that predicts cell positions across frames.

    Maintains a rolling history of positions and velocities (last 5 frames)
    and uses an exponential moving average to smooth velocity estimates.

    Attributes:
        position: Current estimated position (x, y).
        velocity: Current estimated velocity vector.
        age: Total number of ``predict`` calls.
        hits: Total number of ``update`` calls.
        hit_streak: Consecutive frames with a successful update.
        time_since_update: Frames since the last ``update``.
    """

    def __init__(self, initial_pos: Tuple[float, float]) -> None:
        self.position: np.ndarray = np.array(initial_pos, dtype=np.float32)
        self.velocity: np.ndarray = np.array([0.0, 0.0], dtype=np.float32)
        self.positions_history: List[np.ndarray] = [np.array(initial_pos, dtype=np.float32)]
        self.velocities_history: List[np.ndarray] = []

        self.age: int = 0
        self.hits: int = 0
        self.hit_streak: int = 0
        self.time_since_update: int = 0

    def predict(self) -> np.ndarray:
        """Predict the next position using a constant-velocity model.

        Returns:
            Predicted (x, y) position as a float32 array.
        """
        predicted_pos = self.position + self.velocity
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        return predicted_pos

    def update(self, measurement: Tuple[float, float]) -> None:
        """Incorporate a new observed position and refine the velocity estimate.

        Velocity is smoothed with an exponential moving average (alpha = 0.3).
        Only the most recent 5 positions and velocities are retained.

        Args:
            measurement: Observed (x, y) centroid of the cell.
        """
        if len(self.positions_history) > 0:
            new_pos = np.array(measurement, dtype=np.float32)
            old_pos = self.positions_history[-1]
            velocity = new_pos - old_pos

            # Exponential moving average for velocity smoothing
            alpha = 0.3
            self.velocity = alpha * velocity + (1 - alpha) * self.velocity
            self.velocities_history.append(self.velocity.copy())

        self.position = np.array(measurement, dtype=np.float32)
        self.positions_history.append(self.position.copy())

        # Retain only the most recent history window
        if len(self.positions_history) > 5:
            self.positions_history = self.positions_history[-5:]
        if len(self.velocities_history) > 5:
            self.velocities_history = self.velocities_history[-5:]

        self.hits += 1
        self.hit_streak += 1
        self.time_since_update = 0

class Cell:
    """Lightweight wrapper around an OpenCV contour for a single segmented cell.

    Caches the minimum enclosing circle to avoid repeated computation
    during overlap and distance evaluations.

    Attributes:
        cnt: OpenCV contour array.
        randomcellid_montage: Label value from the encoded segmentation mask.
    """

    def __init__(
        self,
        cnt: np.ndarray,
        randomcellid_montage: Optional[int] = None,
    ) -> None:
        self.cnt = cnt
        self.randomcellid_montage = randomcellid_montage
        self._circle_cache: Optional[Tuple[Tuple[float, float], float]] = None
        self._center_cache: Optional[Tuple[float, float]] = None

    def get_circle(self) -> Tuple[Tuple[float, float], float]:
        """Return the minimum enclosing circle ``((cx, cy), radius)``."""
        if self._circle_cache is None:
            self._circle_cache = cv2.minEnclosingCircle(self.cnt)
        return self._circle_cache

    def get_center(self) -> Tuple[float, float]:
        """Return the center ``(cx, cy)`` of the minimum enclosing circle."""
        if self._center_cache is None:
            self._center_cache = cv2.minEnclosingCircle(self.cnt)[0]
        return self._center_cache

    def evaluate_overlap(self, circle2: Tuple[Tuple[float, float], float]) -> bool:
        """Check whether this cell overlaps with another enclosing circle.

        Two cells are considered overlapping when the distance between
        their centres is less than 80 % of the sum of their radii.

        Args:
            circle2: ``((cx, cy), radius)`` of the other cell.

        Returns:
            ``True`` if the circles overlap.
        """
        c2, r2 = circle2
        c1, r1 = self.get_circle()
        dist = np.hypot(c1[0] - c2[0], c1[1] - c2[1])
        return dist < (r1 + r2) * 0.8

    def evaluate_dist(self, circle2: Tuple[Tuple[float, float], float]) -> float:
        """Compute the Euclidean distance between this cell and another circle centre.

        Args:
            circle2: ``((cx, cy), radius)`` of the other cell.

        Returns:
            Distance in pixels between the two circle centres.
        """
        c2, _ = circle2
        c1, _ = self.get_circle()
        return np.hypot(c1[0] - c2[0], c1[1] - c2[1])

def sort_cell_info_by_index(
    time_dict: collections.OrderedDict,
    time_list: List[int],
) -> collections.OrderedDict:
    """Sort cell records within each timepoint by their assigned track ID.

    Args:
        time_dict: Mapping of timepoint -> list of ``[track_id, Cell]`` pairs.
        time_list: Ordered list of timepoint keys.

    Returns:
        The same ``time_dict`` with each timepoint's list sorted by track ID.
    """
    for tp in time_list:
        time_dict[tp] = sorted(
            [(int(idx), obj) for idx, obj in time_dict[tp]],
            key=lambda x: x[0]
        )
    return time_dict

def populate_cell_ind_overlap(
    time_dict: collections.OrderedDict,
    time_list: List[int],
) -> collections.OrderedDict:
    """Assign track IDs by checking enclosing-circle overlap with the previous frame.

    Cells in the first timepoint receive sequential IDs starting at 1.
    In subsequent frames each cell is matched to the first overlapping
    cell from the prior frame; unmatched cells get a new ID.

    Args:
        time_dict: Mapping of timepoint -> list of ``[track_id, Cell]`` pairs.
        time_list: Ordered list of timepoint keys.

    Returns:
        Updated ``time_dict`` with track IDs populated.
    """
    first = time_list[0]
    for i, rec in enumerate(time_dict[first], 1):
        rec[0] = i
    counter = len(time_dict[first]) + 1

    for i in range(1, len(time_list)):
        curr, prev = time_list[i], time_list[i - 1]
        # Pre-compute circles for previous timepoint to avoid repeated calculations
        prev_circles = [(p[0], p[1].get_circle()) for p in time_dict[prev]]
        
        for rec in time_dict[curr]:
            circ = rec[1].get_circle()
            matched = False
            for prev_id, prev_circ in prev_circles:
                if rec[1].evaluate_overlap(prev_circ):
                    rec[0] = prev_id
                    matched = True
                    break
            if not matched:
                rec[0] = counter
                counter += 1

    return time_dict

def populate_cell_ind_closest(
    time_dict: collections.OrderedDict,
    time_list: List[int],
    max_dist: float = 100,
) -> collections.OrderedDict:
    """Assign track IDs by nearest-neighbour matching to the previous frame.

    Each cell is linked to the closest cell in the prior timepoint. If the
    nearest distance exceeds ``max_dist``, a new track ID is assigned.

    Args:
        time_dict: Mapping of timepoint -> list of ``[track_id, Cell]`` pairs.
        time_list: Ordered list of timepoint keys.
        max_dist: Maximum pixel distance for a valid match.

    Returns:
        Updated ``time_dict`` with track IDs populated.
    """
    first = time_list[0]
    for i, rec in enumerate(time_dict[first], 1):
        rec[0] = i
    counter = len(time_dict[first]) + 1

    for i in range(1, len(time_list)):
        curr, prev = time_list[i], time_list[i - 1]
        # Pre-compute circles for previous timepoint to avoid repeated calculations
        prev_circles = [(p[0], p[1].get_circle()) for p in time_dict[prev]]
        
        for rec in time_dict[curr]:
            circ = rec[1].get_circle()
            best = float('inf')
            best_id = None
            for prev_id, prev_circ in prev_circles:
                d = rec[1].evaluate_dist(prev_circ)
                if d < best:
                    best, best_id = d, prev_id
            if best > max_dist:
                rec[0] = counter
                counter += 1
            else:
                rec[0] = best_id

    return time_dict

def populate_cell_ind_motion(
    time_dict: collections.OrderedDict,
    time_list: List[int],
    max_dist: float = 100,
) -> collections.OrderedDict:
    """Assign track IDs using linear motion prediction and the Hungarian algorithm.

    For each frame, existing trackers predict their next position.  A cost
    matrix of predicted-vs-observed distances is built and solved via
    ``scipy.optimize.linear_sum_assignment``.  Unmatched detections start
    new tracks; trackers without updates for 3+ frames are pruned.

    Args:
        time_dict: Mapping of timepoint -> list of ``[track_id, Cell]`` pairs.
        time_list: Ordered list of timepoint keys.
        max_dist: Maximum pixel distance for a valid assignment.

    Returns:
        Updated ``time_dict`` with track IDs populated.
    """
    first = time_list[0]
    
    # Initialize trackers for first timepoint
    trackers = {}
    for i, rec in enumerate(time_dict[first], 1):
        rec[0] = i
        center = rec[1].get_center()
        trackers[i] = MotionTracker(center)
    
    counter = len(time_dict[first]) + 1

    for i in range(1, len(time_list)):
        curr, prev = time_list[i], time_list[i - 1]
        
        # Predict positions for all existing trackers
        predicted_positions = {}
        for tracker_id, tracker in trackers.items():
            predicted_pos = tracker.predict()
            predicted_positions[tracker_id] = predicted_pos
        
        # Get current detections
        current_detections = []
        for rec in time_dict[curr]:
            center = rec[1].get_center()
            current_detections.append(center)
        
        if not current_detections:
            # No detections in current frame, continue with predictions
            continue
            
        # Create cost matrix for Hungarian algorithm
        num_trackers = len(trackers)
        num_detections = len(current_detections)
        
        if num_trackers == 0:
            # No existing trackers, assign new IDs to all detections
            for j, rec in enumerate(time_dict[curr]):
                rec[0] = counter
                center = rec[1].get_center()
                trackers[counter] = MotionTracker(center)
                counter += 1
            continue
        
        # Create cost matrix (distance between predicted and actual positions)
        cost_matrix = np.full((num_trackers, num_detections), max_dist * 2, dtype=np.float32)
        
        tracker_ids = list(trackers.keys())
        for t_idx, tracker_id in enumerate(tracker_ids):
            predicted_pos = predicted_positions[tracker_id]
            for d_idx, detection_pos in enumerate(current_detections):
                distance = np.hypot(predicted_pos[0] - detection_pos[0], 
                                  predicted_pos[1] - detection_pos[1])
                if distance <= max_dist:
                    cost_matrix[t_idx, d_idx] = distance
        
        # Use Hungarian algorithm to find optimal assignments
        if num_trackers <= num_detections:
            # Standard assignment problem
            tracker_indices, detection_indices = linear_sum_assignment(cost_matrix)
        else:
            # More trackers than detections - need to handle differently
            # Transpose the problem: assign detections to trackers
            detection_indices, tracker_indices = linear_sum_assignment(cost_matrix.T)
        
        # Process assignments
        assigned_detections = set()
        assigned_trackers = set()
        
        for t_idx, d_idx in zip(tracker_indices, detection_indices):
            if cost_matrix[t_idx, d_idx] < max_dist:
                tracker_id = tracker_ids[t_idx]
                detection_pos = current_detections[d_idx]
                
                # Update tracker with new measurement
                trackers[tracker_id].update(detection_pos)
                
                # Assign ID to detection
                time_dict[curr][d_idx][0] = tracker_id
                
                assigned_detections.add(d_idx)
                assigned_trackers.add(tracker_id)
        
        # Handle unassigned detections (new cells)
        for d_idx, rec in enumerate(time_dict[curr]):
            if d_idx not in assigned_detections:
                rec[0] = counter
                center = rec[1].get_center()
                trackers[counter] = MotionTracker(center)
                counter += 1
        
        # Remove trackers that haven't been updated for too long
        trackers_to_remove = []
        for tracker_id, tracker in trackers.items():
            if tracker.time_since_update > 3:  # Remove after 3 frames without update
                trackers_to_remove.append(tracker_id)
        
        for tracker_id in trackers_to_remove:
            del trackers[tracker_id]

    return time_dict

class MontageDBTracker:
    """Orchestrates cell tracking across montaged wells using database-stored masks.

    Reads encoded segmentation masks from the PostgreSQL database, applies one
    of three tracking strategies (overlap, proximity, or motion-prediction),
    writes tracked masks to disk, and records per-cell intensity measurements
    in a summary CSV.

    Attributes:
        experiment: Experiment name in the database.
        track_type: ``'overlap'`` or ``'proximity'``.
        max_dist: Maximum pixel distance for proximity / motion matching.
        target_channel: Channels for intensity measurement (not for tracking).
        motion: Whether to use the Hungarian-algorithm motion tracker.
    """

    def __init__(
        self,
        experiment: str,
        track_type: str,
        max_dist: int,
        target_channel: str,
        motion: bool = False,
    ) -> None:
        self.Db = Database()
        self.experiment = experiment
        self.track_type = track_type
        self.max_dist = max_dist
        self.target_channel = target_channel.split(',')
        self.motion = motion
        self.analysisdir = ""
        # Cache for path transformations
        self._path_cache = {}
        
        # Performance monitoring
        self.start_time = None
        self.processed_wells = 0
        self.total_wells = 0
        
        # Tracking statistics
        self.tracking_stats = {
            'well_stats': {},
            'overall_stats': {
                'total_cells_tracked': 0,
                'total_timepoints': 0,
                'total_wells': 0,
                'motion_enabled': motion,
                'tracking_method': track_type
            }
        }
        
        # Dynamic core allocation for parallel processing
        available_cores = cpu_count()
        self.max_cores = max(1, int(available_cores * 0.75))  # Use 75% of available cores

        # Persistent worker pool + shared-memory block. Both are created lazily
        # on the first contour extraction and reused across every timepoint /
        # well processed by this tracker instance. They are torn down in the
        # ``finally`` block of ``run()``.
        self._pool: Optional[Pool] = None
        self._shm: Optional[shared_memory.SharedMemory] = None
        self._shm_capacity: int = 0

        logger.info(f"Initialized MontageDBTracker for experiment {experiment}")
        logger.info(f"Using {self.max_cores} cores out of {available_cores} available (75%)")
        if self.motion:
            logger.info("Motion tracking with linear prediction and Hungarian algorithm enabled")

    def _ensure_pool(self) -> Pool:
        """Create the worker pool on first use; return the cached instance thereafter.

        The pool is shared across every timepoint / well handled by this
        tracker run, so the fork + ``import numpy/cv2`` cost is paid once.
        """
        if self._pool is None:
            self._pool = Pool(
                processes=self.max_cores,
                initializer=_init_label_worker,
            )
            logger.info(f"Started tracking worker pool with {self.max_cores} processes")
        return self._pool

    def _prepare_shm_for_mask(
        self,
        mask: np.ndarray,
    ) -> Tuple[str, Tuple[int, ...], str]:
        """Publish ``mask`` into a reusable shared-memory buffer.

        Reuses the existing block when it has enough capacity (grow-only);
        otherwise closes+unlinks the old block and allocates a new one — in
        which case workers detect the name change on their next task and
        re-attach.

        Args:
            mask: 2-D label image to make visible to worker processes.

        Returns:
            ``(shm_name, shape, dtype_str)`` — the values workers need to
            attach and interpret the buffer.
        """
        # numpy views need contiguous memory; readers usually give us that,
        # but guard against a stray view slipping through.
        if not mask.flags['C_CONTIGUOUS']:
            mask = np.ascontiguousarray(mask)
        nbytes = mask.nbytes
        if self._shm is None or self._shm_capacity < nbytes:
            if self._shm is not None:
                try:
                    self._shm.close()
                    self._shm.unlink()
                except Exception:
                    pass
            self._shm = shared_memory.SharedMemory(create=True, size=nbytes)
            self._shm_capacity = nbytes
        view = np.ndarray(mask.shape, dtype=mask.dtype, buffer=self._shm.buf)
        view[:] = mask
        return self._shm.name, tuple(mask.shape), mask.dtype.str

    def _process_contours_pooled(
        self,
        mask: np.ndarray,
        labels: np.ndarray,
    ) -> List["Cell"]:
        """Extract contours for every label in ``mask`` using the persistent pool.

        Falls back to sequential processing for tiny label counts to avoid
        IPC overhead. For larger counts the mask is published once via
        shared memory and each worker task carries only the integer label
        plus the (small) shm descriptor.

        Args:
            mask: Encoded segmentation mask.
            labels: Non-zero label values to extract.

        Returns:
            List of ``Cell`` objects (only successful contour extractions).
        """
        if len(labels) < 50:
            results: List["Cell"] = []
            for lb in labels:
                cell = process_label_contour((mask, int(lb)))
                if cell:
                    results.append(cell)
            return results

        pool = self._ensure_pool()
        shm_name, shape, dtype_str = self._prepare_shm_for_mask(mask)
        task_args = [(shm_name, shape, dtype_str, int(lb)) for lb in labels]
        results = pool.map(_process_label_shm, task_args)
        return [c for c in results if c is not None]

    def _cleanup_pool(self) -> None:
        """Tear down the worker pool and release the shared-memory buffer.

        Safe to call multiple times. Invoked from ``run()``'s ``finally`` so
        the resources are released whether the run completed normally, hit
        an exception, or was interrupted by SIGINT / SIGTERM (Slurm kill).
        """
        pool = self._pool
        if pool is not None:
            try:
                pool.close()
                pool.join()
            except Exception:
                try:
                    pool.terminate()
                    pool.join()
                except Exception:
                    pass
            self._pool = None
        if self._shm is not None:
            try:
                self._shm.close()
            except Exception:
                pass
            try:
                self._shm.unlink()
            except Exception:
                pass
            self._shm = None
            self._shm_capacity = 0
    
    def _get_channel_path(self, mask_path: str, channel: str) -> str:
        """Derive the aligned/montaged image path for a given channel.

        Results are cached so repeated lookups for the same
        ``(mask_path, channel)`` pair are free.

        Args:
            mask_path: Path to the encoded mask TIFF.
            channel: Target channel name (e.g. ``'Cy5'``).

        Returns:
            Path to the corresponding channel image.
        """
        cache_key = (mask_path, channel, self.selected)
        if cache_key in self._path_cache:
            return self._path_cache[cache_key]
        
        if self.selected:
            aligned_path = mask_path.replace('/CellMasksMontage/', '/AlignedMontages/')
        else:
            aligned_path = mask_path.replace('/CellMasksMontage/', '/MontagedImages/')

        # Extract and modify filename
        filename = os.path.basename(aligned_path)
        dirname = os.path.dirname(aligned_path)
        
        # Replace channel name in filename
        parts = filename.split('_')
        matched = False
        for i, part in enumerate(parts):
            if any(part.startswith(prefix) for prefix in ['Epi-', 'DAPI', 'Cy', 'FITC', 'RFP', 'Confocal-']):
                parts[i] = channel
                matched = True
                break
        if not matched:
            # No recognized channel token: the filename is used unchanged,
            # which means intensities would be measured from the WRONG channel
            # image. Surface it instead of silently proceeding.
            logger.warning(
                f"_get_channel_path: no known channel prefix found in "
                f"'{filename}'; cannot substitute channel '{channel}'. "
                f"Intensity for this channel may be read from the wrong image."
            )
        filename = '_'.join(parts)
        aligned_path = os.path.join(dirname, filename)

        # Convert file extension
        if self.selected:
            aligned_path = aligned_path.replace('_MONTAGE_ALIGNED_ENCODED.tif', '_MONTAGE_ALIGNED.tif')
        else:
            aligned_path = aligned_path.replace('_MONTAGE_ENCODED.tif', '_MONTAGE.tif')
        
        self._path_cache[cache_key] = aligned_path
        return aligned_path

    def gather_encoded_from_db(
        self,
        wells: List[str],
        channel_marker: str = "_MONTAGE_ALIGNED_ENCODED",
    ) -> Tuple[Dict[str, collections.OrderedDict], pd.DataFrame, pd.DataFrame]:
        """Query the database for encoded montage masks and load them into memory.

        Args:
            wells: Wells to process (e.g. ``['A1', 'B2']``).
            channel_marker: Substring that identifies aligned encoded masks
                in the ``alignedmontagemaskpath`` column.

        Returns:
            A 3-tuple of:
              - dict mapping each well to an ``OrderedDict`` of
                ``timepoint -> [[None, Cell], ...]``.
              - DataFrame of grouped ``(well, timepoint, mask_path)`` rows.
              - Full tiledata DataFrame from the database.
        """
        from db_util import Ops
        import argparse

        db_start_time = time()
        opt_inner = argparse.Namespace(
            experiment=self.experiment,
            chosen_wells=','.join(wells),
            wells_toggle='include',
            chosen_timepoints='',
            timepoints_toggle='include',
            chosen_channels='all',
            channels_toggle='include',
            tile=0
        )
        op = Ops(opt_inner)
        tiledata_df = op.get_tiledata_df()

        # Optimized database filtering - check for markers first
        well_mask = tiledata_df['well'].isin(wells)
        
        # Check which marker exists first to avoid redundant operations
        has_aligned = tiledata_df['alignedmontagemaskpath'].str.contains(channel_marker, na=False).any()
        has_encoded = tiledata_df['alignedmontagemaskpath'].str.contains("_MONTAGE_ENCODED", na=False).any()
        
        if has_aligned:
            path_mask = tiledata_df['alignedmontagemaskpath'].str.contains(channel_marker, na=False)
            self.selected = True
        elif has_encoded:
            path_mask = tiledata_df['alignedmontagemaskpath'].str.contains("_MONTAGE_ENCODED", na=False)
            self.selected = False
        else:
            raise ValueError("No montage path contains the required marker.")

        # Apply filters and group efficiently
        filtered_df = tiledata_df[well_mask & path_mask]
        
        # Group by well and timepoint, taking first occurrence
        df = filtered_df.groupby(['well', 'timepoint'], as_index=False).agg({'alignedmontagemaskpath': 'first'})
        
        db_time = time() - db_start_time
        print(f'Reading Masks from the db - {db_time:.1f} seconds')

        results = {}
        for well in wells:
            print(f'Processing well {well}...')
            logger.info(f"Tracking well {well}")
            df_w = df[df['well'] == well]
            if df_w.empty:
                logger.warning(f"No encoded masks for well {well}")
                continue

            print(f'  Found {len(df_w)} mask files for well {well}')
            time_dict = collections.OrderedDict()
            
            # Pre-extract timepoints and check file existence
            mask_data = []
            existing_count = 0
            for _, row in df_w.iterrows():
                mask_path = row['alignedmontagemaskpath']
                if os.path.exists(mask_path):  # Only process existing files
                    tp_label = os.path.basename(mask_path).split('_')[2]
                    tp = int(tp_label.lstrip('T')) if tp_label.startswith('T') else int(tp_label)
                    mask_data.append((tp, mask_path))
                    existing_count += 1
                else:
                    print(f'  WARNING: Missing file {mask_path}')
            
            print(f'  Processing {existing_count} existing mask files...')
            
            # Process masks in batches for better performance
            batch_size = 5  # Smaller batches for better progress visibility
            for i in range(0, len(mask_data), batch_size):
                batch = mask_data[i:i+batch_size]
                print(f'  Processing batch {i//batch_size + 1}/{(len(mask_data)-1)//batch_size + 1} ({len(batch)} files)...')
                
                for j, (tp, mask_path) in enumerate(batch):
                    print(f'    Reading mask {j+1}/{len(batch)}: {os.path.basename(mask_path)}')
                    mask = read_tiff_safe(mask_path)
                    if mask is None:
                        print(f'    ERROR: Could not read {mask_path}')
                        continue
                    
                    if not isinstance(mask, np.ndarray):
                        print(f'    ERROR: Invalid mask data type for {mask_path}')
                        continue
                        
                    labels = np.unique(mask)
                    labels = labels[labels > 0]
                    print(f'    Found {len(labels)} labels in mask')

                    entries = time_dict.setdefault(tp, [])
                    # Optimized: only find contours for labels that exist
                    if len(labels) > 0:
                        print(f'    Processing {len(labels)} labels for contours in parallel...')
                        contour_start_time = time()
                        
                        # Use the persistent, per-tracker worker pool +
                        # shared-memory buffer (created lazily on first call).
                        cells = self._process_contours_pooled(mask, labels)
                        
                        # Add cells to entries
                        for cell in cells:
                            entries.append([None, cell])
                        
                        contour_time = time() - contour_start_time
                        print(f'    Completed processing {len(labels)} labels in {contour_time:.1f}s, found {len(entries)} valid contours')

            if time_dict:
                time_dict = collections.OrderedDict(sorted(time_dict.items()))
                results[well] = time_dict
                print(f'  Completed well {well} with {len(time_dict)} timepoints')

        print(f'Data gathering completed for {len(results)} wells')
        return results, df, tiledata_df

    def get_cell_props(
        self,
        mask: np.ndarray,
        image_stack: Dict[str, np.ndarray],
        label: int,
    ) -> Optional[Tuple[int, int, int, Dict[str, float]]]:
        """Compute centroid, area, and mean intensity for a single cell label.

        Args:
            mask: Encoded segmentation mask.
            image_stack: Mapping of channel name to image array.
            label: Label value identifying the cell in ``mask``.

        Returns:
            ``(cx, cy, area, {channel: mean_intensity})`` or ``None`` when
            the label has zero area (degenerate mask).
        """
        bin_mask = (mask == label).astype(np.uint8)
        M = cv2.moments(bin_mask)
        if M["m00"] == 0:
            return None
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        area = cv2.countNonZero(bin_mask)

        intensities = {}
        for chan, img in image_stack.items():
            intensities[chan] = float(np.mean(img[bin_mask > 0]))
        # Sentinel: if no channel image was available, still emit ONE row per
        # cell so downstream tools see the track (mask was saved) even when
        # intensity could not be measured. Prevents silent CSV drop-outs.
        if not intensities:
            intensities = {'<missing>': float('nan')}

        return cx, cy, area, intensities

    def calculate_tracking_statistics(
        self,
        well: str,
        time_dict: collections.OrderedDict,
        tracked_dict: collections.OrderedDict,
    ) -> Dict[str, Any]:
        """Compute per-well tracking statistics (cell counts, motion, ID switching).

        Args:
            well: Well identifier (e.g. ``'A1'``).
            time_dict: Original timepoint -> cell list mapping.
            tracked_dict: Post-tracking timepoint -> ``[track_id, Cell]`` mapping.

        Returns:
            Dictionary of statistics including per-timepoint counts, motion
            distances, and ID-retention metrics.
        """
        tps = sorted(time_dict.keys())
        
        well_stats = {
            'well': well,
            'total_timepoints': len(tps),
            'timepoint_stats': [],
            'motion_stats': [],
            'id_switching_stats': [],
            'overall_well_stats': {}
        }
        
        # Calculate per-timepoint statistics
        for i, tp in enumerate(tps):
            tp_stats = {
                'timepoint': tp,
                'cell_count': len(tracked_dict[tp]),
                'unique_ids': len(set([rec[0] for rec in tracked_dict[tp]]))
            }
            well_stats['timepoint_stats'].append(tp_stats)
        
        # Calculate motion statistics between consecutive timepoints
        if len(tps) > 1:
            for i in range(1, len(tps)):
                curr_tp, prev_tp = tps[i], tps[i-1]
                motion_stats = self._calculate_motion_between_timepoints(
                    tracked_dict[prev_tp], tracked_dict[curr_tp], prev_tp, curr_tp
                )
                well_stats['motion_stats'].append(motion_stats)
        
        # Calculate ID switching statistics
        id_switching_stats = self._calculate_id_switching(tracked_dict, tps)
        well_stats['id_switching_stats'] = id_switching_stats
        
        # Calculate overall well statistics
        total_cells = sum(len(tracked_dict[tp]) for tp in tps)
        all_ids = set()
        for tp in tps:
            all_ids.update([rec[0] for rec in tracked_dict[tp]])
        
        well_stats['overall_well_stats'] = {
            'total_cells_tracked': total_cells,
            'unique_track_ids': len(all_ids),
            'avg_cells_per_timepoint': total_cells / len(tps) if tps else 0,
            'max_cells_in_timepoint': max(len(tracked_dict[tp]) for tp in tps) if tps else 0,
            'min_cells_in_timepoint': min(len(tracked_dict[tp]) for tp in tps) if tps else 0
        }
        
        return well_stats
    
    def _calculate_motion_between_timepoints(
        self,
        prev_recs: List[List[Any]],
        curr_recs: List[List[Any]],
        prev_tp: int,
        curr_tp: int,
    ) -> Dict[str, Any]:
        """Compute motion-distance statistics for cells tracked between two frames.

        Args:
            prev_recs: ``[track_id, Cell]`` records from the earlier timepoint.
            curr_recs: ``[track_id, Cell]`` records from the later timepoint.
            prev_tp: Earlier timepoint index.
            curr_tp: Later timepoint index.

        Returns:
            Dictionary with mean/std/max/min motion distance and cell count.
        """
        # Create mapping of IDs to positions
        prev_positions = {rec[0]: rec[1].get_center() for rec in prev_recs}
        curr_positions = {rec[0]: rec[1].get_center() for rec in curr_recs}
        
        # Find common IDs
        common_ids = set(prev_positions.keys()) & set(curr_positions.keys())
        
        if not common_ids:
            return {
                'from_timepoint': prev_tp,
                'to_timepoint': curr_tp,
                'cells_with_motion': 0,
                'mean_motion_distance': 0.0,
                'std_motion_distance': 0.0,
                'max_motion_distance': 0.0,
                'min_motion_distance': 0.0
            }
        
        # Calculate motion distances
        motion_distances = []
        for cell_id in common_ids:
            prev_pos = prev_positions[cell_id]
            curr_pos = curr_positions[cell_id]
            distance = np.hypot(curr_pos[0] - prev_pos[0], curr_pos[1] - prev_pos[1])
            motion_distances.append(distance)
        
        motion_distances = np.array(motion_distances)
        
        return {
            'from_timepoint': prev_tp,
            'to_timepoint': curr_tp,
            'cells_with_motion': len(common_ids),
            'mean_motion_distance': float(np.mean(motion_distances)),
            'std_motion_distance': float(np.std(motion_distances)),
            'max_motion_distance': float(np.max(motion_distances)),
            'min_motion_distance': float(np.min(motion_distances))
        }
    
    def _calculate_id_switching(
        self,
        tracked_dict: collections.OrderedDict,
        tps: List[int],
    ) -> List[Dict[str, Any]]:
        """Measure track-ID retention, gain, and loss between consecutive frames.

        Args:
            tracked_dict: Timepoint -> ``[track_id, Cell]`` mapping.
            tps: Sorted list of timepoint indices.

        Returns:
            List of per-transition dictionaries with retention and growth rates.
        """
        if len(tps) < 2:
            return []
        
        switching_stats = []
        
        for i in range(1, len(tps)):
            prev_tp, curr_tp = tps[i-1], tps[i]
            
            # Get IDs from both timepoints
            prev_ids = set([rec[0] for rec in tracked_dict[prev_tp]])
            curr_ids = set([rec[0] for rec in tracked_dict[curr_tp]])
            
            # Calculate switching metrics
            common_ids = prev_ids & curr_ids
            new_ids = curr_ids - prev_ids
            lost_ids = prev_ids - curr_ids
            
            switching_stats.append({
                'from_timepoint': prev_tp,
                'to_timepoint': curr_tp,
                'prev_timepoint_ids': len(prev_ids),
                'curr_timepoint_ids': len(curr_ids),
                'common_ids': len(common_ids),
                'new_ids': len(new_ids),
                'lost_ids': len(lost_ids),
                'id_retention_rate': len(common_ids) / len(prev_ids) if prev_ids else 0,
                'id_growth_rate': len(new_ids) / len(prev_ids) if prev_ids else 0
            })
        
        return switching_stats
    
    def save_tracking_statistics(self) -> None:
        """Persist all accumulated tracking statistics to a single CSV file.

        Writes overall, per-timepoint, motion, and ID-switching rows to
        ``<analysisdir>/<experiment>_tracking-info.csv``.
        """
        if not self.tracking_stats['well_stats']:
            logger.warning("No tracking statistics to save")
            return
        
        # Combine all statistics into a single comprehensive dataset
        all_data = []
        
        for well, well_stats in self.tracking_stats['well_stats'].items():
            # Add overall well statistics
            overall_row = {
                'experiment': self.experiment,
                'well': well,
                'tracking_method': self.tracking_stats['overall_stats']['tracking_method'],
                'motion_enabled': self.tracking_stats['overall_stats']['motion_enabled'],
                'statistic_type': 'overall',
                'timepoint': None,
                'from_timepoint': None,
                'to_timepoint': None,
                **well_stats['overall_well_stats']
            }
            all_data.append(overall_row)
            
            # Add timepoint statistics
            for tp_stat in well_stats['timepoint_stats']:
                timepoint_row = {
                    'experiment': self.experiment,
                    'well': well,
                    'tracking_method': self.tracking_stats['overall_stats']['tracking_method'],
                    'motion_enabled': self.tracking_stats['overall_stats']['motion_enabled'],
                    'statistic_type': 'timepoint',
                    'timepoint': tp_stat['timepoint'],
                    'from_timepoint': None,
                    'to_timepoint': None,
                    'cell_count': tp_stat['cell_count'],
                    'unique_ids': tp_stat['unique_ids'],
                    # Add empty fields for other statistic types
                    'total_cells_tracked': None,
                    'unique_track_ids': None,
                    'avg_cells_per_timepoint': None,
                    'max_cells_in_timepoint': None,
                    'min_cells_in_timepoint': None,
                    'cells_with_motion': None,
                    'mean_motion_distance': None,
                    'std_motion_distance': None,
                    'max_motion_distance': None,
                    'min_motion_distance': None,
                    'prev_timepoint_ids': None,
                    'curr_timepoint_ids': None,
                    'common_ids': None,
                    'new_ids': None,
                    'lost_ids': None,
                    'id_retention_rate': None,
                    'id_growth_rate': None
                }
                all_data.append(timepoint_row)
            
            # Add motion statistics
            for motion_stat in well_stats['motion_stats']:
                motion_row = {
                    'experiment': self.experiment,
                    'well': well,
                    'tracking_method': self.tracking_stats['overall_stats']['tracking_method'],
                    'motion_enabled': self.tracking_stats['overall_stats']['motion_enabled'],
                    'statistic_type': 'motion',
                    'timepoint': None,
                    'from_timepoint': motion_stat['from_timepoint'],
                    'to_timepoint': motion_stat['to_timepoint'],
                    'cells_with_motion': motion_stat['cells_with_motion'],
                    'mean_motion_distance': motion_stat['mean_motion_distance'],
                    'std_motion_distance': motion_stat['std_motion_distance'],
                    'max_motion_distance': motion_stat['max_motion_distance'],
                    'min_motion_distance': motion_stat['min_motion_distance'],
                    # Add empty fields for other statistic types
                    'cell_count': None,
                    'unique_ids': None,
                    'total_cells_tracked': None,
                    'unique_track_ids': None,
                    'avg_cells_per_timepoint': None,
                    'max_cells_in_timepoint': None,
                    'min_cells_in_timepoint': None,
                    'prev_timepoint_ids': None,
                    'curr_timepoint_ids': None,
                    'common_ids': None,
                    'new_ids': None,
                    'lost_ids': None,
                    'id_retention_rate': None,
                    'id_growth_rate': None
                }
                all_data.append(motion_row)
            
            # Add ID switching statistics
            for switching_stat in well_stats['id_switching_stats']:
                switching_row = {
                    'experiment': self.experiment,
                    'well': well,
                    'tracking_method': self.tracking_stats['overall_stats']['tracking_method'],
                    'motion_enabled': self.tracking_stats['overall_stats']['motion_enabled'],
                    'statistic_type': 'id_switching',
                    'timepoint': None,
                    'from_timepoint': switching_stat['from_timepoint'],
                    'to_timepoint': switching_stat['to_timepoint'],
                    'prev_timepoint_ids': switching_stat['prev_timepoint_ids'],
                    'curr_timepoint_ids': switching_stat['curr_timepoint_ids'],
                    'common_ids': switching_stat['common_ids'],
                    'new_ids': switching_stat['new_ids'],
                    'lost_ids': switching_stat['lost_ids'],
                    'id_retention_rate': switching_stat['id_retention_rate'],
                    'id_growth_rate': switching_stat['id_growth_rate'],
                    # Add empty fields for other statistic types
                    'cell_count': None,
                    'unique_ids': None,
                    'total_cells_tracked': None,
                    'unique_track_ids': None,
                    'avg_cells_per_timepoint': None,
                    'max_cells_in_timepoint': None,
                    'min_cells_in_timepoint': None,
                    'cells_with_motion': None,
                    'mean_motion_distance': None,
                    'std_motion_distance': None,
                    'max_motion_distance': None,
                    'min_motion_distance': None
                }
                all_data.append(switching_row)
        
        # Save all data to a single CSV file. Parallel per-well tracking jobs
        # all append to the same experiment-level file, so the write is done
        # under an exclusive file lock (see _append_df_locked).
        if all_data:
            combined_file = os.path.join(self.analysisdir, f"{self.experiment}_tracking-info.csv")
            combined_df = pd.DataFrame(all_data)
            self._append_df_locked(combined_df, combined_file)
            logger.info(f"Saved all tracking statistics to {combined_file}")
            print(f"📊 Saved comprehensive tracking statistics to: {combined_file}")
        else:
            logger.warning("No tracking statistics data to save")

    @staticmethod
    def _append_df_locked(df: pd.DataFrame, path: str) -> None:
        """Append a DataFrame to a CSV under an exclusive file lock.

        TRACKING_MONTAGE runs one Slurm job per well concurrently, and every
        well writes to the same experiment-level CSV. Without locking, the
        check-then-append pattern races: two jobs can both see the file as
        absent and both write a header, or interleave partial rows and lose
        data. This takes an exclusive ``flock`` around the whole
        header-decision + write so each well's rows are appended atomically.
        The header is written only when the file is empty (first writer wins).

        Args:
            df: Rows to append.
            path: Destination CSV path (shared across parallel wells).
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # 'a+' creates the file if absent and lets us check its size under the
        # lock to decide whether a header is needed.
        with open(path, 'a+', newline='') as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                handle.seek(0, os.SEEK_END)
                need_header = handle.tell() == 0
                df.to_csv(handle, header=need_header, index=False)
                handle.flush()
                os.fsync(handle.fileno())
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def run(self, wells: List[str]) -> None:
        """Execute the full tracking pipeline for the given wells.

        Gathers masks from the database, assigns track IDs, writes tracked
        mask TIFFs, computes per-cell intensity measurements, and saves a
        summary CSV and tracking-statistics CSV.

        The worker pool and shared-memory buffer created for parallel
        contour extraction are torn down in a ``finally`` block so an
        exception, KeyboardInterrupt, or Slurm SIGTERM does not leak
        workers or an unlinked ``/dev/shm`` block.

        Args:
            wells: List of well identifiers to process.
        """
        self.start_time = time()
        self.total_wells = len(wells)

        logger.info(f"Starting tracking for wells: {wells}")

        # Make SIGTERM (Slurm's polite kill) raise SystemExit so the
        # ``finally`` below runs and releases the pool + shared memory.
        prev_sigterm = signal.getsignal(signal.SIGTERM)

        def _sigterm_handler(signum, frame):
            logger.warning("Received SIGTERM; cleaning up tracker resources")
            raise SystemExit(f"terminated by signal {signum}")

        try:
            signal.signal(signal.SIGTERM, _sigterm_handler)
        except (ValueError, OSError):
            # Not on the main thread — best-effort; skip installation.
            prev_sigterm = None

        try:
            self._run_body(wells)
        finally:
            self._cleanup_pool()
            if prev_sigterm is not None:
                try:
                    signal.signal(signal.SIGTERM, prev_sigterm)
                except (ValueError, OSError):
                    pass

    def _run_body(self, wells: List[str]) -> None:
        """Internal: the actual tracking work, wrapped by ``run()`` for cleanup."""
        exp_id = self.Db.get_table_uuid('experimentdata', {'experiment': self.experiment})
        all_wells, df, tiledata_df = self.gather_encoded_from_db(wells)

        out_records = []
        n_missing_channel = 0  # counts (well, tp, channel) triples with no image

        for well, time_dict in all_wells.items():
            welldata_id = self.Db.get_table_uuid('welldata', {'experimentdata_id': exp_id, 'well': well})
            tps = sorted(time_dict.keys())
            
            if self.motion:
                tracked = populate_cell_ind_motion(time_dict, tps, max_dist=self.max_dist)
            else:
                tracked = (
                    populate_cell_ind_overlap(time_dict, tps)
                    if self.track_type == 'overlap'
                    else populate_cell_ind_closest(time_dict, tps, max_dist=self.max_dist)
                )
            sorted_td = sort_cell_info_by_index(tracked, tps)
            
            # Calculate tracking statistics for this well
            well_stats = self.calculate_tracking_statistics(well, time_dict, sorted_td)
            self.tracking_stats['well_stats'][well] = well_stats
            
            # Print summary statistics
            print(f'\n📊 Tracking Statistics for Well {well}:')
            print(f'   Total timepoints: {well_stats["total_timepoints"]}')
            print(f'   Total cells tracked: {well_stats["overall_well_stats"]["total_cells_tracked"]}')
            print(f'   Unique track IDs: {well_stats["overall_well_stats"]["unique_track_ids"]}')
            print(f'   Average cells per timepoint: {well_stats["overall_well_stats"]["avg_cells_per_timepoint"]:.1f}')
            if well_stats['motion_stats']:
                avg_motion = np.mean([stat['mean_motion_distance'] for stat in well_stats['motion_stats']])
                print(f'   Average motion distance: {avg_motion:.2f} pixels')
            if well_stats['id_switching_stats']:
                avg_retention = np.mean([stat['id_retention_rate'] for stat in well_stats['id_switching_stats']])
                print(f'   Average ID retention rate: {avg_retention:.2%}')

            for tp, recs in sorted_td.items():
                tp_start_time = time()
                print(f'\n started tracking well {well} + timepoint {tp}')
                
                df_wtp = df[(df['well'] == well) & (df['timepoint'] == tp)]
                if df_wtp.empty:
                    continue
                mask_path = df_wtp['alignedmontagemaskpath'].iloc[0]
              
                self.analysisdir = os.path.dirname(mask_path.split('CellMasksMontage')[0])
                mask = read_tiff_safe(mask_path)
                if mask is None:
                    logger.error(f"Failed to read mask file: {mask_path}")
                    continue
                base_path = mask_path.replace('_MONTAGE_ALIGNED_ENCODED.tif', '')
                
                tracked_mask = np.zeros_like(mask, dtype=np.uint16)
                for new_id, cell in recs:
                    if cell.randomcellid_montage is None:
                        continue
                    tracked_mask[mask == cell.randomcellid_montage] = new_id

                # Construct the output path for the tracked TIFF
            
                tracked_folder = os.path.join(self.analysisdir, 'TrackedCellMasksMontage', well)
               
                os.makedirs(tracked_folder, exist_ok=True)

                if self.selected:
                    tracked_filename = os.path.basename(mask_path).replace(
                        '_MONTAGE_ALIGNED_ENCODED.tif', '_TRACKED.tif'
                    )
                else:
                    tracked_filename = os.path.basename(mask_path).replace(
                        '_MONTAGE_ENCODED.tif', '_TRACKED.tif'
                    )

                tracked_path = os.path.join(tracked_folder, tracked_filename)

                # Save the tracked mask
                if not cv2.imwrite(tracked_path, tracked_mask):
                    raise IOError(f"Failed to write tracked mask: {tracked_path}")
                logger.info(f"Saved tracked mask: {tracked_path}")
                channel_imgs = {}
                object_count = len(recs)

                # Target channel only for calculate the intensities, not tracking
                
                for ch in self.target_channel:
                    # Use optimized path helper method
                    img_path = self._get_channel_path(mask_path, ch)

                    if os.path.exists(img_path):
                        img = read_tiff_safe(img_path)
                        if img is not None:
                            channel_imgs[ch] = img
                        else:
                            logger.error(f"[MISSING-CHANNEL] Could not read aligned image for well={well} tp={tp} ch={ch} path={img_path}")
                            n_missing_channel += 1
                    else:
                        logger.error(f"[MISSING-CHANNEL] Missing aligned image for well={well} tp={tp} ch={ch} path={img_path}")
                        n_missing_channel += 1

                ## KS edit for Galaxy csv format
                for new_id, cell in recs:
                    props = self.get_cell_props(mask, channel_imgs, cell.randomcellid_montage)
                    if props is None:
                        continue
                    cx, cy, area, intensities = props

                    for ch, mean_val in intensities.items():
                        row = {
                            'experiment': self.experiment,
                            'ObjectCount': object_count,
                            'well': well,
                            'ObjectLabelsFound': new_id,
                            'tracked_id': new_id,
                            'MeasurementTag': ch,
                            'BlobArea': area,
                            'BlobCentroidX': cx,
                            'centroid_x': cx,
                            'BlobCentroidY': cy,
                            'centroid_y': cy,
                            'PixelIntensityMean': mean_val,
                            'Sci_WellID': well,
                            'Timepoint': tp,
                            'timepoint': tp,
                            'area': area
                        }
                        out_records.append(row)
                
                # Log timepoint completion
                tp_time = time() - tp_start_time
                print(f'completed well {well} + timepoint {tp} in {tp_time:.1f} seconds')

        if out_records:  # Only process if we have records
            out_df = pd.DataFrame(out_records)
            outfile = os.path.join(self.analysisdir, f"{self.experiment}_tracked_montage_summary.csv")
            # Parallel per-well jobs share this file; append under a lock so
            # concurrent wells cannot race on the header/append (data loss).
            self._append_df_locked(out_df, outfile)
            logger.info(f"Wrote {len(out_records)} tracked records to {outfile}")
        else:
            logger.warning("No tracking records generated")
        
        # Save tracking statistics
        if self.tracking_stats['well_stats']:
            print(f'\n📈 Saving tracking statistics...')
            self.save_tracking_statistics()
            
            # Print overall summary
            total_wells = len(self.tracking_stats['well_stats'])
            total_cells = sum(well_stats['overall_well_stats']['total_cells_tracked'] 
                            for well_stats in self.tracking_stats['well_stats'].values())
            total_timepoints = sum(well_stats['total_timepoints'] 
                                 for well_stats in self.tracking_stats['well_stats'].values())
            
            print(f'\n🎯 OVERALL TRACKING SUMMARY:')
            print(f'   Total wells processed: {total_wells}')
            print(f'   Total cells tracked: {total_cells}')
            print(f'   Total timepoints: {total_timepoints}')
            print(f'   Tracking method: {self.tracking_stats["overall_stats"]["tracking_method"]}')
            print(f'   Motion tracking enabled: {self.tracking_stats["overall_stats"]["motion_enabled"]}')
        
        # Final completion logging
        if n_missing_channel:
            logger.error(f"missing-channel image events: {n_missing_channel} (see [MISSING-CHANNEL] lines above)")
        total_time = time() - self.start_time
        logger.warning(f'TRACKING COMPLETED in {total_time:.2f}s ({total_time/60:.2f} min)')
        print(f'✅ TRACKING COMPLETED in {total_time:.2f}s ({total_time/60:.2f} min)')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Track montage masks stored in DB")
    parser.add_argument('--experiment', required=True, help='Experiment name')
    parser.add_argument('--track_type', choices=['overlap','proximity'], default='overlap', help='Tracking method')
    parser.add_argument('--max_dist', type=int, default=100, help='Max distance for proximity')
    parser.add_argument('--wells', required=True, help='Comma-separated list of wells, e.g. A1,B1')
    parser.add_argument("--target_channel",  type=str, default='Cy5',
                        dest="target_channel",
                        help="Get intensity of this channel.")
    parser.add_argument('--motion', action='store_true', 
                        help='Enable motion tracking with linear prediction and Hungarian algorithm')
    args = parser.parse_args()

    wells = [w.strip() for w in args.wells.split(',')]
    tracker = MontageDBTracker(args.experiment, args.track_type, args.max_dist, args.target_channel, args.motion)
    tracker.run(wells)


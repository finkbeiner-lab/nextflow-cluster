#!/usr/bin/env python
"""Threshold-based cell segmentation on montage images.

Applies a configurable thresholding method (e.g. sd_from_mean, triangle,
otsu) to montage TIFF images, labels connected regions, filters by area,
and persists the results (masks and region properties) to the database.

Processing is parallelised across tiles within each well/timepoint group
using a :class:`concurrent.futures.ThreadPoolExecutor`.

Usage (standalone)::

    python segmentation_montage.py --experiment <name> --chosen_wells <well> ...

Typically invoked by the Nextflow pipeline via ``modules.nf``.
"""

import argparse
import datetime
import gc
import logging
import multiprocessing
import os
from concurrent.futures import ThreadPoolExecutor
from time import time
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import tifffile
from scipy.spatial import cKDTree
from skimage import exposure, filters, measure

from db_util import Ops
from normalization import Normalize
from segmentation_helper_montage import (
    batch_update_celldata_and_intensitycelldata,
    batch_update_celldata_and_intensitycelldata_no_delete,
    save_mask,
    update_celldata_and_intensitycelldata,
)
from sql import Database

# ---------------------------------------------------------------------------
# Module-level logging setup
# ---------------------------------------------------------------------------
logger = logging.getLogger("Segmentation")

now = datetime.datetime.now()
TIMESTAMP = '%d%02d%02d%02d%02d' % (now.year, now.month, now.day, now.hour, now.minute)
print(f'Starting segmentation processing at {now.strftime("%Y-%m-%d %H:%M:%S")}')

fink_log_dir = './finkbeiner_logs'
if not os.path.exists(fink_log_dir):
    os.makedirs(fink_log_dir)
logname = os.path.join(fink_log_dir, f'Segmentation-log_{TIMESTAMP}.log')
fh = logging.FileHandler(logname)
logger.addHandler(fh)
# Also mirror this logger to the experiment-scoped debug log
# (<params.output_path>/pipeline_debug.log) shared across all parallel wells.
from experiment_logger import attach_experiment_log  # noqa: E402
attach_experiment_log(logger, os.environ.get('NEXTFLOW_OUTPUT_PATH', ''), 'SEG_MONTAGE')
logger.warning('Running OPTIMIZED Segmentation from Database.')


class Segmentation:
    """Threshold-based segmentation engine for montage microscopy images.

    Reads tile images from the database (via :class:`Normalization`),
    applies one of several thresholding algorithms, labels connected
    components, filters by area, saves masks to disk, and writes cell /
    intensity-cell records back to the database.

    Args:
        opt: Namespace (typically from ``argparse``) containing at least:
            - ``segmentation_method`` (str): Threshold algorithm name.
            - ``chosen_channels`` (str): Channel(s) to segment.
            - ``chosen_wells`` (str): Well identifier to process.
            - ``sd_scale_factor`` (float): Multiplier for sd_from_mean.
            - ``manual_thresh`` (int): Fixed threshold for "manual" mode.
            - ``lower_area_thresh`` / ``upper_area_thresh`` (int): Area
              bounds for filtering segmented regions.
    """

    def __init__(self, opt: argparse.Namespace) -> None:
        self.opt = opt
        # Cap thread count to avoid excessive memory consumption
        self.thread_lim: int = min(multiprocessing.cpu_count(), 8)
        self.segmentation_method: str = opt.segmentation_method
        assert len(self.opt.chosen_channels) > 0, 'Must select a channel for segmentation'
        logger.warning(f'Segmentation Method: {self.segmentation_method}')
        logger.warning(f'Using {self.thread_lim} threads for parallel processing')
        self.mask_folder_name: str = 'CellMasksMontage'

        # Map method names to callable threshold functions
        self.threshold_func: Dict[str, Any] = dict(
            sd_from_mean=self.sd_from_mean,
            minimum=filters.threshold_minimum,
            yen=filters.threshold_yen,
            local=filters.threshold_local,
            li=filters.threshold_li,
            isodata=filters.threshold_isodata,
            mean=filters.threshold_mean,
            otsu=filters.threshold_otsu,
            sauvola=filters.threshold_sauvola,
            triangle=filters.threshold_triangle,
            manual=None,
            tryall=filters.try_all_threshold,
        )
        # `.get` so segmentation_method='cellpose' (not a threshold) does not KeyError
        self.thresh_func = self.threshold_func.get(self.segmentation_method)
        self._cp_model = None  # lazily-loaded Cellpose-SAM model (cellpose method)

        # Region properties extracted for every labelled cell
        self.region_props: Tuple[str, ...] = (
            'label', 'area', 'centroid_weighted',
            'orientation', 'solidity', 'extent',
            'perimeter', 'eccentricity',
            'intensity_max', 'intensity_mean',
            'intensity_min', 'axis_major_length',
            'axis_minor_length',
        )

        self.proximity_filter_radius: int = getattr(opt, 'proximity_filter_radius', 0)
        if self.proximity_filter_radius > 0:
            logger.warning(f'Proximity filter enabled: radius={self.proximity_filter_radius}px')

        self.Norm = Normalize(self.opt)
        _, self.analysisdir = self.Norm.get_raw_and_analysis_dir()
        logger.warning(f'Save directory: {self.analysisdir}')

        # Performance counters
        self.processed_tiles: int = 0
        self.total_tiles: int = 0
        self.start_time: Optional[float] = None

        self._setup_optimizations()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _setup_optimizations(self) -> None:
        """Initialise caches and pre-allocated structures."""
        # Tracks directories already created so we skip redundant os.path.exists
        self.mask_dirs: Dict[str, bool] = {}
        self.area_thresh_mask: Optional[np.ndarray] = None
        self._cache: Dict[str, Any] = {}

    def _cleanup(self) -> None:
        """Release caches and trigger garbage collection."""
        self._cache.clear()
        gc.collect()

    def _precreate_directories(self, tiledata_df: pd.DataFrame) -> None:
        """Ensure mask output directories exist for every well in *tiledata_df*.

        Args:
            tiledata_df: DataFrame with a ``well`` column.
        """
        wells = tiledata_df['well'].unique()
        for well in wells:
            mask_dir = os.path.join(self.analysisdir, self.mask_folder_name, well)
            if mask_dir not in self.mask_dirs:
                if not os.path.exists(mask_dir):
                    os.makedirs(mask_dir)
                    logger.info(f"Created mask directory: {mask_dir}")
                self.mask_dirs[mask_dir] = True

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Execute the full segmentation workflow and log elapsed time."""
        self.start_time = time()
        if self.segmentation_method == 'cellpose':
            self.run_cellpose()
        else:
            self.run_threshold()

        total_time = time() - self.start_time
        logger.warning(f'Completed threshold in {total_time:.2f}s')
        if self.total_tiles > 0:
            avg_time_per_tile = total_time / self.total_tiles
            logger.warning(f'Average time per tile: {avg_time_per_tile:.2f}s')

        print(f'SEGMENTATION COMPLETED in {total_time:.2f}s ({total_time/60:.2f} min)')
        self._cleanup()

    # ------------------------------------------------------------------
    # Cellpose-SAM segmentation (segmentation_method == 'cellpose')
    # ------------------------------------------------------------------

    def _get_cellpose_model(self) -> Any:
        """Lazily load a single Cellpose-SAM (v4) model (GPU). Imported here so
        the threshold path never imports cellpose."""
        if self._cp_model is None:
            from cellpose import models
            logger.warning('Loading Cellpose-SAM (v4) model (gpu=True)')
            self._cp_model = models.CellposeModel(gpu=True)
        return self._cp_model

    @staticmethod
    def _clean_masks(img: np.ndarray, masks: np.ndarray, k: float) -> np.ndarray:
        """Drop Cellpose cells that are background-level (dim false positives).

        Keeps only cells whose median intensity exceeds background median +
        k * robust-SD (MAD-based), measured on the RAW image.
        """
        bg = img[masks == 0]
        if bg.size == 0 or masks.max() == 0:
            return masks
        bg_med = np.median(bg)
        bg_mad = 1.4826 * np.median(np.abs(bg - bg_med))
        thr = bg_med + k * bg_mad
        out = np.zeros_like(masks)
        for lab in range(1, int(masks.max()) + 1):
            m = masks == lab
            if m.any() and np.median(img[m]) >= thr:
                out[m] = lab
        return out

    @staticmethod
    def _filter_debris(masks: np.ndarray, area_tiny: int, area_max: int, ecc_max: float) -> np.ndarray:
        """Remove small, round dead-cell debris while keeping fibroblasts.

        Drops a label if it is very small (< area_tiny) OR both small
        (< area_max) and round (eccentricity < ecc_max). Keeps elongated
        fibroblasts and large rounded dying cells.
        """
        if masks.max() == 0 or (area_tiny <= 0 and area_max <= 0):
            return masks
        out = masks.copy()
        for rp in measure.regionprops(masks):
            tiny = area_tiny > 0 and rp.area < area_tiny
            small_round = area_max > 0 and rp.area < area_max and rp.eccentricity < ecc_max
            if tiny or small_round:
                out[masks == rp.label] = 0
        return out

    def cellpose_single_image(self, model: Any, img: np.ndarray) -> Tuple[np.ndarray, pd.DataFrame]:
        """Segment one montage image with Cellpose-SAM + CLAHE + cleanup.

        Percentile-clip then CLAHE adaptive contrast (lifts dim spread-cell
        cytoplasm), run Cellpose-SAM, then MAD intensity cleanup and the
        shape debris filter. Returns the labelled mask and its regionprops.
        """
        raw = img.astype(np.float32)
        lo, hi = np.percentile(raw, 1), np.percentile(raw, 99.5)
        if hi > lo:
            stretched = np.clip((raw - lo) / (hi - lo + 1e-9), 0, 1)
            if getattr(self.opt, 'clahe_clip', 0) and self.opt.clahe_clip > 0:
                stretched = exposure.equalize_adapthist(stretched, clip_limit=self.opt.clahe_clip)
            norm = (stretched * 255.0).astype(np.float32)
        else:
            norm = np.zeros_like(raw)
        masks = model.eval(norm, batch_size=self.opt.batch_size, diameter=self.opt.cell_diameter,
                           flow_threshold=self.opt.flow_threshold,
                           cellprob_threshold=self.opt.cell_probability)[0]
        masks = self._clean_masks(raw, masks, self.opt.cleanup_k)
        masks = self._filter_debris(masks, self.opt.debris_area_tiny,
                                    self.opt.debris_area_max, self.opt.debris_ecc_max)
        props = measure.regionprops_table(masks, intensity_image=img, properties=self.region_props)
        return masks, pd.DataFrame(props)

    def run_cellpose(self) -> None:
        """Cellpose-SAM segmentation over the well's montages (serial, GPU).

        Mirrors :meth:`run_threshold` but processes each unique montage once
        (not once per tile row) and runs serially (Cellpose eval is not
        thread-safe). The mask/celldata write path is reused verbatim.
        """
        Db = Database()
        tiledata_df: pd.DataFrame = self.Norm.get_tiledata_df()
        tiledata_df = tiledata_df[tiledata_df['well'] == self.opt.chosen_wells]
        if tiledata_df.empty:
            logger.warning(f'No tile data for well {self.opt.chosen_wells}')
            return
        self.total_tiles = len(tiledata_df)
        print(f'Cellpose: {self.total_tiles} tile rows for well {self.opt.chosen_wells}')
        self._precreate_directories(tiledata_df)
        all_tile_ids: Set = set(tiledata_df['id'].tolist())
        model = self._get_cellpose_model()
        for (well, timepoint), df in tiledata_df.groupby(['well', 'timepoint']):
            self.cellpose_single_group(Db, df, well, timepoint, model)
        if all_tile_ids:
            self.bulk_delete_celldata(Db, all_tile_ids)

    def cellpose_single_group(self, Db: Database, df: pd.DataFrame, well: str,
                              timepoint: Any, model: Any) -> None:
        """Segment one montage once and write masks/celldata for all its tile rows."""
        strt = time()
        df = df.sort_values(by='tile').copy()
        row0 = df.iloc[0]
        img_path = row0.alignedmontagepath if pd.notna(row0.alignedmontagepath) else row0.newimagemontage
        if not img_path or not os.path.exists(img_path):
            print(f'Warning: montage image missing for {well} {timepoint}: {img_path}')
            return
        img = tifffile.imread(img_path)
        masks, props_df = self.cellpose_single_image(model, img)
        mask_dir = os.path.join(self.analysisdir, self.mask_folder_name, well)
        os.makedirs(mask_dir, exist_ok=True)
        maskpath = save_mask(masks, img_path, mask_dir)

        batch_updates: List[Dict[str, Any]] = []
        batch_data: List[Tuple[Any, pd.DataFrame]] = []
        for _, row in df.iterrows():
            batch_updates.append({
                'kwargs': dict(experimentdata_id=row.experimentdata_id, welldata_id=row.welldata_id,
                               channeldata_id=row.channeldata_id, tile=row.tile, timepoint=row.timepoint),
                'data': dict(alignedmontagemaskpath=maskpath),
            })
            if not props_df.empty:
                batch_data.append((row, props_df))
        if batch_updates:
            self.batch_update_tiledata(Db, batch_updates)
        if batch_data:
            self.batch_update_celldata_optimized(Db, batch_data, df)
        try:
            del self.Norm.backgrounds[well][timepoint]
        except (KeyError, AttributeError):
            pass
        gc.collect()
        n_cells = 0 if props_df.empty else len(props_df)
        self.processed_tiles += len(df)
        print(f'Cellpose {well} {timepoint}: {n_cells} cells, {time()-strt:.1f}s '
              f'({self.processed_tiles}/{self.total_tiles} rows)')

    # ------------------------------------------------------------------
    # Thresholding
    # ------------------------------------------------------------------

    def sd_from_mean(self, img: np.ndarray) -> int:
        """Compute a threshold as mean + sd_scale_factor * std.

        Args:
            img: Grayscale image array.

        Returns:
            Integer threshold value.
        """
        img_mean = np.mean(img)
        img_std = np.std(img)
        return int(img_mean + img_std * self.opt.sd_scale_factor)

    def run_threshold(self) -> None:
        """Iterate over tiles for the selected well and apply thresholding.

        Tiles are grouped by (well, timepoint) and processed in parallel
        via :meth:`thresh_single_parallel`.  A bulk delete of prior
        celldata is performed at the end.
        """
        Db = Database()
        tiledata_df: pd.DataFrame = self.Norm.get_tiledata_df()

        # Filter to the single well being processed in this invocation
        well_mask = tiledata_df['well'] == self.opt.chosen_wells
        tiledata_df = tiledata_df[well_mask]

        if tiledata_df.empty:
            logger.warning(
                f"No tile data found for well {self.opt.chosen_wells} "
                f"and timepoint {self.opt.chosen_timepoints}"
            )
            return

        self.total_tiles = len(tiledata_df)
        logger.warning(f"Processing {self.total_tiles} tiles")
        print(f"Processing {self.total_tiles} tiles for well {self.opt.chosen_wells}")

        self._precreate_directories(tiledata_df)

        # Collect tile IDs up-front so we can do a single bulk delete later
        all_tile_ids: Set = set(tiledata_df['id'].tolist())

        grouped = tiledata_df.groupby(['well', 'timepoint'])

        # Create the thread pool once and reuse across every (well, timepoint)
        # group in this run_threshold invocation instead of rebuilding it per
        # iteration.
        with ThreadPoolExecutor(max_workers=self.thread_lim) as executor:
            for (well, timepoint), df in grouped:
                self.thresh_single_parallel(Db, df, well, timepoint, executor=executor)

        # One bulk delete at the end is faster than per-tile deletes
        if all_tile_ids:
            self.bulk_delete_celldata(Db, all_tile_ids)

    def thresh_single_parallel(
        self,
        Db: Database,
        df: pd.DataFrame,
        well: str,
        timepoint: str,
        executor: Optional[ThreadPoolExecutor] = None,
    ) -> None:
        """Segment all tiles for one well/timepoint using a thread pool.

        Args:
            Db: Active database connection.
            df: Subset of tiledata rows for this well/timepoint.
            well: Well identifier (e.g. ``"A1"``).
            timepoint: Timepoint label (e.g. ``"T0"``).
            executor: Caller-owned :class:`ThreadPoolExecutor` for tile
                submission. Required in practice; ``None`` default is kept
                only for signature compatibility. The caller retains
                lifecycle ownership (no shutdown here).
        """
        strt = time()
        df = df.sort_values(by='tile').copy()

        print(f'Processing well {well} at timepoint {timepoint} with {len(df)} tiles')

        batch_updates: List[Dict[str, Any]] = []
        batch_data: List[Tuple[Any, pd.DataFrame]] = []

        # Chunk size balances work granularity vs. thread overhead
        chunk_size = max(1, len(df) // (self.thread_lim * 2))

        futures = []
        for i in range(0, len(df), chunk_size):
            chunk = df.iloc[i:i+chunk_size]
            for _, row in chunk.iterrows():
                future = executor.submit(self.process_single_tile, row, well, timepoint)
                futures.append((future, row))

        for future, row in futures:
            try:
                result = future.result()
                if result:
                    maskpath, props_df, masks = result

                    batch_updates.append({
                        'kwargs': dict(
                            experimentdata_id=row.experimentdata_id,
                            welldata_id=row.welldata_id,
                            channeldata_id=row.channeldata_id,
                            tile=row.tile,
                            timepoint=row.timepoint,
                        ),
                        'data': dict(alignedmontagemaskpath=maskpath),
                    })

                    if not props_df.empty:
                        batch_data.append((row, props_df))

                    self.processed_tiles += 1
                    progress = (self.processed_tiles / self.total_tiles) * 100
                    print(f'Progress: {progress:.1f}% ({self.processed_tiles}/{self.total_tiles})')

            except Exception as exc:
                print(f'Tile {row.tile} generated an exception: {exc}')
                continue

        # Flush accumulated database writes
        if batch_updates:
            self.batch_update_tiledata(Db, batch_updates)

        if batch_data:
            self.batch_update_celldata_optimized(Db, batch_data, df)

        # Free background images for this timepoint to reclaim memory
        try:
            del self.Norm.backgrounds[well][timepoint]
        except KeyError:
            pass

        gc.collect()

        total_time = time() - strt
        print(f'Finished well {well} + timepoint {timepoint} in {total_time:.2f}s')

    def process_single_tile(
        self,
        row: Any,
        well: str,
        timepoint: str,
    ) -> Optional[Tuple[str, pd.DataFrame, np.ndarray]]:
        """Segment a single tile image and return the mask and properties.

        Args:
            row: A tiledata row with image path attributes.
            well: Well identifier.
            timepoint: Timepoint label.

        Returns:
            A tuple of (mask_path, regionprops_df, labelled_mask), or
            ``None`` if the image cannot be loaded or the method is
            ``tryall``.
        """
        # Prefer aligned montage; fall back to raw montage
        img_path: str = row.alignedmontagepath if pd.notna(row.alignedmontagepath) else row.newimagemontage

        if not os.path.exists(img_path):
            print(f"Warning: Image path does not exist: {img_path}")
            return None

        img: np.ndarray = tifffile.imread(img_path)

        # Determine threshold value
        if self.segmentation_method == 'manual':
            thresh = self.opt.manual_thresh
        elif self.segmentation_method == 'tryall':
            return self.handle_tryall_case(row, img)
        else:
            try:
                thresh = self.thresh_func(img)
            except ValueError:
                # Fallback: set threshold above any possible pixel value
                thresh = 65535

        # Binary threshold -> connected-component labelling
        regions = (img > thresh).astype(np.uint8) * 255
        masks: np.ndarray = measure.label(regions)

        props = measure.regionprops_table(
            masks,
            intensity_image=img,
            properties=self.region_props,
        )

        props_df = pd.DataFrame(props)
        props_df, masks = self.filter_by_area_optimized(props_df, masks)
        props_df, masks = self.filter_by_proximity(props_df, masks)

        mask_dir = os.path.join(self.analysisdir, self.mask_folder_name, row.well)
        maskpath: str = save_mask(masks, img_path, mask_dir)

        return maskpath, props_df, masks

    def handle_tryall_case(
        self,
        row: Any,
        img: np.ndarray,
    ) -> None:
        """Run ``skimage.filters.try_all_threshold`` and save the figure.

        This is a diagnostic helper that writes a comparison PNG instead
        of producing a usable mask.

        Args:
            row: Tiledata row (used for well name).
            img: Grayscale image array.

        Returns:
            None (no mask is produced).
        """
        fig, ax = self.thresh_func(img)
        savedir = os.path.join(self.analysisdir, self.mask_folder_name, row.well)
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        fig.savefig(os.path.join(savedir, f'try_all_{row.well}.png'))
        print(f'Saved {self.segmentation_method} segmentation mask to {savedir}')
        return None

    # ------------------------------------------------------------------
    # Area filtering
    # ------------------------------------------------------------------

    def filter_by_area_optimized(
        self,
        props_df: pd.DataFrame,
        labelled_mask: np.ndarray,
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """Remove labelled regions whose area falls outside the configured bounds.

        Uses vectorised boolean indexing and :func:`numpy.isin` for
        efficient label removal.

        Args:
            props_df: DataFrame with at least ``area`` and ``label`` columns.
            labelled_mask: 2-D integer array of labelled regions.

        Returns:
            Tuple of (filtered_props_df, updated_labelled_mask).
        """
        area_mask = (
            (self.opt.upper_area_thresh > props_df.area)
            & (props_df.area > self.opt.lower_area_thresh)
        )
        props_df = props_df[area_mask]

        if props_df.empty:
            labelled_mask.fill(0)
            return props_df, labelled_mask

        filtered_labels: set = set(props_df.label.values)
        all_labels = np.unique(labelled_mask)

        if len(all_labels) > 100:
            logger.debug(f"Area filtering: {len(all_labels)} -> {len(filtered_labels)} cells")

        # Zero out labels that did not pass the area filter
        to_delete = set(all_labels) - filtered_labels
        if to_delete:
            mask = np.isin(labelled_mask, list(to_delete))
            labelled_mask[mask] = 0
        return props_df, labelled_mask

    def filter_by_proximity(
        self,
        props_df: pd.DataFrame,
        labelled_mask: np.ndarray,
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """Remove cells whose centroids are within *proximity_filter_radius* pixels of another cell.

        Uses a KD-tree to efficiently find all centroid pairs closer than
        the configured radius.  Every cell involved in at least one such
        pair is removed (both partners), so only well-isolated cells
        remain for downstream tracking.

        Args:
            props_df: DataFrame with ``centroid_weighted-0`` (y) and
                ``centroid_weighted-1`` (x) columns, plus ``label``.
            labelled_mask: 2-D integer array of labelled regions.

        Returns:
            Tuple of (filtered_props_df, updated_labelled_mask).
        """
        if self.proximity_filter_radius <= 0 or len(props_df) < 2:
            return props_df, labelled_mask

        centroids = props_df[['centroid_weighted-1', 'centroid_weighted-0']].values
        tree = cKDTree(centroids)
        pairs = tree.query_pairs(r=self.proximity_filter_radius)

        if not pairs:
            return props_df, labelled_mask

        # Collect indices of all cells involved in any close pair
        clumped_indices: set = set()
        for i, j in pairs:
            clumped_indices.add(i)
            clumped_indices.add(j)

        clumped_labels: set = set(props_df.iloc[list(clumped_indices)]['label'].values)

        n_before = len(props_df)
        props_df = props_df[~props_df['label'].isin(clumped_labels)]
        n_removed = n_before - len(props_df)

        logger.warning(
            f'Proximity filter (r={self.proximity_filter_radius}px): '
            f'removed {n_removed}/{n_before} clumped cells, '
            f'{len(props_df)} remain'
        )

        # Zero out removed labels in the mask
        if clumped_labels:
            mask = np.isin(labelled_mask, list(clumped_labels))
            labelled_mask[mask] = 0

        return props_df, labelled_mask

    def filter_by_area(
        self,
        props_df: pd.DataFrame,
        labelled_mask: np.ndarray,
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """Remove labelled regions outside the configured area bounds.

        .. deprecated::
            Use :meth:`filter_by_area_optimized` instead.  This legacy
            implementation iterates over labels with a Python loop and is
            significantly slower on large masks.

        Args:
            props_df: DataFrame with ``area`` and ``label`` columns.
            labelled_mask: 2-D integer array of labelled regions.

        Returns:
            Tuple of (filtered_props_df, updated_labelled_mask).
        """
        props_df = props_df[
            (self.opt.upper_area_thresh > props_df.area)
            & (props_df.area > self.opt.lower_area_thresh)
        ]
        filtered_labels = props_df.label.tolist()
        all_labels = np.unique(labelled_mask)
        if len(all_labels) > 100:
            logger.debug(f"Legacy area filtering: {len(all_labels)} -> {len(filtered_labels)} cells")

        to_delete = list(set(all_labels) - set(props_df.label.tolist()))
        props_df = props_df[~props_df.label.isin(to_delete)]
        for lbl in to_delete:
            labelled_mask[labelled_mask == lbl] = 0
        return props_df, labelled_mask

    # ------------------------------------------------------------------
    # Database batch operations
    # ------------------------------------------------------------------

    def batch_update_tiledata(
        self,
        Db: Database,
        batch_updates: List[Dict[str, Any]],
    ) -> None:
        """Write mask paths back to tiledata rows in chunked batches.

        Args:
            Db: Active database connection.
            batch_updates: List of dicts, each containing ``kwargs`` (the
                row filter) and ``data`` (the columns to update).
        """
        chunk_size = 100
        for i in range(0, len(batch_updates), chunk_size):
            chunk = batch_updates[i:i+chunk_size]
            for update_data in chunk:
                Db.update('tiledata', update_data['data'], kwargs=update_data['kwargs'])

    def batch_update_celldata_optimized(
        self,
        Db: Database,
        batch_data: List[Tuple[Any, pd.DataFrame]],
        df: pd.DataFrame,
    ) -> None:
        """Insert celldata and intensitycelldata without per-tile deletes.

        Deletion is deferred to :meth:`bulk_delete_celldata` which runs
        once at the end of the entire well.

        Args:
            Db: Active database connection.
            batch_data: List of (tiledata_row, regionprops_DataFrame) tuples.
            df: Full tiledata DataFrame for this group (unused but kept
                for API consistency).
        """
        if not batch_data:
            return
        batch_update_celldata_and_intensitycelldata_no_delete(batch_data, Db)

    def bulk_delete_celldata(
        self,
        Db: Database,
        tile_ids: Set,
    ) -> None:
        """Delete all celldata and intensitycelldata for a set of tile IDs.

        Processes deletions in chunks to stay within database limits.
        Prefers ``Db.bulk_delete`` when available; otherwise falls back to
        per-ID :meth:`Db.delete_based_on_duplicate_name`.

        Args:
            Db: Active database connection.
            tile_ids: Set of ``tiledata_id`` UUIDs to clear.
        """
        if not tile_ids:
            return

        tile_ids_list = list(tile_ids)
        chunk_size = 50

        for i in range(0, len(tile_ids_list), chunk_size):
            chunk = tile_ids_list[i:i+chunk_size]
            try:
                if hasattr(Db, 'bulk_delete'):
                    Db.bulk_delete('celldata', {'tiledata_id': chunk})
                    Db.bulk_delete('intensitycelldata', {'tiledata_id': chunk})
                else:
                    for tile_id in chunk:
                        Db.delete_based_on_duplicate_name(
                            tablename='celldata',
                            kwargs={'tiledata_id': tile_id},
                        )
                        Db.delete_based_on_duplicate_name(
                            tablename='intensitycelldata',
                            kwargs={'tiledata_id': tile_id},
                        )
            except Exception as exc:
                # Log and continue with next chunk rather than aborting
                logger.error(f"bulk_delete_celldata failed for chunk: {exc}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_dict',
        help='path to pickle',
        default=f'/gladstone/finkbeiner/linsley/GXYTMPS/Nextflow-tmp/GALAXY/YD-Transdiff-XDP-Survival1-102822/GXYTMP/tmp_output.txt'
    )
    parser.add_argument(
        '--outfile',
        help='Tiff image of last tile',
        default=f'/gladstone/finkbeiner/linsley/GXYTMPS/Nextflow-tmp/GALAXY/YD-Transdiff-XDP-Survival1-102822/GXYTMP/tmp_output.tif'
    )

    parser.add_argument('--experiment',default='0907-FB-1-JL-gedi-test', type=str)

    parser.add_argument('--segmentation_method', default='sd_from_mean', choices=['sd_from_mean', 'minimum', 'yen', 'local', 'li', 'isodata', 'mean',
                                                          'otsu', 'sauvola', 'triangle', 'manual', 'tryall', 'cellpose'], type=str,
                        help='Auto segmentation method. "cellpose" = Cellpose-SAM v4 with CLAHE + debris cleanup.')
    # Cellpose-SAM params (only used when segmentation_method='cellpose'); tuned defaults for 20X fibroblast GFP morphology
    parser.add_argument('--batch_size', default=16, type=int, help="Cellpose eval batch size.")
    parser.add_argument('--cell_diameter', default=70, type=int, help="Cellpose diameter (SAM is scale-invariant; value has little effect).")
    parser.add_argument('--flow_threshold', default=0.6, type=float, help="Cellpose flow threshold.")
    parser.add_argument('--cell_probability', default=-1.5, type=float, help="Cellpose cellprob threshold (lower=more/dimmer cells).")
    parser.add_argument('--cleanup_k', default=3.0, type=float, help="Drop cells dimmer than bg median + k*MAD (0 disables).")
    parser.add_argument('--clahe_clip', default=0.03, type=float, help="CLAHE clip limit for adaptive contrast before Cellpose (0 disables).")
    parser.add_argument('--debris_area_tiny', default=800, type=int, help="Always drop mask objects smaller than this many px (0 disables).")
    parser.add_argument('--debris_area_max', default=3000, type=int, help="Objects smaller than this AND round (< debris_ecc_max) are debris (0 disables).")
    parser.add_argument('--debris_ecc_max', default=0.8, type=float, help="Eccentricity below this counts as round-ish debris candidate.")
    parser.add_argument('--img_norm_name', default='subtraction', choices=['division', 'subtraction', 'identity'], type=str,
                        help='Image normalization method using flatfield image.')
    parser.add_argument('--lower_area_thresh', default=50, type=int, help="Lowerbound for cell area. Remove cells with area less than this value.")
    parser.add_argument('--upper_area_thresh', default=36000, type=int, help="Upperbound for cell area. Remove cells with area greater than this value.")
    parser.add_argument('--sd_scale_factor', default=3.5, type=float, help="Standard Deviation (SD) scale factor if using sd_from_mean threshold.")
    parser.add_argument('--manual_thresh', default=0, type=int, help="Threshold if using manual threshold method.")
    parser.add_argument('--proximity_filter_radius', default=0, type=int,
                        help="Remove cells whose centroids are within this many pixels of another cell. 0 to disable.")
    parser.add_argument("--wells_toggle", default='include',
                        help="Chose whether to include or exclude specified wells.")
    parser.add_argument("--timepoints_toggle", default='include',
                        help="Chose whether to include or exclude specified timepoints.")
    parser.add_argument("--channels_toggle", default='include',
                        help="Chose whether to include or exclude specified channels.")
    parser.add_argument("--chosen_wells", "-cw",
                        dest="chosen_wells", default='E4',
                        help="Specify well to process")
    parser.add_argument("--chosen_timepoints", "-ct",
                        dest="chosen_timepoints", default='all',
                        help="Specify timepoints to include or exclude.")
    parser.add_argument("--chosen_channels", "-cc",
                        dest="chosen_channels", default='GFP-DMD1',
                        help="Morphology channel.")
    parser.add_argument('--tile', default=0, type=int, help="Select single tile to segment. Default is to segment all tiles.")

    args = parser.parse_args()
    Seg = Segmentation(args)
    Seg.run()

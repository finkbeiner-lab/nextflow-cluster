#!/usr/bin/env python
"""Cell segmentation with various methods - OPTIMIZED VERSION"""
import tifffile  # Faster than imageio for TIFF files
import argparse
from sql import Database
from skimage import (
    color, feature, filters, measure, morphology, segmentation, util
)
import os
import numpy as np
import pandas as pd
from db_util import Ops
from normalization import Normalize
import logging
import datetime
from time import time
from segmentation_helper_montage import save_mask, update_celldata_and_intensitycelldata, batch_update_celldata_and_intensitycelldata, batch_update_celldata_and_intensitycelldata_no_delete
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
from functools import partial
import pdb
from typing import List, Tuple, Dict, Any, Optional
import gc


logger = logging.getLogger("Segmentation")
# logger.propagate = False
now = datetime.datetime.now()
TIMESTAMP = '%d%02d%02d%02d%02d' % (now.year, now.month, now.day, now.hour, now.minute)
print(f'🚀 Starting segmentation processing at {now.strftime("%Y-%m-%d %H:%M:%S")}')

fink_log_dir = './finkbeiner_logs'
if not os.path.exists(fink_log_dir):
    os.makedirs(fink_log_dir)
logname = os.path.join(fink_log_dir, f'Segmentation-log_{TIMESTAMP}.log')
fh = logging.FileHandler(logname)
# fh.setLevel(logging.DEBUG)
logger.addHandler(fh)
logger.warning('Running OPTIMIZED Segmentation from Database.')


class Segmentation:
    def __init__(self, opt):
        self.opt = opt
        # Use CPU count for optimal parallelization
        self.thread_lim = min(multiprocessing.cpu_count(), 8)  # Cap at 8 to avoid memory issues
        self.segmentation_method = opt.segmentation_method
        assert len(self.opt.chosen_channels) > 0, 'Must select a channel for segmentation'
        logger.warning(f'Segmentation Method: {self.segmentation_method}')
        logger.warning(f'Using {self.thread_lim} threads for parallel processing')
        self.mask_folder_name = 'CellMasksMontage'
        
        # Pre-define threshold functions for faster lookup
        self.threshold_func = dict(sd_from_mean=self.sd_from_mean,
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
                                   tryall=filters.try_all_threshold)
        self.thresh_func = self.threshold_func[self.segmentation_method]
        
        # Pre-define region properties for faster processing
        self.region_props = ('label', 'area', 'centroid_weighted',
                            'orientation', 'solidity', 'extent',
                            'perimeter', 'eccentricity',
                            'intensity_max', 'intensity_mean',
                            'intensity_min', 'axis_major_length',
                            'axis_minor_length')
        
        self.Norm = Normalize(self.opt)
        _, self.analysisdir = self.Norm.get_raw_and_analysis_dir()
        logger.warning(f'Save directory: {self.analysisdir}')
        
        # Performance monitoring
        self.processed_tiles = 0
        self.total_tiles = 0
        self.start_time = None
        
        # Pre-allocate common arrays and data structures
        self._setup_optimizations()

    def _setup_optimizations(self):
        """Setup optimization-related attributes"""
        # Pre-create directories to avoid repeated os.path.exists checks
        self.mask_dirs = {}
        
        # Pre-allocate numpy arrays for common operations
        self.area_thresh_mask = None
        
        # Cache for frequently accessed data
        self._cache = {}

    def run(self):
        self.start_time = time()
        self.run_threshold()
        
        total_time = time() - self.start_time
        logger.warning(f'Completed threshold in {total_time:.2f}s')
        if self.total_tiles > 0:
            avg_time_per_tile = total_time / self.total_tiles
            logger.warning(f'Average time per tile: {avg_time_per_tile:.2f}s')
        
        print(f'✅ SEGMENTATION COMPLETED in {total_time:.2f}s ({total_time/60:.2f} min)')
        
        # Clean up memory
        self._cleanup()

    def _cleanup(self):
        """Clean up memory and caches"""
        self._cache.clear()
        gc.collect()

    def sd_from_mean(self, img):
        # Optimized calculation using numpy operations
        img_mean = np.mean(img)
        img_std = np.std(img)
        return int(img_mean + img_std * self.opt.sd_scale_factor)

    def run_threshold(self):
        Db = Database()
        tiledata_df = self.Norm.get_tiledata_df()

        # Filter by well - use vectorized operations
        well_mask = tiledata_df['well'] == self.opt.chosen_wells
        tiledata_df = tiledata_df[well_mask]

        if tiledata_df.empty:
            logger.warning(f"No tile data found for well {self.opt.chosen_wells} and timepoint {self.opt.chosen_timepoints}")
            return

        self.total_tiles = len(tiledata_df)
        logger.warning(f"Processing {self.total_tiles} tiles")
        print(f"🔬 Processing {self.total_tiles} tiles for well {self.opt.chosen_wells}")

        # Pre-create mask directories for all wells
        self._precreate_directories(tiledata_df)
        
        # Collect all tile IDs for bulk delete at the end
        all_tile_ids = set(tiledata_df['id'].tolist())
        
        grouped = tiledata_df.groupby(['well', 'timepoint'])
        
        # Process each well/timepoint group in parallel
        for (well, timepoint), df in grouped:
            self.thresh_single_parallel(Db, df, well, timepoint)
        
        # Perform bulk delete for all tiles at the end
        if all_tile_ids:
            self.bulk_delete_celldata(Db, all_tile_ids)

    def _precreate_directories(self, tiledata_df):
        """Pre-create all necessary directories to avoid repeated os.path.exists checks"""
        wells = tiledata_df['well'].unique()
        for well in wells:
            mask_dir = os.path.join(self.analysisdir, self.mask_folder_name, well)
            if mask_dir not in self.mask_dirs:
                if not os.path.exists(mask_dir):
                    os.makedirs(mask_dir)
                    logger.info(f"Created mask directory: {mask_dir}")
                self.mask_dirs[mask_dir] = True

    def thresh_single_parallel(self, Db, df, well, timepoint):
        """Process tiles in parallel for a single well/timepoint - OPTIMIZED"""
        strt = time()
        
        # Use numpy sort for better performance
        df = df.sort_values(by='tile').copy()
        
        print(f'🔬 Processing well {well} at timepoint {timepoint} with {len(df)} tiles')
        
        # Pre-allocate batch containers with estimated sizes
        estimated_cells = len(df) * 100  # Estimate 100 cells per tile
        batch_updates = []
        batch_data = []  # Combined batch data for efficient processing
        
        # Process tiles in parallel with optimized chunking
        chunk_size = max(1, len(df) // (self.thread_lim * 2))  # Optimize chunk size
        
        with ThreadPoolExecutor(max_workers=self.thread_lim) as executor:
            # Submit jobs in chunks for better memory management
            futures = []
            for i in range(0, len(df), chunk_size):
                chunk = df.iloc[i:i+chunk_size]
                for _, row in chunk.iterrows():
                    future = executor.submit(self.process_single_tile, row, well, timepoint)
                    futures.append((future, row))
            
            # Process results as they complete
            for future, row in futures:
                try:
                    result = future.result()
                    if result:
                        maskpath, props_df, masks = result
                        
                        # Collect for batch operations
                        batch_updates.append({
                            'kwargs': dict(experimentdata_id=row.experimentdata_id,
                                         welldata_id=row.welldata_id,
                                         channeldata_id=row.channeldata_id,
                                         tile=row.tile,
                                         timepoint=row.timepoint),
                            'data': dict(alignedmontagemaskpath=maskpath)
                        })
                        
                        # Prepare cell data for batch insertion
                        if not props_df.empty:
                            batch_data.append((row, props_df))
                        
                        self.processed_tiles += 1
                        progress = (self.processed_tiles / self.total_tiles) * 100
                        print(f'✅ Progress: {progress:.1f}% ({self.processed_tiles}/{self.total_tiles})')
                        
                except Exception as exc:
                    print(f'❌ Tile {row.tile} generated an exception: {exc}')
                    continue
        
        # Perform optimized batch database operations
        if batch_updates:
            self.batch_update_tiledata(Db, batch_updates)
        
        if batch_data:
            self.batch_update_celldata_optimized(Db, batch_data, df)
        
        # Clean up memory
        try:
            del self.Norm.backgrounds[well][timepoint]
        except KeyError:
            pass
        
        # Force garbage collection
        gc.collect()
        
        total_time = time() - strt
        print(f'✅ Finished well {well} + timepoint {timepoint} in {total_time:.2f}s')

    def process_single_tile(self, row, well, timepoint):
        """Process a single tile - HIGHLY OPTIMIZED for parallel execution"""
        tile_start_time = time()
        
        # Select aligned image if available, otherwise raw
        img_path = row.alignedmontagepath if pd.notna(row.alignedmontagepath) else row.newimagemontage
        
        if not os.path.exists(img_path):
            print(f"Warning: Image path does not exist: {img_path}")
            return None
        
        # Load image with optimized settings
        img = tifffile.imread(img_path)
        
        # Apply thresholding with optimized operations
        if self.segmentation_method == 'manual':
            thresh = self.opt.manual_thresh
        elif self.segmentation_method == 'tryall':
            return self.handle_tryall_case(row, img)
        else:
            try:
                thresh = self.thresh_func(img)
            except ValueError:
                thresh = np.ones_like(img) * 65535
        
        # Create regions and masks with optimized numpy operations
        regions = (img > thresh).astype(np.uint8) * 255
        masks = measure.label(regions)
        
        # Calculate properties with pre-defined list
        props = measure.regionprops_table(masks, intensity_image=img,
                                        properties=self.region_props)
        
        props_df = pd.DataFrame(props)
        props_df, masks = self.filter_by_area_optimized(props_df, masks)
        
        # Save mask using pre-created directory
        mask_dir = os.path.join(self.analysisdir, self.mask_folder_name, row.well)
        maskpath = save_mask(masks, img_path, mask_dir)
        
        return maskpath, props_df, masks

    def handle_tryall_case(self, row, img):
        """Handle the tryall segmentation method separately"""
        fig, ax = self.thresh_func(img)
        savedir = os.path.join(self.analysisdir, self.mask_folder_name, row.well)
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        fig.savefig(os.path.join(savedir, f'try_all_{row.well}.png'))
        print(f'Saved {self.segmentation_method} segmentation mask to {savedir}')
        return None

    def filter_by_area_optimized(self, props_df: pd.DataFrame, labelled_mask):
        """Optimized area filtering using vectorized operations"""
        # Use vectorized boolean indexing for better performance
        area_mask = (self.opt.upper_area_thresh > props_df.area) & (props_df.area > self.opt.lower_area_thresh)
        props_df = props_df[area_mask]
        
        if props_df.empty:
            # If no cells pass filter, return empty mask
            labelled_mask.fill(0)
            return props_df, labelled_mask
        
        # Use numpy operations for faster label removal
        filtered_labels = set(props_df.label.values)
        all_labels = np.unique(labelled_mask)
        
        # Only log if there's significant filtering happening
        if len(all_labels) > 100:  # Only log for tiles with many cells
            logger.debug(f"Area filtering: {len(all_labels)} → {len(filtered_labels)} cells")
        
        # Vectorized operation to remove unwanted labels
        to_delete = set(all_labels) - filtered_labels
        if to_delete:
            # Use numpy's in1d for faster filtering
            mask = np.isin(labelled_mask, list(to_delete))
            labelled_mask[mask] = 0
        return props_df, labelled_mask

    def batch_update_tiledata(self, Db, batch_updates):
        """Perform batch updates to tiledata - OPTIMIZED"""
        # Use bulk update if available, otherwise process in chunks
        chunk_size = 100  # Process in chunks to avoid memory issues
        for i in range(0, len(batch_updates), chunk_size):
            chunk = batch_updates[i:i+chunk_size]
            for update_data in chunk:
                Db.update('tiledata', update_data['data'], kwargs=update_data['kwargs'])

    def batch_update_celldata_optimized(self, Db, batch_data, df):
        """Highly optimized batch operations for celldata and intensitycelldata - NO DELETES"""
        if not batch_data:
            return
        
        # Use the optimized helper function from segmentation_helper_montage
        # Skip individual deletes - they will be done in bulk at the end
        batch_update_celldata_and_intensitycelldata_no_delete(batch_data, Db)

    def bulk_delete_celldata(self, Db, tile_ids):
        """Perform bulk delete for all tiles at once - HIGHLY OPTIMIZED"""
        if not tile_ids:
            return
        
        # Convert to list for database operations
        tile_ids_list = list(tile_ids)
        
        # Process in chunks to avoid database limits
        chunk_size = 50  # Adjust based on your database limits
        for i in range(0, len(tile_ids_list), chunk_size):
            chunk = tile_ids_list[i:i+chunk_size]
            
            # Bulk delete from celldata table
            try:
                # Use IN clause for bulk delete if your database supports it
                if hasattr(Db, 'bulk_delete'):
                    Db.bulk_delete('celldata', {'tiledata_id': chunk})
                    Db.bulk_delete('intensitycelldata', {'tiledata_id': chunk})
                else:
                    # Fall back to individual deletes in chunks
                    for tile_id in chunk:
                        Db.delete_based_on_duplicate_name(tablename='celldata', 
                                                        kwargs={'tiledata_id': tile_id})
                        Db.delete_based_on_duplicate_name(tablename='intensitycelldata', 
                                                        kwargs={'tiledata_id': tile_id})
            except Exception as e:
                # Continue with next chunk
                pass

    def filter_by_area(self, props_df:pd.DataFrame, labelled_mask):
        """Legacy function - kept for backward compatibility"""
        props_df = props_df[(self.opt.upper_area_thresh > props_df.area) & (props_df.area > self.opt.lower_area_thresh) ]
        filtered_labels = props_df.label.tolist()
        all_labels = np.unique(labelled_mask)
        # Only log if there's significant filtering happening
        if len(all_labels) > 100:  # Only log for tiles with many cells
            logger.debug(f"Legacy area filtering: {len(all_labels)} → {len(filtered_labels)} cells")
        # filtered_mask = np.zeros_like(labelled_mask)
        # rows, cols = np.shape(labelled_mask)
        # for i in range(rows):
        #     for j in range(cols):
        #         filtered_mask[i,j] = labelled_mask[i,j] if labelled_mask[i,j] in filtered_labels else 0

        to_delete = list(set(all_labels) - set(props_df.label.tolist()))
        props_df = props_df[~props_df.label.isin(to_delete)]
        for lbl in to_delete:
            labelled_mask[labelled_mask == lbl] = 0
        return props_df, labelled_mask


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
                                                          'otsu', 'sauvola', 'triangle', 'manual', 'tryall'], type=str,
                        help='Auto segmentation method.')
    parser.add_argument('--img_norm_name', default='subtraction', choices=['division', 'subtraction', 'identity'], type=str,
                        help='Image normalization method using flatfield image.')
    parser.add_argument('--lower_area_thresh', default=50, type=int, help="Lowerbound for cell area. Remove cells with area less than this value.")
    parser.add_argument('--upper_area_thresh', default=36000, type=int, help="Upperbound for cell area. Remove cells with area greater than this value.")
    parser.add_argument('--sd_scale_factor', default=3.5, type=float, help="Standard Deviation (SD) scale factor if using sd_from_mean threshold.")
    parser.add_argument('--manual_thresh', default=0, type=int, help="Threshold if using manual threshold method.")
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
   # parser.add_argument('--use_aligned_tiles', action='store_true',
                   # help="Use aligned tile images if available")

    args = parser.parse_args()
    Seg = Segmentation(args)
    Seg.run()
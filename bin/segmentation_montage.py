#!/usr/bin/env python
"""Cell segmentation with various methods"""
import imageio.v2 as imageio
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
from segmentation_helper_montage import save_mask, update_celldata_and_intensitycelldata
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
from functools import partial
import pdb


logger = logging.getLogger("Segmentation")
# logger.propagate = False
now = datetime.datetime.now()
TIMESTAMP = '%d%02d%02d%02d%02d' % (now.year, now.month, now.day, now.hour, now.minute)
print('Timestamp', TIMESTAMP)
fink_log_dir = './finkbeiner_logs'
if not os.path.exists(fink_log_dir):
    os.makedirs(fink_log_dir)
logname = os.path.join(fink_log_dir, f'Segmentation-log_{TIMESTAMP}.log')
fh = logging.FileHandler(logname)
# fh.setLevel(logging.DEBUG)
logger.addHandler(fh)
logger.warning('Running Segmentation from Database.')


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
        self.Norm = Normalize(self.opt)
        _, self.analysisdir = self.Norm.get_raw_and_analysis_dir()
        logger.warning(f'Save directory: {self.analysisdir}')
        
        # Performance monitoring
        self.processed_tiles = 0
        self.total_tiles = 0
        self.start_time = None

    def run(self):
        self.start_time = time()
        self.run_threshold()
        total_time = time() - self.start_time
        logger.warning(f'Completed threshold in {total_time:.2f}s')
        if self.total_tiles > 0:
            avg_time_per_tile = total_time / self.total_tiles
            logger.warning(f'Average time per tile: {avg_time_per_tile:.2f}s')

    def sd_from_mean(self, img):
        thresh = int(np.mean(img) + np.std(img) * self.opt.sd_scale_factor)
        return thresh

    def run_threshold(self):
        Db = Database()
        tiledata_df = self.Norm.get_tiledata_df()

        # Filter by well
        tiledata_df = tiledata_df[tiledata_df['well'] == self.opt.chosen_wells]

        # Filter by timepoint if not "all"
        # if self.opt.chosen_timepoints != 'all':
        #     tiledata_df = tiledata_df[tiledata_df['timepoint'] == int(self.opt.chosen_timepoints)]

        if tiledata_df.empty:
            logger.warning(f"No tile data found for well {self.opt.chosen_wells} and timepoint {self.opt.chosen_timepoints}")
            return

        self.total_tiles = len(tiledata_df)
        logger.warning(f"Processing {self.total_tiles} tiles")

        grouped = tiledata_df.groupby(['well', 'timepoint'])
        
        # Process each well/timepoint group in parallel
        for (well, timepoint), df in grouped:
            self.thresh_single_parallel(Db, df, well, timepoint)

    def thresh_single_parallel(self, Db, df, well, timepoint):
        """Process tiles in parallel for a single well/timepoint"""
        strt = time()
        df.sort_values(by='tile', inplace=True)
        
        print(f'Processing well {well} at timepoint {timepoint} with {len(df)} tiles')
        
        # Prepare batch operations
        batch_updates = []
        batch_celldata = []
        batch_intensitycelldata = []
        
        # Process tiles in parallel
        with ThreadPoolExecutor(max_workers=self.thread_lim) as executor:
            # Submit all tile processing jobs
            future_to_row = {
                executor.submit(self.process_single_tile, row, well, timepoint): row 
                for _, row in df.iterrows()
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_row):
                row = future_to_row[future]
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
                            batch_celldata.extend(self.prepare_celldata_batch(row, props_df))
                            batch_intensitycelldata.extend(self.prepare_intensitycelldata_batch(row, props_df))
                        
                        self.processed_tiles += 1
                        progress = (self.processed_tiles / self.total_tiles) * 100
                        print(f'Progress: {progress:.1f}% ({self.processed_tiles}/{self.total_tiles}) - Completed tile {row.tile}')
                        
                except Exception as exc:
                    print(f'Tile {row.tile} generated an exception: {exc}')
                    continue
        
        # Perform batch database operations
        if batch_updates:
            self.batch_update_tiledata(Db, batch_updates)
        
        if batch_celldata:
            self.batch_update_celldata(Db, batch_celldata, batch_intensitycelldata, df)
        
        try:
            del self.Norm.backgrounds[well][timepoint]
        except KeyError:
            print(f"⚠️  No background found for well {well}, timepoint {timepoint}. Skipping delete.")
        
        print(f'Finished well {well} + timepoint {timepoint} in {time() - strt:.2f}s')

    def process_single_tile(self, row, well, timepoint):
        """Process a single tile - optimized for parallel execution"""
        tile_strt = time()
        
        # Select aligned image if available, otherwise raw
        img_path = row.alignedmontagepath if pd.notna(row.alignedmontagepath) else row.newimagemontage
        
        if not os.path.exists(img_path):
            print(f"Warning: Image path does not exist: {img_path}")
            return None
        
        # Load image
        img = imageio.imread(img_path)
        smoothed_im = img
        
        # Apply thresholding
        if self.segmentation_method == 'manual':
            thresh = self.opt.manual_thresh
        elif self.segmentation_method == 'tryall':
            # Handle tryall case separately
            return self.handle_tryall_case(row, smoothed_im)
        else:
            try:
                thresh = self.thresh_func(smoothed_im)
            except ValueError:
                thresh = np.ones_like(img) * 65535
        
        # Create regions and masks
        regions = np.uint8((smoothed_im > thresh) * 255)
        masks = measure.label(regions)
        
        # Calculate properties
        props = measure.regionprops_table(masks, intensity_image=img,
                                        properties=('label', 'area', 'centroid_weighted',
                                                   'orientation', 'solidity', 'extent',
                                                   'perimeter', 'eccentricity',
                                                   'intensity_max', 'intensity_mean',
                                                   'intensity_min', 'axis_major_length',
                                                   'axis_minor_length'))
        
        props_df = pd.DataFrame(props)
        props_df, masks = self.filter_by_area(props_df, masks)
        
        # Save mask
        savedir = os.path.join(self.analysisdir, self.mask_folder_name, row.well)
        maskpath = save_mask(masks, img_path, savedir)
        
        print(f'Saved {self.segmentation_method} segmentation mask to {maskpath}')
        
        return maskpath, props_df, masks

    def handle_tryall_case(self, row, smoothed_im):
        """Handle the tryall segmentation method separately"""
        fig, ax = self.thresh_func(smoothed_im)
        savedir = os.path.join(self.analysisdir, self.mask_folder_name, row.well)
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        fig.savefig(os.path.join(savedir, f'try_all_{row.well}.png'))
        print(f'Saved {self.segmentation_method} segmentation mask to {savedir}')
        return None

    def prepare_celldata_batch(self, row, props_df):
        """Prepare cell data for batch insertion"""
        celldata_batch = []
        for _, prow in props_df.iterrows():
            celldata_batch.append({
                'experimentdata_id': row.experimentdata_id,
                'welldata_id': row.welldata_id,
                'tiledata_id': row.id,
                'randomcellid_montage': prow.label,
                'centroid_x': prow['centroid_weighted-1'],
                'centroid_y': prow['centroid_weighted-0'],
                'perimeter': prow.perimeter,
                'area': prow.area,
                'solidity': prow.solidity,
                'extent': prow.extent,
                'eccentricity': prow.eccentricity,
                'axis_major_length': prow.axis_major_length,
                'axis_minor_length': prow.axis_minor_length,
            })
        return celldata_batch

    def prepare_intensitycelldata_batch(self, row, props_df):
        """Prepare intensity cell data for batch insertion"""
        intensity_batch = []
        for _, prow in props_df.iterrows():
            intensity_batch.append({
                'experimentdata_id': row.experimentdata_id,
                'welldata_id': row.welldata_id,
                'tiledata_id': row.id,
                'channeldata_id': row.channeldata_id,
                'intensity_max': prow.intensity_max,
                'intensity_mean': prow.intensity_mean,
                'intensity_min': prow.intensity_min
            })
        return intensity_batch

    def batch_update_tiledata(self, Db, batch_updates):
        """Perform batch updates to tiledata"""
        print(f"Performing batch update of {len(batch_updates)} tiles")
        for update_data in batch_updates:
            Db.update('tiledata', update_data['data'], kwargs=update_data['kwargs'])

    def batch_update_celldata(self, Db, batch_celldata, batch_intensitycelldata, df):
        """Perform batch operations for celldata and intensitycelldata"""
        print(f"Performing batch operations for {len(batch_celldata)} cells")
        
        # Clear existing data for all tiles in this batch
        for _, row in df.iterrows():
            check_dct = dict(experimentdata_id=row.experimentdata_id,
                           welldata_id=row.welldata_id,
                           tiledata_id=row.id)
            Db.delete_based_on_duplicate_name(tablename='celldata', kwargs=check_dct)
            Db.delete_based_on_duplicate_name(tablename='intensitycelldata', kwargs=check_dct)
        
        # Batch insert new data
        if batch_celldata:
            from segmentation_helper_montage import convert_numpy_types
            batch_celldata = convert_numpy_types(batch_celldata)
            Db.add_row(tablename='celldata', dct=batch_celldata)
        
        if batch_intensitycelldata:
            from segmentation_helper_montage import convert_numpy_types
            batch_intensitycelldata = convert_numpy_types(batch_intensitycelldata)
            Db.add_row(tablename='intensitycelldata', dct=batch_intensitycelldata)

    def filter_by_area(self, props_df:pd.DataFrame, labelled_mask):
        props_df = props_df[(self.opt.upper_area_thresh > props_df.area) & (props_df.area > self.opt.lower_area_thresh) ]
        filtered_labels = props_df.label.tolist()
        all_labels = np.unique(labelled_mask)
        print('num masks before area filters', len(all_labels))
        # filtered_mask = np.zeros_like(labelled_mask)
        # rows, cols = np.shape(labelled_mask)
        # for i in range(rows):
        #     for j in range(cols):
        #         filtered_mask[i,j] = labelled_mask[i,j] if labelled_mask[i,j] in filtered_labels else 0

        to_delete = list(set(all_labels) - set(props_df.label.tolist()))
        props_df = props_df[~props_df.label.isin(to_delete)]
        for lbl in to_delete:
            labelled_mask[labelled_mask == lbl] = 0
        print('num masks after area filters', len(np.unique(labelled_mask)))
        return props_df, labelled_mask



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_dict',
        help='path to pickle',
        default=f'/gladstone/finkbeiner/linsley/josh/GALAXY/YD-Transdiff-XDP-Survival1-102822/GXYTMP/tmp.pkl'
    )
    parser.add_argument(
        '--outfile',
        help='Tiff image of last tile',
        default=f'/gladstone/finkbeiner/linsley/josh/GALAXY/YD-Transdiff-XDP-Survival1-102822/GXYTMP/tmp_output.tif'
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
    print(args)
    Seg = Segmentation(args)
    Seg.run()

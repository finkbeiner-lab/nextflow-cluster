#!/usr/bin/env python
import sys
import cv2
import numpy as np
import logging
import argparse
import os
from db_util import Ops
from sql import Database
import pandas as pd
import collections
import ast  # Safe way to evaluate a string representation of a list
import tifffile  # For reading TIFF files saved by segmentation script
import datetime
from time import time
from multiprocessing import Pool, cpu_count
from functools import partial


logger = logging.getLogger("TrackingDB")
logging.basicConfig(
    stream=sys.stderr,       # Send logs to stderr (captured by SLURM)
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Add file logging similar to segmentation script
now = datetime.datetime.now()
TIMESTAMP = '%d%02d%02d%02d%02d' % (now.year, now.month, now.day, now.hour, now.minute)
print(f'🚀 Starting tracking processing at {now.strftime("%Y-%m-%d %H:%M:%S")}')

fink_log_dir = './finkbeiner_logs'
if not os.path.exists(fink_log_dir):
    os.makedirs(fink_log_dir)
logname = os.path.join(fink_log_dir, f'Tracking-log_{TIMESTAMP}.log')
fh = logging.FileHandler(logname)
logger.addHandler(fh)
logger.warning('Running OPTIMIZED Tracking from Database.')

def read_tiff_safe(file_path):
    """Optimized TIFF reading with caching and fast fallback"""
    try:
        # Try tifffile first (faster for segmentation-generated files)
        img = tifffile.imread(file_path)
        return img
    except Exception:
        try:
            # Fallback to OpenCV
            img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
            return img if img is not None else None
        except Exception:
            return None

def process_label_contour(args):
    """Process a single label to find its contour - for parallel processing"""
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

def process_contours_parallel(mask, labels, num_processes=None):
    """Process contours in parallel for better performance"""
    if num_processes is None:
        # Use 75% of available cores, similar to segmentation script approach
        available_cores = cpu_count()
        num_processes = max(1, int(available_cores * 0.75))
        print(f'    Using {num_processes} cores out of {available_cores} available (75%)')
    
    if len(labels) < 50:  # Don't parallelize for small numbers
        # Process sequentially for small label counts
        results = []
        for label in labels:
            cell = process_label_contour((mask, label))
            if cell:
                results.append(cell)
        return results
    
    # Process in parallel for large label counts
    args_list = [(mask, label) for label in labels]
    
    with Pool(processes=num_processes) as pool:
        results = pool.map(process_label_contour, args_list)
    
    # Filter out None results
    return [cell for cell in results if cell is not None]

class Cell:
    """A class that makes cells from contours or masks."""
    def __init__(self, cnt, randomcellid_montage=None):
        self.cnt = cnt
        self.randomcellid_montage = randomcellid_montage
        self._circle_cache = None

    def get_circle(self):
        if self._circle_cache is None:
            self._circle_cache = cv2.minEnclosingCircle(self.cnt)
        return self._circle_cache

    def evaluate_overlap(self, circle2):
        c2, r2 = circle2
        c1, r1 = self.get_circle()
        dist = np.hypot(c1[0] - c2[0], c1[1] - c2[1])
        return dist < (r1 + r2) * 0.8

    def evaluate_dist(self, circle2):
        c2, _ = circle2
        c1, _ = self.get_circle()
        return np.hypot(c1[0] - c2[0], c1[1] - c2[1])

def sort_cell_info_by_index(time_dict, time_list):
    for tp in time_list:
        time_dict[tp] = sorted(
            [(int(idx), obj) for idx, obj in time_dict[tp]],
            key=lambda x: x[0]
        )
    return time_dict

def populate_cell_ind_overlap(time_dict, time_list):
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

def populate_cell_ind_closest(time_dict, time_list, max_dist=100):
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

class MontageDBTracker:
    def __init__(self, experiment, track_type, max_dist, target_channel):
        self.Db = Database()
        self.experiment = experiment
        self.track_type = track_type
        self.max_dist = max_dist
        self.target_channel = target_channel.split(',')
        self.analysisdir = ""
        # Cache for path transformations
        self._path_cache = {}
        
        # Performance monitoring
        self.start_time = None
        self.processed_wells = 0
        self.total_wells = 0
        
        # Dynamic core allocation for parallel processing
        available_cores = cpu_count()
        self.max_cores = max(1, int(available_cores * 0.75))  # Use 75% of available cores
        
        logger.info(f"Initialized MontageDBTracker for experiment {experiment}")
        logger.info(f"Using {self.max_cores} cores out of {available_cores} available (75%)")
    
    def _get_channel_path(self, mask_path, channel):
        """Optimized path transformation with caching."""
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
        for i, part in enumerate(parts):
            if any(part.startswith(prefix) for prefix in ['Epi-', 'DAPI', 'Cy', 'FITC', 'RFP', 'Confocal-']):
                parts[i] = channel
                break
        filename = '_'.join(parts)
        aligned_path = os.path.join(dirname, filename)

        # Convert file extension
        if self.selected:
            aligned_path = aligned_path.replace('_MONTAGE_ALIGNED_ENCODED.tif', '_MONTAGE_ALIGNED.tif')
        else:
            aligned_path = aligned_path.replace('_MONTAGE_ENCODED.tif', '_MONTAGE.tif')
        
        self._path_cache[cache_key] = aligned_path
        return aligned_path

    def gather_encoded_from_db(self, wells, channel_marker="_MONTAGE_ALIGNED_ENCODED"):
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
                        
                        # Use parallel processing for contour finding
                        cells = process_contours_parallel(mask, labels, num_processes=self.max_cores)
                        
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

    def get_cell_props(self, mask, image_stack, label):
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

        return cx, cy, area, intensities

    def run(self, wells):
        self.start_time = time()
        self.total_wells = len(wells)
        
        logger.info(f"Starting tracking for wells: {wells}")
        
        exp_id = self.Db.get_table_uuid('experimentdata', {'experiment': self.experiment})
        all_wells, df, tiledata_df = self.gather_encoded_from_db(wells)

        out_records = []

        for well, time_dict in all_wells.items():
            welldata_id = self.Db.get_table_uuid('welldata', {'experimentdata_id': exp_id, 'well': well})
            tps = sorted(time_dict.keys())
            
            tracked = (
                populate_cell_ind_overlap(time_dict, tps)
                if self.track_type == 'overlap'
                else populate_cell_ind_closest(time_dict, tps, max_dist=self.max_dist)
            )
            sorted_td = sort_cell_info_by_index(tracked, tps)

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
                cv2.imwrite(tracked_path, tracked_mask)
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
                            logger.warning(f"[WARN] Could not read aligned image for {ch} at: {img_path}")
                    else:
                        logger.warning(f"[WARN] Missing aligned image for {ch} at: {img_path}")

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

            # More efficient CSV writing
            file_exists = os.path.exists(outfile)
            out_df.to_csv(outfile, mode='a' if file_exists else 'w', 
                         header=not file_exists, index=False)
            logger.info(f"Wrote {len(out_records)} tracked records to {outfile}")
        else:
            logger.warning("No tracking records generated")
        
        # Final completion logging
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
    args = parser.parse_args()

    wells = [w.strip() for w in args.wells.split(',')]
    tracker = MontageDBTracker(args.experiment, args.track_type, args.max_dist, args.target_channel)
    tracker.run(wells)


#!/usr/bin/env python
"""Overlay cell IDs on aligned montage images based on tracked CSV and database experiment info."""

import imageio
import os
from PIL import Image, ImageFont, ImageDraw
import numpy as np
import pandas as pd
import argparse
from sql import Database
import pdb
import datetime
from time import time
from multiprocessing import Pool, cpu_count
from functools import partial


class OverlayBatch:
    def __init__(self, opt):
        self.opt = opt
        self.Db = Database()
        self.experiment_name = opt.experiment_name

        # Fetch analysisdir from experimentdata table
        result = self.Db.get_table_value(
            tablename='experimentdata',
            column='analysisdir',
            kwargs=dict(experiment=self.experiment_name)
        )

        if not result or not result[0][0]:
            raise ValueError(f"[ERROR] analysisdir not found for experiment '{self.experiment_name}'")

        self.experiment_root = result[0][0].rstrip('/')
        print(f"[INFO] Found experiment root: {self.experiment_root}")
        
        aligned_path = os.path.join(self.experiment_root, "AlignedMontages")
        montaged_path = os.path.join(self.experiment_root, "MontagedImages")

        if os.path.isdir(aligned_path) and os.listdir(aligned_path):
            self.montage_root = aligned_path
            self.selected = True
            print(f"[INFO] Using AlignedMontages directory: {aligned_path}")
        else:
            self.montage_root = montaged_path
            self.selected = False
            print(f"[INFO] Using MontagedImages directory: {montaged_path}")

        self.overlay_root = os.path.join(self.experiment_root, "Overlay_Montages")

        self.font = ImageFont.truetype('/usr/share/fonts/dejavu/DejaVuSansMono.ttf', 50)

        summary_path = os.path.join(
            self.experiment_root,
            f"{self.experiment_name}_tracked_montage_summary.csv"
        )
        if not os.path.exists(summary_path):
            raise FileNotFoundError(f"Tracking CSV not found: {summary_path}")
        self.df = pd.read_csv(summary_path)
        self.df['timepoint'] = self.df['timepoint'].astype(int)
        print(f"[INFO] Loaded tracking summary: {summary_path}")
        
        # Dynamic core allocation (75% of available cores)
        self.max_cores = max(1, int(cpu_count() * 0.75))
        print(f"[INFO] Using {self.max_cores} cores for parallel processing")
        
        # Performance monitoring
        self.start_time = None

    def run(self):
        self.start_time = time()
        
        # Collect all timepoint tasks for parallel processing
        timepoint_tasks = []
        
        for well in os.listdir(self.montage_root):
            well_path = os.path.join(self.montage_root, well)
            if not os.path.isdir(well_path):
                continue

            # Filter wells
            if self.opt.wells_toggle == "include" and self.opt.chosen_wells != "all":
                if well not in self.opt.chosen_wells.split(","):
                    continue
            elif self.opt.wells_toggle == "exclude":
                if well in self.opt.chosen_wells.split(","):
                    continue

            for fname in os.listdir(well_path):
                # Only process files from the selected directory type
                if self.selected:
                    # Processing from AlignedMontages - only process aligned files
                    if "_MONTAGE_ALIGNED.tif" not in fname:
                        continue
                else:
                    # Processing from MontagedImages - only process regular montage files
                    if "_MONTAGE.tif" not in fname or "_MONTAGE_ALIGNED.tif" in fname:
                        continue
                
                aligned_path = os.path.join(well_path, fname)
                timepoint = self.extract_timepoint(fname)

                # Filter timepoints
                if self.opt.timepoints_toggle == "include" and self.opt.chosen_timepoints != "all":
                    tp_range = self.parse_timepoints(self.opt.chosen_timepoints)
                    if timepoint not in tp_range:
                        continue
                elif self.opt.timepoints_toggle == "exclude":
                    tp_range = self.parse_timepoints(self.opt.chosen_timepoints)
                    if timepoint in tp_range:
                        continue

                # Add to tasks for parallel processing
                timepoint_tasks.append((well, timepoint, aligned_path))

        print(f"[INFO] Processing {len(timepoint_tasks)} timepoints in parallel using {self.max_cores} cores")
        
        # Debug: Print timepoint details
        unique_timepoints = set()
        for task in timepoint_tasks:
            well, timepoint, aligned_path = task
            unique_timepoints.add(timepoint)
        
        print(f"[DEBUG] Unique timepoints found: {sorted(unique_timepoints)}")
        
        # Prepare data for multiprocessing (avoid pickle issues)
        df_data = self.df.to_dict('records')  # Convert DataFrame to list of dicts
        opt_params = {
            'contrast': self.opt.contrast,
            'shift': self.opt.shift
        }
        
        # Prepare tasks with serializable data
        mp_tasks = []
        for well, timepoint, aligned_path in timepoint_tasks:
            mp_tasks.append((well, timepoint, aligned_path, df_data, self.experiment_name, opt_params, self.overlay_root))
        
        # Process timepoints in parallel
        if len(mp_tasks) > 1:
            with Pool(processes=self.max_cores) as pool:
                results = pool.map(overlay_single_timepoint, mp_tasks)
        else:
            # Fallback to sequential for single timepoint
            results = [overlay_single_timepoint(task) for task in mp_tasks]
        
        # Print results
        for result in results:
            print(result)
        
        # Final completion logging
        total_time = time() - self.start_time
        print(f'Overlay Montage completed in {total_time:.1f} seconds ({total_time/60:.1f} min)')

    def extract_timepoint(self, filename):
        parts = filename.split('_')
        for part in parts:
            if part.startswith('T') and part[1:].isdigit():
                return int(part[1:])
        raise ValueError(f"Could not find timepoint in filename: {filename}")

    def parse_timepoints(self, tp_string):
        if tp_string == "all":
            return list(range(100))  # Assume max 100 timepoints
        elif "-" in tp_string:
            start, end = map(int, tp_string.replace("T", "").split("-"))
            return list(range(start, end + 1))
        else:
            return [int(tp_string.replace("T", ""))]

def overlay_single_timepoint(args):
    """Process a single timepoint - standalone function for parallel execution"""
    well, timepoint, aligned_path, df_data, experiment_name, opt_params, overlay_root = args
    
    try:
        # Recreate pandas DataFrame from the passed data
        df = pd.DataFrame(df_data)
        
        # Filter tracking data for this specific timepoint
        df_filtered = df[
            (df['experiment'] == experiment_name) &
            (df['well'] == well) &
            (df['timepoint'] == timepoint)
        ]

        if df_filtered.empty:
            return f"{well} timepoint {timepoint} - no cells found"

        # Load and process image
        aligned_img = imageio.imread(aligned_path)
        img = aligned_img - np.min(aligned_img)
        img = np.float32((img / np.max(img) * 128 * opt_params['contrast']))
        img[img > 128] = 128
        img = np.uint8(img)

        # Create text overlay
        text_img = np.zeros_like(img)
        text_img = Image.fromarray(text_img)
        draw = ImageDraw.Draw(text_img)
        
        # Load font for this process
        font = ImageFont.truetype('/usr/share/fonts/dejavu/DejaVuSansMono.ttf', 50)

        for _, row in df_filtered.iterrows():
            cellid = int(row['tracked_id'])
            x = int(min(max(row['centroid_x'] + opt_params['shift'], 0), img.shape[1] - 20))
            y = int(min(max(row['centroid_y'] + opt_params['shift'], 0), img.shape[0] - 20))
            draw.text((x, y), str(cellid), (127), font)

        text_img = np.array(text_img)
        overlay_img = np.dstack([img, img + text_img, img])

        # Save overlay
        overlaydir = os.path.join(overlay_root, well)
        os.makedirs(overlaydir, exist_ok=True)

        outname = os.path.basename(aligned_path).replace('.tif', '_OVERLAY.png')
        outpath = os.path.join(overlaydir, outname)
        imageio.v3.imwrite(outpath, overlay_img)
        
        return f"{well} timepoint {timepoint} completed"
        
    except Exception as e:
        return f"{well} timepoint {timepoint} failed: {str(e)}"



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_name', required=True, help="Experiment name")
    parser.add_argument('--target_channel', required=True, help="Channel to use (e.g., Epi-GFP16)")
    parser.add_argument('--chosen_wells', default="all", help="Comma-separated wells (e.g., D3,E4) or 'all'")
    parser.add_argument('--chosen_timepoints', default="all", help="e.g. 'T0', 'T0-T2', or 'all'")
    parser.add_argument('--wells_toggle', default="include", choices=["include", "exclude"])
    parser.add_argument('--timepoints_toggle', default="include", choices=["include", "exclude"])
    parser.add_argument('--channels_toggle', default="include", choices=["include", "exclude"])
    parser.add_argument('--shift', default=20, type=int, help="Pixel shift for label placement")
    parser.add_argument('--contrast', default=1.3, type=float, help="Adjust contrast")
    args = parser.parse_args()

    Ovr = OverlayBatch(args)
    Ovr.run()

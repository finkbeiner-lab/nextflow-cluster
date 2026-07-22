#!/usr/bin/env python
"""Overlay tracked cell IDs onto aligned montage images.

Reads tracked-cell coordinates and IDs from a summary CSV produced by the
tracking pipeline, loads the corresponding montage images, and renders
white text labels with leader lines at each cell centroid.  Output images
are saved as grayscale PNGs.  Supports parallel processing across timepoints,
optional restriction to a subset of cell IDs (via a stable CSV), and
well/timepoint filtering.
"""

import imageio
import os
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from PIL import Image, ImageFont, ImageDraw
import numpy as np
import pandas as pd
import argparse
from sql import Database
import datetime
from time import time
from multiprocessing import Pool, cpu_count


class OverlayBatch:
    """Batch overlay of tracked cell IDs onto montaged microscopy images.

    Reads experiment metadata from the database, loads the tracking summary
    CSV, and writes overlay PNGs (white text on greyscale) for each
    well/timepoint combination.

    Attributes:
        experiment_name: Name of the experiment in the database.
        montage_root: Directory containing aligned or raw montage TIFFs.
        overlay_root: Output directory for overlay PNGs.
        df: Tracking summary DataFrame.
        max_cores: Number of worker processes for parallel overlay.
    """

    def __init__(self, opt: argparse.Namespace) -> None:
        """Initialise the overlay batch from command-line options.

        Args:
            opt: Parsed arguments including experiment name, well/timepoint
                filters, contrast, shift, and optional cell-ID restrictions.
        """
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

        # When filtering by cell IDs (set or CSV), write to Overlay_Montages_selected_ids so we don't overwrite the full overlay
        if getattr(self.opt, 'cell_ids', None) or getattr(self.opt, 'cell_ids_by_well_timepoint', None) is not None:
            self.overlay_root = os.path.join(self.experiment_root, "Overlay_Montages_selected_ids")
            print(f"[INFO] Selected cell IDs (or stable CSV) provided → output: {self.overlay_root}")
        else:
            self.overlay_root = os.path.join(self.experiment_root, "Overlay_Montages")

        self.font = ImageFont.truetype('/usr/share/fonts/dejavu/DejaVuSansMono.ttf', 50)

        summary_path = os.path.join(
            self.experiment_root,
            f"{self.experiment_name}_tracked_montage_summary.csv"
        )
        if not os.path.exists(summary_path):
            raise FileNotFoundError(f"Tracking CSV not found: {summary_path}")
        self.df = pd.read_csv(summary_path)
        n_before = len(self.df)
        self.df['tracked_id'] = pd.to_numeric(self.df['tracked_id'], errors='coerce')
        self.df['timepoint']  = pd.to_numeric(self.df['timepoint'],  errors='coerce')
        self.df = self.df.dropna(subset=['tracked_id', 'timepoint']).copy()
        self.df['tracked_id'] = self.df['tracked_id'].astype(int)
        self.df['timepoint']  = self.df['timepoint'].astype(int)
        n_dropped = n_before - len(self.df)
        if n_dropped:
            print(f"[WARN] Dropped {n_dropped} malformed row(s) from tracking summary "
                  f"(non-numeric tracked_id/timepoint) — likely column-shift corruption")
        print(f"[INFO] Loaded tracking summary: {summary_path} ({len(self.df)} rows)")
        
        # Dynamic core allocation (75% of available cores)
        self.max_cores = max(1, int(cpu_count() * 0.75))
        print(f"[INFO] Using {self.max_cores} cores for parallel processing")
        
        # Performance monitoring
        self.start_time = None

    def run(self) -> None:
        """Generate overlay images for all matching wells and timepoints.

        Distributes work across ``self.max_cores`` processes when there are
        multiple timepoints to render.
        """
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
        
        # Prepare data for multiprocessing (avoid pickle issues).
        # Only pass each task the tracking rows for its own well: previously the
        # ENTIRE multi-well summary DataFrame was embedded into every per-timepoint
        # task tuple and re-pickled per timepoint. Grouping by well up front means
        # only the relevant rows are serialized. overlay_single_timepoint still
        # filters by well/timepoint, so the overlay output is unchanged.
        exp_df = self.df[self.df['experiment'] == self.experiment_name]
        df_records_by_well = {
            str(well): group.to_dict('records')
            for well, group in exp_df.groupby('well')
        }
        opt_params = {
            'contrast': self.opt.contrast,
            'shift': self.opt.shift,
            'cell_ids': getattr(self.opt, 'cell_ids', None),
            'cell_ids_by_well_timepoint': getattr(self.opt, 'cell_ids_by_well_timepoint', None)
        }

        # Prepare tasks with serializable data
        mp_tasks = []
        for well, timepoint, aligned_path in timepoint_tasks:
            df_data = df_records_by_well.get(str(well), [])
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

    def extract_timepoint(self, filename: str) -> int:
        """Parse the integer timepoint from a montage filename.

        Expects a ``T<number>`` token separated by underscores (e.g.
        ``Well_A1_T3_Cy5_MONTAGE_ALIGNED.tif``).

        Args:
            filename: Basename of the montage image file.

        Returns:
            Integer timepoint index.

        Raises:
            ValueError: If no ``T<digits>`` token is found.
        """
        parts = filename.split('_')
        for part in parts:
            if part.startswith('T') and part[1:].isdigit():
                return int(part[1:])
        raise ValueError(f"Could not find timepoint in filename: {filename}")

    def parse_timepoints(self, tp_string: str) -> List[int]:
        """Convert a timepoint specification string to a list of integers.

        Args:
            tp_string: ``'all'``, a single value like ``'T3'``, or a range
                like ``'T0-T7'``.

        Returns:
            List of integer timepoint indices.
        """
        if tp_string == "all":
            # Use every timepoint actually present in the tracked-cell summary.
            # The previous `range(100)` capped experiments at T99 silently.
            return sorted(self.df['timepoint'].astype(int).unique().tolist())
        elif "-" in tp_string:
            start, end = map(int, tp_string.replace("T", "").split("-"))
            return list(range(start, end + 1))
        else:
            return [int(tp_string.replace("T", ""))]

def load_stable_csv(csv_path: str) -> Dict[Tuple[str, int], Set[int]]:
    """Load a stable-cell CSV and group tracked IDs by (well, timepoint).

    The CSV must contain the columns ``well``, ``tracked_id``, and
    ``timepoint``.

    Args:
        csv_path: Path to the CSV file.

    Returns:
        Mapping of ``(well, timepoint)`` to the set of tracked cell IDs
        present in that combination.

    Raises:
        ValueError: If required columns are missing.
    """
    df = pd.read_csv(csv_path)
    required = {'well', 'tracked_id', 'timepoint'}
    if not required.issubset(df.columns):
        raise ValueError(f"Stable CSV must have columns {required}; got {list(df.columns)}")
    df['timepoint'] = df['timepoint'].astype(int)
    out = {}
    for (well, tp), g in df.groupby(['well', 'timepoint']):
        out[(str(well).strip(), int(tp))] = set(g['tracked_id'].astype(int))
    return out


def overlay_single_timepoint(
    args: Tuple[str, int, str, List[Dict], str, Dict[str, Any], str],
) -> str:
    """Render an overlay image for a single well/timepoint combination.

    This is a module-level function (rather than a method) so it can be
    dispatched by ``multiprocessing.Pool.map`` without pickling issues.

    Args:
        args: A 7-tuple of ``(well, timepoint, aligned_image_path,
            tracking_df_as_dicts, experiment_name, opt_params_dict,
            overlay_output_root)``.

    Returns:
        A status string describing success or failure for logging.
    """
    well, timepoint, aligned_path, df_data, experiment_name, opt_params, overlay_root = args
    
    try:
        # Recreate pandas DataFrame from the passed data
        df = pd.DataFrame(df_data)
        if df.empty:
            return f"{well} timepoint {timepoint} - skipped: no cells in tracking summary (data missing?)"
        # Belt-and-suspenders: enforce int dtype so .isin() against int stable-id set matches.
        # Pool serialization can widen dtypes; a single stray string in an upstream write
        # would otherwise cause a silent zero-match filter here.
        df['tracked_id'] = pd.to_numeric(df['tracked_id'], errors='coerce').astype('Int64')
        df = df.dropna(subset=['tracked_id']).copy()
        df['tracked_id'] = df['tracked_id'].astype(int)

        # Filter tracking data for this specific timepoint
        df_filtered = df[
            (df['experiment'] == experiment_name) &
            (df['well'] == well) &
            (df['timepoint'] == timepoint)
        ]

        # Optionally restrict overlay: stable CSV (well, timepoint) -> ids, or global set of ids
        cell_ids_by_well_timepoint = opt_params.get('cell_ids_by_well_timepoint')
        cell_ids = opt_params.get('cell_ids')
        if cell_ids_by_well_timepoint is not None:
            ids_for_this = cell_ids_by_well_timepoint.get((well, timepoint))
            if ids_for_this is not None:
                df_filtered = df_filtered[df_filtered['tracked_id'].isin(ids_for_this)]
            else:
                df_filtered = df_filtered.iloc[0:0]  # empty: no rows for this (well, timepoint) in CSV
        elif cell_ids is not None:
            df_filtered = df_filtered[df_filtered['tracked_id'].isin(cell_ids)]

        if df_filtered.empty:
            return f"{well} timepoint {timepoint} - skipped: no stable/selected cells match at this timepoint"

        # Load and process image
        aligned_img = imageio.imread(aligned_path)
        img = aligned_img - np.min(aligned_img)
        img = np.float32((img / np.max(img) * 128 * opt_params['contrast']))
        img[img > 128] = 128
        img = np.uint8(img)

        # Create text + leader-line overlay (single channel, 255 = white in final grayscale)
        text_img = np.zeros_like(img, dtype=np.uint8)
        text_pil = Image.fromarray(text_img)
        draw = ImageDraw.Draw(text_pil)
        font = ImageFont.truetype('/usr/share/fonts/dejavu/DejaVuSansMono.ttf', 50)

        line_width = max(4, min(8, max(img.shape) // 500))  # 4–8 px so the line is clearly visible
        for _, row in df_filtered.iterrows():
            cellid = int(row['tracked_id'])
            cx, cy = int(row['centroid_x']), int(row['centroid_y'])
            x = int(min(max(cx + opt_params['shift'], 0), img.shape[1] - 20))
            y = int(min(max(cy + opt_params['shift'], 0), img.shape[0] - 20))
            # Leader line from cell centroid to ID label
            draw.line((cx, cy, x, y), fill=255, width=line_width)
            # Small circle at cell so the line clearly starts at the cell
            r = max(3, line_width + 2)
            draw.ellipse((cx - r, cy - r, cx + r, cy + r), outline=255, width=2)
            draw.text((x, y), str(cellid), fill=255, font=font)

        text_img = np.array(text_pil)
        # Merge text/lines (white) onto the grayscale base image
        overlay_img = np.clip(img.astype(np.int32) + text_img, 0, 255).astype(np.uint8)

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
    parser.add_argument(
        '--cell_ids',
        default=None,
        help="Path to stable CSV (columns: well, tracked_id, timepoint) OR comma-separated cell IDs. Omit or 'all' = show all."
    )
    args = parser.parse_args()

    # Resolve cell_ids: stable CSV path -> (well, timepoint) -> set(ids); else comma-separated IDs -> set(ids)
    raw = (args.cell_ids or '').strip().strip("'\"")
    args.cell_ids_by_well_timepoint = None
    if not raw or raw.lower() == 'all':
        args.cell_ids = None
    elif os.path.isfile(raw) or raw.endswith('.csv'):
        path = raw if os.path.isfile(raw) else raw
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Cell IDs CSV not found: {path}")
        args.cell_ids_by_well_timepoint = load_stable_csv(path)
        args.cell_ids = None
        print(f"[INFO] Loaded stable CSV: {path} → {len(args.cell_ids_by_well_timepoint)} (well, timepoint) groups")
    else:
        ids = set()
        for x in raw.split(','):
            s = x.strip().strip("'\"")
            if s:
                ids.add(int(s))
        args.cell_ids = ids
        args.cell_ids_by_well_timepoint = None

    Ovr = OverlayBatch(args)
    Ovr.run()

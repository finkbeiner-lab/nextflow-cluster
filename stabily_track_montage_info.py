#!/usr/bin/env python3
"""
Mark tracked cells as stable or not, and record maximum centroid displacement.
Adapted for montage CSV format with different column names.
"""

import pandas as pd
import numpy as np
import argparse

def mark_stable_cells(input_csv, output_csv, threshold=100):
    # Load CSV
    df = pd.read_csv(input_csv)

    # Number of unique timepoints in experiment
    timepoints = df["timepoint"].unique()
    n_timepoints = len(timepoints)

    stable_flags = {}
    max_disp_values = {}

    # Loop over each (well, tracked_id) - using tracked_id instead of cellid
    # Note: removed tile grouping since it's not in the new format
    for (well, tracked_id), group in df.groupby(["well", "tracked_id"]):
        stable = False
        max_disp = np.nan

        if group["timepoint"].nunique() == n_timepoints:
            # Sort by timepoint for displacement calculation
            g = group.sort_values("timepoint")
            coords = g[["centroid_x", "centroid_y"]].values

            if len(coords) > 1:
                # Compute Euclidean displacements
                displacements = np.sqrt(np.sum(np.diff(coords, axis=0) ** 2, axis=1))
                max_disp = float(np.max(displacements))

                # Check stability
                if np.all(displacements < threshold):
                    stable = True

        stable_flags[(well, tracked_id)] = stable
        max_disp_values[(well, tracked_id)] = max_disp

    # Map results back to dataframe
    df["stably_tracked"] = df.apply(
        lambda row: stable_flags.get((row["well"], row["tracked_id"]), False),
        axis=1
    )
    df["max_displacement"] = df.apply(
        lambda row: max_disp_values.get((row["well"], row["tracked_id"]), np.nan),
        axis=1
    )

    # Save results
    df.to_csv(output_csv, index=False)
    print(f"Saved results with stability flags to {output_csv}")


if __name__ == "__main__":
    # Example usage - update paths as needed
    input_csv = "/gladstone/finkbeiner/kaye/KanchanSarda/GXYTMP/Nextflow/GXYTMP-ALS-Set55/ALS-Set55-04172025TDP43EOS-JAK_tracked_montage_summary.csv"
    output_csv = "/gladstone/finkbeiner/kaye/KanchanSarda/GXYTMP/Nextflow/GXYTMP-ALS-Set55/ALS-Set55-04172025TDP43EOS-JAK_tracked_stable.csv"
    threshold = 100  # pixels

    mark_stable_cells(input_csv, output_csv, threshold)

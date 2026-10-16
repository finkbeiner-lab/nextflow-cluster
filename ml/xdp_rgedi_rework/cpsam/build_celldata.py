#!/usr/bin/env python3
"""Turn ``cpsam_tiles.py`` per-cell files into a Galaxy-format ``cell_data.csv``.

The output has the legacy Galaxy "Blob" long layout that MRID's ``mrid_ratio()``
(``steps/02b_mrid_ratio_rebuild.R``) reads: two rows per cell, MeasurementTag
``FITC`` and ``RFP``, sharing ``ObjectLabelsFound`` within a (well, timepoint).
Intensities are from the Galaxy background-corrected tiles.

Cells with BG FITC mean == 0 are dropped (the GEDI ratio RFP/FITC is undefined);
they are counted in ``qc/dropped_zero_fitc.csv``.

QC written to ``<out-dir>/qc/``:

* ``counts_well_tp.csv``: cells per well x timepoint (+ Galaxy counts when
  ``--galaxy-celldata`` is given).
* ``completeness.csv``: manifest (well, tp) pairs with no percell file.
* ``dropped_zero_fitc.csv``.
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import pandas as pd

GALAXY_COLS = [
    "ObjectCount", "ObjectLabelsFound", "MeasurementTag", "BlobArea", "BlobCentroidX",
    "BlobCentroidY", "PixelIntensityMaximum", "PixelIntensityMinimum", "PixelIntensityMean",
    "PixelIntensityStdDev", "PixelIntensity50Percentile", "Sci_PlateID", "Sci_WellID",
    "RowID", "ColumnID", "Timepoint", "ElapsedHours",
]


def galaxy_rows(pc: pd.DataFrame, plate_id: str, channel: str) -> pd.DataFrame:
    """One Galaxy-format row per cell for one channel.

    Args:
        pc: Per-cell table from ``cpsam_tiles.py``.
        plate_id: ``Sci_PlateID`` value (Galaxy uses ``PID<date>_<EXP>-RGEDI``).
        channel: ``"FITC"`` or ``"RFP"``.

    Returns:
        Rows in ``GALAXY_COLS`` order.
    """
    p = "fitc" if channel == "FITC" else "rfp"
    n_obj = pc.groupby(["well", "tp"])["label"].transform("size")
    return pd.DataFrame({
        "ObjectCount": n_obj.values,
        "ObjectLabelsFound": pc["label"].astype(int).values,
        "MeasurementTag": channel,
        "BlobArea": pc["area"].values,
        "BlobCentroidX": pc["cx"].round(3).values,
        "BlobCentroidY": pc["cy"].round(3).values,
        "PixelIntensityMaximum": pc[f"{p}_max"].values,
        "PixelIntensityMinimum": pc[f"{p}_min"].values,
        "PixelIntensityMean": pc[f"{p}_mean"].values,
        "PixelIntensityStdDev": pc[f"{p}_std"].values,
        "PixelIntensity50Percentile": pc[f"{p}_median"].values,
        "Sci_PlateID": plate_id,
        "Sci_WellID": pc["well"].values,
        "RowID": pc["well"].str[0].values,
        "ColumnID": pc["well"].str[1:].astype(int).values,
        "Timepoint": pc["tp"].astype(int).values,
        "ElapsedHours": "",
    })[GALAXY_COLS]


def galaxy_counts(path: str) -> pd.DataFrame:
    """Galaxy cells per well x timepoint (FITC rows, after MRID's dedup key)."""
    g = pd.read_csv(path, usecols=["ObjectLabelsFound", "MeasurementTag", "Sci_WellID", "Timepoint"])
    g = g[g["MeasurementTag"].str.contains("FITC|GFP", regex=True)]
    g = g.drop_duplicates(["Sci_WellID", "Timepoint", "ObjectLabelsFound"])
    return (g.groupby(["Sci_WellID", "Timepoint"]).size().rename("n_galaxy").reset_index()
            .rename(columns={"Sci_WellID": "well", "Timepoint": "tp"}))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seg-dir", required=True, help="cpsam_tiles.py --out-dir")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--plate-id", required=True, help="e.g. PID20250129_XDP10-RGEDI")
    ap.add_argument("--out", default="", help="default <seg-dir>/cell_data.csv")
    ap.add_argument("--galaxy-celldata", default="", help="Galaxy cell_data.csv for count QC")
    ap.add_argument("--allow-incomplete", action="store_true",
                    help="write cell_data even if manifest items are missing (pilot)")
    args = ap.parse_args()
    out = args.out or os.path.join(args.seg_dir, "cell_data.csv")
    qc = os.path.join(args.seg_dir, "qc")
    os.makedirs(qc, exist_ok=True)

    man = pd.read_csv(args.manifest)
    have = {os.path.basename(f)[:-len(".csv.gz")]
            for f in glob.glob(os.path.join(args.seg_dir, "percell", "*.csv.gz"))}
    man["key"] = man["well"] + "_" + man["timepoint"]
    missing = man[~man["key"].isin(have)]
    missing[["well", "timepoint"]].to_csv(os.path.join(qc, "completeness.csv"), index=False)
    print(f"manifest items {len(man)}; percell present {len(man) - len(missing)}; missing {len(missing)}")
    if len(missing) and not args.allow_incomplete:
        raise SystemExit("incomplete segmentation; see qc/completeness.csv")

    keys = man.loc[man["key"].isin(have), "key"]
    pc = pd.concat([pd.read_csv(os.path.join(args.seg_dir, "percell", f"{k}.csv.gz"))
                    for k in keys], ignore_index=True)
    dup = pc.duplicated(["well", "tp", "label"]).sum()
    if dup:
        raise SystemExit(f"{dup} duplicate (well, tp, label) rows")
    zero = pc["fitc_mean"] <= 0
    (pc[zero].groupby(["well", "timepoint"]).size().rename("n_dropped").reset_index()
     .to_csv(os.path.join(qc, "dropped_zero_fitc.csv"), index=False))
    print(f"cells {len(pc)}; dropped (BG FITC mean 0) {int(zero.sum())}")
    pc = pc[~zero]

    counts = (pc.groupby(["well", "tp", "cell_line", "condition"]).size().rename("n_cpsam")
              .reset_index())
    if args.galaxy_celldata:
        counts = counts.merge(galaxy_counts(args.galaxy_celldata), on=["well", "tp"], how="left")
        counts["ratio_cpsam_galaxy"] = counts["n_cpsam"] / counts["n_galaxy"]
    counts.to_csv(os.path.join(qc, "counts_well_tp.csv"), index=False)

    pc = pc.sort_values(["well", "tp", "label"])
    cd = pd.concat([galaxy_rows(pc, args.plate_id, "FITC"), galaxy_rows(pc, args.plate_id, "RFP")],
                   ignore_index=True)
    cd = cd.sort_values(["Sci_WellID", "Timepoint", "ObjectLabelsFound", "MeasurementTag"])
    tmp = out + ".tmp"
    cd.to_csv(tmp, index=False)
    os.replace(tmp, out)
    print(f"wrote {out}: {len(cd)} rows ({len(pc)} cells x 2 channels)")
    if args.galaxy_celldata:
        print(counts.groupby("tp")[["n_cpsam", "n_galaxy"]].sum().assign(
            ratio=lambda d: (d.n_cpsam / d.n_galaxy).round(2)).to_string())


if __name__ == "__main__":
    main()

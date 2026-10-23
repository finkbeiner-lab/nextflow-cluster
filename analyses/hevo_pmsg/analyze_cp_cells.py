#!/usr/bin/env python
"""Per-cell RFP/GFP and RFP-alone readout from Cellpose-SAM cells, summarized per tile / well / condition.

Usage:
    python analyze_cp_cells.py <cells.csv> <fgbg tiles.csv> <out_dir>

Background per tile comes from the foreground/background run (fg_tile_readout.py): the median of the
tile's dish pixels, or, for tiles with <10% dish, the median over tiles of the same dose column and
timepoint that do have >=10% dish. Per cell:
    ratio      = rfp_mean / gfp_mean                                   (raw, as in the foreground readout)
    rfp_sub    = rfp_mean - red background                             (RFP alone, drift-corrected)
Per tile: median over its cells (+ 90th percentile of rfp_sub, which follows a bright minority).
Per well x timepoint: mean of the 9 tile values, band = IQR across tiles.
"""
import os
import sys

import numpy as np
import pandas as pd

import analyze_fg_tiles as A

A.METRICS.update({
    "ratio": "RFP / GFP per cell (median of cells)",
    "rfp_sub": "RFP − red background, per cell (median)",
    "rfp_sub_p90": "RFP − red background, per cell (90th pct)",
    "n_cells_fc": "Cellpose cells per well, fold vs T0",
})


def main():
    cells_csv, tiles_csv, out = sys.argv[1:4]
    os.makedirs(out, exist_ok=True)
    lay = A.LAYOUT["hevo-pmsG-1"]
    c, t = pd.read_csv(cells_csv), pd.read_csv(tiles_csv)
    t["col"] = t.well.str[1:].astype(int)
    ref = t[(1 - t.fg_frac) >= 0.10].groupby(["col", "tp"])[["bg_rfp_median", "bg_gfp_median"]].median()
    ref.columns = ["ref_rfp", "ref_gfp"]
    t = t.join(ref, on=["col", "tp"])
    own = (1 - t.fg_frac) >= 0.10
    t["bg_rfp"] = np.where(own, t.bg_rfp_median, t.ref_rfp)
    t["bg_gfp"] = np.where(own, t.bg_gfp_median, t.ref_gfp)
    c = c.merge(t[["well", "tp", "tile", "bg_rfp", "bg_gfp"]], on=["well", "tp", "tile"], how="left")
    c["ratio"] = c.rfp_mean / c.gfp_mean
    c["rfp_sub"] = c.rfp_mean - c.bg_rfp
    c["row"], c["col"] = c.well.str[0], c.well.str[1:].astype(int)
    c["line"], c["dose_ms"] = c.row.map(lay["line"]), c.col.map(lay["dose_ms"])
    c["tpi"] = c.tp.str[1:].astype(int); c["hours"] = c.tpi * lay["hours_per_tp"]

    keys = ["well", "line", "dose_ms", "tp", "tpi", "hours"]
    tile = c.groupby(keys + ["tile"]).agg(ratio=("ratio", "median"), rfp_sub=("rfp_sub", "median"),
                                          rfp_sub_p90=("rfp_sub", lambda v: v.quantile(0.9)),
                                          n_cells=("label", "size")).reset_index()
    agg = {}
    for m in ["ratio", "rfp_sub", "rfp_sub_p90"]:
        agg[m] = (m, "mean")
        agg[f"{m}_q25"] = (m, lambda v: v.quantile(0.25)); agg[f"{m}_q75"] = (m, lambda v: v.quantile(0.75))
    agg["n_cells"] = ("n_cells", "sum"); agg["n_tiles"] = ("tile", "size")
    w = tile.groupby(keys).agg(**agg).reset_index().sort_values(["well", "tpi"])
    for m in ["ratio", "rfp_sub", "rfp_sub_p90", "n_cells"]:
        w[f"{m}_fc"] = w[m] / w.well.map(w[w.tpi == 0].set_index("well")[m])
    c.to_csv(f"{out}/cells_bg.csv.gz", index=False)
    tile.to_csv(f"{out}/tile_tp.csv", index=False)
    w.to_csv(f"{out}/well_tp.csv", index=False)

    colors = A.dose_colors(set(w.dose_ms))
    SRC = "Reference Cellpose-SAM cells, per-tile median of cells"
    A.fig_by_line(w, "ratio", False, colors, f"{out}/pmsG-1_cp_ratio.png", "pmsG-1 Cellpose cells: RFP/GFP", SRC)
    A.fig_by_line(w, "rfp_sub", False, colors, f"{out}/pmsG-1_cp_rfp.png", "pmsG-1 Cellpose cells: RFP alone (median cell)", SRC)
    A.fig_by_line(w, "rfp_sub_p90", False, colors, f"{out}/pmsG-1_cp_rfp_p90.png", "pmsG-1 Cellpose cells: RFP alone (brightest 10% of cells)", SRC)
    w["n_cells_fc_q25"] = w["n_cells_fc_q75"] = np.nan
    A.fig_by_line(w.assign(n_cells_fc_fc=w.n_cells_fc), "n_cells_fc", False, colors, f"{out}/pmsG-1_cp_counts.png",
                  "pmsG-1 Cellpose cells: cell count vs T0", SRC)
    print(f"{len(c)} cells, {c.groupby(['well', 'tp']).ngroups} montages; "
          f"{int(c.bg_rfp.isna().sum())} cells without a tile background")


if __name__ == "__main__":
    main()

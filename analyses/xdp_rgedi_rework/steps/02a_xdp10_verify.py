"""Step 02a: verify the XDP10 re-extract against the truncated original cell_data.csv.

The original file stops mid-row. We parse it up to its last complete line, treat
the last (well, timepoint, channel) group as possibly incomplete, and require the
re-extract to contain IDENTICAL rows (same multiset of rows on every shared
column) for every other group. We then check that the re-extract covers the
full plate grid (rows D-M x cols 2-22 x T1-T10 x FITC/RFP).
"""

import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from lib import cfg, data  # noqa: E402
from lib.paths import LOCAL_OUT, RAW_CACHE  # noqa: E402
from lib.provenance import StepRecorder  # noqa: E402

STEP = "02a_xdp10_verify"
GROUP = ["Sci_WellID", "Timepoint", "MeasurementTag"]


def read_truncated(path: Path):
    raw = path.read_bytes()
    cut = raw.rfind(b"\n")
    tail = raw[cut + 1:].decode(errors="replace")
    df = pd.read_csv(io.BytesIO(raw[:cut + 1]), low_memory=False)
    return df, {"bytes": len(raw), "bytes_mod_4096": len(raw) % 4096, "ends_with_newline": raw.endswith(b"\n"),
                "partial_last_line": tail, "rows_parsed": len(df)}


def row_hashes(df: pd.DataFrame, cols) -> pd.Series:
    return pd.util.hash_pandas_object(df[cols], index=False)


def main() -> None:
    rec = StepRecorder(STEP, LOCAL_OUT / STEP)
    staging = json.loads((RAW_CACHE / "staging_manifest.json").read_text())
    p_new = data.staged_path("XDP10_rerun_raw")
    p_old = data.staged_path("XDP10_trunc_raw")
    rec.add_input(p_new, staging["XDP10_rerun_raw"]["sha256"])
    rec.add_input(p_old, staging["XDP10_trunc_raw"]["sha256"])

    old, trunc = read_truncated(p_old)
    new = pd.read_csv(p_new, low_memory=False)
    trunc["last_complete_row"] = old.iloc[-1][GROUP].astype(str).to_dict()
    rec.summary["truncation"] = trunc
    rec.check("XDP10.original_truncated", "Original cell_data.csv ends mid-row (no final newline)",
              not trunc["ends_with_newline"], {k: trunc[k] for k in ("bytes", "bytes_mod_4096", "partial_last_line")})

    shared = [c for c in old.columns if c in new.columns]
    rec.summary["columns"] = {"original": list(old.columns), "reextract": list(new.columns),
                              "only_in_reextract": [c for c in new.columns if c not in old.columns],
                              "only_in_original": [c for c in old.columns if c not in new.columns]}
    for df in (old, new):
        df["Sci_WellID"] = df["Sci_WellID"].astype(str).map(cfg.pad_well)

    # Groups present in the original; drop the last one (may be cut off).
    last_key = tuple(old.iloc[-1][GROUP])
    old["_h"] = row_hashes(old, shared)
    new["_h"] = row_hashes(new, shared)
    g_old = old.groupby(GROUP)["_h"].apply(lambda s: np.sort(s.values))
    g_new = new.groupby(GROUP)["_h"].apply(lambda s: np.sort(s.values))
    complete = [k for k in g_old.index if k != last_key]
    n_identical, mismatches = 0, []
    for k in complete:
        a = g_old.loc[k]
        b = g_new.loc[k] if k in g_new.index else np.array([], dtype=a.dtype)
        if len(a) == len(b) and np.array_equal(a, b):
            n_identical += 1
        else:
            mismatches.append({"group": list(map(str, k)), "rows_original": int(len(a)), "rows_reextract": int(len(b)),
                               "rows_in_common": int(len(np.intersect1d(a, b)))})
    last_a = g_old.loc[last_key]
    last_b = g_new.loc[last_key] if last_key in g_new.index else np.array([])
    rec.summary["overlap"] = {
        "groups_in_original": int(len(g_old)), "groups_compared": len(complete),
        "groups_identical": n_identical, "groups_mismatched": len(mismatches),
        "rows_compared": int(sum(len(g_old.loc[k]) for k in complete)),
        "last_group": {"key": list(map(str, last_key)), "rows_original": int(len(last_a)),
                       "rows_reextract": int(len(last_b)),
                       "original_rows_found_in_reextract": int(np.isin(last_a, last_b).sum())},
    }
    pd.DataFrame(mismatches).to_csv(rec.out_dir / "mismatched_groups.csv", index=False)
    rec.check("XDP10.overlap_identical",
              "Every complete (well, timepoint, channel) group of the original is reproduced row-for-row "
              "(all shared columns identical) in the re-extract", len(mismatches) == 0,
              {"compared": len(complete), "identical": n_identical, "mismatched": len(mismatches)})

    # Grid completeness of the re-extract.
    roles = cfg.well_roles("XDP10")
    tps = cfg.read_timepoints("XDP10").tp.tolist()
    have = new.groupby(GROUP).size().rename("n").reset_index()
    grid = pd.MultiIndex.from_product([roles.well, tps, ["FITC", "RFP"]], names=GROUP).to_frame(index=False)
    grid = grid.merge(have, on=GROUP, how="left")
    missing = grid[grid.n.isna()]
    extra_wells = sorted(set(new.Sci_WellID) - set(roles.well))
    rec.check("XDP10.reextract_complete_grid",
              "Re-extract has cells for every plate well (D-M x 2-22) x T1-T10 x FITC/RFP",
              len(missing) == 0, {"missing_groups": missing[GROUP].astype(str).values.tolist()[:40],
                                  "n_missing": len(missing)})
    rec.check("XDP10.reextract_extra_wells", "Wells in the re-extract that are not on the plate map",
              len(extra_wells) == 0, extra_wells)

    # What the original was missing (evidence for the document).
    old_groups = set(g_old.index)
    lost = grid[[tuple(r) not in old_groups for r in grid[GROUP].itertuples(index=False)]]
    lost_summary = (lost.assign(row=lost.Sci_WellID.str[0])
                    .groupby(["row", "MeasurementTag"]).size().rename("groups").reset_index())
    lost.to_csv(rec.out_dir / "groups_missing_from_original.csv", index=False)
    rec.summary["missing_from_original"] = {
        "groups": len(lost), "wells_affected": sorted(lost.Sci_WellID.unique().tolist()),
        "by_row_channel": lost_summary.to_dict("records")}
    rec.summary["reextract"] = {"rows": len(new), "wells": int(new.Sci_WellID.nunique()),
                                "timepoints": sorted(new.Timepoint.unique().tolist()),
                                "channels": sorted(new.MeasurementTag.unique().tolist())}
    rec.write()
    print(json.dumps(rec.summary["overlap"], indent=1))
    print(json.dumps(rec.summary["missing_from_original"]["by_row_channel"])[:800])
    for c in rec.checks:
        print(c["status"], c["id"], str(c["value"])[:300])


if __name__ == "__main__":
    main()

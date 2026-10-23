# Run 3 recipe (after `xdp10_cpsam_post` has written `cpsam-seg/XDP10/cell_data.csv`)

Run 3 = run 2 with only XDP10's per-cell input replaced (decision D29). Run on the Mac. The share must be mounted.

```bash
cd ~/Desktop/nextflow-cluster/analyses/xdp_rgedi_rework
export XDP_RGEDI_RUN=run3_cpsam-xdp10_<YYYYMMDD>
export XDP_RGEDI_EXP_OVERRIDE=overrides/run3_cpsam_xdp10.yaml
export XDP_RGEDI_PARQUET=/Users/aholub/xdp_rgedi_cache/parquet_$XDP_RGEDI_RUN
PY=/Users/aholub/anaconda3/bin/python3
```

1. Add the staging key now that the file exists. It was left out until then so that a full `00_stage` in another
   session cannot fail on a missing file. Add this line to `config/staging.yaml` under `large:`:
   `XDP10_cpsam_raw: RGEDI-full-rework/cpsam-seg/XDP10/cell_data.csv`
   Then run `$PY steps/00_stage.py --only XDP10_cpsam_raw`.
2. Set up the run folder and parquet folder:
   - `mkdir -p $XDP_RGEDI_PARQUET`
   - copy the five unchanged parquets: `cp ../../../xdp_rgedi_cache/parquet/XDP{1,3,5,8,11}.parquet $XDP_RGEDI_PARQUET/`
     (or use absolute paths);
   - copy `01_metadata/` from run 2, which includes the hand-written `platemap_mrid_XDP10.csv`.
3. Rebuild the ratios with MRID's own `mrid_ratio()`. Leave tp_removal empty:
   ```bash
   Rscript --vanilla steps/02b_mrid_ratio_rebuild.R <raw cache>/RGEDI-full-rework/cpsam-seg/XDP10/cell_data.csv \
       <run3>/01_metadata/platemap_mrid_XDP10.csv <run3>/02b_rebuild/XDP10 "" \
       <raw cache>/MRID-rework/MRID-code/MRID_Ratio_Functions.R
   ```
4. `$PY steps/01b_extract.py XDP10`. Its integrity checks must pass: unique (tp, well, obj), ratio == rfp/gfp, wells
   mapped.
5. Run the remaining steps in run 2's order (see `out/RUNS.md`): 03 → 04 → 05/05b/05c/05d → 06 → 06b → 08 → 07 →
   09 → 10a → 11 → 12 → 12b → 13 → 14 → 15 → 16 → 16b → 17 → 10/10b. Skip 02a and 02c, which are tied to the
   Galaxy files.
6. Check that the non-XDP10 thresholds and MRID inputs match run 2. Then rsync the run to
   `RGEDI-full-rework/$XDP_RGEDI_RUN/` and add a run-3 row to `out/RUNS.md`.
7. Compare XDP10 in run 2 vs run 3: counts, S.A. separation, cutoffs, death fraction, ORs, capture fraction.

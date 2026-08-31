#!/opt/conda/bin/python
"""MINISOG: miniSOG-RGEDI death-signal readout and RFP-vs-NarrowRFP evaluation.

Image-free downstream analysis module. It does **not** measure pixels itself —
that is done upstream by ``bin/intensity.py`` (INTENSITY), which writes per-cell
intensities into ``intensitycelldata`` for each candidate red channel projected
onto the RGEDI-green (``Epi-GFP16``) morphology mask.

For the miniSOG-RGEDI design, blue-light stimulation drives a cumulative death
signal read out in a red channel. Each timepoint acquires a **pre**-stim and a
**post**-stim (``-2``) red image for two candidate sensors:

    RFP16      -> post ``Epi-RFP16-2``,     baseline ``Epi-RFP16`` @T0
    NarrowRFP  -> post ``Epi-NarrowRFP-2``, baseline ``Epi-NarrowRFP`` @T0

The meaningful death accumulates between timepoints, so the death trajectory is
the **post (`-2`) intensity series across all timepoints**, with the **T0 pre
read as the absolute baseline** (before any stimulation).

This module reconstructs a per-track post-stim trajectory (joining
``intensitycelldata`` -> ``celldata.cellid`` -> ``tiledata.timepoint/hours``),
computes continuous per-track features, and compares the two sensors using the
plate's dose-response (blue-light duration by column) and time-course structure
as ground truth. Which sensor gives the *cleaner* death signal is decided by a
composite ``quality_score``. Results go to ``minisogtrackdata`` and
``minisogcomparisondata`` plus CSVs/plots under ``analysisdir/MINISOG/``.

Requires TRACKING to have populated ``celldata.cellid`` and INTENSITY to have
written both sensors' pre and post channels.
"""

import argparse
import datetime
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from db_util import Ops
from sql import Database
import utils as utils

logger = logging.getLogger("MiniSOG")
now = datetime.datetime.now()
TIMESTAMP = '%d%02d%02d%02d%02d' % (now.year, now.month, now.day, now.hour, now.minute)
fink_log_dir = './finkbeiner_logs'
if not os.path.exists(fink_log_dir):
    os.makedirs(fink_log_dir)
fh = logging.FileHandler(os.path.join(fink_log_dir, f'MiniSOG-log_{TIMESTAMP}.log'))
fh.setLevel(20)
logger.addHandler(fh)
logger.warning('Running MiniSOG from Database.')


def _native(value: Any) -> Any:
    """Coerce numpy scalars / NaN to native Python types for DB insertion.

    This branch's ``Database.add_row`` inserts values verbatim; numpy 2.0
    scalars injected into SQL raise, and numpy NaN must become ``None`` for a
    nullable column. Returns a plain ``int``/``float``/``str``/``None``.

    Args:
        value: Any scalar (numpy or native) or None.

    Returns:
        A JSON/SQL-friendly native Python scalar, or ``None`` for NaN/None.
    """
    if value is None:
        return None
    if isinstance(value, (np.generic,)):
        value = value.item()
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        return None
    return value


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation without a scipy dependency.

    Args:
        x: First sample.
        y: Second sample (same length as ``x``).

    Returns:
        Spearman rho, or ``nan`` if fewer than 3 finite pairs or no variance.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return float('nan')
    xr = pd.Series(x[mask]).rank().values
    yr = pd.Series(y[mask]).rank().values
    if np.std(xr) == 0 or np.std(yr) == 0:
        return float('nan')
    return float(np.corrcoef(xr, yr)[0, 1])


def _slope(x: np.ndarray, y: np.ndarray) -> float:
    """Ordinary-least-squares slope of ``y`` vs ``x``.

    Args:
        x: Independent variable.
        y: Dependent variable.

    Returns:
        The slope, or ``nan`` if fewer than 2 finite pairs or no x-variance.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2 or np.std(x[mask]) == 0:
        return float('nan')
    return float(np.polyfit(x[mask], y[mask], 1)[0])


class MiniSOG:
    """Reconstruct per-track death trajectories and compare red sensors.

    Args:
        opt: Configuration namespace (argparse or Nextflow params). Expected
            attributes: ``experiment``, ``sensors``, ``baseline_timepoint``,
            ``min_track_len``, ``chosen_wells``/``wells_toggle``,
            ``chosen_timepoints``/``timepoints_toggle``, and the ``Ops`` toggle
            attributes.
    """

    def __init__(self, opt: Any) -> None:
        self.opt = opt
        # Ops needs these attributes; default any the launcher didn't pass.
        for attr, default in (('chosen_channels', 'all'), ('channels_toggle', 'include'), ('tile', 0)):
            if not hasattr(opt, attr):
                setattr(opt, attr, default)
        self.Op = Ops(opt)
        self.Db = Database()
        self.sensors: List[Tuple[str, str, str]] = self._parse_sensors(opt.sensors)
        self.baseline_tp: int = int(opt.baseline_timepoint)
        self.min_track_len: int = int(opt.min_track_len)
        self.exp_uuid = self.Db.get_table_uuid('experimentdata', dict(experiment=opt.experiment))
        if self.exp_uuid is None:
            raise Exception(f'Experiment not found in database: {opt.experiment}')
        analysisdir = self.Db.get_table_value('experimentdata', 'analysisdir', dict(id=self.exp_uuid))
        self.analysisdir = analysisdir[0][0] if analysisdir else None
        self.plotdir = os.path.join(self.analysisdir, 'MINISOG') if self.analysisdir else None
        # Montage flow writes per-track red intensities to this CSV (not the DB).
        self.tracking_csv = getattr(opt, 'tracking_csv', '') or (
            os.path.join(self.analysisdir, f'{opt.experiment}_tracked_montage_summary.csv')
            if self.analysisdir else '')

    @staticmethod
    def _parse_sensors(spec: str) -> List[Tuple[str, str, str]]:
        """Parse the ``--sensors`` spec 'name:post:baseline,name:post:baseline'.

        Args:
            spec: Comma-separated sensor definitions, each ``name:post:baseline``.

        Returns:
            List of (sensor_name, post_channel, baseline_channel) tuples.

        Raises:
            ValueError: If a sensor definition is malformed.
        """
        sensors = []
        for part in spec.split(','):
            part = part.strip()
            if not part:
                continue
            fields = part.split(':')
            if len(fields) != 3:
                raise ValueError(
                    f"Bad --sensors entry {part!r}; expected 'name:post_channel:baseline_channel'")
            sensors.append((fields[0].strip(), fields[1].strip(), fields[2].strip()))
        if not sensors:
            raise ValueError('No sensors parsed from --sensors')
        return sensors

    def run(self) -> None:
        """Execute the full analysis: load, per-track metrics, comparison, plots."""
        merged = self.load_intensity_frame()
        tracks = []
        for name, post_ch, base_ch in self.sensors:
            per_track = self.compute_track_metrics(merged, name, post_ch, base_ch)
            if per_track.empty:
                logger.warning(f'No tracks for sensor {name} (post={post_ch}). Skipping.')
                print(f'WARNING: no tracks for sensor {name} (post channel {post_ch}).')
                continue
            self.write_track_metrics(per_track)
            tracks.append(per_track)
        if not tracks:
            raise Exception('No per-track metrics produced for any sensor. '
                            'Check that INTENSITY ran for the post channels and TRACKING set cellid.')
        tracks_df = pd.concat(tracks, ignore_index=True)
        comparison = self.compare_sensors(tracks_df)
        self.write_comparison(comparison)
        self.emit_plots(merged, tracks_df, comparison)
        self.export_csvs(tracks_df, comparison)
        print('MiniSOG done.')

    # Unified tidy schema produced by both loaders (one row per cell x channel x tp):
    _TIDY_COLS = ('welldata_id', 'well', 'cellid', 'tile', 'channel', 'channeldata_id',
                  'timepoint', 'hours', 'celltype', 'condition', 'dosage', 'readout')

    def load_intensity_frame(self) -> pd.DataFrame:
        """Load per-track red intensities from the configured source.

        Two upstream flows store the red-channel per-cell intensities differently:

        * **tile flow** (Cellpose/threshold tile segmentation -> tile tracking ->
          ``intensity.py``): written to the DB ``intensitycelldata``, joined to a
          tracked ``celldata.cellid``. Source ``'db'``.
        * **montage flow** (segmentation_montage -> tracking_montage): written to
          ``<analysisdir>/<experiment>_tracked_montage_summary.csv``; the DB gets
          only the morphology channel. Source ``'csv'``.

        ``--intensity_source`` selects the source; ``'auto'`` tries the DB and
        falls back to the CSV. Both return the same tidy schema (:data:`_TIDY_COLS`).

        Returns:
            A tidy DataFrame with a ``readout`` column.
        """
        src = getattr(self.opt, 'intensity_source', 'auto')
        if src == 'csv':
            return self.load_from_csv()
        if src == 'db':
            return self.load_from_db()
        # auto: prefer the DB (canonical), fall back to the tracking CSV
        try:
            return self.load_from_db()
        except Exception as e:
            logger.warning(f'DB intensity source unavailable ({e}); falling back to tracking CSV.')
            print(f'DB intensity source unavailable ({e}); falling back to tracking CSV.')
            return self.load_from_csv()

    def load_from_db(self) -> pd.DataFrame:
        """Load red intensities from the DB ``intensitycelldata`` (tile flow).

        Joins ``intensitycelldata`` -> ``celldata`` (tracked ``cellid``) ->
        ``tiledata`` (``timepoint``/``hours``/``tile``) -> ``channeldata``
        (``channel``) -> ``welldata`` (+``dosagedata``). Track identity is
        ``(welldata_id, tile, cellid)``.

        Returns:
            Tidy DataFrame (:data:`_TIDY_COLS`).

        Raises:
            Exception: If intensitycelldata is empty, tracking never ran
                (``cellid`` all NULL), or none of the sensor post channels are
                present (i.e. reds were not measured into the DB — montage flow).
        """
        Db = self.Db
        exp = self.exp_uuid
        icd = Db.get_df_from_query('intensitycelldata', dict(experimentdata_id=exp))
        if icd.empty:
            raise Exception('intensitycelldata is empty for this experiment (run INTENSITY, tile flow).')
        cell = Db.get_df_from_query('celldata', dict(experimentdata_id=exp))
        if cell['cellid'].notna().sum() == 0:
            raise Exception('celldata.cellid is all NULL — TRACKING has not run.')
        tile = Db.get_df_from_query('tiledata', dict(experimentdata_id=exp))
        chan = Db.get_df_from_query('channeldata', dict(experimentdata_id=exp))
        well = Db.get_df_from_query('welldata', dict(experimentdata_id=exp))
        dose = Db.get_df_from_query('dosagedata', dict(experimentdata_id=exp))

        cell = cell[['id', 'cellid']].rename(columns={'id': 'celldata_id'})
        tile = tile[['id', 'timepoint', 'hours', 'tile']].rename(columns={'id': 'tiledata_id'})
        chan = chan[['id', 'channel']].rename(columns={'id': 'channeldata_id'})
        well = well[['id', 'well', 'celltype', 'condition']].rename(columns={'id': 'welldata_id'})

        df = icd.merge(cell, on='celldata_id', how='inner')
        df = df.merge(tile, on='tiledata_id', how='inner')
        df = df.merge(chan, on='channeldata_id', how='inner')
        df = df.merge(well, on='welldata_id', how='left')
        if not dose.empty:
            dose = dose[['welldata_id', 'dosage']].drop_duplicates('welldata_id')
            df = df.merge(dose, on='welldata_id', how='left')
        else:
            df['dosage'] = np.nan

        post_channels = {s[1] for s in self.sensors}
        if not (df['channel'].isin(post_channels)).any():
            raise Exception(f'None of the sensor post channels {sorted(post_channels)} are in '
                            'intensitycelldata (reds not measured into DB — montage flow?). '
                            'Use --intensity_source csv.')
        df = df[df['cellid'].notna()].copy()
        df['cellid'] = df['cellid'].astype(int)
        df['readout'] = df['intensity_mean'].astype(float)
        df = self._apply_filters(df)
        print(f'Loaded {len(df)} intensitycelldata rows (DB): {df.channel.nunique()} channels, '
              f'{df.well.nunique()} wells, {df.timepoint.nunique()} timepoints.')
        return df[list(self._TIDY_COLS)]

    def load_from_csv(self) -> pd.DataFrame:
        """Load per-track red intensities from the tracking-montage summary CSV.

        The montage pipeline writes red-channel per-cell intensities via
        ``tracking_montage.py`` to
        ``<analysisdir>/<experiment>_tracked_montage_summary.csv`` (columns
        ``well``, ``tracked_id``, ``MeasurementTag`` (=channel), ``Timepoint``,
        ``PixelIntensityMean``). Well metadata and ``timepoint``->``hours`` come
        from the DB. There is no tile dimension in the montage flow, so ``tile``
        is set to 0. Well/timepoint filters from ``self.opt`` are applied.

        Returns:
            Tidy DataFrame (:data:`_TIDY_COLS`).

        Raises:
            Exception: If the tracking CSV is missing or empty.
        """
        csv_path = self.tracking_csv
        if not csv_path or not os.path.exists(csv_path):
            raise Exception(
                f'Tracking-montage summary CSV not found: {csv_path!r}. Run the montage '
                'workflow (DO_STD_WORKFLOW) with the red channels in target_channel first.')
        raw = pd.read_csv(csv_path)
        if raw.empty:
            raise Exception(f'Tracking-montage summary CSV is empty: {csv_path}')
        colmap = {c.lower(): c for c in raw.columns}

        def col(*names: str) -> str:
            for n in names:
                if n.lower() in colmap:
                    return colmap[n.lower()]
            raise Exception(f'Tracking CSV missing expected column (any of {names}); '
                            f'has {list(raw.columns)}')

        df = pd.DataFrame({
            'well': raw[col('well', 'Sci_WellID')].astype(str),
            'cellid': raw[col('tracked_id', 'ObjectLabelsFound')].astype(int),
            'channel': raw[col('MeasurementTag')].astype(str),
            'timepoint': raw[col('timepoint', 'Timepoint')].astype(int),
            'readout': raw[col('PixelIntensityMean')].astype(float),
        })

        Db = self.Db
        exp = self.exp_uuid
        well = Db.get_df_from_query('welldata', dict(experimentdata_id=exp))[
            ['id', 'well', 'celltype', 'condition']].rename(columns={'id': 'welldata_id'})
        chan = Db.get_df_from_query('channeldata', dict(experimentdata_id=exp))[
            ['id', 'channel', 'welldata_id']].rename(columns={'id': 'channeldata_id'})
        tile = Db.get_df_from_query('tiledata', dict(experimentdata_id=exp))[
            ['timepoint', 'hours']].drop_duplicates('timepoint')
        dose = Db.get_df_from_query('dosagedata', dict(experimentdata_id=exp))

        df = df.merge(well, on='well', how='left')
        df = df.merge(chan, on=['channel', 'welldata_id'], how='left')
        df = df.merge(tile, on='timepoint', how='left')
        # fall back to 4h/timepoint spacing if hours not populated in DB
        if df['hours'].isna().any():
            df['hours'] = df['hours'].fillna(4.0 * df['timepoint'].astype(float))
        if not dose.empty:
            dose = dose[['welldata_id', 'dosage']].drop_duplicates('welldata_id')
            df = df.merge(dose, on='welldata_id', how='left')
        else:
            df['dosage'] = np.nan

        df['tile'] = 0  # montage tracked_id is per-well; no tile dimension
        df = self._apply_filters(df)
        print(f'Loaded {len(df)} tracked records: {df.channel.nunique()} channels, '
              f'{df.well.nunique()} wells, {df.timepoint.nunique()} timepoints from {csv_path}.')
        return df[list(self._TIDY_COLS)]

    def _apply_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply well/timepoint include-exclude filters from ``self.opt``."""
        cw = getattr(self.opt, 'chosen_wells', 'all')
        if cw and str(cw).lower() != 'all':
            wells = utils.get_iter_from_user(cw)
            df = Ops.filter_df(df, 'well', wells, self.opt.wells_toggle)
        ct = getattr(self.opt, 'chosen_timepoints', 'all')
        if ct and str(ct).lower() != 'all':
            tps = utils.get_iter_from_user(ct)
            if len(tps) and str(tps[0]).startswith('T'):
                tps = [t[1:] for t in tps]
            tps = [int(t) for t in tps if str(t).isnumeric()]
            df = Ops.filter_df(df, 'timepoint', tps, self.opt.timepoints_toggle)
        return df

    def compute_track_metrics(self, merged: pd.DataFrame, sensor: str,
                              post_ch: str, base_ch: str) -> pd.DataFrame:
        """Compute continuous per-track death features for one sensor.

        Args:
            merged: The tidy frame from :meth:`load_intensity_frame`.
            sensor: Sensor name (e.g. ``'RFP16'``).
            post_ch: Post-stim channel name (the death trajectory).
            base_ch: Pre-stim channel name (T0 provides the baseline).

        Returns:
            One row per (well, tile, cellid) track with the schema columns of
            ``minisogtrackdata`` plus helper columns (``well``, ``celltype``,
            ``dosage``) used by the comparison stage.
        """
        post = merged[merged['channel'] == post_ch].copy()
        if post.empty:
            return pd.DataFrame()
        max_tp = int(merged['timepoint'].max())

        # baseline = pre channel at the baseline timepoint, per track
        base = merged[(merged['channel'] == base_ch) & (merged['timepoint'] == self.baseline_tp)]
        base = base[['welldata_id', 'tile', 'cellid', 'readout']].rename(columns={'readout': 'baseline_t0'})
        base = base.drop_duplicates(['welldata_id', 'tile', 'cellid'])

        rows: List[Dict[str, Any]] = []
        for (wid, tile, cellid), g in post.groupby(['welldata_id', 'tile', 'cellid']):
            g = g.sort_values('timepoint')
            if len(g) < self.min_track_len:
                continue
            tps = g['timepoint'].values.astype(float)
            hrs = g['hours'].values.astype(float)
            y = g['readout'].values.astype(float)
            first = g.iloc[0]

            bmatch = base[(base['welldata_id'] == wid) & (base['tile'] == tile) & (base['cellid'] == cellid)]
            baseline_t0 = float(bmatch['baseline_t0'].iloc[0]) if len(bmatch) else float(np.nanmin(y))

            early = y[:min(3, len(y))]
            baseline_std = float(np.std(early))
            baseline_cv = baseline_std / baseline_t0 if baseline_t0 > 0 else float('nan')
            peak_idx = int(np.nanargmax(y))
            peak = float(y[peak_idx])
            dynamic_range = peak / baseline_t0 if baseline_t0 > 0 else float('nan')
            auc = float(np.trapz(np.clip(y - baseline_t0, 0, None), hrs)) if len(y) > 1 else 0.0
            snr = (peak - baseline_t0) / baseline_std if baseline_std > 0 else float('nan')

            rows.append(dict(
                experimentdata_id=self.exp_uuid,
                welldata_id=wid,
                channeldata_id=first['channeldata_id'],
                cellid=int(cellid),
                tile=int(tile),
                sensor=sensor,
                post_channel=post_ch,
                n_timepoints=int(len(g)),
                first_timepoint=int(tps[0]),
                last_timepoint=int(tps[-1]),
                baseline_t0=baseline_t0,
                baseline_std=baseline_std,
                baseline_cv=baseline_cv,
                peak_intensity=peak,
                peak_timepoint=int(tps[peak_idx]),
                peak_hours=float(hrs[peak_idx]),
                dynamic_range=dynamic_range,
                auc=auc,
                timecourse_slope=_slope(hrs, y),
                timecourse_rho=_spearman(tps, y),
                snr=snr,
                dropout=int(1 if tps[-1] < max_tp else 0),
                # helper columns (not written to minisogtrackdata)
                well=first.get('well'),
                celltype=first.get('celltype'),
                dosage=float(first['dosage']) if pd.notna(first.get('dosage')) else float('nan'),
            ))
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------ writes
    _TRACK_COLS = ('experimentdata_id', 'welldata_id', 'channeldata_id', 'cellid', 'tile',
                   'sensor', 'post_channel', 'n_timepoints', 'first_timepoint', 'last_timepoint',
                   'baseline_t0', 'baseline_std', 'baseline_cv', 'peak_intensity', 'peak_timepoint',
                   'peak_hours', 'dynamic_range', 'auc', 'timecourse_slope', 'timecourse_rho',
                   'snr', 'dropout')

    def write_track_metrics(self, per_track: pd.DataFrame) -> None:
        """Upsert per-track rows into ``minisogtrackdata`` (delete-then-add)."""
        for (wid, chid), _ in per_track.groupby(['welldata_id', 'channeldata_id']):
            self.Db.delete_based_on_duplicate_name(
                'minisogtrackdata', dict(welldata_id=wid, channeldata_id=chid))
        dcts = []
        for _, r in per_track.iterrows():
            dcts.append({c: _native(r[c]) for c in self._TRACK_COLS})
        self.Db.add_row('minisogtrackdata', dcts)
        print(f'Wrote {len(dcts)} rows to minisogtrackdata for sensor {per_track["sensor"].iloc[0]}.')

    def compare_sensors(self, tracks: pd.DataFrame) -> pd.DataFrame:
        """Score each (cell line x sensor) and pick the cleaner sensor.

        Ground truth: within a cell line, the death readout should increase with
        blue-light dose and rise over time. Cleaner = stronger dose-response and
        time-course, higher dynamic range, lower noise and dropout.

        Args:
            tracks: Concatenated per-track metrics for all sensors.

        Returns:
            One row per (celltype, sensor) with comparison stats and ``is_winner``.
        """
        rows: List[Dict[str, Any]] = []
        for (celltype, sensor), sub in tracks.groupby(['celltype', 'sensor']):
            wells = sub.groupby('well').agg(dose=('dosage', 'first'),
                                            med_dr=('dynamic_range', 'median')).dropna(subset=['dose'])
            dose_rho = _spearman(wells['dose'].values, wells['med_dr'].values) if len(wells) >= 3 else float('nan')
            dose_slope = _slope(wells['dose'].values, wells['med_dr'].values) if len(wells) >= 2 else float('nan')
            tc_rho = float(np.nanmedian(sub['timecourse_rho'])) if len(sub) else float('nan')
            dr_med = float(np.nanmedian(sub['dynamic_range'])) if len(sub) else float('nan')
            cv_med = float(np.nanmedian(sub['baseline_cv'])) if len(sub) else float('nan')
            dropout_rate = float(np.nanmean(sub['dropout'])) if len(sub) else float('nan')
            quality = self._quality_score(dose_rho, tc_rho, dr_med, cv_med, dropout_rate)
            rows.append(dict(
                experimentdata_id=self.exp_uuid,
                channeldata_id=None,
                sensor=sensor,
                celltype=celltype,
                n_wells=int(sub['well'].nunique()),
                n_tracks=int(len(sub)),
                dose_response_rho=dose_rho,
                dose_response_slope=dose_slope,
                timecourse_rho_median=tc_rho,
                dynamic_range_median=dr_med,
                baseline_cv_median=cv_med,
                dropout_rate=dropout_rate,
                quality_score=quality,
                is_winner=0,
            ))
        comp = pd.DataFrame(rows)
        # winner per cell line = max quality_score
        for celltype, sub in comp.groupby('celltype'):
            if sub['quality_score'].notna().any():
                win_idx = sub['quality_score'].idxmax()
                comp.loc[win_idx, 'is_winner'] = 1
        return comp

    @staticmethod
    def _quality_score(dose_rho: float, tc_rho: float, dr_med: float,
                       cv_med: float, dropout_rate: float) -> float:
        """Composite 'cleaner death signal' score (higher is better).

        Bounded terms so no single component dominates: dose-response and
        time-course correlations in [-1, 1]; a dynamic-range term in [0, 1);
        penalties for baseline noise and tracking dropout.
        """
        def z(v: float) -> float:
            return 0.0 if (v is None or not np.isfinite(v)) else float(v)
        dr_term = 1.0 - 1.0 / max(z(dr_med), 1.0) if np.isfinite(dr_med) else 0.0
        return (0.35 * z(dose_rho)
                + 0.25 * z(tc_rho)
                + 0.20 * dr_term
                - 0.10 * min(z(cv_med), 1.0)
                - 0.10 * z(dropout_rate))

    _COMP_COLS = ('experimentdata_id', 'channeldata_id', 'sensor', 'celltype', 'n_wells', 'n_tracks',
                  'dose_response_rho', 'dose_response_slope', 'timecourse_rho_median',
                  'dynamic_range_median', 'baseline_cv_median', 'dropout_rate', 'quality_score',
                  'is_winner')

    def write_comparison(self, comparison: pd.DataFrame) -> None:
        """Replace this experiment's ``minisogcomparisondata`` rows."""
        self.Db.delete_based_on_duplicate_name(
            'minisogcomparisondata', dict(experimentdata_id=self.exp_uuid))
        dcts = [{c: _native(r[c]) for c in self._COMP_COLS} for _, r in comparison.iterrows()]
        if dcts:
            self.Db.add_row('minisogcomparisondata', dcts)
        print(f'Wrote {len(dcts)} rows to minisogcomparisondata.')
        # human-readable verdict
        for celltype, sub in comparison.groupby('celltype'):
            win = sub[sub['is_winner'] == 1]
            wname = win['sensor'].iloc[0] if len(win) else 'n/a'
            print(f'  {celltype}: winner = {wname} '
                  f'({", ".join(f"{r.sensor} q={r.quality_score:.3f}" for r in sub.itertuples())})')

    # ------------------------------------------------------------------ output
    def export_csvs(self, tracks: pd.DataFrame, comparison: pd.DataFrame) -> None:
        """Write per-track and comparison CSVs under ``analysisdir/MINISOG``."""
        if not self.plotdir:
            return
        os.makedirs(self.plotdir, exist_ok=True)
        tracks.to_csv(os.path.join(self.plotdir, 'minisog_trackdata.csv'), index=False)
        comparison.to_csv(os.path.join(self.plotdir, 'minisog_comparison.csv'), index=False)
        print(f'Wrote CSVs to {self.plotdir}')

    def emit_plots(self, merged: pd.DataFrame, tracks: pd.DataFrame, comparison: pd.DataFrame) -> None:
        """Emit dose-response, time-course, and quality-score plots.

        Failures here are logged but never abort the run (the DB writes are the
        primary deliverable).
        """
        if not self.plotdir:
            return
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except Exception as e:  # pragma: no cover - environment dependent
            logger.warning(f'matplotlib unavailable, skipping plots: {e}')
            return
        os.makedirs(self.plotdir, exist_ok=True)
        post_channels = [s[1] for s in self.sensors]
        try:
            # 1) time-course: mean post readout vs timepoint, per sensor, per dose
            post = merged[merged['channel'].isin(post_channels)]
            fig, axes = plt.subplots(1, len(self.sensors), figsize=(6 * len(self.sensors), 5), squeeze=False)
            for ax, (name, post_ch, _) in zip(axes[0], self.sensors):
                sub = post[post['channel'] == post_ch]
                for cond, cg in sub.groupby('condition'):
                    tc = cg.groupby('timepoint')['readout'].mean()
                    ax.plot(tc.index, tc.values, marker='o', label=str(cond))
                ax.set_title(f'{name} ({post_ch})')
                ax.set_xlabel('timepoint')
                ax.set_ylabel('mean post intensity')
                ax.legend(title='dose', fontsize=8)
            fig.suptitle('Post-stim death signal over time (by blue-light dose)')
            fig.tight_layout()
            fig.savefig(os.path.join(self.plotdir, 'timecourse_by_dose.png'), dpi=120)
            plt.close(fig)

            # 2) dose-response: median dynamic_range vs dose, per cell line, sensors overlaid
            celltypes = sorted(tracks['celltype'].dropna().unique())
            if celltypes:
                fig, axes = plt.subplots(1, len(celltypes), figsize=(4.5 * len(celltypes), 4.5), squeeze=False)
                for ax, ct in zip(axes[0], celltypes):
                    for name in tracks['sensor'].unique():
                        s = tracks[(tracks['celltype'] == ct) & (tracks['sensor'] == name)]
                        dr = s.groupby('dosage')['dynamic_range'].median()
                        ax.plot(dr.index, dr.values, marker='o', label=name)
                    ax.set_title(str(ct))
                    ax.set_xlabel('blue-light dose (ms)')
                    ax.set_ylabel('median dynamic range')
                    ax.legend(fontsize=8)
                fig.suptitle('Dose-response by cell line: RFP16 vs NarrowRFP')
                fig.tight_layout()
                fig.savefig(os.path.join(self.plotdir, 'dose_response.png'), dpi=120)
                plt.close(fig)

            # 3) quality_score bars per cell line x sensor
            if not comparison.empty:
                piv = comparison.pivot(index='celltype', columns='sensor', values='quality_score')
                ax = piv.plot(kind='bar', figsize=(max(6, 1.5 * len(piv)), 5))
                ax.set_ylabel('quality_score (higher = cleaner)')
                ax.set_title('Sensor quality score by cell line')
                ax.figure.tight_layout()
                ax.figure.savefig(os.path.join(self.plotdir, 'quality_score.png'), dpi=120)
                plt.close(ax.figure)
            print(f'Wrote plots to {self.plotdir}')
        except Exception as e:
            logger.warning(f'Plot emission failed: {e}')
            print(f'WARNING: plot emission failed: {e}')


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='MiniSOG-RGEDI death-signal readout and sensor comparison.')
    parser.add_argument('--input_dict', default='', help='Unused; galaxy-compat stub.')
    parser.add_argument('--outfile', default='', help='Unused; galaxy-compat stub.')
    parser.add_argument('--experiment', required=True, type=str)
    parser.add_argument('--morphology_channel', default='Epi-GFP16', type=str,
                        help='RGEDI-green morphology channel (segmentation/tracking).')
    parser.add_argument('--sensors',
                        default='RFP16:Epi-RFP16-2:Epi-RFP16,NarrowRFP:Epi-NarrowRFP-2:Epi-NarrowRFP',
                        help="Comma list of 'name:post_channel:baseline_channel'.")
    parser.add_argument('--intensity_source', default='auto', choices=['auto', 'db', 'csv'],
                        help="Where red intensities come from: 'db' (intensitycelldata, tile flow), "
                             "'csv' (tracking-montage summary, montage flow), or 'auto' (db then csv).")
    parser.add_argument('--tracking_csv', default='',
                        help='Path to <experiment>_tracked_montage_summary.csv. '
                             'Default: <analysisdir>/<experiment>_tracked_montage_summary.csv.')
    parser.add_argument('--baseline_timepoint', default=0, type=int,
                        help='Timepoint whose pre-stim read is the absolute baseline.')
    parser.add_argument('--min_track_len', default=4, type=int,
                        help='Minimum observed timepoints for a track to be scored.')
    parser.add_argument('--chosen_wells', '-cw', default='all')
    parser.add_argument('--wells_toggle', default='include')
    parser.add_argument('--chosen_timepoints', '-ct', default='all')
    parser.add_argument('--timepoints_toggle', default='include')
    parser.add_argument('--tile', default=0, type=int, help='0 = all tiles.')
    return parser


if __name__ == '__main__':
    args = _build_parser().parse_args()
    print(args)
    MiniSOG(args).run()

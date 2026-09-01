#!/opt/conda/bin/python
"""MINISOG: GEDI-style death quantification for the miniSOG-RGEDI biosensor.

Death is quantified the way the GEDI paper does (Linsley et al., Nat Commun 2021):

* **GEDI ratio** = post-stim red (RGEDI death signal) / GFP (morphology / expression),
  per tracked cell per timepoint. Normalizing by GFP removes cell-size/expression
  variation, giving a far cleaner live/dead separation than raw red.
* **Death threshold** is derived from the T0 all-live population (a configurable
  percentile of the T0 ratio distribution -- the "live ceiling").
* **Time of death** = the first timepoint a cell's ratio crosses the threshold and
  stays above it (sustained). Cells that never cross are censored (alive/unknown).

From these it reports per-cell death metrics (``minisogtrackdata``), a per-cell-line
summary with the dose-response of death (``minisogcomparisondata``), and survival /
dose-response / time-of-death plots + CSVs under ``analysisdir/MINISOG``.

Input: the montage-flow tracking summary CSV (``--intensity_source csv``, which must
contain the GFP channel for the ratio) or the DB ``intensitycelldata`` (tile flow).
Requires TRACKING to have run. Set ``--death_metric raw`` to threshold raw red instead
of the ratio.
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
logger.warning('Running MiniSOG (GEDI) from Database.')


def _native(value: Any) -> Any:
    """Coerce numpy scalars / NaN to native Python types for DB insertion."""
    if value is None:
        return None
    if isinstance(value, (np.generic,)):
        value = value.item()
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        return None
    return value


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation (no scipy); nan if <3 finite pairs or no variance."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return float('nan')
    xr = pd.Series(x[m]).rank().values; yr = pd.Series(y[m]).rank().values
    if np.std(xr) == 0 or np.std(yr) == 0:
        return float('nan')
    return float(np.corrcoef(xr, yr)[0, 1])


def _slope(x: np.ndarray, y: np.ndarray) -> float:
    """OLS slope of y vs x; nan if <2 finite pairs or no x-variance."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 2 or np.std(x[m]) == 0:
        return float('nan')
    return float(np.polyfit(x[m], y[m], 1)[0])


def _km(event_times: np.ndarray, censor_times: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Kaplan-Meier product-limit survival estimate (no lifelines dependency).

    Args:
        event_times: times (hours) at which death occurred.
        censor_times: last-observed times (hours) for cells that never died.

    Returns:
        (t, S) step-function arrays: survival probability S at each time t.
    """
    event_times = np.asarray(event_times, float)
    censor_times = np.asarray(censor_times, float)
    times = np.concatenate([event_times, censor_times])
    ev = np.concatenate([np.ones(len(event_times)), np.zeros(len(censor_times))])
    if len(times) == 0:
        return np.array([0.0]), np.array([1.0])
    ts, ss, S = [0.0], [1.0], 1.0
    for t in np.unique(event_times):
        at_risk = int(np.sum(times >= t))
        d = int(np.sum((event_times == t)))
        if at_risk > 0:
            S *= (1 - d / at_risk)
        ts.append(float(t)); ss.append(S)
    return np.array(ts), np.array(ss)


class MiniSOG:
    """GEDI-style death quantification and per-cell-line dose-response.

    Args:
        opt: config namespace (argparse or Nextflow params). Key attributes:
            ``experiment``, ``sensors`` ('name:post_channel[,...]'), ``gfp_channel``,
            ``death_metric`` ('ratio'|'raw'), ``death_threshold_pct``,
            ``baseline_timepoint``, ``death_persist``, ``min_track_len``,
            ``intensity_source``, well/timepoint filters.
    """

    def __init__(self, opt: Any) -> None:
        self.opt = opt
        for attr, default in (('chosen_channels', 'all'), ('channels_toggle', 'include'), ('tile', 0)):
            if not hasattr(opt, attr):
                setattr(opt, attr, default)
        self.Op = Ops(opt)
        self.Db = Database()
        self.sensors: List[Tuple[str, str]] = self._parse_sensors(opt.sensors)
        self.gfp_channel: str = opt.gfp_channel
        self.death_metric: str = opt.death_metric           # 'ratio' | 'raw'
        self.death_threshold_pct: float = float(opt.death_threshold_pct)
        self.baseline_tp: int = int(opt.baseline_timepoint)
        self.death_persist: int = int(opt.death_persist)
        self.min_track_len: int = int(opt.min_track_len)
        self.exp_uuid = self.Db.get_table_uuid('experimentdata', dict(experiment=opt.experiment))
        if self.exp_uuid is None:
            raise Exception(f'Experiment not found in database: {opt.experiment}')
        analysisdir = self.Db.get_table_value('experimentdata', 'analysisdir', dict(id=self.exp_uuid))
        self.analysisdir = analysisdir[0][0] if analysisdir else None
        self.plotdir = os.path.join(self.analysisdir, 'MINISOG') if self.analysisdir else None
        self.tracking_csv = getattr(opt, 'tracking_csv', '') or (
            os.path.join(self.analysisdir, f'{opt.experiment}_tracked_montage_summary.csv')
            if self.analysisdir else '')

    @staticmethod
    def _parse_sensors(spec: str) -> List[Tuple[str, str]]:
        """Parse '--sensors' spec 'name:post_channel[,name:post_channel]'."""
        out = []
        for part in spec.split(','):
            part = part.strip()
            if not part:
                continue
            f = part.split(':')
            if len(f) < 2:
                raise ValueError(f"Bad --sensors entry {part!r}; expected 'name:post_channel'")
            out.append((f[0].strip(), f[1].strip()))
        if not out:
            raise ValueError('No sensors parsed from --sensors')
        return out

    def run(self) -> None:
        """Load intensities, compute GEDI death per cell, compare, and emit outputs."""
        merged = self.load_intensity_frame()
        all_tracks = []
        for name, post_ch in self.sensors:
            series = self.build_series(merged, post_ch)
            if series.empty:
                logger.warning(f'No data for sensor {name} (post={post_ch}); skipping.')
                print(f'WARNING: no data for sensor {name} ({post_ch}).')
                continue
            thr = self.compute_threshold(series)
            per_track = self.compute_track_metrics(series, name, post_ch, thr)
            if per_track.empty:
                continue
            self.write_track_metrics(per_track)
            all_tracks.append(per_track)
        if not all_tracks:
            raise Exception('No per-track metrics produced. Check GFP channel presence and tracking.')
        tracks = pd.concat(all_tracks, ignore_index=True)
        comparison = self.compare_sensors(tracks)
        self.write_comparison(comparison)
        self.emit_plots(tracks, comparison)
        self.export_csvs(tracks, comparison)
        print('MiniSOG (GEDI) done.')

    # ---------------------------------------------------------------- loading
    _TIDY = ('welldata_id', 'well', 'cellid', 'tile', 'channel', 'channeldata_id',
             'timepoint', 'hours', 'celltype', 'condition', 'dosage', 'readout')

    def load_intensity_frame(self) -> pd.DataFrame:
        """Load per-cell intensities (long form, incl. the GFP channel) from CSV or DB."""
        src = getattr(self.opt, 'intensity_source', 'auto')
        if src == 'csv':
            return self.load_from_csv()
        if src == 'db':
            return self.load_from_db()
        try:
            return self.load_from_db()
        except Exception as e:
            logger.warning(f'DB source unavailable ({e}); using tracking CSV.')
            return self.load_from_csv()

    def load_from_csv(self) -> pd.DataFrame:
        """Load intensities from the montage tracking-summary CSV (must include GFP)."""
        p = self.tracking_csv
        if not p or not os.path.exists(p):
            raise Exception(f'Tracking-montage summary CSV not found: {p!r}.')
        raw = pd.read_csv(p)
        if raw.empty:
            raise Exception(f'Tracking CSV is empty: {p}')
        cm = {c.lower(): c for c in raw.columns}

        def col(*names):
            for n in names:
                if n.lower() in cm:
                    return cm[n.lower()]
            raise Exception(f'Tracking CSV missing column (any of {names}); has {list(raw.columns)}')

        df = pd.DataFrame({
            'well': raw[col('well', 'Sci_WellID')].astype(str),
            'cellid': raw[col('tracked_id', 'ObjectLabelsFound')].astype(int),
            'channel': raw[col('MeasurementTag')].astype(str),
            'timepoint': raw[col('timepoint', 'Timepoint')].astype(int),
            'readout': raw[col('PixelIntensityMean')].astype(float),
        })
        Db, exp = self.Db, self.exp_uuid
        well = Db.get_df_from_query('welldata', dict(experimentdata_id=exp))[
            ['id', 'well', 'celltype', 'condition']].rename(columns={'id': 'welldata_id'})
        chan = Db.get_df_from_query('channeldata', dict(experimentdata_id=exp))[
            ['id', 'channel', 'welldata_id']].rename(columns={'id': 'channeldata_id'})
        tile = Db.get_df_from_query('tiledata', dict(experimentdata_id=exp))[
            ['timepoint', 'hours']].drop_duplicates('timepoint')
        dose = Db.get_df_from_query('dosagedata', dict(experimentdata_id=exp))
        df = df.merge(well, on='well', how='left').merge(chan, on=['channel', 'welldata_id'], how='left')
        df = df.merge(tile, on='timepoint', how='left')
        if df['hours'].isna().any():
            df['hours'] = df['hours'].fillna(4.0 * df['timepoint'].astype(float))
        if not dose.empty:
            df = df.merge(dose[['welldata_id', 'dosage']].drop_duplicates('welldata_id'),
                          on='welldata_id', how='left')
        else:
            df['dosage'] = np.nan
        df['tile'] = 0
        df = self._apply_filters(df)
        print(f'Loaded {len(df)} rows: {df.channel.nunique()} channels, {df.well.nunique()} wells, '
              f'{df.timepoint.nunique()} timepoints from {p}.')
        return df[list(self._TIDY)]

    def load_from_db(self) -> pd.DataFrame:
        """Load intensities from DB intensitycelldata (tile flow); must include GFP + reds."""
        Db, exp = self.Db, self.exp_uuid
        icd = Db.get_df_from_query('intensitycelldata', dict(experimentdata_id=exp))
        if icd.empty:
            raise Exception('intensitycelldata empty (run INTENSITY, tile flow).')
        cell = Db.get_df_from_query('celldata', dict(experimentdata_id=exp))
        if cell['cellid'].notna().sum() == 0:
            raise Exception('celldata.cellid all NULL - TRACKING has not run.')
        tile = Db.get_df_from_query('tiledata', dict(experimentdata_id=exp))
        chan = Db.get_df_from_query('channeldata', dict(experimentdata_id=exp))
        well = Db.get_df_from_query('welldata', dict(experimentdata_id=exp))
        dose = Db.get_df_from_query('dosagedata', dict(experimentdata_id=exp))
        cell = cell[['id', 'cellid']].rename(columns={'id': 'celldata_id'})
        tile = tile[['id', 'timepoint', 'hours', 'tile']].rename(columns={'id': 'tiledata_id'})
        chan = chan[['id', 'channel']].rename(columns={'id': 'channeldata_id'})
        well = well[['id', 'well', 'celltype', 'condition']].rename(columns={'id': 'welldata_id'})
        df = (icd.merge(cell, on='celldata_id', how='inner')
                 .merge(tile, on='tiledata_id', how='inner')
                 .merge(chan, on='channeldata_id', how='inner')
                 .merge(well, on='welldata_id', how='left'))
        if not dose.empty:
            df = df.merge(dose[['welldata_id', 'dosage']].drop_duplicates('welldata_id'),
                          on='welldata_id', how='left')
        else:
            df['dosage'] = np.nan
        need = {p for _, p in self.sensors} | {self.gfp_channel}
        if not df['channel'].isin(need).any():
            raise Exception(f'Required channels {sorted(need)} not in intensitycelldata (montage flow? '
                            'use --intensity_source csv).')
        df = df[df['cellid'].notna()].copy()
        df['cellid'] = df['cellid'].astype(int)
        df['readout'] = df['intensity_mean'].astype(float)
        df = self._apply_filters(df)
        print(f'Loaded {len(df)} intensitycelldata rows (DB).')
        return df[list(self._TIDY)]

    def _apply_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        cw = getattr(self.opt, 'chosen_wells', 'all')
        if cw and str(cw).lower() != 'all':
            df = Ops.filter_df(df, 'well', utils.get_iter_from_user(cw), self.opt.wells_toggle)
        ct = getattr(self.opt, 'chosen_timepoints', 'all')
        if ct and str(ct).lower() != 'all':
            tps = utils.get_iter_from_user(ct)
            if len(tps) and str(tps[0]).startswith('T'):
                tps = [t[1:] for t in tps]
            tps = [int(t) for t in tps if str(t).isnumeric()]
            df = Ops.filter_df(df, 'timepoint', tps, self.opt.timepoints_toggle)
        return df

    # ---------------------------------------------------------------- GEDI series
    def build_series(self, merged: pd.DataFrame, post_ch: str) -> pd.DataFrame:
        """Build the per-cell death-signal series (GEDI ratio = post/GFP, or raw post).

        Returns a frame with one row per (welldata_id, tile, cellid, timepoint) and a
        ``value`` column (the death signal) plus hours + well metadata.
        """
        keys = ['welldata_id', 'tile', 'cellid', 'timepoint']
        post = merged[merged['channel'] == post_ch][keys + ['well', 'hours', 'celltype', 'condition',
                                                            'dosage', 'channeldata_id', 'readout']]
        post = post.rename(columns={'readout': 'post'})
        if self.death_metric == 'raw':
            post['value'] = post['post']
            return post.dropna(subset=['value'])
        gfp = merged[merged['channel'] == self.gfp_channel][keys + ['readout']].rename(columns={'readout': 'gfp'})
        s = post.merge(gfp, on=keys, how='inner')
        s = s[s['gfp'] > 0].copy()
        s['value'] = s['post'] / s['gfp']
        return s.dropna(subset=['value'])

    def compute_threshold(self, series: pd.DataFrame) -> float:
        """Death threshold = percentile of the T0 (all-live) death-signal distribution."""
        t0 = series[series['timepoint'] == self.baseline_tp]['value'].values
        t0 = t0[np.isfinite(t0)]
        if len(t0) < 20:
            logger.warning(f'Few T0 points ({len(t0)}) for threshold; using all-timepoint pool.')
            t0 = series['value'].values
        thr = float(np.percentile(t0, self.death_threshold_pct))
        print(f'  threshold ({self.death_metric}, T0 p{self.death_threshold_pct:g}) = {thr:.4f}')
        return thr

    def _death_timepoint(self, tps: np.ndarray, vals: np.ndarray, thr: float) -> Optional[int]:
        """First timepoint whose value crosses thr and stays above for a full
        ``death_persist``-frame window. A crossing in the last (death_persist-1)
        observed frames cannot be confirmed, so it is NOT called (the cell is
        left censored) -- this avoids inflating deaths at the final timepoints.
        Returns the crossing timepoint, or None."""
        above = vals > thr
        n = len(tps)
        for i in range(n):
            if not above[i]:
                continue
            win = above[i:i + self.death_persist]
            if len(win) >= self.death_persist and win.all():
                return int(tps[i])
        return None

    def compute_track_metrics(self, series: pd.DataFrame, sensor: str, post_ch: str,
                              thr: float) -> pd.DataFrame:
        """Per-track continuous features + GEDI death call, for one sensor."""
        max_tp = int(series['timepoint'].max())
        base = series[series['timepoint'] == self.baseline_tp][
            ['welldata_id', 'tile', 'cellid', 'value']].rename(columns={'value': 'baseline_t0'})
        base = base.drop_duplicates(['welldata_id', 'tile', 'cellid'])
        rows: List[Dict[str, Any]] = []
        for (wid, tile, cellid), g in series.groupby(['welldata_id', 'tile', 'cellid']):
            g = g.sort_values('timepoint')
            if len(g) < self.min_track_len:
                continue
            tps = g['timepoint'].values.astype(float)
            hrs = g['hours'].values.astype(float)
            y = g['value'].values.astype(float)
            first = g.iloc[0]
            bm = base[(base.welldata_id == wid) & (base.tile == tile) & (base.cellid == cellid)]
            baseline_t0 = float(bm['baseline_t0'].iloc[0]) if len(bm) else float(np.nanmin(y))
            early = y[:min(3, len(y))]
            baseline_std = float(np.std(early))
            peak_idx = int(np.nanargmax(y)); peak = float(y[peak_idx])
            dtp = self._death_timepoint(g['timepoint'].values.astype(int), y, thr)
            death_hours = float(4.0 * dtp) if dtp is not None else float('nan')
            # hours for death: use observed hours at that tp when available
            if dtp is not None:
                match = g[g['timepoint'] == dtp]
                if len(match):
                    death_hours = float(match['hours'].iloc[0])
            rows.append(dict(
                experimentdata_id=self.exp_uuid, welldata_id=wid,
                channeldata_id=first['channeldata_id'], cellid=int(cellid), tile=int(tile),
                sensor=sensor, post_channel=post_ch, n_timepoints=int(len(g)),
                first_timepoint=int(tps[0]), last_timepoint=int(tps[-1]),
                baseline_t0=baseline_t0, baseline_std=baseline_std,
                baseline_cv=(baseline_std / baseline_t0 if baseline_t0 > 0 else float('nan')),
                peak_intensity=peak, peak_timepoint=int(tps[peak_idx]), peak_hours=float(hrs[peak_idx]),
                dynamic_range=(peak / baseline_t0 if baseline_t0 > 0 else float('nan')),
                auc=float(np.trapz(np.clip(y - baseline_t0, 0, None), hrs)) if len(y) > 1 else 0.0,
                timecourse_slope=_slope(hrs[:peak_idx + 1], y[:peak_idx + 1]),
                timecourse_rho=_spearman(tps[:peak_idx + 1], y[:peak_idx + 1]),
                snr=((peak - baseline_t0) / baseline_std if baseline_std > 0 else float('nan')),
                dropout=int(1 if tps[-1] < max_tp else 0),
                metric=self.death_metric, threshold=thr,
                died=int(dtp is not None), death_timepoint=(dtp if dtp is not None else None),
                time_to_death=death_hours,
                # helpers (not written to DB)
                well=first.get('well'), celltype=first.get('celltype'),
                dosage=float(first['dosage']) if pd.notna(first.get('dosage')) else float('nan'),
            ))
        return pd.DataFrame(rows)

    # ---------------------------------------------------------------- writes
    _TRACK_COLS = ('experimentdata_id', 'welldata_id', 'channeldata_id', 'cellid', 'tile', 'sensor',
                   'post_channel', 'n_timepoints', 'first_timepoint', 'last_timepoint', 'baseline_t0',
                   'baseline_std', 'baseline_cv', 'peak_intensity', 'peak_timepoint', 'peak_hours',
                   'dynamic_range', 'auc', 'timecourse_slope', 'timecourse_rho', 'snr', 'dropout',
                   'metric', 'threshold', 'died', 'death_timepoint', 'time_to_death')

    def write_track_metrics(self, per_track: pd.DataFrame) -> None:
        for (wid, chid), _ in per_track.groupby(['welldata_id', 'channeldata_id']):
            self.Db.delete_based_on_duplicate_name('minisogtrackdata', dict(welldata_id=wid, channeldata_id=chid))
        dcts = [{c: _native(r[c]) for c in self._TRACK_COLS} for _, r in per_track.iterrows()]
        self.Db.add_row('minisogtrackdata', dcts)
        print(f'Wrote {len(dcts)} rows to minisogtrackdata ({per_track["sensor"].iloc[0]}).')

    def compare_sensors(self, tracks: pd.DataFrame) -> pd.DataFrame:
        """Per (cell line x sensor): death fraction, dose-response of death, time-of-death."""
        rows: List[Dict[str, Any]] = []
        for (celltype, sensor), sub in tracks.groupby(['celltype', 'sensor']):
            thr = float(sub['threshold'].iloc[0])
            n = len(sub); n_died = int(sub['died'].sum())
            pct_dead = float(sub['died'].mean())
            tod = sub[sub['died'] == 1]['time_to_death'].values
            med_tod = float(np.nanmedian(tod)) if len(tod) else float('nan')
            # dose-response of DEATH: %dead vs blue-light dose (one well per dose within a line)
            dg = sub.dropna(subset=['dosage']).groupby('dosage')['died'].mean()
            dose_rho = _spearman(dg.index.values, dg.values) if len(dg) >= 3 else float('nan')
            dose_slope = _slope(dg.index.values, dg.values) if len(dg) >= 2 else float('nan')
            dr_med = float(np.nanmedian(sub['dynamic_range']))
            cv_med = float(np.nanmedian(sub['baseline_cv']))
            dropout_rate = float(np.nanmean(sub['dropout']))
            rows.append(dict(
                experimentdata_id=self.exp_uuid, channeldata_id=None, sensor=sensor, celltype=celltype,
                n_wells=int(sub['well'].nunique()), n_tracks=n,
                dose_response_rho=dose_rho, dose_response_slope=dose_slope,
                timecourse_rho_median=float(np.nanmedian(sub['timecourse_rho'])),
                dynamic_range_median=dr_med, baseline_cv_median=cv_med, dropout_rate=dropout_rate,
                quality_score=float('nan'), is_winner=1,
                threshold=thr, n_died=n_died, pct_dead=pct_dead, median_time_to_death=med_tod,
            ))
        return pd.DataFrame(rows)

    _COMP_COLS = ('experimentdata_id', 'channeldata_id', 'sensor', 'celltype', 'n_wells', 'n_tracks',
                  'dose_response_rho', 'dose_response_slope', 'timecourse_rho_median',
                  'dynamic_range_median', 'baseline_cv_median', 'dropout_rate', 'quality_score',
                  'is_winner', 'threshold', 'n_died', 'pct_dead', 'median_time_to_death')

    def write_comparison(self, comparison: pd.DataFrame) -> None:
        self.Db.delete_based_on_duplicate_name('minisogcomparisondata', dict(experimentdata_id=self.exp_uuid))
        dcts = [{c: _native(r[c]) for c in self._COMP_COLS} for _, r in comparison.iterrows()]
        if dcts:
            self.Db.add_row('minisogcomparisondata', dcts)
        print(f'Wrote {len(dcts)} rows to minisogcomparisondata.')
        for _, r in comparison.sort_values(['sensor', 'celltype']).iterrows():
            print(f'  {r.sensor}/{r.celltype}: {100*r.pct_dead:.0f}% dead (n={r.n_tracks}), '
                  f'median t-death={r.median_time_to_death:.0f}h, dose-response rho={r.dose_response_rho:.2f}')

    # ---------------------------------------------------------------- outputs
    def export_csvs(self, tracks: pd.DataFrame, comparison: pd.DataFrame) -> None:
        if not self.plotdir:
            return
        os.makedirs(self.plotdir, exist_ok=True)
        tracks.to_csv(os.path.join(self.plotdir, 'minisog_trackdata.csv'), index=False)
        comparison.to_csv(os.path.join(self.plotdir, 'minisog_comparison.csv'), index=False)
        print(f'Wrote CSVs to {self.plotdir}')

    def emit_plots(self, tracks: pd.DataFrame, comparison: pd.DataFrame) -> None:
        """Survival (cumulative %dead), dose-response of death, time-of-death distribution."""
        if not self.plotdir:
            return
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except Exception as e:  # pragma: no cover
            logger.warning(f'matplotlib unavailable, skipping plots: {e}')
            return
        os.makedirs(self.plotdir, exist_ok=True)
        try:
            for sensor in tracks['sensor'].unique():
                ts = tracks[tracks['sensor'] == sensor]
                tps = list(range(int(ts['last_timepoint'].max()) + 1))
                fig, ax = plt.subplots(1, 3, figsize=(18, 5))
                # (A) cumulative % dead over time
                n = len(ts)
                cum = [100.0 * ((ts['died'] == 1) & (ts['death_timepoint'] <= t)).sum() / n for t in tps]
                ax[0].plot([t * 4 for t in tps], cum, marker='o')
                ax[0].set_title(f'{sensor}: cumulative % dead'); ax[0].set_xlabel('hours'); ax[0].set_ylabel('% dead')
                # (B) % dead by dose per cell line
                for ct, sub in ts.groupby('celltype'):
                    dd = sub.groupby('dosage')['died'].mean() * 100
                    ax[1].plot(dd.index, dd.values, marker='o', label=str(ct))
                ax[1].set_title('% dead by dose'); ax[1].set_xlabel('blue-light dose (ms)')
                ax[1].set_ylabel('% dead'); ax[1].legend(fontsize=8)
                # (C) time-of-death distribution
                tod = ts[ts['died'] == 1]['time_to_death'].dropna().values
                if len(tod):
                    ax[2].hist(tod, bins=15)
                ax[2].set_title(f'time-of-death (median {np.nanmedian(tod):.0f}h)' if len(tod) else 'time-of-death')
                ax[2].set_xlabel('hours'); ax[2].set_ylabel('# cells')
                fig.suptitle(f'{sensor} GEDI death ({self.death_metric}, thr={ts["threshold"].iloc[0]:.3f})')
                fig.tight_layout(rect=[0, 0, 1, 0.95])
                fig.savefig(os.path.join(self.plotdir, f'gedi_death_{sensor}.png'), dpi=120)
                plt.close(fig)

                # Kaplan-Meier survival (death = event, non-death = censored at last obs)
                figk, axk = plt.subplots(1, 2, figsize=(13, 5))
                for ct, sub in ts.groupby('celltype'):
                    ev = sub[sub['died'] == 1]['time_to_death'].dropna().values
                    cens = sub[sub['died'] == 0]['last_timepoint'].values * 4.0
                    t, S = _km(ev, cens)
                    axk[0].step(t, S, where='post', label=str(ct))
                axk[0].set_title('KM survival by cell line'); axk[0].set_xlabel('hours')
                axk[0].set_ylabel('survival'); axk[0].set_ylim(0, 1.02); axk[0].legend(fontsize=8)
                for dz, sub in ts.dropna(subset=['dosage']).groupby('dosage'):
                    ev = sub[sub['died'] == 1]['time_to_death'].dropna().values
                    cens = sub[sub['died'] == 0]['last_timepoint'].values * 4.0
                    t, S = _km(ev, cens)
                    axk[1].step(t, S, where='post', label=f'{int(dz)}ms')
                axk[1].set_title('KM survival by dose'); axk[1].set_xlabel('hours')
                axk[1].set_ylabel('survival'); axk[1].set_ylim(0, 1.02); axk[1].legend(fontsize=8)
                figk.suptitle(f'{sensor} Kaplan-Meier survival (GEDI, thr={ts["threshold"].iloc[0]:.3f})')
                figk.tight_layout(rect=[0, 0, 1, 0.95])
                figk.savefig(os.path.join(self.plotdir, f'gedi_survival_{sensor}.png'), dpi=120)
                plt.close(figk)
            print(f'Wrote plots to {self.plotdir}')
        except Exception as e:
            logger.warning(f'Plot emission failed: {e}')
            print(f'WARNING: plot emission failed: {e}')


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='MiniSOG-RGEDI GEDI-style death quantification.')
    p.add_argument('--input_dict', default=''); p.add_argument('--outfile', default='')
    p.add_argument('--experiment', required=True)
    p.add_argument('--morphology_channel', default='Epi-GFP16')
    p.add_argument('--sensors', default='RFP16:Epi-RFP16-2',
                   help="Comma list of 'name:post_channel' (NarrowRFP dropped by default).")
    p.add_argument('--gfp_channel', default='Epi-GFP16', help='Morphology channel for the GEDI ratio.')
    p.add_argument('--death_metric', default='ratio', choices=['ratio', 'raw'],
                   help="'ratio' = post/GFP (GEDI); 'raw' = post red only.")
    p.add_argument('--death_threshold_pct', default=99.0, type=float,
                   help='Percentile of the T0 all-live distribution used as the death threshold.')
    p.add_argument('--death_persist', default=2, type=int,
                   help='Sustained frames above threshold required to call death.')
    p.add_argument('--baseline_timepoint', default=0, type=int)
    p.add_argument('--min_track_len', default=4, type=int)
    p.add_argument('--intensity_source', default='auto', choices=['auto', 'db', 'csv'])
    p.add_argument('--tracking_csv', default='')
    p.add_argument('--chosen_wells', '-cw', default='all'); p.add_argument('--wells_toggle', default='include')
    p.add_argument('--chosen_timepoints', '-ct', default='all'); p.add_argument('--timepoints_toggle', default='include')
    p.add_argument('--tile', default=0, type=int)
    return p


if __name__ == '__main__':
    args = _build_parser().parse_args()
    print(args)
    MiniSOG(args).run()

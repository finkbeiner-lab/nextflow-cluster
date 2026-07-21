"""Database operations wrapper for the Galaxy imaging pipeline.

Provides the ``Ops`` class, a higher-level interface on top of
:class:`sql.Database` that handles experiment/well/channel/timepoint
filtering and returns pandas DataFrames ready for downstream analysis
modules.
"""

import datetime
import logging
import os
import sys
from typing import Any, List, Optional, Tuple

import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from sql import Database
import utils as utils

# ---------------------------------------------------------------------------
# Module-level logging setup
# ---------------------------------------------------------------------------
logger_db = logging.getLogger("dbutil")

now = datetime.datetime.now()
TIMESTAMP = '%d%02d%02d%02d%02d' % (now.year, now.month, now.day, now.hour, now.minute)

fink_log_dir = './finkbeiner_logs'
if not os.path.exists(fink_log_dir):
    os.makedirs(fink_log_dir)

logname = os.path.join(fink_log_dir, f'dbutil-log_{TIMESTAMP}.log')
fh = logging.FileHandler(logname)
logger_db.addHandler(fh)
logger_db.info('Initialised db_util logging.')


class Ops:
    """High-level database operations for the imaging pipeline.

    Wraps :class:`sql.Database` queries with experiment-aware filtering
    by wells, channels, and timepoints based on user-supplied options.

    Args:
        opt: Configuration namespace (e.g. from argparse or Nextflow params).
            Expected attributes include ``experiment``, ``chosen_wells``,
            ``wells_toggle``, ``chosen_channels``, ``channels_toggle``,
            ``chosen_timepoints``, ``timepoints_toggle``, and ``tile``.
    """

    def __init__(self, opt: Any) -> None:
        self.opt = opt
        self.experiment: str = opt.experiment

    def get_raw_and_analysis_dir(self) -> Tuple[str, str]:
        """Retrieve the raw image directory and analysis directory for the experiment.

        Returns:
            Tuple of (imagedir, analysisdir) paths.

        Raises:
            Exception: If the experiment name is not found in the database.
        """
        Db = Database()
        _analysisdir = Db.get_table_value(tablename='experimentdata', column='analysisdir', kwargs=dict(experiment=self.experiment))
        _imagedir = Db.get_table_value(tablename='experimentdata', column='imagedir', kwargs=dict(experiment=self.experiment))

        if _imagedir is None:
            raise Exception(f'Experiment name not found in database. {self.experiment}')
        if _analysisdir is None:
            raise Exception(f'Analysis directory not found in database. {self.experiment}')
        imagedir = _imagedir[0][0]
        analysisdir = _analysisdir[0][0]
        return imagedir, analysisdir

    def get_well(self, uuid: Any, Db: Optional[Database] = None) -> Any:
        """Look up a well name by its UUID.

        Args:
            uuid: UUID of the well row.
            Db: Optional pre-existing Database instance to reuse.

        Returns:
            The well identifier (e.g. 'A1').

        Raises:
            Exception: If the UUID is not found in welldata.
        """
        if Db is None:
            Db = Database()
        result = Db.get_table_value(tablename='welldata', column='well', kwargs=dict(id=uuid))
        if result is None:
            raise Exception(f'Well uuid {uuid} not found in welldata database')
        return result[0]

    def add_col_from_table_to_df(self, df: pd.DataFrame, tablename: str, column: str) -> None:
        """Add a column from a related table to a DataFrame.

        .. note:: Not yet implemented.

        Args:
            df: Target DataFrame to augment.
            tablename: Source table containing the desired column.
            column: Column name to add.
        """
        raise NotImplementedError("add_col_from_table_to_df is not yet implemented")

    def get_tiledata_df(self) -> pd.DataFrame:
        """Query tiledata and apply well/channel/timepoint filters from ``self.opt``.

        Merges well names into the tile DataFrame and applies include/exclude
        filtering based on ``chosen_wells``, ``chosen_channels``, and
        ``chosen_timepoints`` configuration parameters.

        Returns:
            Filtered tiledata DataFrame with well and celltype columns joined.
        """
        Db = Database()

        exp_uuid = Db.get_table_uuid('experimentdata', dict(experiment=self.experiment))
        welldata_df = Db.get_df_from_query('welldata', dict(experimentdata_id=exp_uuid))
        channeldata_df = Db.get_df_from_query('channeldata', dict(experimentdata_id=exp_uuid))
        tiledata_df = Db.get_df_from_query('tiledata', dict(experimentdata_id=exp_uuid))
        # Join well names onto tiledata for downstream convenience
        tiledata_df = pd.merge(tiledata_df, welldata_df[['id', 'well', 'celltype']], left_on='welldata_id', right_on='id', how='left', suffixes=[None, '_dontuse'])

        # --- Filter by wells ---
        if self.opt.chosen_wells is not None and len(self.opt.chosen_wells) > 0 and self.opt.chosen_wells.lower() != 'all':
            selected_wells = utils.get_iter_from_user(self.opt.chosen_wells)
            print(f'Selected wells: {selected_wells}')
            welldata_df = self.filter_df(welldata_df, 'well', selected_wells, self.opt.wells_toggle)
            if not len(welldata_df):
                print('Welldata df is empty.')
                logger_db.info('Welldata df is empty.')
        # --- Filter by channels ---
        if self.opt.chosen_channels is not None and len(self.opt.chosen_channels) > 0 and self.opt.chosen_channels.lower() != 'all':
            selected_channels = self.opt.chosen_channels.strip(' ').split(',')
            print(f'Selected channels {selected_channels}')
            channeldata_df = self.filter_df(channeldata_df, 'channel', selected_channels, self.opt.channels_toggle)
            if not len(channeldata_df):
                print('Channeldata df is empty')
                logger_db.info('Channeldata df is empty')
        # --- Filter by timepoints ---
        if self.opt.chosen_timepoints is not None and len(self.opt.chosen_timepoints) > 0 and self.opt.chosen_timepoints.lower() != 'all':
            selected_timepoints = utils.get_iter_from_user(self.opt.chosen_timepoints)
            # Strip leading 'T' prefix (e.g. 'T0' -> '0') if present
            if len(selected_timepoints) and len(selected_timepoints[0]) > 1 and selected_timepoints[0][0] == 'T':
                selected_timepoints = [t[1:] for t in selected_timepoints]
            if len(selected_timepoints) and selected_timepoints[0].isnumeric():
                selected_timepoints = [int(t) for t in selected_timepoints]
            print(f'Selected timepoints {selected_timepoints}')
            logger_db.info(f'Selected timepoints {selected_timepoints}')
            tiledata_df = self.filter_df(tiledata_df, 'timepoint', selected_timepoints, self.opt.timepoints_toggle)
            if not len(tiledata_df):
                logger_db.info('Timepoint df is empty')
                print('Timepoint df is empty')

        # Keep only tiles whose well and channel survived filtering
        tiledata_df = tiledata_df[(tiledata_df.welldata_id.isin(welldata_df.id))
                                  & (tiledata_df.channeldata_id.isin(channeldata_df.id))]
        if self.opt.tile > 0:
            tiledata_df = self.filter_df_by_tile(tiledata_df, self.opt.tile)
        if not len(tiledata_df):
            print('Dataframe is empty after filtering. Check your selected wells, timepoints, channels.')
        logger_db.info(f'Length of tiledata df {len(tiledata_df)}')
        print(f'Length of tiledata df {len(tiledata_df)}')
        return tiledata_df

    def get_trackedmaskpath_from_other_channel(self, Db: Optional[Database], tile: int, timepoint: int) -> Optional[Any]:
        """Find a non-null tracked mask path for a given tile and timepoint.

        Searches across all channels to locate an existing tracked mask that
        can be reused when the current channel has not yet been tracked.

        Args:
            Db: Optional pre-existing Database instance to reuse.
            tile: Tile number.
            timepoint: Timepoint index.

        Returns:
            The first non-None trackedmaskpath found, or ``None``.
        """
        if Db is None:
            Db = Database()
        # Scope the lookup to this experiment: tile numbers repeat across
        # experiments, so filtering by tile/timepoint alone can return a path
        # belonging to a different experiment. welldata_id is not available in
        # this method's signature, so we cannot scope to a specific well here.
        exp_uuid = Db.get_table_uuid('experimentdata', dict(experiment=self.experiment))
        result = Db.get_table_value(tablename='tiledata', column='trackedmaskpath', kwargs=dict(tile=tile,
                                                                                                timepoint=timepoint,
                                                                                                experimentdata_id=exp_uuid))
        trackedmaskpath = next((el for el in result if el is not None), None)
        return trackedmaskpath

    def get_celldata_df(self) -> pd.DataFrame:
        """Return a DataFrame of cell data merged with filtered tile data.

        Returns:
            DataFrame with cell morphology columns joined to tile metadata.
        """
        tiledata_df = self.get_tiledata_df()
        tiledata_df.rename(columns={'id': 'tiledata_id'}, inplace=True)
        Db = Database()
        exp_uuid = Db.get_table_uuid('experimentdata', dict(experiment=self.experiment))
        celldata_df = Db.get_df_from_query('celldata', dict(experimentdata_id=exp_uuid))
        celldata_df = pd.merge(tiledata_df, celldata_df, on='tiledata_id', how='inner', suffixes=[None, '_dontuse'])
        return celldata_df
    
    def get_welldata_df(self) -> pd.DataFrame:
        """Return a DataFrame of all wells for the current experiment.

        Returns:
            DataFrame with welldata rows for the experiment.
        """
        Db = Database()
        exp_uuid = Db.get_table_uuid('experimentdata', dict(experiment=self.experiment))
        welldata_df = Db.get_df_from_query('welldata', dict(experimentdata_id=exp_uuid))
        return welldata_df

    def get_df_for_training(self, tablenames: List[str]) -> pd.DataFrame:
        """Build a merged DataFrame for ML training by joining multiple tables.

        Sequentially merges each table in ``tablenames`` onto the filtered
        tiledata DataFrame.  Supported table names: ``celldata``, ``cropdata``,
        ``dosagedata``, ``channeldata``.

        Args:
            tablenames: Ordered list of table names to merge.

        Returns:
            Merged DataFrame with columns from all requested tables.
        """
        df = self.get_tiledata_df()
        df.rename(columns={'id': 'tiledata_id'}, inplace=True)
        Db = Database()
        exp_uuid = Db.get_table_uuid('experimentdata', dict(experiment=self.experiment))
        for tablename in tablenames:
            df = df.loc[:, ~df.columns.str.contains('_dontuse')]
            table_df = Db.get_df_from_query(tablename, dict(experimentdata_id=exp_uuid))
            if tablename == 'celldata':  # celldata is based on morphology channel
                df = pd.merge(df, table_df, on='tiledata_id', how='inner', suffixes=[None, '_dontuse'])
                df.rename(columns={'id': 'celldata_id'}, inplace=True)
            if tablename == 'cropdata':  # can have multiple channels per cell
                df = pd.merge(table_df, df, on='celldata_id', how='inner', suffixes=[None, '_dontuse'])
                df.rename(columns={'id': 'cropdata_id'}, inplace=True)
            if tablename == 'dosagedata':
                df = pd.merge(df, table_df, on='welldata_id', how='inner', suffixes=[None, '_dontuse'])
            if tablename == 'channeldata':
                if self.opt.chosen_channels is not None and len(self.opt.chosen_channels) > 0 and self.opt.chosen_channels != 'all':
                    selected_channels = self.opt.chosen_channels.strip(' ').split(',')
                    print(f'Selected channels for ML: {selected_channels}')
                    table_df = self.filter_df(table_df, 'channel', selected_channels, self.opt.channels_toggle)
                df = pd.merge(df, table_df, left_on='channeldata_id', right_on='id', how='inner', suffixes=[None, '_dontuse'])
        df = df.loc[:, ~df.columns.str.contains('_dontuse')]
        return df
    
    def get_punctadata_df(self) -> pd.DataFrame:
        """Return a DataFrame of puncta data merged with tile and well data.

        Returns:
            DataFrame with puncta morphology columns joined to tile and well
            metadata.
        """
        tiledata_df = self.get_tiledata_df()
        tiledata_df.rename(columns={'id': 'tiledata_id'}, inplace=True)
        Db = Database()
        exp_uuid = Db.get_table_uuid('experimentdata', dict(experiment=self.experiment))
        punctadata_df = Db.get_df_from_query('punctadata', dict(experimentdata_id=exp_uuid))
        welldata_df = Db.get_df_from_query('welldata', dict(experimentdata_id=exp_uuid))
        punctadata_df = pd.merge(tiledata_df, punctadata_df, on='tiledata_id', how='inner', suffixes=[None, '_dontuse'])
        punctadata_df = pd.merge(punctadata_df, welldata_df,  left_on='welldata_id', right_on='id', how='inner', suffixes=[None, '_dontuse'])
        return punctadata_df

    @staticmethod
    def filter_df_by_tile(df: pd.DataFrame, tile: int) -> pd.DataFrame:
        """Filter a DataFrame to a single tile number.

        Args:
            df: DataFrame containing a ``tile`` column.
            tile: Tile number to keep.

        Returns:
            Filtered DataFrame.
        """
        return df[df.tile == tile]

    @staticmethod
    def filter_single_channel(df: pd.DataFrame, channel: str) -> pd.DataFrame:
        """Filter a DataFrame to a single channel.

        Args:
            df: DataFrame containing a ``channel`` column.
            channel: Channel name to keep.

        Returns:
            Filtered DataFrame.
        """
        return df[df.channel == channel]

    @staticmethod
    def filter_df(df: pd.DataFrame, column_name: str, selected_values: List[Any], toggle: str) -> pd.DataFrame:
        """Include or exclude rows based on column values.

        Args:
            df: Source DataFrame.
            column_name: Column to filter on.
            selected_values: Values to include or exclude.
            toggle: ``'include'`` to keep only matching rows,
                ``'exclude'`` to remove them.

        Returns:
            Filtered DataFrame.
        """
        if toggle == 'include':
            df = df[df[column_name].isin(selected_values)]
        else:
            df = df[~df[column_name].isin(selected_values)]
        return df

from sql import Database
import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

sys.path.append(os.path.dirname(SCRIPT_DIR))
import utils as utils
import pandas as pd
import datetime
import logging

logger_db = logging.getLogger("dbutil")
# logger_db.propagate = False
now = datetime.datetime.now()
TIMESTAMP = '%d%02d%02d%02d%02d' % (now.year, now.month, now.day, now.hour, now.minute)
print('Timestamp', TIMESTAMP)
fink_log_dir = './finkbeiner_logs'
if not os.path.exists(fink_log_dir):
    os.makedirs(fink_log_dir)
logname = os.path.join(fink_log_dir, f'dbutil-log_{TIMESTAMP}.log')
fh = logging.FileHandler(logname)
# fh.setLevel(logging.DEBUG)
logger_db.addHandler(fh)
logger_db.info('Running Segmentation from Database.')

class Ops:
    def __init__(self, opt):
        self.opt = opt
        self.experiment = opt.experiment

    def get_raw_and_analysis_dir(self):
        Db = Database()
        # get imagedir and savedir
        _analysisdir = Db.get_table_value(tablename='experimentdata', column='analysisdir', kwargs=dict(experiment=self.experiment))
        _imagedir = Db.get_table_value(tablename='experimentdata', column='imagedir', kwargs=dict(experiment=self.experiment))

        if _imagedir is None:
            raise Exception(f'Experiment name not found in database. {self.experiment}')
        imagedir = _imagedir[0][0]
        analysisdir = _analysisdir[0][0]
        return imagedir, analysisdir

    def get_well(self, uuid, Db=None):
        if Db is None:
            Db = Database()
        result = Db.get_table_value(tablename='welldata', column='well', kwargs=dict(id=uuid))
        if result is None:
            raise Exception(f'Well uuid {uuid} not found in welldata database')
        return result[0]

    def add_col_from_table_to_df(self, df, tablename, column):
        df[column] = None
        foreign_uuids = df[f'{tablename}_id']

    def get_tiledata_df(self):
        # Read db
        Db = Database()

        exp_uuid = Db.get_table_uuid('experimentdata', dict(experiment=self.experiment))
        welldata_df = Db.get_df_from_query('welldata', dict(experimentdata_id=exp_uuid))
        channeldata_df = Db.get_df_from_query('channeldata', dict(experimentdata_id=exp_uuid))
        tiledata_df = Db.get_df_from_query('tiledata', dict(experimentdata_id=exp_uuid))
        tiledata_df = pd.merge(tiledata_df, welldata_df[['id', 'well', 'celltype']], left_on='welldata_id', right_on='id', how='left', suffixes=[None, '_dontuse'])

        if self.opt.chosen_wells is not None and len(self.opt.chosen_wells) > 0 and self.opt.chosen_wells.lower() !='all':
            selected_wells = utils.get_iter_from_user(self.opt.chosen_wells)
            print(f'Selected wells: {selected_wells}')
            welldata_df = self.filter_df(welldata_df, 'well', selected_wells, self.opt.wells_toggle)
            if not(len(welldata_df)): 
                print('Welldata df is empty.')
                logger_db.info('Welldata df is empty.')
        if self.opt.chosen_channels is not None and len(self.opt.chosen_channels) > 0 and self.opt.chosen_channels.lower() !='all':
            selected_channels = self.opt.chosen_channels.strip(' ').split(',')
            print(f'Selected channels {selected_channels}')
            channeldata_df = self.filter_df(channeldata_df, 'channel', selected_channels, self.opt.channels_toggle)
            if not (len(channeldata_df)): 
                print('Channeldata df is empty')
                logger_db.info('Channeldata df is empty')
        if self.opt.chosen_timepoints is not None and len(self.opt.chosen_timepoints) > 0 and self.opt.chosen_timepoints.lower() !='all':
            selected_timepoints = utils.get_iter_from_user(self.opt.chosen_timepoints)
            if len(selected_timepoints) and len(selected_timepoints[0]) > 1 and selected_timepoints[0][0]=='T':
                selected_timepoints = [t[1:] for t in selected_timepoints]
            if len(selected_timepoints) and selected_timepoints[0].isnumeric():
                selected_timepoints = [int(t) for t in selected_timepoints]
            print(f'Selected timepoints {selected_timepoints}')
            logger_db.info(f'Selected timepoints {selected_timepoints}')
            tiledata_df = self.filter_df(tiledata_df, 'timepoint', selected_timepoints, self.opt.timepoints_toggle)
            if not (len(tiledata_df)): 
                logger_db.info('Timepoint df is empty')
                print('Timepoint df is empty')


        tiledata_df = tiledata_df[(tiledata_df.welldata_id.isin(welldata_df.id))
                                  & (tiledata_df.channeldata_id.isin(channeldata_df.id))]
        if self.opt.tile > 0:
            tiledata_df = self.filter_df_by_tile(tiledata_df, self.opt.tile)
        if not len(tiledata_df):
            print('Dataframe is empty after filtering. Check your selected wells, timepoints, channels.')
        logger_db.info(f'Length of tiledata df {len(tiledata_df)}')
        print(f'Length of tiledata df {len(tiledata_df)}')
        return tiledata_df

    def get_trackedmaskpath_from_other_channel(self, Db: Database, tile: int, timepoint: int):
        if Db is None:
            Db = Database()
        result = Db.get_table_value(tablename='tiledata', column='trackedmaskpath', kwargs=dict(tile=tile,
                                                                                                timepoint=timepoint))
        trackedmaskpath = next((el for el in result if el is not None), None)
        return trackedmaskpath

    def get_celldata_df(self):
        tiledata_df = self.get_tiledata_df()
        tiledata_df.rename(columns={'id': 'tiledata_id'}, inplace=True)
        Db = Database()
        exp_uuid = Db.get_table_uuid('experimentdata', dict(experiment=self.experiment))
        celldata_df = Db.get_df_from_query('celldata', dict(experimentdata_id=exp_uuid))
        celldata_df = pd.merge(tiledata_df, celldata_df, on='tiledata_id', how='inner', suffixes=[None, '_dontuse'])
        return celldata_df
    
    def get_df_for_training(self, tablenames:list):
        df = self.get_tiledata_df()
        df.rename(columns={'id': 'tiledata_id'}, inplace=True)
        Db = Database()
        exp_uuid = Db.get_table_uuid('experimentdata', dict(experiment=self.experiment))
        for tablename in tablenames:
            df = df.loc[:, ~df.columns.str.contains('_dontuse')]
            table_df = Db.get_df_from_query(tablename, dict(experimentdata_id=exp_uuid))
            if tablename=='celldata':  # celldata is based on morphology channel. Outer join includes all
                df = pd.merge(df, table_df, on='tiledata_id', how='inner', suffixes=[None, '_dontuse'])
                df.rename(columns={'id': 'celldata_id'}, inplace=True)
            if tablename=='cropdata':  # can have multiple channels per cell
                df = pd.merge(table_df, df, on='celldata_id', how='inner', suffixes=[None, '_dontuse'])
                df.rename(columns={'id': 'cropdata_id'}, inplace=True)
            if tablename=='dosagedata':
                df = pd.merge(df, table_df, on='welldata_id', how='inner', suffixes=[None, '_dontuse'])
            if tablename=='channeldata':
                if self.opt.chosen_channels is not None and len(self.opt.chosen_channels) > 0 and self.opt.chosen_channels!='all':
                    selected_channels = self.opt.chosen_channels.strip(' ').split(',')
                    print(f'Selected channels for ML: {selected_channels}')
                    table_df = self.filter_df(table_df, 'channel', selected_channels, self.opt.channels_toggle)
                df = pd.merge(df, table_df, left_on='channeldata_id', right_on='id', how='inner', suffixes=[None, '_dontuse'])
        df = df.loc[:, ~df.columns.str.contains('_dontuse')]
        return df
    
    def get_punctadata_df(self):
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
    def filter_df_by_tile(df, tile):
        return df[df.tile == tile]

    @staticmethod
    def filter_single_channel(df, channel):
        return df[df.channel == channel]

    @staticmethod
    def filter_df(df, column_name, selected_values, toggle):
        if toggle == 'include':
            df = df[df[column_name].isin(selected_values)]
        else:
            df = df[~df[column_name].isin(selected_values)]
        return df

#!/opt/conda/bin/python
"""Get csvs from database for experiment"""

from sql import Database
import pandas as pd
import logging
import datetime
import argparse
import os

logger = logging.getLogger("Puncta")
# logger.propagate = False
now = datetime.datetime.now()
TIMESTAMP = '%d%02d%02d%02d%02d' % (now.year, now.month, now.day, now.hour, now.minute)
print('Timestamp', TIMESTAMP)
fink_log_dir = './finkbeiner_logs'
if not os.path.exists(fink_log_dir):
    os.makedirs(fink_log_dir)
logname = os.path.join(fink_log_dir, f'CSV-log_{TIMESTAMP}.log')
fh = logging.FileHandler(logname)
# fh.setLevel(logging.DEBUG)
logger.addHandler(fh)
logger.warn('Registering experiment with database.')

class GetCSVS:
    def __init__(self, opt):
        self.opt = opt

    def run(self):
        Db = Database()
        exp_uuid = Db.get_table_uuid('experimentdata', dict(experiment=self.opt.experiment))
        analysisdir = Db.get_table_value('experimentdata','analysisdir',  dict(id=exp_uuid))
        if analysisdir is None: raise Exception(f'analysisdir is not found from experimentdata, {exp_uuid}.')
        analysisdir = analysisdir[0][0]
        savedir = os.path.join(analysisdir, 'CSVS')
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        experimentdata = Db.get_df_from_query('experimentdata', dict())
        welldata = Db.get_df_from_query('welldata', dict(experimentdata_id=exp_uuid))
        tiledata = Db.get_df_from_query('tiledata', dict(experimentdata_id=exp_uuid))
        celldata = Db.get_df_from_query('celldata', dict(experimentdata_id=exp_uuid))
        channeldata = Db.get_df_from_query('channeldata', dict(experimentdata_id=exp_uuid))
        punctadata = Db.get_df_from_query('punctadata', dict(experimentdata_id=exp_uuid))
        dosagedata = Db.get_df_from_query('dosagedata', dict(experimentdata_id=exp_uuid))
        intensitycelldata = Db.get_df_from_query('intensitycelldata', dict(experimentdata_id=exp_uuid))
        intensitypunctadata = Db.get_df_from_query('intensitypunctadata', dict(experimentdata_id=exp_uuid))

        experimentdata.to_csv(os.path.join(savedir, 'experimentdata.csv'))
        welldata.to_csv(os.path.join(savedir, 'welldata.csv'))
        tiledata.to_csv(os.path.join(savedir, 'tiledata.csv'))
        celldata.to_csv(os.path.join(savedir, 'celldata.csv'))
        channeldata.to_csv(os.path.join(savedir, 'channeldata.csv'))
        punctadata.to_csv(os.path.join(savedir, 'punctadata.csv'))
        dosagedata.to_csv(os.path.join(savedir, 'dosagedata.csv'))
        intensitycelldata.to_csv(os.path.join(savedir, 'intensitycelldata.csv'))
        intensitypunctadata.to_csv(os.path.join(savedir, 'intensitypunctadata.csv'))
        logger.warn(f'Saved csvs to {savedir}.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_dict',
        help='path to pickle, used to link modules in galaxy',
        default=f'/gladstone/finkbeiner/linsley/josh/GALAXY/YD-Transdiff-XDP-Survival1-102822/GXYTMP/tmp.pkl'
    )
    parser.add_argument(
        '--outfile',
        help='Tiff image of last tile',
        default=f'/gladstone/finkbeiner/linsley/josh/GALAXY/YD-Transdiff-XDP-Survival1-102822/GXYTMP/tmp_output.tif'
    )
    parser.add_argument('--experiment', type=str)
    args = parser.parse_args()
    print(args)
    GC = GetCSVS(args)
    GC.run()

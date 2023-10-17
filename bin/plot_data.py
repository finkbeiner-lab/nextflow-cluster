"""Plot table data against dosage"""

from db_util import Ops
from normalization import Normalize
from sql import Database
import argparse
import imageio
import numpy as np
import os
import logging
import datetime
import matplotlib.pyplot as plt
import pandas as pd

logger = logging.getLogger("Plot")
# logger.propagate = False
now = datetime.datetime.now()
TIMESTAMP = '%d%02d%02d%02d%02d' % (now.year, now.month, now.day, now.hour, now.minute)
print('Timestamp', TIMESTAMP)
fink_log_dir = './finkbeiner_logs'
if not os.path.exists(fink_log_dir):
    os.makedirs(fink_log_dir)
logname = os.path.join(fink_log_dir, f'Plot-log_{TIMESTAMP}.log')
fh = logging.FileHandler(logname)
# fh.setLevel(logging.DEBUG)
logger.addHandler(fh)
logger.warn('Plotting.')

class PlotTable:
    def __init__(self, opt):
        self.opt = opt
        self.Op = Ops(self.opt)
        self.tiledata = self.Op.get_tiledata_df()
        self.Db = Database()
        self.exp_uuid = self.Db.get_table_uuid('experimentdata', dict(experiment=self.opt.experiment))
        analysisdir = self.Db.get_table_value('experimentdata','analysisdir',  dict(id=self.exp_uuid))
        if analysisdir is None: raise Exception(f'analysisdir is not found from experimentdata, {self.exp_uuid}.')
        self.analysisdir = analysisdir[0][0]
        self.plotdir = os.path.join(self.analysisdir, 'Plots')
        if not os.path.exists(self.plotdir):
            os.makedirs(self.plotdir)
    def set_treatment_column(self, x):
        d = {}
        treatment_name = x.loc[x.kind=='treatment', 'name'].values[0]
        d = {'treatment': treatment_name}
        return pd.Series(d)
    def plot_puncta(self):
        # group classes
        punctadata = self.Op.get_punctadata_df()
        dosagedata = self.Db.get_df_from_query('dosagedata', dict(experimentdata_id=self.exp_uuid))
        punctadata = pd.merge(punctadata, dosagedata, on='welldata_id', how='inner', suffixes=[None, '_dontuse'])

        s = punctadata.groupby(by=['welldata_id']).apply(self.set_treatment_column)
        print('treatment', s)
        # punctadata.pivot(index=punctadata.id, columns='key')['val']
        punctadata = pd.merge(punctadata, s, on='welldata_id', how='inner', suffixes=[None, '_dontuse'])
        punctadata.to_csv(os.path.join(self.plotdir, 'merged.csv'))
        punctadata = punctadata.groupby(by=['id']).first()
        punctadata.pivot_table(index=['well'], columns='treatment',aggfunc='size').plot(kind='bar')
        plt.ylabel('Puncta Count')
        plt.xlabel('Well')
        plt.savefig(os.path.join(self.plotdir, 'puncta_per_well.png'))

        punctadata.pivot_table(index=['celltype'],columns='treatment',aggfunc='size').plot(kind='bar')
        plt.ylabel('Puncta Count')
        plt.xlabel('Celltype')
        plt.savefig(os.path.join(self.plotdir, 'puncta_per_celltype.png'))
        print('Done.')
        

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
    parser.add_argument("--wells_toggle",
                        help="Chose whether to include or exclude specified wells.")
    parser.add_argument("--timepoints_toggle",
                        help="Chose whether to include or exclude specified timepoints.")
    parser.add_argument("--channels_toggle",
                        help="Chose whether to include or exclude specified channels.")
    parser.add_argument("--chosen_wells", "-cw",
                        dest="chosen_wells", default='',
                        help="Specify wells to include or exclude")
    parser.add_argument("--chosen_timepoints", "-ct",
                        dest="chosen_timepoints", default='',
                        help="Specify timepoints to include or exclude.")
    parser.add_argument("--chosen_channels", "-cc",
                        dest="chosen_channels", default='',
                        help="Specify channels to include or exclude.")
    args = parser.parse_args()
    print(args)
    Pl = PlotTable(args)
    Pl.plot_puncta()

        
#!/opt/conda/bin/python


"""Update path on database after moving from TM
"""
from sql import Database
import os
import datetime
import argparse

class UpdatePath:
    def __init__(self, opt):
        self.opt = opt
        self.Db = Database()
    
    def run(self):
        # Add to database
        now = datetime.datetime.now()
        analysisdate = f'{now.year}-{now.month:02}-{now.day:02}'
        microscope=self.Db.get_table_value(tablename='experimentdata',column='microscope', kwargs=dict(experiment=self.opt.experiment))
        exp_uuid=self.Db.get_table_uuid(tablename='experimentdata',kwargs=dict(experiment=self.opt.experiment))
        print(microscope)
        assert microscope[0][0]=='TM', 'Experiment is not from Thinking Microscope'
        
        self.Db.update(tablename='experimentdata',
                       update_dct=dict(imagedir=os.path.join(f'/gladstone/finkbeiner/robodata/ThinkingMicroscope/{self.opt.experiment}'),
                                                                  analysisdir=os.path.join(f'/gladstone/finkbeiner/linsley/TM_analysis/GXYTMP-{self.opt.experiment}'),
                                                                  analysisdate=analysisdate),
                       kwargs=dict(experiment=self.opt.experiment))
                       
        
                # /gladstone/finkbeiner/robodata/ThinkingMicroscope/20230828-2-msneuron-cry2/F8
        self.Db.update_prefix_path('tiledata', exp_uuid=exp_uuid, old_string=r'D:/Images', 
                                   new_string='/gladstone/finkbeiner/robodata/ThinkingMicroscope')
        self.Db.update_slashes('tiledata', exp_uuid=exp_uuid)
        print('Done')


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_dict',
        help='path to pickle',
        default=f'/gladstone/finkbeiner/linsley/josh/GALAXY/YD-Transdiff-XDP-Survival1-102822/GXYTMP/tmp.pkl'
    )
    parser.add_argument(
        '--outfile',
        help='Text status',
        default=f'/gladstone/finkbeiner/linsley/josh/GALAXY/YD-Transdiff-XDP-Survival1-102822/GXYTMP/tmp_output.txt'
    )
    parser.add_argument('--experiment', default = '20231002-1-MSN-taueosx', type=str)
    args = parser.parse_args()
    print(args)
    Up = UpdatePath(args)
    Up.run()
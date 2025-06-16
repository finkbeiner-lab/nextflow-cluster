#!/opt/conda/bin/python

import imageio
import pickle
import shutil
import numpy as np
import pandas as pd
import os
import sys
import argparse
import time
import datetime
import logging
from sql import Database
from db_util import Ops

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from track_cells_res.minimum_flow import MinimumFlow
from track_cells_res.solver import SolverSmall
from track_cells_res.output_helper import OutputSmall
from track_cells_res.tracking_graph import Graph
import os
import warnings
# import matplotlib.pyplot as plt
import argparse
import time
from glob import glob
import logging
import datetime
from sql import Database
from db_util import Ops
import pdb

logger = logging.getLogger("Tracking")
now = datetime.datetime.now()
TIMESTAMP = '%d%02d%02d%02d%02d' % (now.year, now.month, now.day, now.hour, now.minute)

fink_log_dir = './finkbeiner_logs'
os.makedirs(fink_log_dir, exist_ok=True)
logname = os.path.join(fink_log_dir, f'TRACKING-log_{TIMESTAMP}.log')
fh = logging.FileHandler(logname)
logger = logging.getLogger("Tracking")
logger.addHandler(fh)
logger.warning('Tracking starting')

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
class TrackCells:
    def __init__(self, opt):
        self.opt = opt
        self.celldata = None
        self.filtered_celldata = None
        self.use_proximity = True
        self.use_overlap = True
        self.use_siamese = False
        self.forward_bool = True
        self.reverse_bool = True
        self.midpoint = None
        self.candidates = {}
        self.thread_lim = 6
        self.h, self.w = None, None
        self.randomcellid_str = 'randomcellid'
        self.centroid_x_str = 'centroid_x'
        self.centroid_y_str = 'centroid_y'
        logger.warning('Initialized Track Class.')
        self.Db = Database()

    
    def _resolve_duplicates(self, df, well, timepoint):
        subset = df[(df.well == well) & (df.timepoint == timepoint)]
        dups = subset.cellid[subset.cellid.duplicated()]
        for dup_val in set(dups):
            idxs = subset.index[subset.cellid == dup_val].tolist()
            for idx in idxs[1:]:
                df.at[idx, 'cellid'] = self.next_cellid
                self.next_cellid += 1
        return df


    def remap_labelled_mask(self, labelled_mask, mapping: dict):
        labelled_mask = np.float32(labelled_mask)
        for randomcellid, cellid in mapping.items():
            labelled_mask[labelled_mask == randomcellid] = cellid + .1
        return np.uint16(labelled_mask)

    def run(self):
        Op = Ops(self.opt)
        self.celldata = Op.get_celldata_df()
        self.welldata = Op.get_welldata_df()
        self.celldata.sort_values(by=['timepoint'], inplace=True)
        
        groups = self.celldata.groupby(by=['well'])
        
        for well, df in groups:
            start_well = time.time()
            logger.warning(f'Tracking {well}')
            print(f'Tracking {well}')
            
            experimentdata_id = self.Db.get_table_uuid('experimentdata', dict(experiment=self.opt.experiment))
            welldata_id = self.Db.get_table_uuid('welldata', dict(experimentdata_id=experimentdata_id, well=well))
            
            selected_channels = self.opt.chosen_channels.strip(' ').split(',')
            logger.warning(f'Channels for tracking: {selected_channels}')
            channel = selected_channels[0]
            
            channeldata_id = self.Db.get_table_uuid('channeldata', dict(
                experimentdata_id=experimentdata_id, welldata_id=welldata_id, channel=channel))
            
            timepoints = df.timepoint.unique().tolist()
            timepoints.sort()
            logger.warning(f'Timepoints: {timepoints}')
            
            # Initialize cell IDs for the first timepoint using composite of tile + randomcellid
            df = self.set_cellid_from_randomcellid(df, timepoints[0])
            df = self._resolve_duplicates(df, well, timepoints[0])
            df0 = df[df.timepoint == timepoints[0]]


            if df0.empty:
                print("df is empty")
                continue
            
               # After df0 is created:
            if not self.opt.DEBUG:
                for _, row in df0.iterrows():
                    self.Db.update(
                        'celldata',
                        update_dct={'cellid': int(row['cellid'])},
                        kwargs={
                            'id': row['id']
                        }
                    )
               
            # seed the global next_cellid counter
            self.next_cellid = int(df.cellid.max()) + 1

            # Retrieve montage info
            welldata_row = self.welldata[self.welldata['id'] == welldata_id]
            if welldata_row.empty or 'maskmontage' not in welldata_row.columns or 'imagemontage' not in welldata_row.columns:
                logger.warning(f'No montage data found for well {well}')
                continue
            
            f_prev = welldata_row['imagemontage'].iloc[0]
            f_prev_mask = welldata_row['maskmontage'].iloc[0]
            
            if len(df) < 2:
                continue
            
            start_tp = time.time()
            for prev_timepoint, current_timepoint in zip(timepoints[:-1], timepoints[1:]):
                elapsed_tp = time.time() - start_tp
                start_tp = time.time()
                logger.warning(f'Elapsed time for timepoint {current_timepoint}: {elapsed_tp}')

                if df[df.timepoint == current_timepoint].empty:
                    continue
                if df[df.timepoint == prev_timepoint].empty:
                    max_id = df.cellid.max()
                    cellids = df.loc[df.timepoint == current_timepoint, 'cellid']
                    new_ids = [i + 1 + max_id for i in range(len(cellids))]
                    df.loc[df.timepoint == current_timepoint, 'cellid'] = new_ids
                    tmp_df = df[df.timepoint == current_timepoint]
                    if not self.opt.DEBUG:
                        for _, row in tmp_df.iterrows():
                            self.Db.update('celldata', update_dct=dict(cellid=row.cellid),
                                           kwargs=dict(welldata_id=welldata_id, randomcellid=row.randomcellid))
                    continue
                
                df = self._resolve_duplicates(df, well, prev_timepoint)
                self.get_previous_and_current_tp_df(df, prev_timepoint, current_timepoint)
                f_curr = welldata_row['imagemontage'].iloc[0]
                f_curr_mask = welldata_row['maskmontage'].iloc[0]
                f_curr_mask_relabelled = f_curr_mask.replace("ENCODED_MONTAGE", "TRACKED_MONTAGE")

                logger.warning(f'Tracking {self.opt.experiment} at well {well} for timepoint T{prev_timepoint} → T{current_timepoint}')
                print(f'Tracking {self.opt.experiment} at well {well} for timepoint T{prev_timepoint} → T{current_timepoint}')

                
                
                mapping = self.run_one_timepoint(prev_timepoint, current_timepoint, f_prev, f_prev_mask,
                                                f_curr, f_curr_mask, f_curr_mask_relabelled, well)
                updated_celldata = self.update_mapping_to_dataframe(mapping, df, well, current_timepoint)
                df.loc[updated_celldata.index] = updated_celldata

                # ensure no duplicate IDs at this timepoint
                df = self._resolve_duplicates(df, well, current_timepoint)

                if not self.opt.DEBUG:
                    for random_lbl, lbl_from_prev in mapping.items():
                        self.Db.update('celldata', update_dct=dict(cellid=lbl_from_prev),
                                       kwargs=dict(welldata_id=welldata_id, randomcellid=random_lbl))

                f_prev = f_curr
                f_prev_mask = f_curr_mask

            elapsed_well = time.time() - start_well
            logger.warning(f'Elapsed time for well {well}: {elapsed_well}')
            # Optionally, update the welldata table with the final tracked montage mask
            self.Db.update('welldata',
                        update_dct=dict(trackedmaskmontage=f_curr_mask_relabelled),
                        kwargs=dict(id=welldata_id))



    def set_cellid_from_randomcellid(self, df, timepoint):
        max_rand = int(df['randomcellid'].max())
        shift    = 10 ** (len(str(max_rand)) + 1)
        composite = df['tile'].astype(int)*shift + df['randomcellid'].astype(int)
        df['composite_id'] = composite   # purely for DataFrame logic
        mask = (df['timepoint'] == timepoint)
        df.loc[mask, 'cellid'] = composite.loc[mask]
        return df


        

    def run_one_timepoint(self, prev_timepoint, current_timepoint, f_prev, f_prev_mask,
                      f_curr, f_curr_mask, f_curr_mask_relabelled, well, divisions=1, plot_bool=False):
        logger.warning(f'previous file: {f_prev}')
        logger.warning(f'previous mask: {f_prev_mask}')
        logger.warning(f'current file: {f_curr}')
        logger.warning(f'current mask: {f_curr_mask}')
        logger.warning(f'current mask relabelled: {f_curr_mask_relabelled}')

        mappings = {}

        

        Gr = Graph(celldata=self.filtered_celldata, include_appear=True,
                use_siamese=False, use_proximity=True, appear_cost=int(self.opt.DISTANCE_THRESHOLD),
                voronoi_bool=self.opt.VORONOI_BOOL,
                verbose=self.opt.VERBOSE,
                debug=self.opt.DEBUG)

        prev_img = imageio.v3.imread(f_prev)
        curr_img = imageio.v3.imread(f_curr)
        self.h, self.w = curr_img.shape
        prev, current = Gr.get_prev_and_curr_df(current_timepoint=current_timepoint, prev_timepoint=prev_timepoint)
        current = current.drop_duplicates('randomcellid')

        if not prev.empty and not current.empty:
            logger.warning('Getting edges from pandas dataframes')
            Gr.edges_from_pandas(prev, current, prev_img, curr_img)
            nodelist = list(Gr.g.nodes)
            logger.warning('Running LP')
            decision = self.linear_program(Gr, USE_GT=False)
            final_choice = self.replace_appear_node_with_cell_id(decision, nodelist)
            mappings.update(final_choice)
        else:
            logger.warning('One of the timepoint dataframes is empty')

        mapping = {int(k[1:]): int(v[1:]) for k, v in mappings.items()}

        curr_mask = imageio.v3.imread(f_curr_mask)
        curr_mask_relabelled = self.remap_labelled_mask(curr_mask, mapping)
        imageio.v3.imwrite(f_curr_mask_relabelled, curr_mask_relabelled)
        self.Db.update(
            'welldata',
            update_dct=dict(trackedmaskmontage=f_curr_mask_relabelled),
            kwargs=dict(experimentdata_id=self.Db.get_table_uuid('experimentdata', dict(experiment=self.opt.experiment)),
                        well=well))

        experimentdata_id = self.Db.get_table_uuid('experimentdata', dict(experiment=self.opt.experiment))
        welldata_id = self.Db.get_table_uuid('welldata', dict(experimentdata_id=experimentdata_id, well=well))

        if not self.opt.DEBUG:
            self.Db.update('welldata', update_dct=dict(trackedmaskmontage=f_curr_mask_relabelled), kwargs=dict(id=welldata_id))

        return mapping

    # def set_cellid_from_randomcellid(self, df, timepoint):
    #     df.loc[(df.timepoint == timepoint), 'cellid'] = df.loc[(df.timepoint == timepoint), 'randomcellid']
    #     return df


    
    def replace_appear_node_with_cell_id(self, decision, nodelist):
        max_lbl = 0
        if 'D' in decision.keys():
            del decision['D']
        print('nodelist', nodelist)
        for node in nodelist:  # get max existing label
            if node != 'A' and node != 'D':
                n = int(node[1:])
                if n > max_lbl:
                    max_lbl = n
    # set 'A' appear nodes to max label increasing
        for rnode, lnode in decision.items():
            if lnode == 'A':
                max_lbl += 1
                decision[rnode] = f'L{max_lbl}'
        return decision


    def linear_program(self, Gr, USE_GT=False):
        if USE_GT:
            warnings.warn('Duplicates set with GROUND TRUTH')
        # Initialize solver
        Solve = SolverSmall(debug=self.opt.DEBUG, verbose=False)
        Grs = MinimumFlow(Gr.g, True, self.opt.DEBUG)
        Outs = OutputSmall()

        # Sparse minimum flow linear program
        logger.warning('Getting incidence matrix and vertices')
        a_incidence, a_vertices = Grs.incidence_matrix()  # prepare algorithm input
        logger.warning(f'a_incidence {a_incidence}')
        logger.warning(f'a_vertices {a_vertices}')

        b_flow = Grs.b_flow(a_vertices)
        logger.warning(f'b_flow, {b_flow}')
        c_cost = Grs.c_cost(a_incidence, a_vertices)  # edge cost is distance between nodes
        logger.warning(f'c_cost {c_cost}')
        x = Solve.opto(a_incidence, b_flow, c_cost)  # solve
        decision = Outs.update(Grs.g, a_incidence, x, a_vertices)
        logger.warning(f'Decisions from linear program: {decision}')
        return decision

    def update_mapping_to_dataframe(self, mapping, celldata, well, timepoint):
        for random_lbl, lbl_from_prev_timepoint in mapping.items():
            celldata.loc[
                (celldata.well == well)
                & (celldata.timepoint == timepoint)
                & (celldata.randomcellid == random_lbl),
                'cellid'
            ] = lbl_from_prev_timepoint
        return celldata
    
    def load_celldata(self, celldata_csv):
        self.celldata = pd.read_csv(celldata_csv, low_memory=False)
        self.celldata['cellid'] = -1
        self.celldata.loc[self.celldata.Timepoint == 0, 'cellid'] = self.celldata.loc[
            self.celldata.Timepoint == 0, 'ObjectLabelsFound'
        ]
        self.celldata = self.celldata.drop_duplicates(subset=['randomcellid', 'well', 'timepoint'])

    def filter_celldata_func(self, well, prev_timepoint, current_timepoint):
        logger.warning(f'Filtering celldata with exp: {self.opt.experiment}, well: {well}, previous timepoint: {prev_timepoint}, current timepoint: {current_timepoint}')
        self.filtered_celldata = self.celldata[
            (self.celldata.experiment == self.opt.experiment)
            & (self.celldata.well == well)
            & ((self.celldata.timepoint == prev_timepoint) | (self.celldata.timepoint == current_timepoint))
        ].copy()

      
    
    def get_previous_and_current_tp_df(self, df, prev_timepoint, current_timepoint):
        self.filtered_celldata = df[(df.timepoint == prev_timepoint) | (df.timepoint == current_timepoint)]
    

    # All other methods (replace_appear_node_with_cell_id, linear_program, etc.) remain unchanged.
if __name__ == '__main__':
    DEBUG = 0
    VERBOSE = 0
    SAVEBOOL = 1
    USE_SIAMESE = 1
    USE_PROXIMITY = 1
    SET_RANDOM_IDS = 0
    USE_OVERLAP = 0
    VORONOI_CALC = 0
    VORONOI_BOOL = 0
    FORWARD_BOOL = 1
    REVERSE_BOOL = 1
    GET_ENCODED_MASKS = 0
    TIMEPOINT_BREAK = 0
    BOOTSTRAPPING = 0

    INIT_MINFLOW = 1
    GET_CONFIDENCE = 0
    PCA_BOOL = 0
    GET_TRAINING_RECS = 0
    DO_TRAINING = 0
    THREEEIGHTYFOUR = 0
    GET_EMBEDDING_REC = 0
    SET_EMBEDDING = 0
    _MIDPOINT = 0.0

    weight = 1
    alpha = 1

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dict', default='/path/to/tmp.pkl')
    parser.add_argument('--outfile', default='/path/to/output.pkl')
    parser.add_argument('--experiment', default='20231130-1-MsN-cry2tdp43-optocrisprgedi-update')
    parser.add_argument('--DISTANCE_THRESHOLD', default=300)
    parser.add_argument('--USE_PROXIMITY', default=1, type=int)
    parser.add_argument('--VORONOI_BOOL', default=0, type=str2bool)
    parser.add_argument('--VERBOSE', default=0, type=int)
    parser.add_argument('--DEBUG', default=0, type=int)
    parser.add_argument('--chosen_channels', default='RFP1')
    parser.add_argument('--wells_toggle', default='include')
    parser.add_argument('--timepoints_toggle', default='include')
    parser.add_argument('--channels_toggle', default='include')
    parser.add_argument('--chosen_wells', '-cw', default='B2')
    parser.add_argument('--chosen_timepoints', '-ct', default='T0,T1')
    parser.add_argument('--tile', default=0, type=int)

    args, _ = parser.parse_known_args()

    Track = TrackCells(args)
    Track.run()

    handlers = logger.handlers[:]
    for handler in handlers:
        logger.removeHandler(handler)
        handler.close()
    logging.shutdown()
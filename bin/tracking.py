#!/opt/conda/bin/python

import imageio
import pickle
import shutil
import numpy as np
import pandas as pd
import os
import sys

print('working dir', os.getcwd())
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

# logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("Tracking")
# _logger.propagate = False
now = datetime.datetime.now()
TIMESTAMP = '%d%02d%02d%02d%02d' % (now.year, now.month, now.day, now.hour, now.minute)
print('Timestamp', TIMESTAMP)
fink_log_dir = './finkbeiner_logs'
if not os.path.exists(fink_log_dir):
    os.makedirs(fink_log_dir)
logname = os.path.join(fink_log_dir, f'TRACKING-log_{TIMESTAMP}.log')
fh = logging.FileHandler(logname)
# fh.setLevel(logging.DEBUG)
logger.addHandler(fh)
logger.warning('Tracking starting')


class TrackCells:
    def __init__(self, opt):
        self.opt = opt
        self.celldata = None
        self.use_proximity = True
        self.use_overlap = True
        self.use_siamese = False
        self.forward_bool = True
        self.reverse_bool = True
        self.midpoint = None
        self.candidates = {}
        self.thread_lim = 6
        self.h, self.w = None, None
        self.celldata = None
        self.filtered_celldata = None
        self.randomcellid_str = 'randomcellid'
        self.centroid_x_str = 'centroid_x'
        self.centroid_y_str = 'centroid_y'
        logger.warning('Initialized Track Class.')
        self.Db = Database()

    def remap_labelled_mask(self, labelled_mask, mapping:dict):
        """Use mapping dictionary from tracking to relabel mask"""
        labelled_mask = np.float32(labelled_mask)
        for randomcellid, cellid in mapping.items():
            labelled_mask[labelled_mask == randomcellid] = cellid + .1
        return np.uint16(labelled_mask)

    def run(self):
        Op = Ops(self.opt)
        self.celldata = Op.get_celldata_df()
        self.celldata = self.celldata.sort_values(by=['timepoint'])
        start_well = time.time()
        groups = self.celldata.groupby(by=['well', 'tile'])
        # TODO: thread wells
        for (well, tile), df in groups:
            logger.warning(f'Tracking {well} {tile}')
            print(f'Tracking {well} {tile}')
            experimentdata_id = self.Db.get_table_uuid('experimentdata', dict(experiment=self.opt.experiment))
            welldata_id = self.Db.get_table_uuid('welldata', dict(experimentdata_id=experimentdata_id,
                                                                        well=well))
            selected_channels = self.opt.chosen_channels.strip(' ').split(',')
            # should just be one channel for tracking
            logger.warning(f'channels for tracking {selected_channels}')
            channel = selected_channels[0]
            channeldata_id = self.Db.get_table_uuid('channeldata', dict(experimentdata_id=experimentdata_id,
                                                                        welldata_id=welldata_id, 
                                                                        channel=channel))                                                  
            timepoints = self.celldata[self.celldata.well == well].timepoint.unique().tolist()
            timepoints.sort()
            logger.warning(f'timepoints: {timepoints}')
            # In case celldata start after T0, copy randomlabel to cellid label
            df = self.set_cellid_from_randomcellid(df, timepoints[0])
            df0 = df[df.timepoint == timepoints[0]]
            if df0.empty:
                continue
            tiledata_id_T0 = df0.tiledata_id.iloc[0]
            if not self.opt.DEBUG:
                for i, row in df0.iterrows():
                    self.Db.update('celldata', update_dct=dict(cellid=row.randomcellid),
                                kwargs=dict(tiledata_id=tiledata_id_T0,
                                            randomcellid=row.randomcellid))
                self.Db.update('tiledata', update_dct=dict(trackedmaskpath=df0.maskpath.iloc[0]), kwargs=dict(id=tiledata_id_T0))


            if len(df) < 2:
                continue
            start_tp = time.time()
            for prev_timepoint, current_timepoint in zip(timepoints[:-1], timepoints[1:]):
                elapsed_tp = time.time() - start_tp
                start_tp = time.time()
                logger.warning(f'Elapsed time for timepoint: {current_timepoint} is {elapsed_tp}')

                if df[df.timepoint==current_timepoint].empty:
                    continue
                if df[df.timepoint==prev_timepoint].empty:
                    # Set cellid from randomcellid
                    max_cell_id = df.cellid.max()
                    print('max cell id', max_cell_id)
                    cellids = df.loc[df.timepoint==current_timepoint, 'cellid']
                    new_cellids = [i + 1 + max_cell_id for i in range(len(cellids))]
                    df.loc[df.timepoint==current_timepoint, 'cellid'] = new_cellids
                    tmp_df = df.loc[df.timepoint==current_timepoint]
                    if not self.opt.DEBUG:
                        for i, row in tmp_df.iterrows():
                            self.Db.update('celldata', update_dct=dict(cellid=row.cellid),
                                    kwargs=dict(tiledata_id=row.tiledata_id,
                                                randomcellid=row.randomcellid))
                    continue
                # Updates self.filtered_celldata

                self.get_previous_and_current_tp_df(df, prev_timepoint, current_timepoint)
                
                # print('filtered', self.filtered_celldata)
                tiledata_id = self.filtered_celldata.loc[self.filtered_celldata.timepoint == current_timepoint, 'tiledata_id'].iloc[0]
                # Get files
                f_prev, f_prev_mask = self.filtered_celldata.loc[self.filtered_celldata.timepoint == prev_timepoint, ['filename', 'maskpath']].iloc[0]
                f_curr, f_curr_mask = self.filtered_celldata.loc[self.filtered_celldata.timepoint == current_timepoint, ['filename', 'maskpath']].iloc[0]
                f_curr_mask_relabelled = f_curr_mask.split('_')[:-1] + ['TRACKED.tif']
                f_curr_mask_relabelled = '_'.join(f_curr_mask_relabelled)
                # f_curr_mask_relabelled = f_curr_mask.split('ENCODED.tif')[0] + 'TRACKED.tif'
                logger.warning(f'Tracking {self.opt.experiment} at well {well} at tile {tile} for timepoint T{prev_timepoint} to T{current_timepoint}')
                print(f'Tracking {self.opt.experiment} at well {well} at tile {tile} for timepoint T{prev_timepoint} to T{current_timepoint}')

                mapping = self.run_one_timepoint(prev_timepoint, current_timepoint, f_prev, f_prev_mask,
                                                 f_curr, f_curr_mask, f_curr_mask_relabelled, divisions=1, plot_bool=False)
                updated_celldata = self.update_mapping_to_dataframe(mapping, self.filtered_celldata, well=well, tile=tile,
                                                                    timepoint=current_timepoint)
                # replace celldata cellid with updated tracked labels
                df.loc[updated_celldata.index] = updated_celldata
                assert np.all(updated_celldata.cellid != -1), '-1 in cellid after updating celldata'
                print('mapping', mapping)
                if not self.opt.DEBUG:
                    for random_lbl, lbl_from_prev_timepoint in mapping.items():
                        self.Db.update('celldata', update_dct=dict(cellid=lbl_from_prev_timepoint),
                                    kwargs=dict(tiledata_id=tiledata_id,
                                                randomcellid=random_lbl))

            elapsed_well = time.time() - start_well
            logger.warning(f'Elapsed time for well: {well} is {elapsed_well}')

    def run_one_timepoint(self, prev_timepoint, current_timepoint, f_prev, f_prev_mask,
                          f_curr, f_curr_mask, f_curr_mask_relabelled, divisions=1, plot_bool=False):
        logger.warning(f'previous file: {f_prev}')
        logger.warning(f'previous mask: {f_prev_mask}')
        logger.warning(f'current file: {f_curr}')
        logger.warning(f'current mask: {f_curr_mask}')
        logger.warning(f'current mask relabelled: {f_curr_mask_relabelled}')
        print(f'previous file: {f_prev}')
        print(f'previous mask: {f_prev_mask}')
        print(f'current file: {f_curr}')
        print(f'current mask: {f_curr_mask}')
        print(f'current mask relabelled: {f_curr_mask_relabelled}')
        mappings = {}

        Gr = Graph(celldata=self.filtered_celldata, include_appear=True,
                   use_siamese=False, use_proximity=True, appear_cost=int(self.opt.DISTANCE_THRESHOLD),
                   voronoi_bool=self.opt.VORONOI_BOOL,
                   verbose=self.opt.VERBOSE,
                   debug=self.opt.DEBUG)
        logger.warning(f'Divisions of image in tracking: {divisions}')
        prev_img = imageio.v3.imread(f_prev)
        curr_img = imageio.v3.imread(f_curr)
        self.h, self.w = curr_img.shape
        prev, current = Gr.get_prev_and_curr_df(current_timepoint=current_timepoint, prev_timepoint=prev_timepoint)
        current = current.drop_duplicates('randomcellid')  # todo: should be unnecessary

        # divisions is to break up the image in case the image is too large to process
        for i in range(divisions):
            for j in range(divisions):
                _prev = prev[(prev[self.centroid_x_str] >= i * self.w / divisions)
                             & (prev[self.centroid_x_str] < (i + 1) * self.w / divisions)
                             & (prev[self.centroid_y_str] >= j * self.h / divisions)
                             & (prev[self.centroid_y_str] < (j + 1) * self.h / divisions)]
                _current = current[(current[self.centroid_x_str] >= i * self.w / divisions)
                                   & (current[self.centroid_x_str] < (i + 1) * self.w / divisions)
                                   & (current[self.centroid_y_str] >= j * self.h / divisions)
                                   & (current[self.centroid_y_str] < (j + 1) * self.h / divisions)]
                if not _prev.empty and not _current.empty:
                    logger.warning('Getting edges from pandas dataframes')
                    df_current_tp = Gr.edges_from_pandas(_prev, _current, prev_img, curr_img)
                    nodelist = list(Gr.g.nodes)
                    logger.warning('Running LP')
                    decision = self.linear_program(Gr, USE_GT=False)
                    logger.warning(f'solution {decision}')
                    # replace appear node with new cell ids
                    final_choice = self.replace_appear_node_with_cell_id(decision, nodelist)

                    logger.warning(f'replaced appear and disappear with new neurons \n {i} {j}', final_choice)
                    for rnode, lnode in final_choice.items():
                        assert rnode not in mappings.keys()
                        mappings[rnode] = lnode
                else:
                    logger.warning(f'Length of previous timepoint dataframe: {len(_prev)}')
                    logger.warning(f'Length of current timepoint dataframe: {len(_current)}')
                    logger.warning('One of the timepoint dataframes is empty')

        logger.warning(f'mappings {mappings}')

        # checking mapping
        mapping = {}
        for rnode, lnode in mappings.items():
            r = int(rnode[1:])  # turn R2 into 2
            ell = int(lnode[1:])
            mapping[r] = ell
        logger.warning(f'Mapping previous timepoint to current timepoint: \n {mapping}')

        # prev_mask = imageio.v3.imread(f_prev_mask)
        curr_mask = imageio.v3.imread(f_curr_mask)
        curr_mask_relabelled = self.remap_labelled_mask(curr_mask, mapping)
        imageio.v3.imwrite(f_curr_mask_relabelled, curr_mask_relabelled)
        current_tile_uuid = self.filtered_celldata.loc[self.filtered_celldata.timepoint == current_timepoint, 'tiledata_id'].iloc[0]
        self.Db.update('tiledata', update_dct=dict(trackedmaskpath=f_curr_mask_relabelled), kwargs=dict(id=current_tile_uuid))
        return mapping

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

    def update_mapping_to_dataframe(self, mapping, celldata, well, tile, timepoint):
        """
        mapping: dict linking ObjectLabelsFound to cellid
        celldata: Filtered celldata based on experiment, well, timepoint, tracking channel
        todo: switch to uuid
        """
        for random_lbl, lbl_from_prev_timepoint in mapping.items():
            celldata.loc[(celldata.well == well)
                         & (celldata.tile == tile) & (celldata.timepoint == timepoint)
                         & (celldata.randomcellid == random_lbl), 'cellid'] = lbl_from_prev_timepoint
        return celldata

    def load_celldata(self, celldata_csv):
        self.celldata = pd.read_csv(celldata_csv, low_memory=False)
        self.celldata['cellid'] = -1
        # todo: delete, fails if tracking initializes after T0
        self.celldata.loc[self.celldata.Timepoint == 0, 'cellid'] = self.celldata.loc[self.celldata.Timepoint == 0, 'ObjectLabelsFound']
        self.celldata = self.celldata.drop_duplicates(subset=['randomcellid', 'well', 'timepoint'])  # todo: should be unnecessary

    def set_cellid_from_randomcellid(self, df, timepoint):
        df.loc[(df.timepoint == timepoint), 'cellid'] = df.loc[(df.timepoint == timepoint), 'randomcellid']
        return df

    def filter_celldata_func(self, well, tile, prev_timepoint, current_timepoint):
        # todo: store plate id without PID, just experiment name. Timestamp should be another column.
        logger.warning(f'Filtering celldata with exp: {self.opt.exp}, well: {well}, tile: {tile}, previous timepoint: {prev_timepoint}, current timepoint: {current_timepoint}')
        self.filtered_celldata = self.celldata[(self.celldata.experiment == self.opt.experiment)
                                               & (self.celldata.well == well)
                                               & (self.celldata.tile == tile)
                                               & ((self.celldata.timepoint == prev_timepoint) |
                                                  (self.celldata.timepoint == current_timepoint))].copy()

    def get_previous_and_current_tp_df(self, df, prev_timepoint, current_timepoint):
        self.filtered_celldata = df[(df.timepoint == prev_timepoint) | (df.timepoint == current_timepoint)]


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':

    DEBUG = 1
    VERBOSE = 0
    SAVEBOOL = 1
    USE_SIAMESE = 1
    USE_PROXIMITY = 1
    SET_RANDOM_IDS = 0
    USE_OVERLAP = 0
    VORONOI_CALC = 0  # calculate voronoi
    VORONOI_BOOL = 0  # use voronoi calculations, 0 to use basic proximity decide duplicates by cell area
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

    parser.add_argument(
        '--input_dict',
        help='path to pickle',
        default=f'/gladstone/finkbeiner/linsley/josh/GALAXY/YD-Transdiff-XDP-Survival1-102822/GXYTMP/tmp.pkl'
    )
    parser.add_argument(
        '--outfile',
        help='path to save pickle file',
        default=f'/gladstone/finkbeiner/linsley/josh/GALAXY/YD-Transdiff-XDP-Survival1-102822/GXYTMP/tmp_output.pkl'
    )
    parser.add_argument(
        '--experiment',default='20231026-1-msn-cry2tdp43-updated',
        help='Plate name',
        # default='SB26-30plate1'  # AB-CS47iTDP-Survival LINCS062016A AB-SOD1-KW4-WTC11-Survival
        # default='KS-AB-iMN-TDP43-Survival'  # AB-CS47iTDP-Survival LINCS062016A LINCS092016B
        # default='Sanofi-cmpds-plate8290040841-040121'  # AB-CS47iTDP-Survival LINCS062016A LINCS092016B
    )

    parser.add_argument(
        '--DISTANCE_THRESHOLD', default=300,
        help='Distance to determine a cell is a new cell.'
    )
    parser.add_argument(
        '--USE_PROXIMITY',
        action='store',
        help='0 or 1 for using proximity tracking',
        default=USE_PROXIMITY,
        type=int,
        dest='USE_PROXIMITY')
    parser.add_argument(
        '--VORONOI_BOOL',
        action='store',default=VORONOI_BOOL,
        help='Use Voronoi Regions. Otherwise uses proximity tracking.',
        type=str2bool,
        dest='VORONOI_BOOL')
    parser.add_argument(
        '--VERBOSE',
        action='store',
        help='String name of optimizer for tensorflow',
        default=VERBOSE,
        type=int,
        dest='VERBOSE')
    parser.add_argument(
        '--DEBUG',
        action='store',
        help='String name of optimizer for tensorflow',
        default=DEBUG,
        type=int,
        dest='DEBUG')
    parser.add_argument("--wells_toggle", default='include',
                        help="Chose whether to include or exclude specified wells.")
    parser.add_argument("--timepoints_toggle", default='include',
                        help="Chose whether to include or exclude specified timepoints.")
    parser.add_argument("--channels_toggle", default='include',
                        help="Chose whether to include or exclude specified channels.")
    parser.add_argument("--chosen_wells", "-cw",
                        dest="chosen_wells", default='A1',
                        help="Specify wells to include or exclude")
    parser.add_argument("--chosen_timepoints", "-ct",
                        dest="chosen_timepoints", default='T0,T1',
                        help="Specify timepoints to include or exclude.")
    parser.add_argument("--chosen_channels", "-cc", default='RFP1',
                        dest="chosen_channels",
                        help="Morphology Channel.")
    parser.add_argument('--tile', default=1, type=int, help="Select single tile to segment. Default is to segment all tiles.")

    args, _ = parser.parse_known_args()
    print('args', args)
    logger.warning(f'args: {args}')
    Track = TrackCells(args)
    mapping = Track.run()
    logger.warning('Tracking tasks completed.')
    print(args.outfile)
    logger.warning(args.outfile)
    logger.warning(args.input_dict)

    handlers = logger.handlers[:]
    for handler in handlers:
        logger.removeHandler(handler)
        handler.close()
    logging.shutdown()
    outfile = shutil.copyfile(args.input_dict, args.outfile)
    print('outfile2', outfile)

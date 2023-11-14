"""
Ops for minflow algorithm
Consider solving LInear Program (LP solutions) simplex method, ellipsoid , and in python scipy or pulp
"""

import networkx as nx
import pandas as pd
import numpy as np
from .voronoi import run_voronoi
import time
# import matplotlib.pyplot as plt
import logging
_logger = logging.getLogger("Tracking")



class Graph:
    def __init__(self, celldata, include_appear, appear_cost, use_siamese, use_proximity, voronoi_bool,
                 verbose, debug):
        assert len(celldata), 'Celldata is empty, check filtering'
        self.debug = debug
        self.verbose = verbose
        self.randomcellid_str = 'randomcellid'
        self.centroid_x_str = 'centroid_x'
        self.centroid_y_str = 'centroid_y'


        self.include_appear = include_appear  # include appear disappear nodes
        self.use_siamese = use_siamese
        self.use_proximity = use_proximity
        # Initialize graph
        self.g = nx.DiGraph()
        # Get csv data
        # parent_dir = '/mnt/finkbeinerlab/robodata/DreamChallengeDataSetsNew2017/Tracking_curated_masks'
        # todo: standardize experiment so it doesn't include PID

        self.celldata = celldata.copy()

        self.celldata = self.celldata.dropna(
            subset=[self.randomcellid_str])  # todo: update randomcellid with connected components
        well = pd.unique(self.celldata.well)
        assert len(well) == 1, f'Must be single well for tracking in input dataframe, {well}'

        self.celldata = self.celldata.loc[(self.celldata[self.centroid_x_str] >= 0) & (self.celldata[self.centroid_y_str] >= 0)]
        # todo: exclude centroids outside image
        # self.celldata = self.celldata.loc[(self.celldata.centroid_x < self.p.img_shape[0]) &
        #                                   (self.celldata.centroid_y < self.p.img_shape[1])]
        self.present_wells = list(pd.unique(self.celldata.well))
        assert len(self.present_wells) > 0, 'No wells detected in celldata.'
        self.neighbors = {}
        self.node_to_cellid_dict = {}
        self.voronoi_bool = voronoi_bool

        ######################## Confidence ######################
        self.intersection_over_L_RL = {}  # storing intersection for confidence metric
        self.intersection_over_R_LR = {}
        self.intersection_over_L_LR = {}  # storing intersection for confidence metric
        self.intersection_over_R_RL = {}
        self.distance_dict_LR = {}  # storing centroid distance
        self.distance_dict_RL = {}  # storing centroid distance

        self.data = None
        self.emb_size = 64
        self.emb_lst = [f'emb_{i}' for i in range(self.emb_size)]

        self.base_appear_cost = appear_cost
        # self.base_disappear_cost = 300

    def generate_imgs(self, wellid):
        """Generator for current and next image"""
        cnt = 0
        zstack = self.data.get_group(wellid)
        # for wellid, zstack in self.data:
        if len(pd.unique(zstack.timepoint)) > 1:  # must have multiple timepoints to track
            grp_zstack = zstack.sort_values('timepoint', ascending=True).groupby('timepoint')
            for tp, img in grp_zstack:
                # Set previous (left node) and current (right node)
                if cnt:
                    prev = current
                current = img
                cnt += 1
                if cnt == 1:
                    continue
                yield prev, current

    def get_prev_and_curr_df(self, current_timepoint, prev_timepoint):
        """Get previous and current dataframe"""
        current = self.celldata.loc[self.celldata.timepoint==current_timepoint]
        prev = self.celldata.loc[self.celldata.timepoint==prev_timepoint]
        return prev, current


    # def get_neighbors(self, prev, current):
    #     r_cnt = 0
    #     for i, cell in prev.iterrows():
    #         if cell.Timepoint == 0:
    #             label = 'L' + str(cell.cellid)
    #             self.neighbors[label] = {}
    #             for j, rcell in current.iterrows():
    #                 r_cnt += 1
    #                 rlabel = 'R' + str(r_cnt)
    #                 dist = np.sqrt((cell.centroid_x - rcell.centroid_x) ** 2 + (
    #                         cell.centroid_y - rcell.centroid_y) ** 2)
    #                 self.neighbors[label][rlabel] = dist

    def edges_from_pandas(self, prev_df, current_df, prev_img=None, curr_img=None):
        # print('prev df', prev_df.cellid)
        unique_cellids, cellid_counts = np.unique(prev_df.cellid, return_counts=True)
        _logger.info('unique_cellids {unique_cellids}')
        _logger.info('unique_cellids counts {cellid_counts}')
        if not prev_df.cellid.is_unique:
            _logger.error(prev_df.cellid)
            raise Exception('cell ids are not unique')
        # todo: cellids that are not found are being copied over, appear should handle this!
        # assert current.cellid.is_unique, 'cell ids are not unique, filter by time_imaged'  # should all be unique
        current_df = current_df.drop_duplicates(self.randomcellid_str)  # cellid is tracked and has not been written
        # current = current[current.area > 30]
        # prev = prev[prev.area > 30]
        prev_df = prev_df.sort_values('cellid')  # cellid is tracked, and T0 matches randomcellid
        current_df = current_df.sort_values(self.randomcellid_str)
        prev_x = prev_df[self.centroid_x_str].to_numpy()
        prev_y = prev_df[self.centroid_y_str].to_numpy()
        curr_x = current_df[self.centroid_x_str].to_numpy()
        curr_y = current_df[self.centroid_y_str].to_numpy()

        # Concatenate past and present timepoints to create adjacency matrix for pandas / networkx
        both_centroid_x = np.concatenate((prev_x, curr_x))
        both_centroid_y = np.concatenate((prev_y, curr_y))
        # Create mapping for cellids (previous timepoint) and randomcellids (current timepoint)
        mapping = {cnt: f'L{int(i)}' for cnt, i in enumerate(prev_df.cellid)}
        for cnt, i in enumerate(current_df[self.randomcellid_str]):
            max_idx = len(prev_x)
            mapping[max_idx + cnt] = f'R{int(i)}'
        XPREV, XCURR = np.meshgrid(both_centroid_x, both_centroid_x)
        YPREV, YCURR = np.meshgrid(both_centroid_y, both_centroid_y)

        # Get displacement (delta) along x and y
        XDELTA = XCURR - XPREV
        YDELTA = YCURR - YPREV

        # Euclidean distance
        DIST = np.sqrt(XDELTA ** 2 + YDELTA ** 2)
        # add appear and disappear edges
        N = len(DIST)
        print('tmp', self.base_appear_cost)
        _appear_arr = np.ones(N, ) * self.base_appear_cost
        _disappear_arr = np.ones(N, ) * self.base_appear_cost
        DIST = np.vstack((DIST, _appear_arr, _disappear_arr))
        _appear_arr2 = np.ones((N + 2, 1)) * self.base_appear_cost
        _disappear_arr2 = np.ones((N + 2, 1)) * self.base_appear_cost
        DIST = np.hstack((DIST, _appear_arr2, _disappear_arr2))
        DIST[-2:, -2:] = 0

        # distances that are too far to be considered the same cell, will exclude
        too_far = np.where(DIST > self.base_appear_cost)

        # to dataframe, then graph
        edge_df = pd.DataFrame(DIST)
        _logger.info(f'Edge df: {edge_df}')
        mapping[N] = 'A'
        mapping[N + 1] = 'D'
        g = nx.from_pandas_adjacency(edge_df)
        g = nx.relabel_nodes(g, mapping)
        # _logger.info(self.g['L0']['R0'])
        self.g = nx.DiGraph(g)
        # got graph with edges
        ebunch = []

        ########### APPLY VORONOI ################
        if self.voronoi_bool:
            start = time.time()
            edges_to_keep = run_voronoi(prev_df, current_df, prev_img.shape, curr_img.shape, debug=self.debug)
            start2 = time.time()
            elapsed = start2 - start
            _logger.info(f'voronoi elapsed {elapsed:.3}')
            remove_edge_set = set(self.g.edges) - set(edges_to_keep)
            remove_edge_set = [i for i in remove_edge_set if 'A' not in i and 'D' not in i]
            ebunch = list(remove_edge_set)
        else:
            start2 = time.time()
        ############ REMOVE EDGES #################
        _logger.info(f'Gathered edges to remove from Voronoi intersection.:{time.time() - start2:.3}')
        listOfCoordinates = list(zip(too_far[0], too_far[1]))
        for coord in listOfCoordinates:
            ebunch.append((mapping[coord[0]], mapping[coord[1]]))
        for i in prev_df.cellid:  # todo: check edges just go from L to R
            for j in prev_df.cellid:
                ebunch.append((f'L{i}', f'L{j}'))
            ebunch.append((f'L{i}', 'A'))
            ebunch.append(('A', f'L{i}'))
            ebunch.append(('D', f'L{i}'))

        for i in current_df[self.randomcellid_str]:
            for j in current_df[self.randomcellid_str]:
                ebunch.append((f'R{i}', f'R{j}'))
            ebunch.append((f'R{i}', 'A'))
            ebunch.append((f'R{i}', 'D'))
            ebunch.append(('D', f'R{i}'))
        ebunch.append(('D', 'A'))

        for i in current_df[self.randomcellid_str]:
            for j in prev_df.cellid:
                ebunch.append((f'R{i}', f'L{j}'))

        self.g.remove_edges_from(ebunch)
        _logger.info(f'Removed edges from graph {time.time() - start2:.3}')
        _logger.info(f'Possible edges {self.g.edges}')
        _logger.info(f'Possible nodes {self.g.nodes}')
        return current_df

    def incidence_matrix(self):
        """
        Make incidence matrix. Coupling the matrix is not necessary
        :return:
        """
        # couple_matrix
        # Construct coupled graph matrix from graph structure
        #
        #
        # Inputs:   g           -   current graph structure
        #
        # Outputs:  a_coup      -   coupled incidence matrix
        #           a_vertices  -   order of vertices in coupled matrix
        #

        # order of vertices in incidence matrix
        nodelist = self.g.nodes()
        a_vertices = [n for n in nodelist if 'M' not in n and 'S' not in n]

        # Incidence matrix
        a_sparse = nx.incidence_matrix(self.g, nodelist=nodelist, oriented=True)
        a_dense = a_sparse.todense()

        return a_dense, a_vertices

    def b_flow(self, a_vertices):

        # b_flow
        #
        # Construct flow constraint vector
        # (vector of size |V| x 1 representing the sum of flow for each vertex.
        # Having removed source and drain nodes. Now require:
        # L nodes = -1
        # R nodes = +1
        # A node = -|L|
        # D node = +|L|
        #
        # Inputs:   a_vertices  - order of nodes in coupled matrix
        # Outputs:  b_flow      -   flow constraint vector.
        #

        b = []

        # Total Cells
        l_cells = sum(1 for x in a_vertices if 'L' in x)
        r_cells = sum(1 for x in a_vertices if 'R' in x)

        # run through nodes and adjust flow for source/drain
        for node in a_vertices:
            if 'L' in node:
                b.append(-1)
            elif 'R' in node:
                b.append(1)
            elif 'A' in node:
                b.append(r_cells * (-1))
            elif 'D' in node:
                b.append(l_cells)
            else:
                _logger.info("Coupling matrix problems, there "
                      "remain split/merge vertices")

        return b

    def c_cost(self, a_incidence, a_vertices):

        # c_cost
        # creates vector of costs for edges
        #
        # Inputs:
        #           a_incidence      -  incidence matrix
        #           a_vertices  - order of rows in coupled matrix
        #
        # Outputs:  c           - list of costs for each edge in incidence matrix
        #

        # Initialise cost vector
        c = []

        # For all edges in coupled matrix (iterating over transpose)
        for e in a_incidence.T:

            # Get vertices connected by edge
            vertex_indices = np.nonzero(e)
            v = [a_vertices[i] for i in vertex_indices[1]]

            # Get weights
            cost = 0

            # that only works if they're labelled...

            # get relative distances in a dict?

            # For simple edges
            if len(v) == 2:
                try:
                    cost = self.g[v[0]][v[1]]['weight']
                except KeyError:
                    cost = self.g[v[1]][v[0]]['weight']

            # For coupled edges
            elif len(v) == 4:

                assert 0, 'there should be no coupled edges, no merge or split nodes'

            # Append to cost vector
            if self.verbose:
                l_gt = self.g.nodes[v[0]]['GT_id']
                r_gt = self.g.nodes[v[1]]['GT_id']
                _logger.info(f'{v}, {l_gt}, {r_gt} cost: {cost}')
            c.append(cost)

        return c


if __name__ == '__main__':
    from main.Database.sql import Database

    Db = Database()
    exp = 'tracking2'
    well = 'D3'

    celldata = Db.get_df_from_query(column_name='experiment', identifier=exp)
    celldata = celldata.loc[celldata.channel == 'GFP_DMD']
    _logger.info(celldata.columns)
    use_siamese = False
    voronoi_bool = True
    use_proximity = True
    verbose = True
    debug = False

    Gr = Graph(celldata=celldata, include_appear=True, exp=exp,
               use_siamese=use_siamese, use_proximity=use_proximity,
               voronoi_bool=voronoi_bool,
               verbose=verbose,
               debug=debug)
    Gr.filter_syn()
    for prev, current in Gr.generate_imgs(well):
        _logger.info(prev, current)
        Gr.edges_from_pandas(prev, current)

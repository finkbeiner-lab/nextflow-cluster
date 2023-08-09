"""Output ops for minflow, small is for running this only for duplicates in ops_manual"""
import collections
import csv
import numpy as np
import pandas as pd
import os
import shutil
import logging
logger = logging.getLogger("TM")

# todo: not using most of the functions

class OutputSmall:
    def __init__(self):
        pass

    def df_init(self):
        self.df['graph_id'] = -5
        self.df['graph_id_correct'] = -5
        self.df['distance_travel'] = -5
        self.df['intersection_over_region1'] = -5
        self.df['intersection_over_region2'] = -5
        self.df['confidence'] = -5
        self.df['nearest_neighbor_from_prev'] = -5
        self.df['second_nearest_neighbor_from_prev'] = -5
        self.df['nearest_neighbor_from_current'] = -5
        self.df['second_nearest_neighbor_from_current'] = -5

    def initialise_out(self, g_nodes):

        # initialise_out
        # Function to initialise_out output format
        #
        # inputs:   g   -   current graph structure
        #
        # outputs   out -   out structure
        #

        out = dict()
        out['frame'] = 0

        tracks = collections.OrderedDict()

        for vertex, data in g_nodes.items():
            if 'L' in vertex:
                # initialise_out cell data
                track = self.initialise_track(data)
                cell_id = len(tracks)
                track['cell_id'] = cell_id
                tracks[cell_id] = track

        out['tracks'] = tracks

        return out

    @staticmethod
    def initialise_track(data):

        # initialise_track
        # initialise track for new cell in output

        track = {'frame': [0],
                 'parent': None}

        for key, value in data.items():
            track[key] = [value]
        track['distance_travel'] = [-1]
        track['intersection_over_region1'] = [-10]
        track['intersection_over_region2'] = [-10]
        track['confidence'] = [-1]
        track['nearest_neighbor_from_prev'] = [-1]
        track['second_nearest_neighbor_from_prev'] = [-1]
        track['nearest_neighbor_from_current'] = [-1]
        track['second_nearest_neighbor_from_current'] = [-1]

        return track

    def update_metadata(self, timestamp, input_path, output_path, voronoi_bool, siamese_bool, proximity_bool,
                        overlap_bool, forward_bool, reverse_bool, debug):
        if not os.path.exists(self.p.metadata_csv):
            metadata_dict = {'timestamp': [], 'bootstrap_timestamp': [], 'input_path': [], 'minflow_path': [],
                             'bootstrap_path': [],
                             'Voronoi': [], 'Siamese': [], 'Proximity': [], 'Overlap': [],
                             'model_name': [], 'Has_Embedding': [], 'Debug': [], 'Forward': [], 'Reverse': []}
            metadata = pd.DataFrame(metadata_dict)
        else:
            metadata = pd.read_csv(self.p.metadata_csv)
        if self.bootstrapping:
            metadata.loc[
                metadata.embedded_path == input_path, ['bootstrap_timestamp', 'bootstrap_path', 'Siamese', 'Debug']] = [
                timestamp, output_path, siamese_bool * 1, debug]
        else:
            metadata_dict = {'timestamp': [timestamp], 'bootstrap_timestamp': ['No'], 'input_path': [input_path],
                             'minflow_path': [output_path],
                             'bootstrap_path': ['No'],
                             'Voronoi': [voronoi_bool * 1], 'Siamese': [siamese_bool * 1],
                             'Proximity': [proximity_bool * 1],
                             'Overlap': [overlap_bool * 1], 'model_name': ['No'], 'Has_Embedding': [0],
                             'Debug': [debug], 'Forward': [forward_bool * 1], 'Reverse': [reverse_bool * 1]}
            df = pd.DataFrame(metadata_dict)
            metadata = metadata.append(df)
        return metadata

    def update_cell_data(self, out, vertex, prev, g_nodes, active_cells, intersection_over_L, intersection_over_R,
                         distance_dict_LR, distance_dict_RL):

        # update_cell_data
        # Update output with cell data
        #
        # inputs:   out             - current output data
        #           vertex          - label for new cell data
        #           prev            - label for prev cell data
        #           g_nodes         - nodes in graph with attributes
        #           active_cells    - dict of 'active' cells and prev labels

        # find output row
        for cell_id, l_label in active_cells.items():
            if l_label == prev:
                # l_label Lx, vertex is Rx
                # Update cell information
                features = out['tracks'][cell_id]
                for key, value in features.items():

                    # Append frame number
                    if key == 'frame':
                        features['frame'].append((features['frame'][-1] + 1))

                    # append cell feature data
                    elif isinstance(features[key], list):
                        if key not in ['intersection_over_region1', 'intersection_over_region2', 'confidence',
                                       'distance_travel',
                                       'nearest_neighbor_from_prev', 'second_nearest_neighbor_from_prev',
                                       'nearest_neighbor_from_current', 'second_nearest_neighbor_from_current']:
                            features[key].append(g_nodes[vertex][key])
                try:
                    features['distance_travel'].append(distance_dict_LR[l_label][vertex])
                except KeyError:
                    features['distance_travel'].append(-6)

                try:
                    sorted_distances = sorted(distance_dict_LR[l_label].values())
                except KeyError:
                    sorted_distances = [-7, -7]
                try:
                    features['nearest_neighbor_from_prev'].append(sorted_distances[0])
                except IndexError:
                    features['nearest_neighbor_from_prev'].append(-6)
                try:
                    features['second_nearest_neighbor_from_prev'].append(sorted_distances[1])
                except IndexError:
                    features['second_nearest_neighbor_from_prev'].append(-6)

                try:
                    sorted_distances = sorted(distance_dict_RL[vertex].values())
                except KeyError:
                    sorted_distances = [-7, -7]
                try:
                    features['nearest_neighbor_from_current'].append(sorted_distances[0])
                except IndexError:
                    features['nearest_neighbor_from_current'].append(-6)
                try:
                    features['second_nearest_neighbor_from_current'].append(sorted_distances[1])
                except IndexError:
                    features['second_nearest_neighbor_from_current'].append(-6)

                try:
                    features['intersection_over_region1'].append(intersection_over_L[vertex][l_label])
                    features['intersection_over_region2'].append(intersection_over_R[l_label][vertex])
                    features['confidence'].append(
                        intersection_over_L[vertex][l_label] * intersection_over_R[l_label][vertex])
                except KeyError:
                    features['intersection_over_region1'].append(0)
                    features['intersection_over_region2'].append(0)
                    features['confidence'].append(0)

    def update_disappear(self, out, vertex, prev, g_nodes, active_cells, intersection_over_L, intersection_over_R,
                         distance_dict_LR, distance_dict_RL):

        # update_cell_data
        # Update output with cell data
        #
        # inputs:   out             - current output data
        #           vertex          - label for new cell data
        #           prev            - label for prev cell data
        #           g_nodes         - nodes in graph with attributes
        #           active_cells    - dict of 'active' cells and prev labels

        # Disappear confidence should be max intersection of Lx with any Rx

        # find output row
        for cell_id, l_label in active_cells.items():
            if l_label == prev:

                # Update cell information
                features = out['tracks'][cell_id]
                for key, value in features.items():

                    # Append frame number
                    if key == 'frame':
                        features['frame'].append((features['frame'][-1] + 1))

                    # append cell feature data
                    elif isinstance(features[key], list):
                        if key not in ['intersection_over_region1', 'intersection_over_region2', 'confidence',
                                       'distance_travel',
                                       'nearest_neighbor_from_prev', 'second_nearest_neighbor_from_prev',
                                       'nearest_neighbor_from_current', 'second_nearest_neighbor_from_current']:
                            features[key].append(g_nodes[vertex][key])
                features['intersection_over_region1'].append(-2)
                features['intersection_over_region2'].append(-2)
                features['confidence'].append(-2)
                features['distance_travel'].append(-2)
                features['nearest_neighbor_from_prev'].append(-2)
                features['second_nearest_neighbor_from_prev'].append(-2)
                features['nearest_neighbor_from_current'].append(-2)
                features['second_nearest_neighbor_from_current'].append(-2)

    def update_appear(self, current_out, vertex, g_nodes, intersection_over_L, intersection_over_R, distance_dict_LR,
                      distance_dict_RL):

        # update_appear
        # Update output for cell appearances
        #
        # Inputs:   current_out     - current output data (list of lists)
        #           vertex          - label for new cell data (string)
        #           g_nodes         - nodes in graph with associated attributes
        #           parent_id       - parent cell labels (if any)
        #
        # Appear confidence should be max intersection of Rx with any Lx

        # Initialise data structure for new track
        new_cell_track = self.initialise_track(g_nodes[vertex])
        max_cell_id = 0
        for cell_id_key, features in current_out['tracks'].items():
            hot_pred_id = np.max(features['GT_id'])
            if hot_pred_id > max_cell_id:
                max_cell_id = hot_pred_id
        new_pred_id = max_cell_id + 1
        cell_id = len(current_out['tracks'])
        new_cell_track['cell_id'] = cell_id
        new_cell_track['GT_id'][0] = new_pred_id

        # Adjust frame to moment appeared
        new_cell_track['frame'] = [current_out['frame']]

        # Add to output
        current_out['tracks'][cell_id] = new_cell_track
        current_out['tracks'][cell_id]['intersection_over_region1'][0] = -3
        current_out['tracks'][cell_id]['intersection_over_region2'][0] = -3
        current_out['tracks'][cell_id]['confidence'][0] = -3
        current_out['tracks'][cell_id]['distance_travel'][0] = -3
        current_out['tracks'][cell_id]['nearest_neighbor_from_prev'][0] = -3
        current_out['tracks'][cell_id]['second_nearest_neighbor_from_prev'][0] = -3
        current_out['tracks'][cell_id]['nearest_neighbor_from_current'][0] = -3
        current_out['tracks'][cell_id]['second_nearest_neighbor_from_current'][0] = -3

    def reduce_a_pyomo(self, a, x):
        # reduce_a
        # Reduces incidence matrix to only included vertices and edges
        #
        # Inputs:   a   - dense coupled incidence matrix
        #           x   - Optimisation solution
        #
        # Outputs:  a_sol - incidence matrix with only included edges/vertices
        #                   list of lists
        #

        # Extract edges to delete from solution
        included_edges = [j for j in range(len(x)) if x[j].value == 1]

        if not included_edges:
            raise ValueError('Optimiser did not find a solution')

        # Remove edges not included from incidence matrix
        a_reduced = a[:, included_edges]
        a_sol = a_reduced.tolist()

        return a_sol

    def reduce_a(self, a, x):
        # reduce_a
        # Reduces incidence matrix to only included vertices and edges
        #
        # Inputs:   a   - dense coupled incidence matrix
        #           x   - Optimisation solution
        #
        # Outputs:  a_sol - incidence matrix with only included edges/vertices
        #                   list of lists
        #

        # Extract edges to delete from solution
        included_edges = [j for j in range(len(x)) if x[j] == 1]  # indices

        if not included_edges:
            raise ValueError('Optimiser did not find a solution')

        # Remove edges not included from incidence matrix
        a_reduced = a[:, included_edges]
        a_sol = a_reduced.tolist()

        return a_sol

    def label_active(self, g_nodes, out):

        # label_active
        # Data structure for cells that are 'active' in the tracking.
        # Holds what the cell's L label is in the previous image.
        #
        # Inputs:   g_nodes     - nodes in graph
        #           out         - current out data
        #

        active_cells = dict()

        # For each L node
        for vertex in g_nodes:
            if 'L' in vertex:
                # find track associated and label with cell name
                for cell_id, data in out['tracks'].items():
                    if (data['centroid_x'][-1] == g_nodes[vertex]['centroid_x']) & (
                            data['centroid_y'][-1] == g_nodes[vertex]['centroid_y']):
                        active_cells[cell_id] = vertex

        return active_cells

    def update_manual_small(self, g, out, final_choice, intersection_over_L_RL, intersection_over_R_LR, distance_dict_LR,
                      distance_dict_RL):

        # update_out
        # Given optimisation solution update the output data structure
        #
        # Inputs:   g           -   graph structure
        #           a_matrix    -   coupled incidence matrix
        #           x           -   solution from optimisation
        #           out         -   current output data structure
        #           c_vertices  -   order of vertices in coupled matrix
        #           intersection_over_L -
        #
        # Outputs:  update_out  -   updated output data
        #

        # Reduce incidence matrix to sol list of lists

        # Check if first output, else associate L cells with tracks
        # todo: active cells loops unnecessarily
        if not out:
            out = self.initialise_out(g.node)

        # Create dict of 'active cells' and their l_labels in previous frame
        active_cells = self.label_active(g.nodes, out)

        l_nodes = []
        for label in g.nodes():
            if 'L' in label:
                l_nodes.append(label)
        # Update frame number
        out['frame'] += 1

        # Update connections
        for r_node, l_node in final_choice.items():

            # Find new cells and the edge going to that cell

            # Cell moved
            if 'L' in str(l_node):
                self.update_cell_data(out, r_node, l_node, g.nodes, active_cells, intersection_over_L_RL,
                                      intersection_over_R_LR, distance_dict_LR, distance_dict_RL)

                try:
                    l_nodes.remove(l_node)
                except ValueError:
                    pass
        for r_node, l_node in final_choice.items():
            # Cell appeared
            if l_node == 'new':
                self.update_appear(out, r_node, g.nodes, intersection_over_L_RL, intersection_over_R_LR,
                                   distance_dict_LR, distance_dict_RL)
        if len(l_nodes) > 0:
            for l_node in l_nodes:
                if 'L' in str(l_node):
                    self.update_disappear(out, 'D', l_node, g.nodes, active_cells, intersection_over_L_RL,
                                          intersection_over_R_LR, distance_dict_LR, distance_dict_RL)

        return out

    def update(self, g, a_matrix, x, c_vertices):

        # update_out
        # Given optimisation solution update the output data structure
        #
        # Inputs:   g           -   graph structure
        #           a_matrix    -   coupled incidence matrix
        #           x           -   solution from optimisation
        #           out         -   current output data structure
        #           c_vertices  -   order of vertices in coupled matrix
        #           intersection_over_L -
        #
        # Outputs:  update_out  -   updated output data
        #

        # Reduce incidence matrix to sol list of lists
        res = {}
        # a_sol = self.reduce_a_pyomo(a_matrix, x)
        a_sol = self.reduce_a(a_matrix, x)

        # Check if first output, else associate L cells with tracks
        # out = self.initialise_out(g.node)

        # Create dict of 'active cells' and their l_labels in previous frame
        # active_cells = self.label_active(g.nodes, out)

        # Update frame number
        # out['frame'] += 1

        # Update connections
        for row, vertex in enumerate(c_vertices):

            # Find new cells and the edge going to that cell
            if 'R' in vertex:

                edge = a_sol[row].index(1)

                # Find labels for predecessors
                predecessors = [c_vertices[i] for i, v in enumerate(a_sol) if
                                v[edge] == -1]  # -1 in a_sol matrix mean L node

                # Simple cell movement
                if len(predecessors) == 1:
                    prev = predecessors.pop()
                    res[vertex] = prev
            elif 'D' in vertex:
                if np.any(a_sol[row]) == 1:
                    edge = a_sol[row].index(1)

                    # Find labels for predecessors
                    predecessors = [c_vertices[i] for i, v in enumerate(a_sol) if v[edge] == -1]

                    # Simple cell movement
                    if len(predecessors) == 1:
                        prev = predecessors.pop()
                        res[vertex] = prev

        return res

    def save_csv(self, output_data, save_path):

        # save_csv
        # save tracking output to csv file
        #
        # Inputs:   output_data - output data structure (list of lists)
        #           save_path   - dir to save csv inside
        #

        # Set save path
        save_out = save_path + '/output_data.csv'

        with open(save_out, 'wb') as f:
            writer = csv.writer(f)
            writer.writerows(output_data)

    def update_master_small(self, output_data):
        if output_data is not None:
            well = output_data['tracks'][0]['well'][0]
            # exp = output_data['tracks'][0]['experiment'][0]
            partial = self.df[(self.df.Sci_WellID == well)].copy()
            for j, track in output_data['tracks'].items():
                dict_len = len(track['timepoint'])
                for i in range(dict_len):
                    if i == 0:
                        partial.loc[track['index'][i],
                                    ['graph_id',
                                     'distance_travel',
                                     'intersection_over_region1',
                                     'intersection_over_region2',
                                     'confidence',
                                     'nearest_neighbor_from_prev',
                                     'second_nearest_neighbor_from_prev',
                                     'nearest_neighbor_from_current',
                                     'second_nearest_neighbor_from_current',
                                     'siamese_candidate']] = [track['GT_id'][i],
                                                              track['distance_travel'][i],
                                                              track['intersection_over_region1'][i],
                                                              track['intersection_over_region2'][i],
                                                              track['confidence'][i],
                                                              track['nearest_neighbor_from_prev'][i],
                                                              track[
                                                                  'second_nearest_neighbor_from_prev'][
                                                                  i],
                                                              track['nearest_neighbor_from_current'][
                                                                  i],
                                                              track[
                                                                  'second_nearest_neighbor_from_current'][
                                                                  i],
                                                              track['siamese_candidate'][i]
                                                              ]
                    else:
                        ground_truth_id = track['GT_id'][i]
                        if ground_truth_id != -1:  # -1 GT ID means track disappear, nothing to record
                            partial.loc[track['index'][i],
                                        ['graph_id',
                                         'distance_travel',
                                         'intersection_over_region1',
                                         'intersection_over_region2',
                                         'confidence',
                                         'nearest_neighbor_from_prev',
                                         'second_nearest_neighbor_from_prev',
                                         'nearest_neighbor_from_current',
                                         'second_nearest_neighbor_from_current',
                                         'siamese_candidate']] = [track['GT_id'][i - 1],
                                                                  track['distance_travel'][i],
                                                                  track['intersection_over_region1'][
                                                                      i],
                                                                  track['intersection_over_region2'][
                                                                      i],
                                                                  track['confidence'][i],
                                                                  track[
                                                                      'nearest_neighbor_from_prev'][
                                                                      i],
                                                                  track[
                                                                      'second_nearest_neighbor_from_prev'][
                                                                      i],
                                                                  track[
                                                                      'nearest_neighbor_from_current'][
                                                                      i],
                                                                  track[
                                                                      'second_nearest_neighbor_from_current'][
                                                                      i],
                                                                  track['siamese_candidate'][i]
                                                                  ]
            # assert partial.loc[partial.graph_id==-1]
            partial.graph_id_correct = partial.graph_id == partial.ObjectTrackID
            idx = partial.index
            self.df.loc[idx] = partial
            if self.savebool:
                self.df.to_csv(self.graph_csv, index=False)
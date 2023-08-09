import networkx as nx
import numpy as np
import logging

logger = logging.getLogger("Tracking")


class MinimumFlow:
    def __init__(self, g, verbose, debug):
        self.debug = debug
        self.verbose = verbose
        self.include_appear = True  # include appear disappear nodes
        self.g = g.copy()

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
                logger.info("Coupling matrix problems, there "
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
            v = [a_vertices[i] for i in vertex_indices[0]]

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
            if self.verbose and 0:
                if 'L' in v[0] and 'R' in v[1]:
                    l_x = self.g.nodes[v[0]]['centroid_x']
                    l_y = self.g.nodes[v[0]]['centroid_y']
                    r_x = self.g.nodes[v[1]]['centroid_x']
                    r_y = self.g.nodes[v[1]]['centroid_y']
                    logger.info(f'{v}, {v[0]}, {v[1]} cost: {cost}, {l_x} vs {r_x} {l_y} vs {r_y}')
            c.append(cost)

        return c
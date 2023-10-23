"""
Solver for minflow Linear Program based on gurobi
"""

from pyomo.environ import *
import numpy as np
from gurobipy import *
# from pyutilib.services import register_executable
# import pyutilib.subprocess.GlobalData

# pyutilib.subprocess.GlobalData.DEFINE_SIGNAL_HANDLERS_DEFAULT = False

# register_executable(name='gurobi')  # glpsol
# import logging
# logger = logging.getLogger("Tracking")

class SolverSmall:

    def __init__(self, debug, verbose):
        self.debug = debug
        self.verbose = verbose

    def opto(self, a_coup, b_flow, c_cost):
        # opto
        # Function to set up the model and call the optimisation engine
        #
        # Inputs:   a_coup      -   coupled incidence matrix, mxn
        #           b_flow      -   sum of flow for each vertex, m
        #           c_cost      -   vector of edge costs, n
        #
        # Outputs:  sol         -   solution vector
        #

        # build model
        # logger.info('Initializing model')
        b = np.array(b_flow)
        a = np.array(a_coup)

        model = Model()
        x = model.addMVar((len(c_cost),), vtype=GRB.BINARY)
        model.setObjective(sum(c_cost[j] * x[j] for j in range(len(c_cost))), sense=GRB.MINIMIZE)

        # model.addConstr(np.array(a_coup) @ x == np.array(b_flow), name='flow')
        for j in range(len(b)):
            if -1 <=b[j]<=1:
                model.addConstr(a[j,:] @ x== b[j])
            else:
                model.addConstr(a[j,:] @ x <= abs(b[j]))

        # model.addConstr(~(-1 <= b and b <= 1) >> a_coup @ x <= b)
        # logger.info('updating LP model')
        model.update()
        obj = model.getObjective()
        constrs = model.getConstrs()
        # for constr in constrs:
        #     print(constr)
        # logger.info('optimizing')
        model.optimize()
        # model.printAttr('X')

        # model = self.model_construct(a_coup, b_flow, c_cost)

        # if self.debug and self.verbose:
        #     # model.pprint()
        #     print('constrain')
        #     model.constrain.pprint()
        #     print('xx')
        #     print('objective')
        #     model.objective.pprint()
        #     model.x.pprint()
        #     model.x.display()
        #     print('model a')
        #     # for i in range(50):
        #     #     print(f'model a: 2 {i}', model.a[2, i])
        #     model.a.pprint()
        #     print('model b')
        #     model.b.pprint()
        #     # print('model c cost')
        #     # model.c.pprint()

        # solve
        # sol = self.solve(model)

        return model.X

if __name__=='__main__':
    Solve = SolverSmall()
    Solve.opto()
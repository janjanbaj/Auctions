"""
FILE DESCRIPTION:

This file implements the class WDP (Winner Determination Problem). This class is used for solving a winner determination problem given a finite sample of submitted XOR bids..
WDP has the following functionalities:
    0.CONSTRUCTOR: __init__(self, bids)
       bids = list of numpy nxd arrays representing elicited bundle-value pairs from each bidder. n=number of elicited bids, d = number of items + 1(value for that bundle).
    1.METHOD: initialize_mip(self, verbose=False)
        verbose = boolean, level of verbosity when initializing the MIP for the logger.
        This method initializes the winner determination problem as a MIP.
    2.METHOD: solve_mip(self)
        This method solves the MIP of the winner determination problem and sets the optimal allocation.
    3.METHOD: log_solve_details
        This method logs Solution details.
    4.METHOD: __repr__(self)
        Echoe on on your python shell when it evaluates an instances of this class.
    5.METHOD: print_optimal_allocation(self)
        This method printes the optimal allocation x_star in a nice way.
"""

# Libs
import numpy as np
import pandas as pd
import logging
# Gurobi: Here, Gurobipy is used for solving the deep neural network-based Winner Determination Problem.
import gurobipy as gp
from gurobipy import GRB
# documentation
# https://www.gurobi.com/documentation/9.5/refman/py_python_api_details.html
# %%


class MLCA_WDP:

    def __init__(self, bids):

        self.bids = bids  # list of numpy nxd arrays representing elicited bundle-value pairs from each bidder. n=number of elicited bids, d = number of items + 1(value for that bundle).
        self.N = len(bids)  # number of bidders
        self.M = bids[0].shape[1] - 1  # number of items
        self.Mip = gp.Model(name="WDP")  # gurobi model
        self.Mip.setParam("Threads", 1)  # Set Threads to 1 for better parallelization
        self.K = [x.shape[0] for x in bids]  # number of elicited bids per bidder
        self.z = {}  # decision variables. z(i,k) = 1 <=> bidder i gets the kth bundle out of 1,...,K[i] from his set of bundle-value pairs
        self.x_star = np.zeros((self.N, self.M), dtype=int)  # optimal allocation of the winner determination problem

    def initialize_mip(self, verbose=0):

        for i in range(0, self.N):  # over bidders i \in N
            # add decision variables
            self.z.update({(i, k): self.Mip.addVar(vtype=GRB.BINARY, name="z({},{})".format(i, k)) for k in range(0, self.K[i])})  # z(i,k) = 1 <=> bidder i gets bundle k \in Ki
            # add allocation constraints for z(i,k)
            self.Mip.addConstr((gp.quicksum(self.z[(i, k)] for k in range(self.K[i])) <= 1), name="CT Allocation Bidder {}".format(i))

        # add intersection constraints of buzndles for z(i,k)
        for m in range(0, self.M):  # over items m \in M
            self.Mip.addConstr((gp.quicksum(self.z[(i, k)]*self.bids[i][k, m] for i in range(0, self.N) for k in range(0, self.K[i])) <= 1), name="CT Intersection Item {}".format(m))

        # add objective
        objective = gp.quicksum(self.z[(i, k)]*self.bids[i][k, self.M] for i in range(0, self.N) for k in range(0, self.K[i]))
        self.Mip.setObjective(objective, GRB.MAXIMIZE)

        if verbose==1:
            constrs = self.Mip.getConstrs()
            for m in range(0, self.Mip.NumConstrs):
                    logging.debug('({}) %s'.format(m), constrs[m])
            logging.debug('\nMip initialized')

    def solve_mip(self, verbose=0):
        self.Mip.optimize()
        if verbose==1:
            self.log_solve_details(self.Mip)
        # set the optimal allocation
        for i in range(0, self.N):
            for k in range(0, self.K[i]):
                if self.z[(i, k)].X != 0:
                    self.x_star[i, :] = self.z[(i, k)].X*self.bids[i][k, :-1]

    def log_solve_details(self, solved_mip):
        # Gurobi stores details as model attributes
        logging.debug('Status  : %s', solved_mip.Status)
        logging.debug('Time    : %s sec', round(solved_mip.Runtime))
        logging.debug('Problem : %s', "MIP")
        try:
            logging.debug('Rel. Gap: {} %'.format(solved_mip.MIPGap))
        except AttributeError:
            logging.debug('Rel. Gap: N/A')
        logging.debug('N. Iter : %s', solved_mip.IterCount)
        logging.debug('Hit Lim.: %s', solved_mip.Status == GRB.TIME_LIMIT)
        try:
            logging.debug('Objective Value: %s', solved_mip.ObjVal)
        except AttributeError:
            logging.debug('Objective Value: N/A')

    def summary(self):
        print('################################ OBJECTIVE ################################')
        try:
            print('Objective Value: ', self.Mip.ObjVal, '\n')
        except Exception:
            print("Not yet solved!\n")
        print('############################# SOLVE STATUS ################################')
        print(f'Status: {self.Mip.Status}')
        try:
            print(f'Runtime: {self.Mip.Runtime}\n')
        except AttributeError:
            pass
        print('########################### ALLOCATED BIDDERs ############################')
        try:
            for i in range(0, self.N):
                for k in range(0, self.K[i]):
                    if self.z[(i, k)].X != 0:
                        print('z({},{})='.format(i, k), int(self.z[(i, k)].X))
        except Exception:
            print("Not yet solved!\n")
        print('########################### OPT ALLOCATION ###############################')
        self.print_optimal_allocation()
        return(' ')

    def print_optimal_allocation(self):
        D = pd.DataFrame(self.x_star)
        D.columns = ['Item_{}'.format(j) for j in range(1, self.M+1)]
        print(D)
        print('\nItems allocated:')
        print(D.sum(axis=0))

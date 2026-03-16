"""
See example_toy{i}_mip_test.py for examples of how to use this class.

"""

# Libs
import logging
from collections import OrderedDict

import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas as pd

# # CPLEX: Here, Gurobipy is used for solving the deep neural network-based Winner Determination Problem.
# documentation: http://ibmdecisionoptimization.github.io/py_python_api_details/mp/docplex.mp.model.html
# %% Neural Net Optimization Class


class MVNN_MIP_TORCH_NEW:
    '''
    This implements the class MVNN_MIP_TORCH.
    This class is used for solving MIP reformulations of the neural network-based Winner Determination Problems
        for Monotone-Value Neural Networks (MVNNS).
    The MVNNs are implemented via PYTORCH.
    '''

    def __init__(self,
                 models,
                 L=None):

        self.M = list(models[list(models.keys())[0]].parameters())[0].shape[1]  # number of items in the value model = dimension of input layer
        self.Models = models  # dict of pytorch models
        self.sorted_bidders = list(self.Models.keys())  # sorted list of bidders
        self.sorted_bidders.sort()
        self.N = len(models)  # number of bidders
        self.Mip = gp.Model(name="MVNN_MIP_NEW")  # docplex instance
        self.Mip.setParam("Threads", 1)  # Set Threads to 1 for better parallelization

        self.z = {}  # continous MIP variable
        self.alpha = {}  # binary MIP variable
        self.beta = {}  # binary MIP variable

        self.x_star = np.ones(shape=(self.N, self.M), dtype=int) * (-1)  # optimal allocation (-1=not yet solved)
        self.soltime = None  # timing

        self.case_counter = OrderedDict(
            [(bidder_name, {'Case1': 0, 'Case2': 0, 'Case3': 0, 'Case4': 0, 'Case5': 0}) for bidder_name, model in
             self.Models.items()])  # counts the removed UNN or ReLU cts
        self.ts = {}  # layerwise cutoffs per MVNN

    def calc_preactivated_box_bounds(self,
                                     verbose=False):

        # BOX-bounds for y variable (preactivated!!!!) as column vectors
        # Initialize
        upper_bound_input = [1] * self.M
        lower_bound_input = [0] * self.M
        self.upper_box_bounds = OrderedDict(list(
            (bidder_name, [np.array(upper_bound_input, dtype=int).reshape(-1, 1)]) for bidder_name in
            self.sorted_bidders))
        self.lower_box_bounds = OrderedDict(list(
            (bidder_name, [np.array(lower_bound_input, dtype=int).reshape(-1, 1)]) for bidder_name in
            self.sorted_bidders))
        # Propagate through Networks
        for bidder_name in self.sorted_bidders:
            weights, biases, ts = self._get_model_weights(bidder_name)
            for i in range(len(biases)):
                W, b = weights[i], biases[i].reshape(-1, 1)
                if i == 0:
                    self.upper_box_bounds[bidder_name].append(W @ self.upper_box_bounds[bidder_name][-1] + b)
                    self.lower_box_bounds[bidder_name].append(W @ self.lower_box_bounds[bidder_name][-1] + b)
                else:
                    t = ts[i - 1].reshape(-1, 1)
                    self.upper_box_bounds[bidder_name].append(
                        W @ self.phi(self.upper_box_bounds[bidder_name][-1], t) + b)
                    self.lower_box_bounds[bidder_name].append(
                        W @ self.phi(self.lower_box_bounds[bidder_name][-1], t) + b)
            if verbose:
                print(f'preactivated Upper Box Bounds for {bidder_name}:')
                print(self.upper_box_bounds[bidder_name])
            if verbose:
                print(f'preactivated Lower Box Bounds for {bidder_name}:')
                print(self.lower_box_bounds[bidder_name])


    def phi(self, x, t):
        # Bounded ReLU (bReLU) activation function for MVNNS with cutoff t (vectorized)
        return np.minimum(t, np.maximum(0, x)).reshape(-1, 1)

    def print_optimal_allocation(self):
        D = pd.DataFrame(self.x_star)
        D.columns = ['Item_{}'.format(j) for j in range(1, self.M + 1)]
        D.loc['Sum'] = D.sum(axis=0)
        print(D)

    def solve_mip(self,
                  log_output=False,
                  time_limit=None,
                  mip_relative_gap=None,
                  integrality_tol=None,
                  feasibility_tol=None,
                  mip_start=None):
        # add a warm start
        self.Mip.Params.LogToConsole = 1 if log_output else 0
        if mip_start is not None:
            for v in self.Mip.getVars():
                if v.VarName in mip_start:
                    v.Start = mip_start[v.VarName]
        if time_limit is not None:
            self.Mip.Params.TimeLimit = time_limit
        if mip_relative_gap is not None:
            self.Mip.Params.MIPGap = mip_relative_gap
        if integrality_tol is not None:
            self.Mip.Params.IntFeasTol = integrality_tol
        if feasibility_tol is not None:
            self.Mip.Params.FeasibilityTol = feasibility_tol

        logging.info('')
        logging.info('Solve MIP')
        logging.info('-----------------------------------------------')
        logging.info(f'MIP warm-start {bool(mip_start)}')
        logging.info('MIP time Limit of %s', getattr(self.Mip.Params, "TimeLimit", None))
        logging.info('MIP relative gap %s', getattr(self.Mip.Params, "MIPGap", None))
        logging.info('MIP integrality tol %s', getattr(self.Mip.Params, "IntFeasTol", None))
        logging.info('MIP feasibility tol %s', getattr(self.Mip.Params, "FeasibilityTol", None))

        self.Mip.optimize()
        
        unsatisfied_constraints = []
        if self.Mip.Status not in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL]:
            unsatisfied_constraints = ["Model is infeasible or unbounded"]
            
        logging.info(f'MIP unsatisfied constraints: {unsatisfied_constraints}')
        assert len(unsatisfied_constraints) == 0, \
            f'Solution does not satisfy {len(unsatisfied_constraints)} constraint(s).'
            
        Sol = {}
        if self.Mip.Status in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL] and self.Mip.SolCount > 0:
            for v in self.Mip.getVars():
                Sol[v.VarName] = v.X

        # get solution details
        try:
            self.soltime = self.Mip.Runtime
        except Exception:
            self.soltime = None
        mip_log = self.log_solve_details(self.Mip)
        # set the optimal allocation
        for i in range(0, self.N):
            for j in range(0, self.M):
                self.x_star[i, j] = int(self.z[(i, 0, j)].X)

            logging.debug('MIP Solution')
            logging.debug(f'Bidder {i} - {self.x_star[i, :].flatten().tolist()}')

        return Sol, mip_log

    def log_solve_details(self,
                          solved_mip):
        logging.info('\nSolve Details')
        logging.info('-----------------------------------------------')
        logging.info('Problem : %s', "MIP")
        logging.info('Status  : %s', solved_mip.Status)
        try:
            logging.info('Time    : %s sec', round(solved_mip.Runtime))
        except AttributeError:
            pass
        try:
            logging.info('Rel. Gap: {} %'.format(solved_mip.MIPGap))
        except AttributeError:
            logging.info('Rel. Gap: N/A %')
        logging.debug('N. Iter : %s', solved_mip.IterCount)
        logging.debug('Hit Lim.: %s', solved_mip.Status == GRB.TIME_LIMIT)
        try:
            logging.debug('Objective Value: %s', solved_mip.ObjVal)
        except AttributeError:
            logging.debug('Objective Value: N/A')
        logging.debug('')
        logging.debug('IA Case Statistics:')
        for bidder_name, v in self.case_counter.items():
            logging.debug(bidder_name)
            for k, v2 in v.items():
                logging.debug(f' - {k}: {v2}')
        logging.debug('\n')
        gap = solved_mip.MIPGap if hasattr(solved_mip, "MIPGap") else 0.0
        time_sec = solved_mip.Runtime if hasattr(solved_mip, "Runtime") else 0.0
        return {'n_iter': solved_mip.IterCount, 'rel. gap': gap,
                'hit_limit': solved_mip.Status == GRB.TIME_LIMIT, 'time': time_sec,
                'case_counter': self.case_counter}

    def summary(self):
        print('\n################################ SUMMARY ################################')
        print('----------------------------- OBJECTIVE --------------------------------')
        print(self.Mip.get_objective_expr(), '\n')
        try:
            print('Objective Value: ', self.Mip.objective_value, '\n')
        except Exception:
            print("Objective Value: Not yet solved!\n")
        print('----------------------------- SOLVE STATUS -----------------------------')
        print(self.Mip.get_solve_details())
        print(self.Mip.get_statistics())
        print()
        print('IA Case Statistics:')
        for bidder_name, v in self.case_counter.items():
            print(bidder_name)
            for k, v2 in v.items():
                print(f' - {k}: {v2}')
        print('\n')
        try:
            print(self.Mip.get_solve_status(), '\n')
        except AttributeError:
            print("Not yet solved!\n")
        print('----------------------------- OPT ALLOCATION ----------------------------')
        self.print_optimal_allocation()
        print('#########################################################################')
        print('\n')
        return (' ')

    def print_mip_constraints(self):
        print('\n############################### CONSTRAINTS ###############################')
        print('Notation: {variableName}_B{BidderId}_({HiddenLayer,Unit})')
        print('BidderId in {0,1,...,nBidders}')
        print('HiddenLayer in {1,...,nHiddenLayers}')
        print('Unit in {0,1,...,nUnits}')
        print()
        k = 0
        for m in range(0, self.Mip.number_of_constraints):
            if self.Mip.get_constraint_by_index(m) is not None:
                print('({}):   '.format(k), self.Mip.get_constraint_by_index(m))
                k = k + 1
        print('#########################################################################')
        print('\n')

    def _get_model_weights(self,
                           key,
                           layer_type=['layers']):
        weights = []
        biases = []
        ts = []
        for v in self.Models[key].layers:
            weights.append(v.weight.detach().cpu().numpy())
            biases.append(v.bias.detach().cpu().numpy())
            ts.append(v.ts.detach().cpu().numpy())

        weights.append(self.Models[key].output_layer.weight.detach().cpu().numpy())
        return weights, biases, ts

    def _get_model_layers(self,
                          key,
                          exc_layer_type=[]):
        layers = []
        for k, v in dict(self.Models[key].named_parameters()).items():
            if 'bias' not in k:
                if all(exc_type not in k for exc_type in exc_layer_type):
                    layers.append(v)
        return layers

    def _clean_weights(self,
                       Wb):
        for v in range(0, len(Wb) - 2, 2):
            Wb[v][abs(Wb[v]) <= 1e-8] = 0
            Wb[v + 1][abs(Wb[v + 1]) <= 1e-8] = 0
            zero_rows = np.where(np.logical_and((Wb[v] == 0).all(axis=0), Wb[v + 1] == 0))[0]
            if len(zero_rows) > 0:
                logging.debug('Clean Weights (rows) %s', zero_rows)
                Wb[v] = np.delete(Wb[v], zero_rows, axis=1)
                Wb[v + 1] = np.delete(Wb[v + 1], zero_rows)
                Wb[v + 2] = np.delete(Wb[v + 2], zero_rows, axis=0)
        return (Wb)

    def _add_matrix_constraints(self,
                                i,
                                verbose=False):
        layer = 1
        key = self.sorted_bidders[i]
        if verbose is True:
            logging.debug('\nAdd Matrix constraints: %s', key)

        # Wb = self._clean_weights(self._get_model_weights(key)) with weights cleaning
        weights, biases, ts = self._get_model_weights(key)

        for layer_idx in range(0, len(biases)):  # loop over layers
            if verbose is True:
                logging.debug('\nLayer: %s', layer)
            W = weights[layer_idx]
            if verbose is True:
                logging.debug('W: %s', W.shape)
            b = biases[layer_idx]
            # Negate the bias like in the layer
            if verbose is True:
                logging.debug('b: %s', b.shape)
            R, J = W.shape
            # decision variables
            if layer_idx == 0:
                self.z.update({(i, 0, j): self.Mip.addVar(vtype=GRB.BINARY, name=f"x_B{i}_(0,{j})") for j in
                               range(0, J)})  # binary variables for allocation
            self.z.update({(i, layer, r): self.Mip.addVar(vtype=GRB.CONTINUOUS, name=f"z_B{i}_({layer},{r})") for r in range(0,
                                                                                                               R)})  # output value variables after activation: remark one could add lb=0, ub=ts[layer-1] globally, not sure if this makes it faster or slower.

            # BUILD CONSTRAINTS FOR EACH NODE IN BIDDER {key}'s MVNN
            for r in range(0, R):
                # CASE 1 -> REMOVAL:
                if self.lower_box_bounds[key][layer][r, 0] >= ts[layer - 1][r]:
                    self.z[(i, layer, r)] = ts[layer - 1][r]
                    self.case_counter[key]['Case1'] += 1
                # CASE 2 -> REMOVAL:
                elif self.upper_box_bounds[key][layer][r, 0] <= 0:
                    self.z[(i, layer, r)] = 0
                    self.case_counter[key]['Case2'] += 1
                # CASE 3 -> REMOVAL:
                elif (self.lower_box_bounds[key][layer][r, 0] >= 0 and self.lower_box_bounds[key][layer][r, 0] <= ts[
                    layer - 1][r]) and (
                        self.upper_box_bounds[key][layer][r, 0] >= 0 and self.upper_box_bounds[key][layer][r, 0] <= ts[
                    layer - 1][r]):
                    # PRECALCULATION:
                    aff_linear_output = gp.quicksum(W[r, j] * self.z[(i, layer - 1, j)] for j in range(0, J)) + b[r]
                    # self.Mip.addConstr(self.z[(i, layer , r)] ==  aff_linear_output, name=f'Bidder{i}_Node({layer},{r})_Case3_CT1')
                    self.z[(i, layer, r)] = aff_linear_output
                    self.case_counter[key]['Case3'] += 1
                # CASE 4 -> REMOVAL:
                elif self.lower_box_bounds[key][layer][r, 0] >= 0:
                    # PRECALCULATION:
                    aff_linear_output = gp.quicksum(W[r, j] * self.z[(i, layer - 1, j)] for j in range(0, J)) + b[r]
                    # Initialize binary variable beta
                    self.beta.update({(i, layer, r): self.Mip.addVar(vtype=GRB.BINARY, name=f"beta_B{i}_({layer},{r})")})
                    # TYPE 1 Constraints for the whole network (except the output layer)
                    self.Mip.addConstr(self.z[(i, layer, r)] <= ts[layer - 1][r],
                                            name=f'Bidder{i}_Node({layer},{r})_Case4_CT1')
                    # TYPE 2 Constraints for the whole network (except the output layer)
                    self.Mip.addConstr(self.z[(i, layer, r)] <= aff_linear_output,
                                            name=f'Bidder{i}_Node({layer},{r})_Case4_CT2')
                    # TYPE 3 Constraints for the whole network (except the output layer)
                    self.Mip.addConstr(self.z[(i, layer, r)] >= self.beta[(i, layer, r)] * ts[layer - 1][r],
                                            name=f'Bidder{i}_Node({layer},{r})_Case4_CT3')
                    # TYPE 4 Constraints for the whole network (except the output layer)
                    self.Mip.addConstr(self.z[(i, layer, r)] >= aff_linear_output + (
                            ts[layer - 1][r] - self.upper_box_bounds[key][layer][r, 0]) * self.beta[(i, layer, r)],
                                            name=f'Bidder{i}_Node({layer},{r})_Case4_CT4')
                    self.case_counter[key]['Case4'] += 1
                # CASE 5 -> REMOVAL:
                elif self.upper_box_bounds[key][layer][r, 0] <= ts[layer - 1][r]:
                    # PRECALCULATION:
                    aff_linear_output = gp.quicksum(W[r, j] * self.z[(i, layer - 1, j)] for j in range(0, J)) + b[r]
                    # Initialize binary variable alpha
                    self.alpha.update({(i, layer, r): self.Mip.addVar(vtype=GRB.BINARY, name=f"alpha_B{i}_({layer},{r})")})
                    # TYPE 1 Constraints for the whole network (except the output layer)
                    self.Mip.addConstr(self.z[(i, layer, r)] <= self.alpha[(i, layer, r)] * ts[layer - 1][r],
                                            name=f'Bidder{i}_Node({layer},{r})_Case5_CT1')
                    # TYPE 2 Constraints for the whole network (except the output layer)
                    self.Mip.addConstr(
                        self.z[(i, layer, r)] <= aff_linear_output - self.lower_box_bounds[key][layer][r, 0] * (
                                1 - self.alpha[(i, layer, r)]), name=f'Bidder{i}_Node({layer},{r})_Case5_CT2')
                    # TYPE 3 Constraints for the whole network (except the output layer)
                    self.Mip.addConstr(self.z[(i, layer, r)] >= 0,
                                            name=f'Bidder{i}_Node({layer},{r})_Case5_CT3')
                    # TYPE 4 Constraints for the whole network (except the output layer)
                    self.Mip.addConstr(self.z[(i, layer, r)] >= aff_linear_output,
                                            name=f'Bidder{i}_Node({layer},{r})_Case5_CT4')
                    self.case_counter[key]['Case5'] += 1
                # DEFAULT CASE -> NO REMOVAL:
                else:
                    # PRECALCULATION:
                    aff_linear_output = gp.quicksum(W[r, j] * self.z[(i, layer - 1, j)] for j in range(0, J)) + b[r]
                    # Initialize binary variables alpha and beta
                    self.alpha.update({(i, layer, r): self.Mip.addVar(vtype=GRB.BINARY, name=f"alpha_B{i}_({layer},{r})")})
                    self.beta.update({(i, layer, r): self.Mip.addVar(vtype=GRB.BINARY, name=f"beta_B{i}_({layer},{r})")})
                    # TYPE 1 Constraints for the whole network (except the output layer)
                    self.Mip.addConstr(self.z[(i, layer, r)] <= self.alpha[(i, layer, r)] * ts[layer - 1][r],
                                            name=f'Bidder{i}_Node({layer},{r})_Default_CT1')
                    # TYPE 2 Constraints for the whole network (except the output layer)
                    self.Mip.addConstr(
                        self.z[(i, layer, r)] <= aff_linear_output - self.lower_box_bounds[key][layer][r, 0] * (
                                1 - self.alpha[(i, layer, r)]), name=f'Bidder{i}_Node({layer},{r})_Default_CT2')
                    # TYPE 3 Constraints for the whole network (except the output layer)
                    self.Mip.addConstr(self.z[(i, layer, r)] >= self.beta[(i, layer, r)] * ts[layer - 1][r],
                                            name=f'Bidder{i}_Node({layer},{r})_Default_CT3')
                    # TYPE 4 Constraints for the whole network (except the output layer)
                    self.Mip.addConstr(self.z[(i, layer, r)] >= aff_linear_output + (
                            ts[layer - 1][r] - self.upper_box_bounds[key][layer][r, 0]) * self.beta[(i, layer, r)],
                                            name=f'Bidder{i}_Node({layer},{r})_Default_CT4')

            layer += 1

        # Final output layer of bidder {key}'s MVNN
        if hasattr(self.Models[key], 'lin_skip_layer'):
            W = weights[-1]
            R, J = W.shape
            output_classic = gp.quicksum(W[0, j] * self.z[(i, layer - 1, j)] for j in range(0, J))
            lin_skip_W = self.Models[key].lin_skip_layer.weight.detach().cpu().numpy()
            R, J = lin_skip_W.shape
            output_lin_skip_conection = gp.quicksum(lin_skip_W[0, j] * self.z[(i, 0, j)] for j in range(0, J))
            self.z.update({(i, layer, 0): output_classic + output_lin_skip_conection})
        else:
            W = weights[-1]
            R, J = W.shape
            self.z.update({(i, layer, r): gp.quicksum(W[r, j] * self.z[(i, layer - 1, j)] for j in range(0, J)) for r in
                           range(0, R)})

    def initialize_mip(self,
                       verbose=False,
                       bidder_specific_constraints=None,
                       GSVM_specific_constraints=False,
                       national_circle_complement=None):
        # pay attention here order is important, thus first sort the keys of bidders!
        logging.info('')
        logging.info('Initialize MIP')
        logging.info('-----------------------------------------------')
        logging.debug('Sorted active bidders in MIP: %s', self.sorted_bidders)
        logging.debug('Calculate tight MVNN BOX-bounds:')

        # calculate tight BOX-bounds for MVNN
        self.calc_preactivated_box_bounds(verbose=verbose)

        # for each bidder i encode MVNN as MIP
        for i in range(0, self.N):
            self._add_matrix_constraints(i, verbose=verbose)
        # allocation constraints for x^i's
        for j in range(0, self.M):
            self.Mip.addConstr((gp.quicksum(self.z[(i, 0, j)] for i in range(0, self.N)) <= 1),
                                    name="FeasabilityCT_item_{}".format(j))
        # add bidder specific constraints
        if bidder_specific_constraints is not None:
            self._add_bidder_specific_constraints(bidder_specific_constraints)

        #  GSVM specific allocation constraints for regional and local bidder
        if GSVM_specific_constraints and national_circle_complement is not None:
            for i in range(0, self.N):
                # regional bidder
                if self.sorted_bidders[i] in ['Bidder_0', 'Bidder_1', 'Bidder_2', 'Bidder_3', 'Bidder_4', 'Bidder_5']:
                    logging.debug('Adding GSVM specific constraints for regional {}.'.format(self.sorted_bidders[i]))
                    self.Mip.addConstr((gp.quicksum(self.z[(i, 0, j)] for j in range(0, self.M)) <= 4),
                                            name="GSVM_CT_RegionalBidder{}".format(i))
                # national bidder
                elif self.sorted_bidders[i] in ['Bidder_6']:
                    logging.debug(
                        'Adding GSVM specific constraints for national {} with national circle complement {}.'.format(
                            self.sorted_bidders[i], national_circle_complement))
                    self.Mip.addConstr(
                        (gp.quicksum(self.z[(i, 0, j)] for j in national_circle_complement) == 0),
                        name="GSVM_CT_NationalBidder{}".format(i))
                else:
                    raise NotImplementedError(
                        'GSVM only implmented in default version for Regional Bidders:[Bidder_0,..,Bidder_5] and National Bidder: [Bidder_6]. You entered {}'.format(
                            self.sorted_bidders[i]))

        # add objective: sum of 1dim outputs of neural network per bidder z[(i,K_i,0)]
        objective = gp.quicksum(
            self.Models[self.sorted_bidders[i]]._target_max * self.z[
                (i, self.Models[self.sorted_bidders[i]]._num_hidden_layers + 1, 0)] for i in range(0, self.N))
        self.Mip.setObjective(objective, GRB.MAXIMIZE)
        logging.info('MIP initialized')

    def _add_bidder_specific_constraints(self,
                                         bidder_specific_constraints):
        for bidder_key, bundles in bidder_specific_constraints.items():
            bidder_id = np.where([x == bidder_key for x in self.sorted_bidders])[0][0]
            logging.debug('Adding bidder specific constraints')
            logging.debug(f'Id: {bidder_id}, Key: {bidder_key}')
            for idx, bundle in enumerate(bundles):
                logging.debug(f'{bundle}')
                # NEW LINEAR CUT
                self.Mip.addConstr(
                    gp.quicksum(self.z[(bidder_id, 0, j)]*int(2*(bundle[j]-0.5)) for j in range(0, self.M)) <= int(np.sum(bundle)-1),
                    name="BidderSpecificCT_Bidder{}_No{}".format(bidder_id, idx))
                '''
                # OLD INTEGER CUT
                self.Mip.addConstr(
                    (gp.quicksum((self.z[(bidder_id, 0, j)] == bundle[j]) for j in range(0, self.M)) <= self.M - 1),
                    name="BidderSpecificCT_Bidder{}_No{}".format(bidder_id, idx))
                '''

    def get_bidder_key_position(self,
                                bidder_key):
        return np.where([x == bidder_key for x in self.sorted_bidders])[0][0]

    def reset_mip(self):
        self.Mip = gp.Model(name="MVNN_MIP_NEW")

    def print_bounds(self):
        print('\n######################### BOUNDS & IA Case Statistics #########################')
        print('(preactivated) Upper Bounds z:')
        for k, v in pd.DataFrame({k: pd.Series(l) for k, l in self.upper_box_bounds.items()}).fillna('-').items():
            print(k)
            print(v.to_string())
        print('')
        print('(preactivated) Lower Bounds z:')
        for k, v in pd.DataFrame({k: pd.Series(l) for k, l in self.lower_box_bounds.items()}).fillna('-').items():
            print(k)
            print(v.to_string())
        print('\nIA Case Statistics:')
        for k, v in self.case_counter.items():
            print(f'     {k}:{v}')
        print('#############################################################################')
        print('\n')

# mlca_src/gurobi_wrapper.py

import warnings

import gurobipy as gp
from gurobipy import GRB


class ConstraintWrapper:
    """
    Handles the dual nature of equality in docplex:
    1. Acts as a constraint (TempConstr) when passed to add_constraint.
    2. Acts as a linear expression (0 or 1) when summed.
    """

    def __init__(self, constr, expr):
        self.constr = constr
        self.expr = expr

    def __add__(self, other):
        return self.expr + other

    def __radd__(self, other):
        return other + self.expr


class VarWrapper:
    """
    Wraps Gurobi Vars and LinExprs to provide:
    1. NoneType protection (treats None as 0).
    2. Docplex-like API (solution_value, name).
    3. Sticky wrapping (arithmetic returns wrappers).
    """

    def __init__(self, item, name=None):
        self.item = item  # Can be gurobipy.Var or gurobipy.LinExpr
        self._name = name

    def __hash__(self):
        return hash(self._name)

    @property
    def var(self):
        """Access underlying Gurobi object."""
        return self.item

    @property
    def solution_value(self):
        """Safely returns value from solution."""
        try:
            if isinstance(self.item, gp.Var):
                return self.item.X
            elif isinstance(self.item, gp.LinExpr):
                return self.item.getValue()
            return 0.0
        except:
            return 0.0

    @property
    def name(self):
        if self._name:
            return self._name
        if isinstance(self.item, gp.Var):
            return self.item.VarName
        return "Expr"

    def _unwrap(self, other):
        """Converts inputs to Gurobi objects, treating None as 0."""
        if other is None:
            return 0
        if isinstance(other, VarWrapper):
            return other.item
        if isinstance(other, ConstraintWrapper):
            return other.expr
        return other

    # --- Arithmetic (Returns VarWrapper to maintain protection) ---
    def __add__(self, other):
        return VarWrapper(self.item + self._unwrap(other))

    def __radd__(self, other):
        return VarWrapper(self._unwrap(other) + self.item)

    def __sub__(self, other):
        return VarWrapper(self.item - self._unwrap(other))

    def __rsub__(self, other):
        return VarWrapper(self._unwrap(other) - self.item)

    def __mul__(self, other):
        return VarWrapper(self.item * self._unwrap(other))

    def __rmul__(self, other):
        return VarWrapper(self._unwrap(other) * self.item)

    def __truediv__(self, other):
        return VarWrapper(self.item / self._unwrap(other))

    # --- Comparison (Returns Gurobi TempConstr or ConstraintWrapper) ---
    def __eq__(self, other):
        val = self._unwrap(other)
        constr = self.item == val

        # Safely determine if 'val' is 0 or 1 (handles standard ints, floats, AND numpy types)
        # Calling float() on a Gurobi Var or LinExpr raises a TypeError, which we catch.
        try:
            val_float = float(val)
            if val_float == 1.0:
                expr = self.item
            elif val_float == 0.0:
                expr = 1 - self.item
            else:
                expr = 0
        except (TypeError, ValueError):
            # If val is another Gurobi variable/expression, we don't need indicator logic
            expr = 0

        return ConstraintWrapper(constr, expr)

    def __le__(self, other):
        return self.item <= self._unwrap(other)

    def __ge__(self, other):
        return self.item >= self._unwrap(other)

    def __repr__(self):
        return str(self.item)


class SolveDetails:
    def __init__(self, model: gp.Model):
        model.update()
        self._model = model
        self.status = self._get_status_string()
        try:
            self.time = model.Runtime
        except:
            self.time = 0.0

        has_sol = getattr(model, "SolCount", 0) > 0
        try:
            self.mip_relative_gap = model.MIPGap if has_sol else float("inf")
        except (AttributeError, gp.GurobiError):
            self.mip_relative_gap = float("inf")
        self.nb_iterations = getattr(model, "IterCount", 0)
        self._time = self.time
        self.problem_type = "MIP"
        self.num_constrs = len(model.getConstrs())
        self.num_vars = len(model.getVars())

    def _get_status_string(self):
        s = self._model.Status
        if s == GRB.OPTIMAL:
            return "solution found"
        if s == GRB.TIME_LIMIT:
            return "time limit exceeded"
        if s == GRB.INFEASIBLE:
            return "infeasible"
        if s == GRB.UNBOUNDED:
            return "unbounded"
        return f"status {s}"

    def has_hit_limit(self):
        return self._model.Status == GRB.TIME_LIMIT


class Solution:
    def __init__(self, model_wrapper):
        self._model_wrapper = model_wrapper
        self.solve_details = model_wrapper.get_solve_details()

    @property
    def objective_value(self):
        return self._model_wrapper.objective_value

    def as_dict(self, keep_zeros=False):
        results = {}
        for v in self._model_wrapper._vars_list:
            val = v.solution_value
            if keep_zeros or abs(val) > 1e-6:
                results[v] = val
        return results


class ParameterWrapper:
    def __init__(self, model):
        self._model = model

    class GapHandler:
        def __init__(self, model):
            self._model = model

        def get(self):
            return self._model.Params.MIPGap

        def __repr__(self):
            return str(self._model.Params.MIPGap)

    @property
    def mipgap(self):
        return self.GapHandler(self._model)

    @mipgap.setter
    def mipgap(self, value):
        self._model.Params.MIPGap = value

    class IntegralityHandler:
        def __init__(self, model):
            self._model = model

        def set(self, value):
            self._model.Params.IntFeasTol = value

        def get(self):
            return self._model.Params.IntFeasTol

    @property
    def integrality(self):
        return self.IntegralityHandler(self._model)


class MIPParameters:
    def __init__(self, model):
        self.tolerances = ParameterWrapper(model)
        self.cuts = type("Cuts", (), {"flowcovers": 0, "mircuts": 0})()


class ParametersRoot:
    def __init__(self, model):
        self.mip = MIPParameters(model)


class Model:
    def __init__(self, name=""):
        self.mdl = gp.Model(name)
        # Prevent outputs from Gurobi! Noise!!
        self.mdl.setParam("OutputFlag", 0)
        self.parameters = ParametersRoot(self.mdl)
        self._constraints_list = []
        self._vars_list = []
        self._current_solution = None

    def binary_var(self, name=""):
        v = self.mdl.addVar(vtype=GRB.BINARY, name=name)
        wrapper = VarWrapper(v, name)
        self._vars_list.append(wrapper)
        return wrapper

    def continuous_var(self, lb=0, name=""):
        v = self.mdl.addVar(vtype=GRB.CONTINUOUS, lb=lb, name=name)
        wrapper = VarWrapper(v, name)
        self._vars_list.append(wrapper)
        return wrapper

    def sum(self, args):
        # Unwrap args for quicksum, handling ConstraintWrapper
        clean_args = []
        for x in args:
            if isinstance(x, VarWrapper):
                clean_args.append(x.item)
            elif isinstance(x, ConstraintWrapper):
                clean_args.append(x.expr)
            else:
                clean_args.append(x)
        return VarWrapper(gp.quicksum(clean_args))

    def add_constraint(self, ct, ctname=""):
        # Handle ConstraintWrapper from __eq__
        if isinstance(ct, ConstraintWrapper):
            ct = ct.constr

        c = self.mdl.addConstr(ct, name=ctname)
        self._constraints_list.append(c)
        return c

    def maximize(self, expr):
        real_expr = expr.item if isinstance(expr, VarWrapper) else expr
        self.mdl.setObjective(real_expr, GRB.MAXIMIZE)

    @property
    def solution(self):
        return self._current_solution

    def solve(self, log_output=False):
        self.mdl.setParam("LogToConsole", 1 if log_output else 0)
        self.mdl.optimize()
        if self.mdl.SolCount > 0:
            self._current_solution = Solution(self)
        else:
            self._current_solution = None
        return self._current_solution

    def get_solve_details(self):
        return SolveDetails(self.mdl)

    @property
    def solve_details(self):
        return self.get_solve_details()

    def set_time_limit(self, limit):
        self.mdl.setParam("TimeLimit", limit)

    def get_time_limit(self):
        return self.mdl.Params.TimeLimit

    def add_mip_start(self, mip_start):
        if mip_start:
            for v, val in mip_start.items():
                if isinstance(v, VarWrapper):
                    v.var.Start = val
                elif isinstance(v, gp.Var):
                    v.Start = val

    @property
    def number_of_constraints(self):
        return self.mdl.NumConstrs

    def get_constraint_by_index(self, idx):
        if 0 <= idx < len(self._constraints_list):
            return self._constraints_list[idx]
        return None

    @property
    def objective_value(self):
        try:
            return self.mdl.ObjVal
        except:
            raise Exception("No objective value available")

    def get_solve_status(self):
        return self.mdl.Status

    def get_statistics(self):
        return f"Gurobi Stats: {self.mdl.NumVars} vars, {self.mdl.NumConstrs} constrs"

    def get_objective_expr(self):
        return "Gurobi Objective Expression"

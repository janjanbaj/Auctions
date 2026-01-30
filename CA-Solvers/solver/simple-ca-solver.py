# parse file
from pulp import *

# Quick Functions:
add_constraints = lambda problem, constraints: [problem.addConstraint(c) for c in constraints]

# Parse File
with open("../../CATS/0000.txt", "r") as file:
    data = file.read()

data = list(filter(lambda x: '\t' in x ,data.split("\n")))
data = list(map(lambda x: x.split("\t"), data))
bid_val = {i[0]: i[1] for i in data}
bid_items = {i[0]: i[2:-1] for i in data}


item_overlap = {}
for bid, items in bid_items.items():
    for item in items:
        if item_overlap.get(item) is None:
            item_overlap[item] = []
        item_overlap[item].append(bid)

# Build Up LP
bid_vars = {i: LpVariable(f"bid_{i}",0 , 1, LpInteger) for i in bid_val.keys()}

non_overlaping_constraints = [lpSum([bid_vars[i] for i in bids]) <= 1  for item, bids in item_overlap.items()]

obj_function = lpSum([bid_vars[i] * float(bid_val[i]) for i in bid_val.keys()])

problem = LpProblem("Combinatorial Auction", LpMaximize)

# Add Objective Function to Maximize Profit
problem += (obj_function, "Max Profit")

add_constraints(problem, non_overlaping_constraints)

print(problem)

problem.solve()

for v in problem.variables():
    if v.varValue == 1:
        print(v.name, "=", v.varValue)


# Equivalent. Check if the LP file created by CATs is the same as the one our script did.
from docplex.mp.model_reader import ModelReader


m = ModelReader.read("../../CATS/0000.lp")
m.export_as_mps("./0000.mps")

variables, pulp_model = LpProblem.fromMPS("./0000.mps")

# You can now work with the pulp_model object as usual
print(f"Successfully read problem: {pulp_model.name}")

pulp_model.solve()


for v in pulp_model.variables():
    if v.varValue == 1:
        print(v.name, "=", v.varValue)

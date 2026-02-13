# Helper Functions and Classes:


## 1. CATs makes an LP Model that I want to solve using PULP: 
For the standard CATS model, it is the case that using the ``` -cplex``` flag, we generate a CPLEX compatible ```*.lp``` file. I currently use PuLP in Python for solving LPs so the following method helps me convert ```sh *.lp -> *.mps```.

- Install docplex and cplex: ```sh pip install cplex docplex```
- Use the following:
```python
from docplex.mp.model_reader import ModelReader
# Specify the path to your CPLEX LP file
lp_file_path = 'your_model.lp'
mps_file_path = 'your_model.mps'

# Read the LP file using docplex
m = ModelReader.read(lp_file_path)

# Export the model as an MPS file
m.export_as_mps(mps_file_path)

# Read the MPS file into a PuLP LpProblem object
# The function returns a dictionary of variables and the LpProblem object
variables, pulp_model = pl.LpProblem.fromMPS(mps_file_path)

# You can now work with the pulp_model object as usual
print(f"Successfully read problem: {pulp_model.name}")
# Example: print the objective
print(f"Objective: {pulp_model.objective}")

```



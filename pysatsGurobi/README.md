- An Extension to PySATs that uses Gurobi to Solve MIPs for all models.
- Major changes are made in the simple_model.py file.
- This change in PySATs is meant to be a drop in replacement for those who were unable to get the CPLEX license to solve larger instances of the MIPs required for SATs. In this work we use Gurobi.
- We still require the Community Edition of CPLEX to be compatible with older libraries that use CPLEX as is. 
- The workings of this library is as follows:
  - In ```simple_model.py``` we have modified ```get_efficient_allocation``` such that we export the MIP as a *.lp file:
      ```py        
        export_lp = os.path.abspath(f"sats_export_{uuid.uuid4().hex}.lp")
        java_path = autoclass("java.nio.file.Paths").get(export_lp)
        solver.exportToDisk(imip, java_path)
     ```
  Then, we wrap the values back into Java so that we do not break assumptions.

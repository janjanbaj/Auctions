# Optimization of MLCA Simulation Speed

We implemented a series of optimizations to significantly speed up MLCA simulations by parallelizing computationally intensive tasks and tuning the underlying MIP solver.

## Key Changes

### 1. Parallelized Auction Rounds
In [mlca_mechanism.py](file:///home/janeet/Code/Auctions/BOCA/mlca/mlca_mechanism.py), we replaced the sequential loops for generating marginal and main economy queries with a parallelized approach.
- **Gather-Parallel-Scatter**: The simulation now identifies all unique economies in each round, solves their "Main" optimizations in parallel, and then handles any necessary "Bidder Specific" optimizations also in parallel.
- **Log Management**: Combined the interleaved logs from parallel workers into a coherent [log.txt](file:///home/janeet/Code/Auctions/BOCA/results/LSVM/0.6/999/mean_model/log.txt) output.

### 2. Parallelized VCG Payments
In [mlca_economies.py](file:///home/janeet/Code/Auctions/BOCA/mlca/mlca_economies.py), the [calculate_vcg_payments](file:///home/janeet/Code/Auctions/BOCA/mlca/mlca_economies.py#2392-2454) method now solves the Winner Determination Problem (WDP) for all marginal economies in parallel. This significantly reduces the time spent at the end of each simulation.

### 3. Gurobi Threading Optimization
We modified [mlca_wdp.py](file:///home/janeet/Code/Auctions/BOCA/mlca/mlca_wdp.py) and [mvnn_mip_torch_new.py](file:///home/janeet/Code/Auctions/BOCA/mlca/mvnn_mip_torch_new.py) to set Gurobi's `Threads` parameter to `1`.
- **Why**: This prevents Gurobi from oversubscribing CPU cores during parallel execution, allowing multiple independent MIPs to run efficiently on separate cores simultaneously.

## Verification Results

We verified the implementation with a simulation run (`LSVM` domain).
- **Parallel Execution Confirmed**: Logs show multiple MIP solvers initializing and running concurrently.
- **Performance**: The "Main" optimizations for multiple economies are now solved in a single batch, utilizing all available CPU cores.
- **Functional Equivalence**: The logic for sampling and query generation remains functionally equivalent to the original sequential implementation.

## Files Modified
- [mlca_wdp.py](file:///home/janeet/Code/Auctions/BOCA/mlca/mlca_wdp.py): Set Gurobi Threads to 1.
- [mvnn_mip_torch_new.py](file:///home/janeet/Code/Auctions/BOCA/mlca/mvnn_mip_torch_new.py): Set Gurobi Threads to 1.
- [mlca_economies.py](file:///home/janeet/Code/Auctions/BOCA/mlca/mlca_economies.py): Implemented [solve_optimizations_parallel](file:///home/janeet/Code/Auctions/BOCA/mlca/mlca_economies.py#2281-2294), [_worker_optimization_step](file:///home/janeet/Code/Auctions/BOCA/mlca/mlca_economies.py#2182-2249), and updated [calculate_vcg_payments](file:///home/janeet/Code/Auctions/BOCA/mlca/mlca_economies.py#2392-2454).
- [mlca_mechanism.py](file:///home/janeet/Code/Auctions/BOCA/mlca/mlca_mechanism.py): Refactored the auction round loop for parallelization and added missing `OrderedDict` import.

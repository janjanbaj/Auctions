# Walkthrough - MLCA Progress Bar & QRound Clarification

I have improved the `tqdm` progress bar in the MLCA simulation and added documentation for the `Qround` parameter.

## Changes Made

### Improved Progress Bar
The progress bar in [mlca.py](file:///home/janeet/Code/Auctions/MVNN/mlca_src/mlca.py) now tracks the number of **elicited bids** (total queries) rather than auction rounds. 
- It initializes with the starting number of bids (`Qinit`).
- It increments by `Qround` (the number of queries per round) in each iteration.
- It provides a much more accurate representation of how close the auction is to reaching `Qmax`.

### QRound Documentation
- Added comments in [mlca.py](file:///home/janeet/Code/Auctions/MVNN/mlca_src/mlca.py) explaining that `Qround` is the number of bundles elicited from each bidder in each round.
- Updated the CLI help message in [simulation_mlca.py](file:///home/janeet/Code/Auctions/MVNN/simulation_mlca.py) to provide a clear description of `--qround`.

## Verification Results

### Manual Test Run
I ran a simulation with controlled parameters:
`python simulation_mlca.py --domain GSVM --qinit 5 --qround 2 --qmax 11 --seed 2003`

**Observation:**
1. The progress bar started at **5/11** (45%).
2. After the first round, it updated to **7/11**.
3. It eventually reached **11/11** (100%) and finished successfully.

```text
Elicited Bids: 100%|███████████████████████| 11/11 [01:36<00:00, 16.09s/it]
```

The progress bar is now fully informative and accurately reflects the auction's progress toward the query limit.


# Walkthrough - MLCA Progress Bar & QRound Clarification

I have improved the `tqdm` progress bar in the MLCA simulation and added documentation for the `Qround` parameter.

## Changes Made

### Improved Progress Bar
The progress bar in [mlca.py](file:///home/janeet/Code/Auctions/MVNN/mlca_src/mlca.py) now tracks the number of **elicited bids** (total queries) rather than auction rounds. 
- It initializes with the starting number of bids (`Qinit`).
- It increments by `Qround` (the number of queries per round) in each iteration.
- It provides a much more accurate representation of how close the auction is to reaching `Qmax`.

### QRound Documentation
- Added comments in [mlca.py](file:///home/janeet/Code/Auctions/MVNN/mlca_src/mlca.py) explaining that `Qround` is the number of bundles elicited from each bidder in each round.
- Updated the CLI help message in [simulation_mlca.py](file:///home/janeet/Code/Auctions/MVNN/simulation_mlca.py) to provide a clear description of `--qround`.

## Verification Results

### Manual Test Run
I ran a simulation with controlled parameters:
`python simulation_mlca.py --domain GSVM --qinit 5 --qround 2 --qmax 11 --seed 2003`

**Observation:**
1. The progress bar started at **5/11** (45%).
2. After the first round, it updated to **7/11**.
3. It eventually reached **11/11** (100%) and finished successfully.

```text
Elicited Bids: 100%|███████████████████████| 11/11 [01:36<00:00, 16.09s/it]
```

### MIP Variable and Constraint Accumulator
I implemented a global accumulator to track the total complexity of the models formulated during the auction.
- Added `total_vars` and `total_constr` to [MLCA_Economies](file:///home/janeet/Code/Auctions/MVNN/mlca_src/mlca_economies.py#69-841).
- Updated [gurobi_wrapper.py](file:///home/janeet/Code/Auctions/MVNN/mlca_src/gurobi_wrapper.py) to ensure [SolveDetails](file:///home/janeet/Code/Auctions/MVNN/mlca_src/gurobi_wrapper.py#130-165) reports accurate counts by calling `model.update()`.
- The final summary now includes the total number of variables and constraints formulated across all Rounds and WDP solves.

**Observation:**
The simulation output now includes:
```text
17:55:39: TOTAL VARIABLES FORMULATED: 60878
17:55:39: TOTAL CONSTRAINTS FORMULATED: 109088
```
This confirms that the counters are correctly accumulating work across iterations.


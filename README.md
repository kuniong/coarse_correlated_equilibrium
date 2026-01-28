# Opt-In Queue Regulation via Coarse Correlated Equilibrium Recommendations

This project implements the computational experiments for a callback-pool queueing model with information design, comparing **Correlated Equilibrium (CE)**, **Coarse Correlated Equilibrium (CCE)**, and **Selfish Full-Information (FI)** benchmarks.

## Overview

The model studies a queueing system where arriving customers can:
- **Join (J)**: Enter the queue immediately
- **Defer (D)**: Join a callback pool and wait for a callback
- **Balk (B)**: Take an outside option

The platform uses information design to recommend actions, and customers may comply with recommendations under different obedience constraints (CE vs CCE).

## Project Structure

```
CCE/
├── algorithms/              # Core algorithms
│   ├── __init__.py
│   ├── selfish_fi.py       # Algorithm 1: Selfish FI observable benchmark
│   └── ce_cce_solver.py    # Algorithm 2: Fixed-point solver for CE/CCE
├── utils/                   # Helper utilities
│   ├── __init__.py
│   ├── lp_solver.py        # Linear program formulation
│   ├── recursion.py        # Tagged-customer recursion
│   ├── metrics.py          # Metrics extraction
│   └── plotting.py         # Visualization utilities
├── experiments.ipynb        # Jupyter notebook for running experiments
└── README.md                # This file
```

## Two Core Algorithms

### 1. Algorithm 1: `selfish_FI_observable` (Selfish Full-Information Benchmark)

**Location**: `algorithms/selfish_fi.py`

**Purpose**: Computes the equilibrium when customers observe the full system state (n,m) and selfishly choose the action maximizing their utility.

**Algorithm**: Policy iteration with endogenous deferral utilities:
1. Initialize deferral utility guess u(D,n,m)
2. Compute greedy best-response policy for each state
3. Evaluate endogenous u(D,n,m) via tagged-customer recursion
4. Update policy and utilities until convergence

**Key Features**:
- Deterministic policy (one action per state)
- Fixed-point iteration with endogenous callback values
- Stationary distribution computed via CTMC balance equations

**Usage**:
```python
from algorithms import selfish_FI_observable

result = selfish_FI_observable(
    lam=0.8,      # Arrival rate λ
    mu=1.0,       # Service rate μ
    gamma=0.8,    # Callback intensity γ
    N=15,         # Max in-system occupancy
    M=10,         # Max callback pool size
    R=18.0,       # Service value (join)
    C=2.2,        # Waiting cost rate (join)
    rD=8.0,       # Service value (deferred)
    Cd=0.8,       # Waiting cost rate (deferred)
)

print(f"Welfare: {result['welfare']:.4f}")
print(f"p(J): {result['pJ']:.4f}, p(D): {result['pD']:.4f}, p(B): {result['pB']:.4f}")
```

### 2. Algorithm 2: `fixed_point_solve_CE_CCE` (CE/CCE Information Design)

**Location**: `algorithms/ce_cce_solver.py`

**Purpose**: Solves the nonlinear optimization problem for welfare-maximizing information design under CE or CCE obedience constraints.

**Algorithm**: Fixed-point iteration alternating between:
1. **LP Step**: Solve occupancy-measure LP given current u(D,n,m) table
2. **Recursion Step**: Evaluate new u(D,n,m) via tagged-customer recursion
3. **Update**: Apply damped update to u(D,n,m) until convergence

**Key Features**:
- Supports both CE (message-contingent) and CCE (ex ante) obedience
- Endogenous deferral utilities via tagged-customer recursion
- Damped fixed-point iteration for stability
- Warm-start capability for parameter sweeps

**Usage**:
```python
from algorithms import fixed_point_solve_CE_CCE
import pulp as pl

solver = pl.PULP_CBC_CMD(msg=False)

sol = fixed_point_solve_CE_CCE(
    mode="CE",        # or "CCE"
    lam=0.8,
    mu=1.0,
    gamma=0.8,
    N=15,
    M=10,
    R=18.0,
    C=2.2,
    rD=8.0,
    Cd=0.8,
    max_iter=40,      # Max fixed-point iterations
    tol=1e-6,         # Convergence tolerance
    damping=0.6,      # Damping factor (0.6 = 60% new, 40% old)
    solver=solver,
    warm_start=None,  # Optional: previous solution's uD table
)

print(f"Status: {sol['status']}")
print(f"Welfare: {sol['welfare']:.4f}")
print(f"Fixed-point iterations: {len(sol['fixed_point_history'])}")
```

## Installation

### Requirements

```bash
pip install numpy matplotlib pulp
```

### Optional: Install CBC Solver

For faster LP solving, install the CBC solver:
- **Windows**: Download from [COIN-OR](https://github.com/coin-or/Cbc/releases)
- **Mac/Linux**: `brew install cbc` or `apt-get install coinor-cbc`

## Running Experiments

Open and run `experiments.ipynb`:

```bash
jupyter notebook experiments.ipynb
```

The notebook runs three sensitivity experiments:
1. **Traffic intensity** (ρ = λ/μ): Varies from 0.6 to 0.98
2. **Service value ratio** (R/C): Varies from 2.0 to 9.0
3. **Callback intensity** (γ): Varies from 0.05 to 0.80

Each experiment produces 4 plots comparing CE, CCE, and Selfish FI:
- Welfare
- p(J) - Join probability
- p(D) - Defer probability
- p(B) - Balk probability



## Contact
Hung Q. Nguyen
nguyen.quoc.hung.xu@alumni.tsukuba.ac.jp


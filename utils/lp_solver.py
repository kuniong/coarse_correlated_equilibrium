"""
LP solver for the callback-pool queueing model with given endogenous deferral utilities.

This module provides the linear program formulation for both CE (correlated equilibrium)
and CCE (coarse correlated equilibrium) designs when the deferral utility u(D,n,m)
is treated as a fixed input table.
"""

import pulp as pl
from typing import Dict, Any, List, Tuple, Optional


def solve_correlated_queue_lp_callback_pool_given_uD(
    mode: str,                      # "CE" or "CCE"
    lam: float,
    mu: float,
    gamma: float,
    N: int,
    M: int,
    actions: List[str],
    uJ: Dict[Tuple[int, int], float],
    uB: Dict[Tuple[int, int], float],
    uD: Dict[Tuple[int, int], float],        # used in objective AND obedience
    solver: Optional[pl.LpSolver] = None,
    eps_support: float = 1e-12,
) -> Dict[str, Any]:
    """
    Solve the occupancy-measure LP for CE or CCE given fixed utility tables.
    
    Parameters:
    -----------
    mode : str
        Either "CE" (message-contingent obedience) or "CCE" (ex ante obedience)
    lam : float
        External arrival rate (λ)
    mu : float
        Service rate (μ)
    gamma : float
        Callback trigger intensity (γ)
    N : int
        Truncation level for in-system occupancy
    M : int
        Truncation level for callback pool size
    actions : List[str]
        List of actions ["J", "D", "B"]
    uJ : Dict[Tuple[int, int], float]
        Join utility table u(J, n, m)
    uB : Dict[Tuple[int, int], float]
        Balk utility table u(B, n, m)
    uD : Dict[Tuple[int, int], float]
        Deferral utility table u(D, n, m) (endogenous, but treated as input here)
    solver : Optional[pl.LpSolver]
        PuLP solver instance (default: CBC)
    eps_support : float
        Threshold for treating π_{n,m} as positive
    
    Returns:
    --------
    Dict containing:
        - mode: str
        - status: str (LP solver status)
        - params: dict of model parameters
        - x: occupancy measure x_{n,m,a}
        - pi: marginal state probabilities π_{n,m}
        - alpha: recommendation policy α_{n,m}(a)
        - welfare: objective value (arrival-based welfare)
    """
    if solver is None:
        solver = pl.PULP_CBC_CMD(msg=False)

    mode = mode.upper()
    if mode not in ("CE", "CCE"):
        raise ValueError("mode must be 'CE' or 'CCE'")

    A = actions
    states = [(n, m) for n in range(N + 1) for m in range(M + 1)]

    # x[n,m,a] = Pr(L=n,K=m,recommended/compliant action=a) in steady state (normalized to 1)
    x = pl.LpVariable.dicts(
        "x", (list(range(N + 1)), list(range(M + 1)), A),
        lowBound=0, cat="Continuous"
    )
    prob = pl.LpProblem(f"{mode}_CallbackPool_LP_FP", pl.LpMaximize)

    pi_expr = {(n, m): pl.lpSum(x[n][m][a] for a in A) for (n, m) in states}

    def u(a: str, n: int, m: int) -> float:
        if a == "J":
            return uJ[(n, m)]
        if a == "D":
            return uD[(n, m)]
        if a == "B":
            return uB[(n, m)]
        raise KeyError(a)

    # Paper objective: SW(x) = λ Σ x * u(a,n,m)
    prob += lam * pl.lpSum(
        x[n][m][a] * u(a, n, m)
        for (n, m) in states for a in A
    )

    # Normalization: Σ x = 1
    prob += pl.lpSum(x[n][m][a] for (n, m) in states for a in A) == 1.0, "norm"

    # Truncation feasibility: x_{N,m,J}=0 and x_{n,M,D}=0
    if "J" in A:
        for m in range(M + 1):
            prob += x[N][m]["J"] == 0.0, f"no_J_at_capacity_m{m}"
    if "D" in A:
        for n in range(N + 1):
            prob += x[n][M]["D"] == 0.0, f"no_D_at_poolcap_n{n}"

    # Stationary global balance (flow form)
    for n in range(N + 1):
        for m in range(M + 1):
            pi_nm = pi_expr[(n, m)]

            out_terms = []
            if n > 0:
                out_terms.append(mu * pi_nm)
            if "J" in A and n < N:
                out_terms.append(lam * x[n][m]["J"])
            if "D" in A and m < M:
                out_terms.append(lam * x[n][m]["D"])
            if m > 0:
                out_terms.append(gamma * m * pi_nm)
            outflow = pl.lpSum(out_terms) if out_terms else 0.0

            in_terms = []
            if n + 1 <= N:
                in_terms.append(mu * pi_expr[(n + 1, m)])
            if "J" in A and n > 0:
                in_terms.append(lam * x[n - 1][m]["J"])
            if "D" in A and m > 0:
                in_terms.append(lam * x[n][m - 1]["D"])
            if n > 0 and (m + 1) <= M:
                in_terms.append(gamma * (m + 1) * pi_expr[(n - 1, m + 1)])
            if n == N and (m + 1) <= M:
                in_terms.append(gamma * (m + 1) * pi_expr[(N, m + 1)])
            inflow = pl.lpSum(in_terms) if in_terms else 0.0

            prob += inflow == outflow, f"flow_n{n}_m{m}"

    # Obedience constraints (linear in x given u table)
    if mode == "CE":
        for a_rec in A:
            lhs = pl.lpSum(x[n][m][a_rec] * u(a_rec, n, m) for (n, m) in states)
            for a_dev in A:
                rhs = pl.lpSum(x[n][m][a_rec] * u(a_dev, n, m) for (n, m) in states)
                prob += lhs >= rhs, f"CE_{a_rec}_dev_{a_dev}"
    else:
        lhs_total = pl.lpSum(
            x[n][m][a] * u(a, n, m)
            for (n, m) in states for a in A
        )
        for a_dev in A:
            rhs = pl.lpSum(pi_expr[(n, m)] * u(a_dev, n, m) for (n, m) in states)
            prob += lhs_total >= rhs, f"CCE_dev_{a_dev}"

    status = prob.solve(solver)
    status_str = pl.LpStatus[status]
    if status_str not in ("Optimal", "Feasible"):
        raise RuntimeError(f"{mode} LP status: {status_str}")

    x_val = {(n, m, a): float(pl.value(x[n][m][a])) for (n, m) in states for a in A}
    pi_val = {(n, m): sum(x_val[(n, m, a)] for a in A) for (n, m) in states}

    alpha_val: Dict[Tuple[int, int], Dict[str, Optional[float]]] = {}
    for (n, m) in states:
        alpha_val[(n, m)] = {}
        if pi_val[(n, m)] > eps_support:
            for a in A:
                alpha_val[(n, m)][a] = x_val[(n, m, a)] / pi_val[(n, m)]
        else:
            for a in A:
                alpha_val[(n, m)][a] = None

    welfare = lam * sum(
        x_val[(n, m, a)] * u(a, n, m)
        for (n, m) in states for a in A
    )

    return {
        "mode": mode,
        "status": status_str,
        "params": {"lambda": lam, "mu": mu, "gamma": gamma, "N": N, "M": M, "actions": A},
        "x": x_val,
        "pi": pi_val,
        "alpha": alpha_val,
        "welfare": float(welfare),
    }

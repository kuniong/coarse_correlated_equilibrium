"""
Fixed-point solver for CE/CCE designs with endogenous deferral utilities.

This module implements the nonlinear fixed-point algorithm for solving
the CE (correlated equilibrium) and CCE (coarse correlated equilibrium)
design problems when deferral utilities are endogenous.
"""

import pulp as pl
from typing import Dict, Any, List, Tuple, Optional
from utils.lp_solver import solve_correlated_queue_lp_callback_pool_given_uD
from utils.recursion import evaluate_uD_via_recursion


def fixed_point_solve_CE_CCE(
    mode: str,
    lam: float,
    mu: float,
    gamma: float,
    N: int,
    M: int,
    R: float,
    C: float,
    rD: float,
    Cd: float,
    max_iter: int = 40,
    tol: float = 1e-6,
    damping: float = 0.6,
    solver: Optional[pl.LpSolver] = None,
    warm_start: Optional[Dict[Tuple[int, int], float]] = None,
) -> Dict[str, Any]:
    """
    Solve the nonlinear CE or CCE design problem via fixed-point iteration.
    
    The algorithm alternates between:
    1. Solving the LP given current u(D,n,m) table
    2. Evaluating u(D,n,m) via tagged-customer recursion under the resulting policy
    3. Updating u(D,n,m) with damping until convergence
    
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
    R : float
        Service value for immediate join
    C : float
        Waiting-cost rate for immediate join
    rD : float
        Deferred-service value (r_D in paper)
    Cd : float
        Waiting-cost rate for deferred customers (C_D in paper)
    max_iter : int
        Maximum number of fixed-point iterations
    tol : float
        Convergence tolerance (sup-norm on u(D,n,m) changes)
    damping : float
        Under-relaxation factor in (0,1] (0.6 = 60% new, 40% old)
    solver : Optional[pl.LpSolver]
        PuLP solver instance (default: CBC)
    warm_start : Optional[Dict[Tuple[int, int], float]]
        Initial guess for u(D,n,m) table
    
    Returns:
    --------
    Dict containing:
        - mode: str
        - status: str (LP solver status from last iteration)
        - params: dict of model parameters
        - x: occupancy measure x_{n,m,a}
        - pi: marginal state probabilities π_{n,m}
        - alpha: recommendation policy α_{n,m}(a)
        - welfare: objective value (arrival-based welfare)
        - uD_endogenous: final converged u(D,n,m) table
        - fixed_point_history: list of iteration diagnostics
    """
    actions = ["J", "D", "B"]

    uJ = {(n, m): R - C * (n + 1) / mu for n in range(N + 1) for m in range(M + 1)}
    uB = {(n, m): 0.0 for n in range(N + 1) for m in range(M + 1)}

    if warm_start is not None:
        uD = dict(warm_start)
    else:
        uD = {(n, m): (rD - Cd * (n + 1) / mu) for n in range(N + 1) for m in range(M + 1)}
        for n in range(N + 1):
            uD[(n, M)] = -1e9

    hist = []
    sol = None

    for it in range(max_iter):
        sol = solve_correlated_queue_lp_callback_pool_given_uD(
            mode=mode, lam=lam, mu=mu, gamma=gamma, N=N, M=M,
            actions=actions,
            uJ=uJ, uB=uB, uD=uD,
            solver=solver
        )

        uD_eval, _ = evaluate_uD_via_recursion(
            lam=lam, mu=mu, gamma=gamma, N=N, M=M,
            alpha=sol["alpha"], rD=rD, Cd=Cd
        )

        supdiff = 0.0
        for n in range(N + 1):
            for m in range(M):
                new_val = (1 - damping) * uD[(n, m)] + damping * uD_eval[(n, m)]
                supdiff = max(supdiff, abs(new_val - uD[(n, m)]))
                uD[(n, m)] = new_val

        hist.append({"iter": it, "welfare": sol["welfare"], "supdiff": supdiff})
        if supdiff < tol:
            break

    sol["fixed_point_history"] = hist
    sol["uD_endogenous"] = uD
    return sol

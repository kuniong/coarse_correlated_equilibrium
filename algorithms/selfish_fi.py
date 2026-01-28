"""
Selfish full-information (FI) observable benchmark.

This module implements the selfish best-response policy when customers
can observe the full system state (n, m) before making their decision.
"""

import numpy as np
from typing import Dict, Tuple
from utils.recursion import evaluate_uD_via_recursion


def stationary_ctmc_under_deterministic_policy(
    lam: float, mu: float, gamma: float, N: int, M: int,
    a_pol: Dict[Tuple[int, int], str],
) -> Dict[Tuple[int, int], float]:
    """
    Compute stationary distribution under a deterministic action policy.
    
    Parameters:
    -----------
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
    a_pol : Dict[Tuple[int, int], str]
        Deterministic policy mapping (n,m) -> action
    
    Returns:
    --------
    Dict[(n,m)] -> float : Stationary probability π_{n,m}
    """
    states = [(n, m) for n in range(N + 1) for m in range(M + 1)]
    idx = {(n, m): i for i, (n, m) in enumerate(states)}
    S = len(states)
    Q = np.zeros((S, S), dtype=float)

    for (n, m) in states:
        i = idx[(n, m)]
        out_rate = 0.0

        if n > 0:
            j = idx[(n - 1, m)]
            Q[i, j] += mu
            out_rate += mu

        a = a_pol[(n, m)]
        if a == "J" and n < N:
            j = idx[(n + 1, m)]
            Q[i, j] += lam
            out_rate += lam
        elif a == "D" and m < M:
            j = idx[(n, m + 1)]
            Q[i, j] += lam
            out_rate += lam

        if m > 0:
            rate_cb = gamma * m
            j = idx[(n + 1, m - 1)] if n < N else idx[(N, m - 1)]
            Q[i, j] += rate_cb
            out_rate += rate_cb

        Q[i, i] -= out_rate

    A = Q.T.copy()
    b = np.zeros(S, dtype=float)
    A[-1, :] = 1.0
    b[-1] = 1.0
    try:
        pi = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        pi, *_ = np.linalg.lstsq(A, b, rcond=None)

    pi = np.maximum(pi, 0.0)
    s = pi.sum()
    if s <= 0:
        return {(n, m): float("nan") for (n, m) in states}
    pi /= s
    return {(n, m): float(pi[idx[(n, m)]]) for (n, m) in states}


def selfish_FI_observable(
    lam: float,
    mu: float,
    gamma: float,
    N: int,
    M: int,
    R: float,
    C: float,
    rD: float,
    Cd: float,
    max_iter: int = 80,
    tol: float = 1e-10,
    damping: float = 1.0,
) -> Dict[str, float]:
    """
    Compute selfish full-information observable equilibrium.
    
    This implements policy iteration where customers observe the full state (n,m)
    and selfishly choose the action that maximizes their utility, with endogenous
    deferral utilities computed via the tagged-customer recursion.
    
    Parameters:
    -----------
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
        Maximum fixed-point iterations
    tol : float
        Convergence tolerance
    damping : float
        Damping factor for value updates (1.0 = no damping)
    
    Returns:
    --------
    Dict with keys:
        - welfare: float (arrival-based social welfare)
        - pJ: float (probability of joining)
        - pD: float (probability of deferring)
        - pB: float (probability of balking)
    """
    uJ = {(n, m): R - C * (n + 1) / mu for n in range(N + 1) for m in range(M + 1)}
    uB0 = 0.0

    uD_guess = {(n, m): (rD - Cd * (n + 1) / mu) for n in range(N + 1) for m in range(M + 1)}
    for n in range(N + 1):
        uD_guess[(n, M)] = -1e9

    def greedy_action(n: int, m: int) -> str:
        vals = []
        if n < N:
            vals.append(("J", uJ[(n, m)]))
        if m < M:
            vals.append(("D", uD_guess[(n, m)]))
        vals.append(("B", 0.0))
        # deterministic tie-breaking: B ≺ D ≺ J (prefer higher; break ties by this order)
        order = {"B": 0, "D": 1, "J": 2}
        best_a, best_v = vals[0]
        for a, v in vals[1:]:
            if v > best_v + 1e-15:
                best_a, best_v = a, v
            elif abs(v - best_v) <= 1e-15 and order[a] > order[best_a]:
                best_a, best_v = a, v
        return best_a

    a_pol = {(n, m): greedy_action(n, m) for n in range(N + 1) for m in range(M + 1)}

    for _ in range(max_iter):
        alpha = {}
        for n in range(N + 1):
            for m in range(M + 1):
                a = a_pol[(n, m)]
                alpha[(n, m)] = {"J": 1.0 if a == "J" else 0.0,
                                 "D": 1.0 if a == "D" else 0.0,
                                 "B": 1.0 if a == "B" else 0.0}

        uD_eval, _ = evaluate_uD_via_recursion(
            lam=lam, mu=mu, gamma=gamma, N=N, M=M,
            alpha=alpha, rD=rD, Cd=Cd
        )

        changed = 0
        maxdiff = 0.0
        for n in range(N + 1):
            for m in range(M + 1):
                old = uD_guess[(n, m)]
                uD_guess[(n, m)] = (1 - damping) * old + damping * uD_eval[(n, m)]
                maxdiff = max(maxdiff, abs(uD_guess[(n, m)] - old))

                new_a = greedy_action(n, m)
                if new_a != a_pol[(n, m)]:
                    a_pol[(n, m)] = new_a
                    changed += 1

        if changed == 0 and maxdiff < tol:
            break

    pi = stationary_ctmc_under_deterministic_policy(lam, mu, gamma, N, M, a_pol)

    pJ = sum(pi[(n, m)] for n in range(N + 1) for m in range(M + 1) if a_pol[(n, m)] == "J")
    pD = sum(pi[(n, m)] for n in range(N + 1) for m in range(M + 1) if a_pol[(n, m)] == "D")
    pB = sum(pi[(n, m)] for n in range(N + 1) for m in range(M + 1) if a_pol[(n, m)] == "B")

    welfare = 0.0
    for n in range(N + 1):
        for m in range(M + 1):
            a = a_pol[(n, m)]
            if a == "J":
                u_here = uJ[(n, m)]
            elif a == "D":
                u_here = uD_guess[(n, m)]
            else:
                u_here = 0.0
            welfare += pi[(n, m)] * u_here
    welfare *= lam

    return {"welfare": float(welfare), "pJ": float(pJ), "pD": float(pD), "pB": float(pB)}

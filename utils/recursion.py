"""
Tagged-customer recursion evaluator for endogenous deferral utilities.

This module solves the linear system for V^{cb}(n,m), the expected utility
of a tagged customer in the callback pool, as defined in the paper.
"""

import numpy as np
from typing import Dict, Tuple, Optional


def evaluate_uD_via_recursion(
    lam: float,
    mu: float,
    gamma: float,
    N: int,
    M: int,
    alpha: Dict[Tuple[int, int], Dict[str, Optional[float]]],
    rD: float,
    Cd: float,
) -> Tuple[Dict[Tuple[int, int], float], Dict[Tuple[int, int], float]]:
    """
    Solve linear system for V^{cb}(n,m), n=0..N, m=1..M, where m includes tagged customer.

    Matches Eq. (Vcb_recursion) in the paper with:
      v(n,m) = rD - Cd*(n+1)/mu for n<=N-1, and v(N,m)=0 (blocked => exits with 0)
      no running flow disutility while in callback pool

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
    alpha : Dict[Tuple[int, int], Dict[str, Optional[float]]]
        Recommendation policy α_{n,m}(a)
    rD : float
        Deferred-service value (r_D in paper)
    Cd : float
        Waiting-cost rate for deferred customers (C_D in paper)

    Returns:
    --------
    Tuple of:
        - uD: Dict[(n,m)] = V^{cb}(n,m+1) for m<=M-1; uD[(n,M)] = -inf sentinel
        - V_dict: Dict[(n,m)] = V^{cb}(n,m) for m>=1
    """
    idx, rev = {}, []
    k = 0
    for n in range(N + 1):
        for m in range(1, M + 1):
            idx[(n, m)] = k
            rev.append((n, m))
            k += 1

    K = k
    A = np.zeros((K, K), dtype=float)
    b = np.zeros(K, dtype=float)

    def get_alpha_prob(n: int, m: int, a: str) -> float:
        d = alpha.get((n, m), None)
        if d is None or d.get(a) is None:
            return 0.0 if a != "B" else 1.0
        return float(d[a])

    for (n, m) in rev:
        row = idx[(n, m)]
        aJ = get_alpha_prob(n, m, "J")
        aD = get_alpha_prob(n, m, "D")
        aB = get_alpha_prob(n, m, "B")

        if n == N:
            aJ = 0.0
        if m == M:
            aD = 0.0

        s = aJ + aD + aB
        if s <= 0.0:
            aB, s = 1.0, 1.0
        aJ, aD, aB = aJ / s, aD / s, aB / s

        q = lam + (mu if n > 0 else 0.0) + gamma * m

        v_here = (rD - Cd * (n + 1) / mu) if n <= N - 1 else 0.0

        # Eq form:
        # V = (mu/q)V(n-1,m) + (lam/q)[aB V + aJ V(n+1,m) + aD V(n,m+1)] + (gamma/q)v + (gamma(m-1)/q)V(min{n+1,N},m-1)
        A[row, row] = 1.0 - (lam * aB) / q
        b[row] = (gamma / q) * v_here

        if n > 0:
            A[row, idx[(n - 1, m)]] -= (mu / q)

        if n < N:
            A[row, idx[(n + 1, m)]] -= (lam * aJ) / q

        if m < M:
            A[row, idx[(n, m + 1)]] -= (lam * aD) / q

        if m >= 2:
            nxt = (n + 1, m - 1) if n < N else (N, m - 1)
            A[row, idx[nxt]] -= (gamma * (m - 1)) / q

    V_vec = np.linalg.solve(A, b)
    V_dict = {(n, m): float(V_vec[idx[(n, m)]]) for (n, m) in rev}

    uD: Dict[Tuple[int, int], float] = {}
    for n in range(N + 1):
        for m in range(M + 1):
            if m == M:
                uD[(n, m)] = -1e9
            else:
                uD[(n, m)] = V_dict[(n, m + 1)]
    return uD, V_dict

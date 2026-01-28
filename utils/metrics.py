"""
Metrics computation utilities for solution analysis.

This module provides helper functions to extract key metrics from
LP solutions and benchmark results.
"""

from typing import Dict, Any


def arrival_action_frequencies_from_x(sol: Dict[str, Any]) -> Dict[str, float]:
    """
    Extract arrival action frequencies (unconditional recommendation rates).
    
    Parameters:
    -----------
    sol : Dict[str, Any]
        Solution dict from LP solver containing 'x' and 'params'
    
    Returns:
    --------
    Dict[str, float] with keys "J", "D", "B" representing p(a) = Σ_{n,m} x_{n,m,a}
    """
    p = sol["params"]
    N, M = p["N"], p["M"]
    x = sol["x"]
    out = {}
    for a in p["actions"]:
        out[a] = sum(x[(n, m, a)] for n in range(N + 1) for m in range(M + 1))
    return out


def metrics_from_solution(sol: Dict[str, Any]) -> Dict[str, float]:
    """
    Extract key metrics from a solution.
    
    Parameters:
    -----------
    sol : Dict[str, Any]
        Solution dict from LP solver
    
    Returns:
    --------
    Dict with keys:
        - welfare: float
        - pJ: float (probability of joining)
        - pD: float (probability of deferring)
        - pB: float (probability of balking)
    """
    pA = arrival_action_frequencies_from_x(sol)
    return {
        "welfare": float(sol["welfare"]),
        "pJ": float(pA.get("J", 0.0)),
        "pD": float(pA.get("D", 0.0)),
        "pB": float(pA.get("B", 0.0)),
    }

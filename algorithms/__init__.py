"""
Algorithms package for callback-pool queueing model.

This package contains the two core algorithms:
- selfish_fi: Selfish full-information observable benchmark
- ce_cce_solver: Fixed-point solver for CE/CCE designs
"""

from .selfish_fi import selfish_FI_observable
from .ce_cce_solver import fixed_point_solve_CE_CCE

__all__ = [
    'selfish_FI_observable',
    'fixed_point_solve_CE_CCE',
]

"""CRIT v0.4: utility-sensitive transfer identifiability."""

from crit_v04.identifiability import (
    AmbiguityRegretResult,
    TransferIdentifiabilityResult,
    composition_automorphisms,
    composition_isomorphisms,
    minimax_transfer_regret,
    one_hot_orbit_regret,
    transfer_identifiability_gap,
    transport_utility,
)
from crit_v04.synthetic import run_identifiability_benchmark

__all__ = [
    "AmbiguityRegretResult",
    "TransferIdentifiabilityResult",
    "composition_automorphisms",
    "composition_isomorphisms",
    "minimax_transfer_regret",
    "one_hot_orbit_regret",
    "run_identifiability_benchmark",
    "transfer_identifiability_gap",
    "transport_utility",
]

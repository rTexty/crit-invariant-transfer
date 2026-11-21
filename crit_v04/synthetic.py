"""Exact finite families for the CRIT transfer-identifiability theorem."""

from __future__ import annotations

from itertools import permutations
from numbers import Integral
from typing import Any

import numpy as np

from crit_v04.identifiability import (
    composition_automorphisms,
    transfer_identifiability_gap,
)


def _composition_from_table(table: np.ndarray) -> np.ndarray:
    size = int(table.shape[0])
    tensor = np.zeros((size, size, size))
    for first in range(size):
        for second in range(size):
            tensor[first, second, int(table[first, second])] = 1.0
    return tensor


def _rigid_composition(
    size: int, rng: np.random.Generator, max_attempts: int = 10_000
) -> tuple[np.ndarray, int]:
    identity = (tuple(range(size)),)
    for attempt in range(1, max_attempts + 1):
        table = rng.integers(0, size, size=(size, size))
        composition = _composition_from_table(table)
        if composition_automorphisms(composition) == identity:
            return composition, attempt
    raise RuntimeError(f"failed to construct a rigid magma of size {size}")


def run_identifiability_benchmark(
    *,
    min_size: int = 2,
    max_size: int = 5,
    seed: int = 20260831,
) -> dict[str, Any]:
    """Verify rigid separation and a non-rigid counterexample exactly."""

    for name, value in (("min_size", min_size), ("max_size", max_size)):
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise ValueError(f"{name} must be an integer")
    if min_size < 2 or max_size > 5 or min_size > max_size:
        raise ValueError("benchmark sizes must satisfy 2 <= min_size <= max_size <= 5")
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")

    rng = np.random.default_rng(seed)
    rigid_cases: list[dict[str, Any]] = []
    errors: list[float] = []
    duality_gaps: list[float] = []
    constraint_violations: list[float] = []
    simplex_errors: list[float] = []
    for size in range(min_size, max_size + 1):
        composition, attempts = _rigid_composition(size, rng)
        automorphisms = composition_automorphisms(composition)
        blind = tuple(permutations(range(size)))
        utility = np.zeros(size)
        utility[0] = 1.0
        certificate = transfer_identifiability_gap(utility, blind, automorphisms)
        expected = 1.0 - 1.0 / size
        error = abs(certificate.gap - expected)
        errors.append(error)
        duality_gaps.extend(
            (certificate.blind.duality_gap, certificate.informed.duality_gap)
        )
        constraint_violations.extend(
            (
                certificate.blind.constraint_violation,
                certificate.informed.constraint_violation,
            )
        )
        simplex_errors.extend(
            (certificate.blind.simplex_error, certificate.informed.simplex_error)
        )
        rigid_cases.append(
            {
                "size": size,
                "automorphism_count": len(automorphisms),
                "blind_regret": certificate.blind.regret,
                "informed_regret": certificate.informed.regret,
                "gap": certificate.gap,
                "expected_gap": expected,
                "formula_error": error,
                "search_attempts": attempts,
            }
        )

    counter_size = min(max_size, max(min_size, 3))
    table = np.repeat(np.arange(counter_size)[:, None], counter_size, axis=1)
    left_zero = _composition_from_table(table)
    left_zero_automorphisms = composition_automorphisms(left_zero)
    blind = tuple(permutations(range(counter_size)))
    utility = np.zeros(counter_size)
    utility[0] = 1.0
    counterexample = transfer_identifiability_gap(
        utility, blind, left_zero_automorphisms
    )
    return {
        "rigid_cases": rigid_cases,
        "all_rigid": all(case["automorphism_count"] == 1 for case in rigid_cases),
        "max_formula_error": max(errors),
        "max_duality_gap": max(duality_gaps),
        "max_constraint_violation": max(constraint_violations),
        "max_simplex_error": max(simplex_errors),
        "left_zero_counterexample": {
            "size": counter_size,
            "automorphism_count": len(left_zero_automorphisms),
            "blind_regret": counterexample.blind.regret,
            "informed_regret": counterexample.informed.regret,
            "gap": counterexample.gap,
        },
        "seed": int(seed),
    }

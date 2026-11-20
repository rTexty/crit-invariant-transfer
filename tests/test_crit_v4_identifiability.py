from __future__ import annotations

from itertools import permutations

import numpy as np
import pytest

from crit_v04.identifiability import (
    composition_automorphisms,
    composition_isomorphisms,
    minimax_transfer_regret,
    one_hot_orbit_regret,
    transfer_identifiability_gap,
    transport_utility,
)


def deterministic_composition(table: list[list[int]]) -> np.ndarray:
    size = len(table)
    tensor = np.zeros((size, size, size))
    for first in range(size):
        for second in range(size):
            tensor[first, second, table[first][second]] = 1.0
    return tensor


def test_transport_utility_follows_source_to_target_permutation() -> None:
    utility = np.array([9.0, 4.0, -2.0])

    transported = transport_utility(utility, (2, 0, 1))

    assert transported == pytest.approx([4.0, -2.0, 9.0])
    with pytest.raises(ValueError):
        transported.flags.writeable = True


def test_rigid_composition_has_only_identity_automorphism() -> None:
    composition = deterministic_composition([[0, 0, 1], [1, 1, 2], [2, 0, 2]])

    automorphisms = composition_automorphisms(composition)

    assert automorphisms == ((0, 1, 2),)


def test_composition_automorphisms_preserve_all_three_axes() -> None:
    z3 = deterministic_composition([[0, 1, 2], [1, 2, 0], [2, 0, 1]])

    automorphisms = composition_automorphisms(z3)

    assert automorphisms == ((0, 1, 2), (0, 2, 1))


def test_cross_domain_isomorphism_respects_prior_hypothesis_set() -> None:
    source = deterministic_composition([[0, 0, 1], [1, 1, 2], [2, 0, 2]])
    true_mapping = (2, 0, 1)
    target = np.empty_like(source)
    target[np.ix_(true_mapping, true_mapping, true_mapping)] = source

    unrestricted = composition_isomorphisms(source, target)
    restricted = composition_isomorphisms(
        source, target, hypotheses=((0, 1, 2), true_mapping)
    )

    assert unrestricted == (true_mapping,)
    assert restricted == (true_mapping,)


@pytest.mark.parametrize("size", [2, 3, 4, 5])
def test_full_permutation_ambiguity_has_exact_one_hot_regret(size: int) -> None:
    utility = np.zeros(size)
    utility[0] = 1.0
    symmetric_group = tuple(permutations(range(size)))

    result = minimax_transfer_regret(utility, symmetric_group)

    assert result.regret == pytest.approx(1.0 - 1.0 / size, abs=1e-10)
    assert result.decision == pytest.approx(np.full(size, 1.0 / size), abs=1e-9)
    assert one_hot_orbit_regret(size, symmetric_group, 0) == pytest.approx(
        result.regret
    )


def test_rigid_composition_closes_the_transfer_identifiability_gap() -> None:
    utility = np.array([1.0, 0.0, 0.0])
    blind = tuple(permutations(range(3)))
    informed = ((0, 1, 2),)

    result = transfer_identifiability_gap(utility, blind, informed)

    assert result.blind.regret == pytest.approx(2.0 / 3.0)
    assert result.informed.regret == pytest.approx(0.0)
    assert result.gap == pytest.approx(2.0 / 3.0)


def test_utility_invariant_under_remaining_symmetry_needs_no_identification() -> None:
    utility = np.array([0.0, 1.0, 1.0])
    z3_automorphisms = ((0, 1, 2), (0, 2, 1))

    result = minimax_transfer_regret(utility, z3_automorphisms)

    assert result.regret == pytest.approx(0.0)
    assert result.decision[1] + result.decision[2] == pytest.approx(1.0)


def test_utility_sensitive_ambiguity_remains_when_optimum_moves_in_orbit() -> None:
    utility = np.array([0.0, 1.0, 0.0])
    z3_automorphisms = ((0, 1, 2), (0, 2, 1))

    result = minimax_transfer_regret(utility, z3_automorphisms)

    assert result.regret == pytest.approx(0.5)
    assert result.decision == pytest.approx([0.0, 0.5, 0.5], abs=1e-9)


@pytest.mark.parametrize(
    "utility,expected",
    [
        ([1e-12, 0.0], 5e-13),
        ([1e12, 1e12 + 1.0], 0.5),
        ([-7.0, -3.0], 2.0),
        ([5.0, 5.0], 0.0),
        ([-1e308, 1e308], 1e308),
    ],
)
def test_minimax_certificate_is_affine_scale_stable(
    utility: list[float], expected: float
) -> None:
    result = minimax_transfer_regret(utility, ((0, 1), (1, 0)))

    assert result.regret == pytest.approx(expected, rel=1e-9, abs=1e-15)
    assert result.decision == pytest.approx([0.5, 0.5], abs=1e-9)
    assert result.duality_gap <= 1e-9 * max(1.0, abs(expected))
    assert result.simplex_error <= 1e-10
    assert result.constraint_violation <= 1e-9


@pytest.mark.parametrize(
    "composition",
    [
        np.zeros((2, 2, 2)),
        -np.ones((2, 2, 2)),
        np.full((2, 2, 2), 0.6),
    ],
)
def test_composition_certificate_requires_probability_rows(
    composition: np.ndarray,
) -> None:
    with pytest.raises(ValueError):
        composition_automorphisms(composition)


def test_exact_automorphism_search_has_a_factorial_budget() -> None:
    size = 9
    table = [[first for _ in range(size)] for first in range(size)]

    with pytest.raises(ValueError, match="max_actions"):
        composition_automorphisms(deterministic_composition(table))


def test_hypothesis_and_action_budgets_count_consumed_inputs() -> None:
    duplicates = ((0, 1) for _ in range(11))
    with pytest.raises(ValueError, match="max_hypotheses"):
        minimax_transfer_regret([1.0, 0.0], duplicates, max_hypotheses=10)

    oversized = np.zeros(257)
    with pytest.raises(ValueError, match="max_actions"):
        minimax_transfer_regret(oversized, (tuple(range(257)),))


@pytest.mark.parametrize(
    "utility,hypotheses",
    [
        ([], ((0,),)),
        ([0.0, np.nan], ((0, 1),)),
        ([0.0, 1.0], ()),
        ([0.0, 1.0], ((0, 0),)),
        ([0.0, 1.0], ((0, 1, 2),)),
    ],
)
def test_invalid_ambiguity_problems_are_rejected(
    utility: list[float], hypotheses: tuple[tuple[int, ...], ...]
) -> None:
    with pytest.raises(ValueError):
        minimax_transfer_regret(utility, hypotheses)

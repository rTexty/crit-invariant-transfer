from __future__ import annotations

import pytest

from crit_v04.synthetic import run_identifiability_benchmark


@pytest.mark.slow
def test_rigid_algebra_family_realizes_exact_separation_theorem() -> None:
    summary = run_identifiability_benchmark(min_size=2, max_size=5, seed=20260831)

    assert summary["all_rigid"] is True
    assert summary["max_formula_error"] < 1e-9
    assert summary["max_duality_gap"] < 1e-9
    assert summary["max_constraint_violation"] < 1e-9
    assert summary["max_simplex_error"] < 1e-9
    for case in summary["rigid_cases"]:
        size = case["size"]
        assert case["automorphism_count"] == 1
        assert case["blind_regret"] == pytest.approx(1.0 - 1.0 / size)
        assert case["informed_regret"] == pytest.approx(0.0)
        assert case["gap"] == pytest.approx(1.0 - 1.0 / size)


def test_nonrigid_algebra_is_an_explicit_zero_gap_counterexample() -> None:
    summary = run_identifiability_benchmark(min_size=3, max_size=3, seed=7)

    counterexample = summary["left_zero_counterexample"]
    assert counterexample["automorphism_count"] == 6
    assert counterexample["blind_regret"] == pytest.approx(2.0 / 3.0)
    assert counterexample["informed_regret"] == pytest.approx(2.0 / 3.0)
    assert counterexample["gap"] == pytest.approx(0.0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min_size": 1},
        {"min_size": 4, "max_size": 3},
        {"max_size": 6},
        {"seed": -1},
    ],
)
def test_benchmark_controls_are_bounded(kwargs: dict[str, int]) -> None:
    with pytest.raises(ValueError):
        run_identifiability_benchmark(**kwargs)

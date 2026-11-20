"""Finite symmetry and minimax-regret certificates for transferable actions."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from numbers import Integral, Real
from typing import Iterable, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import linprog


FloatArray = NDArray[np.float64]
Permutation = tuple[int, ...]


def _readonly(value: ArrayLike) -> FloatArray:
    copied = np.array(value, dtype=np.float64, order="C", copy=True)
    return np.frombuffer(copied.tobytes(), dtype=np.float64).reshape(copied.shape)


def _utility(value: ArrayLike) -> FloatArray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf" or raw.dtype.kind == "b":
        raise ValueError("utility must contain real numeric values")
    result = np.array(raw, dtype=np.float64, copy=True)
    if result.ndim != 1 or result.size == 0 or not np.all(np.isfinite(result)):
        raise ValueError("utility must be a nonempty finite vector")
    return result


def _permutation(value: Sequence[int], size: int) -> Permutation:
    raw = tuple(value)
    if any(isinstance(item, bool) or not isinstance(item, Integral) for item in raw):
        raise ValueError("each hypothesis must contain integer indices")
    result = tuple(int(item) for item in raw)
    if len(result) != size or sorted(result) != list(range(size)):
        raise ValueError(f"each hypothesis must permute {size} actions")
    return result


def _positive_limit(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _hypotheses(
    values: Iterable[Sequence[int]],
    size: int,
    max_hypotheses: int = 100_000,
) -> tuple[Permutation, ...]:
    limit = _positive_limit(max_hypotheses, "max_hypotheses")
    unique: dict[Permutation, None] = {}
    for consumed, value in enumerate(values, start=1):
        if consumed > limit:
            raise ValueError("number of inputs exceeds max_hypotheses")
        unique[_permutation(value, size)] = None
    result = tuple(unique)
    if not result:
        raise ValueError("at least one transfer hypothesis is required")
    return result


def transport_utility(utility: ArrayLike, mapping: Sequence[int]) -> FloatArray:
    """Push a source utility through an old-index to target-index permutation."""

    source = _utility(utility)
    permutation = _permutation(mapping, source.size)
    target = np.empty_like(source)
    target[list(permutation)] = source
    return _readonly(target)


def _composition(value: ArrayLike, name: str) -> FloatArray:
    tensor = np.asarray(value)
    if tensor.dtype.kind not in "iuf" or tensor.dtype.kind == "b":
        raise ValueError(f"{name} must contain real numeric values")
    tensor = np.array(tensor, dtype=np.float64, copy=True)
    if (
        tensor.ndim != 3
        or tensor.shape[0] == 0
        or tensor.shape != (tensor.shape[0],) * 3
        or not np.all(np.isfinite(tensor))
    ):
        raise ValueError(f"{name} must be a finite nonempty cubic tensor")
    if np.any(tensor < 0.0) or not np.allclose(
        np.sum(tensor, axis=2), 1.0, atol=1e-12, rtol=0.0
    ):
        raise ValueError(f"{name} must contain normalized probability rows")
    return tensor


def _tolerance(value: float) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not np.isfinite(float(value))
        or value < 0.0
    ):
        raise ValueError("tolerance must be finite and nonnegative")
    return float(value)


def composition_automorphisms(
    composition: ArrayLike,
    *,
    tolerance: float = 0.0,
    max_actions: int = 8,
) -> tuple[Permutation, ...]:
    """Enumerate symmetries, with factorial work bounded by ``max_actions``.

    With positive ``tolerance`` the approximate matches need not form a group.
    """

    tensor = _composition(composition, "composition")
    tolerance = _tolerance(tolerance)
    limit = _positive_limit(max_actions, "max_actions")
    if tensor.shape[0] > limit:
        raise ValueError("composition size exceeds max_actions")
    result: list[Permutation] = []
    for candidate in permutations(range(tensor.shape[0])):
        index = np.asarray(candidate, dtype=np.intp)
        transformed = tensor[np.ix_(index, index, index)]
        if np.allclose(transformed, tensor, atol=tolerance, rtol=0.0):
            result.append(candidate)
    return tuple(result)


def composition_isomorphisms(
    source: ArrayLike,
    target: ArrayLike,
    *,
    hypotheses: Iterable[Sequence[int]] | None = None,
    tolerance: float = 0.0,
    max_actions: int = 8,
    max_hypotheses: int = 100_000,
) -> tuple[Permutation, ...]:
    """Filter hypotheses by composition, with bounded factorial exact search."""

    left = _composition(source, "source composition")
    right = _composition(target, "target composition")
    if left.shape != right.shape:
        raise ValueError("source and target composition sizes must match")
    tolerance = _tolerance(tolerance)
    limit = _positive_limit(max_actions, "max_actions")
    if left.shape[0] > limit:
        raise ValueError("composition size exceeds max_actions")
    hypothesis_limit = _positive_limit(max_hypotheses, "max_hypotheses")
    candidates: Iterable[Permutation]
    if hypotheses is None:
        candidates = permutations(range(left.shape[0]))
    else:
        candidates = _hypotheses(hypotheses, left.shape[0], hypothesis_limit)
    result: list[Permutation] = []
    for count, candidate in enumerate(candidates, start=1):
        if count > hypothesis_limit:
            raise ValueError("number of hypotheses exceeds max_hypotheses")
        index = np.asarray(candidate, dtype=np.intp)
        if np.allclose(
            right[np.ix_(index, index, index)],
            left,
            atol=tolerance,
            rtol=0.0,
        ):
            result.append(candidate)
    return tuple(result)


def one_hot_orbit_regret(
    action_count: int,
    hypotheses: Iterable[Sequence[int]],
    source_action: int,
    *,
    max_hypotheses: int = 100_000,
    max_actions: int = 256,
) -> float:
    """Closed form minimax regret for utility concentrated on one source action."""

    if (
        isinstance(action_count, bool)
        or not isinstance(action_count, Integral)
        or action_count < 1
    ):
        raise ValueError("action_count must be a positive integer")
    action_limit = _positive_limit(max_actions, "max_actions")
    if action_count > action_limit:
        raise ValueError("action_count exceeds max_actions")
    if (
        isinstance(source_action, bool)
        or not isinstance(source_action, Integral)
        or source_action < 0
        or source_action >= action_count
    ):
        raise ValueError("source_action must index the action set")
    candidates = _hypotheses(hypotheses, int(action_count), max_hypotheses)
    orbit = {candidate[int(source_action)] for candidate in candidates}
    return 1.0 - 1.0 / len(orbit)


@dataclass(frozen=True, slots=True, eq=False)
class AmbiguityRegretResult:
    regret: float
    decision: FloatArray
    adversary: FloatArray
    hypotheses: tuple[Permutation, ...]
    scenario_utilities: FloatArray
    duality_gap: float
    simplex_error: float
    constraint_violation: float


def minimax_transfer_regret(
    utility: ArrayLike,
    hypotheses: Iterable[Sequence[int]],
    *,
    max_hypotheses: int = 100_000,
    max_actions: int = 256,
) -> AmbiguityRegretResult:
    """Numerically solve the normalized primal and dual ambiguity games."""

    source = _utility(utility)
    action_limit = _positive_limit(max_actions, "max_actions")
    if source.size > action_limit:
        raise ValueError("utility size exceeds max_actions")
    candidates = _hypotheses(hypotheses, source.size, max_hypotheses)
    original_scenarios = np.stack(
        [transport_utility(source, candidate) for candidate in candidates]
    )
    utility_minimum = float(np.min(source))
    utility_maximum = float(np.max(source))
    if utility_maximum == utility_minimum:
        decision = np.full(source.size, 1.0 / source.size)
        adversary = np.full(len(candidates), 1.0 / len(candidates))
        return AmbiguityRegretResult(
            0.0,
            _readonly(decision),
            _readonly(adversary),
            candidates,
            _readonly(original_scenarios),
            0.0,
            0.0,
            0.0,
        )

    with np.errstate(over="ignore", invalid="ignore"):
        direct_range = utility_maximum - utility_minimum
    if np.isfinite(direct_range):
        scenarios = (original_scenarios - utility_minimum) / direct_range
        scale_first = direct_range
        scale_second = 1.0
    else:
        magnitude = float(np.max(np.abs(source)))
        scaled_scenarios = original_scenarios / magnitude
        scaled_minimum = float(np.min(scaled_scenarios))
        scaled_range = float(np.max(scaled_scenarios) - scaled_minimum)
        scenarios = (scaled_scenarios - scaled_minimum) / scaled_range
        scale_first = scaled_range
        scale_second = magnitude
    scenario_optima = np.max(scenarios, axis=1)
    loss = scenario_optima[:, None] - scenarios
    variable_count = source.size + 1
    objective = np.zeros(variable_count)
    objective[-1] = 1.0
    inequalities = np.concatenate((loss, -np.ones((len(candidates), 1))), axis=1)
    equality = np.zeros((1, variable_count))
    equality[0, : source.size] = 1.0
    solution = linprog(
        objective,
        A_ub=inequalities,
        b_ub=np.zeros(len(candidates)),
        A_eq=equality,
        b_eq=[1.0],
        bounds=[(0.0, 1.0)] * source.size + [(0.0, None)],
        method="highs",
    )
    if not solution.success or solution.x is None:
        raise RuntimeError(f"minimax regret optimization failed: {solution.message}")
    primal = np.asarray(solution.x, dtype=np.float64)
    if primal.shape != (variable_count,) or not np.all(np.isfinite(primal)):
        raise RuntimeError("minimax solver returned a malformed primal solution")
    decision = np.maximum(primal[: source.size], 0.0)
    decision_mass = float(np.sum(decision))
    if not np.isfinite(decision_mass) or decision_mass <= 0.0:
        raise RuntimeError("minimax solver returned an invalid decision")
    decision /= decision_mass
    normalized_regret = max(0.0, float(np.max(loss @ decision)))
    if abs(normalized_regret - float(primal[-1])) > 1e-7:
        raise RuntimeError("minimax primal epigraph is numerically inconsistent")

    dual_objective = np.zeros(len(candidates) + 1)
    dual_objective[-1] = -1.0
    dual_inequalities = np.concatenate((-loss.T, np.ones((source.size, 1))), axis=1)
    dual_equality = np.zeros((1, len(candidates) + 1))
    dual_equality[0, : len(candidates)] = 1.0
    dual_solution = linprog(
        dual_objective,
        A_ub=dual_inequalities,
        b_ub=np.zeros(source.size),
        A_eq=dual_equality,
        b_eq=[1.0],
        bounds=[(0.0, 1.0)] * len(candidates) + [(0.0, None)],
        method="highs",
    )
    if not dual_solution.success or dual_solution.x is None:
        raise RuntimeError(f"minimax dual optimization failed: {dual_solution.message}")
    dual = np.asarray(dual_solution.x, dtype=np.float64)
    if dual.shape != (len(candidates) + 1,) or not np.all(np.isfinite(dual)):
        raise RuntimeError("minimax solver returned a malformed dual solution")
    adversary = np.maximum(dual[: len(candidates)], 0.0)
    adversary_mass = float(np.sum(adversary))
    if not np.isfinite(adversary_mass) or adversary_mass <= 0.0:
        raise RuntimeError("minimax solver returned an invalid adversary")
    adversary /= adversary_mass
    normalized_dual_value = float(dual[-1])
    normalized_duality_gap = abs(normalized_regret - normalized_dual_value)
    duality_gap = normalized_duality_gap * scale_first * scale_second
    simplex_error = max(
        abs(float(np.sum(decision)) - 1.0),
        abs(float(np.sum(adversary)) - 1.0),
    )
    constraint_violation = max(
        0.0,
        float(np.max(loss @ decision - normalized_regret)),
        float(np.max(normalized_dual_value - loss.T @ adversary)),
    )
    if normalized_duality_gap > 1e-7 or constraint_violation > 1e-7:
        raise RuntimeError("minimax primal and dual solutions disagree")
    return AmbiguityRegretResult(
        normalized_regret * scale_first * scale_second,
        _readonly(decision),
        _readonly(adversary),
        candidates,
        _readonly(original_scenarios),
        duality_gap,
        simplex_error,
        constraint_violation,
    )


@dataclass(frozen=True, slots=True)
class TransferIdentifiabilityResult:
    blind: AmbiguityRegretResult
    informed: AmbiguityRegretResult
    gap: float


def transfer_identifiability_gap(
    utility: ArrayLike,
    blind_hypotheses: Iterable[Sequence[int]],
    informed_hypotheses: Iterable[Sequence[int]],
    *,
    max_hypotheses: int = 100_000,
    max_actions: int = 256,
) -> TransferIdentifiabilityResult:
    """Measure avoidable minimax regret removed by a richer observation model."""

    blind = minimax_transfer_regret(
        utility,
        blind_hypotheses,
        max_hypotheses=max_hypotheses,
        max_actions=max_actions,
    )
    informed = minimax_transfer_regret(
        utility,
        informed_hypotheses,
        max_hypotheses=max_hypotheses,
        max_actions=max_actions,
    )
    if not set(informed.hypotheses).issubset(blind.hypotheses):
        raise ValueError("informed hypotheses must refine the blind hypotheses")
    gap = blind.regret - informed.regret
    numerical_scale = max(1.0, abs(blind.regret), abs(informed.regret))
    uncertainty = (
        blind.duality_gap
        + informed.duality_gap
        + 16.0 * np.finfo(float).eps * numerical_scale
    )
    if gap < -uncertainty:
        raise RuntimeError("refining transfer hypotheses increased minimax regret")
    return TransferIdentifiabilityResult(blind, informed, max(0.0, gap))

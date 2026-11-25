"""Independent fluid-source and finite FIFO-target models for CRIT v0.8."""

from __future__ import annotations
from dataclasses import dataclass
from itertools import permutations
from typing import Mapping
import numpy as np


@dataclass(frozen=True)
class Metrics:
    cost: float
    throughput: int
    rejected: int
    overflow: int
    mean_latency: float
    p95_latency: float
    unfinished: int
    occupancy: float
    q_trajectory: tuple[float, ...] = ()


def _finite(name, x):
    a = np.asarray(x)
    if a.size == 0 or not np.all(np.isfinite(a)):
        raise ValueError(f"{name} must be non-empty and finite")
    return a


def source_run(arrivals, service_rate, threshold, *, arrival_fraction=1.0, cap=None):
    a = _finite("arrivals", arrivals).astype(float)
    cap = float(threshold if cap is None else cap)
    if (
        np.any(a < 0)
        or not np.isfinite(service_rate)
        or service_rate < 0
        or not np.isfinite(cap)
        or not (0 < cap and np.isfinite(threshold) and 0 <= threshold <= cap and np.isfinite(arrival_fraction) and 0 <= arrival_fraction <= 1)
    ):
        raise ValueError("invalid source parameters")
    q = 0.0
    lost = 0.0
    trace = []
    for raw in a:
        offered = arrival_fraction * raw
        admitted = min(offered, max(0.0, cap - q))
        lost += offered - admitted
        q = float(np.clip(q + admitted - service_rate, 0.0, cap))
        trace.append(q / cap)
    occ = float(np.mean(trace))
    total = float(np.sum(a) or 1.0)
    return {
        "occupancy": occ,
        "lost": lost,
        "cost": occ + 0.8 * lost / total,
        "q_trajectory": tuple(trace),
    }


def source_policy(x, h, *, source_high=0.95, source_low=0.55):
    if not (np.isfinite(x) and 0 <= x <= 1 and 0 <= h <= 1):
        raise ValueError("x and h must lie in [0,1]")
    return float(source_high if x < h else source_low)


def source_rollout(
    arrivals, service_rate, cap, h, *, source_high=0.95, source_low=0.55
):
    """Exact common-policy fluid dynamics: clip(q+a*arrival-service, 0, cap)."""
    a = _finite("arrivals", arrivals).astype(float)
    if (a.ndim != 1 or np.any(a < 0) or not np.isfinite(cap) or cap <= 0
            or not np.isfinite(service_rate) or service_rate < 0
            or not np.isfinite(h) or not 0 <= h <= 1
            or not np.isfinite(source_high) or not np.isfinite(source_low)
            or not 0 <= source_high <= 1 or not 0 <= source_low <= 1):
        raise ValueError("invalid source rollout parameters")
    q = 0.0
    trace = []
    lost = 0.0
    for arrival in a:
        fraction = source_policy(
            q / cap, h, source_high=source_high, source_low=source_low
        )
        offered = fraction * arrival
        # Source chronology is simultaneous balance: admit the offered fluid,
        # apply service, then count fluid above capacity as overflow. Rejected
        # demand is the unoffered (1-fraction) share.
        lost += (1.0 - fraction) * arrival
        q_next = float(np.clip(q + offered - service_rate, 0.0, cap))
        lost += max(0.0, q + offered - service_rate - cap)
        q = q_next
        trace.append(q / cap)
    total = float(np.sum(a) or 1.0)
    return {
        "q_trajectory": tuple(trace),
        "occupancy": float(np.mean(trace)),
        "lost": lost,
        "cost": float(np.mean(trace)) + 0.8 * lost / total,
    }


def source_fingerprint(arrivals, service_rate=2.0, threshold=4.0, scale=12.0):
    return np.asarray(
        source_run(arrivals, service_rate, threshold, cap=float(scale))["q_trajectory"]
    )


def source_signature(arrivals, action_fraction, *, cap=12.0, reset=(0.25, 0.75)):
    """Fluid counterpart of the two reset one-step target fingerprints."""
    a = _finite("arrivals", arrivals).astype(float)
    reset_values = np.asarray(reset, dtype=float)
    if (a.ndim != 1 or np.any(a < 0) or not np.isfinite(action_fraction)
            or not 0 <= action_fraction <= 1 or not np.isfinite(cap) or cap <= 0
            or reset_values.size == 0 or not np.all(np.isfinite(reset_values))
            or np.any((reset_values < 0) | (reset_values > 1))):
        raise ValueError("invalid source signature parameters")
    return np.asarray(
        [min(cap, x * cap + action_fraction * a[0]) / cap - x for x in reset_values]
    )


def _check_target(arrivals, uniforms, service_uniforms, capacity):
    a = _finite("arrivals", arrivals)
    u = _finite("uniforms", uniforms)
    su = _finite("service_uniforms", service_uniforms)
    if a.ndim != 1 or np.any(a < 0) or np.any(a != np.floor(a)):
        raise ValueError("arrivals must be nonnegative integers")
    if (
        u.ndim != 2
        or u.shape[0] != len(a)
        or u.shape[1] == 0
        or np.any((u < 0) | (u > 1))
    ):
        raise ValueError("invalid uniforms")
    if (
        su.ndim != 2
        or su.shape[0] != len(a)
        or su.shape[1] == 0
        or np.any((su < 0) | (su > 1))
    ):
        raise ValueError("invalid service uniforms")
    if int(capacity) != capacity or capacity <= 0 or np.any(a > u.shape[1]):
        raise ValueError("invalid capacity or insufficient uniforms")


def _target_actions(
    arrivals,
    uniforms,
    service_uniforms,
    actions,
    *,
    service_rate=2,
    queue_capacity=12,
    admission_levels=(0.55, 0.95),
    action_permutation=(0, 1),
    service_shift=1.0,
):
    _check_target(arrivals, uniforms, service_uniforms, queue_capacity)
    if (tuple(action_permutation) not in ((0, 1), (1, 0))
            or len(actions) != len(arrivals)
            or any(isinstance(x, (bool, np.bool_)) or not np.isfinite(x)
                   or int(x) != x or int(x) not in (0, 1) for x in actions)):
        raise ValueError("invalid actions or permutation")
    levels = tuple(float(x) for x in admission_levels)
    if len(levels) != 2 or any(not (0 <= x <= 1) for x in levels):
        raise ValueError("invalid admission levels")
    if (isinstance(service_rate, (bool, np.bool_)) or not np.isfinite(service_rate)
            or int(service_rate) != service_rate or service_rate < 0
            or not np.isfinite(service_shift) or service_shift <= 0):
        raise ValueError("invalid service parameters")
    q = []
    lat = []
    rejected = overflow = 0
    trace = []
    for t, raw in enumerate(np.asarray(arrivals, int)):
        frac = levels[action_permutation[int(actions[t])]]
        for i in range(raw):
            if uniforms[t, i] >= frac:
                rejected += 1
            elif len(q) >= queue_capacity:
                overflow += 1
            else:
                q.append(t)
        count = min(
            len(q),
            int(
                np.floor(
                    service_uniforms[t, 0] * (int(service_rate) + 1) / float(service_shift)
                )
            ),
        )
        for _ in range(count):
            lat.append(float(t - q.pop(0) + 1))
        trace.append(len(q) / queue_capacity)
    x = np.asarray(lat, float)
    throughput = len(lat)
    total = int(np.sum(arrivals))
    unfinished = len(q)
    assert rejected + overflow + throughput + unfinished == total
    return Metrics(
        float(np.mean(trace) + 0.8 * (rejected + overflow) / float(total or 1)),
        throughput,
        rejected,
        overflow,
        float(x.mean()) if len(x) else 0.0,
        float(np.percentile(x, 95)) if len(x) else 0.0,
        unfinished,
        float(np.mean(trace)),
        tuple(trace),
    )


def target_trace(arrivals, uniforms, service_uniforms, action, **kwargs):
    if isinstance(action, (bool, np.bool_)) or int(action) != action or int(action) not in (0, 1):
        raise ValueError("action must be 0 or 1")
    return np.asarray(
        _target_actions(
            arrivals, uniforms, service_uniforms, [action] * len(arrivals), **kwargs
        ).q_trajectory
    )


def target_fingerprint(
    arrivals, uniforms, service_uniforms, action, *, reset=(0.25, 0.75), **kwargs
):
    if isinstance(action, (bool, np.bool_)) or int(action) != action or int(action) not in (0, 1):
        raise ValueError("action must be 0 or 1")
    _check_target(
        arrivals, uniforms, service_uniforms, kwargs.get("queue_capacity", 12)
    )
    cap_value = kwargs.get("queue_capacity", 12)
    if not np.isfinite(cap_value) or int(cap_value) != cap_value or cap_value <= 0:
        raise ValueError("invalid queue capacity")
    cap = int(cap_value)
    levels = kwargs.get("admission_levels", (0.55, 0.95))
    perm = tuple(kwargs.get("action_permutation", (0, 1)))
    if perm not in ((0, 1), (1, 0)):
        raise ValueError("invalid action permutation")
    reset_values = np.asarray(reset, dtype=float)
    if reset_values.size == 0 or not np.all(np.isfinite(reset_values)) or np.any((reset_values < 0) | (reset_values > 1)):
        raise ValueError("invalid reset")
    out = []
    for x in reset_values:
        q0 = int(round(x * cap))
        accepted = sum(
            uniforms[0, i] < float(levels[perm[int(action)]])
            for i in range(int(arrivals[0]))
        )
        out.append((min(cap, q0 + accepted) - q0) / cap)
    return np.asarray(out)


def calibrate_fingerprints(seeds_data: Mapping[int, tuple], **kwargs):
    result = {0: [], 1: []}
    for a, u, su in seeds_data.values():
        for action in (0, 1):
            result[action].append(target_fingerprint(a, u, su, action, **kwargs))
    return {k: np.mean(v, axis=0) for k, v in result.items()}


def fit_action_map(target_fingerprints, source_fingerprints):
    if set(source_fingerprints) != {"low", "high"} or set(target_fingerprints) != {
        0,
        1,
    }:
        raise ValueError("expected two actions")
    shapes = []
    for values in list(target_fingerprints.values()) + list(source_fingerprints.values()):
        arr = np.asarray(values, dtype=float)
        if arr.size == 0 or not np.all(np.isfinite(arr)):
            raise ValueError("fingerprints must be non-empty and finite")
        shapes.append(arr.shape)
    if len(set(shapes)) != 1:
        raise ValueError("fingerprints must have the same shape")
    scores = []
    for p in permutations((0, 1)):
        m = dict(zip(("low", "high"), p))
        score = float(
            sum(
                np.mean(
                    (
                        np.asarray(target_fingerprints[a])
                        - np.asarray(source_fingerprints[label])
                    )
                    ** 2
                )
                for label, a in m.items()
            )
        )
        scores.append((score, m))
    scores.sort(key=lambda z: (z[0], tuple(z[1].values())))
    return scores[0][1], scores[0][0], scores[1][0] - scores[0][0]


def make_randomness(seed, horizon=60, max_arrivals=8):
    if horizon <= 0 or max_arrivals <= 0:
        raise ValueError("horizon and max_arrivals must be positive")
    rng = np.random.default_rng(seed)
    return (
        np.minimum(rng.poisson(4.0, horizon), max_arrivals).astype(int),
        rng.random((horizon, max_arrivals)),
        rng.random((horizon, 1)),
    )


def threshold_policy(source_seeds, thresholds, *, service_rate=0.0, cap=None,
                     arrivals_by_seed=None, return_scores=False):
    if not source_seeds or not thresholds:
        raise ValueError("source_seeds and thresholds must be non-empty")
    if service_rate < 0 or not np.isfinite(service_rate):
        raise ValueError("invalid source service rate")
    if any(not np.isfinite(h) or not 0 <= h <= 1 for h in thresholds):
        raise ValueError("thresholds must lie in [0,1]")
    cap = float(max(thresholds) if cap is None else max(cap, max(thresholds)))
    data = arrivals_by_seed or {s: make_randomness(s)[0] for s in source_seeds}
    scores = {
        float(h): float(np.mean([
            source_rollout(data[s], service_rate, cap, h)["cost"]
            for s in source_seeds
        ])) for h in thresholds
    }
    winner = min(scores, key=lambda h: (scores[h], h))
    return (winner, scores) if return_scores else winner


def run_policy(
    arrivals, uniforms, service_uniforms, source_threshold, action_map, **kwargs
):
    cap_value = kwargs.get("queue_capacity", 12)
    rate_value = kwargs.get("service_rate", 2)
    if (not np.isfinite(cap_value) or int(cap_value) != cap_value or cap_value <= 0
            or isinstance(rate_value, (bool, np.bool_)) or not np.isfinite(rate_value)
            or int(rate_value) != rate_value or rate_value < 0):
        raise ValueError("invalid target configuration")
    cap = int(cap_value)
    levels = kwargs.get("admission_levels", (0.55, 0.95))
    rate = int(rate_value)
    shift = float(kwargs.get("service_shift", 1.0))
    perm = tuple(kwargs.get("action_permutation", (0, 1)))
    if perm not in ((0, 1), (1, 0)):
        raise ValueError("invalid action permutation")
    if (not np.isfinite(source_threshold) or not 0 <= source_threshold <= 1
            or set(action_map) != {"low", "high"}
            or any(isinstance(action_map[k], (bool, np.bool_)) or not np.isfinite(action_map[k])
                   or int(action_map[k]) != action_map[k] or int(action_map[k]) not in (0, 1)
                   for k in ("low", "high"))):
        raise ValueError("invalid policy")
    q = []
    actions = []
    for t, raw in enumerate(np.asarray(arrivals, int)):
        label = "high" if len(q) / cap < source_threshold else "low"
        action = int(action_map[label])
        actions.append(action)
        frac = float(levels[perm[action]])
        for i in range(raw):
            if uniforms[t, i] < frac and len(q) < cap:
                q.append(t)
        count = min(len(q), int(np.floor(service_uniforms[t, 0] * (rate + 1) / shift)))
        del q[:count]
    return _target_actions(arrivals, uniforms, service_uniforms, actions, **kwargs)


def _target(
    arrivals,
    uniforms,
    service_uniforms,
    *,
    admission,
    service_rate,
    queue_capacity,
    **kwargs,
):
    return _target_actions(
        arrivals,
        uniforms,
        service_uniforms,
        [0] * len(arrivals),
        service_rate=service_rate,
        queue_capacity=queue_capacity,
        admission_levels=(admission, admission),
        **kwargs,
    )

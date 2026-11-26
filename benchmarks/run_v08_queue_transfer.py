"""Frozen paired descriptive v0.8 queue-transfer benchmark (not a theorem)."""

from __future__ import annotations
import json
import sys
import argparse
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from crit_v08.queue_transfer import (
    make_randomness,
    source_signature,
    fit_action_map,
    threshold_policy,
    target_fingerprint,
    run_policy,
    _target,
    _target_actions,
)

REGIMES = {
    "balanced": (2, 12, 1.0),
    "overload": (1, 8, 1.0),
    "service_shift": (2, 12, 2.0),
}
FIELDS = ("cost", "throughput", "rejected", "overflow", "mean_latency", "p95_latency", "unfinished")
METHODS = ("fixed_high", "identity", "random", "native", "transfer", "oracle")
THRESHOLDS = (0.1, 0.3, 0.5, 0.7)


def _bootstrap_ci(values, *, seed, draws=2000):
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or not len(x):
        raise ValueError("values must be a non-empty vector")
    rng = np.random.default_rng(seed)
    means = np.mean(x[rng.integers(0, len(x), size=(draws, len(x)))], axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def _native_threshold(train, *, rate, cap, shift, permutation):
    candidates = THRESHOLDS
    scores = []
    for h in candidates:
        for mapping in ({"low": 0, "high": 1}, {"low": 1, "high": 0}):
            vals = [run_policy(*make_randomness(seed), float(h), mapping,
                    service_rate=rate, queue_capacity=cap, service_shift=shift,
                    action_permutation=permutation).cost for seed in train]
            scores.append((float(np.mean(vals)), float(h), mapping))
    return min(scores, key=lambda item: (item[0], item[1], tuple(item[2].values())))


def run_benchmark(seed=20260905, seeds_per_regime=20):
    source_seeds = [seed + 100000 + i for i in range(16)]
    source_rng = {s: np.random.default_rng(s) for s in source_seeds}
    source_arrivals = {s: source_rng[s].uniform(2.0, 6.0, 60) for s in source_seeds}
    source_fit, source_candidate_costs = threshold_policy(
        source_seeds, THRESHOLDS, service_rate=2.0, cap=12.0,
        arrivals_by_seed=source_arrivals, return_scores=True)
    out = {"seed": seed, "seeds_per_regime": seeds_per_regime,
           "source_fit_seeds": source_seeds, "source_threshold_grid": list(THRESHOLDS),
           "source_fit_config": {"service_rate": 2.0, "capacity": 12.0,
                                 "arrival": "Uniform(2,6), horizon=60"},
           "source_candidate_costs": source_candidate_costs,
           "regimes": {}}
    for ri, (name, (rate, cap, shift)) in enumerate(REGIMES.items()):
        cal = [seed + 1000 * ri + i for i in range(8)]
        train = [seed + 1000 * ri + 40 + i for i in range(12)]
        test = [seed + 1000 * ri + 100 + i for i in range(seeds_per_regime)]
        perm = tuple(np.random.default_rng(seed + ri).permutation(2))
        cal_data = {s: make_randomness(s) for s in cal}
        target_fp = {
            a: np.mean(
                [
                    target_fingerprint(
                        *cal_data[s],
                        a,
                        action_permutation=perm,
                        service_rate=rate,
                        queue_capacity=cap,
                        service_shift=shift,
                    )
                    for s in cal
                ],
                axis=0,
            )
            for a in (0, 1)
        }
        # Source calibration is independent of target calibration and uses no target reward.
        source_data = {s: make_randomness(s + 50000) for s in cal}
        sf = {
            "low": np.mean(
                [source_signature(source_data[s][0], 0.55, cap=cap) for s in cal], axis=0
            ),
            "high": np.mean(
                [source_signature(source_data[s][0], 0.95, cap=cap) for s in cal], axis=0
            ),
        }
        amap, score, gap = fit_action_map(target_fp, sf)
        oracle_map = {"low": perm.index(0), "high": perm.index(1)}
        threshold = source_fit
        native_cost, native_h, native_map = _native_threshold(
            train, rate=rate, cap=cap, shift=shift, permutation=perm)
        per_method = {k: [] for k in METHODS}
        for s in test:
            a, u, su = make_randomness(s)
            kw = {"service_rate": rate, "queue_capacity": cap, "service_shift": shift}
            perm_test = perm
            rng = np.random.default_rng(s + 99)
            actions = rng.integers(0, 2, len(a))
            methods = {
                "fixed_high": _target(a, u, su, admission=0.95, **kw),
                "identity": run_policy(
                    a,
                    u,
                    su,
                    threshold,
                    {"low": 0, "high": 1},
                    action_permutation=perm_test,
                    **kw,
                ),
                "random": _target_actions(
                    a, u, su, actions, action_permutation=perm_test, **kw
                ),
                "native": run_policy(a, u, su, native_h, native_map, action_permutation=perm_test, **kw),
                "transfer": run_policy(
                    a, u, su, threshold, amap, action_permutation=perm_test, **kw
                ),
                "oracle": run_policy(
                    a,
                    u,
                    su,
                    threshold,
                    oracle_map,
                    action_permutation=perm_test,
                    **kw,
                ),
            }
            for key, value in methods.items():
                per_method[key].append({field: float(getattr(value, field)) for field in FIELDS})
        metrics = {k: {z: float(np.mean([x[z] for x in vs])) for z in FIELDS}
                   for k, vs in per_method.items()}
        ci = {k: {z: _bootstrap_ci([x[z] for x in vs], seed=seed + ri * 100 + j)
                  for j, z in enumerate(FIELDS)} for k, vs in per_method.items()}
        paired_ci = {baseline: _bootstrap_ci(
            [a["cost"] - b["cost"] for a, b in zip(per_method["transfer"], per_method[baseline])],
            seed=seed + 10000 + ri * 100 + i)
            for i, baseline in enumerate(("fixed_high", "identity", "random", "native", "oracle"))}
        paired_differences = {baseline: [
            float(a["cost"] - b["cost"])
            for a, b in zip(per_method["transfer"], per_method[baseline])
        ] for baseline in ("fixed_high", "identity", "random", "native", "oracle")}
        out["regimes"][name] = {
            "calibration_seeds": cal, "target_train_seeds": train,
            "test_seeds": test, "permutation_fixed": True, "action_permutation": [int(x) for x in perm],
            "fingerprint_score": float(score), "ambiguity_gap": float(gap), "ambiguous": bool(gap <= 1e-12),
            "inferred_map": amap, "oracle_map": oracle_map, "inferred_map_equals_oracle": bool(amap == oracle_map),
            "source_threshold": float(threshold), "native_threshold": float(native_h),
            "native_map": native_map, "native_train_cost": float(native_cost),
            "metrics": metrics, "per_policy_bootstrap95_ci": ci,
            "paired_cost_difference_transfer_minus_baseline_ci95": paired_ci,
            "paired_cost_differences_transfer_minus_baseline": paired_differences,
            "per_seed": per_method,
        }
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = json.dumps(run_benchmark(), indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)

from __future__ import annotations

import importlib
import json
import runpy
import sys
from collections import Counter, defaultdict
from inspect import signature
from types import SimpleNamespace

import numpy as np
import pytest

import crit_v08.queue_transfer as queue_transfer


DEFAULT_SEED = 20260905
REGIME_ORDER = ("balanced", "overload", "service_shift")
METHOD_ORDER = ("fixed_high", "identity", "random", "native", "transfer", "oracle")
FIELD_ORDER = (
    "cost",
    "throughput",
    "rejected",
    "overflow",
    "mean_latency",
    "p95_latency",
    "unfinished",
)
BASELINE_ORDER = ("fixed_high", "identity", "random", "native", "oracle")


@pytest.fixture(scope="module")
def runner_module():
    return importlib.import_module("benchmarks.run_v08_queue_transfer")


@pytest.fixture(scope="module")
def quick_result(runner_module):
    return runner_module.run_benchmark(seeds_per_regime=2)


def _regime_seeds(seed: int, regime_index: int, seeds_per_regime: int) -> dict[str, list[int]]:
    return {
        "calibration": [seed + 1000 * regime_index + i for i in range(8)],
        "train": [seed + 1000 * regime_index + 40 + i for i in range(12)],
        "test": [seed + 1000 * regime_index + 100 + i for i in range(seeds_per_regime)],
    }


def _bootstrap_ci(values, *, seed: int, draws: int = 2000) -> list[float]:
    x = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = np.mean(x[rng.integers(0, len(x), size=(draws, len(x)))], axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def test_publication_defaults_and_method_tables(runner_module) -> None:
    parameters = signature(runner_module.run_benchmark).parameters

    assert parameters["seed"].default == DEFAULT_SEED
    assert parameters["seeds_per_regime"].default == 20
    assert tuple(runner_module.REGIMES) == REGIME_ORDER
    assert tuple(runner_module.METHODS) == METHOD_ORDER
    assert tuple(runner_module.FIELDS) == FIELD_ORDER
    assert tuple(runner_module.THRESHOLDS) == (0.1, 0.3, 0.5, 0.7)


def test_quick_benchmark_structure_and_disjoint_seed_streams(quick_result) -> None:
    assert quick_result["seed"] == DEFAULT_SEED
    assert quick_result["seeds_per_regime"] == 2
    assert list(quick_result["regimes"]) == list(REGIME_ORDER)

    source_fit = quick_result["source_fit_seeds"]
    assert source_fit == [DEFAULT_SEED + 100000 + i for i in range(16)]
    assert quick_result["source_threshold_grid"] == [0.1, 0.3, 0.5, 0.7]
    assert quick_result["source_fit_config"] == {
        "service_rate": 2.0,
        "capacity": 12.0,
        "arrival": "Uniform(2,6), horizon=60",
    }

    all_allocated_seeds = set(source_fit)
    for regime_index, regime_name in enumerate(REGIME_ORDER):
        regime = quick_result["regimes"][regime_name]
        expected = _regime_seeds(DEFAULT_SEED, regime_index, 2)
        assert regime["calibration_seeds"] == expected["calibration"]
        assert regime["target_train_seeds"] == expected["train"]
        assert regime["test_seeds"] == expected["test"]

        regime_seed_groups = {
            "calibration": set(expected["calibration"]),
            "train": set(expected["train"]),
            "test": set(expected["test"]),
        }
        group_names = list(regime_seed_groups)
        for left_index, left_name in enumerate(group_names):
            for right_name in group_names[left_index + 1 :]:
                assert regime_seed_groups[left_name].isdisjoint(regime_seed_groups[right_name])
        flattened = set().union(*regime_seed_groups.values())
        assert flattened.isdisjoint(source_fit)
        assert flattened.isdisjoint(all_allocated_seeds - set(source_fit))
        all_allocated_seeds.update(flattened)

        assert list(regime["per_seed"]) == list(METHOD_ORDER)
        assert list(regime["metrics"]) == list(METHOD_ORDER)
        assert list(regime["per_policy_bootstrap95_ci"]) == list(METHOD_ORDER)
        for method in METHOD_ORDER:
            assert len(regime["per_seed"][method]) == 2
            for record in regime["per_seed"][method]:
                assert tuple(record) == FIELD_ORDER
                assert all(np.isfinite(value) for value in record.values())


def test_action_permutation_maps_thresholds_and_ambiguity_metadata(
    quick_result, runner_module
) -> None:
    assert set(quick_result["source_candidate_costs"]) == {0.1, 0.3, 0.5, 0.7}
    assert all(np.isfinite(value) for value in quick_result["source_candidate_costs"].values())
    assert quick_result["source_fit_seeds"] == sorted(quick_result["source_fit_seeds"])

    expected_source_costs = {}
    for threshold in runner_module.THRESHOLDS:
        seed_costs = []
        for seed in quick_result["source_fit_seeds"]:
            arrivals = np.random.default_rng(seed).uniform(2.0, 6.0, 60)
            seed_costs.append(queue_transfer.source_rollout(arrivals, 2.0, 12.0, threshold)["cost"])
        expected_source_costs[float(threshold)] = float(np.mean(seed_costs))
    for threshold, expected_cost in expected_source_costs.items():
        assert quick_result["source_candidate_costs"][threshold] == pytest.approx(expected_cost)
    expected_source_threshold = min(
        expected_source_costs,
        key=lambda threshold: (expected_source_costs[threshold], threshold),
    )

    for regime_index, regime_name in enumerate(REGIME_ORDER):
        regime = quick_result["regimes"][regime_name]
        expected_permutation = [
            int(x) for x in np.random.default_rng(DEFAULT_SEED + regime_index).permutation(2)
        ]
        assert regime["permutation_fixed"] is True
        assert regime["action_permutation"] == expected_permutation
        assert sorted(regime["action_permutation"]) == [0, 1]

        permutation = regime["action_permutation"]
        expected_oracle = {"low": permutation.index(0), "high": permutation.index(1)}
        assert regime["oracle_map"] == expected_oracle
        for mapping_name in ("inferred_map", "native_map", "oracle_map"):
            mapping = regime[mapping_name]
            assert set(mapping) == {"low", "high"}
            assert set(mapping.values()) == {0, 1}
            assert all(isinstance(value, int) for value in mapping.values())

        assert regime["source_threshold"] == pytest.approx(expected_source_threshold)
        assert regime["native_threshold"] in runner_module.THRESHOLDS
        assert np.isfinite(regime["native_train_cost"])
        assert regime["fingerprint_score"] >= 0.0
        assert regime["ambiguity_gap"] >= 0.0
        assert regime["ambiguous"] == bool(regime["ambiguity_gap"] <= 1e-12)
        assert regime["inferred_map_equals_oracle"] == bool(
            regime["inferred_map"] == regime["oracle_map"]
        )


def test_cost_means_and_paired_differences_use_matching_seed_records(quick_result) -> None:
    for regime_name in REGIME_ORDER:
        regime = quick_result["regimes"][regime_name]
        for method in METHOD_ORDER:
            records = regime["per_seed"][method]
            for field in FIELD_ORDER:
                expected_mean = float(np.mean([record[field] for record in records]))
                assert regime["metrics"][method][field] == pytest.approx(expected_mean)

        paired = regime["paired_cost_differences_transfer_minus_baseline"]
        assert list(paired) == list(BASELINE_ORDER)
        for baseline in BASELINE_ORDER:
            expected = [
                transfer["cost"] - control["cost"]
                for transfer, control in zip(
                    regime["per_seed"]["transfer"], regime["per_seed"][baseline]
                )
            ]
            actual = paired[baseline]
            assert len(actual) == 2
            assert actual == pytest.approx(expected, rel=0.0, abs=1e-15)
            assert float(np.mean(actual)) == pytest.approx(
                regime["metrics"]["transfer"]["cost"] - regime["metrics"][baseline]["cost"]
            )


def test_bootstrap_intervals_are_ordered_seed_aggregates(quick_result) -> None:
    for regime_index, regime_name in enumerate(REGIME_ORDER):
        regime = quick_result["regimes"][regime_name]
        for method in METHOD_ORDER:
            for field_index, field in enumerate(FIELD_ORDER):
                values = [record[field] for record in regime["per_seed"][method]]
                expected = _bootstrap_ci(
                    values, seed=DEFAULT_SEED + regime_index * 100 + field_index
                )
                actual = regime["per_policy_bootstrap95_ci"][method][field]
                assert actual == pytest.approx(expected)
                assert actual[0] <= actual[1]

        paired_ci = regime["paired_cost_difference_transfer_minus_baseline_ci95"]
        assert list(paired_ci) == list(BASELINE_ORDER)
        paired = regime["paired_cost_differences_transfer_minus_baseline"]
        for baseline_index, baseline in enumerate(BASELINE_ORDER):
            expected = _bootstrap_ci(
                paired[baseline],
                seed=DEFAULT_SEED + 10000 + regime_index * 100 + baseline_index,
            )
            actual = paired_ci[baseline]
            assert actual == pytest.approx(expected)
            assert actual[0] <= actual[1]


def test_methods_share_randomness_and_fixed_permutation_real_calls(
    runner_module, monkeypatch
) -> None:
    original_make_randomness = runner_module.make_randomness
    original_target_fingerprint = runner_module.target_fingerprint
    original_run_policy = runner_module.run_policy
    original_target = runner_module._target
    original_target_actions = runner_module._target_actions
    original_threshold_policy = runner_module.threshold_policy
    randomness_seeds = []
    fingerprint_calls = []
    top_level_calls = []
    threshold_calls = []
    randomness_snapshots = {}

    def snapshot_randomness(args):
        key = id(args[0])
        if key not in randomness_snapshots:
            randomness_snapshots[key] = (
                args[0].copy(),
                args[1].copy(),
                args[2].copy(),
                args[0],
                args[1],
                args[2],
            )

    def observe_make_randomness(*args, **kwargs):
        randomness_seeds.append(args[0])
        return original_make_randomness(*args, **kwargs)

    def observe_target_fingerprint(*args, **kwargs):
        snapshot_randomness(args)
        fingerprint_calls.append({"args": args, "kwargs": kwargs})
        return original_target_fingerprint(*args, **kwargs)

    def observe_run_policy(*args, **kwargs):
        snapshot_randomness(args)
        top_level_calls.append({"kind": "run_policy", "args": args, "kwargs": kwargs})
        return original_run_policy(*args, **kwargs)

    def observe_target(*args, **kwargs):
        snapshot_randomness(args)
        top_level_calls.append({"kind": "_target", "args": args, "kwargs": kwargs})
        return original_target(*args, **kwargs)

    def observe_target_actions(*args, **kwargs):
        snapshot_randomness(args)
        top_level_calls.append({"kind": "_target_actions", "args": args, "kwargs": kwargs})
        return original_target_actions(*args, **kwargs)

    def observe_threshold_policy(*args, **kwargs):
        threshold_calls.append({"args": args, "kwargs": kwargs})
        return original_threshold_policy(*args, **kwargs)

    monkeypatch.setattr(runner_module, "make_randomness", observe_make_randomness)
    monkeypatch.setattr(runner_module, "target_fingerprint", observe_target_fingerprint)
    monkeypatch.setattr(runner_module, "run_policy", observe_run_policy)
    monkeypatch.setattr(runner_module, "_target", observe_target)
    monkeypatch.setattr(runner_module, "_target_actions", observe_target_actions)
    monkeypatch.setattr(runner_module, "threshold_policy", observe_threshold_policy)

    result = runner_module.run_benchmark(seeds_per_regime=2)

    for snapshot in randomness_snapshots.values():
        (
            arrivals_copy,
            uniforms_copy,
            service_copy,
            arrivals,
            uniforms,
            service_uniforms,
        ) = snapshot
        assert np.array_equal(arrivals, arrivals_copy)
        assert np.array_equal(uniforms, uniforms_copy)
        assert np.array_equal(service_uniforms, service_copy)

    assert len(threshold_calls) == 1
    source_fit_call = threshold_calls[0]
    expected_source_seeds = [DEFAULT_SEED + 100000 + i for i in range(16)]
    assert list(source_fit_call["args"][0]) == expected_source_seeds
    assert tuple(source_fit_call["args"][1]) == (0.1, 0.3, 0.5, 0.7)
    assert source_fit_call["kwargs"]["service_rate"] == 2.0
    assert source_fit_call["kwargs"]["cap"] == 12.0
    assert source_fit_call["kwargs"]["return_scores"] is True
    source_arrivals = source_fit_call["kwargs"]["arrivals_by_seed"]
    assert list(source_arrivals) == expected_source_seeds
    for seed in expected_source_seeds:
        expected_arrivals = np.random.default_rng(seed).uniform(2.0, 6.0, 60)
        assert np.array_equal(source_arrivals[seed], expected_arrivals)

    counts = Counter(randomness_seeds)
    for regime_index, regime_name in enumerate(REGIME_ORDER):
        expected = _regime_seeds(DEFAULT_SEED, regime_index, 2)
        for seed in expected["test"]:
            assert counts[seed] == 1
        for seed in expected["calibration"]:
            assert counts[seed] == 1
            assert counts[seed + 50000] == 1
        for seed in expected["train"]:
            assert counts[seed] == 8
        assert set(expected["calibration"]).isdisjoint(expected["train"])
        assert set(expected["calibration"]).isdisjoint(expected["test"])
        assert set(expected["train"]).isdisjoint(expected["test"])
        assert {seed + 50000 for seed in expected["calibration"]}.isdisjoint(
            set(expected["calibration"]) | set(expected["train"]) | set(expected["test"])
        )

    config_to_permutation = {
        runner_module.REGIMES[regime_name]: tuple(
            int(x) for x in np.random.default_rng(DEFAULT_SEED + regime_index).permutation(2)
        )
        for regime_index, regime_name in enumerate(REGIME_ORDER)
    }
    for call in top_level_calls:
        if call["kind"] in {"run_policy", "_target_actions"}:
            config = (
                call["kwargs"]["service_rate"],
                call["kwargs"]["queue_capacity"],
                call["kwargs"]["service_shift"],
            )
            assert tuple(call["kwargs"]["action_permutation"]) == config_to_permutation[config]

    groups_by_arrivals = defaultdict(list)
    for call in top_level_calls:
        groups_by_arrivals[id(call["args"][0])].append(call)
    shared_groups = [group for group in groups_by_arrivals.values() if len(group) == 6]
    assert len(shared_groups) == len(REGIME_ORDER) * 2

    expected_kinds = [
        "_target",
        "run_policy",
        "_target_actions",
        "run_policy",
        "run_policy",
        "run_policy",
    ]
    for regime_index, regime_name in enumerate(REGIME_ORDER):
        regime = result["regimes"][regime_name]
        expected_permutation = tuple(
            int(x) for x in np.random.default_rng(DEFAULT_SEED + regime_index).permutation(2)
        )
        assert tuple(regime["action_permutation"]) == expected_permutation
        expected_config = runner_module.REGIMES[regime_name]
        for seed_offset in range(2):
            group = shared_groups[regime_index * 2 + seed_offset]
            assert [call["kind"] for call in group] == expected_kinds
            first = group[0]
            assert first["kwargs"]["admission"] == 0.95
            for call in group:
                assert call["args"][0] is first["args"][0]
                assert call["args"][1] is first["args"][1]
                assert call["args"][2] is first["args"][2]
                assert call["kwargs"]["service_rate"] == expected_config[0]
                assert call["kwargs"]["queue_capacity"] == expected_config[1]
                assert call["kwargs"]["service_shift"] == expected_config[2]

            permutation_calls = [
                call for call in group if call["kind"] in {"run_policy", "_target_actions"}
            ]
            assert len(permutation_calls) == 5
            permutations = [call["kwargs"]["action_permutation"] for call in permutation_calls]
            assert all(tuple(permutation) == expected_permutation for permutation in permutations)
            assert all(permutation is permutations[0] for permutation in permutations)

            identity, random_call, native, transfer, oracle = group[1:6]
            assert identity["args"][4] == {"low": 0, "high": 1}
            assert native["args"][4] == regime["native_map"]
            assert transfer["args"][4] == regime["inferred_map"]
            assert oracle["args"][4] == regime["oracle_map"]
            assert identity["args"][3] == pytest.approx(regime["source_threshold"])
            assert transfer["args"][3] == pytest.approx(regime["source_threshold"])
            assert oracle["args"][3] == pytest.approx(regime["source_threshold"])
            assert native["args"][3] == pytest.approx(regime["native_threshold"])

            test_seed = regime["test_seeds"][seed_offset]
            expected_random_actions = np.random.default_rng(test_seed + 99).integers(
                0, 2, len(first["args"][0])
            )
            assert np.array_equal(random_call["args"][3], expected_random_actions)

    fingerprint_calls_by_config = defaultdict(list)
    for call in fingerprint_calls:
        config = (
            call["kwargs"]["service_rate"],
            call["kwargs"]["queue_capacity"],
            call["kwargs"]["service_shift"],
        )
        fingerprint_calls_by_config[config].append(call)
    for regime_index, regime_name in enumerate(REGIME_ORDER):
        expected_permutation = tuple(
            int(x) for x in np.random.default_rng(DEFAULT_SEED + regime_index).permutation(2)
        )
        calls = fingerprint_calls_by_config[runner_module.REGIMES[regime_name]]
        assert len(calls) == 16
        assert all(
            tuple(call["kwargs"]["action_permutation"]) == expected_permutation for call in calls
        )
        action_zero = [call for call in calls if call["args"][3] == 0]
        action_one = [call for call in calls if call["args"][3] == 1]
        assert len(action_zero) == len(action_one) == 8
        for zero_call, one_call in zip(action_zero, action_one):
            assert zero_call["args"][0] is one_call["args"][0]
            assert zero_call["args"][1] is one_call["args"][1]
            assert zero_call["args"][2] is one_call["args"][2]


def test_quick_benchmark_repeatable(runner_module, quick_result) -> None:
    repeated = runner_module.run_benchmark(seeds_per_regime=2)

    assert repeated == quick_result


def test_cli_output_writes_default_json_without_stdout(
    runner_module, monkeypatch, tmp_path, capsys
) -> None:
    def fast_randomness(seed, horizon=60, max_arrivals=8):
        return (
            np.full(2, 2, dtype=int),
            np.full((2, 8), 0.5),
            np.full((2, 1), 0.5),
        )

    def fast_metric(*args, **kwargs):
        return SimpleNamespace(
            cost=1.0,
            throughput=1,
            rejected=0,
            overflow=0,
            mean_latency=1.0,
            p95_latency=1.0,
            unfinished=0,
        )

    def fast_threshold_policy(source_seeds, thresholds, **kwargs):
        return float(thresholds[0]), {float(threshold): 1.0 for threshold in thresholds}

    def fast_fingerprint(*args, **kwargs):
        return np.asarray([0.0, 0.0])

    monkeypatch.setattr(queue_transfer, "make_randomness", fast_randomness)
    monkeypatch.setattr(queue_transfer, "run_policy", fast_metric)
    monkeypatch.setattr(queue_transfer, "_target", fast_metric)
    monkeypatch.setattr(queue_transfer, "_target_actions", fast_metric)
    monkeypatch.setattr(queue_transfer, "threshold_policy", fast_threshold_policy)
    monkeypatch.setattr(queue_transfer, "target_fingerprint", fast_fingerprint)
    monkeypatch.setattr(queue_transfer, "source_signature", fast_fingerprint)
    monkeypatch.setattr(sys, "path", list(sys.path))

    script = runner_module.__file__
    output = tmp_path / "queue-result.json"
    monkeypatch.setattr(sys, "argv", [script, "--output", str(output)])

    runpy.run_path(script, run_name="__main__")

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    assert [path.name for path in tmp_path.iterdir()] == ["queue-result.json"]
    text = output.read_text(encoding="utf-8")
    assert text.endswith("\n")
    payload = json.loads(text)
    assert payload["seed"] == DEFAULT_SEED
    assert payload["seeds_per_regime"] == 20
    assert list(payload["regimes"]) == list(REGIME_ORDER)
    for regime_index, regime_name in enumerate(REGIME_ORDER):
        regime = payload["regimes"][regime_name]
        expected = _regime_seeds(DEFAULT_SEED, regime_index, 20)
        assert regime["calibration_seeds"] == expected["calibration"]
        assert regime["target_train_seeds"] == expected["train"]
        assert regime["test_seeds"] == expected["test"]
        assert len(regime["test_seeds"]) == 20
        assert set(regime["per_seed"]) == set(METHOD_ORDER)
        assert all(len(regime["per_seed"][method]) == 20 for method in METHOD_ORDER)

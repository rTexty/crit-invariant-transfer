"""Compact historical references and fresh public-CLI reproduction checks."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "results" / "reference"
SNAPSHOT = "8e354594f61200fedca1266068794919a4eb1d64"
REGIMES = ("balanced", "overload", "service_shift")
METHODS = {"fixed_high", "identity", "random", "native", "transfer", "oracle"}
CASE_FIELDS = {"size", "automorphism_count", "blind_regret", "informed_regret", "gap"}


def _identifiability(payload):
    result = payload["result"]
    return {
        "seed": result["seed"],
        "all_rigid": result["all_rigid"],
        "rigid_cases": [{key: case[key] for key in CASE_FIELDS}
                        for case in result["rigid_cases"]],
        "left_zero_counterexample": result["left_zero_counterexample"],
    }


def _queue(payload):
    return {
        "seed": payload["seed"],
        "seeds_per_regime": payload["seeds_per_regime"],
        "source_fit_seeds": payload["source_fit_seeds"],
        "regimes": {
            name: {
                **{key: regime[key] for key in (
                    "calibration_seeds", "target_train_seeds", "test_seeds",
                    "inferred_map_equals_oracle",
                )},
                "metrics": {method: {"cost": metrics["cost"]}
                            for method, metrics in regime["metrics"].items()},
                "mean_paired_cost_difference_transfer_minus_native": sum(
                    regime["paired_cost_differences_transfer_minus_baseline"]["native"]
                ) / len(regime["paired_cost_differences_transfer_minus_baseline"]["native"]),
            }
            for name, regime in payload["regimes"].items()
        },
    }


def _compare(actual, expected, path="result"):
    assert type(actual) is type(expected), path
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys(), path
        for key in expected:
            _compare(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(expected, list):
        assert len(actual) == len(expected), path
        for index, (left, right) in enumerate(zip(actual, expected)):
            _compare(left, right, f"{path}[{index}]")
    elif isinstance(expected, float):
        assert math.isfinite(actual) and math.isfinite(expected), path
        assert math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-10), path
    else:
        assert actual == expected, path


def _load(name):
    return json.loads((REFERENCE / name).read_text(encoding="utf-8"))


def test_reference_manifest_and_schemas():
    manifest = _load("manifest.json")
    assert manifest["archive_snapshot"] == SNAPSHOT
    assert manifest["status"] == "Historical synthetic reference subsets; not new observations"
    assert manifest["tolerances"] == {"rtol": 1e-9, "atol": 1e-10}
    assert manifest["environment"]["python"] == "3.14.4"
    assert manifest["environment"]["numpy"] == "2.4.6"
    assert manifest["environment"]["scipy"] == "1.18.0"
    assert set(manifest["references"]) == {"identifiability.json", "queue-transfer.json"}
    for name, entry in manifest["references"].items():
        assert entry["sha256"] == hashlib.sha256((REFERENCE / name).read_bytes()).hexdigest()
        assert len(entry["archive_sha256"]) == 64
        int(entry["archive_sha256"], 16)
        source = "crit_v04_identifiability" if name == "identifiability.json" else "crit_v08_queue_transfer"
        module = "run_v04_identifiability" if name == "identifiability.json" else "run_v08_queue_transfer"
        assert entry["archive_path"] == f".ecc/benchmarks/{source}.json"
        assert entry["command"] == f"uv run --locked python -m benchmarks.{module} --output artifacts/{name}"

    ident = _load("identifiability.json")
    assert set(ident) == {"seed", "all_rigid", "rigid_cases", "left_zero_counterexample"}
    assert type(ident["seed"]) is int and ident["seed"] == 20260831
    assert ident["all_rigid"] is True
    assert len(ident["rigid_cases"]) == 4
    for size, case in enumerate(ident["rigid_cases"], 2):
        assert set(case) == CASE_FIELDS
        assert type(case["size"]) is int and case["size"] == size
        assert type(case["automorphism_count"]) is int and case["automorphism_count"] == 1
        assert case["informed_regret"] == 0.0
        assert case["blind_regret"] == pytest.approx(1 - 1 / size, rel=1e-9, abs=1e-10)
        assert case["gap"] == case["blind_regret"]
    counter = ident["left_zero_counterexample"]
    assert set(counter) == CASE_FIELDS
    assert counter["size"] == 3 and counter["automorphism_count"] == 6
    assert counter["blind_regret"] == counter["informed_regret"] == pytest.approx(2 / 3)
    assert counter["gap"] == 0.0

    queue = _load("queue-transfer.json")
    assert set(queue) == {"seed", "seeds_per_regime", "source_fit_seeds", "regimes"}
    assert type(queue["seed"]) is int and queue["seed"] == 20260905
    assert type(queue["seeds_per_regime"]) is int and queue["seeds_per_regime"] == 20
    assert queue["source_fit_seeds"] == list(range(20360905, 20360921))
    assert set(queue["regimes"]) == set(REGIMES)
    for index, name in enumerate(REGIMES):
        regime = queue["regimes"][name]
        assert set(regime) == {
            "calibration_seeds", "target_train_seeds", "test_seeds", "metrics",
            "inferred_map_equals_oracle", "mean_paired_cost_difference_transfer_minus_native",
        }
        base = queue["seed"] + 1000 * index
        for field, offset, count in (("calibration_seeds", 0, 8),
                                     ("target_train_seeds", 40, 12), ("test_seeds", 100, 20)):
            assert regime[field] == list(range(base + offset, base + offset + count))
            assert all(type(seed) is int for seed in regime[field])
        assert regime["inferred_map_equals_oracle"] is True
        assert set(regime["metrics"]) == METHODS
        for metrics in regime["metrics"].values():
            assert set(metrics) == {"cost"}
            assert type(metrics["cost"]) is float and math.isfinite(metrics["cost"])
        delta = regime["mean_paired_cost_difference_transfer_minus_native"]
        assert delta == pytest.approx(
            regime["metrics"]["transfer"]["cost"] - regime["metrics"]["native"]["cost"],
            rel=1e-9, abs=1e-10,
        )
        assert delta == 0.0 if name == "balanced" else delta > 0.0


@pytest.mark.parametrize("actual,expected", [
    (True, 1), (20260906, 20260905), ([1], [1, 2]), ({"a": 1}, {"b": 1}),
    (float("nan"), 0.0), (float("inf"), 0.0), (1.000001, 1.0), ("overload", "balanced"),
])
def test_comparison_rejects_category_and_numeric_drift(actual, expected):
    with pytest.raises(AssertionError):
        _compare(actual, expected)


def test_comparison_accepts_declared_numeric_tolerance():
    _compare({"x": [1.0 + 5e-10, 5e-11]}, {"x": [1.0, 0.0]})


@pytest.mark.reproduction
@pytest.mark.parametrize("module,name,compact", [
    ("run_v04_identifiability", "identifiability.json", _identifiability),
    ("run_v08_queue_transfer", "queue-transfer.json", _queue),
])
def test_cli_reproduces_historical_reference(module, name, compact, tmp_path):
    output = tmp_path / name
    completed = subprocess.run(
        [sys.executable, "-m", f"benchmarks.{module}", "--output", str(output)],
        cwd=ROOT, capture_output=True, text=True, timeout=180, check=True,
    )
    assert completed.stdout == ""
    assert completed.stderr == ""
    _compare(compact(json.loads(output.read_text(encoding="utf-8"))), _load(name))

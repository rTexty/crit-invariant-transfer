import numpy as np
import pytest
from crit_v08.queue_transfer import (
    _target_actions,
    _target,
    calibrate_fingerprints,
    fit_action_map,
    make_randomness,
    run_policy,
    source_policy,
    source_rollout,
    source_run,
    source_signature,
    target_fingerprint,
    target_trace,
    threshold_policy,
)


def test_source_conservation_and_reproducibility():
    a, u, s = make_randomness(4, 20)
    r = source_run(a, 2, 3)
    assert r["lost"] + sum(min(x, 3) for x in a) >= sum(a) - 1
    assert np.array_equal(make_randomness(4, 20)[0], a)


def test_hidden_map_not_needed_by_fit_and_permutation_equivariance():
    a, u, s = make_randomness(2, 30)
    fp = {0: target_trace(a, u, s, 0), 1: target_trace(a, u, s, 1)}
    sf = {"low": fp[0], "high": fp[1]}
    m, score, gap = fit_action_map(fp, sf)
    assert set(m) == {"low", "high"}
    assert score == pytest.approx(0)
    assert gap >= 0
    fp2 = {1: fp[0], 0: fp[1]}
    m2, _, _ = fit_action_map(fp2, {"low": fp2[1], "high": fp2[0]})
    assert m2["low"] == 1 and m2["high"] == 0


def test_transfer_executes_and_accounts_fifo_metrics():
    a, u, s = make_randomness(8, 40)
    fp = {0: target_trace(a, u, s, 0), 1: target_trace(a, u, s, 1)}
    m, _, _ = fit_action_map(fp, {"low": fp[0], "high": fp[1]})
    metric = run_policy(a, u, s, threshold_policy([1, 2], [0.2, 0.4, 0.6], cap=12), m)
    assert (
        metric.throughput + metric.rejected + metric.overflow + metric.unfinished
        == sum(a)
    )
    assert metric.p95_latency >= metric.mean_latency
    assert metric.unfinished >= 0


def test_target_shift_changes_descriptive_metrics():
    a, u, s = make_randomness(9, 40)
    x = _target(
        a, u, s, admission=0.95, service_rate=2, queue_capacity=12, service_shift=1
    )
    y = _target(
        a, u, s, admission=0.95, service_rate=2, queue_capacity=12, service_shift=2
    )
    assert y.mean_latency >= x.mean_latency


def test_policy_uses_hand_computed_target_occupancy_and_service_shift():
    # Deliberately opaque, hand-computed queue: the transferred controller must
    # use current target occupancy and the shifted service draw, not source q or
    # an unshifted service rate.  The fixed action replay is an independent
    # target-simulator oracle, not a second call to run_policy.
    arrivals = np.array([2, 0, 2, 0])
    uniforms = np.array([[0.7, 0.8]] * 4)
    service_uniforms = np.array([[0.0], [0.9], [0.0], [0.9]])
    common = dict(
        service_rate=3,
        queue_capacity=2,
        service_shift=2,
        action_permutation=(1, 0),
    )
    expected_actions = [0, 1, 1, 1]
    expected_trace = (1.0, 0.5, 0.5, 0.0)
    expected = dict(
        cost=0.9,
        throughput=2,
        rejected=2,
        overflow=0,
        unfinished=0,
        occupancy=0.5,
        q_trajectory=expected_trace,
    )

    transferred = run_policy(
        arrivals, uniforms, service_uniforms, 0.5,
        {"high": 0, "low": 1}, **common
    )
    oracle = _target_actions(
        arrivals, uniforms, service_uniforms, expected_actions, **common
    )
    for field, value in expected.items():
        actual = getattr(transferred, field)
        expected_actual = pytest.approx(value) if isinstance(value, float) else value
        assert actual == expected_actual
        oracle_actual = getattr(oracle, field)
        assert oracle_actual == expected_actual
    assert transferred.mean_latency <= transferred.p95_latency


def test_source_rollout_matches_independent_hand_computation():
    arrivals = np.array([5.0, 1.0, 8.0])
    # q/cap is .9375 after t=0, so t=1 and t=2 must use the low fraction.
    result = source_rollout(
        arrivals, service_rate=0.0, cap=4.0, h=0.5,
        source_high=0.75, source_low=0.25,
    )
    expected_trace = (0.9375, 1.0, 1.0)
    # Rejected fluid is 1.25 + .75 + 6.0; overflow is 2.0.
    expected_lost = 10.0
    expected_occupancy = sum(expected_trace) / 3
    expected_cost = expected_occupancy + 0.8 * expected_lost / 14.0
    assert result["q_trajectory"] == pytest.approx(expected_trace)
    assert result["lost"] == pytest.approx(expected_lost)
    assert result["occupancy"] == pytest.approx(expected_occupancy)
    assert result["cost"] == pytest.approx(expected_cost)


def test_threshold_fit_returns_actual_source_argmin():
    arrivals = np.array([3.0, 3.0, 3.0])
    winner, scores = threshold_policy(
        [17], [0.1, 1.0], service_rate=2.0, cap=4.0,
        arrivals_by_seed={17: arrivals}, return_scores=True,
    )
    assert scores[0.1] == pytest.approx(0.3783333333333333)
    assert scores[1.0] == pytest.approx(0.465)
    assert winner == min(scores, key=lambda threshold: (scores[threshold], threshold))


def test_fingerprint_fit_uses_independent_source_signature_and_recovers_permutation():
    target_arrivals = np.array([4, 0])
    target_uniforms = np.array([[0.1, 0.2, 0.6, 0.9], [0.1, 0.1, 0.1, 0.1]])
    target_service = np.ones((2, 1))
    target = {
        action: target_fingerprint(
            target_arrivals, target_uniforms, target_service, action,
            reset=(0.25, 0.75), queue_capacity=4,
            admission_levels=(0.5, 0.75), action_permutation=(1, 0),
        )
        for action in (0, 1)
    }
    # Independent source arrivals; these are not target traces or target q.
    source = {
        "low": source_signature(np.array([4.0, 99.0]), 0.5, cap=4.0, reset=(0.25, 0.75)),
        "high": source_signature(np.array([4.0, 1.0]), 0.75, cap=4.0, reset=(0.25, 0.75)),
    }
    mapping, score, _ = fit_action_map(target, source)
    assert mapping == {"low": 1, "high": 0}
    assert score == pytest.approx(0.0)

    test_data = make_randomness(91, 8)
    kwargs = dict(service_rate=2, queue_capacity=4, service_shift=2,
                  action_permutation=(1, 0))
    transferred = run_policy(*test_data, 0.5, mapping, **kwargs)
    oracle = run_policy(*test_data, 0.5, {"low": 1, "high": 0}, **kwargs)
    assert transferred == oracle


def test_calibration_aggregate_includes_all_eight_seeds():
    seeds_data = {
        seed: (np.array([seed - 99, 0]), np.full((2, 8), 0.1), np.ones((2, 1)))
        for seed in range(100, 108)
    }
    actual = calibrate_fingerprints(
        seeds_data, queue_capacity=8, admission_levels=(0.5, 0.75),
    )
    expected = {
        action: np.mean([
            target_fingerprint(*seeds_data[seed], action,
                               queue_capacity=8,
                               admission_levels=(0.5, 0.75))
            for seed in seeds_data
        ], axis=0)
        for action in (0, 1)
    }
    for action in (0, 1):
        assert actual[action] == pytest.approx(expected[action])


def test_fit_reports_ambiguous_zero_gap_without_using_reward():
    target = {0: np.array([0.0, 0.0]), 1: np.array([0.0, 0.0])}
    source = {"low": target[0], "high": target[1]}
    mapping, score, gap = fit_action_map(target, source)
    assert mapping == {"low": 0, "high": 1}
    assert score == pytest.approx(0.0)
    assert gap == pytest.approx(0.0)


def test_source_threshold_fit_uses_policy_and_counts_rejection_and_overflow():
    arrivals = np.array([2.0, 2.0, 2.0])
    low = source_rollout(arrivals, service_rate=1.0, cap=4.0, h=0.1)
    high = source_rollout(arrivals, service_rate=1.0, cap=4.0, h=0.9)
    assert low["cost"] != high["cost"]
    assert low["lost"] > 0 and high["lost"] > 0
    winner, scores = threshold_policy([1], [0.1, 0.5], cap=4, arrivals_by_seed={1: arrivals}, return_scores=True)
    assert winner in scores and set(scores) == {0.1, 0.5}


def test_validation_rejects_bad_source_and_target_inputs():
    with pytest.raises(ValueError):
        source_policy(1.7, 0.5)
    with pytest.raises(ValueError):
        threshold_policy([1], [1.7])
    a, u, s = make_randomness(1, 3)
    with pytest.raises(ValueError):
        target_fingerprint(a, u, s, 1.7)
    with pytest.raises(ValueError):
        fit_action_map({0: np.array([]), 1: np.array([1.0])}, {"low": np.array([]), "high": np.array([1.0])})
    with pytest.raises(ValueError):
        _target_actions(a, np.full((3, 8), 0.5), np.full((3, 1), 0.5), [0.5, 0, 1])
    with pytest.raises(ValueError):
        source_run(a, np.nan, 3)

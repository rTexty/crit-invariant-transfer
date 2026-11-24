# Queue strategy transfer: reproduction protocol

This is a synthetic engineering experiment, separate from the
[finite ambiguity theorem](../theory/identifiability.md). It asks whether a
threshold fitted on a fluid reservoir can control a stochastic FIFO queue
after matching two opaque admission actions using occupancy fingerprints.
The common normalized occupancy observation is human-authored. Neither domain
discovery nor real-world utility is established by this experiment.

This document specifies a reproduction of the **revised historical experiment**.
It is a subsequent public description, not a new preregistration. It specifies
the design without importing observed costs or the selected threshold.
The queue implementation and executable reproduction will be published in
later steps; this documentation step does not report a new run.

## Models and objective

The source is a continuous reservoir, coded independently from the finite FIFO
target queue. Both start empty and run for 60 steps. The policy observes the
current occupancy divided by capacity. For normalized threshold `h`, it selects
`high` when occupancy is strictly below `h`, and `low` otherwise. Source action
fractions are `high = 0.95` and `low = 0.55`.

In each source step, the policy offers its chosen fraction of incoming fluid,
subtracts service and clips the resulting level to `[0, capacity]`. Lost demand
counts both the unoffered fraction and overflow above capacity after service.
The source objective is mean post-step normalized occupancy plus `0.8` times
lost demand divided by total incoming demand.

For the target, incoming requests receive independent admission uniforms.
Requests rejected by the admission decision and requests admitted but blocked
by a full queue are counted separately. Arrivals are processed before service.
Service removes the oldest requests, up to
`floor(service_uniform * (service_rate + 1) / service_shift)` per step.
The queue capacity includes all requests retained in this simulator.

The frozen target objective, minimized by privileged target tuning, is

```text
cost = mean post-service normalized occupancy
       + 0.8 * (rejected + overflow) / max(total arrivals, 1).
```

Cost is dimensionless and lower is better. It is a simulator objective, not
money, a production metric or theorem-level regret. Also retain throughput,
rejected requests, overflow requests, unfinished requests at the horizon,
and mean and 95th-percentile latency among completed requests. Latency is
`completion step - arrival step + 1`, in simulator steps; if no request
completes, the implementation stores zero for both latency summaries.
Unfinished work is reported separately, so completed latency is censored.
Check request accounting in every run:

```text
total arrivals = rejected + overflow + throughput + unfinished.
```

## Fixed configurations and disjoint data

Use master seed `S = 20260905`. Fit the source threshold **once**, globally,
on 16 source-only seeds `S + 100000 + i`, `i = 0,...,15`. Each generates
60 independent `Uniform(2,6)` fluid arrivals, with capacity `12` and service
`2`. Minimize mean source cost over the grid `[0.1, 0.3, 0.5, 0.7]`;
break ties toward the smaller threshold. No target rewards enter this fit.

Use these target regimes in the listed order, indexed by `r`:

| `r` | Regime | Service rate | Capacity | Service shift |
| --- | --- | --- | --- | --- |
| 0 | `balanced` | 2 | 12 | 1.0 |
| 1 | `overload` | 1 | 8 | 1.0 |
| 2 | `service_shift` | 2 | 12 | 2.0 |

Names label fixed cases; `balanced` is not a queue-stability assertion.
For each regime, use three disjoint sets:

| Purpose | Seeds | Count |
| --- | --- | --- |
| Target calibration | `S + 1000*r + i`, `i = 0,...,7` | 8 |
| Privileged target training | `S + 1000*r + 40 + i`, `i = 0,...,11` | 12 |
| Frozen target evaluation | `S + 1000*r + 100 + i`, `i = 0,...,19` | 20 |

For each target seed, NumPy's `default_rng` draws 60 Poisson arrivals with
mean `4`, capped at `8` per step, then admission uniforms of shape `(60,8)`
and service uniforms of shape `(60,1)`. Generate these arrays once per test
seed and reuse them across all six policies. Do not refit on test seeds.

Draw one permutation of `(0,1)` from `default_rng(S + r)` per regime.
It maps opaque target action labels to admission levels `(0.55,0.95)` and
stays fixed across calibration, training and evaluation. Map recovery has
three regime-level cases, not 60 independent mapping trials.

## Calibration and transferred policy

Reset probes use occupancies `0.25` and `0.75`. In the target, initialize
the queue length to `round(reset * capacity)`. For each opaque action,
measure the normalized occupancy increment after first-step admissions,
clipped at capacity, **before service**. Average the two-component fingerprint
over all eight calibration seeds. These are reset interventions, not complete
queue rollouts; generated service randomness does not enter the fingerprint.

For the source action signatures, use independent streams with seeds
`calibration_seed + 50000`. They use the same capped-Poisson generator, separate
from the Uniform arrivals used to fit the threshold. At each reset `x`, the
source signature component is
`min(capacity, x*capacity + action_fraction*first_arrival)/capacity - x`.
Use the regime's capacity and average over the eight independent streams.

Enumerate both bijections from `{low, high}` to `{0,1}`. Score each as the sum,
over source labels, of mean squared differences between source and matched
target fingerprint components. Select the smallest score; lexicographic
order of `(map[low], map[high])` breaks ties. Report the winning score and
second-best-minus-best gap. Flag a gap `<= 1e-12` as ambiguous: deterministic
tie-breaking supplies a diagnostic choice, not evidence of identification.
The score and gap are not confidence or certification measures.

Freeze the inferred map and source-fitted threshold before evaluation. At
each target decision, select the source label from the **current target**
occupancy and translate it through the inferred map. Do not use a source
reference trajectory to choose target actions.

## Six methods and their information access

| Method | Definition and access |
| --- | --- |
| `fixed_high` | Always admit with probability `0.95`. This is a semantic simulator control with direct access to the high admission level, not a learner discovering an opaque label. |
| `identity` | Use the source-fitted threshold with the uncalibrated map `low -> 0`, `high -> 1`. No target reward tuning. |
| `random` | Independently draw opaque labels uniformly at each step using `default_rng(test_seed + 99)`; no target reward tuning. |
| `native` | Privileged target control: tune both threshold and orientation on the 12 target-train seeds using target costs. Search the same four thresholds and both maps; break cost ties by threshold, then `(map[low], map[high])`. |
| `transfer` | Use only source-fitted threshold, reset occupancy fingerprints and current target occupancy. No target reward fitting or access to the planted map. |
| `oracle` | Use the source-fitted threshold with the planted action map supplied by the evaluator. This is a mapping diagnostic, not an optimal target-policy oracle. |

The evaluator owns the simulator configuration, planted permutation, test
randomness and all outcome metrics. The transfer procedure receives the
specified occupancy abstraction, reset intervention data and source signatures;
capacity is used for normalization and signature construction. It does not
receive the planted permutation or target train/test costs to select its map
or threshold. `native` receives target training costs and searches opaque
orientation; `oracle` receives the true map; `fixed_high` bypasses opaque-map
learning. Label these advantages when comparing costs.

## Paired descriptive reporting

Keep all 20 test records per method and regime, the seed lists, source grid
costs, selected thresholds and maps, fingerprint score, gap and ambiguity flag.
Report per-regime means. Retain per-policy descriptive 95% percentile bootstrap
intervals for cost and the six other reported metrics, using 2,000 resamples
of 20 seed records with replacement. Metric order is
`cost, throughput, rejected, overflow, mean_latency, p95_latency, unfinished`;
for metric index `j`, bootstrap with seed `S + 100*r + j`.

For each baseline in order `fixed_high, identity, random, native, oracle`,
form the 20 **paired** differences `transfer cost - baseline cost` on matching
test seeds. A positive difference favors the baseline. Bootstrap these
differences with 2,000 resamples and seed `S + 10000 + 100*r + baseline_index`.
Use NumPy percentiles `2.5` and `97.5` with default linear interpolation.
Retain the individual differences and their intervals; do not subtract
separately bootstrapped policy intervals. These descriptive intervals concern
seed variation under one fixed calibration and map per regime. They do not
quantify uncertainty across refitted calibrations or unseen domains, and do
not correct for multiple comparisons. Report failures and negative transfer.

## Revision record and provenance

The initial draft was invalid and its artifact was discarded. It used one
calibration seed, a reference rollout to choose target actions and independently
changing test permutations. The revised historical protocol fixed one
permutation per regime, used all eight calibration seeds, based transferred
actions on current target occupancy and flagged zero-gap ambiguity. It was
frozen before the historical rerun, **after** that invalid draft; this is not
a clean preregistration. A new public run reproduces this revised design.

Primary source: `research/CRIT_v0_8_PROTOCOL.md` at local archive snapshot
`8e354594f61200fedca1266068794919a4eb1d64`. Details of seed ranges, training,
simulation and bootstrap conventions were checked against
`benchmarks/run_v08_queue_transfer.py` and `crit_v08/queue_transfer.py` at the
same snapshot. These are provenance paths in the read-only archive, not public
runtime dependencies. This adaptation publishes no raw archive artifacts.
The [information limit of one-step probes](../theory/limitations.md) remains
separate from empirical map recovery and policy performance.

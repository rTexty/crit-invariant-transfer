# Synthetic queue strategy transfer: results and limits

**Status:** descriptive results of the revised historical CRIT v0.8 experiment.
The public reference values are verified by the C14 reproduction tests. This
report adds no observations and makes no claim of practical deployment or
extension of the [finite ambiguity theorem](../theory/identifiability.md).

## Design and comparison budget

The [reproduction protocol](queue-transfer-protocol.md) specifies independent
source reservoir and target FIFO simulators, a human-authored normalized
occupancy observation, and two opaque admission actions. The source threshold
was fitted once on 16 source-only seeds, selecting `h = 0.1` from
`[0.1, 0.3, 0.5, 0.7]`. Each regime has eight target calibration seeds,
12 target training seeds for the privileged comparator, and 20 disjoint test
seeds. Every rollout lasts 60 simulator steps.

The master seed is `S = 20260905`. For regime index `r = 0, 1, 2` in the
order below, calibration uses `S + 1000*r + i` for `i = 0,...,7`, training uses
`S + 1000*r + 40 + i` for `i = 0,...,11`, and testing uses
`S + 1000*r + 100 + i` for `i = 0,...,19`. Source fitting uses
`S + 100000 + i` for `i = 0,...,15`. The complete lists are retained in the
[compact reference](../../results/reference/queue-transfer.json).

Within each test seed, all six methods receive the **same** arrival, admission
uniform and service uniform arrays. The random policy additionally draws its
actions with `default_rng(test_seed + 99)`. Comparisons pair costs by test seed;
training and calibration do not use test outcomes.

| Method | Information and tuning budget |
| --- | --- |
| `fixed_high` | Direct semantic access to admission probability `0.95`; bypasses learning the opaque label. No target reward fitting. |
| `identity` | Source-fitted threshold and assumed map `low -> 0, high -> 1`; no target reward fitting or reset-map inference. |
| `random` | Uniform opaque action at each step; no target reward fitting. |
| `native` | Privileged access to target training costs: searches four thresholds and both action orientations on 12 target-train seeds per regime. |
| `transfer` | Source-fitted threshold plus reset occupancy fingerprints averaged over eight calibration seeds per regime; uses current target occupancy, with no target reward fitting or planted-map access. |
| `oracle` | Source-fitted threshold and the evaluator's planted map; a mapping diagnostic, not an optimal target-policy oracle. |

The transfer calibration also uses independent source-signature streams at
`calibration_seed + 50000`, as specified in the protocol. Calibration and source
fitting consume information; they are not free samples. The target reward and
map privileges differ across comparators, so this is not an equal-information
sample-efficiency test.

## Frozen cost results

Cost is dimensionless and lower is better:

```text
cost = mean post-service normalized occupancy
       + 0.8 * (rejected + overflow) / max(total arrivals, 1).
```

Each cell is a mean over 20 test seeds, rounded to six decimal places. All six
methods and all three regimes are shown; full precision is in the reference.

| Regime | `fixed_high` | `identity` | `random` | `native` | `transfer` | `oracle` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `balanced` | 1.422855 | 1.409509 | 1.399638 | 1.374910 | 1.374910 | 1.374910 |
| `overload` | 1.608855 | 1.599063 | 1.599376 | 1.589167 | 1.598502 | 1.598502 |
| `service_shift` | 1.643840 | 1.630507 | 1.631479 | 1.619919 | 1.630507 | 1.630507 |

The next table uses the reference's **mean of paired differences**,
`transfer cost - native cost`. Positive values mean transfer is worse.

| Regime | Mean paired difference | Interpretation relative to `native` | Inferred map equals planted map |
| --- | ---: | --- | --- |
| `balanced` | 0.000000000000000 | Tie | true |
| `overload` | 0.009334733893557 | Loss in mean cost | true |
| `service_shift` | 0.010587677435144 | Loss in mean cost | true |

Thus the transferred strategy ties the privileged target-tuned baseline in
one regime and loses in two. This is a comparison of means, not a claim that
transfer loses on every seed. Map recovery succeeds in **3/3 regime cases**:
one inferred map is fixed per regime across 20 evaluation seeds. Those 60
rollouts are not 60 independent mapping trials. Equality with `oracle` checks
the implementation of the planted action correspondence; it does not establish
generalization or optimal target control.

## Interpretation and limits

The regime names are labels. Even `balanced` does not assert queue stability;
the historical report describes all three finite-horizon cases as overloaded.
Costs are arbitrary frozen simulator objectives, not economic or production
metrics. The source abstraction and candidate threshold grid were supplied by
the researcher; the experiment does not discover a shared representation from
raw domains.

Mean and 95th-percentile latency in the runner concern **completed requests
only**, measured as `completion step - arrival step + 1`. Requests unfinished
at step 60 are excluded from latency and reported separately. A run with no
completions stores zero for both latency summaries; that is a convention, not
zero waiting time for all arrivals. Throughput, rejection, overflow and
unfinished work must be read together with latency. Request accounting is
`arrivals = rejected + overflow + throughput + unfinished`. The compact C14
reference retains cost, not these latency or accounting records.

The full runner offers paired descriptive bootstrap intervals under the
[protocol](queue-transfer-protocol.md). They describe seed variation conditional
on a fixed calibration and map per regime, not uncertainty across new domains,
refitted calibrations or independent experiments. No significance or broad
load/service-shift robustness claim follows here.

The historical protocol was revised after an invalid draft. This is not a clean
preregistration. Matching reset fingerprints can still conceal different
optimal continuations, as the separate
[two-world counterexample](../theory/limitations.md) demonstrates. Exact map
recovery and successful strategy transfer are distinct outcomes.

## Evidence and provenance

Numerical source: [queue-transfer reference](../../results/reference/queue-transfer.json),
with source and published SHA-256 hashes, reproduction command, numerical
tolerances and environment in the [manifest](../../results/reference/manifest.json).
The [reference tests](../../tests/test_reference_results.py) check the compact
schema and hashes and rerun both public CLIs under the `reproduction` marker.
The reproduction target is Python 3.14.4, NumPy 2.4.6 and SciPy 1.18.0;
manifest environment metadata is separate from historical queue-run metadata.

Historical interpretation and design were checked against
`research/CRIT_v0_8.md`, `research/CRIT_v0_8_PROTOCOL.md`,
`benchmarks/run_v08_queue_transfer.py`, and
`.ecc/benchmarks/crit_v08_queue_transfer.json` at archive snapshot
`8e354594f61200fedca1266068794919a4eb1d64`. These are provenance identifiers,
not public runtime dependencies. No raw archive or external data is published
by this report. See the [evidence register](../evidence.md) for claim boundaries.

# Evidence register

This register separates mathematical statements, synthetic checks and empirical
interpretation. It covers the evidence currently published here. A successful
mapping check does not establish a useful transferred policy. Numerical
verification reproduces a synthetic setup; it is not external validation.

| Conclusion | Status | Assumptions / design | Metric and result | Sources and verification | Limit |
| --- | --- | --- | --- | --- | --- |
| Finite action ambiguity determines one-step minimax regret in the one-hot utility case. | Theorem, with proof. | Nonempty finite admissible bijections; exact utility transport; one useful source action; randomized single decision; no extra map-distinguishing feedback. | Regret `1 - 1 / number of possible images`; TIG is the decrease after intersecting hypotheses with composition constraints. | [Formal model](theory/formal-model.md), [proof](theory/identifiability.md). | Does not bound sequential queue-policy cost, noisy calibration or arbitrary utility by the same closed form. |
| Rigid composition can remove decision-relevant ambiguity. | Synthetic computational check of the theorem. | Generated rigid examples with 2, 3, 4 and 5 actions; initially all permutations admissible. | Informed regret `0`; blind regret and TIG `0.5, 2/3, 0.75, 0.8`. | [Exact reference](../results/reference/identifiability.json), [solver tests](../tests/test_crit_v4_identifiability.py), [synthetic tests](../tests/test_crit_v4_synthetic.py), [reference reproduction](../tests/test_reference_results.py). | Constructed finite cases, not measured transfer between real domains. |
| Composition need not improve a decision. | Analytical counterexample and synthetic check. | Three-action left-zero law; all six permutations preserve composition. | Blind and informed regret `2/3`; TIG `0`. | [Limits](theory/limitations.md), [exact reference](../results/reference/identifiability.json), [reference reproduction](../tests/test_reference_results.py). | A counterexample to universal improvement, not a frequency estimate. |
| Reset fingerprints need not identify an optimal continuation. | Symbolic two-world counterexample. | Identical reset observations; opposite second-step optima in two deterministic worlds. | Randomized minimax expected cost `1`; world-aware cost `0`. | [Explicit construction and derivation](theory/limitations.md). | A logical witness, not an outcome measured in the queue simulator. |
| Reset fingerprints recover the planted queue action maps in the reported cases. | Descriptive synthetic result from revised historical v0.8; public reproduction available. | Eight calibration seeds and one fixed planted permutation per regime; supplied occupancy abstraction. | `inferred_map_equals_oracle = true` in all three regimes; `transfer` and `oracle` have equal mean cost. | [Protocol](experiments/queue-transfer-protocol.md), [results](experiments/queue-transfer-results.md), [queue reference](../results/reference/queue-transfer.json), [reference reproduction](../tests/test_reference_results.py). | `3/3` regimes, not 60 independent map trials; no unseen-domain or optimal-policy claim. |
| Map recovery does not yield a cost advantage over the target-tuned comparator here. | Descriptive synthetic policy comparison; includes negative transfer. | Six methods, 20 paired test seeds per regime; source fit on 16 seeds; privileged `native` tuning on 12 target-train seeds per regime. | Dimensionless cost, lower better. Mean paired `transfer - native`: `0` in `balanced`, `+0.009334733893557434` in `overload`, `+0.010587677435144393` in `service_shift`. | [All-method cost table and information budgets](experiments/queue-transfer-results.md), [queue reference](../results/reference/queue-transfer.json), [hashes and tolerances](../results/reference/manifest.json), [reference reproduction](../tests/test_reference_results.py). | Revised protocol after invalid draft; unequal comparator information; fixed simulator objective and finite horizon; no practical benefit or significance claim. |
| Queue latency summaries do not describe every arriving request. | Measurement definition in simulator and protocol. | Horizon 60; completed-request latency only; unfinished work counted separately. | Mean / p95 completed latency in simulator steps; zero summaries if no completion. | [Protocol and accounting](experiments/queue-transfer-protocol.md), [implementation](../crit_v08/queue_transfer.py), [interpretation](experiments/queue-transfer-results.md). | Censored to completions; compact cost reference does not retain latency records; no latency superiority claim. |

## Verification scope and provenance

The [manifest](../results/reference/manifest.json) identifies archive snapshot
`8e354594f61200fedca1266068794919a4eb1d64`, original artifact hashes, compact
reference hashes, reproduction commands, environment and tolerances.
Categorical fields compare exactly; numerical reference comparisons use
`rtol = 1e-9`, `atol = 1e-10`. Dates and platform are metadata.

The theorem and left-zero example derive from `research/CRIT_v0_4.md` at that
snapshot. The queue report derives from `research/CRIT_v0_8.md` and its revised
protocol at the same snapshot. Historical source paths are provenance, not
links to public files or dependencies on the private archive. The two-world
witness was adapted separately in the published limitations document.

The public tests verify the two examples and selected reference values. They do
not establish research priority, a new theorem for sequential transfer,
production readiness or performance on real systems. Prior work is discussed
in [related work](theory/related-work.md); broader limits are stated in
[limitations](theory/limitations.md). Later research stages require their own
published sources and verification before adding empirical claims here.

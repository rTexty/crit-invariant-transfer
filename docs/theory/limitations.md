# Limits of the transfer guarantees

The [finite ambiguity bound](identifiability.md) applies to a single
randomized decision with exact utility transport and a nonempty finite
set of admissible action bijections. Its closed form also requires unit
utility on exactly one source action. In the [general model](formal-model.md),
agreement on observed responses constrains the correspondence. Using that
correspondence to transfer a policy also requires information about the
target dynamics and objective.

## Composition can leave the optimal action ambiguous

Suppose both domains have three actions, every bijection is initially
admissible, and each domain has the composition law $x\circ y=x$. For any
bijection $\phi$, both $\phi(x\circ y)$ and $\phi(x)\circ\phi(y)$ equal
$\phi(x)$. Every bijection therefore satisfies the composition constraints,
and all three target images of the useful source action remain possible.
The bound gives regret $2/3$ before and after the comparison, so TIG is
zero.

## Identical one-step observations can hide opposite continuations

Consider two deterministic worlds, $W_0$ and $W_1$. At reset the observed
state is $q_0=0$. The first action $a_0\in\{0,1\}$ yields the observation
$q_1=a_0$ in both worlds. A calibration protocol that resets after each first
action distinguishes the two action signatures but receives identical data
in the two worlds.

At the second decision the hidden bit $m$ equals $a_0$ in $W_0$ and $1-a_0$
in $W_1$. The second action $a_1\in\{0,1\}$ incurs cost $0$ if $a_1=m$ and
$2$ otherwise.

| First action $a_0$ | Observation $q_1$ in either world | Optimal $a_1$ in $W_0$ | Optimal $a_1$ in $W_1$ |
| --- | --- | --- | --- |
| 0 | 0 | 0 | 1 |
| 1 | 1 | 1 | 0 |

Let $p$ be the probability that a policy chooses $a_1=a_0$, including any
randomness in its first action. The same $p$ applies in both worlds because
the policy receives the same history distribution before choosing $a_1$.
Expected costs are $2(1-p)$ in $W_0$ and $2p$ in $W_1$. Their sum is $2$, so
one of them is at least $1$. Choosing the second action uniformly achieves
cost $1$ in both worlds. The minimax expected cost is therefore $1$.
A policy told the world before the second action can choose $a_1=m$ and
pay $0$ in either world.

A two-step calibration probe can distinguish the worlds. For example, the
sequence $(a_0,a_1)=(0,0)$ costs $0$ in $W_0$ and $2$ in $W_1$. This symbolic
example concerns the information available in reset calibration. It is
separate from measured queue-simulator outcomes.

## Implications for empirical transfer

A recovered action map identifies a correspondence under the stated
protocol. Policy cost is a separate outcome because a transferred decision
also depends on how the target responds over time and how the target
objective values those responses. With noisy fingerprints, even the
retention of the true map in the admissible set requires a separate
statistical argument. The exact one-step proof supplies neither a
finite-sample error bound nor a guarantee for sequential policies.

Provenance: the left-zero example comes from CRIT v0.4 and the two-world
counterexample from CRIT v0.8, both at archive snapshot
`8e354594f61200fedca1266068794919a4eb1d64`.

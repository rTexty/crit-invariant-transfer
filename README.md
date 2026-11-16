# Mathematical Formalization of Cross-Domain Decision Strategy Transfer Based on Structural-Functional Invariants

**CRIT · Research code and mathematical notes**

When can a decision strategy learned in one domain be reused in another? CRIT (*Causal-Relational Intertwining Transport*) studies which relationships and action responses must be preserved for the available evidence to justify a choice in the target domain.

## Research question

Several action correspondences can explain the same observations. The relevant question is whether those correspondences imply different decisions. Recovering every label is unnecessary when all admissible correspondences agree on the best action. If they disagree, a transfer method must account for the remaining ambiguity.

CRIT examines relationships between roles, how systems respond to actions or compose them, and which outcomes matter for a decision. The result below asks how much decision quality can be lost when the available information leaves the action mapping ambiguous.

## A finite result

Consider finite source and target action sets and a nonempty set of admissible bijections between them. Exactly one source action has utility 1; the others have utility 0. Each bijection transports that utility exactly, so under each possible mapping exactly one target action also has utility 1. Let $m$ be the number of distinct possible target images of the useful action. For a single target choice with no additional evidence about the mapping, the minimum worst-case expected regret over randomized choices is

$$
R^* = 1 - \frac{1}{m}.
$$

Regret measures the shortfall in expected utility relative to a choice that knows the mapping. A uniform choice among the $m$ possible images attains the bound: it chooses the correct action with probability $1/m$ under every admissible mapping. For three images the regret is $2/3$; for one it is zero. Sequential decisions and noisy evidence require separate analysis.

## A separate synthetic example

A threshold strategy fitted in a reservoir model is reused in a simulated first-in, first-out queue. The two target actions have opaque identifiers; the example matches them to source actions using occupancy responses. The inferred mapping matched the planted mapping in each of the three tested regimes.

The strategy was evaluated on 20 test seeds per regime. The cost is mean occupancy plus $0.8$ times the fraction of dropped requests. Lower is better.

| Regime | Transferred strategy | Target-tuned baseline |
| --- | ---: | ---: |
| `balanced` | 1.374910 | 1.374910 |
| `overload` | 1.598502 | 1.589167 |
| `service_shift` | 1.630507 | 1.619919 |

The strategy tied a baseline tuned with target rewards in `balanced` and had higher cost in the other two regimes. The regime names are nominal: all three simulated queues experienced overload in the tested horizon. These are descriptive results from a revised synthetic protocol.

## Publication status

This repository is being prepared as a research artifact. The first release is planned to include the finite derivation, a runnable ambiguity benchmark, the queue experiment with all controls, and instructions for reproducing the reference results from a clean checkout. The scope is mathematical analysis and synthetic evaluation; external validation on real systems is open.

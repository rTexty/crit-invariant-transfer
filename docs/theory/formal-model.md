# A formal model of cross-domain strategy transfer

CRIT asks which features two domains must preserve for evidence from one domain
to justify a decision in the other. This note defines a general comparison and
then narrows it to the finite decision problem used in the first analysis. The
general comparison is a modeling choice; its preservation conditions do not by
themselves give a performance guarantee for transferred policies.

## Domains and observable structure

Use $d\in\{S,T\}$ to denote the source and target. In domain $d$, let $V_d$
denote observable roles, $A_d$ a discrete set of admissible actions, and $\mathcal H_d$ decision
histories. A history contains the actions and observations available before the
next choice. The analyst specifies which aspects of each domain can be compared:

| Information | Notation | Meaning |
| --- | --- | --- |
| Relations | $R_d(v,w)$ | Declared features of an ordered pair of roles, such as direction, sign, or a dependency. |
| Responses | $B_d(a,v,t)$ | The distribution of a specified observation of role $v$ at time $t$ under intervention $a$, with the initial conditions and observation scale fixed. |
| Composition | $C_d(a,b)$ | The action or program obtained by executing $a$ followed by $b$, where that operation is defined and represented in $A_d$. |
| Utility | $U_d(a)$ | The value assigned to a one-step action under the stated decision objective. |

An action correspondence $\phi:A_S\to A_T$ can be considered alongside a role
correspondence $\tau:V_S\to V_T$. These maps are evaluated against a declared
set of observations and comparisons. If the relation and response features
share units and observation times, examples of exact preservation are

$$
R_S(v,w)=R_T(\tau(v),\tau(w)),\qquad
B_S(a,v,t)=B_T(\phi(a),\tau(v),t).
$$

When comparing composition, the compared pairs must be defined in both
domains. Exact preservation for such a pair means

$$
\phi(C_S(a,b))=C_T(\phi(a),\phi(b)).
$$

The same role and action maps must satisfy the selected comparisons jointly.
If the domains use different units or time scales, the allowed transformations
must be specified before testing preservation. For example, a change of action
labels preserves composition when the operation is relabeled consistently. It
need not preserve the measured response or the value of the resulting decision.

## Functional measurements and decision objectives

A response fingerprint is a chosen statistic of the responses available under a
stated observation protocol. It might contain mean occupancy after each trial
intervention, for example. Approximate agreement requires a specified statistic,
distance, and tolerance; sampling uncertainty and the choice of tolerance are
separate from exact equality in the model.

Two actions can have equal fingerprints and differ under an unmeasured probe or
at a longer horizon. They can also have equal measured responses and differ in
utility because the objectives differ. Thus a structural match, a functional
measurement, and a statement about decision quality are separate claims.

## What it means to transfer a strategy

Let $\pi_S(\cdot\mid h_S)$ be a distribution over source actions at a source
history $h_S$. Write $\Delta(A)$ for the set of probability distributions on an
action set $A$. When a history map $\beta:\mathcal H_T\to\mathcal H_S$ and an
action map $\phi$ are available, one candidate transferred strategy is

$$
\pi_{S\to T}(\cdot\mid h_T)
=\phi_{\#}\pi_S(\cdot\mid\beta(h_T)).
$$

Here $\phi_{\#}$ sends the probability assigned to source actions to their target
images and sums the masses of actions with the same image. The expression defines
a candidate policy, not a claim that the history map can always be constructed or
that the policy retains its source value. Its target performance depends on the
target dynamics, admissible actions, and objective.

## Finite one-step setting

For the exact ambiguity problem, $A_S$ and $A_T$ are finite, nonempty and the
same size. Let $\operatorname{Bij}(A_S,A_T)$ be their set of bijections. The
observable comparisons restrict the unknown action map to a set
$H_0\subseteq\operatorname{Bij}(A_S,A_T)$. If exact composition is also observed,
retain only maps satisfying the composition comparisons, giving a subset
$H_C\subseteq H_0$. The analysis uses a nonempty admissible set $H$, for example
$H_0$ or a nonempty $H_C$. An empty set means that the assumed constraints do not
admit a map; the one-step decision model below does not cover that case.

Let $U_S:A_S\to\mathbb R$ be the known source utility. Under an admissible
bijection $\phi$, assume that the target utility is transported exactly:

$$
U_\phi(b)=U_S(\phi^{-1}(b)),\qquad b\in A_T.
$$

A decision rule must choose a distribution $q\in\Delta(A_T)$ before learning
which $\phi\in H$ is true; no further observation or action feedback is available
before that choice. The one-useful-action case in the [README](../../README.md)
then asks how far $q$ can fall short of the best action under the true mapping.
This setup does not assume that the mapping is unique. The relevant question is
which target actions remain plausible optima under the declared utility.

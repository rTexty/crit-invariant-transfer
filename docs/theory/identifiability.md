# The finite transfer ambiguity bound

Use the finite one-step setting of the [formal model](formal-model.md).
Let $A_S$ and $A_T$ be finite nonempty action sets of equal size, and let
$H_0\subseteq\operatorname{Bij}(A_S,A_T)$ contain the bijections consistent
with the initial observations. Write $C_S$ and $C_T$ for the source and target
composition laws. Retain the bijections that satisfy the exact composition
comparisons specified by the observation protocol.

$$
H_C=\{\phi\in H_0 :
\phi(C_S(a,b))=C_T(\phi(a),\phi(b))
\text{ for every compared pair }(a,b)\}\subseteq H_0.
$$

Each comparison uses a pair for which both compositions are defined. Assume
$H_C$ is nonempty when comparing the two sets. An empty $H_C$ means the observed
constraints admit no bijection. The argument below uses finite sets of maps and
requires no closure under composition.

## Minimax regret and the Transfer Identifiability Gap

For a known source utility $U\colon A_S\to\mathbb R$, assume that the target
utility under the true bijection $\phi\in H$ is exactly
$U_\phi(b)=U(\phi^{-1}(b))$. Here $H$ is a nonempty admissible set such as
$H_0$ or $H_C$. The decision maker chooses a distribution
$q\in\Delta(A_T)$ over target actions before receiving any further observation
that distinguishes the maps in $H$. Write $q_b$ for the probability of
choosing $b$. Relative to the best action with knowledge of $\phi$, the
worst-case expected regret minimized over $q$ is

$$
\mathfrak R(H,U)=\min_{q\in\Delta(A_T)}\max_{\phi\in H}
\left[\max_{b\in A_T} U_\phi(b) -
\sum_{b\in A_T}q_b U_\phi(b)\right].
$$

The Transfer Identifiability Gap (TIG) is the reduction in this value after
imposing the composition constraints.

$$
\operatorname{TIG}(U;H_0,H_C)=\mathfrak R(H_0,U)-\mathfrak R(H_C,U)\ge 0.
$$

For every $q$, the maximum over $H_C$ is at most the maximum over $H_0$.
Taking the minimum over the same distributions preserves the inequality.

## Exact bound for a single useful action

Suppose $U(a)=1$ for a single source action $a_*$ and $U(a)=0$ otherwise.
Let $O_H(a_*)=\{\phi(a_*):\phi\in H\}$ be the set of possible target images
of that action. Then

$$
\mathfrak R(H,U)=1-\frac{1}{|O_H(a_*)|}.
$$

For a given $\phi$, the best target utility is $1$, and drawing from $q$
gives expected utility $q_{\phi(a_*)}$. The worst-case regret is therefore
$1-\min_{b\in O_H(a_*)}q_b$. Put $k=|O_H(a_*)|$. The probabilities on these
$k$ actions sum to at most $1$, so their minimum is at most $1/k$. This
gives a lower bound of $1-1/k$ for every distribution. The uniform
distribution on $O_H(a_*)$ assigns each image probability $1/k$ and attains
the bound. Equality requires all $k$ probabilities to equal $1/k$, so this
distribution is the unique minimax rule. It follows that

$$
\operatorname{TIG}(U;H_0,H_C)
=\frac{1}{|O_{H_C}(a_*)|}-\frac{1}{|O_{H_0}(a_*)|}.
$$

In this utility model, reducing the number of admissible bijections changes
neither the optimal distribution nor the regret if every previously possible
image remains possible.

## Three-action example

Let $A_S=\{a_*,a_2,a_3\}$ and $A_T=\{b_1,b_2,b_3\}$. If $H_0$ contains all
six bijections, each target action is a possible image of $a_*$. At least
one of the three probabilities is at most $1/3$. A uniform draw achieves
$1/3$ under every map, so $\mathfrak R(H_0,U)=2/3$.

Take total composition laws $C_S(a,b)=a_*$ and $C_T(x,y)=b_1$ for all
source pairs $(a,b)$ and target pairs $(x,y)$. Their exact comparison reduces
to $\phi(a_*)=b_1$. The other two source actions can still be matched in
either order, so $H_C$ contains two bijections. Choosing $b_1$ with
probability $1$ achieves the best utility under both maps, giving
$\mathfrak R(H_C,U)=0$ and $\operatorname{TIG}(U;H_0,H_C)=2/3$.

If an additional observation instead eliminates just one of the six initial
bijections, all three images are still possible because each occurs in two
of the original maps. The minimax rule remains uniform on the three actions,
and the regret remains $2/3$.

## Scope

The closed form assumes exact utility transport, unit utility at a single
source action, a nonempty finite set of bijections, and a single randomized
choice without further feedback. The general regret definition also covers
utilities spread across multiple actions, for which the full utility vectors
enter the optimization. Noisy observations, approximate utility transport,
and sequential decisions with feedback require separate analysis.

# Related work

The [formal model](formal-model.md) separates three inputs to transfer:
relations among roles, observed responses to actions, and decision utility.
The works below provide comparisons for different parts of that model.

## Relational and behavioral comparison

- Relational alignment draws on Gentner's structure-mapping account of
  analogy, in which systems of relations guide alignment between a base and
  a target. In CRIT, relation preservation is a constraint on candidate role
  maps. [Gentner (1983)](https://onlinelibrary.wiley.com/doi/10.1207/s15516709cog0702_3)
- Gromov–Wasserstein compares metric measure spaces through their internal
  distances. Fused Gromov–Wasserstein jointly compares structure and features,
  and Gromov–Wasserstein learning has been used to align graphs and learn node
  embeddings. These are comparators for finding structural correspondences.
  CRIT's one-step analysis evaluates a decision given a set of admissible
  action correspondences.
  [Mémoli (2011)](https://link.springer.com/article/10.1007/s10208-011-9093-5),
  [Vayer et al. (2019)](https://proceedings.mlr.press/v97/titouan19a.html),
  [Xu et al. (2019)](https://proceedings.mlr.press/v97/xu19b.html)
- Coalgebraic bisimulation relates states through their behavior. Linear
  realization theory constructs state-space models from input–output
  information. Both motivate distinguishing an observation law from its
  internal realization. The assumptions on CRIT's observations are defined
  by the probe protocol.
  [Rutten (2000)](https://www.cs.cornell.edu/courses/cs6861/2024sp/Handouts/Rutten.pdf),
  [Ho and Kalman (1966)](https://ntrs.nasa.gov/citations/19670049337)
- Bisimulation metrics for continuous-state Markov decision processes link
  behavioral similarity to continuity of optimal discounted value under the
  paper's assumptions. Such metrics provide a comparison for sequential
  decision problems with specified transition and reward models.
  [Ferns, Panangaden, and Precup (2011)](https://epubs.siam.org/doi/10.1137/10080484X)

## Causal abstraction and intervention transfer

- Exact causal transformations require predictions under corresponding
  interventions to agree across model levels. Causal abstraction refines
  which transformations qualify, with extensions for approximate agreement,
  soft interventions, and lossy representations. This line of work motivates
  requiring preservation under declared interventions. CRIT's finite model
  imposes exact action-composition comparisons on admissible bijections.
  [Rubenstein et al. (2017)](https://proceedings.mlr.press/r15/rubenstein-17a.html),
  [Beckers and Halpern (2019)](https://ojs.aaai.org/index.php/AAAI/article/view/4117),
  [Beckers, Eberhardt, and Halpern (2020)](https://proceedings.mlr.press/v115/beckers20a.html),
  [Massidda et al. (2023)](https://proceedings.mlr.press/v213/massidda23a.html),
  [Xia and Bareinboim (2025)](https://proceedings.mlr.press/v267/xia25a.html)
- Causal Optimal Transport of Abstractions (COTA) learns abstraction maps
  from observational and interventional data using multi-marginal optimal
  transport with causal constraints. Graph-adapted causal optimal transport
  defines distances between causal models through transport plans that
  respect a specified causal graph. These approaches address different
  correspondence and comparison problems within causal models.
  [Felekis et al. (2024)](https://proceedings.mlr.press/v236/felekis24a.html),
  [Cheridito and Eckstein (2023 preprint, revised 2024)](https://arxiv.org/abs/2303.14085)
- Causally abstracted multi-armed bandits study sequential transfer across
  causal models related by an abstraction map. CRIT's finite theorem
  fixes the information available before a single target choice and
  measures its worst-case loss across the remaining maps.
  [Zennaro et al. (2024)](https://proceedings.mlr.press/v244/zennaro24a.html)

## Symmetry and decision ambiguity

SMDP homomorphisms aggregate state–action pairs while preserving expected
rewards and aggregate dynamics. EqR learns representations equivariant to
state–action symmetries for reinforcement learning. In causal representation
learning, interventional observations can constrain latent factors up to
specified transformations, such as permutation and scaling under the
conditions for perfect interventions.
[Ravindran and Barto (2003)](https://all.cs.umass.edu/pubs/2003/ravindran_b_IJCAI03.pdf),
[Mondal et al. (2022)](https://proceedings.mlr.press/v162/mondal22a.html),
[Ahuja et al. (2023)](https://proceedings.mlr.press/v202/ahuja23a.html)

The [finite ambiguity analysis](identifiability.md) defines the Transfer
Identifiability Gap (TIG) as the difference in minimax regret before and
after adding exact composition constraints. With unit utility on one
source action, the proof makes regret depend on the number of its possible
target images. This links the remaining correspondence ambiguity to the
value of a target decision. The [continuation counterexample](limitations.md)
shows how one-step observations can leave a different decision unresolved.

The CRIT v0.1 bibliography also draws motivation from studies of relational
integration in human analogy and from a review of memory generalization.
These motivate the choice of research question. The finite regret result
follows from its mathematical assumptions.
[Green et al. (2006)](https://bpb-us-e1.wpmucdn.com/sites.dartmouth.edu/dist/4/26/files/2012/09/Green-2006-BrainRes.pdf),
[Zeithamova and Bowman (2020)](https://pmc.ncbi.nlm.nih.gov/articles/PMC7655622/)

Provenance: adapted from the literature discussion in the CRIT v0.1 and
v0.4 research notes at archive snapshot
`8e354594f61200fedca1266068794919a4eb1d64`.

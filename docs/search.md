# Information-set search

This document defines Sam, the deterministic round-local opponent used for
local gameplay, and BGC-128, the information-safe teacher used by the active
dataset miner. Sam-32 and earlier neural-search experiments are historical
evidence indexed under [reports](../reports/README.md); they are not active
alternatives.

## Objective

Every completed search simulation returns the exact engine result:

```text
z(player) = (round_score[player] - round_score[other]) / 150
```

The result is in `[-1, 1]`, and changing player perspective negates it. The
engine score already includes line order, ranked-line resolution, multipliers,
Vampires, and ties. Search never substitutes a hand-authored move score or an
estimate of an incomplete coffin.

Search ends with the current round. All eight hand cards are consumed within
that round, so a placement cannot change later stock or preserve a card for a
later round. The current game-score differential is not part of the search
payoff.

## Information boundary

At a decision, the acting player may receive:

- Their role, dealer status, remaining cards, and stable hand slots.
- The public coffin and accepted current-round moves.
- Completed public scores.
- Hand and stock counts.
- Their legal card-destination actions.

They may not receive the other player's remaining hand, private hand-slot
indexes, stock order, game seed, authoritative hidden assignment, model state,
or an enclosing search's sampled-world identity.

Queen uses the authoritative coffin orientation. King uses its transpose, so
the acting player's scoring lines are horizontal in both views. The typed
information state, 875-bit projection, legal mask, and 32-action map are
defined in the [engine–opponent contract](engine-model-contract.md).

Equal player-visible information states have the same fingerprint regardless
of authoritative hidden-card arrangements.

## Determinization

A search request samples only worlds consistent with the relevant acting
player's information:

1. Fix all public cards and the acting player's remaining hand.
2. Form the unseen pool from the canonical deck minus known cards.
3. Sample the required opponent-hand cards uniformly without replacement.
4. Reconstruct stable hand slots from the sampled and publicly played cards.
5. Assign the remaining unseen cards to a deterministic simulated stock.
6. Rebuild an engine-valid immutable simulation state.

The sample seed derives from the information-state fingerprint, controller
configuration, request identity, and simulation index. It never derives from
the authoritative opponent hand, stock order, engine seed, or private move
slots.

## Sam nested response search

Sam uses one information-set UCT search for the real decision and a separate
actor-local UCT search for every non-forced simulated continuation decision.

```text
outer simulations:           32
actor-response simulations:  32
outer exploration:           sqrt(2)
response exploration:        sqrt(2)
terminal evaluation:         exact completed-round differential
```

One outer simulation:

1. Samples an engine-valid world from the real actor's information state.
2. Selects or expands one legal outer action with UCT.
3. For each later non-forced decision, constructs a fresh information state
   for that simulated actor.
4. Runs a 32-simulation actor-local search from that information state.
5. Applies only the actor-local search's selected legal action to the unchanged
   outer sampled world.
6. Continues until the engine completes the round.
7. Backs up the exact result from the real actor's perspective.

The actor-local search samples its own hidden assignments. It does not receive
or preserve the outer search's hidden allocation. Its cache identity contains
the actor's information-state fingerprint and complete response configuration,
not the root hand, stock order, engine seed, or sampled-world identity.

Each node stores values from its acting player's perspective. Values are
negated when backed up across player perspectives. Both outer and actor-local
selection use:

```text
Q(s, a) + c * sqrt(log(N(s)) / N(s, a))
```

Every legal strategic action is visited before the exploration rule is used.
The selected real group has the highest visit count; mean value and canonical
group index break ties. A forced one-action placement bypasses both search
budgets and is applied directly.

## Authoritative early-turn symmetry

Sam-32 applies only the destination groups below. Coffin positions are numbered
1 through 9 in reading order. `C` marks an occupied position. Card identity,
rank, suit, color, and Vampire status do not affect grouping.

Center only:

```text
1 2 3
4 C 6
7 8 9

representative 2 -> (2, 8)
representative 4 -> (4, 6)
```

Center plus position 2:

```text
1 C 3
4 C 6
7 8 9

representative 1 -> (1, 3)
representative 4 -> (4, 6)
representative 8 -> (8)
```

Center plus position 8:

```text
1 2 3
4 C 6
7 C 9

representative 2 -> (2)
representative 4 -> (4, 6)
representative 9 -> (7, 9)
```

Center plus position 4:

```text
1 2 3
C C 6
7 8 9

representative 1 -> (1, 7)
representative 2 -> (2, 8)
representative 6 -> (6)
```

Center plus position 6:

```text
1 2 3
4 C C
7 8 9

representative 2 -> (2, 8)
representative 3 -> (3, 9)
representative 4 -> (4)
```

Completed horizontal center line:

```text
1 2 3
C C C
7 8 9

representative 1 -> (1, 7)
representative 2 -> (2, 8)
representative 3 -> (3, 9)
```

Completed vertical center line:

```text
1 C 3
4 C 6
7 C 9

representative 1 -> (1, 3)
representative 4 -> (4, 6)
representative 7 -> (7, 9)
```

Every other occupied-position pattern retains each legal destination as its own
group. The two three-card cases are recognized only from their exact board
patterns.

Legal actions are partitioned first by hand card and then by destination group.
Different hand cards always remain different strategic actions. Outer and
actor-response UCT expansion, visits, values, selection, and backup operate at
the strategic-group level.

After a paired group is selected, a versioned derived fair coin selects one
concrete destination. The coin cannot change group statistics. An unpaired
group maps directly to its sole destination, and the engine validates the
concrete action before transition.

## Sam and Sam-32

The historical Sam controllers share the same 32×32 nested-search budgets and
exact terminal scoring.

- **Sam** is the strong local gameplay controller.
- **Sam-32** adds the authoritative destination grouping at outer and
  actor-response nodes and emits one selected strategic group for training.

The group selection was the historical Sam teacher target. Its visit
distributions and action values remained private diagnostics. Branching,
artifact identity, cache, and resume behavior are defined in the historical
[Sam dataset miner](sam-dataset-miner.md).

## Active belief-greedy teacher

Belief-greedy search is the active dataset teacher. It keeps
one outer information-set UCT tree and replaces every nested actor-response
tree with a deterministic, one-ply expected-score comparison.

At each non-forced simulated response:

1. Project the acting player's `SearchInformationState` from the sampled
   simulation. The response evaluator receives no other simulation data.
2. Enumerate the exact strategic action groups defined above.
3. For each of eight deterministic belief completions, sample an opponent
   remaining hand uniformly from that actor's unseen-card pool. The same hand
   sample and completion ordering are shared across every candidate group.
4. Apply each candidate representative action, assign the remaining known and
   sampled round cards to the open coffin positions with the shared completion
   ordering, and score the completed coffin with the engine.
5. Average the exact normalized actor-relative round differential for each
   group and select the maximum. Canonical representative index breaks an
   exact tie; the existing derived fair coin resolves a paired destination.

The response policy builds no inner search tree. Its selected action is
deterministic for an information-state fingerprint and configuration digest;
the Monte Carlo samples integrate uncertainty about hidden cards rather than
reading the outer world's hidden assignment. The outer tree still completes
every simulation through the engine and backs up exact round results.

The companion balanced miner creates one root-search distribution at each learned placement and
then advances the trajectory through one of five deterministic profiles:
all-teacher, one deviation per round, two deviations per round, mixed teacher
and alternative actions, or all-alternative actions. Alternatives are sampled
uniformly from legal strategic groups excluding the teacher group. A complete
six-round game therefore contains exactly six rows for each placement 1–7 and
42 rows total; forced eighth placements create no row.

The active collection profile uses 128 outer simulations, eight belief
completions per response, four complete-game workers, and the five trajectory
profiles above. It saves exact strategic-group root visits summing to 128; the
most-visited group remains audit data rather than a one-hot label. Historical
corpus artifacts are never mixed with belief-greedy rows.

## Iterative policy-adversary direction

The first balanced corpus is labeled entirely by the 128-outer belief-greedy
controller. A standalone policy learns its normalized strategic-group visit
distribution with masked distributional cross-entropy.

For D1, a frozen copy of `pi0` replaced the hand-written belief-greedy response
choice at simulated opponent nodes.
The outer 128-simulation information-set tree, actor-local visible inputs,
engine transitions, and exact completed-round backup remain unchanged. The
policy supplies one inexpensive continuation action; it does not receive the
authoritative sampled hand and it does not become a value cutoff or a live
hybrid controller.

Each iteration retains its own dataset and model identity:

```text
128-outer belief-greedy corpus D0
    -> standalone policy π0
    -> 128-outer search with π0 simulated responses, corpus D1
    -> standalone policy pi1
    -> user selection for production
```

The deployed artifact is the standalone `pi1` classifier. Iterative collection
made its teacher continuations reflect learned play while reducing response-
evaluation work. This training sequence is complete.

For the completed controller comparison, base BGC-128 used the one-ply
belief-greedy response above and candidate BGC-128 used `pi0` for simulated
continuation responses. Every other outer-search property is identical:
128 simulations, information-state sampling, exact engine transitions and
terminal scoring, request seeds, symmetry, legal actions, forced placements,
and backup. Each model call received only that simulated actor's own
information state. `pi0` supplied no value cutoff, root prior, or hidden-state
estimate.

The retained accepted-policy adapter loads only the checkpoint named by its
selection manifest. It verifies the checkpoint and state
dictionary, training snapshot and configuration, evaluation fixtures and all
three reports, source identities, and the observation, action, symmetry, and
representative-mask contracts. One invocation accepts only the acting
simulated player's `SearchInformationState`, performs one policy forward pass,
sets illegal and non-proxy probabilities to zero, selects the canonical masked
argmax, and applies the established paired-destination coin. There is no
belief-greedy, random, Sam, or other policy fallback.

The candidate outer planner subclasses the existing belief-greedy outer UCT
and overrides only its actor-response method. Its outer configuration remains
128 simulations and retains the base search digest and request seed derivation;
its separately versioned controller digest additionally binds the accepted
checkpoint and response adapter. Base BGC-128 remains explicitly constructible
for the required comparison.

## Standalone policy projection

The standalone classifier does not score or average complete strategic groups.
Before training or inference, an external projection builds a
representative-action mask from the exact table above:

- The designated representative remains legal for every legal hand card.
- The paired non-representative destination is masked.
- An unpaired group retains its sole action.
- An unlisted occupied-position pattern retains every engine-legal action.

Each strategic group's exact root visits are assigned to its designated proxy
action and normalized by the 128-simulation total. The selected representative
is retained for audit metrics but is not converted into a one-hot loss. At
inference, argmax selects one representative action and the existing
`derive_strategic_destination_choice_seed` stream, using scope
`sam-policy-argmax-result-v1` and choice index zero, resolves a paired concrete
destination. That coin does not affect the learned class.

## Determinism and safety

A search result records controller and schema digests, request identity, legal
strategic groups, group visits and means, selected group, selected concrete
action, simulation counts, node counts, latency, and a concise continuation.
Diagnostics are server-private.

Versioned seed namespaces separate request identity, outer determinization,
outer selection, actor-response request, actor-response determinization,
actor-response selection, and concrete destination choice. Equal information
states, configurations, and request seeds reproduce the same result.

Search operates only on immutable engine states. Interruption cannot mutate the
authoritative game or commit a partial opponent turn. Retrying the same claimed
turn reproduces its input and action.

Tests enforce:

- Per-actor information isolation under hidden-card substitutions.
- Exact legal-action and symmetry partitions.
- Exact terminal values and perspective signs.
- Deterministic visits, values, groups, and concrete actions.
- Forced-placement bypass.
- Input-state immutability and interruption safety.
- No private search data in API responses or browser state.

## Historical evidence

The original 500-simulation planner, shallow Teacher v2, response-ranker
hybrids, guided search, expert iteration, and Sam-128 remain preserved under
[historical reports](../reports/README.md). Direct browser comparison selected
nested Sam over the weaker or shallower controllers. Sam-128 supplied cost
evidence before Sam-32 was selected for continuous collection.

Nested Sam remains a strong explicit local comparison. Search is not part of
the final deployed move path; standalone `pi1` is the selected release
controller.

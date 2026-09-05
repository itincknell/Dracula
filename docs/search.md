# Search, symmetry, and policy move selection

Standalone `pi1` remains the selected production opponent. The repository also
retains the two BGC-128 controllers that produced its training lineage: the
original belief-greedy continuation and the phase-two policy continuation.
They are maintained search and evaluation code, not Lambda gameplay fallbacks.

## Information boundary

The policy receives a typed `SearchInformationState` for the acting player.
It contains that player's remaining cards, the public coffin and history,
public scores and lifecycle facts, counts for hidden locations, and legal
actions. It contains no opponent hand, stock order, engine seed,
determinization, search tree, or model state.

Queen retains global grid coordinates. King uses the self-inverse transpose
defined by the engine contract, so the acting player's scoring lines are rows.
Equal player-visible states have the same fingerprint regardless of hidden
card locations.

## Authoritative early-turn symmetry

The following table is exhaustive. Coffin positions are numbered 1 through 9
in reading order. `C` marks an occupied position. Card identity, rank, suit,
color, and Vampire status do not affect grouping.

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

Every other occupied-position pattern retains each legal destination as its
own group. The two three-card cases are recognized only from their exact board
patterns. Different hand cards always remain separate actions.

## BGC-128 outer search

`BGCInformationSetSearch` runs 128 round-local UCT simulations. Each simulation
samples a complete hidden world using only the root player's information state.
Whenever another simulated actor must decide, that actor receives a newly
projected `SearchInformationState`; the continuation policy never receives the
sampled opponent hand, stock order, or root tree.

UCT expands, selects, records visits, and backs up values by strategic group.
Mirrored concrete destinations never become separate tree choices. After a
group is selected, the existing deterministic fair coin chooses its concrete
member for the engine transition. Root visit targets therefore contain one
count per representative action and zero counts for non-proxy members. Exact
completed-round engine score differentials supply terminal values.

The retained configuration is:

- 128 outer simulations;
- exploration constant `sqrt(2)`;
- the exhaustive destination-symmetry table in this document;
- maximum visit count, then mean value, then canonical representative index for
  the final root choice.

## Continuation policies

The original BGC continuation evaluates every legal strategic group over eight
shared samples of the acting player's unseen cards. For each group it places the
candidate card at the representative destination, fills the remaining round in
a deterministic sampled order, and uses exact engine scoring. It chooses the
highest mean actor-relative differential, breaking an exact tie by canonical
representative index.

The phase-two continuation replaces only that response calculation with one
policy inference. It uses the current 659-bit observation encoder, final model
artifact loader, representative mask, canonical argmax, and paired-destination
coin. Outer sampling, UCT, symmetry, transitions, and terminal scoring remain
the same. The adapter is artifact-neutral: the D1 experiment supplied `pi0`;
the same final tensor contract can load another verified policy artifact for a
controlled comparison.

## Representative actions

The external action projection keeps one proxy destination per strategic
group for every current hand card. Non-proxy members of listed pairs are
masked; unpaired and unlisted legal actions remain available. The model never
averages logits across paired destinations.

The policy selects the greatest legal representative logit, with flattened
action index as the deterministic tie-breaker. A separately derived fair coin
then selects the concrete member of a paired group. That coin cannot change
the representative action or model logits, and the engine revalidates the
resolved concrete move.

## Determinism and safety

The active path constructs the 659-bit observation directly from actor-visible
data, derives candidate rows from in-hand card membership in fixed card-ID
order, and maps the selected candidate back through the typed information
state. Retrying the same game history reproduces the same representative and
concrete actions. Diagnostics remain server-private.

BGC request, determinization, tree-selection, continuation, and concrete-choice
seeds use separate versioned namespaces. Search results and sampled principal
continuations are private diagnostics and are not part of any public API.

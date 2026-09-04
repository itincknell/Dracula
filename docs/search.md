# Symmetry and policy move selection

Search controllers are not part of the selected product. This document owns
the strategic destination grouping retained by standalone `pi1`. Historical
search designs and measurements remain in [reports](../reports/README.md).

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

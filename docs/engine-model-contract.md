# Engine–opponent contract

This document defines the deterministic boundary between the game engine and
the selected standalone `pi1` policy. HTTP replay belongs in
[stateless gameplay API](stateless-api.md); model structure belongs in
[neural model](neural-model.md).

## Card identity and deterministic randomness

The engine uses the 52 standard cards followed by `V1` and `V2`, ordered by
clubs, diamonds, hearts, spades, then Vampires. Rank order within a suit is
Ace through King. This 54-card order defines card indexes, hand sorting,
serialized state, observation features, and fixtures.

The engine creates one local `random.Random` instance from the caller's game
seed. That generator shuffles the deck and then selects the initial dealer.
The stable seed conversion uses SHA-256 so results do not depend on Python's
process-randomized string hash. Search and fair coins likewise use independent
local generators constructed from the facts that identify their operation.

## Engine ownership

`EngineState` is immutable and contains the private seed, stock, both hands,
coffin, current moves, completed rounds, scores, active player, dealer, and
lifecycle status. The engine owns:

- pair dealing and canonical four-slot hands;
- orthogonally adjacent legal placements;
- immutable move application;
- exact line and round scoring;
- six-round lifecycle transitions;
- state validation and card conservation;
- canonical private serialization and fingerprints.

`legal_moves` orders moves by stable hand slot and global grid index.
`apply_move` validates lifecycle, actor, slot, vacancy, and adjacency. The
eighth placement scores the round. `advance_after_round` alternates the dealer
and deals the next round, or completes the game after round six.

Private engine move records retain the original hand slot for deterministic
validation. Public move records retain only actor, card, destination, and turn
number; an opponent's former slot is not public information.

## Player-visible information

The policy receives `SearchInformationState`, never `EngineState`. Its fields
are:

```text
player, round, dealer, active player, turn number
public completed rounds and scores
acting player's canonical four-slot hand
player-relative coffin and public current moves
played-card IDs and unseen-card IDs
opponent remaining-card count and stock count
engine legal mask, bool[4,8]
```

Unseen IDs identify a belief set, not hidden locations. The type contains no
opponent hand, stock order, engine seed, determinization, search tree, or model
state. Public cards, own cards, and unseen cards partition the 54-card deck.

Queen retains global grid coordinates. King uses the self-inverse transpose
`index -> 3 * (index % 3) + index // 3`, so each actor's scoring lines are
rows. Coffin positions and legal destinations use the same orientation.

## Direct policy observation

The selected runtime constructs this Boolean tensor directly from the typed
information state:

```text
observation[0:486]     = player-relative coffin, bool[9,54]
observation[486:540]   = played cards, bool[54]
observation[540:594]   = own remaining cards, bool[54]
observation[594:648]   = unseen cards, bool[54]
observation[648:654]   = round index, one-hot bool[6]
observation[654:658]   = own decision index, one-hot bool[4]
observation[658]       = acting player is dealer, bool
```

The total is `486 + 162 + 11 = 659` bits. The three card-status vectors
partition the deck. Player identity, scores, move order, hidden locations,
and legal-mask bits are not model features.

Earlier training data carried a redundant `bool[4,54]` stable-slot prefix.
The sealed selected corpus was physically converted to the exact 659-bit
layout. The active runtime does not construct or compact that older tensor.
The retained corpus reader accepts only the converted dataset format.

## Candidate rows and action mapping

The policy returns raw `float32[4,8]` logits. The eight columns correspond to
player-relative non-center coffin positions in this order:

```text
0, 1, 2, 3, 5, 6, 7, 8
```

The set bits in the 54-bit in-hand vector determine the one to four current
cards in fixed card-ID order. Those cards occupy temporary candidate rows
zero through three. Padding rows are illegal. Candidate row identity is not a
model input and has no learned embedding.

The typed information state retains stable engine slots. After selection, the
runtime maps the candidate card back to its occupied engine slot, combines it
with the selected destination, and verifies that the resulting 32-index action
is present in the engine action table.

## Symmetry and legality

The full engine mask remains external to the model. The authoritative table in
[symmetry and move selection](search.md#authoritative-early-turn-symmetry)
partitions legal destinations into strategic groups. One designated proxy per
group remains true in the representative mask for each current card. Paired
non-proxy destinations are masked; unlisted patterns retain every legal
destination independently.

Inference replaces illegal and non-proxy logits with negative infinity,
selects the greatest representative logit with canonical flattened-index
tie-breaking, and maps that candidate action back to an engine-slot action.
A deterministic local fair coin resolves paired destinations.
The engine validates the concrete move before transition.

Forced eighth placements bypass model inference and apply the engine's sole
legal move directly.

## Training boundary

The retained trainer reads only the sealed converted `pi1` corpus. Each row
contains a 659-bit observation, full legal mask, exact strategic groups,
representative visits summing to 128, and audit metadata. It reconstructs the
representative mask and trains with distributional cross-entropy over legal
proxy actions. Values, returns, selected-action one-hot loss, illegal-action
loss, PPO state, recurrence, and critics are absent.

## Required invariants

- Direct observation construction matches the converted selected corpus bit
  layout exactly.
- Candidate ordering follows fixed card-ID order and is independent of stable
  engine-slot position.
- Queen and King use the same player-relative model contract.
- Representative masks exactly partition current legal strategic choices.
- Fixed game history and artifact bytes reproduce logits and actions.
- The model never receives authoritative hidden locations.
- Public API responses contain no observation, mask, logit, weight, or private
  engine object.

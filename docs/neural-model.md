# Search-guided neural model

The learned opponent is a later optimization of the validated search player.
The gates in [information-set search](search.md#search-only-gates) have passed.
Model implementation begins after the remaining architecture values in this
document are finalized.

## Role

One shared network will learn two outputs from a player-relative information
state:

```text
policy logits p: float32[32]
round value v:   float32 scalar in [-1, 1]
```

The policy head approximates the root visit distribution produced by search.
The value head approximates the normalized terminal round-score differential
from the acting player's perspective. The heads share one encoder; there is no
independent critic.

The network does not own legality, state transition, determinization, scoring,
or action selection. The engine-derived mask is applied outside the model. The
model receives no opponent hand, stock order, seed, authoritative state,
search determinization, or private diagnostics.

## Information-state input

The model input is derived from `SearchInformationState` in the
[engine–opponent contract](engine-model-contract.md#player-relative-information-state).
The current 875-bit observation plus its separate legal mask is a valid starting
point for the first round-local model:

- Four stable positions for the acting player's current hand.
- The player-relative 3×3 coffin with card identity.
- Played, in-hand, and unseen status for every canonical card.
- Dealer and current decision progress.
- The 32-entry legal mask.

Queen uses the authoritative orientation and King uses its transpose, including
public move destinations and action indexes. The input therefore does not need
a Queen/King feature.

The projection is Markov-sufficient for the version 1 uniform belief and
opponent model. Public move history remains in the search decision record for
engine reconstruction and audit but is not a neural feature unless the search
begins conditioning hidden-card beliefs or opponent behavior on that history.
Completed-round layouts and cumulative scores remain outside the first
round-local model. If deployment later optimizes game-win probability rather
than additive score differential, that is a new model contract.

The tensor contract must be mechanically revalidated against the typed
information state before implementation. The shared architecture and exact
parameter count remain to be finalized from the measured search dataset shape
and throughput.

## Action contract

The action space remains the fixed Cartesian product of four own hand slots and
eight non-center coffin positions:

```text
action_index = 8 * hand_slot + policy_position_index
policy positions = [0, 1, 2, 3, 5, 6, 7, 8]
```

The network returns unmasked logits. Training and inference replace illegal
logits with negative infinity before softmax. A forced final placement bypasses
search and model inference and supplies no policy-loss sample.

## Feed-forward architecture

The first search-guided model is feed-forward. Its encoder will use shared card
embeddings and structured encoders for hand positions, coffin positions,
unseen-card status, and context. A shared fused body feeds:

- A 32-logit policy head.
- A bounded scalar value head.

Recurrence is not justified in the first model. The explicit information state
contains the current round history needed by search, and the round-local target
does not reward adaptation across rounds. Removing recurrence also removes
game-scoped hidden-state persistence, 24-step replay, and backpropagation through
game trajectories. Recurrent opponent adaptation requires separate evidence and
a new contract.

Exact embedding widths, body width, normalization, parameter budget, and
initialization are selected after search throughput and dataset shape are
measured. The initial model must remain small enough for batched local training
and low-latency CPU inference on the target laptop.

## Training targets

For a non-forced decision with legal root visit counts `N(a)`:

```text
pi(a) = N(a) / sum_b N(b)
z = (round_score[player] - round_score[other]) / 150

L = -sum_a pi(a) * log softmax(masked_logits)(a)
    + value_weight * (v - z)^2
    + l2_weight * ||parameters||^2
```

Only legal actions have nonzero `pi`. The target is the search distribution,
not the sampled move and not a hand-authored move score. The same exact terminal
`z` is attached to every non-forced decision by that player in the round.

Search reaches the end of the round during the first expert-iteration phase, so
its backups use exact engine scoring rather than the network value. After the
model is demonstrably accurate, its value may evaluate cut-off leaves and its
policy may supply search priors. That change requires a comparison showing
equal or stronger play at lower measured cost.

## Dataset boundary

Each example contains:

```text
contract versions and search configuration digest
information-state tensor and digest
legal mask
search visit distribution
selected action
exact terminal round return
fixture, role, dealer, round, and decision indexes
```

Examples come from actual self-play decision roots. Internal determinizations,
opponent private hands, stock orders, and tree nodes are not training examples.
Rows are immutable after sealing and split by fixture seed so no deck fixture
appears in both training and validation.

## Acceptance

Before neural guidance can affect search or gameplay:

- Tensor encoding is invariant under Queen/King normalization and hidden-card
  substitutions outside the player's information set.
- Masked policy probabilities are finite, normalized, and zero for illegal
  actions.
- Value outputs are finite and bounded.
- Policy and value losses train the shared body in one backward pass.
- Batch and individual forwards agree within device tolerances.
- CPU and MPS optimization are benchmarked; the faster sustained device with no
  unsafe memory or swap growth is selected.
- A fixed control evaluation shows guided search is no weaker than the
  search-only teacher at the selected latency budget.

## Artifact boundary

A candidate artifact contains its state dictionary, architecture and tensor
schemas, card and action order, source revision, dataset and search-report
digests, training configuration, and weight digest. One checkpoint is selected
manually from absolute evaluation evidence.

The deployment contract is not selected yet. Search latency and strength will
determine whether the application serves search, search guided by the network,
or a distilled network alone. No deployment-specific hidden state or endpoint
shape is part of the model design.

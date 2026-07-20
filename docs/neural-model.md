# Neural model

This document defines the learned opponent policy. The deterministic state,
projection, and action bridge belong in the
[engine–model contract](engine-model-contract.md); application serving and
persistence belong in [architecture](architecture.md); training systems and
model comparison belong in [model training](model-training.md).

## Policy boundary

The policy is a recurrent neural network. On each turn owned by a
policy-controlled player, it consumes the current `PolicyGameView`, the
legal-action mask, and its prior hidden state. It returns raw action logits and
a new hidden state. The engine selects or supplies the resulting move. The
policy does not receive a move log because the current coffin and card-status
tensors encode the current game state.

The policy never receives the human hand, stock order, authoritative hidden-card
locations, game seed, scoring implementation, or persistence access. The game
service generates the view and legal-action mask, validates the returned action,
and remains the sole authority for applying it.

The legal-action mask is an explicit policy input, not only a post-processing
guard. The policy is expected to learn to suppress illegal actions from that
input. The hard output mask remains authoritative so sampling can never select
an illegal action.

## Version 1 observation contract

The observation is binary. It is player-relative: Queen uses the authoritative
coffin orientation and King uses its transpose, so a policy player's scoring
lines are always the three horizontal rows of its view. The action mapping uses
the same transform. The policy therefore receives no Queen/King role field.

The fixed card-index order is Clubs, Diamonds, Hearts, Spades; within each suit
the ranks are Ace, 2 through 10, Jack, Queen, King; the two remaining indexes
are `Vampire-0` and `Vampire-1`. The Vampire indexes distinguish physical cards
only; they have identical game meaning. Every card-indexed tensor uses this
order.

For global grid coordinates `(row, column)`, Queen uses `(row, column)` and
King uses `(column, row)`. In row-major index form, the King transform is
`index -> 3 * (index % 3) + index // 3`. It is self-inverse. The center remains
index `4`.

| Field | Shape | Meaning |
| --- | --- | --- |
| Hand positions | `4 × 54` | One card-identity vector per canonical round hand slot; a played slot becomes all zeroes |
| Coffin positions | `9 × 54` | One card-identity vector per player-relative grid position, including the center |
| Played status | `54` | Cards publicly played in the current or completed rounds |
| In-hand status | `54` | Cards currently in the policy player's hand |
| Hidden status | `54` | Cards whose exact location is unavailable to the policy player |
| Round | `6` | One-hot round index, zero through five |
| Own decision progress | `4` | One-hot own decision index: `1000` first through `0001` fourth |
| Dealer | `1` | Dealer or non-dealer |

Fields concatenate in the table order into `observation: bool[875]`. The legal
mask is a separate `bool[4, 8]` model input. At the framework boundary, Boolean
observation and mask values convert to `float32` only where an operation requires
floating-point inputs; their stored and fixture form remains Boolean.

At the start of a round, the four cards are sorted by their fixed card indexes
and assigned to hand rows zero through three. Rows never compact. An occupied
row contains exactly one `true`; playing that card changes the row to all
`false`. This is a canonical schema convention, not information learned by the
policy.

For every card index `c`, the status invariant is:

```text
played[c] + in_hand[c] + hidden[c] == 1
```

`played` contains completed-round cards and cards currently in the coffin.
`hidden` contains the stock and the other player's hand. The positional tensors
deliberately overlap with status information: status identifies the policy-visible
class, while hand and coffin tensors identify positions. All mappings are
versioned with the model.

## Action tensor

The action space is the fixed Cartesian product of four hand slots and eight
non-center coffin positions:

```text
policy logits: 4 × 8
legal mask:    4 × 8
```

The eight policy-grid positions are `[0, 1, 2, 3, 5, 6, 7, 8]` in the
player-relative grid. `action_index = 8 * hand_slot + policy_position_index`.
The service transforms the selected policy position back to the authoritative
grid before validation and application.

The policy produces `raw_logits: float32[4, 8]`. The raw values remain available
to the training loss. For action selection, illegal logits are replaced with
negative infinity, the result is flattened, and one `float32` softmax is applied
across all 32 positions. An active policy turn must have at least one legal
action; the legal mask must exactly match the deterministic engine's legal moves.

The eighth placement has exactly one legal action. The policy runs on that turn
to advance the dealer's hidden state, and the engine applies the unique action.
A round therefore has eight recurrent state updates: four for each player. The
non-dealer's four updates and the dealer's first three are learned decisions.
The dealer's fourth update has `own_decision_progress = 0001` and is a forced
recurrent transition. It contributes no actor, entropy, critic, or
illegal-action loss.

## Recurrence

The production policy retains one hidden state for Dracula. Self-play maintains
separate hidden states for both players even when they share network weights.
Each state begins as the all-zero model state when its game begins and persists
through all six rounds. It receives a view only when that player is active; the
updated coffin is present when the player receives its next view.

Each player receives four recurrent updates in every round. The non-dealer's
fourth update precedes the dealer's final placement. The dealer's fourth update
precedes its own unique final placement. The next round uses the retained state
from that player's fourth update; no round-result feature is injected into the
policy input.

The hidden-state shape and serialization format are part of the model artifact.
A hidden state is valid only for the exact model version that produced it. It is
updated only when the corresponding move is accepted.

Serving invokes the policy once for each policy-controlled turn, including a
forced recurrent transition. Training replays ordered 24-step player-game
trajectories. Its state-update mask is true at every step; its actor-loss mask
is false at the three forced recurrent transitions.

## Validation

The encoder rejects a view when its shape or Boolean dtype is wrong; card IDs or
hand-slot occupancy are inconsistent; card statuses do not partition the deck;
the player-relative coffin conflicts with authoritative state; round, dealer, or
own-decision fields disagree with the lifecycle; or the legal mask differs from
engine legality. A forced recurrent transition requires exactly one legal action
and a dealer progress value of `0001`.

## Serving artifact

The single manually selected policy is exported as:

```text
model.tar.gz
  manifest.json
  policy_state.pt
```

`policy_state.pt` is the policy `state_dict`. The CPU inference image contains
the corresponding model class and loads weights only. `manifest.json` contains:

- Artifact and policy IDs, policy architecture version, and parameter count.
- Observation, action, card-order, grid-order, and hidden-state schema versions.
- PyTorch version, inference-image digest, source revision, and training-run ID.
- Weight-file SHA-256 digest and the selected model's comparison-report reference.
- Supported request contract and the all-zero initial hidden state definition.

The Model Registry package binds the immutable S3 model archive to the immutable
ECR image digest. The package ARN, package version, artifact ID, archive digest,
and schema versions identify the deployed policy and are recorded on every game.

## Serving inference contract

The container is stateless. Every request supplies the prior hidden state and
every successful response returns its successor:

```python
class PolicyInferenceRequest:
    contract_version: Literal["policy-inference-v1"]
    artifact_id: str
    observation: list[bool]       # exactly 875 values
    legal_mask: list[list[bool]]  # exactly 4 rows of 8 values
    hidden_state: str             # base64 little-endian float32[128]


class PolicyInferenceResponse:
    contract_version: Literal["policy-inference-v1"]
    artifact_id: str
    raw_logits: list[list[float]] # exactly 4 rows of 8 values
    next_hidden_state: str        # base64 little-endian float32[128]
```

The application uses `application/json`. It stores hidden state in the same
512-byte representation used by the request and response. At game creation it
uses 512 zero bytes. The hidden bytes are valid only with the package and hidden
schema versions stored in the game's policy session.

The container verifies the contract and artifact IDs, exact tensor shapes,
Boolean input values, at least one legal action, and finite hidden-state values.
It adds a batch dimension, converts model inputs to CPU `float32`, and runs the
policy in evaluation and inference modes. It returns unmasked logits and the
next hidden state after checking both for finite values.

The game service compares the legal mask with engine legality before invocation.
After invocation it verifies response versions, shapes, encoding, and finite
values; applies the authoritative hard mask; and resolves the action with the
deployment's fixed inference profile. A forced transition still invokes the
model to advance hidden state, while the engine supplies its unique action.

## Policy architecture version 1

The policy uses a structured state encoder, one 128-unit GRU layer, and a
pair-scoring action head. It transforms `PolicyInput` and the prior recurrent
state into `raw_logits: float32[4,8]` and the next recurrent state.

### Structured encoder

The encoder has a shared learned card embedding matrix `E_card[54,32]`. A
one-hot card vector selects its 32-feature representation. The two Vampires
retain their independent card indexes and learn their representations from
training data.

For each hand slot `i`, the encoder creates:

```text
H_i = GELU(Linear_41_to_64(
    card_embedding(hand[i]) || hand_slot_embedding[i] || occupied(hand[i])
))
```

`hand_slot_embedding` has shape `4 × 8`. An empty hand row supplies a zero card
embedding and an occupied value of zero.

For each player-relative coffin cell `j`, it creates:

```text
G_j = GELU(Linear_41_to_64(
    card_embedding(coffin[j]) || grid_position_embedding[j] || occupied(coffin[j])
))
```

`grid_position_embedding` has shape `9 × 8`. The hand and grid encoders use
separate linear layers. Their shared card embedding preserves card identity
across both surfaces.

The three 54-bit status vectors remain explicit card-indexed inputs:

```text
S = GELU(Linear_162_to_96(played || in_hand || hidden))
C = GELU(Linear_11_to_16(round || own_decision_progress || dealer))
```

The fused state is:

```text
F = H_0 || H_1 || H_2 || H_3 || G_0 || ... || G_8 || S || C   # 944 features
Z = GELU(Linear_256_to_128(LayerNorm_256(GELU(Linear_944_to_256(F)))))
```

`S` retains a distinct learned weight for every card and status coordinate. The
encoder therefore carries card identity through status information as well as
through hand and coffin positions.

### Recurrent core

The GRU input concatenates the fused state and the legal mask:

```text
U_t = Z_t || flatten(legal_mask_t)    # 160 features
R_t = GRUCell_160_to_128(U_t, R_(t-1))
```

`R_0` is the all-zero 128-feature hidden state defined by the recurrence
contract. The GRU advances on every policy-controlled turn. Training unrolls a
complete 24-step player-game trajectory without detaching at a round boundary.

### Pair-scoring action head

For each hand slot `i` and policy-grid position `k`, let
`j = [0, 1, 2, 3, 5, 6, 7, 8][k]`. The shared pair head produces:

```text
P_i,k = H_i || G_j || R_t || legal_mask[i,k]        # 257 features
raw_logit_i,k = Linear_128_to_1(
    LayerNorm_128(GELU(Linear_257_to_128(P_i,k)))
)
```

The 32 raw logits flatten in hand-slot-major order. Action selection replaces
masked logits with negative infinity and applies one softmax across all 32
positions. The raw logits remain available to the thresholded illegal-probability
loss.

### Parameters and initialization

The policy has **443,145 trainable parameters**:

| Component | Parameters |
| --- | ---: |
| Card, hand-slot, and grid-position embeddings | 1,832 |
| Hand and coffin encoders | 5,376 |
| Status and context encoders | 15,840 |
| Fusion layers and normalization | 275,328 |
| GRU cell | 111,360 |
| Pair-scoring head and normalization | 33,409 |

Linear matrices use Xavier-uniform initialization and zero biases. Card and
position embeddings sample from a zero-mean normal distribution with standard
deviation `1 / sqrt(embedding_width)`. GRU input matrices use Xavier-uniform
initialization, recurrent matrices use orthogonal initialization per gate, and
GRU biases initialize to zero. Layer-normalization scales initialize to one and
biases to zero. Dropout is zero.

### Acceptance

The required tests cover exact forward-pass shapes, parameter count,
Queen/King transpose equivalence, canonical-hand permutation equivalence,
action-table and legal-mask agreement, forced-transition routing, game-scoped
hidden-state continuity, finite forward/backward values, and actor-loss masking
for 24-step player-game batches. The local benchmark measures full-iteration
throughput and memory on the target laptop.

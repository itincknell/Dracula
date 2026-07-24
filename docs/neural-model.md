# Search-guided neural model

The learned opponent will compress the approved Teacher v2 search player. One
feed-forward network predicts a policy over the fixed action space and a
player-relative round value. It has no recurrent state and does not own
legality, transitions, determinization, scoring, or final action selection.

## Contract versions

The initial artifact records:

```text
model_schema_version       = dracula-policy-value-v1
observation_schema_version = dracula-observation-v1
action_schema_version      = dracula-action-map-v1
value_schema_version       = dracula-round-value-v1
```

The model receives `observation: bool[875]` and returns:

```text
policy_logits: float32[4, 8]
round_value:   float32 scalar in [-1, 1]
```

Batch input is `bool[B, 875]`; batch output is `float32[B, 4, 8]` and
`float32[B]`. The legal mask is a separate `bool[4, 8]` or `bool[B, 4, 8]`
value applied outside the model.

## Observation

The observation is the player-relative round-local projection already produced
by the engine bridge:

| Range | Shape | Meaning |
| --- | --- | --- |
| `0:216` | `4 × 54` | Acting player's stable hand slots |
| `216:702` | `9 × 54` | Player-relative coffin positions |
| `702:756` | `54` | Played cards |
| `756:810` | `54` | Cards remaining in the acting player's hand |
| `810:864` | `54` | Unseen cards |
| `864:870` | `6` | One-hot round index |
| `870:874` | `4` | One-hot acting-player decision index |
| `874` | `1` | Acting player is dealer |

The three card-status vectors partition all 54 cards. Empty hand and coffin
positions contain all zeros. Queen uses authoritative grid orientation; King
uses the existing transpose so the acting player's scoring lines are always
rows. Player identity is therefore not a feature.

The layout totals `216 + 486 + 162 + 11 = 875` bits. It was revalidated against
`SearchInformationState` across 3,072 active states from 64 complete seeded
games: 2,688 learned decisions and 384 forced placements. Hand slots, coffin
positions, card partitions, lifecycle fields, and legal masks agreed at every
state.

The projection is sufficient for the uniform belief and round-local objective.
Public move history remains available to search for determinization and audit,
but it is not a model feature. Completed rounds and cumulative scores do not
affect the additive round-value target.

## Architecture

One card embedding is shared by separate hand and coffin encoders:

```text
card embedding:          54 × 32
hand-slot embedding:      4 × 8
coffin-position embedding:9 × 8

hand position:    (32 card + 8 slot + 1 occupied) -> Linear(41, 64) -> GELU
coffin position:  (32 card + 8 position + 1 occupied) -> Linear(41, 64) -> GELU
card status:      162 -> Linear(162, 96) -> GELU
context:           11 -> Linear(11, 16) -> GELU

fusion: 4×64 + 9×64 + 96 + 16 = 944
shared body: Linear(944, 256) -> GELU -> LayerNorm(256)
             -> Linear(256, 128) -> GELU
```

For each position, matrix multiplication of its one-hot card row by the shared
card-embedding table produces the card vector; an empty row produces a zero
vector. The slot or grid-position embedding and occupied flag remain present.
The status and context inputs use their flattened table order. The fusion order
is the four hand features, nine coffin features, status feature, then context
feature.

The policy head scores each hand-slot and destination pair with shared weights:

```text
pair = hand_feature[64] + destination_feature[64] + shared_state[128]
pair -> Linear(256, 128) -> GELU -> LayerNorm(128)
     -> Linear(128, 1)
```

Applying the pair head to four hand slots and the eight non-center positions
produces 32 raw logits. It receives no legal-mask bit. External masking replaces
illegal logits with negative infinity before log-softmax or softmax.

For the bounded response-distillation experiment, the same shared body and
policy head rank Teacher v2 strategic action groups. A group score is the
arithmetic mean of its concrete member logits. Training uses weighted pairwise
logistic ranking: each non-tied group pair is weighted by the absolute
difference between its four-completion mean terminal differentials. Exact ties
contribute zero. Inference selects the highest legal group score and then the
lowest canonical representative index. Paired destinations continue to use the
search layer's derived fair coin. No softmax, root visits, or value prediction
participates in this experiment.

The value head is:

```text
shared_state[128] -> Linear(128, 64) -> GELU -> Linear(64, 1) -> tanh
```

There is no dropout, recurrence, attention, or batch normalization. The exact
trainable parameter count is **339,978**:

| Component | Parameters |
| --- | ---: |
| Shared embeddings | 1,832 |
| Local encoders | 21,216 |
| Shared body | 275,328 |
| Policy head | 33,281 |
| Value head | 8,321 |

## Initialization

The model seed is the unsigned big-endian integer represented by the first
eight bytes of:

```text
derive_seed(
    "dracula-model-initialization-v2",
    run_root_seed,
    model_id,
    decimal(initialization_ordinal),
)
```

Linear weights use Xavier uniform initialization with gain `1.0` and zero
biases. Embeddings use
a zero-mean normal distribution with standard deviation
`1 / sqrt(embedding_width)`. Layer-normalization weights initialize to one and
biases to zero. Initialization uses an isolated PyTorch generator and does not
change global random state.

## Targets and loss

For a non-forced decision with root visit counts `N(a)`:

```text
pi(a) = N(a) / sum_b N(b)
z = (round_score[player] - round_score[other]) / 150

L_policy = -sum_a pi(a) * log_softmax(masked_logits)(a)
L_value  = (v - z)^2
L        = L_policy + L_value
```

Illegal actions have zero target probability. A malformed target with positive
illegal mass is rejected before loss calculation. The exact completed-round
`z` attaches to every non-forced decision made by that player during the round.
Forced placements produce no training example.

Both loss weights are `1.0`. Regularization is AdamW decoupled weight decay of
`1e-4` applied to every trainable parameter; it is not added a second time to
the reported loss.

## Acceptance

Implementation acceptance requires:

- Exact tensor and parameter counts.
- Deterministic initialization and CPU inference.
- Batch and individual forward equivalence.
- Queen/King normalized-input equivalence.
- Invariance under authoritative hidden-card substitutions.
- Finite outputs, losses, and gradients.
- Policy probability zero on illegal actions and unit mass on legal actions.
- Player-relative values that reverse sign with the target perspective.
- Policy and value losses both updating the shared body.
- CPU and MPS agreement within `1e-5` absolute and `1e-4` relative tolerance.
- No engine scoring, private state, determinization, or action selection in the
  model package.

## Artifact

A model artifact contains the state dictionary, exact parameter count, all
contract versions, card and action order, initialization identity, source
revision, training configuration, dataset and search-report digests, optimizer
compatibility version, and state-dictionary digest. Loading rejects missing,
extra, incorrectly shaped, or non-finite parameters.

The artifact has no hidden-state field. Deployment remains undecided until
search-only, guided-search, and standalone-model strength and latency are
measured.

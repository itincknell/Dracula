# Standalone BGC-128 policy model

The selected release model is a standalone policy classifier trained from
BGC-128 lineage normalized root-search visit distributions. It receives one player-visible
card-set observation and returns one raw score for each current-card candidate
and non-center destination pair. It does not predict action values, returns, scores, or any
other search diagnostic.

The model has no value head, critic, recurrence, PPO objective, or hybrid-search
deployment role. It does not own legality, symmetry detection, concrete paired-destination
selection, engine transitions, determinization, or scoring.

## Tensor interface

The single-state interface is:

```text
observation:       bool[659]
raw_policy_logits: float32[4, 8]
```

The batched interface is:

```text
observation:       bool[B, 659]
raw_policy_logits: float32[B, 4, 8]
```

Parameters and outputs are float32. The model receives no legal-mask bit.
Legality and representative-action reduction remain external.

Action index is `candidate_row * 8 + destination_table_index`. Candidate rows
contain the one to four current hand cards in fixed canonical card-ID order;
padding rows have no legal action and no learned row embedding. Destination-table order is global grid
indexes `0, 1, 2, 3, 5, 6, 7, 8`; the occupied center at grid index `4` has no
action.

## Observation

The observation is the established player-relative projection:

| Range | Shape | Meaning |
| --- | --- | --- |
| `0:486` | `9 × 54` | Player-relative coffin positions |
| `486:540` | `54` | Played cards |
| `540:594` | `54` | Cards remaining in the acting player's hand |
| `594:648` | `54` | Unseen cards |
| `648:654` | `6` | One-hot round index |
| `654:658` | `4` | One-hot acting-player decision index |
| `658` | `1` | Acting player is dealer |

The three card-status vectors partition all 54 cards. Empty hand and coffin
positions contain all zeros. Queen uses authoritative grid orientation. King
uses the established transpose so the acting player's scoring lines are rows.
Player identity is not a feature.

The layout totals `486 + 162 + 11 = 659` bits. Fixed preprocessing extracts the
set bits of the in-hand vector in canonical card-ID order and packs them into
four candidate rows. Canonical order is not learned. The projection contains no
authoritative opponent hand, stock order, engine seed, determinization, search
tree, model state, or policy history.

## Architecture

One card embedding is shared by the hand and coffin encoders:

```text
card embedding:             54 × 32
coffin-position embedding:   9 × 8

current card candidate:
    32 card
    -> Linear(32, 64)
    -> GELU

hand set:
    sum current-card features
    -> Linear(64, 256)
    -> GELU

coffin position:
    32 card + 8 position + 1 occupied
    -> Linear(41, 64)
    -> GELU

card status:
    162
    -> Linear(162, 96)
    -> GELU

context:
    11
    -> Linear(11, 16)
    -> GELU
```

The in-hand membership vector selects one to four exact card embeddings. The
shared candidate encoder runs on each card and the resulting features are
summed before the hand-set encoder. This summary is permutation-invariant.
Candidate rows only serialize the current cards for scoring; they have no
embedding or learned identity.

The fusion order is the 256-wide hand-set feature, nine coffin features, status
feature, then context feature:

```text
fusion width = 256 + 9×64 + 96 + 16 = 944

shared body:
    Linear(944, 512)
    -> GELU
    -> LayerNorm(512)
    -> Linear(512, 256)
    -> GELU
```

The shared pair scorer receives the exact candidate-card feature, encoded destination, and
shared state:

```text
pair input = hand feature 64 + destination feature 64 + shared state 256
           = 384

pair head:
    Linear(384, 256)
    -> GELU
    -> LayerNorm(256)
    -> Linear(256, 1)
```

The pair head runs for each of four current-card candidate rows and the eight non-center coffin
positions, producing 32 raw logits. There is no dropout, attention, recurrence,
batch normalization, value head, or auxiliary output.

Every GELU is `GELU(approximate="none")`. Both LayerNorm modules use learnable
affine parameters and epsilon `1e-5`.

The exact trainable parameter count is **754,601**:

| Component | Parameters |
| --- | ---: |
| Shared embeddings | 1,800 |
| Card, hand-set, coffin, status, and context encoders | 37,280 |
| Shared body | 616,192 |
| Pair head | 99,329 |
| **Total** | **754,601** |

## Representative-action legality

The dataset row retains the full engine-legal `bool[4,8]` mask and the exact
strategic-group partition. The training projection reconstructs the
authoritative destination groups from the observation and full mask, verifies
that they equal the sealed row groups, and derives a separate representative
mask.

For every legal hand card:

- A listed symmetry pattern enables only the table's designated proxy
  destination for each strategic group.
- The non-proxy member of a paired group is false in the representative mask.
- An unpaired group enables its sole destination.
- An unlisted occupied-position pattern enables every engine-legal destination
  independently.

The reduction applies independently to each hand card. Different hand cards
remain different actions. The center position has no action logit. The complete
authoritative table is in
[symmetry and move selection](search.md#authoritative-early-turn-symmetry).

Each strategic group's exact root visit count is assigned to its designated
proxy action. No mirrored-destination averaging or pooling occurs. Counts sum
to the configured 128 outer simulations. Before a row enters a batch,
validation requires:

1. The stored groups exactly partition the full engine-legal actions.
2. The reconstructed representatives exactly match the authoritative table.
3. The representative mask contains one true action per strategic group.
4. Every positive visit count belongs to a true representative action.
5. Every non-proxy paired member is false.

## Objective

Let `m` be the external representative mask, `N(a)` the exact group visit
count, and `pi(a) = N(a) / 128`:

```text
masked_logits[a] = raw_policy_logits[a] when m[a] is true
masked_logits[a] = -infinity           when m[a] is false

log_p = log_softmax(masked_logits)
L_policy = -sum(pi[a] * log_p[a])
```

The target is the normalized root-search distribution over legal
representative actions. There is no selected-action-only loss, illegal-action
penalty, legality auxiliary task, temperature transformation, label smoothing,
action-value target, score target, return target, value loss, PPO term, or
other auxiliary loss. AdamW weight decay is the only regularization and is not
added to the reported cross-entropy.

Forced eighth placements have no row and no loss.

## Standalone inference

Standalone inference performs these steps:

1. Build the established player-visible observation and full engine-legal mask.
2. Derive and validate the representative mask from the authoritative symmetry
   table.
3. Run the model once.
4. Apply the representative mask outside the model.
5. Select the maximum legal representative-action logit. An exact tie resolves
   to the lowest canonical action index.
6. If the selected group is paired, seed a local fair coin from the game and
   decision identity and choose one of its two concrete destinations.
7. Validate and apply the concrete action through the engine.

The fair-coin result is not a model target. The selected proxy remains the
strategic choice regardless of which paired destination the coin resolves.
Standalone argmax inference is the selected production controller. The visit
distribution supplied training information; deployment selects the highest
masked representative logit.

## Initialization

Training supplies one integer initialization seed. Initialization uses an
isolated PyTorch generator and does not change global random state:

- Linear weights use Xavier uniform initialization with gain `1.0`.
- Linear biases are zero.
- Embeddings use a zero-mean normal distribution with standard deviation
  `1 / sqrt(embedding_width)`.
- LayerNorm weights are one and biases are zero.

## Mechanical acceptance

Implementation acceptance covers:

- Exact input, output, batch, and parameter counts.
- Float32 parameters and outputs.
- Deterministic initialization from one recorded integer seed.
- Batch and individual forward equivalence.
- Queen/King normalized-input equivalence.
- Hidden-card substitution invariance.
- Exact representative masks for every authoritative pattern and every legal
  hand card.
- All engine-legal actions retained independently for unlisted patterns.
- Finite logits, cross-entropy, and gradients.
- Shared body and pair-head parameter updates.
- Zero training probability on masked actions.
- Deterministic proxy argmax and paired-destination fair-coin resolution.
- CPU operation and MPS agreement within `1e-5` absolute and `1e-4` relative
  tolerance.
- No engine scoring, search, private state, or action application in the model
  package.

These checks establish model and artifact validity. The completed `pi1`
training and user selection are summarized in
[model training](model-training.md#selected-pi1-run).

## Artifact

The deployment artifact contains one format marker and the model state
dictionary. Its complete file SHA-256 identifies the selected release bytes.
Loading rejects missing, extra, incorrectly shaped, incorrectly typed, or
non-finite parameters. Training provenance remains in the training report and
is not duplicated inside the runtime artifact.

The artifact contains no value parameters, critic state, hidden state, search
state, or fair-coin outcome. Previous policy/value, response-ranker, and hybrid
artifacts remain historical evidence and are not accepted by this loader.

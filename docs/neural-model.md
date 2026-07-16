# Neural model

This document defines the learned opponent policy. Application serving and
persistence belong in [architecture](architecture.md); training systems and
promotion belong in [model training](model-training.md).

## Policy boundary

The policy is a recurrent neural network. On each opponent turn it consumes the
current `PolicyGameView`, the legal-action mask, and its prior hidden state. It
returns one action and a new hidden state. It does not receive a move log because
the current coffin and card-status tensors encode the current game state.

The policy never receives the human hand, stock order, authoritative hidden-card
locations, game seed, scoring implementation, or persistence access. The game
service generates the view and legal-action mask, validates the returned action,
and remains the sole authority for applying it.

The legal-action mask is an explicit policy input, not only a post-processing
guard. The policy is expected to learn to suppress illegal actions from that
input. The hard output mask remains authoritative so sampling can never select
an illegal action.

## Observation tensor

The observation is binary and uses stable card IDs and position mappings:

| Field | Shape | Meaning |
| --- | --- | --- |
| Hand positions | `4 × 54` | One card-identity vector per stable round hand slot; a played slot becomes all zeroes |
| Coffin positions | `9 × 54` | One card-identity vector per grid position, including the center |
| Played status | `54` | Cards publicly played in the current or completed rounds |
| In-hand status | `54` | Cards currently in the policy player's hand |
| Hidden status | `54` | Cards whose exact location is unavailable to the policy player |
| Round | `6` | One-hot round number |
| Turn | `9` | One-hot count of cards placed after the center card, from zero through eight |
| Role | `1` | Queen or King |
| Dealer | `1` | Dealer or non-dealer |

For every card index `c`, the status invariant is:

```text
played[c] + in_hand[c] + hidden[c] == 1
```

The positional tensors deliberately overlap with status information: status
identifies the policy-visible ownership class, while the hand and coffin tensors
identify actionable positions. All mappings are versioned with the model.

## Action tensor

The action space is the fixed Cartesian product of four hand slots and eight
non-center coffin positions:

```text
policy logits: 4 × 8
legal mask:    4 × 8
```

Illegal logits are replaced with negative infinity. The result is flattened and
one softmax is applied across all 32 positions. The eight-position mapping is
stable; it excludes center grid index `4`. An active policy turn must have at
least one legal action.

The eighth placement of a round has exactly one legal action. The engine applies
that action without invoking the policy. A round therefore contains seven
policy decisions: four by the non-dealer and three by the dealer. The forced
placement remains part of the authoritative trajectory and terminal round
calculation, but not the policy or entropy losses.

## Recurrence

The production policy retains one hidden state for Dracula. Self-play maintains
separate hidden states for both players even when they share network weights.
Each player's recurrence begins from its own initial round view and receives only
that player's subsequent views.

When the dealer's fourth placement is forced, its recurrent state is not
advanced. No later decision depends on that state because recurrence resets at
the round boundary.

The hidden-state shape and serialization format are part of the model artifact.
A hidden state is valid only for the exact model version that produced it. It is
updated only when the corresponding move is accepted.

## Artifact contract

A promotable artifact contains:

- Network weights and executable inference graph.
- Observation, action, card-order, grid-order, and hidden-state schema versions.
- Framework and export versions.
- Training-run and source revision identifiers.
- Inference defaults and supported runtime capabilities.
- Evaluation summary and content hash.

> **TODO MODEL-001 — Finalize the tensor and recurrence contract.** Confirm exact
> dtypes, dimensions, mappings, zero-slot behavior, initialization, padding,
> hidden-state boundaries, legal-mask injection, forced-placement handling, and
> validation errors.
>
> **Complete when:** Frozen fixtures encode and decode representative initial,
> middle, final, Vampire, and sixth-round views, legal masks, and the forced
> placement without private-state leakage or mapping ambiguity.

> **TODO MODEL-002 — Select the recurrent policy architecture.** Compare the
> candidate recurrent cell, shared card projection, hidden width, layer count,
> mask-conditioned policy head, initialization, and parameter count. The critic
> is a separate training model specified in `TRAIN-002`.
>
> **Complete when:** The selected architecture has a reproducible specification,
> parameter-count report, initialization contract, and exact forward-pass tensor
> flow. Runtime benchmarking belongs to `TRAIN-001`.

> **TODO MODEL-003 — Finalize inference and artifact behavior.** Define action
> sampling, temperature, inference seed, hidden-state serialization, export
> format, numerical tolerances, and runtime compatibility.
>
> **Complete when:** Local and serving-container inference agree on frozen inputs
> within documented tolerances and produce compatible hidden states.

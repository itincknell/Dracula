# Model training

## Completed lineage

Model training is complete for the release controller:

```text
balanced BGC-128 visit corpus D0
    -> compact card-set migration
    -> standalone policy pi0
    -> pi0 continuation corpus D1
    -> standalone policy pi1
    -> user selection for production
```

The deployment build does not train, fine-tune, search, or promote models.
Training code and sealed local runs remain reproducibility evidence.

## Selected pi1 run

`pi1` was trained from the immutable D1 compact-card snapshot:

| Item | Value |
| --- | --- |
| Training rows | 806,610 |
| Validation rows | 89,670 |
| Parameters | 754,601 |
| Objective | Representative-masked distributional cross-entropy |
| Best epoch | 12 |
| Best validation cross-entropy | 1.99895417 |
| Validation top-1 / top-2 / top-3 | 0.4879 / 0.7139 / 0.8179 |
| Artifact SHA-256 | `70c76f2eb64600eab2297640278a6c94d4336ab8e73bf941a6d96237f69f5b5c` |

The target was the normalized 128-simulation strategic-group visit
distribution mapped to representative actions. Illegal and non-proxy logits
were masked before log-softmax. There was no illegal-action penalty, one-hot
selected-action loss, value output, return target, PPO term, recurrence, or
critic.

The exact architecture and serving semantics are in
[neural model](neural-model.md).

## Evaluation evidence

Local fixed evidence includes:

- `pi1` versus standalone `pi0`: 18 wins in 24 games and mean score
  differential `+19.75`.
- `pi1` versus BGC-128 with `pi0` continuations: 13 wins in 24 games and mean
  score differential `-1.125`.
- `pi1` versus BGC-128 with `pi1` continuations: 9 wins in 24 games and mean
  score differential `-19.25`.
- `pi1` self-play over 54 games: Queen 28 wins, King 26 wins, with Queen mean
  score differential `+1.96` and an interval spanning zero.
- No illegal action in the fixed matchup evidence.

These measurements are retained as evidence, not an automatic deployment gate.
The user subsequently selected standalone `pi1` after manual play.

## Reproduction paths

The ignored local evidence is rooted at:

```text
runs/bgc-policy-d1-card-set-001/
runs/bgc-policy-pi1-001/
runs/pi1-evaluation-001/
```

The selected artifact path is:

```text
runs/bgc-policy-pi1-001/artifacts/unaccepted-candidate.pt
```

The historical filename records its status before manual selection. Release
packaging copies the exact bytes into a production artifact location and
verifies the digest; it does not retrain or rewrite the state dictionary.

## Historical training

Earlier PPO, policy/value, Teacher v2, response-ranking, expert-iteration,
Sam-selected-action, and `pi0` acceptance protocols remain indexed in
[reports](../reports/README.md). They are inactive.

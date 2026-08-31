# Teacher v2 response ranker

## Result

The bounded response-ranking experiment completed, but the selected model did
not learn a useful general response policy from four training games. It reduced
validation ranking loss slightly relative to its deterministic untrained
initialization, while pairwise ordering remained effectively random and
top-group accuracy did not improve. This artifact is valid experimental
evidence; it is not a gameplay candidate.

## Configuration

| Item | Value |
| --- | --- |
| Training games | 4 |
| Validation games | 2 |
| Training examples | 12,930 |
| Validation examples | 6,463 |
| Dataset digest | `110dd3658af4ab33ccd6a9eefe44f530b2f8fb1f17321e917448cdaacb17ffec` |
| Model parameters | 339,978 |
| Optimized parameters | 331,657 |
| Frozen parameters | 8,321-value-head parameters |
| Optimizer | AdamW, learning rate `0.0003`, weight decay `0.0001` |
| Batch size | 256 |
| Gradient clipping | Global norm `1.0` |
| Device | CPU |

The model uses the existing structured encoder, shared feed-forward body, and
32-logit policy head. Strategic-group scores are arithmetic means of concrete
action logits. Training minimizes the absolute-value-gap-weighted pairwise
logistic loss. Root visits and the value head do not participate.

Checkpoint selection used only validation ranking loss. The run completed
eight epochs and stopped at the configured minimum after validation had failed
to improve for five epochs. Epoch 0 was selected:

| Measurement | Value |
| --- | ---: |
| Best validation loss | 0.646767 |
| Untrained validation loss | 0.650864 |
| Epoch 7 training loss | 0.610833 |
| Epoch 7 validation loss | 0.666340 |
| Training time | 7.44 seconds |
| Peak process RSS | 419.2 MiB |

The widening train-validation gap is consistent with overfitting this small
four-game training split.

## Validation metrics

Pairwise rank accuracy measures non-tied Teacher v2 group pairs. Rank
correlation is the mean per-example Spearman correlation where the Teacher
values are not constant. Regret is the Teacher v2 value of its best group minus
the value of the model-selected group.

| Metric | Untrained | Selected |
| --- | ---: | ---: |
| Top-group accuracy | 23.69% | 23.24% |
| Top-two recall | 46.28% | 46.56% |
| Top-three recall | 54.91% | 55.79% |
| Mean Teacher-value regret | 0.09956 | 0.09610 |
| Pairwise rank accuracy | 49.94% | 50.14% |
| Mean rank correlation | -0.0098 | 0.0108 |

The small regret and loss improvements are real but insufficient. Pairwise
accuracy remains approximately 50%, and the model does not recover the
Teacher's selected group more often than its untrained control.

### Placement

Placement 1 has no examples because the shallow-response evaluator is entered
only after the outer search has selected its root action.

| Placement | Examples | Top group | Top two | Top three | Regret | Pairwise | Rank correlation |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0 | — | — | — | — | — | — |
| 2 | 352 | 7.39% | 15.34% | 22.44% | 0.16129 | 51.18% | 0.0333 |
| 3 | 449 | 8.02% | 15.81% | 23.39% | 0.15641 | 49.80% | 0.0007 |
| 4 | 876 | 6.28% | 14.73% | 24.09% | 0.15120 | 49.69% | -0.0025 |
| 5 | 1,260 | 10.87% | 23.73% | 38.02% | 0.12018 | 50.46% | 0.0143 |
| 6 | 1,625 | 17.91% | 34.15% | 51.14% | 0.08821 | 50.56% | 0.0159 |
| 7 | 1,901 | 50.34% | 100.00% | 100.00% | 0.03516 | 50.40% | 0.0080 |

Placement 7 contains two groups, so top-two and top-three recall are
automatically 100% and do not demonstrate ranking quality.

### Role and dealer status

| Split | Examples | Top group | Top two | Top three | Regret | Pairwise | Rank correlation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Queen | 3,235 | 22.10% | 46.55% | 55.36% | 0.09772 | 50.14% | 0.0088 |
| King | 3,228 | 24.38% | 46.56% | 56.23% | 0.09446 | 50.13% | 0.0128 |
| Dealer | 2,853 | 13.04% | 25.87% | 39.29% | 0.11657 | 50.14% | 0.0123 |
| Non-dealer | 3,610 | 31.30% | 62.91% | 68.84% | 0.07992 | 50.13% | 0.0095 |

The dealer/non-dealer difference primarily reflects different group counts and
game states; neither split shows learned pairwise ranking above noise.

## Inference and artifacts

CPU latency includes the shared model forward pass and, for the single-state
measurement, strategic-group aggregation:

| Measurement | Value |
| --- | ---: |
| Single-state median | 0.227 ms |
| Single-state p95 | 0.330 ms |
| Batch-256 median | 3.037 ms |
| Batch-256 per example | 0.0119 ms |

The distinct response-ranker artifact is
`runs/teacher-v2-response-ranker-001/artifacts/response-ranker.pt`.
It is 1,373,598 bytes and has SHA-256
`f6504bfb246e56e09d0e98d6a9f668d9744113045678450c84c5e48fc8cc9f75`.
Its state-dict digest is
`717f04df89f654933d2785f7b60d5a915a7a7399b5a4b7cf6d413e762465501c`.

The selected checkpoint is
`runs/teacher-v2-response-ranker-001/checkpoints/final.pt`, with SHA-256
`e6e3d399f6f2f272d9e669570be439e6e22286f01ba0c267d42c3d52949add5b`.
Detailed metrics are in
`runs/teacher-v2-response-ranker-001/metrics/final-evaluation.json`.

## Verification

- The ranker reads the sealed four-game training and two-game validation
  manifests without resplitting them.
- Dense minibatch loss equals the locked scalar pairwise-loss contract.
- Losses, outputs, gradients, optimizer state, and exported weights are finite.
- The value head is excluded from the optimizer and receives no gradients.
- An interrupted full-corpus CPU run replayed its partial epoch and produced
  byte-identical selected weights, best epoch, and best validation loss.
- The exported artifact reproduces checkpoint logits exactly.
- Artifact loading binds the model, observation, action, response-example,
  action-group, ranking, dataset, split-manifest, initialization, and
  configuration identities.
- Dataset and artifact validation reject unknown or private authoritative
  fields. No determinization, opponent hand, stock order, engine seed, search
  tree, recurrence, PPO state, separate critic, or population state enters the
  ranker.
- The policy/value, response observer, response dataset, response ranker, and
  existing supervised-training suites pass: 48 tests.

## Assessment

The experiment demonstrates that neural response inference would be cheap, but
this checkpoint does not preserve Teacher v2 response quality. Offline accuracy
does not support replacing the four-completion shallow response evaluator, and
no gameplay-competence claim follows from these results.

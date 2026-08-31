# Teacher v2 response ranker 002

## Result

The first substantive response-ranker run completed from the 48-game training
split and selected epoch 0 by held-out validation loss. A controlled
interruption resumed exactly, and the exported artifact reproduces the selected
checkpoint's inference.

The selected model produced small improvements over the new untrained
initialization and prior smoke ranker on validation ranking loss, pairwise
accuracy, rank correlation, and Teacher-value regret. Top-group and top-three
results did not improve consistently. These results are evidence for manual
review, not an acceptance or rejection decision.

## Configuration

| Item | Value |
| --- | --- |
| Training examples | 154,839 from 48 games |
| Validation examples | 38,652 from 12 disjoint games |
| Dataset digest | `bf902e36916397e3a589fe081c9dc102f05998e3c783bc98a6dffa0f412d3148` |
| Model parameters | 339,978 |
| Optimized parameters | 331,657 |
| Frozen parameters | 8,321 value-head parameters |
| Optimizer | AdamW, learning rate `0.0003`, weight decay `0.0001` |
| Batch size | 256 |
| Gradient clipping | Global norm `1.0` |
| Device | CPU |

The shared structured encoder, feed-forward body, and 32-logit policy head are
unchanged. Strategic-group logits are arithmetic means over concrete actions.
The objective is absolute-value-gap-weighted pairwise logistic ranking. Root
visits, round-value loss, PPO, recurrence, a separate critic, and population
training are absent.

The resolved configuration binds:

```text
training manifest    025f15d989dfe30cca4ed3bd4fadbe7b3fa6ae3c7fb680002957e1d0ea3d2045
validation manifest  5a2aea5e1999b757b021e8ea57eff9d62b62519a5b1cb8422f2b29e4f6e08e69
resolved config      18371c01aca5bc80a3bba9988baf97e1de2d9ba37f302a83943b22a572c68520
```

## Training

Epoch 0 produced the lowest validation loss. The required eight-epoch minimum
completed, then early stopping applied after five validation epochs without
the configured minimum improvement.

| Epoch | Training loss | Validation loss | Top group | Regret | Max gradient norm | Seconds |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.645668 | **0.646408** | 23.54% | 0.09696 | 0.0967 | 9.72 |
| 1 | 0.645500 | 0.646622 | 23.45% | 0.09703 | 0.0248 | 9.60 |
| 2 | 0.645196 | 0.646886 | 24.12% | 0.09755 | 0.0455 | 9.64 |
| 3 | 0.644076 | 0.646754 | 24.76% | 0.09637 | 0.0858 | 9.55 |
| 4 | 0.641452 | 0.649559 | 24.51% | 0.09751 | 0.1290 | 9.70 |
| 5 | 0.636475 | 0.651489 | 24.39% | 0.09658 | 0.1946 | 9.60 |
| 6 | 0.628867 | 0.656122 | 24.17% | 0.09652 | 0.3039 | 9.60 |
| 7 | 0.617743 | 0.667060 | 24.58% | 0.09627 | 0.4251 | 9.81 |

Completed optimization took 77.4 seconds. Peak process RSS was 751.6 MiB.
The divergence between later training and validation loss is why checkpoint
selection retained epoch 0.

## Held-out comparison

All three models were evaluated over the same 38,652 validation rows from the
new dataset.

| Metric | Untrained initialization | Prior smoke ranker | Selected checkpoint |
| --- | ---: | ---: | ---: |
| Ranking loss | 0.649409 | 0.646662 | **0.646408** |
| Top-group accuracy | **23.91%** | 23.52% | 23.54% |
| Top-two recall | 47.52% | 47.43% | **47.60%** |
| Top-three recall | 56.77% | **56.83%** | 56.81% |
| Pairwise rank accuracy | 49.95% | 50.13% | **50.20%** |
| Rank correlation | -0.0028 | 0.0041 | **0.0094** |
| Mean Teacher-value regret | 0.09800 | 0.09838 | **0.09696** |
| Teacher group missing from top two | 20,285 | 20,318 | **20,252** |

The selected checkpoint's improvement is clearest in loss and regret. Its
top-group accuracy is below the untrained initialization, while pairwise
accuracy remains close to 50%.

## Placement results

Placement 1 is outside the shallow-response observer boundary. Placement 7 has
only two strategic groups, so top-two and top-three recall are necessarily
100%.

Selected checkpoint:

| Placement | Examples | Top group | Top two | Top three | Pairwise | Rank correlation | Regret | Missing top two |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 2,092 | 7.84% | 17.30% | 24.62% | 50.17% | 0.0040 | 0.16776 | 1,730 |
| 3 | 2,694 | 8.83% | 17.67% | 27.06% | 50.06% | 0.0030 | 0.15606 | 2,218 |
| 4 | 5,230 | 7.65% | 15.05% | 22.64% | 50.22% | 0.0075 | 0.15373 | 4,443 |
| 5 | 7,491 | 13.59% | 26.81% | 40.61% | 50.35% | 0.0121 | 0.11677 | 5,483 |
| 6 | 9,768 | 17.04% | 34.71% | 52.33% | 50.12% | 0.0040 | 0.09172 | 6,378 |
| 7 | 11,377 | 49.36% | 100.00% | 100.00% | 50.87% | 0.0173 | 0.03532 | 0 |

Top-group comparison:

| Placement | Untrained | Prior smoke | Selected |
| ---: | ---: | ---: | ---: |
| 2 | 7.70% | **9.56%** | 7.84% |
| 3 | 8.13% | 7.68% | **8.83%** |
| 4 | 7.55% | 7.42% | **7.65%** |
| 5 | 13.08% | 13.24% | **13.59%** |
| 6 | **17.19%** | 16.71% | 17.04% |
| 7 | **51.05%** | 49.85% | 49.36% |

Mean Teacher-value regret:

| Placement | Untrained | Prior smoke | Selected |
| ---: | ---: | ---: | ---: |
| 2 | 0.16853 | **0.16697** | 0.16776 |
| 3 | 0.15923 | 0.16197 | **0.15606** |
| 4 | **0.15332** | 0.15562 | 0.15373 |
| 5 | 0.11959 | 0.12045 | **0.11677** |
| 6 | 0.09235 | **0.09140** | 0.09172 |
| 7 | 0.03574 | 0.03584 | **0.03532** |

## Role and dealer status

Selected checkpoint:

| Split | Examples | Top group | Top two | Top three | Pairwise | Rank correlation | Regret | Missing top two |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Queen | 19,322 | 23.70% | 47.73% | 57.03% | 50.24% | 0.0097 | 0.09679 | 10,099 |
| King | 19,330 | 23.38% | 47.48% | 56.60% | 50.17% | 0.0091 | 0.09714 | 10,153 |
| Dealer | 17,090 | 13.04% | 26.56% | 39.85% | 50.19% | 0.0051 | 0.12001 | 12,551 |
| Non-dealer | 21,562 | 31.87% | 64.28% | 70.25% | 50.22% | 0.0133 | 0.07870 | 7,701 |

The dealer/non-dealer difference tracks the different strategic-group counts
and state distribution. Detailed control results for every placement, role,
and dealer split are retained in `metrics/control-comparison.json`.

## Recovery and verification

The requested run was interrupted during epoch 0 before minibatch 8. Its state
recorded `completed_epoch = -1`; only the immutable iteration-start checkpoint
and untrained control existed, with no partial epoch metrics or temporary file.

Resume replayed epoch 0 from the deterministic shuffle. A clean same-seed run
under `.local` produced:

```text
selected state digest     aade4221c12460742609fe4c6adc2470ec88c3198afe54d473f768adf6b67140
best epoch                0
best validation loss      0.6464078884236772
optimizer state           tensor-exact
non-timing epoch history  exact
frozen value head         tensor-exact
```

All model outputs, losses, gradients, optimizer tensors, checkpoints, and
reported metrics are finite. The largest pre-clipping gradient norm was
`0.4251`, below the configured limit. The exported artifact reproduces the
selected checkpoint's logits exactly over a 256-row validation batch.

## Inference and artifacts

| Measurement | Result |
| --- | ---: |
| Single-state median | 0.239 ms |
| Single-state p95 | 0.564 ms |
| Batch-256 median | 3.416 ms |
| Batch-256 per example | 0.0133 ms |

The artifact is
`runs/teacher-v2-response-ranker-002/artifacts/response-ranker.pt`.
It is 1,373,598 bytes and has SHA-256
`8ac77576c23061e35bbf0f1c4d9141ece86cfb3a8d50f603c16ce647d97d5585`.
Its state-dictionary digest is
`aade4221c12460742609fe4c6adc2470ec88c3198afe54d473f768adf6b67140`.

The selected final checkpoint has SHA-256
`eec2207eaca07a4fcbb117c27ae1340b3cc05270b0e9e5fbad600c6c29b6a7df`.
The complete run occupies 14 MiB. Detailed control evaluation is in
`runs/teacher-v2-response-ranker-002/metrics/control-comparison.json`.

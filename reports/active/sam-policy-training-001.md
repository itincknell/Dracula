# Standalone Sam policy training 001

## Outcome

The first complete standalone imitation experiment finished normally. It
trained the 738,569-parameter `SamPolicyModel` on the fixed
`sam-policy-snapshot-001` corpus snapshot and stopped after epoch 8 under the
configured minimum-epoch and patience rules.

Epoch 3 has the lowest held-out cross-entropy and is the selected checkpoint:

- Selected validation cross-entropy: **0.916986**
- Selected validation top-1 agreement with Sam: **58.47%**
- Selected validation top-2 agreement with Sam: **88.21%**
- Selected validation top-3 agreement with Sam: **91.74%**
- Untrained validation cross-entropy: **0.985684**
- Untrained validation top-1 / top-2 / top-3: **43.30% / 83.28% / 87.17%**

These are imitation measurements, not gameplay-strength measurements. No
gameplay integration or candidate acceptance occurred.

## Frozen inputs and configuration

- Snapshot: [`sam-policy-snapshot-001`](../../runs/sam-policy-snapshot-001/corpus-snapshot.json)
- Snapshot digest:
  `7939d0f67b91ae74d3ec6b1f587b2baa4602c3563d3834882250ee079d6edf4e`
- Dataset digest:
  `98e01da7319243de4e4bf1b078f3d1bd10e585be3b4d5ece777e214e34987195`
- Overlay split digest:
  `7e6f0a1e6e99c9f1ad275e2519f55f89c90f1436c94db841523fd1812cd03786`
- Training: 14 complete decks, 458,724 rows
- Validation: 2 complete decks, 65,532 rows
- Model contract digest:
  `258498fa041277aa193232be315a92d53b12e4c2b9c8dfea275a52a60ba45a52`
- Optimizer configuration digest:
  `10545ca9139f8fd4f3552e2d41a54ae12345a34c56fe660a4d5278a90ad5e44d`
- Resolved run digest:
  `248d6f1d82bec31f23d48e93eccb82a9ce116eff1aec8935cbc1f644f84b5c06`

The resolved run binds the snapshot digest. The snapshot binds the frozen
corpus manifest and deck manifests, which in turn bind:

- Sam-32 teacher configuration:
  `790c5c664ecd60377b82d2da0ba62d4c343a9d017915d933d2c0a4de587bf329`
- Action schema:
  `60bf0f1b09b6f5ecfc14fec3369d5c6da0e712e424c3e6b268552a2c509ac178`
- Symmetry schema:
  `d5985b2775fcd1ee2191ffa6d2a755532df898902303530728269ead1074a661`
- Collection configuration:
  `d681a05fd1e9d53d07ee5e158166d47ccf55e1dbfb472910718712ee626caada`

Training used CPU, batch size 256, AdamW at `3e-4`, betas
`(0.9, 0.999)`, epsilon `1e-8`, weight decay `1e-4`, and global gradient
clipping at 1.0. There was no label smoothing, placement weighting, value
head, value loss, search-visit target, recurrence, PPO, critic, or hybrid
objective.

Training source revision was
`ec6e3d395bc89e9b0cb4bf31eb15438119281c60` with tree digest
`0de9f525f512f3f77eb2e64ab2ee84097c2417dbf8d182a8cb5a77d232fab87c`.
The frozen corpus source tree digest was
`c5494eb05db31d1a0a8c9765623de8b7e3742a41ef31fef574b7437ec4d89222`.

## Training history

| Epoch | Train CE | Validation CE | Val top-1 | Val top-2 | Val top-3 | Max pre-clip gradient | Seconds |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.923212 | 0.984077 | 46.38% | 85.45% | 89.74% | 0.395 | 477.8 |
| 2 | 0.795609 | 0.935223 | 55.57% | 87.24% | 90.88% | 0.851 | 519.1 |
| **3** | **0.706484** | **0.916986** | **58.47%** | **88.21%** | **91.74%** | 1.222 | 523.4 |
| 4 | 0.643332 | 0.923105 | 59.99% | 88.86% | 92.17% | 1.366 | 427.0 |
| 5 | 0.591159 | 0.946475 | 60.60% | 89.37% | 92.62% | 1.537 | 377.6 |
| 6 | 0.544927 | 0.967626 | 60.43% | 89.33% | 92.63% | 1.648 | 377.7 |
| 7 | 0.499886 | 1.004471 | 60.50% | 89.38% | 92.69% | 1.940 | 373.7 |
| 8 | 0.470151 | 1.038700 | 60.61% | 89.15% | 92.47% | 2.063 | 533.1 |

The minimum eight epochs completed. Epochs 4–8 failed to improve validation
cross-entropy by `1e-4`, satisfying patience 5 at epoch 8. Cross-entropy
selection retained epoch 3 even though later top-1 agreement was higher.

## Selected-checkpoint validation

The comparison uses the same 65,532 held-out rows for the untrained
initialization and selected epoch-3 checkpoint.

| Placement | Rows | Untrained CE | Selected CE | Untrained top-1 | Selected top-1 | Selected top-2 | Selected top-3 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 12 | 2.0736 | 1.9448 | 25.00% | 33.33% | 58.33% | 75.00% |
| 2 | 48 | 2.4835 | 2.3444 | 6.25% | 10.42% | 33.33% | 50.00% |
| 3 | 192 | 2.4456 | 2.3640 | 10.94% | 16.67% | 28.12% | 39.58% |
| 4 | 768 | 2.6104 | 2.5119 | 8.72% | 17.06% | 28.78% | 39.19% |
| 5 | 3,072 | 2.0295 | 1.9130 | 13.44% | 26.92% | 46.06% | 58.92% |
| 6 | 12,288 | 1.7674 | 1.6061 | 18.78% | 35.49% | 56.50% | 71.18% |
| 7 | 49,152 | 0.6922 | 0.6502 | 52.00% | 67.05% | 100.00% | 100.00% |

| Split | Rows | Untrained CE | Selected CE | Untrained top-1 | Selected top-1 | Selected top-2 | Selected top-3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Queen | 32,766 | 0.9846 | 0.8894 | 44.37% | 60.00% | 88.41% | 91.96% |
| King | 32,766 | 0.9867 | 0.9446 | 42.22% | 56.94% | 88.02% | 91.51% |
| Dealer | 13,104 | 1.8195 | 1.6619 | 18.15% | 34.32% | 54.79% | 69.22% |
| Non-dealer | 52,428 | 0.7773 | 0.7308 | 49.58% | 64.50% | 96.57% | 97.37% |
| **All** | **65,532** | **0.9857** | **0.9170** | **43.30%** | **58.47%** | **88.21%** | **91.74%** |

## Runtime and artifacts

- Snapshot verification and loading: 759.9 seconds
- Eight complete epochs: 3,609.3 seconds total
- Mean epoch: 451.2 seconds; range 373.7–533.1 seconds
- Total run, including loading and post-run verification: 4,407.7 seconds
  (73 minutes 28 seconds)
- Effective complete-epoch throughput: 860–1,228 training examples/second,
  including full training and validation evaluation
- Peak trainer RSS: 561,807,360 bytes (535.8 MiB)
- Machine-wide swap at start/end: 109,177,733 / 293,402,050 bytes
- Machine-wide swap growth during the run: 184,224,317 bytes (175.7 MiB);
  this includes the concurrent collector and other laptop activity
- Complete run artifact directory: 32,723,408 bytes
- Exported policy: 2,966,453 bytes

Standalone CPU forward latency under the concurrent collector:

- Single state: mean 0.499 ms, p50 0.462 ms, p95 0.713 ms
- Batch of 256: mean 20.049 ms, or 0.0783 ms per state
- Full validation pass per-state batched inference: 0.214 ms and 0.186 ms
  in the two repeated passes

## Mechanical verification

All checkpoints loaded through the strict checkpoint reader and reproduced
their embedded model and optimizer digests:

| Checkpoint | Epoch | File digest | Model-state digest |
| --- | ---: | --- | --- |
| `initial.pt` | 0 | `af7181af…c2f8a` | `5a25249c…60c24` |
| `best-validation.pt` | 3 | `f88db1ba…663f8` | `edd353f5…26afe` |
| `latest.pt` | 8 | `d1be4f40…d5097` | `bf38438b…6efca` |
| `final.pt` | 8 | `949f46bb…771b` | `bf38438b…6efca` |

Two deterministic validation passes matched exactly for every loss, count,
placement, role, dealer-status, and agreement metric. Timing was excluded from
the equality assertion. Exported inference matched the selected checkpoint
exactly.

- Selected checkpoint:
  [`best-validation.pt`](../../runs/sam-policy-training-001/checkpoints/best-validation.pt)
- Exported artifact:
  [`policy.pt`](../../runs/sam-policy-training-001/artifacts/policy.pt)
- Artifact file digest:
  `5d595935e92ed0b378b2507345850e050484b6afbb84eba710c8fb14751483ab`
- Artifact model-state digest:
  `edd353f5209ba98220a5c69985184be583ff2bcce3e73ab3a1aca4f3de326afe`
- Run summary:
  [`summary.json`](../../runs/sam-policy-training-001/metrics/summary.json)
- Progress log:
  [`output-sam-policy-training-001`](../../output-sam-policy-training-001)

Standalone evaluation:

```sh
.venv/bin/python -m dracula.sam_policy_training validate \
  --run runs/sam-policy-training-001
```

Artifact verification:

```sh
.venv/bin/python -m dracula.sam_policy_training export \
  --run runs/sam-policy-training-001
```

## Observed limitations

- The validation overlay contains only two complete deck roots. It has only
  12 placement-1 rows and 48 placement-2 rows, so those placement estimates
  have high sampling uncertainty.
- Natural branch frequencies place 49,152 of 65,532 validation rows (75%) at
  placement 7. Aggregate top-2 and top-3 agreement therefore emphasize the
  two-action final learned placement.
- Validation cross-entropy improved through epoch 3 and then worsened while
  top-1 agreement continued rising. Later models became more confident,
  including on held-out disagreements.
- The snapshot audit found repeated visible observations with conflicting Sam
  labels, consistent with the teacher's sampled search variation. This
  imposes irreducible classification noise for a deterministic standalone
  argmax policy.
- Offline imitation agreement does not establish gameplay strength. This
  stage intentionally performed no gameplay integration or comparison.

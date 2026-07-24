# Teacher v2 supervised-training smoke

## Assessment

The supervised policy/value trainer passes its mechanical smoke gate against
the sealed Teacher v2 dataset. Dataset isolation, optimization, checkpointing,
CPU resume, export, and device selection behave as specified. The
339,978-parameter architecture and its losses were not changed.

This run is not candidate evidence. It contains only eight training games and
two validation games, and the preceding collection smoke found that the
32-visit policy targets are nearly uniform in early and middle play.

## Configuration

```text
dataset                 runs/teacher-v2-smoke-001
dataset digest          b231051c1ee7474bb197cb692baa7ae0803d736720a25799eb80b7399216a9cc
training examples       336 from 8 games
validation examples      84 from 2 games
device                   CPU
batch size               256
optimizer                AdamW
learning rate            3e-4
weight decay             1e-4
gradient clipping        global norm 1.0
epochs                    20
```

The fixture sets have no overlap. Both manifests identify
`shallow-response-teacher-v2-32x4`; no version 1 dataset enters the run.

## Training result

| Epoch | Training policy loss | Training value loss | Validation policy loss | Validation value loss | Policy accuracy | Value MAE | Maximum gradient norm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 2.207935 | 0.247192 | 2.210354 | 0.183446 | 3.57% | 0.367965 | 6.254214 |
| 7, selected | 2.203199 | 0.018562 | 2.205634 | 0.034385 | 11.90% | 0.139802 | 0.419656 |
| 19 | 2.202196 | 0.001715 | 2.205209 | 0.036144 | 9.52% | 0.148718 | 0.218663 |

Every loss, metric, gradient norm, and epoch duration was finite. Mean epoch
time was 0.042 seconds; the measured range was 0.033–0.079 seconds.

Validation total loss improved from 2.393800 to 2.240019, with epoch 7 retained
as best. Most learning came from the value head. Validation policy
cross-entropy changed only from 2.210354 to 2.205634 at the selected epoch,
consistent with the nearly uniform search targets found in the dataset smoke.

The latest checkpoint differs from initialization in every tested model
section:

| Section | Changed parameters | Maximum absolute change |
| --- | ---: | ---: |
| Shared body input | 241,664 | 0.009959 |
| Policy output head | 128 | 0.009287 |
| Value output head | 64 | 0.006123 |

## Recovery and artifacts

The controlled run was interrupted before epoch 5 began, after epoch 4 had
committed. It retained no temporary artifact. Resume reproduced all 20
deterministic minibatch orders and matched the clean run bit-for-bit for:

- The epoch-latest model and optimizer state.
- The best-validation checkpoint.
- The final checkpoint.

The iteration-start, every epoch metric and report, epoch-latest,
best-validation, final, and exported model all bind the Teacher v2 dataset
digest. Exported inference exactly matches the best-validation checkpoint on
the held-out tensors.

```text
best model state       78eab11b26683287f1d0d8fa67cd92ad312357d77cc694353690e0cb014aa211
exported artifact      runs/teacher-v2-training-smoke-001/teacher-v2-training-smoke.pt
```

No PPO ratio, behavior log probability, recurrence, separate critic, or
population metadata appears in the resolved configuration, checkpoints, or
export.

## Device benchmark

The standard three-warm-up and five-timed-epoch profile produced:

| Device | Median epoch | Examples/second | Peak RSS | Swap growth |
| --- | ---: | ---: | ---: | ---: |
| CPU | 0.0182 s | 18,512 | 405.9 MiB | 12.1 MiB |
| MPS | 0.0961 s | 3,497 | 467.8 MiB | 92.0 MiB |

Both devices produced finite results and initial outputs within the documented
tolerance. CPU was 5.29 times faster for this workload and used less measured
memory. CPU remains the correct default.

## Corrections made during validation

Per-epoch metrics and reports now record the dataset digest directly. The
`stopped_early` result now remains false when training reaches the configured
maximum epoch, even if the stale-validation count also meets the early-stop
condition on that final epoch. Neither correction changes model computation.

## Conclusion

The existing supervised trainer is ready to train a contract-valid Teacher v2
corpus and recover exactly on CPU. The current 32×4 smoke corpus remains
unsuitable as production training data because its policy targets do not
encode the teacher's early- and middle-round preferences strongly enough. No
full collection or production training was started.

Detailed validation output is in
`.local/teacher-v2-training-smoke-validation.json`; training progress is in
`output-teacher-v2-training-smoke`.

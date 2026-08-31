# Teacher v2 response-ranking dataset 001

## Result

The first substantive response-ranking corpus completed and passed artifact,
privacy, determinism, and recovery validation. No model training was performed,
and this report makes no deployment or competence decision.

## Configuration

```text
Teacher                    symmetry-aware Teacher v2
Outer simulations          32
Response completions       4 per strategic group
Workers                    4 complete-game processes
PyTorch threads            1 per worker
Training games             48
Validation games           12
Root seed                  dracula-teacher-v2-response-dataset-001-v1
```

Each game includes both Queen and King play over six alternating-dealer rounds.
Training and validation fixtures use separate deterministic split namespaces
and have no overlapping fixture IDs.

## Dataset

| Split | Games | Unique response examples | Manifest digest |
| --- | ---: | ---: | --- |
| Training | 48 | 154,839 | `025f15d989dfe30cca4ed3bd4fadbe7b3fa6ae3c7fb680002957e1d0ea3d2045` |
| Validation | 12 | 38,652 | `5a2aea5e1999b757b021e8ea57eff9d62b62519a5b1cb8422f2b29e4f6e08e69` |
| Total | 60 | 193,491 | `bf902e36916397e3a589fe081c9dc102f05998e3c783bc98a6dffa0f412d3148` |

The collection configuration digest is
`31c51ed309b189015b05d990c3ba8bdeeb87f6f8c5a73d5ba0264994de11ccce`.
The sealed run occupies 221.8 MiB.

Every row contains a boolean 875-element observation, a boolean 4×8 legal
mask, exact strategic-action groups, float64 four-completion mean terminal
differentials, selected representative and concrete actions, placement, role,
dealer status, and deduplication identity. The corpus contains no root-visit
examples or round-return training targets.

## Response-state distribution

The response evaluator is entered after an outer-search root action, so
placement 1 does not produce a shallow-response example. Forced placement 8 is
also absent by contract.

| Placement | Examples | Mean groups | Mean score spread | Mean top-two gap |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 10,494 | 12.00 | 0.3332 | 0.0534 |
| 3 | 13,479 | 11.95 | 0.3119 | 0.0508 |
| 4 | 26,173 | 13.45 | 0.3084 | 0.0461 |
| 5 | 37,489 | 7.51 | 0.2375 | 0.0469 |
| 6 | 48,887 | 5.83 | 0.1809 | 0.0373 |
| 7 | 56,969 | 2.00 | 0.0734 | 0.0734 |

Across all examples, the mean strategic-group count was 6.82:

```text
2 groups   29.44%
4 groups    2.10%
6 groups   27.89%
8 groups   14.65%
9 groups    3.62%
12 groups  12.28%
15 groups  10.02%
```

The overall score spread had mean `0.1949`, median `0.1733`, and p95
`0.4683`. The top-two score gap had mean `0.0528`, median `0.0317`, and p95
`0.1767`.

Selected groups covered all 32 representative action indexes. The five most
frequent indexes were 18, 21, 26, 29, and 10, each accounting for
4.19–4.31% of examples.

Actor/dealer coverage was:

```text
Queen dealer       42,945
Queen non-dealer   53,773
King dealer        42,609
King non-dealer    54,164
```

The unequal dealer and non-dealer example totals reflect the different
numbers and shapes of shallow response states encountered, not an imbalance
in game fixtures.

## Deduplication

The searches made 203,331 shallow-response requests and sealed 193,491 unique
cache identities:

```text
total duplicate requests removed   9,840
duplicate request rate              4.84%
observer emissions                194,166
repeated emitted identities           675
```

No cache identity occurs twice within a shard, across shards, or between the
training and validation manifests.

## Recovery and deterministic replay

An actual search interruption was requested after the first of two recovery
fixtures sealed. The interrupted directory retained one complete shard and no
temporary shard. Resuming completed the remaining fixture and produced dataset
digest
`dd91c9a09ae703c25545c1983bcd432232639e7f19942daed7851c93cbd69ea8`,
identical to a clean collection from the same configuration.

A completed-run resume of this 60-game corpus revalidated all shards and
returned the original corpus digest without rewriting the metrics artifact.
The collector now has regression coverage for both planner-level interruption
translation and completed-run metric preservation.

## Contract verification

- All 60 shard content digests and both manifest digests were recomputed by
  the strict loaders.
- Every row has the required tensor shapes and CPU dtypes.
- Every strategic group partitions the server-derived legal concrete actions
  exactly.
- Every selected group is the maximum recorded mean with canonical
  representative tie-breaking.
- Every selected concrete action belongs to its selected group.
- Training and validation fixture IDs are disjoint.
- Schema and configuration bindings match in resolved configuration, shards,
  and manifests.
- No determinization, opponent hand, stock order, engine seed, sampled outer
  state, search tree, policy hidden state, or model data is present.
- No temporary, partial, or stop-request file remains.
- The focused engine, bridge, information-state, symmetry, search, and
  response suites pass: 121 tests.

## Performance and artifacts

| Measurement | Result |
| --- | ---: |
| Wall time | 3,180.0 s (53 min 0 s) |
| Aggregate worker search time | 12,548.7 s |
| Collection throughput | 219,044 examples/hour |
| Game throughput | 67.9 games/hour |
| Peak RSS per worker | 189.1 MiB |
| Four-worker RSS upper bound | 756.4 MiB |
| Sealed disk use | 221.8 MiB |

The process-level RSS value is recorded independently for each worker; the
four-worker figure is a conservative sum, not a sampled process-tree peak.
The browser gameplay server remained live during collection, so these timings
include contention from any concurrent manual play.

Artifacts:

```text
runs/teacher-v2-response-dataset-001/
output-teacher-v2-response-dataset-001
```

Important artifact hashes:

```text
resolved configuration  b8a7bbdc4e11eded723e9fbdb28f20f5ee83623ab275492b4c136a7de2df4445
fixture splits          2f98d13bdd686f38720f2ef53b1222c25fd0753fcc1a460e75d6a75b06fe4aef
metrics                 cf2a9859acfd694e0905875bc6f927a6cbb7d4fd9dafe68117e60bc974148017
progress log            930ad1bcef5af102e7eb31bf05406f176742568c71c15af398d2da680d3221bd
```

## Limitations

The four-completion group means are Monte Carlo estimates and therefore retain
sampling noise, especially where the top-two gap is small. Sixty complete
games provide substantially broader response-state coverage than the smoke
corpus but do not establish how a trained ranker will generalize to rare
states. The 12-game validation split supports first-run comparison; it is not
an independent gameplay-strength result. Those questions belong to subsequent
training and controlled gameplay evaluation.

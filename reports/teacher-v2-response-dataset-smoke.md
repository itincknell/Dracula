# Teacher v2 response-ranking dataset smoke

## Result

The response observer produced a deterministic, private, resumable dataset
without changing Teacher v2 search behavior. No model training was performed.

## Configuration

```text
Teacher                    symmetry-aware Teacher v2
Outer simulations          32
Response completions       4 per strategic group
Workers                    4 complete-game processes
Training games             4
Validation games           2
Root seed                  teacher-v2-response-dataset-smoke-v1
```

Each game used both Queen and King and alternated the dealer for six rounds.
Training and validation fixtures were derived through distinct split
components and had no overlapping fixture IDs.

## Dataset

| Split | Games | Unique response examples | Manifest digest |
| --- | ---: | ---: | --- |
| Training | 4 | 12,930 | `ec5ca6b4cba8899a2909f727b124004bf6f2715bc7f81044828e92041374bae8` |
| Validation | 2 | 6,463 | `8c1a5c1b8666606f084c7b127b3ebc58611037a1e4954ae05b2c29e874559ba1` |
| Total | 6 | 19,393 | `110dd3658af4ab33ccd6a9eefe44f530b2f8fb1f17321e917448cdaacb17ffec` |

The collection configuration digest is
`549cbbf2d85e739e3f4f9e4056d040b0869e55baa6c440343faa8ead58ebe5da`.
The sealed data occupies 22.2 MiB.

Every row stores the 875-bit observation, legal mask, flattened strategic
groups, float64 four-completion means, selected representative and concrete
actions, placement, actor role, dealer status, and cache identity. It contains
no root visits or round-return target.

## Response-state distribution

The shallow evaluator is entered after the outer root action, so placement 1
does not produce an inner-response example.

| Placement | Examples | Mean groups | Mean score spread | Mean top-two gap |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 1,054 | 12.00 | 0.3451 | 0.0552 |
| 3 | 1,342 | 12.00 | 0.3170 | 0.0500 |
| 4 | 2,625 | 13.48 | 0.3137 | 0.0455 |
| 5 | 3,772 | 7.53 | 0.2437 | 0.0473 |
| 6 | 4,886 | 5.85 | 0.1873 | 0.0378 |
| 7 | 5,714 | 2.00 | 0.0736 | 0.0736 |

Across all examples, the mean group count was 6.83. Group-count frequencies
were:

```text
2 groups   29.46%
4 groups    1.93%
6 groups   27.82%
8 groups   14.90%
9 groups    3.34%
12 groups  12.54%
15 groups  10.01%
```

The overall score spread had mean `0.1994`, median `0.1767`, and p95 `0.4817`.
The top-two score gap had mean `0.0530`, median `0.0317`, and p95 `0.1750`.

Selected groups were distributed across all 32 representative action indexes.
The most frequent were indexes 26, 18, 29, 10, and 21, each representing
4.10–4.39% of examples. No single group dominated the corpus.

Actor/dealer coverage was:

```text
Queen dealer       4,210
Queen non-dealer   5,484
King dealer        4,355
King non-dealer    5,344
```

## Deduplication

The searches made 20,356 shallow-response requests. They emitted 19,473 unique
cache-miss observations within their request-local caches. Dataset-level cache
identity deduplication sealed 19,393 rows:

```text
request-local cache hits                    883
repeated identities across root searches     80
total duplicate requests removed             963
duplicate rate                              4.73%
```

No cache identity occurs twice within a shard, across shards in one split, or
between the training and validation manifests.

## Contract verification

- The observer-disabled and observer-enabled planners return identical visits,
  values, actions, diagnostics, and configuration digests.
- Every group value is copied directly from the evaluator's four-completion
  mean and retained as float64.
- Every selected group is the maximum mean with canonical representative
  tie-breaking.
- Each row's strategic groups partition its legal concrete actions exactly.
- Placement 8 is absent.
- Both splits bind the search, response, destination-symmetry, observation,
  action, response-example, action-group, and ranking schemas and both
  configuration digests.
- Shard validation rejects private or unknown fields. No determinization,
  authoritative hand, stock order, engine seed, sampled outer state, tree,
  policy hidden state, or model data is sealed.
- An interruption test leaves only completed game shards; resume skips those
  shards and reproduces the uninterrupted dataset digest.

A clean second collection used the same fixtures and configuration in another
directory. Worker completion order differed, but both manifest digests and the
combined dataset digest matched exactly.

## Performance and artifacts

| Measurement | Primary | Clean reproduction |
| --- | ---: | ---: |
| Wall time | 338.3 s | 357.9 s |
| Aggregate worker search time | 1,026.4 s | 1,094.1 s |
| Peak RSS per worker | 188.1 MiB | 190.5 MiB |

This collector records per-worker RSS. The prior process-tree benchmark of the
same 32×4 four-worker search profile measured 847 MiB peak total RSS.

Primary artifacts:

```text
.local/response-distillation/teacher-v2-response-dataset-smoke-20260723/
output-teacher-v2-response-dataset
```

Clean reproduction:

```text
.local/response-distillation/teacher-v2-response-dataset-smoke-20260723-repro/
output-teacher-v2-response-dataset-repro
```

Detailed distributions are in
`.local/response-distillation/teacher-v2-response-dataset-smoke-20260723/response-distillation/statistics.json`.

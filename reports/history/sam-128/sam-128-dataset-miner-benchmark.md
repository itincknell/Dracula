# Sam-128 dataset-miner benchmark

Date: 2026-07-24
Machine: Mac15,12; Apple M3; 8 CPU cores; 8 GiB unified memory
System: macOS 14.7.4
Teacher: symmetry-aware nested Teacher v2, 128 outer simulations × 128
actor-response simulations
Teacher configuration digest:
`20933f828ed85af78b58b39439ea10bb43a9ebe7ecc82b8bf351cdcebe2a68f2`
Benchmark root:
`.local/sam-128-dataset-miner-benchmark-001`

## Result

The completed miner is deterministic and structurally correct. The benchmark
did not start a production corpus.

Four workers were fastest on the measured six-fixture workload, completing it
in 118.1 seconds: 2.02 times the one-worker throughput. Six workers were
slightly slower at 122.2 seconds and raised peak benchmark-process RSS from
822 MB to 1.18 GB.

The placement-weighted projection for four workers is:

| Output | Time | Logical bytes | Allocated disk |
| --- | ---: | ---: | ---: |
| One 5,461-example round | 2 h 45 min | 14.7 MB | 32.3 MB |
| One 32,766-example deck | 16 h 30 min | 87.9 MB | 193.9 MB |
| One million examples | 21.0 days | 2.68 GB | 5.92 GB |

The projection assumes unique information states and no cache savings. It uses
the actual branching weights for placements 1–7, not an equal mix of the seven
placements. It extrapolates from fixed states rather than a complete real
Sam-128 round; it should be treated as a capacity estimate.

Every worker-count run observed system swap growth. Four workers added 183 MB
at peak during its measurement. The machine already had about 7.3 GB of swap
in use, so this benchmark does not establish that the miner caused all
observed system movement. No macOS thermal or performance warning was
reported. Matched deterministic reruns were a median 0.8% slower than their
initial runs, with a worst measured slowdown of 2.1%; there was no measured
thermal-throughput collapse.

## Teacher latency

The ordinary-state rows pair the initial query with the exact deterministic
rerun. Terminal evaluations are identical across each pair.

| Placement | Strategic groups | Initial | Rerun | Terminal evaluations |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 8 | 140.50 s | 140.47 s | 83,072 |
| 2 | 12 | 101.05 s | 101.92 s | 67,200 |
| 3 | 15 | 68.68 s | 67.99 s | 51,200 |
| 4 | 12 | 40.64 s | 40.97 s | 34,304 |
| 5 | 6 | 19.41 s | 19.82 s | 16,896 |
| 6 | 6 | 10.06 s | 10.13 s | 12,928 |
| 7 | 2 | 0.100 s | 0.101 s | 128 |

Across all 19 ordinary, symmetry, and repeated queries:

```text
mean:     63.15 s
p50:      67.45 s
p95:     140.50 s
maximum: 140.50 s
```

### Authoritative early symmetry patterns

| Pattern | Placement | Strategic groups | Latency | Terminal evaluations |
| --- | ---: | ---: | ---: | ---: |
| Center only | 1 | 8 | 140.50 s | 83,072 |
| Center plus position 2 | 2 | 12 | 101.05 s | 67,200 |
| Center plus position 8 | 2 | 12 | 100.88 s | 67,200 |
| Center plus position 4 | 2 | 12 | 100.97 s | 67,200 |
| Center plus position 6 | 2 | 12 | 101.22 s | 67,200 |
| Horizontal three-card line | 3 | 9 | 67.45 s | 50,432 |
| Vertical three-card line | 3 | 9 | 67.44 s | 50,432 |

The four center-plus-adjacent cases differed by 0.34 seconds. The horizontal
and vertical line cases differed by 0.01 seconds. Symmetry orientation did not
introduce a measurable cost asymmetry.

### Cold and warm cache

The placement-4 cache probe ran the same exact query twice:

```text
cold search and verified cache write: 40.601 s
verified warm-cache lookup:             0.000402 s
cache misses / hits:                    1 / 1
```

The cached teacher group and concrete action matched the cold result.

## Real Sam-128 branch prefixes

Each depth reused the content-addressed results sealed by shallower depths.
`Teacher queries` therefore counts new Sam-128 searches; `cache hits` counts
rows resolved from earlier prefix work.

| Depth | Examples | Frontier branches | New teacher queries | Cache hits | Terminal evaluations | Wall time | Examples/hour |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 4 | 1 | 0 | 83,072 | 139.0 s | 25.9 |
| 2 | 5 | 16 | 4 | 1 | 268,800 | 400.1 s | 45.0 |
| 3 | 21 | 64 | 16 | 5 | 813,440 | 1,072.0 s | 70.5 |
| 4 | 85 | 256 | 64 | 21 | 2,210,944 | 2,570.2 s | 119.1 |

The prefix process peaked at 203 MB RSS. These prefixes stopped at their
frontiers, so they intentionally produced no terminal round leaves.

## Complete structural traversal

The deterministic injected teacher traversed one complete six-round deck with
the production `k = 4` branch structure:

```text
examples:                    32,766
rounds:                           6
terminal leaves:             49,152
teacher calls/cache entries: 32,766
sealed data shards:               30
wall time:                    413.6 s
peak RSS:                       220 MB
```

The deck has 8,192 terminal leaves per round. Placement seven has two legal
strategic actions, so terminal arity is two at that final learned placement.

Artifact sizing:

```text
logical corpus bytes:             87,911,934
logical bytes/example:                 2,683
filesystem allocated bytes:      193,921,024
allocated bytes/example:                5,918
sealed-shard payload bytes:        59,602,432
mean payload bytes/sealed shard:    1,986,748
```

Filesystem allocation is materially larger than logical content because the
content-addressed teacher cache uses one small file per unique state.

## Process parallelism

Every run used the same six disjoint real-teacher fixtures at placements 2–7.
Every process used one PyTorch thread. All four worker counts produced the same
result digest for every fixture.

| Workers | Wall time | Queries/hour | Speedup | Mean aggregate CPU | Peak worker RSS | Peak total RSS | Peak swap growth |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 238.6 s | 90.5 | 1.00× | 99.5% | 202 MB | 392 MB | 148 MB |
| 2 | 136.4 s | 158.4 | 1.75× | 195.1% | 199 MB | 435 MB | 67 MB |
| 4 | 118.1 s | 182.9 | 2.02× | 250.6% | 197 MB | 822 MB | 183 MB |
| 6 | 122.2 s | 176.8 | 1.95× | 256.7% | 192 MB | 1.18 GB | 112 MB |

The measured worker-count fixture has one state from each placement 2–7. The
examples/hour projections below reweight those latencies to the branch tree's
actual `1, 4, 16, 64, 256, 1,024, 4,096` placement counts.

| Workers | Projected examples/hour | Projected round | Projected deck | Projected million |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 982 | 5 h 33 min | 33 h 21 min | 42.4 days |
| 2 | 1,719 | 3 h 11 min | 19 h 4 min | 24.2 days |
| 4 | 1,985 | 2 h 45 min | 16 h 30 min | 21.0 days |
| 6 | 1,919 | 2 h 51 min | 17 h 4 min | 21.7 days |

Four workers gave the highest measured throughput. Two workers had the lowest
observed swap growth and used about half the peak total RSS. This report does
not change the configured worker count.

## Reproducibility and artifacts

One fixture at every placement was rerun. Selected group, concrete action,
visits, action values, continuation, terminal counts, and diagnostic digest
matched exactly in all seven cases. Worker counts also produced identical
per-fixture result digests.

Raw sealed results:

| Artifact | Content digest |
| --- | --- |
| `teacher-measurements.json` | `0d7542f43d1e233fa54338b8a886ca47f9ec8426cb1ef8cbd97dd7e351689ebd` |
| `prefix-measurements.json` | `351dd8254233b7c64fa0efc583d90ada5918a6632f9d50b2051b82acaacf41e6` |
| `structural-measurement.json` | `07ad9b041154d6b425ce782b2b23ecb18144b5a3ec3e04892f0514c176806113` |
| `worker-measurements.json` | `67076210546df88a195c9b36b12f3d0feb235ac2ef795fd9a7b6ed36e20ca912` |
| `summary.json` | `f817f830a0f9714c4fede994276aff5d59513d8cca9c37e63d9d41a9a423c96f` |

The preserved SQLite database was not opened or modified by the benchmark.

## Verification

```text
tests/test_sam_teacher.py + tests/test_sam_miner.py: 90 passed
structural corpus verify:
  decks:          1
  rounds:         6
  rows:           32,766
  shards:         30
  terminal leaves: 49,152
  corpus manifest:
    cb7aba78e7baf01ed2e57eba1541009b583d21570b1e904fd4d2b65f6e5b576e
```

Every top-level benchmark artifact was reloaded and checked against its sealed
content digest before this report was written.

## Preview restoration

The benchmark recorded and restored the pre-measurement local controller:

```text
opponent mode:              search-v2-nested
outer simulations:          32
actor-response simulations: 32
exploration:                sqrt(2)
database:                   .local/dracula.sqlite3
API:                        http://127.0.0.1:8000
frontend:                   http://127.0.0.1:4173
```

After restoration, `/health` returned `200 OK` with narration disabled and the
frontend returned `200 OK`.

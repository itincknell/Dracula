# Teacher v2 collection parallelism

## Decision

Use four complete-game worker processes for the 240-game Teacher v2 collection.
Each worker uses one PyTorch intra-op thread and one inter-op thread.

Six workers completed the fixture set 20% faster than four, but used 40% more
peak total RSS, ended with 200 MiB additional swap, and increased aggregate
search time by 50%. Four workers were the fastest configuration without
sustained swap growth.

## Method

The benchmark ran July 23, 2026 on the project M3 MacBook Air with eight CPU
cores and 8 GB RAM. Local FastAPI and Vite processes were stopped first. Each
worker count processed the same six full games from root seed
`teacher-v2-parallelism-v1` using the approved 32×4 search profile and a new
decision-cache directory.

```bash
.venv/bin/python tools/archive/teacher-v2/benchmark_teacher_parallelism.py \
  --output-root .local/teacher-v2-parallelism-20260723 \
  --games 6 --workers 1 2 4 6
```

The harness sampled the complete collector process tree every 250 ms. All runs
produced trajectory digest
`342a4abb1106d9b520ed8839e01bbc9ed3207fc0366702a07e445ba3231c0513`.
Every sealed worker metric reported PyTorch thread counts `(1, 1)`.

## Results

| Workers | Wall time | Aggregate search | Examples/hour | Mean CPU | Peak process RSS | Peak total RSS | Peak swap growth | End swap growth |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1,056.0 s | 1,053.6 s | 860 | 99% | 194 MiB | 194 MiB | 346 MiB | +291 MiB |
| 2 | 601.8 s | 1,180.0 s | 1,511 | 196% | 194 MiB | 574 MiB | 37 MiB | +33 MiB |
| 4 | 487.6 s | 1,540.7 s | 1,866 | 313% | 190 MiB | 847 MiB | 152 MiB | -8 MiB |
| 6 | 391.3 s | 2,306.4 s | 2,326 | 580% | 191 MiB | 1,183 MiB | 213 MiB | +200 MiB |

System swap was already approximately 8.1 GiB before the first measurement, so
peak movement includes machine-wide activity. The end-to-start measurement is
the selection constraint: four workers released its transient swap, while six
retained an additional 200 MiB. macOS recorded no formal thermal warning, but
aggregate search time per game rose from 176 seconds at one worker to 384
seconds at six, showing clear throughput collapse under the six-worker load.

Historical runs and `.local/dracula.sqlite3` were not modified. Benchmark
artifacts are under `.local/teacher-v2-parallelism-20260723`.

## Interruption

A live two-worker probe received `SIGINT` during active searches. It exited with
code 130 in 0.30 seconds, recorded phase `interrupted`, left zero game shards,
removed the stop marker, and left no worker process running.

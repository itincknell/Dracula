# Teacher v2 shallow-response implementation

The active Teacher v2 planner retains version 1 outer UCT and uses a shallow
greedy Monte Carlo policy for non-forced continuation decisions. For each
completion index, every legal candidate starts from the same actor-local
determinization. Candidate worlds then continue with uniform legal play and
receive the engine's exact actor-relative round differential.

The nested actor-local UCT implementation is no longer used by active search
code or configuration. Version 1 behavior remains fixed by its golden result
digests.

## Mechanical validation

- 338 Python tests passed.
- Package compilation, dependency validation, and wheel construction passed.
- Response tests cover shared determinizations, exhaustive candidate coverage,
  actor-view privacy, exact values, deterministic selection, cache accounting,
  interruption, forced transitions, and result-boundary compatibility.

## Cost benchmark

Measurements use one process and 32 outer simulations on the project M3
MacBook Air. Terminal counts include the 32 outer simulations.

| Stage | Completions/action | Latency | Terminal evaluations | Evaluations/second |
| --- | ---: | ---: | ---: | ---: |
| Early | 1 | 9.967 s | 1,051 | 105.5 |
| Early | 2 | 20.231 s | 2,168 | 107.2 |
| Early | 4 | 38.254 s | 4,224 | 110.4 |
| Middle | 1 | 0.607 s | 96 | 158.1 |
| Middle | 2 | 0.879 s | 160 | 182.0 |
| Middle | 4 | 1.361 s | 288 | 211.6 |
| Late | 1 | 0.158 s | 32 | 202.8 |
| Late | 2 | 0.165 s | 32 | 194.0 |
| Late | 4 | 0.165 s | 32 | 193.7 |

Peak process RSS was 204.5 MiB. Late-round cost is independent of response
configuration in this fixture because only the forced placement follows the
root decision.

These measurements establish computational cost only. The 500-outer target and
strategic acceptance gates have not run, so Teacher v2 is not approved for
human testing or teacher collection.

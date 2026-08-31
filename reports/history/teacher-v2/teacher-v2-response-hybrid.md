# Teacher v2 response-ranker hybrid

## Result

The response ranker is integrated behind two explicit experimental Teacher v2
response modes. Pure symmetry-aware 32×4 Teacher v2 remains the unchanged
control.

The experiment demonstrates substantial computational savings:

- `student-direct` removes all inner terminal response evaluations.
- `student-top-2` removes 69.7% of inner terminal evaluations and halves median
  decision latency across the fixed benchmark.

These are efficiency results only. The ranker had near-random held-out ranking
quality, and the hybrids frequently selected different root actions from pure
Teacher v2. Neither hybrid is an accepted gameplay controller.

## Controllers

All three controllers use 32 outer UCT simulations and exact engine scoring at
the end of every simulated round.

| Mode | Inner response |
| --- | --- |
| Pure | Four completions for every legal strategic group |
| Student-direct | Select the model's highest-ranked group directly |
| Student-top-2 | Four completions for only the two highest-ranked groups |

The network does not provide a root prior or terminal value. Its value head is
unused. Paired destinations retain the existing derived fair coin.

Hybrid cache identity binds the actor information-state fingerprint,
response-ranker artifact SHA-256, shortlist size, and response configuration.
The ranker receives only the acting player's 875-bit information projection and
legal strategic groups.

## Benchmark

Each controller ran in a separate single-threaded process over the same nine
fixed information states: three fixtures each at placements 1, 4, and 6. Each
mode received the same raw request seed for each state.

| Measurement | Pure | Student-direct | Student-top-2 |
| --- | ---: | ---: | ---: |
| Decisions | 9 | 9 | 9 |
| Outer terminal evaluations | 288 | 288 | 288 |
| Response requests | 819 | 819 | 819 |
| Unique responses | 812 | 814 | 814 |
| Model calls | 0 | 814 | 814 |
| Candidate groups evaluated | 5,367 | 0 | 1,628 |
| Inner terminal evaluations | 21,468 | 0 | 6,512 |
| Terminal-evaluation savings | — | 100.0% | 69.7% |
| Median latency | 0.844 s | 0.097 s | 0.420 s |
| p95 latency | 5.951 s | 0.201 s | 1.475 s |
| Peak process RSS | 191.2 MiB | 199.2 MiB | 199.7 MiB |

### Placement latency

| Placement | Pure | Student-direct | Student-top-2 |
| ---: | ---: | ---: | ---: |
| 1 | 5.941 s | 0.193 s | 1.458 s |
| 4 | 0.844 s | 0.097 s | 0.420 s |
| 6 | 0.097 s | 0.051 s | 0.110 s |

### Placement terminal evaluations

| Placement | Pure | Student-direct | Student-top-2 | Top-2 savings |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 16,836 | 0 | 4,032 | 76.1% |
| 4 | 3,904 | 0 | 1,752 | 55.1% |
| 6 | 728 | 0 | 728 | 0.0% |

Top-2 provides no late-state reduction when only two strategic groups remain,
and its model overhead made the placement-6 median slightly slower than pure
Teacher v2.

Student-direct matched pure Teacher v2's selected root action on four of nine
states. Student-top-2 matched on three of nine. These disagreements are not a
strength evaluation, but they are consistent with the ranker's weak held-out
metrics and prevent an efficiency-only acceptance.

Raw results are in `.local/response-hybrid-benchmark.json`, SHA-256
`999049faaed57c0b338a7a1b6b9ddae3126a3a22715bd8e0e3bc005a7caaa17a`.
The benchmark used response-ranker artifact SHA-256
`f6504bfb246e56e09d0e98d6a9f668d9744113045678450c84c5e48fc8cc9f75`.

## Verification

- The pure configuration digest and search results remain unchanged.
- Student ranking covers every legal strategic group exactly.
- Top-2 uses one sampled actor-valid world per completion for both candidates.
- Student-direct performs no shallow terminal evaluation.
- Forced placements bypass both model inference and response evaluation.
- Repeated requests reproduce visits, values, and concrete actions.
- Artifact digest and shortlist size change request identity.
- The model boundary accepts `SearchInformationState`, never authoritative
  engine state, opponent hands, stock order, or engine seeds.
- Outer simulations continue through engine transitions to exact completed
  round scores.
- Experimental gameplay selection requires an explicit mode and artifact path.

## Local selection

```text
DRACULA_OPPONENT_MODE=search-v2-student-direct
DRACULA_OPPONENT_MODE=search-v2-student-top-2
DRACULA_RESPONSE_RANKER_ARTIFACT=runs/teacher-v2-response-ranker-001/artifacts/response-ranker.pt
DRACULA_SEARCH_SIMULATIONS=32
DRACULA_SEARCH_RESPONSE_COMPLETIONS=4
```

Pure Teacher v2 remains `DRACULA_OPPONENT_MODE=search-v2`.

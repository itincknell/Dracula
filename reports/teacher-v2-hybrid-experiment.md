# Teacher v2 hybrid experiment

## Conclusion

Student-top-2 is the only hybrid that preserved recognizable Teacher v2 competence in this controlled comparison while materially reducing computation. It finished 11–13 against pure Teacher v2 across 12 identical decks in both roles, split 144 rounds exactly 50/50, and trailed by 1.19 points per round. On the fixed decision matrix it removed 71.7% of inner terminal evaluations and reduced p50 latency from 3.396 seconds to 0.969 seconds.

This is evidence for using top-2 as an experimental gameplay controller, not an automatic promotion. It selected the same strategic group as pure Teacher v2 on 45.2% of fixed decisions and lost two defensive fixture passes that pure retained. The hybrid is therefore preserving outcomes more successfully than it is imitating individual choices.

Student-direct was faster but clearly weaker. Student-top-3 improved fixture coverage but lost 5–19 in paired games; evaluating an extra noisy candidate did not improve the complete policy. No larger shortlist is justified by this experiment.

## Method

- Controllers: pure symmetry-aware 32×4 Teacher v2, student-direct, student-top-2, and the conditionally added student-top-3.
- Fixed decisions: all 12 constructive fixtures, all 14 defensive fixtures, and 24 role/dealer-balanced placement states repeated over five deterministic request seeds.
- Games: 12 fixed decks, each played with the hybrid as Queen and King against pure Teacher v2. Dealer alternated across all six rounds.
- Every paired fixed decision used the same information state and raw request seed. All modes retained 32 outer simulations and exact engine terminal scoring.
- Teacher-value regret is the best pure-search group mean minus the pure-search mean for the hybrid-selected group. It is diagnostic, not a promotion threshold.
- Response-ranker artifact: `f6504bfb246e56e09d0e98d6a9f668d9744113045678450c84c5e48fc8cc9f75`.
- Raw result: `.local/teacher-v2-hybrid-experiment/final.json` (`sha256:dad23e48d9116b2ce865f6dfc5b420b84e1c5c9651aa0d2096a5e86283827a40`).

## Strategic fixtures

The table reports strategic-group equivalence, so a fair-coin-selected mirrored destination is not counted as a different strategic choice.

| Suite | Fixture | Placement | Expected actions | Pure | Direct | Top-2 | Top-3 |
| --- | --- | ---: | --- | --- | --- | --- | --- |
| constructive | `dealer-late-forced` | 8 | `25` | pass; group 25 | pass; group 25 | pass; group 25 | pass; group 25 |
| constructive | `king-early-dealer-vampire` | 2 | `25` | pass; group 25 | pass; group 25 | pass; group 25 | pass; group 25 |
| constructive | `king-early-nondealer-suit` | 3 | `11` | fail; group 9 | fail; group 17 | fail; group 9 | fail; group 9 |
| constructive | `king-late-block` | 7 | `10` | pass; group 10 | pass; group 10 | pass; group 10 | pass; group 10 |
| constructive | `king-middle-preserve-card` | 5 | `10` | fail; group 8 | pass; group 10 | pass; group 10 | pass; group 10 |
| constructive | `king-middle-suit` | 6 | `24` | pass; group 24 | fail; group 7 | pass; group 24 | pass; group 24 |
| constructive | `queen-early-dealer-suit` | 2 | `3,11` | pass; group 11 | pass; group 11 | pass; group 3 | pass; group 3 |
| constructive | `queen-early-nondealer-suit` | 3 | `3` | pass; group 3 | pass; group 3 | pass; group 3 | pass; group 3 |
| constructive | `queen-late-offense-defense` | 7 | `20` | pass; group 20 | pass; group 20 | pass; group 20 | pass; group 20 |
| constructive | `queen-middle-avoid-vampire` | 6 | `4` | pass; group 4 | pass; group 4 | pass; group 4 | pass; group 4 |
| constructive | `queen-middle-color-exploit` | 6 | `29` | pass; group 29 | fail; group 18 | pass; group 29 | pass; group 29 |
| constructive | `queen-middle-offense-defense` | 5 | `27` | pass; group 27 | pass; group 27 | pass; group 27 | pass; group 27 |
| defensive | `king-early-dealer-constructive` | 2 | `19` | fail; group 1 | pass; group 19 | pass; group 19 | pass; group 19 |
| defensive | `king-early-nondealer-safe-intersection` | 3 | `4,23` | pass; group 23 | fail; group 3 | fail; group 15 | pass; group 4 |
| defensive | `king-late-block` | 7 | `2` | pass; group 2 | pass; group 2 | pass; group 2 | pass; group 2 |
| defensive | `king-late-dealer-forced` | 8 | `26` | pass; group 26 | pass; group 26 | pass; group 26 | pass; group 26 |
| defensive | `king-middle-avoid-suit` | 5 | `25` | fail; group 29 | fail; group 9 | fail; group 9 | pass; group 25 |
| defensive | `king-middle-denial-over-offense` | 5 | `5` | fail; group 4 | fail; group 28 | fail; group 4 | fail; group 4 |
| defensive | `queen-early-dealer-safe-intersection` | 2 | `28` | fail; group 8 | fail; group 24 | fail; group 24 | fail; group 1 |
| defensive | `queen-early-nondealer-avoid-suit` | 3 | `1,9` | fail; group 5 | fail; group 13 | fail; group 21 | fail; group 15 |
| defensive | `queen-late-block` | 7 | `7` | pass; group 7 | pass; group 7 | pass; group 7 | pass; group 7 |
| defensive | `queen-late-dealer-forced` | 8 | `29` | pass; group 29 | pass; group 29 | pass; group 29 | pass; group 29 |
| defensive | `queen-middle-avoid-color` | 5 | `18` | pass; group 18 | fail; group 5 | pass; group 18 | pass; group 18 |
| defensive | `queen-middle-avoid-vampire-destruction` | 5 | `23` | pass; group 23 | fail; group 20 | fail; group 0 | fail; group 0 |
| defensive | `queen-middle-constructive-over-empty-block` | 6 | `14` | pass; group 14 | pass; group 14 | pass; group 14 | pass; group 14 |
| defensive | `queen-middle-defensive-vampire` | 6 | `31` | pass; group 31 | fail; group 21 | pass; group 31 | pass; group 31 |

Totals:

| Controller | Constructive | Defensive |
| --- | ---: | ---: |
| pure | 10/12 | 9/14 |
| student-direct | 9/12 | 6/14 |
| student-top-2 | 11/12 | 8/14 |
| student-top-3 | 11/12 | 10/14 |

Top-3 was run because top-2 lost the `king-early-nondealer-safe-intersection` and `queen-middle-avoid-vampire-destruction` defensive actions that pure Teacher v2 passed. Top-3 recovered the safe-intersection fixture and added the `king-middle-avoid-suit` pass, but it still missed the Vampire-destruction fixture.

## Paired game results

| Hybrid subject | Record vs pure | Game VP | Round VP | Mean round differential |
| --- | ---: | ---: | ---: | ---: |
| student-direct | 5-19 | 20.8% | 37.2% | -8.65 |
| student-top-2 | 11-13 | 45.8% | 50.0% | -1.19 |
| student-top-3 | 5-19 | 20.8% | 42.4% | -6.58 |

### Role splits

| Hybrid | Queen games | King games | Queen round VP / diff | King round VP / diff |
| --- | ---: | ---: | ---: | ---: |
| student-direct | 2-10 | 3-9 | 36.1% / -7.04 | 38.2% / -10.26 |
| student-top-2 | 7-5 | 4-8 | 50.0% / +1.17 | 50.0% / -3.56 |
| student-top-3 | 2-10 | 3-9 | 40.3% / -5.42 | 44.4% / -7.74 |

### Dealer splits

| Hybrid | Dealer round VP / diff | Non-dealer round VP / diff |
| --- | ---: | ---: |
| student-direct | 30.6% / -9.64 | 43.8% / -7.67 |
| student-top-2 | 45.8% / -4.35 | 54.2% / +1.96 |
| student-top-3 | 44.4% / -7.65 | 40.3% / -5.50 |

Each role contains 12 games and 72 rounds. Each dealer split contains 72 rounds.

## Efficiency

The primary latency comparison uses the identical 146 fixed decisions for each controller.

| Controller | p50 | p95 | Max | Inner terminals | Reduction | Model calls | Model time | ms/call | Peak RSS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| pure | 3.396s | 7.537s | 9.846s | 364,116 | 0.0% | 0 | 0.000s | — | 204.3 MiB |
| student-direct | 0.182s | 0.402s | 0.544s | 0 | 100.0% | 12,904 | 6.752s | 0.523 | 214.1 MiB |
| student-top-2 | 0.969s | 2.091s | 3.234s | 103,032 | 71.7% | 12,879 | 7.259s | 0.564 | 215.6 MiB |
| student-top-3 | 1.144s | 2.380s | 3.589s | 140,280 | 61.5% | 12,875 | 6.695s | 0.520 | 213.3 MiB |

Gameplay timings below cover the hybrid-controlled decisions in the 24 paired games. Concurrent full-game evaluation increases tail latency relative to isolated fixed decisions.

| Hybrid | Decisions | p50 | p95 | Max | Inner terminals | Model calls / time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| student-direct | 504 | 0.228s | 0.578s | 0.766s | 0 | 38,836 / 25.12s |
| student-top-2 | 504 | 1.046s | 4.604s | 5.297s | 311,056 | 38,882 / 28.42s |
| student-top-3 | 504 | 1.360s | 6.127s | 7.121s | 420,688 | 38,888 / 28.50s |

Neural inference itself remained small: roughly 0.52–0.56 ms per call on the fixed matrix. Search continuations, not the network, dominate hybrid latency.

## Agreement and teacher-value regret

| Placement | Direct agreement / regret | Top-2 agreement / regret | Top-3 agreement / regret |
| ---: | ---: | ---: | ---: |
| 1 | 20.0% / 0.119 | 30.0% / 0.133 | 40.0% / 0.112 |
| 2 | 13.6% / 0.186 | 20.5% / 0.126 | 11.4% / 0.156 |
| 3 | 23.5% / 0.114 | 29.4% / 0.110 | 35.3% / 0.080 |
| 4 | 20.0% / 0.094 | 20.0% / 0.099 | 30.0% / 0.090 |
| 5 | 25.0% / 0.064 | 62.5% / 0.042 | 56.2% / 0.017 |
| 6 | 40.0% / 0.130 | 100.0% / 0.000 | 100.0% / 0.000 |
| 7 | 100.0% / 0.000 | 100.0% / 0.000 | 100.0% / 0.000 |
| Overall | 30.8% / 0.117 | 45.2% / 0.084 | 44.5% / 0.081 |

Agreement becomes exact at placements 6 and 7 for top-2 and top-3 because the remaining legal response space is small. Early placement disagreement is substantial, consistent with the ranker's approximately random offline pairwise accuracy.

## Every strategic-group disagreement

`TQ(T)` is pure Teacher v2's value for its selected group; `TQ(H)` is pure Teacher v2's value for the hybrid-selected group. `HQ(T)` and `HQ(H)` are the hybrid search's corresponding values. Full group visits and values for both controllers remain in the raw artifact.

| Mode | Source | P | Actor | Dealer | Teacher group | Hybrid group | TQ(T) | TQ(H) | HQ(T) | HQ(H) | Regret |
| --- | --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| student-direct | `constructive:king-early-nondealer-suit` | 3 | king | queen | 9 | 17 | 0.058 | -0.087 | 0.020 | 0.067 | 0.144 |
| student-direct | `constructive:king-middle-preserve-card` | 5 | king | queen | 8 | 10 | -0.138 | -0.140 | -0.138 | -0.105 | 0.002 |
| student-direct | `constructive:king-middle-suit` | 6 | king | king | 24 | 7 | 0.407 | -0.127 | 0.407 | 0.407 | 0.533 |
| student-direct | `constructive:queen-middle-color-exploit` | 6 | queen | queen | 29 | 18 | 0.200 | 0.100 | 0.200 | 0.200 | 0.100 |
| student-direct | `defensive:king-early-dealer-constructive` | 2 | king | king | 1 | 19 | 0.178 | 0.147 | 0.020 | 0.253 | 0.031 |
| student-direct | `defensive:king-early-nondealer-safe-intersection` | 3 | king | queen | 23 | 3 | 0.253 | 0.093 | 0.191 | 0.407 | 0.160 |
| student-direct | `defensive:king-middle-avoid-suit` | 5 | king | queen | 29 | 9 | -0.080 | -0.200 | -0.267 | -0.033 | 0.160 |
| student-direct | `defensive:king-middle-denial-over-offense` | 5 | king | queen | 4 | 28 | 0.144 | -0.007 | 0.107 | 0.127 | 0.151 |
| student-direct | `defensive:queen-early-dealer-safe-intersection` | 2 | queen | queen | 8 | 24 | -0.033 | -0.058 | -0.107 | 0.049 | 0.024 |
| student-direct | `defensive:queen-early-nondealer-avoid-suit` | 3 | queen | king | 5 | 13 | -0.049 | -0.083 | 0.003 | 0.173 | 0.034 |
| student-direct | `defensive:queen-middle-avoid-color` | 5 | queen | king | 18 | 5 | -0.027 | -0.096 | 0.040 | 0.040 | 0.069 |
| student-direct | `defensive:queen-middle-avoid-vampire-destruction` | 5 | queen | king | 23 | 20 | 0.116 | -0.060 | 0.173 | 0.188 | 0.176 |
| student-direct | `defensive:queen-middle-defensive-vampire` | 6 | queen | queen | 31 | 21 | 0.013 | -0.567 | 0.013 | 0.260 | 0.580 |
| student-direct | `p1-center-only-actor-king-dealer-queen#0` | 1 | king | queen | 9 | 27 | 0.172 | 0.015 | -0.047 | 0.183 | 0.157 |
| student-direct | `p1-center-only-actor-king-dealer-queen#1` | 1 | king | queen | 1 | 17 | 0.183 | 0.023 | 0.043 | 0.213 | 0.159 |
| student-direct | `p1-center-only-actor-king-dealer-queen#2` | 1 | king | queen | 1 | 17 | 0.216 | 0.077 | 0.215 | 0.295 | 0.139 |
| student-direct | `p1-center-only-actor-king-dealer-queen#3` | 1 | king | queen | 1 | 17 | 0.244 | 0.141 | -0.180 | 0.177 | 0.103 |
| student-direct | `p1-center-only-actor-king-dealer-queen#4` | 1 | king | queen | 9 | 27 | 0.156 | 0.097 | 0.113 | 0.169 | 0.059 |
| student-direct | `p1-center-only-actor-queen-dealer-king#1` | 1 | queen | king | 17 | 1 | 0.113 | -0.092 | 0.017 | 0.232 | 0.205 |
| student-direct | `p1-center-only-actor-queen-dealer-king#3` | 1 | queen | king | 17 | 3 | 0.087 | -0.053 | -0.167 | 0.152 | 0.140 |
| student-direct | `p1-center-only-actor-queen-dealer-king#4` | 1 | queen | king | 17 | 1 | 0.047 | -0.184 | 0.067 | 0.119 | 0.231 |
| student-direct | `p2-center-plus-2-actor-king-dealer-king#0` | 2 | king | king | 0 | 8 | 0.278 | 0.000 | 0.018 | 0.177 | 0.278 |
| student-direct | `p2-center-plus-2-actor-king-dealer-king#1` | 2 | king | king | 19 | 30 | 0.183 | 0.084 | -0.049 | 0.049 | 0.200 |
| student-direct | `p2-center-plus-2-actor-king-dealer-king#2` | 2 | king | king | 19 | 6 | 0.260 | -0.029 | -0.113 | 0.109 | 0.289 |
| student-direct | `p2-center-plus-2-actor-king-dealer-king#3` | 2 | king | king | 27 | 8 | 0.253 | -0.117 | 0.022 | 0.084 | 0.370 |
| student-direct | `p2-center-plus-2-actor-king-dealer-king#4` | 2 | king | king | 22 | 27 | 0.309 | -0.260 | 0.171 | 0.178 | 0.569 |
| student-direct | `p2-center-plus-2-actor-queen-dealer-queen#0` | 2 | queen | queen | 14 | 24 | -0.033 | -0.280 | -0.273 | 0.113 | 0.247 |
| student-direct | `p2-center-plus-2-actor-queen-dealer-queen#1` | 2 | queen | queen | 8 | 24 | 0.038 | -0.180 | 0.011 | 0.162 | 0.218 |
| student-direct | `p2-center-plus-2-actor-queen-dealer-queen#2` | 2 | queen | queen | 16 | 0 | 0.044 | -0.167 | -0.213 | 0.069 | 0.211 |
| student-direct | `p2-center-plus-2-actor-queen-dealer-queen#3` | 2 | queen | queen | 11 | 14 | -0.102 | -0.260 | -0.093 | -0.053 | 0.158 |
| student-direct | `p2-center-plus-2-actor-queen-dealer-queen#4` | 2 | queen | queen | 19 | 3 | -0.004 | -0.007 | 0.062 | 0.140 | 0.002 |
| student-direct | `p2-center-plus-4-actor-king-dealer-king#0` | 2 | king | king | 1 | 4 | 0.209 | -0.467 | -0.031 | 0.184 | 0.676 |
| student-direct | `p2-center-plus-4-actor-king-dealer-king#1` | 2 | king | king | 16 | 8 | 0.164 | -0.122 | -0.029 | 0.156 | 0.287 |
| student-direct | `p2-center-plus-4-actor-king-dealer-king#2` | 2 | king | king | 8 | 16 | 0.253 | -0.198 | -0.013 | 0.095 | 0.451 |
| student-direct | `p2-center-plus-4-actor-king-dealer-king#3` | 2 | king | king | 1 | 16 | 0.096 | -0.033 | -0.040 | -0.031 | 0.129 |
| student-direct | `p2-center-plus-4-actor-king-dealer-king#4` | 2 | king | king | 16 | 4 | 0.124 | -0.129 | 0.100 | 0.118 | 0.253 |
| student-direct | `p2-center-plus-4-actor-queen-dealer-queen#0` | 2 | queen | queen | 1 | 16 | 0.049 | 0.011 | -0.216 | 0.091 | 0.038 |
| student-direct | `p2-center-plus-4-actor-queen-dealer-queen#1` | 2 | queen | queen | 1 | 16 | -0.042 | -0.118 | -0.110 | 0.102 | 0.076 |
| student-direct | `p2-center-plus-4-actor-queen-dealer-queen#2` | 2 | queen | queen | 12 | 16 | -0.082 | -0.087 | -0.033 | 0.131 | 0.004 |
| student-direct | `p2-center-plus-6-actor-king-dealer-king#0` | 2 | king | king | 19 | 27 | 0.111 | -0.044 | -0.103 | 0.116 | 0.156 |
| student-direct | `p2-center-plus-6-actor-king-dealer-king#1` | 2 | king | king | 27 | 11 | 0.051 | 0.002 | -0.029 | 0.011 | 0.049 |
| student-direct | `p2-center-plus-6-actor-king-dealer-king#2` | 2 | king | king | 18 | 27 | 0.123 | -0.042 | 0.011 | 0.140 | 0.240 |
| student-direct | `p2-center-plus-6-actor-king-dealer-king#3` | 2 | king | king | 1 | 2 | 0.207 | -0.060 | -0.090 | 0.140 | 0.267 |
| student-direct | `p2-center-plus-6-actor-king-dealer-king#4` | 2 | king | king | 26 | 27 | 0.253 | -0.002 | -0.062 | 0.073 | 0.256 |
| student-direct | `p2-center-plus-6-actor-queen-dealer-queen#0` | 2 | queen | queen | 3 | 1 | -0.044 | -0.162 | -0.200 | 0.136 | 0.118 |
| student-direct | `p2-center-plus-6-actor-queen-dealer-queen#1` | 2 | queen | queen | 25 | 27 | 0.024 | -0.257 | -0.167 | 0.029 | 0.281 |
| student-direct | `p2-center-plus-6-actor-queen-dealer-queen#2` | 2 | queen | queen | 18 | 9 | -0.027 | -0.240 | -0.149 | 0.091 | 0.213 |
| student-direct | `p2-center-plus-6-actor-queen-dealer-queen#3` | 2 | queen | queen | 11 | 27 | 0.087 | -0.113 | -0.093 | 0.096 | 0.200 |
| student-direct | `p2-center-plus-6-actor-queen-dealer-queen#4` | 2 | queen | queen | 9 | 17 | 0.016 | -0.180 | -0.153 | 0.051 | 0.196 |
| student-direct | `p2-center-plus-8-actor-king-dealer-king#0` | 2 | king | king | 23 | 7 | 0.229 | -0.093 | -0.197 | 0.291 | 0.322 |
| student-direct | `p2-center-plus-8-actor-king-dealer-king#2` | 2 | king | king | 9 | 27 | 0.256 | 0.171 | -0.123 | 0.244 | 0.084 |
| student-direct | `p2-center-plus-8-actor-king-dealer-king#4` | 2 | king | king | 1 | 27 | 0.276 | -0.043 | -0.073 | 0.322 | 0.319 |
| student-direct | `p2-center-plus-8-actor-queen-dealer-queen#0` | 2 | queen | queen | 11 | 17 | 0.120 | -0.109 | 0.024 | 0.049 | 0.229 |
| student-direct | `p2-center-plus-8-actor-queen-dealer-queen#1` | 2 | queen | queen | 9 | 17 | 0.051 | -0.178 | -0.193 | -0.056 | 0.229 |
| student-direct | `p2-center-plus-8-actor-queen-dealer-queen#2` | 2 | queen | queen | 7 | 9 | 0.087 | -0.144 | -0.173 | -0.029 | 0.231 |
| student-direct | `p2-center-plus-8-actor-queen-dealer-queen#3` | 2 | queen | queen | 11 | 9 | 0.016 | -0.127 | -0.080 | 0.102 | 0.142 |
| student-direct | `p2-center-plus-8-actor-queen-dealer-queen#4` | 2 | queen | queen | 15 | 7 | -0.067 | -0.164 | -0.177 | 0.158 | 0.098 |
| student-direct | `p3-corner-growth-control-actor-king-dealer-queen#0` | 3 | king | queen | 8 | 10 | 0.229 | 0.160 | 0.040 | 0.262 | 0.069 |
| student-direct | `p3-corner-growth-control-actor-king-dealer-queen#1` | 3 | king | queen | 14 | 16 | 0.216 | -0.010 | -0.150 | 0.273 | 0.226 |
| student-direct | `p3-corner-growth-control-actor-king-dealer-queen#2` | 3 | king | queen | 16 | 12 | 0.224 | 0.064 | 0.100 | 0.271 | 0.160 |
| student-direct | `p3-corner-growth-control-actor-king-dealer-queen#4` | 3 | king | queen | 18 | 16 | 0.258 | 0.229 | 0.017 | 0.251 | 0.029 |
| student-direct | `p3-corner-growth-control-actor-queen-dealer-king#0` | 3 | queen | king | 22 | 20 | 0.036 | -0.117 | -0.113 | 0.142 | 0.152 |
| student-direct | `p3-corner-growth-control-actor-queen-dealer-king#1` | 3 | queen | king | 16 | 21 | -0.011 | -0.083 | 0.033 | 0.231 | 0.072 |
| student-direct | `p3-corner-growth-control-actor-queen-dealer-king#2` | 3 | queen | king | 26 | 21 | -0.076 | -0.117 | 0.007 | 0.309 | 0.041 |
| student-direct | `p3-corner-growth-control-actor-queen-dealer-king#3` | 3 | queen | king | 30 | 18 | 0.011 | -0.130 | -0.007 | 0.173 | 0.141 |
| student-direct | `p3-corner-growth-control-actor-queen-dealer-king#4` | 3 | queen | king | 30 | 8 | 0.089 | -0.170 | 0.070 | 0.116 | 0.259 |
| student-direct | `p3-horizontal-line-actor-king-dealer-queen#1` | 3 | king | queen | 9 | 24 | 0.120 | 0.032 | 0.102 | 0.148 | 0.088 |
| student-direct | `p3-horizontal-line-actor-king-dealer-queen#3` | 3 | king | queen | 26 | 9 | 0.093 | -0.118 | 0.040 | 0.135 | 0.211 |
| student-direct | `p3-horizontal-line-actor-king-dealer-queen#4` | 3 | king | queen | 26 | 9 | 0.040 | -0.087 | 0.040 | 0.187 | 0.127 |
| student-direct | `p3-horizontal-line-actor-queen-dealer-king#0` | 3 | queen | king | 16 | 24 | 0.033 | -0.196 | -0.049 | 0.112 | 0.229 |
| student-direct | `p3-horizontal-line-actor-queen-dealer-king#1` | 3 | queen | king | 10 | 26 | -0.032 | -0.342 | 0.053 | 0.097 | 0.311 |
| student-direct | `p3-horizontal-line-actor-queen-dealer-king#2` | 3 | queen | king | 10 | 18 | 0.049 | -0.176 | -0.018 | 0.108 | 0.225 |
| student-direct | `p3-horizontal-line-actor-queen-dealer-king#4` | 3 | queen | king | 18 | 24 | 0.027 | -0.133 | -0.091 | 0.237 | 0.160 |
| student-direct | `p3-vertical-line-actor-king-dealer-queen#1` | 3 | king | queen | 16 | 27 | 0.280 | 0.144 | 0.288 | 0.302 | 0.136 |
| student-direct | `p3-vertical-line-actor-king-dealer-queen#2` | 3 | king | queen | 8 | 16 | 0.285 | 0.173 | 0.200 | 0.340 | 0.112 |
| student-direct | `p3-vertical-line-actor-queen-dealer-king#0` | 3 | queen | king | 19 | 29 | 0.116 | -0.067 | -0.072 | 0.192 | 0.183 |
| student-direct | `p3-vertical-line-actor-queen-dealer-king#1` | 3 | queen | king | 21 | 11 | 0.165 | 0.132 | 0.064 | 0.240 | 0.033 |
| student-direct | `p3-vertical-line-actor-queen-dealer-king#2` | 3 | queen | king | 8 | 27 | 0.135 | -0.109 | 0.100 | 0.135 | 0.244 |
| student-direct | `p3-vertical-line-actor-queen-dealer-king#3` | 3 | queen | king | 19 | 11 | 0.072 | 0.057 | 0.058 | 0.251 | 0.015 |
| student-direct | `p3-vertical-line-actor-queen-dealer-king#4` | 3 | queen | king | 21 | 27 | 0.210 | -0.104 | 0.080 | 0.183 | 0.314 |
| student-direct | `p4-continued-corner-growth-actor-king-dealer-king#1` | 4 | king | king | 21 | 14 | 0.202 | 0.064 | -0.157 | 0.056 | 0.138 |
| student-direct | `p4-continued-corner-growth-actor-king-dealer-king#2` | 4 | king | king | 14 | 10 | 0.013 | -0.044 | -0.036 | 0.191 | 0.058 |
| student-direct | `p4-continued-corner-growth-actor-king-dealer-king#4` | 4 | king | king | 13 | 14 | 0.056 | -0.097 | -0.002 | 0.113 | 0.152 |
| student-direct | `p4-continued-corner-growth-actor-queen-dealer-queen#0` | 4 | queen | queen | 14 | 26 | -0.091 | -0.131 | -0.102 | 0.053 | 0.040 |
| student-direct | `p4-continued-corner-growth-actor-queen-dealer-queen#1` | 4 | queen | queen | 13 | 26 | -0.027 | -0.064 | 0.004 | 0.031 | 0.038 |
| student-direct | `p4-continued-corner-growth-actor-queen-dealer-queen#2` | 4 | queen | queen | 26 | 22 | -0.051 | -0.302 | 0.004 | 0.016 | 0.251 |
| student-direct | `p4-continued-corner-growth-actor-queen-dealer-queen#3` | 4 | queen | queen | 22 | 26 | -0.096 | -0.136 | -0.107 | 0.040 | 0.040 |
| student-direct | `p4-continued-corner-growth-actor-queen-dealer-queen#4` | 4 | queen | queen | 13 | 26 | -0.071 | -0.297 | -0.100 | 0.053 | 0.226 |
| student-direct | `p5-continued-corner-growth-actor-king-dealer-queen#0` | 5 | king | queen | 30 | 28 | 0.269 | 0.252 | 0.271 | 0.276 | 0.017 |
| student-direct | `p5-continued-corner-growth-actor-king-dealer-queen#2` | 5 | king | queen | 18 | 28 | 0.287 | 0.250 | 0.267 | 0.303 | 0.037 |
| student-direct | `p5-continued-corner-growth-actor-king-dealer-queen#3` | 5 | king | queen | 18 | 28 | 0.190 | 0.126 | 0.259 | 0.268 | 0.064 |
| student-direct | `p5-continued-corner-growth-actor-king-dealer-queen#4` | 5 | king | queen | 18 | 28 | 0.240 | 0.180 | 0.240 | 0.240 | 0.060 |
| student-direct | `p5-continued-corner-growth-actor-queen-dealer-king#0` | 5 | queen | king | 30 | 18 | -0.041 | -0.261 | -0.048 | -0.001 | 0.220 |
| student-direct | `p5-continued-corner-growth-actor-queen-dealer-king#3` | 5 | queen | king | 30 | 18 | 0.022 | 0.018 | 0.037 | 0.038 | 0.004 |
| student-direct | `p5-continued-corner-growth-actor-queen-dealer-king#4` | 5 | queen | king | 30 | 18 | -0.031 | -0.090 | -0.040 | 0.003 | 0.059 |
| student-direct | `p6-continued-corner-growth-actor-king-dealer-king#0` | 6 | king | king | 18 | 28 | 0.001 | -0.075 | -0.052 | 0.017 | 0.076 |
| student-direct | `p6-continued-corner-growth-actor-king-dealer-king#1` | 6 | king | king | 31 | 18 | 0.020 | -0.141 | -0.180 | 0.044 | 0.161 |
| student-direct | `p6-continued-corner-growth-actor-king-dealer-king#4` | 6 | king | king | 18 | 20 | -0.002 | -0.198 | -0.103 | 0.106 | 0.196 |
| student-direct | `p6-continued-corner-growth-actor-queen-dealer-queen#1` | 6 | queen | queen | 26 | 20 | 0.022 | -0.049 | 0.029 | 0.033 | 0.072 |
| student-direct | `p6-continued-corner-growth-actor-queen-dealer-queen#2` | 6 | queen | queen | 26 | 23 | 0.022 | -0.076 | 0.009 | 0.022 | 0.098 |
| student-direct | `p6-continued-corner-growth-actor-queen-dealer-queen#4` | 6 | queen | queen | 26 | 23 | 0.029 | -0.101 | 0.031 | 0.053 | 0.130 |
| student-top-2 | `constructive:king-middle-preserve-card` | 5 | king | queen | 8 | 10 | -0.138 | -0.140 | -0.138 | -0.105 | 0.002 |
| student-top-2 | `constructive:queen-early-dealer-suit` | 2 | queen | queen | 11 | 3 | 0.402 | 0.353 | 0.208 | 0.355 | 0.049 |
| student-top-2 | `defensive:king-early-dealer-constructive` | 2 | king | king | 1 | 19 | 0.178 | 0.147 | -0.071 | 0.171 | 0.031 |
| student-top-2 | `defensive:king-early-nondealer-safe-intersection` | 3 | king | queen | 23 | 15 | 0.253 | 0.131 | 0.142 | 0.204 | 0.122 |
| student-top-2 | `defensive:king-middle-avoid-suit` | 5 | king | queen | 29 | 9 | -0.080 | -0.200 | -0.225 | -0.000 | 0.160 |
| student-top-2 | `defensive:queen-early-dealer-safe-intersection` | 2 | queen | queen | 8 | 24 | -0.033 | -0.058 | -0.020 | -0.013 | 0.024 |
| student-top-2 | `defensive:queen-early-nondealer-avoid-suit` | 3 | queen | king | 5 | 21 | -0.049 | -0.400 | -0.067 | 0.133 | 0.351 |
| student-top-2 | `defensive:queen-middle-avoid-vampire-destruction` | 5 | queen | king | 23 | 0 | 0.116 | 0.027 | 0.180 | 0.185 | 0.089 |
| student-top-2 | `p1-center-only-actor-king-dealer-queen#0` | 1 | king | queen | 9 | 1 | 0.172 | 0.000 | 0.133 | 0.213 | 0.172 |
| student-top-2 | `p1-center-only-actor-king-dealer-queen#2` | 1 | king | queen | 1 | 17 | 0.216 | 0.077 | 0.003 | 0.180 | 0.139 |
| student-top-2 | `p1-center-only-actor-king-dealer-queen#3` | 1 | king | queen | 1 | 9 | 0.244 | -0.030 | -0.173 | 0.116 | 0.274 |
| student-top-2 | `p1-center-only-actor-queen-dealer-king#1` | 1 | queen | king | 17 | 1 | 0.113 | -0.092 | -0.080 | 0.101 | 0.205 |
| student-top-2 | `p1-center-only-actor-queen-dealer-king#2` | 1 | queen | king | 17 | 11 | 0.137 | 0.067 | -0.062 | 0.039 | 0.071 |
| student-top-2 | `p1-center-only-actor-queen-dealer-king#3` | 1 | queen | king | 17 | 11 | 0.087 | -0.082 | 0.045 | 0.075 | 0.168 |
| student-top-2 | `p1-center-only-actor-queen-dealer-king#4` | 1 | queen | king | 17 | 9 | 0.047 | -0.249 | 0.082 | 0.075 | 0.296 |
| student-top-2 | `p2-center-plus-2-actor-king-dealer-king#0` | 2 | king | king | 0 | 6 | 0.278 | 0.162 | -0.027 | 0.167 | 0.116 |
| student-top-2 | `p2-center-plus-2-actor-king-dealer-king#1` | 2 | king | king | 19 | 16 | 0.183 | -0.070 | -0.111 | 0.047 | 0.354 |
| student-top-2 | `p2-center-plus-2-actor-king-dealer-king#2` | 2 | king | king | 19 | 14 | 0.260 | 0.229 | -0.064 | 0.216 | 0.031 |
| student-top-2 | `p2-center-plus-2-actor-king-dealer-king#3` | 2 | king | king | 27 | 6 | 0.253 | 0.122 | -0.091 | 0.224 | 0.131 |
| student-top-2 | `p2-center-plus-2-actor-king-dealer-king#4` | 2 | king | king | 22 | 6 | 0.309 | 0.260 | -0.150 | 0.140 | 0.049 |
| student-top-2 | `p2-center-plus-2-actor-queen-dealer-queen#0` | 2 | queen | queen | 14 | 22 | -0.033 | -0.377 | -0.024 | 0.082 | 0.343 |
| student-top-2 | `p2-center-plus-2-actor-queen-dealer-queen#1` | 2 | queen | queen | 8 | 22 | 0.038 | 0.024 | -0.040 | 0.051 | 0.013 |
| student-top-2 | `p2-center-plus-2-actor-queen-dealer-queen#2` | 2 | queen | queen | 16 | 22 | 0.044 | -0.044 | -0.213 | 0.024 | 0.089 |
| student-top-2 | `p2-center-plus-2-actor-queen-dealer-queen#3` | 2 | queen | queen | 11 | 22 | -0.102 | -0.273 | -0.250 | -0.084 | 0.171 |
| student-top-2 | `p2-center-plus-2-actor-queen-dealer-queen#4` | 2 | queen | queen | 19 | 14 | -0.004 | -0.287 | -0.082 | -0.033 | 0.282 |
| student-top-2 | `p2-center-plus-4-actor-king-dealer-king#0` | 2 | king | king | 1 | 24 | 0.209 | 0.078 | -0.031 | 0.208 | 0.131 |
| student-top-2 | `p2-center-plus-4-actor-king-dealer-king#2` | 2 | king | king | 8 | 1 | 0.253 | 0.087 | 0.091 | 0.109 | 0.167 |
| student-top-2 | `p2-center-plus-4-actor-king-dealer-king#3` | 2 | king | king | 1 | 16 | 0.096 | -0.033 | -0.067 | 0.098 | 0.129 |
| student-top-2 | `p2-center-plus-4-actor-king-dealer-king#4` | 2 | king | king | 16 | 9 | 0.124 | -0.193 | 0.000 | 0.073 | 0.318 |
| student-top-2 | `p2-center-plus-4-actor-queen-dealer-queen#1` | 2 | queen | queen | 1 | 4 | -0.042 | -0.180 | -0.087 | 0.011 | 0.138 |
| student-top-2 | `p2-center-plus-4-actor-queen-dealer-queen#2` | 2 | queen | queen | 12 | 17 | -0.082 | -0.189 | -0.160 | 0.047 | 0.107 |
| student-top-2 | `p2-center-plus-4-actor-queen-dealer-queen#3` | 2 | queen | queen | 4 | 1 | -0.076 | -0.164 | -0.111 | -0.040 | 0.089 |
| student-top-2 | `p2-center-plus-4-actor-queen-dealer-queen#4` | 2 | queen | queen | 8 | 4 | -0.004 | -0.031 | -0.058 | -0.013 | 0.027 |
| student-top-2 | `p2-center-plus-6-actor-king-dealer-king#0` | 2 | king | king | 19 | 27 | 0.111 | -0.044 | -0.073 | 0.087 | 0.156 |
| student-top-2 | `p2-center-plus-6-actor-king-dealer-king#1` | 2 | king | king | 27 | 11 | 0.051 | 0.002 | -0.060 | 0.222 | 0.049 |
| student-top-2 | `p2-center-plus-6-actor-king-dealer-king#2` | 2 | king | king | 18 | 27 | 0.123 | -0.042 | 0.080 | 0.116 | 0.240 |
| student-top-2 | `p2-center-plus-6-actor-king-dealer-king#4` | 2 | king | king | 26 | 11 | 0.253 | 0.000 | -0.087 | 0.133 | 0.253 |
| student-top-2 | `p2-center-plus-6-actor-queen-dealer-queen#1` | 2 | queen | queen | 25 | 19 | 0.024 | -0.009 | -0.073 | -0.042 | 0.033 |
| student-top-2 | `p2-center-plus-6-actor-queen-dealer-queen#2` | 2 | queen | queen | 18 | 17 | -0.027 | -0.158 | -0.133 | 0.053 | 0.131 |
| student-top-2 | `p2-center-plus-6-actor-queen-dealer-queen#3` | 2 | queen | queen | 11 | 17 | 0.087 | -0.216 | -0.047 | 0.018 | 0.302 |
| student-top-2 | `p2-center-plus-6-actor-queen-dealer-queen#4` | 2 | queen | queen | 9 | 10 | 0.016 | -0.173 | -0.160 | 0.011 | 0.189 |
| student-top-2 | `p2-center-plus-8-actor-king-dealer-king#1` | 2 | king | king | 7 | 19 | 0.269 | 0.100 | 0.076 | 0.080 | 0.169 |
| student-top-2 | `p2-center-plus-8-actor-king-dealer-king#2` | 2 | king | king | 9 | 7 | 0.256 | -0.097 | 0.044 | 0.116 | 0.352 |
| student-top-2 | `p2-center-plus-8-actor-king-dealer-king#4` | 2 | king | king | 1 | 9 | 0.276 | 0.113 | -0.011 | 0.153 | 0.162 |
| student-top-2 | `p2-center-plus-8-actor-queen-dealer-queen#0` | 2 | queen | queen | 11 | 15 | 0.120 | -0.329 | 0.069 | 0.109 | 0.449 |
| student-top-2 | `p2-center-plus-8-actor-queen-dealer-queen#1` | 2 | queen | queen | 9 | 31 | 0.051 | -0.104 | -0.151 | 0.038 | 0.156 |
| student-top-2 | `p2-center-plus-8-actor-queen-dealer-queen#4` | 2 | queen | queen | 15 | 7 | -0.067 | -0.164 | -0.013 | 0.078 | 0.098 |
| student-top-2 | `p3-corner-growth-control-actor-king-dealer-queen#0` | 3 | king | queen | 8 | 16 | 0.229 | 0.067 | 0.097 | 0.280 | 0.162 |
| student-top-2 | `p3-corner-growth-control-actor-king-dealer-queen#1` | 3 | king | queen | 14 | 16 | 0.216 | -0.010 | 0.023 | 0.291 | 0.226 |
| student-top-2 | `p3-corner-growth-control-actor-king-dealer-queen#2` | 3 | king | queen | 16 | 8 | 0.224 | 0.077 | 0.093 | 0.278 | 0.148 |
| student-top-2 | `p3-corner-growth-control-actor-king-dealer-queen#3` | 3 | king | queen | 8 | 30 | 0.316 | 0.133 | 0.258 | 0.280 | 0.182 |
| student-top-2 | `p3-corner-growth-control-actor-king-dealer-queen#4` | 3 | king | queen | 18 | 10 | 0.258 | 0.173 | 0.187 | 0.231 | 0.084 |
| student-top-2 | `p3-corner-growth-control-actor-queen-dealer-king#0` | 3 | queen | king | 22 | 29 | 0.036 | -0.053 | 0.027 | 0.147 | 0.089 |
| student-top-2 | `p3-corner-growth-control-actor-queen-dealer-king#1` | 3 | queen | king | 16 | 8 | -0.011 | -0.070 | 0.027 | 0.116 | 0.059 |
| student-top-2 | `p3-corner-growth-control-actor-queen-dealer-king#2` | 3 | queen | king | 26 | 29 | -0.076 | -0.123 | -0.387 | 0.136 | 0.048 |
| student-top-2 | `p3-corner-growth-control-actor-queen-dealer-king#3` | 3 | queen | king | 30 | 12 | 0.011 | -0.073 | -0.140 | 0.084 | 0.084 |
| student-top-2 | `p3-corner-growth-control-actor-queen-dealer-king#4` | 3 | queen | king | 30 | 13 | 0.089 | -0.027 | -0.040 | 0.049 | 0.116 |
| student-top-2 | `p3-horizontal-line-actor-king-dealer-queen#0` | 3 | king | queen | 17 | 24 | 0.052 | -0.043 | -0.002 | 0.100 | 0.095 |
| student-top-2 | `p3-horizontal-line-actor-king-dealer-queen#3` | 3 | king | queen | 26 | 17 | 0.093 | -0.127 | 0.035 | 0.087 | 0.220 |
| student-top-2 | `p3-horizontal-line-actor-queen-dealer-king#1` | 3 | queen | king | 10 | 18 | -0.032 | -0.240 | -0.018 | 0.070 | 0.208 |
| student-top-2 | `p3-horizontal-line-actor-queen-dealer-king#2` | 3 | queen | king | 10 | 18 | 0.049 | -0.176 | -0.052 | 0.067 | 0.225 |
| student-top-2 | `p3-vertical-line-actor-king-dealer-queen#0` | 3 | king | queen | 11 | 16 | 0.245 | 0.155 | 0.056 | 0.238 | 0.090 |
| student-top-2 | `p3-vertical-line-actor-king-dealer-queen#1` | 3 | king | queen | 16 | 29 | 0.280 | 0.109 | 0.317 | 0.327 | 0.171 |
| student-top-2 | `p3-vertical-line-actor-king-dealer-queen#2` | 3 | king | queen | 8 | 21 | 0.285 | -0.124 | 0.220 | 0.283 | 0.410 |
| student-top-2 | `p3-vertical-line-actor-king-dealer-queen#4` | 3 | king | queen | 27 | 8 | 0.235 | 0.187 | 0.145 | 0.300 | 0.048 |
| student-top-2 | `p3-vertical-line-actor-queen-dealer-king#0` | 3 | queen | king | 19 | 21 | 0.116 | -0.007 | 0.188 | 0.190 | 0.123 |
| student-top-2 | `p3-vertical-line-actor-queen-dealer-king#2` | 3 | queen | king | 8 | 21 | 0.135 | 0.033 | 0.163 | 0.185 | 0.102 |
| student-top-2 | `p3-vertical-line-actor-queen-dealer-king#3` | 3 | queen | king | 19 | 21 | 0.072 | -0.062 | 0.085 | 0.220 | 0.134 |
| student-top-2 | `p3-vertical-line-actor-queen-dealer-king#4` | 3 | queen | king | 21 | 29 | 0.210 | -0.040 | 0.093 | 0.293 | 0.250 |
| student-top-2 | `p4-continued-corner-growth-actor-king-dealer-king#0` | 4 | king | king | 29 | 26 | 0.111 | -0.133 | -0.009 | 0.080 | 0.244 |
| student-top-2 | `p4-continued-corner-growth-actor-king-dealer-king#1` | 4 | king | king | 21 | 22 | 0.202 | -0.038 | 0.020 | 0.104 | 0.240 |
| student-top-2 | `p4-continued-corner-growth-actor-king-dealer-king#2` | 4 | king | king | 14 | 10 | 0.013 | -0.044 | -0.080 | 0.176 | 0.058 |
| student-top-2 | `p4-continued-corner-growth-actor-queen-dealer-queen#0` | 4 | queen | queen | 14 | 26 | -0.091 | -0.131 | -0.187 | 0.029 | 0.040 |
| student-top-2 | `p4-continued-corner-growth-actor-queen-dealer-queen#1` | 4 | queen | queen | 13 | 26 | -0.027 | -0.064 | -0.027 | 0.022 | 0.038 |
| student-top-2 | `p4-continued-corner-growth-actor-queen-dealer-queen#2` | 4 | queen | queen | 26 | 20 | -0.051 | -0.382 | -0.169 | -0.027 | 0.331 |
| student-top-2 | `p4-continued-corner-growth-actor-queen-dealer-queen#3` | 4 | queen | queen | 22 | 26 | -0.096 | -0.136 | -0.431 | 0.073 | 0.040 |
| student-top-2 | `p4-continued-corner-growth-actor-queen-dealer-queen#4` | 4 | queen | queen | 13 | 28 | -0.071 | -0.071 | -0.071 | -0.044 | 0.000 |
| student-top-2 | `p5-continued-corner-growth-actor-king-dealer-queen#0` | 5 | king | queen | 30 | 18 | 0.269 | 0.252 | 0.193 | 0.263 | 0.017 |
| student-top-2 | `p5-continued-corner-growth-actor-queen-dealer-king#0` | 5 | queen | king | 30 | 18 | -0.041 | -0.261 | -0.085 | -0.020 | 0.220 |
| student-top-2 | `p5-continued-corner-growth-actor-queen-dealer-king#1` | 5 | queen | king | 30 | 18 | -0.037 | -0.223 | -0.073 | -0.050 | 0.186 |
| student-top-3 | `constructive:king-middle-preserve-card` | 5 | king | queen | 8 | 10 | -0.138 | -0.140 | -0.138 | -0.105 | 0.002 |
| student-top-3 | `constructive:queen-early-dealer-suit` | 2 | queen | queen | 11 | 3 | 0.402 | 0.353 | 0.208 | 0.343 | 0.049 |
| student-top-3 | `defensive:king-early-dealer-constructive` | 2 | king | king | 1 | 19 | 0.178 | 0.147 | -0.153 | 0.120 | 0.031 |
| student-top-3 | `defensive:king-early-nondealer-safe-intersection` | 3 | king | queen | 23 | 4 | 0.253 | 0.178 | 0.142 | 0.178 | 0.076 |
| student-top-3 | `defensive:king-middle-avoid-suit` | 5 | king | queen | 29 | 25 | -0.080 | -0.040 | -0.225 | -0.040 | 0.000 |
| student-top-3 | `defensive:queen-early-dealer-safe-intersection` | 2 | queen | queen | 8 | 1 | -0.033 | -0.283 | -0.042 | -0.013 | 0.250 |
| student-top-3 | `defensive:queen-early-nondealer-avoid-suit` | 3 | queen | king | 5 | 15 | -0.049 | -0.173 | -0.067 | 0.142 | 0.124 |
| student-top-3 | `defensive:queen-middle-avoid-vampire-destruction` | 5 | queen | king | 23 | 0 | 0.116 | 0.027 | 0.180 | 0.185 | 0.089 |
| student-top-3 | `p1-center-only-actor-king-dealer-queen#0` | 1 | king | queen | 9 | 17 | 0.172 | 0.067 | 0.075 | 0.180 | 0.105 |
| student-top-3 | `p1-center-only-actor-king-dealer-queen#2` | 1 | king | queen | 1 | 11 | 0.216 | 0.073 | 0.100 | 0.216 | 0.143 |
| student-top-3 | `p1-center-only-actor-king-dealer-queen#3` | 1 | king | queen | 1 | 9 | 0.244 | -0.030 | 0.003 | 0.151 | 0.274 |
| student-top-3 | `p1-center-only-actor-queen-dealer-king#0` | 1 | queen | king | 11 | 19 | 0.056 | -0.238 | -0.028 | 0.027 | 0.294 |
| student-top-3 | `p1-center-only-actor-queen-dealer-king#1` | 1 | queen | king | 17 | 1 | 0.113 | -0.092 | -0.088 | 0.067 | 0.205 |
| student-top-3 | `p1-center-only-actor-queen-dealer-king#4` | 1 | queen | king | 17 | 3 | 0.047 | -0.055 | -0.278 | 0.071 | 0.102 |
| student-top-3 | `p2-center-plus-2-actor-king-dealer-king#0` | 2 | king | king | 0 | 3 | 0.278 | 0.064 | 0.004 | 0.140 | 0.213 |
| student-top-3 | `p2-center-plus-2-actor-king-dealer-king#1` | 2 | king | king | 19 | 22 | 0.183 | 0.116 | 0.011 | 0.218 | 0.169 |
| student-top-3 | `p2-center-plus-2-actor-king-dealer-king#2` | 2 | king | king | 19 | 14 | 0.260 | 0.229 | -0.020 | 0.162 | 0.031 |
| student-top-3 | `p2-center-plus-2-actor-king-dealer-king#3` | 2 | king | king | 27 | 6 | 0.253 | 0.122 | -0.077 | 0.158 | 0.131 |
| student-top-3 | `p2-center-plus-2-actor-king-dealer-king#4` | 2 | king | king | 22 | 6 | 0.309 | 0.260 | -0.033 | 0.104 | 0.049 |
| student-top-3 | `p2-center-plus-2-actor-queen-dealer-queen#0` | 2 | queen | queen | 14 | 19 | -0.033 | -0.176 | -0.177 | -0.022 | 0.142 |
| student-top-3 | `p2-center-plus-2-actor-queen-dealer-queen#1` | 2 | queen | queen | 8 | 22 | 0.038 | 0.024 | -0.343 | 0.089 | 0.013 |
| student-top-3 | `p2-center-plus-2-actor-queen-dealer-queen#2` | 2 | queen | queen | 16 | 19 | 0.044 | -0.013 | -0.353 | 0.093 | 0.058 |
| student-top-3 | `p2-center-plus-2-actor-queen-dealer-queen#3` | 2 | queen | queen | 11 | 14 | -0.102 | -0.260 | -0.040 | -0.033 | 0.158 |
| student-top-3 | `p2-center-plus-2-actor-queen-dealer-queen#4` | 2 | queen | queen | 19 | 14 | -0.004 | -0.287 | -0.233 | 0.049 | 0.282 |
| student-top-3 | `p2-center-plus-4-actor-king-dealer-king#0` | 2 | king | king | 1 | 4 | 0.209 | -0.467 | 0.044 | 0.064 | 0.676 |
| student-top-3 | `p2-center-plus-4-actor-king-dealer-king#1` | 2 | king | king | 16 | 9 | 0.164 | -0.080 | -0.087 | 0.142 | 0.244 |
| student-top-3 | `p2-center-plus-4-actor-king-dealer-king#2` | 2 | king | king | 8 | 1 | 0.253 | 0.087 | -0.011 | 0.104 | 0.167 |
| student-top-3 | `p2-center-plus-4-actor-king-dealer-king#3` | 2 | king | king | 1 | 16 | 0.096 | -0.033 | 0.100 | 0.203 | 0.129 |
| student-top-3 | `p2-center-plus-4-actor-king-dealer-king#4` | 2 | king | king | 16 | 8 | 0.124 | -0.223 | -0.042 | 0.222 | 0.348 |
| student-top-3 | `p2-center-plus-4-actor-queen-dealer-queen#0` | 2 | queen | queen | 1 | 8 | 0.049 | -0.129 | 0.007 | 0.049 | 0.178 |
| student-top-3 | `p2-center-plus-4-actor-queen-dealer-queen#1` | 2 | queen | queen | 1 | 8 | -0.042 | -0.084 | -0.071 | -0.029 | 0.042 |
| student-top-3 | `p2-center-plus-4-actor-queen-dealer-queen#2` | 2 | queen | queen | 12 | 4 | -0.082 | -0.136 | -0.147 | -0.004 | 0.053 |
| student-top-3 | `p2-center-plus-4-actor-queen-dealer-queen#3` | 2 | queen | queen | 4 | 1 | -0.076 | -0.164 | -0.111 | -0.040 | 0.089 |
| student-top-3 | `p2-center-plus-4-actor-queen-dealer-queen#4` | 2 | queen | queen | 8 | 12 | -0.004 | -0.367 | -0.082 | -0.022 | 0.362 |
| student-top-3 | `p2-center-plus-6-actor-king-dealer-king#0` | 2 | king | king | 19 | 10 | 0.111 | -0.143 | 0.064 | 0.067 | 0.254 |
| student-top-3 | `p2-center-plus-6-actor-king-dealer-king#1` | 2 | king | king | 27 | 1 | 0.051 | -0.102 | -0.011 | 0.147 | 0.153 |
| student-top-3 | `p2-center-plus-6-actor-king-dealer-king#2` | 2 | king | king | 18 | 10 | 0.123 | 0.198 | -0.087 | 0.220 | 0.000 |
| student-top-3 | `p2-center-plus-6-actor-king-dealer-king#3` | 2 | king | king | 1 | 26 | 0.207 | -0.031 | 0.169 | 0.213 | 0.238 |
| student-top-3 | `p2-center-plus-6-actor-queen-dealer-queen#0` | 2 | queen | queen | 3 | 19 | -0.044 | -0.170 | -0.127 | -0.002 | 0.126 |
| student-top-3 | `p2-center-plus-6-actor-queen-dealer-queen#2` | 2 | queen | queen | 18 | 3 | -0.027 | -0.087 | -0.098 | 0.007 | 0.060 |
| student-top-3 | `p2-center-plus-6-actor-queen-dealer-queen#3` | 2 | queen | queen | 11 | 9 | 0.087 | -0.056 | -0.056 | 0.042 | 0.142 |
| student-top-3 | `p2-center-plus-6-actor-queen-dealer-queen#4` | 2 | queen | queen | 9 | 1 | 0.016 | -0.138 | -0.163 | 0.024 | 0.153 |
| student-top-3 | `p2-center-plus-8-actor-king-dealer-king#0` | 2 | king | king | 23 | 3 | 0.229 | -0.044 | 0.111 | 0.167 | 0.273 |
| student-top-3 | `p2-center-plus-8-actor-king-dealer-king#1` | 2 | king | king | 7 | 9 | 0.269 | 0.131 | 0.124 | 0.249 | 0.138 |
| student-top-3 | `p2-center-plus-8-actor-king-dealer-king#4` | 2 | king | king | 1 | 27 | 0.276 | -0.043 | 0.089 | 0.324 | 0.319 |
| student-top-3 | `p2-center-plus-8-actor-queen-dealer-queen#0` | 2 | queen | queen | 11 | 3 | 0.120 | -0.200 | -0.031 | 0.122 | 0.320 |
| student-top-3 | `p2-center-plus-8-actor-queen-dealer-queen#1` | 2 | queen | queen | 9 | 3 | 0.051 | -0.190 | -0.243 | -0.004 | 0.241 |
| student-top-3 | `p2-center-plus-8-actor-queen-dealer-queen#2` | 2 | queen | queen | 7 | 9 | 0.087 | -0.144 | -0.150 | -0.036 | 0.231 |
| student-top-3 | `p2-center-plus-8-actor-queen-dealer-queen#3` | 2 | queen | queen | 11 | 17 | 0.016 | -0.200 | -0.058 | -0.029 | 0.216 |
| student-top-3 | `p2-center-plus-8-actor-queen-dealer-queen#4` | 2 | queen | queen | 15 | 17 | -0.067 | -0.171 | -0.157 | 0.067 | 0.104 |
| student-top-3 | `p3-corner-growth-control-actor-king-dealer-queen#0` | 3 | king | queen | 8 | 10 | 0.229 | 0.160 | 0.130 | 0.289 | 0.069 |
| student-top-3 | `p3-corner-growth-control-actor-king-dealer-queen#1` | 3 | king | queen | 14 | 8 | 0.216 | 0.070 | 0.182 | 0.262 | 0.146 |
| student-top-3 | `p3-corner-growth-control-actor-king-dealer-queen#2` | 3 | king | queen | 16 | 8 | 0.224 | 0.077 | 0.077 | 0.209 | 0.148 |
| student-top-3 | `p3-corner-growth-control-actor-queen-dealer-king#0` | 3 | queen | king | 22 | 21 | 0.036 | -0.027 | -0.017 | 0.100 | 0.062 |
| student-top-3 | `p3-corner-growth-control-actor-queen-dealer-king#1` | 3 | queen | king | 16 | 8 | -0.011 | -0.070 | 0.020 | 0.162 | 0.059 |
| student-top-3 | `p3-corner-growth-control-actor-queen-dealer-king#2` | 3 | queen | king | 26 | 21 | -0.076 | -0.117 | -0.100 | 0.093 | 0.041 |
| student-top-3 | `p3-corner-growth-control-actor-queen-dealer-king#4` | 3 | queen | king | 30 | 12 | 0.089 | -0.077 | -0.040 | 0.102 | 0.166 |
| student-top-3 | `p3-horizontal-line-actor-king-dealer-queen#1` | 3 | king | queen | 9 | 26 | 0.120 | -0.089 | 0.048 | 0.055 | 0.209 |
| student-top-3 | `p3-horizontal-line-actor-king-dealer-queen#2` | 3 | king | queen | 9 | 26 | 0.088 | -0.063 | -0.018 | 0.097 | 0.152 |
| student-top-3 | `p3-horizontal-line-actor-king-dealer-queen#3` | 3 | king | queen | 26 | 17 | 0.093 | -0.127 | -0.276 | 0.037 | 0.220 |
| student-top-3 | `p3-horizontal-line-actor-king-dealer-queen#4` | 3 | king | queen | 26 | 17 | 0.040 | -0.122 | 0.028 | 0.140 | 0.162 |
| student-top-3 | `p3-horizontal-line-actor-queen-dealer-king#1` | 3 | queen | king | 10 | 8 | -0.032 | -0.102 | -0.073 | 0.025 | 0.070 |
| student-top-3 | `p3-horizontal-line-actor-queen-dealer-king#2` | 3 | queen | king | 10 | 24 | 0.049 | -0.171 | -0.008 | 0.007 | 0.220 |
| student-top-3 | `p3-horizontal-line-actor-queen-dealer-king#4` | 3 | queen | king | 18 | 26 | 0.027 | -0.142 | -0.056 | 0.063 | 0.169 |
| student-top-3 | `p3-vertical-line-actor-king-dealer-queen#0` | 3 | king | queen | 11 | 13 | 0.245 | 0.088 | 0.104 | 0.295 | 0.157 |
| student-top-3 | `p3-vertical-line-actor-king-dealer-queen#4` | 3 | king | queen | 27 | 8 | 0.235 | 0.187 | 0.232 | 0.253 | 0.048 |
| student-top-3 | `p3-vertical-line-actor-queen-dealer-king#0` | 3 | queen | king | 19 | 21 | 0.116 | -0.007 | -0.116 | 0.239 | 0.123 |
| student-top-3 | `p3-vertical-line-actor-queen-dealer-king#1` | 3 | queen | king | 21 | 16 | 0.165 | 0.125 | 0.003 | 0.107 | 0.040 |
| student-top-3 | `p3-vertical-line-actor-queen-dealer-king#2` | 3 | queen | king | 8 | 19 | 0.135 | 0.008 | -0.037 | 0.167 | 0.127 |
| student-top-3 | `p3-vertical-line-actor-queen-dealer-king#3` | 3 | queen | king | 19 | 21 | 0.072 | -0.062 | -0.098 | 0.083 | 0.134 |
| student-top-3 | `p4-continued-corner-growth-actor-king-dealer-king#2` | 4 | king | king | 14 | 10 | 0.013 | -0.044 | 0.140 | 0.189 | 0.058 |
| student-top-3 | `p4-continued-corner-growth-actor-king-dealer-king#4` | 4 | king | king | 13 | 29 | 0.056 | -0.029 | 0.089 | 0.111 | 0.084 |
| student-top-3 | `p4-continued-corner-growth-actor-queen-dealer-queen#0` | 4 | queen | queen | 14 | 26 | -0.091 | -0.131 | -0.193 | 0.029 | 0.040 |
| student-top-3 | `p4-continued-corner-growth-actor-queen-dealer-queen#1` | 4 | queen | queen | 13 | 20 | -0.027 | -0.377 | -0.218 | -0.067 | 0.350 |
| student-top-3 | `p4-continued-corner-growth-actor-queen-dealer-queen#2` | 4 | queen | queen | 26 | 20 | -0.051 | -0.382 | -0.227 | 0.004 | 0.331 |
| student-top-3 | `p4-continued-corner-growth-actor-queen-dealer-queen#3` | 4 | queen | queen | 22 | 26 | -0.096 | -0.136 | -0.293 | -0.068 | 0.040 |
| student-top-3 | `p4-continued-corner-growth-actor-queen-dealer-queen#4` | 4 | queen | queen | 13 | 28 | -0.071 | -0.071 | -0.213 | -0.009 | 0.000 |
| student-top-3 | `p5-continued-corner-growth-actor-king-dealer-queen#0` | 5 | king | queen | 30 | 28 | 0.269 | 0.252 | 0.246 | 0.248 | 0.017 |
| student-top-3 | `p5-continued-corner-growth-actor-king-dealer-queen#3` | 5 | king | queen | 18 | 30 | 0.190 | 0.186 | 0.256 | 0.316 | 0.004 |
| student-top-3 | `p5-continued-corner-growth-actor-queen-dealer-king#0` | 5 | queen | king | 30 | 20 | -0.041 | -0.204 | -0.083 | -0.046 | 0.163 |
| student-top-3 | `p5-continued-corner-growth-actor-queen-dealer-king#3` | 5 | queen | king | 30 | 18 | 0.022 | 0.018 | -0.028 | 0.047 | 0.004 |

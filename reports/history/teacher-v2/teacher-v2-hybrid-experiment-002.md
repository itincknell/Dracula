# Teacher v2 hybrid experiment 002

## Result

The larger run-1 response ranker did not produce a clear hybrid improvement over the prior smoke ranker. Run-1 top-2 improved the fixed defensive fixtures to 13/14, versus 8/14 for prior top-2 and 9/14 for pure Teacher v2, but it finished 8–16 against pure over the 24 paired games. Prior top-2 repeated its historical 11–13 record exactly. Run-1 direct finished 5–19.

The computational result is stable. Both top-2 controllers removed about 72% of pure Teacher v2's inner terminal evaluations and cut isolated median decision latency by roughly 3.6×. Run-1 direct removed all inner response completions and was about 19× faster, but its game result and score differential were substantially worse.

The evidence is mixed rather than promotive: run 1 learned a slightly different ranking that helped the curated defensive fixtures but regressed on this small absolute game sample and on fixed-state agreement/regret. Selection for manual testing remains the user's decision.

## Method

- Fixed decisions: 12 constructive fixtures, 14 defensive fixtures, and 24 role/dealer-balanced placement states repeated over five request seeds, for 146 decisions per controller.
- Games: 12 deterministic deck seeds, each hybrid versus pure Teacher v2 in both roles, for 24 six-round games and 144 rounds per hybrid. Dealer assignment alternated across rounds.
- Every controller retained 32 outer simulations and exact engine terminal scoring. Top-2 retained four-completion shallow evaluation for the model's two selected response groups.
- The independently rerun pure matrices, and the rerun prior-ranker matrix versus its historical artifact, are tensor/search-result exact after excluding elapsed-time and RSS fields.
- Prior ranker: `f6504bfb246e56e09d0e98d6a9f668d9744113045678450c84c5e48fc8cc9f75`.
- Run-1 ranker: `8ac77576c23061e35bbf0f1c4d9141ece86cfb3a8d50f603c16ce647d97d5585`.
- Prior raw artifact: `.local/teacher-v2-hybrid-experiment-002/prior-smoke.json` (`sha256:3e82da0353be932d911d7b97de9db658c9e1756d5df60bcc5fed73b0b16ae215`).
- Run-1 raw artifact: `.local/teacher-v2-hybrid-experiment-002/run1.json` (`sha256:f15613cb21a756965adc1ed05fbb0ca30025fd7be81f5b2fd93d4a2ba6fa80b5`).

## Strategic fixtures

| Suite | Fixture | P | Expected | Pure | Prior top-2 | Run-1 top-2 | Run-1 direct |
| --- | --- | ---: | --- | --- | --- | --- | --- |
| constructive | `dealer-late-forced` | 8 | `25` | pass; g25 | pass; g25 | pass; g25 | pass; g25 |
| constructive | `king-early-dealer-vampire` | 2 | `25` | pass; g25 | pass; g25 | fail; g3 | fail; g3 |
| constructive | `king-early-nondealer-suit` | 3 | `11` | fail; g9 | fail; g9 | pass; g11 | pass; g11 |
| constructive | `king-late-block` | 7 | `10` | pass; g10 | pass; g10 | pass; g10 | pass; g10 |
| constructive | `king-middle-preserve-card` | 5 | `10` | fail; g8 | pass; g10 | fail; g16 | fail; g16 |
| constructive | `king-middle-suit` | 6 | `24` | pass; g24 | pass; g24 | pass; g24 | pass; g24 |
| constructive | `queen-early-dealer-suit` | 2 | `3,11` | pass; g11 | pass; g3 | pass; g11 | pass; g11 |
| constructive | `queen-early-nondealer-suit` | 3 | `3` | pass; g3 | pass; g3 | pass; g3 | pass; g3 |
| constructive | `queen-late-offense-defense` | 7 | `20` | pass; g20 | pass; g20 | pass; g20 | pass; g20 |
| constructive | `queen-middle-avoid-vampire` | 6 | `4` | pass; g4 | pass; g4 | pass; g4 | pass; g4 |
| constructive | `queen-middle-color-exploit` | 6 | `29` | pass; g29 | pass; g29 | pass; g29 | pass; g29 |
| constructive | `queen-middle-offense-defense` | 5 | `27` | pass; g27 | pass; g27 | pass; g27 | pass; g27 |
| defensive | `king-early-dealer-constructive` | 2 | `19` | fail; g1 | pass; g19 | pass; g19 | pass; g19 |
| defensive | `king-early-nondealer-safe-intersection` | 3 | `4,23` | pass; g23 | fail; g15 | pass; g23 | pass; g4 |
| defensive | `king-late-block` | 7 | `2` | pass; g2 | pass; g2 | pass; g2 | pass; g2 |
| defensive | `king-late-dealer-forced` | 8 | `26` | pass; g26 | pass; g26 | pass; g26 | pass; g26 |
| defensive | `king-middle-avoid-suit` | 5 | `25` | fail; g29 | fail; g9 | pass; g25 | fail; g29 |
| defensive | `king-middle-denial-over-offense` | 5 | `5` | fail; g4 | fail; g4 | pass; g5 | pass; g5 |
| defensive | `queen-early-dealer-safe-intersection` | 2 | `28` | fail; g8 | fail; g24 | pass; g28 | fail; g16 |
| defensive | `queen-early-nondealer-avoid-suit` | 3 | `1,9` | fail; g5 | fail; g21 | fail; g23 | fail; g21 |
| defensive | `queen-late-block` | 7 | `7` | pass; g7 | pass; g7 | pass; g7 | pass; g7 |
| defensive | `queen-late-dealer-forced` | 8 | `29` | pass; g29 | pass; g29 | pass; g29 | pass; g29 |
| defensive | `queen-middle-avoid-color` | 5 | `18` | pass; g18 | pass; g18 | pass; g18 | fail; g5 |
| defensive | `queen-middle-avoid-vampire-destruction` | 5 | `23` | pass; g23 | fail; g0 | pass; g23 | fail; g2 |
| defensive | `queen-middle-constructive-over-empty-block` | 6 | `14` | pass; g14 | pass; g14 | pass; g14 | pass; g14 |
| defensive | `queen-middle-defensive-vampire` | 6 | `31` | pass; g31 | pass; g31 | pass; g31 | fail; g21 |

Totals:

| Controller | Constructive | Defensive |
| --- | ---: | ---: |
| pure | 10/12 | 9/14 |
| prior top-2 | 11/12 | 8/14 |
| run-1 top-2 | 10/12 | 13/14 |
| run-1 direct | 10/12 | 8/14 |

## Paired game results

| Hybrid | Record vs pure | Game VP (95% Wilson) | Round VP | Mean round differential |
| --- | ---: | ---: | ---: | ---: |
| prior top-2 | 11-13 | 45.8% (27.9%–64.9%) | 50.0% | -1.19 |
| run-1 top-2 | 8-16 | 33.3% (18.0%–53.3%) | 41.7% | -4.33 |
| run-1 direct | 5-19 | 20.8% (9.2%–40.5%) | 43.1% | -7.62 |

The intervals describe binomial sampling uncertainty only. With 12 decks, the game results distinguish obvious weakness from parity more reliably than they distinguish the two top-2 rankers from each other.

### Role and dealer splits

| Hybrid | Queen games | King games | Queen round VP / diff | King round VP / diff |
| --- | ---: | ---: | ---: | ---: |
| prior top-2 | 7-5 | 4-8 | 50.0% / +1.17 | 50.0% / -3.56 |
| run-1 top-2 | 5-7 | 3-9 | 45.8% / -1.68 | 37.5% / -6.99 |
| run-1 direct | 4-8 | 1-11 | 44.4% / -4.96 | 41.7% / -10.28 |

| Hybrid | Dealer round VP / diff | Non-dealer round VP / diff |
| --- | ---: | ---: |
| prior top-2 | 45.8% / -4.35 | 54.2% / +1.96 |
| run-1 top-2 | 44.4% / -1.96 | 38.9% / -6.71 |
| run-1 direct | 41.7% / -7.54 | 44.4% / -7.69 |

## Fixed-state agreement and Teacher-value regret

| Placement | Prior top-2 | Run-1 top-2 | Run-1 direct |
| ---: | ---: | ---: | ---: |
| 1 | 30.0% / 0.133 | 40.0% / 0.112 | 0.0% / 0.182 |
| 2 | 20.5% / 0.126 | 13.6% / 0.184 | 15.9% / 0.175 |
| 3 | 29.4% / 0.110 | 20.6% / 0.110 | 11.8% / 0.155 |
| 4 | 20.0% / 0.099 | 30.0% / 0.092 | 20.0% / 0.073 |
| 5 | 62.5% / 0.042 | 50.0% / 0.047 | 56.2% / 0.051 |
| 6 | 100.0% / 0.000 | 100.0% / 0.000 | 46.7% / 0.082 |
| 7 | 100.0% / 0.000 | 100.0% / 0.000 | 100.0% / 0.000 |
| Overall | 45.2% / 0.084 | 41.1% / 0.100 | 31.5% / 0.120 |

Each cell is strategic-group agreement with pure Teacher v2 / mean pure-Teacher value regret. Run-1 top-2 agreed less often and had higher regret than prior top-2 on this matrix.

### Teacher group absent from neural top two

| Ranker | Held-out response states | Missing | Omission rate |
| --- | ---: | ---: | ---: |
| Prior smoke | 38,652 | 20,318 | 52.6% |
| Run 1 | 38,652 | 20,252 | 52.4% |

This is measured at the actual shallow-response boundary over the shared 38,652-row held-out response set. It is not inferred from root actions. Run 1 reduced omissions by 66 rows (0.17 percentage points), a small offline change.

## Efficiency

The fixed matrix contains the same 146 decisions for every controller. Each hybrid is compared with the pure timing from its contemporaneous run; pure behavior and work counts were identical across the two runs.

| Controller | p50 | p95 | Max | Inner terminals | Reduction | Model calls / time | ms/call | Peak RSS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| pure (prior run) | 3.389s | 7.148s | 9.147s | 364,116 | 0.0% | 0 / 0.000s | — | 205.4 MiB |
| pure (run-1 run) | 3.959s | 9.036s | 10.268s | 364,116 | 0.0% | 0 / 0.000s | — | 205.7 MiB |
| prior top-2 | 0.956s | 2.170s | 3.115s | 103,032 | 71.7% | 12,879 / 7.600s | 0.590 | 217.0 MiB |
| run-1 top-2 | 1.091s | 2.289s | 3.506s | 102,688 | 71.8% | 12,836 / 8.097s | 0.631 | 217.0 MiB |
| run-1 direct | 0.209s | 0.480s | 0.642s | 0 | 100.0% | 12,870 / 7.750s | 0.602 | 197.1 MiB |

### Response-cache work on the fixed matrix

| Controller | Requests | Unique evaluations | Cache hits | Hit rate | Candidate actions evaluated |
| --- | ---: | ---: | ---: | ---: | ---: |
| pure | 13,347 | 12,888 | 459 | 3.4% | 91,029 |
| prior top-2 | 13,347 | 12,879 | 468 | 3.5% | 25,758 |
| run-1 top-2 | 13,347 | 12,836 | 511 | 3.8% | 25,672 |
| run-1 direct | 13,347 | 12,870 | 477 | 3.6% | 0 |

### Concurrent gameplay latency

| Hybrid | Decisions | p50 | p95 | Max | Inner terminals | Model calls / time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| prior top-2 | 504 | 1.037s | 4.521s | 5.507s | 311,056 | 38,882 / 27.91s |
| run-1 top-2 | 504 | 0.977s | 4.205s | 5.657s | 310,192 | 38,774 / 26.70s |
| run-1 direct | 504 | 0.251s | 0.632s | 0.745s | 0 | 38,701 / 27.35s |

## Every meaningful strategic-fixture disagreement

All 26 fixtures are shown above, including pass/fail changes. The table below exhaustively lists every fixture where a hybrid selected a different strategic group from pure Teacher v2. `Q` values are the exact normalized four-completion/outer-search means stored in the raw artifacts; the report prints six decimal places. All 266 fixed-state disagreements, including repeated synthetic placement states, remain in the raw artifacts with full-precision group visits and values.

| Controller | Fixture | P | Pure group | Hybrid group | Pure Q(pure) | Pure Q(hybrid) | Hybrid Q(pure) | Hybrid Q(hybrid) | Regret |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| prior top-2 | `constructive:king-middle-preserve-card` | 5 | 8 | 10 | -0.138333 | -0.140000 | -0.138333 | -0.105000 | 0.001667 |
| prior top-2 | `constructive:queen-early-dealer-suit` | 2 | 11 | 3 | 0.402222 | 0.353333 | 0.208333 | 0.355000 | 0.048889 |
| prior top-2 | `defensive:king-early-dealer-constructive` | 2 | 1 | 19 | 0.177778 | 0.146667 | -0.071111 | 0.171111 | 0.031111 |
| prior top-2 | `defensive:king-early-nondealer-safe-intersection` | 3 | 23 | 15 | 0.253333 | 0.131111 | 0.142222 | 0.204444 | 0.122222 |
| prior top-2 | `defensive:king-middle-avoid-suit` | 5 | 29 | 9 | -0.080000 | -0.200000 | -0.225000 | -0.000000 | 0.160000 |
| prior top-2 | `defensive:queen-early-dealer-safe-intersection` | 2 | 8 | 24 | -0.033333 | -0.057778 | -0.020000 | -0.013333 | 0.024444 |
| prior top-2 | `defensive:queen-early-nondealer-avoid-suit` | 3 | 5 | 21 | -0.048889 | -0.400000 | -0.066667 | 0.133333 | 0.351111 |
| prior top-2 | `defensive:queen-middle-avoid-vampire-destruction` | 5 | 23 | 0 | 0.116000 | 0.026667 | 0.180000 | 0.185333 | 0.089333 |
| run-1 top-2 | `constructive:king-early-dealer-vampire` | 2 | 25 | 3 | 0.078667 | -0.646667 | 0.015556 | 0.100000 | 0.751667 |
| run-1 top-2 | `constructive:king-early-nondealer-suit` | 3 | 9 | 11 | 0.057778 | 0.033333 | 0.040000 | 0.588889 | 0.024444 |
| run-1 top-2 | `constructive:king-middle-preserve-card` | 5 | 8 | 16 | -0.138333 | -0.223333 | -0.138333 | -0.113333 | 0.085000 |
| run-1 top-2 | `defensive:king-early-dealer-constructive` | 2 | 1 | 19 | 0.177778 | 0.146667 | -0.020000 | 0.204444 | 0.031111 |
| run-1 top-2 | `defensive:king-middle-avoid-suit` | 5 | 29 | 25 | -0.080000 | -0.040000 | -0.200000 | -0.040000 | 0.000000 |
| run-1 top-2 | `defensive:king-middle-denial-over-offense` | 5 | 4 | 5 | 0.144444 | 0.064444 | -0.006667 | 0.064444 | 0.080000 |
| run-1 top-2 | `defensive:queen-early-dealer-safe-intersection` | 2 | 8 | 28 | -0.033333 | -0.266667 | -0.186667 | -0.015556 | 0.233333 |
| run-1 top-2 | `defensive:queen-early-nondealer-avoid-suit` | 3 | 5 | 23 | -0.048889 | -0.113333 | -0.120000 | 0.148889 | 0.064444 |
| run-1 direct | `constructive:king-early-dealer-vampire` | 2 | 25 | 3 | 0.078667 | -0.646667 | 0.151111 | 0.430667 | 0.751667 |
| run-1 direct | `constructive:king-early-nondealer-suit` | 3 | 9 | 11 | 0.057778 | 0.033333 | 0.040000 | 0.273333 | 0.024444 |
| run-1 direct | `constructive:king-middle-preserve-card` | 5 | 8 | 16 | -0.138333 | -0.223333 | -0.138333 | -0.123333 | 0.085000 |
| run-1 direct | `defensive:king-early-dealer-constructive` | 2 | 1 | 19 | 0.177778 | 0.146667 | 0.046667 | 0.255556 | 0.031111 |
| run-1 direct | `defensive:king-early-nondealer-safe-intersection` | 3 | 23 | 4 | 0.253333 | 0.177778 | 0.122222 | 0.508333 | 0.075556 |
| run-1 direct | `defensive:king-middle-denial-over-offense` | 5 | 4 | 5 | 0.144444 | 0.064444 | -0.006667 | 0.023333 | 0.080000 |
| run-1 direct | `defensive:queen-early-dealer-safe-intersection` | 2 | 8 | 16 | -0.033333 | -0.197778 | 0.053333 | 0.086667 | 0.164444 |
| run-1 direct | `defensive:queen-early-nondealer-avoid-suit` | 3 | 5 | 21 | -0.048889 | -0.400000 | -0.086667 | 0.240000 | 0.351111 |
| run-1 direct | `defensive:queen-middle-avoid-color` | 5 | 18 | 5 | -0.026667 | -0.096000 | 0.040000 | 0.040000 | 0.069333 |
| run-1 direct | `defensive:queen-middle-avoid-vampire-destruction` | 5 | 23 | 2 | 0.116000 | -0.086667 | 0.265000 | 0.300000 | 0.202667 |
| run-1 direct | `defensive:queen-middle-defensive-vampire` | 6 | 31 | 21 | 0.013333 | -0.566667 | 0.260000 | 0.260000 | 0.580000 |

## Verification

- Deterministic replay: 146/146 pure decisions matched across both fresh runs and the historical control after excluding only timing/RSS. The prior top-2 rerun matched all 146 decisions and all 24 complete games from the historical experiment.
- Pure Teacher v2 invariance: selected actions, strategic groups, visits, action means, response-work counters, and exact terminal values were unchanged.
- Privacy and legality: the focused search/model suite passed. Student inference accepts only actor-relative information-state tensors and legal masks; no authoritative hand, stock, seed, tree, or private model state enters it.
- Exact scoring: every outer terminal value came from the deterministic engine. No model value head or scoring approximation was enabled.
- Legal actions: no fixed decision or game produced an illegal action, and forced placements bypassed response inference.

## Interpretation

Run 1 is not a demonstrated upgrade to the hybrid. Its curated defensive fixture result is stronger, but its 8–16 game record, 41.1% fixed-state agreement, and 0.100 mean Teacher-value regret are worse than the prior top-2 ranker's 11–13, 45.2%, and 0.084. The top-two omission rate barely changed. The game sample is small enough that the two top-2 records are not a precise strength ranking, but it provides no positive evidence that the larger response dataset improved gameplay.

Both top-2 hybrids retain the useful efficiency result: about 72% fewer inner terminal evaluations and roughly 3.6× lower isolated median latency. Whether the run-1 controller's defensive fixture behavior is worth manual browser comparison, or whether the prior top-2 controller remains the better experimental candidate, is deferred to the user.

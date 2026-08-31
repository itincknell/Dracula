# Belief-greedy balanced-miner benchmark

Date: 2026-08-23
Machine: Apple M3 MacBook Air, 8 GB unified memory
Source revision: `ec6e3d395bc89e9b0cb4bf31eb15438119281c60`

## Implemented controller

The controller retains one information-set UCT tree for the real decision.
Every later simulated actor uses a one-ply belief-greedy response rather than
launching a nested UCT tree. For each legal strategic group, the response
evaluator uses eight shared hidden-hand completions, completes the coffin, and
maximizes the engine's exact normalized actor-relative round differential.
Only the acting player's information state enters that evaluation.

The balanced miner emits one Sam label at every learned placement. Each
complete six-round game contains exactly six rows for each placement 1–7 and
42 rows total. Forced eighth placements are applied but not stored. The twelve
benchmark fixtures cycle through all-teacher, one-deviation, two-deviation,
mixed, and all-alternative trajectories; alternative groups are sampled
deterministically and uniformly from the legal nonteacher groups.

## Measurement

Both configurations used the same twelve deck ordinals, root seed, trajectory
profiles, eight belief completions, and four complete-game workers. Each worker
used one PyTorch thread. Twelve games form three fully occupied worker waves,
avoiding a partially occupied final wave in the daily projection.

| Metric | 32 outer | 128 outer |
| --- | ---: | ---: |
| Measured games | 12 | 12 |
| Measured rows | 504 | 504 |
| Wall time | 44.548 s | 176.937 s |
| Mean complete-game time | 14.503 s | 58.377 s |
| Rows/second | 11.314 | 2.848 |
| Peak RSS per worker | 197.1 MB | 185.1 MB |
| Response potential evaluations | 2,127,248 | 7,487,312 |
| Response requests | 40,685 | 150,994 |

The 128-outer configuration measured 3.97 times slower than 32 outer.

## Projected rows in 24 hours

| Learned placement | 32 outer | 128 outer |
| --- | ---: | ---: |
| 1 | 139,642 | 35,158 |
| 2 | 139,642 | 35,158 |
| 3 | 139,642 | 35,158 |
| 4 | 139,642 | 35,158 |
| 5 | 139,642 | 35,158 |
| 6 | 139,642 | 35,158 |
| 7 | 139,642 | 35,158 |
| **Total** | **977,494** | **246,107** |

These are direct extrapolations from the measured fully occupied wall-clock
rates. They include search, varied trajectory advancement, observation
packing, process scheduling, and sealed game writes. They do not include a
24-hour thermal-soak correction; actual sustained output can differ with
temperature and competing system load.

## Verification

- Each configuration sealed 12 game artifacts and one canonical manifest.
- Every game contained 42 rows and exactly six rows at each placement 1–7.
- Each configuration contained 252 Queen and 252 King rows.
- Every selected label belonged to a legal strategic group; groups partitioned
  the engine legal mask exactly.
- Rows contained the visible packed observation, legal mask, group membership,
  teacher representative, and public metadata. They contained no opponent
  hand, stock order, engine seed, determinization, search tree, visit target,
  action value, or return target.
- Replaying fixture zero independently at both budgets reproduced the sealed
  game content digest exactly.
- Focused search and miner tests: 11 passed.

## Artifacts

- Raw benchmark: `.local/belief-greedy-balanced-benchmark-001/benchmark.json`
- Raw tree size: 876 KB
- Raw benchmark SHA-256:
  `9135b27c2c1a9349add653ac5884fe342043a957322eb76d176d5a7073863f2a`
- 32-outer manifest SHA-256:
  `11ab61c72d88bc199ea93da07fb744e9bea0ca605072e030c1bf2d48ec197435`
- 128-outer manifest SHA-256:
  `6756ceb62e3dd06215e57c04dfe48c1dd12a3a127fb2caaf317d978f922db958`

No continuing collection was started by this benchmark.

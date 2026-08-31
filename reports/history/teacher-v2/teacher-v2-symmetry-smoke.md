# Teacher v2 symmetry smoke

## Result

**Fail.** Symmetry reduction lowers early-turn computation and raises selected-group agreement, but it does not materially concentrate the 32-visit target and its action-value rankings remain unstable.

No corpus collection or neural training was performed. The existing dataset configuration must remain unchanged.

## Configuration

- Fixtures: 24 fixed role-balanced states covering placements 1–7
- Request seeds: 5 per state and planner
- Searches: 240
- Outer simulations: 32
- Response completions per strategic action: 4
- Outer exploration: `sqrt(2)`
- Terminal value: exact normalized round differential
- Baseline: destination symmetry disabled
- Reduced: authoritative destination symmetry enabled
- Baseline search digest: `a6add599d8324f8a3098ee64635705078c00102f76d7804aa01d9ee8b1251fd5`
- Reduced search digest: `63a0ec904f71aa739c91e3a728fa7450f9431fa4e7e672694295dcc588ec0ecb`
- Baseline response digest: `97af09f17f00748013b9a7004b67302d605b0e65026a5d9a4c797d3852b8c0a9`
- Reduced response digest: `d5ad44d941ee32bf3d50e3bf7276c3611ab17404240f73ae5f96bb31bd29d1bc`

The two planners used identical fixtures and request seeds. Configuration differs only in the symmetry contract and its derived digest.

## Per-placement measurements

| Placement | State | Mode | Concrete | Groups | Entropy | Top-two | Seed agreement | Value-rank | Terminals | Latency |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | placement-1-opening | baseline | 16.0 | 8.0 | 100.0% | 0.0% | 40.0% | 0.271 | 7095.6 | 9.957s |
| 1 | placement-1-opening | reduced | 16.0 | 8.0 | 99.4% | 1.2% | 60.0% | 0.411 | 5591.6 | 7.379s |
| 2 | placement-2-adjacent | baseline | 20.0 | 12.0 | 97.5% | 0.2% | 37.5% | 0.163 | 4853.5 | 5.975s |
| 2 | placement-2-adjacent | reduced | 20.0 | 12.0 | 99.3% | 0.5% | 47.5% | 0.150 | 4143.9 | 4.946s |
| 3 | placement-3-line | baseline | 18.0 | 9.0 | 99.3% | 0.0% | 45.0% | 0.330 | 3520.0 | 3.765s |
| 3 | placement-3-line | reduced | 18.0 | 9.0 | 99.4% | 1.1% | 60.0% | 0.448 | 3232.0 | 3.495s |
| 3 | placement-3-no-symmetry | baseline | 15.0 | 15.0 | 99.6% | 0.0% | 40.0% | 0.118 | 3116.0 | 3.341s |
| 3 | placement-3-no-symmetry | reduced | 15.0 | 15.0 | 99.6% | 0.0% | 20.0% | 0.168 | 3116.0 | 3.356s |
| 4 | placement-4-no-symmetry | baseline | 12.0 | 12.0 | 99.2% | 0.9% | 60.0% | 0.587 | 1424.0 | 1.213s |
| 4 | placement-4-no-symmetry | reduced | 12.0 | 12.0 | 99.3% | 0.6% | 40.0% | 0.422 | 1415.2 | 1.222s |
| 5 | placement-5-no-symmetry | baseline | 6.0 | 6.0 | 99.5% | 0.6% | 90.0% | 0.543 | 728.8 | 0.448s |
| 5 | placement-5-no-symmetry | reduced | 6.0 | 6.0 | 99.3% | 1.9% | 60.0% | 0.563 | 732.8 | 0.448s |
| 6 | placement-6-no-symmetry | baseline | 6.0 | 6.0 | 99.5% | 1.6% | 80.0% | 0.391 | 247.2 | 0.131s |
| 6 | placement-6-no-symmetry | reduced | 6.0 | 6.0 | 99.4% | 2.2% | 80.0% | 0.342 | 249.6 | 0.133s |
| 7 | placement-7-no-symmetry | baseline | 2.0 | 2.0 | 99.9% | 1.9% | 80.0% | 0.400 | 0.0 | 0.033s |
| 7 | placement-7-no-symmetry | reduced | 2.0 | 2.0 | 99.9% | 1.9% | 80.0% | 0.400 | 0.0 | 0.033s |

Placement 1, placement 2, conditional placement-3 line states, and the placement-3 non-symmetry control remain separate. Placements 4–7 are reported individually.

## Early-turn signal

### placement-1-opening

- Selected-group agreement across seeds: 40.0% → 60.0%
- Normalized visit entropy: 100.0% → 99.4%
- Top-two visit margin: 0.0% → 1.2%
- Action-value rank agreement: 0.271 → 0.411
- Terminal-evaluation saving: 21.2%
- Mean runtime saving: 25.9%
- Strategic-action disagreements after projecting mirrors: 7/10

### placement-2-adjacent

- Selected-group agreement across seeds: 37.5% → 47.5%
- Normalized visit entropy: 97.5% → 99.3%
- Top-two visit margin: 0.2% → 0.5%
- Action-value rank agreement: 0.163 → 0.150
- Terminal-evaluation saving: 14.6%
- Mean runtime saving: 17.2%
- Strategic-action disagreements after projecting mirrors: 33/40

### placement-3-line

- Selected-group agreement across seeds: 45.0% → 60.0%
- Normalized visit entropy: 99.3% → 99.4%
- Top-two visit margin: 0.0% → 1.1%
- Action-value rank agreement: 0.330 → 0.448
- Terminal-evaluation saving: 8.2%
- Mean runtime saving: 7.2%
- Strategic-action disagreements after projecting mirrors: 18/20

The symmetry-aware target still assigns nearly equal visits to the strategic groups. The selected action is therefore determined by very small count differences. Cross-seed value-rank correlations of 0.411, 0.150, and 0.448 show that noisy action values remain the deeper limitation.

## Concrete fair-coin results

- placement-1-opening: `{"2-8": {"count": 10, "frequencies": {"2": 0.6, "8": 0.4}, "positions": {"2": 6, "8": 4}}}`
- placement-2-adjacent: `{"1-3": {"count": 4, "frequencies": {"1": 0.5, "3": 0.5}, "positions": {"1": 2, "3": 2}}, "1-7": {"count": 6, "frequencies": {"1": 0.3333333333333333, "7": 0.6666666666666666}, "positions": {"1": 2, "7": 4}}, "2-8": {"count": 5, "frequencies": {"2": 0.8, "8": 0.2}, "positions": {"2": 4, "8": 1}}, "3-9": {"count": 2, "frequencies": {"3": 1.0}, "positions": {"3": 2}}, "4-6": {"count": 9, "frequencies": {"4": 0.3333333333333333, "6": 0.6666666666666666}, "positions": {"4": 3, "6": 6}}, "7-9": {"count": 2, "frequencies": {"7": 0.5, "9": 0.5}, "positions": {"7": 1, "9": 1}}}`
- placement-3-line: `{"1-3": {"count": 5, "frequencies": {"1": 0.2, "3": 0.8}, "positions": {"1": 1, "3": 4}}, "1-7": {"count": 1, "frequencies": {"1": 1.0}, "positions": {"1": 1}}, "2-8": {"count": 5, "frequencies": {"2": 0.2, "8": 0.8}, "positions": {"2": 1, "8": 4}}, "3-9": {"count": 4, "frequencies": {"3": 0.25, "9": 0.75}, "positions": {"3": 1, "9": 3}}, "4-6": {"count": 5, "frequencies": {"4": 0.6, "6": 0.4}, "positions": {"4": 3, "6": 2}}}`

Concrete frequencies describe only searches whose selected strategic group was paired. The contract tests separately prove that both members of every pair are reachable and that the coin cannot change group visits or values.

## Privacy, legality, and reproducibility

- Replayed searches: 14 (both planners at one fixture for every placement)
- Visits, values, selected strategic groups, concrete actions, principal continuations, counters, and deterministic diagnostics matched exactly: True
- Raw forbidden private fields present: []
- Search inputs were typed player information states; raw results contain no authoritative engine state, hidden hand, stock order, determinization, seed, tree, or model state.
- Every selected concrete action passed the search result's engine-legality validation.

## Gate

- higher selected group agreement: **pass**
- meaningfully concentrated visits: **fail**
- stable action value rankings: **fail**
- exact strategic meaning: **pass**
- privacy and legality: **pass**
- reproducibility: **pass**
- lower early computational cost: **pass**

The smoke fails because visit concentration and action-value stability fail. Symmetry reduction removes duplicate spatial choices correctly, but 32 UCT visits still provide little evidence beyond initial group coverage, while shallow-response Monte Carlo values vary substantially across hidden samples.

## Runtime and artifacts

- Comparison wall time: 217.6 seconds
- Worker processes: 4
- Comparison: `.local/teacher-v2-symmetry-smoke-20260723/comparison.json`
- Comparison SHA-256: `82f2721dc03a3687eb02d640f6280d35fbefb611d2ec7a968fb25311e84713c9`
- Reproducibility: `.local/teacher-v2-symmetry-smoke-20260723/reproducibility.json`
- Reproducibility SHA-256: `89cbda5b1191ed3368ff4f6c3a836ebc81393d68e91c4e5b94fbc287c15cc500`
- Sealed manifest: `.local/teacher-v2-symmetry-smoke-20260723/manifest.json`
- Baseline profile: `teacher-v2-32x4-no-destination-symmetry`
- Reduced profile: `teacher-v2-32x4-authoritative-destination-symmetry`

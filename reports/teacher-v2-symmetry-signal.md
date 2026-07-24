# Teacher v2 symmetry signal measurement

## Method

24 deterministic role-balanced states were measured with 5 request seeds under the frozen 32x4 baseline and symmetry-reduced 32x4 Teacher v2.

Baseline concrete visits and visit-weighted action values were projected into the authoritative destination groups before comparison. Initial coverage remains the cost paid by each planner: concrete actions for baseline and strategic groups for the reduced planner.

## Findings

The reduction halves initial root coverage at placement 1 and the conditional placement-3 line states, and reduces placement-2 coverage from 62.5% to 37.5%.

That saved work does not produce a clearly concentrated 32-visit target. Reduced-search normalized entropy remains 99.4% at placement 1, 99.3% at placement 2, and 99.4% in the conditional placement-3 line states. Mean top-two margins remain at or below 1.2%.

Selected-group agreement improves in those three categories, while action-value rank agreement improves at placement 1 and the conditional placement-3 line states but not placement 2. The signal result is therefore mixed: symmetry reduces redundant computation and modestly improves some stability measures, but does not make the 32-visit distribution a strong policy target.

Mean latency falls from 11.819s to 8.743s at placement 1, from 6.752s to 5.570s at placement 2, and from 4.066s to 3.756s in conditional placement-3 line states.

Paired-position frequencies below are descriptive; five request seeds per state are sufficient for signal comparison, not a statistical test of coin fairness.

## Root signal by placement

| Placement | State | Mode | Concrete | Groups | Coverage | Entropy | Max share | Top-two | Selected agreement | Value-rank agreement |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | placement-1-opening | baseline | 16.0 | 8.0 | 50.0% | 100.0% | 12.5% | 0.0% | 40.0% | 0.271 |
| 1 | placement-1-opening | reduced | 16.0 | 8.0 | 25.0% | 99.4% | 15.6% | 1.2% | 60.0% | 0.411 |
| 2 | placement-2-adjacent | baseline | 20.0 | 12.0 | 62.5% | 97.5% | 12.5% | 0.2% | 37.5% | 0.163 |
| 2 | placement-2-adjacent | reduced | 20.0 | 12.0 | 37.5% | 99.3% | 9.9% | 0.5% | 47.5% | 0.150 |
| 3 | placement-3-line | baseline | 18.0 | 9.0 | 56.2% | 99.3% | 12.5% | 0.0% | 45.0% | 0.330 |
| 3 | placement-3-line | reduced | 18.0 | 9.0 | 28.1% | 99.4% | 13.6% | 1.1% | 60.0% | 0.448 |
| 3 | placement-3-no-symmetry | baseline | 15.0 | 15.0 | 46.9% | 99.6% | 9.4% | 0.0% | 40.0% | 0.118 |
| 3 | placement-3-no-symmetry | reduced | 15.0 | 15.0 | 46.9% | 99.6% | 9.4% | 0.0% | 20.0% | 0.168 |
| 4 | placement-4-no-symmetry | baseline | 12.0 | 12.0 | 37.5% | 99.2% | 10.3% | 0.9% | 60.0% | 0.587 |
| 4 | placement-4-no-symmetry | reduced | 12.0 | 12.0 | 37.5% | 99.3% | 10.0% | 0.6% | 40.0% | 0.422 |
| 5 | placement-5-no-symmetry | baseline | 6.0 | 6.0 | 18.8% | 99.5% | 19.4% | 0.6% | 90.0% | 0.543 |
| 5 | placement-5-no-symmetry | reduced | 6.0 | 6.0 | 18.8% | 99.3% | 20.3% | 1.9% | 60.0% | 0.563 |
| 6 | placement-6-no-symmetry | baseline | 6.0 | 6.0 | 18.8% | 99.5% | 20.0% | 1.6% | 80.0% | 0.391 |
| 6 | placement-6-no-symmetry | reduced | 6.0 | 6.0 | 18.8% | 99.4% | 20.3% | 2.2% | 80.0% | 0.342 |
| 7 | placement-7-no-symmetry | baseline | 2.0 | 2.0 | 6.2% | 99.9% | 50.9% | 1.9% | 80.0% | 0.400 |
| 7 | placement-7-no-symmetry | reduced | 2.0 | 2.0 | 6.2% | 99.9% | 50.9% | 1.9% | 80.0% | 0.400 |

Value-rank agreement is mean pairwise Spearman correlation across the five request seeds.

## Search work by placement

| Placement | State | Mode | Response requests | Cache hits | Candidates | Terminals | Mean latency | p95 latency | Peak RSS |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | placement-1-opening | baseline | 176.0 | 0.0 | 1773.9 | 7095.6 | 11.819s | 12.891s | 189.0 MiB |
| 1 | placement-1-opening | reduced | 168.0 | 0.0 | 1397.9 | 5591.6 | 8.743s | 9.627s | 189.0 MiB |
| 2 | placement-2-adjacent | baseline | 148.0 | 0.0 | 1213.4 | 4853.5 | 6.752s | 7.167s | 189.0 MiB |
| 2 | placement-2-adjacent | reduced | 140.0 | 0.0 | 1036.0 | 4143.9 | 5.570s | 5.944s | 189.0 MiB |
| 3 | placement-3-line | baseline | 114.0 | 0.0 | 880.0 | 3520.0 | 4.066s | 4.332s | 189.0 MiB |
| 3 | placement-3-line | reduced | 105.0 | 0.0 | 808.0 | 3232.0 | 3.756s | 3.919s | 189.0 MiB |
| 3 | placement-3-no-symmetry | baseline | 111.0 | 0.0 | 779.0 | 3116.0 | 3.683s | 4.110s | 189.0 MiB |
| 3 | placement-3-no-symmetry | reduced | 111.0 | 0.0 | 779.0 | 3116.0 | 3.709s | 4.369s | 189.0 MiB |
| 4 | placement-4-no-symmetry | baseline | 76.0 | 0.0 | 356.0 | 1424.0 | 1.355s | 1.742s | 188.9 MiB |
| 4 | placement-4-no-symmetry | reduced | 76.0 | 0.0 | 353.8 | 1415.2 | 1.384s | 1.734s | 188.9 MiB |
| 5 | placement-5-no-symmetry | baseline | 38.0 | 0.1 | 182.2 | 728.8 | 0.510s | 0.540s | 189.0 MiB |
| 5 | placement-5-no-symmetry | reduced | 38.0 | 0.0 | 183.2 | 732.8 | 0.507s | 0.552s | 189.0 MiB |
| 6 | placement-6-no-symmetry | baseline | 32.0 | 1.1 | 61.8 | 247.2 | 0.137s | 0.158s | 188.9 MiB |
| 6 | placement-6-no-symmetry | reduced | 32.0 | 0.8 | 62.4 | 249.6 | 0.140s | 0.152s | 188.9 MiB |
| 7 | placement-7-no-symmetry | baseline | 0.0 | 0.0 | 0.0 | 0.0 | 0.035s | 0.036s | 188.3 MiB |
| 7 | placement-7-no-symmetry | reduced | 0.0 | 0.0 | 0.0 | 0.0 | 0.037s | 0.044s | 188.3 MiB |

## Paired-position selection

### placement-1-opening

- baseline: `{"2-8": {"count": 8, "frequencies": {"2": 0.75, "8": 0.25}, "positions": {"2": 6, "8": 2}}, "4-6": {"count": 2, "frequencies": {"6": 1.0}, "positions": {"6": 2}}}`
- reduced: `{"2-8": {"count": 10, "frequencies": {"2": 0.6, "8": 0.4}, "positions": {"2": 6, "8": 4}}}`
- Cross-mode selected-group agreement: 30.0%

### placement-2-adjacent

- baseline: `{"1-3": {"count": 4, "frequencies": {"1": 0.75, "3": 0.25}, "positions": {"1": 3, "3": 1}}, "1-7": {"count": 5, "frequencies": {"1": 0.6, "7": 0.4}, "positions": {"1": 3, "7": 2}}, "2-8": {"count": 4, "frequencies": {"8": 1.0}, "positions": {"8": 4}}, "3-9": {"count": 5, "frequencies": {"3": 0.2, "9": 0.8}, "positions": {"3": 1, "9": 4}}, "4-6": {"count": 10, "frequencies": {"4": 0.7, "6": 0.3}, "positions": {"4": 7, "6": 3}}, "7-9": {"count": 2, "frequencies": {"7": 1.0}, "positions": {"7": 2}}}`
- reduced: `{"1-3": {"count": 4, "frequencies": {"1": 0.5, "3": 0.5}, "positions": {"1": 2, "3": 2}}, "1-7": {"count": 6, "frequencies": {"1": 0.3333333333333333, "7": 0.6666666666666666}, "positions": {"1": 2, "7": 4}}, "2-8": {"count": 5, "frequencies": {"2": 0.8, "8": 0.2}, "positions": {"2": 4, "8": 1}}, "3-9": {"count": 2, "frequencies": {"3": 1.0}, "positions": {"3": 2}}, "4-6": {"count": 9, "frequencies": {"4": 0.3333333333333333, "6": 0.6666666666666666}, "positions": {"4": 3, "6": 6}}, "7-9": {"count": 2, "frequencies": {"7": 0.5, "9": 0.5}, "positions": {"7": 1, "9": 1}}}`
- Cross-mode selected-group agreement: 17.5%

### placement-3-line

- baseline: `{"1-3": {"count": 2, "frequencies": {"1": 0.5, "3": 0.5}, "positions": {"1": 1, "3": 1}}, "1-7": {"count": 3, "frequencies": {"1": 0.3333333333333333, "7": 0.6666666666666666}, "positions": {"1": 1, "7": 2}}, "2-8": {"count": 4, "frequencies": {"2": 0.5, "8": 0.5}, "positions": {"2": 2, "8": 2}}, "3-9": {"count": 3, "frequencies": {"9": 1.0}, "positions": {"9": 3}}, "4-6": {"count": 2, "frequencies": {"6": 1.0}, "positions": {"6": 2}}, "7-9": {"count": 6, "frequencies": {"7": 0.5, "9": 0.5}, "positions": {"7": 3, "9": 3}}}`
- reduced: `{"1-3": {"count": 5, "frequencies": {"1": 0.2, "3": 0.8}, "positions": {"1": 1, "3": 4}}, "1-7": {"count": 1, "frequencies": {"1": 1.0}, "positions": {"1": 1}}, "2-8": {"count": 5, "frequencies": {"2": 0.2, "8": 0.8}, "positions": {"2": 1, "8": 4}}, "3-9": {"count": 4, "frequencies": {"3": 0.25, "9": 0.75}, "positions": {"3": 1, "9": 3}}, "4-6": {"count": 5, "frequencies": {"4": 0.6, "6": 0.4}, "positions": {"4": 3, "6": 2}}}`
- Cross-mode selected-group agreement: 10.0%

### placement-3-no-symmetry

- baseline: `{}`
- reduced: `{}`
- Cross-mode selected-group agreement: 10.0%

### placement-4-no-symmetry

- baseline: `{}`
- reduced: `{}`
- Cross-mode selected-group agreement: 30.0%

### placement-5-no-symmetry

- baseline: `{}`
- reduced: `{}`
- Cross-mode selected-group agreement: 60.0%

### placement-6-no-symmetry

- baseline: `{}`
- reduced: `{}`
- Cross-mode selected-group agreement: 90.0%

### placement-7-no-symmetry

- baseline: `{}`
- reduced: `{}`
- Cross-mode selected-group agreement: 100.0%

## Runtime

- Searches: 240
- Worker processes: 4
- Wall time: 244.3 seconds

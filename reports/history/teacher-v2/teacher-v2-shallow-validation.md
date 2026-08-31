# Shallow Teacher v2 validation

Validation completed July 22, 2026. No configuration passed the human-test
gate. Version 1 remains the local opponent and neural training remains gated.

## Fixture evidence

The diagnostic solver reproduced the expected action for all 12 learned
defensive fixtures under exact adversarial continuation analysis. It also
supported the expected action set for all 11 learned constructive fixtures
under exact perfect-information continuation analysis. The three forced
fixtures retain their sole engine-legal action. No fixture expectation or
reward was changed during validation.

| Controller | Outer | Completions/action | Constructive | Defensive | Fixture p95 | Maximum |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Frozen v1 | 500 | uniform response | 100.0% | 78.3% | 3.074 s | 3.111 s |
| Shallow v2 | 32 | 1 | 81.8% | 66.7% | 3.275 s | 3.617 s |
| Shallow v2 | 32 | 2 | 63.6% | 66.7% | 6.468 s | 6.873 s |
| Shallow v2 | 32 | 4 | 90.9% | 91.7% | 13.164 s | 13.754 s |
| Shallow v2 | 64 | 4 | 81.8% | 75.0% | 17.086 s | 17.664 s |
| Shallow v2 | 500 | 1 | 100.0% | 66.7% | 14.707 s | 16.114 s |
| Shallow v2 | 500 | 2 | 81.8% | 58.3% | 25.782 s | 26.525 s |
| Shallow v2 | 500 | 4 | 100.0% | 83.3% | 49.716 s | 52.222 s |

Each constructive percentage contains 33 decisions across three fixed request
seeds. Each defensive percentage contains 60 decisions across five fixed
request seeds. Frozen v1's constructive result is its sealed 33-decision
baseline.

The only profile inside the relative latency gate was `32×1`, whose p95 was
less than 1.5 times the frozen v1 p95. It failed both strategic fixture gates.
The strongest defensive profile, `32×4`, missed one constructive fixture and
one defensive fixture family and exceeded the relative latency limit. At the
500-simulation target, four completions preserved construction and modestly
exceeded v1's defensive rate, but missed two defensive fixture families and
exceeded both the p95 and 30-second maximum limits.

## Target-budget cost

Single-worker measurements use the same representative early, middle, and late
positions for each profile.

| Completions/action | Early | Middle | Late | Early terminal evaluations/s |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 11.020 s | 1.822 s | 0.902 s | 483.4 |
| 2 | 19.087 s | 1.889 s | 0.911 s | 566.9 |
| 4 | 34.846 s | 1.988 s | 0.900 s | 605.8 |

The 500-simulation profiles reused 71.2%–71.7% of shallow response requests.
Peak measured process RSS was 209.3 MiB, below the 1 GiB limit.

## Failed-action evidence

At `500×4`, the two failed defensive positions were:

- `queen-early-dealer-safe-intersection`: action 17 received 37 visits with
  mean value `-0.0959`; expected action 28 received 19 visits with mean value
  `-0.3312`. The direct four-completion response estimate also preferred action
  25 at `0.1017` over expected action 28 at `0.0500`.
- `king-early-nondealer-safe-intersection`: action 15 received 64 visits with
  mean value `0.1509`; expected actions 4 and 23 received 55 and 44 visits with
  means `0.1143` and `0.0606`. The direct response estimate preferred expected
  action 4, showing that the outer sampled continuations reversed the shallow
  response ranking.

The complete per-action visits, means, exact values, timings, and deterministic
request records are in
`runs/teacher-v2-shallow-validation-001/teacher-v2-validation.json`.

## Absolute controls

Teacher v2 game, round, role, dealer, and paired-interval results are not
reported because no profile passed the prerequisite constructive, defensive,
and latency gates. Starting the 12-pair permanent controls and 30-pair v1
non-regression matrix after that failure would not make any profile eligible.
Frozen v1 retains its sealed 24–0 results against both random legal play and
`policy-2-v20`.

## Diagnosis

The simulation fast path removed redundant whole-deck validation and preserved
all frozen v1 golden results. It improved early `32×1` latency from 9.967
seconds to 2.662 seconds, so engine overhead is no longer the primary failure.

The remaining problem is the response estimate. Each information state caches
only one, two, or four uniform-completion samples. Raising the outer budget
reuses those estimates; it does not make the response policy converge.
Increasing the completion count changes the biased random-continuation target
and does not improve it monotonically. The misses at `500×4` show both failure
modes: one direct response ranking is wrong, while another correct direct
ranking is reversed by the outer continuation distribution.

Additional completion counts would increase latency without resolving the
one-ply random-continuation semantics. Fixing the strategic deficiency requires
a different response model, not a bounded parameter adjustment to this
candidate.

## Decision

No shallow-response Teacher v2 configuration is selected for human testing.
The tracker remains at the validation gate, the version 1 controller remains
the default, and neural teacher collection does not begin.

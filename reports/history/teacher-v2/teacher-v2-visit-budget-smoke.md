# Teacher v2 visit-budget smoke

## Result

Neither 64 nor 128 outer simulations produces a useful early-round policy
target from root visit counts. The full Teacher v2 corpus remains blocked.

Increasing the budget improves late-round concentration and modestly improves
middle-round action agreement at 128. Early-round visits remain almost uniform,
the top-two separation shrinks, and the selected action remains unstable across
request seeds.

This does not invalidate the manually approved Teacher v2 opponent. It shows
that root visits do not retain the information responsible for its strong play
under the current UCT configuration.

## Method

The assessment reused the six states from the 32-visit smoke corpus:

- Two early-round states with 16 legal actions.
- Two middle-round states with 15 legal actions.
- Two late-round states with two legal actions.

Each state ran under the same five deterministic request-seed indexes at 32,
64, and 128 outer simulations. The four-completion shallow response policy,
information state, exploration constant, engine, and scoring remained
unchanged. Each budget therefore represents 30 complete searches.

## Target quality

| Stage | Budget | Initial coverage | Normalized entropy | Top-two margin | Top-action agreement |
| --- | ---: | ---: | ---: | ---: | ---: |
| Early | 32 | 50.0% | 99.88% | 0.63% | 30% |
| Early | 64 | 25.0% | 99.65% | 0.47% | 30% |
| Early | 128 | 12.5% | 99.71% | 0.23% | 40% |
| Middle | 32 | 46.9% | 99.58% | 0.00% | 70% |
| Middle | 64 | 23.4% | 99.69% | 0.16% | 70% |
| Middle | 128 | 11.7% | 99.58% | 1.02% | 90% |
| Late | 32 | 6.3% | 95.04% | 19.38% | 100% |
| Late | 64 | 3.1% | 92.92% | 23.75% | 100% |
| Late | 128 | 1.6% | 87.99% | 30.47% | 100% |

Entropy near 100% means that the recorded visits are spread almost evenly
among legal moves. At 128 simulations the most-visited early action receives
only 7.66% of visits on average; a uniform distribution over 16 actions would
give each action 6.25%.

Middle-round agreement at 128 is better, but it does not compensate for the
unchanged early-round target. The intended model must help prioritize the
largest early search space.

## Cost

| Budget | Searches | Search wall time | Relative to 32 | Peak RSS |
| ---: | ---: | ---: | ---: | ---: |
| 32 | 30 | 141.7 s | 1.00× | 191.0 MiB |
| 64 | 30 | 262.1 s | 1.85× | 189.5 MiB |
| 128 | 30 | 503.4 s | 3.55× | 191.2 MiB |

The cost scales approximately with simulations while memory remains flat. Both
new processes reported zero swap operations.

## Artifacts

```text
64 results  .local/teacher-v2-target-quality-64.json
64 log      output-teacher-v2-target-64
64 digest   79fd9aeac8e1fdc1ee10613606b2773ce19856f698aeac94d36c55421097855f

128 results .local/teacher-v2-target-quality-128.json
128 log     output-teacher-v2-target-128
128 digest  4731b4bc930c4f0c6d793c7c768e009677acf6d7edc93c848353e28ff90a3ea8
```

## Conclusion

Do not collect a 64- or 128-visit full corpus using normalized visit counts as
the policy target. Paying for more outer simulations does not expose a stable
early-round preference.

The next bounded experiment should inspect the root action-score estimates that
Teacher v2 already computes. If their rankings are stable across request seeds,
derive and validate a policy target from those scores. If the score rankings
are also unstable, increase or aggregate the samples used to estimate each
action rather than increasing UCT visits.

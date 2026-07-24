# Search-training redesign

## Status

This is a historical report for the version 1 teacher design. Its collection
commands and 500-simulation dataset plan are superseded by
[Teacher v2 smoke collection](teacher-v2-smoke-collection.md) and the active
[model-training contract](../docs/model-training.md).

The information-set search opponent, shared policy/value model, deterministic
teacher collector, supervised optimizer, neural-guided PUCT planner, and
single-model expert iterator are validated mechanically. Full-scale data
collection and candidate evaluation remain.

## Rejected PPO experiment

The five-policy PPO experiment did not establish strategic competence. Its best
checkpoint, `policy-2-v20`, remained near random legal performance in the final
tournament and lost both recorded human games by large margins. Relative
population rank and critic validation therefore did not provide an absolute
strength signal.

## Active search

The active opponent uses POMCP-style root sampling through the end of the
current round. Root-player decisions use UCT. Simulated opponent decisions use
uniform legal actions from the opponent's sampled information state. Every
transition and terminal score comes from the deterministic engine.

Search observes the acting player's role-normalized coffin, remaining hand,
public moves, dealer and turn state, completed public scores, unseen-card
membership, and legal actions. It cannot observe the real opponent hand, stock
order, engine seed, opponent hand slots, policy state, or authoritative private
fingerprint.

Each simulation samples opponent cards and irrelevant stock order uniformly
from the unseen-card pool. Sampling begins from `SearchInformationState`, uses
versioned deterministic seeds, conserves all 54 cards, and reconstructs an
engine-valid simulation state. Equal player information produces equal samples
and search results for the same seed.

## Neural learner

One 339,978-parameter feed-forward network will predict:

- 32 raw policy logits over the fixed hand-slot and coffin-position action map.
- One bounded player-relative value estimating normalized terminal round-score
  differential.

Search visit counts will be normalized over legal root actions to form the
policy target. The exact terminal target is:

```text
(acting_player_round_score - other_player_round_score) / 150
```

The same target attaches to each non-forced decision by that player in the
round. Forced placements produce no policy sample.

The initial dataset uses 240 games and 500-simulation search. After warm-start
training, guided self-play uses 100-simulation PUCT with network policy priors
and exact full-round scoring. The value head does not truncate search until a
separate accuracy, strength, and latency gate passes.

## Retained implementation

- Deterministic engine, scoring, serialization, and invariance tests.
- Player-relative information state and hidden-card sampler.
- Search-only UCT planner, exhaustive diagnostic, strategic fixtures, and
  absolute-control evaluation.
- The 339,978-parameter feed-forward policy/value model, external masking,
  dual-head loss, deterministic initialization, and strict artifact contract.
- Deterministic full-game search-teacher collection, public decision caches,
  sealed tensor shards, split manifests, inspection, and phase resume.
- Deterministic supervised optimization, held-out validation, CPU/MPS device
  selection, atomic checkpoint recovery, reports, and model export.
- Information-safe PUCT using model priors, per-actor sampled views, exact
  terminal backup, deterministic diagnostics, and an experimental gated value
  cutoff.
- Historical PPO runs, reports, checkpoints, recurrent model loader, and local
  adapter as immutable comparison evidence.
- FastAPI, SQLite, React gameplay, local search and guided-search executors, and
  recorded local results.

## Removed from active use

- `dracula-train` and `dracula-tournament` package commands.
- Six population-PPO TOML configurations under `configs/`.
- The local archived-control loader's import dependency on the population
  training configuration module.

The PPO collection, optimization, evaluation, and checkpoint modules remain in
the repository only because historical-control tests and the current local
`policy-2-v20` gameplay adapter still require them. They are not active training
architecture.

## Current evidence

At 500 simulations per decision, search passed all 33 seeded strategic fixture
decisions and matched all 21 decisions backed by exhaustive continuation
analysis. On 12 fixed decks with both role assignments it achieved:

| Control | Games | Game wins | Round wins | Mean round differential |
| --- | ---: | ---: | ---: | ---: |
| Random legal | 24 | 24 | 92.4% | +31.32 |
| `policy-2-v20` | 24 | 24 | 86.8% | +30.35 |
| 100-simulation search | 24 | 18 | 61.1% | +5.90 |

Under four concurrent validation workers, 500-simulation fixture decisions had
15.7-second p50 and 26.5-second p95 latency. The isolated process peak was about
203 MiB including Python and PyTorch. The 2,000-simulation budget added no
fixture passes over 500.

One full six-round teacher game produced 42 examples at each measured budget:

| Simulations | Wall time | Examples/hour | Simulations/second | Peak RSS |
| ---: | ---: | ---: | ---: | ---: |
| 100 | 57.9 s | 2,610 | 72.8 | 199 MB |
| 500 | 309.2 s | 489 | 68.0 | 201 MB |

These are single-worker measurements. Four-worker collection is implemented and
validated separately; full-run duration depends on sustained thermal behavior.

A two-game 500-simulation smoke corpus supplied 42 training and 42 held-out
examples. Supervised training selected epoch 28 and stopped after epoch 33 with
a held-out total loss of 2.2213. This validates mechanics only; it is not a
candidate-strength result.

| Optimization device | Examples/second | Median epoch | Peak RSS | Swap growth |
| --- | ---: | ---: | ---: | ---: |
| CPU | 7,756 | 5.4 ms | 297 MB | 0 |
| MPS | 953 | 44.1 ms | 440 MB | 0 |

CPU is the measured default for the current batch shape.

The two-game smoke model was also exercised as guided-search infrastructure.
On one matched seed per strategic fixture, full-round exact search produced:

| Simulations | UCT fixtures | Guided fixtures | UCT mean | Guided mean |
| ---: | ---: | ---: | ---: | ---: |
| 20 | 8/12 | 9/12 | 0.34 s | 0.40 s |
| 50 | 9/12 | 10/12 | 0.87 s | 1.17 s |
| 100 | 10/12 | 12/12 | 2.24 s | 2.35 s |

Peak process RSS was about 203 MB. This is a plumbing and tactical-fixture
result, not candidate-strength evidence. The smoke model's held-out value MSE
was `0.0690`, worse than the zero predictor's `0.0344`; model-value cutoffs
therefore remain disabled without running the later strength and latency gates.

## Commands

Run the complete implemented suite:

```bash
.venv/bin/pytest -q
```

Reproduce search validation from the sealed caches, or recompute it after
removing `runs/search-validation-001`:

```bash
.venv/bin/dracula-search-validate \
  --output-directory runs/search-validation-001 \
  --policy-archive runs/training-004/archives/policy-2-policy-2-v20.pt \
  --workers 4 \
  --sensitivity-repetitions 3 \
  --game-pairs 12
```

Collect the full initial teacher corpus and run supervised warm-start training:

```bash
.venv/bin/dracula-teacher full \
  --output runs/search-teacher-001 \
  --root-seed dracula-search-teacher-warmstart-v1

.venv/bin/dracula-supervised train --config configs/search-warmstart.toml
```

`dracula-supervised resume`, `validate`, and `export` operate from the immutable
resolved run directory. `configs/search-warmstart-smoke.toml` reproduces the
mechanical smoke run.

Run local gameplay against the search opponent:

```bash
make preview
```

Run the historical comparison control with `make preview-control`.

## Remaining measured limitations

- The full 240-game teacher corpus and candidate warm start have not run.
- The expert-iteration smoke and sustained trials are below the contractual
  collection and absolute-evaluation sample sizes.
- A guided candidate completed local gameplay and retained strategic behavior
  at 100 simulations, but has not established full-game non-inferiority to the
  500-simulation teacher.
- The validated 500-simulation Python search is too slow for an unqualified
  interactive deployment claim.

## Readiness assessment

The complete training path is ready for the full sealed teacher corpus and
contract-size expert run. No guided candidate is ready for acceptance until the
full absolute controls establish the required strength and latency result.

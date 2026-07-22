# Search-training redesign

## Status

The information-set search opponent is validated. Search-guided neural training
is not ready because the shared policy/value architecture and guided-search
training contract remain incomplete.

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

## Planned neural learner

One feed-forward network will predict:

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

## Retained implementation

- Deterministic engine, scoring, serialization, and invariance tests.
- Player-relative information state and hidden-card sampler.
- Search-only UCT planner, exhaustive diagnostic, strategic fixtures, and
  absolute-control evaluation.
- Historical PPO runs, reports, checkpoints, recurrent model loader, and local
  adapter as immutable comparison evidence.
- FastAPI, SQLite, React gameplay, the local search executor, and recorded local
  results.

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

There is no active search-guided training command. The retired PPO command was
removed rather than presented as the new trainer.

Run local gameplay against the search opponent:

```bash
make preview
```

Run the historical comparison control with `make preview-control`.

## Remaining measured limitations

- Embedding widths, shared-body shape, normalization, exact initialization,
  parameter count, and loss weights are not finalized.
- Neural-guided information-set selection, batched leaf evaluation, root
  exploration, and move-temperature behavior are not specified or implemented.
- Replay, dual-head optimization, atomic resume, CPU/MPS comparison, and a
  sustained search-guided trial do not exist.
- No search-guided neural checkpoint exists for absolute evaluation or local
  gameplay. Search-only gameplay is available.
- The validated 500-simulation Python search is too slow for an unqualified
  interactive deployment claim.

## Readiness assessment

Not ready for sustained search-guided self-play. The exact blockers are the
unfinished neural architecture, guided-search contract, replay/training system,
and search-backed application adapter. Engine and search-only readiness gates
are complete.

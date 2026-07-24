# Expert-iteration readiness

## Result

This is a historical readiness report for the version 1 teacher path. Its
500-simulation collection command and dataset references are superseded by
[Teacher v2 training readiness](teacher-v2-training-readiness.md).

The single-model expert-iteration suite is implemented end to end. Collection,
replay, optimization, checkpoint recovery, absolute evaluation, manual
acceptance, export, and local guided gameplay are mechanically ready. No new
candidate is accepted: the full teacher corpus, full warm start, and contractual
absolute-evaluation samples have not been run.

## Implemented path

One accepted policy/value artifact serves both roles. Each iteration freezes
that artifact, collects role-balanced guided-search games, stores normalized
search visits and exact player-relative round returns, samples the permanent
teacher plus the five latest accepted iterations, and trains one candidate.
Collection, optimization, and evaluation commit atomically. Resume preserves
fixtures, replay order, optimizer state, model state, and CPU results.

Evaluation compares identical decks in both role assignments against random
legal play, `policy-2-v20`, frozen search-only play, and retained accepted
checkpoints. It also runs every non-forced strategic fixture and reports paired
bootstrap intervals. Acceptance and rejection are explicit commands.

## Verification

- Python: 293 tests passed, including interruption and exact-resume coverage
  for every expert-iteration phase.
- Frontend: 70 tests, TypeScript, lint, and the production-build privacy check
  passed.
- Browser: the six-test application suite passed. A separate guided-search game
  completed all six rounds through the production frontend; its network and
  browser-state privacy assertions passed.
- Guided game evidence: Dracula won 399-224, with five of six round wins. This
  is an integration result, not a strength estimate.
- Package imports, bytecode compilation, dependency checks, Markdown links,
  tracker-ID placement, and diff formatting passed.

The three-game sustained trial collected 126 examples from 12,600 simulations
in 184.2 seconds. Median and p95 collection decision latency were 1.86 and 5.44
seconds under two workers; peak RSS was 198 MiB. Five CPU epochs selected epoch
3 at held-out loss 2.2738. Its one-pair-per-control evaluation was deliberately
too small for acceptance:

| Control | Games | Candidate wins | Mean round differential |
| --- | ---: | ---: | ---: |
| Accepted smoke model | 2 | 1 | -0.83 |
| `policy-2-v20` | 2 | 2 | +35.25 |
| Random legal | 2 | 2 | +39.42 |
| Search-only, 100 simulations | 2 | 1 | +2.58 |

At 100 simulations, the sustained candidate passed all 11 non-forced strategic
fixtures. The same candidate passed 7/11 at 20 simulations and 10/11 at 50.
Single-process fixture latency was 2.05 seconds p50 and 4.08 seconds p95 at 100
simulations. This preserves the tested tactical behavior at one fifth of the
teacher budget, but does not establish non-inferiority to the 500-simulation
teacher across full games.

CPU remains the optimization default: 7,391 examples/s versus 845 on MPS for
the measured replay shape. CPU used 287 MiB peak RSS with no swap growth; MPS
used 441 MiB and added 105 MiB of swap.

## Commands

Produce the full teacher and warm-start artifacts:

```bash
.venv/bin/dracula-teacher full \
  --output runs/search-teacher-001 \
  --root-seed dracula-search-teacher-warmstart-v1
.venv/bin/dracula-supervised train --config configs/search-warmstart.toml
```

Run the full configured iteration:

```bash
.venv/bin/dracula-expert run --config configs/expert.toml
.venv/bin/dracula-expert resume --output runs/expert-001
.venv/bin/dracula-expert evaluate --output runs/expert-001
```

Make the manual decision and export an accepted artifact:

```bash
.venv/bin/dracula-expert accept --output runs/expert-001
.venv/bin/dracula-expert reject --output runs/expert-001
.venv/bin/dracula-expert export \
  --output runs/expert-001 \
  --destination runs/accepted-policy-value.pt
```

Run the validated candidate locally behind guided search:

```bash
make preview OPPONENT=guided \
  GUIDED_ARTIFACT=/absolute/path/to/candidate-model.pt \
  GUIDED_SIMULATIONS=100
```

Run verification:

```bash
.venv/bin/pytest -q
npm --prefix frontend run check
npm --prefix frontend run test:e2e
```

Run one complete guided browser game with the selected smoke candidate:

```bash
DRACULA_E2E_OPPONENT=guided \
DRACULA_POLICY_VALUE_ARTIFACT=/absolute/path/to/candidate-model.pt \
DRACULA_GUIDED_SIMULATIONS=100 \
npm --prefix frontend run test:e2e -- \
  --grep 'completes all six rounds as Queen'
```

## Remaining blockers

1. Collect the contract-size 240-game, 500-simulation teacher corpus and train
   the full warm-start model.
2. Run at least one 108-training-game and 12-validation-game expert iteration.
3. Run the configured 12-pair random and PPO controls and 60-pair search and
   prior-accepted controls. The smoke intervals are not evidence.
4. Confirm that guided search is non-inferior to frozen 500-simulation search
   at full-game scale while retaining its latency advantage.

The suite is ready to produce this evidence. The candidate-readiness gate is
not yet passed, so local gameplay should continue to default to frozen
500-simulation search.

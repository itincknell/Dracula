# Dracula

Dracula is a six-round browser card game backed by a deterministic Python engine.
Local gameplay uses SQLite, FastAPI, and the same React client contracts intended
for deployment. The default local opponent is the validated information-set
search controller.

## Local gameplay

The search opponent runs 500 simulations per learned move by default.

Start the development servers:

```bash
make dev
```

Start FastAPI with a production frontend build served by Vite preview:

```bash
make preview
```

Adjust the search budget when measuring strength and latency:

```bash
make preview SEARCH_SIMULATIONS=100
```

Run the experimental response-ranker hybrid with:

```bash
make preview OPPONENT=search-v2-student-top-2 \
  SEARCH_SIMULATIONS=32 \
  SEARCH_RESPONSE_COMPLETIONS=4
```

Its pure Teacher v2 comparison control uses the same search settings with
`OPPONENT=search-v2`. The hybrid requires a compatible response-ranker artifact;
override `RESPONSE_RANKER_ARTIFACT` to select another one explicitly.

Run the archived `policy-2-v20` comparison control with `make preview-control`.
Another compatible archive can be selected with
`make preview-control CANDIDATE=/absolute/path/to/policy.pt`.

Run a policy/value artifact behind guided search with:

```bash
make preview OPPONENT=guided \
  GUIDED_ARTIFACT=/absolute/path/to/policy-value.pt \
  GUIDED_SIMULATIONS=100
```

Open <http://127.0.0.1:5173> for development or
<http://127.0.0.1:4173> for preview. Stop either command with `Ctrl-C`; its API
process is stopped by the command trap.

For separate development terminals, run `make dev-api` and
`make dev-frontend`. Stop either process with `Ctrl-C`.
`API_PORT`, `DEV_PORT`, and `PREVIEW_PORT` may be overridden when a default
local port is occupied.

Reset local game state after stopping the servers:

```bash
make reset-local
```

Run the complete local checks:

```bash
.venv/bin/pytest -q
npm --prefix frontend run check
make test-e2e
```

The archived-control inference profile defaults to masked argmax. Set
`INFERENCE_PROFILE=sample-temperature-1-v1` only when intentionally testing the
deterministic sampled profile.

## Expert iteration

The recoverable expert-iteration command supports `smoke`, `run`, `resume`,
`evaluate`, `accept`, `reject`, and `export`:

```bash
.venv/bin/dracula-expert smoke --config configs/expert-smoke.toml
.venv/bin/dracula-expert resume --output runs/expert-smoke-001
.venv/bin/dracula-expert evaluate --output runs/expert-smoke-001
.venv/bin/dracula-expert accept --output runs/expert-smoke-001
.venv/bin/dracula-expert reject --output runs/expert-smoke-001
.venv/bin/dracula-expert export \
  --output runs/expert-smoke-001 \
  --destination runs/accepted-policy-value.pt
```

`configs/expert.toml` contains the full 108-training-game, 12-validation-game,
100-simulation iteration and permanent absolute-control sample sizes. Candidate
acceptance is always an explicit command.

## Recorded game results

Completed and in-progress games remain in `.local/dracula.sqlite3`. Print the
aggregate Markdown report without stopping the application:

```bash
make play-report
```

Redirect that command to save a snapshot. Inspect one game's six round results
using the full game ID from the aggregate report:

```bash
make play-game GAME_ID=00000000-0000-0000-0000-000000000000
```

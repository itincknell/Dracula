# Dracula

Dracula is a six-round browser card game backed by a deterministic Python
engine. Local gameplay uses SQLite, FastAPI, and React. The selected release
opponent is the standalone `pi1` feed-forward policy.

## Repository map

| Path | Ownership |
| --- | --- |
| `src/`, `tests/` | Python engine, search, service, miner, and regression suites |
| `frontend/` | React application, selected card assets, and browser tests |
| `docs/`, `contracts/` | Active design contracts and versioned public API fixtures |
| `configs/`, `tools/` | Indexed historical experiment inputs and reproducibility utilities |
| `queries/` | Read-only local gameplay reports |
| `reports/` | Active evidence, historical experiment indexes, and maintenance handoffs |

`Cards (large)`, `Cards (medium)`, and `Dracula.png` are retained source
assets; runtime card assets live under `frontend/public/cards/`. Ignored
`.local/`, `runs/`, and `output*` paths contain protected local state,
collection artifacts, archives, and logs rather than project source.

## Local gameplay

Start FastAPI and the Vite development server:

```bash
make dev
```

Start FastAPI with a production frontend build:

```bash
make preview
```

Open <http://127.0.0.1:5173> for development or
<http://127.0.0.1:4173> for preview. Stop either command with `Ctrl-C`.

`pi1` is the default. Nested Sam remains an explicit comparison:

```bash
make preview-sam
```

Run the historical standalone Sam imitation with `make preview-policy`, or the
archived neural-policy control with `make preview-control`. Select another
implemented controller only by setting `OPPONENT`; invalid or incomplete
controller configuration fails instead of falling back to `pi1`.

For separate development terminals, use `make dev-api` and
`make dev-frontend`. `API_PORT`, `DEV_PORT`, and `PREVIEW_PORT` may be
overridden. Reset local game state after stopping the servers with
`make reset-local`.

## Verification

```bash
make test
make test-e2e
```

## Historical Sam-32 corpus

The stopped corpus remains reproducible from an ignored frozen checkout. It
used symmetry-aware Sam-32, four deck workers, four children when available, a
repeating 27/2/2 training/validation/test split, and a 1 GiB free-disk guard.

```bash
make corpus-health
make corpus-inspect
make corpus-verify
make corpus-stop
make corpus-resume
```

These commands are retained for historical inspection and controlled resume.

## Recorded games

Completed and in-progress games remain in `.local/dracula.sqlite3`.

```bash
make play-report
make play-game GAME_ID=00000000-0000-0000-0000-000000000000
```

The [design index](docs/README.md), [active tracker](docs/design-tracker.md),
and [report index](reports/README.md) describe the current contracts and
historical evidence.

## Deployment target

The locked release serves the frontend from
`https://ian-tincknell.com/Dracula/` on GitHub Pages and the API from
`https://api.ian-tincknell.com/` through API Gateway and a Lambda container.
Gameplay is stateless: the browser carries the initial seed and accepted
command history, and Lambda replays or locally caches the reconstructed state.
Direct Bedrock narration runs only at opening, rounds 1–5 transitions, and the
final game result. See [deployment](docs/deployment.md).

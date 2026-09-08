# Dracula

Dracula is a six-round browser card game backed by a deterministic Python
engine. The active local and production-shaped paths use stateless FastAPI,
React, and the standalone `pi1` feed-forward policy.

## Repository map

| Path | Ownership |
| --- | --- |
| `src/dracula/game/` | Cards, immutable engine state, rules, scoring, and lifecycle |
| `src/dracula/decision/`, `src/dracula/search/` | Player-visible decisions, symmetry, and retained information-set UCT |
| `src/dracula/policy/` | Standalone policy runtime, artifact, and offline training pipeline |
| `src/dracula/api/` | Stateless gameplay, narration, local preview, and production composition |
| `tests/` | Python regression and privacy suites |
| `frontend/` | React application, selected card assets, and browser tests |
| `docs/` | Active design and operating contracts |
| `tools/` | Release and deployment utilities |

`Cards (large)` and `Cards (medium)` are retained card-source assets; runtime
card and Dracula portrait assets live under `frontend/public/`. Ignored
`runs/bgc-policy-pi1-001/artifacts/pi1-policy.pt` is the ignored selected model
artifact required by local play and release builds. Rebuildable output remains
ignored rather than part of the repository.

## Local gameplay

Start FastAPI and the Vite development server:

```bash
make dev
```

Start FastAPI with a production frontend build:

```bash
make preview
```

Exercise the approved narration timing and portrait states with deterministic
local dialogue:

```bash
make preview-dialogue
```

Exercise the same preview with live Amazon Bedrock narration after installing
the optional AWS SDK and configuring credentials through Boto3's normal
credential chain:

```bash
.venv/bin/pip install -e '.[bedrock]'
make preview-bedrock
```

This calls Amazon Nova Lite in `us-east-1`; gameplay and policy inference remain
local. The AWS identity needs `bedrock:InvokeModel` permission. See the
[narrator contract](docs/narrator.md#local-bedrock-preview).

Open <http://127.0.0.1:5173> for development or
<http://127.0.0.1:4173> for preview. Stop either command with `Ctrl-C`.

The selected standalone `pi1` artifact is the only gameplay controller. A
missing or incompatible artifact fails explicitly; there is no fallback.

For separate development terminals, use `make dev-api` and
`make dev-frontend`. `API_PORT`, `DEV_PORT`, and `PREVIEW_PORT` may be
overridden. Stop any preview with `Ctrl-C`.

## Verification

```bash
make test
make test-e2e
make web-build
make web-test
```

`web-build` verifies the `/Dracula/` asset base, production API origin, and
absence of local URLs. `web-test` runs a complete stateless six-round browser
game with reload and narration coverage; neither command publishes anything.

The [design index](docs/README.md) describes the current contracts and
operating instructions.

## Deployment target

The locked release serves the frontend from
`https://ian-tincknell.com/Dracula/` through Cloudflare, API Gateway, and one
Lambda container. FastAPI serves both the built frontend and `/Dracula/api`.
The personal website remains on GitHub Pages; Dracula has no Pages release.
Gameplay is stateless: the browser carries the initial seed and accepted
command history, and Lambda replays or locally caches the reconstructed state.
Direct Bedrock narration runs only at opening, rounds 1–5 transitions, and the
final game result. See [deployment](docs/deployment.md) and the implemented
[stateless gameplay API](docs/stateless-api.md).

The frontend workflow validates without publishing. Use
`make preview-app NARRATION_ENABLED=true` for the combined FastAPI application
at <http://127.0.0.1:4173/Dracula/> with real Bedrock narration.
The [deployment contract](docs/deployment.md) covers build, staging, public
cutover, rollback, and teardown.

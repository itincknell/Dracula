# Bottom-up code-review agenda

## Review basis

This agenda describes the standalone `pi1` repository after removal of retired
model and search implementations. The cleanup begins from baseline commit
`fbc2fceaf3bb95983fc26252cf373cbc26ae0a92`; the final cleanup commit is added
after validation.

Current maintained surface before the final commit:

| Area | Modules/files | Lines |
| --- | ---: | ---: |
| Python package | 43 | 10,365 |
| Python tests | 14 files including the browser server fixture | 3,284 |
| Frontend runtime and styles | 29 | 4,469 |
| Frontend tests and fixtures | 13 test files | covered by 99 tests |

Selected artifact SHA-256:
`70c76f2eb64600eab2297640278a6c94d4336ab8e73bf941a6d96237f69f5b5c`.

> **Comment rule:** Comments explain non-obvious intent, invariants,
> constraints, or reasons. They remain concise and professional. They do not
> narrate straightforward code.

Every tranche includes an inline-comment and docstring review before approval.

## Approval ledger

- [x] NORTHSTARS, rules, and cards.
- [x] Engine types, scoring, and dealing.
- [x] State validation, serialization, transitions, and lifecycle.
- [ ] Everything after the engine boundary.

## 1. Rules and cards — approved

- [x] [NORTHSTARS](/Users/itincknell/Projects/Dracula/NORTHSTARS) — project
  priorities and engineering constraints.
- [x] [rules.md](/Users/itincknell/Projects/Dracula/docs/rules.md) — authoritative
  game rules.
- [x] [cards.py](/Users/itincknell/Projects/Dracula/src/dracula/cards.py) — 114
  lines; immutable card identities, indexes, validation, and canonical sorting.
  Inputs are card IDs; outputs are frozen `Card` values or canonical tuples.
  It has no private/public distinction and is covered by `test_foundation.py`.

## 2. Engine types and scoring — approved

- [x] [engine_types.py](/Users/itincknell/Projects/Dracula/src/dracula/engine_types.py)
  — 276 lines; engine enums, immutable values, exceptions, adjacency, and
  transitions. It is imported throughout the engine and services. Frozen state
  shape and zero-based row-major coffin indexing are key invariants.
- [x] [scoring.py](/Users/itincknell/Projects/Dracula/src/dracula/scoring.py) —
  179 lines; exact line arithmetic, Vampire override, multiplier priority, and
  ranked tie resolution. Coffin cards enter; immutable score structures leave.
  `test_engine.py` and `test_foundation.py` cover it.
- [x] [engine_dealing.py](/Users/itincknell/Projects/Dracula/src/dracula/engine_dealing.py)
  — 94 lines; deterministic shuffle, dealer selection, and two-card packet
  deal. It depends on cards and randomness; engine validation and lifecycle
  call it. Packet order and canonical hands are compatibility-sensitive.

## 3. Validation, serialization, transitions, and lifecycle — approved

- [x] [engine_validation.py](/Users/itincknell/Projects/Dracula/src/dracula/engine_validation.py)
  — 376 lines; complete lifecycle, deal, card-conservation, move-history, and
  score validation. It handles private `EngineState`; callers receive success
  or typed violations. Complexity is intentional because all invariants meet
  at this boundary.
- [x] [engine_serialization.py](/Users/itincknell/Projects/Dracula/src/dracula/engine_serialization.py)
  — 125 lines; canonical private serialization and fingerprints. Dictionary
  order and exact JSON encoding are compatibility invariants.
- [x] [engine.py](/Users/itincknell/Projects/Dracula/src/dracula/engine.py) — 323
  lines; gameplay facade for creation, legal moves, immutable transitions,
  round advancement, and re-exports. Inputs and outputs are private immutable
  engine values. `test_engine.py` holds golden deal, legality, scoring, and
  lifecycle coverage.

## 4. Deterministic replay and privacy — next review tranche

Review in this order:

| Review | Module | Purpose and public interface | Boundaries, tests, and risk |
| --- | --- | --- | --- |
| [ ] | [randomness.py](/Users/itincknell/Projects/Dracula/src/dracula/randomness.py) — 120 lines | `derive_seed`, counter stream, unbiased `randbelow`, deterministic shuffle | Text/integers in; bytes or deterministic choices out. Seed namespace and rejection sampling are non-obvious compatibility rules. Covered by `test_foundation.py`. |
| [ ] | [bridge.py](/Users/itincknell/Projects/Dracula/src/dracula/bridge.py) — 258 lines | Player-relative 32-action mapping, `PolicyTurnContext`, forced/learned turn resolution | Engine state stays private; action table is the legal bridge. Ordering and Queen/King transpose are invariant. Covered by `test_bridge.py`. |
| [ ] | [search/information.py](/Users/itincknell/Projects/Dracula/src/dracula/search/information.py) — 540 lines | `SearchInformationState`, public history, projection, validation, canonical fingerprint | Erases opponent slots, stock order, and seed. Played-card union must remain globally canonical because it affects deterministic paired moves. Covered by `test_search_information.py`. |
| [ ] | [search/__init__.py](/Users/itincknell/Projects/Dracula/src/dracula/search/__init__.py) — 39 lines | Narrow public exports for information and symmetry only | Must not re-export removed search planners. Import tests cover it. |

Flow:

```text
EngineState --validated projection--> SearchInformationState
     |                                      |
     `-- canonical replay/fingerprint ------'
```

## 5. Symmetry, pi1, and move selection

| Review | Module | Purpose and public interface | Boundaries, tests, and risk |
| --- | --- | --- | --- |
| [ ] | [search/symmetry.py](/Users/itincknell/Projects/Dracula/src/dracula/search/symmetry.py) — 165 lines | Exhaustive seven-pattern destination grouping | Occupancy and legal destinations in; designated proxies/groups out. No inferred geometry is permitted. Covered by `test_search_symmetry.py`. |
| [ ] | [strategic_actions.py](/Users/itincknell/Projects/Dracula/src/dracula/strategic_actions.py) — 243 lines | Card-specific strategic groups and deterministic paired-member selection | Accepts actor-visible information only. Distinct cards never merge; groups partition legality exactly. Covered by symmetry and model tests. |
| [ ] | [action_contract.py](/Users/itincknell/Projects/Dracula/src/dracula/action_contract.py) — 310 lines | Representative masks, proxy mapping, masked argmax, fair-coin primitives | Legal/group tensors in; representative selection out. Masked actions must have zero probability and canonical ties. Covered by symmetry/model/trainer tests. |
| [ ] | [policy_observation.py](/Users/itincknell/Projects/Dracula/src/dracula/policy_observation.py) — 135 lines | Direct `bool[659]` encoding and candidate-to-engine row mapping | Typed actor view in; tensor or action index out. No permanent slot feature enters the model. Pre-cleanup hashes and round trips are tested. |
| [ ] | [bgc_policy_model.py](/Users/itincknell/Projects/Dracula/src/dracula/bgc_policy_model.py) — 225 lines | `BGCPolicyModel`, deterministic initialization, card candidates, `float32[4,8]` logits | 659-bit visible observation in; raw logits out. Exact 754,601 parameters and no value head are invariant. Covered by `test_bgc_policy_model_v2.py`. |
| [ ] | [bgc_policy.py](/Users/itincknell/Projects/Dracula/src/dracula/bgc_policy.py) — 378 lines | Strict model metadata, state-dictionary digest, atomic artifact save/load | Artifact bytes are private server data. Exact schema equality prevents guessed migrations. Covered by model, trainer, packaging tests. |
| [ ] | [active_policy.py](/Users/itincknell/Projects/Dracula/src/dracula/active_policy.py) — 225 lines | Selected artifact runtime and `ActivePolicyExecutor` | One actor-visible inference selects a proxy; fair coin resolves concrete move; engine action table revalidates it. The old seed literal is retained solely for exact behavior. |

Model boundary:

```text
SearchInformationState -> bool[659] -> pi1 -> logits[4,8]
        + legal groups -> representative mask -> masked argmax
        + typed own hand -> engine slot -> deterministic paired destination
```

## 6. Stateless API and narration

| Review | Module | Purpose and public interface | Boundaries, tests, and risk |
| --- | --- | --- | --- |
| [ ] | [api/policy.py](/Users/itincknell/Projects/Dracula/src/dracula/api/policy.py) — 84 lines | Controller-neutral executor/request/result protocol | `SearchInformationState` crosses to pi1; engine state does not. A fixed empty hidden-state field remains only for stateful-record compatibility. |
| [ ] | [api/stateless_contracts.py](/Users/itincknell/Projects/Dracula/src/dracula/api/stateless_contracts.py) — 243 lines | Recovery envelope, commands, public game view, narration wire models | Public JSON only. Seed/history are intentionally client-visible; model and engine-private fields are excluded. |
| [ ] | [api/stateless_http.py](/Users/itincknell/Projects/Dracula/src/dracula/api/stateless_http.py) — 81 lines | Bounded request-body middleware and uniform stateless errors | Raw HTTP in; validated bounded body/error out. Chunk handling is the main subtlety. |
| [ ] | [api/stateless_projection.py](/Users/itincknell/Projects/Dracula/src/dracula/api/stateless_projection.py) — 128 lines | Private-engine to public-human projection | Opponent hand/actions never cross. Covered by stateless API privacy tests. |
| [ ] | [api/stateless_service.py](/Users/itincknell/Projects/Dracula/src/dracula/api/stateless_service.py) — 440 lines | Replay cache, canonical envelope digest, replay, commands, opponent turns | Seed/history commands in; reconstructed private state stays internal; public response out. Cache is optional and must not affect correctness. |
| [ ] | [api/stateless_routes.py](/Users/itincknell/Projects/Dracula/src/dracula/api/stateless_routes.py) — 139 lines | `/health`, `/games`, `/games/command`, `/games/resume`, `/narration` handlers | Thin transport over service/narration boundaries. Covered by `test_stateless_gameplay_api.py` and narration tests. |
| [ ] | [api/stateless_app.py](/Users/itincknell/Projects/Dracula/src/dracula/api/stateless_app.py) — 41 lines | Production-neutral FastAPI assembly | Explicit policy and narration dependencies; no database fallback. |
| [ ] | [api/narration.py](/Users/itincknell/Projects/Dracula/src/dracula/api/narration.py) — 483 lines | Three cue classes, grounded public facts, prompt construction, output safety | Replays envelope but sends only public cue facts. Narration failure cannot alter gameplay. Complexity centers on exact score-movement language. |
| [ ] | [api/bedrock.py](/Users/itincknell/Projects/Dracula/src/dracula/api/bedrock.py) — 180 lines | Nova request payload, bounded Bedrock call, plain-text parsing | Public prompt in; optional bounded text out. Credentials/provider failures remain outside gameplay transaction. |
| [ ] | [api/production_config.py](/Users/itincknell/Projects/Dracula/src/dracula/api/production_config.py) — 70 lines | Strict environment parsing for artifact, narration, and replay cache | No account IDs or secrets are defaults. |
| [ ] | [api/production.py](/Users/itincknell/Projects/Dracula/src/dracula/api/production.py) — 33 lines | Lambda FastAPI entry point | Loads the pinned artifact and optional Bedrock adapter; never selects a fallback controller. |
| [ ] | [production_logging.py](/Users/itincknell/Projects/Dracula/src/dracula/production_logging.py) — 30 lines | Structured JSON formatter | Log records in; one JSON object out. Logs must omit seed/history/private/model data. |

### Narrow local compatibility API

These modules preserve recorded SQLite games and transaction tests but are not
copied into the Lambda image:

| Review | Module | Ownership and boundary |
| --- | --- | --- |
| [ ] | [api/contracts.py](/Users/itincknell/Projects/Dracula/src/dracula/api/contracts.py) — 289 lines | Shared public score/view primitives plus local-stateful wire models. |
| [ ] | [api/presentation.py](/Users/itincknell/Projects/Dracula/src/dracula/api/presentation.py) — 230 lines | Shared scoring presentation projection. |
| [ ] | [api/session.py](/Users/itincknell/Projects/Dracula/src/dracula/api/session.py) — 451 lines | Validated local persisted session encoding; private SQLite data only. |
| [ ] | [api/repository.py](/Users/itincknell/Projects/Dracula/src/dracula/api/repository.py) — 371 lines | In-memory and SQLite transactional storage; local only. |
| [ ] | [api/service.py](/Users/itincknell/Projects/Dracula/src/dracula/api/service.py) — 748 lines | Local stateful transaction orchestration; no production caller. |
| [ ] | [api/local_controllers.py](/Users/itincknell/Projects/Dracula/src/dracula/api/local_controllers.py) — 42 lines | Explicit local resolution of `bgc-policy` only. |
| [ ] | [api/app.py](/Users/itincknell/Projects/Dracula/src/dracula/api/app.py) — 243 lines | Explicit local-stateful/stateless assembly; normal Make targets select stateless. |
| [ ] | [api/local_preview.py](/Users/itincknell/Projects/Dracula/src/dracula/api/local_preview.py) — 95 lines | Stateless preview with deterministic dummy dialogue. |
| [ ] | [api/__init__.py](/Users/itincknell/Projects/Dracula/src/dracula/api/__init__.py) — 16 lines | Lazy local app exports; no eager server construction. |

## 7. Frontend state and presentation

Review runtime files in this order:

| Review | Module | Purpose, boundaries, and coverage |
| --- | --- | --- |
| [ ] | [contractPrimitives.ts](/Users/itincknell/Projects/Dracula/frontend/src/contractPrimitives.ts) — 281 lines | Shared public cards, scores, phases, and strict validators. |
| [ ] | [statefulContracts.ts](/Users/itincknell/Projects/Dracula/frontend/src/statefulContracts.ts) — 281 lines | Shared `HumanGameView` presentation shape and retained local API validators. |
| [ ] | [statelessContracts.ts](/Users/itincknell/Projects/Dracula/frontend/src/statelessContracts.ts) — 277 lines | Seed/history envelope, commands, stateless response and narration validation. |
| [ ] | [gameControllerContract.ts](/Users/itincknell/Projects/Dracula/frontend/src/gameControllerContract.ts) — 49 lines | View-facing controller interface used by presentation components. |
| [ ] | [statelessApi.ts](/Users/itincknell/Projects/Dracula/frontend/src/statelessApi.ts) — 127 lines | Fetch transport, JSON/error validation, configured origin. |
| [ ] | [statelessRecovery.ts](/Users/itincknell/Projects/Dracula/frontend/src/statelessRecovery.ts) — 42 lines | Browser persistence of seed and accepted history only. |
| [ ] | [statelessProjection.ts](/Users/itincknell/Projects/Dracula/frontend/src/statelessProjection.ts) — 112 lines | Converts stateless public response into the presentation view. |
| [ ] | [statelessNarration.ts](/Users/itincknell/Projects/Dracula/frontend/src/statelessNarration.ts) — 87 lines | Cue eligibility, stale-response protection, and failure isolation. |
| [ ] | [statelessGameStore.ts](/Users/itincknell/Projects/Dracula/frontend/src/statelessGameStore.ts) — 374 lines | New game, commands, replay, persistence, and narration sequencing. Immediate human placement and minimum opponent delay are key timing invariants. |
| [ ] | [scoringStateMachine.ts](/Users/itincknell/Projects/Dracula/frontend/src/scoringStateMachine.ts) — 300 lines | Pure scoring-animation timeline and transitions. |
| [ ] | [cardAssets.ts](/Users/itincknell/Projects/Dracula/frontend/src/cardAssets.ts) — 44 lines | Immutable runtime card asset mapping. |
| [ ] | [siteConfig.ts](/Users/itincknell/Projects/Dracula/frontend/src/siteConfig.ts) — 24 lines | `/Dracula/` links and site destinations. |
| [ ] | [CardFace.tsx](/Users/itincknell/Projects/Dracula/frontend/src/CardFace.tsx) — 5 lines | Shared card image component. |
| [ ] | [DraculaCommentary.tsx](/Users/itincknell/Projects/Dracula/frontend/src/DraculaCommentary.tsx) — 172 lines | Portrait state, typed text, and final-loss face timing. |
| [ ] | [ScoringPresentation.tsx](/Users/itincknell/Projects/Dracula/frontend/src/ScoringPresentation.tsx) — 377 lines | Scoring sequence, final score card, round/game continuation controls. |
| [ ] | [gameplay.tsx](/Users/itincknell/Projects/Dracula/frontend/src/gameplay.tsx) — 256 lines | Board, hand, role/status, cheat sheet, and public controls. |
| [ ] | [RulesPage.tsx](/Users/itincknell/Projects/Dracula/frontend/src/RulesPage.tsx) — 114 lines | Static public rules page. |
| [ ] | [App.tsx](/Users/itincknell/Projects/Dracula/frontend/src/App.tsx) — 273 lines | Hash route, controller creation, game/start/rules composition. |
| [ ] | [main.tsx](/Users/itincknell/Projects/Dracula/frontend/src/main.tsx) — 21 lines | Browser bootstrap and font/style imports. |

`api.ts` (135 lines) and `gameStore.ts` (269 lines) are retained only as local
stateful test adapters. They are not imported by the production application.
`contracts.ts` (46 lines) is the test-facing compatibility export surface.

Presentation styles, reviewed in cascade order:

- [ ] [styles.css](/Users/itincknell/Projects/Dracula/frontend/src/styles.css) — 5 lines.
- [ ] [base.css](/Users/itincknell/Projects/Dracula/frontend/src/styles/base.css) — 201 lines.
- [ ] [gameplay.css](/Users/itincknell/Projects/Dracula/frontend/src/styles/gameplay.css) — 326 lines.
- [ ] [scoring.css](/Users/itincknell/Projects/Dracula/frontend/src/styles/scoring.css) — 110 lines.
- [ ] [rules.css](/Users/itincknell/Projects/Dracula/frontend/src/styles/rules.css) — 26 lines.
- [ ] [responsive.css](/Users/itincknell/Projects/Dracula/frontend/src/styles/responsive.css) — 120 lines.

The approved desktop/mobile geometry, animation timing, no-auto-scroll rule,
portrait behavior, and score alignment are invariants. Component, usability,
contract, state-machine, and Playwright suites cover them.

## 8. Deployment and operations

- [ ] [Dockerfile](/Users/itincknell/Projects/Dracula/deployment/container/Dockerfile)
  and pinned requirement files — Lambda Web Adapter, Python runtime, embedded
  `pi1`, read-only runtime.
- [ ] [build_lambda_context.py](/Users/itincknell/Projects/Dracula/tools/build_lambda_context.py)
  — verifies the exact artifact digest and copies only the 34-module production
  import closure.
- [ ] [application.yaml](/Users/itincknell/Projects/Dracula/infrastructure/application.yaml)
  and [parameters](/Users/itincknell/Projects/Dracula/infrastructure/parameters/staging.json)
  — Regional HTTP API, Lambda, Bedrock permission, logs, alarms, and budgets.
- [ ] [validate_lambda_container.py](/Users/itincknell/Projects/Dracula/tools/validate_lambda_container.py)
  — local cold/warm/replay/container validation.
- [ ] [build_release_candidate.py](/Users/itincknell/Projects/Dracula/tools/build_release_candidate.py)
  — source, model, frontend, image, and infrastructure release identity.
- [ ] [pages.yml](/Users/itincknell/Projects/Dracula/.github/workflows/pages.yml)
  — pull-request verification and explicit manual Pages publication.
- [ ] [Makefile](/Users/itincknell/Projects/Dracula/Makefile),
  [pyproject.toml](/Users/itincknell/Projects/Dracula/pyproject.toml), and
  [package.json](/Users/itincknell/Projects/Dracula/frontend/package.json) —
  one current local, test, training, and release command surface.

External boundaries to check line by line: Bedrock Runtime request, API Gateway
request/response, Cloudflare CNAME, GitHub Pages publication, and browser local
storage. No production database, SageMaker endpoint, signature, or hidden
fallback exists.

## 9. Tests, retained trainer, compatibility reader, docs, and rollback

### Current training support

- [ ] [bgc_policy_migration.py](/Users/itincknell/Projects/Dracula/src/dracula/bgc_policy_migration.py)
  — 260 lines; narrow read-only verifier for the physically converted 659-bit
  selected corpus. It accepts no historical source-row format.
- [ ] [source_identity.py](/Users/itincknell/Projects/Dracula/src/dracula/source_identity.py)
  — 66 lines; Git revision and tracked-source digest for trainer artifacts.
- [ ] [bgc_policy_training.py](/Users/itincknell/Projects/Dracula/src/dracula/bgc_policy_training.py)
  — 1,476 lines; current dataset loading, masked distributional loss,
  deterministic CPU optimization/resume, metrics, checkpoints, export, and CLI.
  It is the largest retained module; phase boundaries are explicit and tests
  cover the high-risk artifact and recovery paths.

### Test modules

- [ ] Engine/foundation: `test_foundation.py`, `test_engine.py`.
- [ ] Projection/policy: `test_bridge.py`, `test_search_information.py`,
  `test_search_symmetry.py`, `test_bgc_policy_model_v2.py`.
- [ ] Current training: `test_bgc_policy_training.py`.
- [ ] Local compatibility API: `test_api.py`, `test_gameplay_api.py`.
- [ ] Stateless product: `test_stateless_gameplay_api.py`,
  `test_bedrock_narration.py`, `test_local_preview.py`.
- [ ] Release: `test_lambda_packaging.py`, frontend's 13 test files, and
  `e2e/production-stateless.spec.ts`.

Each test is checked for retained-behavior value; no deleted implementation is
kept solely to satisfy an obsolete test.

### Route checklist

- [ ] `GET /health`
- [ ] `POST /games`
- [ ] `POST /games/command`
- [ ] `POST /games/resume`
- [ ] `POST /narration`

### Public and browser-state checklist

- [ ] Public envelope: `seed`, ordered accepted `history`.
- [ ] Public game: lifecycle, roles, public coffin/moves/scores, human hand and
  human legal moves.
- [ ] Narration: cue type, ready/unavailable status, bounded text.
- [ ] Browser persistence: seed and accepted history only.
- [ ] Confirm absence of engine objects, opponent hand, stock, observations,
  masks, logits, weights, diagnostics, and narrator prompts.

### Environment variables

- [ ] `DRACULA_POLICY_ARTIFACT`
- [ ] `DRACULA_NARRATION_ENABLED`
- [ ] `DRACULA_REPLAY_CACHE_ENTRIES`
- [ ] Bedrock model/region/timeout/output-limit variables documented in
  `deployment.md`
- [ ] Local ports, API proxy, Pages origin, and base path

### Rollback

- [ ] Repoint Lambda to the prior immutable image digest.
- [ ] Roll back the CloudFormation stack/update.
- [ ] Redeploy the prior Pages artifact.
- [ ] Restore or remove the API custom-domain CNAME as documented.
- [ ] Run health, CORS, replay, policy digest, and complete-game smoke checks.

## Recommended restart point

The user-approved engine boundary is complete. Resume review with
[randomness.py](/Users/itincknell/Projects/Dracula/src/dracula/randomness.py),
then `bridge.py` and `search/information.py` in tranche 4.

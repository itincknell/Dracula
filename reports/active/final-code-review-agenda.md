# Bottom-up code-review agenda

## Review basis

This agenda describes the standalone `pi1` product plus the retained BGC-128
search lineage. Sam, shallow-search, PPO, and earlier model implementations
remain deleted. The current review basis is commit
`7395741a85847784ec413ae6c322ca95ac2f91f4` plus the uncommitted, validated
review changes listed by Git.

Current maintained surface before the final commit:

| Area | Modules/files | Lines |
| --- | ---: | ---: |
| Python package | 61 | 13,145 |
| Python tests and browser server fixture | 17 | 5,088 |
| Frontend production runtime and styles | 28 | 3,784 |
| Frontend unit tests | 10 | 1,668; covered by 85 tests |
| Frontend test support | 2 | 340 |
| Browser test | 1 | 204; covered by 2 tests |
| Release and frontend build tools | 8 | 1,299 |

Selected artifact SHA-256:
`70c76f2eb64600eab2297640278a6c94d4336ab8e73bf941a6d96237f69f5b5c`.

> **Comment rule:** Comments explain non-obvious intent, invariants,
> constraints, or reasons. They remain concise and professional. They do not
> narrate straightforward code.

Every tranche includes an inline-comment and docstring review before approval.

For progress accounting, the maintained implementation contains 92 modules
and 17,958 lines: the Python package, production frontend and styles, three
release tools, and five frontend build/configuration scripts. Tests,
declarative infrastructure, documentation, and generated files are tracked
separately. The approved boundary covers 15 of 92 modules and 3,117 of 17,958
lines. Approving tranche 6 would cover 20 modules and 4,795 lines.

## Approval ledger

- [x] NORTHSTARS, rules, and cards.
- [x] Engine types, scoring, and dealing.
- [x] State validation, serialization, transitions, and lifecycle.
- [x] Deterministic replay and privacy.
- [x] Symmetry and strategic-action representation.
- [ ] Everything after the strategic-action boundary.

## 1. Rules and cards — approved

- [x] [NORTHSTARS](/Users/itincknell/Projects/Dracula/NORTHSTARS) — project
  priorities and engineering constraints.
- [x] [rules.md](/Users/itincknell/Projects/Dracula/docs/rules.md) — authoritative
  game rules.
- [x] [cards.py](/Users/itincknell/Projects/Dracula/src/dracula/cards.py) — 114
  118 lines; immutable card identities, indexes, validation, and canonical sorting.
  Inputs are card IDs; outputs are frozen `Card` values or canonical tuples.
  It has no private/public distinction and is covered by `test_foundation.py`.

## 2. Engine types and scoring — approved

- [x] [engine_types.py](/Users/itincknell/Projects/Dracula/src/dracula/engine_types.py)
  — 297 lines; engine enums, immutable values, exceptions, adjacency, and
  transitions. It is imported throughout the engine and services. Frozen state
  shape and zero-based row-major coffin indexing are key invariants.
- [x] [scoring.py](/Users/itincknell/Projects/Dracula/src/dracula/scoring.py) —
  191 lines; exact line arithmetic, Vampire override, multiplier priority, and
  ranked tie resolution. Coffin cards enter; immutable score structures leave.
  `test_engine.py` and `test_foundation.py` cover it.
- [x] [engine_dealing.py](/Users/itincknell/Projects/Dracula/src/dracula/engine_dealing.py)
  — 98 lines; deterministic shuffle, dealer selection, and two-card packet
  deal. It depends on cards and randomness; engine validation and lifecycle
  call it. Packet order and canonical hands are compatibility-sensitive.

## 3. Validation, serialization, transitions, and lifecycle — approved

- [x] [engine_validation.py](/Users/itincknell/Projects/Dracula/src/dracula/engine_validation.py)
  — 377 lines; complete lifecycle, deal, card-conservation, move-history, and
  score validation. It handles private `EngineState`; callers receive success
  or typed violations. Complexity is intentional because all invariants meet
  at this boundary.
- [x] [engine_serialization.py](/Users/itincknell/Projects/Dracula/src/dracula/engine_serialization.py)
  — 210 lines; canonical private serialization, validated decoding for local
  SQLite recovery, and fingerprints. Dictionary order and exact JSON encoding
  are compatibility invariants.
- [x] [engine.py](/Users/itincknell/Projects/Dracula/src/dracula/engine.py) — 313
  lines; gameplay facade for creation, legal moves, immutable transitions,
  round advancement, and re-exports. Inputs and outputs are private immutable
  engine values. `test_engine.py` holds golden deal, legality, scoring, and
  lifecycle coverage.

## 4. Deterministic replay and privacy — approved

Review in this order:

| Review | Module | Purpose and public interface | Boundaries, tests, and risk |
| --- | --- | --- | --- |
| [x] | [randomness.py](/Users/itincknell/Projects/Dracula/src/dracula/randomness.py) — 124 lines | `derive_seed`, counter stream, unbiased `randbelow`, deterministic shuffle | Text/integers in; bytes or deterministic choices out. Seed namespace and rejection sampling are non-obvious compatibility rules. Covered by `test_foundation.py`. |
| [x] | [bridge.py](/Users/itincknell/Projects/Dracula/src/dracula/bridge.py) — 262 lines | Player-relative 32-action mapping, `PolicyTurnContext`, forced/learned turn resolution | Engine state stays private; action table is the legal bridge. Ordering and Queen/King transpose are invariant. Covered by `test_bridge.py`. |
| [x] | [search/information.py](/Users/itincknell/Projects/Dracula/src/dracula/search/information.py) — 554 lines | `SearchInformationState`, public history, projection, validation, canonical fingerprint | Erases opponent slots, stock order, and seed. The simulation projection applies the same erasure without repeating whole-deck validation at every simulated move. Covered by `test_search_information.py` and `test_bgc_search.py`. |
| [x] | [search/__init__.py](/Users/itincknell/Projects/Dracula/src/dracula/search/__init__.py) — 45 lines | Narrow public exports for information and symmetry only | Heavyweight BGC controllers require explicit module imports, preventing cycles and hidden runtime dependencies. |

Flow:

```text
EngineState --validated projection--> SearchInformationState
     |                                      |
     `-- canonical replay/fingerprint ------'
```

## 5. Symmetry and strategic-action representation — approved

| Review | Module | Purpose and public interface | Boundaries, tests, and risk |
| --- | --- | --- | --- |
| [x] | [search/symmetry.py](/Users/itincknell/Projects/Dracula/src/dracula/search/symmetry.py) — 120 lines | Exhaustive seven-pattern destination grouping | Occupancy and legal destinations in; designated proxies/groups out. No inferred geometry is permitted. Covered by `test_search_symmetry.py`. |
| [x] | [strategic_actions.py](/Users/itincknell/Projects/Dracula/src/dracula/strategic_actions.py) — 121 lines | Card-specific strategic groups and deterministic paired-member selection | Accepts actor-visible information only. Distinct cards never merge; groups partition legality exactly. Covered by symmetry and model tests. |
| [x] | [action_contract.py](/Users/itincknell/Projects/Dracula/src/dracula/action_contract.py) — 112 lines | Representative masks, masked logits, and canonical argmax | Legal/group tensors in; representative selection out. Masked actions must have zero probability and canonical ties. Covered by symmetry/model/trainer tests. |
| [x] | [policy_observation.py](/Users/itincknell/Projects/Dracula/src/dracula/policy_observation.py) — 179 lines | Direct `bool[659]` encoding, engine/model row translation, and the shared masked-inference path | Typed actor view in; tensor or selected strategic group out. No permanent slot feature enters the model. Observation hashes, card extraction, and action round trips are tested. |

Action boundary:

```text
SearchInformationState -> legal card/destination actions
     -> authoritative destination groups -> one proxy per strategic action
     -> representative mask + candidate-card row mapping
```

## 6. Retained BGC-128 search and continuation policies — next review tranche

This is maintained training-lineage and comparison code, not a production
gameplay fallback. Review in dependency order:

| Review | Module | Purpose and public interface | Boundaries, tests, and risk |
| --- | --- | --- | --- |
| [ ] | [search/contracts.py](/Users/itincknell/Projects/Dracula/src/dracula/search/contracts.py) — 147 lines | Search result values, canonical configuration digests, and the actor-visible continuation protocol | Continuation policies accept only `SearchInformationState`. Principal continuations are private diagnostics because sampled cards reveal a determinization. |
| [ ] | [search/determinization.py](/Users/itincknell/Projects/Dracula/src/dracula/search/determinization.py) — 382 lines | Reconstruct one complete sampled world from actor-visible information | The sampled opponent hand and stock never leave search. The reconstructed state must project exactly back to the supplied information state. |
| [ ] | [search/bgc.py](/Users/itincknell/Projects/Dracula/src/dracula/search/bgc.py) — 724 lines | Symmetry-aware information-set UCT with the retained 128-simulation default | Its cohesive search loop is separated internally into sampling, selection, continuation, backup, and result phases; it samples a private world per simulation and uses exact completed-round scores. |
| [ ] | [search/belief_greedy.py](/Users/itincknell/Projects/Dracula/src/dracula/search/belief_greedy.py) — 292 lines | Baseline BGC continuation: one-ply choice using eight shared belief completions | Every legal strategic group is scored against the same sampled worlds. This is the close successor to the original retained BGC configuration. |
| [ ] | [search/policy_continuation.py](/Users/itincknell/Projects/Dracula/src/dracula/search/policy_continuation.py) — 133 lines | Phase-two continuation using one verified current-format policy artifact | Delegates the shared 659-bit masking and selection path, then applies its own deterministic continuation seed and fair coin. |

Search variants:

```text
root actor-visible state -> BGC UCT (128 simulations) -> exact round score
                              |
                              +-- baseline: 8-completion belief-greedy responses
                              `-- phase two: one pi0 inference per response
```

## 7. Pi1 model, artifact, and production move selection

| Review | Module | Purpose and public interface | Boundaries, tests, and risk |
| --- | --- | --- | --- |
| [ ] | [bgc_policy_model.py](/Users/itincknell/Projects/Dracula/src/dracula/bgc_policy_model.py) — 339 lines | `BGCPolicyModel`, deterministic initialization, named architecture widths, card-candidate extraction, and `float32[4,8]` logits | 659-bit visible observation in; raw logits out. Exact 754,601 parameters and no value head are invariant. Its forward path is separated into hand, coffin, shared-state, and pair-scoring phases. |
| [ ] | [bgc_policy.py](/Users/itincknell/Projects/Dracula/src/dracula/bgc_policy.py) — 439 lines | Strict model metadata, state-dictionary digest, atomic artifact save/load | Artifact bytes are an untrusted load boundary. Envelope, metadata, configuration, and tensors are verified once in separate phases before runtime use. |
| [ ] | [active_policy.py](/Users/itincknell/Projects/Dracula/src/dracula/active_policy.py) — 183 lines | Selected artifact runtime and `ActivePolicyExecutor` | Delegates the shared observation/mask/inference path; fair coin resolves the concrete move; the engine action table revalidates it. The fixed seed namespace preserves validated π1 move resolution. |

Model boundary:

```text
SearchInformationState -> bool[659] -> pi1 -> logits[4,8]
        + legal groups -> representative mask -> masked argmax
        + typed own hand -> engine slot -> deterministic paired destination
```

## 8. Stateless API and narration

| Review | Module | Purpose and public interface | Boundaries, tests, and risk |
| --- | --- | --- | --- |
| [ ] | [api/policy.py](/Users/itincknell/Projects/Dracula/src/dracula/api/policy.py) — 88 lines | Controller-neutral executor/request/result protocol | `SearchInformationState` crosses to pi1; engine state does not. A fixed empty hidden-state field remains only for stateful-record compatibility. |
| [ ] | [api/stateless_contracts.py](/Users/itincknell/Projects/Dracula/src/dracula/api/stateless_contracts.py) — 248 lines | Recovery envelope, commands, public game view, narration wire models | Public JSON only. Seed/history are intentionally client-visible; model and engine-private fields are excluded. |
| [ ] | [api/stateless_http.py](/Users/itincknell/Projects/Dracula/src/dracula/api/stateless_http.py) — 85 lines | Bounded request-body middleware and uniform stateless errors | Raw HTTP in; validated bounded body/error out. Chunk handling is the main subtlety. |
| [ ] | [api/stateless_projection.py](/Users/itincknell/Projects/Dracula/src/dracula/api/stateless_projection.py) — 128 lines | Private-engine to public-human projection | Opponent hand/actions never cross. Covered by stateless API privacy tests. |
| [ ] | [api/stateless_replay.py](/Users/itincknell/Projects/Dracula/src/dracula/api/stateless_replay.py) — 117 lines | Canonical envelope identity and bounded process-local replay cache | Cache entries are private reconstructed states; eviction or a cold process cannot alter correctness. |
| [ ] | [api/stateless_service.py](/Users/itincknell/Projects/Dracula/src/dracula/api/stateless_service.py) — 359 lines | Deterministic replay, command validation, and opponent turns | Seed/history commands in; reconstructed private state stays internal; public response out. It delegates optional cache ownership to `stateless_replay`. |
| [ ] | [api/stateless_routes.py](/Users/itincknell/Projects/Dracula/src/dracula/api/stateless_routes.py) — 143 lines | `/health`, `/games`, `/games/command`, `/games/resume`, `/narration` handlers | Thin transport over service/narration boundaries. Covered by `test_stateless_gameplay_api.py` and narration tests. |
| [ ] | [api/stateless_app.py](/Users/itincknell/Projects/Dracula/src/dracula/api/stateless_app.py) — 45 lines | Production-neutral FastAPI assembly | Explicit policy and narration dependencies; no database fallback. |
| [ ] | [api/narration_cues.py](/Users/itincknell/Projects/Dracula/src/dracula/api/narration_cues.py) — 275 lines | Three cue classes, grounded public facts, and deterministic prompt construction | Replays the envelope but derives only public cue facts. Exact combo and score-movement language belongs here. |
| [ ] | [api/narration.py](/Users/itincknell/Projects/Dracula/src/dracula/api/narration.py) — 226 lines | Provider-neutral narration orchestration, output safety, failure isolation, and test adapter | Narration failure cannot alter gameplay; provider output is bounded plain text or an unavailable result. |
| [ ] | [api/bedrock.py](/Users/itincknell/Projects/Dracula/src/dracula/api/bedrock.py) — 186 lines | Nova request payload, bounded Bedrock call, plain-text parsing | Public prompt in; optional bounded text out. Credentials/provider failures remain outside gameplay transaction. |
| [ ] | [api/production_config.py](/Users/itincknell/Projects/Dracula/src/dracula/api/production_config.py) — 80 lines | Strict environment parsing for artifact, narration, and replay cache | No account IDs or secrets are defaults. |
| [ ] | [api/production.py](/Users/itincknell/Projects/Dracula/src/dracula/api/production.py) — 37 lines | Lambda FastAPI entry point | Loads the pinned artifact and optional Bedrock adapter; never selects a fallback controller. |
| [ ] | [production_logging.py](/Users/itincknell/Projects/Dracula/src/dracula/production_logging.py) — 34 lines | Structured JSON formatter | Log records in; one JSON object out. Logs must omit seed/history/private/model data. |

### Narrow local compatibility API

These modules preserve recorded SQLite games and transaction tests but are not
copied into the Lambda image:

| Review | Module | Ownership and boundary |
| --- | --- | --- |
| [ ] | [api/contracts.py](/Users/itincknell/Projects/Dracula/src/dracula/api/contracts.py) — 293 lines | Shared public score/view primitives plus local-stateful wire models. |
| [ ] | [api/presentation.py](/Users/itincknell/Projects/Dracula/src/dracula/api/presentation.py) — 234 lines | Shared scoring presentation projection. |
| [ ] | [api/session.py](/Users/itincknell/Projects/Dracula/src/dracula/api/session.py) — 255 lines | Validated local persisted session encoding; private SQLite data only. |
| [ ] | [api/repository.py](/Users/itincknell/Projects/Dracula/src/dracula/api/repository.py) — 382 lines | In-memory and SQLite transactional storage; local only. |
| [ ] | [api/local_session_events.py](/Users/itincknell/Projects/Dracula/src/dracula/api/local_session_events.py) — 255 lines | Local session creation and public event history derived from engine transitions. |
| [ ] | [api/local_policy_turn.py](/Users/itincknell/Projects/Dracula/src/dracula/api/local_policy_turn.py) — 69 lines | Actor-visible local policy invocation and move-result validation. |
| [ ] | [api/local_projection.py](/Users/itincknell/Projects/Dracula/src/dracula/api/local_projection.py) — 125 lines | Local browser-safe view projection and signed move tokens. |
| [ ] | [api/local_routes.py](/Users/itincknell/Projects/Dracula/src/dracula/api/local_routes.py) — 128 lines | Local-only stateful FastAPI route declarations. |
| [ ] | [api/service.py](/Users/itincknell/Projects/Dracula/src/dracula/api/service.py) — 564 lines | Local stateful transaction, idempotency, and policy-claim orchestration; no production caller. |
| [ ] | [api/local_controllers.py](/Users/itincknell/Projects/Dracula/src/dracula/api/local_controllers.py) — 46 lines | Explicit local resolution of `bgc-policy` only. |
| [ ] | [api/app.py](/Users/itincknell/Projects/Dracula/src/dracula/api/app.py) — 126 lines | Explicit local-stateful/stateless assembly; normal Make targets select stateless. |
| [ ] | [api/local_preview.py](/Users/itincknell/Projects/Dracula/src/dracula/api/local_preview.py) — 99 lines | Stateless preview with deterministic dummy dialogue. |
| [ ] | [api/__init__.py](/Users/itincknell/Projects/Dracula/src/dracula/api/__init__.py) — 5 lines | Side-effect-free package marker; callers import the owning app module directly. |

## 9. Frontend state and presentation

Review runtime files in dependency order:

| Review | Module | Purpose, boundaries, and coverage |
| --- | --- | --- |
| [ ] | [contractPrimitives.ts](/Users/itincknell/Projects/Dracula/frontend/src/contractPrimitives.ts) — 237 lines | Public cards, scores, phases, and shared wire-value validators. |
| [ ] | [statelessContracts.ts](/Users/itincknell/Projects/Dracula/frontend/src/statelessContracts.ts) — 271 lines | Exact validation of external seed/history, game, health, error, and narration responses. |
| [ ] | [gameView.ts](/Users/itincknell/Projects/Dracula/frontend/src/gameView.ts) — 47 lines | Minimal trusted presentation view and presentation error shape. |
| [ ] | [gameControllerContract.ts](/Users/itincknell/Projects/Dracula/frontend/src/gameControllerContract.ts) — 52 lines | View-facing controller and observable store interfaces used by components. |
| [ ] | [statelessApi.ts](/Users/itincknell/Projects/Dracula/frontend/src/statelessApi.ts) — 132 lines | The sole browser HTTP transport and raw-response validation boundary. |
| [ ] | [statelessRecovery.ts](/Users/itincknell/Projects/Dracula/frontend/src/statelessRecovery.ts) — 47 lines | Browser persistence and validation of seed plus accepted history only. |
| [ ] | [statelessProjection.ts](/Users/itincknell/Projects/Dracula/frontend/src/statelessProjection.ts) — 78 lines | Trusted response projection plus deliberate human/opponent timing previews. |
| [ ] | [statelessNarration.ts](/Users/itincknell/Projects/Dracula/frontend/src/statelessNarration.ts) — 92 lines | Three-cue eligibility, stale-response protection, reveal timing, and failure isolation. |
| [ ] | [statelessGameStore.ts](/Users/itincknell/Projects/Dracula/frontend/src/statelessGameStore.ts) — 361 lines | New game, commands, replay, persistence, narration sequencing, immediate human placement, and minimum Dracula delay. |
| [ ] | [scoringStateMachine.ts](/Users/itincknell/Projects/Dracula/frontend/src/scoringStateMachine.ts) — 297 lines | Pure scoring model, frame sequence, and locked playback timing. |
| [ ] | [cardAssets.ts](/Users/itincknell/Projects/Dracula/frontend/src/cardAssets.ts) — 49 lines | Generated card-ID-to-asset mapping and human-readable card names. |
| [ ] | [siteConfig.ts](/Users/itincknell/Projects/Dracula/frontend/src/siteConfig.ts) — 29 lines | `/Dracula/` routes and personal-site destinations. |
| [ ] | [CardFace.tsx](/Users/itincknell/Projects/Dracula/frontend/src/CardFace.tsx) — 10 lines | Shared card image component. |
| [ ] | [DraculaCommentary.tsx](/Users/itincknell/Projects/Dracula/frontend/src/DraculaCommentary.tsx) — 170 lines | Portrait state, typed text, and static final-loss face behavior. |
| [ ] | [ScoringWorkspaces.tsx](/Users/itincknell/Projects/Dracula/frontend/src/ScoringWorkspaces.tsx) — 240 lines | Pure score-calculation, comparison, and final-score workspaces. |
| [ ] | [ScoringPresentation.tsx](/Users/itincknell/Projects/Dracula/frontend/src/ScoringPresentation.tsx) — 157 lines | Scoring timer progression and round/game continuation controls. |
| [ ] | [gameplay.tsx](/Users/itincknell/Projects/Dracula/frontend/src/gameplay.tsx) — 244 lines | Board, hand, role/status, scoreboard, and commentary composition. |
| [ ] | [RulesPage.tsx](/Users/itincknell/Projects/Dracula/frontend/src/RulesPage.tsx) — 119 lines | Static public rules page. |
| [ ] | [SeenCardsExpando.tsx](/Users/itincknell/Projects/Dracula/frontend/src/SeenCardsExpando.tsx) — 96 lines | Public seen-card ledger and expandable cheat-sheet presentation. |
| [ ] | [App.tsx](/Users/itincknell/Projects/Dracula/frontend/src/App.tsx) — 192 lines | Hash routing, controller construction, start page, and page composition. |
| [ ] | [main.tsx](/Users/itincknell/Projects/Dracula/frontend/src/main.tsx) — 26 lines | Browser bootstrap and ordered font/style imports. |

The session-backed browser API, store, contracts, and compatibility export were
removed because no maintained runtime imported them. Local SQLite gameplay
remains a Python-only compatibility surface. `testGameController.ts` and
`scoringTestFixtures.ts` are explicit test support, not production adapters.

Presentation styles, reviewed in cascade order:

- [ ] [styles.css](/Users/itincknell/Projects/Dracula/frontend/src/styles.css) — 10 lines.
- [ ] [base.css](/Users/itincknell/Projects/Dracula/frontend/src/styles/base.css) — 206 lines.
- [ ] [gameplay.css](/Users/itincknell/Projects/Dracula/frontend/src/styles/gameplay.css) — 331 lines.
- [ ] [scoring.css](/Users/itincknell/Projects/Dracula/frontend/src/styles/scoring.css) — 115 lines.
- [ ] [rules.css](/Users/itincknell/Projects/Dracula/frontend/src/styles/rules.css) — 31 lines.
- [ ] [responsive.css](/Users/itincknell/Projects/Dracula/frontend/src/styles/responsive.css) — 125 lines.

The approved desktop/mobile geometry, animation timing, no-auto-scroll rule,
portrait behavior, and score alignment are invariants. Component, usability,
contract, state-machine, and Playwright suites cover them.

## 10. Deployment and operations

- [ ] [Dockerfile](/Users/itincknell/Projects/Dracula/deployment/container/Dockerfile)
  — 37 lines; Lambda Web Adapter, Python runtime, embedded `pi1`, and read-only
  runtime. Review the three pinned requirement files and logging configuration
  immediately afterward.
- [ ] [build_lambda_context.py](/Users/itincknell/Projects/Dracula/tools/build_lambda_context.py)
  — 235 lines; verifies the exact artifact digest and copies only the 34-module
  production import closure.
- [ ] [application.yaml](/Users/itincknell/Projects/Dracula/infrastructure/application.yaml)
  — 432 lines; Regional HTTP API, Lambda, Bedrock permission, logs, alarms, and
  budgets. Review the 30-line ECR template and both 20-line parameter files
  with it.
- [ ] [validate_lambda_container.py](/Users/itincknell/Projects/Dracula/tools/validate_lambda_container.py)
  — 482 lines; local cold/warm/replay/container validation.
- [ ] [build_release_candidate.py](/Users/itincknell/Projects/Dracula/tools/build_release_candidate.py)
  — 312 lines; source, model, frontend, image, and infrastructure release identity.
- [ ] [pages.yml](/Users/itincknell/Projects/Dracula/.github/workflows/pages.yml)
  — 65 lines; pull-request verification and explicit manual Pages publication.
- [ ] [Makefile](/Users/itincknell/Projects/Dracula/Makefile),
  [pyproject.toml](/Users/itincknell/Projects/Dracula/pyproject.toml), and
  [package.json](/Users/itincknell/Projects/Dracula/frontend/package.json) —
  122, 34, and 47 lines; one current local, test, training, and release command
  surface. Review `vite.config.ts`, `playwright.config.ts`, ESLint, and the two
  frontend build scripts before the package manifest.

External boundaries to check line by line: Bedrock Runtime request, API Gateway
request/response, Cloudflare CNAME, GitHub Pages publication, and browser local
storage. No production database, SageMaker endpoint, signature, or hidden
fallback exists.

## 11. Tests, retained trainer, compatibility reader, docs, and rollback

### Current training support

- [ ] [bgc_policy_migration.py](/Users/itincknell/Projects/Dracula/src/dracula/bgc_policy_migration.py)
  — 486 lines; narrow read-only verifier and decoder for the physically
  converted 659-bit selected corpus. It accepts no historical source-row format.
- [ ] [source_identity.py](/Users/itincknell/Projects/Dracula/src/dracula/source_identity.py)
  — 70 lines; Git revision and tracked-source digest for trainer artifacts.
- [ ] [bgc_policy_training_contracts.py](/Users/itincknell/Projects/Dracula/src/dracula/bgc_policy_training_contracts.py)
  — 178 lines; immutable trainer configuration, result values, and locked
  optimizer constants.
- [ ] [bgc_policy_data.py](/Users/itincknell/Projects/Dracula/src/dracula/bgc_policy_data.py)
  — 149 lines; strict JSON, digest, privacy, and packed-Boolean decoding helpers.
- [ ] [bgc_policy_metrics.py](/Users/itincknell/Projects/Dracula/src/dracula/bgc_policy_metrics.py)
  — 267 lines; representative-masked distributional loss and evaluation metrics.
- [ ] [bgc_policy_checkpoints.py](/Users/itincknell/Projects/Dracula/src/dracula/bgc_policy_checkpoints.py)
  — 259 lines; atomic, identity-bound training checkpoints and resolved config seal.
- [ ] [bgc_policy_training_config.py](/Users/itincknell/Projects/Dracula/src/dracula/bgc_policy_training_config.py)
  — 251 lines; strict TOML loading and immutable data/model/optimizer/source identity.
- [ ] [bgc_policy_optimization.py](/Users/itincknell/Projects/Dracula/src/dracula/bgc_policy_optimization.py)
  — 486 lines; deterministic minibatches, interruption-safe resume, epochs, and
  checkpoint selection.
- [ ] [bgc_policy_training.py](/Users/itincknell/Projects/Dracula/src/dracula/bgc_policy_training.py)
  — 471 lines; training-run coordination, selected-checkpoint export, validation,
  and reporting.
- [ ] [bgc_policy_training_cli.py](/Users/itincknell/Projects/Dracula/src/dracula/bgc_policy_training_cli.py)
  — 113 lines; command parsing, dispatch, terminal output, and stable exit codes.

### Test modules

- [ ] Engine/foundation: `test_foundation.py`, `test_engine.py`.
- [ ] Projection/policy: `test_bridge.py`, `test_search_information.py`,
  `test_search_symmetry.py`, `test_bgc_policy_model_v2.py`.
- [ ] Retained search: `test_bgc_search.py`.
- [ ] Current training: `test_bgc_policy_training.py`.
- [ ] Local compatibility API: `test_api.py`, `test_gameplay_api.py`.
- [ ] Stateless product: `test_stateless_gameplay_api.py`,
  `test_bedrock_narration.py`, `test_local_preview.py`.
- [ ] Release: `test_lambda_packaging.py`, frontend's 10 unit-test files, and
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

The strategic-action boundary is approved. Resume review with
[search/contracts.py](/Users/itincknell/Projects/Dracula/src/dracula/search/contracts.py),
then follow tranche 6 through determinization, BGC search, the belief-greedy
continuation, and the policy continuation. These five modules are the complete
retained search lineage and remain separate from production standalone `pi1`.

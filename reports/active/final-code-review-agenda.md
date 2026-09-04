# Bottom-up code-review agenda

Date: 2026-09-02

This agenda restarts the user-led line review from the refactored repository.
It does not approve any code. Review proceeds in nine manageable tranches; each
tranche ends with its tests and an explicit approval checkpoint.

## Frozen review basis

| Item | Identity |
| --- | --- |
| Git commit beneath the working tree | `81b2504d4cc679e40e36a1fd63160037e6dfbb0d` |
| Dirty scope before this agenda refresh | 59 tracked paths changed; 69 untracked status entries; no staged files |
| Authored review manifest | 249 files, SHA-256 `e4d0cece0c2dcbf232eb5509b4b742e294cca15378f951c856338c0b86812666` |
| Selected `pi1` artifact | SHA-256 `70c76f2eb64600eab2297640278a6c94d4336ab8e73bf941a6d96237f69f5b5c` |
| Local Lambda image | `sha256:be17039c28629b8a142e7fdd5c85088d258b7a718dfcdb9fe6421cc3af4678a4` (996,560,893 bytes) |
| Infrastructure tree | SHA-256 `77d527765405010309fc6a1261fd9df46dccd8896e25419acce7b6d620dee6a2` |
| Production frontend tree | SHA-256 `23646d2d96a0a795704b353af0403b90448ddbc04b51a59fc6025435f79588f5` |
| Python validation | 688 passed |
| Frontend validation | 99 unit/component and 2 browser tests passed; TypeScript, ESLint, and production build passed |

The manifest covers `NORTHSTARS`, root build metadata, active docs/configs,
Python source and tests, frontend source/e2e/scripts, tools, deployment,
infrastructure, and workflows. Reports are review evidence and are not included
in the source digest.

## Comment and docstring rule

> **Comments explain non-obvious intent, invariants, constraints, or reasons.
> They remain concise and professional. They do not narrate straightforward
> code.**

Every tranche includes a comment and docstring review before approval. A low
comment count is not itself a defect. Public docstrings add value only when
caller-visible behavior, ownership, side effects, exceptions, privacy, or
compatibility cannot be read directly from the name and signature.

## Review notation

Each module entry records its purpose and public interfaces; input/output flow;
dependencies and callers; invariants and public/private data; tests; and known
complexity. Check the box only after reading the file line by line and reviewing
its comments/docstrings against the rule above.

---

## Tranche 1 — Rules and cards

- [ ] [NORTHSTARS](../../NORTHSTARS) — **7 lines.**
  Scope and priority order. Input: product decisions; output: constraints for
  every contract. All code and review decisions depend on it. Public data only.
  Complexity: none; ambiguity here propagates everywhere.
- [ ] [rules.md](../../docs/rules.md) — **164
  lines.** Authoritative six-round rules, adjacency, Vampire behavior, scoring,
  multipliers, dealer alternation, and tie breaks. Input: game design; output:
  engine/UI behavior. Covered by foundation, engine, and scoring-presentation
  suites. Public data. Risk: prose/code disagreement.
- [ ] [cards.py](../../src/dracula/cards.py) — **114
  lines.** Card domain and canonical 54-card order. Public interfaces: `Suit`,
  `Color`, frozen/slotted `Card`, `card_by_id`, `card_by_index`,
  `sort_card_ids`. Inputs card IDs/indices; outputs immutable card values.
  Depends only on the standard library; called by engine, bridge, model, search,
  API, and training. Invariants: 52 ranked cards plus two Vampires, stable IDs,
  suit/color mapping. Handles public or private card identity depending on its
  caller but owns no visibility policy. Tests: `test_foundation.py`,
  `test_engine.py`, bridge/search/model suites. Complexity: low; canonical order
  is artifact-sensitive.

**Tranche approval checks**

- [ ] Every rule used below is present in `rules.md`.
- [ ] Card order and Vampire representation are intentional and stable.
- [ ] Comments and docstrings meet the stated rule.

---

## Tranche 2 — Engine types and scoring

- [ ] [engine_types.py](../../src/dracula/engine_types.py)
  — **276 lines.** Immutable engine values and exceptions. Public interfaces:
  player/status/orientation/reason enums, rule exceptions, `Move`, line/round
  result records, deal/state values, and `other_player`. Inputs validated domain
  fields; outputs frozen state objects. Depends on cards; called by every engine,
  bridge, search, API, and evaluation path. Invariants: nine coffin positions,
  four hand slots, immutable tuples/mappings, explicit lifecycle. Contains
  private hands and stock in `EngineState`; never a public transport object.
  Tests: foundation, engine, API privacy. Complexity: moderate because these
  shapes define most contracts.
- [ ] [scoring.py](../../src/dracula/scoring.py) —
  **179 lines.** Pure scoring and outcome resolution. Public interfaces:
  `score_line`, `score_coffin`, `resolve_round_scores`, `make_round_result`,
  `resolve_game_outcome`. Inputs completed coffins/history; outputs exact line,
  round, and game results. Depends on cards/types; called by engine and
  diagnostic search. Invariants: Vampire zero, exact multiplier order, player
  orientations, ranked-line and final tie breaks. Scoring values are public
  after round completion; inputs can originate in private engine state. Tests:
  foundation, engine, strategic and API scoring tests. Complexity: high-value
  rules boundary despite small size.
- [ ] [engine_dealing.py](../../src/dracula/engine_dealing.py)
  — **94 lines.** Deterministic shuffle/deal primitives. Public interfaces:
  `shuffled_deck`, `initial_dealer`, `deal_round`, `create_initial_deal`,
  `as_hand`, `starting_coffin`. Inputs seed/deck/round; outputs ordered private
  hands, stock, dealer, and center card. Depends on cards/randomness/types;
  called by engine lifecycle and replay. Invariants: exact shuffle stream and
  card conservation. Private data. Tests: foundation and golden engine/replay
  tests. Complexity: low code, high determinism sensitivity.

**Tranche approval checks**

- [ ] Data classes are immutable where state ownership requires it.
- [ ] Scoring matches every rule example and tie-break.
- [ ] Dealing never depends on process-global randomness.
- [ ] Comments and docstrings meet the stated rule.

---

## Tranche 3 — Validation, serialization, transitions, and lifecycle

- [ ] [engine_validation.py](../../src/dracula/engine_validation.py)
  — **350 lines.** Structural, history, deal, lifecycle, and card-conservation
  validation. Public interface: `validate_state`. Input: complete private
  `EngineState`; output: validated return or precise exception. Depends on
  cards/types; called at construction, transition, replay, and artifact
  boundaries. Invariants: legal counts, history/coffin agreement, active player,
  dealer, hands/stock conservation, completed-round consistency. Private data.
  Tests: `test_engine.py`, malformed API/replay tests. Complexity: highest
  concentration of engine invariants.
- [ ] [engine_serialization.py](../../src/dracula/engine_serialization.py)
  — **116 lines.** Canonical private state serialization and fingerprinting.
  Public interfaces: `canonical_state_data`, `canonical_state_json`,
  `state_fingerprint`. Input: validated state; outputs stable JSON-compatible
  data/text/hash. Depends on cards/types; called by search, persistence,
  diagnostics, and golden fixtures. Invariants: canonical key/list order and
  complete private identity. Private data; output must not cross public API.
  Tests: engine golden/fingerprint, repository and search invariance. Complexity:
  compatibility-sensitive.
- [ ] [engine.py](../../src/dracula/engine.py) — **299
  lines.** Stable gameplay facade plus legal moves, immutable transitions, round
  advancement, and lifecycle orchestration. Public interfaces: `create_game`,
  `legal_moves`, `legal_simulation_moves`, `apply_move`,
  `apply_simulation_move`, `advance_after_round`, `derive_game_outcome`, and
  direct compatibility re-exports. Inputs commands/private states; outputs new
  states and results. Depends on the five engine owners; called by all gameplay,
  replay, search, API, and fixtures. Invariants: no mutation, canonical move
  order, adjacency, forced eighth placement, six rounds, exact exceptions.
  Private state internally; callers choose projections. Tests: full engine,
  bridge, search, API, transaction, and golden suites. Complexity: central
  transition boundary, now intentionally small.

**Tranche approval checks**

- [ ] Every state-changing function returns a new valid state.
- [ ] Serialization remains canonical and private.
- [ ] `engine.py` is a readable facade, not a second implementation.
- [ ] Comments and docstrings meet the stated rule.

---

## Tranche 4 — Deterministic replay and privacy

- [ ] [randomness.py](../../src/dracula/randomness.py)
  — **108 lines.** Versioned SHA-256 seed derivation and deterministic byte
  stream. Interfaces: `derive_seed`, `derive_pytorch_seed`, `seed_hex`,
  `seed_integer`, `Sha256CounterStream`, `shuffled`. Structured inputs produce
  isolated deterministic streams. Called by dealing, policy, search, mining,
  and training. Invariants: namespace/domain separation and no global RNG
  mutation. Seed material is private except the accepted browser game seed.
  Tests: foundation and every reproducibility suite. Complexity: compact but
  changing literals changes artifacts/actions.
- [ ] [search/information.py](../../src/dracula/search/information.py)
  — **727 lines.** Actor-visible game/history projection, validation,
  fingerprinting, and model input conversion. Interfaces include public-history
  records, `SearchInformationState`, projection constructors, fingerprints, and
  policy-input conversion. Complete engine/simulation state enters; only the
  acting player's information leaves. Depends on engine/bridge; called by all
  planners, miners, and policy evaluation. Invariants: actor-local hand only,
  public history only, indistinguishable hidden worlds share identity. Handles
  the privacy boundary. Tests: `test_search_information.py`, bridge/search/
  privacy suites. Complexity: large because it retains historical encodings.
- [ ] [search/determinization.py](../../src/dracula/search/determinization.py)
  — **219 lines.** Private hidden-world sampling extracted from information
  projection. Public interface: `sample_determinization`. Actor information and
  derived seed enter; sampled opponent hand/stock/simulation state leave only
  to search. Depends on engine dealing/information; called by information-set
  planners. Invariants: consistency with visible facts and no authoritative
  hidden arrangement. Strictly private data. Tests: information and search
  invariance. Complexity: privacy-critical.
- [ ] [bridge.py](../../src/dracula/bridge.py) — **451
  lines.** Player-relative observation, legal mask, action/move mapping, and turn
  context. Interfaces: `PolicyInput`, `PolicyTurnContext`, coordinate mappings,
  action mappings, encoders, and context builders. Engine state plus actor enter;
  bool observation/mask and legal moves leave. Depends on engine/cards; called
  by active policy, search, datasets, and API. Invariants: Queen/King
  normalization, 32 action indices, legal mask exactness, no opponent hand in
  model input. Tests: `test_bridge.py`, policy/search/privacy suites. Complexity:
  artifact-sensitive feature contract.
- [ ] [bridge_legacy.py](../../src/dracula/bridge_legacy.py)
  — **259 lines.** Recurrent hidden-state compatibility extracted from the
  active bridge. Interfaces: `PolicyTransition`, hidden byte conversion,
  validation, transition construction. Inputs historical policy states; outputs
  archived transition records. Called by PPO/legacy adapters only. Invariants:
  exact byte shape and historical serialization. Private model state. Tests:
  bridge, local policy, collection/training. Complexity: compatibility code;
  no production caller.
- [ ] [engine-model-contract.md](../../docs/engine-model-contract.md)
  — **402 lines.** Authoritative boundary tying engine, actor information,
  symmetry, observation, action, and privacy. Reviewed jointly with the five
  modules above. Public contract describing private-data exclusions. Tests are
  the bridge/information/privacy suites. Risk: duplicated claims drifting from
  executable shapes.

**Replay and privacy flow**

```text
seed + accepted commands -> engine facade -> private EngineState
                                      |-> public API projection
                                      `-> acting-player information -> pi1/search
```

- [ ] Same seed/history reproduces the same private state and public response.
- [ ] Only actor-visible information crosses the policy/search boundary.
- [ ] Determinizations never enter public responses or model inputs.
- [ ] Comments and docstrings meet the stated rule.

---

## Tranche 5 — Symmetry, pi1, and move selection

- [ ] [search/symmetry.py](../../src/dracula/search/symmetry.py)
  — **154 lines.** Exhaustive early-turn destination grouping. Interfaces:
  `DestinationSymmetryGroup`, `destination_symmetry_groups`. Occupied pattern
  plus legal destinations enter; designated representative/member groups leave.
  Called by strategic action construction. Invariants: only the seven table
  patterns, no inferred symmetry, card-independent groups. Public board data.
  Tests: `test_search_symmetry.py` and strategic symmetry suites. Complexity:
  low, exact-contract sensitive.
- [ ] [strategic_actions.py](../../src/dracula/strategic_actions.py)
  — **223 lines.** Strategic card/destination groups and paired-destination
  resolution. Interfaces: `StrategicActionGroup`, `legal_action_indices`,
  `strategic_action_groups`, derived destination seed, concrete selection.
  Information state/legal actions enter; groups or one concrete legal action
  leave. Depends on symmetry/randomness; called by active policy and all grouped
  searches. Invariants: hand cards remain distinct, exact proxy, fair coin does
  not alter group choice. Actor-visible data. Tests: symmetry, Sam, active
  policy. Complexity: shared compatibility boundary.
- [ ] [action_contract.py](../../src/dracula/action_contract.py)
  — **293 lines.** Representative mask/proxy mapping and standalone logit
  selection. Interfaces: representative group/projection values, projection
  builder, mask/probability functions, `select_representative_action`, and
  resolution. Inputs engine mask/groups/logits; outputs one legal proxy/group/
  concrete action. Depends on strategic actions; called by model training and
  active inference. Invariants: exactly one proxy per group, masked actions get
  zero probability, canonical ties. Actor-visible/model data. Tests: Sam/BGC
  policy and search-policy suites. Complexity: serving/training agreement.
- [ ] [bgc_policy_model.py](../../src/dracula/bgc_policy_model.py)
  — **307 lines.** Compact permutation-invariant card-set policy network.
  Interfaces: 875→659 migration helpers, candidate/action mappings,
  `BGCPolicyModel`. Bool observation enters; float32 4×8 logits leave. Depends on
  PyTorch/cards/randomness; called by training/artifact runtime. Invariants:
  754,601 parameters, shared card scorer, no value head, deterministic init.
  Actor-visible features only. Tests: `test_bgc_policy_model_v2.py`, policy
  training/search policy. Complexity: tensor shape and artifact-sensitive.
- [ ] [bgc_policy.py](../../src/dracula/bgc_policy.py)
  — **356 lines.** Strict `pi0`/`pi1` policy artifact metadata, digest, save, and
  load. Interfaces: metadata/loaded records and build/save/load functions.
  State dictionaries and metadata enter; verified model/artifact identity
  leaves. Depends on model/digests; called by active policy, training, packaging,
  evaluation. Invariants: exact shapes, finite parameters, schema and digest
  agreement. Private weights, no game data. Tests: BGC policy/training/search/
  packaging. Complexity: provenance-critical.
- [ ] [active_policy.py](../../src/dracula/active_policy.py)
  — **216 lines.** Selected standalone `pi1` runtime and service executor.
  Interfaces: `ActivePolicyDecision`, `ActivePolicyRuntime`,
  `ActivePolicyExecutor`. Actor-visible turn context enters; representative and
  concrete legal action plus latency leave. Depends only on compact artifact,
  information, action contract, and deterministic seed. Called by production
  and explicit local `bgc-policy`. Invariants: one inference, canonical argmax,
  frozen paired-destination namespace, exact selected artifact. Handles model
  weights privately; logs only safe diagnostics. Tests: `test_search_policy.py`,
  stateless/API/packaging/e2e. Complexity: release-critical.
- [ ] [search_policy.py](../../src/dracula/search_policy.py)
  — **479 lines.** Explicit adapters for v1, shallow, nested Sam, BGC, pi0-BGC,
  and the active `pi1` compatibility wrapper. Public interfaces are the six
  `Inline*Executor` classes. Policy-turn requests enter; legal actions/results
  leave. Depends on specific controller modules; called only by explicit local
  selection and tests. Invariants: no silent fallback and exact controller
  settings. Actor-visible inputs; search internals private. Tests:
  `test_search_policy.py` and controller suites. Complexity: broad by design as
  the comparison surface.
- [ ] [policy_adapter.py](../../src/dracula/policy_adapter.py)
  — **179 lines.** Historical generic policy inference contract and masked
  action selector. Interfaces: adapter protocols/metadata, inference records,
  profiles, validators, `select_masked_action`. Called by archived policies and
  retained API compatibility, not production `pi1`. Invariants: legal masking,
  finite logits, recurrent-byte shape. Actor-visible plus private model state.
  Tests: local/model/API suites. Complexity: compatibility only.
- [ ] [local_controllers.py](../../src/dracula/api/local_controllers.py)
  — **232 lines.** Explicit local opponent resolver. Interface:
  `resolve_local_controller`. Environment/configuration enters; one named
  executor leaves. Depends on active and historical adapters; called by local
  app only. Invariants: missing artifact/configuration fails; `pi1` is default;
  no production fallback. Tests: API/search-policy/local preview. Data: model
  paths/configuration, no game state. Complexity: deliberate compatibility map.
- [ ] [neural-model.md](../../docs/neural-model.md)
  — **298 lines.** Selected 659-bit, 754,601-parameter architecture and
  representative-action inference contract.
- [ ] [model-training.md](../../docs/model-training.md)
  — **84 lines.** D0→`pi0`→D1→selected `pi1` provenance and objective.
- [ ] [search.md](../../docs/search.md) — **376
  lines.** Information-safe search and authoritative symmetry history; active
  only for explicit controls and training provenance.

**Tranche approval checks**

- [ ] Training and inference share one representative-action implementation.
- [ ] The compact model remains hand-permutation invariant and card-specific.
- [ ] `pi1` receives no authoritative opponent hand or search state.
- [ ] Paired-member choice is deterministic but outside model logits.
- [ ] Comments and docstrings meet the stated rule.
## Tranche 6 — Stateless API and narration

- [ ] [api/policy.py](../../src/dracula/api/policy.py)
  — **66 lines.** Neutral policy execution protocol and service response. Main
  interfaces: `PolicyTurnRequest`, `PolicyTurnResult`, `PolicyExecutor`,
  `UnavailablePolicyExecutor`, `ServiceResponse`. Engine turn context enters;
  action/result or typed service error leaves. Called by stateful/stateless
  services and all executors. Invariants: no transport coupling and no implicit
  controller. Actor-visible input; recurrent state only for compatibility.
  Tests: API, search policy, local policy. Complexity: low.
- [ ] [api/stateless_contracts.py](../../src/dracula/api/stateless_contracts.py)
  — **211 lines.** Strict production Pydantic commands, envelope, public game,
  narration, health, and error models. JSON enters/leaves FastAPI. Depends on
  shared API contracts; called by routes/service/frontend contract tests.
  Invariants: unknown fields rejected, seed/history only recovery, bounded
  command union, no model/private engine fields. Public data only. Tests:
  stateless/API/privacy/e2e. Complexity: public compatibility surface.
- [ ] [api/stateless_service.py](../../src/dracula/api/stateless_service.py)
  — **399 lines.** Deterministic replay, bounded LRU cache, command application,
  and required Dracula turns. Interfaces: replay/policy errors, `ReplayedGame`,
  cache records, `canonical_envelope_digest`, `StatelessGameplayService`.
  Envelope/command enter; replayed private state and projected response leave to
  trusted callers. Depends on engine, actor information, policy, projection;
  called by routes/narration/tests. Invariants: cache is optional, retries do
  not double-apply, malformed order fails, forced moves bypass policy. Handles
  private reconstructed state internally. Tests: complete stateless suite.
  Complexity: release-critical state machine.
- [ ] [api/stateless_projection.py](../../src/dracula/api/stateless_projection.py)
  — **122 lines.** Pure private-engine-to-public-view projection. Interface:
  `PublicGameSource`, `project_stateless_game`. Replayed state and human role
  enter; public status, board, human hand, scores, rounds, phase, legal moves
  leave. Depends on engine/presentation/contracts; called by stateless service.
  Invariant: opponent hand, stock, seed, masks, logits, and engine object never
  leave. Tests: stateless/privacy/API. Complexity: privacy-critical.
- [ ] [api/stateless_http.py](../../src/dracula/api/stateless_http.py)
  — **77 lines.** ASGI request-size middleware. Interface:
  `StatelessRequestBodyLimitMiddleware`. HTTP body chunks enter; bounded stream
  or 413 response leaves. Called by stateless app. Invariants: 64 KiB limit,
  streaming-safe rejection. Public transport bytes. Tests: stateless malformed/
  oversized requests. Complexity: low, protocol-sensitive.
- [ ] [api/stateless_routes.py](../../src/dracula/api/stateless_routes.py)
  — **129 lines.** Thin FastAPI route registration. Interface:
  `configure_stateless_routes`. Validated requests enter; service/narrator
  responses or mapped public errors leave. Depends on contracts/service/
  narration; called by stateless app. Invariants: route handlers delegate,
  narration failure remains nonfatal, no exception leaks. Public transport.
  Tests: stateless, narration, packaging. Complexity: low.
- [ ] [api/stateless_app.py](../../src/dracula/api/stateless_app.py)
  — **41 lines.** Stateless FastAPI assembly. Interface: `create_stateless_app`.
  Policy/narrator/cache/config enter; configured ASGI app leaves. Depends on
  middleware/routes/logging. Called by production and explicit local mode.
  Invariants: no repository/database and explicit CORS. Tests: API/packaging.
  Public transport only. Complexity: low composition root.
- [ ] [api/production_config.py](../../src/dracula/api/production_config.py)
  — **66 lines.** Central production environment parser. Interface:
  `ProductionSettings`. Environment strings enter; validated policy,
  narration, cache, and CORS settings leave. Called only by production app.
  Invariants: required `pi1` path, no local controller selector, bounded cache.
  Configuration data only. Tests: API/packaging. Complexity: low.
- [ ] [api/production.py](../../src/dracula/api/production.py)
  — **31 lines.** Lambda ASGI entry point. Interface:
  `create_production_app` and module `app`. Settings enter; stateless app with
  selected `pi1` and optional Bedrock leaves. Depends on active policy,
  production settings, stateless assembly. Invariant: no stateful fallback.
  Tests: API/Lambda packaging/container. Complexity: low, release-critical.
- [ ] [api/narration.py](../../src/dracula/api/narration.py)
  — **457 lines.** Approved prompt, grounded public cue construction, eligibility,
  fake adapter, and failure-isolated narration service. Interfaces include cue/
  prompt/provider records, `build_narration_prompt`,
  `derive_grounded_narration_cue`, `NarrationService`. Replayed game plus cue
  type enter; bounded text/unavailable response leaves. Depends on public engine
  results/contracts; called by routes/local preview. Invariants: opening,
  rounds 1–5, final only; no raw scores/seed/history/private cards sent to
  provider; failure never mutates gameplay. Tests: 518-line narration suite and
  e2e. Complexity: substantial grounding branches.
- [ ] [api/bedrock.py](../../src/dracula/api/bedrock.py)
  — **174 lines.** Bedrock Runtime transport and strict Converse parsing.
  Interfaces: `parse_bedrock_text`, `BedrockRuntimeAdapter`. Grounded prompt
  enters; bounded plain text/token/latency record leaves. Depends lazily on
  Botocore/Boto3; called only when narration enabled. Invariants: one allowlisted
  model, timeout, no retries, one text block, safe logs. External request sees
  grounded public facts only. Tests: Bedrock fake/parse/configuration tests.
  Complexity: provider wire variability.
- [ ] [production_logging.py](../../src/dracula/production_logging.py)
  — **30 lines.** JSON logging formatter. Interface: `JsonLogFormatter`.
  `LogRecord` enters; one structured line leaves. Called by container logging
  configuration. Invariants: bounded known fields and no private payload.
  Tests: packaging/container. Complexity: low.
- [ ] [api/presentation.py](../../src/dracula/api/presentation.py)
  — **218 lines.** Shared phase, score, played-move, and round-record projection.
  Interfaces: `phase_for_state`, `player_score`, `played_move`, `round_record`.
  Engine state/results enter; stateful-compatible public records leave. Called
  by both public projections. Invariants: orientation and scoring presentation
  agree across API modes. Public output from private state. Tests: gameplay API
  and stateless suite. Complexity: moderate mapping code.
- [ ] [api/contracts.py](../../src/dracula/api/contracts.py)
  — **247 lines.** Shared/stateful public response models. Interfaces include
  health, score, move, line, round, phase, game, event, and error models. Python
  values enter; strict JSON schemas leave. Called by presentation/stateful API
  and frontend compatibility. Invariants: closed schemas and no private fields.
  Public data. Tests: API/gameplay/frontend contracts. Complexity: compatibility
  surface.
- [ ] [api/repository.py](../../src/dracula/api/repository.py)
  — **366 lines.** Explicit local-only in-memory/SQLite session persistence and
  optimistic update errors. Interfaces: repository protocol and two
  implementations. Session records enter/leave local storage. Called by local
  app/service only. Invariants: atomic revision, request-id consistency,
  preserved recorded games. Handles complete local session/private engine
  state; never production. Tests: gameplay API/transactions. Complexity:
  SQLite transaction semantics.
- [ ] [api/session.py](../../src/dracula/api/session.py)
  — **421 lines.** Local persistent-session values, validation, canonical
  serialization, request fingerprints, and idempotency records. Interfaces:
  session records and serialization/response helpers. Private engine/policy
  state enters; local repository data leaves. Depends on engine/contracts/
  policy/presentation; called by local service/repository. Invariants: revision,
  complete reconstruction, exact historical format. Tests: gameplay API and
  policy recovery. Complexity: compatibility-heavy local persistence.
- [ ] [api/service.py](../../src/dracula/api/service.py)
  — **734 lines.** Retained repository-backed gameplay transaction service.
  Interface: `GameplayService`. Stateful requests/repository sessions enter;
  public stateful responses leave. Depends on engine/policy/session/repository;
  called by local routes/tests only. Invariants: request idempotency,
  optimistic concurrency, policy claims, score lifecycle. Handles private
  session state. Tests: `test_gameplay_api.py`, API/local policy. Complexity:
  large historical local path, intentionally outside production.
- [ ] [api/app.py](../../src/dracula/api/app.py) —
  **241 lines.** Compatible local application factory and stateful route
  assembly. Interface: `create_app` and module `app`. Environment/repository/
  controller choice enter; local or explicit stateless ASGI app leaves. Depends
  on local resolver/service/repository or stateless assembly. Invariants: local
  default is explicit; invalid modes fail. Tests: API/gameplay/local preview.
  Data depends on selected mode. Complexity: retained dual-mode composition.
- [ ] [api/local_preview.py](../../src/dracula/api/local_preview.py)
  — **89 lines.** Deterministic dummy narration preview. Interfaces:
  `LocalDummyNarrationAdapter`, `create_local_preview_app`, module `app`. Called
  by `make preview-dialogue`. Invariants: same cue timing/grounding, no Bedrock,
  explicit local only. Public cue data. Tests: local preview and frontend e2e.
  Complexity: low.
- [ ] [api/__init__.py](../../src/dracula/api/__init__.py)
  — **16 lines.** Side-effect-free package compatibility exports. No stateful
  app construction on import. Tests: static import/API. Complexity: low.
- [ ] [stateless-api.md](../../docs/stateless-api.md)
  — **180 lines.** Authoritative recovery envelope, routes, cache, replay,
  errors, and privacy contract.
- [ ] [narrator.md](../../docs/narrator.md) — **132
  lines.** Three cue classes, approved voice, Nova Lite selection, grounding,
  transport, timeout, and logging contract.

**Tranche approval checks**

- [ ] Every route delegates to one tested service boundary.
- [ ] Production imports cannot construct SQLite or a comparison controller.
- [ ] Every public response field is justified and private fields are absent.
- [ ] Narration is grounded, optional, and transaction-independent.
- [ ] Comments and docstrings meet the stated rule.

---

## Tranche 7 — Frontend state and presentation

### Contracts, transport, and recovery

- [ ] [contractPrimitives.ts](../../frontend/src/contractPrimitives.ts)
  — **281 lines.** Shared player, status, score, move, round, phase, and strict
  validation primitives. Wire JSON enters; typed public presentation values
  leave. Called by both API families and UI. Invariants: strict object/number/
  enum validation. Public data. Tests: contracts/API/store. Complexity: central
  validator helpers.
- [ ] [statefulContracts.ts](../../frontend/src/statefulContracts.ts)
  — **281 lines.** Retained local stateful request/response/event types and
  validators. Called by `api.ts`/`gameStore.ts`. Invariants: exact local API
  schema; no production use. Public browser data. Tests: contracts/stateful
  client/store. Complexity: compatibility only.
- [ ] [statelessContracts.ts](../../frontend/src/statelessContracts.ts)
  — **277 lines.** Production envelope, command, game, narration, health, and
  error types/validators. Called by stateless transport/store/recovery. Input:
  unknown JSON; output: strict types. Invariants: seed/history envelope and no
  model/private fields. Public browser data. Tests: contracts/stateless API/
  store/e2e. Complexity: public wire surface.
- [ ] [contracts.ts](../../frontend/src/contracts.ts)
  — **46 lines.** Explicit compatibility re-export of the original public
  frontend contract surface. No validators are reimplemented. Called mainly by
  tests/legacy imports. Tests: `contracts.test.ts`. Complexity: low.
- [ ] [gameControllerContract.ts](../../frontend/src/gameControllerContract.ts)
  — **49 lines.** UI-facing controller snapshot and command interface shared by
  stateful/stateless stores. Controller state/actions enter components; view
  snapshots leave. Invariants: presentation does not know transport mode.
  Public data. Tests: app/gameplay/stores. Complexity: low.
- [ ] [statelessApi.ts](../../frontend/src/statelessApi.ts)
  — **127 lines.** Production fetch transport and strict response/error parsing.
  Interfaces: stateless client methods. Typed request enters; validated response
  leaves. Depends on stateless contracts and build-time API origin; called by
  stateless store. Invariants: no localhost production default, JSON/error
  failures explicit. Public recovery/game data. Tests: `statelessApi.test.ts`,
  production verifier/e2e. Complexity: low.
- [ ] [statelessRecovery.ts](../../frontend/src/statelessRecovery.ts)
  — **42 lines.** Browser storage of the confirmed recovery envelope only.
  Interfaces: read/write/clear helpers. Public seed/history enters/leaves local
  storage. Called by stateless store. Invariants: malformed state clears safely;
  no model/game object/logits. Tests: stateless store/e2e. Complexity: low.
- [ ] [statelessProjection.ts](../../frontend/src/statelessProjection.ts)
  — **112 lines.** Pure stateless wire-to-view projection and optimistic human/
  delayed-opening previews. Response or confirmed snapshot enters; view state
  leaves. Called by stateless store. Invariants: human card appears immediately,
  Dracula waits for minimum delay, no mutation. Tests: stateless store.
  Complexity: timing-sensitive pure transformations.
- [ ] [statelessNarration.ts](../../frontend/src/statelessNarration.ts)
  — **87 lines.** Narration request eligibility, generation tokens, stale-result
  protection, and display state. Envelope/cue enter; ready/unavailable text state
  leaves. Called by stateless store. Invariants: three cues, scoring-time fetch,
  reveal afterward, failure nonblocking. Public text/envelope. Tests: stateless
  store, commentary, e2e. Complexity: asynchronous ordering.
- [ ] [statelessGameStore.ts](../../frontend/src/statelessGameStore.ts)
  — **374 lines.** Production gameplay controller sequencing transport,
  confirmed recovery, optimistic placement, minimum Dracula delay, scoring, and
  narration. Implements the shared controller interface. Inputs user actions/
  API responses; outputs subscribed snapshots. Depends on the four owners above;
  called by `App`. Invariants: one in-flight action, stale response protection,
  confirmed-only persistence, no autoscroll. Public browser state. Tests:
  454-line store suite and e2e. Complexity: main frontend state machine.
- [ ] [api.ts](../../frontend/src/api.ts) — **135
  lines.** Retained local stateful fetch client. Called by `gameStore`; no
  production selection. Invariants: strict stateful parsing/error mapping.
  Public local session data. Tests: `api.test.ts`. Complexity: compatibility.
- [ ] [gameStore.ts](../../frontend/src/gameStore.ts)
  — **269 lines.** Retained stateful controller for local compatibility. API
  events/actions enter; shared controller snapshots leave. Called only when
  explicitly selected. Invariants: revision/request handling and UI contract.
  Public local data. Tests: `gameStore.test.ts`. Complexity: compatibility.

### Presentation and timing

- [ ] [App.tsx](../../frontend/src/App.tsx) — **273
  lines.** Hash-route selection, start/game/rules shells, controller lifecycle,
  and approved page composition. Input: controller snapshots/browser route;
  output: React UI. Depends on stateless store by default and presentation
  components. Invariants: `/Dracula/` hash paths, explicit start roles, approved
  mobile/desktop order. Public view only. Tests: App/gameplay/usability/e2e.
  Complexity: top-level composition.
- [ ] [gameplay.tsx](../../frontend/src/gameplay.tsx)
  — **256 lines.** Turn status, scores, coffin, hand, cheat sheet, and action
  components. Public view/actions enter; accessible controls/markup leave.
  Depends on card assets/contracts. Called by App/scoring. Invariants: legal
  highlights, exact grid, seen cards, immediate placement, no hidden state.
  Tests: gameplay/usability/e2e. Complexity: moderate presentation set.
- [ ] [DraculaCommentary.tsx](../../frontend/src/DraculaCommentary.tsx)
  — **172 lines.** Portrait selection and retro dialogue typing. Public score/
  round/narration/timing enter; portrait and text presentation leave. Called by
  App. Invariants: static angriest loss portrait, winning grin only in later
  rounds, face transition at text, stable mobile image geometry. Tests:
  254-line commentary suite/usability/e2e. Complexity: animation/timing.
- [ ] [scoringStateMachine.ts](../../frontend/src/scoringStateMachine.ts)
  — **300 lines.** Pure server scoring-step conversion and deterministic frame
  timeline. Scoring records/timing enter; immutable presentation models/frames
  leave. Called by scoring component. Invariants: engine values only, locked
  cadence, transition/final holds. Public data. Tests: 93-line state-machine
  suite and scoring component. Complexity: temporal sequence.
- [ ] [ScoringPresentation.tsx](../../frontend/src/ScoringPresentation.tsx)
  — **377 lines.** Executes scoring frames, line emphasis, calculations, round
  totals, next-round/play-again actions, and narration reveal. Inputs public
  scoring model/controller actions; output approved animation UI. Depends on
  state machine/gameplay components. Invariants: number-column alignment,
  reserved mobile dialogue space, fixed status anchor, one completion callback.
  Tests: 301-line component suite, usability, e2e. Complexity: highest frontend
  presentation risk.
- [ ] [scoringTestFixtures.ts](../../frontend/src/scoringTestFixtures.ts)
  — **189 lines.** Deterministic public scoring fixtures shared by frontend
  tests. Inputs none; outputs representative scoring records. Test-only public
  data. Complexity: fixture fidelity.
- [ ] [RulesPage.tsx](../../frontend/src/RulesPage.tsx)
  — **114 lines.** In-app rules rendering and navigation. Public static content
  to React UI. Called by App. Invariant: agrees with `docs/rules.md`. Tests:
  App/usability/e2e. Complexity: low.
- [ ] [CardFace.tsx](../../frontend/src/CardFace.tsx)
  — **5 lines.** Shared card image wrapper. Card ID/alt/class enter; image leaves.
  Depends on asset map; called by gameplay/scoring. Public visible card only.
  Tests: gameplay/card assets. Complexity: none.
- [ ] [cardAssets.ts](../../frontend/src/cardAssets.ts)
  — **44 lines.** Generated-manifest-backed canonical card asset lookup.
  Interface: card image helpers/constants. Called by card UI. Invariants: all 54
  cards and back exist beneath Vite base. Public assets. Tests:
  `cardAssets.test.ts` and production verifier. Complexity: low.
- [ ] [siteConfig.ts](../../frontend/src/siteConfig.ts)
  — **24 lines.** Base-aware `/Dracula/` destinations for home/rules/about/
  contact. Called by App/Rules. Invariant: no hard-coded root breakage. Tests:
  `siteConfig.test.ts`. Complexity: low.
- [ ] [main.tsx](../../frontend/src/main.tsx) — **21
  lines.** React DOM entry plus font and ordered stylesheet imports. Called by
  Vite. Invariants: one root and approved fonts. Public UI. Build/e2e coverage.
- [ ] [vite-env.d.ts](../../frontend/src/vite-env.d.ts)
  — **15 lines.** Types the build-time API/base variables. No runtime output.
  TypeScript coverage.
- [ ] [styles.css](../../frontend/src/styles.css)
  — **5 lines.** Fixed cascade import order.
- [ ] [base.css](../../frontend/src/styles/base.css)
  — **202 lines.** Fonts, tokens, shell, buttons, links, and shared framing.
- [ ] [gameplay.css](../../frontend/src/styles/gameplay.css)
  — **326 lines.** Board, cards, hand, Dracula bar, score, cheat sheet, and
  gameplay animation declarations.
- [ ] [scoring.css](../../frontend/src/styles/scoring.css)
  — **110 lines.** Scoring overlay, line math, totals, and score-card alignment.
- [ ] [rules.css](../../frontend/src/styles/rules.css)
  — **26 lines.** Rules-only typography and grid.
- [ ] [responsive.css](../../frontend/src/styles/responsive.css)
  — **120 lines.** Container/mobile/reduced-motion geometry. Inputs viewport and
  motion preference; output responsive cascade. Invariants across all style
  modules: exact approved values/order, no card resampling flicker, no mobile
  overflow/autoscroll, stable dialogue/score/status positions. Public UI. Tests:
  usability, scoring, commentary, and browser geometry. Complexity: cascade and
  narrow-height interaction; review all six style files together.

**Frontend state flow**

```text
statelessApi -> statelessGameStore -> App/components
       ^              |                  |
recovery envelope     `-> narration     `-> scoring timeline
```

- [ ] Confirm browser persistence contains only seed and accepted history.
- [ ] Trace immediate human placement and delayed Dracula placement.
- [ ] Trace round-5 transition and round-6 final narration separately.
- [ ] Check approved desktop/mobile layout declarations without redesign.
- [ ] Comments and docstrings meet the stated rule.

---

## Tranche 8 — Deployment and operations

### Command and build surface

- [ ] [README.md](../../README.md) — **119 lines.**
  Developer/operator entry point for local `pi1`, explicit controls, tests,
  historical corpus inspection, SQLite reports, and deployment topology.
  Depends on Make targets/docs. Public documentation. Risk: stale command.
- [ ] [Makefile](../../Makefile) — **242 lines.**
  Main local command surface. Inputs variables/artifact paths; outputs servers,
  tests, builds, reports, corpus operations, or release context. Depends on
  Python/npm/Docker/tools. Invariants: `bgc-policy` default, comparisons
  explicit, shared local environment exact, protected data never removed by a
  verification target. Tests: dry-run of all targets and local preview/package
  suites. Complexity: many retained explicit modes.
- [ ] [pyproject.toml](../../pyproject.toml) — **39
  lines.** Python package, dependencies, test configuration, and six maintained
  data/evaluation entry points. Inputs packaging tools; output wheel/CLI
  metadata. Invariants: Python 3.12+, no production database/cloud SDK core
  dependency. Package/import/CLI checks. Complexity: low.
- [ ] [frontend/package.json](../../frontend/package.json)
  — **48 lines.** Locked frontend commands and dependency ownership. Inputs npm
  scripts; outputs tests/build/preview. Depends on lockfile. Invariants:
  production verify follows build, card preparation precedes test/build.
- [ ] [vite.config.ts](../../frontend/vite.config.ts)
  — **39 lines.** Vite base, API proxy, and production environment validation.
  Inputs build variables; output config. Invariants: production origin explicit,
  local proxy preserved, `/Dracula/` supported. Tests: site config/build/e2e.
- [ ] [playwright.config.ts](../../frontend/playwright.config.ts)
  — **64 lines.** Production-shaped ASGI/Vite browser harness. Inputs selected
  artifact and fixed ports; output one-worker Chrome run. Invariants: no server
  reuse, `/Dracula/` base, reduced-motion deterministic timing. Test: e2e run.
- [ ] [prepare-card-assets.mjs](../../frontend/scripts/prepare-card-assets.mjs)
  — **74 lines.** Validates/copies immutable card/Vampire assets and writes the
  generated manifest. Inputs source packs and public assets; output ignored
  generated card tree/manifest. Invariants: exact 54-card coverage and no source
  mutation. Tests: card assets/build.
- [ ] [verify-production-build.mjs](../../frontend/scripts/verify-production-build.mjs)
  — **50 lines.** Inspects built HTML/assets for base/API/local-origin errors.
  Input `frontend/dist`; output pass/fail. Called by `verify:build` and Pages.
  Invariants: `/Dracula/`, production API, no localhost. Build tests.

### Lambda and release tools

- [ ] [build_lambda_context.py](../../tools/build_lambda_context.py)
  — **189 lines.** Creates isolated ignored container context after verifying
  selected `pi1` SHA-256. Inputs authored runtime/container files and ignored
  artifact; output generated context. Called by Make/release. Invariants: fail
  closed, no datasets/SQLite/history, deterministic manifest. Tests:
  `test_lambda_packaging.py`. Private weights copied only to ignored output.
- [ ] [validate_lambda_container.py](../../tools/validate_lambda_container.py)
  — **495 lines.** Runs read-only container and exercises health, game, replay,
  cache, narration fake, logs, memory, and artifact identity. Input image;
  output ignored JSON/log measurements. Depends on Docker/HTTP. Invariants: no
  host persistence or private response fields. Packaging tests/report.
  Complexity: integration harness.
- [ ] [build_release_candidate.py](../../tools/build_release_candidate.py)
  — **308 lines.** Seals source/diff/model/image/infrastructure/frontend/report
  identities. Input reviewed tree and image; output ignored manifest. Depends on
  Git/Docker/hash helpers. Invariants: deterministic file selection, exact `pi1`,
  only two remaining user decisions recorded. Tests: packaging. Complexity:
  provenance-sensitive.

### Container and AWS definitions

- [ ] [Dockerfile](../../deployment/container/Dockerfile)
  — **37 lines.** arm64 Lambda Python image with Lambda Web Adapter, runtime,
  `pi1`, and stateless ASGI entry. Inputs generated context; output immutable
  image. Invariants: digest-pinned bases/adapter, non-root runtime, read-only
  compatibility, no database. Container validation. Private weights embedded.
- [ ] [requirements-constraints.txt](../../deployment/container/requirements-constraints.txt)
  — **30 lines**; [requirements-lambda.txt](../../deployment/container/requirements-lambda.txt)
  — **5 lines**; [requirements-torch.txt](../../deployment/container/requirements-torch.txt)
  — **2 lines.** Pinned Lambda/Web/API/AWS and CPU-Torch dependency inputs.
  Output deterministic install resolution. Invariant: no SageMaker/database.
  Package/container tests. Review together.
- [ ] [logging.json](../../deployment/container/logging.json)
  — **37 lines.** Uvicorn/application JSON log wiring. Input log records; output
  CloudWatch-ready lines. Invariant: no request bodies/private game state.
  Packaging/container tests.
- [ ] [ecr.yaml](../../infrastructure/ecr.yaml) — **30
  lines.** Immutable scan-on-push ECR repository. Inputs project/repository
  parameters; outputs URI/ARN. Called by release commands. Invariants: images
  retained and mutable tags rejected. Static/package checks.
- [ ] [application.yaml](../../infrastructure/application.yaml)
  — **432 lines.** Lambda, IAM, HTTP API/routes/CORS/logs/alarms/optional SNS,
  budget, Regional domain, and mapping. Inputs image and deployment parameters;
  outputs API/domain/resource IDs. Invariants: least-privilege Bedrock model ARN,
  no database, bounded resources, explicit conditions. Static parsing and
  packaging tests. Complexity: largest deployment contract.
- [ ] [staging.json](../../infrastructure/parameters/staging.json)
  — **20 lines**; [production.json](../../infrastructure/parameters/production.json)
  — **20 lines.** Generic non-secret 18-parameter defaults. Account-bound image,
  Nova Lite ARN, ACM, notification, and domain values are rendered outside Git.
  Invariants: 2,048 MB/20 s/concurrency 2 current staging values; narration and
  custom domain disabled until rendering. Static/package checks.
- [ ] [infrastructure README](../../infrastructure/README.md)
  — **20 lines.** Ownership of the sole CloudFormation mechanism and selected
  personal account/`us-east-1`/Nova Lite inputs.
- [ ] [Pages workflow](../../.github/workflows/pages.yml)
  — **65 lines.** Pull-request verification and explicit manual Pages publish.
  Inputs reviewed Git/lockfile; output verified `frontend/dist` only on approved
  dispatch. Invariants: no AWS credential, `/Dracula/` and production API fixed,
  PRs never publish. Build/e2e/workflow review.
- [ ] [deployment.md](../../docs/deployment.md) —
  **356 lines.** Authoritative build, staging, cutover, rollback, teardown, and
  selected/remaining decision contract.
- [ ] [architecture.md](../../docs/architecture.md)
  — **223 lines.** Final stateless topology and refactored module ownership.
- [ ] [frontend-experience.md](../../docs/frontend-experience.md)
  — **394 lines.** Approved responsive, timing, portrait, scoring, and narration
  presentation contract.
- [ ] [remaining-user-decisions.md](../../reports/active/remaining-user-decisions.md)
  — **56 lines.** Only production resource/notification approval and final
  go-live remain.

### Explicit environment surface

- [ ] Production application: `DRACULA_POLICY_ARTIFACT`,
  `DRACULA_NARRATION_ENABLED`, `DRACULA_REPLAY_CACHE_ENTRIES`,
  `DRACULA_ALLOWED_ORIGIN`, `DRACULA_BEDROCK_REGION`,
  `DRACULA_BEDROCK_MODEL_ID`, `DRACULA_NARRATION_TIMEOUT_SECONDS`, and
  `DRACULA_NARRATION_MAX_TOKENS`.
- [ ] Frontend build: `VITE_API_ORIGIN`, `VITE_BASE_PATH`; local proxy only:
  `DRACULA_API_PROXY_TARGET`.
- [ ] Release shell: `AWS_REGION=us-east-1`, account/profile credentials,
  Bedrock ARN, image URI, resource settings, ACM ARN, Cloudflare token/zone and
  `CLOUDFLARE_PROXY=true`. Secrets stay outside Git.

### AWS resource checklist

- [ ] Immutable ECR repository and retained image digests.
- [ ] One arm64 Lambda and least-privilege execution role.
- [ ] One Regional API Gateway HTTP API, stage, integrations, five routes, and
  Lambda permission.
- [ ] Lambda/API log groups and error alarms; optional SNS and Budget.
- [ ] Optional Regional custom domain, ACM certificate input, and API mapping.
- [ ] Proxied Cloudflare `api` CNAME; ACM validation CNAME remains DNS-only.
- [ ] GitHub Pages is separate static hosting and receives no AWS credential.

### Rollback checklist

- [ ] API: redeploy the previous recorded immutable ECR image digest.
- [ ] Narration: deploy with `NarrationEnabled=false`; gameplay remains live.
- [ ] Frontend: dispatch Pages from the previous reviewed commit/build digest.
- [ ] Domain: remove mapping only after directing DNS safely; retain execute-api
  endpoint for diagnosis.
- [ ] Staging: delete application stack and wait; retain ECR images.
- [ ] Production: delete only after explicit authorization; separately handle
  DNS, ACM, ECR contents, alarms, and budget.

**Tranche approval checks**

- [ ] Commands match actual CLI help and Make dry runs.
- [ ] Build context contains exactly one verified model and no local state.
- [ ] Every AWS permission/resource is required by the locked topology.
- [ ] Rollback is possible without retraining or reconstructing a dataset.
- [ ] Comments and docstrings meet the stated rule.

---

## Tranche 9 — Tests, maintained historical readers, and rollback evidence

Production behavior is complete after tranche 8. This tranche verifies the
offline provenance and compatibility surface that keeps selected `pi1` and
historical artifacts inspectable. These modules are not production fallbacks.

### Current training-lineage and evaluation modules

- [ ] [bgc_policy_training.py](../../src/dracula/bgc_policy_training.py)
  — **2,605 lines.** D0/D1 snapshot verification, visit projection,
  distributional training, deterministic resume/checkpoints/export, CLI.
  Sealed manifests/rows/config enter; candidate artifacts/metrics leave.
  Depends on BGC model/artifact, action contract, engine information; called by
  installed training CLI. Invariants: visits sum 128, representative-only
  targets, no value/illegal/one-hot loss, deck split isolation. Dataset rows are
  actor-visible only; weights private. Tests: 394-line training suite. Risk:
  large artifact/recovery implementation.
- [ ] [bgc_policy_migration.py](../../src/dracula/bgc_policy_migration.py)
  — **451 lines.** Physical 875→659 observation/action migration for the frozen
  D0 snapshot. Sealed source snapshot enters; immutable compact snapshot leaves.
  Depends on compact model/training schemas; called by `build_pi1_dataset.py`.
  Invariants: row count/order/targets/splits/digests preserved. Actor-visible
  rows only. Tests: BGC model/training. Risk: one-time provenance path.
- [ ] [bgc_policy_evaluation.py](../../src/dracula/bgc_policy_evaluation.py)
  — **2,722 lines.** Standalone controls, paired matches, confidence intervals,
  acceptance evidence, throughput, and CLI. Artifacts/fixed fixtures enter;
  games/reports/optional accepted bundle leave. Depends on engine, information,
  BGC search/policies. Called by installed evaluation CLI and retained tools.
  Invariants: paired roles/decks, exact scoring, actor-local policy, atomic
  acceptance. Private simulated state stays offline. Tests: 500-line evaluation
  suite. Risk: broad evaluation surface.
- [ ] [accepted_pi0.py](../../src/dracula/accepted_pi0.py)
  — **255 lines.** Strict accepted-`pi0` manifest loader and continuation adapter.
  Acceptance bundle/actor information enter; legal continuation leaves. Called
  by D1 evaluation/miner only. Invariants: every digest and acceptance result
  verified, no fallback. Private weights; actor-visible game input. Tests: BGC
  evaluation/D1 miner. Risk: historical naming can be confused with selected
  `pi1`.
- [ ] [bgc_pi0_miner.py](../../src/dracula/bgc_pi0_miner.py)
  — **1,228 lines.** D1 collector using accepted `pi0` continuations and BGC-128
  root visits. Accepted bundle/config enter; balanced sealed game corpus leaves.
  Depends on accepted adapter/search/artifact utilities; installed CLI. Invariants:
  D0/D1 separation, 42 rows/game, visits sum 128, deterministic workers/resume,
  disk guard/privacy. Tests: 494-line miner suite. Risk: completed large offline
  pipeline.
- [ ] [belief_greedy_miner.py](../../src/dracula/belief_greedy_miner.py)
  — **1,171 lines.** Balanced BGC D0 collector/benchmark and continuous corpus
  operations. Deterministic trajectory profiles/config enter; sealed 42-row
  games/manifests leave. Depends on BGC search/engine; installed CLI. Invariants:
  equal placements, five-game blocks, exact visits, atomic resume/privacy. Tests:
  135-line miner suite. Risk: completed operational code retained for provenance.
- [ ] [search/belief_greedy.py](../../src/dracula/search/belief_greedy.py)
  — **1,007 lines.** BGC outer UCT and one-ply belief-greedy response evaluator.
  Actor information/config/seed enter; visits/action/diagnostics leave. Called by
  D0 miner/evaluation/explicit local comparison. Invariants: shared belief
  completions, actor-relative exact scoring, information safety, deterministic
  selection. Private simulated states. Tests: belief-greedy search/miner/BGC
  evaluation. Risk: performance-sensitive historical teacher.
- [ ] [build_pi1_dataset.py](../../tools/build_pi1_dataset.py)
  — **199 lines.** Reproduces compact D1 snapshot migration. Inputs sealed D1;
  outputs selected training snapshot/report. Invariants: source immutability and
  digest verification. Tests live in migration/training suites. Offline private
  manifests, actor-visible rows. Risk: path defaults point to ignored evidence.
- [ ] [evaluate_sam_policy.py](../../tools/evaluate_sam_policy.py)
  — **733 lines.** Historical standalone Sam fixture/game evaluation. Artifact
  and fixed decks enter; Markdown/JSON evidence leaves. Uses engine/controls;
  no production caller. Invariants: paired roles and legal actions. Tests:
  policy/evaluation suites. Risk: historical name.
- [ ] [run_pi1_incremental_matchups.py](../../tools/run_pi1_incremental_matchups.py)
  — **170 lines.** Runs and reports fixed incremental `pi1` matchup batches.
  Artifacts/decks enter; progress/report leave. Depends on BGC evaluation. No
  production data. Risk: experimental operator utility.
- [ ] [run_sam_policy_training_smoke.py](../../tools/run_sam_policy_training_smoke.py)
  — **1,015 lines.** Historical Sam-policy snapshot/training/device smoke
  orchestrator. Sealed sample/config enter; smoke artifacts/report leave. Depends
  on Sam trainer. Invariants: bounded, deterministic, no current training
  selection. Risk: large one-off retained for evidence.

### Historical artifact readers and explicit controls

Each entry below is maintained only enough to import, read its artifacts, and
pass its named regression tests. Inputs and outputs remain offline; any private
engine/model state stays inside the historical computation. None is called by
`dracula.api.production`.

- [ ] [sam_miner.py](../../src/dracula/sam_miner.py)
  — **3,934 lines.** Branched Sam corpus/cache/shards/resume/parallel CLI;
  deterministic config → sealed historical corpus. Tests: `test_sam_miner.py`.
  Invariants: branch counts/digests/privacy. Risk: largest retained module.
- [ ] [sam_policy.py](../../src/dracula/sam_policy.py)
  — **719 lines.** Historical 738,569-parameter policy/artifact contract;
  observation → logits, state dict ↔ verified artifact. Tests:
  `test_sam_policy.py`. Invariant: unchanged historical weights/schema.
- [ ] [sam_policy_training.py](../../src/dracula/sam_policy_training.py)
  — **2,679 lines.** Historical one-hot Sam trainer/snapshot CLI; sealed rows →
  checkpoints/artifact. Tests: `test_sam_policy_training.py`. Invariants:
  deterministic resume and proxy target.
- [ ] [standalone_policy.py](../../src/dracula/standalone_policy.py)
  — **302 lines.** Historical Sam standalone executor; actor information → legal
  action. Tests: `test_standalone_policy.py`. Invariants: mask and fair coin.
- [ ] [search/nested_strategic.py](../../src/dracula/search/nested_strategic.py)
  — **699 lines.** Sam 32×32 nested information-set search; actor information →
  group/action/visits. Tests: strategic/search-policy. Invariants: actor-relative
  values and exact terminal scoring.
- [ ] [search/sam_teacher.py](../../src/dracula/search/sam_teacher.py)
  — **1,164 lines.** Grouped Sam-128 teacher; information state → deterministic
  visits/group/action. Tests: `test_sam_teacher.py`/miner. Invariants: 128×128
  contract and information isolation.
- [ ] [search/planner.py](../../src/dracula/search/planner.py)
  — **398 lines.** Frozen v1 information-set UCT; information → visits/action.
  Tests: `test_search_planner.py`. Invariants: golden behavior and normalized
  terminal return.
- [ ] [search/strategic.py](../../src/dracula/search/strategic.py)
  — **1,588 lines.** Shallow Teacher v2 and response-student modes; information
  → grouped search result/diagnostics. Tests: strategic/hybrid/symmetry.
  Invariants: exact completed-round scoring and active helper re-exports.
- [ ] [search/guided.py](../../src/dracula/search/guided.py)
  — **760 lines.** Historical policy/value-guided information-set search;
  information/model priors → visits/action. Tests: `test_guided_search.py`.
  Invariant: no production selection.
- [ ] [guided_policy.py](../../src/dracula/guided_policy.py)
  — **83 lines.** Guided-search service adapter; policy request → action/result.
  Tests: `test_guided_policy.py`. Explicit local mode only.
- [ ] [policy_value.py](../../src/dracula/policy_value.py)
  — **669 lines.** Historical feed-forward policy/value model, loss, optimizer,
  artifact. Tests: `test_policy_value.py`. Invariant: incompatible with `pi1`.
- [ ] [response_distillation.py](../../src/dracula/response_distillation.py)
  — **385 lines.** Response group targets/observer/ranking loss; response
  evaluations → ranking examples/loss. Tests: response-distillation/hybrid.
- [ ] [response_dataset.py](../../src/dracula/response_dataset.py)
  — **1,497 lines.** Historical response shards/manifests/resume CLI; captured
  response states → sealed dataset. Tests: `test_response_dataset.py`.
- [ ] [response_ranker.py](../../src/dracula/response_ranker.py)
  — **1,776 lines.** Historical pairwise response-ranker trainer/artifact;
  response dataset → checkpoint/ranker. Tests: `test_response_ranker.py`.
- [ ] [search/response_student.py](../../src/dracula/search/response_student.py)
  — **138 lines.** Response-ranker group evaluator used by the historical hybrid;
  actor information/groups → scores. Tests: hybrid/response suites.
- [ ] [models.py](../../src/dracula/models.py) —
  **345 lines.** Historical recurrent `Policy` and separate `Critic`; observation
  and hidden state → logits/value. Tests: `test_models.py`. Incompatible with
  active policy.
- [ ] [local_policy.py](../../src/dracula/local_policy.py)
  — **290 lines.** Archived recurrent artifact loader/service executor; policy
  request/hidden bytes → action/new hidden state. Tests: `test_local_policy.py`.
- [ ] [legacy_policy_contract.py](../../src/dracula/legacy_policy_contract.py)
  — **11 lines.** Historical recurrent shape constants. Tests: models/bridge.
- [ ] [collection.py](../../src/dracula/collection.py)
  — **1,291 lines.** PPO rollout schedules/trajectories/shards; fixtures/policies
  → historical actor/critic data. Tests: `test_collection.py`.
- [ ] [optimization.py](../../src/dracula/optimization.py)
  — **960 lines.** Historical PPO preparation/loss/optimizer/replay; trajectories
  → updated population/reports. Tests: `test_optimization.py`.
- [ ] [training_config.py](../../src/dracula/training_config.py)
  — **533 lines.** Historical PPO TOML resolution/canonical manifest. Tests:
  training/optimization. Invariant: archived run compatibility.
- [ ] [training.py](../../src/dracula/training.py)
  — **1,260 lines.** Historical PPO phase orchestration/recovery; config/run →
  population artifacts. Tests: `test_training.py`.
- [ ] [evaluation.py](../../src/dracula/evaluation.py)
  — **406 lines.** Historical population evaluation schedules/metrics; policies
  and fixtures → results. Tests: optimization/training/evaluation callers.
- [ ] [tournament.py](../../src/dracula/tournament.py)
  — **701 lines.** Archived candidate tournament CLI; policy set/fixtures →
  report/artifacts. Tests: `test_tournament.py`.
- [ ] [supervised.py](../../src/dracula/supervised.py)
  — **1,380 lines.** Historical policy/value search-distillation trainer;
  teacher dataset → checkpoints/export. Tests: `test_supervised.py`.
- [ ] [teacher.py](../../src/dracula/teacher.py) —
  **1,881 lines.** Historical search-teacher collector/shards/manifests; fixtures
  → training rows. Tests: `test_teacher.py`.
- [ ] [expert.py](../../src/dracula/expert.py) —
  **1,588 lines.** Historical expert-iteration collection/training/replay/
  acceptance orchestration. Tests: `test_expert.py`.
- [ ] [expert_evaluation.py](../../src/dracula/expert_evaluation.py)
  — **708 lines.** Historical expert candidate/controller/fixture evaluation.
  Tests: `test_expert_evaluation.py`.
- [ ] [search/defensive_fixtures.py](../../src/dracula/search/defensive_fixtures.py)
  — **317 lines.** Defensive fixed-state definitions/replay. Tests: strategic
  validation. Synthetic private states; no runtime caller.
- [ ] [search/strategic_fixtures.py](../../src/dracula/search/strategic_fixtures.py)
  — **514 lines.** Constructive/defensive fixture semantics/features. Tests:
  strategic validation. Synthetic private states.
- [ ] [search/diagnostic.py](../../src/dracula/search/diagnostic.py)
  — **201 lines.** Perfect-information round solvers for diagnostic evidence.
  Tests: search validation. Invariant: cannot enter gameplay.
- [ ] [search/signal_measurement.py](../../src/dracula/search/signal_measurement.py)
  — **1,124 lines.** Historical symmetry/visit signal harness; fixed states →
  raw/Markdown metrics. Tests: `test_search_signal_measurement.py`.
- [ ] [search/strategic_validation.py](../../src/dracula/search/strategic_validation.py)
  — **1,011 lines.** Teacher v2 validation/benchmark CLI. Tests:
  `test_strategic_validation.py`. Offline only.
- [ ] [search/validation.py](../../src/dracula/search/validation.py)
  — **1,216 lines.** Frozen v1 validation/comparison CLI. Tests:
  `test_search_validation.py`. Offline only.
- [ ] [search/__init__.py](../../src/dracula/search/__init__.py)
  — **200 lines.** Lazy compatibility export map. Inputs attribute imports;
  outputs owning-module symbols without eager historical imports. Tests: static
  imports/all search suites. Invariant: side-effect-free narrow imports.
- [ ] [cli.py](../../src/dracula/cli.py) — **85
  lines.** Unregistered archived PPO CLI retained for old runs. Tests: training
  paths/imports. No production caller.
- [ ] [__init__.py](../../src/dracula/__init__.py)
  — **1 line.** Package marker; no import side effects.

### Backend test modules

All entries expose pytest cases rather than runtime APIs. Their inputs are
deterministic fixtures/artifacts; outputs are assertions. Private state appears
only inside tests targeting privacy/search/engine boundaries. Review each after
its target implementation.

| Review | Test module | Lines | Primary invariants |
| --- | --- | ---: | --- |
| [ ] | [test_foundation.py](../../tests/test_foundation.py) | 266 | cards, seed streams, immutable basics |
| [ ] | [test_engine.py](../../tests/test_engine.py) | 418 | deals, legality, scoring, lifecycle, serialization, golden behavior |
| [ ] | [test_bridge.py](../../tests/test_bridge.py) | 468 | player-relative encoding, mapping, forced moves, legacy transition |
| [ ] | [test_search_information.py](../../tests/test_search_information.py) | 314 | actor privacy, fingerprints, determinization |
| [ ] | [test_search_symmetry.py](../../tests/test_search_symmetry.py) | 163 | authoritative table only |
| [ ] | [test_bgc_policy_model_v2.py](../../tests/test_bgc_policy_model_v2.py) | 112 | compact model shape, permutation invariance, mappings |
| [ ] | [test_search_policy.py](../../tests/test_search_policy.py) | 322 | exact selected `pi1` and explicit controller behavior |
| [ ] | [test_api.py](../../tests/test_api.py) | 108 | app assembly, public contracts, health |
| [ ] | [test_stateless_gameplay_api.py](../../tests/test_stateless_gameplay_api.py) | 394 | replay, retries, cache, privacy, complete games |
| [ ] | [test_bedrock_narration.py](../../tests/test_bedrock_narration.py) | 518 | cues, grounding, provider parsing/failure/privacy |
| [ ] | [test_lambda_packaging.py](../../tests/test_lambda_packaging.py) | 182 | build context, artifact, infrastructure, logging, release manifest |
| [ ] | [test_local_preview.py](../../tests/test_local_preview.py) | 45 | dummy narration composition |
| [ ] | [test_gameplay_api.py](../../tests/test_gameplay_api.py) | 562 | local repository/API transactions and recovery |
| [ ] | [frontend_production_app.py](../../tests/frontend_production_app.py) | 34 | production e2e ASGI harness |
| [ ] | [test_belief_greedy_search.py](../../tests/test_belief_greedy_search.py) | 180 | BGC response/search semantics |
| [ ] | [test_belief_greedy_miner.py](../../tests/test_belief_greedy_miner.py) | 135 | balanced D0 games/artifacts |
| [ ] | [test_bgc_policy_training.py](../../tests/test_bgc_policy_training.py) | 394 | migration, visit loss, resume, export |
| [ ] | [test_bgc_policy_evaluation.py](../../tests/test_bgc_policy_evaluation.py) | 500 | paired controls, confidence, acceptance |
| [ ] | [test_bgc_pi0_miner.py](../../tests/test_bgc_pi0_miner.py) | 494 | accepted adapter and D1 separation/recovery |
| [ ] | [test_sam_policy.py](../../tests/test_sam_policy.py) | 764 | historical model, masks, artifact |
| [ ] | [test_sam_policy_training.py](../../tests/test_sam_policy_training.py) | 402 | historical trainer/resume/export |
| [ ] | [test_standalone_policy.py](../../tests/test_standalone_policy.py) | 274 | historical standalone inference |
| [ ] | [test_sam_miner.py](../../tests/test_sam_miner.py) | 1,469 | branch counts, shards, privacy, resume, parallelism |
| [ ] | [test_sam_teacher.py](../../tests/test_sam_teacher.py) | 522 | Sam-128 grouped search |
| [ ] | [test_search_planner.py](../../tests/test_search_planner.py) | 287 | frozen v1 golden search |
| [ ] | [test_strategic_search.py](../../tests/test_strategic_search.py) | 546 | shallow Teacher v2 semantics |
| [ ] | [test_strategic_search_symmetry.py](../../tests/test_strategic_search_symmetry.py) | 431 | pooling, fair coin, invariance |
| [ ] | [test_guided_search.py](../../tests/test_guided_search.py) | 278 | historical guided search |
| [ ] | [test_guided_policy.py](../../tests/test_guided_policy.py) | 105 | guided adapter |
| [ ] | [test_hybrid_response_search.py](../../tests/test_hybrid_response_search.py) | 370 | response hybrid modes/privacy |
| [ ] | [test_response_distillation.py](../../tests/test_response_distillation.py) | 281 | ranking examples/loss |
| [ ] | [test_response_dataset.py](../../tests/test_response_dataset.py) | 401 | response shards/manifests/resume |
| [ ] | [test_response_ranker.py](../../tests/test_response_ranker.py) | 345 | ranker trainer/artifact |
| [ ] | [test_search_signal_measurement.py](../../tests/test_search_signal_measurement.py) | 225 | symmetry signal harness |
| [ ] | [test_search_validation.py](../../tests/test_search_validation.py) | 191 | v1 validation harness |
| [ ] | [test_strategic_validation.py](../../tests/test_strategic_validation.py) | 140 | strategic acceptance fixtures |
| [ ] | [test_models.py](../../tests/test_models.py) | 421 | recurrent policy/critic compatibility |
| [ ] | [test_local_policy.py](../../tests/test_local_policy.py) | 424 | archived policy serving/recovery |
| [ ] | [test_policy_value.py](../../tests/test_policy_value.py) | 448 | historical feed-forward policy/value |
| [ ] | [test_collection.py](../../tests/test_collection.py) | 439 | historical rollouts/shards |
| [ ] | [test_optimization.py](../../tests/test_optimization.py) | 435 | PPO math/replay/artifacts |
| [ ] | [test_training.py](../../tests/test_training.py) | 449 | PPO orchestration/recovery |
| [ ] | [test_supervised.py](../../tests/test_supervised.py) | 369 | policy/value supervised trainer |
| [ ] | [test_teacher.py](../../tests/test_teacher.py) | 578 | teacher collection/privacy/resume |
| [ ] | [test_expert.py](../../tests/test_expert.py) | 308 | expert iteration/replay/recovery |
| [ ] | [test_expert_evaluation.py](../../tests/test_expert_evaluation.py) | 40 | expert evaluation adapter |
| [ ] | [test_tournament.py](../../tests/test_tournament.py) | 190 | archived candidate tournament |

### Frontend test modules

These Vitest/Playwright modules consume typed public fixtures and assert DOM,
state, timing, transport, accessibility, and layout behavior. They must never
depend on private engine or model data.

| Review | Test module | Lines | Primary invariants |
| --- | --- | ---: | --- |
| [ ] | [contracts.test.ts](../../frontend/src/contracts.test.ts) | 37 | compatibility validator exports |
| [ ] | [statelessApi.test.ts](../../frontend/src/statelessApi.test.ts) | 95 | stateless routes/origin/error parsing |
| [ ] | [statelessGameStore.test.ts](../../frontend/src/statelessGameStore.test.ts) | 454 | transactions, recovery, delay, narration |
| [ ] | [api.test.ts](../../frontend/src/api.test.ts) | 88 | retained stateful client |
| [ ] | [gameStore.test.ts](../../frontend/src/gameStore.test.ts) | 157 | retained stateful controller |
| [ ] | [scoringStateMachine.test.ts](../../frontend/src/scoringStateMachine.test.ts) | 93 | scoring frames/timing |
| [ ] | [ScoringPresentation.test.tsx](../../frontend/src/ScoringPresentation.test.tsx) | 301 | scoring layout/callbacks/narration reveal |
| [ ] | [DraculaCommentary.test.tsx](../../frontend/src/DraculaCommentary.test.tsx) | 254 | face states/typing/stability |
| [ ] | [cardAssets.test.ts](../../frontend/src/cardAssets.test.ts) | 64 | complete asset mapping |
| [ ] | [siteConfig.test.ts](../../frontend/src/siteConfig.test.ts) | 34 | base-aware routes |
| [ ] | [gameplay.test.tsx](../../frontend/src/gameplay.test.tsx) | 243 | interaction/presentation components |
| [ ] | [usability.test.tsx](../../frontend/src/usability.test.tsx) | 165 | accessibility/responsive CSS contract |
| [ ] | [App.test.ts](../../frontend/src/App.test.ts) | 12 | hash route resolution |
| [ ] | [production-stateless.spec.ts](../../frontend/e2e/production-stateless.spec.ts) | 209 | six-round reload/narration/mobile/desktop browser flow |

### Configuration and document provenance

- [ ] [configs/README.md](../../configs/README.md)
  and the four root TOML files: **25 + 78 lines.** Verify `pi1` provenance,
  retained `pi0`/D1/Sam settings, installed CLI mapping, and explicit history.
- [ ] [docs/README.md](../../docs/README.md),
  [product-and-scope.md](../../docs/product-and-scope.md),
  [decisions.md](../../docs/decisions.md), and
  [evaluation-and-operations.md](../../docs/evaluation-and-operations.md)
  — **29, 52, 70, and 86 lines.** Verify active index, locked scope,
  architectural decisions, and release evidence requirements.
- [ ] [bgc-dataset-miner.md](../../docs/bgc-dataset-miner.md)
  — **147 lines** and [sam-dataset-miner.md](../../docs/sam-dataset-miner.md)
  — **376 lines.** Verify selected `pi1` corpus lineage versus historical Sam
  miner; neither is a deployment path.
- [ ] [design-tracker.md](../../docs/design-tracker.md)
  — **66 lines.** Sole tracker-ID location and actual release state.
- [ ] [reports README](../../reports/README.md) —
  **117 lines.** Active evidence/history/maintenance index; reports do not
  reactivate historical implementations.

### Final review and rollback record

- [ ] Read [refactoring-pass-002.md](../../reports/maintenance/refactoring-pass-002.md)
  against the actual diff and validation output.
- [ ] Confirm every historical reader above has an explicit retain/archive/
  remove-later disposition; do not infer removal merely from nonproduction use.
- [ ] Confirm all 688 Python, 99 frontend, and 2 browser tests cover the reviewed
  boundaries and identify any missing test before approval.
- [ ] Recheck package imports, wheel, links, tracker scope, secrets, ignored/
  untracked files, and diff format after review corrections.
- [ ] Build one clean reviewed commit, reseal the release candidate, deploy
  hosted staging, review measurements, and only then request go-live approval.
- [ ] Comments and docstrings meet the stated rule in every maintained module.

---

## Review completion record

- [ ] Tranche 1 approved: rules and cards.
- [ ] Tranche 2 approved: engine types and scoring.
- [ ] Tranche 3 approved: validation, serialization, transitions, lifecycle.
- [ ] Tranche 4 approved: deterministic replay and privacy.
- [ ] Tranche 5 approved: symmetry, `pi1`, and move selection.
- [ ] Tranche 6 approved: stateless API and narration.
- [ ] Tranche 7 approved: frontend state and presentation.
- [ ] Tranche 8 approved: deployment and operations.
- [ ] Tranche 9 approved: tests, historical readers, and rollback.
- [ ] Both remaining user decisions are resolved at their required time.
- [ ] Review corrections pass the complete validation surface.
- [ ] Final commit and release candidate are created only after approval.

The next action is to restart the line-by-line review at **Tranche 1 — Rules
and cards**.

---

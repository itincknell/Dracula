# Final-sprint restrained refactoring

Date: 2026-08-31

## Result

The active stateless deployment path is less coupled while preserving game,
policy, narration, recovery, and presentation behavior. The selected `pi1`
weights and digest are unchanged. No deployment, commit, push, database change,
game-rule change, or deferred user decision occurred.

## Modules changed

- `src/dracula/active_policy.py` is the single active standalone-policy runtime
  and gameplay adapter. Production no longer imports the historical `pi0`
  evaluation module to load or run `pi1`.
- `src/dracula/api/production_config.py` owns the active production environment:
  `DRACULA_POLICY_ARTIFACT`, `DRACULA_NARRATION_ENABLED`, and
  `DRACULA_REPLAY_CACHE_ENTRIES`.
- `src/dracula/api/production.py` constructs the stateless application directly
  from that configuration. Local comparison-controller selection is not part of
  the Lambda entry point.
- `src/dracula/api/stateless_projection.py` is the public projection boundary;
  private replay state no longer shares projection implementation with the
  replay service.
- `src/dracula/api/stateless_routes.py` owns the small FastAPI stateless route
  handlers and delegates every operation to gameplay or narration services.
- `src/dracula/api/stateless_service.py` now contains replay and command
  application rather than browser projection code.
- `src/dracula/search_policy.py` retains its historical
  `InlineBGCPolicyExecutor` name as a compatibility reader, but delegates to the
  active policy adapter instead of duplicating inference.
- `frontend/src/statelessRecovery.ts` owns browser recovery-envelope storage,
  validation, cloning, and clearing.
- `frontend/src/statelessNarration.ts` owns opening, round-transition, and final
  narration request timing, deferred reveal, cancellation, and stale-request
  suppression.
- `frontend/src/statelessGameStore.ts` remains the gameplay-state coordinator
  and delegates those two concerns.
- The Makefile, Playwright configuration, container, and CloudFormation template
  use the neutral active artifact setting. Historical `pi0` settings remain only
  on explicitly selected comparison paths.

## Duplication and naming removed

- One active artifact loader and inference implementation replaces the runtime
  dependency on `StandalonePi0Opponent`.
- One canonical representative-action projection remains in `sam_policy.py`;
  the active runtime calls it rather than introducing another mask.
- One canonical paired-destination seed and concrete-action selector remains in
  strategic search; the active runtime calls it unchanged.
- Stateless transport registration moved out of the general local application
  factory, removing four large closure handlers from `create_app`.
- Stateless public view construction moved out of deterministic replay.
- Browser storage parsing and narration generation bookkeeping moved out of the
  gameplay controller.
- Active production configuration no longer calls the selected model a `pi0`
  artifact. The old deterministic seed namespace is intentionally retained
  because changing it would change paired-destination choices.

## Behavior-preservation evidence

- A focused regression compares the new active runtime against the historical
  runtime for the same artifact, information state, game identity, and turn.
  Representative and concrete actions match exactly.
- Focused API, policy, stateless, narration, packaging, and infrastructure
  tests: 60 passed before the complete run.
- Complete Python suite:
  `PYTHONHASHSEED=0 .venv/bin/pytest -q` — 683 passed in 633.10 seconds.
- Frontend:
  `npm --prefix frontend run check` — 12 files and 84 tests passed; TypeScript,
  ESLint, and `/Dracula/` production build verification passed.
- Browser:
  `make pages-test` — two Playwright tests passed, including a complete
  six-round stateless game, recovery, narration timing, failure isolation, and
  mobile/desktop coverage.
- Package/import:
  `compileall`, `pip check`, and an isolated `pip wheel` build passed.
- Infrastructure:
  `.venv/bin/cfn-lint infrastructure/ecr.yaml infrastructure/application.yaml`
  passed and representative Make targets parsed.
- Container:
  `make lambda-context`, `make lambda-build`, and `make lambda-validate` passed
  with artifact SHA-256
  `70c76f2eb64600eab2297640278a6c94d4336ab8e73bf941a6d96237f69f5b5c`.
  The arm64 image completed four games and 124 requests, selected 448 legal
  policy actions, exercised warm and cold replay, and reported no privacy or
  write-boundary failure. Image size was 996,560,893 bytes; peak measured
  memory was 203,738,317 bytes.
- Markdown links passed across 70 files and `git diff --check` passed.

## Complexity retained intentionally

- The general local application still resolves explicit historical and search
  comparison modes. Removing them would break archived evidence and local
  comparisons; production bypasses that resolver.
- Historical classes and artifact readers retain their old names and schemas so
  historical reports remain reproducible. They are not production fallbacks.
- Replay validation and deterministic command application remain together in
  one service because they enforce one ordered state transition contract.
- Narration prompt construction and the Bedrock adapter were already separated
  behind one protocol and one request builder; no refactor was warranted.
- Presentation components and engine rules were untouched.

No active regression or mechanical defect remains from this refactoring pass.

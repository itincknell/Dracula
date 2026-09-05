# Behavior-preserving refactoring audit 002

Date: 2026-09-02

## Scope and baseline

This audit covers the current working tree at Git commit
`81b2504d4cc679e40e36a1fd63160037e6dfbb0d`. It inspected 71 Python source
modules (51,643 lines), 47 Python test modules (16,741 lines), 34 frontend
source files (6,539 lines), the production browser test, active documentation,
configuration, packaging, infrastructure, workflows, and build tools.

The working tree already contains substantial intentional final-sprint and UI
work: 40 tracked paths differ from `HEAD`, with 1,307 insertions and 786
deletions, and the stateless API, deployment, narration, frontend, tests, and
reports include eligible untracked files. This audit did not edit source or
reinterpret those changes.

Protected local databases, model artifacts, datasets, archives, source assets,
generated builds, logs, and other ignored runtime files were left untouched.
The root Grok image and video are treated as user assets, not cleanup targets.

## Code classification

| Class | Current ownership | Treatment in this pass |
| --- | --- | --- |
| Active production code | `cards.py`, `engine.py`, `randomness.py`, the actor-information and symmetry boundaries, compact BGC model/artifact loading, `active_policy.py`, stateless API/projection/replay/narration, production logging/configuration, the stateless frontend, Lambda container, CloudFormation, Pages workflow, and release tools | Refactor where the dependency or responsibility problem below is concrete; preserve every public result and selected `pi1` action |
| Maintained training, evaluation, or operational code | BGC policy training/evaluation/migration, accepted-`pi0` and D1 readers, belief-greedy collection/search, explicit local comparison controllers, local SQLite gameplay, root configurations, package entry points, and current-lineage tools | Keep runnable and tested; remove mechanical debris, but do not redesign completed algorithms or artifact schemas |
| Compatibility code required for existing artifacts | PPO/recurrent training modules, policy/value and guided search, Teacher v2 and response-ranker modules, nested Sam and Sam-miner readers, old standalone-policy readers, `dracula.cli`, stateful browser API/store, and historical fixture/report generators | Preserve import and artifact-reading behavior; do not perform stylistic decomposition merely because files are large |
| Historical evidence | `reports/history/`, `configs/archive/`, `tools/archive/`, `.local/archive/`, sealed ignored runs, and prior output logs | No source-style refactor; retain exact evidence and digests |
| Mechanically dead code | Seven unused imports were found: `Counter` and `seed_hex` in `bgc_policy_evaluation.py`; `BGCCommittedCorpusSnapshot` in `bgc_policy_migration.py`; three unused schema/size imports in `bgc_policy_training.py`; and `build_representative_action_projection` in `sam_policy_training.py` | Remove and let import/package tests prove the change. No complete module or public symbol is proven dead. `dracula.cli` is unregistered but retained intentionally for archived PPO runs |

## Concrete findings

### 1. The engine contains separable domain responsibilities

`src/dracula/engine.py` is 1,047 lines. Its size is a consequence of five
separate responsibilities rather than one unusually long cohesive algorithm:

- domain enums, errors, and immutable value objects (lines 40–235);
- deterministic dealing, creation, and move legality (238–374);
- line/coffin scoring, transitions, and outcomes (377–676);
- structural, history, deal, and card-conservation validation (679–951);
- canonical serialization and fingerprints (954–1,047).

`validate_state` alone spans 173 lines and enforces several independent
invariant families. That is the strongest refactoring candidate. The stable
`dracula.engine` import surface must remain as a facade so callers and sealed
artifact readers do not change.

A restrained split is `engine_types.py`, `scoring.py`,
`engine_validation.py`, and `engine_serialization.py`, with deal/legality and
lifecycle orchestration remaining in `engine.py`. Each extracted module owns a
substantial coherent concern; this avoids arbitrary one-function modules.

### 2. Package initializers broaden the runtime dependency graph

`src/dracula/search/__init__.py` is a 259-line eager re-export barrel. Importing
one search submodule executes imports for the planner, guided search, shallow
Teacher v2, response student, nested Sam, and belief-greedy search. A production
`active_policy` import currently loads 35 Dracula modules, including historical
`policy_value`, response-distillation, guided, Sam-teacher, and belief-greedy
modules that do not participate in standalone `pi1` inference.

`src/dracula/api/__init__.py` eagerly imports and constructs the general app
when any API submodule is imported. No repository source imports `app` or
`create_app` from this package facade; callers already use
`dracula.api.app` explicitly.

Make both package initializers side-effect free. Change internal callers to
import their owning modules directly. Retain only deliberate, tested
compatibility re-exports. This removes static dependency cycles and unnecessary
Lambda imports without altering search or API behavior.

### 3. Active action handling lives inside historical implementations

Production `pi1` uses strategic-group construction and paired-destination
resolution from the 1,802-line shallow-search module, plus representative-mask
projection from the 1,069-line historical Sam-policy module. The implementations
are not duplicated, but their ownership makes the production path depend on
large historical systems.

Extract the existing types and functions—without rewriting their algorithms—
into one small action-contract module covering strategic groups,
representative proxies, and concrete paired-destination resolution. Historical
modules re-export those names for compatibility. The exact seed namespace,
canonical tie ordering, symmetry table, proxy mask, and fair-coin outcome must
remain byte-for-byte stable.

The literal seed namespace `dracula-pi0-standalone-request-v1` in
`active_policy.py` is deliberately frozen because changing it changes concrete
paired destinations. Rename the Python constant to identify it as a retained
legacy namespace and document the compatibility reason; do not change the
literal.

### 4. Search information mixes projection with determinization

`search/information.py` is 906 lines and owns public-history validation,
actor-local projection, canonical fingerprints, seed derivation, and hidden
world sampling. The first four concerns define the production information
boundary; determinization is used by retained search controllers.

Move determinization construction and its private reconstruction helpers into
`search/determinization.py`. Keep the actor-visible data types, projection,
validation, canonical representation, and fingerprint in `information.py`.
Preserve compatibility imports and all seed derivations. No information field
or sampling rule changes.

### 5. The application factory mixes configuration, controller selection, and two APIs

`api/app.py:create_app` spans 373 lines. It parses environment variables,
resolves eleven production/historical opponent modes, creates either the
stateless or stateful service, and declares all stateful routes. Production
imports this general local factory even though it needs only stateless `pi1`.

Separate:

- the policy request/response/protocol types currently housed in the stateful
  `api/service.py`;
- explicit local comparison-controller resolution;
- stateful route registration;
- stateless production app construction.

Keep `api/app.py` as the compatible local ASGI entry point and factory. Make
`api/production.py` depend only on the stateless assembly path and active
policy. Do not remove comparison modes or introduce a fallback.

`api/session.py` also combines local persistent-session serialization and
idempotency with scoring/phase projection reused by stateless production.
Extract the shared presentation helpers into a neutral projection module;
leave local persistence in `session.py`.

### 6. A few other boundaries merit small, targeted changes

- `bridge.py` combines actor-visible encoding/action mapping with recurrent
  hidden-state transition compatibility. Extract the legacy transition/hidden
  state portion while retaining `bridge.py` re-exports. The active 875-to-659
  `pi1` conversion and stable engine-slot semantics do not change.
- `api/narration.py` contains grounded cue construction, Bedrock wire parsing,
  the provider adapter, and request orchestration. Split the Bedrock-specific
  adapter from pure cue construction/service behavior. Keep the existing
  adapter protocol and the three-cue cadence exactly.
- The current selected artifact retains the historical filename and metadata
  status `unaccepted-candidate`. The release build pins its SHA-256 and the
  user selected it. Preserve the artifact and loader check; add concise
  compatibility documentation rather than rewriting artifact metadata.
- No silent production fallback was found. Missing production policy or
  narration dependencies surface as unconfigured/unavailable behavior, and
  explicit comparison modes remain explicit.

### 7. Frontend responsibilities can be separated without redesigning the UI

- `frontend/src/contracts.ts` is 796 lines and contains both stateful-v1 and
  stateless types plus two full validator families. Split common presentation
  types, stateful compatibility contracts, and stateless production contracts;
  retain a compatibility re-export if needed by existing tests.
- `statelessGameStore.ts` is 472 lines and combines wire-to-view projection,
  optimistic placement previews, and controller orchestration. Extract the
  pure projection/preview functions; keep lifecycle and networking in the
  controller. Recovery and narration are already appropriately separated.
- `styles.css` is 786 lines and contains base theme, game surface, scoring,
  rules, animation, container-responsive, mobile, and reduced-motion layers.
  Split it by those existing concerns with one explicit import order. Do not
  rename classes, change values, revisit the approved design, or add a styling
  framework.
- `ScoringPresentation.tsx`, `scoringStateMachine.ts`, `App.tsx`, and
  `gameplay.tsx` are cohesive at their current size. Keep them intact unless an
  extraction above leaves an obviously self-contained helper.
- The repeated local runtime environment list in the `dev`, `preview`, and
  `dev-api` Make targets is concrete duplication. Factor the shared assignment
  list once while keeping target names, defaults, quoting, and behavior.

### 8. Comment and docstring quality is mostly restrained

Every Python module has a module docstring. Existing inline comments in the
active engine, cards, randomness, API, and frontend explain invariants or
environment limitations; no `TODO`, `FIXME`, tutorial narration, or
generated-sounding commentary was found in active source. Low comment counts
are not themselves defects.

Apply the review standard only to changed modules:

> Comments explain non-obvious intent, invariants, constraints, or reasons.
> They remain concise and professional. They do not narrate straightforward
> code.

Add concise docstrings at public boundaries whose contracts are not evident
from their signatures: engine validation/serialization, coordinate and action
mapping, information-state projection/fingerprinting, active-policy decision,
stateless replay/command application, and narration failure isolation. Do not
add docstrings to obvious enums, passive data holders, trivial route closures,
or simple React render helpers. Recheck existing comments after moves so none
describe obsolete import behavior.

### 9. Supporting documentation has known drift

- `configs/README.md` says historical Python commands are not installed, while
  `pyproject.toml` currently defines six maintained collector/training/evaluation
  entry points.
- Root `tools/` contains current-lineage migration/evaluation scripts that are
  absent from `tools/README.md`; they are reproducibility tools, not dead code.
- Active deployment documents and the decision packet still call the AWS
  region, Bedrock model, narration voice, `/Dracula/` capitalization,
  Cloudflare proxy mode, and disclosure policy unresolved. The user has since
  selected the personal AWS account, `us-east-1`, Amazon Nova Lite
  (`amazon.nova-lite-v1:0`), the arcade/villain narration direction,
  `/Dracula/`, proxied Cloudflare DNS, and no public cheatability disclosure.
- Ignored generated Lambda/frontend build artifacts currently exist under
  `build/`; they are not Git candidates and remain untouched by this audit.

## Ordered implementation plan

### 1. Python domain and runtime changes

1. Capture pre-refactor golden outputs for engine state JSON/fingerprints,
   scoring, information-state fingerprints, symmetry groups, representative
   masks, `pi1` representative/concrete actions, stateless response JSON, and
   local stateful routes.
2. Extract engine types, scoring, validation, and canonical serialization while
   keeping `dracula.engine` as the stable facade. Refactor `validate_state`
   internally by invariant family without weakening any check.
3. Extract the shared strategic-action/proxy/fair-coin contract. Update active
   policy to import only this boundary; retain historical re-exports and the
   frozen seed literal.
4. Split search determinization from actor-local information projection and
   make `dracula.search` initialization side-effect free. Replace broad package
   imports with direct owner-module imports.
5. Extract policy execution contracts and shared presentation helpers from the
   stateful API. Separate local opponent selection and stateful route wiring
   from stateless production assembly. Make `dracula.api` initialization
   side-effect free.
6. Separate Bedrock transport from grounded narration behavior and isolate
   legacy bridge transition state from active encoding.
7. Remove the seven proven unused imports. Do not restructure completed PPO,
   Teacher v2, Sam-miner, response-ranker, or sealed-corpus implementations.
8. Apply the comment/docstring standard only to touched code, then run an
   import-direction and public-symbol compatibility audit.

### 2. Frontend and supporting-file changes

1. Split shared, stateful-compatibility, and stateless-production TypeScript
   contracts and validators while preserving all runtime validation.
2. Extract stateless response projection and optimistic previews from the
   controller. Preserve the recovery envelope, narration scheduling, minimum
   Dracula delay, scoring animation, and synchronous opponent settlement.
3. Split CSS into ordered base, gameplay, scoring/animation, rules, and
   responsive layers using the exact approved declarations and selectors.
4. Factor repeated Makefile runtime environment assignments. Keep standalone
   `pi1` as the default and every comparison mode explicit.
5. Preserve all portrait/card/favicon assets and verify their generated asset
   map is identical.

### 3. Documentation and final validation

1. Update module ownership in architecture, API, deployment, and review docs.
2. Record the already-made AWS, Bedrock, narration, path, proxy, and disclosure
   decisions without changing remaining resource-sizing or go-live decisions.
3. Correct the config/tool indexes and regenerate the bottom-up code-review
   agenda from the refactored tree.
4. Run terminology, internal-link, tracker-ID, ignored/untracked, secret-path,
   and `git diff --check` validation. Do not alter protected ignored artifacts.

### 4. Exact behavior-preservation tests

- Engine: full engine/foundation suites plus exact before/after canonical state,
  fingerprint, deal, legality, scoring, lifecycle, and exception fixtures.
- Information/action boundary: bridge, actor-information, privacy, symmetry,
  representative-mask, fair-coin, search, and selected-artifact action fixtures;
  compare exact seeds, groups, masks, and actions.
- API: local stateful tests and complete stateless six-round games in both roles
  and dealer assignments; compare exact status codes and public JSON on cold
  replay, cache hit, duplicate, stale, malformed, and reload requests.
- Narration: exact cue eligibility/grounding, timeout/failure isolation, and
  private-field exclusion with fake and disabled adapters.
- Frontend: all unit/component tests, TypeScript, ESLint, production build,
  stateless Playwright games, reload stages, mobile/desktop layout contracts,
  narration timing, and reduced-motion behavior.
- Packaging: import every maintained module, build the wheel, validate CLI
  help, run `cfn-lint`, rebuild the Lambda context/image, verify the fixed
  `pi1` digest, and rerun the read-only container validation.
- Final: full Python suite, exact artifact hashes, Markdown links, tracker-ID
  confinement, production URL scan, and `git diff --check`.

## Audit conclusion

The repository warrants a substantial but bounded refactor. The engine split,
runtime dependency cleanup, active action-contract extraction, API assembly
separation, and focused frontend splits directly improve reviewability and
ownership. Refactoring every large historical miner, trainer, or report tool
would be stylistic churn and is explicitly excluded. No behavior change,
artifact migration, UI redesign, or new framework is part of the plan.

## Stage 2 implementation: Python

The Python refactoring is complete. It preserves the existing public behavior
while narrowing the ownership and import boundaries identified above.

### Engine ownership

`dracula.engine` remains the gameplay facade and direct home of move legality,
transitions, and lifecycle orchestration. Its 1,047 lines were reduced to 299
by moving existing cohesive responsibilities without changing their
algorithms:

- `engine_types.py` owns immutable domain values, enums, constants, and engine
  exceptions.
- `engine_dealing.py` owns deterministic shuffling, dealing, starting hands,
  and starting coffins.
- `scoring.py` owns line, coffin, round, and game scoring.
- `engine_validation.py` owns structural, history, deal, and card-conservation
  invariants.
- `engine_serialization.py` owns canonical private serialization and state
  fingerprints.

The facade directly re-exports the established engine API. An exact golden
fixture spanning deals, legal-move order, transitions, round advancement,
scores, canonical serialization, and fingerprints remained byte-identical:
`21bd36f0afffa96b16ccc1c87177b0f69801b9a5c1812c96a09a37559189743c`
before and after the split.

### Information and action ownership

- `action_contract.py` now owns the representative-action mask, proxy mapping,
  output selection, and concrete group resolution used by standalone `pi1`.
- `strategic_actions.py` owns strategic action groups and the existing seeded
  paired-destination choice. Historical search and model modules re-export the
  moved names for compatibility.
- `search/determinization.py` now owns hidden-world reconstruction and sampling;
  `search/information.py` retains the actor-visible projection, validation,
  canonical representation, and fingerprint boundary.
- `dracula.search` now uses lazy compatibility exports. Importing the package
  alone no longer loads every historical planner and model.
- `active_policy.py` imports only the selected model, information, action, and
  symmetry boundaries. The legacy request-seed literal is unchanged because it
  determines concrete paired destinations.

All 14 recorded `pi1` representative-group and concrete-action sequences are
identical before and after the extraction.

### API, narration, and bridge ownership

- `api/policy.py` owns policy execution contracts and hidden-state sizing.
- `api/presentation.py` owns shared phase and scoring projection.
- `api/local_controllers.py` owns explicit local comparison-controller
  selection, with no silent fallback.
- `api/stateless_app.py` provides the small production stateless assembly path;
  `api/app.py` remains the compatible local/stateful entry point.
- `api/bedrock.py` owns Bedrock wire parsing and transport while
  `api/narration.py` retains deterministic public cue construction and failure
  isolation.
- `bridge_legacy.py` owns retained recurrent hidden-state transition support;
  `bridge.py` remains the active observation and action bridge with lazy legacy
  exports.
- `dracula.api` is side-effect free. Importing it no longer constructs an app.

The seven mechanically unused imports listed in the audit were removed. Clear
historical planners, miners, trainers, and artifact readers were otherwise left
intact.

### Comments and docstrings

Touched modules were reviewed against the locked standard:

> Comments explain non-obvious intent, invariants, constraints, or reasons.
> They remain concise and professional. They do not narrate straightforward
> code.

Comments now cover only compatibility-sensitive seed literals, deterministic
dealing, scoring tie handling, sealed-deal validation, card conservation, lazy
compatibility imports, and test-adapter safety. Redundant narration was not
added. Public docstrings are limited to boundaries whose ownership or
compatibility behavior is not evident from their names and signatures.

### Focused verification

The required focused verification completed successfully:

- Foundation, engine, bridge, information, symmetry, search-policy, Sam-policy,
  and standalone-policy group: **187 passed**.
- Search planner, guided search, strategic search, Sam teacher, symmetry, and
  retained validation group: **107 passed**.
- Maintained BGC policy model/training/evaluation, `pi0` miner,
  belief-greedy miner, Sam miner, and Sam-policy training group:
  **111 passed**.
- Stateful API, stateless API, Bedrock narration, and local preview group:
  **46 passed**.
- Bridge, collection, and retained training compatibility group:
  **45 passed**.
- All **85** Python package modules import successfully from a fresh process.
- `pip check`, bytecode compilation, and `git diff --check` pass.
- Importing `dracula.search` loads only that package; importing `dracula.api`
  does not load the app factory.

One checkpoint identity assertion failed during an earlier long-running test
while the source tree was still being edited. The isolated assertion and the
entire 111-test maintained-data group passed after the tree became stable; no
checkpoint validation was weakened.

The final full repository suite was intentionally not run in this stage. The
queued frontend/supporting-file refactor runs next, followed by the final
combined validation stage.

## Stage 3 implementation: frontend and supporting files

The frontend refactoring is complete. It changes module ownership and test
synchronization without changing the approved interface, timing, assets, or
production configuration.

### Contract and controller ownership

The former 796-line `contracts.ts` remains as the compatibility facade but now
delegates to three focused owners:

- `contractPrimitives.ts` contains presentation types and their shared strict
  validators.
- `statefulContracts.ts` contains the retained stateful API types and
  validators.
- `statelessContracts.ts` contains the production recovery, gameplay,
  narration, health, and error contracts and validators.

Maintained source imports now target the owning module directly. Existing
tests and compatibility callers can continue to import the same public names
from `contracts.ts`; internal validation helpers are not added to that public
facade.

Pure wire-to-view projection, immediate human-placement preview, and delayed
opponent-opening preview moved from `statelessGameStore.ts` into
`statelessProjection.ts`. The controller retains network sequencing,
recovery persistence, publication, narration coordination, and lifecycle
state. Its size decreased from 472 to 370 lines. Existing tests already cover
the immediate human placement, opening and later-round Dracula delays,
rejected-command rollback, reload recovery, narration timing, and synchronous
opponent settlement.

The API transport, recovery store, narration coordinator, Dracula portrait,
scoring state machine, scoring presentation, application, and gameplay
components were reviewed and left intact because each already has one clear
responsibility. No framework or generic transport abstraction was added.

### Styles and supporting configuration

The 786-line stylesheet had five established declaration regions. It now has
one five-line import surface with ordered `base`, `gameplay`, `scoring`,
`rules`, and `responsive` files. Declaration text and order are unchanged.
The concatenated nonblank stylesheet digest is identical before and after:
`076b1f4391ae831371b60523a8d6168d6c8d8f1a028d99498438ff05ec5d7b13`.
The compiled production CSS remains exactly `index-B9yj0M26.css` at 27.20 kB.
Card, Vampire, portrait, and favicon file digests are unchanged.

The repeated local gameplay environment assignments in the `dev`, `preview`,
and `dev-api` Make recipes are now one `LOCAL_GAMEPLAY_ENV` definition. Target
names, values, quoting, explicit comparison modes, and command expansion are
unchanged. Dry runs cover those targets plus Pages and Lambda builds.

Vite configuration, asset preparation, production-output verification,
Playwright configuration, dependency metadata, the Pages workflow, Lambda
container files, CloudFormation templates, and environment parameter files
were reviewed. No additional responsibility or duplication justified a
change.

### Browser-test synchronization

The production browser test had two stale synchronization assumptions: it
could inspect or reload before the locked minimum Dracula delay had committed
the response to browser recovery storage, and a text query matched both the
accessible and animated copies inside `RetroDialogue`. The test now waits for
the exact recovery envelope and enabled hand before continuing, and it asserts
dialogue through the single dialogue container. Runtime timing and markup are
unchanged.

### Focused verification

- Frontend unit and component tests: **99 passed** across 13 files.
- TypeScript and ESLint: passed.
- Production Vite build and output verification: passed with `/Dracula/` and
  `https://api.ian-tincknell.com`; no local URL appears in production output.
- Production Playwright suite: **2 passed**, including a complete six-round
  stateless game with reloads and narration timing, plus mobile/desktop
  geometry, narration failure, and no-autoscroll coverage.
- A live check at 1440x900 and 390x844 found no horizontal overflow and retained
  the approved mobile element order and 90x90 portrait geometry.
- Affected API, stateless gameplay, local preview, and Lambda packaging tests:
  **29 passed**.
- Both CloudFormation templates parse, both parameter files parse with their
  expected 18 entries, and the infrastructure contract tests pass.
- Python bytecode compilation, package imports, `pip check`, Make dry runs, and
  `git diff --check` pass.

The local dummy-dialogue preview was rebuilt and relaunched at
`http://127.0.0.1:4173/Dracula/`. No deployment, training, mining, commit, or
push occurred.

## Final closeout

### Before and after

The refactor preserved stable facades while assigning cohesive ownership:

| Former concentration | Final structure |
| --- | --- |
| `engine.py`, 1,047 lines | `engine.py`, 299-line legality/transition/lifecycle facade; `engine_types.py` 222; `engine_dealing.py` 79; `scoring.py` 174; `engine_validation.py` 350; `engine_serialization.py` 116 |
| `search/information.py`, 906 lines | 727-line actor-visible information contract plus 219-line private determinization owner |
| Active action behavior inside historical search/model files | `strategic_actions.py`, 223 lines, owns groups and paired destinations; `action_contract.py`, 293 lines, owns representative projection, masking, and output selection |
| `bridge.py`, 631 lines | 451-line active encoding/action bridge plus 259-line recurrent compatibility module |
| `api/app.py`, 523 lines | 241-line compatible local factory; production assembly, stateless routes, local controller selection, policy contracts, and presentation have separate owners |
| `api/narration.py`, 553 lines | 457-line grounded cue/service owner plus 174-line Bedrock transport owner |
| `contracts.ts`, 796 lines | 46-line compatibility facade over 281-line primitives, 281-line stateful contracts, and 277-line stateless contracts |
| `statelessGameStore.ts`, 472 lines | 374-line controller plus 112-line pure stateless projection module |
| `styles.css`, 786 lines | Five-line ordered import surface over base, gameplay, scoring, rules, and responsive owners; declaration content and order remain identical |

The production dependency direction is now engine → actor-visible projection →
representative action contract → `pi1` runtime → stateless service → public
projection/routes. Bedrock and local repository-backed gameplay sit behind
separate explicit boundaries. The frontend similarly separates wire contracts,
transport, persistence, projection, narration timing, controller state, and
presentation.

### Comments, docstrings, and dead code

All touched modules were reviewed against the required rule:

> Comments explain non-obvious intent, invariants, constraints, or reasons.
> They remain concise and professional. They do not narrate straightforward
> code.

Concise comments and public docstrings now identify deterministic compatibility
seeds, card conservation, canonical serialization, private/public information
boundaries, replay idempotency, atomic artifact behavior, stale-response
protection, and animation coordination. Redundant or tutorial-style comments
were not added; no `TODO`, `FIXME`, `XXX`, or `HACK` marker remains in maintained
Python or frontend source.

Seven imports identified mechanically by the audit were removed. Repository
import searches, package entry-point inspection, all-module imports, and the
complete test suite did not prove any whole maintained module dead. Historical
artifact readers and explicit comparison controllers therefore remain intact.

### Final validation

- Complete Python suite: **688 passed** in 697.56 seconds.
- Static import pass: all **84** discovered `dracula` modules imported.
- Package build: `pip wheel . --no-deps` produced a 422 KiB wheel; `pip check`
  found no broken requirements. The optional `build` frontend is not installed
  in the project environment, so the standards-based pip wheel path was used.
- Frontend: **99 passed** across 13 files; TypeScript and ESLint passed.
- Production frontend: Vite build and verifier passed with `/Dracula/`,
  `https://api.ian-tincknell.com`, and no development origin in output.
- Browser: **2 passed**, including a complete six-round stateless game with
  reload/narration timing and the mobile/desktop narration-failure geometry
  case.
- Infrastructure: both CloudFormation documents and both 18-entry parameter
  files parse; the eight packaging/infrastructure tests pass. `cfn-lint` is not
  installed in this environment; no live AWS call was substituted for it.
- Commands: all six installed CLI `--help` surfaces, `make help`, and dry runs
  of every active gameplay, verification, Pages, Lambda, and release target
  pass.
- Documentation: 115 local Markdown links resolve; tracker IDs remain confined
  to `docs/design-tracker.md`.
- Privacy and determinism: the complete suite covers golden engine results,
  actor-local information, exact `pi1` actions, stateless cold/warm replay,
  public response schemas, narration grounding, and hidden-field rejection.
- Selected artifact SHA-256 remains
  `70c76f2eb64600eab2297640278a6c94d4336ab8e73bf941a6d96237f69f5b5c`.
- Secret-pattern and authored absolute-workstation-path scans pass. The only
  absolute examples found were third-party files under ignored
  `frontend/node_modules`.
- Git has no staged file, `git diff --check` passes, and generated build, run,
  database, log, and dependency trees remain ignored.

### Current diff scope and remaining concerns

The working tree intentionally combines the uncommitted final deployment
sprint, user-approved UI work, repository hygiene, and this refactor. Relative
to `81b2504d4cc679e40e36a1fd63160037e6dfbb0d`, 59 tracked paths differ
(1,421 insertions and 4,563 deletions before this report and agenda refresh),
with 69 untracked status entries representing authored source, tests,
infrastructure, reports, selected frontend assets, and two retained root Grok
source assets. No run directory, model binary, database, build context, test
result, or output log is staged.

The remaining concerns are operational rather than refactoring defects:

- Hosted AWS staging still requires refreshing the selected personal-account
  credentials; managed Lambda and Nova Lite measurements do not yet exist.
- Production resource/notification values and final go-live authorization
  remain user decisions after hosted staging.
- Large historical training and miner modules remain complex because sealed
  artifacts depend on their readers. They are isolated from the production
  request path and are retained deliberately.
- The tree needs user-led line review, then one clean commit and resealed
  release candidate. No deployment has occurred.

The refreshed
[bottom-up review agenda](../active/final-code-review-agenda.md) uses current
line counts and the final module owners. Its nine tranches cover all 194 Python,
frontend, test, tool, deployment, infrastructure, and workflow files in scope;
the coverage audit found no orphan. The final Markdown pass resolved 339 local
links with no missing target.

## Maintained frontend and deployment follow-up

The line-review follow-up removed the unused session-backed browser client,
controller, wire models, facade, and their tests. The maintained frontend now
has one external-response validator (`statelessContracts`), one presentation
view (`gameView`), one transport (`statelessApi`), and one gameplay controller
(`statelessGameStore`). Test components use an explicit test controller rather
than exercising an obsolete transport. The production projection now extends
the already validated public view only with UI move IDs; tests import recovery
and projection helpers from their owning modules rather than through the game
controller.

The scoring timeline no longer contains its unreachable session-narrator wait
branch. Narration remains a separate request coordinated by
`statelessNarration`; the production scoring sequence and its timings are
unchanged. No CSS, card asset, portrait asset, responsive rule, animation, or
approved layout value changed. The unused `VITE_API_BASE_URL` declaration and
the old contract-fixture TypeScript include were removed.

Deployment cleanup removed an unused generic tree copier, centralized the
selected artifact digest in the context builder, and merged two identical HTTP
request helpers in the container validator. An absolute Lambda-context output
also exposed and received a focused fix: the build completed correctly but its
success message had assumed the output was inside the repository. The CLI now
reports either repository-relative or absolute destinations and has a
regression test.

Validation after these changes:

- Complete Python suite: **181 passed**.
- Frontend: **85 passed** across 10 files; TypeScript, ESLint, and the production
  build passed.
- Browser: **2 passed**, including a complete six-round stateless game and the
  narration-failure responsive-layout case.
- Package: all **42** discovered submodules imported, `pip check` passed, and a
  416 KiB wheel built.
- Deployment: **8** packaging tests passed; a fresh 41-file Lambda context
  reproduced the selected artifact digest. Both CloudFormation files and both
  18-entry parameter files parsed locally. AWS `ValidateTemplate` could not run
  because the configured personal-account token is invalid; no resource was
  created or changed.
- Documentation: **180** local Markdown links resolve; tracker-ID confinement,
  active frontend terminology, and `git diff --check` pass.

The refreshed review agenda records the current commit, active module owners,
line counts, and the unchanged approval boundary. Generated build output,
runtime data, model binaries, caches, and local databases remain ignored and
unstaged.

## Look-ahead readability cleanup

The review-driven follow-up inspected the unapproved search, training, local
API, stateless API, narration, and frontend paths before the user reached them.
It made the following responsibility changes without altering contracts:

- BGC policy training now separates external row decoding, metric mathematics,
  mutable optimization, atomic checkpoint persistence, immutable training
  contracts, resolved configuration, command-line transport, and run
  coordination into focused `bgc_policy_*` modules. The former 346-line
  training procedure is now a 69-line coordinator over named phases; the
  public entry point remains compatible while argument parsing lives outside
  the training coordinator.
- The local SQLite service now delegates session/event construction, policy
  invocation, public projection and move tokens, and route declarations to
  `local_session_events`, `local_policy_turn`, `local_projection`, and
  `local_routes`. `api.service` retains transaction, idempotency, and claim
  ownership.
- Stateless command application now separates placement and round-advance
  validation. Grounded narration derives opening, transition, and final cues in
  separate helpers.
- Frontend scoring timer/controller behavior remains in
  `ScoringPresentation`; pure scoring workspaces moved to `ScoringWorkspaces`.
  The public card ledger moved from page assembly to `SeenCardsExpando`.
- BGC simulation and determinization were divided into explicit sampling,
  selection, continuation, backup, and result-construction phases. Comments now
  explain hidden-world ownership, shared belief samples, UCT backup, checkpoint
  identity, interruption boundaries, optimistic placement, and minimum opponent
  timing.

No cohesive search algorithm, external validation boundary, declarative route
table, or tensor architecture was split merely to reduce a line count. Exact
duplicate-function inspection found only protocol stubs and the intentionally
separate local/stateless HTTP response adapters.

Validation after the look-ahead cleanup:

- Complete Python suite: **222 passed**.
- Frontend: **85 passed**; TypeScript, ESLint, and production build passed.
- Static import pass: all **60** discovered package submodules imported.
- `pip check` and `git diff --check` passed.
- Selected policy, BGC, API, replay, privacy, narration, local repository, and
  complete-game tests are included in the passing suite.
- Browser validation: both Playwright scenarios passed, including a complete
  six-round stateless game and narration-failure layout coverage.

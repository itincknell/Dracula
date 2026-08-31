# Repository hygiene inventory

Audit date: July 25, 2026
Scope: read-only inspection of the Dracula repository and live Sam-32 corpus

## Summary

The repository contains a coherent active engine, web application, nested
Sam-32 opponent, continuous corpus miner, and policy-imitation contract. Those
active paths are mixed with four generations of historical learning work:
recurrent PPO, version 1 search validation, shallow Teacher v2, response
ranking, and expert iteration.

The repository uses approximately:

| Area | Size | Assessment |
| --- | ---: | --- |
| `runs/` | 6.0 GiB | Mostly historical immutable evidence; active Sam-32 corpus is 1.2 GiB |
| `.venv/` | 720 MiB | Active local environment; rebuildable but retained |
| `.local/` | 360 MiB | Mix of protected games and historical scratch artifacts |
| `frontend/` | 148 MiB | About 228 KiB source/assets plus ignored dependencies and build output |
| `output*` | 9.2 MiB | Historical console logs except the active Sam-32 log |
| `src/` | 3.3 MiB | Active and historical Python implementations |
| `tests/` | 2.9 MiB | Active regression and historical artifact coverage |
| `build/` | 1.4 MiB | Rebuildable packaging output |
| `.pytest_cache/` | 72 KiB | Rebuildable test cache |
| `reports/` | 216 KiB | One active readiness report and historical experiment reports |
| `docs/` | 176 KiB | Active contracts, with several stale operational statements |
| Immutable card packs | 776 KiB | Protected sources; selected frontend assets are 228 KiB |

There are 227 tracked files, 16 modified tracked files, and 16 untracked files.
The working tree represents substantial intentional implementation work and
must not be reset or broadly rewritten.

## Live Sam-32 collector

The collector is healthy and protected.

| Field | Value |
| --- | --- |
| Corpus | `runs/sam-32-continuous-corpus-001` |
| Log | `output-sam-32-continuous-corpus-001` |
| PID file | `.local/sam-32-continuous-corpus-001.pid` |
| Parent PID at audit | `73252` |
| Worker count | Four search workers plus one resource tracker |
| Teacher | Symmetry-aware nested Sam, 32 outer × 32 actor-response simulations |
| Branching | Maximum `k = 4` |
| Split cycle | 27 training, two validation, two test decks |
| Disk floor | 1 GiB plus 256 MiB in-flight reserve per worker |
| Configuration digest | `d681a05fd1e9d53d07ee5e158166d47ccf55e1dbfb472910718712ee626caada` |
| Source revision | `ec6e3d395bc89e9b0cb4bf31eb15438119281c60` |
| Source-tree digest | `c5494eb05db31d1a0a8c9765623de8b7e3742a41ef31fef574b7437ec4d89222` |
| Source identity match | Passed |
| Committed state at audit | Four decks, 131,064 rows |
| Durable staging at audit | Twelve rounds, 65,532 rows |
| Log errors | None |
| Free disk | 119 GiB |

The collector source identity covers exactly:

- `NORTHSTARS`
- `pyproject.toml`
- 48 Python files under `src/`
- 14 Markdown files under `docs/`

The continuous parent creates a new spawned process pool for each four-deck
batch. Changing Python source while the current process remains attached to
this working tree can therefore change code imported by a later worker batch.
Changing any sealed source or document also prevents exact resume because the
source-tree check will fail. Committing unchanged contents changes Git HEAD and
also breaks the sealed revision check.

Until the collector is isolated, none of the listed identity files may change.
The requested inventory report is safe because `reports/` is outside the
source-identity set.

Exact recovery currently requires:

1. Git HEAD `ec6e3d395bc89e9b0cb4bf31eb15438119281c60`.
2. Exact contents producing source-tree digest
   `c5494eb05db31d1a0a8c9765623de8b7e3742a41ef31fef574b7437ec4d89222`.
3. The existing absolute corpus path.
4. The existing virtual environment or a compatible Python 3.12 environment.
5. `PYTHONPATH=src`.
6. `dracula.sam_miner resume --output
   runs/sam-32-continuous-corpus-001`.

## Repository map

### Active source and contracts

| Path | Role |
| --- | --- |
| `NORTHSTARS` | Project priorities and documentation standard |
| `src/dracula/cards.py`, `engine.py`, `randomness.py` | Deterministic rules foundation |
| `src/dracula/bridge.py` | Player-relative observation and action boundary |
| `src/dracula/search/information.py` | Information-safe player state and determinizations |
| `src/dracula/search/nested_strategic.py` | Nested 32×32 Sam controller used for local play |
| `src/dracula/search/sam_teacher.py` | Symmetry-aware nested dataset teacher implementation |
| `src/dracula/search/symmetry.py` | Authoritative early-destination groups |
| `src/dracula/sam_miner.py` | Active continuous branched corpus miner |
| `src/dracula/policy_value.py` | Feed-forward model body and policy head intended for Sam imitation |
| `src/dracula/api/`, `search_policy.py` | FastAPI gameplay and injected opponent execution |
| `frontend/src/`, `frontend/e2e/` | Browser gameplay and tests |
| `contracts/v1/` | Versioned public API fixtures |
| `queries/` | Local gameplay result reports |
| `docs/rules.md` | Authoritative game rules |
| `docs/sam-dataset-miner.md` | Active collection contract |
| `docs/neural-model.md`, `model-training.md` | Active policy-imitation design |
| `docs/architecture.md`, `frontend-experience.md` | Active application contracts |

### Active runtime artifacts

| Path | Treatment |
| --- | --- |
| `runs/sam-32-continuous-corpus-001` | Protected; never move, compress, delete, or inspect destructively while live |
| `output-sam-32-continuous-corpus-001` | Protected active append-only log |
| `.local/sam-32-continuous-corpus-001.pid` | Protected live process identity |
| `.local/dracula.sqlite3*` | Protected recorded human games |
| `.venv/` | Retain for the collector and local development |
| `frontend/node_modules/` | Retain unless reinstall cost is deliberately accepted |
| `runs/training-004/archives/policy-2-policy-2-v20.pt` | Retain as the active fixed PPO comparison control until its path is migrated |
| `runs/search-warmstart-smoke-001/warm-start-smoke.pt` | Retain while guided comparison mode remains exposed |
| `runs/teacher-v2-response-ranker-001/artifacts/response-ranker.pt` | Retain while historical student comparison mode remains exposed |

### Immutable source assets

| Path | Treatment |
| --- | --- |
| `Cards (large)/` | Protected card source pack used by the deterministic preparation script |
| `Cards (medium)/` | Protected alternate source pack; quarantined but not disposable |
| `frontend/public/cards/` | Protected selected 54-card browser asset set and attribution |
| `Dracula.png` | Unreferenced by source; retain until its intended portrait role is resolved |
| `.local/V1.svg.png`, `.local/V2.svg.png` | Uncertain generated Vampire sources; retain until compared with tracked SVG/PNG assets |

## Historical implementation map

The following code remains mechanically covered but is not the active learning
architecture:

| Area | Modules and commands | Current reason to retain |
| --- | --- | --- |
| Recurrent PPO | `models.py`, `collection.py`, `optimization.py`, `training.py`, `training_config.py`, `cli.py`, `legacy_policy_contract.py` | Reads and evaluates historical checkpoints and comparison controls |
| Version 1 search | `search/planner.py`, `search/validation.py`, `evaluation.py` | Fixed absolute control and regression baseline |
| Shallow Teacher v2 | `search/strategic.py`, response fixtures and validation | Historical behavior and privacy comparison |
| Response distillation | `response_dataset.py`, `response_distillation.py`, `response_ranker.py`, `search/response_student.py` | Reads historical datasets and supports an explicit comparison controller |
| Guided search | `search/guided.py`, `guided_policy.py`, `supervised.py`, `policy_value.py` | Historical prototype plus reusable future model code |
| Expert iteration | `expert.py`, `expert_evaluation.py` | Historical prototype, not active training |
| Tournament | `tournament.py` | Historical PPO comparison evidence |

These paths are candidates for later simplification, not immediate deletion.
The import surface, artifact readers, comparison commands, tests, and reports
must be evaluated together before removing any module.

## Historical artifacts proposed for archive

All destinations below are under
`.local/archive/2026-07-25/`. Each archive must carry an input manifest, SHA-256
digest, verified member listing, and tested restoration before originals are
removed.

### PPO runs

| Source | Proposed destination |
| --- | --- |
| `runs/training-001` | `runs/ppo/training-001.tar.zst` |
| `runs/training-002` | `runs/ppo/training-002.tar.zst` |
| `runs/training-003` | `runs/ppo/training-003.tar.zst` |
| `runs/training-004` except the selected `policy-2-v20` control | `runs/ppo/training-004.tar.zst` |
| `output`, `output-002`, `output-003`, `output-004` | `logs/ppo/` |
| `output-tournament-003`, `output-tournament-004` | `logs/ppo/` |

The selected `policy-2-v20` artifact must first be copied to a stable comparison
control path and verified by direct inference before the original
`training-004` tree is removed.

### Version 1 and guided-search records

| Source | Proposed destination |
| --- | --- |
| `runs/search-validation-001` | `runs/search-v1/search-validation-001.tar.zst` |
| `runs/search-teacher-smoke-100` | `runs/search-v1/search-teacher-smoke-100.tar.zst` |
| `runs/search-teacher-smoke-500` | `runs/search-v1/search-teacher-smoke-500.tar.zst` |
| `runs/search-teacher-warmstart-smoke-500` | `runs/search-v1/search-teacher-warmstart-smoke-500.tar.zst` |
| `runs/readiness-full-20260720` | `runs/guided/readiness-full-20260720.tar.zst` |
| `runs/search-warmstart-smoke-001` | Retain until the guided comparison artifact path is migrated |

### Shallow Teacher v2 and response-ranking records

| Source | Proposed destination |
| --- | --- |
| `runs/teacher-v2-validation-001` | `runs/teacher-v2/validation-001.tar.zst` |
| `runs/teacher-v2-shallow-validation-001` | `runs/teacher-v2/shallow-validation-001.tar.zst` |
| `runs/teacher-v2-shallow-benchmark-001` | `runs/teacher-v2/shallow-benchmark-001.tar.zst` |
| `runs/teacher-v2-smoke-001` | `runs/teacher-v2/smoke-001.tar.zst` |
| `runs/teacher-v2-response-dataset-001` | `runs/teacher-v2/response-dataset-001.tar.zst` |
| `runs/teacher-v2-training-smoke-001` | `runs/teacher-v2/training-smoke-001.tar.zst` |
| `runs/teacher-v2-response-ranker-001` | Retain until the comparison artifact path is migrated |
| `runs/teacher-v2-response-ranker-002` | `runs/teacher-v2/response-ranker-002.tar.zst` after artifact verification |
| Every `output-teacher-v2-*` and `output-response-hybrid-benchmark` | `logs/teacher-v2/` |

### Expert-iteration records

| Source | Proposed destination |
| --- | --- |
| `runs/expert-smoke-001` | `runs/expert-iteration/expert-smoke-001.tar.zst` |
| `runs/expert-sustained-smoke-001` | `runs/expert-iteration/expert-sustained-smoke-001.tar.zst` |
| `configs/expert*.toml` | Retain in the repository until active/historical config organization |

### Sam-128 records

| Source | Proposed destination |
| --- | --- |
| `runs/sam-128-prefix-smoke-001` | `runs/sam-128/prefix-smoke-001.tar.zst` |
| `.local/sam-128-dataset-miner-benchmark-001` | `local/sam-128/benchmark-001.tar.zst` |
| `.local/sam-128-dataset-miner-readiness-001` | `local/sam-128/readiness-001.tar.zst` |
| `.local/sam-128-readiness-package` | `local/sam-128/readiness-package.tar.zst` |
| `.local/sam-128-readiness-prefix-001` | `local/sam-128/readiness-prefix.tar.zst` |
| `output-sam-128-million-corpus-001` | `logs/sam-128/` |
| `runs/sam-128-million-corpus-001` | Remove after confirming it is empty |

### Local scratch and test data

Archive under `local/teacher-v2/`, `local/browser-tests/`, or
`local/resume-tests/`:

- `.local/response-distillation`
- `.local/response-ranker-resume-check*`
- `.local/teacher-v2-*` except any currently selected comparison artifact
- `.local/manual-teacher-v2-nested`
- `.local/manual-response-ranker-002`
- `.local/response-hybrid-benchmark.json`
- `.local/browser-ui-20260722.sqlite3`
- `.local/scoring-browser-20260722.sqlite3*`
- `.local/dracula-e2e.sqlite3*`
- `.local/dracula-guided-e2e.sqlite3*`
- `.local/dracula-guided-validation.sqlite3*`

The four non-collector PID files are stale and can be removed after their
associated logs are archived.

## Report organization proposal

There are no byte-identical Markdown reports, but 16 of the 21 existing reports
have no link from active docs. Preserve them by moving with `apply_patch` and
updating links:

```text
reports/
  README.md
  active/
    sam-32-continuous-corpus-readiness.md
  history/
    ppo/
    search-v1/
    teacher-v2/
    sam-128/
  maintenance/
    repo-hygiene-*.md
```

Suggested placement:

- `sam-32-continuous-corpus-readiness.md` → `active/`
- `sam-128-*` → `history/sam-128/`
- `teacher-v2-*`, `search-training-redesign.md`,
  `expert-iteration-readiness.md` → appropriate historical search or Teacher v2
  subdirectory
- This inventory and subsequent cleanup reports → `maintenance/`

Historical contents should not be rewritten as current requirements.

## Rebuildable or generated material

These are candidates for direct removal after confirming no active process has
an open file beneath them:

- `.DS_Store`
- `.pytest_cache/`
- `build/`
- `dist/` if present
- `frontend/dist/`
- `frontend/playwright-report/`
- `frontend/test-results/`
- Python `__pycache__/`
- `src/dracula_game.egg-info/`
- `tools/__pycache__/`
- Empty cache directories inside historical runs after those runs are archived

`frontend/node_modules/` is rebuildable but should remain unless reclaiming
148 MiB is worth a reinstall. `.venv/` is also rebuildable but is required by
the live collector and must remain.

## Candidate removals after archival

| Path | Reason | Prerequisite |
| --- | --- | --- |
| Root historical `output*` files | Completed console logs clutter the root | Archive digest and readable copy |
| `runs/sam-128-million-corpus-001` | Empty abandoned prepared run | Confirm no references and retain readiness report |
| Four stale `.local/*.pid` files | Their processes do not exist | Archive associated logs |
| One-off tools under `tools/` | Eight of nine have no external filename reference | Preserve report inputs/outputs and determine whether reproduction still matters |
| Historical configs | Point to abandoned datasets or training modes | Move to `configs/archive/`, do not silently delete |
| `Notes on UI` | Untracked note duplicates frontend requirements | Merge useful content into the frontend contract after source isolation |
| `Dracula.png` | No repository reference | Retain until visually compared with intended portrait requirement |

No active source module is approved for deletion by this audit.

## Stale and contradictory surfaces

### Confirmed operational drift

1. `README.md` says the default opponent is version 1 search at 500 simulations.
   The user-selected local opponent and active contract are nested Sam at
   32 outer × 32 actor-response simulations.
2. `Makefile` sets `OPPONENT=search` and `SEARCH_SIMULATIONS=500`, reproducing
   the stale README default.
3. `docs/design-tracker.md` calls the already-running corpus `Gated` and says
   the current work is to start it after authorization.
4. `reports/sam-32-continuous-corpus-readiness.md` says collection has not
   started and documents mutable-working-tree launch commands.
5. `README.md` presents expert iteration as an active top-level workflow even
   though active training is Sam policy imitation.
6. Several active documents retain search-guided model language inherited from
   the superseded expert-iteration plan. It must either be marked historical or
   removed where it competes with Sam imitation.

### Naming debt

The active teacher is Sam-32, but stable internal schema names and many source
messages in `src/dracula/search/sam_teacher.py` still say Sam-128. The dataset
contract explicitly preserves one historical namespace for compatibility.
Renaming schema identifiers would be a contract change; user-facing comments
and diagnostics can be clarified after the collector is isolated.

### Healthy consistency results

- No broken Markdown links were found across `README.md`, `docs/`, and
  `reports/`.
- No tracker ID appears outside `docs/design-tracker.md`.
- No exact duplicate Markdown report was found.
- Immutable card-source references consistently select `Cards (large)` and
  explicitly reject automatic use of `Cards (medium)`.

## Configurations and tools

All nine root-level tool scripts are experiment-specific. Eight have no
external filename reference; `benchmark_teacher_parallelism.py` has one.
They should be treated as historical reproduction tools until their output
reports and raw manifests are archived.

The nine TOML files under `configs/` belong to warm-start, Teacher v2,
response-ranker, or expert-iteration experiments. None configures the active
Sam-32 corpus. They should move under `configs/archive/` unless an explicit
comparison command still consumes them.

The following console entry points are historical or comparison surfaces, not
active collection:

- `dracula-search-validate`
- `dracula-search-signal`
- `dracula-teacher`
- `dracula-response-dataset`
- `dracula-response-ranker`
- `dracula-supervised`
- `dracula-expert`

`dracula-sam-miner` is active. Entry-point removal must wait for source
isolation and import/test evidence.

## Verification before destructive work

### Collector isolation

- Recreate the sealed revision and exact source-tree digest in a dedicated
  ignored Git checkout.
- Run the source-identity verifier from that checkout.
- Stop the current process with SIGINT and wait for all workers.
- Verify committed manifests and sealed staging artifacts.
- Resume from the frozen checkout.
- Confirm new workers import frozen, not mutable, source.

### Run archival

- Record original path, byte count, purpose, and report association.
- Hash every source file or a canonical input manifest.
- Create one low-priority archive at a time.
- Verify archive listing and digest.
- Restore representative files and compare bytes.
- Preserve resolved configurations and artifact manifests.
- Remove the original only after verification.

### Model controls

- Copy selected policy artifacts to stable control paths.
- Verify artifact SHA-256.
- Run direct inference equivalence before changing Makefile paths.
- Preserve policy metadata and contract versions.

### SQLite records

- Stop any application process that holds the database.
- Include the database and matching WAL/SHM state in one consistent backup.
- Run SQLite integrity checks against the backup.
- Compare aggregate game counts and completed results.
- Never run `make reset-local` as cleanup.

### Report moves

- Use `apply_patch` moves.
- Update every active and historical link.
- Rerun the Markdown link checker.
- Preserve report contents and evidence dates.

### Cache removal

- Confirm no live process has an open file under the cache.
- Remove only regenerated content.
- Rebuild or rerun one representative command afterward.

## Classification summary

### Active source or contract

Engine, bridge, information state, nested Sam, Sam teacher, symmetry, Sam miner,
policy imitation model, FastAPI, frontend, public contracts, rules, architecture,
and active collection/training docs.

### Active runtime artifact

Continuous Sam-32 corpus, active log and PID, SQLite human-game records, virtual
environment, frontend dependencies, and explicitly exposed comparison artifacts.

### Immutable source asset

Both card packs, selected frontend card assets and attribution, and—pending
resolution—the root Dracula portrait.

### Historical evidence to archive

PPO runs and tournaments; search version 1 validations; shallow Teacher v2
data, rankers, and benchmarks; expert-iteration runs; Sam-128 smoke and
benchmarks; completed output logs; test databases; and resume fixtures.

### Rebuildable cache or generated output

Build directories, frontend distribution and test reports, package metadata,
bytecode, pytest cache, and `.DS_Store`.

### Candidate for removal

Empty Sam-128 run directory, stale PID files, verified archived root logs,
rebuildable caches, merged UI note, and one-off scripts only after their
reproducibility value is resolved.

### Uncertain and therefore retained

Historical artifact readers and tests, explicit comparison controller artifacts,
`Dracula.png`, generated Vampire source images, both card packs, and any tool or
configuration whose consumer cannot yet be mechanically excluded.

No cleanup action was performed during this audit.

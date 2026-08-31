# Repository layout cleanup

Cleanup date: July 25, 2026

## Result

The main checkout now presents the active engine, nested Sam opponent, browser
application, Sam corpus miner, tests, and contracts as its primary surface.
Historical experiment inputs and one-off utilities remain available without
appearing as active defaults.

The frozen Sam-32 collector checkout, active corpus, output log, PID, SQLite
game database, comparison artifacts, immutable card packs, and selected
frontend assets were not moved or modified.

## Layout changes

- Moved the nine superseded expert-iteration, search-guided, Teacher v2
  training, and response-ranker TOML files under `configs/archive/`.
- Moved the nine one-off Teacher v2 and Sam-128 measurement and report scripts
  under `tools/archive/`.
- Added concise indexes for `configs/`, `tools/`, and `reports/`.
- Retained all historical readers in `src/dracula`; no source module was
  removed.
- Removed the seven historical experiment console entry points from package
  metadata. Their readers remain explicitly runnable with
  `python -m dracula.<module>`. `dracula-sam-miner` is the sole installed
  project command.
- Removed empty abandoned directories and regenerated Python/package caches
  created during validation.
- Updated historical references required by the configuration and tool moves.

## Frontend note merge

The standalone `Notes on UI` file was incorporated into
`docs/frontend-experience.md` and removed.

The active contract now states that:

- human turns identify Queen or King and row or column orientation;
- opponent scoring is labeled **Dracula's Score**;
- hand cards retain a 72-pixel readable minimum;
- the custom `V1.svg` and `V2.svg` Vampire designs remain visually distinct
  from Jacks.

The note requesting slower scoring was not retained because the user later
restored the faster presentation. The frontend implementation already matched
the other requirements. One stale browser-test expectation for **My Score** was
corrected, and the card attribution now describes the custom Vampire assets.

## Active commands

The `Makefile` now defaults local gameplay to nested Sam with 32 outer and 32
actor-response simulations. Other controllers remain explicit `OPPONENT`
choices; none is a fallback.

| Command | Purpose |
| --- | --- |
| `make dev` | Run FastAPI and the Vite development server |
| `make preview` | Build and preview local gameplay against nested Sam 32×32 |
| `make preview-control` | Explicitly select the archived policy comparison |
| `make test` | Run the complete Python and frontend checks |
| `make corpus-health` | Show live processes and committed corpus counts |
| `make corpus-inspect` | Inspect the corpus through the frozen source |
| `make corpus-verify` | Verify corpus hashes through the frozen source |
| `make corpus-stop` | Send `SIGINT` and wait for all collector processes |
| `make corpus-resume` | Resume in the foreground from frozen source |

All repository paths in tracked commands are relative to `CURDIR`; no absolute
machine path was added to tracked source or configuration.

## Mechanical code audit

A syntax-tree import scan found eight unused imports in the API service,
bridge, historical evaluation/expert readers, and search modules. They were
removed. The scan is now clean.

The old `dracula.cli` module has no active import or package entry point, but it
was retained because it reads and operates on archived PPO training artifacts.
The other historical learning modules remain covered by tests or explicitly
selected comparison controllers. No dependency was removable: FastAPI,
Pydantic, PyTorch, and Uvicorn all remain part of the active runtime.

The response-ranker test previously depended on an ignored local smoke dataset
that was archived during cleanup. It now uses an equivalent deterministic,
self-contained dataset projection while continuing to test split isolation,
ranking loss, frozen value-head behavior, artifact compatibility, and metrics.

## Verification

- Imported all 48 Python modules successfully.
- Built the wheel successfully; its only console script is
  `dracula-sam-miner`.
- `pip check` reported no broken Python requirements.
- Frontend dependency inspection reported no missing packages.
- Focused engine, bridge, information-state, symmetry, search, Sam teacher,
  miner, API, and repository tests: **324 passed**.
- Self-contained response-ranker tests: **6 passed**.
- Frontend unit tests: **73 passed**.
- TypeScript, ESLint, production build, and production localhost-string check:
  passed.
- All active and comparison `OPPONENT` selections passed Makefile validation.
- Corpus stop/resume commands passed dry-run validation; neither was executed.
- Archived TOML files parsed and archived Python utilities passed syntax
  parsing.
- Package diff-format checks passed.
- No stale pre-move config or tool paths remain outside the historical hygiene
  inventory.
- No absolute user path occurs in tracked source, configuration, tests,
  contracts, frontend, or active command files.

Frozen-source corpus inspection and verification both passed:

| Measure | Value |
| --- | ---: |
| Committed decks | 4 |
| Rows | 131,064 |
| Rounds | 24 |
| Shards | 120 |
| Terminal leaves | 196,608 |
| Corpus content digest | `e56bb000a98dbd8da0e951f1c5a1aa3290b0173d97e5bb9e53cfd868d47c0603` |

At final verification the collector parent remained live with four active
workers and one resource tracker, all isolated from the main source tree.

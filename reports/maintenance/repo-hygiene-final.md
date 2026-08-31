# Final repository-hygiene handoff

Audit completed: July 25, 2026 at 11:51 EDT

## Result

The Dracula repository is mechanically clean and ready for handoff. The root
now exposes the deterministic engine, FastAPI and React application, nested
Sam opponent, continuous Sam-32 corpus miner, tests, active contracts, and
indexed evidence without presenting historical experiments as active work.

Historical records are archived and restorable. Protected game records,
immutable card resources, comparison artifacts, and the continuous corpus
remain intact. The live collector is isolated from this working tree and
continued computing throughout the final audit.

No feature, training, deployment, narrator, or model-design work was started.
No commit was created.

## Before and after

| Surface | Before cleanup | Final state |
| --- | --- | --- |
| Root | Active project files mixed with historical logs and an unexplained UI note | Six tracked entry files, indexed project directories, one ignored active corpus log, and an explicit ownership map |
| Local opponent | README and Makefile still selected the original 500-simulation controller | Nested Sam 32×32 is the explicit local default |
| Collection | Continuous collector depended on mutable working-tree imports | Collector and all workers use a verified frozen checkout |
| `runs/` | Approximately 6.0 GiB spanning several experiment generations | Active corpus plus three explicitly retained comparison dependencies |
| Configurations | Superseded TOML files appeared active | Historical inputs are indexed under `configs/archive/` |
| Utilities | One-off report and benchmark scripts appeared active | Reproducibility tools are indexed under `tools/archive/` |
| Reports | Historical and operational evidence shared one flat directory | Active, historical, and maintenance evidence have separate indexed locations |
| Documentation | Several competing PPO, search, and distillation plans remained active | One concise Sam-32 collection and one-hot policy-imitation design |
| Generated output | Historical output logs, caches, build state, and test artifacts remained | Verified historical records are archived; rebuildable output was removed |

The final root contains:

- Project entry points: `NORTHSTARS`, `README.md`, `Makefile`,
  `pyproject.toml`, and `.gitignore`.
- Active implementation and verification: `src/`, `tests/`, and `frontend/`.
- Contracts and operations: `docs/`, `contracts/`, `configs/`, `tools/`,
  `queries/`, and `reports/`.
- Retained source assets: `Cards (large)`, `Cards (medium)`, and
  `Dracula.png`.
- Ignored protected state: `.local/`, `runs/`, and the active `output*` log.

## Space reclaimed

| Measure | Bytes |
| --- | ---: |
| Historical originals archived | 4,184,170,519 |
| Verified compressed archives | 3,434,452,575 |
| Rebuildable output removed directly | 141,632,680 |
| Net space reclaimed | 891,350,624 |

The cleanup reclaimed approximately 850 MiB while retaining the historical
payloads. Current large local areas are the 3.20 GiB verified archive, the
2.37 GiB `runs/` tree including active in-flight collection, the 720 MiB
virtual environment, and 146 MiB of frontend dependencies.

## Active architecture and commands

The engine and scoring implementation own game truth. FastAPI and SQLite own
local sessions and transactions. React renders only the public API view. Nested
Sam 32×32 makes local opponent decisions through player-relative information
states. The symmetry-aware Sam-32 miner continuously produces one-hot
strategic-group imitation rows. The feed-forward policy head is the next
planned consumer; training is not running.

Primary commands:

```bash
make dev
make preview
make preview-control
make test
make test-e2e
make play-report
make corpus-health
make corpus-inspect
make corpus-verify
make corpus-stop
make corpus-resume
```

`make preview` resolves nested Sam at 32 outer and 32 actor-response
simulations. Comparison controllers require explicit selection and never serve
as a silent fallback. `dracula-sam-miner` is the sole installed project entry
point.

## Archive and restore

Historical payloads are under:

`.local/archive/2026-07-25/`

The machine-readable inventory is:

`.local/archive/2026-07-25/manifest.json`

Its current SHA-256 remains:

`4cb49be791937b7c68c8bd23fdcd809285e20ca3668212a0a4f107fe4248d006`

The manifest indexes 11 zstd archives and 81 original paths. Every archive has
an input manifest, SHA-256, member listing, and representative restoration
record. Restore an item without altering the archive:

```bash
mkdir -p /path/to/restore
zstd -dc .local/archive/2026-07-25/<archive>.tar.zst |
  tar -xf - -C /path/to/restore <member-path>
```

The final audit retested the search-v1 archive and restored
`runs/search-validation-001/fixture-records.json`. Its 104,625 bytes reproduced
SHA-256
`acf65ecaf3a49068ba9f55397bf966bd63a0350dde8892c80afd77004b8c9c64`.

## Validation

The complete post-cleanup pass produced:

| Check | Result |
| --- | --- |
| Full Python suite | 543 passed |
| Final focused engine, bridge, information-state, symmetry, Sam, miner, API, and transaction suite | 250 passed in 154.52 seconds |
| Frontend suite | 10 files and 73 tests passed |
| TypeScript, ESLint, production build | Passed |
| Static package imports | Package root and all 47 submodules imported |
| Package dependencies and wheel | Passed; only `dracula-sam-miner` is exported |
| API and dataset privacy | Passed; forbidden private fields absent |
| SQLite integrity | `ok`; 25 recorded games, including 19 complete |
| Archive digests and zstd integrity | All 11 passed |
| Representative archive restoration | Passed byte-for-byte |
| Markdown paths and anchors | Passed |
| Tracker-ID confinement | Passed |
| Active terminology | No stale default or finite-corpus claim |
| Make targets and CLI help | Passed |
| Diff formatting | Passed |

The final directory audit found no orphaned active configuration, dead
installed entry point, accidental absolute source path, or unexplained root
file. Historical schema names remain where changing them would break artifact
compatibility.

## Current corpus health

| Field | Final audit |
| --- | --- |
| Parent PID | `45590` |
| Children | Four active search workers and one resource tracker |
| Configuration digest | `d681a05fd1e9d53d07ee5e158166d47ccf55e1dbfb472910718712ee626caada` |
| Source revision | `ec6e3d395bc89e9b0cb4bf31eb15438119281c60` |
| Source-tree digest | `c5494eb05db31d1a0a8c9765623de8b7e3742a41ef31fef574b7437ec4d89222` |
| Committed decks | 4 |
| Committed rows | 131,064 |
| Committed terminal leaves | 196,608 |
| Durable in-flight boundary | 16 sealed rounds, 87,376 rows |
| Temporary or partial staging files | 0 |
| Parent and child RSS | Approximately 126 MiB |
| System memory free | 42% |
| Free disk | Approximately 121 GiB |

The parent and every child use
`.local/collector-source/sam-32-continuous-corpus-001` as their working
directory and its absolute `src` as `PYTHONPATH`. Source identity recomputation
matched the resolved configuration exactly, and no frozen identity input is a
symlink.

Committed corpus inspection and verification passed. The active log contains
the documented source-isolation interrupt and no traceback, digest, corruption,
or disk error. A 10-second final sample added 38 content-addressed worker-cache
entries, confirming ongoing computation toward the next four-deck commit.

System swap is elevated at approximately 5.39 GiB of 6 GiB. Current memory
pressure is not high, collector RSS is small, and no collector failure or
throughput stop is present. Swap remains an operational metric worth monitoring
rather than a repository defect.

## Retained historical material

- Eleven compressed archive classes covering PPO, search version 1, guided
  search, shallow Teacher v2, response ranking, expert iteration, Sam-128,
  historical logs, and local test evidence.
- Historical artifact readers and regression modules needed for explicit
  comparisons and archive inspection.
- The selected PPO, guided-search, and response-ranker comparison artifacts
  under `runs/training-004`, `runs/search-warmstart-smoke-001`, and
  `runs/teacher-v2-response-ranker-001`.
- Historical configurations, tools, and reports under clearly named archive
  and history directories.
- Both source card packs, selected frontend assets, the retained Dracula
  portrait source, recorded SQLite games, and generated Vampire source images.

None is an active default or competing tracker item.

## Proposed Git commit

The working tree contains one intentional cumulative change set: the completed
Sam/miner implementation, related tests and service integration, repository
layout cleanup, historical moves, active documentation rewrite, report
hierarchy, and maintenance evidence. Ignored corpus data, archives, databases,
logs, frontend build output, dependencies, and frozen source are excluded.

Exact proposed staging scope:

```bash
git add \
  .gitignore Makefile README.md pyproject.toml \
  configs docs \
  frontend/e2e/gameplay.spec.ts \
  frontend/public/cards/ATTRIBUTION.md \
  reports src tests tools
```

Recommended commit message:

```text
feat: finalize the continuous Sam-32 miner and repository handoff
```

## Remaining concern

There is no unresolved mechanical repository defect. The only current
operational watch item is elevated system swap while collection runs; the
collector itself remains healthy, isolated, deterministic, and well below its
disk guard.

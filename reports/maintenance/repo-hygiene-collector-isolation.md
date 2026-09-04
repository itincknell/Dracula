# Sam-32 collector source isolation

Migration completed: July 25, 2026 at 10:16 EDT

## Result

The active Sam-32 continuous collector now runs from an ignored, detached,
physical source checkout:

`<repository>/.local/collector-source/sam-32-continuous-corpus-001`

The checkout is independent of the mutable main working tree. The collector
continues to use the original corpus, virtual environment, log, and PID file.

## Sealed identity

| Field | Value |
| --- | --- |
| Configuration digest | `d681a05fd1e9d53d07ee5e158166d47ccf55e1dbfb472910718712ee626caada` |
| Git revision | `ec6e3d395bc89e9b0cb4bf31eb15438119281c60` |
| Source-tree schema | `dracula-source-tree-v1` |
| Source-tree digest | `c5494eb05db31d1a0a8c9765623de8b7e3742a41ef31fef574b7437ec4d89222` |
| Python | Python 3.12.9 from the existing project virtual environment |
| Frozen `PYTHONPATH` | `.local/collector-source/sam-32-continuous-corpus-001/src` |

The frozen checkout's source-identity calculation reproduced the sealed Git
revision and tree digest exactly. Its `NORTHSTARS`, `pyproject.toml`,
`src/**/*.py`, and `docs/**/*.md` contain no symlinks. Representative source
files have different inodes from their main-working-tree counterparts.

The checkout is intentionally dirty relative to the detached revision because
the corpus was originally sealed from that revision plus the exact then-current
working-tree contents.

## Migration boundary

Before interruption:

- Parent PID: `73252`
- Worker PIDs: `85122`, `85123`, `85124`, `85125`
- One multiprocessing resource tracker: `73255`
- Committed decks: 4
- Committed rows: 131,064
- Committed terminal leaves: 196,608
- Corpus content digest:
  `e56bb000a98dbd8da0e951f1c5a1aa3290b0173d97e5bb9e53cfd868d47c0603`
- Corpus file SHA-256:
  `f783136ec1ac815892ab4fc0c2167f29b7130a493ebbcddc066f138ac33e4158`
- Durable in-flight boundary: three sealed rounds in each of deck ordinals
  4 through 7
- Log byte position before interruption: 947

`SIGINT` stopped the parent and all five children cleanly. The collection state
became `interrupted`. The committed corpus manifest remained byte-identical,
and no temporary, partial, or stop-request file remained. Frozen-source
verification then accepted all four decks, 24 rounds, 120 shards, 131,064
cache entries, and 131,064 rows.

## Resumed process

- Parent PID: `45590`
- Resource tracker PID at verification: `45739`
- Worker PIDs at verification: `45740`, `45741`, `45742`, `45743`
- PID file: `.local/sam-32-continuous-corpus-001.pid`
- Corpus: `runs/sam-32-continuous-corpus-001`
- Log: `output-sam-32-continuous-corpus-001`
- Log byte position after the recorded interrupt: 969

All parent and child current-working directories were the frozen checkout.
The parent and all four workers carried the frozen absolute `PYTHONPATH`.
A direct import resolved `dracula.sam_miner` from the frozen checkout and
reproduced the sealed source identity.

At the post-resume check, the four workers were active near full CPU
utilization. Combined parent, tracker, and worker resident memory was
approximately 315 MiB, system memory reported 36% free, and the data volume had
approximately 120 GiB free.

The committed manifest still contained exactly four unique fixture IDs and four
unique fixture indexes (`0` through `3`). Its digest, row count, and terminal
leaf count were unchanged, proving that migration did not duplicate or
partially commit a deck. The workers resumed the same four staging directories
at their sealed round boundaries.

The state file remains `interrupted` while the resumed in-flight batch is being
completed; the collector writes `collecting` with the next ordinal when that
entire four-deck batch seals.

## Recovery

The exact verification and resume commands are recorded in
`.local/collector-source/README.md`.

The live parent and future spawned worker pools now use only the frozen
checkout's cwd and `PYTHONPATH`. Changes to `NORTHSTARS`, `pyproject.toml`,
`src/`, or `docs/` in the main working tree cannot alter imports for this
collector or prevent its recovery.

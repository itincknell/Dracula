# Standalone pi1 runtime cleanup

## Scope

The cleanup begins from pushed baseline
`fbc2fceaf3bb95983fc26252cf373cbc26ae0a92`. Git history remains the archive
for executable PPO, policy/value, search-teacher, response-ranker, hybrid,
Sam, miner, evaluator, and superseded trainer implementations.

The installed product retains the deterministic engine, actor-visible
information state, authoritative symmetry rules, selected standalone `pi1`,
stateless API, narration, frontend, release tooling, and the current 659-bit
corpus reader/trainer. The only model-data compatibility code retained is
`bgc_policy_migration.py`, a read-only verifier for the physically converted
corpus used by the selected training line.

## Direct observation path

`policy_observation.py` now constructs the selected input directly:

```text
9 x 54 coffin          486
3 x 54 card status     162
lifecycle context       11
total                  659 bits
```

Candidate card rows come from current in-hand membership in canonical card-ID
order. `SearchInformationState.own_hand` maps a selected candidate card back
to its occupied stable engine slot. The active runtime no longer constructs a
`bool[4,54]` hand-slot prefix, compacts an 875-bit tensor, or converts action
rows through that removed tensor.

One correction discovered by the complete-game comparison preserved the old
information-state identity exactly: played cards from completed rounds and the
current coffin must be sorted as one canonical union. Segment-wise sorting
left the 659 input unchanged but altered the fair-coin seed on some paired
openings. The union ordering is now regression-tested.

## Behavior equivalence

Before deletion, 21 fixed states spanning Queen, King, and late-round play
proved that direct 659-bit observations equal the former compacted tensors bit
for bit. The fixture bundle digest was
`4c10eb8d8ebd061b78238bbda02c30f1b0fda65ed038caef980a2a93187317cd`.

The selected artifact remained byte-identical:

```text
70c76f2eb64600eab2297640278a6c94d4336ab8e73bf941a6d96237f69f5b5c
```

For the seven fixed `pi1-direct-queen` decisions, observation hashes, logit
hashes, representative actions, and concrete actions match the pre-cleanup
baseline exactly. The first decision remains representative 17 and concrete
22; all seven tuples are enforced by the local-artifact regression test.

Complete stateless games using the first public legal human move at every turn
also match the baseline:

| Fixture | Transcript SHA-256 | Final human / opponent |
| --- | --- | ---: |
| `pi1-game-queen` | `83428a09bd403c3b61ca6d1a59f017c5dfae890278c5da3ada2ba5b278626a2e` | 198 / 269 |
| `pi1-game-king` | `d52706456a89c07849191ebd3b74130f7170b400275f4e4d8239b4a4cacf49bf` | 234 / 306 |

The existing stateless suites cover cache hit/miss equality, reload replay,
malformed histories, forced moves, public projection, and private-field
exclusion.

## Removed execution surface

- 42 of the baseline's 85 installed Python modules were removed.
- 33 tests dedicated solely to retired implementations were removed.
- Retired collector, evaluator, report-generator, migration, and smoke tools
  were removed.
- Superseded experiment configurations and five obsolete console entry points
  were removed.
- The Makefile now exposes only selected `pi1` gameplay, current verification,
  current training, and release operations.
- The Lambda builder copies a 34-module production import closure instead of
  the entire source tree; trainer and local-stateful modules are excluded.

## Validation

At the pre-commit validation point:

- Python: 180 retained tests passed before the final artifact regression was
  added; the final total is recorded below after the last full run.
- Frontend: 13 files and 99 tests passed, followed by TypeScript, ESLint, and
  the production `/Dracula/` build.
- Browser: both production Playwright scenarios passed, including a complete
  six-round game, replay, narration timing, failure isolation, mobile, and
  desktop checks.
- Packaging: the wheel built and core imports succeeded.
- Lambda: its generated context imported the production app and excluded
  trainer, migrated-corpus reader, stateful service, tests, runs, databases,
  and generated Python metadata.

Final pre-commit results:

- Python: **181 passed**.
- Frontend: **99 passed** across 13 files; TypeScript, ESLint, and production
  build passed.
- Browser: **2 Playwright scenarios passed**, including a complete six-round
  game.
- Direct observation, selected-artifact logits/actions, complete-game
  transcripts, cache/reload replay, privacy, package imports, wheel build,
  Lambda import closure, Markdown links, tracker-ID placement, secret scan,
  and diff formatting passed.

The retained migrated-corpus reader also verified the selected sealed snapshot
without modifying it: 806,610 training rows and 89,670 validation rows, with
each of placements 1–7 represented by 115,230 and 12,810 rows respectively.
Its snapshot, dataset, and split digests verified, and the reader's recursive
privacy inspection passed.

# Standalone Sam policy gameplay evaluation 001

## Scope

This pass mechanically verified the epoch-3 standalone Sam classifier, added
it as the explicit `sam-policy` gameplay controller, and compared it with
random legal play, the archived PPO control, and nested Sam 32×32. No
competence threshold or automatic promotion rule was applied.

The evaluation used 12 fixed deck seeds in both roles for each control:
24 games and 144 completed rounds per comparison. Raw results are stored at
`.local/sam-policy-evaluation-001/results.json`.

## Artifact and frozen data

- Model: `SamPolicyModel`, 738,569 parameters
- Snapshot: 16 complete decks and 524,256 rows
- Training overlay: 14 decks and 458,724 rows
- Validation overlay: 2 decks and 65,532 rows
- Selected checkpoint: epoch 3
- Artifact:
  `runs/sam-policy-training-001/artifacts/policy.pt`
- Artifact file digest:
  `5d595935e92ed0b378b2507345850e050484b6afbb84eba710c8fb14751483ab`
- Model-state digest:
  `edd353f5209ba98220a5c69985184be583ff2bcce3e73ab3a1aca4f3de326afe`
- Snapshot digest:
  `7939d0f67b91ae74d3ec6b1f587b2baa4602c3563d3834882250ee079d6edf4e`
- Artifact size: 2,966,453 bytes (2.83 MiB)
- Raw evaluation digest:
  `7ba9e0eff0033ccc05c15d4b8122d1dfa69dd7230f1837dbb3456c94b7146967`

Strict artifact loading reproduced all metadata, schema, parameter-count, state
shape, finite-value, source, training, snapshot, and state-dictionary digests.
The exported state dictionary is tensor-exact with
`checkpoints/best-validation.pt`, and direct versus exported inference is
tensor-exact.

## Offline imitation result

The selected checkpoint's held-out cross-entropy was `0.916986`, compared with
`0.985684` for its untrained initialization.

| Split | Rows | Top-1 | Top-2 | Top-3 |
| --- | ---: | ---: | ---: | ---: |
| All | 65,532 | 58.47% | 88.21% | 91.74% |
| Queen | 32,766 | 60.00% | 88.41% | 91.96% |
| King | 32,766 | 56.94% | 88.02% | 91.51% |
| Dealer | 13,104 | 34.32% | 54.79% | 69.22% |
| Non-dealer | 52,428 | 64.50% | 96.57% | 97.37% |

Natural branch frequency places 49,152 of the 65,532 validation rows at
placement 7. Placement-1 and placement-2 results contain only 12 and 48 rows,
respectively.

## Strategic fixtures

Fixture success is measured at the strategic-group level. A paired concrete
destination selected by the deterministic fair coin therefore retains the
same strategic meaning as its proxy.

- Constructive/strategic catalog: standalone `7/12`; nested Sam `11/12`.
- Defensive catalog: standalone `5/14`; nested Sam `12/14`.
- The defensive catalog includes two forced placements. Excluding them,
  standalone passed `3/12` and nested Sam passed `10/12`.
- Standalone and nested Sam selected the same strategic group on `10/26`
  fixtures, including both forced placements.

| Catalog | Fixture | Accepted action | Standalone | Nested Sam | Standalone accepted | Same group as Sam |
| --- | --- | --- | --- | --- | --- | --- |
| Constructive | `queen-early-nondealer-suit` | `7C@4` | `7C@4` | `7C@4` | yes | yes |
| Constructive | `queen-early-dealer-suit` | `9C@4`, `QC@4` | `QC@4` | `KD@9` | yes | no |
| Constructive | `king-early-nondealer-suit` | `KC@4` | `KC@9` | `KC@4` | no | no |
| Constructive | `king-early-dealer-vampire` | `V1@2` | `AH@2` | `V1@2` | no | no |
| Constructive | `queen-middle-offense-defense` | `JS@4` | `JS@4` | `JS@4` | yes | yes |
| Constructive | `queen-middle-color-exploit` | `JS@7` | `3S@7` | `JS@7` | no | no |
| Constructive | `king-middle-suit` | `9S@1` | `9S@9` | `9S@1` | no | no |
| Constructive | `queen-middle-avoid-vampire` | `9C@6` | `9C@6` | `9C@6` | yes | yes |
| Constructive | `king-middle-preserve-card` | `6S@3` | `6S@4` | `6S@3` | no | no |
| Constructive | `king-late-block` | `3C@3` | `3C@3` | `3C@3` | yes | yes |
| Constructive | `queen-late-offense-defense` | `4S@6` | `4S@6` | `4S@6` | yes | yes |
| Constructive | `dealer-late-forced` | `JS@2` | `JS@2` | `JS@2` | yes | yes |
| Defensive | `king-early-dealer-constructive` | `KD@4` | `V2@4` | `KD@4` | no | no |
| Defensive | `queen-early-dealer-safe-intersection` | `JS@6` | `3S@6` | `AD@1` | no | no |
| Defensive | `queen-early-nondealer-avoid-suit` | `6C@2`, `QD@2` | `6C@2` | `QD@1` | yes | no |
| Defensive | `king-early-nondealer-safe-intersection` | `8C@6`, `JD@9` | `8C@2` | `8C@6` | no | no |
| Defensive | `king-middle-avoid-suit` | `QH@2` | `3H@2` | `QH@2` | no | no |
| Defensive | `queen-middle-avoid-color` | `2S@3` | `2S@4` | `2S@3` | no | no |
| Defensive | `queen-middle-defensive-vampire` | `V2@9` | `6S@9` | `V2@9` | no | no |
| Defensive | `queen-middle-avoid-vampire-destruction` | `10H@9` | `10H@9` | `10H@9` | yes | yes |
| Defensive | `king-middle-denial-over-offense` | `6D@7` | `6D@2` | `6D@7` | no | no |
| Defensive | `queen-middle-constructive-over-empty-block` | `7C@8` | `7C@8` | `7C@8` | yes | yes |
| Defensive | `queen-late-block` | `3H@9` | `3H@1` | `3H@9` | no | no |
| Defensive | `king-late-block` | `6D@3` | `6D@2` | `6D@3` | no | no |
| Defensive | `queen-late-dealer-forced` | `V2@7` | `V2@7` | `V2@7` | yes | yes |
| Defensive | `king-late-dealer-forced` | `JH@3` | `JH@3` | `JH@3` | yes | yes |

The standalone fixture decisions had p50/p95/maximum latency of
`2.11/2.88/3.15 ms`. Nested Sam's fixture decisions had
`2.52/25.21/26.22 s`.

## Role-balanced games

The score differential is standalone score minus control score, averaged over
all rounds.

| Control | Games W-L-T | Rounds W-L-T | Mean round differential |
| --- | ---: | ---: | ---: |
| Random legal | 22-2-0 | 94-49-1 | +10.65 |
| Archived PPO | 20-4-0 | 93-49-2 | +11.88 |
| Nested Sam 32×32 | 1-23-0 | 41-101-2 | -17.61 |

### Queen and King

| Control | Queen W-L-T | Queen differential | King W-L-T | King differential |
| --- | ---: | ---: | ---: | ---: |
| Random legal | 11-1-0 | +11.88 | 11-1-0 | +9.42 |
| Archived PPO | 10-2-0 | +12.12 | 10-2-0 | +11.64 |
| Nested Sam 32×32 | 1-11-0 | -21.68 | 0-12-0 | -13.54 |

### Dealer and non-dealer rounds

| Control | Dealer W-L-T | Dealer differential | Non-dealer W-L-T | Non-dealer differential |
| --- | ---: | ---: | ---: | ---: |
| Random legal | 44-27-1 | +9.57 | 50-22-0 | +11.72 |
| Archived PPO | 51-20-1 | +13.56 | 42-29-1 | +10.21 |
| Nested Sam 32×32 | 24-47-1 | -11.93 | 17-54-1 | -23.29 |

Across all 1,512 learned standalone decisions, latency was `1.83 ms` p50,
`3.33 ms` p95, and `21.99 ms` maximum while the continuous corpus collector
and nested evaluation were running. The evaluation process peak RSS was
242,122,752 bytes (230.9 MiB). The completed training process peak was
535.8 MiB. The live FastAPI process used 226,752 KiB RSS immediately after
loading the artifact and 206,896 KiB after one minute; the production frontend
fell from 99,296 to 78,880 KiB over the same interval.

## Mechanical and privacy verification

- 183 focused Python tests passed in 147.04 seconds.
- 73 frontend tests passed across 10 files.
- TypeScript, ESLint, production build, and the no-localhost build check passed.
- The package built as a wheel and included the standalone controller.
- Complete API games using the selected real artifact finished all six rounds
  as Queen and King; each included 24 opponent turns and three forced turns.
- In-memory and SQLite transaction, retry, stale-version, and recovery tests
  passed.
- Representative-only masks passed every authoritative symmetry case.
- Deterministic paired-destination resolution exercised both concrete members.
- No evaluation or API turn selected an illegal action.
- Public responses contained no artifact path or digest, raw logits, legal or
  representative masks, request seed, hidden bytes, stock order, opponent
  hand, search state, or state dictionary.
- The continuous Sam-32 corpus and SQLite gameplay database were not modified
  by evaluation.

## Evidence and uncertainty

The classifier is decisively stronger than the two weak absolute controls and
decisively weaker than nested Sam on these fixed decks. It reproduces Sam's
held-out label as top-1 on 58.47% of rows, but the aggregate is dominated by
placement 7 and does not imply equivalent construction or defense. The 26
fixture results expose the same gap directly, especially in the non-forced
defensive catalog.

This report records evidence only. Manual browser comparison remains the next
step selected by the user.

## Manual-test runtime

- Browser: `http://127.0.0.1:4173`
- API log: `output-sam-policy-api-001`
- Frontend log: `output-sam-policy-frontend-001`
- Standalone start: `make preview-policy`
- Nested Sam comparison: `make preview`

The runtime uses the existing `.local/dracula.sqlite3` database with narration
disabled. At launch it retained 25 game sessions and 1,610 public events. The
continuous corpus remained healthy at 20 decks, 655,320 rows, and four active
collection workers.

# Sam-128 dataset-miner readiness

Date: 2026-07-24
Prepared corpus: `runs/sam-128-million-corpus-001`
Progress log: `output-sam-128-million-corpus-001`

## State

The Sam-128 teacher, recursive miner, sealed artifacts, deterministic recovery,
and process-level parallelism are implemented and mechanically validated
against the active documented contract. Collection has not started.

The configured child count is a maximum. Placements one through six create
four children. Placement seven exposes two legal strategic actions and creates
both. The resulting counts remain exactly 5,461 labeled states per round and
32,766 per deck.

New collections also seal the Git revision and a deterministic digest of the
collection-relevant source and design tree. Mining and resume reject source
drift before search begins.

## Verification

The complete Python suite passed:

```text
537 passed in 613.65 seconds
```

This includes:

- Engine lifecycle, legality, scoring, tie resolution, conservation, and
  deterministic replay.
- Engine-model bridge, Queen/King transforms, action mapping, and
  information-state invariance.
- Every authoritative destination-symmetry pattern and unlisted-state
  fallback.
- Frozen nested 32×32 Sam golden behavior.
- Sam-128 grouping, 128×128 accounting, exact engine terminal values,
  information isolation, deterministic seeds, fair coins, and forced moves.
- Recursive branch counts, input immutability, legal actions, and all-Sam
  six-round trunk consumption.
- Packed rows, schema/configuration rejection, cache validation, atomic shards,
  manifests, privacy, fixture splits, and canonical ordering.
- Sequential/parallel corpus equivalence, worker thread limits, interruption,
  sealed-subtree resume, and clean-run digest equivalence.

Additional checks:

```text
dependency check:              clean
package modules imported:      47
wheel build:                   passed
wheel SHA-256:
  511ba4ab6f4d4d6a38801a4b695e0f0cdc70eace2c6912120e518356442b7bcd
internal documentation links:  57 passed across 14 files
diff-format check:             passed
existing benchmark artifacts:  33 sealed artifacts revalidated
existing structural corpus:    32,766 rows reverified
```

## Contract audit

| Requirement | Result |
| --- | --- |
| Four children when available | Placements 1–6 create four; placement 7 creates both of its two legal strategic actions |
| Three alternatives | Exactly three distinct, teacher-excluding alternatives at placements 1–6; one at placement 7 |
| Uniform sampling | Partial Fisher-Yates over canonical groups using unbiased `randbelow`; 4,096-seed audit preserved distinctness/exclusion and exercised every available group |
| Counts | 5,461 examples and 8,192 leaves per round; 32,766 examples and 49,152 leaves per deck |
| Sam-128 budget | 128 outer × 128 actor-response simulations; configuration digest `20933f828ed85af78b58b39439ea10bb43a9ebe7ecc82b8bf351cdcebe2a68f2` |
| Symmetry | Only the seven exact occupied-position patterns in the authoritative table; all other destinations remain independent |
| Saved label | The row group index and representative exactly match the validated Sam-128 cache selection |
| Training target | One selected strategic group only; no visits, action values, return, shallow-response label, or value target |
| Privacy | Exact row fields contain only packed player-visible input, legal groups, the selected label, public collection identity, and contract digests |
| Recovery | Interrupted and resumed runs reproduce clean corpus digests; partial subtrees never become committed artifacts |
| Parallelism | Worker count is excluded from corpus identity; one- and multi-worker tests produce identical rows, actions, shards, splits, and manifests |
| Source identity | Resolved configuration seals Git HEAD plus the digest of `NORTHSTARS`, `pyproject.toml`, `src/**/*.py`, and `docs/**/*.md`; mining and resume reject drift |
| Historical preservation | 772 pre-existing run files retained identical path, size, and modification metadata; SQLite main, WAL, and SHM files retained identical SHA-256 hashes |

The alternative-sampling audit observed an expected inclusion count of
1,755.43 per available group over 4,096 derived seeds. Observed counts ranged
from 1,717 to 1,836, with a maximum relative deviation of 4.59%. The exact
uniformity property comes from the unbiased partial Fisher-Yates algorithm;
the repeated-seed measurement is a distribution smoke check.

## Real-teacher replay

A fresh production prefix was collected at placement one:

```text
path:
  .local/sam-128-readiness-prefix-001
configuration digest:
  fec99e5865407c5968e3cdd3a02f7cf90707d8b3dab2cad9ba8eeeb1d0f22613
teacher digest:
  20933f828ed85af78b58b39439ea10bb43a9ebe7ecc82b8bf351cdcebe2a68f2
rows:                    1
frontier children:       4
legal strategic groups:  8
selected group index:    4
selected representative: 17
cold elapsed:            140.618 seconds
cached replay elapsed:     0.006 seconds
content digest:
  5e92df5e940799f71ceaf1199654535ae4a3e89319d5f3b81c674c2a1bf8cf17
file digest:
  33edbfcf11e9d45afd22463d59faef81039bbed012551702630014cd9a278e11
```

The replay artifact matched the original byte-for-byte. Its label matched the
sealed Sam-128 result, and its exact field set passed the privacy validator.

## Measured capacity

The benchmark selected four workers as the fastest measured configuration:

```text
four-worker speedup:       2.02× over one worker
projected throughput:      1,985 examples/hour
projected 5,461-row round: 2 h 45 min
projected 32,766-row deck: 16 h 30 min
projected million rows:    21.0 days
peak benchmark RSS:        822 MB
peak worker RSS:           197 MB
observed peak swap growth: 183 MB
```

Six workers were slower and used 1.18 GB peak total RSS. The machine reported
no thermal warning, and deterministic repeat timings showed a median 0.8%
slowdown rather than a throughput collapse.

The prepared 31-deck corpus contains 1,015,746 expected examples. At the
placement-weighted four-worker rate:

```text
projected collection time: 21.3 days
logical artifact bytes:    2.73 GB
allocated filesystem use:  6.01 GB
currently available disk:  118 GiB
```

These are extrapolations from fixed-state teacher timings and a complete
deterministic-stub structural traversal. Cache reuse can reduce actual search
work; a different distribution of reached states can change it.

## Prepared corpus

```text
directory:
  runs/sam-128-million-corpus-001
run ID:
  sam-128-million-corpus-001
root seed:
  sam-128-million-corpus-root-v1-2026-07-24
configuration digest:
  555910ecb31d1df5f9a163d2714c244575c77b73370095a63196a8225a0d99c2
branch configuration digest:
  e55be9755bc622310b1392c60cbd9f969e4e87b0b155997cfac20be10c473c33
Git revision:
  ec6e3d395bc89e9b0cb4bf31eb15438119281c60
source-tree digest:
  19d5132ae69a2ce0e57ecb30335e27453a486016d6b860ad2bab905b6296d786
workers:             4
children k:          4
training decks:     27 =   884,682 examples
validation decks:    2 =    65,532 examples
test decks:          2 =    65,532 examples
total decks:        31 = 1,015,746 examples
```

All 31 fixture IDs are unique and the three deck-level splits are disjoint.
The state is `initialized`. There is no corpus manifest, deck, round, subtree,
or teacher-cache artifact in this directory; collection has not started.

Resolved artifacts:

```text
resolved-config.json SHA-256:
  ebcc8209651ed6b6733be73e87dcc681559ed7080ab7b29125ec095c7344b131
fixture-splits.json SHA-256:
  89e6677f6c8d99a88df33628d0b00030d19b78a14a918662da8de1928d5411d9
state.json SHA-256:
  31d50c0014da77c61e32aef37a6529a66dd1526967d7501af4fa3cf5b032ecee
```

## Commands

The initialize command has already been run. It is included for exact
reconstruction and must not be rerun against the prepared directory:

```bash
PYTHONPATH=src .venv/bin/python -m dracula.sam_miner initialize \
  --output runs/sam-128-million-corpus-001 \
  --run-id sam-128-million-corpus-001 \
  --root-seed sam-128-million-corpus-root-v1-2026-07-24 \
  --training-decks 27 \
  --validation-decks 2 \
  --test-decks 2 \
  --child-count 4 \
  --workers 4 \
  --teacher-profile sam-128
```

Start:

```bash
nohup /bin/zsh -c 'trap - INT; exec env PYTHONPATH=src PYTHONUNBUFFERED=1 .venv/bin/python -m dracula.sam_miner mine --output runs/sam-128-million-corpus-001' \
  > output-sam-128-million-corpus-001 2>&1 < /dev/null &
echo $! > .local/sam-128-million-corpus-001.pid
```

Monitor every five seconds:

```bash
while :; do
  date
  printf 'sealed decks: '
  find runs/sam-128-million-corpus-001/decks -name deck-manifest.json 2>/dev/null | wc -l
  printf 'sealed rounds, including worker staging: '
  find runs/sam-128-million-corpus-001/decks runs/sam-128-million-corpus-001/.workers \
    -name round-manifest.json 2>/dev/null | wc -l
  printf 'sealed subtrees, including worker staging: '
  find runs/sam-128-million-corpus-001/decks runs/sam-128-million-corpus-001/.workers \
    -name 'subtree-*.json' 2>/dev/null | wc -l
  tail -n 20 output-sam-128-million-corpus-001
  sleep 5
done
```

Stop cleanly:

```bash
kill -INT "$(cat .local/sam-128-million-corpus-001.pid)"
while kill -0 "$(cat .local/sam-128-million-corpus-001.pid)" 2>/dev/null; do
  sleep 1
done
```

Resume:

```bash
nohup /bin/zsh -c 'trap - INT; exec env PYTHONPATH=src PYTHONUNBUFFERED=1 .venv/bin/python -m dracula.sam_miner resume --output runs/sam-128-million-corpus-001' \
  >> output-sam-128-million-corpus-001 2>&1 < /dev/null &
echo $! > .local/sam-128-million-corpus-001.pid
```

Inspect, verify, and summarize after the corpus manifest seals:

```bash
PYTHONPATH=src .venv/bin/python -m dracula.sam_miner inspect \
  --output runs/sam-128-million-corpus-001

PYTHONPATH=src .venv/bin/python -m dracula.sam_miner verify \
  --output runs/sam-128-million-corpus-001

PYTHONPATH=src .venv/bin/python -m dracula.sam_miner summarize \
  --output runs/sam-128-million-corpus-001
```

## Resolved clarifications

- The child count is a maximum: four children when at least four strategic
  actions exist, otherwise every legal strategic action. Placement seven
  therefore has two children without changing the documented corpus counts.
- Newly initialized collections seal both Git revision and exact
  collection-relevant source-tree digest. Older unbound artifacts remain
  inspectable and verifiable, but cannot resume collection.
- Coarse CLI logging is understood operational behavior, not a readiness gap.

No collection authorization is inferred by this report.

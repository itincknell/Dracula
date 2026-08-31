# Sam dataset miner

This document preserves the historical recursively branched Sam-32
selected-action corpus contract. The active balanced visit-target contract is
[BGC dataset miner](bgc-dataset-miner.md). Search mechanics remain in
[information-set search](search.md), and the player-visible input remains in the
[engine–opponent contract](engine-model-contract.md).

## Controllers

**Sam** is the nested actor-local UCT controller used for local play:

```text
outer simulations:           32
actor-response simulations:  32
outer exploration:           sqrt(2)
response exploration:        sqrt(2)
terminal evaluation:         exact normalized round differential
```

**Sam-32** is the dataset teacher. It uses the same 32-outer and 32-response
nested actor-local algorithm and information boundary selected through manual
play. Sam-32 adds the authoritative destination symmetry to the historical
live Sam implementation; this does not retroactively change that controller.

Sam-32 operates on strategic actions. One strategic action is one remaining
hand card paired with one destination group from the authoritative early-turn
symmetry table in
[information-set search](search.md#authoritative-early-turn-symmetry).
Different hand cards always remain different actions. Unlisted board patterns
retain independent destinations.

At every search node, group visits and returns are pooled across paired
destinations. Selection uses maximum visits, then maximum mean terminal return,
then canonical representative action index. The existing derived fair coin
selects one concrete member after its strategic group is selected.

## Configurable branching

`k` is the maximum number of children created at a non-forced state. The
default benchmark configuration is:

```text
k = 4
teacher children = 1
uniform alternative children = k - 1 = 3
```

The miner creates `min(k, legal_group_count)` children. Placements one through
six expose at least four legal strategic actions and therefore create four
children under the default configuration. Placement seven has one remaining
hand card and two empty coffin cells, so it has exactly two legal strategic
actions and creates both children. `k` is a positive integer recorded in the
immutable resolved configuration and every cache, shard, and manifest digest.

For every non-forced decision state, the miner:

1. Constructs the acting player's established `SearchInformationState`.
2. Runs Sam-32 once and saves one example whose label is its selected
   strategic group.
3. Applies the Sam-32 group to create child zero, the teacher child.
4. Samples `min(k, legal_group_count) - 1` other strategic groups uniformly
   without replacement after excluding the teacher group.
5. Applies each sampled group to create the remaining children, up to `k - 1`
   alternatives.
6. Recursively processes every child through the remaining learned placements.

Alternative actions create states for later Sam-32 queries. They never become
labels for their parent state. A forced eighth placement creates no example and
one engine continuation.

Branching terminates when the engine completes the current round. The path that
selects child zero at every node is the all-Sam trunk. Only its terminal state
advances the seeded deck to the next round. Alternative branches end with the
round in which they were created. This repeats for all six rounds.

For a round with seven learned placements and placement-seven arity `m`:

```text
examples(k) =
    7                              when k = 1
    (k^7 - 1) / (k - 1)           when k > 1

placement-seven children m = min(k, 2)
terminal leaves(k) = k^6 * m
examples per deck(k) = 6 * examples(k)
```

The default `k = 4` benchmark therefore produces exactly:

```text
placement 1 examples:      1
placement 2 examples:      4
placement 3 examples:      16
placement 4 examples:      64
placement 5 examples:      256
placement 6 examples:      1,024
placement 7 examples:      4,096
round examples:            5,461
round terminal leaves:      8,192
six-round deck examples:   32,766
```

## Group and child identity

Legal concrete actions retain the fixed 32-action order. A strategic group is:

```text
representative_action_index
member_action_indexes       # canonical ascending order, length one or two
```

Groups sort by representative action index. The teacher label is the index of
the selected group in that canonical list.

A branch path is the ordered sequence:

```text
round number
placement number
actor
parent path digest
child ordinal               # zero is teacher; 1..k-1 are alternatives
strategic representative
concrete action index
```

The path digest is SHA-256 over canonical JSON containing the branch schema
version and those fields. It identifies collection structure without entering
the model input.

For child zero, the concrete action is Sam-32's deterministic
`sam-128-root-result` fair-coin choice. The cache retains that choice with the
selected group. For an alternative paired group, concrete selection uses the
miner-derived fair coin containing the information-state fingerprint, group
representative, round, placement, parent path digest, child ordinal, and branch
configuration digest. Concrete selection cannot alter group selection.

## Deterministic seeds

All streams use the derivation format in the
[engine–opponent contract](engine-model-contract.md#deterministic-seeds-and-shuffle).

| Namespace | Components after namespace |
| --- | --- |
| `dracula-sam-miner-deck-v1` | Corpus root, split, deck index |
| `dracula-sam-128-request-v1` | Information-state fingerprint, Sam-32 configuration digest; the stable internal namespace predates the budget change |
| `dracula-sam-miner-alternatives-v1` | Corpus root, deck fixture ID, round, placement, parent path digest, information-state fingerprint, `k` |
| `dracula-sam-miner-destination-v1` | Alternative branch configuration digest, information-state fingerprint, group representative, round, placement, parent path digest, child ordinal |
| `dracula-sam-miner-shuffle-v1` | Corpus root, split, epoch |

Alternative sampling starts from the canonical legal-group list after removing
the teacher group. A SHA-256 counter stream performs a partial Fisher-Yates
shuffle and takes its first `k - 1` groups. Repeating a resolved collection
reproduces every group, concrete action, branch, and row.

## Source identity

Every newly initialized run seals:

```text
Git HEAD revision
source-tree schema version
source-tree SHA-256 digest
```

The tree digest covers `NORTHSTARS`, `pyproject.toml`, every Python file under
`src/`, and every Markdown contract under `docs/`. It therefore binds both the
committed base revision and relevant uncommitted source or contract changes.
The collection configuration digest transitively carries that identity into
cache, row, shard, and manifest bindings.

Mining and resume recompute the identity from the source checkout that launches
the collector and fail before search when either the revision or tree digest
differs. The active continuous run launches from a private frozen checkout
containing exact copies of every identity input. The mutable main repository is
not on its worker `PYTHONPATH`.

Legacy artifacts without source identity remain inspectable and verifiable, but
cannot resume collection.

## Dataset row

Each non-forced state produces:

```text
dataset_schema_version
collection_configuration_digest
teacher_schema_version
teacher_configuration_digest
branch_schema_version
branch_configuration_digest
observation_schema_version
action_schema_version
symmetry_schema_version
information_state_fingerprint
observation                    packed bool[875]
legal_mask                     packed bool[4, 8]
strategic_groups               canonical member action indexes
teacher_group_index
teacher_group_representative
round_number
placement_number               1 through 7
player                         Queen or King
dealer                         Queen or King
deck_fixture_id
branch_path_digest
```

The row contains no visit counts, return estimates, round-return target,
determinizations, public history, authoritative state, opponent hand, stock
order, game seed, search tree, model state, or policy hidden state.

## Teacher cache

The content-addressed teacher cache key contains:

```text
information_state_fingerprint
teacher schema version
Sam-32 configuration digest
symmetry schema version
action schema version
```

The cached value contains the canonical legal group partition, selected group,
and Sam-32's selected concrete member for the teacher child. A cache entry
validates its key, group partition, selected legal group, selected legal
concrete member, and content digest before use.

Live Sam, historical Sam-128, and Sam-32 corpus caches cannot mix. Shallow
Teacher v2, response-ranker, root-visit, policy/value, and PPO artifacts are
incompatible with this dataset contract.

## Fixtures and splits

Continuous collection repeats a deterministic split cycle containing 27
training decks, two validation decks, and two test decks. Those values are
cycle weights, not terminal corpus sizes. Each split derives deck fixtures
under its own split component in `dracula-sam-miner-deck-v1`; indexes increase
without a configured endpoint. A fixture ID appears in exactly one split.

Rows inherit their deck's split. Splits never occur by randomly assigning rows
from one branch tree.

The active continuous run resolves:

```text
run ID:                  sam-32-continuous-corpus-001
root seed:               sam-32-continuous-corpus-root-v1-2026-07-24
teacher:                 Sam-32
outer/response budgets:  32 / 32
k:                       4
workers:                 4
split cycle:             27 training, 2 validation, 2 test
termination:             manual interrupt or disk guard
minimum free disk:       1 GiB
in-flight reserve:       256 MiB per worker
```

## Shards, manifests, and resume

Each round has one root-row shard and `k` first-child subtree shards. A subtree
contains all examples below one placement-one child. Rows sort by placement and
branch-path digest. Each shard records its schema and configuration digests,
row count, placement counts, canonical content digest, and envelope file
digest.

A round manifest seals only after its root shard and all `k` subtree shards
validate. It records the all-Sam terminal fingerprint needed to reproduce the
next deal. A deck manifest seals only after all six round manifests validate.
After each complete worker batch, the corpus manifest atomically advances to
include every contiguous deck ordinal and sorts entries by split and deck
index.

Writes use a temporary sibling, flush and validate their content, then rename
atomically. Interruption discards only the unsealed subtree. Resume reconstructs
that subtree from the deck fixture and branch path; it never reads a serialized
authoritative engine state.

Complete seeded decks are the only process-parallel work unit. The resolved
configuration accepts one through eight workers; worker count is an operational
setting and is excluded from corpus identity. Each worker:

- uses one PyTorch intra-op thread and one inter-op thread;
- owns a deterministic fixture-specific staging tree and cache;
- runs the unchanged sequential round and subtree miner; and
- polls one parent-owned stop marker at existing interruption boundaries.

Workers do not create nested pools, Python search threads, or merged UCT trees.
The parent validates each completed deck and every content-addressed cache entry,
merges equal cache entries with conflict detection, then atomically renames the
sealed deck into the corpus. The corpus manifest always sorts decks by split and
fixture index, independent of worker completion order. Interrupted staging trees
remain available for exact resume from their sealed subtree boundaries; no
partial deck enters the corpus.

Collection continues until interrupted or the disk guard stops it. The active
guard preserves at least 1 GiB of free disk and reserves another 256 MiB for
each configured worker before starting a batch. Free space is polled while a
batch runs; a low-disk signal stops workers at existing subtree interruption
boundaries. Only previously sealed decks enter the corpus manifest. Resume
reuses sealed worker subtrees and continues from the first missing deck
ordinal.

The active process runs from the frozen checkout recorded alongside its PID and
resolved configuration under `.local/collector-source/`. Stopping and resuming
must use that checkout until this corpus ends. Editing the main source tree does
not alter an active or newly spawned worker.

The `smoke` command seals a configurable prefix ending at placements one
through six. It uses the same query, cache, label, branch, and fair-coin path as
full mining but stops before any round can complete.

## Training snapshots

The standalone trainer consumes a versioned snapshot of the committed,
gap-free deck prefix rather than the moving corpus manifest. A snapshot
references sealed deck manifests and their hashes; it copies no row and changes
no corpus artifact or original split label.

For the first committed-data experiment, the two highest included deck ordinals
form a validation overlay and every earlier ordinal forms training. Decks are
indivisible. The full snapshot contract is in
[standalone Sam-32 policy training](model-training.md#committed-corpus-snapshot).

## Supervised target

The standalone classifier produces 32 raw logits arranged as four hand slots
by eight non-center destinations. The sealed row retains the full engine-legal
mask and exact strategic groups. Training reconstructs a separate
representative-action mask from the authoritative symmetry table.

For every legal hand card, a listed symmetry pattern retains only the
designated proxy destination for each group. Non-proxy paired members are
masked. Unlisted patterns retain every engine-legal action independently.
Sam-32's selected group is represented by its designated proxy action `a*`:

```text
L_policy = cross_entropy(
    raw_policy_logits masked to legal representative actions,
    a*,
)
```

There is no mirrored-logit averaging or pooling. After standalone argmax
selects a proxy action, the existing derived fair coin chooses the concrete
member of a paired group. The concrete result does not enter the target.

Training uses no visit distribution, action-value or score estimate, return,
value head, value loss, PPO objective, recurrence, critic, policy population,
hybrid search, or automatic promotion. The exact model and loss are defined in
[standalone Sam-32 policy model](neural-model.md).

## Verification

The miner must establish:

- Exactly `min(k, legal_group_count)` children: `k` at placements one through
  six and both legal children at placement seven, with one teacher child and
  distinct uniform alternatives.
- Exact formulas and default `k = 4` counts above.
- Exact Sam-32 labels, strategic grouping, legality, and concrete fair-coin
  resolution.
- Identical rows and manifests under repeated runs, resume, and supported worker
  counts.
- Deck-level split isolation.
- Cache and artifact contract rejection across controller versions.
- Input immutability and exact six-round stock consumption on the all-Sam
  trunk.
- Player-view invariance under authoritative hidden-card substitutions.
- Absence of private search and engine data from sealed rows.

Corpus collection and training remain separate phases. A sealed corpus prefix
may be inspected and verified while collection continues; training uses only a
user-selected verified prefix.

# BGC balanced dataset miner

This document defines the active turn-balanced corpus produced by the
128-outer belief-greedy controller. Search mechanics and information isolation
are defined in [information-set search](search.md); observation and action
schemas remain in the [engine–opponent contract](engine-model-contract.md).

## Teacher and coverage

```text
outer UCT simulations:       128
belief completions/response: 8
workers:                      4 complete-game processes
rows/game:                    42
rows/placement/game:          6 at each learned placement 1-7
forced placement 8:           transition only, no row
```

Complete games cycle through teacher, one-deviation, two-deviation, mixed, and
alternative trajectory profiles. Nonteacher actions are sampled uniformly
from legal strategic groups excluding the teacher group. This preserves equal
placement counts while exposing the teacher to states outside its own preferred
continuations.

## Row target

Every frozen D0 source row contains:

- Packed player-visible `bool[875]` observation.
- Packed full engine-legal `bool[4,8]` mask.
- Exact strategic groups and designated representatives.
- Exact root visit count for every group, summing to 128.
- Most-visited selected representative for audit and trajectory advancement.
- Round, placement, actor, dealer, fixture, trajectory profile, and schema
  identity.

The exact eligible prefix is then physically migrated once for neural
training. Migration removes the redundant `4 × 54` stable-slot block, producing
a packed `bool[659]` observation. Current hand cards are placed in temporary
candidate rows in canonical card-ID order, and masks, groups, representatives,
visits, and the audit selection are relabeled from stable engine slots to those
candidate rows. The migration preserves all 128 visits and every complete-game
split assignment. It writes a separately versioned, independently verified
corpus rather than mutating D0.

Training maps each group count to its designated representative action and
normalizes by 128. Illegal and non-proxy actions receive zero target mass. The
model sets illegal and non-proxy logits to negative infinity before
log-softmax, so those actions receive zero probability. The loss is
distributional policy cross-entropy:

```text
pi(a) = N(a) / 128
L = -sum(pi(a) * log(p(a)))
```

The selected group is not converted into a one-hot target. It remains only for
audit, agreement metrics, and trajectory advancement. The dataset contains no
action values, returns, scores, determinizations, hidden hands, stock order, or
search tree. Training adds no illegal-action penalty, legality task,
temperature transform, label smoothing, value target, or auxiliary objective.

## First snapshot eligibility

The first `D0` source snapshot sealed the exact gap-free prefix of 11,905
complete games and 500,010 rows without modifying collection. Training reads
only its separately sealed card-set migration, never the moving source corpus.

The five consecutive trajectory profiles form one indivisible split block. A
versioned deterministic assignment places 2,143 blocks in training and 238 in
validation:

```text
training:   10,715 games, 450,030 rows, 64,290 per placement
validation:  1,190 games,  49,980 rows,  7,140 per placement
```

Game and fixture identities are disjoint across the overlays. A snapshot
command must refuse to seal if the committed prefix, block completeness,
digests, totals, placement balance, or fixture isolation is not exact.

## Persistence and recovery

Each complete game seals atomically. A cumulative content-addressed manifest
lists a gap-free game prefix in canonical ordinal order. The immutable resolved
configuration binds the root seed, search digest, worker count, disk floor,
Git revision, and exact source-tree digest.

D0 stopped after sealing its active worker batch when it received `SIGINT`.
Its final 12,080-game, 507,360-row manifest verifies. The immutable training
snapshot references only the contracted first 11,905 games. The existing
resume path can still verify the resolved configuration, every committed game
digest, row schema, visit total, placement balance, and manifest totals, but D0
is not currently collecting.

The stopped selected-action pilot uses version-one row and manifest schemas.
It remains readable historical evidence but cannot mix with this visit-target
corpus.

## Completed D1 collection

The separately versioned D1 collector used the user-selected `pi0` bundle.
Initialization and every resume verified that bundle's model, training, source,
symmetry, mask, and selection digests before committing output.

D1 retains the D0 outer UCT budget, exact engine transitions and round scoring,
strategic grouping, visit targets, five trajectory profiles, 42-row game
balance, four complete-game workers, atomic game commitment, and disk guard.
Only simulated non-forced continuation responses change from eight-completion
belief-greedy evaluation to one accepted-policy inference. The base response
controller remains available only through its explicit D0 or evaluation path.

D1 uses independent dataset, row, game-shard, search, response, cache,
manifest, resolved-configuration, and report schemas. D1 writes the compact
`bool[659]` card-set observation and candidate-row action encoding directly;
it does not recreate the redundant stable-slot tensor. Every row, game, and
manifest binds the accepted checkpoint, acceptance manifest, D1 controller,
and base outer-search digests. A verifier rejects D0 rows or manifests and D1
artifacts belonging to another accepted policy. D0 is never opened for writing
by the D1 command.

The retained template is
[`configs/bgc-d1-template.toml`](../configs/bgc-d1-template.toml). The completed
collection used:

```bash
dracula-bgc-pi0-miner initialize \
  --output runs/bgc-pi0-d1-corpus-001 \
  --accepted-pi0 runs/pi0-accepted-001 \
  --root-seed bgc-pi0-d1-corpus-001-root-v1 \
  --workers 4 --minimum-free-disk-gib 1
dracula-bgc-pi0-miner continuous \
  --output runs/bgc-pi0-d1-corpus-001 \
  --accepted-pi0 runs/pi0-accepted-001 \
  --root-seed bgc-pi0-d1-corpus-001-root-v1 \
  --workers 4 --minimum-free-disk-gib 1
dracula-bgc-pi0-miner inspect \
  --output runs/bgc-pi0-d1-corpus-001
dracula-bgc-pi0-miner verify \
  --output runs/bgc-pi0-d1-corpus-001 \
  --accepted-pi0 runs/pi0-accepted-001
```

`SIGINT` stops after the current complete worker batch seals. Reissuing the
same `continuous` command verifies the gap-free prefix and resumes at the next
game ordinal. D1 is now stopped and supplied the immutable `pi1` training
snapshot; it is not part of deployment.

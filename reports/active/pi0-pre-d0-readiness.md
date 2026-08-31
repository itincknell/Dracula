# π0 pre-D0-finish readiness

Audit time: 2026-08-23 20:48 EDT.

## Verdict

The machinery is ready to seal the contractual D0 snapshot and run the π0
training, evaluation, conditional acceptance, and D1 sequence when D0 reaches
11,905 committed games. D0 is still collecting normally. No real snapshot,
training run, candidate evaluation, accepted checkpoint, or D1 corpus was
created during this review.

Two operational defects found by the disposable command rehearsal were fixed:

- Validation no longer rewrites a sealed relative run path to a different
  canonical spelling before checking the immutable resolved configuration.
- A D1 command without a valid accepted π0 now exits with code 2 and the concise
  contractual refusal instead of an internal traceback.

Both repairs have regression tests.

## Contract and implementation review

The active path is mechanically bound to these decisions:

- Input is the player-visible `bool[875]` observation; the model contains
  exactly 738,569 trainable parameters and returns 32 raw policy logits.
- Each version-two BGC row contains group visits summing to 128. Loading maps
  those visits to authoritative proxy actions and divides by 128.
- The full engine mask and reconstructed representative mask are both
  verified. Illegal and non-proxy logits become negative infinity before
  log-softmax and therefore have zero probability and zero gradient.
- Optimization uses only distributional policy cross-entropy. The active
  trainer contains no selected-action one-hot, illegal-action, value, return,
  PPO, recurrent, or critic objective.
- D0 snapshotting accepts only the gap-free first 11,905 games arranged as
  2,381 complete five-profile blocks. The deterministic overlay assigns 2,143
  blocks to training and 238 to validation.
- Random legal and eight-completion one-ply belief-greedy controls receive only
  the acting player's information state. Paired evaluation uses both roles and
  the deterministic 2,000-resample 95% interval.
- A comparison extends from 60 to 180 deck seeds only when its primary point
  estimate is positive and its interval includes zero.
- Acceptance is atomic and requires all three contractual comparisons to pass.
  Failed, inconclusive, stale, incomplete, forged, or interrupted evidence
  creates no accepted directory.
- The accepted-policy adapter changes only the simulated continuation response
  in BGC-128. Outer UCT, 128 visits, actor-local information, deterministic
  seeds, exact engine transitions, exact terminal scoring, forced moves, and
  backup remain unchanged.
- D1 has independent row, dataset, game, cache, search, response, report,
  resolved-configuration, and manifest schemas. Every artifact binds the
  accepted checkpoint and acceptance manifest. D0/D1 mixing is rejected.

## Verification evidence

Commands and results:

```text
nice -n 10 .venv/bin/pytest -q
630 passed in 1275.12s

nice -n 10 .venv/bin/pytest -q \
  tests/test_bgc_policy_training.py \
  tests/test_bgc_policy_evaluation.py \
  tests/test_bgc_pi0_miner.py
25 passed in 33.88s after the two repairs

.venv/bin/python -m pip check
No broken requirements found.

.venv/bin/python -m pip wheel --no-deps . -w /tmp/dracula-readiness-wheel-20260823
Wheel built successfully.

.venv/bin/python -m compileall -q src tests
Passed.
```

The focused pre-repair readiness selection passed 215 tests. The complete run
then covered all engine, scoring, bridge, information-state, symmetry, search,
miner, model, API, transaction, privacy, recovery, and historical invariance
suites. The affected training and D1 suites were rerun after repair.

Package entry-point help was verified for `dracula-bgc-policy-training`,
`dracula-bgc-policy-evaluation`, `dracula-bgc-pi0-miner`, and
`dracula-belief-greedy-miner`. Fifty-one Markdown files passed path and heading
anchor validation. Tracker IDs remain confined to the design tracker; active
stale-terminology and diff-format checks passed.

Serialized D0 games and the disposable D1 corpus were scanned for private
hands, stock order, engine seeds, determinizations, search trees, model state,
action values, and returns. None were present. Public-response privacy and API
transaction/retry behavior also passed the complete test suite.

## Disposable command rehearsal

The rehearsal used a disposable version-two corpus under
`/tmp/dracula-pi0-readiness-20260823-v1` with ten complete games, 420 rows, all
seven placements, both roles, both dealer states, every trajectory profile,
and a five-game training/five-game validation split.

It verified:

- Eligibility inspection and production snapshot refusal at 420 rows.
- Refusal left no partial snapshot directory.
- A custom ten-game test snapshot verified every row and digest.
- Training interrupted after one sealed minibatch and resumed through the CLI.
- Validation and exported inference succeeded after resume.
- Synthetic production-shaped comparison evidence exercised pass, fail,
  inconclusive, and extension mechanics in tests.
- A synthetic passing bundle sealed atomically and initialized an empty,
  separately versioned D1 corpus; D1 inspect and verify reproduced its digests.
- The real training command refused because the real snapshot is absent.
- The real D1 command refused with `D1 requires a valid accepted pi0 bundle`;
  it left no output directory.

The synthetic data and synthetic passing evidence are mechanical fixtures only.
They are not π0 competence evidence and live outside the repository.

## D0 health

At the audit time:

```text
PID:                         1507
worker processes:            4 active, each compute-bound
committed games:             692
committed rows:              29,064
rows at each placement:      4,152
remaining games:             11,213
remaining rows:              470,946
corpus manifest digest:      62f72c2b1ae55c28b5585fa7120938233939f7a7e870373d0529d80ae502bb85
resolved configuration:      8f50986137c38d003385ad8b4dfec56162c5e5a721c4f93a1bb2cb06c0d3bb1a
source revision:             ec6e3d395bc89e9b0cb4bf31eb15438119281c60
source-tree digest:          b9e4be610583b9f461b29432bede49303593a06a30946b2b0ffc3f30d7c687ba
corpus disk use:             24 MiB
free disk:                   about 80 GiB
system memory free:          44%
system swap currently used:  about 7.68 GiB
```

The large existing swap allocation reflects system-wide history and did not
prevent normal collection. The four current workers used about 0.47 GiB total
RSS at the sampled point, no pages were throttled, and batches continued to
seal. The recent sustained projection was about 211,855 rows/day. At that rate,
snapshot eligibility is about 53.3 hours away; a practical current estimate is
52–55 hours if the machine remains similarly loaded.

The frozen checkout independently reproduced the sealed revision and tree
digest. Worker file handles point into
`.local/collector-source/bgc-128-visit-corpus-001`, and the recovery command in
that checkout's `README.md` contains no stale source path.

## Future execution sequence

Keep the main source tree unchanged from training through evaluation and
acceptance because artifacts bind its exact source-tree digest.

### 1. Recheck D0 eligibility

```bash
.venv/bin/dracula-bgc-policy-training eligibility \
  --corpus runs/bgc-128-visit-corpus-001
```

### 2. Seal the immutable D0 snapshot

```bash
.venv/bin/dracula-bgc-policy-training snapshot \
  --corpus runs/bgc-128-visit-corpus-001 \
  --output runs/bgc-policy-d0-snapshot-001
```

### 3. Inspect and verify the snapshot

`inspect` performs full manifest, row, schema, split, digest, and privacy
verification.

```bash
.venv/bin/dracula-bgc-policy-training inspect \
  --snapshot runs/bgc-policy-d0-snapshot-001/snapshot.json
```

### 4. Train π0

```bash
.venv/bin/dracula-bgc-policy-training train \
  --config configs/bgc-policy-pi0.toml \
  2>&1 | tee output-bgc-policy-pi0
```

### 5. Resume interrupted training

```bash
.venv/bin/dracula-bgc-policy-training resume \
  --run runs/bgc-policy-pi0 \
  2>&1 | tee -a output-bgc-policy-pi0
```

### 6. Validate and verify the selected unaccepted checkpoint

```bash
.venv/bin/dracula-bgc-policy-training validate \
  --run runs/bgc-policy-pi0
.venv/bin/dracula-bgc-policy-training export \
  --run runs/bgc-policy-pi0
```

The candidate path is
`runs/bgc-policy-pi0/artifacts/unaccepted-candidate.pt`.

### 7. Run both standalone comparisons

```bash
mkdir -p runs/pi0-evaluation-001
.venv/bin/dracula-bgc-policy-evaluation fixtures \
  --root-seed pi0-evaluation-root-v1 \
  --output runs/pi0-evaluation-001/fixtures.json
.venv/bin/dracula-bgc-policy-evaluation standalone \
  --candidate runs/bgc-policy-pi0/artifacts/unaccepted-candidate.pt \
  --fixtures runs/pi0-evaluation-001/fixtures.json \
  --control random \
  --output runs/pi0-evaluation-001/standalone-random-legal.json
.venv/bin/dracula-bgc-policy-evaluation standalone \
  --candidate runs/bgc-policy-pi0/artifacts/unaccepted-candidate.pt \
  --fixtures runs/pi0-evaluation-001/fixtures.json \
  --control one-ply \
  --output runs/pi0-evaluation-001/standalone-one-ply-belief-greedy.json
```

### 8. Compare base BGC-128 with π0-BGC-128

```bash
.venv/bin/dracula-bgc-policy-evaluation controller \
  --candidate runs/bgc-policy-pi0/artifacts/unaccepted-candidate.pt \
  --fixtures runs/pi0-evaluation-001/fixtures.json \
  --output runs/pi0-evaluation-001/pi0-bgc-vs-base-bgc.json
```

### 9. Seal acceptance only on complete success

```bash
.venv/bin/dracula-bgc-policy-evaluation evidence \
  --candidate runs/bgc-policy-pi0/artifacts/unaccepted-candidate.pt \
  --fixtures runs/pi0-evaluation-001/fixtures.json \
  --random-report runs/pi0-evaluation-001/standalone-random-legal.json \
  --one-ply-report runs/pi0-evaluation-001/standalone-one-ply-belief-greedy.json \
  --controller-report runs/pi0-evaluation-001/pi0-bgc-vs-base-bgc.json \
  --output runs/pi0-evaluation-001/validation-evidence.json
.venv/bin/dracula-bgc-policy-evaluation accept \
  --candidate runs/bgc-policy-pi0/artifacts/unaccepted-candidate.pt \
  --fixtures runs/pi0-evaluation-001/fixtures.json \
  --evidence runs/pi0-evaluation-001/validation-evidence.json \
  --output runs/pi0-accepted-001
```

A failed or inconclusive result prints that status and leaves
`runs/pi0-accepted-001` absent.

### 10. Initialize D1

```bash
.venv/bin/dracula-bgc-pi0-miner initialize \
  --output runs/bgc-pi0-d1-corpus-001 \
  --accepted-pi0 runs/pi0-accepted-001 \
  --root-seed bgc-pi0-d1-corpus-001-root-v1 \
  --workers 4 --minimum-free-disk-gib 1
```

### 11. Start, monitor, stop, resume, inspect, and verify D1

Start:

```bash
nohup .venv/bin/dracula-bgc-pi0-miner continuous \
  --output runs/bgc-pi0-d1-corpus-001 \
  --accepted-pi0 runs/pi0-accepted-001 \
  --root-seed bgc-pi0-d1-corpus-001-root-v1 \
  --workers 4 --minimum-free-disk-gib 1 \
  >> output-bgc-pi0-d1-corpus-001 2>&1 &
echo $! > .local/bgc-pi0-d1-corpus-001.pid
```

Monitor and inspect:

```bash
tail -f output-bgc-pi0-d1-corpus-001
.venv/bin/dracula-bgc-pi0-miner inspect \
  --output runs/bgc-pi0-d1-corpus-001
```

Stop at a sealed worker-batch boundary:

```bash
kill -INT "$(cat .local/bgc-pi0-d1-corpus-001.pid)"
```

Resume by reissuing the exact `nohup ... continuous` command above. Verify:

```bash
.venv/bin/dracula-bgc-pi0-miner verify \
  --output runs/bgc-pi0-d1-corpus-001 \
  --accepted-pi0 runs/pi0-accepted-001
```

## Remaining external sequence

Only moving-corpus work and real evidence remain: D0 must continue sealing its
contractual prefix, then the exact snapshot, training, evaluation, conditional
acceptance, and D1 commands above can run. The implementation does not presume
that π0 will pass; it preserves an unaccepted result when it does not.

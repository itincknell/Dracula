# Teacher v2 training readiness

## Verdict

The collection, recovery, privacy, versioning, and supervised-training
mechanics are ready. Full collection is not ready because root visit counts do
not produce an informative policy target in early play at 32, 64, or 128
simulations.

The manually approved 32×4 controller can remain the gameplay reference. The
blocking issue concerns distillation: the stored target is its root visit
distribution, not its selected action or response values.

## Prepared full run

The reserved paths are fresh:

```text
dataset       runs/teacher-v2-001
training      runs/teacher-v2-training-001
collection log output-teacher-v2
training log output-teacher-v2-training
training config configs/archive/teacher-v2/teacher-v2-training.toml
```

The proposed collection contains 216 training games and 24 validation games,
uses four complete-game workers, and excludes absolute-evaluation fixtures.
At 42 learned decisions per game it would seal:

```text
training examples   9,072
validation examples 1,008
total examples     10,080
outer simulations 322,560
```

The expected collection configuration digest is
`dcb4f657ea48b9a4daed194a6b9b9dd72992498f73d64becbccaf3763571fa38`.
The collector writes and validates its immutable resolved configuration before
the first game. The supervised trainer resolves
`configs/archive/teacher-v2/teacher-v2-training.toml` against the sealed dataset digest before
optimization.

## Capacity

The ten-game clean smoke run sustained 2,102 examples per hour. The four-worker
benchmark sustained 1,866 examples per hour on a shorter cold fixture set.
These rates project 4.8 to 5.4 hours before sustained thermal effects. A
realistic unattended range is 5 to 7 hours.

Four workers peaked at 847 MiB total RSS and released transient swap by the end
of the benchmark. Six workers retained 200 MiB of additional swap and increased
aggregate search time per game by 50 percent, so four remains the selected
count. CPU training peaked at 406 MiB and was 5.29 times faster than MPS.

The ten-game dataset occupies 704 KiB. Linear projection gives about 16.5 MiB
for 240 shards and manifests. Training checkpoints, metrics, reports, and the
export are about 15 MiB at the current model size. Reserve 50 MiB for the
completed dataset and training run, or 100 MiB including temporary and
diagnostic headroom.

## Evidence

- Interrupted collection resumes only from sealed game boundaries and
  reproduced the clean dataset digest exactly.
- Version 1 caches, shards, and manifests are rejected by format, dataset,
  search, response, configuration, approval, and profile bindings.
- Training and validation fixtures are disjoint.
- Twenty-four validation games provide 1,008 rows and balance both roles. This
  is adequate for first-pass held-out checkpoint selection, but not for a
  strength claim; absolute evaluation remains separate.
- The smoke trainer updated the shared body and both heads. Value validation
  MSE fell from `0.183446` to `0.034385`; policy cross-entropy moved only from
  `2.210354` to `2.205634`.
- The policy result matches the target-quality finding. At 32 visits,
  normalized visit entropy was `99.13%` early and `99.43%` middle, with
  top-two margins of `0.21%` and `0.64%`. Early top-action agreement across
  request seeds was only `30%`.

The fixed-state follow-up tested 64 and 128 outer visits. At 128, early
normalized entropy remained `99.71%`, the top-two margin fell to `0.23%`, and
top-action agreement reached only `40%`. The full results are in
[Teacher v2 visit-budget smoke](teacher-v2-visit-budget-smoke.md).

The next bounded experiment should evaluate the root action-score estimates
that Teacher v2 already computes. Stable score rankings can supply a more
faithful policy target than visits. Unstable score rankings require more or
aggregated action evaluations rather than more UCT visits.

## Prepared commands

These commands are prepared but the collection command must remain withheld
until the visit-target gate passes.

```bash
nohup .venv/bin/dracula-teacher full \
  --output runs/teacher-v2-001 \
  --run-id teacher-v2-001 \
  --root-seed dracula-teacher-v2-full-v1 \
  --simulation-budget 32 \
  --training-games 216 \
  --validation-games 24 \
  --absolute-fixtures 0 \
  --workers 4 \
  > output-teacher-v2 2>&1 & echo $! > .local/teacher-v2.pid
```

```bash
nohup .venv/bin/dracula-teacher resume \
  --output runs/teacher-v2-001 \
  >> output-teacher-v2 2>&1 & echo $! > .local/teacher-v2.pid
```

```bash
.venv/bin/dracula-teacher inspect --output runs/teacher-v2-001
```

```bash
nohup .venv/bin/dracula-supervised train \
  --config configs/archive/teacher-v2/teacher-v2-training.toml \
  > output-teacher-v2-training 2>&1 & echo $! > .local/teacher-v2-training.pid
```

```bash
nohup .venv/bin/dracula-supervised resume \
  --output runs/teacher-v2-training-001 \
  >> output-teacher-v2-training 2>&1 & \
  echo $! > .local/teacher-v2-training.pid
```

```bash
.venv/bin/dracula-supervised validate \
  --output runs/teacher-v2-training-001 \
  --checkpoint runs/teacher-v2-training-001/checkpoints/best-validation.pt
```

```bash
.venv/bin/dracula-supervised export \
  --output runs/teacher-v2-training-001 \
  --checkpoint runs/teacher-v2-training-001/checkpoints/best-validation.pt \
  --destination runs/teacher-v2-training-001/teacher-v2-warm-start.pt
```

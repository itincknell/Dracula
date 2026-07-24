# Search and model training

This document defines local teacher collection, policy/value optimization, and
single-model expert iteration. Application state and HTTP behavior belong in
[architecture](architecture.md).

## Retired experiment

The five-policy PPO experiment did not establish strategic competence. Its
artifacts under `runs/training-001` through `runs/training-004` remain immutable
comparison evidence. PPO, the separate critic, critic burn-in, recurrence,
population training, replacement, and population-relative selection are not
part of the active trainer.

The validated 500-simulation search configuration in
[information-set search](search.md#validation-result) remains a frozen baseline.
The user approved the 32-outer, four-completion Teacher v2 profile after direct
browser comparison on July 23, 2026. This approval authorizes teacher
collection despite the preserved failed automated gate report. The collector's
version, privacy, and resume tests pass. Full policy/value collection is
deferred while a bounded experiment tests whether the policy head can replace
Teacher v2's inner response enumeration without weakening play.

## Deterministic namespaces

All seeds use the derivation and formatting in the
[engine–opponent contract](engine-model-contract.md#deterministic-seeds-and-shuffle).

| Namespace | Components after namespace |
| --- | --- |
| `dracula-search-fixture-v1` | Run root, phase, split, fixture index |
| `dracula-search-request-v1` | Fixture ID, information-state digest, player, round, turn, search-config digest |
| `dracula-search-determinization-v1` | Search-request digest, simulation index |
| `dracula-search-opponent-v1` | Search-request digest, simulation index, rollout ply |
| `dracula-search-expansion-v1` | Search-request digest, simulation index, node digest |
| `dracula-strategic-search-request-v2` | Fixture ID, root information-state digest, player, round, turn, v2 configuration digest |
| `dracula-strategic-search-determinization-v2` | Strategic-search request digest, outer simulation index |
| `dracula-strategic-search-expansion-v2` | Strategic-search request digest, outer simulation index, root node digest |
| `dracula-greedy-response-request-v1` | Actor information-state digest, response-configuration digest |
| `dracula-greedy-response-determinization-v1` | Response-request digest, completion index |
| `dracula-greedy-response-rollout-v1` | Response-request digest, completion index, candidate action, rollout ply, acting information-state digest |
| `dracula-self-play-action-v1` | Run root, iteration, fixture ID, round, placement |
| `dracula-model-initialization-v2` | Run root, model ID, initialization ordinal |
| `dracula-replay-sampling-v1` | Run root, iteration, epoch, batch, row |
| `dracula-absolute-evaluation-v1` | Evaluation suite, fixture index, controller pair, role assignment |

`phase` is `teacher`, `expert`, or `evaluation`; `split` is `training`,
`validation`, or `absolute`. Indexes are zero-based unsigned decimal values.
Every artifact stores the full derived digest and all stable components needed
to reproduce it.

## Planned initial teacher dataset

The initial dataset contains **240 independent six-round games**:

```text
216 training games   -> 9,072 examples
 24 validation games -> 1,008 examples
240 total games      -> 10,080 examples
```

One game supplies 42 non-forced decisions: 21 from Queen and 21 from King.
Every game therefore balances roles without replaying the same deck. Training
and validation game seeds use distinct split components under
`dracula-search-fixture-v1`. Absolute-evaluation seeds use a separate namespace
and never enter teacher collection or replay.

At every non-forced decision, the approved Teacher v2 configuration runs 32
outer simulations with UCT constant `sqrt(2)`. Every non-forced opponent action
and every root rollout action after expansion uses the shallow greedy response
policy from that actor's information state. It compares every legal action over
the approved number of shared hidden-card samples and uniform round
completions, then selects the maximum mean exact actor-relative differential.
The outer root visit distribution is not an active policy target. The complete
240-game policy/value corpus remains deferred until the bounded response
experiment is resolved.

The 32-visit smoke corpus found nearly uniform early- and middle-round visit
targets because mandatory legal-action coverage consumes much of the outer
budget. Manual approval establishes the controller's playing strength; it does
not establish that 32 visits encode a useful distillation target. Full
collection requires a fixed-state budget measurement showing materially lower
target entropy, wider top-two visit margins, and stable top actions before the
collection profile is sealed.

Fixed-state measurements at 64 and 128 visits did not satisfy that gate. At
128, early-round normalized entropy remained `99.71%`, the mean top-two margin
was `0.23%`, and the top action agreed across only `40%` of request seeds. Root
visit counts are therefore not the active policy target for full collection.

## Bounded response-ranking experiment

An optional observer captures one example at each unique shallow-response cache
miss without changing search. Each example contains the actor's model-visible
`bool[875]` observation, `bool[4,8]` legal mask, exact strategic groups, every
group's four-completion mean terminal differential, selected group, fair-coin
concrete action, placement, role, dealer status, configuration digests, and
response-cache identity.

The observer never receives a determinization, authoritative opponent hand,
stock order, engine seed, outer sampled state, tree, policy hidden state,
logits, or model data. Observer-disabled and observer-enabled searches must
produce identical results and configuration behavior.

The current model's shared body and policy head produce 32 concrete logits.
Group logits are arithmetic means over their concrete members. Training uses
all non-tied group pairs:

```text
weight(i, j) = abs(q_i - q_j)
loss(i, j) = weight(i, j)
             * softplus(-sign(q_i - q_j) * (logit_i - logit_j))
```

The batch loss is normalized by total pair weight. Exact ties have zero weight.
Inference ranks legal groups by model score and then canonical representative
index; paired concrete destinations retain the existing derived fair coin. The
value head, root visits, softmax temperature, and round-return regression are
outside this experiment.

The first six-game response dataset smoke passed the instrumentation,
invariance, privacy, resume, and reproducibility checks. Its evidence is in the
[response-dataset report](../reports/teacher-v2-response-dataset-smoke.md).
Model training remains a separate next step.

### Deferred full-collection mechanics

The dataset manifest records the Teacher v2 search and response schemas, search
and response configuration digests, validation-report digest, and manual
approval identity. Version 1 results, other v2 profiles, and mixed teacher
contracts cannot enter the initial dataset.

Self-play samples the accepted real move from visit counts at placements one
through four and uses maximum visits at placements five through seven. The
eighth placement is forced. Derived action seeds make the complete dataset
reproducible.

Teacher collection uses four CPU worker processes, selected by the
[M3 parallelism benchmark](../reports/teacher-v2-parallelism.md). Each worker
owns complete games and search trees and uses one PyTorch intra-op and inter-op
thread. No mutable engine, model, or random-stream state crosses workers. A
completed game is written as one atomic CPU tensor shard. A partial game is
discarded and reproduced from its fixture seed. The sealed dataset manifest
sorts shards by fixture index and records every content digest.

The active collection command is `dracula-teacher`. Its `smoke` and `full`
subcommands create a new run, `resume` continues the immutable resolved
configuration, and `inspect` validates manifests and shards before reporting
their totals. Decision caches contain only information-state digests, visits,
selected actions, and timing; they are removed after their full-game shard is
sealed.

The active optimizer command is `dracula-supervised`. `smoke` and `train` read
the same strict TOML contract; run size is determined by the sealed dataset and
epoch controls. `resume`, `validate`, and `export` operate from an immutable
resolved run directory. The full and smoke configurations are
`configs/search-warmstart.toml` and `configs/search-warmstart-smoke.toml`.

The expert-iteration command is `dracula-expert`. `smoke` and `run` create a
cycle; `resume` restarts its current atomic phase; `evaluate` resumes absolute
comparison; `accept` and `reject` record the manual decision; and `export`
copies the currently accepted artifact after validation. The full and smoke
configurations are `configs/expert.toml` and `configs/expert-smoke.toml`.

## Deferred full-dataset row

Each training example contains:

```text
observation:             bool[875]
legal_mask:              bool[4, 8]
search_policy:           float32[32]
selected_action_index:   int64
round_return:            float32
information_state_digest
search_config_digest
fixture ID, split, player, dealer, round, placement
contract versions
```

`search_policy` sums to one over legal actions and is zero elsewhere.
`round_return` is in `[-1, 1]`. The sealed row contains no full information
state, public history, determinization, opponent hand, stock order, engine seed,
tree node, or model diagnostic. Forced placements remain in the game replay but
not in the training tensor shard.

## Optimization

The model and loss are defined in [neural model](neural-model.md). Training uses:

```text
optimizer:          AdamW
learning rate:      3e-4, constant
betas:              (0.9, 0.999)
epsilon:            1e-8
weight decay:       1e-4
batch size:         256
global grad norm:   1.0
maximum epochs:     50
minimum epochs:     8
early-stop patience:5 completed epochs
minimum improvement:1e-4 in validation total loss
```

There is no learning-rate scheduler, dropout, mixed precision, gradient
accumulation, or data-loader subprocess in the first implementation. Tensors
remain on CPU and only the active minibatch moves to the optimization device.
Forward, loss, and optimizer arithmetic use `float32`.

One epoch draws 9,072 training examples. For the warm start this is one
deterministic shuffle without replacement. During expert iteration, the sampler
chooses a retained source shard uniformly and then a row uniformly from that
shard, with replacement, using `dracula-replay-sampling-v1`. Validation evaluates
every retained validation row without replacement.

The best checkpoint is the epoch with the lowest validation
`policy_cross_entropy + value_mse`. Training stops after the minimum epoch when
five consecutive completed epochs fail to improve that metric by `1e-4`, or
after epoch 50. The candidate is the best checkpoint, not necessarily the final
epoch. Its optimizer state is restored from that same epoch. Gameplay
acceptance remains a separate manual decision based on absolute evaluation.

## Device selection

Search and dataset collection run on CPU. Optimization accepts `cpu`, `mps`, or
`auto`. Before the first full run, `auto` executes the same three warm-up and
five timed smoke epochs on CPU and MPS. It selects MPS only when:

- Every output, loss, and gradient is finite.
- CPU/MPS validation outputs agree within `1e-5` absolute and `1e-4` relative
  tolerance before optimization.
- Median MPS epoch time is at least 10% lower than CPU.
- Process resident memory remains below 6 GiB and swap grows by no more than
  512 MiB during the benchmark.

Otherwise `auto` resolves to CPU. The resolved device is immutable for a run.
CPU is the bit-reproducible reference; MPS resume preserves fixtures and batch
order but uses the documented numerical tolerances.

## Expert iteration

The default run performs at most 20 iterations with one accepted network. Each
iteration begins from an immutable accepted model and its optimizer state:

1. Freeze the accepted checkpoint for collection.
2. Generate 120 independent guided-search games: 108 training and 12
   validation games.
3. Use the 100-simulation full-round PUCT profile in
   [information-set search](search.md#neural-guided-search) at every non-forced
   real decision.
4. Seal 4,536 training and 504 validation examples with exact round returns.
5. Optimize a candidate against the bounded replay window.
6. Evaluate the best-validation candidate against permanent controls and the
   accepted checkpoint.
7. Retain the candidate and report for manual acceptance or rejection.

Rejection restores the iteration-start model and optimizer exactly. Acceptance
makes the candidate the sole current model for the next iteration. Collection
never mixes model versions within a game.

## Replay window

The original teacher training and validation shards remain permanent. Replay
also retains the five most recent **accepted** guided-search iterations. Rejected
candidate data does not enter later replay. Source-uniform sampling prevents a
large shard from dominating and keeps the teacher represented as the policy
changes.

Examples are immutable after sealing. A fixture ID may appear in only one split
and one source iteration. Dataset manifests reject duplicate fixture IDs,
overlapping split digests, incompatible schemas, illegal target mass, and
non-finite values.

## Absolute evaluation and manual acceptance

Permanent controls are:

- Uniform random legal play without a neural model.
- Archived `policy-2-v20` using its recorded `argmax-v1` profile.
- Frozen 500-simulation search-only play.
- The manually approved 32×4 Teacher v2 configuration.
- The previously accepted policy/value checkpoint after the warm start.
- The strategic fixture suite.

Random and PPO control comparisons retain the established minimum of 12
independent decks with both controller-role assignments. Guided-search
non-inferiority comparisons use 60 independent decks with both assignments:
120 games against the approved Teacher v2 configuration and 120 against the
previous accepted checkpoint. Version 1 remains in the fixed control matrix.
Every controller sees identical decks and receives its own valid information
state.

A report marks a candidate eligible for manual acceptance only when:

- It passes every strategic fixture.
- The paired 95% lower confidence bound for victory-percentage difference is
  above zero against random legal play and `policy-2-v20`.
- The paired 95% lower bound is above zero against the frozen version 1 search
  and at least `-0.05` against Teacher v2 and the previous accepted checkpoint.
- No Queen, King, dealer, or non-dealer split loses more than 10 percentage
  points relative to the accepted checkpoint.
- Mean guided-search decision latency is at most 80% of Teacher v2's latency on
  the paired games.
- All privacy, legality, finite-output, and latency checks pass.

Eligibility does not promote a model. The report includes game and round wins,
score differential, role splits, paired intervals, strategic fixtures,
decision latency, simulations per second, and peak memory. The user accepts or
rejects the candidate explicitly.

## Locked training sequence

1. Implement shallow-response Teacher v2 without modifying the version 1
   planner or evidence.
2. Run the defensive, constructive, privacy, absolute-performance, latency,
   and memory gates, preserve the failed result, then expose 32×4 for local
   human comparison.
3. Record the explicit manual approval of the 32×4 profile while retaining the
   failed automated report.
4. Capture unique Teacher v2 response examples and test the policy head as a
   pairwise response ranker; do not use the rejected 32-, 64-, or 128-visit
   distributions.
5. Compare the hybrid controller against 32×4 Teacher v2 at identical outer
   budgets. Continue to a full corpus only if it reduces cost without material
   competence loss.
6. Warm-start one model with the fixed optimizer and early-stopping contract.
7. Evaluate the best-validation checkpoint against the permanent controls and
   accept or reject it manually. Expert iteration requires an accepted warm
   start.
8. For each accepted-model iteration, collect 120 games with 100-simulation
   full-round PUCT, update from the bounded replay window, and evaluate the
   best-validation candidate.
9. Continue only after explicit acceptance. Keep exact terminal scoring until
   the value-cutoff gate passes independently.
10. Compare version 1, Teacher v2, guided search, and standalone inference before
   choosing a deployment profile.

## Configuration

The supervised optimizer and expert iterator each read one TOML file and
resolve it to an immutable JSON manifest. Unknown or missing keys and values
outside these contracts fail before artifacts are written. The expert
configuration additionally fixes the accepted artifact, replay bounds,
collection profile, permanent controls, and every evaluation sample size.

```toml
[run]
run_id = "warmstart-001"
root_seed = "replace-me"
output_directory = "runs/warmstart-001"
source_revision = "git-revision"

[dataset]
teacher_directory = "runs/search-teacher-v2-001"
search_report_digest = "0000000000000000000000000000000000000000000000000000000000000000"

[model]
model_id = "dracula-policy-value"
initialization_ordinal = 0

[optimization]
device = "auto"
batch_size = 256
learning_rate = 0.0003
betas = [0.9, 0.999]
epsilon = 1e-8
weight_decay = 0.0001
gradient_norm = 1.0
minimum_epochs = 8
maximum_epochs = 50
early_stop_patience = 5
minimum_improvement = 0.0001
```

The complete expert defaults are recorded in `configs/expert.toml`: 108
training games, 12 validation games, four collection workers, 100 simulations,
five accepted replay iterations, 9,072 replay draws per epoch, and the fixed
12-pair or 60-pair control comparisons. `configs/expert-smoke.toml` reduces
only workload sizes and is never acceptance evidence.

## Artifacts and recovery

```text
runs/<run-id>/
  resolved-config.json
  state.json
  device-benchmark.json
  checkpoints/
    iteration-start.pt
    epoch-latest.pt
    best-validation.pt
    final.pt
  metrics/<epoch>.json
  metrics/summary.json
  reports/<epoch>.md
  reports/summary.md
```

An expert run adds an immutable `accepted.json` pointer and one directory per
iteration containing sealed collection manifests, the iteration-start and
candidate checkpoints, a versioned policy/value artifact, evaluation records,
and the concise evaluation report. Rejected iteration data remains evidence
but is absent from later replay manifests.

Files are written beside their destination with a temporary suffix, flushed,
validated, and atomically renamed. `state.json` advances only after the new
artifact and its digest validate.

- Collection resumes from the last sealed complete-game shard. A partial game
  restarts from its fixture seed.
- Optimization resumes from `epoch-latest.pt`, which contains model, optimizer,
  epoch, deterministic sampler position, resolved device, and best-validation
  state.
- Evaluation resumes from the immutable final checkpoint and skips completed fixture
  pairs whose result hashes validate.
- Acceptance writes a new immutable accepted pointer; rejection restores the
  iteration-start pointer. Neither mutates prior checkpoints.

An interrupted phase cannot expose a partial dataset, checkpoint, report, or
accepted model. Resume on CPU must reproduce fixtures, samples, batches,
parameters, and metrics exactly.

## Readiness

The suite is ready for sustained expert iteration when:

- Model tensor, parameter, initialization, loss, and device tests pass.
- Approved Teacher v2 collection seals and resumes without privacy or split
  violations.
- One optimization smoke run improves held-out fit and resumes exactly on CPU.
- Guided PUCT reproduces fixed visits and actions and retains exact terminal
  scoring.
- One complete iteration survives interruption at collection, optimization,
  and evaluation boundaries.
- A sustained local trial stays within the memory and swap limits.
- Absolute evaluation can reject a weaker candidate despite lower training
  loss.

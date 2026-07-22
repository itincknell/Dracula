# Search and model training

This document defines the local path from search-only validation to a single
search-guided policy/value model. It does not participate in live game state or
HTTP request handling.

## Evidence from the retired approach

The PPO system remains implemented and its artifacts remain under `runs/` as
historical evidence. It is not the active training architecture.

The final `training-004` tournament used 46,080 games and 3,072 games per
matchup. The leading checkpoint, `policy-2-v20`, achieved 50.8% against uniform
random legal play; the other candidates ranged from 48.0% to 51.7%. Adjacent
candidate ordering was unresolved at the configured 95% threshold. In two
recorded human games, `policy-2-v20` lost both, lost ten of twelve rounds, and
finished an average 173 points behind.

These results show that relative population ranking and critic improvement did
not establish strategic competence. PPO, the separate critic, critic burn-in,
five-policy collection, recurrent trajectory replay, and population-relative
selection are retired from the active plan.

## Stage 1: search-only validation

The [information-set search](search.md) reaches the end of the current round and
uses exact engine scoring. Its tactical and absolute-control gates passed on
July 22, 2026.

The required work is ordered:

1. Build the information-state projection, public move record, uniform
   determinization sampler, and privacy fixtures.
2. Build deterministic POMCP-style root-sampled UCT with a uniform legal
   opponent model and private diagnostics.
3. Verify late-round decisions against exhaustive enumeration and run tactical
   behavior fixtures.
4. Benchmark 100, 500, and 2,000 simulations per move on the target laptop.
5. Freeze one viable search configuration and compare it with uniform random
   legal play and the archived `policy-2-v20` candidate on fixed role-balanced
   fixtures.

Search-guided data generation may begin after the neural architecture and
training configuration are finalized. The search validation result is recorded
in [information-set search](search.md#validation-result).

## Deterministic namespaces

All seeds use the derivation and formatting defined in the
[engine–opponent contract](engine-model-contract.md#deterministic-seeds-and-shuffle).
The active namespaces are:

| Namespace | Components after namespace |
| --- | --- |
| `dracula-search-fixture-v1` | Suite ID, lane root, game counter |
| `dracula-search-request-v1` | Fixture ID, information-state digest, player, round, turn, search-config digest |
| `dracula-search-determinization-v1` | Search-request digest, simulation index |
| `dracula-search-opponent-v1` | Search-request digest, simulation index, rollout ply |
| `dracula-search-expansion-v1` | Search-request digest, simulation index, node digest |
| `dracula-model-initialization-v2` | Run root, model ID, initialization ordinal |
| `dracula-expert-self-play-v1` | Run root, iteration, lane root, game counter |
| `dracula-dataset-shuffle-v1` | Run root, dataset digest, training epoch |
| `dracula-absolute-evaluation-v1` | Evaluation suite, fixture ID, controller ID, role |

Stable identifiers, zero-based counters, full digests, and resolved namespace
versions are stored in each manifest. Retry reuses the same identifiers and
therefore reproduces the same CPU result.

## Search fixtures and absolute controls

Evaluation fixtures are disjoint from search development and neural training.
Every controller uses the same deck fixtures, plays both roles, and receives the
same simulation budget and deterministic tie-breaking where applicable.

The permanent absolute controls are:

- Uniform random legal play implemented without a neural model.
- The archived `policy-2-v20` PPO candidate under its recorded `argmax-v1`
  profile.
- The frozen search-only configuration that passes the first gate, once one
  exists.

Primary strength is victory percentage:

```text
(wins + 0.5 * ties) / completed_games
```

Reports also include cumulative score differential, round-score differential,
Queen and King splits, dealer splits, worst-control performance, paired fixture
bootstrap intervals, latency, and peak memory. A search candidate beats a
control only when the paired 95% bootstrap lower bound for the
victory-percentage difference is above zero across at least 12 independent deck
seeds and 24 role-balanced games. Resampling uses the deck seed as the block;
the fixed sample has no interim stopping.

Tactical fixtures are separate from rank. They include exact or exhaustively
verified positions for completing a suit or color multiplier, blocking a visible
construction, exploiting an opponent card, retaining flexible destinations,
and avoiding a Vampire-destroyed line. Search diagnostics must show the root
action visits and representative continuations that support each result.

## Stage 2: expert iteration

After every search-only gate passes, train one shared policy/value network as
defined in [neural model](neural-model.md).

One expert-iteration cycle is:

1. Freeze the current model version and search configuration.
2. Generate complete self-play games. At every non-forced real decision, run
   information-set search from the acting player's view.
3. Select the real move from root visit counts using the configured self-play
   temperature. Search simulations use only information-safe opponent inputs.
4. Attach the exact normalized round return after each round completes.
5. Seal `(information state, legal mask, visit distribution, return)` examples
   into a replay window.
6. Train the shared model by policy cross-entropy and value mean squared error.
7. Evaluate the new checkpoint against all permanent controls and the prior
   accepted checkpoint on the fixed absolute fixture suite.
8. Accept or reject the checkpoint manually from the complete report.

There is one current model, not a population. Self-play uses the latest accepted
model for both roles with separate information states. The model may provide
search priors and an opponent rollout policy only after an ablation shows that
each use improves strength or latency. Full-round terminal scoring remains the
search value until a value-cutoff comparison passes.

## Replay window and splits

Examples are grouped by complete game fixture and round. A fixture belongs to
exactly one of training, validation, or absolute evaluation. The initial replay
window retains a configurable number of the most recent accepted iterations and
samples iterations uniformly before sampling rows, preventing a large recent
iteration from silently replacing all earlier data.

The sealed dataset contains only model-visible information and targets. Search
trees, determinizations, authoritative hidden state, and stock order remain in
private diagnostic artifacts and never enter model rows.

Validation reports:

- Policy cross-entropy and top-one agreement with search.
- Value MSE and mean absolute error against exact normalized round return.
- Calibration by predicted-value bucket.
- Metrics split by role, dealer, decision depth, multiplier outcome, and
  Vampire presence.

Loss metrics diagnose approximation; only absolute gameplay controls determine
whether a checkpoint is stronger.

## Local compute

The target machine is an M3 MacBook Air with 8 GB unified memory. Python 3.12
and PyTorch remain the local stack.

- Engine simulation and tree search run on CPU.
- Search processes one decision at a time initially and releases its tree after
  the decision artifact is sealed.
- Model optimization uses configurable `cpu`, `mps`, or `auto`; both paths use
  the same implementation.
- Dataset tensors remain on CPU and only the active minibatch moves to the
  optimization device.
- `float32`, eager execution, and an in-process loader are the initial defaults.
- Batch size changes only between runs after measuring memory and swap.

The benchmark records decisions per second, simulations per second, engine
placements per second, search-node count, process memory, MPS allocation, swap
growth, thermal slowdown, and finite model outputs and gradients. `auto` selects
MPS only after its sustained epoch throughput exceeds CPU without unsupported
operations or unsafe memory behavior.

## Configuration

A run reads one TOML file and writes an immutable resolved JSON manifest. Search
and neural phases use separate commands but share fixture and version sections.

```toml
[run]
run_id = "search-001"
root_seed = "..."
output_directory = "../runs"

[search]
simulation_budget = 500
exploration_constant = 1.0
opponent_model = "uniform-legal-v1"
decision = "max-visits-v1"

[fixtures]
development_lane_roots = []
evaluation_lane_roots = []
games_per_lane = 0

[compute]
search_device = "cpu"
optimization_device = "auto"
model_batch_size = 256
```

The resolved configuration records all seed namespaces, contract versions,
fixture roots, controller digests, search limits, device resolution, and source
revision. Unknown keys and contradictory settings fail at startup.

## Artifacts and recovery

Search-only runs use:

```text
runs/<run-id>/
  resolved-config.json
  state.json
  decisions/<fixture>/<turn>.json
  metrics/<phase>.json
  reports/search-validation.md
```

Expert-iteration runs add:

```text
  checkpoints/<iteration>/iteration-start.pt
  checkpoints/<iteration>/candidate.pt
  datasets/<iteration>.pt
  reports/<iteration>.md
```

Artifacts are written to a temporary name, validated, and atomically renamed
before `state.json` advances. A partial search decision restarts from its
deterministic request seed. A partial dataset collection restarts the current
fixture. A training interruption reloads the iteration-start checkpoint and
sealed dataset. An evaluation interruption reuses the immutable candidate and
fixture schedule.

No partial artifact becomes committed state. Resumption must reproduce fixture
IDs, search decisions, dataset digests, and CPU metrics without manual repair.

## Manual selection

Checkpoint acceptance is manual. The report compares the candidate with random
legal, `policy-2-v20`, frozen search-only play, and the prior accepted neural
checkpoint. A regression against any permanent control blocks acceptance unless
the report identifies a predeclared latency-strength tradeoff.

The project deploys one opponent at one difficulty. Checkpoints exist to retain
evidence and recover from regression, not to maintain a training population.

## Deployment gate

Deployment design remains open until these measurements exist:

- Search-only strength and per-move latency at the three benchmark budgets.
- Guided-search strength and latency after expert iteration.
- Standalone-network strength and inference latency.
- Serialized model size and the memory required by one concurrent search.

Those results determine whether production runs search, guided search, or the
network alone. No cloud inference product, request shape, or concurrency tier is
selected in advance.

## Verification

The suite is ready for sustained expert iteration only when:

- Search privacy, determinization, action mapping, terminal payoff, replay, and
  exhaustive late-round tests pass.
- Search-only tactical and absolute-control gates pass.
- Dataset splits prevent fixture overlap and sealed rows contain no hidden data.
- One smoke expert-iteration cycle changes model parameters, improves fit on a
  held-out dataset, and resumes from every phase boundary.
- A full local cycle completes within measured memory and swap limits.
- Absolute evaluation can reject a weaker candidate even when its training loss
  improves.

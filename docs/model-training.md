# Model training

This document defines local self-play training and model comparison for the
recurrent opponent policy. It does not participate in live game state or web
request handling.

## Self-play system

The training environment runs the authoritative pure Python engine and its
[engine–model contract](engine-model-contract.md). A collection window has five
active policy versions and twelve lanes. Self matches are excluded.

Each lane `o` owns a distinct root seed. For non-negative game counter `g`, its
game seed is the SHA-256 digest of the UTF-8 sequence
`dracula-fixture-v1`, that lane's root seed, and the decimal `g`, separated by
zero bytes. The lane passes that seed to `create_game`. The lane's root seed
separates it from every other lane; incrementing `g` starts a new full game in
that lane.

One collection generation is the twelve lane games at one game counter. For
each lane, every unordered pair from the five active policies plays one
six-round game from that lane's seed. There are therefore:

```text
C(5, 2) × 12 = 120 engine games per collection generation
```

For an ordered pair `(A, B)`, where `A` sorts before `B` by stable policy ID,
`A` is Queen when `((o + g) % 2) == 0`; otherwise `B` is Queen. Every pair
therefore plays six Queen and six King games in one collection generation. Each
player is dealer for three rounds and non-dealer for three rounds in every game.

The collector derives a sampling seed for each learned action from the match
fixture ID, learner policy version, player role, round number, and recurrent
step index. The deck and action samples are consequently reproducible without
exposing either seed to a policy.

Each actor update accumulates four complete collection generations before model
weights change. A five-model population therefore produces 480 engine games per
update window. One policy plays four opponents in each of twelve lanes, yielding
48 player-game trajectories and 1,008 actor-loss samples per generation. Four
generations yield 192 complete trajectories and 4,032 actor-loss samples for
that policy's PPO update.

The fixture record contains collection generation, lane, game counter, lane-root
seed digest, game seed, ordered policy pair and versions, Queen/King assignment,
and sampling-seed derivation version. A match fixture starts both participant
hidden states from the all-zero model state. Hidden state and policy data never
cross from one match fixture to another.

Each simulated player has a separate recurrent hidden state and receives only
its legitimate `PolicyGameView`. The policy is invoked for every turn owned by
that player. On the dealer's fourth turn, the engine applies the unique legal
move and the policy invocation advances hidden state. Each player therefore has
24 ordered recurrent state updates per game: four in each of six rounds.

Twenty-one of those updates are learned decisions: four in each round in which
the player is non-dealer and three in each round in which the player is dealer.
The three dealer-final updates use a false actor-loss mask. Collected transitions
are owned by the learner and exact policy version that generated them. A learned
transition records its observation, legal mask, selected action, old action log
probability, critic estimate, learner and opponent identities and versions,
fixture identity, and eventual round return.

An optimizer batch contains complete, ordered player-game trajectories generated
by one learner policy version. Its members span all twelve lanes and all four
opponents. The policy is replayed from the all-zero game state across all 24
updates, without detaching hidden state at a round boundary. Actor trajectories
remain in their generating policy-version bucket. The shared critic trains from
the aggregate state/return rows collected in the window.

## Learning objective

The player policy and critic are conceptually and operationally separate:

- The recurrent policy produces masked move probabilities and selects actions.
- One shared, player-agnostic critic estimates the expected normalized
  round-local return from a perspective-normalized observation. It trains from
  states collected across all current players, opponents, and seeded fixtures.
- The critic is used only to estimate advantages during training. It is not an
  input to policy action selection and is not deployed with the policy.

The engine supplies one rules-defined round result for both players after the
eighth placement. For player `p` in round `r`, the return is:

```text
R_r,p = (round_score_r,p - round_score_r,other_player) / 150
```

Every learned decision owned by `p` in round `r` receives `R_r,p`. The maximum
possible line and round score is 150, so the return lies in `[-1, 1]`. This
preserves the rules-defined score difference while giving policy and critic
targets a fixed scale.

At collection time, the critic estimate `V_old(observation_t)` is stored with
each learned decision. Its fixed actor advantage is
`A_t = R_r,p - V_old(observation_t)`. The advantage is an observed round result
minus its expected value; it is not a deterministic move score. Critic updates
regress against `R_r,p` independently. Later-round losses can train recurrent
representations through retained hidden state, while an earlier action's policy
term receives only its own round's return.

## Critic model

The critic is an independently parameterized feed-forward nonlinear regressor.
It receives `observation: bool[875]` and produces one unconstrained scalar. It
uses the policy's structured input decomposition with separate weights:

```text
card embedding:       54 × 32
hand slot encoder:    (32 card + 8 slot + 1 occupied) -> 64, for four slots
coffin cell encoder:  (32 card + 8 position + 1 occupied) -> 64, for nine cells
status encoder:       162 -> 64
context encoder:      11 -> 16
fused state:          912 -> 128 -> 64 -> 1
```

The first fused layer uses GELU followed by 128-feature layer normalization;
the second uses GELU; the value head is linear. This model has **143,273
trainable parameters**. It does not share actor weights, use a recurrent state,
or consume the legal mask.

For `N` learned decisions in one learner batch, its loss is:

```text
L_critic = (1 / N) * Σ_t (V(observation_t) - R_round,t)^2
```

Mean squared error makes `V` estimate the conditional expected return. The
round return already has a fixed `[-1, 1]` scale, so critic targets are not
normalized again within a batch.

## Policy update and loss schedule

PPO updates one learner policy version at a time. For every learned action, the
collector stores the masked behavior log probability `log_prob_old,t`. During an
update, the policy replays the complete ordered trajectory from its all-zero
hidden state and calculates `log_prob_new,t` under the current weights:

```text
ratio_t = exp(log_prob_new,t - log_prob_old,t)
advantage_t = R_round,t - V_old(observation_t)
advantage_normalized_t = (advantage_t - mean(advantage)) / (std(advantage) + 1e-8)
L_PPO = -(1 / N) * Σ_t min(
    ratio_t * advantage_normalized_t,
    clip(ratio_t, 0.8, 1.2) * advantage_normalized_t
)
```

The sums in `L_critic` and `L_PPO` include the 21 learned decisions in each
player-game trajectory. The three forced recurrent transitions remain in the
GRU unroll and have a false actor-loss mask. Actor advantages remain fixed for
all four PPO epochs; critic predictions are recomputed for each critic update.

For the same learned steps, let `q_t` be the 32-way softmax of the raw policy
logits before hard legality masking, and let `m_t` be the Boolean legal mask.
The illegal probability and thresholded penalty are:

```text
p_illegal,t = Σ_i q_t,i * (1 - m_t,i)
L_illegal = (1 / N) * Σ_t max(0, p_illegal,t - 0.001)^2
```

Entropy uses the hard-masked policy distribution over legal actions:

```text
H = (1 / N) * Σ_t -Σ_i policy_probability_t,i * log(policy_probability_t,i)
L_policy = actor_weight * L_PPO - entropy_coefficient * H + 1.0 * L_illegal
```

The critic begins on broad random play. `actor_weight` is zero until both of
these conditions hold: at least 10,000 completed training rounds have been
collected, and the critic's mean squared error improves by at least five percent
over a zero-value predictor on three consecutive held-out collection windows.
It then increases linearly from zero to one during the next 10,000 completed
rounds. `entropy_coefficient` is `0.005` through critic burn-in and the actor
ramp, then `0.001` after `actor_weight` reaches one. The illegal penalty and
entropy term remain active throughout burn-in.

The policy optimizer is Adam with learning rate `3e-4`; the critic optimizer is
Adam with learning rate `1e-3`. Both use beta values `(0.9, 0.999)`, epsilon
`1e-8`, and global gradient-norm clipping at `0.5`. Each learner batch receives
four PPO epochs. Each epoch applies both policy and critic updates. It shuffles
complete trajectories and preserves all 24 ordered recurrent steps. The target
laptop starts with 16 player-game trajectories per minibatch; the configured
maximum is 64. An update stops before a further epoch when
`mean(log_prob_old - log_prob_new)` across learned steps reaches `0.015`.

Training samples the hard-masked policy at temperature `1.0`. Exploration comes
from the entropy term; action sampling does not inject separate random moves.

The training environment supplies no manually coded strategic move score. The
deterministic engine supplies legal actions and terminal rules outcomes only.

## Integrated training iteration

One training iteration collects four complete collection generations from a
frozen active population, updates each policy and the shared critic, then runs
one held-out evaluation pass. No policy or critic weight changes while a
collection window is in progress.

### Iteration boundary

At the start of an iteration, the trainer snapshots the five active policy
versions and the critic parameters. These are the behavior policies and
`V_old` critic for the entire window. Let `g_0` be the next game counter in
every lane. The window contains game counters `g_0` through `g_0 + 3`.

For every counter and lane, the trainer creates the ten unordered pairs of the
five policy IDs and assigns Queen with the established `((o + g) % 2) == 0`
rule. It runs the resulting full six-round fixtures using the policy and critic
snapshots. A fixture may execute concurrently with another fixture, provided
that its engine state, two hidden states, sampling stream, and trajectory data
remain independent.

The window has these fixed sizes:

| Quantity | Per policy | Entire population |
| --- | ---: | ---: |
| Complete player-game trajectories | 192 | 960 |
| Recurrent state updates | 4,608 | 23,040 |
| Learned action and critic rows | 4,032 | 20,160 |
| Engine games | — | 480 |

Each policy's 192 trajectories contain all four opponents, all twelve lanes,
and all four game counters. The 20,160 critic rows are the learned rows from
both participants in every engine game. Forced recurrent transitions remain in
the 23,040 state updates but contribute no actor or critic row.

### Fixture collection

For each fixture, the trainer creates the game from its game seed, initializes
one zero hidden state for each participant, and runs the engine to terminal
state. For an active player turn, it performs this sequence:

```text
context = build_policy_turn_context(engine_state, active_player)
policy = behavior_policy[active_player]
hidden_out, raw_logits = policy(context.input, hidden_in[active_player])

if context.kind is learned:
    probabilities = masked_softmax(raw_logits, context.input.legal_mask, T=1.0)
    action = sample(probabilities, fixture-derived sampling seed)
    critic_value = V_old(context.input.observation)
    move = context.action_table[action]
else:
    action = the unique legal action index
    critic_value = absent
    move = context.forced_move

transition = apply_move(engine_state, move)
record the ordered policy transition
engine_state = transition.state
hidden_in[active_player] = hidden_out
```

The recorded transition contains the pre-action input, behavior version,
selected action, masked behavior log probability when learned, and the frozen
critic estimate when learned. The two hidden states remain local to the fixture
and are discarded after its ordered trajectories have been assembled.

When the eighth move produces an `EngineRoundResult`, the trainer calculates
both rules-defined round returns. It attaches each player's return to that
player's learned transitions in the completed round and advances the engine to
the next round. A terminal sixth-round result completes the fixture. The
trainer rejects a fixture if it does not produce exactly 24 ordered transitions
per participant, 21 learned rows per participant, and six completed round
results.

### Batch assembly

After all 480 fixtures finish, the trainer assembles five actor buckets keyed
by their behavior policy version. Each contains 192 complete 24-step
trajectories and 4,032 learned rows. For each bucket, it calculates
`R_round - V_old(observation)` and standardizes those 4,032 advantages once.
The stored behavior log probabilities, frozen critic estimates, returns, and
standardized advantages remain unchanged for all PPO epochs.

The shared critic pool contains the 20,160 learned observation/return pairs
from all five actor buckets. It carries no recurrent state and may mix policy
versions in a regression minibatch.

### Optimization

The trainer performs four optimization passes over the assembled window. In
each pass it first processes every non-halted actor bucket in stable policy-ID
order. It shuffles complete trajectories, partitions them using the configured
trajectory minibatch size, replays each trajectory from its zero hidden state,
and preserves all 24 recurrent steps in order. The policy loss uses the three
loss masks and coefficients defined above. Only the policy represented by that
bucket receives gradients.

After an actor pass, the trainer calculates its approximate KL over all learned
rows in that actor bucket. If it reaches `0.015`, that policy takes no further
PPO passes in this iteration. Its already collected data remains unchanged.

Each optimization pass also performs one shuffled regression pass over the
shared critic pool. Critic minibatches use the learned rows from the same number
of complete player-game trajectories as the actor configuration. They minimize
the stated MSE against the stored round returns. The critic completes all four
regression passes even when one or more actor policies stop early; its updated
predictions do not replace the stored `V_old` values in the actor objective.

After optimization, every active policy receives a new policy version and the
critic receives a new critic version. New collection begins only after all six
updated versions are available. No game, recurrent state, actor trajectory, or
critic target crosses this iteration boundary.

### Evaluation

The trainer then evaluates the five updated policies on the configured held-out
fixture roots. Evaluation uses the same full-game role balancing and
temperature-one masked sampling as collection, with sampling seeds derived from
the held-out fixture identity. It starts fresh hidden states for every game.
The pass reports each policy's victory percentage and does not contribute rows
to the next training iteration.

### Reference run

With five policies and twelve lanes, one iteration beginning at `g_0` runs the
following sequence:

```text
collect g_0, g_0 + 1, g_0 + 2, and g_0 + 3
    4 counters × 12 lanes × C(5, 2) pairs = 480 full engine games
    per policy: 4 opponents × 12 lanes × 4 counters = 192 trajectories
    per policy: 192 trajectories × 21 learned steps = 4,032 actor rows

freeze the five 4,032-row actor objectives and assemble 20,160 critic rows
run up to four PPO passes for each policy and four MSE passes for the critic
evaluate the resulting five policy versions on the held-out fixture set
```

## Manual evaluation and population changes

The five active policies train only against each other. Evaluation uses a
configured held-out fixture set with root seeds distinct from collection lanes.
An evaluation pass plays each active policy against every other active policy
over that fixture set with the same Queen/King balancing rule as collection.
Each game starts new participant hidden states and executes all six rounds.

The ranking metric is victory percentage:

```text
victory_percentage = (wins + 0.5 × ties) / completed_games
```

The configured evaluation window defaults to one post-update evaluation pass.

Population changes occur manually between training runs. Victory percentages
support the decision to archive a high-performing checkpoint as a regression
reference or replace an active policy with a randomly initialized policy.
Collection then resumes with five policies.

The five-policy population is a training mechanism, not a set of product
difficulty levels. When training concludes, one checkpoint is manually selected
as the strongest deployment candidate. Candidate comparisons reuse the same
held-out fixtures, roles, and action-selection profile so deck ordering does not
favor one candidate. Victory percentage remains the primary ranking metric;
per-fixture results remain available for manual review.

## Local ML stack

Training uses native arm64 Python 3.12 and PyTorch. The deterministic engine,
fixture scheduler, action sampling, and trajectory storage run on CPU. Policy
collection and policy and critic optimization select a PyTorch device through
configuration.

The target machine has 8 GB of unified memory. Version 1 therefore uses
`float32`, processes one learner policy at a time, keeps full collection buffers
on CPU, transfers only the active minibatch to the optimization device, and
uses eager PyTorch execution with in-process data loading.

The default configuration uses CPU for collection and MPS for optimization.
Collection consists of small recurrent calls interleaved with CPU engine work;
optimization performs the larger batched forward and backward passes that can
benefit from MPS. Both phases use the same PyTorch implementation.

The trajectory minibatch is 16 complete player games. Supported values are one
through 64; changes are made explicitly between runs after measuring the full
24-step backward pass on the target machine.

The initial benchmark measures collection and optimization separately on CPU
and MPS. It records phase duration and throughput, process memory, MPS memory,
swap growth, finite losses and gradients, and numerical agreement for logits,
critic values, and hidden states. The committed device configuration must
complete a sustained iteration without unsupported operations, excessive swap,
or material thermal degradation.

## Local training suite

The suite runs as one process with ordered collection, optimization, evaluation,
and commit phases. Collection seals the complete training window before weights
change. Optimization updates the five policies sequentially, updates the shared
critic, validates all six models, and writes one population checkpoint.
Evaluation reads that checkpoint and produces the manual comparison report.

### Configuration

A run reads one TOML file. The suite resolves defaults and `auto` device values
once at startup and writes an immutable JSON configuration manifest. At minimum,
the configuration contains:

| Section | Fields |
| --- | --- |
| Run | Run ID, root seed, iteration count, output directory |
| Population | Five policy IDs and checkpoints, critic checkpoint |
| Fixtures | Twelve collection lane roots, held-out evaluation roots, next game counter |
| Compute | Collection device, optimization device, trajectory minibatch size |
| Training | Optimizer settings, PPO settings, critic burn-in and actor-weight schedule |
| Versions | Rules, engine, observation, action map, policy, critic, and training formats |

The compute settings are:

```toml
[compute]
collection_device = "cpu"       # cpu | mps | auto
optimization_device = "mps"     # cpu | mps | auto
trajectory_batch_size = 16      # 1..64 complete trajectories
```

`auto` selects MPS when it is available and passes a policy-and-critic operation
probe; otherwise it selects CPU. The resolved device remains fixed for the run.
An out-of-memory failure does not change the batch size automatically.

Random operations use independently derived seeds. Model initialization,
fixture action sampling, trajectory shuffling, critic-row shuffling, and
evaluation sampling each use the run root seed, a versioned namespace, and
stable operation identifiers. Resumption therefore reconstructs the same
streams without depending on an unrecorded mutable generator.

### Memory and process boundaries

The suite retains observations and masks as Boolean or unsigned-byte CPU
tensors and converts the active minibatch to `float32` at the model boundary.
Complete trajectories remain ordered and on CPU. Only the policy or critic
being optimized and its active minibatch occupy the optimization device.

The initial process has no worker processes or multiprocessing data loader.
Collection may batch simultaneously available policy views while preserving
each fixture's engine, hidden states, and sampling seed. Policy optimization
processes one learner bucket at a time. Critic optimization uses the shared
CPU row pool.

### Artifacts and checkpoints

Each run uses this logical layout:

```text
runs/<run-id>/
  resolved-config.json
  state.json
  checkpoints/<iteration>/iteration-start.pt
  checkpoints/<iteration>/post-update.pt
  collections/<iteration>.pt
  metrics/<iteration>.json
  reports/<iteration>.md
```

`state.json` identifies the last committed phase and references artifacts by
content hash. Artifact files are written to a temporary name, validated, and
atomically renamed before `state.json` advances.

An iteration-start checkpoint contains the five policy states and optimizer
states, critic state and optimizer state, policy and critic versions, training
and schedule counters, lane roots and next game counter, seed-derivation
version, and resolved configuration and contract versions.

The sealed collection artifact contains the five ordered actor buckets, the
shared critic rows, fixture identities, behavior-policy versions, selected
actions, old log probabilities, frozen critic estimates, returns, and masks
required to reproduce the update. It contains no retained fixture hidden
states. The artifact is deleted after the iteration and its report commit
successfully unless retained explicitly for diagnosis.

The post-update checkpoint contains the same resumable training state with all
six updated model versions and the next game counter. A manually archived
checkpoint contains the selected policy artifact, its model and tensor
contracts, its resolved training manifest, and the comparison report that
motivated retention.

### Recovery and commit rules

Recovery begins at a complete phase boundary:

- A collection failure reloads the iteration-start checkpoint and recollects
  the four-generation window.
- An optimization failure reloads the iteration-start checkpoint and sealed
  collection, then repeats all policy and critic updates.
- An evaluation failure reloads the post-update checkpoint and repeats
  evaluation.

The suite does not resume a partial game or minibatch. A post-update population
becomes the active population only after all policy and critic states are
finite, expected tensor shapes and version contracts match, the checkpoint
commits, and the next game counter advances. Any validation failure leaves the
prior committed population active.

### Metrics and comparison report

Each iteration records:

- Victory percentage and mean round return for each policy, including Queen and
  King splits.
- Policy loss, entropy, approximate KL, pre-mask illegal probability, and
  gradient norm for each policy.
- Critic mean squared error, zero-predictor mean squared error, and gradient
  norm.
- Collection, optimization, and evaluation duration and throughput.
- Peak process memory, peak MPS allocation, swap growth, and validation
  failures.

Metrics are stored as structured JSON. The Markdown report identifies the run,
population and contract versions, resolved compute configuration, victory
percentages, training diagnostics, runtime and memory measurements, and any
failed phase. It supports manual checkpoint archiving and random policy
replacement; it does not select or replace a policy.

### Verification

The suite is accepted after a smoke configuration and one full five-policy,
twelve-lane, four-generation iteration complete on the target laptop. Recovery
tests interrupt collection, optimization, and evaluation and confirm that each
phase restarts from its documented boundary. Repeated recovery produces the
same fixture schedule and committed model versions, with neural values matching
the configured numerical tolerance. The full run must avoid sustained swap
growth and produce a complete comparison report.

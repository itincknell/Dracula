# Model training

This document defines the separate system used to train, evaluate, and promote
the recurrent opponent policy. It does not participate in live game state or web
request handling.

## Self-play system

The training environment runs the authoritative pure Python engine without AWS
or narration. Its primary configuration dimensions are:

- `X`: independently initialized current player policies.
- `Y`: concurrent seeded game streams.

Each current player plays each seeded fixture against each current opponent.
Fixtures expose the same deals to different opponents and different deals to
the same opponents. Role, dealer, and play-order balancing are part of the
fixture schedule. `X` and `Y` are increased only to the limit supported by
measured collection, optimization, and memory throughput.

Seed streams may start at configured offsets by burning cards before the first
game; the current proposal is `6 × offset` cards. A stream starts another game
from its configured seed when one finishes. `TRAIN-004` must define whether
that means continuation of one random stream or reproducible reseeding, prevent
unintended overlapping fixtures, and define exactly how offsets map to deals.

Each simulated player has a separate recurrent hidden state and receives only
its legitimate `PolicyGameView`. The policy is invoked for the seven non-forced
placements. The engine applies the eighth placement directly and uses the
complete round to calculate terminal return.

Collected optimizer samples are owned by the learner and exact policy version
that generated them. Each decision record contains the observation, legal mask,
input hidden state, selected action, old action log-probability, critic estimate,
learner and opponent identities and versions, fixture identity, and eventual
terminal round return. The forced placement is an environment transition, not
an actor-loss sample.

Recurrent updates preserve complete per-player decision sequences or carry the
exact initial hidden state for a fragment. Minibatches mix fixtures, opponents,
roles, dealers, and game stages while retaining learner/version ownership.

## Learning objective

The player policy and critic are conceptually and operationally separate:

- The recurrent policy produces masked move probabilities and selects actions.
- One shared, player-agnostic critic estimates expected terminal round return
  from a perspective-normalized state and trains from states collected across
  all current players, opponents, and seeded fixtures.
- The critic is used only to estimate advantages during training. It is not an
  input to policy action selection and is not deployed with the policy.

The deterministic engine supplies one terminal round return to each player
after the forced placement. Its exact definition and normalization remain open;
the current candidate is an opposing value derived from the two rules-defined
round results. The training design uses *return* for this signal to distinguish
it from model logits and rules-defined scores.

The critic has its own regression objective against observed or bootstrapped
returns. A move's advantage is derived from the terminal return and critic
estimates; the critic does not directly label a move. Terminal Monte Carlo and
generalized advantage estimation are the candidate estimators.

PPO is the candidate policy-update algorithm. For each collected action it
compares the action probability under the updated policy with the saved
probability under the generating policy. The clipped surrogate limits a single
update from changing that ratio too far. Each policy's actions contribute only
to that policy's learner update; both sides may contribute to their respective
learner batches when both are current policies.

The combined training configuration includes distinct terms for:

- PPO policy loss using estimated advantages.
- Critic value regression loss.
- Optional entropy regularization over legal actions.
- Thresholded illegal-probability penalty measured before the hard output mask.
- A scheduled policy round-objective weight.

The critic initially trains on the broad random play produced by untrained
policies. The policy round-objective weight begins at zero or a low value and
increases after a configured critic burn-in condition. The critic continues to
train throughout self-play. Burn-in exit, weight schedule, PPO clip range,
target KL, update epochs, value coefficient, entropy coefficient, illegal-loss
threshold and coefficient, gradient clipping, and optimizer settings remain
configuration decisions.

The training environment supplies no manually coded strategic move score. The
deterministic engine supplies legal actions and terminal rules outcomes only.

## Population lifecycle

Current policies train against other current policies. At a configured cadence,
performance is estimated on controlled fixtures. The lowest-performing eligible
policies may be archived and replaced by randomly initialized policies. This
preserves opponent variety and introduces learners under different critic and
population states.

Replacement must not use noisy short-window ranks as fact. The design must set
minimum evidence, tie handling, replacement count and cadence, archive metadata,
critic behavior across replacement, and whether archived policies remain
evaluation references or active opponents. External random action injection,
temperature, and entropy are alternative exploration controls and are not yet
selected.

## Training suite

The suite will provide:

- Vectorized environment collection and legal-action masking.
- Population creation, matchmaking, checkpointing, and resumption.
- Recurrent trajectory storage with policy versions and behavior probabilities.
- Separate policy and critic optimization, validation, periodic tournaments,
  and regression detection.
- Structured metrics and reproducible configuration snapshots.
- Artifact export, evaluation attachment, and Model Registry submission.

Every run records source, engine, rules, tensor schema, model architecture,
framework, hardware, seeds, population, opponent-selection policy,
hyperparameters, checkpoints, duration, and cost.

## Local compute

The first benchmark compares native CPU execution, PyTorch MPS, TensorFlow with
`tensorflow-metal`, and MLX on the actual environment-collection and
recurrent-update workload. It measures complete rounds per second, policy-update
throughput, memory, numerical stability, portability, and implementation effort.

The selected local environment must use native Apple Silicon Python and support
long-running resumable jobs. Local checkpoints use the same logical artifact
metadata as cloud training.

## AWS training

SageMaker Training is the cloud execution target when local throughput is
insufficient or a cloud parity run is required. The training image is stored in
ECR; checkpoints, metrics, and final artifacts are stored in S3 and CloudWatch.
Managed Spot Training is preferred for interruptible long runs after checkpoint
recovery is verified.

Training jobs have explicit maximum runtime, instance count, checkpoint
interval, and cost metadata. SageMaker Studio is not required for unattended
jobs and must not remain running as an incidental dependency.

## Evaluation and promotion

Evaluation tournaments include random legal play, independently trained current
policies, and a fixed historical checkpoint pool. Current-versus-current results
alone are not promotion evidence. Reports separate Queen/King role, dealer,
round, policy version, invalid outputs, score differential, win rate, latency,
and inference stability.

A candidate that passes the selected thresholds is registered as a versioned
model package in SageMaker Model Registry. Promotion records the approved model
package and serving configuration. Production games pin that exact version for
their lifetime.

> **TODO TRAIN-001 — Select the local ML stack and system requirements.** Build
> the representative CPU, PyTorch MPS, TensorFlow Metal, and MLX benchmark and
> record supported OS, Python, framework, memory, and toolchain versions.
>
> **Complete when:** Results identify the default local framework and a portable
> cloud path using end-to-end rounds-per-second and update-throughput evidence.

> **TODO TRAIN-002 — Define terminal return and the critic.** Select the terminal
> round-return formula and normalization, perspective transform, critic inputs
> and architecture, value targets, recurrence behavior, burn-in workload, and
> burn-in exit criteria.
>
> **Complete when:** Equations and deterministic fixtures map complete rounds to
> both players' returns, critic targets, and perspective-normalized inputs
> without ambiguity.

> **TODO TRAIN-003 — Define the policy update.** Confirm or replace PPO; define
> advantage estimation, old-policy data, clip range, target KL, update epochs,
> entropy, raw illegal-probability penalty, round-objective ramp, sequence
> handling, optimizer, and numerical safeguards.
>
> **Complete when:** Equations, tensor shapes, pseudocode, and deterministic
> cases specify one separate critic update and one learner-policy update without
> implying that the policy consults the critic during action selection.

> **TODO TRAIN-004 — Define seeded collection and matchmaking.** Specify `X` and
> `Y`, seed continuation or reseeding, burn offsets, fixture identity, role and
> dealer balancing, full versus sampled round robin, batch mixing, and exact
> trajectory fields.
>
> **Complete when:** A small seeded schedule reproduces every deal and matchup,
> avoids unintended fixture overlap, and attributes every action to its learner,
> opponent, versions, behavior probability, and terminal return.

> **TODO TRAIN-005 — Define population lifecycle and exploration.** Select
> initialization, ranking evidence, replacement eligibility and cadence,
> archive retention and use, critic treatment of replacements, exploration
> controls, and collapse/regression response.
>
> **Complete when:** A seeded population schedule deterministically identifies
> evaluation, archive, replacement, and exploration actions from recorded
> results and configuration.

> **TODO TRAIN-006 — Specify one end-to-end training iteration.** Integrate the
> tensor contract, seven-decision trajectories, collector, critic update, policy
> update, and population schedule into one ordered dataflow with ownership and
> version boundaries.
>
> **Complete when:** Executable pseudocode and a deterministic miniature run
> specify every state transition and tensor from deal creation through updated
> policy and critic checkpoints.

> **TODO TRAIN-007 — Specify the training suite and resumption contract.** Define
> process boundaries, configuration schema, checkpoint contents, metrics,
> failure recovery, and artifact layout.
>
> **Complete when:** A stopped local run resumes without losing policy or critic
> models, optimizers, schedulers, random generators, seeded streams, population,
> matchmaking, or evaluation state.

> **TODO TRAIN-008 — Finalize SageMaker training and cost controls.** Choose the
> container, instance candidates, Spot behavior, storage, IAM, network boundary,
> runtime caps, checkpoint interval, and budget alarms.
>
> **Complete when:** A short parity job reproduces local behavior, restores from
> an interrupted checkpoint, and emits a complete itemized cost record.

> **TODO EVAL-001 — Define competence and promotion criteria.** Select tournament
> opponents, seeds, sample sizes, metrics, uncertainty reporting, regression
> limits, and approval thresholds.
>
> **Complete when:** The protocol can reject a deliberately regressed policy and
> promote a candidate without reference to current-versus-self win rate alone.

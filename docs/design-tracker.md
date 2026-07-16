# Design tracker

This is the single index for unfinished MVP design. **Open** items are ready to
design, **blocked** items require another item in the current gate, and
**deferred** items are intentionally held until the current gate closes. An item
is complete when its stated evidence exists.

## Current design gate

The project does not advance into implementation planning, framework
benchmarking, serving, application evaluation, or operations until the neural
training architecture is fully specified. The gate contains:

- `MODEL-001`: tensor, legal-mask, recurrence, and forced-placement contract.
- `MODEL-002`: recurrent policy architecture.
- `TRAIN-002`: terminal return and independent critic.
- `TRAIN-003`: policy update and loss schedule.
- `TRAIN-004`: seeded collection and matchmaking.
- `TRAIN-005`: population lifecycle and exploration.
- `TRAIN-006`: one integrated training iteration.

Closing the gate requires all seven items to be complete and mutually
consistent. `TRAIN-006` is the integration check; it cannot close by referring
unresolved behavior to implementation.

## Critical path

```text
Rules and game state -> tensor and sequence contract
                              |
                              v
                  policy architecture + critic and return
                              |
                              v
                PPO/update + seeded collection + population
                              |
                              v
                   specified training iteration [GATE]
                              |
                              v
                framework benchmark + training suite build
                              |
                              v
                 evaluation -> registry -> inference -> release

Application API -> policy-turn contract ----------^
Public events -> narrator prompt and evaluation
```

## Product, rules, and experience

| ID | Status | Work | Depends on | Completion evidence |
| --- | --- | --- | --- | --- |
| `PROD-001` | Complete | Define the revised release in [product and scope](product-and-scope.md). | None | Public access, trained-policy requirement, supported clients, release checks, and smoke test are defined |
| `RULE-001` | Complete | Maintain authoritative [rules](rules.md) and examples. | Authoritative rules | Project variant and implementation clarifications are documented with examples and sources |
| `UX-001` | Complete | Maintain responsive gameplay and scoring in [frontend experience](frontend-experience.md). | `RULE-001` | Start, desktop/mobile layouts, turns, commentary, scoring, completion, recovery, assets, and usability are defined |

## Application and serving

| ID | Status | Work | Depends on | Completion evidence |
| --- | --- | --- | --- | --- |
| `ARCH-001` | Complete | Maintain authoritative schemas, lifecycle, events, and persistence in [architecture](architecture.md). | `RULE-001` | State, projections, invariants, transactions, replay, retention, and adapters are defined |
| `ARCH-002` | Deferred | Finalize policy-turn orchestration and inference schemas in [architecture](architecture.md). | Training architecture gate, `MODEL-003` | Idempotency, hidden-state commit, sampling ownership, retry, timeout, and invalid-output behavior are unambiguous |
| `SERVE-001` | Deferred | Validate SageMaker Serverless Inference configuration and rollout in [evaluation and operations](evaluation-and-operations.md). | Training architecture gate, `MODEL-003`, `EVAL-001`, `ARCH-002` | Approved model version meets cold-start, warm-latency, memory, cost, rollback, and concurrency thresholds |

## Neural model

| ID | Status | Work | Depends on | Completion evidence |
| --- | --- | --- | --- | --- |
| `MODEL-001` | Open | Finalize tensor, legal-mask, recurrence, and seven-decision contracts in [neural model](neural-model.md). | `RULE-001`, `ARCH-001` | Versioned fixtures prove exact shapes, mappings, invariants, information boundaries, and forced final placement |
| `MODEL-002` | Blocked | Select the recurrent policy architecture in [neural model](neural-model.md). | `MODEL-001` | The recurrent cell, projections, mask injection, widths, layers, initialization, heads, and parameter count are reproducibly specified |
| `MODEL-003` | Deferred | Finalize inference and artifact behavior in [neural model](neural-model.md). | Training architecture gate, `TRAIN-001` | Local and container inference agree within documented tolerances and exchange compatible hidden state |

## Training and promotion

| ID | Status | Work | Depends on | Completion evidence |
| --- | --- | --- | --- | --- |
| `TRAIN-001` | Deferred | Select the local ML stack and system requirements in [model training](model-training.md). | Training architecture gate | CPU, PyTorch MPS, TensorFlow Metal, and MLX benchmark identifies the default local and portable cloud path |
| `TRAIN-002` | Open | Define terminal return and the independent critic in [model training](model-training.md). | `RULE-001` | Complete rounds map unambiguously to both returns, critic targets, and perspective-normalized inputs |
| `TRAIN-003` | Blocked | Define the policy update and loss schedule in [model training](model-training.md). | `TRAIN-002` | Equations and deterministic cases specify separate critic and learner-policy updates, including PPO configuration if selected |
| `TRAIN-004` | Open | Define seeded collection and matchmaking in [model training](model-training.md). | `RULE-001` | A reproducible schedule defines seeds, offsets, fixtures, balance, batches, and trajectory ownership |
| `TRAIN-005` | Blocked | Define population lifecycle and exploration in [model training](model-training.md). | `TRAIN-002`, `TRAIN-004` | Ranking evidence deterministically drives configured archive, replacement, and exploration actions |
| `TRAIN-006` | Blocked | Specify one end-to-end training iteration in [model training](model-training.md). | `MODEL-001`, `MODEL-002`, `TRAIN-002`, `TRAIN-003`, `TRAIN-004`, `TRAIN-005` | Executable pseudocode and a miniature run account for every state transition, tensor, version, and update |
| `TRAIN-007` | Deferred | Specify training-suite boundaries, checkpoints, metrics, and resumption in [model training](model-training.md). | Training architecture gate, `TRAIN-001` | A stopped run can resume every policy, critic, optimizer, scheduler, random stream, population, and evaluation state |
| `TRAIN-008` | Deferred | Finalize SageMaker Training and cost controls in [model training](model-training.md). | `TRAIN-001`, `TRAIN-007` | A bounded parity job restores from interruption and emits a complete cost record |
| `EVAL-001` | Deferred | Define policy competence and promotion criteria in [model training](model-training.md). | Training architecture gate, `TRAIN-007` | Versioned tournaments can reject a regression and promote a candidate using fixed external references |

## Narrator and delivery

| ID | Status | Work | Depends on | Completion evidence |
| --- | --- | --- | --- | --- |
| `NARRATOR-001` | Complete | Maintain narrator cadence and public-data boundaries in [narrator](narrator.md) and [architecture](architecture.md). | `ARCH-001` | Trigger matrix and blocking/non-blocking presentation behavior are defined |
| `NARRATOR-002` | Deferred | Define the Dracula persona and prompt in [narrator](narrator.md). | Training architecture gate, `NARRATOR-001`, `EVAL-002` | Versioned public-only cases pass grounding, privacy, cadence, repetition, length, and tone checks |
| `EVAL-002` | Deferred | Define application, narrator, and cost acceptance in [evaluation and operations](evaluation-and-operations.md). | Training architecture gate, `PROD-001` | Every release claim has a metric, threshold, fixture or workload, and reproducible cost method |
| `OPS-001` | Deferred | Finalize environment, security, observability, retention, and budgets in [evaluation and operations](evaluation-and-operations.md). | Training architecture gate, `PROD-001` | Choices are represented in versioned infrastructure and boundary tests |
| `OPS-002` | Deferred | Write local, CI, deployment, recovery, rollback, and teardown instructions in [evaluation and operations](evaluation-and-operations.md). | `SERVE-001`, `OPS-001` | Clean local and AWS environments succeed without unstated steps |

## Completion

The design sub-project is complete when every row is complete, the application
and model builds have explicit contracts, the approved policy has reproducible
training and promotion evidence, serving and narration meet release thresholds,
and local/deployment instructions are executable.

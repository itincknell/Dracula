# Design tracker

This is the single index for unfinished MVP design. **Blocked** items require an
earlier decision or external input; **open** items are ready to design. An item
is complete when its stated evidence exists.

## Critical path

```text
Rules -> state/tool contracts -> turn/API behavior -> implementation
                       |                    |
                       v                    +------> narrator event flow
                 model choices ------> evaluation          |
                                            |
                                            v
                               UX scoring flow and deployment
```

## Product and rules

| ID | Status | Work | Depends on | Completion evidence |
| --- | --- | --- | --- | --- |
| `PROD-001` | Complete | Define release-ready in [product and scope](product-and-scope.md). | None | Public access, supported clients, release checks, and smoke test are defined |
| `RULE-001` | Complete | Complete authoritative [rules](rules.md) and examples. | Authoritative rules | Standard rules and implementation clarifications are documented with examples and sources |

## Architecture and experience

| ID | Status | Work | Depends on | Completion evidence |
| --- | --- | --- | --- | --- |
| `ARCH-001` | Complete | Finalize schemas, visibility, lifecycle, and invariants in [architecture](architecture.md). | `RULE-001` | Schemas, lifecycle transitions, projections, and invariants are defined |
| `ARCH-002` | Complete | Finalize turn orchestration and API behavior in [architecture](architecture.md). | `ARCH-001` | Success, concurrency, idempotency, recovery, and failure behavior are defined |
| `ARCH-003` | Complete | Finalize events, replay, retention, and persistence adapters in [architecture](architecture.md). | `ARCH-001` | Event transactions, replay, retention, and adapter contracts are defined |
| `UX-001` | Complete | Define the responsive game flow in [frontend experience](frontend-experience.md). | `RULE-001`, `ARCH-002` | Start, desktop/mobile layouts, turns, narration, scoring, completion, and recovery are defined |
| `UX-002` | Open | Define usability and asset acceptance in [frontend experience](frontend-experience.md). | `PROD-001` | Testable layout, input, accessibility, asset, and link checklist |
| `UX-004` | Open | Design the scripted round-scoring presentation in [frontend experience](frontend-experience.md). | `ARCH-001`, `NARRATOR-001` | Reviewed storyboard covers lines, multipliers, ties, totals, and narration points |

## Agents and evaluation

| ID | Status | Work | Depends on | Completion evidence |
| --- | --- | --- | --- | --- |
| `AGENT-001` | Open | Specify Strands tool contracts in [agents and models](agents-and-models.md). | `RULE-001`, `ARCH-001` | Local, deterministic, non-mutating contract tests |
| `AGENT-002` | Blocked | Choose models, prompts, strategy, and AgentCore/Strands configuration in [agents and models](agents-and-models.md). | `EVAL-001` | Evaluated, justified, fully persisted configuration |
| `AGENT-003` | Blocked | Define retries, limits, and invalid output handling in [agents and models](agents-and-models.md). | `AGENT-001`, `EVAL-001`, `ARCH-002` | Failure tests prove at most one applied move |
| `NARRATOR-001` | Complete | Finalize narrator triggers, cadence, ordering, and scoring integration in [agents and models](agents-and-models.md). | `ARCH-001` | Public event matrix and blocking/non-blocking presentation behavior are defined |
| `NARRATOR-002` | Blocked | Define Dracula persona and prompt in [agents and models](agents-and-models.md). | `NARRATOR-001`, `EVAL-002` | Public-only golden cases pass for grounding, cadence, and tone |
| `EVAL-001` | Open | Define strategic questions and success thresholds in [evaluation and operations](evaluation-and-operations.md). | `RULE-001`, `PROD-001` | Metrics and paired-seed protocol support each model decision |
| `EVAL-002` | Open | Define narrator and cost evaluation in [evaluation and operations](evaluation-and-operations.md). | `RULE-001` | Versioned narrator cases and reproducible cost method |

## Delivery

| ID | Status | Work | Depends on | Completion evidence |
| --- | --- | --- | --- | --- |
| `OPS-001` | Open | Finalize environment, limits, observability, and security in [evaluation and operations](evaluation-and-operations.md). | `PROD-001` | Choices represented in CDK/configuration and boundary tests |
| `OPS-002` | Blocked | Write local, CI, deployment, recovery, and teardown instructions in [evaluation and operations](evaluation-and-operations.md). | `ARCH-003`, `OPS-001` | Clean local and AWS setup succeeds without unstated steps |

## Completion

The design sub-project is complete when every row is complete, the rules contain
no inferred behavior, configuration choices are justified, UI and API flows are
testable, and local/deployment instructions are executable.

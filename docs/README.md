# Dracula design docs

This folder contains the MVP design. The [design tracker](design-tracker.md)
indexes open design work.

| Document | Purpose |
| --- | --- |
| [Product and scope](product-and-scope.md) | North stars, goals, non-goals, acceptance |
| [Rules](rules.md) | Authoritative game rules and examples |
| [Architecture](architecture.md) | Web application, opponent orchestration, state, API, and persistence |
| [Engine–opponent contract](engine-model-contract.md) | Deterministic engine, information-state projection, determinization, and action mapping |
| [Information-set search](search.md) | Round-local search, hidden-information handling, feasibility, and gates |
| [Neural model](neural-model.md) | Gated shared policy/value model and training targets |
| [Model training](model-training.md) | Search validation, expert iteration, checkpoints, and absolute comparison |
| [Narrator](narrator.md) | Dracula commentary behavior, prompt, and Bedrock configuration |
| [Frontend experience](frontend-experience.md) | Responsive game interface and interaction flow |
| [Evaluation and operations](evaluation-and-operations.md) | Application evaluation, deployment, observability, security, and local development |
| [Decisions](decisions.md) | Accepted design choices and omitted features |

Sources of truth are [`NORTHSTARS`](../NORTHSTARS) and the documents in this
folder. The tracker records the status and completion condition for each open
item.

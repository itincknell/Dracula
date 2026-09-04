# Dracula design

These documents are the active first-pass contracts. The
[design tracker](design-tracker.md) records current work and priority;
[reports](../reports/README.md) retain operational evidence and historical
experiments.

| Document | Purpose |
| --- | --- |
| [Product and scope](product-and-scope.md) | North stars, goals, non-goals, acceptance |
| [Rules](rules.md) | Authoritative game rules and examples |
| [Architecture](architecture.md) | Stateless web application, `pi1`, replay, API, and narration boundaries |
| [Stateless gameplay API](stateless-api.md) | Production envelope, replay, cache, routes, errors, and privacy |
| [Deployment](deployment.md) | GitHub Pages, Cloudflare, API Gateway, Lambda, Bedrock, and release phases |
| [Engine–opponent contract](engine-model-contract.md) | Deterministic engine, information-state projection, and action mapping |
| [Symmetry and move selection](search.md) | Authoritative symmetry table and deterministic standalone action selection |
| [Neural model](neural-model.md) | Selected standalone 754,601-parameter `pi1` architecture and action semantics |
| [Model training](model-training.md) | Retained `pi1` corpus reader, trainer, and completed-run evidence |
| [Narrator](narrator.md) | Locked Bedrock cue cadence and public-input boundary |
| [Frontend experience](frontend-experience.md) | Responsive game interface and interaction flow |
| [Evaluation and operations](evaluation-and-operations.md) | Release verification, observability, cost, and local development |
| [Decisions](decisions.md) | Accepted design choices and omitted features |

[`NORTHSTARS`](../NORTHSTARS) governs scope and style. The deterministic game
rules, engine contract, symmetry contract, model contract, and resolved runtime
configuration are authoritative within their stated boundaries. Historical
reports are evidence, not active alternatives.

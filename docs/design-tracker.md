# Design tracker

This is the index for unfinished MVP work. Tracker IDs appear only here.

## Current item

`WEB-001` builds the playable browser application from the completed frontend
and API contracts. The React application and FastAPI boundary must support local
development and the deployed application without separate gameplay code paths.

## Remaining sequence

```text
playable web application -> narrator configuration -> release
```

| ID | Status | Work | Complete when |
| --- | --- | --- | --- |
| `WEB-001` | Open | Implement the React, TypeScript, and Vite gameplay surface and the local FastAPI application boundary defined in [frontend experience](frontend-experience.md) and [architecture](architecture.md). Keep the API client configurable so the same browser build can use local FastAPI or the deployed API. | A user can select Queen or King, complete a six-round game through drag and drop, follow scoring, resize between desktop and mobile layouts, reload into the current game, and run locally with narration disabled |
| `NARRATOR-002` | Deferred | Finalize the Dracula prompt, model configuration, and narrator acceptance cases in [narrator](narrator.md), then connect it to the existing commentary hooks. | Public-only fixtures pass grounding, privacy, brevity, repetition, cadence, and tone checks |
| `RELEASE-001` | Deferred | Finalize application acceptance, observability, security, budgets, deployment, rollback, and teardown in [evaluation and operations](evaluation-and-operations.md). | A clean environment can deploy, complete the smoke test, recover, roll back, and tear down without unstated steps |

## Completed foundations

| ID | Area | Evidence |
| --- | --- | --- |
| `PROD-001` | Product and scope | [Product and scope](product-and-scope.md) defines the MVP and release-ready behavior |
| `RULE-001` | Rules | [Rules](rules.md) defines the project variant and scoring examples |
| `UX-001` | Frontend | [Frontend experience](frontend-experience.md) defines interaction, responsive layout, scoring, recovery, and assets |
| `ARCH-001` | Application architecture | [Architecture](architecture.md) defines state, lifecycle, API, events, and persistence |
| `ENGINE-001` | Engine–model bridge | [Engine–model contract](engine-model-contract.md) defines deterministic transitions and policy mapping |
| `MODEL-001` | Model inputs and recurrence | [Neural model](neural-model.md) defines tensors, masks, recurrence, and forced transitions |
| `MODEL-002` | Policy architecture | [Neural model](neural-model.md) defines the structured encoder, GRU, and action head |
| `TRAIN-001` | Local ML stack | [Model training](model-training.md) selects PyTorch, configurable CPU/MPS optimization, float32, and the target laptop constraints |
| `TRAIN-002` through `TRAIN-006` | Training architecture | [Model training](model-training.md) defines return, critic, PPO, collection, population, and one integrated iteration |
| `TRAIN-007` | Local training suite | [Model training](model-training.md) defines configuration, compute and memory boundaries, checkpoints, recovery, metrics, and manual comparison reports |
| `SERVE-001` | Policy serving | [Neural model](neural-model.md), [architecture](architecture.md), and [evaluation and operations](evaluation-and-operations.md) define the artifact, stateless inference, turn transaction, Serverless deployment, and validation |
| `NARRATOR-001` | Narrator integration | [Narrator](narrator.md) and [architecture](architecture.md) define public-data boundaries and cadence |

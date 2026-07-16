# Evaluation and operations

Model training and competence evaluation are defined in
[model training](model-training.md). This document covers the deployed web
application, policy-serving integration, narrator, and operational release.

## Application evaluation

Deterministic tests cover dealing, legal moves, projections, scoring examples,
state invariants, lifecycle transitions, event replay, conditional persistence,
and idempotency. Adapter contract tests run against in-memory, SQLite, and
DynamoDB implementations.

Policy-serving tests cover observation and mask construction, pinned model
versions, hidden-state loading and transactional advancement, stale and duplicate
requests, invalid actions, timeouts, endpoint errors, and recovery. No failure
may apply more than one move or advance hidden state without its accepted move.

The deployment smoke test loads the frontend, passes `/health`, creates a seeded
game, completes all six rounds through the deployed policy and narrator, checks
recovery from one interrupted turn, and confirms useful logs without private
state.

## Narrator and cost evaluation

Narrator cases use versioned public event projections. They assess factual
grounding, private-state leakage, brevity, repetition, character, cadence, and
graceful timeout or failure behavior.

Cost reports use timestamped regional prices and raw usage. They separate
SageMaker training, model storage and registry artifacts, Serverless Inference,
Bedrock narration, Lambda, API Gateway, DynamoDB, Amplify, S3, and CloudWatch.
Training reports belong to the training run; release reports include expected
interactive traffic and cold-start contingencies.

> **TODO EVAL-002 — Define application, narrator, and cost acceptance.** Select
> workloads, cases, metrics, thresholds, sample sizes, timestamped pricing input,
> and release interpretation.
>
> **Complete when:** Every release claim has a reproducible test or workload and
> raw usage deterministically produces the reported cost.

## Serving validation

The initial endpoint is SageMaker Serverless Inference using an approved Model
Registry package. Validation measures container size, model-load time, cold and
warm latency, memory configuration, concurrency, throttling, error behavior,
numerical parity, and per-turn cost. Provisioned Concurrency and a real-time
endpoint are excluded unless measurements show that on-demand latency fails the
release threshold.

Promotion never changes the model version of an active game. Deployment either
keeps the prior version addressable until affected games expire or prevents new
games from using it while existing games finish. Rollback restores the previous
approved model and serving configuration for new games.

> **TODO SERVE-001 — Validate Serverless Inference and model rollout.** Choose
> container, memory, concurrency, timeout, invocation schema, version routing,
> monitoring, deployment, and rollback behavior.
>
> **Complete when:** The approved artifact passes parity, cold/warm latency,
> memory, concurrency, failure, cost, and rollback checks at release load.

## Deployment

AWS CDK deploys Amplify Hosting, API Gateway, FastAPI on Lambda through Mangum,
the asynchronous narrator Lambda, DynamoDB, IAM, CloudWatch, S3 evaluation
storage, budget alerts, and the SageMaker Serverless endpoint integration. The
model build owns its ECR training/inference images, S3 artifacts, SageMaker
Training jobs, and Model Registry versions.

Only the game service may invoke policy inference. DynamoDB and SageMaker are
not browser-accessible. Every returned policy action is revalidated. Invocation,
retry, output, and wall-time limits are configuration. Add per-IP/API rate
limits before public launch and AWS Budget alerts aligned with the revised cost
model.

Logs and traces include game and turn identifiers, policy and schema versions,
endpoint latency and status, action validity, narrator model and prompt versions,
tokens, cost, and failures. They exclude hands, hidden-card locations, stock
order, recurrent hidden vectors, private prompts, and model internals.

## Local development

The pure Python engine runs without AWS. SQLite supplies local persistence,
in-memory adapters support unit tests, and local policy and narrator adapters
implement the production interfaces. A deterministic policy stub supports API
and frontend tests; the exported learned artifact supports local end-to-end
inference after `MODEL-003` is complete.

Verification covers engine rules, model-view encoding, API contracts, policy
integration, frontend gameplay, narrator cases, deployment smoke tests, and
seeded replay.

> **TODO OPS-001 — Finalize environment and operational controls.** Select AWS
> region, environments, naming, IAM, endpoint protection, CORS, secrets and
> configuration handling, rate limits, retention enforcement, alarms, runtime
> limits, and budget thresholds.
>
> **Complete when:** Versioned infrastructure expresses the choices and tests
> prove public callers cannot reach private data or services.

> **TODO OPS-002 — Write executable development and deployment instructions.**
> Choose language versions, package managers, commands, local adapters, CI gates,
> deployment verification, model promotion, rollback, recovery, and teardown.
>
> **Complete when:** A clean checkout can run and test locally, and a clean AWS
> environment can be deployed, smoke-tested, rolled back, and removed without
> unstated steps.

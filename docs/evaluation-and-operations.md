# Evaluation and operations

Model training and model comparison are defined in
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

Cost reports use timestamped regional prices and raw usage. They separate model
storage and registry artifacts, Serverless Inference, Bedrock narration, Lambda,
API Gateway, DynamoDB, Amplify, S3, and CloudWatch. Release reports include
expected interactive traffic and cold-start contingencies.

## Serving validation

One manually selected policy and one action-selection profile serve the game.
Before packaging, masked argmax and temperature-one sampling play the same
held-out fixtures against the same opponents and roles. The profile with the
higher victory percentage is selected; a tie selects argmax. A sampled profile
uses the deterministic per-turn seed defined in [architecture](architecture.md).

Artifact validation checks the weight digest, manifest and image versions,
parameter count, exact state-dictionary keys and shapes, and a successful model
load using weights-only deserialization. The custom CPU container loads one
model in one worker and exposes `/ping` and `/invocations`.

Parity validation replays versioned full-game policy trajectories through the
local CPU adapter and the container. For all 24 recurrent steps it requires:

- Response versions and tensor shapes match the serving contract.
- Logits and hidden states are finite and agree with `rtol=1e-5` and
  `atol=1e-5`.
- The service resolves the same masked action from both results.
- Forced transitions advance hidden state while retaining the engine's unique
  action.

The initial on-demand Serverless configuration is 1 GB memory, maximum
concurrency five, and zero provisioned concurrency. Validation measures image
and archive size, model-load time, observed cold starts, 500 warm sequential
turns, and bursts of five concurrent turns. It records warm latency percentiles,
cold-start overhead, throttling and error behavior, peak memory, and inference
cost per turn and per 24-turn game. Memory increases one supported tier at a
time only when the measured latency improvement justifies the additional cost.

Integration cases cover asynchronous worker redelivery, SageMaker timeout and
throttling, malformed and non-finite output, a stale claimed version, and a
transaction conflict. Each case must leave either one accepted move with its
one successor hidden state or the unchanged prior turn and hidden state.

The MVP publishes one Model Registry package to one Serverless endpoint and
does not rotate it while games are active. A later model replacement uses a
maintenance boundary: game creation closes, active games finish, the endpoint
configuration changes, parity and smoke tests run, and game creation reopens.
Rollback restores the prior Registry package and endpoint configuration before
reopening.

## Deployment

AWS CDK deploys Amplify Hosting, API Gateway, FastAPI on Lambda through Mangum,
the asynchronous policy and narrator Lambdas, DynamoDB, IAM, CloudWatch, S3
storage, budget alerts, and the SageMaker Serverless endpoint integration. The
model build owns its inference image, selected artifact, and Model Registry
version.

Only the policy-worker role may invoke the account-scoped SageMaker endpoint.
The browser has access to neither SageMaker nor DynamoDB. The Serverless endpoint
does not depend on VPC placement or container-local sessions. Invocation, retry,
output, and wall-time limits are configuration. Per-IP/API rate limits and AWS
Budget alerts are active before public launch.

Logs and traces include game and turn identifiers, policy and schema versions,
endpoint latency and status, action validity, narrator model and prompt versions,
tokens, cost, and failures. They exclude hands, hidden-card locations, stock
order, recurrent hidden vectors, private prompts, and model internals.

## Local development

The pure Python engine runs without AWS. SQLite supplies local persistence,
in-memory adapters support unit tests, and local policy and narrator adapters
implement the production interfaces. A deterministic policy stub supports API
and frontend tests. The learned-policy adapter loads the same archive and
implements the same stateless JSON contract as the SageMaker container.

Verification covers engine rules, model-view encoding, API contracts, policy
integration, frontend gameplay, narrator cases, deployment smoke tests, and
seeded replay.

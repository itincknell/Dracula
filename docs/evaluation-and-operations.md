# Evaluation and operations

## Selected release controller

The user selected standalone `pi1` as the production move controller after
local play and fixed evaluation. The 754,601-parameter model runs one masked
argmax inference for every non-forced Dracula move. It is packaged inside the
Lambda image; there is no SageMaker call or production search.

The selected local artifact has SHA-256 digest
`d35196cf4513589def0ffb3c4c7c268e78652a46dab8ea41001ff2c648265203`.
Release packaging verifies this digest before building the container.

## Application verification

Final-sprint tests cover:

- Seed-and-history replay through all six rounds.
- Strict bounded parsing and rejection of impossible command histories.
- Identical cache-hit and cache-miss reconstruction.
- Deterministic retry and intentional branching from an older envelope.
- Queen/King normalization and legal representative-action masking.
- Deterministic paired-destination resolution.
- Exact engine scoring and scoring-presentation inputs.
- Browser storage reload during human, opponent, and scoring phases.
- Opening, rounds 1–5 transition, and final-result narration cues.
- Absence of a round-six transition cue.
- Bedrock delay, timeout, and failure without game-state corruption.
- No model logits, tensors, masks, or prompts in public responses.

The seed and action history are intentionally visible to the client. Tests no
longer assert that the browser cannot inspect hidden cards or stock derived from
the seed. Information-state privacy remains mandatory inside `pi1`: the model
receives only Dracula's player-relative observation.

## Deployment verification

Staging exercises:

1. `/health` under the API Gateway custom domain.
2. GitHub Pages asset loading under `/Dracula/`.
3. CORS restricted to the deployed frontend origin.
4. Cold model initialization and warm reuse.
5. Requests deliberately spread across fresh Lambda environments.
6. Complete games as Queen and King.
7. Reload and browser-storage recovery.
8. Bedrock calls at only the configured cue points.
9. Rate limits and request-size bounds.
10. Logs, latency/error metrics, alarms, and budget notification.
11. Image rollback and infrastructure teardown.

## Operational data

Normal logs contain request identity, route, duration, cache hit or miss,
history length, opponent latency, action legality, Bedrock cue class, token
usage, and failure category. Logs omit the game seed, action history, hands,
stock, model input, masks, logits, weights, and complete narrator prompt.

No game database, backup, migration, or retention process exists. Browser-local
games are outside AWS recovery. CloudWatch logs follow a declared retention
period. ECR image retention keeps the active release and rollback image.

## Cost boundary

AWS cost comes from API Gateway requests, Lambda duration and memory, ECR image
storage, CloudWatch, ACM-supported endpoints, and Bedrock input/output tokens.
There is no database or SageMaker endpoint cost. Cost validation measures cold
and warm request duration, model-load memory, narration tokens per complete
game, expected traffic, and a bounded abuse case.

## Local development

The normal local preview uses the stateless API and exact `pi1` artifact.
SQLite remains only for explicit historical-record inspection and the retained
local-stateful compatibility tests; it is not a production adapter.

```bash
make dev
make preview
make test
make test-e2e
```

The final deployment topology and remaining implementation phases are in
[deployment](deployment.md).

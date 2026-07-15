# Evaluation and operations

## Evaluation

The headless runner executes seeded games against random, heuristic, search,
model-only, and heuristic-assisted model strategies. Profile comparisons use
identical seeds. Runs can stop and resume without losing completed games.

Per-game JSONL records result, score, invalid moves, tool calls, retries, latency,
tokens, estimated cost, failures, seed, and resolved configuration. Aggregate
Markdown/CSV reports include win rate with confidence interval, score difference,
legality, cost, latency, round/seat effects, and pairwise comparisons. Raw results
are stored in S3. Narration is disabled in strategic batches unless it is being
tested.

> **TODO EVAL-001 — Define focused success criteria.** State the hypotheses,
> paired-seed protocol, sample size, and release thresholds for legality,
> completion, latency, cost, and model choice.
>
> **Complete when:** Each model/configuration decision has a metric, comparison,
> threshold, and interpretation; the runner can resume a 500-game batch.

> **TODO EVAL-002 — Define narrator and cost evaluation.** Specify public-safe
> narrator cases and criteria, plus the timestamped pricing method for models,
> tools, runtime, retries, and failures.
>
> **Complete when:** Versioned cases assess grounding, leakage, brevity,
> repetition, and tone, and raw usage deterministically produces cost estimates.

## Deployment

AWS CDK deploys Amplify Hosting, API Gateway, Lambda/FastAPI via Mangum, the
asynchronous narrator Lambda, DynamoDB, private AgentCore Runtime, IAM,
CloudWatch, S3 evaluation storage, and budget alerts. Prefer one supported US
region.

Only Lambda may invoke AgentCore. DynamoDB is not browser-accessible. Every model
move is validated. Tool calls, retries, tokens, and wall time are bounded. Add
per-IP/API rate limits before public launch and AWS Budget alerts at $25, $50,
and $90.

Logs and traces include game/turn identifiers, profile and prompt versions, seed,
move validity, tokens, latency, cost, and failures. They exclude hidden cards,
deck order, private prompts, and reasoning.

## Local development

The pure Python engine runs without AWS or LLMs against random and heuristic
bots. SQLite supplies local persistence, in-memory adapters support unit tests,
and agent/model substitutes implement the production interfaces. Verification
covers engine rules, golden examples, projections, API contracts, tools,
frontend gameplay, deployment smoke tests, and seeded runs.

> **TODO OPS-001 — Finalize environment and security choices.** Select region,
> environments, naming, IAM, endpoint protection, CORS, secrets/config handling,
> rate limits, retention enforcement, alarms, and runtime limits.
>
> **Complete when:** Versioned CDK/configuration expresses the choices and tests
> prove public callers cannot reach private data or services.

> **TODO OPS-002 — Write executable development and deployment instructions.**
> Choose Python/Node versions, package managers, commands, local adapters, CI
> gates, deployment verification, rollback, recovery, and teardown.
>
> **Complete when:** A clean checkout can run and test locally, and a clean AWS
> environment can be deployed, smoke-tested, and removed without unstated steps.

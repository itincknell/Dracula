# Evaluation and operations

Search and model comparison are defined in
[model training](model-training.md). This document covers application
verification, local operation, and the evidence required before deployment is
designed.

## Application evaluation

Deterministic tests cover dealing, legal moves, projections, scoring, state
invariants, lifecycle transitions, event replay, conditional persistence, and
idempotency. Adapter contract tests run against in-memory and SQLite storage.

Opponent integration tests cover:

- Exact agreement between authoritative legality and the opponent action table.
- Absence of opponent hand slots, hidden cards, stock order, seeds, and search
  diagnostics from public responses.
- Determinization invariance under authoritative hidden-state substitutions.
- Deterministic retry of one claimed opponent turn.
- Invalid, timed-out, or stale decisions leaving the engine unchanged.
- Forced final placements bypassing search and committing exactly once.

The local smoke test loads the frontend, passes `/health`, completes seeded
Queen and King games, reloads during every lifecycle phase, and confirms that
narration-disabled play completes without fabricated commentary.

## Opponent evaluation

The version 1 baseline, Teacher v2 gates, fixed fixtures, controls, statistical
rules, tactical cases, and later neural comparisons are defined in
[information-set search](search.md#teacher-v2-gates) and
[model training](model-training.md#absolute-evaluation-and-manual-acceptance).

Application acceptance additionally measures per-turn wall time, timeout rate,
outer simulations, shallow-response requests, candidate actions, terminal
evaluations, response-cache behavior, peak process memory, retained tree memory
after a decision, and reproducibility after retry. A controller is not eligible
for deployment until its worst supported decision budget fits a declared
request-execution profile.

The initial guided-search candidate uses 100 full-round simulations per move.
Its model, PUCT, replay, and manual-acceptance criteria are fixed in
[neural model](neural-model.md) and
[model training](model-training.md#absolute-evaluation-and-manual-acceptance).
Network value cutoffs remain experimental until they satisfy the separate gate
in [information-set search](search.md#value-cutoff-gate).

## Narrator and cost evaluation

Narrator cases use versioned public event projections. They assess factual
grounding, private-state leakage, brevity, repetition, character, cadence, and
graceful timeout or failure behavior.

Cost modeling begins after the opponent execution profile is selected. It uses
timestamped regional prices and measured request duration, memory, concurrency,
storage, narration tokens, and traffic. Release reports separate one-time model
or artifact storage from per-game compute and include cold-start contingencies.

## Deployment gate

Deployment topology remains open until controller measurements determine:

- Whether production runs search, network-guided search, or a standalone
  distilled network.
- CPU and memory required by one decision at the selected strength budget.
- Expected warm and cold latency and the concurrency needed for public traffic.
- Artifact size and whether a persistent model process is necessary.
- Retry behavior and the maximum safe execution time.

The selected design must preserve the `OpponentEngine` boundary in
[architecture](architecture.md#opponent-decision-contract). The browser never
accesses persistence or opponent compute directly. Only an application worker
may invoke the eventual opponent service.

Infrastructure, retention, observability, rate limits, budget alerts, rollback,
and teardown are specified after this gate. No AWS inference product or memory
tier is an active requirement before then.

## Privacy and observability

Public responses, browser state, narrator input, and normal logs exclude hands
other than the human's own, opponent hand slots, hidden-card assignments, stock
order, seeds, search determinizations, search trees, and model tensors.

Operational records may include game and turn IDs, public engine version,
controller and schema versions, search budget, decision latency, node count,
result status, action validity, narrator model and prompt versions, token usage,
and failures. Private diagnostic artifacts use restricted storage and are
referenced by digest rather than copied into logs.

## Local development

The deterministic engine, SQLite persistence, FastAPI service, React frontend,
and narrator-disabled mode run without AWS. In-memory adapters support unit
tests. Local gameplay can select the manually approved 32×4 Teacher v2
controller. Version 1 remains an explicit comparison control, and the failed
automated v2 report remains historical evidence. A selected policy/value
artifact can run behind guided search through the same opponent boundary. The
archived-policy adapter remains available only as the fixed PPO control.

Run the search opponent with:

```bash
make dev
```

Use `SEARCH_SIMULATIONS` and `SEARCH_EXPLORATION` to override its resolved
configuration. Run `make preview-control` to select the historical PPO control.
Run the manually approved Teacher v2 controller explicitly with:

```bash
make preview OPPONENT=search-v2 \
  SEARCH_SIMULATIONS=32 \
  SEARCH_RESPONSE_COMPLETIONS=4
```

Version 1 remains the default and no controller is selected as a silent
fallback.

Run the following to inspect a candidate without changing the default
controller:

```bash
make preview OPPONENT=guided \
  GUIDED_ARTIFACT=<path> \
  GUIDED_SIMULATIONS=<budget>
```
Missing or invalid controller configuration fails at startup; it never silently
selects another opponent.

Verification covers engine rules, information-state privacy, search replay,
API contracts, opponent integration, frontend gameplay, narrator cases, and
seeded end-to-end replay.

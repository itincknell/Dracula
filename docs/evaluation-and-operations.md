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

The search-only gates, fixed fixtures, controls, statistical rule, tactical
cases, and later neural comparisons are defined in
[information-set search](search.md#search-only-gates) and
[model training](model-training.md#search-fixtures-and-absolute-controls).

Application acceptance additionally measures per-turn wall time, timeout rate,
peak process memory, retained tree memory after a decision, and reproducibility
after retry. A controller is not eligible for deployment until its worst
supported decision budget fits a declared request-execution profile.

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
tests. Local gameplay uses the validated search controller by default. The
archived-policy adapter remains available only as the fixed PPO control.

Run the search opponent with:

```bash
make dev
```

Use `SEARCH_SIMULATIONS` and `SEARCH_EXPLORATION` to override its resolved
configuration. Run `make preview-control` to select the historical control.
Missing or invalid controller configuration fails at startup; it never silently
selects another opponent.

Verification covers engine rules, information-state privacy, search replay,
API contracts, opponent integration, frontend gameplay, narrator cases, and
seeded end-to-end replay.

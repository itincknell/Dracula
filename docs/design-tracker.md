# Design tracker

This is the index for unfinished MVP work. Tracker IDs appear only here.

## Current item

Revalidate the round-local observation and define the shared policy/value model
that will learn from the validated search teacher.

## Active sequence

```text
information state -> search engine -> search validation
    -> policy/value model -> expert iteration -> serving decision
    -> narrator -> release
```

| ID | Status | Work | Complete when |
| --- | --- | --- | --- |
| `SEARCH-001` | Completed | Implement public move projection, player-relative information state, uniform determinization, and opponent-view privacy fixtures. | Equal player views produce equal information states; every sample conserves the deck; no authoritative hidden field or opponent slot enters search |
| `SEARCH-002` | Completed | Implement deterministic POMCP-style root-sampled UCT, uniform legal opponent rollouts, exact round payoff, replay, and private diagnostics. | Exhaustive late-round fixtures agree; retries reproduce samples, visits, values, and actions |
| `SEARCH-003` | Completed | Benchmark 100, 500, and 2,000 simulations per move and run tactical and absolute-control evaluations. | Search demonstrates constructive and defensive play and its paired 95% lower bound exceeds random legal and `policy-2-v20` on fixed role-balanced fixtures |
| `MODEL-003` | Current | Revalidate the 875-bit projection and implement the feed-forward shared policy/value model. | Search gates pass; model encoding preserves the round-local information core; policy and value acceptance tests pass |
| `TRAIN-008` | Gated | Implement single-model expert iteration, replay windows, policy/value optimization, recovery, and absolute checkpoint comparison. | One reproducible local cycle improves held-out fit, survives recovery tests, and does not regress against permanent controls |
| `SERVE-002` | Gated | Compare search-only, guided-search, and standalone-model strength, latency, memory, and cost; select the production opponent profile. | One profile satisfies the competence and application latency gates with a complete deployment contract |
| `NARRATOR-002` | Deferred | Finalize the Dracula prompt, model configuration, and narrator acceptance cases in [narrator](narrator.md). | Public-only fixtures pass grounding, privacy, brevity, repetition, cadence, and tone checks |
| `RELEASE-001` | Deferred | Finalize deployment, observability, security, budgets, rollback, teardown, and public smoke acceptance in [evaluation and operations](evaluation-and-operations.md). | A clean environment can deploy, complete the smoke test, recover, roll back, and tear down without unstated steps |

## Completed foundations

| Area | Evidence |
| --- | --- |
| Product and rules | [Product and scope](product-and-scope.md) and [rules](rules.md) define the game and release behavior |
| Deterministic engine | The pure engine implements dealing, legality, scoring, lifecycle, serialization, and invariance tests |
| Local gameplay | FastAPI, SQLite, React, responsive scoring, reload recovery, and narrator-disabled completion work against the archived controller adapter |
| Search redesign | [Information-set search](search.md), [neural model](neural-model.md), [model training](model-training.md), [architecture](architecture.md), and [decisions](decisions.md) define the search-first contracts and gates |
| Search information boundary | Immutable player views, deterministic hidden-card sampling, engine-validated simulation states, per-actor projections, and privacy invariance tests implement the first search gate |
| Search validation | The 500-simulation planner passed every strategic fixture and defeated random legal play and `policy-2-v20` across the fixed 24-game role-balanced control sets |
| Historical PPO evidence | `runs/training-001` through `runs/training-004`, the final tournament, and recorded human games are retained as regression evidence; they are not active architecture |

# Design tracker

This is the index for unfinished MVP work. Tracker IDs appear only here.

## Current item

Measure whether a policy-head response ranker can replace Teacher v2's expensive
inner enumeration without reducing the approved controller's competence.

## Active sequence

```text
version 1 baseline -> shallow Teacher v2 implementation -> automated gates
    -> human testing and approval -> bounded response-ranker experiment
    -> policy/value corpus if useful -> serving decision -> narrator -> release
```

| ID | Status | Work | Complete when |
| --- | --- | --- | --- |
| `SEARCH-001` | Completed | Implement public move projection, player-relative information state, uniform determinization, and opponent-view privacy fixtures. | Equal player views produce equal information states; every sample conserves the deck; no authoritative hidden field or opponent slot enters search |
| `SEARCH-002` | Completed | Implement deterministic POMCP-style root-sampled UCT, uniform legal opponent rollouts, exact round payoff, replay, and private diagnostics. | Exhaustive late-round fixtures agree; retries reproduce samples, visits, values, and actions |
| `SEARCH-003` | Completed | Benchmark 100, 500, and 2,000 simulations per move and run tactical and absolute-control evaluations. | Search demonstrates constructive and defensive play and its paired 95% lower bound exceeds random legal and `policy-2-v20` on fixed role-balanced fixtures |
| `SEARCH-004` | Completed | Implement the information-safe nested-response prototype without changing version 1. | Actor-view invariance, world isolation, exact values, deterministic accounting, retries, and result compatibility pass |
| `SEARCH-005` | Completed | Evaluate the nested-response prototype on the defensive suite and representative latency benchmarks. | Evidence shows the prototype's small defensive gain does not justify its latency, and the prototype is removed from the active design |
| `SEARCH-006` | Completed | Replace the prototype with shallow greedy response evaluation using shared determinizations and one, two, or four uniform completions per legal action. | Candidate fairness, privacy, exact values, deterministic selection, caching, accounting, interruption, and version 1 regression tests pass |
| `SEARCH-007` | Completed | Run the constructive, defensive, privacy, absolute-performance, latency, and memory gates and retain the result. | The failed automated result is sealed without being represented as a pass |
| `SERVE-003` | Completed | Expose the 32×4 Teacher v2 adapter for local human comparison and record the user's decision. | The user completes the comparison and explicitly accepts 32×4 as the neural teacher |
| `MODEL-003` | Completed | Implement the locked feed-forward shared policy/value model over the revalidated 875-bit projection. | Exact architecture, parameter, initialization, masking, loss, invariance, gradient, and artifact tests pass |
| `TRAIN-008` | Current | Capture unique shallow-response examples, train the existing policy head with weighted pairwise ranking, and compare the resulting hybrid against approved 32×4 Teacher v2. | The observer is behaviorally inert and private; the hybrid materially lowers decision cost without a material competence regression |
| `SERVE-002` | Gated | Compare search-only, guided-search, and standalone-model strength, latency, memory, and cost; select the production opponent profile. | One profile satisfies the competence and application latency gates with a complete deployment contract |
| `NARRATOR-002` | Deferred | Finalize the Dracula prompt, model configuration, and narrator acceptance cases in [narrator](narrator.md). | Public-only fixtures pass grounding, privacy, brevity, repetition, cadence, and tone checks |
| `RELEASE-001` | Deferred | Finalize deployment, observability, security, budgets, rollback, teardown, and public smoke acceptance in [evaluation and operations](evaluation-and-operations.md). | A clean environment can deploy, complete the smoke test, recover, roll back, and tear down without unstated steps |

## Completed foundations

| Area | Evidence |
| --- | --- |
| Product and rules | [Product and scope](product-and-scope.md) and [rules](rules.md) define the game and release behavior |
| Deterministic engine | The pure engine implements dealing, legality, scoring, lifecycle, serialization, and invariance tests |
| Local gameplay | FastAPI, SQLite, React, responsive scoring, reload recovery, and narrator-disabled completion work against the validated search controller; the archived adapter remains a comparison control |
| Search redesign | [Information-set search](search.md), [neural model](neural-model.md), [model training](model-training.md), [architecture](architecture.md), and [decisions](decisions.md) define the search-first contracts and gates |
| Search information boundary | Immutable player views, deterministic hidden-card sampling, engine-validated simulation states, per-actor projections, and privacy invariance tests implement the first search gate |
| Search validation | The frozen 500-simulation version 1 planner passed every original strategic fixture and defeated random legal play and `policy-2-v20` across the fixed 24-game role-balanced control sets |
| Teacher v2 contract | [Information-set search](search.md#teacher-v2-algorithm) fixes shallow greedy response evaluation, shared candidate samples, information isolation, exact values, seeds, accounting, compatibility, and pre-human gates |
| Nested-response prototype | The rejected prototype and its validation artifacts remain historical evidence; its information boundary passed, but its defensive gain did not justify its measured latency |
| Shallow Teacher v2 | The separately versioned planner uses shared actor-local samples, exhaustive candidate evaluation, exact terminal values, deterministic caching and accounting, and preserves the version 1 golden results; see the [implementation report](../reports/teacher-v2-shallow-implementation.md) |
| Policy/value model | The 339,978-parameter feed-forward model implements typed information-state encoding, dual heads, external masking, deterministic initialization, losses, AdamW configuration, and strict artifacts |
| Teacher v2 approval | The failed automated validation remains historical evidence; completed browser play established 32 outer simulations and four response completions as the explicitly approved neural teacher |
| Teacher v2 smoke evidence | The 32×4 collector is deterministic, private, and resumable, but 32, 64, and 128 root visits remain unsuitable policy targets; the bounded response-ranker experiment replaces that target |
| Visit-budget assessment | Fixed-state 64- and 128-visit searches remain nearly uniform and unstable in early play; normalized root visits are rejected as the full-corpus policy target |
| Response-ranking dataset | The six-game smoke sealed 19,393 unique response states with exact grouped values, disjoint splits, privacy enforcement, game-boundary resume, and a reproduced dataset digest; see the [dataset report](../reports/teacher-v2-response-dataset-smoke.md) |
| Search teacher collection | Full-game collection mechanics are implemented, deterministic, private, and resumable; full policy/value collection is deferred until the bounded response-ranker experiment is measured |
| Supervised warm-start optimizer | Strict TOML, deterministic sealed-data batching, CPU/MPS selection, dual-head optimization, atomic recovery, held-out metrics, validation, and export pass the smoke and resume suites |
| Guided search | Information-safe model evaluation, PUCT priors, actor-relative value conversion, exact-terminal backup, forced-transition bypass, reproducibility, batching, and interruption tests pass; model-value cutoffs remain gated |
| Expert-iteration mechanics | Guided self-play, sealed replay sources, batched dual-head optimization, versioned accepted and candidate checkpoints, phase recovery, manual acceptance, absolute controls, and the local guided adapter pass the smoke contract |
| Historical PPO evidence | `runs/training-001` through `runs/training-004`, the final tournament, and recorded human games are retained as regression evidence; they are not active architecture |

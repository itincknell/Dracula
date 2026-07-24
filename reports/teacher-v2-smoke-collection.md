# Teacher v2 smoke collection

Follow-up measurements at 64 and 128 visits are recorded in
[Teacher v2 visit-budget smoke](teacher-v2-visit-budget-smoke.md). Neither
budget corrected the early-round target, so the recommendation below has been
completed and root visits are rejected as the full-corpus policy target.

## Assessment

The smoke corpus is deterministic, resumable, private, and contract-valid.
The approved 32×4 search profile does **not**, however, produce sufficiently
informative root-visit distributions for full-game policy distillation.
Early- and middle-round targets are nearly uniform and vary substantially
across request seeds. The full 240-game collection should not start with the
current 32-visit policy target.

This finding concerns the neural target, not the manually observed strength of
the Teacher v2 controller. Its selected move also uses root mean values as a
tie-break, while the model target contains visits only.

## Collection

```text
profile                     shallow-response-teacher-v2-32x4
outer simulations           32
response completions/action 4
workers                     4
training games              8
validation games            2
examples/game               42
training examples           336
validation examples         84
total examples              420
total root simulations      13,440
```

Training and validation fixture IDs are disjoint. Every game contains 21 Queen
and 21 King examples, for 210 examples per role across the corpus.

The initial process was interrupted after two complete games. At interruption,
the run state was `interrupted`, no temporary shard existed, and no partial
game was sealed. Resume reused 74 version-bound decision caches and completed
the remaining games. All caches were removed after their games sealed.

A separate cold-cache collection used the same fixtures and configuration but
completed games in a different process order. It produced the same dataset
digest:

```text
collection configuration  0f9ab6842b72f9b609a19fb40bf56a2166cc3a9a268a2172909bce1cec680dba
training manifest         77a50f53ee8e2507b49d60b8b7c539b1f2e9e5977022fda852a1ad6f9120eb95
validation manifest       f83fc85821da1fc184ac8259ae5f88b35cd16496aa37a652e1c885c40024a83f
resumed dataset           b231051c1ee7474bb197cb692baa7ae0803d736720a25799eb80b7399216a9cc
clean reference dataset   b231051c1ee7474bb197cb692baa7ae0803d736720a25799eb80b7399216a9cc
```

The interrupted and resumed collection took approximately 714.5 seconds
end-to-end and produced about 2,116 examples/hour. The clean reference took
719.1 seconds and produced 2,102 examples/hour. Clean decision latency was
2.19 seconds at p50 and 24.13 seconds at p95. Aggregate worker search time was
2,488 seconds.

Peak resident memory was 189.5 MiB for one clean worker. The prior four-worker
process-tree benchmark measured 847.2 MiB total; this smoke run records
per-worker rather than process-tree RSS.

## Contract verification

All ten shards passed these checks:

- `observation` is `bool[875]`.
- `legal_mask` is `bool[4,8]`.
- `search_visits` has 32 entries, sums exactly to 32, and assigns at least one
  visit to every legal root action.
- `search_policy` is the exact float32 normalization of the visits, sums to
  one, and has zero illegal mass.
- The selected action is legal and has a positive visit count.
- Engine replay reproduces every stored player-relative round return.
- Placement eight is absent.
- The shards and manifests bind the Teacher v2 profile, approval, search and
  response schemas, and configuration digests.
- No determinization, opponent hand, stock order, engine seed, search tree,
  model state, or authoritative state is sealed.
- Legacy v1 shard and decision-cache formats are rejected.
- No temporary shard or unsealed cache remains.

## Visit-target quality

The outer search visits every legal root action once before revisiting any
action. At 32 simulations, this mandatory coverage consumes much of the early
and middle target budget.

Whole-corpus results:

| Stage | Examples | Mean legal actions | Visits consumed by initial coverage | Normalized entropy | Maximum-action share | Top-two margin |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Early | 120 | 18.00 | 56.25% | 99.13% | 6.46% | 0.21% |
| Middle | 180 | 11.71 | 36.58% | 99.43% | 11.79% | 0.64% |
| Late | 120 | 3.87 | 12.08% | 99.23% | 37.32% | 4.45% |

Normalized entropy is measured relative to a uniform distribution over that
row's legal actions. Values near 100% mean that visits contain little action
preference.

Six replayed states—two each from early, middle, and late play—were searched
again under five independently derived request seeds:

| Stage | Initial coverage | Normalized entropy | Maximum-action share | Top-two margin | Top-action agreement |
| --- | ---: | ---: | ---: | ---: | ---: |
| Early | 50.00% | 99.88% | 6.88% | 0.63% | 30% |
| Middle | 46.88% | 99.58% | 9.38% | 0.00% | 70% |
| Late | 6.25% | 95.04% | 59.69% | 19.38% | 100% |

One early state selected five different actions across five seeds. The other
selected four. One middle state was stable, while the other selected four
different actions. Both late states selected the same action under every seed.

## Conclusion

The exact value targets remain valid, and late-round policy targets contain
useful preference information. The 32-visit distribution is not a suitable
general policy target: early and middle rows are dominated by mandatory action
coverage and search variance. Cross-entropy training on this corpus would
mostly teach a near-uniform legal policy rather than the controller's
strategic choice.

Follow-up measurements tested 64 and 128 simulations on the same fixed states.
Neither produced materially lower early-round entropy, wider margins, or stable
top actions. The visit-budget report supersedes increasing outer visits as the
next step.

The sealed smoke corpus is under `runs/teacher-v2-smoke-001`. Detailed
per-shard hashes and per-seed measurements are in
`.local/teacher-v2-smoke-assessment.json`; collection progress is in
`output-teacher-v2-smoke`.

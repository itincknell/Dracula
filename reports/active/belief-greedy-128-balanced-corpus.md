# BGC-128 balanced corpus

This selected-action pilot began and ended on 2026-08-23. It was stopped at a
sealed boundary when the active target changed from the single selected group
to the complete normalized root-search visit distribution.

## Configuration

```text
teacher:                  128-outer belief-greedy controller
belief completions:        8 per response action
workers:                   4 complete-game processes
rows per game:             42
rows per placement/game:   6 for each placement 1-7
trajectory profiles:       teacher, 1-deviation, 2-deviation, mixed, alternative
disk floor:                1 GiB
termination:               graceful SIGINT or disk floor
```

Every committed game is atomic and the cumulative manifest is gap-free. A
graceful interrupt finishes the active four-game batch, seals it, and stops.
Resume verifies every committed game and continues at the next ordinal.

## Identity and paths

- Corpus: `runs/belief-greedy-128-balanced-corpus-001`
- Progress log: `output-belief-greedy-128-balanced-corpus-001`
- PID: `.local/belief-greedy-128-balanced-corpus-001.pid`
- Frozen source: `.local/collector-source/belief-greedy-128-balanced-corpus-001`
- Source revision: `ec6e3d395bc89e9b0cb4bf31eb15438119281c60`
- Source-tree digest: `bcc929563f1a138c6e5fa6041dbce2d394b8e94c03a07094d3d8844511a45b17`
- Resolved-configuration digest: `3b86348b9af86164957d568fbc4a0913bafa5b7be71b0024b5a71bcd870e87c6`

## Operation

Stream sealed-batch progress:

```sh
tail -n 30 -f output-belief-greedy-128-balanced-corpus-001
```

Inspect committed state:

```sh
.venv/bin/python -m dracula.belief_greedy_miner inspect \
  --output runs/belief-greedy-128-balanced-corpus-001
```

Stop cleanly:

```sh
kill -INT "$(cat .local/belief-greedy-128-balanced-corpus-001.pid)"
```

The exact frozen-source recovery command is in
`.local/collector-source/belief-greedy-128-balanced-corpus-001/README.md`.

## Training direction

This `D0` corpus supplies one selected representative-action label per visible
state, balanced across all seven learned placements. The next standalone
policy is trained with masked categorical cross-entropy. A later explicitly
versioned `D1` experiment may use a frozen `P0` policy for simulated response
actions inside the same 128-outer search. The outer engine search and exact
terminal scoring remain intact; deployment remains standalone policy
inference.

The earlier benchmark projected approximately 246,107 rows per 24 hours at
this configuration. Live sealed-batch throughput is authoritative once enough
batches have accumulated.

The first live four-game batch sealed in 57.05 seconds with 168 rows: 24 at
each placement. Its initial extrapolation was 254,417 rows per 24 hours, close
to the independent benchmark. The first corpus-manifest digest was
`c2363bcdb35b2a87ef48b01a1593d971dd1255a147ea0c0909907ea60856bc26`.

The final retained pilot contains 524 complete games and 22,008 rows: 3,144
rows at each learned placement. Its final manifest digest is
`26b9c8e284e9bd0c4f70463c88a1a32558711802284ae9a5aec9bac07de9f57f`.
These rows retain only the selected group and will not mix with the replacement
visit-target corpus.

# BGC-128 visit-target corpus

Collection began on 2026-08-23 after the selected-action pilot stopped at a
sealed boundary. It stopped cleanly after the exact training prefix was sealed.
This replacement preserves the full root-search policy target instead of
converting the most-visited action into a one-hot label.

## Configuration

```text
teacher:                    128-outer belief-greedy controller
belief completions:          8 per response action
workers:                     4 complete-game processes
rows/game:                   42
rows/placement/game:         6 at each placement 1-7
trajectory profiles:         teacher, 1-deviation, 2-deviation, mixed, alternative
target:                      exact group visits, normalized by 128 during training
termination:                 graceful SIGINT or 1 GiB disk floor
```

Each row stores exact visits aligned with its strategic groups. The visits sum
to 128, every group has at least one root visit, and the selected group has the
maximum count. Action values, returns, hidden deals, determinizations, and
search trees remain absent.

## Identity and operation

- Corpus: `runs/bgc-128-visit-corpus-001`
- Progress: `output-bgc-128-visit-corpus-001`
- Frozen source: `.local/collector-source/bgc-128-visit-corpus-001`
- Source-tree digest: `b9e4be610583b9f461b29432bede49303593a06a30946b2b0ffc3f30d7c687ba`
- Resolved-configuration digest: `8f50986137c38d003385ad8b4dfec56162c5e5a721c4f93a1bb2cb06c0d3bb1a`
- Final committed source corpus: 12,080 games and 507,360 rows
- Training source snapshot: `runs/bgc-policy-d0-source-snapshot-001`
- Snapshot prefix: 11,905 games and 500,010 rows
- Snapshot digest: `129ed9c2673cec2db26c76c58b959d2f1221db43ec07b0e7d7903ecee51f3eee`

```sh
.venv/bin/python -m dracula.belief_greedy_miner inspect \
  --output runs/bgc-128-visit-corpus-001
```

The selected-action pilot remains sealed at
`runs/belief-greedy-128-balanced-corpus-001` with 524 games and 22,008 rows.
Its version-one schema cannot mix with this version-two corpus.

The first replacement batch sealed four games and 168 rows in 60.48 seconds,
an initial projection of 239,971 rows per 24 hours. It contains 24 rows at each
placement. Every inspected row had group visits summing exactly to 128. The
first manifest digest is
`c454ca7a4660a7bf6f998b82a0bc41d3f2196da0346e36ce926cfaccae08fa77`.

The card-set migration and active `pi0` run are recorded in
[D0 card-set migration and pi0 launch](bgc-policy-card-set-migration-001.md).

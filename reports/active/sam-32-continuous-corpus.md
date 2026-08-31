# Active Sam-32 continuous corpus

The continuous corpus is running with the resolved configuration sealed under
`runs/sam-32-continuous-corpus-001/`.

```text
teacher:                    symmetry-aware nested Sam-32
outer/response simulations: 32 / 32
workers:                    4
children:                   k = 4 when available
split cycle:                27 training / 2 validation / 2 test
minimum free disk:          1 GiB
in-flight reserve:          256 MiB per worker
```

The collector and newly spawned workers load code from
`.local/collector-source/sam-32-continuous-corpus-001/`, whose revision and
source-tree digest match the resolved configuration. The main working tree is
not on their `PYTHONPATH`.

Current counts and health are intentionally read from live state rather than
copied into this document:

```bash
make corpus-health
make corpus-inspect
make corpus-verify
```

Graceful control uses:

```bash
make corpus-stop
make corpus-resume
```

The PID file, progress log, frozen-checkout recovery command, and isolation
verification are recorded in
[`../maintenance/repo-hygiene-collector-isolation.md`](../maintenance/repo-hygiene-collector-isolation.md).
The pre-launch readiness evidence remains under
[`../history/sam-32/sam-32-continuous-corpus-readiness.md`](../history/sam-32/sam-32-continuous-corpus-readiness.md).

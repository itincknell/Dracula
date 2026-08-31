# Sam-32 continuous corpus readiness

Prepared run: `runs/sam-32-continuous-corpus-001`
Progress log: `output-sam-32-continuous-corpus-001`

## Resolved configuration

```text
teacher:                 symmetry-aware nested Sam
outer simulations:       32
response simulations:    32
maximum children:        4
workers:                  4
split cycle:              27 training, 2 validation, 2 test
mode:                     continuous
minimum free disk:        1 GiB
in-flight disk reserve:   256 MiB per worker
configuration digest:     d681a05fd1e9d53d07ee5e158166d47ccf55e1dbfb472910718712ee626caada
teacher digest:           790c5c664ecd60377b82d2da0ba62d4c343a9d017915d933d2c0a4de587bf329
source-tree digest:       c5494eb05db31d1a0a8c9765623de8b7e3742a41ef31fef574b7437ec4d89222
```

There is no configured row or deck limit. Before starting a four-worker batch,
the collector requires more than 2 GiB free: the 1 GiB floor plus four 256 MiB
in-flight reserves. While workers run, it polls the 1 GiB floor and asks them
to stop at existing subtree interruption boundaries if the floor is reached.
Only sealed decks in the contiguous fixture prefix enter the corpus manifest.

## Verification

```text
complete Python suite:      543 passed
final focused miner suite:   65 passed
static imports:              passed
package dependencies:        passed
documentation links:         passed
diff format:                 passed
```

Tests cover continuous fixture derivation, disk-floor stop, manual stop and
resume, gap-free manifest commitment, source-tree binding, and identical
continuous prefixes with one and two workers.

## Commands

Start:

```bash
mkdir -p .local
nohup /bin/zsh -c 'trap - INT; exec env PYTHONPATH=src PYTHONUNBUFFERED=1 .venv/bin/python -m dracula.sam_miner mine --output runs/sam-32-continuous-corpus-001' \
  >> output-sam-32-continuous-corpus-001 2>&1 < /dev/null &
echo $! > .local/sam-32-continuous-corpus-001.pid
```

Monitor every five seconds:

```bash
while true; do
  clear
  date
  tail -n 30 output-sam-32-continuous-corpus-001
  find runs/sam-32-continuous-corpus-001/decks -name deck-manifest.json 2>/dev/null | wc -l
  df -h .
  sleep 5
done
```

Stop cleanly:

```bash
kill -INT "$(cat .local/sam-32-continuous-corpus-001.pid)"
while kill -0 "$(cat .local/sam-32-continuous-corpus-001.pid)" 2>/dev/null; do
  sleep 1
done
```

Resume:

```bash
nohup /bin/zsh -c 'trap - INT; exec env PYTHONPATH=src PYTHONUNBUFFERED=1 .venv/bin/python -m dracula.sam_miner resume --output runs/sam-32-continuous-corpus-001' \
  >> output-sam-32-continuous-corpus-001 2>&1 < /dev/null &
echo $! > .local/sam-32-continuous-corpus-001.pid
```

Inspect, verify, or summarize after at least one deck has sealed:

```bash
PYTHONPATH=src .venv/bin/python -m dracula.sam_miner inspect --output runs/sam-32-continuous-corpus-001
PYTHONPATH=src .venv/bin/python -m dracula.sam_miner verify --output runs/sam-32-continuous-corpus-001
PYTHONPATH=src .venv/bin/python -m dracula.sam_miner summarize --output runs/sam-32-continuous-corpus-001
```

Collection has not started.

# D0 card-set migration and pi0 launch

The eligible D0 prefix was frozen, physically rewritten into the compact
card-set schema, verified, and used to launch the first BGC visit-distillation
run.

## Frozen source

```text
source corpus at snapshot: 12,040 games / 505,680 rows
final source corpus:       12,080 games / 507,360 rows
snapshot path:            runs/bgc-policy-d0-source-snapshot-001
snapshot prefix:          11,905 games / 500,010 rows
snapshot digest:          129ed9c2673cec2db26c76c58b959d2f1221db43ec07b0e7d7903ecee51f3eee
training:                  10,715 games / 450,030 rows
validation:                 1,190 games /  49,980 rows
```

Each placement contributes 64,290 training and 7,140 validation rows. The D0
collector stopped gracefully after snapshot creation; its additional 175
sealed games remain preserved outside this experiment.

## Physical migration

```text
path:                      runs/bgc-policy-d0-card-set-001
corpus manifest digest:    6da6988aea6d720afb07324ea4258e08c821494d9430789d2c917fcc0af7d5a2
observation schema:        dracula-observation-card-set-v2
action schema:             dracula-card-candidate-action-map-v2
size:                      469 MiB
```

All 500,010 rows were rewritten. The migration removed the redundant 216
stable-slot bits, retained the 659-bit player-visible card-set projection, and
relabeled masks, groups, representatives, visits, and audit selections into
canonical current-card candidate rows. It preserved every 128-visit total,
game boundary, split assignment, role, dealer state, and placement count.

## Model and run

```text
model:                     BGCPolicyModel
parameters:                754,601 float32
input:                     bool[659]
output:                    float32[4,8]
objective:                 representative-masked visit-distribution cross-entropy
training rows:             450,030
validation rows:            49,980
run:                       runs/bgc-policy-pi0-001
progress:                  output-bgc-policy-pi0-001
configuration:             configs/bgc-policy-pi0.toml
```

The run used the locked AdamW configuration on CPU and early-stopped after 19
epochs. Epoch 14 was selected with held-out cross-entropy `2.00077879`.
Validation agreement with the most-visited BGC action was 43.06% at top one,
64.35% at top two, and 75.63% at top three. The exported unaccepted artifact is
`runs/bgc-policy-pi0-001/artifacts/unaccepted-candidate.pt`, with SHA-256
`f7709ca53edc2e64c7f396a6dcfad69950618afa77f3d33a46796e41c95a73b5`.
Gameplay evaluation and acceptance remain separate operations.

```bash
.venv/bin/dracula-bgc-policy-training train \
  --config configs/bgc-policy-pi0.toml

.venv/bin/dracula-bgc-policy-training resume \
  --run runs/bgc-policy-pi0-001

.venv/bin/dracula-bgc-policy-training validate \
  --run runs/bgc-policy-pi0-001
```

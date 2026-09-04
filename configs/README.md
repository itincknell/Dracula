# Configurations

Completed collection and training configurations are immutable and stored with
their ignored runs. Local gameplay is configured through the documented
`Makefile` variables; standalone `pi1` is the default.

`bgc-policy-pi0.toml` and `bgc-policy-pi1.toml` retain the completed BGC
visit-distillation configurations.

`bgc-d1-template.toml` retains the completed D1 controller, schema, worker, and
disk-guard settings for reproducibility.

`archive/` retains superseded experiment configurations needed to interpret or
reproduce historical evidence. They are not active defaults. Six maintained
data and evaluation commands are installed from `pyproject.toml`:

- `dracula-belief-greedy-miner`
- `dracula-bgc-policy-evaluation`
- `dracula-bgc-policy-training`
- `dracula-bgc-pi0-miner`
- `dracula-sam-miner`
- `dracula-sam-policy-training`

Other historical commands remain available explicitly through
`python -m dracula.<module>`.

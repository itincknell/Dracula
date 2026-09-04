# Tools

Active operator commands live in the package entry points and root `Makefile`.
The maintained root utilities have narrow release or provenance roles:

- `build_lambda_context.py` verifies and stages the selected `pi1` artifact.
- `validate_lambda_container.py` exercises the production container locally.
- `build_release_candidate.py` records source, image, infrastructure, and
  frontend identities.
- `build_pi1_dataset.py`, `evaluate_sam_policy.py`,
  `run_pi1_incremental_matchups.py`, and `run_sam_policy_training_smoke.py`
  retain current training-lineage reproduction and evaluation behavior.

`archive/` contains one-off historical experiment and report utilities. It is
not part of the active deployment command surface.

# Tools

Active operator commands live in the package entry points and root `Makefile`.
The maintained root utilities have narrow release or provenance roles:

- `build_lambda_context.py` verifies and stages the selected `pi1` artifact.
- `validate_lambda_container.py` exercises the production container locally.
- `build_release_candidate.py` records source, image, infrastructure, and
  frontend identities.
Retired mining, evaluation, and one-off report utilities remain available from
Git history rather than the installed application.

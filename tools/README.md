# Tools

Active operator commands live in the package entry points and root `Makefile`.
The maintained root utilities have narrow release or provenance roles:

- `build_lambda_context.py` stages the verified policy, runtime import closure,
  and built frontend. Run `make lambda-context` to build the frontend first.
- `validate_lambda_container.py` exercises the production container locally.
- `lambda_validation_gameplay.py` supplies its focused stateless HTTP game
  exerciser; it is not a separate operator command.
- `build_release_candidate.py` records source, image, infrastructure, and
  frontend identities.

See [deployment](../docs/deployment.md) for combined-image commands and
[the Worker](../deployment/cloudflare/README.md) for the prepared public route.
Retired mining, evaluation, and one-off report utilities remain available from
Git history rather than the installed application.

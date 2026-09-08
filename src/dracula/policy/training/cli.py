"""Expose the standalone visit-distillation trainer as a command-line surface.

Argument parsing and terminal output belong here rather than in the training
coordinator. The commands delegate to verified training functions and translate
known configuration, artifact, and interruption failures into stable exit codes.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from dracula.policy.model import PolicyModelError
from dracula.policy.training.config import (
    config_from_resolved,
    load_policy_training_config,
)
from dracula.policy.training.contracts import (
    PolicyTrainingError,
    PolicyTrainingInterrupted,
)


def _parser() -> argparse.ArgumentParser:
    """Define the supported inspect, train, resume, validate, and export commands."""

    parser = argparse.ArgumentParser(
        description="Information-set UCT visit-distribution policy distillation"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser(
        "inspect", help="verify and inspect a sealed policy snapshot"
    )
    inspect.add_argument("--snapshot", required=True)
    smoke = commands.add_parser(
        "smoke", help="run bounded synthetic-fixture optimization"
    )
    smoke.add_argument("--config", required=True)
    smoke.add_argument("--epochs", type=int, default=2)
    train = commands.add_parser("train", help="start real snapshot optimization")
    train.add_argument("--config", required=True)
    resume = commands.add_parser("resume", help="resume the latest checkpoint")
    resume.add_argument("--run", required=True)
    validate = commands.add_parser(
        "validate", help="evaluate the best unaccepted candidate"
    )
    validate.add_argument("--run", required=True)
    export = commands.add_parser(
        "export", help="verify or copy the unaccepted candidate artifact"
    )
    export.add_argument("--run", required=True)
    export.add_argument("--output")
    return parser


def _run_command(arguments: argparse.Namespace) -> object:
    """Dispatch one parsed command without mixing terminal concerns into training."""

    # The late import keeps this interface from participating in the training
    # module's import graph when callers only request command help.
    from dracula.policy.training.runner import (
        export_policy_run,
        inspect_policy_snapshot,
        train_policy,
        validate_policy_run,
    )

    if arguments.command == "inspect":
        return inspect_policy_snapshot(arguments.snapshot)
    if arguments.command in {"train", "smoke"}:
        config = load_policy_training_config(arguments.config)
        return asdict(
            train_policy(
                config,
                smoke_epochs=(
                    arguments.epochs if arguments.command == "smoke" else None
                ),
            )
        )
    if arguments.command == "resume":
        output = Path(arguments.run).expanduser().resolve()
        config, smoke_epochs = config_from_resolved(output)
        return asdict(
            train_policy(config, resume=True, smoke_epochs=smoke_epochs)
        )
    if arguments.command == "validate":
        return asdict(validate_policy_run(arguments.run))
    return {
        "artifact": str(export_policy_run(arguments.run, arguments.output))
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Run the training CLI and return its process exit status."""

    arguments = _parser().parse_args(argv)
    try:
        result = _run_command(arguments)
    except PolicyTrainingInterrupted as error:
        print(json.dumps({"status": "interrupted", "message": str(error)}))
        return 75
    except (PolicyTrainingError, PolicyModelError) as error:
        print(str(error), file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

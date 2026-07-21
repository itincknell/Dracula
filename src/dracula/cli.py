"""Minimal command line interface for local training operations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dracula.training import TrainingSuite, create_training_run


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dracula-train")
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="create and execute a configured run")
    run.add_argument("config", type=Path)

    resume = commands.add_parser("resume", help="resume a run from its committed phase")
    resume.add_argument("run_directory", type=Path)

    evaluate = commands.add_parser("evaluate", help="evaluate the active population")
    evaluate.add_argument("run_directory", type=Path)

    archive = commands.add_parser("archive", help="archive one active policy")
    archive.add_argument("run_directory", type=Path)
    archive.add_argument("policy_id")
    archive.add_argument("--output", type=Path)

    replace = commands.add_parser("replace", help="randomly replace one active policy")
    replace.add_argument("run_directory", type=Path)
    replace.add_argument("policy_id")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "run":
        suite = create_training_run(arguments.config)
        state = suite.run()
        print(json.dumps(state, indent=2, sort_keys=True))
    elif arguments.command == "resume":
        suite = TrainingSuite.open(arguments.run_directory)
        state = suite.resume()
        print(json.dumps(state, indent=2, sort_keys=True))
    elif arguments.command == "evaluate":
        result = TrainingSuite.open(arguments.run_directory).evaluate_current()
        print(
            json.dumps(
                {
                    "games": len(result.fixture_results),
                    "policies": [
                        {
                            "policy_id": item.policy.policy_id,
                            "version": item.policy.version,
                            "victory_percentage": item.overall.victory_percentage,
                        }
                        for item in result.policy_metrics
                    ],
                    "critic_validation": {
                        "rows": result.critic_validation.row_count,
                        "critic_mse": result.critic_validation.critic_mse,
                        "zero_predictor_mse": result.critic_validation.zero_predictor_mse,
                        "passed": result.critic_validation.passed,
                    },
                },
                indent=2,
                sort_keys=True,
            )
        )
    elif arguments.command == "archive":
        path = TrainingSuite.open(arguments.run_directory).archive_policy(
            arguments.policy_id, arguments.output
        )
        print(path)
    elif arguments.command == "replace":
        identity = TrainingSuite.open(arguments.run_directory).replace_policy(
            arguments.policy_id
        )
        print(f"{identity.policy_id} {identity.version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

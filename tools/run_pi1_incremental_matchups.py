#!/usr/bin/env python3
"""Run the three requested π1 matchups with atomic per-deck progress."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path

from dracula.bgc_policy_evaluation import (
    BGCSearchOpponent,
    EvaluationProtocol,
    StandalonePi0Opponent,
    _comparison_metrics,
    _play_game,
    _record_from_json,
    create_evaluation_fixture_manifest,
)
from dracula.engine import EnginePlayer
from dracula.search import BeliefGreedySearchConfig


SCHEMA = "dracula-pi1-incremental-matchup-v1"
ROOT_SEED = "dracula-pi1-evaluation-001-v1"


def canonical(value: object) -> bytes:
    return json.dumps(
        value, allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def digest(value: object) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    try:
        with temporary.open("wb") as handle:
            handle.write(canonical(value) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def run(matchup: str, pi1_path: Path, pi0_path: Path, output: Path) -> None:
    protocol = EvaluationProtocol(
        primary_deck_count=12,
        extension_deck_count=1,
        bootstrap_samples=2_000,
        outer_simulation_budget=128,
        belief_completion_count=8,
    )
    fixtures = create_evaluation_fixture_manifest(
        output.parent / "fixtures.json", root_seed=ROOT_SEED, protocol=protocol
    )
    pi1 = StandalonePi0Opponent.from_artifact(pi1_path)
    pi0 = StandalonePi0Opponent.from_artifact(pi0_path)
    config = BeliefGreedySearchConfig(
        outer_simulation_budget=128, belief_completion_count=8
    )
    controls = {
        "pi1-vs-pi1-bgc": BGCSearchOpponent.pi0(pi1, config),
        "pi1-vs-pi0-bgc": BGCSearchOpponent.pi0(pi0, config),
        "pi1-vs-pi0": pi0,
    }
    if matchup not in controls:
        raise ValueError(f"unknown matchup: {matchup}")
    control = controls[matchup]
    pi1_digest = file_digest(pi1_path)
    pi0_digest = file_digest(pi0_path)
    records = []
    if output.exists():
        document = json.loads(output.read_text(encoding="utf-8"))
        content = document.get("content")
        if (
            not isinstance(content, dict)
            or document.get("content_digest") != digest(content)
            or content.get("schema_version") != SCHEMA
            or content.get("matchup") != matchup
            or content.get("pi1_artifact_digest") != pi1_digest
            or content.get("pi0_artifact_digest") != pi0_digest
            or content.get("fixture_manifest_digest") != fixtures.digest
        ):
            raise ValueError("existing π1 matchup progress is incompatible")
        records = [_record_from_json(value) for value in content["games"]]
    for fixture in fixtures.fixtures[len(records) // 2 : 12]:
        for role in EnginePlayer:
            records.append(
                _play_game(
                    fixture,
                    subject=pi1,
                    control=control,
                    subject_role=role,
                    should_stop=None,
                )
            )
        metrics = _comparison_metrics(
            records,
            comparison_id=matchup,
            fixture_manifest_digest=fixtures.digest,
            bootstrap_samples=protocol.bootstrap_samples,
        )
        content = {
            "complete": len(records) == 24,
            "completed_deck_count": len(records) // 2,
            "control": {"digest": control.digest, "identity": control.identity},
            "fixture_manifest_digest": fixtures.digest,
            "games": [asdict(record) for record in records],
            "matchup": matchup,
            "metrics": metrics,
            "pi0_artifact_digest": pi0_digest,
            "pi1_artifact_digest": pi1_digest,
            "schema_version": SCHEMA,
            "subject": {"digest": pi1.digest, "identity": pi1.identity},
            "target_deck_count": 12,
        }
        atomic(output, {"content": content, "content_digest": digest(content)})
        print(
            json.dumps(
                {
                    "completed_decks": len(records) // 2,
                    "matchup": matchup,
                    "paired_score_differential": metrics[
                        "paired_score_differential"
                    ],
                    "paired_score_differential_ci_95": metrics[
                        "paired_score_differential_ci_95"
                    ],
                    "queen_score_differential": metrics["role_splits"]["queen"][
                        "mean_score_differential"
                    ],
                    "king_score_differential": metrics["role_splits"]["king"][
                        "mean_score_differential"
                    ],
                },
                sort_keys=True,
            ),
            flush=True,
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--matchup",
        choices=("pi1-vs-pi1-bgc", "pi1-vs-pi0-bgc", "pi1-vs-pi0"),
        required=True,
    )
    parser.add_argument("--pi1", required=True, type=Path)
    parser.add_argument("--pi0", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run(args.matchup, args.pi1.resolve(), args.pi0.resolve(), args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

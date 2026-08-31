#!/usr/bin/env python3
"""Seal a card-set policy-training view over a completed D1 corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from dracula.bgc_policy_migration import (
    ACTION_SCHEMA_VERSION,
    CORPUS_SCHEMA_VERSION,
    GAME_SCHEMA_VERSION,
    LOADER_SCHEMA_VERSION,
    MIGRATION_SCHEMA_VERSION,
    OBSERVATION_SCHEMA_VERSION,
    ROW_SCHEMA_VERSION,
)
from dracula.randomness import derive_seed


SPLIT_NAMESPACE = "dracula-bgc-pi1-block-split-v1"
SNAPSHOT_SCHEMA = "dracula-bgc-pi1-d1-snapshot-v1"


def canonical(value: object) -> bytes:
    return json.dumps(
        value, allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def digest(value: object) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, value: object) -> None:
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


def verified(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(value, dict)
        or set(value) != {"content", "content_digest"}
        or not isinstance(value["content"], dict)
        or value["content_digest"] != digest(value["content"])
    ):
        raise ValueError(f"artifact failed digest verification: {path}")
    return value


def build(source: Path, output: Path) -> dict[str, object]:
    source = source.resolve()
    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("π1 dataset output must be empty")
    manifest = verified(source / "corpus-manifest.json")
    resolved = verified(source / "resolved-config.json")
    entries = manifest["content"]["games"]
    if not isinstance(entries, list) or len(entries) < 10:
        raise ValueError("D1 corpus has too few complete games")
    usable_game_count = len(entries) - len(entries) % 5
    block_count = usable_game_count // 5
    validation_block_count = max(1, round(block_count * 0.10))
    ranked_blocks = sorted(
        range(block_count),
        key=lambda block: (
            derive_seed(
                SPLIT_NAMESPACE,
                str(manifest["content_digest"]),
                str(block),
            ),
            block,
        ),
    )
    validation_blocks = set(ranked_blocks[:validation_block_count])
    source_content = resolved["content"]
    snapshot_content = {
        "corpus_manifest_digest": manifest["content_digest"],
        "excluded_trailing_games": len(entries) - usable_game_count,
        "source_revision": source_content["source_revision"],
        "source_tree_digest": source_content["source_tree_digest"],
        "snapshot_schema_version": SNAPSHOT_SCHEMA,
        "source_corpus_directory": str(source),
        "split_namespace": SPLIT_NAMESPACE,
        "training_block_count": block_count - validation_block_count,
        "validation_block_count": validation_block_count,
        "usable_game_count": usable_game_count,
    }
    snapshot = {
        "content": snapshot_content,
        "content_digest": digest(snapshot_content),
    }
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "source-snapshot.json", snapshot)
    games: list[dict[str, object]] = []
    placement_counts = {
        overlay: {str(placement): 0 for placement in range(1, 8)}
        for overlay in ("training", "validation")
    }
    for ordinal, entry in enumerate(entries[:usable_game_count]):
        source_path = source / str(entry["path"])
        document = verified(source_path)
        content = document["content"]
        rows = content.get("rows")
        if (
            content.get("ordinal") != ordinal
            or not isinstance(rows, list)
            or len(rows) != 42
        ):
            raise ValueError(f"D1 game {ordinal} is malformed")
        overlay = "validation" if ordinal // 5 in validation_blocks else "training"
        projected_rows = []
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError(f"D1 game {ordinal} has a malformed row")
            projected = dict(row)
            projected["row_schema_version"] = ROW_SCHEMA_VERSION
            projected_rows.append(projected)
            placement_counts[overlay][str(projected["placement_number"])] += 1
        projected_content = {
            "fixture_id": content["fixture_id"],
            "game_schema_version": GAME_SCHEMA_VERSION,
            "ordinal": ordinal,
            "overlay": overlay,
            "rows": projected_rows,
            "source_content_digest": document["content_digest"],
            "trajectory_profile": content["trajectory_profile"],
        }
        projected_document = {
            "content": projected_content,
            "content_digest": digest(projected_content),
        }
        relative = f"games/{ordinal:06d}.json"
        destination = output / relative
        atomic_json(destination, projected_document)
        games.append(
            {
                "content_digest": projected_document["content_digest"],
                "file_digest": file_digest(destination),
                "fixture_id": content["fixture_id"],
                "ordinal": ordinal,
                "overlay": overlay,
                "path": relative,
                "row_count": 42,
                "trajectory_profile": content["trajectory_profile"],
            }
        )
    corpus_content = {
        "action_schema_version": ACTION_SCHEMA_VERSION,
        "corpus_schema_version": CORPUS_SCHEMA_VERSION,
        "game_count": len(games),
        "games": games,
        "loader_schema_version": LOADER_SCHEMA_VERSION,
        "migration_schema_version": MIGRATION_SCHEMA_VERSION,
        "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
        "placement_counts": placement_counts,
        "row_count": len(games) * 42,
        "source_snapshot_digest": snapshot["content_digest"],
        "source_snapshot_path": str((output / "source-snapshot.json").resolve()),
    }
    corpus = {"content": corpus_content, "content_digest": digest(corpus_content)}
    atomic_json(output / "corpus-manifest.json", corpus)
    return {
        "corpus_digest": corpus["content_digest"],
        "excluded_games": len(entries) - usable_game_count,
        "games": len(games),
        "rows": len(games) * 42,
        "training_rows": sum(placement_counts["training"].values()),
        "validation_rows": sum(placement_counts["validation"].values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    print(json.dumps(build(arguments.source, arguments.output), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

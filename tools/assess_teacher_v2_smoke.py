"""Validate a sealed Teacher v2 smoke corpus and measure target concentration."""

from __future__ import annotations

import argparse
import copy
import json
import math
import resource
import tempfile
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import torch

import dracula.teacher as teacher
from dracula.bridge import PolicyTurnKind, build_policy_turn_context
from dracula.engine import (
    EnginePlayer,
    EngineStatus,
    advance_after_round,
    apply_move,
    create_game,
)
from dracula.randomness import derive_seed, seed_hex
from dracula.search import (
    GREEDY_RESPONSE_SCHEMA_VERSION,
    STRATEGIC_SEARCH_SCHEMA_VERSION,
    StrategicInformationSetSearch,
    StrategicSearchConfig,
    derive_strategic_search_request_seed,
    information_state_from_engine,
)
from dracula.teacher import (
    APPROVED_TEACHER_PROFILE,
    EXAMPLES_PER_GAME,
    FixtureSplit,
    TeacherCollectionError,
    derive_teacher_fixture,
    inspect_teacher_dataset,
    load_teacher_config,
    load_teacher_manifest,
    load_teacher_shard,
)

QUALITY_SEED_NAMESPACE = "dracula-teacher-target-quality-fixture-v1"
_FORBIDDEN_KEYS = {
    "authoritative_state",
    "determination",
    "determinization",
    "engine_seed",
    "game_seed",
    "hands",
    "model_state",
    "opponent_hand",
    "policy_hidden_state",
    "sampled_stock",
    "search_tree",
    "stock",
    "tree",
}


def _artifact_keys(value: object):
    if isinstance(value, dict):
        for key, nested in value.items():
            yield str(key)
            yield from _artifact_keys(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            yield from _artifact_keys(nested)


def _fixture_shards(output: Path):
    for split in (FixtureSplit.TRAINING, FixtureSplit.VALIDATION):
        manifest = load_teacher_manifest(output, split)["manifest"]
        for entry in manifest["shards"]:
            yield split, entry, load_teacher_shard(output / entry["relative_path"])


def _replay_and_validate_returns(config, split, shard) -> dict[tuple[int, int], object]:
    fixture_index = int(shard["metadata"]["fixture_index"])
    fixture = derive_teacher_fixture(config.root_seed, split, fixture_index)
    columns = shard["columns"]
    rows = {
        (round_number, placement_number): index
        for index, (round_number, placement_number) in enumerate(
            zip(
                columns["round_numbers"].tolist(),
                columns["placement_numbers"].tolist(),
                strict=True,
            )
        )
    }
    captured: dict[tuple[int, int], object] = {}
    state = create_game(fixture._game_seed)
    while state.status is not EngineStatus.GAME_COMPLETE:
        while state.status is EngineStatus.PLAYING:
            placement = len(state.current_round_moves) + 1
            key = (state.round_number, placement)
            if placement in {1, 4, 7}:
                captured[key] = information_state_from_engine(state)
            context = build_policy_turn_context(state, state.active_player)
            if context.kind is PolicyTurnKind.FORCED_RECURRENT_TRANSITION:
                move = context.forced_move
            else:
                row = rows[key]
                move = context.action_table[
                    int(columns["selected_action_indices"][row].item())
                ]
            if move is None:
                raise AssertionError("replay selected no engine move")
            state = apply_move(state, move).state
        result = state.pending_round_result
        if result is None:
            raise AssertionError("replayed round has no result")
        queen_return = (
            result.round_scores[EnginePlayer.QUEEN]
            - result.round_scores[EnginePlayer.KING]
        ) / 150.0
        for player, expected in (
            (EnginePlayer.QUEEN, queen_return),
            (EnginePlayer.KING, -queen_return),
        ):
            expected_float32 = torch.tensor(expected, dtype=torch.float32).item()
            indexes = [
                index
                for index, (round_number, row_player) in enumerate(
                    zip(
                        columns["round_numbers"].tolist(),
                        columns["players"],
                        strict=True,
                    )
                )
                if round_number == state.round_number
                and row_player == player.value
            ]
            if not all(
                columns["round_returns"][index].item() == expected_float32
                for index in indexes
            ):
                raise AssertionError("sealed return differs from engine score")
        state = advance_after_round(state)
    return captured


def _validate_v1_rejection(config, shard) -> dict[str, bool]:
    results = {"shard": False, "cache": False}
    with tempfile.TemporaryDirectory(prefix="dracula-v1-rejection-") as directory:
        changed = copy.deepcopy(shard)
        changed["format_version"] = teacher.LEGACY_TEACHER_SHARD_FORMAT_VERSION
        path = Path(directory) / "legacy.pt"
        torch.save(changed, path)
        try:
            load_teacher_shard(path)
        except TeacherCollectionError:
            results["shard"] = True

    columns = shard["columns"]
    fixture = derive_teacher_fixture(
        config.root_seed,
        FixtureSplit(shard["metadata"]["split"]),
        int(shard["metadata"]["fixture_index"]),
    )
    value = {
        "format_version": teacher.LEGACY_TEACHER_CACHE_FORMAT_VERSION,
        "teacher_profile": APPROVED_TEACHER_PROFILE,
        "teacher_profile_digest": config.teacher_profile_digest,
        "approval_identity": teacher.TEACHER_APPROVAL_IDENTITY,
        "validation_report_digest": teacher.TEACHER_VALIDATION_REPORT_DIGEST,
        "fixture_id": fixture.fixture_id,
        "split": fixture.split.value,
        "round_number": int(columns["round_numbers"][0].item()),
        "placement_number": int(columns["placement_numbers"][0].item()),
        "information_state_digest": columns["information_state_digests"][0],
        "search_schema_version": STRATEGIC_SEARCH_SCHEMA_VERSION,
        "response_schema_version": GREEDY_RESPONSE_SCHEMA_VERSION,
        "search_config_digest": config.search_config.digest,
        "response_config_digest": config.search_config.response_config.digest,
        "action_visits": columns["search_visits"][0].tolist(),
        "selected_action_index": int(columns["selected_action_indices"][0].item()),
        "search_seconds": 0.0,
    }
    try:
        teacher._validate_decision_cache(
            value,
            fixture=fixture,
            config=config,
            information_digest=columns["information_state_digests"][0],
            legal_mask=columns["legal_masks"][0],
            round_number=int(columns["round_numbers"][0].item()),
            placement_number=int(columns["placement_numbers"][0].item()),
        )
    except TeacherCollectionError:
        results["cache"] = True
    return results


def _validate_dataset(output: Path, reference: Path | None) -> tuple[dict, dict]:
    config = load_teacher_config(output)
    inspection = inspect_teacher_dataset(output)
    if (inspection.training_games, inspection.validation_games) != (8, 2):
        raise AssertionError("smoke split counts differ from 8/2")
    if inspection.examples != 420:
        raise AssertionError("smoke corpus does not contain exactly 420 examples")
    if inspection.simulations != 420 * 32:
        raise AssertionError("smoke simulation total differs")

    fixture_ids: dict[str, set[str]] = {}
    player_counts = Counter()
    all_keys: set[str] = set()
    captured_by_fixture: dict[tuple[str, int], dict[tuple[int, int], object]] = {}
    sealed_target_rows: list[dict[str, float | int | str]] = []
    first_shard = None
    shard_digests: list[str] = []
    for split, entry, shard in _fixture_shards(output):
        if first_shard is None:
            first_shard = shard
        columns = shard["columns"]
        metadata = shard["metadata"]
        if columns["observations"].shape != (EXAMPLES_PER_GAME, 875):
            raise AssertionError("observation shape differs")
        if columns["observations"].dtype is not torch.bool:
            raise AssertionError("observation dtype differs")
        if columns["legal_masks"].shape != (EXAMPLES_PER_GAME, 4, 8):
            raise AssertionError("legal-mask shape differs")
        if columns["legal_masks"].dtype is not torch.bool:
            raise AssertionError("legal-mask dtype differs")
        if columns["search_visits"].shape != (EXAMPLES_PER_GAME, 32):
            raise AssertionError("visit shape differs")
        if not torch.all(columns["search_visits"].sum(dim=1) == 32):
            raise AssertionError("visit totals differ")
        if not torch.equal(
            columns["search_policies"],
            columns["search_visits"].to(torch.float32) / 32.0,
        ):
            raise AssertionError("normalized policy differs from exact visits")
        masks = columns["legal_masks"].flatten(start_dim=1)
        selected = columns["selected_action_indices"]
        if torch.any(columns["search_visits"][~masks] != 0):
            raise AssertionError("visits contain illegal mass")
        if torch.any(columns["search_visits"][masks] == 0):
            raise AssertionError("initial root-action coverage is incomplete")
        if not torch.all(masks.gather(1, selected[:, None]).squeeze(1)):
            raise AssertionError("selected action is illegal")
        if 8 in columns["placement_numbers"].tolist():
            raise AssertionError("forced placement entered the dataset")
        player_counts.update(columns["players"])
        fixture_ids.setdefault(split.value, set()).add(metadata["fixture_id"])
        all_keys.update(_artifact_keys(shard))
        for key, expected in {
            "teacher_profile": APPROVED_TEACHER_PROFILE,
            "teacher_profile_digest": config.teacher_profile_digest,
            "search_schema_version": STRATEGIC_SEARCH_SCHEMA_VERSION,
            "response_schema_version": GREEDY_RESPONSE_SCHEMA_VERSION,
            "search_config_digest": config.search_config.digest,
            "response_config_digest": config.search_config.response_config.digest,
            "collection_config_digest": config.digest,
        }.items():
            if metadata[key] != expected:
                raise AssertionError(f"shard {key} differs")
        fixture_index = int(metadata["fixture_index"])
        fixture = derive_teacher_fixture(
            config.root_seed, split, fixture_index
        )
        shard_path = output / str(entry["relative_path"])
        if fixture._game_seed.encode("utf-8") in shard_path.read_bytes():
            raise AssertionError("private game seed appears in a sealed shard")
        captured_by_fixture[(split.value, fixture_index)] = (
            _replay_and_validate_returns(config, split, shard)
        )
        shard_digests.append(shard["content_digest"])
        for index in range(EXAMPLES_PER_GAME):
            row_visits = tuple(
                int(value) for value in columns["search_visits"][index].tolist()
            )
            nonzero = sorted(
                (count for count in row_visits if count), reverse=True
            )
            legal_count = int(masks[index].sum().item())
            entropy = _entropy(row_visits)
            placement = int(columns["placement_numbers"][index].item())
            stage = (
                "early"
                if placement <= 2
                else "middle"
                if placement <= 5
                else "late"
            )
            sealed_target_rows.append(
                {
                    "stage": stage,
                    "legal_action_count": legal_count,
                    "initial_coverage_visit_share": legal_count / 32.0,
                    "visit_entropy_nats": entropy,
                    "normalized_visit_entropy": (
                        entropy / math.log(legal_count)
                        if legal_count > 1
                        else 0.0
                    ),
                    "maximum_visit_share": nonzero[0] / 32.0,
                    "top_two_visit_margin": (
                        (nonzero[0] - nonzero[1]) / 32.0
                        if len(nonzero) > 1
                        else 1.0
                    ),
                }
            )

    if player_counts != Counter({"queen": 210, "king": 210}):
        raise AssertionError("Queen and King examples are not balanced")
    if not fixture_ids["training"].isdisjoint(fixture_ids["validation"]):
        raise AssertionError("training and validation fixtures overlap")
    if _FORBIDDEN_KEYS.intersection(all_keys):
        raise AssertionError("sealed artifacts contain a forbidden private key")
    if first_shard is None:
        raise AssertionError("smoke corpus contains no shard")
    rejection = _validate_v1_rejection(config, first_shard)
    if not all(rejection.values()):
        raise AssertionError("v1 artifact rejection failed")

    reference_digest = None
    if reference is not None:
        reference_inspection = inspect_teacher_dataset(reference)
        reference_digest = reference_inspection.dataset_digest
        if reference_digest != inspection.dataset_digest:
            raise AssertionError("resumed and clean dataset digests differ")

    sealed_target_summary = {}
    for stage in ("early", "middle", "late"):
        rows = [row for row in sealed_target_rows if row["stage"] == stage]
        sealed_target_summary[stage] = {
            "example_count": len(rows),
            **{
                key: sum(float(row[key]) for row in rows) / len(rows)
                for key in (
                    "legal_action_count",
                    "initial_coverage_visit_share",
                    "visit_entropy_nats",
                    "normalized_visit_entropy",
                    "maximum_visit_share",
                    "top_two_visit_margin",
                )
            },
        }

    return (
        {
            "inspection": {
                **asdict(inspection),
                "metrics": asdict(inspection.metrics),
            },
            "reference_dataset_digest": reference_digest,
            "shard_content_digests": shard_digests,
            "split_fixture_counts": {
                key: len(value) for key, value in fixture_ids.items()
            },
            "player_examples": dict(player_counts),
            "sealed_target_summary": sealed_target_summary,
            "forbidden_key_intersection": sorted(
                _FORBIDDEN_KEYS.intersection(all_keys)
            ),
            "v1_rejection": rejection,
        },
        captured_by_fixture,
    )


def _entropy(visits: tuple[int, ...]) -> float:
    total = sum(visits)
    return -sum(
        (count / total) * math.log(count / total)
        for count in visits
        if count
    )


def _measure_target_quality(
    config,
    captured,
    request_seed_count: int,
    search_config: StrategicSearchConfig,
) -> dict:
    target_specs = (
        ("training", 0, 1, 1, "early"),
        ("validation", 0, 4, 1, "early"),
        ("training", 0, 1, 4, "middle"),
        ("validation", 0, 4, 4, "middle"),
        ("training", 0, 1, 7, "late"),
        ("validation", 0, 4, 7, "late"),
    )
    records = []
    started = time.perf_counter()
    for split, fixture_index, round_number, placement, stage in target_specs:
        information = captured[(split, fixture_index)][
            (round_number, placement)
        ]
        legal_count = sum(
            bool(value)
            for row in information.legal_mask
            for value in row
        )
        state_runs = []
        fixture = derive_teacher_fixture(
            config.root_seed, FixtureSplit(split), fixture_index
        )
        for seed_index in range(request_seed_count):
            quality_fixture_id = seed_hex(
                derive_seed(
                    QUALITY_SEED_NAMESPACE,
                    fixture.fixture_id,
                    stage,
                    str(round_number),
                    str(placement),
                    str(seed_index),
                )
            )
            request_seed = derive_strategic_search_request_seed(
                quality_fixture_id,
                information,
                search_config.digest,
            )
            result = StrategicInformationSetSearch(search_config).search(
                information, request_seed
            )
            visits = result.action_visits
            nonzero = sorted((count for count in visits if count), reverse=True)
            entropy = _entropy(visits)
            state_runs.append(
                {
                    "seed_index": seed_index,
                    "selected_action_index": result.selected_action_index,
                    "visit_entropy_nats": entropy,
                    "normalized_visit_entropy": (
                        entropy / math.log(legal_count) if legal_count > 1 else 0.0
                    ),
                    "maximum_visit_share": (
                        nonzero[0] / search_config.outer_simulation_budget
                    ),
                    "top_two_visit_margin": (
                        (nonzero[0] - nonzero[1])
                        / search_config.outer_simulation_budget
                        if len(nonzero) > 1
                        else 1.0
                    ),
                    "elapsed_seconds": result.elapsed_seconds,
                }
            )
        action_counts = Counter(
            item["selected_action_index"] for item in state_runs
        )
        records.append(
            {
                "split": split,
                "fixture_index": fixture_index,
                "round_number": round_number,
                "placement_number": placement,
                "stage": stage,
                "player": information.player.value,
                "legal_action_count": legal_count,
                "initial_coverage_visit_share": (
                    legal_count / search_config.outer_simulation_budget
                ),
                "mean_visit_entropy_nats": sum(
                    item["visit_entropy_nats"] for item in state_runs
                )
                / len(state_runs),
                "mean_normalized_visit_entropy": sum(
                    item["normalized_visit_entropy"] for item in state_runs
                )
                / len(state_runs),
                "mean_maximum_visit_share": sum(
                    item["maximum_visit_share"] for item in state_runs
                )
                / len(state_runs),
                "mean_top_two_visit_margin": sum(
                    item["top_two_visit_margin"] for item in state_runs
                )
                / len(state_runs),
                "top_action_agreement": max(action_counts.values())
                / len(state_runs),
                "distinct_top_actions": len(action_counts),
                "runs": state_runs,
            }
        )

    stage_results = {}
    for stage in ("early", "middle", "late"):
        selected = [record for record in records if record["stage"] == stage]
        stage_results[stage] = {
            key: sum(record[key] for record in selected) / len(selected)
            for key in (
                "initial_coverage_visit_share",
                "mean_visit_entropy_nats",
                "mean_normalized_visit_entropy",
                "mean_maximum_visit_share",
                "mean_top_two_visit_margin",
                "top_action_agreement",
            )
        }
    return {
        "outer_simulation_budget": search_config.outer_simulation_budget,
        "response_completions_per_action": (
            search_config.response_completions_per_action
        ),
        "search_config_digest": search_config.digest,
        "request_seeds_per_state": request_seed_count,
        "state_count": len(records),
        "search_count": len(records) * request_seed_count,
        "wall_seconds": time.perf_counter() - started,
        "peak_process_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "stages": stage_results,
        "states": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--request-seeds", type=int, default=5)
    parser.add_argument("--target-budget", type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.request_seeds < 2:
        raise ValueError("target quality requires multiple request seeds")
    dataset, captured = _validate_dataset(args.dataset, args.reference)
    config = load_teacher_config(args.dataset)
    search_config = StrategicSearchConfig(
        outer_simulation_budget=(
            config.simulation_budget
            if args.target_budget is None
            else args.target_budget
        ),
        response_completions_per_action=(
            config.response_completions_per_action
        ),
        outer_exploration_constant=config.exploration_constant,
    )
    quality = _measure_target_quality(
        config,
        captured,
        args.request_seeds,
        search_config,
    )
    result = {
        "dataset": dataset,
        "target_quality": quality,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

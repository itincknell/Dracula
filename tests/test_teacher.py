"""Deterministic search-teacher collection and artifact invariants."""

from __future__ import annotations

import copy
import json
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

import dracula.teacher as teacher
from dracula.bridge import PolicyTurnKind, build_policy_turn_context
from dracula.engine import EnginePlayer, EngineStatus, advance_after_round, apply_move, create_game
from dracula.search import (
    GREEDY_RESPONSE_SCHEMA_VERSION,
    STRATEGIC_SEARCH_SCHEMA_VERSION,
    derive_strategic_search_request_seed,
    information_state_fingerprint,
    information_state_from_engine,
)
from dracula.teacher import (
    APPROVED_TEACHER_PROFILE,
    EXAMPLES_PER_GAME,
    EXAMPLES_PER_PLAYER,
    FixtureSplit,
    TEACHER_APPROVAL_IDENTITY,
    TeacherCollectionConfig,
    TeacherCollectionError,
    TeacherCollectionInterrupted,
    collect_teacher_dataset,
    derive_teacher_fixture,
    fixture_schedule,
    inspect_teacher_dataset,
    load_teacher_manifest,
    load_teacher_shard,
    main,
)


def _fast_contract_search(information, fixture, config, should_stop):
    """Produce deterministic legal visits while artifact tests exercise collection."""

    del fixture, should_stop
    legal = tuple(
        index
        for index, allowed in enumerate(
            value for row in information.legal_mask for value in row
        )
        if allowed
    )
    quotient, remainder = divmod(config.simulation_budget, len(legal))
    visits = [0] * 32
    for ordinal, index in enumerate(legal):
        visits[index] = quotient + int(ordinal < remainder)
    selected = min(legal, key=lambda index: (-visits[index], index))
    return SimpleNamespace(
        information_state_fingerprint=information_state_fingerprint(information),
        config_digest=config.search_config.digest,
        simulation_count=config.simulation_budget,
        action_visits=tuple(visits),
        selected_action_index=selected,
    )


def _config(output: Path) -> TeacherCollectionConfig:
    return TeacherCollectionConfig(
        run_id="teacher-contract",
        root_seed="teacher-contract-root",
        output_directory=str(output),
        simulation_budget=32,
        training_games=1,
        validation_games=0,
        absolute_fixtures=3,
        workers=1,
    )


@pytest.fixture(scope="module")
def collected(tmp_path_factory):
    output = tmp_path_factory.mktemp("teacher-baseline")
    config = _config(output)
    with patch.object(teacher, "_run_teacher_search", _fast_contract_search):
        inspection = collect_teacher_dataset(config)
    fixture = derive_teacher_fixture(config.root_seed, FixtureSplit.TRAINING, 0)
    shard_path = (
        output
        / "teacher"
        / "games"
        / FixtureSplit.TRAINING.value
        / f"{fixture.fixture_id}.pt"
    )
    return config, inspection, fixture, shard_path, load_teacher_shard(shard_path)


# One full game must yield 21 learned decisions per role and no forced rows.
def test_full_game_rows_are_role_balanced_and_exclude_forced_placements(collected) -> None:
    config, inspection, _, _, shard = collected
    columns = shard["columns"]
    assert inspection.examples == EXAMPLES_PER_GAME == 42
    assert inspection.simulations == EXAMPLES_PER_GAME * config.simulation_budget
    assert columns["observations"].shape == (42, 875)
    assert columns["legal_masks"].shape == (42, 4, 8)
    assert columns["search_visits"].shape == (42, 32)
    assert columns["search_policies"].shape == (42, 32)
    assert columns["players"].count(EnginePlayer.QUEEN.value) == EXAMPLES_PER_PLAYER
    assert columns["players"].count(EnginePlayer.KING.value) == EXAMPLES_PER_PLAYER
    assert 8 not in columns["placement_numbers"].tolist()
    assert set(columns["placement_numbers"].tolist()) == set(range(1, 8))


# Search visits are the exact policy targets and cannot assign mass outside legality.
def test_visit_targets_are_exact_normalized_and_selected_actions_are_legal(collected) -> None:
    config, _, _, _, shard = collected
    columns = shard["columns"]
    visits = columns["search_visits"]
    policies = columns["search_policies"]
    masks = columns["legal_masks"].flatten(start_dim=1)
    selected = columns["selected_action_indices"]

    assert torch.all(visits.sum(dim=1) == config.simulation_budget)
    assert torch.equal(
        policies,
        visits.to(torch.float32) / float(config.simulation_budget),
    )
    assert torch.allclose(
        policies.sum(dim=1), torch.ones(policies.shape[0]), atol=1e-6, rtol=0.0
    )
    assert torch.count_nonzero(visits[~masks]) == 0
    assert torch.count_nonzero(policies[~masks]) == 0
    assert torch.all(masks.gather(1, selected[:, None]).squeeze(1))
    assert torch.all(visits.gather(1, selected[:, None]).squeeze(1) > 0)


# Replaying public selected actions against the engine must reproduce every terminal target.
def test_round_returns_exactly_match_engine_results_and_reverse_perspective(collected) -> None:
    _, _, fixture, _, shard = collected
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
    state = create_game(fixture._game_seed)
    while state.status is not EngineStatus.GAME_COMPLETE:
        while state.status is EngineStatus.PLAYING:
            placement = len(state.current_round_moves) + 1
            context = build_policy_turn_context(state, state.active_player)
            if context.kind is PolicyTurnKind.FORCED_RECURRENT_TRANSITION:
                move = context.forced_move
            else:
                row = rows[(state.round_number, placement)]
                move = context.action_table[
                    int(columns["selected_action_indices"][row].item())
                ]
            assert move is not None
            state = apply_move(state, move).state
        result = state.pending_round_result
        assert result is not None
        queen_return = (
            result.round_scores[EnginePlayer.QUEEN]
            - result.round_scores[EnginePlayer.KING]
        ) / 150.0
        for player, expected in (
            (EnginePlayer.QUEEN, queen_return),
            (EnginePlayer.KING, -queen_return),
        ):
            indexes = [
                index
                for index, (round_number, row_player) in enumerate(
                    zip(
                        columns["round_numbers"].tolist(),
                        columns["players"],
                        strict=True,
                    )
                )
                if round_number == state.round_number and row_player == player.value
            ]
            assert all(
                columns["round_returns"][index].item()
                == pytest.approx(expected)
                for index in indexes
            )
        state = advance_after_round(state)


# Split namespaces must prevent any deck fixture from crossing data boundaries.
def test_training_validation_and_absolute_fixture_splits_are_disjoint(tmp_path) -> None:
    config = TeacherCollectionConfig(
        run_id="split-test",
        root_seed="split-test-root",
        output_directory=str(tmp_path),
        simulation_budget=32,
        training_games=3,
        validation_games=2,
        absolute_fixtures=4,
        workers=1,
    )
    fixtures = fixture_schedule(config)
    ids = [fixture.fixture_id for fixture in fixtures]
    private_seeds = [fixture._game_seed for fixture in fixtures]
    assert len(ids) == len(set(ids)) == 9
    assert len(private_seeds) == len(set(private_seeds)) == 9
    for split in FixtureSplit:
        split_ids = {item.fixture_id for item in fixtures if item.split is split}
        other_ids = {item.fixture_id for item in fixtures if item.split is not split}
        assert split_ids.isdisjoint(other_ids)


# Sealed shards expose model inputs and public identifiers but no sampled world.
def test_sealed_artifact_has_no_private_state_or_engine_seed(collected) -> None:
    config, _, fixture, shard_path, shard = collected
    forbidden = {
        "authoritative_state",
        "determination",
        "determinization",
        "engine_seed",
        "game_seed",
        "hands",
        "opponent_hand",
        "sampled_stock",
        "search_tree",
        "stock",
        "tree",
    }

    def keys(value):
        if isinstance(value, dict):
            for key, nested in value.items():
                yield key
                yield from keys(nested)
        elif isinstance(value, (list, tuple)):
            for nested in value:
                yield from keys(nested)

    assert forbidden.isdisjoint(keys(shard))
    assert fixture._game_seed.encode("utf-8") not in shard_path.read_bytes()
    split_document = json.loads(
        (config.output_path / "teacher" / "fixture-splits.json").read_text()
    )
    assert set(split_document["splits"]["fixtures"]) == {
        split.value for split in FixtureSplit
    }
    assert len(split_document["splits"]["fixtures"]["absolute"]) == 3
    assert fixture._game_seed not in json.dumps(split_document)
    assert set(shard["columns"]) == {
        "observations",
        "legal_masks",
        "search_visits",
        "search_policies",
        "selected_action_indices",
        "round_returns",
        "information_state_digests",
        "fixture_ids",
        "splits",
        "players",
        "dealers",
        "round_numbers",
        "placement_numbers",
    }


# A restart reuses public decision caches and seals the same deterministic dataset.
def test_interruption_resume_matches_clean_collection(collected, tmp_path) -> None:
    baseline_config, baseline, _, _, _ = collected
    interrupted_config = _config(tmp_path / "interrupted")
    assert interrupted_config.digest == baseline_config.digest

    def stop_after_one_cached_decision() -> bool:
        return any(
            interrupted_config.output_path.glob("teacher/cache/**/*.json")
        )

    with pytest.raises(TeacherCollectionInterrupted):
        with patch.object(teacher, "_run_teacher_search", _fast_contract_search):
            collect_teacher_dataset(
                interrupted_config,
                should_stop=stop_after_one_cached_decision,
            )
    assert not any(interrupted_config.output_path.glob("teacher/games/**/*.pt"))
    assert len(list(interrupted_config.output_path.glob("teacher/cache/**/*.json"))) == 1
    assert json.loads(
        (interrupted_config.output_path / "state.json").read_text()
    )["phase"] == "interrupted"

    cache = json.loads(
        next(interrupted_config.output_path.glob("teacher/cache/**/*.json")).read_text()
    )
    assert cache["teacher_profile"] == APPROVED_TEACHER_PROFILE
    assert cache["approval_identity"] == TEACHER_APPROVAL_IDENTITY
    assert cache["search_schema_version"] == STRATEGIC_SEARCH_SCHEMA_VERSION
    assert cache["response_schema_version"] == GREEDY_RESPONSE_SCHEMA_VERSION
    with patch.object(teacher, "_run_teacher_search", _fast_contract_search):
        resumed = collect_teacher_dataset(interrupted_config, resume=True)
    assert resumed.dataset_digest == baseline.dataset_digest
    assert resumed.examples == baseline.examples
    assert resumed.simulations == baseline.simulations
    assert resumed.metrics.cache_hits == 1


# Tensor artifacts and manifests must reject mutation instead of silently drifting.
def test_artifact_round_trip_and_hash_validation(collected, tmp_path) -> None:
    config, inspection, _, shard_path, shard = collected
    assert inspect_teacher_dataset(config.output_path).dataset_digest == inspection.dataset_digest
    manifest = load_teacher_manifest(config.output_path, FixtureSplit.TRAINING)
    assert manifest["manifest"]["example_count"] == EXAMPLES_PER_GAME
    metadata = shard["metadata"]
    assert metadata["teacher_profile"] == APPROVED_TEACHER_PROFILE
    assert metadata["approval_identity"] == TEACHER_APPROVAL_IDENTITY
    assert metadata["search_schema_version"] == STRATEGIC_SEARCH_SCHEMA_VERSION
    assert metadata["response_schema_version"] == GREEDY_RESPONSE_SCHEMA_VERSION
    assert (
        manifest["manifest"]["teacher_profile_digest"]
        == metadata["teacher_profile_digest"]
    )
    assert (
        manifest["manifest"]["response_config_digest"]
        == metadata["response_config_digest"]
    )

    changed = copy.deepcopy(shard)
    changed["columns"]["search_visits"][0, 0] += 1
    changed_path = tmp_path / "changed.pt"
    torch.save(changed, changed_path)
    with pytest.raises(TeacherCollectionError):
        load_teacher_shard(changed_path)

    truncated_path = tmp_path / "truncated.pt"
    truncated_path.write_bytes(shard_path.read_bytes()[:100])
    with pytest.raises(TeacherCollectionError):
        load_teacher_shard(truncated_path)


def test_worker_bound_and_inspection_cli(collected, capsys) -> None:
    config, inspection, _, _, _ = collected
    six_workers = TeacherCollectionConfig(
        run_id="six-workers",
        root_seed="root",
        output_directory=str(config.output_path / "six"),
        simulation_budget=32,
        training_games=6,
        validation_games=0,
        workers=6,
    )
    assert six_workers.workers == 6
    with pytest.raises(TeacherCollectionError, match="workers"):
        TeacherCollectionConfig(
            run_id="too-many-workers",
            root_seed="root",
            output_directory=str(config.output_path / "invalid"),
            simulation_budget=32,
            training_games=1,
            validation_games=0,
            workers=7,
        )
    assert main(["inspect", "--output", str(config.output_path)]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["dataset_digest"] == inspection.dataset_digest


def test_collection_worker_uses_one_pytorch_thread(collected) -> None:
    config, _, _, shard_path, _ = collected
    metrics = json.loads(shard_path.with_suffix(".metrics.json").read_text())
    assert metrics["torch_num_threads"] == 1
    assert metrics["torch_num_interop_threads"] == 1


# A pool interruption signals active workers before waiting and leaves no shard.
def test_parallel_interruption_requests_worker_stop_and_cancels_pending(
    tmp_path,
    monkeypatch,
) -> None:
    config = TeacherCollectionConfig(
        run_id="parallel-interrupt",
        root_seed="parallel-interrupt-root",
        output_directory=str(tmp_path),
        simulation_budget=32,
        training_games=2,
        validation_games=0,
        workers=2,
    )
    fixtures = fixture_schedule(config)[:2]
    futures = []
    shutdown_calls = []

    class Future:
        def __init__(self) -> None:
            self.cancelled = False

        def result(self):
            raise KeyboardInterrupt

        def cancel(self) -> None:
            self.cancelled = True

    class Executor:
        def __init__(self, **kwargs) -> None:
            assert kwargs["max_workers"] == 2
            assert kwargs["initializer"] is teacher._configure_collection_worker

        def submit(self, function, fixture, submitted_config):
            assert function is teacher._collect_fixture_worker
            assert submitted_config == config
            future = Future()
            futures.append(future)
            return future

        def shutdown(self, *, wait, cancel_futures=False) -> None:
            shutdown_calls.append((wait, cancel_futures))
            if cancel_futures:
                assert (
                    config.output_path
                    / "teacher"
                    / teacher._STOP_REQUEST_FILENAME
                ).exists()

    monkeypatch.setattr(teacher, "ProcessPoolExecutor", Executor)
    monkeypatch.setattr(
        teacher, "as_completed", lambda submitted: iter(submitted)
    )
    with pytest.raises(KeyboardInterrupt):
        teacher._collect_parallel_fixtures(
            fixtures, config, [], len(fixtures), None
        )
    assert futures and all(future.cancelled for future in futures)
    assert shutdown_calls == [(True, True)]
    assert not (
        config.output_path / "teacher" / teacher._STOP_REQUEST_FILENAME
    ).exists()
    assert not any(config.output_path.glob("teacher/games/**/*.pt"))


def test_collection_cli_reports_interruption_without_traceback(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    def interrupt(*args, **kwargs):
        raise TeacherCollectionInterrupted

    monkeypatch.setattr(teacher, "collect_teacher_dataset", interrupt)
    result = main(
        [
            "full",
            "--output",
            str(tmp_path),
            "--root-seed",
            "cli-interruption-root",
            "--training-games",
            "1",
            "--validation-games",
            "0",
            "--absolute-fixtures",
            "0",
        ]
    )
    captured = capsys.readouterr()
    assert result == 130
    assert captured.out == ""
    assert captured.err == "teacher_collection interrupted\n"


# New collection must instantiate the approved strategic planner and request seed.
def test_active_planner_is_strategic_32x4(tmp_path) -> None:
    config = _config(tmp_path)
    fixture = derive_teacher_fixture(
        config.root_seed, FixtureSplit.TRAINING, 0
    )
    information = information_state_from_engine(create_game(fixture._game_seed))
    sentinel = object()
    captured = {}

    class Planner:
        def __init__(self, search_config):
            captured["config"] = search_config

        def search(self, observed, request_seed, should_stop):
            captured["information"] = observed
            captured["seed"] = request_seed
            captured["stop"] = should_stop
            return sentinel

    with patch.object(teacher, "StrategicInformationSetSearch", Planner):
        result = teacher._run_teacher_search(
            information, fixture, config, None
        )
    assert result is sentinel
    assert captured["config"].outer_simulation_budget == 32
    assert captured["config"].response_completions_per_action == 4
    assert captured["information"] == information
    assert captured["seed"] == derive_strategic_search_request_seed(
        fixture.fixture_id, information, config.search_config.digest
    )


# A legacy manifest cannot be spliced into a Teacher v2 dataset.
def test_training_and_validation_manifests_reject_mixed_teacher_versions(
    tmp_path,
) -> None:
    output = tmp_path / "mixed"
    config = _config(output)
    with patch.object(teacher, "_run_teacher_search", _fast_contract_search):
        collect_teacher_dataset(config)
    legacy_manifest = {
        "format_version": teacher.LEGACY_TEACHER_MANIFEST_FORMAT_VERSION,
        "dataset_schema_version": (
            teacher.LEGACY_TEACHER_DATASET_SCHEMA_VERSION
        ),
        "split": FixtureSplit.VALIDATION.value,
        "collection_config_digest": "0" * 64,
        "search_config_digest": "1" * 64,
        "fixture_count": 0,
        "example_count": 0,
        "simulation_count": 0,
        "shards": [],
    }
    (output / "teacher" / "validation-manifest.json").write_text(
        json.dumps(
            {
                "manifest": legacy_manifest,
                "manifest_digest": teacher._json_digest(legacy_manifest),
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(TeacherCollectionError, match="mix teacher contracts"):
        inspect_teacher_dataset(output)


# Cache and shard format bumps prevent v1 material from entering v2 resume.
def test_v1_cache_and_shard_contracts_cannot_enter_v2(
    collected,
    tmp_path,
) -> None:
    _, _, _, _, shard = collected
    mixed_shard = copy.deepcopy(shard)
    mixed_shard["format_version"] = (
        teacher.LEGACY_TEACHER_SHARD_FORMAT_VERSION
    )
    mixed_path = tmp_path / "mixed-shard.pt"
    torch.save(mixed_shard, mixed_path)
    with pytest.raises(TeacherCollectionError, match="metadata shape"):
        load_teacher_shard(mixed_path)

    config = _config(tmp_path / "mixed-cache")

    def stop_after_cache() -> bool:
        return any(config.output_path.glob("teacher/cache/**/*.json"))

    with pytest.raises(TeacherCollectionInterrupted):
        with patch.object(teacher, "_run_teacher_search", _fast_contract_search):
            collect_teacher_dataset(config, should_stop=stop_after_cache)
    cache_path = next(config.output_path.glob("teacher/cache/**/*.json"))
    cache = json.loads(cache_path.read_text())
    cache["format_version"] = teacher.LEGACY_TEACHER_CACHE_FORMAT_VERSION
    cache_path.write_text(json.dumps(cache), encoding="utf-8")
    with pytest.raises(TeacherCollectionError, match="does not match"):
        with patch.object(teacher, "_run_teacher_search", _fast_contract_search):
            collect_teacher_dataset(config, resume=True)


def test_unapproved_new_collection_profile_is_rejected(tmp_path) -> None:
    with pytest.raises(TeacherCollectionError, match="approved 32x4"):
        TeacherCollectionConfig(
            run_id="unapproved",
            root_seed="root",
            output_directory=str(tmp_path),
            simulation_budget=500,
            training_games=1,
            validation_games=0,
        )

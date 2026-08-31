"""Structural and sealed-artifact invariants for the Sam corpus miner."""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import replace
from pathlib import Path

import pytest

import dracula.sam_miner as miner
from dracula.engine import EngineStatus, apply_move, create_game, legal_moves
from dracula.search import (
    SAM_TEACHER_SEARCH_SCHEMA_VERSION,
    SamTeacherInformationSetSearch,
    SamTeacherSearchConfig,
    SearchInterrupted,
    derive_sam_teacher_destination_seed,
    derive_sam_teacher_request_seed,
    information_state_from_engine,
    information_state_from_simulation,
    sample_determinization,
    sam_teacher_action_groups,
    select_concrete_action_index,
)
from dracula.randomness import derive_seed


class CountingTeacher:
    def __init__(self) -> None:
        self.calls = 0
        self._stub = miner.DeterministicSamTeacherStub()
        self.schema_version = self._stub.schema_version
        self.configuration_digest = self._stub.configuration_digest
        self.controller_profile = self._stub.controller_profile

    def select(
        self, information, should_stop=None
    ) -> miner.SamMinerTeacherSelection:
        self.calls += 1
        return self._stub.select(information, should_stop)


def _config(
    path: Path,
    *,
    root_seed: str = "sam-miner-tests",
    child_count: int = 1,
    training: int = 1,
    validation: int = 0,
    test: int = 0,
    workers: int = 1,
    continuous: bool = False,
    minimum_free_disk_bytes: int = 0,
) -> miner.SamMinerConfig:
    teacher = miner.DeterministicSamTeacherStub()
    source = miner.resolve_source_identity()
    return miner.SamMinerConfig(
        run_id="sam-miner-test",
        root_seed=root_seed,
        output_directory=str(path),
        training_decks=training,
        validation_decks=validation,
        test_decks=test,
        child_count=child_count,
        workers=workers,
        continuous=continuous,
        minimum_free_disk_bytes=minimum_free_disk_bytes,
        teacher_schema_version=teacher.schema_version,
        teacher_configuration_digest=teacher.configuration_digest,
        teacher_controller_profile=teacher.controller_profile,
        source_tree_schema_version=source.schema_version,
        source_revision=source.revision,
        source_tree_digest=source.tree_digest,
    )


def _first_state(config: miner.SamMinerConfig):
    fixture = miner.derive_deck_fixture(
        config.root_seed, miner.DeckSplit.TRAINING, 0
    )
    return fixture, create_game(fixture._engine_seed)


def _advance(state, count: int):
    for _ in range(count):
        state = apply_move(
            state, legal_moves(state, state.active_player)[0]
        ).state
    return state


def _reduced_real(
    path: Path,
) -> tuple[
    miner.SamMinerConfig,
    miner.Sam128MinerTeacher,
    miner.DeckFixture,
    object,
]:
    search_config = SamTeacherSearchConfig(8, 8)
    teacher = miner.Sam128MinerTeacher(
        search_config,
        controller_profile=miner.SAM_MINER_REDUCED_TEST_PROFILE,
    )
    source = miner.resolve_source_identity()
    config = miner.SamMinerConfig(
        run_id="sam-reduced-real",
        root_seed="sam-reduced-real",
        output_directory=str(path),
        training_decks=1,
        validation_decks=0,
        test_decks=0,
        child_count=4,
        teacher_schema_version=teacher.schema_version,
        teacher_configuration_digest=teacher.configuration_digest,
        teacher_controller_profile=teacher.controller_profile,
        teacher_outer_simulation_budget=8,
        teacher_response_simulation_budget=8,
        source_tree_schema_version=source.schema_version,
        source_revision=source.revision,
        source_tree_digest=source.tree_digest,
    )
    fixture, initial = _first_state(config)
    return config, teacher, fixture, _advance(initial, 5)


class ProductionShapeTeacher:
    """Fast unit-test double with the exact immutable production identity."""

    schema_version = SAM_TEACHER_SEARCH_SCHEMA_VERSION
    configuration_digest = SamTeacherSearchConfig(32, 32).digest
    controller_profile = miner.SAM_MINER_CONTROLLER_PROFILE

    def select(self, information, should_stop=None):
        if should_stop is not None and should_stop():
            raise miner.SamMinerInterrupted("production-shape interruption")
        groups = sam_teacher_action_groups(information)
        selected = groups[0]
        request = derive_sam_teacher_request_seed(
            information, self.configuration_digest
        )
        concrete = select_concrete_action_index(
            information,
            selected.representative_action_index,
            derive_sam_teacher_destination_seed(
                request,
                "sam-128-root-result",
                information,
                selected.representative_action_index,
                0,
            ),
        )
        return miner.SamMinerTeacherSelection(
            selected.representative_action_index,
            concrete,
        )


def test_packed_row_round_trip_is_exact(tmp_path: Path) -> None:
    """Packed columns must recover all 875+32 booleans without padding data."""

    config = _config(tmp_path)
    fixture, state = _first_state(config)
    teacher = CountingTeacher()
    result = miner.mine_round_tree(
        state,
        fixture_id=fixture.fixture_id,
        teacher=teacher,
        config=config,
    )
    row = result.rows[0]
    restored = miner.SamMinerRow.from_dict(row.to_dict())
    information = information_state_from_engine(state)
    expected_observation = tuple(
        bool(value)
        for value in miner.policy_input_from_information_state(
            information
        ).observation.tolist()
    )

    assert restored == row
    assert len(row.observation_packed) == 110
    assert len(row.legal_mask_packed) == 4
    assert miner.unpack_observation(row) == expected_observation
    assert miner.unpack_legal_mask(row) == information.legal_mask
    assert json.loads(miner._canonical_json(row.to_dict())) == row.to_dict()


def test_default_round_tree_has_engine_valid_exact_counts(
    tmp_path: Path,
) -> None:
    """The last learned node has two actions; prior branching still yields 5,461 rows."""

    config = _config(tmp_path, child_count=4)
    fixture, state = _first_state(config)
    result = miner.mine_round_tree(
        state,
        fixture_id=fixture.fixture_id,
        teacher=CountingTeacher(),
        config=config,
    )

    assert Counter(row.placement_number for row in result.rows) == {
        1: 1,
        2: 4,
        3: 16,
        4: 64,
        5: 256,
        6: 1_024,
        7: 4_096,
    }
    assert len(result.rows) == 5_461
    assert result.terminal_leaf_count == 8_192
    assert result.trunk_terminal_state.status is EngineStatus.ROUND_COMPLETE
    assert len(result.trunk_terminal_state.current_round_moves) == 8


def test_teacher_cache_hit_avoids_duplicate_query_and_replays_action(
    tmp_path: Path,
) -> None:
    """A content-addressed hit must not invoke the expensive teacher again."""

    config = _config(tmp_path)
    fixture, state = _first_state(config)
    information = information_state_from_engine(state)
    groups = sam_teacher_action_groups(information)
    teacher = CountingTeacher()

    first = miner.resolve_teacher(information, groups, teacher, config)
    second = miner.resolve_teacher(information, groups, teacher, config)
    concrete_first = miner._concrete_action(
        information,
        first.selected_representative_action_index,
        config=config,
        parent_path_digest=miner._root_path_digest(
            fixture.fixture_id, 1
        ),
        child_ordinal=0,
    )
    concrete_second = miner._concrete_action(
        information,
        second.selected_representative_action_index,
        config=config,
        parent_path_digest=miner._root_path_digest(
            fixture.fixture_id, 1
        ),
        child_ordinal=0,
    )

    assert teacher.calls == 1
    assert first.cache_hit is False
    assert second.cache_hit is True
    assert first.selected_group_index == second.selected_group_index
    assert concrete_first == concrete_second


def test_cache_rejects_wrong_controller_even_with_valid_outer_digest(
    tmp_path: Path,
) -> None:
    """A rehashed incompatible artifact cannot masquerade as production Sam."""

    config = _config(tmp_path)
    _, state = _first_state(config)
    information = information_state_from_engine(state)
    groups = sam_teacher_action_groups(information)
    teacher = CountingTeacher()
    cached = miner.resolve_teacher(information, groups, teacher, config)
    path = miner._cache_path(config, cached.cache_key)
    document = miner._load_json(path)
    content = dict(document["content"])
    content["controller_profile"] = "sam-32x32"
    miner._atomic_json(
        path,
        miner._envelope(miner.SAM_MINER_CACHE_FORMAT_VERSION, content),
    )

    with pytest.raises(miner.SamMinerError, match="identity"):
        miner.resolve_teacher(information, groups, teacher, config)


def test_branching_is_deterministic_and_inputs_remain_immutable(
    tmp_path: Path,
) -> None:
    """Derived alternatives and destination coins cannot mutate or drift."""

    first_config = _config(tmp_path / "first", child_count=2)
    second_config = _config(tmp_path / "second", child_count=2)
    fixture, state = _first_state(first_config)
    before = miner.state_fingerprint(state)
    first = miner.mine_round_tree(
        state,
        fixture_id=fixture.fixture_id,
        teacher=CountingTeacher(),
        config=first_config,
    )
    second = miner.mine_round_tree(
        state,
        fixture_id=fixture.fixture_id,
        teacher=CountingTeacher(),
        config=second_config,
    )

    assert miner.state_fingerprint(state) == before
    assert tuple(row.to_dict() for row in first.rows) == tuple(
        row.to_dict() for row in second.rows
    )
    assert first.terminal_leaf_count == second.terminal_leaf_count
    assert (
        miner.state_fingerprint(first.trunk_terminal_state)
        == miner.state_fingerprint(second.trunk_terminal_state)
    )


def test_alternatives_are_distinct_and_exclude_teacher(
    tmp_path: Path,
) -> None:
    """The three random alternatives are sampled as groups without replacement."""

    config = _config(tmp_path, child_count=4)
    fixture, state = _first_state(config)
    information = information_state_from_engine(state)
    groups = sam_teacher_action_groups(information)
    teacher = CountingTeacher()
    cached = miner.resolve_teacher(information, groups, teacher, config)
    alternatives = miner._alternative_groups(
        groups,
        cached.selected_group_index,
        information=information,
        config=config,
        fixture_id=fixture.fixture_id,
        parent_path_digest=miner._root_path_digest(
            fixture.fixture_id, 1
        ),
    )

    representatives = {
        group.representative_action_index for group in alternatives
    }
    assert len(alternatives) == 3
    assert len(representatives) == 3
    assert cached.selected_representative_action_index not in representatives


def test_forced_placement_produces_no_row(tmp_path: Path) -> None:
    """The eighth move is retained only as the single engine continuation."""

    config = _config(tmp_path)
    fixture, state = _first_state(config)
    teacher = CountingTeacher()
    path = miner._root_path_digest(fixture.fixture_id, 1)
    current = state
    for _ in range(7):
        information = information_state_from_engine(current)
        groups = sam_teacher_action_groups(information)
        selected = miner.resolve_teacher(
            information, groups, teacher, config
        )
        group = groups[selected.selected_group_index]
        concrete = miner._concrete_action(
            information,
            group.representative_action_index,
            config=config,
            parent_path_digest=path,
            child_ordinal=0,
        )
        child_path = miner._child_path_digest(
            round_number=1,
            placement_number=len(current.current_round_moves) + 1,
            actor=information.player,
            parent_path_digest=path,
            child_ordinal=0,
            strategic_representative=(
                group.representative_action_index
            ),
            concrete_action_index=concrete,
        )
        current = miner._apply_action(current, information, concrete)
        path = child_path
    result = miner.mine_branch_subtree(
        current,
        fixture_id=fixture.fixture_id,
        path_digest=path,
        teacher=teacher,
        config=config,
    )

    assert result.rows == ()
    assert result.terminal_leaf_count == 1


def test_atomic_corpus_and_manifests_round_trip(tmp_path: Path) -> None:
    """Every accepted layer must revalidate from the bytes on disk."""

    config = _config(tmp_path)
    inspection = miner.mine_corpus(
        config, miner.DeterministicSamTeacherStub()
    )
    verified = miner.verify_corpus(tmp_path)

    assert inspection == verified
    assert inspection.deck_count == 1
    assert inspection.round_count == 6
    assert inspection.shard_count == 12
    assert inspection.row_count == 42
    assert inspection.terminal_leaf_count == 6
    assert not tuple(tmp_path.rglob("*.tmp-*"))


def test_interruption_resumes_from_sealed_subtree_boundary(
    tmp_path: Path,
) -> None:
    """A stop after the root seal must neither lose nor duplicate that row."""

    config = _config(tmp_path)
    calls = 0

    def stop() -> bool:
        nonlocal calls
        calls += 1
        return calls >= 4

    with pytest.raises(miner.SamMinerInterrupted):
        miner.mine_corpus(
            config,
            miner.DeterministicSamTeacherStub(),
            should_stop=stop,
        )
    fixture = miner.fixture_schedule(config)[0]
    round_directory = miner._round_directory(config, fixture, 1)
    assert (round_directory / "root.json").exists()
    assert not (round_directory / "subtree-00.json").exists()
    assert not (round_directory / "round-manifest.json").exists()
    assert not tuple(tmp_path.rglob("*.tmp-*"))

    resumed = miner.mine_corpus(
        config,
        miner.DeterministicSamTeacherStub(),
        resume=True,
    )
    assert resumed.row_count == 42
    root = miner.load_subtree_shard(
        round_directory / "root.json", config
    )
    assert root["content"]["row_count"] == 1


def test_resume_and_clean_run_have_identical_corpus_digest(
    tmp_path: Path,
) -> None:
    """Execution history must not enter the canonical corpus identity."""

    interrupted = _config(tmp_path / "interrupted")
    calls = 0

    def stop() -> bool:
        nonlocal calls
        calls += 1
        return calls >= 4

    with pytest.raises(miner.SamMinerInterrupted):
        miner.mine_corpus(
            interrupted,
            miner.DeterministicSamTeacherStub(),
            should_stop=stop,
        )
    resumed = miner.mine_corpus(
        interrupted,
        miner.DeterministicSamTeacherStub(),
        resume=True,
    )
    clean = _config(tmp_path / "clean")
    direct = miner.mine_corpus(
        clean, miner.DeterministicSamTeacherStub()
    )

    assert resumed.corpus_manifest_digest == direct.corpus_manifest_digest
    assert resumed.row_count == direct.row_count


def _artifact_snapshot(root: Path, directory: str) -> dict[str, bytes]:
    base = root / directory
    return {
        str(path.relative_to(base)): path.read_bytes()
        for path in sorted(base.rglob("*"))
        if path.is_file()
    }


def test_worker_count_is_operational_not_corpus_identity(
    tmp_path: Path,
) -> None:
    """Scheduling can change process count but never labels or artifact hashes."""

    sequential = _config(
        tmp_path / "sequential",
        training=2,
        validation=1,
        workers=1,
    )
    parallel = _config(
        tmp_path / "parallel",
        training=2,
        validation=1,
        workers=2,
    )

    sequential_result = miner.mine_corpus(
        sequential, miner.DeterministicSamTeacherStub()
    )
    parallel_result = miner.mine_corpus(
        parallel, miner.DeterministicSamTeacherStub()
    )

    assert sequential.digest == parallel.digest
    assert (
        sequential_result.corpus_manifest_digest
        == parallel_result.corpus_manifest_digest
    )
    assert _artifact_snapshot(
        sequential.output_path, "decks"
    ) == _artifact_snapshot(parallel.output_path, "decks")
    assert _artifact_snapshot(
        sequential.output_path, "cache"
    ) == _artifact_snapshot(parallel.output_path, "cache")
    assert (
        (sequential.output_path / "corpus-manifest.json").read_bytes()
        == (parallel.output_path / "corpus-manifest.json").read_bytes()
    )
    assert (
        (sequential.output_path / "fixture-splits.json").read_bytes()
        == (parallel.output_path / "fixture-splits.json").read_bytes()
    )
    assert not (parallel.output_path / ".workers").exists()


def test_parallel_interruption_resumes_to_clean_corpus(
    tmp_path: Path,
) -> None:
    """A stopped pool may retain staging seals but cannot commit partial decks."""

    interrupted = _config(
        tmp_path / "interrupted",
        training=2,
        validation=1,
        workers=2,
    )
    with pytest.raises(miner.SamMinerInterrupted):
        miner.mine_corpus(
            interrupted,
            miner.DeterministicSamTeacherStub(),
            should_stop=lambda: True,
        )
    state = miner._load_json(interrupted.output_path / "state.json")
    assert state["phase"] == "interrupted"
    assert not tuple(
        interrupted.output_path.glob(
            "decks/*/*/deck-manifest.json"
        )
    )
    assert not tuple(interrupted.output_path.rglob("*.tmp-*"))

    resumed = miner.mine_corpus(
        interrupted,
        miner.DeterministicSamTeacherStub(),
        resume=True,
    )
    clean = _config(
        tmp_path / "clean",
        training=2,
        validation=1,
        workers=2,
    )
    direct = miner.mine_corpus(
        clean, miner.DeterministicSamTeacherStub()
    )

    assert resumed.corpus_manifest_digest == direct.corpus_manifest_digest
    assert _artifact_snapshot(
        interrupted.output_path, "decks"
    ) == _artifact_snapshot(clean.output_path, "decks")
    assert _artifact_snapshot(
        interrupted.output_path, "cache"
    ) == _artifact_snapshot(clean.output_path, "cache")
    assert not (interrupted.output_path / ".workers").exists()


def test_parallel_resume_reuses_sealed_worker_subtree(
    tmp_path: Path,
) -> None:
    """A worker resumes its own fixture path instead of restarting the deck."""

    resumed_config = _config(
        tmp_path / "resumed",
        training=2,
        workers=2,
    )
    miner.initialize_corpus(resumed_config)
    fixture = miner.fixture_schedule(resumed_config)[0]
    staging_config = miner._worker_config(resumed_config, fixture)
    calls = 0

    def stop_after_root() -> bool:
        nonlocal calls
        calls += 1
        return calls >= 4

    with pytest.raises(miner.SamMinerInterrupted):
        miner._mine_deck_artifacts(
            fixture,
            teacher=miner.DeterministicSamTeacherStub(),
            config=staging_config,
            should_stop=stop_after_root,
        )
    root = (
        miner._round_directory(staging_config, fixture, 1) / "root.json"
    )
    root_bytes = root.read_bytes()

    resumed = miner.mine_corpus(
        resumed_config,
        miner.DeterministicSamTeacherStub(),
        resume=True,
    )
    clean_config = _config(
        tmp_path / "clean-staged-control",
        training=2,
        workers=1,
    )
    clean = miner.mine_corpus(
        clean_config, miner.DeterministicSamTeacherStub()
    )

    committed_root = (
        miner._round_directory(resumed_config, fixture, 1) / "root.json"
    )
    assert committed_root.read_bytes() == root_bytes
    assert resumed.corpus_manifest_digest == clean.corpus_manifest_digest
    assert _artifact_snapshot(
        resumed_config.output_path, "decks"
    ) == _artifact_snapshot(clean_config.output_path, "decks")


@pytest.mark.parametrize("workers", range(1, 9))
def test_supported_worker_counts_are_resolved(
    tmp_path: Path,
    workers: int,
) -> None:
    """All target-machine process counts are valid immutable run settings."""

    assert _config(tmp_path / str(workers), workers=workers).workers == workers


@pytest.mark.parametrize("workers", (0, 9))
def test_worker_count_outside_supported_range_is_rejected(
    tmp_path: Path,
    workers: int,
) -> None:
    """The M3 collection profile intentionally supports one through eight."""

    with pytest.raises(miner.SamMinerError, match="worker count"):
        _config(tmp_path, workers=workers)


def test_worker_thread_count_is_verified_before_commit(
    tmp_path: Path,
) -> None:
    """A worker that creates a PyTorch pool cannot commit its deck."""

    config = _config(tmp_path, workers=2)
    fixture = miner.fixture_schedule(config)[0]
    result = miner.DeckWorkerResult(
        fixture=fixture,
        staging_directory=str(miner._worker_output_path(config, fixture)),
        deck_content_digest="0" * 64,
        deck_file_digest="1" * 64,
        torch_thread_count=2,
        torch_interop_thread_count=1,
    )
    with pytest.raises(miner.SamMinerError, match="PyTorch thread"):
        miner._merge_worker_deck(result, config)
    assert not (config.output_path / "decks").exists()


def test_initialize_cli_persists_worker_count(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The process count is an explicit immutable local-run setting."""

    assert (
        miner.main(
            (
                "initialize",
                "--output",
                str(tmp_path),
                "--run-id",
                "parallel-cli",
                "--root-seed",
                "parallel-cli",
                "--training-decks",
                "1",
                "--validation-decks",
                "0",
                "--test-decks",
                "0",
                "--child-count",
                "1",
                "--workers",
                "4",
                "--teacher-profile",
                "deterministic-stub",
            )
        )
        == 0
    )
    capsys.readouterr()
    assert miner.load_config(tmp_path).workers == 4


def test_mixed_schema_or_configuration_shard_is_rejected(
    tmp_path: Path,
) -> None:
    """A valid shard from another branch configuration cannot enter this run."""

    config = _config(tmp_path)
    miner.mine_corpus(config, miner.DeterministicSamTeacherStub())
    fixture = miner.fixture_schedule(config)[0]
    shard = miner._round_directory(config, fixture, 1) / "root.json"
    incompatible = replace(config, child_count=2)

    with pytest.raises(miner.SamMinerError, match="bindings"):
        miner.load_subtree_shard(shard, incompatible)


def test_deck_splits_are_disjoint_and_reproducible(tmp_path: Path) -> None:
    """Branch descendants never cross the deck-root split boundary."""

    config = _config(
        tmp_path, training=3, validation=2, test=2
    )
    first = miner.fixture_schedule(config)
    second = miner.fixture_schedule(config)
    by_split = {
        split: {
            fixture.fixture_id
            for fixture in first
            if fixture.split is split
        }
        for split in miner.DeckSplit
    }

    assert first == second
    assert len({fixture.fixture_id for fixture in first}) == len(first)
    assert by_split[miner.DeckSplit.TRAINING].isdisjoint(
        by_split[miner.DeckSplit.VALIDATION]
    )
    assert by_split[miner.DeckSplit.TRAINING].isdisjoint(
        by_split[miner.DeckSplit.TEST]
    )
    assert by_split[miner.DeckSplit.VALIDATION].isdisjoint(
        by_split[miner.DeckSplit.TEST]
    )


def test_continuous_split_cycle_is_deterministic_and_unbounded(
    tmp_path: Path,
) -> None:
    """An ordinal, not a terminal deck count, determines every fixture."""

    config = _config(
        tmp_path,
        training=3,
        validation=1,
        test=1,
        continuous=True,
        minimum_free_disk_bytes=1,
    )
    fixtures = tuple(
        miner.continuous_deck_fixture(config, ordinal)
        for ordinal in range(10)
    )

    assert [fixture.split for fixture in fixtures] == [
        miner.DeckSplit.TRAINING,
        miner.DeckSplit.TRAINING,
        miner.DeckSplit.TRAINING,
        miner.DeckSplit.VALIDATION,
        miner.DeckSplit.TEST,
    ] * 2
    assert [fixture.index for fixture in fixtures] == [
        0,
        1,
        2,
        0,
        0,
        3,
        4,
        5,
        1,
        1,
    ]
    assert len({fixture.fixture_id for fixture in fixtures}) == len(fixtures)
    assert [
        miner._continuous_fixture_ordinal(config, fixture)
        for fixture in fixtures
    ] == list(range(10))
    with pytest.raises(miner.SamMinerError, match="no finite"):
        miner.fixture_schedule(config)


def test_continuous_disk_floor_seals_an_empty_stoppable_corpus(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Low disk stops cleanly before a deck or private state is committed."""

    config = _config(
        tmp_path,
        continuous=True,
        minimum_free_disk_bytes=1 << 30,
    )
    monkeypatch.setattr(
        miner, "_free_disk_bytes", lambda _config: (1 << 30) - 1
    )
    inspection = miner.mine_corpus(
        config, miner.DeterministicSamTeacherStub()
    )

    assert inspection.deck_count == 0
    assert inspection.row_count == 0
    assert miner._load_json(tmp_path / "state.json") == {
        "format_version": miner.SAM_MINER_STATE_FORMAT_VERSION,
        "collection_config_digest": config.digest,
        "phase": "disk-floor",
        "next_deck_ordinal": 0,
        "free_disk_bytes": (1 << 30) - 1,
    }
    assert miner.verify_corpus(tmp_path) == inspection


def test_continuous_stop_and_resume_match_uninterrupted_prefix(
    tmp_path: Path,
) -> None:
    """Stopping after sealed decks cannot change the eventual corpus prefix."""

    resumed = _config(
        tmp_path / "resumed",
        root_seed="continuous-resume",
        training=2,
        validation=1,
        test=1,
        continuous=True,
        minimum_free_disk_bytes=1,
    )

    def stop_at(root: Path, count: int):
        return lambda: len(
            tuple(root.glob("decks/*/*/deck-manifest.json"))
        ) >= count

    first = miner.mine_corpus(
        resumed,
        miner.DeterministicSamTeacherStub(),
        should_stop=stop_at(resumed.output_path, 1),
    )
    assert first.deck_count == 1
    second = miner.mine_corpus(
        resumed,
        miner.DeterministicSamTeacherStub(),
        resume=True,
        should_stop=stop_at(resumed.output_path, 2),
    )
    assert second.deck_count == 2

    direct = _config(
        tmp_path / "direct",
        root_seed="continuous-resume",
        training=2,
        validation=1,
        test=1,
        continuous=True,
        minimum_free_disk_bytes=1,
    )
    control = miner.mine_corpus(
        direct,
        miner.DeterministicSamTeacherStub(),
        should_stop=stop_at(direct.output_path, 2),
    )

    assert second.corpus_manifest_digest == control.corpus_manifest_digest
    assert _artifact_snapshot(
        resumed.output_path, "decks"
    ) == _artifact_snapshot(direct.output_path, "decks")
    assert miner._load_json(resumed.output_path / "state.json")["phase"] == (
        "stopped"
    )


def test_continuous_parallel_prefix_matches_sequential_prefix(
    tmp_path: Path,
) -> None:
    """Worker batching cannot change an infinite fixture stream or its prefix."""

    def stop_at(root: Path, count: int):
        return lambda: len(
            tuple(root.glob("decks/*/*/deck-manifest.json"))
        ) >= count

    sequential = _config(
        tmp_path / "sequential-continuous",
        root_seed="continuous-workers",
        training=2,
        validation=1,
        test=1,
        child_count=1,
        workers=1,
        continuous=True,
        minimum_free_disk_bytes=1,
    )
    parallel = replace(
        sequential,
        output_directory=str(tmp_path / "parallel-continuous"),
        workers=2,
    )

    sequential_result = miner.mine_corpus(
        sequential,
        miner.DeterministicSamTeacherStub(),
        should_stop=stop_at(sequential.output_path, 2),
    )
    parallel_result = miner.mine_corpus(
        parallel,
        miner.DeterministicSamTeacherStub(),
        should_stop=stop_at(parallel.output_path, 2),
    )

    assert sequential.digest == parallel.digest
    assert (
        sequential_result.config_digest,
        sequential_result.corpus_manifest_digest,
        sequential_result.deck_count,
        sequential_result.round_count,
        sequential_result.shard_count,
        sequential_result.row_count,
        sequential_result.terminal_leaf_count,
        sequential_result.cache_entry_count,
    ) == (
        parallel_result.config_digest,
        parallel_result.corpus_manifest_digest,
        parallel_result.deck_count,
        parallel_result.round_count,
        parallel_result.shard_count,
        parallel_result.row_count,
        parallel_result.terminal_leaf_count,
        parallel_result.cache_entry_count,
    )
    assert _artifact_snapshot(
        sequential.output_path, "decks"
    ) == _artifact_snapshot(parallel.output_path, "decks")
    assert miner.verify_corpus(sequential.output_path) == sequential_result
    assert miner.verify_corpus(parallel.output_path) == parallel_result


def test_continuous_disk_limits_are_operational_not_corpus_identity(
    tmp_path: Path,
) -> None:
    """Changing safety headroom cannot change any scheduled deck or row."""

    config = _config(
        tmp_path,
        continuous=True,
        minimum_free_disk_bytes=1 << 30,
    )
    adjusted = replace(
        config,
        minimum_free_disk_bytes=2 << 30,
        inflight_disk_reserve_per_worker_bytes=512 << 20,
    )

    assert config.digest == adjusted.digest
    assert [
        miner.continuous_deck_fixture(config, ordinal)
        for ordinal in range(20)
    ] == [
        miner.continuous_deck_fixture(adjusted, ordinal)
        for ordinal in range(20)
    ]


def test_continuous_manifest_uses_only_gap_free_committed_prefix(
    tmp_path: Path,
) -> None:
    """A later worker may seal first, but it cannot create a manifest gap."""

    config = _config(
        tmp_path,
        training=2,
        validation=1,
        test=1,
        continuous=True,
        minimum_free_disk_bytes=1,
    )
    fixtures = [
        miner.continuous_deck_fixture(config, ordinal)
        for ordinal in range(4)
    ]

    def document(ordinal: int) -> dict[str, object]:
        fixture = fixtures[ordinal]
        return {
            "content": {
                "split": fixture.split.value,
                "fixture_index": fixture.index,
            }
        }

    prefix = miner._continuous_committed_prefix(
        config, [document(0), document(2), document(3)]
    )
    assert prefix == [document(0)]
    assert miner._continuous_committed_prefix(
        config, [document(3), document(1), document(0), document(2)]
    ) == [document(index) for index in range(4)]


def test_production_configuration_is_exactly_32_by_32(
    tmp_path: Path,
) -> None:
    """The corpus identity cannot silently change the approved teacher."""

    config = miner.SamMinerConfig(
        "production",
        "production-root",
        str(tmp_path),
        1,
        0,
        0,
    )
    assert config.teacher_search_config == SamTeacherSearchConfig(32, 32)
    assert (
        config.teacher_configuration_digest
        == SamTeacherSearchConfig(32, 32).digest
    )
    with pytest.raises(miner.SamMinerError, match="exact 32x32"):
        replace(
            config,
            teacher_outer_simulation_budget=64,
            teacher_configuration_digest=SamTeacherSearchConfig(
                64, 32
            ).digest,
        )


def test_new_configuration_seals_revision_and_source_tree(
    tmp_path: Path,
) -> None:
    """A run binds both its committed base and relevant working-tree bytes."""

    first = miner.resolve_source_identity()
    second = miner.resolve_source_identity()
    config = miner._production_config(
        run_id="source-bound",
        root_seed="source-bound",
        output_directory=str(tmp_path),
        training_decks=1,
        validation_decks=0,
        test_decks=0,
        child_count=4,
    )

    assert first == second
    assert config.source_tree_schema_version == first.schema_version
    assert config.source_revision == first.revision
    assert config.source_tree_digest == first.tree_digest
    miner.initialize_corpus(config)
    resolved = json.loads(
        (tmp_path / "resolved-config.json").read_text(encoding="utf-8")
    )
    assert resolved["format_version"] == miner.SAM_MINER_CONFIG_FORMAT_VERSION
    assert resolved["source_revision"] == first.revision
    assert resolved["source_tree_digest"] == first.tree_digest


def test_collection_rejects_missing_or_changed_source_identity(
    tmp_path: Path,
) -> None:
    """Search cannot run from unbound or subsequently changed source."""

    bound = _config(tmp_path / "bound")
    miner._verify_source_identity(bound, required=True)
    with pytest.raises(miner.SamMinerError, match="source tree differs"):
        miner._verify_source_identity(
            replace(bound, source_tree_digest="0" * 64),
            required=True,
        )

    legacy = replace(
        bound,
        output_directory=str(tmp_path / "legacy"),
        source_tree_schema_version=None,
        source_revision=None,
        source_tree_digest=None,
    )
    with pytest.raises(miner.SamMinerError, match="must seal"):
        miner.initialize_corpus(legacy)


def test_real_adapter_group_and_concrete_match_direct_search(
    tmp_path: Path,
) -> None:
    """The adapter label must be the unmodified direct search decision."""

    config, teacher, _, state = _reduced_real(tmp_path)
    information = information_state_from_engine(state)
    request = derive_sam_teacher_request_seed(
        information, config.teacher_configuration_digest
    )
    direct = SamTeacherInformationSetSearch(
        config.teacher_search_config
    ).search(information, request)
    selected = teacher.select(information)

    assert selected.selected_representative_action_index == (
        direct.selected_representative_action_index
    )
    assert (
        selected.selected_concrete_action_index
        == direct.selected_action_index
    )


def test_cache_enabled_and_disabled_prefixes_are_identical(
    tmp_path: Path,
) -> None:
    """Caching may remove searches but cannot alter labels or branch paths."""

    cached_config, cached_teacher, fixture, state = _reduced_real(
        tmp_path / "cached"
    )
    uncached_config, uncached_teacher, _, uncached_state = _reduced_real(
        tmp_path / "uncached"
    )
    root = miner._root_path_digest(fixture.fixture_id, 1)
    cached = miner.mine_branch_prefix(
        state,
        fixture_id=fixture.fixture_id,
        path_digest=root,
        maximum_placement=6,
        teacher=cached_teacher,
        config=cached_config,
        use_teacher_cache=True,
    )
    repeated = miner.mine_branch_prefix(
        state,
        fixture_id=fixture.fixture_id,
        path_digest=root,
        maximum_placement=6,
        teacher=cached_teacher,
        config=cached_config,
        use_teacher_cache=True,
    )
    uncached = miner.mine_branch_prefix(
        uncached_state,
        fixture_id=fixture.fixture_id,
        path_digest=root,
        maximum_placement=6,
        teacher=uncached_teacher,
        config=uncached_config,
        use_teacher_cache=False,
    )

    assert tuple(row.to_dict() for row in cached.rows) == tuple(
        row.to_dict() for row in uncached.rows
    )
    assert (
        cached.frontier_path_digests
        == repeated.frontier_path_digests
        == uncached.frontier_path_digests
    )
    assert cached_teacher.query_count == 1
    assert uncached_teacher.query_count == 1


def test_teacher_cache_is_shared_by_equal_information_not_fixture(
    tmp_path: Path,
) -> None:
    """Cache identity is the player view, never its branch or deck origin."""

    config, teacher, _, state = _reduced_real(tmp_path)
    information = information_state_from_engine(state)
    groups = sam_teacher_action_groups(information)
    first = miner.resolve_teacher(
        information, groups, teacher, config
    )
    second = miner.resolve_teacher(
        information, groups, teacher, config
    )

    assert first.cache_hit is False
    assert second.cache_hit is True
    assert teacher.query_count == 1
    assert first.selected_group_index == second.selected_group_index
    assert (
        first.selected_concrete_action_index
        == second.selected_concrete_action_index
    )


def test_real_teacher_label_is_invariant_to_hidden_world_substitution(
    tmp_path: Path,
) -> None:
    """Equal actor views from different sampled worlds have one cached label."""

    config, teacher, _, state = _reduced_real(tmp_path)
    information = information_state_from_engine(state)
    first_world = sample_determinization(
        information,
        derive_seed("sam-miner-hidden-test-v1", "a"),
    ).state
    second_world = sample_determinization(
        information,
        derive_seed("sam-miner-hidden-test-v1", "b"),
    ).state
    first = information_state_from_simulation(first_world)
    second = information_state_from_simulation(second_world)
    assert first == second == information

    first_result = miner.resolve_teacher(
        first, sam_teacher_action_groups(first), teacher, config
    )
    second_result = miner.resolve_teacher(
        second, sam_teacher_action_groups(second), teacher, config
    )
    assert first_result.selected_group_index == second_result.selected_group_index
    assert teacher.query_count == 1


def test_teacher_and_three_alternatives_are_all_legal(
    tmp_path: Path,
) -> None:
    """The direct label and all sampled branches must enter the engine legally."""

    config, teacher, fixture, state = _reduced_real(tmp_path)
    information = information_state_from_engine(state)
    groups = sam_teacher_action_groups(information)
    selected = miner.resolve_teacher(
        information, groups, teacher, config
    )
    alternatives = miner._alternative_groups(
        groups,
        selected.selected_group_index,
        information=information,
        config=config,
        fixture_id=fixture.fixture_id,
        parent_path_digest=miner._root_path_digest(
            fixture.fixture_id, 1
        ),
    )
    concrete_actions = [selected.selected_concrete_action_index]
    concrete_actions.extend(
        miner._concrete_action(
            information,
            group.representative_action_index,
            config=config,
            parent_path_digest=miner._root_path_digest(
                fixture.fixture_id, 1
            ),
            child_ordinal=ordinal,
        )
        for ordinal, group in enumerate(alternatives, start=1)
    )

    assert len(concrete_actions) == 4
    assert len(set(group.representative_action_index for group in alternatives)) == 3
    for action_index in concrete_actions:
        next_state = miner._apply_action(
            state, information, action_index
        )
        assert len(next_state.current_round_moves) == 6


def test_invalid_or_interrupted_real_result_cannot_enter_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Search failure returns before cache or subtree commitment."""

    config, teacher, _, state = _reduced_real(tmp_path)
    information = information_state_from_engine(state)
    groups = sam_teacher_action_groups(information)

    def interrupted(*_args, **_kwargs):
        raise SearchInterrupted("stop")

    monkeypatch.setattr(teacher._planner, "search", interrupted)
    with pytest.raises(miner.SamMinerInterrupted):
        miner.resolve_teacher(information, groups, teacher, config)
    assert not tuple((tmp_path / "cache").rglob("*.json"))

    class InvalidResult:
        pass

    monkeypatch.setattr(
        teacher._planner,
        "search",
        lambda *_args, **_kwargs: InvalidResult(),
    )
    with pytest.raises(miner.SamMinerError, match="incompatible"):
        miner.resolve_teacher(information, groups, teacher, config)
    assert not tuple((tmp_path / "cache").rglob("*.json"))


def test_nonfinite_or_drifted_diagnostics_are_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hash-compatible output still fails if values or identity drift."""

    config, teacher, _, state = _reduced_real(tmp_path)
    information = information_state_from_engine(state)
    request = derive_sam_teacher_request_seed(
        information, config.teacher_configuration_digest
    )
    valid = SamTeacherInformationSetSearch(
        config.teacher_search_config
    ).search(information, request)
    original_values = tuple(valid.mean_action_values)
    original_visits = tuple(valid.action_visits)
    malformed_values = original_values
    legal_index = next(
        index for index, value in enumerate(malformed_values)
        if value is not None
    )
    mutable = list(malformed_values)
    mutable[legal_index] = math.nan
    object.__setattr__(valid, "mean_action_values", tuple(mutable))
    monkeypatch.setattr(
        teacher._planner, "search", lambda *_args, **_kwargs: valid
    )

    with pytest.raises(miner.SamMinerError, match="malformed"):
        teacher.select(information)

    object.__setattr__(valid, "mean_action_values", original_values)
    visits = list(original_visits)
    visits[legal_index] += 1
    object.__setattr__(valid, "action_visits", tuple(visits))
    with pytest.raises(miner.SamMinerError, match="visits"):
        teacher.select(information)

    object.__setattr__(valid, "action_visits", original_visits)
    object.__setattr__(valid, "config_digest", "0" * 64)
    with pytest.raises(miner.SamMinerError, match="identity"):
        teacher.select(information)

    object.__setattr__(
        valid, "config_digest", config.teacher_configuration_digest
    )
    object.__setattr__(
        valid, "information_state_fingerprint", "f" * 64
    )
    with pytest.raises(miner.SamMinerError, match="identity"):
        teacher.select(information)


def test_prefix_smoke_artifact_is_private_and_digest_verified(
    tmp_path: Path,
) -> None:
    """A production-shaped prefix seals rows only and never completes a round."""

    config = miner._production_config(
        run_id="prefix-smoke",
        root_seed="prefix-smoke",
        output_directory=str(tmp_path),
        training_decks=1,
        validation_decks=0,
        test_decks=0,
        child_count=4,
    )
    miner.initialize_corpus(config)
    inspection = miner.mine_real_teacher_smoke(
        config,
        ProductionShapeTeacher(),
        maximum_placement=2,
    )
    document = miner.load_prefix_smoke(
        tmp_path / inspection.relative_path, config
    )

    assert inspection.row_count == 5
    assert inspection.frontier_count == 16
    assert document["content_digest"] == inspection.content_digest
    miner._assert_private_fields_absent(document)
    assert all(
        row["placement_number"] <= 2
        for row in document["content"]["rows"]
    )


@pytest.mark.parametrize("forbidden", sorted(miner._FORBIDDEN_KEYS))
def test_privacy_validator_rejects_every_forbidden_field(
    forbidden: str,
) -> None:
    """No future schema extension may quietly seal private search evidence."""

    with pytest.raises(miner.SamMinerError, match="forbidden"):
        miner._assert_private_fields_absent({forbidden: "private"})


def test_manifest_deck_order_is_canonical(tmp_path: Path) -> None:
    """Manifest identity depends on fixture order, not task completion order."""

    config = _config(
        tmp_path, training=1, validation=1, test=1
    )
    miner.initialize_corpus(config)
    documents = [
        miner._mine_deck_artifacts(
            fixture,
            teacher=miner.DeterministicSamTeacherStub(),
            config=config,
        )
        for fixture in miner.fixture_schedule(config)
    ]
    forward = miner._write_corpus_manifest(config, documents)
    reverse = miner._write_corpus_manifest(config, tuple(reversed(documents)))

    assert forward["content_digest"] == reverse["content_digest"]
    assert forward["content"]["decks"] == reverse["content"]["decks"]


def test_cli_initialize_mine_inspect_verify_and_summarize(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The complete recovery surface remains usable without production Sam."""

    output = tmp_path / "cli"
    common = [
        "--output",
        str(output),
    ]
    assert (
        miner.main(
            [
                "initialize",
                *common,
                "--run-id",
                "cli",
                "--root-seed",
                "cli-root",
                "--training-decks",
                "1",
                "--validation-decks",
                "0",
                "--test-decks",
                "0",
                "--child-count",
                "1",
                "--teacher-profile",
                "deterministic-stub",
            ]
        )
        == 0
    )
    for command in ("mine", "inspect", "verify", "summarize"):
        assert miner.main([command, *common]) == 0
        assert "row_count" in capsys.readouterr().out

"""Accepted-pi0 response and separately versioned D1 collection contracts."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

import dracula.bgc_pi0_miner as d1
import dracula.bgc_policy_evaluation as evaluation
from dracula.accepted_pi0 import (
    AcceptedPi0BGCSearch,
    AcceptedPi0ContinuationAdapter,
)
from dracula.belief_greedy_miner import (
    BalancedMinerConfig,
    _continuous_resolved_document,
    _manifest_document,
    resolve_source_identity,
)
from dracula.bgc_policy import load_bgc_policy_artifact, save_bgc_policy_artifact
from dracula.bgc_policy_model import BGCPolicyModel
from dracula.bridge import move_for_action_index
from dracula.engine import (
    EnginePlayer,
    EngineStatus,
    advance_after_round,
    apply_move,
    create_game,
    legal_moves,
    other_player,
)
from dracula.randomness import derive_seed
from dracula.search import (
    BeliefGreedyInformationSetSearch,
    BeliefGreedySearchConfig,
    derive_belief_greedy_request_seed,
    information_state_from_engine,
    sample_determinization,
    select_concrete_action_index,
    strategic_action_groups,
)


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()


def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, separators=(",", ":"), sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _evaluation_records(
    fixtures: evaluation.EvaluationFixtureManifest,
) -> tuple[evaluation.GameEvaluationRecord, ...]:
    records = []
    for fixture in fixtures.fixtures[: fixtures.protocol.primary_deck_count]:
        for role in (EnginePlayer.QUEEN, EnginePlayer.KING):
            rounds = tuple(
                evaluation.RoundEvaluationRecord(
                    round_number,
                    round_number % 2 == 0,
                    21 if round_number == 1 else 20,
                    20,
                )
                for round_number in range(1, 7)
            )
            records.append(
                evaluation.GameEvaluationRecord(
                    fixture.fixture_id,
                    role.value,
                    _digest({"fixture": fixture.fixture_id, "role": role.value}),
                    True,
                    False,
                    121,
                    120,
                    rounds,
                    42,
                    0.042,
                    0.002,
                    128_000_000,
                    0,
                )
            )
    return tuple(records)


@pytest.fixture(scope="module")
def accepted_bundle(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("accepted-pi0-d1")
    candidate = root / "candidate.pt"
    source = resolve_source_identity()
    model = BGCPolicyModel(
        run_root_seed="accepted-pi0-test-root",
        model_id="accepted-pi0-test-model",
        initialization_ordinal=0,
    )
    save_bgc_policy_artifact(
        candidate,
        model,
        source_revision=source.revision,
        source_tree_digest=source.tree_digest,
        training_configuration={"purpose": "D1 deterministic acceptance test"},
        corpus_snapshot_digest=_digest("D1-test-snapshot"),
        dataset_digest=_digest("D1-test-dataset"),
    )
    fixtures = evaluation.create_evaluation_fixture_manifest(
        root / "fixtures.json",
        root_seed="D1-accepted-evaluation-fixtures",
    )
    loaded = load_bgc_policy_artifact(candidate)
    artifact_digest = _file_digest(candidate)
    identities = evaluation._expected_controller_identities(
        loaded, artifact_digest, fixtures.protocol
    )
    records = _evaluation_records(fixtures)
    reports = {}
    for comparison_id, identity in identities.items():
        report = root / f"{comparison_id}.json"
        evaluation._seal_comparison_records(
            comparison_id=comparison_id,
            comparison_type=(
                "controller" if comparison_id == "pi0-bgc-vs-base-bgc" else "standalone"
            ),
            records=records,
            subject_identity=identity[0],
            subject_digest=identity[1],
            control_identity=identity[2],
            control_digest=identity[3],
            fixture_manifest=fixtures,
            artifact_digest=artifact_digest,
            output_path=report,
            extended=False,
        )
        reports[comparison_id] = report
    evidence = root / "validation-evidence.json"
    decision = evaluation.build_validation_evidence(
        candidate_artifact_path=candidate,
        fixture_manifest=fixtures,
        comparison_paths=reports,
        output_path=evidence,
    )
    assert decision.status == "passed"
    accepted = root / "accepted"
    accepted_decision = evaluation.accept_pi0_candidate(
        candidate_artifact_path=candidate,
        validation_evidence_path=evidence,
        fixture_manifest=fixtures,
        output_directory=accepted,
    )
    assert accepted_decision.status == "passed"
    return accepted


@pytest.fixture()
def d1_config(accepted_bundle: Path) -> d1.D1MinerConfig:
    return d1.D1MinerConfig.resolve(
        root_seed="D1-miner-test-root",
        accepted_pi0_directory=accepted_bundle,
        worker_count=1,
    )


def _synthetic_game(config: d1.D1MinerConfig, ordinal: int) -> d1.MinedD1Game:
    fixture = d1._fixture_id(config, ordinal)
    profile = d1._profile(ordinal)
    state = create_game(d1._deck_seed(config, ordinal))
    rows = []
    while state.status is not EngineStatus.GAME_COMPLETE:
        if state.status is EngineStatus.ROUND_COMPLETE:
            state = advance_after_round(state)
            continue
        assert state.active_player is not None
        placement = len(state.current_round_moves) + 1
        if placement == 8:
            moves = legal_moves(state, state.active_player)
            assert len(moves) == 1
            state = apply_move(state, moves[0]).state
            continue
        information = information_state_from_engine(state)
        groups = strategic_action_groups(information, True)
        selected = groups[0].representative_action_index
        visits = [1] * len(groups)
        visits[0] += 128 - len(groups)
        rows.append(
            d1._row(
                information,
                groups,
                tuple(visits),
                selected,
                fixture,
                profile,
                config,
            )
        )
        concrete = select_concrete_action_index(
            information,
            selected,
            derive_seed(
                "d1-synthetic-action-v1",
                str(ordinal),
                str(state.round_number),
                str(placement),
            ),
            True,
        )
        state = apply_move(
            state, move_for_action_index(information.player, concrete)
        ).state
    content = {
        "acceptance_manifest_digest": config.acceptance_manifest_digest,
        "accepted_checkpoint_digest": config.accepted_checkpoint_digest,
        "cache_schema_version": d1.D1_CACHE_SCHEMA_VERSION,
        "configuration_content_digest": config.content_digest,
        "controller_digest": config.controller_digest,
        "dataset_schema_version": d1.D1_DATASET_SCHEMA_VERSION,
        "fixture_id": fixture,
        "game_schema_version": d1.D1_GAME_SCHEMA_VERSION,
        "ordinal": ordinal,
        "response_schema_version": d1.D1_RESPONSE_SCHEMA_VERSION,
        "rows": [row.canonical_data() for row in rows],
        "search_schema_version": d1.D1_SEARCH_SCHEMA_VERSION,
        "trajectory_profile": profile.value,
    }
    return d1.MinedD1Game(
        ordinal,
        fixture,
        profile,
        tuple(rows),
        d1._digest(content),
        0.0,
        0.0,
        0,
        0,
        128_000_000,
    )


def test_accepted_bundle_and_adapter_are_digest_bound_information_only_and_exact(
    accepted_bundle: Path,
) -> None:
    bundle = evaluation.load_accepted_pi0_bundle(accepted_bundle)
    adapter = AcceptedPi0ContinuationAdapter(bundle)
    root = information_state_from_engine(create_game("D1-adapter-hidden-boundary"))
    hidden_a = sample_determinization(root, derive_seed("d1-hidden-v1", "a")).state
    hidden_b = sample_determinization(root, derive_seed("d1-hidden-v1", "b")).state
    assert hidden_a.hands[other_player(root.player)] != hidden_b.hands[
        other_player(root.player)
    ]
    view_a = information_state_from_engine(hidden_a)
    view_b = information_state_from_engine(hidden_b)
    assert view_a == view_b == root

    calls = 0
    original_forward = adapter.model.forward

    def counted_forward(observation):
        nonlocal calls
        calls += 1
        return original_forward(observation)

    adapter.model.forward = counted_forward  # type: ignore[method-assign]
    first = adapter.evaluate(view_a)
    assert calls == 1
    repeated = adapter.evaluate(view_b)
    assert calls == 2
    assert first == repeated
    group = next(
        item.group
        for item in first.group_diagnostics
        if item.group.representative_action_index
        == first.selected_representative_action_index
    )
    assert first.selected_action_index in group.member_action_indices
    move = move_for_action_index(root.player, first.selected_action_index)
    assert move in legal_moves(create_game("D1-adapter-hidden-boundary"), root.player)


def test_candidate_preserves_base_outer_search_and_exact_root_visits(
    accepted_bundle: Path,
) -> None:
    adapter = AcceptedPi0ContinuationAdapter.from_directory(accepted_bundle)
    config = BeliefGreedySearchConfig(
        outer_simulation_budget=128, belief_completion_count=8
    )
    candidate = AcceptedPi0BGCSearch(adapter, config)
    base = BeliefGreedyInformationSetSearch(config)
    assert candidate.config == base.config == config
    assert candidate.__class__._actor_response is not base.__class__._actor_response
    information = information_state_from_engine(create_game("D1-root-visits"))
    request = derive_belief_greedy_request_seed(
        "D1-root-visit-fixture", information, config.digest
    )
    result = candidate.search(information, request)
    assert result.simulation_count == 128
    assert sum(item.visits for item in result.group_diagnostics) == 128
    assert all(item.visits > 0 for item in result.group_diagnostics)
    assert -1.0 <= result.principal_continuation.terminal_value <= 1.0
    assert move_for_action_index(
        information.player, result.selected_action_index
    ) in legal_moves(create_game("D1-root-visits"), information.player)


def test_d1_rows_are_balanced_private_exact_and_separate_from_d0(
    tmp_path: Path,
    d1_config: d1.D1MinerConfig,
) -> None:
    output = tmp_path / "D1"
    d1.initialize_d1_corpus(
        d1_config, output=output, minimum_free_disk_bytes=1
    )
    entries: list[dict[str, object]] = []
    game = _synthetic_game(d1_config, 0)
    d1.commit_d1_games(output, d1_config, entries, (game,))
    verified = d1.verify_d1_corpus(
        output, d1_config.accepted_pi0_directory
    )
    assert verified["verified_game_count"] == 1
    assert verified["row_count"] == 42
    document = json.loads((output / "games/000000.json").read_text())
    rows = document["content"]["rows"]
    assert {row["placement_number"] for row in rows} == set(range(1, 8))
    assert all(sum(row["strategic_group_visits"]) == 128 for row in rows)
    encoded = json.dumps(document).lower()
    for forbidden in (
        "authoritative_state",
        "determinization",
        "engine_seed",
        "model_state",
        "opponent_hand",
        "search_tree",
        "stock_order",
    ):
        assert forbidden not in encoded

    for field, replacement in (
        ("row_schema_version", "dracula-belief-greedy-balanced-row-v2"),
        ("cache_schema_version", "dracula-belief-greedy-response-cache-v0"),
    ):
        mixed = json.loads(json.dumps(document))
        mixed["content"]["rows"][0][field] = replacement
        mixed["content_digest"] = d1._digest(mixed["content"])
        mixed_path = tmp_path / f"mixed-{field}.json"
        _write(mixed_path, mixed)
        with pytest.raises(d1.D1MinerError):
            d1._verify_game(mixed_path, config=d1_config, expected_ordinal=0)

    # A D0 manifest cannot be inspected or resumed as D1.
    d0 = tmp_path / "D0"
    d0.mkdir()
    d0_config = BalancedMinerConfig("D0-separation", 128, 8, 1)
    source = resolve_source_identity()
    _write(
        d0 / "resolved-config.json",
        _continuous_resolved_document(
            d0_config, minimum_free_disk_bytes=1, source=source
        ),
    )
    _write(d0 / "corpus-manifest.json", _manifest_document(d0_config, []))
    with pytest.raises(d1.D1MinerError, match="not a D1"):
        d1.verify_d1_corpus(d0, d1_config.accepted_pi0_directory)


def test_atomic_complete_game_resume_is_canonical(
    tmp_path: Path,
    d1_config: d1.D1MinerConfig,
) -> None:
    resumed = tmp_path / "resumed"
    direct = tmp_path / "direct"
    for output in (resumed, direct):
        d1.initialize_d1_corpus(
            d1_config, output=output, minimum_free_disk_bytes=1
        )
    first = _synthetic_game(d1_config, 0)
    second = _synthetic_game(d1_config, 1)
    resumed_entries: list[dict[str, object]] = []
    d1.commit_d1_games(resumed, d1_config, resumed_entries, (first,))
    resumed_entries = d1._load_entries(resumed, d1_config)
    d1.commit_d1_games(resumed, d1_config, resumed_entries, (second,))
    direct_entries: list[dict[str, object]] = []
    # Worker completion order is irrelevant to the sealed canonical order.
    d1.commit_d1_games(direct, d1_config, direct_entries, (second, first))
    assert (resumed / "corpus-manifest.json").read_bytes() == (
        direct / "corpus-manifest.json"
    ).read_bytes()
    assert not tuple(resumed.rglob("*.partial"))


def test_missing_failed_or_tampered_acceptance_cannot_initialize_d1(
    tmp_path: Path,
    accepted_bundle: Path,
) -> None:
    missing_output = tmp_path / "missing-output"
    with pytest.raises(d1.D1MinerError, match="valid accepted"):
        d1.D1MinerConfig.resolve(
            root_seed="missing",
            accepted_pi0_directory=tmp_path / "missing-accepted",
        )
    assert not missing_output.exists()

    tampered = tmp_path / "tampered"
    shutil.copytree(accepted_bundle, tampered)
    manifest_path = tampered / "acceptance-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["content"]["evaluation_result"] = "failed"
    rewritten = evaluation._envelope(
        evaluation.ACCEPTANCE_SCHEMA_VERSION, manifest["content"]
    )
    manifest_path.write_text(json.dumps(rewritten), encoding="utf-8")
    with pytest.raises(d1.D1MinerError, match="valid accepted"):
        d1.D1MinerConfig.resolve(
            root_seed="failed",
            accepted_pi0_directory=tampered,
        )

    corrupted = tmp_path / "corrupted"
    shutil.copytree(accepted_bundle, corrupted)
    checkpoint = corrupted / "accepted-pi0.pt"
    checkpoint.write_bytes(checkpoint.read_bytes() + b"corrupt")
    with pytest.raises(d1.D1MinerError, match="valid accepted"):
        d1.D1MinerConfig.resolve(
            root_seed="corrupt",
            accepted_pi0_directory=corrupted,
        )


def test_cli_reports_missing_acceptance_without_an_internal_traceback(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = tmp_path / "refused-D1"
    result = d1.main(
        [
            "initialize",
            "--output",
            str(output),
            "--accepted-pi0",
            str(tmp_path / "missing-accepted"),
            "--root-seed",
            "missing-acceptance-cli",
        ]
    )
    captured = capsys.readouterr()
    assert result == 2
    assert captured.err.strip() == "D1 requires a valid accepted pi0 bundle"
    assert "Traceback" not in captured.err
    assert not output.exists()


def test_adapter_interruption_has_no_fallback_or_partial_action(
    accepted_bundle: Path,
) -> None:
    adapter = AcceptedPi0ContinuationAdapter.from_directory(accepted_bundle)
    information = information_state_from_engine(create_game("D1-interruption"))
    with pytest.raises(Exception, match="interrupted"):
        adapter.evaluate(information, should_stop=lambda: True)


def test_interrupted_worker_batch_commits_no_partial_game(
    tmp_path: Path,
    d1_config: d1.D1MinerConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "interrupted-D1"

    def interrupted(_config, _ordinals):
        raise RuntimeError("synthetic worker interruption")

    monkeypatch.setattr(d1, "mine_d1_games", interrupted)
    with pytest.raises(RuntimeError, match="worker interruption"):
        d1.run_d1_collection(
            d1_config,
            output=output,
            minimum_free_disk_bytes=1,
            max_games=1,
        )
    verified = d1.verify_d1_corpus(output, d1_config.accepted_pi0_directory)
    assert verified["game_count"] == 0
    assert verified["row_count"] == 0
    if (output / "games").exists():
        assert not tuple((output / "games").glob("*.json"))

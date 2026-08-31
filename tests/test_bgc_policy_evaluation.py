"""Mechanical evaluation and atomic pi0-acceptance contract tests."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import pytest

import dracula.bgc_policy_evaluation as evaluation
from dracula.belief_greedy_miner import resolve_source_identity
from dracula.bgc_policy import load_bgc_policy_artifact, save_bgc_policy_artifact
from dracula.bgc_policy_model import BGCPolicyModel
from dracula.engine import EnginePlayer, create_game, legal_moves
from dracula.search import (
    BeliefGreedyResponseEvaluator,
    BeliefGreedySearchConfig,
    information_state_from_engine,
    strategic_action_groups,
)


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()


@pytest.fixture(scope="module")
def candidate(tmp_path_factory: pytest.TempPathFactory) -> Path:
    directory = tmp_path_factory.mktemp("pi0-evaluation-candidate")
    path = directory / "unaccepted-pi0.pt"
    source = resolve_source_identity()
    model = BGCPolicyModel(
        run_root_seed="pi0-evaluation-test-root",
        model_id="pi0-evaluation-test",
        initialization_ordinal=0,
    )
    save_bgc_policy_artifact(
        path,
        model,
        source_revision=source.revision,
        source_tree_digest=source.tree_digest,
        training_configuration={"purpose": "synthetic evaluation mechanics"},
        corpus_snapshot_digest=_digest("synthetic-snapshot"),
        dataset_digest=_digest("synthetic-dataset"),
    )
    return path


@pytest.fixture()
def protocol() -> evaluation.EvaluationProtocol:
    return evaluation.EvaluationProtocol(
        primary_deck_count=2,
        extension_deck_count=2,
        bootstrap_samples=200,
        outer_simulation_budget=8,
        belief_completion_count=2,
    )


@pytest.fixture()
def fixtures(
    tmp_path: Path, protocol: evaluation.EvaluationProtocol
) -> evaluation.EvaluationFixtureManifest:
    return evaluation.create_evaluation_fixture_manifest(
        tmp_path / "fixtures.json",
        root_seed="pi0-evaluation-fixtures-test-root",
        protocol=protocol,
    )


def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _records(
    fixture_manifest: evaluation.EvaluationFixtureManifest,
    differences: list[int],
) -> tuple[evaluation.GameEvaluationRecord, ...]:
    records = []
    if len(differences) > len(fixture_manifest.fixtures):
        raise AssertionError("test differences exceed fixture manifest")
    for ordinal, difference in enumerate(differences):
        fixture = fixture_manifest.fixtures[ordinal]
        for role in (EnginePlayer.QUEEN, EnginePlayer.KING):
            rounds = tuple(
                evaluation.RoundEvaluationRecord(
                    round_number,
                    round_number % 2 == 0,
                    20 + (difference if round_number == 1 else 0),
                    20,
                )
                for round_number in range(1, 7)
            )
            subject_score = sum(item.subject_score for item in rounds)
            opponent_score = sum(item.opponent_score for item in rounds)
            records.append(
                evaluation.GameEvaluationRecord(
                    fixture.fixture_id,
                    role.value,
                    _digest(
                        {
                            "difference": difference,
                            "fixture": fixture.fixture_id,
                            "role": role.value,
                        }
                    ),
                    subject_score > opponent_score,
                    subject_score == opponent_score,
                    subject_score,
                    opponent_score,
                    rounds,
                    42,
                    0.042,
                    0.002,
                    128_000_000,
                    0,
                )
            )
    return tuple(records)


def _reports(
    tmp_path: Path,
    candidate: Path,
    fixtures: evaluation.EvaluationFixtureManifest,
    differences: list[int],
) -> dict[str, Path]:
    loaded = load_bgc_policy_artifact(candidate)
    artifact_digest = _file_digest(candidate)
    identities = evaluation._expected_controller_identities(
        loaded, artifact_digest, fixtures.protocol
    )
    extended = len(differences) == fixtures.protocol.total_deck_count
    records = _records(
        fixtures,
        differences,
    )
    result = {}
    for comparison_id, identity in identities.items():
        path = tmp_path / f"{comparison_id}.json"
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
            output_path=path,
            extended=extended,
        )
        result[comparison_id] = path
    return result


def test_random_one_ply_and_pi0_are_deterministic_legal_and_information_only(
    candidate: Path,
) -> None:
    state = create_game("pi0-opponent-mechanics")
    information = information_state_from_engine(state)
    groups = strategic_action_groups(information, True)

    random = evaluation.RandomLegalOpponent()
    first = random.decide(information, fixture_id="fixture-a", decision_index=0)
    second = random.decide(information, fixture_id="fixture-a", decision_index=0)
    assert first.representative_action_index == second.representative_action_index
    assert first.concrete_action_index == second.concrete_action_index

    one_ply = evaluation.OnePlyBeliefGreedyOpponent(8)
    direct = BeliefGreedyResponseEvaluator(
        BeliefGreedySearchConfig(belief_completion_count=8)
    ).evaluate(information)
    greedy = one_ply.decide(information, fixture_id="ignored", decision_index=0)
    assert greedy.representative_action_index == direct.selected_representative_action_index
    assert greedy.concrete_action_index == direct.selected_action_index
    assert all(item.visits == 8 for item in direct.group_diagnostics)

    pi0 = evaluation.StandalonePi0Opponent.from_artifact(candidate)
    policy = pi0.decide(information, fixture_id="fixture-a", decision_index=0)
    assert policy.representative_action_index in {
        group.representative_action_index for group in groups
    }
    for decision in (first, greedy, policy):
        move = evaluation.move_for_action_index(
            information.player, decision.concrete_action_index
        )
        assert move in legal_moves(state, information.player)
    # Every opponent API accepts SearchInformationState, not EngineState or a
    # sampled private world; this is the enforceable privacy boundary.
    assert "information" in evaluation.RandomLegalOpponent.decide.__annotations__
    assert "information" in evaluation.StandalonePi0Opponent.decide.__annotations__


def test_full_game_records_exact_round_scores_and_role_pairs(
    fixtures: evaluation.EvaluationFixtureManifest,
    candidate: Path,
) -> None:
    random = evaluation.RandomLegalOpponent()
    policy = evaluation.StandalonePi0Opponent.from_artifact(candidate)
    records = tuple(
        evaluation._play_game(
            fixtures.fixtures[0],
            subject=policy,
            control=random,
            subject_role=role,
            should_stop=None,
        )
        for role in (EnginePlayer.QUEEN, EnginePlayer.KING)
    )
    assert {record.subject_role for record in records} == {"queen", "king"}
    assert all(len(record.rounds) == 6 for record in records)
    assert all(
        record.subject_score == sum(item.subject_score for item in record.rounds)
        and record.opponent_score == sum(item.opponent_score for item in record.rounds)
        for record in records
    )
    assert len(evaluation._paired_vectors(records)) == 1
    replay = evaluation._play_game(
        fixtures.fixtures[0],
        subject=policy,
        control=random,
        subject_role=EnginePlayer.QUEEN,
        should_stop=None,
    )
    assert replay.action_trace_digest == records[0].action_trace_digest
    assert replay.subject_score == records[0].subject_score
    assert replay.opponent_score == records[0].opponent_score


def test_incremental_controller_seals_each_role_pair_and_resumes(
    tmp_path: Path,
    fixtures: evaluation.EvaluationFixtureManifest,
) -> None:
    output = tmp_path / "controller-progress.json"
    subject = evaluation.RandomLegalOpponent()
    control = evaluation.RandomLegalOpponent()
    stopped = {"value": False}
    updates: list[dict[str, object]] = []

    def record_and_stop(update: dict[str, object]) -> None:
        updates.append(update)
        stopped["value"] = True

    with pytest.raises(evaluation.Pi0EvaluationInterrupted):
        evaluation.run_incremental_controller_comparison(
            subject=subject,
            control=control,
            fixture_manifest=fixtures,
            artifact_digest=_digest("incremental-artifact"),
            output_path=output,
            target_deck_count=2,
            should_stop=lambda: stopped["value"],
            progress_callback=record_and_stop,
        )
    first = json.loads(output.read_text())["content"]
    assert first["completed_deck_count"] == 1
    assert first["complete"] is False
    assert len(first["games"]) == 2
    assert len(first["metrics"]["paired_score_differential_ci_95"]) == 2

    stopped["value"] = False
    resumed_updates: list[dict[str, object]] = []
    evaluation.run_incremental_controller_comparison(
        subject=subject,
        control=control,
        fixture_manifest=fixtures,
        artifact_digest=_digest("incremental-artifact"),
        output_path=output,
        target_deck_count=2,
        should_stop=lambda: False,
        progress_callback=resumed_updates.append,
    )
    final = json.loads(output.read_text())["content"]
    assert final["completed_deck_count"] == 2
    assert final["complete"] is True
    assert len(final["games"]) == 4
    assert [update["completed_deck_count"] for update in updates] == [1]
    assert [update["completed_deck_count"] for update in resumed_updates] == [2]


def test_pi0_controller_changes_only_actor_response_and_remains_reproducible(
    candidate: Path,
) -> None:
    information = information_state_from_engine(create_game("pi0-controller-root"))
    config = BeliefGreedySearchConfig(
        outer_simulation_budget=8, belief_completion_count=2
    )
    policy = evaluation.StandalonePi0Opponent.from_artifact(candidate)
    controller = evaluation.BGCSearchOpponent.pi0(policy, config)
    first = controller.decide(information, fixture_id="controller-fixture", decision_index=0)
    second = controller.decide(information, fixture_id="controller-fixture", decision_index=0)
    assert first.representative_action_index == second.representative_action_index
    assert first.concrete_action_index == second.concrete_action_index
    move = evaluation.move_for_action_index(
        information.player, first.concrete_action_index
    )
    assert move in legal_moves(create_game("pi0-controller-root"), information.player)
    assert controller.planner.config == config
    assert isinstance(controller.planner, evaluation.Pi0ContinuationBeliefGreedySearch)


def test_paired_interval_and_extension_rules_are_deterministic(
    fixtures: evaluation.EvaluationFixtureManifest,
) -> None:
    positive = _records(fixtures, [5, 5, 5, 5])
    first = evaluation.paired_confidence_interval(
        positive,
        comparison_id="standalone-random-legal",
        fixture_manifest_digest=fixtures.digest,
        bootstrap_samples=200,
    )
    second = evaluation.paired_confidence_interval(
        positive,
        comparison_id="standalone-random-legal",
        fixture_manifest_digest=fixtures.digest,
        bootstrap_samples=200,
    )
    assert first == second == (5.0, 5.0)
    positive_metrics = evaluation._comparison_metrics(
        positive[:4],
        comparison_id="standalone-random-legal",
        fixture_manifest_digest=fixtures.digest,
        bootstrap_samples=200,
    )
    negative_metrics = evaluation._comparison_metrics(
        _records(fixtures, [-5, -5, -5, -5])[:4],
        comparison_id="standalone-random-legal",
        fixture_manifest_digest=fixtures.digest,
        bootstrap_samples=200,
    )
    ambiguous_metrics = evaluation._comparison_metrics(
        _records(fixtures, [10, -5, 10, -5])[:4],
        comparison_id="standalone-random-legal",
        fixture_manifest_digest=fixtures.digest,
        bootstrap_samples=200,
    )
    assert evaluation._needs_extension(positive_metrics) is False
    assert evaluation._needs_extension(negative_metrics) is False
    assert evaluation._needs_extension(ambiguous_metrics) is True


@pytest.mark.parametrize(
    ("differences", "expected"),
    [([5, 5], "passed"), ([-5, -5], "failed"), ([10, -5, 10, -5], "inconclusive")],
)
def test_acceptance_pass_fail_and_inconclusive_paths(
    tmp_path: Path,
    candidate: Path,
    fixtures: evaluation.EvaluationFixtureManifest,
    differences: list[int],
    expected: str,
) -> None:
    reports = _reports(tmp_path, candidate, fixtures, differences)
    evidence_path = tmp_path / "validation-evidence.json"
    decision = evaluation.build_validation_evidence(
        candidate_artifact_path=candidate,
        fixture_manifest=fixtures,
        comparison_paths=reports,
        output_path=evidence_path,
    )
    assert decision.status == expected
    accepted = tmp_path / "accepted"
    result = evaluation.accept_pi0_candidate(
        candidate_artifact_path=candidate,
        validation_evidence_path=evidence_path,
        fixture_manifest=fixtures,
        output_directory=accepted,
        require_production_protocol=False,
    )
    assert result.status == expected
    if expected == "passed":
        assert (accepted / "accepted-pi0.pt").is_file()
        assert (accepted / "validation-evidence.json").is_file()
        assert len(tuple((accepted / "comparisons").glob("*.json"))) == 3
        manifest = json.loads((accepted / "acceptance-manifest.json").read_text())
        assert manifest["content"]["evaluation_result"] == "passed"
    else:
        assert not accepted.exists()


def test_interruption_forgery_staleness_and_omitted_extension_commit_nothing(
    tmp_path: Path,
    candidate: Path,
    fixtures: evaluation.EvaluationFixtureManifest,
) -> None:
    output = tmp_path / "interrupted-comparison.json"
    with pytest.raises(evaluation.Pi0EvaluationInterrupted):
        evaluation.run_paired_comparison(
            comparison_id="standalone-random-legal",
            comparison_type="standalone",
            subject=evaluation.StandalonePi0Opponent.from_artifact(candidate),
            control=evaluation.RandomLegalOpponent(),
            fixture_manifest=fixtures,
            artifact_digest=_file_digest(candidate),
            output_path=output,
            should_stop=lambda: True,
        )
    assert not output.exists()

    loaded = load_bgc_policy_artifact(candidate)
    identities = evaluation._expected_controller_identities(
        loaded, _file_digest(candidate), fixtures.protocol
    )["standalone-random-legal"]
    omitted = tmp_path / "omitted-extension.json"
    with pytest.raises(evaluation.Pi0EvaluationError, match="extension"):
        evaluation._seal_comparison_records(
            comparison_id="standalone-random-legal",
            comparison_type="standalone",
            records=_records(fixtures, [10, -5, 10, -5])[:4],
            subject_identity=identities[0],
            subject_digest=identities[1],
            control_identity=identities[2],
            control_digest=identities[3],
            fixture_manifest=fixtures,
            artifact_digest=_file_digest(candidate),
            output_path=omitted,
            extended=False,
        )
    assert not omitted.exists()

    reports = _reports(tmp_path / "valid", candidate, fixtures, [5, 5])
    stale_reports = dict(reports)
    stale_path = tmp_path / "stale-report.json"
    stale_document = json.loads(reports["standalone-random-legal"].read_text())
    stale_content = stale_document["content"]
    stale_content["source_tree_digest"] = "0" * 64
    stale_path.write_text(
        json.dumps(
            evaluation._envelope(
                evaluation.EVALUATION_REPORT_SCHEMA_VERSION, stale_content
            )
        ),
        encoding="utf-8",
    )
    stale_reports["standalone-random-legal"] = stale_path
    with pytest.raises(evaluation.Pi0EvaluationError, match="source evidence is stale"):
        evaluation.build_validation_evidence(
            candidate_artifact_path=candidate,
            fixture_manifest=fixtures,
            comparison_paths=stale_reports,
            output_path=tmp_path / "stale-evidence.json",
        )

    evidence_path = tmp_path / "evidence.json"
    evaluation.build_validation_evidence(
        candidate_artifact_path=candidate,
        fixture_manifest=fixtures,
        comparison_paths=reports,
        output_path=evidence_path,
    )
    forged = json.loads(evidence_path.read_text())
    forged["content"]["overall_status"] = "failed"
    evidence_path.write_text(json.dumps(forged), encoding="utf-8")
    with pytest.raises(evaluation.Pi0EvaluationError):
        evaluation.load_validation_evidence(
            evidence_path,
            candidate_artifact_path=candidate,
            fixture_manifest=fixtures,
        )

    # A stop after evidence verification but before the atomic rename leaves no
    # partially accepted directory.
    evidence_path.unlink()
    evaluation.build_validation_evidence(
        candidate_artifact_path=candidate,
        fixture_manifest=fixtures,
        comparison_paths=reports,
        output_path=evidence_path,
    )
    accepted = tmp_path / "interrupted-acceptance"
    calls = iter((False, False, True))
    with pytest.raises(evaluation.Pi0EvaluationInterrupted):
        evaluation.accept_pi0_candidate(
            candidate_artifact_path=candidate,
            validation_evidence_path=evidence_path,
            fixture_manifest=fixtures,
            output_directory=accepted,
            require_production_protocol=False,
            should_stop=lambda: next(calls, True),
        )
    assert not accepted.exists()


def test_reports_reject_private_or_contradictory_game_evidence(
    fixtures: evaluation.EvaluationFixtureManifest,
) -> None:
    value = json.loads(json.dumps(asdict(_records(fixtures, [5, 5, 5, 5])[0])))
    value["subject_score"] += 1
    with pytest.raises(evaluation.Pi0EvaluationError, match="contradicts"):
        evaluation._record_from_json(value)
    with pytest.raises(evaluation.Pi0EvaluationError, match="private"):
        evaluation._assert_report_privacy({"search_tree": []})

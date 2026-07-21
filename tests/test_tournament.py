"""Candidate round-robin tournament invariants."""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import pytest
import torch

from dracula.collection import PolicyVersion
from dracula.evaluation import EvaluationFixtureResult
from dracula.tournament import (
    RANDOM_LEGAL_IDENTITY,
    TournamentContractViolation,
    UniformLegalPolicy,
    _markdown_report,
    _tournament_document,
    build_candidate_tournament_schedule,
)


def _candidates(count: int = 3) -> tuple[PolicyVersion, ...]:
    return tuple(
        PolicyVersion(f"policy-{index}", f"policy-{index}-v7")
        for index in range(count)
    )


# The control is an action generator: it owns no weights or adaptive state.
def test_random_legal_controller_is_uniform_parameterless_and_stateless() -> None:
    controller = UniformLegalPolicy()
    observation = torch.zeros(875, dtype=torch.bool)
    legal_mask = torch.zeros(4, 8, dtype=torch.bool)
    legal_mask[0, :3] = True
    hidden = controller.initial_hidden()
    logits, hidden_out = controller(observation, legal_mask, hidden)
    assert tuple(controller.parameters()) == ()
    assert logits.shape == (4, 8)
    assert torch.equal(logits, torch.zeros_like(logits))
    assert torch.equal(hidden_out, hidden)
    assert hidden_out.data_ptr() != hidden.data_ptr()


# Every model pair and every model-control pair sees the identical deck set.
def test_candidate_schedule_is_complete_balanced_round_robin() -> None:
    candidates = _candidates()
    schedule = build_candidate_tournament_schedule(
        candidates,
        ("lane-0", "lane-1"),
        generation_count=2,
    )
    assert len(schedule.fixtures) == 24
    expected_seeds = {fixture.game_seed for fixture in schedule.fixtures}
    assert len(expected_seeds) == 4

    expected_pairs = {
        frozenset(pair)
        for pair in (
            *tuple(combinations(candidates, 2)),
            *((candidate, RANDOM_LEGAL_IDENTITY) for candidate in candidates),
        )
    }
    actual_pairs = {
        frozenset((fixture.policy_a, fixture.policy_b))
        for fixture in schedule.fixtures
    }
    assert actual_pairs == expected_pairs
    for pair in expected_pairs:
        fixtures = [
            fixture
            for fixture in schedule.fixtures
            if frozenset((fixture.policy_a, fixture.policy_b)) == pair
        ]
        assert {fixture.game_seed for fixture in fixtures} == expected_seeds
        first = min(pair)
        assert sum(fixture.queen == first for fixture in fixtures) == 2
        assert sum(fixture.king == first for fixture in fixtures) == 2


def test_candidate_schedule_rejects_controls_and_duplicate_policy_ids() -> None:
    with pytest.raises(TournamentContractViolation, match="control"):
        build_candidate_tournament_schedule(
            (_candidates(1)[0], RANDOM_LEGAL_IDENTITY),
            ("lane-0",),
        )
    with pytest.raises(TournamentContractViolation, match="policy IDs"):
        build_candidate_tournament_schedule(
            (
                PolicyVersion("policy", "policy-v1"),
                PolicyVersion("policy", "policy-v2"),
            ),
            ("lane-0",),
        )


def _decisive_results(
    candidates: tuple[PolicyVersion, ...],
    schedule,
) -> tuple[EvaluationFixtureResult, ...]:
    strength = {
        candidates[0]: 3,
        candidates[1]: 2,
        candidates[2]: 1,
        RANDOM_LEGAL_IDENTITY: 0,
    }
    results = []
    for fixture in schedule.fixtures:
        winner = max((fixture.queen, fixture.king), key=strength.__getitem__)
        queen_return = 0.1 if winner == fixture.queen else -0.1
        results.append(
            EvaluationFixtureResult(
                fixture_id=fixture.fixture_id,
                queen=fixture.queen,
                king=fixture.king,
                winner=winner,
                queen_round_returns=(queen_return,) * 6,
                king_round_returns=(-queen_return,) * 6,
            )
        )
    return tuple(results)


# Ranking excludes the random control and retains every opponent-specific view.
def test_document_ranks_candidate_round_robin_and_reports_consistency(
    tmp_path: Path,
) -> None:
    candidates = _candidates()
    lanes = ("lane-0", "lane-1")
    schedule = build_candidate_tournament_schedule(
        candidates, lanes, generation_count=2
    )
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    results = _decisive_results(candidates, schedule)

    document = _tournament_document(
        run_id="run",
        checkpoint=checkpoint,
        lane_roots=lanes,
        schedule=schedule,
        results=results,
        candidates=candidates,
        bootstrap_samples=100,
    )
    rows = document["candidates"]
    assert [row["policy_id"] for row in rows] == [
        "policy-0",
        "policy-1",
        "policy-2",
    ]
    assert [row["macro_victory_percentage"] for row in rows] == [1.0, 0.5, 0.0]
    assert all(len(row["opponents"]) == 2 for row in rows)
    assert all(row["versus_random_legal"]["overall"]["games"] == 4 for row in rows)
    assert document["ranking"]["random_control_included"] is False
    assert document["ranking"]["resolved"] is True
    assert document["random_legal_controller"]["learned_parameters"] == 0
    assert [row["tier"] for row in rows] == [1, 2, 3]
    assert [row["first_place_probability"] for row in rows] == [1.0, 0.0, 0.0]
    for rank_index in range(3):
        assert sum(row["rank_probabilities"][rank_index] for row in rows) == 1.0

    report = _markdown_report(document)
    assert "Head-to-head victory matrix" in report
    assert "Opponent detail" in report
    assert "Random-control games do not affect rank" in report
    assert "initial" not in report.lower()


# Bootstrap sampling is derived from immutable tournament evidence.
def test_bootstrap_analysis_is_reproducible(tmp_path: Path) -> None:
    candidates = _candidates()
    lanes = ("lane-0", "lane-1")
    schedule = build_candidate_tournament_schedule(
        candidates, lanes, generation_count=2
    )
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    arguments = {
        "run_id": "run",
        "checkpoint": checkpoint,
        "lane_roots": lanes,
        "schedule": schedule,
        "results": _decisive_results(candidates, schedule),
        "candidates": candidates,
        "bootstrap_samples": 100,
    }
    first = _tournament_document(**arguments)
    second = _tournament_document(**arguments)
    assert first == second

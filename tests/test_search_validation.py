"""Strategic fixture and validation-harness acceptance tests."""

from __future__ import annotations

from dataclasses import asdict

import pytest

from dracula.engine import EnginePlayer, state_fingerprint
from dracula.search import information_state_from_engine
from dracula.search.strategic_fixtures import (
    STRATEGIC_FIXTURES,
    FixtureEvidence,
    RoundStage,
    StrategicBehavior,
    fixture_action_features,
    replay_fixture,
    validate_fixture_semantics,
)
from dracula.search.validation import (
    DecisionTiming,
    GameComparisonRecord,
    RoundComparison,
    ValidationConfig,
    _ControllerSpec,
    _comparison_metrics,
    _exact_fixture_evidence,
    _fixture_catalog,
    _play_game_task,
    _read_game_cache,
    _write_game_cache,
)


# The catalog must cover every requested behavior rather than relabeling one tactic.
def test_fixture_catalog_covers_behaviors_roles_and_round_stages() -> None:
    assert len({fixture.fixture_id for fixture in STRATEGIC_FIXTURES}) == len(
        STRATEGIC_FIXTURES
    )
    assert {
        behavior for fixture in STRATEGIC_FIXTURES for behavior in fixture.behaviors
    } == set(StrategicBehavior)
    assert {fixture.player for fixture in STRATEGIC_FIXTURES} == set(EnginePlayer)
    assert {fixture.stage for fixture in STRATEGIC_FIXTURES} == set(RoundStage)
    for stage in (RoundStage.EARLY, RoundStage.MIDDLE):
        assert {
            fixture.dealer for fixture in STRATEGIC_FIXTURES if fixture.stage is stage
        } == {False, True}
    # The alternating rules make the dealer's only late turn the forced eighth move.
    assert any(
        fixture.stage is RoundStage.LATE
        and fixture.dealer
        and fixture.evidence is FixtureEvidence.FORCED
        for fixture in STRATEGIC_FIXTURES
    )


# A fixture is a replayable engine history, not an arbitrary hand-authored board.
@pytest.mark.parametrize("fixture", STRATEGIC_FIXTURES, ids=lambda value: value.fixture_id)
def test_strategic_fixture_replays_identically_and_matches_its_semantics(fixture) -> None:
    first = replay_fixture(fixture)
    repeated = replay_fixture(fixture)
    assert state_fingerprint(first) == state_fingerprint(repeated)
    assert len(first.completed_rounds) == 5
    assert first.round_number == 6
    assert first.active_player is fixture.player
    validate_fixture_semantics(fixture)
    assert set(fixture.expected_action_indices) <= set(fixture_action_features(fixture))


# Exact fixtures derive their labels from all remaining continuations and engine scores.
def test_exhaustive_fixture_evidence_reproduces_expected_actions_and_values() -> None:
    evidence = _exact_fixture_evidence()
    expected = {
        fixture.fixture_id: fixture
        for fixture in STRATEGIC_FIXTURES
        if fixture.evidence is FixtureEvidence.EXHAUSTIVE
    }
    assert {row["fixture_id"] for row in evidence} == set(expected)
    for row in evidence:
        fixture = expected[str(row["fixture_id"])]
        assert row["selected_action_index"] in fixture.expected_action_indices
        assert float(row["value"]) == pytest.approx(fixture.evidence_value)
        assert float(row["margin"]) == pytest.approx(fixture.evidence_margin)
        assert int(row["evaluated_states"]) > 0


# Human-readable evidence names the visible position and exact card-placement answer.
def test_fixture_catalog_contains_only_replayable_visible_fields() -> None:
    catalog = _fixture_catalog()
    assert len(catalog) == len(STRATEGIC_FIXTURES)
    for row in catalog:
        assert row["title"] and row["rationale"]
        assert len(row["own_hand"]) == 4
        assert len(row["coffin"]) == 9
        assert row["expected_actions"]
        assert set(row).isdisjoint(
            {"opponent_hand", "stock", "game_seed", "simulation_deck"}
        )


# Comparison metrics keep game, round, role, dealer, latency, and throughput separate.
def test_comparison_metrics_report_required_splits_and_paired_interval() -> None:
    decisions = (
        DecisionTiming("search", 100, 1, 0, 100, 1.0),
        DecisionTiming("search", 100, 1, 2, 100, 0.5),
    )
    records = (
        GameComparisonRecord(
            "search-vs-control",
            "deck-a",
            EnginePlayer.QUEEN,
            True,
            False,
            (
                RoundComparison(1, True, 40, 20),
                RoundComparison(2, False, 30, 30),
            ),
            decisions,
        ),
        GameComparisonRecord(
            "search-vs-control",
            "deck-a",
            EnginePlayer.KING,
            True,
            False,
            (
                RoundComparison(1, False, 50, 10),
                RoundComparison(2, True, 30, 20),
            ),
            decisions,
        ),
    )
    metrics = _comparison_metrics("search-vs-control", records, 50)
    assert metrics["overall"]["victory_percentage"] == 1.0
    assert metrics["queen"]["games"] == metrics["king"]["games"] == 1
    assert metrics["rounds"]["count"] == 4
    assert metrics["decision_latency_seconds"]["count"] == 4
    assert metrics["simulations_per_second"] == pytest.approx(400 / 3)
    assert metrics["paired_game_advantage_ci_95"] == [1.0, 1.0]


def test_validation_configuration_rejects_ambiguous_budget_contracts() -> None:
    assert asdict(ValidationConfig())["selected_budget"] == 500
    assert asdict(ValidationConfig())["game_pairs"] == 12
    with pytest.raises(ValueError):
        ValidationConfig(budgets=(500, 100))
    with pytest.raises(ValueError):
        ValidationConfig(budgets=(100, 500), selected_budget=2_000)


# The full-game harness itself must cross all six engine rounds before costly runs start.
def test_game_comparison_harness_completes_a_role_balanced_engine_game() -> None:
    result = _play_game_task(
        (
            "random-smoke",
            "search-validation-game-smoke",
            EnginePlayer.QUEEN,
            _ControllerSpec("random"),
            _ControllerSpec("random"),
        )
    )
    assert len(result.rounds) == 6
    assert result.subject_role is EnginePlayer.QUEEN
    assert result.decisions == ()


# Long control runs resume only when the schedule and policy artifact still match.
def test_game_cache_round_trips_complete_role_paired_evidence(tmp_path) -> None:
    config = ValidationConfig()
    record = GameComparisonRecord(
        "search-500-vs-random-legal",
        "deck-a",
        EnginePlayer.KING,
        True,
        False,
        (RoundComparison(1, False, 50, 20),),
        (DecisionTiming("search", 500, 1, 0, 500, 2.5),),
    )
    _write_game_cache(tmp_path, config, "archive-digest", (record,))
    assert _read_game_cache(tmp_path, config, "archive-digest") == [record]
    assert _read_game_cache(tmp_path, config, "different-digest") == []


# The acceptance state supplied to search is the same privacy projection used in play.
def test_fixture_search_boundary_contains_no_diagnostic_private_state() -> None:
    state = replay_fixture(STRATEGIC_FIXTURES[0])
    information = information_state_from_engine(state)
    assert not hasattr(information, "opponent_hand")
    assert not hasattr(information, "stock")
    assert not hasattr(information, "simulation_deck")

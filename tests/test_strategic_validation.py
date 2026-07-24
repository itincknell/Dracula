"""Teacher v2 defensive-fixture and absolute-validation contracts."""

from __future__ import annotations

from dataclasses import asdict

import pytest

from dracula.engine import EnginePlayer, legal_moves, state_fingerprint
from dracula.search.defensive_fixtures import (
    DEFENSIVE_FIXTURES,
    DefensiveBehavior,
    replay_defensive_fixture,
)
from dracula.search.strategic_fixtures import (
    STRATEGIC_FIXTURES,
    FixtureEvidence,
    RoundStage,
)
from dracula.search.strategic_validation import (
    StrategicProfile,
    StrategicValidationConfig,
    _constructive_exact_evidence,
    _exact_evidence,
    _fixture_task,
)


# The catalog covers each requested defensive situation rather than relabeling
# one late-round block as evidence for every behavior and lifecycle position.
def test_defensive_catalog_covers_behaviors_roles_stages_and_dealer_states() -> None:
    assert len({fixture.fixture_id for fixture in DEFENSIVE_FIXTURES}) == len(
        DEFENSIVE_FIXTURES
    )
    assert {
        behavior
        for fixture in DEFENSIVE_FIXTURES
        for behavior in fixture.behaviors
    } == set(DefensiveBehavior)
    assert {fixture.player for fixture in DEFENSIVE_FIXTURES} == set(EnginePlayer)
    assert {fixture.stage for fixture in DEFENSIVE_FIXTURES} == set(RoundStage)
    for stage in RoundStage:
        assert {fixture.dealer for fixture in DEFENSIVE_FIXTURES if fixture.stage is stage} == {
            False,
            True,
        }
    for player in EnginePlayer:
        assert any(
            fixture.stage is RoundStage.LATE
            and fixture.player is player
            and fixture.dealer
            and DefensiveBehavior.FORCED_TRANSITION in fixture.behaviors
            for fixture in DEFENSIVE_FIXTURES
        )


# Every fixture is a reproducible public engine history with legal expected actions.
@pytest.mark.parametrize(
    "fixture", DEFENSIVE_FIXTURES, ids=lambda fixture: fixture.fixture_id
)
def test_defensive_fixture_replay_is_stable_and_expected_actions_are_legal(fixture) -> None:
    first = replay_defensive_fixture(fixture)
    second = replay_defensive_fixture(fixture)
    assert state_fingerprint(first) == state_fingerprint(second)
    assert first.active_player is fixture.player
    # The action mapping is player-relative, so it must use the bridge helper.
    from dracula.bridge import action_index_for_move

    legal_indexes = {
        action_index_for_move(move, fixture.player)
        for move in legal_moves(first, fixture.player)
    }
    assert set(fixture.expected_action_indices) <= legal_indexes


# Exhaustive alpha-beta proof, not a hand-authored card feature, fixes every label.
def test_exact_adversarial_evidence_reproduces_all_defensive_labels() -> None:
    evidence = _exact_evidence()
    strategic = [
        fixture
        for fixture in DEFENSIVE_FIXTURES
        if DefensiveBehavior.FORCED_TRANSITION not in fixture.behaviors
    ]
    assert len(evidence) == len(strategic)
    assert {row["fixture_id"] for row in evidence} == {
        fixture.fixture_id for fixture in strategic
    }
    assert all(int(row["evaluated_states"]) > 0 for row in evidence)


# The diagnostic also proves the previously high-budget constructive labels in
# the fixture's concrete hidden world; no expected action rests on prose alone.
def test_exact_diagnostic_supports_every_constructive_label() -> None:
    evidence = _constructive_exact_evidence()
    strategic = [
        fixture
        for fixture in STRATEGIC_FIXTURES
        if fixture.evidence is not FixtureEvidence.FORCED
    ]
    assert len(evidence) == len(strategic)
    assert {row["fixture_id"] for row in evidence} == {
        fixture.fixture_id for fixture in strategic
    }
    assert all(int(row["evaluated_states"]) > 0 for row in evidence)


# A late fixture keeps this test quick while exercising the persisted visit evidence.
def test_fixture_runner_records_v2_visits_values_latency_and_exact_answer() -> None:
    record = _fixture_task(
        (
            "defensive",
            "king-late-block",
            "v2-8x2c",
            0,
            StrategicProfile(8, 2),
        )
    )
    assert record.passed
    assert record.selected_action_index == 2
    assert sum(record.visits) == 8
    assert record.total_terminal_evaluations >= record.outer_simulations
    assert record.latency_seconds > 0
    assert record.peak_rss_mib > 0


def test_validation_configuration_locks_absolute_sample_minimums() -> None:
    config = StrategicValidationConfig()
    assert asdict(config)["selected_profile"] == {
        "outer_simulations": 500,
        "response_completions": 1,
    }
    assert {
        profile.response_completions for profile in config.benchmark_profiles
    } == {1, 2, 4}
    assert config.control_game_pairs == 12
    assert config.version_one_game_pairs == 30
    with pytest.raises(ValueError):
        StrategicValidationConfig(control_game_pairs=11)
    with pytest.raises(ValueError):
        StrategicValidationConfig(version_one_game_pairs=29)

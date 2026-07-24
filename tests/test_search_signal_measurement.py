"""Symmetry-aware Teacher v2 signal-measurement invariants."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from types import SimpleNamespace

import pytest

from dracula.engine import EnginePlayer
from dracula.search.signal_measurement import (
    aggregate_result_to_groups,
    build_signal_fixtures,
    deterministic_result_digest,
)
from dracula.search.strategic import (
    StrategicInformationSetSearch,
    StrategicSearchConfig,
    strategic_action_groups,
)


EXPECTED_OCCUPIED = {
    "placement-1-opening": {(5,)},
    "placement-2-adjacent": {
        (2, 5),
        (5, 8),
        (4, 5),
        (5, 6),
    },
    "placement-3-line": {
        (4, 5, 6),
        (2, 5, 8),
    },
    "placement-3-no-symmetry": {(2, 4, 5)},
    "placement-4-no-symmetry": {(1, 2, 4, 5)},
    "placement-5-no-symmetry": {(1, 2, 4, 5, 7)},
    "placement-6-no-symmetry": {(1, 2, 4, 5, 7, 8)},
    "placement-7-no-symmetry": {(1, 2, 3, 4, 5, 7, 8)},
}


def _occupied(fixture) -> tuple[int, ...]:
    return tuple(
        index + 1
        for index, card_id in enumerate(fixture.information.coffin)
        if card_id is not None
    )


# Every requested placement and special case is represented in both legal role assignments.
def test_fixed_fixture_matrix_is_complete_and_role_balanced() -> None:
    fixtures = build_signal_fixtures()

    assert len(fixtures) == 24
    assert {
        fixture.placement_number for fixture in fixtures
    } == set(range(1, 8))
    for category, occupied_patterns in EXPECTED_OCCUPIED.items():
        selected = [
            fixture for fixture in fixtures if fixture.category == category
        ]
        assert {_occupied(fixture) for fixture in selected} == occupied_patterns
        pattern_roles = defaultdict(set)
        pattern_dealers = defaultdict(set)
        for fixture in selected:
            pattern_roles[_occupied(fixture)].add(
                fixture.information.player
            )
            pattern_dealers[_occupied(fixture)].add(
                fixture.information.dealer
            )
        assert all(roles == set(EnginePlayer) for roles in pattern_roles.values())
        assert all(
            dealers == set(EnginePlayer)
            for dealers in pattern_dealers.values()
        )


# Listed openings reduce destinations exactly; later controls retain singleton destinations.
def test_fixture_groups_apply_only_the_authoritative_table() -> None:
    fixtures = build_signal_fixtures()
    expected_destination_group_counts = {
        "placement-1-opening": 2,
        "placement-2-adjacent": 3,
        "placement-3-line": 3,
    }

    for fixture in fixtures:
        groups = strategic_action_groups(fixture.information)
        available_cards = sum(
            card_id is not None for card_id in fixture.information.own_hand
        )
        if fixture.category in expected_destination_group_counts:
            assert len(groups) == (
                available_cards
                * expected_destination_group_counts[fixture.category]
            )
        else:
            assert all(len(group.member_action_indices) == 1 for group in groups)


# Disabling reduction retains the exact pre-symmetry 32x4 contract identities.
def test_frozen_control_keeps_pre_symmetry_configuration_digests() -> None:
    config = StrategicSearchConfig(
        outer_simulation_budget=32,
        response_completions_per_action=4,
        destination_symmetry_enabled=False,
    )

    assert config.response_config.digest == (
        "97af09f17f00748013b9a7004b67302d605b0e65026a5d9a4c797d3852b8c0a9"
    )
    assert config.digest == (
        "a6add599d8324f8a3098ee64635705078c00102f76d7804aa01d9ee8b1251fd5"
    )


# Projection conserves every visit and uses visit-weighted concrete values.
def test_baseline_visits_and_values_aggregate_exactly() -> None:
    fixture = next(
        fixture
        for fixture in build_signal_fixtures()
        if fixture.fixture_id.startswith("p1-center-only")
    )
    information = fixture.information
    groups = strategic_action_groups(information)
    visits = [0] * 32
    values: list[float | None] = [None] * 32
    for action_index, allowed in enumerate(
        value for row in information.legal_mask for value in row
    ):
        if allowed:
            visits[action_index] = 2
            values[action_index] = action_index / 32.0
    selected = groups[0].member_action_indices[-1]
    result = SimpleNamespace(
        action_visits=tuple(visits),
        mean_action_values=tuple(values),
        selected_action_index=selected,
        simulation_count=sum(visits),
    )
    original_visits = result.action_visits
    original_values = result.mean_action_values

    aggregate = aggregate_result_to_groups(information, result)

    assert aggregate.group_visits == (4,) * len(groups)
    for group_index, group in enumerate(groups):
        expected = sum(
            visits[index] * float(values[index])
            for index in group.member_action_indices
        ) / sum(visits[index] for index in group.member_action_indices)
        assert aggregate.group_mean_values[group_index] == pytest.approx(expected)
    assert aggregate.selected_group_index == 0
    assert result.action_visits == original_visits
    assert result.mean_action_values == original_values


# Reading and aggregating a search result cannot perturb a repeated planner call.
def test_measurement_does_not_change_search_behavior() -> None:
    fixture = next(
        fixture
        for fixture in build_signal_fixtures()
        if fixture.category == "placement-7-no-symmetry"
    )
    config = StrategicSearchConfig(
        outer_simulation_budget=32,
        response_completions_per_action=4,
        destination_symmetry_enabled=False,
    )
    planner = StrategicInformationSetSearch(config)
    request_seed = b"\x91" * 32

    first = planner.search(fixture.information, request_seed)
    aggregate_result_to_groups(fixture.information, first)
    second = planner.search(fixture.information, request_seed)

    assert first.selected_action_index == second.selected_action_index
    assert first.action_visits == second.action_visits
    assert first.mean_action_values == second.mean_action_values
    assert first.principal_continuation == second.principal_continuation
    assert first.response_request_count == second.response_request_count
    assert first.response_cache_hit_count == second.response_cache_hit_count
    assert (
        first.response_candidate_action_count
        == second.response_candidate_action_count
    )
    assert (
        first.response_terminal_evaluation_count
        == second.response_terminal_evaluation_count
    )
    assert deterministic_result_digest(first) == deterministic_result_digest(
        second
    )

    # Runtime and process RSS are observations, not seeded search evidence.
    changed_measurements = replace(
        first,
        elapsed_seconds=first.elapsed_seconds + 1.0,
        peak_resident_memory_bytes=(
            first.peak_resident_memory_bytes + 1
        ),
    )
    assert deterministic_result_digest(first) == deterministic_result_digest(
        changed_measurements
    )


# Each fixture's five-seed comparisons must use the same request seed in both modes.
def test_request_seed_does_not_depend_on_measurement_mode() -> None:
    from dracula.search import signal_measurement

    fixture = build_signal_fixtures()[0]
    seeds = [
        signal_measurement._request_seed(fixture, index)
        for index in range(5)
    ]

    assert len(set(seeds)) == 5
    assert seeds == [
        signal_measurement._request_seed(fixture, index)
        for index in range(5)
    ]

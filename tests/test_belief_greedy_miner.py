"""Balanced belief-greedy mining counts, artifacts, and privacy."""

from __future__ import annotations

from collections import Counter

import pytest

from dracula.belief_greedy_miner import (
    BalancedMinerConfig,
    ROWS_PER_GAME,
    ROWS_PER_PLACEMENT_PER_GAME,
    TRAJECTORY_PROFILE_CYCLE,
    mine_balanced_game,
)


@pytest.fixture(scope="module")
def mined_game():
    return mine_balanced_game(
        BalancedMinerConfig(
            root_seed="balanced-miner-unit-test",
            outer_simulation_budget=32,
            belief_completion_count=1,
            worker_count=1,
        ),
        0,
    )


def test_complete_game_is_exactly_balanced_by_learned_placement(
    mined_game,
) -> None:
    assert len(mined_game.rows) == ROWS_PER_GAME == 42
    assert Counter(row.placement_number for row in mined_game.rows) == {
        placement: ROWS_PER_PLACEMENT_PER_GAME
        for placement in range(1, 8)
    }
    assert Counter(row.player.value for row in mined_game.rows) == {
        "queen": 21,
        "king": 21,
    }
    # Alternating dealers expose each role three times at every placement.
    for placement in range(1, 8):
        rows = [
            row
            for row in mined_game.rows
            if row.placement_number == placement
        ]
        assert Counter(row.player.value for row in rows) == {
            "queen": 3,
            "king": 3,
        }


def test_rows_bind_the_teacher_group_and_partition_every_legal_action(
    mined_game,
) -> None:
    for row in mined_game.rows:
        flattened = tuple(
            member for group in row.strategic_groups for member in group
        )
        assert len(flattened) == len(set(flattened))
        assert row.selected_group_representative in (
            row.strategic_group_representatives
        )
        assert len(row.strategic_group_representatives) == len(
            row.strategic_groups
        )
        # The soft policy target retains all root-search evidence instead of
        # turning a close visit result into a binary winning label.
        assert len(row.strategic_group_visits) == len(row.strategic_groups)
        assert sum(row.strategic_group_visits) == 32
        assert all(visits >= 1 for visits in row.strategic_group_visits)
        selected_index = row.strategic_group_representatives.index(
            row.selected_group_representative
        )
        assert row.strategic_group_visits[selected_index] == max(
            row.strategic_group_visits
        )
        assert all(
            representative in group
            for representative, group in zip(
                row.strategic_group_representatives,
                row.strategic_groups,
                strict=True,
            )
        )
        legal_bits = tuple(
            bool(row.legal_mask_packed[index // 8] & (1 << (7 - index % 8)))
            for index in range(32)
        )
        assert set(flattened) == {
            index for index, allowed in enumerate(legal_bits) if allowed
        }


def test_game_artifact_contains_no_authoritative_or_search_payload(
    mined_game,
) -> None:
    encoded = str(mined_game.artifact_data()).lower()
    forbidden = (
        "opponent_hand",
        "stock_order",
        "engine_seed",
        "determinization",
        "search_tree",
        "action_values",
        "round_return",
    )
    assert not any(field in encoded for field in forbidden)


def test_fixture_replay_reproduces_every_training_row(mined_game) -> None:
    repeated = mine_balanced_game(
        BalancedMinerConfig(
            root_seed="balanced-miner-unit-test",
            outer_simulation_budget=32,
            belief_completion_count=1,
            worker_count=1,
        ),
        0,
    )
    assert repeated.content_digest == mined_game.content_digest
    assert repeated.rows == mined_game.rows


def test_profile_cycle_contains_teacher_and_varied_trajectory_sources() -> None:
    assert tuple(profile.value for profile in TRAJECTORY_PROFILE_CYCLE) == (
        "teacher",
        "one-deviation",
        "two-deviations",
        "mixed",
        "alternative",
    )

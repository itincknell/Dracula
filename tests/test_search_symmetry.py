"""Verify symmetry groups, strategic actions, masks, and concrete resolution.

Tests cover every authorized board pattern, unlisted fallbacks, card-specific
grouping, representative proxies, deterministic coins, and role equivalence.
"""

from __future__ import annotations

from collections.abc import Iterable

import pytest
import torch

from dracula.action_contract import (
    PolicyActionContractError,
    build_representative_action_mask,
)
from dracula.active_policy import ActivePolicyRuntime
from dracula.bgc_policy_model import BGCPolicyModel
from dracula.bridge import transpose_grid_index
from dracula.engine import apply_move, create_game, legal_moves
from dracula.search import (
    DestinationSymmetryError,
    destination_symmetry_groups,
)
from dracula.search.information import information_state_from_engine
from dracula.strategic_actions import (
    select_concrete_action_index,
    strategic_action_groups,
)


def _coffin(
    occupied_positions: Iterable[int],
    *,
    card_prefix: str = "card",
) -> tuple[str | None, ...]:
    result: list[str | None] = [None] * 9
    for ordinal, position in enumerate(occupied_positions):
        result[position - 1] = f"{card_prefix}-{ordinal}"
    return tuple(result)


def _one_based_groups(
    coffin: tuple[str | None, ...],
) -> tuple[tuple[int, tuple[int, ...]], ...]:
    return tuple(
        (
            group.representative_grid_index + 1,
            tuple(index + 1 for index in group.member_grid_indices),
        )
        for group in destination_symmetry_groups(coffin)
    )


def _information_for_occupied_positions(occupied_positions: set[int]):
    """Reach one authorized pattern while varying destinations, not hand cards."""

    frontier = [create_game("strategic-action-contract")]
    for _depth in range(3):
        for state in frontier:
            information = information_state_from_engine(state)
            occupied = {
                index + 1
                for index, card_id in enumerate(information.coffin)
                if card_id is not None
            }
            if occupied == occupied_positions:
                return information
        next_frontier = []
        for state in frontier:
            assert state.active_player is not None
            options = legal_moves(state, state.active_player)
            selected_slot = min(move.hand_slot for move in options)
            next_frontier.extend(
                apply_move(state, move).state
                for move in options
                if move.hand_slot == selected_slot
            )
        frontier = next_frontier
    raise AssertionError(f"authorized pattern is unreachable: {occupied_positions}")


AUTHORIZED_CASES = (
    ({5}, ((2, (2, 8)), (4, (4, 6)))),
    ({2, 5}, ((1, (1, 3)), (4, (4, 6)), (8, (8,)))),
    ({5, 8}, ((2, (2,)), (4, (4, 6)), (9, (7, 9)))),
    ({4, 5}, ((1, (1, 7)), (2, (2, 8)), (6, (6,)))),
    ({5, 6}, ((2, (2, 8)), (3, (3, 9)), (4, (4,)))),
    ({4, 5, 6}, ((1, (1, 7)), (2, (2, 8)), (3, (3, 9)))),
    ({2, 5, 8}, ((1, (1, 3)), (4, (4, 6)), (7, (7, 9)))),
)


# Every authorized board has exactly the documented representatives and members.
@pytest.mark.parametrize(("occupied", "expected"), AUTHORIZED_CASES)
def test_authorized_destination_groups_are_exact(occupied, expected) -> None:
    assert _one_based_groups(_coffin(occupied)) == expected


# Rank, suit, color, and Vampire substitutions cannot affect spatial grouping.
@pytest.mark.parametrize(("occupied", "expected"), AUTHORIZED_CASES)
def test_card_substitutions_do_not_change_groups(occupied, expected) -> None:
    first = _coffin(occupied, card_prefix="Vampire")
    second = _coffin(occupied, card_prefix="Queen-of-hearts")
    assert _one_based_groups(first) == expected
    assert _one_based_groups(second) == expected


def _transpose_coffin(
    coffin: tuple[str | None, ...],
) -> tuple[str | None, ...]:
    result: list[str | None] = [None] * 9
    for index, card in enumerate(coffin):
        result[transpose_grid_index(index)] = card
    return tuple(result)


def _transpose_groups(
    coffin: tuple[str | None, ...],
) -> set[frozenset[int]]:
    return {
        frozenset(
            transpose_grid_index(index)
            for index in group.member_grid_indices
        )
        for group in destination_symmetry_groups(coffin)
    }


# Queen/King transpose preserves groups; each normalized table keeps its listed representative.
@pytest.mark.parametrize(("occupied", "_expected"), AUTHORIZED_CASES)
def test_player_relative_transpose_preserves_groups(occupied, _expected) -> None:
    coffin = _coffin(occupied)
    transposed = _transpose_coffin(coffin)
    actual = {
        frozenset(group.member_grid_indices)
        for group in destination_symmetry_groups(transposed)
    }
    assert actual == _transpose_groups(coffin)


@pytest.mark.parametrize(("occupied", "_expected"), AUTHORIZED_CASES)
def test_strategic_groups_partition_legality_and_masks_retain_only_proxies(
    occupied,
    _expected,
) -> None:
    information = _information_for_occupied_positions(occupied)
    destination_groups = destination_symmetry_groups(information.coffin)
    groups = strategic_action_groups(information)
    legal = {
        index
        for index, allowed in enumerate(
            value for row in information.legal_mask for value in row
        )
        if allowed
    }
    members = tuple(
        action_index for group in groups for action_index in group.member_action_indices
    )
    assert len(members) == len(set(members))
    assert set(members) == legal

    available_slots = tuple(
        slot for slot, card_id in enumerate(information.own_hand) if card_id is not None
    )
    expected_destinations = tuple(
        (group.representative_grid_index, group.member_grid_indices)
        for group in destination_groups
    )
    for hand_slot in available_slots:
        card_groups = tuple(group for group in groups if group.hand_slot == hand_slot)
        assert tuple(
            (group.representative_grid_index, group.member_grid_indices)
            for group in card_groups
        ) == expected_destinations
    assert len(groups) == len(available_slots) * len(destination_groups)

    engine_mask = torch.tensor(information.legal_mask, dtype=torch.bool)
    representative_mask = build_representative_action_mask(engine_mask, groups)
    assert set(torch.nonzero(representative_mask.flatten()).flatten().tolist()) == {
        group.representative_action_index for group in groups
    }
    for group in groups:
        assert representative_mask.flatten()[group.representative_action_index]
        assert all(engine_mask.flatten()[member] for member in group.member_action_indices)
        assert all(
            not representative_mask.flatten()[member]
            for member in group.member_action_indices
            if member != group.representative_action_index
        )


def test_paired_destination_choice_is_reproducible_and_group_preserving() -> None:
    information = information_state_from_engine(create_game("paired-choice-contract"))
    groups = strategic_action_groups(information)
    selected_group = next(
        group for group in groups if len(group.member_action_indices) == 2
    )
    selected_members: set[int] = set()

    for choice_index in range(32):
        first_action = select_concrete_action_index(
            selected_group,
            "paired-choice-test",
            choice_index,
        )
        second_action = select_concrete_action_index(
            selected_group,
            "paired-choice-test",
            choice_index,
        )
        assert first_action == second_action
        assert first_action in selected_group.member_action_indices
        assert next(
            group for group in groups if first_action in group.member_action_indices
        ) == selected_group
        selected_members.add(first_action)

    assert selected_members == set(selected_group.member_action_indices)


def test_active_policy_selects_a_legal_group_for_every_authorized_pattern() -> None:
    runtime = ActivePolicyRuntime(
        BGCPolicyModel(),
        artifact_digest="a" * 64,
    )
    for decision_index, (occupied, _expected) in enumerate(AUTHORIZED_CASES):
        information = _information_for_occupied_positions(occupied)
        groups = strategic_action_groups(information)
        decision = runtime.decide(
            information,
            fixture_id="authorized-pattern-runtime",
            decision_index=decision_index,
        )
        selected_group = next(
            group
            for group in groups
            if group.representative_action_index
            == decision.representative_action_index
        )
        assert decision.concrete_action_index in selected_group.member_action_indices


# Any unlisted occupied pattern falls back to independent legal destinations.
@pytest.mark.parametrize(
    "occupied",
    (
        {1, 5},
        {1, 2, 5},
        {1, 4, 5},
        {2, 3, 5},
        {2, 4, 5, 6},
        {1, 2, 4, 5, 7},
    ),
)
def test_unlisted_patterns_return_only_singletons(occupied) -> None:
    groups = destination_symmetry_groups(_coffin(occupied))
    assert groups
    assert all(
        group.member_grid_indices == (group.representative_grid_index,)
        for group in groups
    )


def _expected_legal_destinations(occupied: set[int]) -> set[int]:
    result = set()
    for destination in range(1, 10):
        if destination in occupied:
            continue
        row, column = divmod(destination - 1, 3)
        if any(
            abs(row - divmod(other - 1, 3)[0])
            + abs(column - divmod(other - 1, 3)[1])
            == 1
            for other in occupied
        ):
            result.add(destination)
    return result


# The table is exhaustive: every other one of the 512 occupancy masks is singleton-only.
def test_every_unlisted_occupancy_mask_uses_exact_legal_singletons() -> None:
    authorized = {frozenset(occupied) for occupied, _expected in AUTHORIZED_CASES}
    for occupancy_mask in range(1 << 9):
        occupied = {
            position
            for position in range(1, 10)
            if occupancy_mask & (1 << (position - 1))
        }
        if frozenset(occupied) in authorized:
            continue
        groups = destination_symmetry_groups(_coffin(occupied))
        actual = {
            group.representative_grid_index + 1
            for group in groups
        }
        assert actual == _expected_legal_destinations(occupied)
        assert all(
            group.member_grid_indices == (group.representative_grid_index,)
            for group in groups
        )


def test_invalid_coffin_length_is_rejected() -> None:
    with pytest.raises(
        DestinationSymmetryError,
        match="coffin must contain nine positions",
    ):
        destination_symmetry_groups((None,) * 8)


def test_action_contracts_reject_malformed_values_at_their_boundaries() -> None:
    information = information_state_from_engine(create_game("action-boundary"))
    groups = strategic_action_groups(information)
    legal_mask = torch.tensor(information.legal_mask, dtype=torch.bool)

    with pytest.raises(PolicyActionContractError, match=r"bool\[4,8\]"):
        build_representative_action_mask(legal_mask.float(), groups)
    with pytest.raises(PolicyActionContractError, match="partition"):
        build_representative_action_mask(legal_mask, groups[:-1])

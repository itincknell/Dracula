"""Contract tests for the exhaustive early-turn destination groups."""

from __future__ import annotations

from collections.abc import Iterable

import pytest

from dracula.bridge import transpose_grid_index
from dracula.search import (
    DestinationSymmetryError,
    destination_symmetry_groups,
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

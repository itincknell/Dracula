"""Exact destination groups for the authorized early-turn symmetries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

COFFIN_POSITION_COUNT = 9
# The table is a versioned gameplay/search contract, not a general geometric
# symmetry algorithm.
DESTINATION_SYMMETRY_SCHEMA_VERSION = (
    "dracula-early-destination-symmetry-v1"
)


class DestinationSymmetryError(ValueError):
    """The coffin cannot produce a valid destination grouping."""


@dataclass(frozen=True, slots=True)
class DestinationSymmetryGroup:
    """One designated proxy destination and its equivalent concrete members."""

    representative_grid_index: int
    member_grid_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        if (
            type(self.representative_grid_index) is not int
            or not 0 <= self.representative_grid_index < COFFIN_POSITION_COUNT
        ):
            raise DestinationSymmetryError("representative grid index is invalid")
        if (
            not isinstance(self.member_grid_indices, tuple)
            or not self.member_grid_indices
            or any(
                type(index) is not int
                or not 0 <= index < COFFIN_POSITION_COUNT
                for index in self.member_grid_indices
            )
            or len(set(self.member_grid_indices)) != len(self.member_grid_indices)
        ):
            raise DestinationSymmetryError("member grid indexes are invalid")
        if self.representative_grid_index not in self.member_grid_indices:
            raise DestinationSymmetryError(
                "representative must belong to its destination group"
            )


def _group(
    representative_position: int,
    *member_positions: int,
) -> DestinationSymmetryGroup:
    """Convert the rules table's one-based positions to engine indexes."""

    return DestinationSymmetryGroup(
        representative_position - 1,
        tuple(position - 1 for position in member_positions),
    )


# Keys use the rules document's one-based coffin positions. These seven cases
# are exhaustive; all other legal destinations remain independent.
_AUTHORIZED_GROUPS = {
    frozenset({5}): (
        _group(2, 2, 8),
        _group(4, 4, 6),
    ),
    frozenset({2, 5}): (
        _group(1, 1, 3),
        _group(4, 4, 6),
        _group(8, 8),
    ),
    frozenset({5, 8}): (
        _group(2, 2),
        _group(4, 4, 6),
        _group(9, 7, 9),
    ),
    frozenset({4, 5}): (
        _group(1, 1, 7),
        _group(2, 2, 8),
        _group(6, 6),
    ),
    frozenset({5, 6}): (
        _group(2, 2, 8),
        _group(3, 3, 9),
        _group(4, 4),
    ),
    frozenset({4, 5, 6}): (
        _group(1, 1, 7),
        _group(2, 2, 8),
        _group(3, 3, 9),
    ),
    frozenset({2, 5, 8}): (
        _group(1, 1, 3),
        _group(4, 4, 6),
        _group(7, 7, 9),
    ),
}


def _orthogonally_adjacent(first: int, second: int) -> bool:
    """Return whether two row-major coffin indexes share an edge."""

    first_row, first_column = divmod(first, 3)
    second_row, second_column = divmod(second, 3)
    row_distance = abs(first_row - second_row)
    column_distance = abs(first_column - second_column)
    return row_distance + column_distance == 1


def _legal_destinations(occupied: frozenset[int]) -> tuple[int, ...]:
    """Return empty positions sharing an edge with the current coffin."""

    return tuple(
        destination
        for destination in range(COFFIN_POSITION_COUNT)
        if destination not in occupied
        and any(
            _orthogonally_adjacent(destination, occupied_index)
            for occupied_index in occupied
        )
    )


def destination_symmetry_groups(
    coffin: Sequence[object | None],
) -> tuple[DestinationSymmetryGroup, ...]:
    """Return the exact authorized groups, or singleton legal destinations."""

    if len(coffin) != COFFIN_POSITION_COUNT:
        raise DestinationSymmetryError("coffin must contain nine positions")
    occupied = frozenset(
        index for index, card in enumerate(coffin) if card is not None
    )
    legal_destinations = _legal_destinations(occupied)
    authorized = _AUTHORIZED_GROUPS.get(
        frozenset(index + 1 for index in occupied)
    )
    if authorized is None:
        return tuple(
            DestinationSymmetryGroup(destination, (destination,))
            for destination in legal_destinations
        )
    grouped_destinations = tuple(
        index
        for group in authorized
        for index in group.member_grid_indices
    )
    if (
        len(set(grouped_destinations)) != len(grouped_destinations)
        or set(grouped_destinations) != set(legal_destinations)
    ):
        raise DestinationSymmetryError(
            "authorized groups do not partition the legal destinations"
        )
    return authorized


__all__ = (
    "DESTINATION_SYMMETRY_SCHEMA_VERSION",
    "DestinationSymmetryError",
    "DestinationSymmetryGroup",
    "destination_symmetry_groups",
)

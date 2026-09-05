"""Apply the exhaustive table of authorized destination symmetries.

Grouping depends only on the exact occupied-position pattern. Unlisted boards
receive no inferred rotation or reflection and keep each destination separate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from dracula.bridge import COFFIN_POSITION_COUNT
from dracula.engine_types import empty_adjacent_grid_indices

class DestinationSymmetryError(ValueError):
    """The coffin cannot produce a valid destination grouping."""


@dataclass(frozen=True, slots=True)
class DestinationSymmetryGroup:
    """One designated proxy destination and its equivalent concrete members."""

    representative_grid_index: int
    member_grid_indices: tuple[int, ...]


def _group(
    representative_position: int,
    member_positions: tuple[int, ...],
) -> DestinationSymmetryGroup:
    """Convert the rules table's one-based positions to engine indexes."""

    return DestinationSymmetryGroup(
        representative_position - 1,
        tuple(position - 1 for position in member_positions),
    )


# Keys and values use the rules document's one-based coffin positions. Each
# value is written as ``_group(proxy, (equivalent concrete positions))``.
# These seven cases are exhaustive; every other legal destination stands alone.
_AUTHORIZED_GROUPS = {
    # Center only.
    frozenset({5}): (
        _group(2, (2, 8)),
        _group(4, (4, 6)),
    ),
    # Above center.
    frozenset({2, 5}): (
        _group(1, (1, 3)),
        _group(4, (4, 6)),
        _group(8, (8,)),
    ),
    # Below center.
    frozenset({5, 8}): (
        _group(2, (2,)),
        _group(4, (4, 6)),
        _group(9, (7, 9)),
    ),
    # Left of center.
    frozenset({4, 5}): (
        _group(1, (1, 7)),
        _group(2, (2, 8)),
        _group(6, (6,)),
    ),
    # Right of center.
    frozenset({5, 6}): (
        _group(2, (2, 8)),
        _group(3, (3, 9)),
        _group(4, (4,)),
    ),
    # Horizontal center line.
    frozenset({4, 5, 6}): (
        _group(1, (1, 7)),
        _group(2, (2, 8)),
        _group(3, (3, 9)),
    ),
    # Vertical center line.
    frozenset({2, 5, 8}): (
        _group(1, (1, 3)),
        _group(4, (4, 6)),
        _group(7, (7, 9)),
    ),
}


def destination_symmetry_groups(
    coffin: Sequence[object | None],
) -> tuple[DestinationSymmetryGroup, ...]:
    """Return the exact authorized groups, or singleton legal destinations."""

    if len(coffin) != COFFIN_POSITION_COUNT:
        raise DestinationSymmetryError("coffin must contain nine positions")
    occupied = frozenset(
        index for index, card in enumerate(coffin) if card is not None
    )
    authorized = _AUTHORIZED_GROUPS.get(
        frozenset(index + 1 for index in occupied)
    )
    if authorized is not None:
        return authorized
    return tuple(
        DestinationSymmetryGroup(destination, (destination,))
        for destination in empty_adjacent_grid_indices(coffin)
    )


__all__ = (
    "DestinationSymmetryError",
    "DestinationSymmetryGroup",
    "destination_symmetry_groups",
)

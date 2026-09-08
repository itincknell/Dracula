"""Combine hand-card choices with strategically distinct destinations.

Different cards always remain different actions. Destination symmetry can pool
concrete placements, after which a separate deterministic coin chooses a member.
"""

from __future__ import annotations

from dataclasses import dataclass

from dracula.decision.bridge import POLICY_GRID_INDICES, POLICY_POSITION_COUNT
from dracula.randomness import deterministic_random
from dracula.decision.information import SearchInformationState
from dracula.decision.symmetry import (
    DestinationSymmetryGroup,
    destination_symmetry_groups,
)


@dataclass(frozen=True, slots=True)
class StrategicActionGroup:
    """Pair one hand card with one strategic destination group.

    Action indexes are derived so the grid and flattened representations cannot
    drift apart.
    """

    hand_slot: int
    destination_group: DestinationSymmetryGroup

    @property
    def representative_grid_index(self) -> int:
        return self.destination_group.representative_grid_index

    @property
    def member_grid_indices(self) -> tuple[int, ...]:
        return self.destination_group.member_grid_indices

    @property
    def representative_action_index(self) -> int:
        return _action_index(self.hand_slot, self.representative_grid_index)

    @property
    def member_action_indices(self) -> tuple[int, ...]:
        return tuple(
            _action_index(self.hand_slot, grid_index)
            for grid_index in self.member_grid_indices
        )


def _action_index(hand_slot: int, grid_index: int) -> int:
    """Combine one hand slot and non-center destination into the policy index."""

    return hand_slot * POLICY_POSITION_COUNT + POLICY_GRID_INDICES.index(grid_index)


def strategic_action_groups(
    information: SearchInformationState,
) -> tuple[StrategicActionGroup, ...]:
    """Pair every available hand card with every strategic destination."""

    destination_groups = destination_symmetry_groups(information.coffin)
    # SearchInformationState already guarantees that its hand, coffin, and
    # legal mask agree; applying each destination group per card partitions it.
    return tuple(
        StrategicActionGroup(hand_slot, destination_group)
        for hand_slot, card_id in enumerate(information.own_hand)
        if card_id is not None
        for destination_group in destination_groups
    )


def strategic_group_for_representative(
    groups: tuple[StrategicActionGroup, ...],
    representative_action_index: int,
) -> StrategicActionGroup:
    """Return the group identified by one representative action index."""

    for group in groups:
        if group.representative_action_index == representative_action_index:
            return group
    raise ValueError("representative action does not identify a strategic group")


def select_concrete_action_index(
    group: StrategicActionGroup,
    *decision_identity: str | int,
) -> int:
    """Resolve a group reproducibly after its strategic choice is complete."""

    members = group.member_action_indices
    if len(members) == 1:
        return members[0]
    # The coin is applied only after policy selection, so it cannot change the
    # chosen card or strategic destination group.
    return members[deterministic_random(*decision_identity).randrange(2)]

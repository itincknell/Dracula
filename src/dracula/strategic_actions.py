"""Information-safe strategic action grouping and destination resolution."""

from __future__ import annotations

from dataclasses import dataclass

from dracula.action_contract import ACTION_COUNT, HAND_SLOT_COUNT, POLICY_GRID_INDICES
from dracula.randomness import Sha256CounterStream, derive_seed, seed_hex
from dracula.search.information import (
    SearchInformationState,
    information_state_fingerprint,
)
from dracula.search.planner import SearchContractViolation
from dracula.search.symmetry import destination_symmetry_groups

# These literals determine persisted group identity and concrete paired choices.
STRATEGIC_DESTINATION_CHOICE_NAMESPACE = "dracula-strategic-destination-choice-v1"
STRATEGIC_DESTINATION_CHOICE_PROFILE = "derived-fair-coin-after-group-selection-v1"

_POLICY_POSITION_BY_GRID_INDEX = {
    grid_index: position for position, grid_index in enumerate(POLICY_GRID_INDICES)
}


@dataclass(frozen=True, slots=True)
class StrategicActionGroup:
    """One hand card paired with one strategically distinct destination group.

    The representative is the search/model proxy. Member actions retain the
    same hand slot and name the one or two legal concrete destinations.
    """

    hand_slot: int
    representative_action_index: int
    representative_grid_index: int
    member_action_indices: tuple[int, ...]
    member_grid_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        if type(self.hand_slot) is not int or not 0 <= self.hand_slot < HAND_SLOT_COUNT:
            raise SearchContractViolation("strategic group hand slot is invalid")
        if (
            type(self.representative_action_index) is not int
            or not 0 <= self.representative_action_index < ACTION_COUNT
            or self.representative_action_index // len(POLICY_GRID_INDICES)
            != self.hand_slot
        ):
            raise SearchContractViolation(
                "strategic group representative action is invalid"
            )
        if (
            type(self.representative_grid_index) is not int
            or self.representative_grid_index
            != POLICY_GRID_INDICES[
                self.representative_action_index % len(POLICY_GRID_INDICES)
            ]
        ):
            raise SearchContractViolation(
                "strategic group representative destination is invalid"
            )
        if (
            not isinstance(self.member_action_indices, tuple)
            or not self.member_action_indices
            or len(set(self.member_action_indices)) != len(self.member_action_indices)
            or any(
                type(action_index) is not int
                or not 0 <= action_index < ACTION_COUNT
                or action_index // len(POLICY_GRID_INDICES) != self.hand_slot
                for action_index in self.member_action_indices
            )
        ):
            raise SearchContractViolation("strategic group member actions are invalid")
        if (
            not isinstance(self.member_grid_indices, tuple)
            or len(self.member_grid_indices) != len(self.member_action_indices)
            or tuple(
                POLICY_GRID_INDICES[action_index % len(POLICY_GRID_INDICES)]
                for action_index in self.member_action_indices
            )
            != self.member_grid_indices
            or self.representative_action_index not in self.member_action_indices
        ):
            raise SearchContractViolation(
                "strategic group member destinations are invalid"
            )


def legal_action_indices(information: SearchInformationState) -> tuple[int, ...]:
    """Return legal concrete action indexes in canonical flattened order."""

    return tuple(
        index
        for index, allowed in enumerate(
            value for row in information.legal_mask for value in row
        )
        if allowed
    )


def strategic_action_groups(
    information: SearchInformationState,
    destination_symmetry_enabled: bool = True,
) -> tuple[StrategicActionGroup, ...]:
    """Partition legal actions by hand slot and the authoritative symmetry table."""

    if not isinstance(information, SearchInformationState):
        raise SearchContractViolation(
            "strategic grouping requires a player information state"
        )
    if type(destination_symmetry_enabled) is not bool:
        raise SearchContractViolation("destination symmetry flag must be Boolean")
    legal = set(legal_action_indices(information))
    if not destination_symmetry_enabled:
        return tuple(
            StrategicActionGroup(
                hand_slot=action_index // len(POLICY_GRID_INDICES),
                representative_action_index=action_index,
                representative_grid_index=POLICY_GRID_INDICES[
                    action_index % len(POLICY_GRID_INDICES)
                ],
                member_action_indices=(action_index,),
                member_grid_indices=(
                    POLICY_GRID_INDICES[action_index % len(POLICY_GRID_INDICES)],
                ),
            )
            for action_index in sorted(legal)
        )

    groups: list[StrategicActionGroup] = []
    # Destination groups are duplicated independently for each available card;
    # symmetry never merges distinct hand-card choices.
    for hand_slot in range(HAND_SLOT_COUNT):
        if information.own_hand[hand_slot] is None:
            continue
        for destination_group in destination_symmetry_groups(information.coffin):
            try:
                representative_position = _POLICY_POSITION_BY_GRID_INDEX[
                    destination_group.representative_grid_index
                ]
                member_positions = tuple(
                    _POLICY_POSITION_BY_GRID_INDEX[grid_index]
                    for grid_index in destination_group.member_grid_indices
                )
            except KeyError as error:
                raise SearchContractViolation(
                    "the center cannot be a strategic destination"
                ) from error
            representative_action = (
                hand_slot * len(POLICY_GRID_INDICES) + representative_position
            )
            groups.append(
                StrategicActionGroup(
                    hand_slot=hand_slot,
                    representative_action_index=representative_action,
                    representative_grid_index=destination_group.representative_grid_index,
                    member_action_indices=tuple(
                        hand_slot * len(POLICY_GRID_INDICES) + position
                        for position in member_positions
                    ),
                    member_grid_indices=destination_group.member_grid_indices,
                )
            )
    grouped = [
        action_index for group in groups for action_index in group.member_action_indices
    ]
    # A strategic reduction is valid only when every concrete legal action is
    # represented exactly once.
    if len(set(grouped)) != len(grouped) or set(grouped) != legal:
        raise SearchContractViolation(
            "strategic action groups do not partition legal actions"
        )
    return tuple(groups)


def derive_strategic_destination_choice_seed(
    request_seed: bytes,
    scope: str,
    information: SearchInformationState,
    representative_action_index: int,
    choice_index: int,
) -> bytes:
    """Derive a paired-member choice independently of strategic selection."""

    if not isinstance(scope, str) or not scope:
        raise ValueError("destination choice scope must be nonempty")
    if (
        type(representative_action_index) is not int
        or not 0 <= representative_action_index < ACTION_COUNT
    ):
        raise ValueError("representative action index is invalid")
    if type(choice_index) is not int or choice_index < 0:
        raise ValueError("choice index must be a non-negative integer")
    return derive_seed(
        STRATEGIC_DESTINATION_CHOICE_NAMESPACE,
        seed_hex(request_seed),
        scope,
        information_state_fingerprint(information),
        str(representative_action_index),
        str(choice_index),
    )


def select_concrete_action_index(
    information: SearchInformationState,
    representative_action_index: int,
    choice_seed: bytes,
    destination_symmetry_enabled: bool = True,
) -> int:
    """Resolve a selected strategic group without changing that selection."""

    seed_hex(choice_seed)
    try:
        group = next(
            candidate
            for candidate in strategic_action_groups(
                information, destination_symmetry_enabled
            )
            if candidate.representative_action_index == representative_action_index
        )
    except StopIteration as error:
        raise SearchContractViolation(
            "selected representative is not a legal strategic action"
        ) from error
    if len(group.member_action_indices) == 1:
        selected = group.member_action_indices[0]
    elif len(group.member_action_indices) == 2:
        # This derived coin resolves presentation-equivalent destinations only
        # after search or inference has selected the strategic group.
        selected = group.member_action_indices[
            Sha256CounterStream(choice_seed).randbelow(2)
        ]
    else:
        raise SearchContractViolation(
            "authorized destination groups must contain one or two members"
        )
    if selected not in legal_action_indices(information):
        raise SearchContractViolation(
            "resolved strategic destination is not currently legal"
        )
    return selected

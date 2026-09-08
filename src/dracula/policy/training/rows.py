"""Validate and decode one sealed visit-distribution training row.

The corpus manifest establishes which game files belong to a split. This module
owns the narrower row boundary inside those files: packed visible observations,
legal strategic groups, visit counts, and the selected audit action become one
compact trusted value. Private engine and search fields have already been
rejected when the containing game document is opened.
"""

from __future__ import annotations

from dataclasses import dataclass

from dracula.decision.bridge import ACTION_COUNT
from dracula.policy.training.data import decode_packed, unpack_bytes
from dracula.policy.training.contracts import (
    PolicyTrainingError,
    OUTER_SIMULATION_BUDGET,
)

OBSERVATION_BYTE_COUNT = 83
LEGAL_MASK_BYTE_COUNT = 4


@dataclass(frozen=True, slots=True)
class DecodedPolicyRow:
    """One trusted training example in compact tensor-ready form.

    The 659 observation bits occupy 83 bytes and the 32 action bits occupy four
    bytes. ``visits`` is a dense 32-entry tuple: only each strategic group's
    representative action carries that group's UCT visit count. Player and
    dealer use zero for Queen and one for King.
    """

    observation: bytes
    legal_mask: bytes
    representative_mask: bytes
    visits: tuple[int, ...]
    selected_action: int
    placement: int
    player: int
    dealer: int


def _pack_mask(actions: set[int]) -> bytes:
    """Pack one 32-action membership set in canonical action order."""

    packed = bytearray(LEGAL_MASK_BYTE_COUNT)
    for action in actions:
        packed[action // 8] |= 1 << (7 - action % 8)
    return bytes(packed)


def _group_fields(
    raw: dict[str, object],
) -> tuple[list[object], list[object], list[object]]:
    """Read the three parallel strategic-group arrays from an external row."""

    groups = raw.get("strategic_groups")
    representatives = raw.get("strategic_group_representatives")
    visits = raw.get("strategic_group_visits")
    if (
        not isinstance(groups, list)
        or not isinstance(representatives, list)
        or not isinstance(visits, list)
        or not groups
        or len(groups) != len(representatives)
        or len(groups) != len(visits)
    ):
        raise PolicyTrainingError("strategic groups are malformed")
    return groups, representatives, visits


def _validate_group(
    group: object,
    representative: object,
    visit_count: object,
    *,
    prior_members: set[int],
    prior_representatives: set[int],
) -> tuple[list[int], int, int]:
    """Validate one group before adding it to the legal-action partition."""

    if (
        not isinstance(group, list)
        or not group
        or any(type(member) is not int for member in group)
        or len(group) != len(set(group))
        or not all(0 <= member < ACTION_COUNT for member in group)
        or type(representative) is not int
        or representative not in group
        or representative in prior_representatives
        or type(visit_count) is not int
        or not 0 <= visit_count <= OUTER_SIMULATION_BUDGET
        or bool(prior_members.intersection(group))
    ):
        raise PolicyTrainingError("strategic group entry is malformed")
    return group, representative, visit_count


def _project_group_visits(
    raw: dict[str, object], legal_mask: bytes
) -> tuple[tuple[int, ...], bytes, int]:
    """Project each strategic group's visits onto its representative action."""

    groups, representatives, group_visits = _group_fields(raw)
    # The model scores a fixed 4×8 action surface. Group visits are therefore
    # expanded into 32 entries while paired non-representative destinations
    # deliberately retain zero visits.
    dense = [0] * ACTION_COUNT
    representative_actions: set[int] = set()
    members: set[int] = set()
    for values in zip(groups, representatives, group_visits, strict=True):
        group, representative, visit_count = _validate_group(
            *values,
            prior_members=members,
            prior_representatives=representative_actions,
        )
        representative_actions.add(representative)
        members.update(group)
        dense[representative] = visit_count

    # Strategic groups must partition the complete engine-legal mask. This is
    # the row boundary that prevents silently dropping or duplicating an action.
    legal_actions = {
        index
        for index, allowed in enumerate(unpack_bytes(legal_mask, ACTION_COUNT))
        if allowed
    }
    if members != legal_actions or sum(dense) != OUTER_SIMULATION_BUDGET:
        raise PolicyTrainingError("legality or visits differ")

    selected_action = raw.get("selected_group_representative")
    if type(selected_action) is not int or selected_action not in representative_actions:
        raise PolicyTrainingError("selected group is invalid")
    return tuple(dense), _pack_mask(representative_actions), selected_action


def decode_policy_row(
    raw: object,
    *,
    fixture_id: object,
    trajectory_profile: object,
) -> DecodedPolicyRow:
    """Validate one external row before it enters the tensor dataset."""

    if (
        not isinstance(raw, dict)
        or raw.get("fixture_id") != fixture_id
        or raw.get("trajectory_profile") != trajectory_profile
        or raw.get("player") not in {"queen", "king"}
        or raw.get("dealer") not in {"queen", "king"}
        or type(raw.get("round_number")) is not int
        or not 1 <= raw["round_number"] <= 6
        or type(raw.get("placement_number")) is not int
        or not 1 <= raw["placement_number"] <= 7
    ):
        raise PolicyTrainingError("dataset row is incompatible")

    observation = decode_packed(
        raw.get("observation_packed"),
        byte_count=OBSERVATION_BYTE_COUNT,
        bit_count=659,
        label="observation",
    )
    legal_mask = decode_packed(
        raw.get("legal_mask_packed"),
        byte_count=LEGAL_MASK_BYTE_COUNT,
        bit_count=ACTION_COUNT,
        label="legal mask",
    )
    visits, representative_mask, selected_action = _project_group_visits(
        raw, legal_mask
    )
    # Role strings are converted once so metric slicing can remain vectorized.
    return DecodedPolicyRow(
        observation=observation,
        legal_mask=legal_mask,
        representative_mask=representative_mask,
        visits=visits,
        selected_action=selected_action,
        placement=raw["placement_number"],
        player=0 if raw["player"] == "queen" else 1,
        dealer=0 if raw["dealer"] == "queen" else 1,
    )

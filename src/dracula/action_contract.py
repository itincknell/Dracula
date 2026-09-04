"""Strategic action groups, representative masks, and concrete resolution."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import torch
from torch import Tensor

from dracula.randomness import Sha256CounterStream, seed_hex
from dracula.search.symmetry import DESTINATION_SYMMETRY_SCHEMA_VERSION

# Training and inference artifacts bind these proxy and concrete-choice rules.
REPRESENTATIVE_MASK_SCHEMA_VERSION = "dracula-sam-representative-mask-v1"
INFERENCE_DESTINATION_SCOPE = "sam-policy-argmax-result-v1"

HAND_SLOT_COUNT = 4
COFFIN_POSITION_COUNT = 9
POLICY_POSITION_COUNT = 8
ACTION_COUNT = HAND_SLOT_COUNT * POLICY_POSITION_COUNT
POLICY_GRID_INDICES = (0, 1, 2, 3, 5, 6, 7, 8)

_AUTHORIZED_PAIRED_DESTINATIONS = frozenset(
    {
        (0, (0, 2)),
        (0, (0, 6)),
        (1, (1, 7)),
        (2, (2, 8)),
        (3, (3, 5)),
        (6, (6, 8)),
        (8, (6, 8)),
    }
)


class SamPolicyContractError(ValueError):
    """A policy input, mask projection, or artifact violates its contract."""


class StrategicGroupLike(Protocol):
    """Structural input accepted from search without importing its group type."""

    hand_slot: int
    representative_action_index: int
    representative_grid_index: int
    member_action_indices: tuple[int, ...]
    member_grid_indices: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class RepresentativeActionGroup:
    """One legal proxy action and the concrete actions it represents."""

    hand_slot: int
    representative_action_index: int
    member_action_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        if (
            type(self.hand_slot) is not int
            or not 0 <= self.hand_slot < HAND_SLOT_COUNT
            or type(self.representative_action_index) is not int
            or not 0 <= self.representative_action_index < ACTION_COUNT
            or self.representative_action_index // POLICY_POSITION_COUNT != self.hand_slot
        ):
            raise SamPolicyContractError("representative action identity is invalid")
        if (
            not isinstance(self.member_action_indices, tuple)
            or len(self.member_action_indices) not in (1, 2)
            or tuple(sorted(self.member_action_indices)) != self.member_action_indices
            or len(set(self.member_action_indices)) != len(self.member_action_indices)
            or self.representative_action_index not in self.member_action_indices
            or any(
                type(action_index) is not int
                or not 0 <= action_index < ACTION_COUNT
                or action_index // POLICY_POSITION_COUNT != self.hand_slot
                for action_index in self.member_action_indices
            )
        ):
            raise SamPolicyContractError("representative group members are invalid")


@dataclass(frozen=True, slots=True, eq=False)
class RepresentativeActionProjection:
    """Representative-only mask plus its validated proxy-to-member mapping."""

    mask: Tensor
    groups: tuple[RepresentativeActionGroup, ...]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.mask, Tensor)
            or self.mask.dtype is not torch.bool
            or self.mask.shape != (HAND_SLOT_COUNT, POLICY_POSITION_COUNT)
            or int(self.mask.sum().item()) != len(self.groups)
        ):
            raise SamPolicyContractError("representative projection mask is invalid")
        representatives = tuple(group.representative_action_index for group in self.groups)
        if (
            tuple(sorted(representatives)) != representatives
            or len(set(representatives)) != len(representatives)
            or any(
                not bool(self.mask.flatten()[representative].item())
                for representative in representatives
            )
        ):
            raise SamPolicyContractError("representative projection groups are invalid")

    def group_for(self, representative_action_index: int) -> RepresentativeActionGroup:
        """Return the unique group named by a legal representative action."""

        if type(representative_action_index) is not int:
            raise SamPolicyContractError("representative action index must be an integer")
        try:
            return next(
                group
                for group in self.groups
                if group.representative_action_index == representative_action_index
            )
        except StopIteration as error:
            raise SamPolicyContractError("action is not a legal representative") from error


def _validate_engine_legal_mask(legal_mask: Tensor) -> tuple[int, ...]:
    if (
        not isinstance(legal_mask, Tensor)
        or legal_mask.dtype is not torch.bool
        or legal_mask.shape != (HAND_SLOT_COUNT, POLICY_POSITION_COUNT)
    ):
        raise SamPolicyContractError("engine legal mask must be bool[4,8]")
    legal = tuple(
        index for index, allowed in enumerate(legal_mask.flatten().tolist()) if allowed
    )
    if not legal:
        raise SamPolicyContractError("engine legal mask must contain an action")
    return legal


def _copy_and_validate_group(group: StrategicGroupLike) -> RepresentativeActionGroup:
    """Detach a search group and verify its proxy against the fixed action map."""

    try:
        hand_slot = group.hand_slot
        representative = group.representative_action_index
        representative_grid = group.representative_grid_index
        members = group.member_action_indices
        member_grids = group.member_grid_indices
    except AttributeError as error:
        raise SamPolicyContractError("strategic group fields are incomplete") from error
    copied = RepresentativeActionGroup(hand_slot, representative, members)
    if (
        type(representative_grid) is not int
        or representative_grid
        != POLICY_GRID_INDICES[representative % POLICY_POSITION_COUNT]
        or not isinstance(member_grids, tuple)
        or member_grids
        != tuple(
            POLICY_GRID_INDICES[member % POLICY_POSITION_COUNT] for member in members
        )
    ):
        raise SamPolicyContractError("strategic group grid and action indexes disagree")
    if len(members) == 2 and (
        representative_grid,
        member_grids,
    ) not in _AUTHORIZED_PAIRED_DESTINATIONS:
        raise SamPolicyContractError(
            "paired destinations are outside the authoritative table"
        )
    return copied


def build_representative_action_projection(
    engine_legal_mask: Tensor,
    strategic_groups: Sequence[StrategicGroupLike],
) -> RepresentativeActionProjection:
    """Retain exactly one proxy action for every authoritative group."""

    legal = _validate_engine_legal_mask(engine_legal_mask)
    if not isinstance(strategic_groups, Sequence) or not strategic_groups:
        raise SamPolicyContractError("strategic groups must be a nonempty sequence")
    groups = tuple(
        sorted(
            (_copy_and_validate_group(group) for group in strategic_groups),
            key=lambda group: group.representative_action_index,
        )
    )
    members = tuple(member for group in groups for member in group.member_action_indices)
    if len(set(members)) != len(members) or set(members) != set(legal):
        raise SamPolicyContractError(
            "strategic groups must partition engine-legal actions"
        )

    legal_slots = {action_index // POLICY_POSITION_COUNT for action_index in legal}
    if {group.hand_slot for group in groups} != legal_slots:
        raise SamPolicyContractError(
            "strategic groups do not cover every legal hand card"
        )
    destination_shape: tuple[tuple[int, tuple[int, ...]], ...] | None = None
    # Every legal hand card must see the same destination reduction. This
    # prevents card identity from changing the authoritative symmetry table.
    for hand_slot in sorted(legal_slots):
        shape = tuple(
            (
                group.representative_action_index % POLICY_POSITION_COUNT,
                tuple(
                    member % POLICY_POSITION_COUNT
                    for member in group.member_action_indices
                ),
            )
            for group in groups
            if group.hand_slot == hand_slot
        )
        if destination_shape is None:
            destination_shape = shape
        elif shape != destination_shape:
            raise SamPolicyContractError(
                "destination grouping must be identical for every legal card"
            )

    mask = torch.zeros_like(engine_legal_mask)
    flat_mask = mask.flatten()
    for group in groups:
        flat_mask[group.representative_action_index] = True
    return RepresentativeActionProjection(mask, groups)


def resolve_representative_action(
    projection: RepresentativeActionProjection,
    representative_action_index: int,
    choice_seed: bytes,
) -> int:
    """Resolve a paired proxy with the established SHA-256 fair coin."""

    if not isinstance(projection, RepresentativeActionProjection):
        raise SamPolicyContractError(
            "concrete resolution requires a representative projection"
        )
    seed_hex(choice_seed)
    group = projection.group_for(representative_action_index)
    if len(group.member_action_indices) == 1:
        return group.member_action_indices[0]
    return group.member_action_indices[
        Sha256CounterStream(choice_seed).randbelow(2)
    ]


def _validate_logits_and_mask(logits: Tensor, representative_mask: Tensor) -> None:
    """Validate finite single or batched logits against same-shaped masks."""

    if (
        not isinstance(logits, Tensor)
        or logits.dtype is not torch.float32
        or not torch.isfinite(logits).all()
    ):
        raise SamPolicyContractError("policy logits must be finite float32")
    if (
        not isinstance(representative_mask, Tensor)
        or representative_mask.dtype is not torch.bool
        or representative_mask.shape != logits.shape
    ):
        raise SamPolicyContractError(
            "representative mask must be Boolean with the logits shape"
        )
    if logits.device != representative_mask.device:
        raise SamPolicyContractError(
            "logits and representative mask must share a device"
        )
    if logits.ndim == 2:
        if logits.shape != (HAND_SLOT_COUNT, POLICY_POSITION_COUNT) or not representative_mask.any():
            raise SamPolicyContractError("single logits and mask must have shape [4,8]")
    elif logits.ndim == 3:
        if (
            logits.shape[1:] != (HAND_SLOT_COUNT, POLICY_POSITION_COUNT)
            or logits.shape[0] < 1
            or not representative_mask.flatten(start_dim=1).any(dim=1).all()
        ):
            raise SamPolicyContractError("batched logits and mask must have shape [B,4,8]")
    else:
        raise SamPolicyContractError("logits must have shape [4,8] or [B,4,8]")


def apply_representative_mask(logits: Tensor, representative_mask: Tensor) -> Tensor:
    """Apply proxy legality outside the standalone model."""

    _validate_logits_and_mask(logits, representative_mask)
    return logits.masked_fill(~representative_mask, -torch.inf)


def representative_policy_probabilities(
    logits: Tensor, representative_mask: Tensor
) -> Tensor:
    """Normalize probability only across legal representative actions."""

    masked = apply_representative_mask(logits, representative_mask)
    if logits.ndim == 2:
        return torch.softmax(masked.reshape(ACTION_COUNT), dim=0).reshape(logits.shape)
    return torch.softmax(masked.flatten(start_dim=1), dim=1).reshape(logits.shape)


def select_representative_action(
    logits: Tensor, representative_mask: Tensor
) -> int | Tensor:
    """Select masked argmax; flattening supplies the canonical tie-break."""

    masked = apply_representative_mask(logits, representative_mask)
    if logits.ndim == 2:
        return int(torch.argmax(masked.reshape(ACTION_COUNT)).item())
    return torch.argmax(masked.flatten(start_dim=1), dim=1)

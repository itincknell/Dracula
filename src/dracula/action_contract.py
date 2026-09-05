"""Build representative-only policy masks and select masked actions.

The module verifies that strategic groups partition engine legality, exposes one
proxy per group, and gives illegal or non-proxy logits no chance of selection.
"""

from __future__ import annotations

import torch
from torch import Tensor

from dracula.bridge import (
    ACTION_COUNT,
    HAND_SLOT_COUNT,
    POLICY_POSITION_COUNT,
)
from dracula.strategic_actions import StrategicActionGroup

# These literal values are part of π1's artifact and deterministic move identity.
# Renaming their Python constants is safe; changing their values is not.
REPRESENTATIVE_MASK_SCHEMA_VERSION = "dracula-sam-representative-mask-v1"
POLICY_DESTINATION_SCOPE = "sam-policy-argmax-result-v1"


class PolicyActionContractError(ValueError):
    """A policy mask or output violates its action contract."""


def _validate_engine_legal_mask(legal_mask: Tensor) -> tuple[int, ...]:
    if (
        not isinstance(legal_mask, Tensor)
        or legal_mask.dtype is not torch.bool
        or legal_mask.shape != (HAND_SLOT_COUNT, POLICY_POSITION_COUNT)
    ):
        raise PolicyActionContractError("engine legal mask must be bool[4,8]")
    legal = tuple(
        index for index, allowed in enumerate(legal_mask.flatten().tolist()) if allowed
    )
    if not legal:
        raise PolicyActionContractError("engine legal mask must contain an action")
    return legal


def build_representative_action_mask(
    engine_legal_mask: Tensor,
    strategic_groups: tuple[StrategicActionGroup, ...],
) -> Tensor:
    """Retain one proxy for each group after checking engine-mask agreement."""

    legal = _validate_engine_legal_mask(engine_legal_mask)
    members = tuple(
        member for group in strategic_groups for member in group.member_action_indices
    )
    if len(set(members)) != len(members) or set(members) != set(legal):
        raise PolicyActionContractError(
            "strategic groups must partition engine-legal actions"
        )

    mask = torch.zeros_like(engine_legal_mask)
    flat_mask = mask.flatten()
    for group in strategic_groups:
        flat_mask[group.representative_action_index] = True
    return mask


def _validate_single_policy_inputs(
    logits: Tensor, representative_mask: Tensor
) -> None:
    """Validate the model output and mask used for one gameplay decision."""

    if (
        not isinstance(logits, Tensor)
        or logits.dtype is not torch.float32
        or logits.shape != (HAND_SLOT_COUNT, POLICY_POSITION_COUNT)
    ):
        raise PolicyActionContractError("policy logits must be float32[4,8]")
    if not bool(torch.isfinite(logits).all().item()):
        raise PolicyActionContractError("policy logits must be finite")
    if (
        not isinstance(representative_mask, Tensor)
        or representative_mask.dtype is not torch.bool
        or representative_mask.shape != logits.shape
    ):
        raise PolicyActionContractError(
            "representative mask must be Boolean with the logits shape"
        )
    if logits.device != representative_mask.device:
        raise PolicyActionContractError(
            "logits and representative mask must share a device"
        )
    if not bool(representative_mask.any().item()):
        raise PolicyActionContractError("representative mask must contain an action")


def apply_representative_mask(logits: Tensor, representative_mask: Tensor) -> Tensor:
    """Give excluded actions zero probability in the later softmax."""

    return logits.masked_fill(~representative_mask, -torch.inf)


def select_representative_action(
    logits: Tensor, representative_mask: Tensor
) -> int:
    """Select one masked action; flattening supplies the canonical tie-break."""

    _validate_single_policy_inputs(logits, representative_mask)
    masked = apply_representative_mask(logits, representative_mask)
    return int(torch.argmax(masked.reshape(ACTION_COUNT)).item())

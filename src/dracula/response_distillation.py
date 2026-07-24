"""Immutable examples and policy-logit ranking for response distillation."""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from typing import Protocol

import torch
from torch import Tensor
from torch.nn import functional as F

from dracula.policy_value import (
    ACTION_COUNT,
    ACTION_SCHEMA_VERSION,
    OBSERVATION_SCHEMA_VERSION,
    OBSERVATION_SIZE,
)

RESPONSE_EXAMPLE_SCHEMA_VERSION = "dracula-response-ranking-example-v1"
RESPONSE_ACTION_GROUP_SCHEMA_VERSION = "dracula-response-action-group-v1"
RESPONSE_RANKING_SCHEMA_VERSION = "dracula-pairwise-response-ranking-v1"
RESPONSE_GROUP_LOGIT_PROFILE = "arithmetic-mean-concrete-logits-v1"
RESPONSE_PAIR_WEIGHT_PROFILE = "absolute-terminal-value-difference-v1"

_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_ROLE_NAMES = frozenset(("queen", "king"))


class ResponseDistillationContractError(ValueError):
    """Captured response evidence or ranking input violates the contract."""


@dataclass(frozen=True, slots=True)
class ResponseCacheIdentity:
    """The same identity used by the live shallow-response cache."""

    information_state_fingerprint: str
    response_config_digest: str

    def __post_init__(self) -> None:
        _validate_digest(
            self.information_state_fingerprint,
            "information-state fingerprint",
        )
        _validate_digest(self.response_config_digest, "response configuration")

    @property
    def digest(self) -> str:
        encoded = (
            f"{RESPONSE_EXAMPLE_SCHEMA_VERSION}\0"
            f"{self.information_state_fingerprint}\0"
            f"{self.response_config_digest}"
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class ResponseActionGroupTarget:
    """One strategically distinct action and its exact Teacher v2 estimate."""

    hand_slot: int
    representative_action_index: int
    member_action_indices: tuple[int, ...]
    mean_terminal_differential: float

    def __post_init__(self) -> None:
        if type(self.hand_slot) is not int or not 0 <= self.hand_slot < 4:
            raise ResponseDistillationContractError("group hand slot is invalid")
        if (
            type(self.representative_action_index) is not int
            or not 0 <= self.representative_action_index < ACTION_COUNT
            or self.representative_action_index // 8 != self.hand_slot
        ):
            raise ResponseDistillationContractError(
                "group representative action is invalid"
            )
        if (
            not isinstance(self.member_action_indices, tuple)
            or not self.member_action_indices
            or tuple(sorted(self.member_action_indices))
            != self.member_action_indices
            or len(set(self.member_action_indices))
            != len(self.member_action_indices)
            or self.representative_action_index
            not in self.member_action_indices
            or any(
                type(index) is not int
                or not 0 <= index < ACTION_COUNT
                or index // 8 != self.hand_slot
                for index in self.member_action_indices
            )
        ):
            raise ResponseDistillationContractError(
                "group concrete actions are invalid"
            )
        if (
            type(self.mean_terminal_differential) not in (int, float)
            or not math.isfinite(self.mean_terminal_differential)
            or not -1.0 <= self.mean_terminal_differential <= 1.0
        ):
            raise ResponseDistillationContractError(
                "group terminal differential must be finite and in [-1, 1]"
            )


@dataclass(frozen=True, slots=True)
class ResponseDistillationExample:
    """Privacy-limited evidence captured after one unique response evaluation."""

    schema_version: str
    ranking_schema_version: str
    observation_schema_version: str
    action_schema_version: str
    observation: tuple[bool, ...]
    legal_mask: tuple[tuple[bool, ...], ...]
    groups: tuple[ResponseActionGroupTarget, ...]
    selected_representative_action_index: int
    selected_concrete_action_index: int
    placement_number: int
    actor_role: str
    actor_is_dealer: bool
    completions_per_action: int
    search_config_digest: str
    response_config_digest: str
    cache_identity: ResponseCacheIdentity

    def __post_init__(self) -> None:
        if self.schema_version != RESPONSE_EXAMPLE_SCHEMA_VERSION:
            raise ResponseDistillationContractError(
                "response example schema is incompatible"
            )
        if self.ranking_schema_version != RESPONSE_RANKING_SCHEMA_VERSION:
            raise ResponseDistillationContractError(
                "response ranking schema is incompatible"
            )
        if self.observation_schema_version != OBSERVATION_SCHEMA_VERSION:
            raise ResponseDistillationContractError(
                "response observation schema is incompatible"
            )
        if self.action_schema_version != ACTION_SCHEMA_VERSION:
            raise ResponseDistillationContractError(
                "response action schema is incompatible"
            )
        if (
            not isinstance(self.observation, tuple)
            or len(self.observation) != OBSERVATION_SIZE
            or any(type(value) is not bool for value in self.observation)
        ):
            raise ResponseDistillationContractError(
                "response observation must be immutable bool[875]"
            )
        if (
            not isinstance(self.legal_mask, tuple)
            or len(self.legal_mask) != 4
            or any(
                not isinstance(row, tuple)
                or len(row) != 8
                or any(type(value) is not bool for value in row)
                for row in self.legal_mask
            )
        ):
            raise ResponseDistillationContractError(
                "response legal mask must be immutable bool[4,8]"
            )
        if not isinstance(self.groups, tuple) or not self.groups:
            raise ResponseDistillationContractError(
                "response example must contain action groups"
            )
        if any(
            not isinstance(group, ResponseActionGroupTarget)
            for group in self.groups
        ):
            raise ResponseDistillationContractError(
                "response example contains an invalid group"
            )
        representatives = tuple(
            group.representative_action_index for group in self.groups
        )
        if representatives != tuple(sorted(representatives)) or len(
            set(representatives)
        ) != len(representatives):
            raise ResponseDistillationContractError(
                "response groups must have canonical unique order"
            )
        legal_indices = {
            row * 8 + column
            for row, values in enumerate(self.legal_mask)
            for column, allowed in enumerate(values)
            if allowed
        }
        grouped_indices = {
            index for group in self.groups for index in group.member_action_indices
        }
        if grouped_indices != legal_indices or len(grouped_indices) != sum(
            len(group.member_action_indices) for group in self.groups
        ):
            raise ResponseDistillationContractError(
                "response groups must partition the legal mask"
            )
        if self.selected_representative_action_index not in representatives:
            raise ResponseDistillationContractError(
                "selected response group is absent"
            )
        selected_group = self.groups[
            representatives.index(self.selected_representative_action_index)
        ]
        if self.selected_concrete_action_index not in (
            selected_group.member_action_indices
        ):
            raise ResponseDistillationContractError(
                "selected concrete action is outside the selected group"
            )
        expected_representative = min(
            self.groups,
            key=lambda group: (
                -group.mean_terminal_differential,
                group.representative_action_index,
            ),
        ).representative_action_index
        if self.selected_representative_action_index != expected_representative:
            raise ResponseDistillationContractError(
                "selected response group is not the maximum-mean group"
            )
        if (
            type(self.placement_number) is not int
            or not 1 <= self.placement_number <= 7
        ):
            raise ResponseDistillationContractError(
                "response placement must be a learned placement from 1 through 7"
            )
        if self.actor_role not in _ROLE_NAMES:
            raise ResponseDistillationContractError("response actor role is invalid")
        if type(self.actor_is_dealer) is not bool:
            raise ResponseDistillationContractError(
                "response dealer status must be Boolean"
            )
        if self.completions_per_action != 4:
            raise ResponseDistillationContractError(
                "the bounded experiment requires four completions per action"
            )
        _validate_digest(self.search_config_digest, "search configuration")
        _validate_digest(self.response_config_digest, "response configuration")
        if not isinstance(self.cache_identity, ResponseCacheIdentity):
            raise ResponseDistillationContractError(
                "response cache identity is invalid"
            )
        if (
            self.cache_identity.response_config_digest
            != self.response_config_digest
        ):
            raise ResponseDistillationContractError(
                "response example and cache configuration disagree"
            )


class ResponseExampleObserver(Protocol):
    """Receives only an immutable, privacy-limited response example."""

    def __call__(self, example: ResponseDistillationExample) -> None: ...


@dataclass(frozen=True, slots=True)
class ResponseRankingLoss:
    total: Tensor
    contributing_pair_count: int
    total_pair_weight: float


def _validate_digest(value: str, label: str) -> None:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise ResponseDistillationContractError(
            f"{label} digest must be 64 lowercase hexadecimal characters"
        )


def _validate_logits(logits: Tensor) -> Tensor:
    if (
        not isinstance(logits, Tensor)
        or logits.dtype is not torch.float32
        or logits.shape != (4, 8)
        or not torch.isfinite(logits).all()
    ):
        raise ResponseDistillationContractError(
            "response logits must be finite float32[4,8]"
        )
    return logits.reshape(ACTION_COUNT)


def response_group_logits(
    logits: Tensor,
    groups: tuple[ResponseActionGroupTarget, ...],
) -> Tensor:
    """Average concrete logits so a mirrored pair gets no size advantage."""

    flat = _validate_logits(logits)
    if not isinstance(groups, tuple) or not groups or any(
        not isinstance(group, ResponseActionGroupTarget) for group in groups
    ):
        raise ResponseDistillationContractError(
            "response ranking requires action groups"
        )
    return torch.stack(
        tuple(
            flat[list(group.member_action_indices)].mean() for group in groups
        )
    )


def response_pairwise_ranking_loss(
    logits: Tensor,
    example: ResponseDistillationExample,
) -> ResponseRankingLoss:
    """Weighted logistic ranking over all non-tied strategic group pairs."""

    if not isinstance(example, ResponseDistillationExample):
        raise ResponseDistillationContractError(
            "response ranking requires a response example"
        )
    scores = response_group_logits(logits, example.groups)
    weighted_losses: list[Tensor] = []
    weights: list[float] = []
    for left_index, left in enumerate(example.groups):
        for right_index in range(left_index + 1, len(example.groups)):
            right = example.groups[right_index]
            difference = (
                left.mean_terminal_differential
                - right.mean_terminal_differential
            )
            if difference == 0:
                continue
            weight = abs(difference)
            sign = 1.0 if difference > 0 else -1.0
            weighted_losses.append(
                weight
                * F.softplus(
                    -sign * (scores[left_index] - scores[right_index])
                )
            )
            weights.append(weight)
    if not weights:
        zero = logits.sum() * 0.0
        return ResponseRankingLoss(zero, 0, 0.0)
    total_weight = sum(weights)
    return ResponseRankingLoss(
        torch.stack(weighted_losses).sum() / total_weight,
        len(weights),
        total_weight,
    )


def select_response_group(
    logits: Tensor,
    groups: tuple[ResponseActionGroupTarget, ...],
) -> int:
    """Rank legal strategic groups without softmax or visit statistics."""

    scores = response_group_logits(logits, groups).detach().cpu().tolist()
    return min(
        range(len(groups)),
        key=lambda index: (
            -float(scores[index]),
            groups[index].representative_action_index,
        ),
    )


__all__ = (
    "RESPONSE_ACTION_GROUP_SCHEMA_VERSION",
    "RESPONSE_EXAMPLE_SCHEMA_VERSION",
    "RESPONSE_GROUP_LOGIT_PROFILE",
    "RESPONSE_PAIR_WEIGHT_PROFILE",
    "RESPONSE_RANKING_SCHEMA_VERSION",
    "ResponseActionGroupTarget",
    "ResponseCacheIdentity",
    "ResponseDistillationContractError",
    "ResponseDistillationExample",
    "ResponseExampleObserver",
    "ResponseRankingLoss",
    "response_group_logits",
    "response_pairwise_ranking_loss",
    "select_response_group",
)

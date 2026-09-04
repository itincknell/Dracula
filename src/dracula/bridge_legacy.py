"""Compatibility records for the retired recurrent policy training path."""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass

import torch
from torch import Tensor

from dracula.bridge import (
    ACTION_COUNT,
    DEALER_INDEX,
    HIDDEN_STATE_BYTES,
    PROGRESS_START,
    ROUND_COUNT,
    ROUND_START,
    BridgeContractViolation,
    PolicyInput,
    PolicyTurnContext,
    PolicyTurnKind,
    _policy_inputs_equal,
    _validate_fingerprint,
    action_index_for_move,
    build_policy_turn_context,
    validate_policy_turn_context,
)
from dracula.engine import EnginePlayer, EngineState, EngineTransition, apply_move
from dracula.models import HIDDEN_SIZE


def _finite_optional(value: float | None, label: str) -> None:
    if value is not None and (
        type(value) not in (int, float) or not math.isfinite(value)
    ):
        raise BridgeContractViolation(f"{label} must be finite when present")


def _validate_identifier(value: object, label: str) -> None:
    if not isinstance(value, str) or not value or "\0" in value:
        raise BridgeContractViolation(f"{label} must be a nonempty string without NUL")


def _validate_hidden_bytes(value: object, label: str) -> None:
    if not isinstance(value, bytes) or len(value) != HIDDEN_STATE_BYTES:
        raise BridgeContractViolation(
            f"{label} must be exactly {HIDDEN_STATE_BYTES} little-endian float32 bytes"
        )
    if not all(
        math.isfinite(number) for number in struct.unpack(f"<{HIDDEN_SIZE}f", value)
    ):
        raise BridgeContractViolation(f"{label} contains a non-finite value")


@dataclass(frozen=True, slots=True, eq=False)
class PolicyTransition:
    """Sealed recurrent training record retained for historical artifacts."""

    fixture_id: str
    learner_id: str
    learner_policy_version: str
    opponent_id: str
    opponent_policy_version: str
    player: EnginePlayer
    state_fingerprint: str
    round_number: int
    own_decision_index: int
    recurrent_step_index: int
    kind: PolicyTurnKind
    policy_input: PolicyInput
    policy_hidden_in: bytes
    action_index: int
    action_log_probability: float | None
    policy_hidden_out: bytes
    critic_value: float | None
    round_return: float | None
    actor_loss_mask: bool

    def __post_init__(self) -> None:
        for value, label in (
            (self.fixture_id, "fixture ID"),
            (self.learner_id, "learner ID"),
            (self.learner_policy_version, "learner policy version"),
            (self.opponent_id, "opponent ID"),
            (self.opponent_policy_version, "opponent policy version"),
        ):
            _validate_identifier(value, label)
        if not isinstance(self.player, EnginePlayer):
            raise BridgeContractViolation("transition player must be an EnginePlayer")
        _validate_fingerprint(self.state_fingerprint)
        if type(self.round_number) is not int or not 1 <= self.round_number <= ROUND_COUNT:
            raise BridgeContractViolation(
                "transition round number must be between 1 and 6"
            )
        if (
            type(self.own_decision_index) is not int
            or not 0 <= self.own_decision_index < 4
        ):
            raise BridgeContractViolation(
                "transition own decision index must be between 0 and 3"
            )
        expected_step = (self.round_number - 1) * 4 + self.own_decision_index
        if self.recurrent_step_index != expected_step:
            raise BridgeContractViolation(
                "recurrent step index does not match round progress"
            )
        if not isinstance(self.kind, PolicyTurnKind):
            raise BridgeContractViolation("transition has an invalid kind")
        if not isinstance(self.policy_input, PolicyInput):
            raise BridgeContractViolation("transition input must be a PolicyInput")
        _validate_hidden_bytes(self.policy_hidden_in, "policy hidden input")
        _validate_hidden_bytes(self.policy_hidden_out, "policy hidden output")
        if type(self.action_index) is not int or not 0 <= self.action_index < ACTION_COUNT:
            raise BridgeContractViolation("action index must be between 0 and 31")
        if not bool(self.policy_input.legal_mask.flatten()[self.action_index].item()):
            raise BridgeContractViolation("transition action index is masked")
        _finite_optional(self.action_log_probability, "action log probability")
        _finite_optional(self.critic_value, "critic value")
        _finite_optional(self.round_return, "round return")
        if self.round_return is not None and not -1.0 <= self.round_return <= 1.0:
            raise BridgeContractViolation("round return must be between -1 and 1")
        if type(self.actor_loss_mask) is not bool:
            raise BridgeContractViolation("actor loss mask must be Boolean")

        round_field = self.policy_input.observation[ROUND_START:PROGRESS_START]
        progress = self.policy_input.observation[PROGRESS_START:DEALER_INDEX]
        if not bool(round_field[self.round_number - 1].item()):
            raise BridgeContractViolation("transition round disagrees with policy input")
        if not bool(progress[self.own_decision_index].item()):
            raise BridgeContractViolation("transition progress disagrees with policy input")

        legal_count = int(self.policy_input.legal_mask.sum().item())
        if self.kind is PolicyTurnKind.FORCED_RECURRENT_TRANSITION:
            if legal_count != 1 or self.own_decision_index != 3:
                raise BridgeContractViolation(
                    "forced transition requires the unique fourth action"
                )
            if not bool(self.policy_input.observation[DEALER_INDEX].item()):
                raise BridgeContractViolation("forced transition must belong to the dealer")
            if any(
                value is not None
                for value in (
                    self.action_log_probability,
                    self.critic_value,
                    self.round_return,
                )
            ) or self.actor_loss_mask:
                raise BridgeContractViolation(
                    "forced transition cannot carry loss samples"
                )
        else:
            if legal_count < 2:
                raise BridgeContractViolation(
                    "learned transition requires multiple legal actions"
                )
            if self.action_log_probability is None or self.critic_value is None:
                raise BridgeContractViolation(
                    "learned transition requires behavior and critic values"
                )
            if not self.actor_loss_mask:
                raise BridgeContractViolation(
                    "learned transition must contribute actor loss"
                )


def hidden_state_to_bytes(hidden_state: Tensor) -> bytes:
    """Encode one legacy float32 recurrent state in fixed little-endian form."""

    if not isinstance(hidden_state, Tensor) or hidden_state.shape != (HIDDEN_SIZE,):
        raise BridgeContractViolation(f"hidden state must have shape ({HIDDEN_SIZE},)")
    if hidden_state.dtype is not torch.float32:
        raise BridgeContractViolation("hidden state must have float32 dtype")
    values = hidden_state.detach().cpu().tolist()
    if not all(math.isfinite(value) for value in values):
        raise BridgeContractViolation("hidden state contains a non-finite value")
    return struct.pack(f"<{HIDDEN_SIZE}f", *values)


def hidden_state_from_bytes(hidden_state: bytes) -> Tensor:
    """Decode one validated legacy recurrent state without sharing storage."""

    _validate_hidden_bytes(hidden_state, "hidden state")
    return torch.tensor(
        struct.unpack(f"<{HIDDEN_SIZE}f", hidden_state), dtype=torch.float32
    )


def validate_policy_transition(transition: PolicyTransition, state: EngineState) -> None:
    """Verify that a historical transition matches its authoritative state."""

    if not isinstance(transition, PolicyTransition):
        raise BridgeContractViolation("transition must be a PolicyTransition")
    context = build_policy_turn_context(state, transition.player)
    if (
        transition.state_fingerprint != context.state_fingerprint
        or transition.round_number != context.round_number
        or transition.own_decision_index != context.own_decision_index
        or transition.kind is not context.kind
        or not _policy_inputs_equal(transition.policy_input, context.input)
        or context.action_table[transition.action_index] is None
    ):
        raise BridgeContractViolation(
            "policy transition does not match its engine state"
        )


def build_policy_transition(
    *,
    context: PolicyTurnContext,
    engine_transition: EngineTransition,
    fixture_id: str,
    learner_id: str,
    learner_policy_version: str,
    opponent_id: str,
    opponent_policy_version: str,
    recurrent_step_index: int,
    policy_hidden_in: bytes,
    policy_hidden_out: bytes,
    action_log_probability: float | None,
    critic_value: float | None,
    round_return: float | None = None,
) -> PolicyTransition:
    """Build and validate one compatibility record from an accepted transition."""

    validate_policy_turn_context(engine_transition.previous_state, context)
    expected_engine_transition = apply_move(
        engine_transition.previous_state, engine_transition.move
    )
    if engine_transition != expected_engine_transition:
        raise BridgeContractViolation(
            "engine transition does not match deterministic application"
        )
    action_index = action_index_for_move(engine_transition.move, context.player)
    if context.action_table[action_index] != engine_transition.move:
        raise BridgeContractViolation(
            "engine transition was not authorized by the context"
        )
    if (
        engine_transition.played_move.player is not engine_transition.move.player
        or engine_transition.played_move.hand_slot != engine_transition.move.hand_slot
        or engine_transition.played_move.global_grid_index
        != engine_transition.move.global_grid_index
    ):
        raise BridgeContractViolation("engine transition move records disagree")

    transition = PolicyTransition(
        fixture_id=fixture_id,
        learner_id=learner_id,
        learner_policy_version=learner_policy_version,
        opponent_id=opponent_id,
        opponent_policy_version=opponent_policy_version,
        player=context.player,
        state_fingerprint=context.state_fingerprint,
        round_number=context.round_number,
        own_decision_index=context.own_decision_index,
        recurrent_step_index=recurrent_step_index,
        kind=context.kind,
        policy_input=context.input,
        policy_hidden_in=policy_hidden_in,
        action_index=action_index,
        action_log_probability=action_log_probability,
        policy_hidden_out=policy_hidden_out,
        critic_value=critic_value,
        round_return=round_return,
        actor_loss_mask=context.kind is PolicyTurnKind.LEARNED,
    )
    validate_policy_transition(transition, engine_transition.previous_state)
    return transition

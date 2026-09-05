"""Bridge one claimed local opponent turn through the configured policy.

The local transaction service owns claim and commit semantics. This module owns
the intervening inference boundary: it builds only the opponent's visible input,
invokes the selected policy, validates recurrent compatibility state when used,
and converts the returned action through the legal action table.
"""

from __future__ import annotations

from dracula.api.policy import (
    PolicyDescriptor,
    PolicyExecutor,
    PolicyTurnRequest,
)
from dracula.api.session import GameSession, validate_hidden_state
from dracula.engine import EngineTransition, other_player


def execute_claimed_policy_turn(
    claimed: GameSession,
    policy_executor: PolicyExecutor,
) -> tuple[EngineTransition, bytes | None]:
    """Run one policy inference against the immutable state in a claimed turn."""

    state = claimed.engine_state
    opponent = other_player(claimed.human_role)
    persisted = claimed.policy_session
    descriptor = PolicyDescriptor(
        policy_id=persisted.policy_id,
        policy_version=persisted.policy_version,
        artifact_id=persisted.artifact_id,
        artifact_sha256=persisted.artifact_sha256,
        observation_schema_version=persisted.observation_schema_version,
        action_schema_version=persisted.action_schema_version,
        hidden_state_schema_version=persisted.hidden_state_schema_version,
        inference_profile=persisted.inference_profile,
    )

    # These heavier modules are needed only for an actual inference request;
    # ordinary local health and state reads retain a lightweight import path.
    from dracula.bridge import apply_policy_action, build_policy_turn_context
    from dracula.search import information_state_from_engine

    context = build_policy_turn_context(state, opponent)
    information = information_state_from_engine(state, opponent)
    result = policy_executor.invoke(
        PolicyTurnRequest(
            game_id=claimed.game_id,
            policy=descriptor,
            turn_number=len(state.current_round_moves) + 1,
            player=opponent,
            round_number=state.round_number,
            turn_kind=context.kind,
            policy_input=None,
            action_table=context.action_table,
            information_state=information,
            hidden_state=persisted.hidden_state,
        )
    )
    if result.hidden_state is not None:
        validate_hidden_state(result.hidden_state)
    return (
        apply_policy_action(state, context, result.action_index),
        result.hidden_state,
    )


__all__ = ("execute_claimed_policy_turn",)

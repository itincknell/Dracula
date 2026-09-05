"""Build persisted local sessions and their append-only public event history.

Local SQLite gameplay records both immutable engine state and public lifecycle
events. This module owns that deterministic presentation bookkeeping so the
transaction service can focus on idempotency, repository conflicts, and policy
turn claims.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from uuid import NAMESPACE_URL, UUID, uuid5

from dracula.api.contracts import PublicEvent
from dracula.api.policy import PolicyDescriptor, zero_hidden_state
from dracula.api.presentation import phase_for_state
from dracula.api.session import GameSession, PolicySession
from dracula.cards import CARD_SCHEMA_VERSION
from dracula.engine import (
    ENGINE_VERSION,
    RULES_VERSION,
    EnginePlayer,
    EngineState,
    EngineStatus,
    EngineTransition,
    other_player,
)
from dracula.randomness import RANDOMNESS_SCHEMA_VERSION


def make_event(
    game_id: UUID,
    sequence: int,
    event_type: str,
    occurred_at: datetime,
    prior_version: int,
    resulting_version: int,
    request_id: str | None,
    payload: Mapping[str, int | str | bool | None],
) -> PublicEvent:
    """Create one deterministic append-only public event record."""

    return PublicEvent(
        game_id=game_id,
        event_id=str(uuid5(NAMESPACE_URL, f"dracula:{game_id}:{sequence}")),
        sequence=sequence,
        event_type=event_type,  # type: ignore[arg-type]
        occurred_at=occurred_at,
        prior_version=prior_version,
        resulting_version=resulting_version,
        request_id=request_id,
        payload=dict(payload),
    )


def initial_session(
    *,
    game_id: UUID,
    human_role: EnginePlayer,
    state: EngineState,
    descriptor: PolicyDescriptor,
    narration_enabled: bool,
    now: datetime,
    request_id: str,
    move_secret: bytes,
) -> GameSession:
    """Build a new local session and its initial public lifecycle events."""

    events = (
        make_event(
            game_id,
            1,
            "game_created",
            now,
            0,
            0,
            request_id,
            {
                "human_role": human_role.value,
                "opponent_role": other_player(human_role).value,
                "policy_id": descriptor.policy_id,
                "policy_version": descriptor.policy_version,
                "artifact_id": descriptor.artifact_id,
                "engine_version": ENGINE_VERSION,
                "rules_version": RULES_VERSION,
                "card_schema_version": CARD_SCHEMA_VERSION,
                "randomness_schema_version": RANDOMNESS_SCHEMA_VERSION,
                "observation_schema_version": descriptor.observation_schema_version,
                "action_schema_version": descriptor.action_schema_version,
                "hidden_state_schema_version": descriptor.hidden_state_schema_version,
                "inference_profile": descriptor.inference_profile,
                "narration_enabled": narration_enabled,
            },
        ),
        make_event(
            game_id,
            2,
            "round_started",
            now,
            0,
            0,
            request_id,
            {
                "round_number": 1,
                "center_card": state.coffin[4],
                "dealer": state.dealer.value,
                "first_player": state.active_player.value,
            },
        ),
    )
    return GameSession(
        game_id=game_id,
        version=0,
        revision=0,
        human_role=human_role,
        engine_state=state,
        policy_session=PolicySession(
            policy_id=descriptor.policy_id,
            policy_version=descriptor.policy_version,
            artifact_id=descriptor.artifact_id,
            artifact_sha256=descriptor.artifact_sha256,
            observation_schema_version=descriptor.observation_schema_version,
            action_schema_version=descriptor.action_schema_version,
            hidden_state_schema_version=descriptor.hidden_state_schema_version,
            inference_profile=descriptor.inference_profile,
            hidden_state=zero_hidden_state(),
        ),
        move_secret=move_secret,
        phase=phase_for_state(state, human_role),
        events=events,
        idempotency_records=(),
    )


def move_events(
    session: GameSession,
    transition: EngineTransition,
    request_id: str,
    resulting_version: int,
    now: datetime,
) -> tuple[PublicEvent, ...]:
    """Build the public events caused by one accepted move."""

    sequence = len(session.events) + 1
    events = [
        make_event(
            session.game_id,
            sequence,
            "move_accepted",
            now,
            session.version,
            resulting_version,
            request_id,
            {
                "player": transition.played_move.player.value,
                "card_id": transition.played_move.card_id,
                "hand_slot": transition.played_move.hand_slot,
                "position": transition.played_move.global_grid_index,
                "turn_number": transition.played_move.turn_number,
                "round_number": transition.state.round_number,
            },
        )
    ]
    result = transition.round_result
    if result is None:
        return tuple(events)

    # Scoring events follow dealer order because the client animates the dealer's
    # row or column calculation before the other player's calculation.
    for player in (result.dealer, other_player(result.dealer)):
        sequence += 1
        event_type = (
            "row_score_ready"
            if player is EnginePlayer.QUEEN
            else "column_score_ready"
        )
        events.append(
            make_event(
                session.game_id,
                sequence,
                event_type,
                now,
                session.version,
                resulting_version,
                request_id,
                {
                    "round_number": result.round_number,
                    "player": player.value,
                    "round_score": result.round_scores[player],
                },
            )
        )
    events.append(
        make_event(
            session.game_id,
            sequence + 1,
            "round_completed",
            now,
            session.version,
            resulting_version,
            request_id,
            {
                "round_number": result.round_number,
                "queen_score": result.round_scores.queen,
                "king_score": result.round_scores.king,
            },
        )
    )
    return tuple(events)


def round_advance_event(
    session: GameSession,
    next_state: EngineState,
    request_id: str,
    resulting_version: int,
    now: datetime,
) -> PublicEvent:
    """Build the single public event emitted after advancing a scored round."""

    if next_state.status is EngineStatus.GAME_COMPLETE:
        outcome = phase_for_state(next_state, session.human_role)
        return make_event(
            session.game_id,
            len(session.events) + 1,
            "game_completed",
            now,
            session.version,
            resulting_version,
            request_id,
            {
                "outcome": outcome.outcome,  # type: ignore[union-attr]
                "queen_total": next_state.total_scores.queen,
                "king_total": next_state.total_scores.king,
            },
        )
    return make_event(
        session.game_id,
        len(session.events) + 1,
        "round_started",
        now,
        session.version,
        resulting_version,
        request_id,
        {
            "round_number": next_state.round_number,
            "center_card": next_state.coffin[4],
            "dealer": next_state.dealer.value,
            "first_player": next_state.active_player.value,
        },
    )


__all__ = ("initial_session", "make_event", "move_events", "round_advance_event")

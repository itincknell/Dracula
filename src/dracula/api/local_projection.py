"""Project private local sessions into browser-safe gameplay views.

Local gameplay exposes opaque, version-bound move tokens rather than accepting
raw engine moves. This module owns those tokens and the final public projection;
session persistence and validation remain in :mod:`dracula.api.session`.
"""

from __future__ import annotations

import base64
import hashlib
import hmac

from dracula.api.contracts import HumanGameView, HumanTurnPhase, LegalMove, RoundRecord
from dracula.api.presentation import played_move, player_score, round_record
from dracula.api.session import GameSession, validate_session
from dracula.engine import EngineStatus, PlayerValues, legal_moves, other_player


def move_id_for(
    session: GameSession, hand_slot: int, position: int, card_id: str
) -> str:
    """Create an opaque token bound to one current local-session move."""

    message = (
        f"move-v1\0{session.game_id}\0{session.version}\0"
        f"{hand_slot}\0{position}\0{card_id}"
    ).encode("utf-8")
    digest = hmac.new(session.move_secret, message, hashlib.sha256).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


def resolve_human_move_id(
    session: GameSession, move_id: str
) -> tuple[int, int] | None:
    """Resolve a token only while it still names a current legal human move."""

    state = session.engine_state
    if (
        state.status is not EngineStatus.PLAYING
        or state.active_player is not session.human_role
    ):
        return None
    for move in legal_moves(state, session.human_role):
        card_id = state.hands[session.human_role][move.hand_slot]
        assert card_id is not None
        candidate = move_id_for(
            session, move.hand_slot, move.global_grid_index, card_id
        )
        if hmac.compare_digest(candidate, move_id):
            return move.hand_slot, move.global_grid_index
    return None


def project_human_game(
    session: GameSession, *, narration_enabled: bool = False
) -> HumanGameView:
    """Project a validated private local session into its browser-safe view."""

    # Some callers project a newly constructed session before its repository
    # commit, so this boundary still validates the complete private record.
    validate_session(session)
    state = session.engine_state
    running = PlayerValues(queen=0, king=0)
    completed: list[RoundRecord] = []
    for result in state.completed_rounds:
        completed.append(round_record(result, session.human_role, running))
        running = PlayerValues(
            queen=running.queen + result.round_scores.queen,
            king=running.king + result.round_scores.king,
        )
    pending = (
        None
        if state.pending_round_result is None
        else round_record(state.pending_round_result, session.human_role, running)
    )

    legal: list[LegalMove] = []
    if (
        state.status is EngineStatus.PLAYING
        and state.active_player is session.human_role
        and isinstance(session.phase, HumanTurnPhase)
    ):
        for move in legal_moves(state, session.human_role):
            card_id = state.hands[session.human_role][move.hand_slot]
            assert card_id is not None
            legal.append(
                LegalMove(
                    move_id=move_id_for(
                        session, move.hand_slot, move.global_grid_index, card_id
                    ),
                    card_id=card_id,
                    hand_slot=move.hand_slot,
                    position=move.global_grid_index,
                )
            )
    return HumanGameView(
        game_id=session.game_id,
        version=session.version,
        status=state.status.value,
        round_number=state.round_number,
        turn_number=len(state.current_round_moves),
        dealer=state.dealer.value,
        active_player=(
            None if state.active_player is None else state.active_player.value
        ),
        human_role=session.human_role.value,
        opponent_role=other_player(session.human_role).value,
        coffin=state.coffin,
        current_round_moves=tuple(
            played_move(move) for move in state.current_round_moves
        ),
        pending_round_result=pending,
        completed_rounds=tuple(completed),
        total_scores=player_score(state.total_scores, session.human_role),
        phase=session.phase,
        latest_event_sequence=len(session.events),
        narration_enabled=narration_enabled,
        human_hand=state.hands[session.human_role],
        legal_moves=tuple(legal),
        events=session.events,
    )


__all__ = ("move_id_for", "project_human_game", "resolve_human_move_id")

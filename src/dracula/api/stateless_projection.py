"""Convert reconstructed private engine state into the public game view.

The projection exposes the human hand, public coffin, scores, phases, and legal
human commands while withholding Dracula's hand and policy/search diagnostics.
"""

from __future__ import annotations

from typing import Protocol

from dracula.api.presentation import phase_for_state, player_score, round_record
from dracula.api.stateless_contracts import (
    StatelessHumanGameView,
    StatelessLegalMove,
    StatelessRoundRecord,
    VisiblePlayedMove,
)
from dracula.engine import (
    EnginePlayer,
    EnginePlayedMove,
    EngineRoundResult,
    EngineState,
    EngineStatus,
    PlayerValues,
    legal_moves,
    other_player,
)


class PublicGameSource(Protocol):
    """Minimum trusted private container accepted by the public projector."""

    state: EngineState
    human_role: EnginePlayer


def _visible_move(move: EnginePlayedMove) -> VisiblePlayedMove:
    return VisiblePlayedMove(
        player=move.player.value,
        card_id=move.card_id,
        position=move.global_grid_index,
        turn_number=move.turn_number,
    )


def _stateless_round(
    result: EngineRoundResult,
    human_role: EnginePlayer,
    previous_queen: int,
    previous_king: int,
) -> StatelessRoundRecord:
    """Convert one verified result into browser scoring and animation facts."""

    projected = round_record(
        result,
        human_role,
        PlayerValues(queen=previous_queen, king=previous_king),
    )
    return StatelessRoundRecord(
        round_number=projected.round_number,
        dealer=projected.dealer,
        coffin=projected.coffin,
        moves=tuple(_visible_move(move) for move in result.moves),
        line_scores=projected.line_scores,
        scoring_sequence=projected.scoring_sequence,
        round_scores=projected.round_scores,
    )


def project_stateless_game(game: PublicGameSource) -> StatelessHumanGameView:
    """Expose only the browser-safe view of a replayed game.

    ``game`` is structural here to keep this projection module independent from
    the replay service's private container type.
    """

    state = game.state
    human_role = game.human_role
    opponent = other_player(human_role)
    queen_total = 0
    king_total = 0
    completed: list[StatelessRoundRecord] = []
    for result in state.completed_rounds:
        completed.append(_stateless_round(result, human_role, queen_total, king_total))
        queen_total += result.round_scores.queen
        king_total += result.round_scores.king
    pending = (
        None
        if state.pending_round_result is None
        else _stateless_round(
            state.pending_round_result,
            human_role,
            queen_total,
            king_total,
        )
    )
    # Legal actions are exposed only for the human's current turn; the opponent
    # hand and its possible actions never enter the public projection.
    legal = ()
    if state.status is EngineStatus.PLAYING and state.active_player is human_role:
        legal = tuple(
            StatelessLegalMove(
                card_id=state.hands[human_role][move.hand_slot],  # type: ignore[arg-type]
                hand_slot=move.hand_slot,
                position=move.global_grid_index,
            )
            for move in legal_moves(state, human_role)
        )
    return StatelessHumanGameView(
        status=state.status.value,
        round_number=state.round_number,
        turn_number=len(state.current_round_moves),
        dealer=state.dealer.value,
        active_player=None if state.active_player is None else state.active_player.value,
        human_role=human_role.value,
        opponent_role=opponent.value,
        coffin=state.coffin,
        current_round_moves=tuple(_visible_move(move) for move in state.current_round_moves),
        pending_round_result=pending,
        completed_rounds=tuple(completed),
        total_scores=player_score(state.total_scores, human_role),
        phase=phase_for_state(state, human_role),
        human_hand=state.hands[human_role],
        legal_moves=legal,
    )


__all__ = ("project_stateless_game",)

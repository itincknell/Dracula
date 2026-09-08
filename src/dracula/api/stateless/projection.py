"""Convert reconstructed private engine state into the public game view.

The projection exposes the human hand, public coffin, scores, phases, and legal
human commands while withholding Dracula's hand and policy/search diagnostics.
"""

from __future__ import annotations

from dracula.api.presentation import (
    phase_for_state,
    player_score,
    round_presentation,
)
from dracula.api.stateless.contracts import (
    StatelessHumanGameView,
    StatelessLegalMove,
    StatelessRoundRecord,
    VisiblePlayedMove,
)
from dracula.game.engine import (
    EnginePlayer,
    EnginePlayedMove,
    EngineRoundResult,
    EngineState,
    EngineStatus,
    PlayerValues,
    legal_moves,
    other_player,
)


def _visible_move(move: EnginePlayedMove) -> VisiblePlayedMove:
    """Remove the private stable-hand-slot detail from an accepted move."""

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

    projected = round_presentation(
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


def _completed_rounds(
    state: EngineState,
    human_role: EnginePlayer,
) -> tuple[StatelessRoundRecord, ...]:
    """Project rounds while carrying the totals used by scoring animation."""

    queen_total = 0
    king_total = 0
    completed: list[StatelessRoundRecord] = []
    for result in state.completed_rounds:
        # Animation for each archived round begins from the totals accumulated
        # by the rounds before it, so carry those totals forward in order.
        completed.append(_stateless_round(result, human_role, queen_total, king_total))
        queen_total += result.round_scores.queen
        king_total += result.round_scores.king
    return tuple(completed)


def _pending_round(
    state: EngineState,
    human_role: EnginePlayer,
) -> StatelessRoundRecord | None:
    """Project the scored round while excluding it from prior running totals."""

    result = state.pending_round_result
    if result is None:
        return None
    # ROUND_COMPLETE totals already include the pending result. The animation
    # needs the totals that were visible immediately before that result.
    return _stateless_round(
        result,
        human_role,
        state.total_scores.queen - result.round_scores.queen,
        state.total_scores.king - result.round_scores.king,
    )


def _human_legal_moves(
    state: EngineState,
    human_role: EnginePlayer,
) -> tuple[StatelessLegalMove, ...]:
    """Expose placements only while the reconstructed human owns the turn."""

    if state.status is not EngineStatus.PLAYING or state.active_player is not human_role:
        return ()
    # ``legal_moves`` returns engine moves whose card is addressed by its stable
    # hand slot. The browser receives that slot plus the visible card so it can
    # render and submit the exact choice without receiving an EngineMove object.
    return tuple(
        StatelessLegalMove(
            card_id=state.hands[human_role][move.hand_slot],  # type: ignore[arg-type]
            hand_slot=move.hand_slot,
            position=move.global_grid_index,
        )
        for move in legal_moves(state, human_role)
    )


def project_stateless_game(
    state: EngineState,
    human_role: EnginePlayer,
) -> StatelessHumanGameView:
    """Expose only the browser-safe fields for one reconstructed engine state."""

    opponent = other_player(human_role)
    # Every field is selected explicitly. Avoiding a generic dataclass dump is
    # what keeps the seed, stock, opponent hand, and policy data out of JSON.
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
        pending_round_result=_pending_round(state, human_role),
        completed_rounds=_completed_rounds(state, human_role),
        total_scores=player_score(state.total_scores, human_role),
        phase=phase_for_state(state, human_role),
        human_hand=state.hands[human_role],
        legal_moves=_human_legal_moves(state, human_role),
    )

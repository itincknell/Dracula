"""Pure line, coffin, round, and game scoring rules."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

from dracula.cards import Card, card_by_id
from dracula.engine_types import (
    COFFIN_SIZE,
    EnginePlayer,
    EngineRoundResult,
    EngineState,
    GameOutcome,
    GameOutcomeReason,
    LineOrientation,
    LineScore,
    MalformedState,
    MultiplierReason,
    PlayerValues,
)


def _line_values(cards: Sequence[Card], orientation: LineOrientation) -> tuple[int, int, int]:
    if orientation is LineOrientation.ROW:
        return tuple(card.horizontal_value for card in cards)  # type: ignore[return-value]
    return tuple(card.vertical_value for card in cards)  # type: ignore[return-value]


def score_line(
    card_ids: Sequence[str],
    orientation: LineOrientation,
    line_index: int = 0,
) -> LineScore:
    """Score one row or column with directional values and multiplier precedence."""

    try:
        orientation = LineOrientation(orientation)
    except (TypeError, ValueError) as error:
        raise ValueError(f"unknown line orientation: {orientation!r}") from error
    if type(line_index) is not int or not 0 <= line_index < 3:
        raise ValueError("line index must be between 0 and 2")
    if len(card_ids) != 3:
        raise ValueError("a line must contain exactly three cards")

    cards = tuple(card_by_id(card_id) for card_id in card_ids)
    ids = tuple(card.card_id for card in cards)
    values = _line_values(cards, orientation)
    base_value = sum(values)
    # A Vampire overrides both arithmetic and all ordinary multiplier rules.
    if any(card.is_vampire for card in cards):
        multiplier = 0
        reason = MultiplierReason.VAMPIRE
    else:
        suit_counts = Counter(card.suit for card in cards)
        colors = {card.color for card in cards}
        # Only one multiplier applies, so test the strongest condition first.
        if len(suit_counts) == 1:
            multiplier = 5
            reason = MultiplierReason.SAME_SUIT
        elif len(colors) == 1:
            multiplier = 3
            reason = MultiplierReason.SAME_COLOR
        elif max(suit_counts.values()) >= 2:
            multiplier = 2
            reason = MultiplierReason.SUIT_PAIR
        else:
            multiplier = 1
            reason = MultiplierReason.NONE
    return LineScore(
        orientation=orientation,
        line_index=line_index,
        cards=ids,  # type: ignore[arg-type]
        values=values,
        base_value=base_value,
        multiplier=multiplier,
        multiplier_reason=reason,
        total=base_value * multiplier,
    )


def _line_cards(
    coffin: Sequence[str], indices: tuple[int, int, int]
) -> tuple[str, str, str]:
    return coffin[indices[0]], coffin[indices[1]], coffin[indices[2]]


def score_coffin(
    coffin: Sequence[str],
) -> PlayerValues[tuple[LineScore, LineScore, LineScore]]:
    """Score the Queen rows and King columns of a completed coffin."""

    if len(coffin) != COFFIN_SIZE:
        raise ValueError("a completed coffin must contain nine cards")
    for card_id in coffin:
        card_by_id(card_id)
    rows = ((0, 1, 2), (3, 4, 5), (6, 7, 8))
    columns = ((0, 3, 6), (1, 4, 7), (2, 5, 8))
    queen_lines = tuple(
        score_line(_line_cards(coffin, indices), LineOrientation.ROW, line_index)
        for line_index, indices in enumerate(rows)
    )
    king_lines = tuple(
        score_line(_line_cards(coffin, indices), LineOrientation.COLUMN, line_index)
        for line_index, indices in enumerate(columns)
    )
    return PlayerValues(queen=queen_lines, king=king_lines)  # type: ignore[arg-type]


def resolve_round_scores(
    queen_line_scores: Sequence[int], king_line_scores: Sequence[int]
) -> PlayerValues[int]:
    """Select both round scores by discarding tied ranked tiers in lockstep."""

    if len(queen_line_scores) != 3 or len(king_line_scores) != 3:
        raise ValueError("each player must have exactly three line scores")
    if any(type(value) is not int or value < 0 for value in (*queen_line_scores, *king_line_scores)):
        raise ValueError("line scores must be non-negative integers")
    queen_sorted = sorted(queen_line_scores, reverse=True)
    king_sorted = sorted(king_line_scores, reverse=True)
    # The third-ranked scores are the required fallback if both higher tiers
    # tie; otherwise the first non-tied tier supplies both recorded scores.
    selected_index = 2
    for index in range(2):
        if queen_sorted[index] != king_sorted[index]:
            selected_index = index
            break
    return PlayerValues(queen=queen_sorted[selected_index], king=king_sorted[selected_index])


def make_round_result(state: EngineState) -> EngineRoundResult:
    """Derive the auditable public result from a completed private round state."""

    if any(card_id is None for card_id in state.coffin):
        raise MalformedState("a round result requires a completed coffin")
    coffin = state.coffin  # type: ignore[assignment]
    line_scores = score_coffin(coffin)
    round_scores = resolve_round_scores(
        tuple(line.total for line in line_scores.queen),
        tuple(line.total for line in line_scores.king),
    )
    return EngineRoundResult(
        round_number=state.round_number,
        dealer=state.dealer,
        coffin=coffin,
        moves=state.current_round_moves,
        line_scores=line_scores,
        round_scores=round_scores,
    )


def resolve_game_outcome(
    total_scores: PlayerValues[int], sixth_round_scores: PlayerValues[int]
) -> GameOutcome:
    """Resolve the game by total score, then the sixth-round tie-break."""

    for label, scores in (("total", total_scores), ("sixth-round", sixth_round_scores)):
        if not isinstance(scores, PlayerValues) or any(
            type(value) is not int or value < 0 for value in (scores.queen, scores.king)
        ):
            raise ValueError(f"{label} scores must be non-negative player values")
    if total_scores.queen != total_scores.king:
        winner = (
            EnginePlayer.QUEEN
            if total_scores.queen > total_scores.king
            else EnginePlayer.KING
        )
        reason = GameOutcomeReason.TOTAL_SCORE
    elif sixth_round_scores.queen != sixth_round_scores.king:
        winner = (
            EnginePlayer.QUEEN
            if sixth_round_scores.queen > sixth_round_scores.king
            else EnginePlayer.KING
        )
        reason = GameOutcomeReason.SIXTH_ROUND_SCORE
    else:
        winner = None
        reason = GameOutcomeReason.TIE
    return GameOutcome(winner, reason, total_scores, sixth_round_scores)

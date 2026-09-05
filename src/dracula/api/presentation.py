"""Project engine scoring and lifecycle facts into public presentation values.

Both stateless production and local gameplay use these pure helpers so phase,
line-score, and round-result representations cannot drift between transports.
"""

from __future__ import annotations

from dracula.api.contracts import (
    GameCompletePhase,
    HumanTurnPhase,
    LineScore as ApiLineScore,
    OpponentTurnPhase,
    PlayedMove,
    PlayerScore,
    ResumablePhase,
    RoundRecord,
    ScoringPhase,
    ScoringStep,
)
from dracula.cards import Suit, card_by_id
from dracula.engine import (
    EnginePlayedMove,
    EnginePlayer,
    EngineRoundResult,
    EngineState,
    EngineStatus,
    LineScore,
    MultiplierReason,
    PlayerValues,
    derive_game_outcome,
    other_player,
)


def phase_for_state(state: EngineState, human_role: EnginePlayer) -> ResumablePhase:
    """Project engine lifecycle into the browser's resumable presentation phase."""

    if state.status is EngineStatus.GAME_COMPLETE:
        outcome = derive_game_outcome(state)
        winner = (
            "tie"
            if outcome.winner is None
            else "human"
            if outcome.winner is human_role
            else "opponent"
        )
        return GameCompletePhase(outcome=winner)
    if state.status is EngineStatus.ROUND_COMPLETE:
        return ScoringPhase(round_number=state.round_number, next_step_index=0)
    if state.active_player is human_role:
        return HumanTurnPhase()
    return OpponentTurnPhase(status="ready")


def player_score(values: PlayerValues[int], human_role: EnginePlayer) -> PlayerScore:
    """Relabel Queen/King values as human/opponent values."""

    return PlayerScore(
        human=values[human_role], opponent=values[other_player(human_role)]
    )


def played_move(move: EnginePlayedMove) -> PlayedMove:
    """Project an accepted engine move into the shared public API shape."""

    return PlayedMove(
        player=move.player.value,
        card_id=move.card_id,
        hand_slot=move.hand_slot,
        position=move.global_grid_index,
        turn_number=move.turn_number,
    )


def _multiplier_label(line: LineScore) -> str:
    cards = tuple(card_by_id(card_id) for card_id in line.cards)
    if line.multiplier_reason is MultiplierReason.VAMPIRE:
        return "Vampire"
    if line.multiplier_reason is MultiplierReason.NONE:
        return "No Multiplier"
    if line.multiplier_reason is MultiplierReason.SAME_COLOR:
        return f"3× {cards[0].color.value.title()}"
    suit_counts = {suit: sum(card.suit is suit for card in cards) for suit in Suit}
    suit = max(suit_counts, key=suit_counts.get)  # type: ignore[arg-type]
    prefix = "2×" if line.multiplier_reason is MultiplierReason.SUIT_PAIR else "3×"
    names = {
        Suit.CLUBS: "Clubs",
        Suit.DIAMONDS: "Diamonds",
        Suit.HEARTS: "Hearts",
        Suit.SPADES: "Spades",
    }
    return f"{prefix} {names[suit]}"


def _highlighted_cards(line: LineScore) -> tuple[str, ...]:
    cards = tuple(card_by_id(card_id) for card_id in line.cards)
    if line.multiplier_reason is MultiplierReason.NONE:
        return ()
    if line.multiplier_reason is MultiplierReason.VAMPIRE:
        return tuple(card.card_id for card in cards if card.is_vampire)
    if line.multiplier_reason is MultiplierReason.SUIT_PAIR:
        counts = {suit: sum(card.suit is suit for card in cards) for suit in Suit}
        repeated = max(counts, key=counts.get)  # type: ignore[arg-type]
        return tuple(card.card_id for card in cards if card.suit is repeated)
    return line.cards


def _api_line(line: LineScore) -> ApiLineScore:
    return ApiLineScore(
        direction=line.orientation.value,
        index=line.line_index,
        card_ids=line.cards,
        base_value=line.base_value,
        multiplier=line.multiplier,  # type: ignore[arg-type]
        multiplier_reason=line.multiplier_reason.value,
        multiplier_label=_multiplier_label(line),
        highlighted_card_ids=_highlighted_cards(line),
        total=line.total,
    )


def _line_scoring_steps(result: EngineRoundResult) -> list[ScoringStep]:
    """Build line-calculation steps in the dealer-first presentation order."""

    steps: list[ScoringStep] = []
    for player in (result.dealer, other_player(result.dealer)):
        lines = result.line_scores[player]
        ranked_indices = sorted(
            range(3), key=lambda index: (-lines[index].total, index)
        )
        rank_by_index = {
            line_index: rank + 1 for rank, line_index in enumerate(ranked_indices)
        }
        for line_index, line in enumerate(lines):
            steps.append(
                ScoringStep(
                    kind="score_line",
                    player=player.value,
                    line=_api_line(line),
                    details={
                        "value_1": line.values[0],
                        "value_2": line.values[1],
                        "value_3": line.values[2],
                        "base_value": line.base_value,
                        "multiplier": line.multiplier,
                        "total": line.total,
                        "rank": rank_by_index[line_index],
                    },
                )
            )
    return steps


def _score_comparison_steps(
    result: EngineRoundResult,
    human_role: EnginePlayer,
) -> tuple[list[ScoringStep], int]:
    """Compare ranked line totals until the round's deciding rank is known."""

    human_totals = sorted(
        (line.total for line in result.line_scores[human_role]), reverse=True
    )
    opponent = other_player(human_role)
    opponent_totals = sorted(
        (line.total for line in result.line_scores[opponent]), reverse=True
    )
    steps: list[ScoringStep] = []
    selected_rank = 2
    for rank in range(3):
        tied = human_totals[rank] == opponent_totals[rank]
        steps.append(
            ScoringStep(
                kind="compare_candidates",
                player=None,
                line=None,
                details={
                    "rank": rank + 1,
                    "human_score": human_totals[rank],
                    "opponent_score": opponent_totals[rank],
                    "tied": tied,
                },
            )
        )
        if rank < 2 and not tied:
            selected_rank = rank
            break
    return steps, selected_rank


def _scoring_sequence(
    result: EngineRoundResult,
    human_role: EnginePlayer,
    previous_totals: PlayerValues[int],
) -> tuple[ScoringStep, ...]:
    """Build the deterministic line, tie-break, and total-update animation script."""

    # Dealer-first line animation is a presentation choice, independent of
    # Queen/King enum order or which role belongs to the human.
    steps = _line_scoring_steps(result)
    comparisons, selected_rank = _score_comparison_steps(result, human_role)
    steps.extend(comparisons)
    opponent = other_player(human_role)
    steps.append(
        ScoringStep(
            kind="select_round_score",
            player=None,
            line=None,
            details={
                "rank": selected_rank + 1,
                "human_score": result.round_scores[human_role],
                "opponent_score": result.round_scores[opponent],
            },
        )
    )
    steps.append(
        ScoringStep(
            kind="update_total",
            player=None,
            line=None,
            details={
                "human_previous": previous_totals[human_role],
                "human_round": result.round_scores[human_role],
                "human_total": previous_totals[human_role]
                + result.round_scores[human_role],
                "opponent_previous": previous_totals[opponent],
                "opponent_round": result.round_scores[opponent],
                "opponent_total": previous_totals[opponent]
                + result.round_scores[opponent],
            },
        )
    )
    return tuple(steps)


def round_record(
    result: EngineRoundResult,
    human_role: EnginePlayer,
    previous_totals: PlayerValues[int],
) -> RoundRecord:
    """Project one completed result and its deterministic scoring presentation."""

    line_order = (result.dealer, other_player(result.dealer))
    return RoundRecord(
        round_number=result.round_number,
        dealer=result.dealer.value,
        coffin=result.coffin,
        moves=tuple(played_move(move) for move in result.moves),
        line_scores=tuple(
            _api_line(line)
            for player in line_order
            for line in result.line_scores[player]
        ),
        scoring_sequence=_scoring_sequence(result, human_role, previous_totals),
        round_scores=player_score(result.round_scores, human_role),
    )

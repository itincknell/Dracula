"""Derive narration prompts exclusively from authoritative public game facts.

This module owns cue eligibility and the exact fact vocabulary supplied to the
narration provider. It may inspect a reconstructed private game internally, but
its output is an allowlisted mapping containing no seed, command history, hidden
cards, policy data, or engine objects.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Mapping

from dracula.api.stateless.contracts import NarrationCueType
from dracula.api.stateless.replay import ReplayedGame
from dracula.api.stateless.service import StatelessReplayError
from dracula.game.cards import Suit, card_by_id
from dracula.game.engine import (
    EnginePlayer,
    EngineRoundResult,
    EngineStatus,
    LineScore,
    MultiplierReason,
    derive_game_outcome,
    other_player,
)
from dracula.game.scoring import round_score_deciding_rank


@dataclass(frozen=True, slots=True)
class GroundedNarrationCue:
    """Allowlisted public facts plus the replay-stable phrasing choice.

    ``wording_variant`` changes only whether the prompt describes the leader as
    winning or the trailer as losing. It never changes the underlying facts.
    """

    cue_type: NarrationCueType
    facts: Mapping[str, object]
    wording_variant: int = 0


def _wording_variant(game: ReplayedGame, cue_type: NarrationCueType) -> int:
    """Choose one replay-stable summary perspective from the visible game seed."""

    choice = random.Random(
        f"{game.policy_game_key}|{game.state.round_number}|{cue_type}"
    )
    return choice.randrange(2)


def _round_winner_player(
    result: EngineRoundResult,
    human_role: EnginePlayer,
) -> EnginePlayer | None:
    """Return the winning engine role, or ``None`` for an exact round tie."""

    opponent = other_player(human_role)
    human_score = result.round_scores[human_role]
    opponent_score = result.round_scores[opponent]
    if human_score == opponent_score:
        return None
    return human_role if human_score > opponent_score else opponent


def _result_label(
    winner: EnginePlayer | None,
    human_role: EnginePlayer,
) -> str:
    """Translate an engine winner into the narrator's human/Dracula vocabulary."""

    if winner is None:
        return "tie"
    return "human" if winner is human_role else "dracula"


def _best_line(lines: tuple[LineScore, LineScore, LineScore]) -> LineScore:
    # Earlier line indexes break exact score and multiplier ties consistently
    # with the order used by the scoring presentation.
    return max(
        lines,
        key=lambda line: (line.total, line.multiplier, -line.line_index),
    )


def _winning_combination(
    result: EngineRoundResult,
    winner: EnginePlayer,
) -> str:
    """Describe only the round winner's strongest rules-derived line."""

    line = _best_line(result.line_scores[winner])
    cards = tuple(card_by_id(card_id) for card_id in line.cards)
    if line.multiplier_reason is MultiplierReason.SAME_SUIT:
        suit = cards[0].suit
        if suit is None:
            raise ValueError("same-suit narration line has no suit")
        return f"three {suit.name.title()} for a 5x multiplier"
    elif line.multiplier_reason is MultiplierReason.SAME_COLOR:
        color = cards[0].color
        if color is None:
            raise ValueError("same-color narration line has no color")
        return f"three {color.value} cards for a 3x multiplier"
    elif line.multiplier_reason is MultiplierReason.SUIT_PAIR:
        suit = next(
            candidate
            for candidate in Suit
            if sum(card.suit is candidate for card in cards) >= 2
        )
        return f"two {suit.name.title()} for a 2x multiplier"
    return "no multiplier"


def _round_tie_break(result: EngineRoundResult) -> str | None:
    """Describe which ranked line decided the round after earlier ties."""

    queen_totals = tuple(line.total for line in result.line_scores.queen)
    king_totals = tuple(line.total for line in result.line_scores.king)
    deciding_rank = round_score_deciding_rank(queen_totals, king_totals)
    if deciding_rank == 1:
        return (
            "the players' best lines tied, and their second-best lines "
            "decided the round"
        )
    if deciding_rank == 2:
        return (
            "the players' best and second-best lines tied, and their third-best "
            "lines decided the round"
        )
    return None


def _actor_name(player: EnginePlayer, human_role: EnginePlayer) -> str:
    """Name an engine role without exposing Queen or King to the narrator."""

    return "the human" if player is human_role else "Dracula"


def _dracula_attitude(game: ReplayedGame) -> str:
    """Name the tone matching the portrait displayed with this cue."""

    state = game.state
    dracula = other_player(game.human_role)
    difference = state.total_scores[dracula] - state.total_scores[game.human_role]
    if state.status is EngineStatus.GAME_COMPLETE and difference < 0:
        return "angrier"
    # Round-transition dialogue introduces the coming round and changes the
    # portrait at the same moment its first character appears.
    mood_round = (
        state.round_number + 1
        if state.status is EngineStatus.ROUND_COMPLETE
        else state.round_number
    )
    if mood_round >= 4 and difference <= -50:
        return "angrier"
    if mood_round >= 4 and difference > 0:
        return "amused"
    if difference < 0:
        return "angry"
    return "imperious"


def _vampires_played(
    result: EngineRoundResult,
    human_role: EnginePlayer,
) -> tuple[str, ...]:
    """Describe each player who placed at least one Vampire this round."""

    descriptions: list[str] = []
    for player in (human_role, other_player(human_role)):
        count = sum(
            move.card_id in {"V1", "V2"}
            for move in result.moves
            if move.player is player
        )
        if count:
            noun = "vampire" if count == 1 else "vampires"
            descriptions.append(
                f"{_actor_name(player, human_role)} played {count} {noun}"
            )
    return tuple(descriptions)


def _score_movement(
    game: ReplayedGame,
    result: EngineRoundResult,
) -> str:
    """Describe the lead change using totals immediately before and after a round."""

    human = game.human_role
    dracula = other_player(human)
    prior_human = game.state.total_scores[human] - result.round_scores[human]
    prior_dracula = game.state.total_scores[dracula] - result.round_scores[dracula]
    prior_difference = prior_dracula - prior_human
    current_difference = (
        game.state.total_scores[dracula] - game.state.total_scores[human]
    )

    if current_difference == 0:
        if prior_difference == 0:
            return "Dracula and the human remain tied"
        catcher = dracula if prior_difference < 0 else human
        return f"{_actor_name(catcher, human)} ties the game"

    leader = dracula if current_difference > 0 else human
    trailer = other_player(leader)
    if prior_difference == 0 or prior_difference * current_difference < 0:
        return f"{_actor_name(leader, human)} takes the lead"
    if abs(current_difference) > abs(prior_difference):
        return (
            f"{_actor_name(leader, human)} extends the lead over "
            f"{_actor_name(trailer, human)}"
        )
    if abs(current_difference) < abs(prior_difference):
        return (
            f"{_actor_name(trailer, human)} catches up with "
            f"{_actor_name(leader, human)} but still trails"
        )
    return f"{_actor_name(leader, human)} keeps the lead"


def _current_leader(game: ReplayedGame) -> str:
    """Return the public leader after the currently reconstructed round."""

    human_score = game.state.total_scores[game.human_role]
    dracula_score = game.state.total_scores[other_player(game.human_role)]
    if human_score == dracula_score:
        return "tie"
    return "human" if human_score > dracula_score else "dracula"


def _opening_cue(game: ReplayedGame) -> GroundedNarrationCue:
    """Validate and construct the cue available at the first playing state."""

    state = game.state
    if (
        state.round_number != 1
        or state.completed_rounds
        or state.status is not EngineStatus.PLAYING
    ):
        raise StatelessReplayError(
            "ineligible_cue",
            "opening narration is available only at the start of a game",
            status_code=409,
        )
    return GroundedNarrationCue(
        "opening",
        {
            "dracula_attitude": "imperious",
            "dracula_role": other_player(game.human_role).value,
            "first_player": other_player(state.dealer).value,
            "human_role": game.human_role.value,
        },
        _wording_variant(game, "opening"),
    )


def _round_transition_cue(game: ReplayedGame) -> GroundedNarrationCue:
    """Construct a completed-round cue without exposing the next deal."""

    state = game.state
    # The pending result remains fixed until advance_round, so narration may be
    # generated during the scoring animation without racing a gameplay write.
    if (
        state.status is not EngineStatus.ROUND_COMPLETE
        or state.pending_round_result is None
        or not 1 <= state.round_number <= 5
    ):
        raise StatelessReplayError(
            "ineligible_cue",
            "round-transition narration requires a completed round from 1 to 5",
            status_code=409,
        )
    result = state.pending_round_result
    winner = _round_winner_player(result, game.human_role)
    facts: dict[str, object] = {
        "dracula_attitude": _dracula_attitude(game),
        "leader": _current_leader(game),
        "round": state.round_number,
        "round_result": _result_label(winner, game.human_role),
        "score_movement": _score_movement(game, result),
    }
    vampires_played = _vampires_played(result, game.human_role)
    if vampires_played:
        facts["vampires_played"] = vampires_played
    if winner is not None:
        facts["winning_combination"] = _winning_combination(result, winner)
        tie_break = _round_tie_break(result)
        if tie_break is not None:
            facts["round_tie_break"] = tie_break
    return GroundedNarrationCue(
        "round_transition",
        facts,
        _wording_variant(game, "round_transition"),
    )


def _final_result_cue(game: ReplayedGame) -> GroundedNarrationCue:
    """Construct the sole narration cue available after round six."""

    state = game.state
    if state.status is not EngineStatus.GAME_COMPLETE:
        raise StatelessReplayError(
            "ineligible_cue",
            "final-result narration requires a completed six-round game",
            status_code=409,
        )
    outcome = derive_game_outcome(state)
    return GroundedNarrationCue(
        "final_result",
        {
            "dracula_attitude": _dracula_attitude(game),
            "final_result": _result_label(outcome.winner, game.human_role),
            "tie_break": outcome.reason.value,
        },
        _wording_variant(game, "final_result"),
    )


def derive_grounded_narration_cue(
    game: ReplayedGame,
    cue_type: NarrationCueType,
) -> GroundedNarrationCue:
    """Validate cue timing and expose only its required public facts."""

    if cue_type == "opening":
        return _opening_cue(game)
    if cue_type == "round_transition":
        return _round_transition_cue(game)
    return _final_result_cue(game)

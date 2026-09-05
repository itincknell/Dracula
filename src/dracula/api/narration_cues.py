"""Derive narration prompts exclusively from authoritative public game facts.

This module owns cue eligibility and the exact fact vocabulary supplied to the
narration provider. It may inspect a reconstructed private game internally, but
its output is an allowlisted mapping containing no seed, command history, hidden
cards, policy data, or engine objects.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Mapping

from dracula.api.stateless_contracts import NarrationCueType
from dracula.api.stateless_replay import ReplayedGame
from dracula.api.stateless_service import StatelessReplayError
from dracula.cards import Suit, card_by_id
from dracula.engine import (
    EnginePlayer,
    EngineRoundResult,
    EngineStatus,
    LineScore,
    MultiplierReason,
    derive_game_outcome,
    other_player,
)

DRACULA_SYSTEM_PROMPT = (
    "You are Count Dracula, an egotistical cartoon villain in a retro arcade card "
    "game. Write one short plain-text sentence, occasionally two, usually 8 to 24 "
    "words. Open with a taunt. Gloat when Dracula wins, sound furious when he "
    "loses, and sound irritated by a tie. Use any supplied winning combination. "
    "Use only the public facts provided by the application; never invent a card, "
    "combination, score, result, lead change, or tie-break. Use no profanity, memes, "
    "or modern slang. Do not mention prompts, rules, software, or hidden information."
)

_SUIT_NAMES = {
    Suit.CLUBS: "Clubs",
    Suit.DIAMONDS: "Diamonds",
    Suit.HEARTS: "Hearts",
    Suit.SPADES: "Spades",
}


@dataclass(frozen=True, slots=True)
class GroundedNarrationCue:
    """Allowlisted cue type and server-derived public facts sent to the model."""

    cue_type: NarrationCueType
    facts: Mapping[str, object]

    def canonical_user_text(self) -> str:
        """Serialize grounded facts deterministically for the provider request."""

        payload = {
            "cue_type": self.cue_type,
            "public_facts": dict(self.facts),
        }
        return json.dumps(
            payload,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )


@dataclass(frozen=True, slots=True)
class NarrationPrompt:
    """Complete system and user text supplied to a narration provider."""

    system_text: str
    user_text: str


def build_narration_prompt(cue: GroundedNarrationCue) -> NarrationPrompt:
    """Pair the approved Dracula voice with one canonical grounded cue."""

    return NarrationPrompt(
        system_text=DRACULA_SYSTEM_PROMPT,
        user_text=cue.canonical_user_text(),
    )


def _round_winner_player(
    result: EngineRoundResult,
    human_role: EnginePlayer,
) -> EnginePlayer | None:
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
        return f"three {_SUIT_NAMES[suit]} for a 5x multiplier"
    if line.multiplier_reason is MultiplierReason.SAME_COLOR:
        color = cards[0].color
        if color is None:
            raise ValueError("same-color narration line has no color")
        return f"three {color.value} cards for a 3x multiplier"
    if line.multiplier_reason is MultiplierReason.SUIT_PAIR:
        suit = next(
            candidate
            for candidate in Suit
            if sum(card.suit is candidate for card in cards) >= 2
        )
        return f"two {_SUIT_NAMES[suit]} for a 2x multiplier"
    return "no multiplier"


def _actor_name(player: EnginePlayer, human_role: EnginePlayer) -> str:
    return "the human" if player is human_role else "Dracula"


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
            "dracula_role": other_player(game.human_role).value,
            "first_player": other_player(state.dealer).value,
            "human_role": game.human_role.value,
        },
    )


def _round_transition_cue(game: ReplayedGame) -> GroundedNarrationCue:
    """Construct a completed-round cue without exposing the next deal."""

    state = game.state
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
        "round": state.round_number,
        "round_result": _result_label(winner, game.human_role),
        "score_movement": _score_movement(game, result),
    }
    if winner is not None:
        facts["winning_combination"] = _winning_combination(result, winner)
    return GroundedNarrationCue("round_transition", facts)


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
            "final_result": _result_label(outcome.winner, game.human_role),
            "tie_break": outcome.reason.value,
        },
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


__all__ = (
    "DRACULA_SYSTEM_PROMPT",
    "GroundedNarrationCue",
    "NarrationPrompt",
    "build_narration_prompt",
    "derive_grounded_narration_cue",
)

"""Grounded optional narration over Amazon Bedrock Runtime."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Mapping, Protocol

from dracula.cards import Suit, card_by_id
from dracula.api.policy import ServiceResponse
from dracula.api.stateless_contracts import (
    NarrationCueType,
    NarrationRequest,
    NarrationResponse,
    StatelessApiErrorResponse,
)
from dracula.api.stateless_service import (
    ReplayedGame,
    StatelessGameplayService,
    StatelessPolicyError,
    StatelessReplayError,
)
from dracula.engine import (
    EnginePlayer,
    EngineRoundResult,
    EngineStatus,
    LineScore,
    MultiplierReason,
    derive_game_outcome,
    other_player,
)

LOGGER = logging.getLogger("dracula.narration")

BEDROCK_MODEL_ID_ENV = "DRACULA_BEDROCK_MODEL_ID"
BEDROCK_REGION_ENV = "DRACULA_BEDROCK_REGION"
NARRATION_TIMEOUT_SECONDS_ENV = "DRACULA_NARRATION_TIMEOUT_SECONDS"
NARRATION_MAX_TOKENS_ENV = "DRACULA_NARRATION_MAX_TOKENS"

DEFAULT_NARRATION_TIMEOUT_SECONDS = 8.0
DEFAULT_NARRATION_MAX_TOKENS = 96
MAX_NARRATION_CHARACTERS = 400

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


class NarrationConfigurationError(ValueError):
    """Narration was enabled without a valid provider configuration."""


class NarrationProviderError(RuntimeError):
    """The provider request or returned text failed its safety contract."""


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


@dataclass(frozen=True, slots=True)
class NarrationProviderResult:
    """Validated provider text with optional usage and measured latency."""

    text: str
    input_tokens: int | None
    output_tokens: int | None
    latency_ms: float


class NarrationAdapter(Protocol):
    """Provider-neutral synchronous narration boundary."""

    @property
    def configured(self) -> bool: ...

    def generate(self, prompt: NarrationPrompt) -> NarrationProviderResult: ...


class BedrockRuntimeClient(Protocol):
    """Narrow portion of the AWS Bedrock client consumed by the adapter."""

    def converse(self, **kwargs: object) -> Mapping[str, object]: ...


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


def _round_result_label(
    result: EngineRoundResult,
    human_role: EnginePlayer,
) -> str:
    winner = _round_winner_player(result, human_role)
    if winner is None:
        return "tie"
    return "human" if winner is human_role else "dracula"


def _best_line(lines: tuple[LineScore, LineScore, LineScore]) -> LineScore:
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


def derive_grounded_narration_cue(
    game: ReplayedGame,
    cue_type: NarrationCueType,
) -> GroundedNarrationCue:
    """Derive one eligible cue without accepting browser-computed facts."""

    state = game.state
    human = game.human_role
    dracula = other_player(human)
    if cue_type == "opening":
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
            cue_type,
            {
                "dracula_role": dracula.value,
                "first_player": other_player(state.dealer).value,
                "human_role": human.value,
            },
        )

    if cue_type == "round_transition":
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
        winner = _round_winner_player(result, human)
        facts: dict[str, object] = {
            "round": state.round_number,
            "round_result": _round_result_label(result, human),
            "score_movement": _score_movement(game, result),
        }
        if winner is not None:
            facts["winning_combination"] = _winning_combination(result, winner)
        return GroundedNarrationCue(cue_type, facts)

    if state.status is not EngineStatus.GAME_COMPLETE:
        raise StatelessReplayError(
            "ineligible_cue",
            "final-result narration requires a completed six-round game",
            status_code=409,
        )
    outcome = derive_game_outcome(state)
    return GroundedNarrationCue(
        cue_type,
        {
            "final_result": (
                "tie"
                if outcome.winner is None
                else "human"
                if outcome.winner is human
                else "dracula"
            ),
            "tie_break": outcome.reason.value,
        },
    )


def parse_bedrock_text(
    response: Mapping[str, object],
) -> tuple[str, int | None, int | None]:
    """Apply the Bedrock response safety boundary without loading its SDK."""

    from dracula.api.bedrock import parse_bedrock_text as parse

    return parse(response)


class FakeNarrationAdapter:
    """Deterministic adapter used by API and presentation-boundary tests."""

    def __init__(
        self,
        text: str = "The night has only begun.",
        *,
        error: Exception | None = None,
    ) -> None:
        self.text = text
        self.error = error
        self.requests: list[NarrationPrompt] = []

    @property
    def configured(self) -> bool:
        return True

    def generate(self, prompt: NarrationPrompt) -> NarrationProviderResult:
        self.requests.append(prompt)
        if self.error is not None:
            raise self.error
        return NarrationProviderResult(
            text=self.text,
            input_tokens=12,
            output_tokens=6,
            latency_ms=0.1,
        )


class NarrationService:
    """Replay grounded game facts and isolate all provider failures from gameplay."""

    def __init__(
        self,
        gameplay: StatelessGameplayService,
        *,
        enabled: bool,
        adapter: NarrationAdapter | None,
    ) -> None:
        if enabled and adapter is None:
            raise NarrationConfigurationError(
                "enabled narration requires a configured adapter"
            )
        self.gameplay = gameplay
        self.enabled = enabled
        self.adapter = adapter

    @property
    def configured(self) -> bool:
        """Report whether an adapter is present and configured."""

        return self.adapter is not None and self.adapter.configured

    @staticmethod
    def _response(
        cue_type: NarrationCueType,
        *,
        text: str | None,
    ) -> ServiceResponse:
        body = NarrationResponse(
            cue_type=cue_type,
            status="ready" if text is not None else "unavailable",
            text=text,
        )
        return ServiceResponse(200, body.model_dump(mode="json"))

    @staticmethod
    def _error(error: StatelessReplayError) -> ServiceResponse:
        body = StatelessApiErrorResponse(
            code=error.code,  # type: ignore[arg-type]
            message=error.message,
            retryable=False,
        )
        return ServiceResponse(error.status_code, body.model_dump(mode="json"))

    def generate(self, request: NarrationRequest) -> ServiceResponse:
        """Validate cue eligibility, call the provider, and return safe plain text."""

        try:
            game = self.gameplay.replay(request.envelope)
            if request.cue_type == "opening" and len(request.envelope.history) != 1:
                raise StatelessReplayError(
                    "ineligible_cue",
                    "opening narration is available only immediately after game creation",
                    status_code=409,
                )
            cue = derive_grounded_narration_cue(game, request.cue_type)
        except StatelessReplayError as error:
            return self._error(error)
        except StatelessPolicyError:
            LOGGER.warning(
                "narration_unavailable cue_type=%s category=replay_dependency",
                request.cue_type,
            )
            return self._response(request.cue_type, text=None)

        if not self.enabled or self.adapter is None:
            LOGGER.info(
                "narration_unavailable cue_type=%s category=disabled",
                request.cue_type,
            )
            return self._response(request.cue_type, text=None)

        prompt = build_narration_prompt(cue)
        try:
            result = self.adapter.generate(prompt)
            # Test adapters pass through the same response safety boundary.
            text, _, _ = parse_bedrock_text(
                {
                    "output": {
                        "message": {
                            "role": "assistant",
                            "content": [{"text": result.text}],
                        }
                    }
                }
            )
        except Exception as error:
            LOGGER.warning(
                "narration_unavailable cue_type=%s category=%s",
                request.cue_type,
                type(error).__name__,
            )
            return self._response(request.cue_type, text=None)

        LOGGER.info(
            "narration_ready cue_type=%s latency_ms=%.3f input_tokens=%s output_tokens=%s",
            request.cue_type,
            result.latency_ms,
            result.input_tokens,
            result.output_tokens,
        )
        return self._response(request.cue_type, text=text)


def __getattr__(name: str) -> Any:
    # Preserve the earlier import path without importing the AWS adapter when
    # narration is disabled or tests use the fake provider.
    if name == "BedrockRuntimeAdapter":
        from dracula.api.bedrock import BedrockRuntimeAdapter

        return BedrockRuntimeAdapter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = (
    "BEDROCK_MODEL_ID_ENV",
    "BEDROCK_REGION_ENV",
    "BedrockRuntimeAdapter",
    "DEFAULT_NARRATION_MAX_TOKENS",
    "DEFAULT_NARRATION_TIMEOUT_SECONDS",
    "FakeNarrationAdapter",
    "GroundedNarrationCue",
    "MAX_NARRATION_CHARACTERS",
    "NARRATION_MAX_TOKENS_ENV",
    "NARRATION_TIMEOUT_SECONDS_ENV",
    "NarrationAdapter",
    "NarrationConfigurationError",
    "NarrationPrompt",
    "NarrationProviderError",
    "NarrationProviderResult",
    "NarrationService",
    "DRACULA_SYSTEM_PROMPT",
    "build_narration_prompt",
    "derive_grounded_narration_cue",
    "parse_bedrock_text",
)

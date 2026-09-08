"""Coordinate optional narration without entering the gameplay transaction.

The service replays the submitted game, delegates public-fact construction to
``cues``, and calls one provider-neutral adapter. Eligibility errors
are public request failures; provider failures produce an unavailable response
and never alter an accepted move.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol

from dracula.api.contracts import ServiceResponse
from dracula.api.narration.cues import (
    GroundedNarrationCue,
    derive_grounded_narration_cue,
)
from dracula.api.narration.prompt import NarrationPrompt, build_narration_prompt
from dracula.api.stateless.contracts import (
    NarrationCueType,
    NarrationRequest,
    NarrationResponse,
)
from dracula.api.stateless.service import (
    StatelessGameplayService,
    StatelessPolicyError,
    StatelessReplayError,
    stateless_replay_error_response,
)

LOGGER = logging.getLogger("uvicorn.error")

class NarrationProviderError(RuntimeError):
    """The provider request or returned text failed its safety contract."""


@dataclass(frozen=True, slots=True)
class NarrationProviderResult:
    """Provider text with optional usage and measured latency."""

    text: str
    input_tokens: int | None
    output_tokens: int | None
    latency_ms: float


class NarrationAdapter(Protocol):
    """Provider-neutral synchronous narration boundary.

    A ``Protocol`` requires these attributes without forcing adapters to inherit
    from this class. Both the Bedrock adapter and local preview adapter can
    therefore be supplied to ``NarrationService``.
    """

    def generate(self, prompt: NarrationPrompt) -> NarrationProviderResult: ...


def validate_narration_text(text: object) -> str:
    """Return bounded plain text suitable for the public narration response."""

    if not isinstance(text, str):
        raise NarrationProviderError("narration provider returned non-text output")
    cleaned = text.strip()
    has_control_character = any(
        ord(character) < 32 and character not in "\n\t" for character in cleaned
    )
    if not cleaned or has_control_character:
        raise NarrationProviderError("narration provider returned invalid text")
    return cleaned


class NarrationService:
    """Replay grounded game facts and isolate all provider failures from gameplay."""

    def __init__(
        self,
        gameplay: StatelessGameplayService,
        *,
        adapter: NarrationAdapter | None,
    ) -> None:
        self.gameplay = gameplay
        self.adapter = adapter

    @property
    def enabled(self) -> bool:
        """Return whether this service has a provider to call."""

        return self.adapter is not None

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

    def _ground_cue(self, request: NarrationRequest) -> GroundedNarrationCue:
        """Replay the submitted identity and derive its eligible public facts."""

        game = self.gameplay.replay(request.envelope)
        if request.cue_type == "opening" and len(request.envelope.history) != 1:
            raise StatelessReplayError(
                "ineligible_cue",
                "opening narration is available only immediately after game creation",
                status_code=409,
            )
        return derive_grounded_narration_cue(game, request.cue_type)

    @staticmethod
    def _log_unavailable(
        cue_type: NarrationCueType,
        category: str,
    ) -> None:
        # Provider and replay failures are intentionally detached from the move
        # transaction: narration can disappear, but accepted play cannot change.
        LOGGER.warning(
            "narration_unavailable cue_type=%s category=%s",
            cue_type,
            category,
        )

    def _unavailable(
        self,
        cue_type: NarrationCueType,
        category: str,
    ) -> ServiceResponse:
        self._log_unavailable(cue_type, category)
        return self._response(cue_type, text=None)

    def _provider_text(
        self,
        cue_type: NarrationCueType,
        cue: GroundedNarrationCue,
        adapter: NarrationAdapter,
    ) -> str | None:
        """Call the provider and contain all provider-side failures."""

        prompt = build_narration_prompt(cue)
        try:
            result = adapter.generate(prompt)
            text = validate_narration_text(result.text)
        except Exception as error:
            self._log_unavailable(cue_type, type(error).__name__)
            return None

        LOGGER.info(
            "narration_ready cue_type=%s latency_ms=%.3f input_tokens=%s "
            "output_tokens=%s text=%r",
            cue_type,
            result.latency_ms,
            result.input_tokens,
            result.output_tokens,
            text,
        )
        return text

    def generate(self, request: NarrationRequest) -> ServiceResponse:
        """Validate cue eligibility, call the provider, and return safe plain text."""

        adapter = self.adapter
        if adapter is None:
            LOGGER.info(
                "narration_unavailable cue_type=%s category=disabled",
                request.cue_type,
            )
            return self._response(request.cue_type, text=None)

        # The cue is tied to the submitted envelope. If the browser advances
        # while generation is in flight, it can discard this separate response.
        try:
            cue = self._ground_cue(request)
        except StatelessReplayError as error:
            return stateless_replay_error_response(error)
        except StatelessPolicyError:
            return self._unavailable(request.cue_type, "replay_dependency")
        text = self._provider_text(request.cue_type, cue, adapter)
        return self._response(request.cue_type, text=text)

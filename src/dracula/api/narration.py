"""Coordinate optional narration without entering the gameplay transaction.

The service replays the submitted game, delegates public-fact construction to
``narration_cues``, and calls one provider-neutral adapter. Eligibility errors
are public request failures; provider failures produce an unavailable response
and never alter an accepted move.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol, cast

from dracula.api.narration_cues import (
    DRACULA_SYSTEM_PROMPT,
    GroundedNarrationCue,
    NarrationPrompt,
    build_narration_prompt,
    derive_grounded_narration_cue,
)
from dracula.api.policy import ServiceResponse
from dracula.api.stateless_contracts import (
    NarrationCueType,
    NarrationRequest,
    NarrationResponse,
)
from dracula.api.stateless_service import (
    StatelessGameplayService,
    StatelessPolicyError,
    StatelessReplayError,
    stateless_replay_error_response,
)

LOGGER = logging.getLogger("dracula.narration")

BEDROCK_MODEL_ID_ENV = "DRACULA_BEDROCK_MODEL_ID"
BEDROCK_REGION_ENV = "DRACULA_BEDROCK_REGION"
NARRATION_TIMEOUT_SECONDS_ENV = "DRACULA_NARRATION_TIMEOUT_SECONDS"
NARRATION_MAX_TOKENS_ENV = "DRACULA_NARRATION_MAX_TOKENS"

DEFAULT_NARRATION_TIMEOUT_SECONDS = 8.0
DEFAULT_NARRATION_MAX_TOKENS = 96
MAX_NARRATION_CHARACTERS = 400


class NarrationConfigurationError(ValueError):
    """Narration was enabled without a valid provider configuration."""


class NarrationProviderError(RuntimeError):
    """The provider request or returned text failed its safety contract."""


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


def validate_narration_text(text: object) -> str:
    """Return bounded plain text suitable for the public narration response."""

    if not isinstance(text, str):
        raise NarrationProviderError("narration provider returned non-text output")
    cleaned = text.strip()
    if (
        not cleaned
        or len(cleaned) > MAX_NARRATION_CHARACTERS
        or any(ord(character) < 32 and character not in "\n\t" for character in cleaned)
    ):
        raise NarrationProviderError("narration provider returned invalid text")
    return cleaned


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
            return stateless_replay_error_response(error)
        except StatelessPolicyError:
            LOGGER.warning(
                "narration_unavailable cue_type=%s category=replay_dependency",
                request.cue_type,
            )
            return self._response(request.cue_type, text=None)

        if not self.enabled:
            LOGGER.info(
                "narration_unavailable cue_type=%s category=disabled",
                request.cue_type,
            )
            return self._response(request.cue_type, text=None)

        prompt = build_narration_prompt(cue)
        try:
            adapter = cast(NarrationAdapter, self.adapter)
            result = adapter.generate(prompt)
            text = validate_narration_text(result.text)
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


__all__ = (
    "BEDROCK_MODEL_ID_ENV",
    "BEDROCK_REGION_ENV",
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
    "validate_narration_text",
)

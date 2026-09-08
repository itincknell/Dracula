"""Send one narration prompt through Amazon Bedrock Runtime.

The AWS SDK exposes Bedrock's ``Converse`` operation as ``client.converse``.
This module translates the project's two-part narration prompt into that AWS
request, limits generated tokens, and accepts only one plain-text assistant
response. Provider errors become ``NarrationProviderError`` and remain separate
from the already-completed gameplay request.
"""

from __future__ import annotations

import math
import os
import time
from typing import Mapping, Protocol

from dracula.api.narration.service import (
    NarrationProviderError,
    NarrationProviderResult,
)
from dracula.api.narration.prompt import NarrationPrompt

BEDROCK_MODEL_ID_ENV = "DRACULA_BEDROCK_MODEL_ID"
BEDROCK_REGION_ENV = "DRACULA_BEDROCK_REGION"
NARRATION_TIMEOUT_SECONDS_ENV = "DRACULA_NARRATION_TIMEOUT_SECONDS"
NARRATION_MAX_TOKENS_ENV = "DRACULA_NARRATION_MAX_TOKENS"

DEFAULT_NARRATION_TIMEOUT_SECONDS = 8.0
DEFAULT_NARRATION_MAX_TOKENS = 96
NARRATION_TEMPERATURE = 0.5


class NarrationConfigurationError(ValueError):
    """Bedrock configuration cannot create a usable narration adapter."""


class BedrockRuntimeClient(Protocol):
    """The single AWS client method required by this module.

    ``Protocol`` describes behavior rather than a concrete base class. A boto3
    client and the test fake both satisfy it by providing ``converse``.
    ``**kwargs`` means the method receives the named AWS request parameters.
    """

    def converse(self, **kwargs: object) -> Mapping[str, object]: ...


def _assistant_text(response: Mapping[str, object]) -> str:
    """Extract one plain-text answer from Bedrock's nested response mapping.

    A successful Converse response is shaped as
    ``output.message.content = [{"text": "..."}]``. Rejecting additional
    blocks prevents tool calls or unfamiliar provider content from reaching the
    browser as if they were narration.
    """

    try:
        output = response["output"]
        if not isinstance(output, Mapping):
            raise TypeError
        message = output["message"]
        if not isinstance(message, Mapping) or message.get("role") != "assistant":
            raise TypeError
        content = message["content"]
        if not isinstance(content, list) or len(content) != 1:
            raise TypeError
        block = content[0]
        if not isinstance(block, Mapping) or set(block) != {"text"}:
            raise TypeError
        text = block["text"]
        if not isinstance(text, str):
            raise TypeError
    except (KeyError, TypeError) as error:
        # Collapse every structural mismatch into one provider-level failure;
        # raw provider content is not included in the public error.
        raise NarrationProviderError("Bedrock returned malformed narration") from error
    return text


def _usage_counts(response: Mapping[str, object]) -> tuple[int | None, int | None]:
    """Read optional non-negative token counts without trusting their presence."""

    usage = response.get("usage")
    input_tokens: int | None = None
    output_tokens: int | None = None
    if isinstance(usage, Mapping):
        # Usage is diagnostic metadata, not required for a valid narration.
        # ``type(...) is int`` excludes Boolean values, which Python otherwise
        # treats as integers.
        raw_input = usage.get("inputTokens")
        raw_output = usage.get("outputTokens")
        if type(raw_input) is int and raw_input >= 0:
            input_tokens = raw_input
        if type(raw_output) is int and raw_output >= 0:
            output_tokens = raw_output
    return input_tokens, output_tokens


def _parse_bedrock_response(
    response: Mapping[str, object],
) -> tuple[str, int | None, int | None]:
    """Extract one assistant text block and optional token counts."""

    text = _assistant_text(response)
    input_tokens, output_tokens = _usage_counts(response)
    return text, input_tokens, output_tokens


def _conversation_messages(prompt: NarrationPrompt) -> list[dict[str, object]]:
    """Place examples in prior turns and leave the live cue unanswered."""

    messages: list[dict[str, object]] = []
    for example_user, example_assistant in prompt.example_turns:
        messages.append(
            {"role": "user", "content": [{"text": example_user}]}
        )
        messages.append(
            {"role": "assistant", "content": [{"text": example_assistant}]}
        )
    messages.append(
        {"role": "user", "content": [{"text": prompt.user_text}]}
    )
    return messages


class BedrockRuntimeAdapter:
    """Validated synchronous adapter around Bedrock Runtime Converse.

    Lambda invokes the synchronous FastAPI route in a worker thread, so this
    adapter uses boto3's ordinary blocking client rather than introducing a
    second asynchronous AWS library.
    """

    def __init__(
        self,
        client: BedrockRuntimeClient,
        *,
        model_id: str,
        max_tokens: int = DEFAULT_NARRATION_MAX_TOKENS,
    ) -> None:
        if not model_id.strip():
            raise NarrationConfigurationError("Bedrock model ID must be nonempty")
        if type(max_tokens) is not int or max_tokens < 1:
            raise NarrationConfigurationError("narration max tokens must be positive")
        self.client = client
        self.model_id = model_id
        self.max_tokens = max_tokens

    @classmethod
    def from_environment(cls) -> BedrockRuntimeAdapter:
        """Create a Bedrock client from deployment environment variables.

        Boto3 obtains AWS credentials from Lambda's execution role. They are
        never read from application configuration or passed through this API.
        """

        model_id = os.getenv(BEDROCK_MODEL_ID_ENV, "").strip()
        region = os.getenv(BEDROCK_REGION_ENV, "").strip()
        if not model_id:
            raise NarrationConfigurationError(
                f"{BEDROCK_MODEL_ID_ENV} must select one Bedrock model"
            )
        if not region:
            raise NarrationConfigurationError(
                f"{BEDROCK_REGION_ENV} must select one AWS region"
            )
        timeout = _positive_float_environment(
            NARRATION_TIMEOUT_SECONDS_ENV, DEFAULT_NARRATION_TIMEOUT_SECONDS
        )
        max_tokens = _positive_int_environment(
            NARRATION_MAX_TOKENS_ENV, DEFAULT_NARRATION_MAX_TOKENS
        )
        try:
            # These packages exist in the Lambda image but are optional during
            # policy-only local imports, so importing them is deferred until an
            # enabled adapter is actually constructed.
            import boto3
            from botocore.config import Config
        except ImportError as error:
            raise NarrationConfigurationError(
                "Bedrock narration requires boto3 and botocore in the runtime image"
            ) from error
        client = boto3.client(
            "bedrock-runtime",
            region_name=region,
            config=Config(
                # Connection setup is capped separately; response generation
                # may use the full configured read timeout.
                connect_timeout=min(timeout, 2.0),
                read_timeout=timeout,
                # Narration is optional. SDK retries could outlive the scoring
                # animation, so one network attempt is made per cue request.
                retries={"mode": "standard", "total_max_attempts": 1},
            ),
        )
        return cls(client, model_id=model_id, max_tokens=max_tokens)

    def generate(self, prompt: NarrationPrompt) -> NarrationProviderResult:
        """Issue one Converse request and return provider text plus diagnostics."""

        started = time.perf_counter()
        try:
            # The camelCase names and nested message structure are Bedrock's
            # public Converse API fields, passed through unchanged by boto3.
            response = self.client.converse(
                modelId=self.model_id,
                system=[{"text": prompt.system_text}],
                messages=_conversation_messages(prompt),
                # Modest variation keeps repeated score patterns from producing
                # identical taunts; the token cap still bounds length and cost.
                inferenceConfig={
                    "maxTokens": self.max_tokens,
                    "temperature": NARRATION_TEMPERATURE,
                },
            )
        except Exception as error:
            # Boto3 exposes several service and networking exception classes.
            # The rest of the application needs only one failure category and
            # never uses narration failure to roll back gameplay.
            raise NarrationProviderError("Bedrock narration request failed") from error
        text, input_tokens, output_tokens = _parse_bedrock_response(response)
        return NarrationProviderResult(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=(time.perf_counter() - started) * 1000.0,
        )


def _positive_float_environment(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError as error:
        raise NarrationConfigurationError(f"{name} must be positive") from error
    if not math.isfinite(value) or value <= 0:
        raise NarrationConfigurationError(f"{name} must be positive")
    return value


def _positive_int_environment(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as error:
        raise NarrationConfigurationError(f"{name} must be positive") from error
    if value < 1:
        raise NarrationConfigurationError(f"{name} must be positive")
    return value

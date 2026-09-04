"""Amazon Bedrock Runtime transport for grounded Dracula narration."""

from __future__ import annotations

import math
import os
import time
from typing import Mapping

from dracula.api.narration import (
    BEDROCK_MODEL_ID_ENV,
    BEDROCK_REGION_ENV,
    DEFAULT_NARRATION_MAX_TOKENS,
    DEFAULT_NARRATION_TIMEOUT_SECONDS,
    MAX_NARRATION_CHARACTERS,
    NARRATION_MAX_TOKENS_ENV,
    NARRATION_TIMEOUT_SECONDS_ENV,
    BedrockRuntimeClient,
    NarrationConfigurationError,
    NarrationPrompt,
    NarrationProviderError,
    NarrationProviderResult,
)


def parse_bedrock_text(
    response: Mapping[str, object],
) -> tuple[str, int | None, int | None]:
    """Accept only one bounded assistant text block from Converse."""

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
        raise NarrationProviderError("Bedrock returned malformed narration") from error

    cleaned = text.strip()
    if (
        not cleaned
        or len(cleaned) > MAX_NARRATION_CHARACTERS
        or any(ord(character) < 32 and character not in "\n\t" for character in cleaned)
    ):
        raise NarrationProviderError("Bedrock returned invalid narration text")

    usage = response.get("usage")
    input_tokens: int | None = None
    output_tokens: int | None = None
    if isinstance(usage, Mapping):
        raw_input = usage.get("inputTokens")
        raw_output = usage.get("outputTokens")
        if type(raw_input) is int and raw_input >= 0:
            input_tokens = raw_input
        if type(raw_output) is int and raw_output >= 0:
            output_tokens = raw_output
    return cleaned, input_tokens, output_tokens


class BedrockRuntimeAdapter:
    """Small synchronous wrapper around Bedrock Runtime Converse."""

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

    @property
    def configured(self) -> bool:
        """Return true for an adapter whose constructor validated all settings."""

        return True

    @classmethod
    def from_environment(cls) -> BedrockRuntimeAdapter:
        """Create a no-retry Bedrock client from validated runtime settings."""

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
                connect_timeout=min(timeout, 2.0),
                read_timeout=timeout,
                retries={"mode": "standard", "total_max_attempts": 1},
            ),
        )
        return cls(client, model_id=model_id, max_tokens=max_tokens)

    def generate(self, prompt: NarrationPrompt) -> NarrationProviderResult:
        """Issue one deterministic Converse request and validate its response."""

        started = time.perf_counter()
        try:
            response = self.client.converse(
                modelId=self.model_id,
                system=[{"text": prompt.system_text}],
                messages=[
                    {"role": "user", "content": [{"text": prompt.user_text}]}
                ],
                inferenceConfig={"maxTokens": self.max_tokens, "temperature": 0.0},
            )
        except Exception as error:
            raise NarrationProviderError("Bedrock narration request failed") from error
        text, input_tokens, output_tokens = parse_bedrock_text(response)
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

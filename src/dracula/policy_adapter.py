"""Deployment-independent policy inference and action-selection contracts."""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from typing import Protocol

from dracula.randomness import derive_seed

POLICY_INFERENCE_CONTRACT_VERSION = "policy-inference-v1"
POLICY_ARCHIVE_FORMAT_VERSION = "dracula-policy-archive-v1"
HIDDEN_STATE_SCHEMA_VERSION = "dracula-hidden-state-v1"
HIDDEN_STATE_FLOATS = 128
HIDDEN_STATE_BYTES = HIDDEN_STATE_FLOATS * 4
SERVING_SAMPLING_NAMESPACE = "dracula-serving-action-sampling-v1"


class PolicyContractError(ValueError):
    """Policy artifact, request, response, or profile violates its contract."""


@dataclass(frozen=True, slots=True)
class PolicyArtifactMetadata:
    artifact_id: str
    artifact_sha256: str
    policy_id: str
    policy_version: str
    architecture_version: str
    observation_schema_version: str
    action_schema_version: str
    hidden_state_schema_version: str
    parameter_count: int


@dataclass(frozen=True, slots=True)
class PolicyInferenceRequest:
    contract_version: str
    artifact_id: str
    observation: tuple[bool, ...]
    legal_mask: tuple[tuple[bool, ...], ...]
    hidden_state: bytes

    def __post_init__(self) -> None:
        if self.contract_version != POLICY_INFERENCE_CONTRACT_VERSION:
            raise PolicyContractError("unsupported policy inference request version")
        if not isinstance(self.artifact_id, str) or not self.artifact_id:
            raise PolicyContractError("artifact ID must be nonempty")
        if len(self.observation) != 875 or any(
            type(value) is not bool for value in self.observation
        ):
            raise PolicyContractError("observation must contain exactly 875 Boolean values")
        if len(self.legal_mask) != 4 or any(
            len(row) != 8 or any(type(value) is not bool for value in row)
            for row in self.legal_mask
        ):
            raise PolicyContractError("legal mask must be Boolean[4,8]")
        if not any(value for row in self.legal_mask for value in row):
            raise PolicyContractError("legal mask must contain at least one legal action")
        validate_hidden_bytes(self.hidden_state)


@dataclass(frozen=True, slots=True)
class PolicyInferenceResponse:
    contract_version: str
    artifact_id: str
    raw_logits: tuple[tuple[float, ...], ...]
    next_hidden_state: bytes

    def __post_init__(self) -> None:
        if self.contract_version != POLICY_INFERENCE_CONTRACT_VERSION:
            raise PolicyContractError("unsupported policy inference response version")
        if not isinstance(self.artifact_id, str) or not self.artifact_id:
            raise PolicyContractError("artifact ID must be nonempty")
        validate_raw_logits(self.raw_logits)
        validate_hidden_bytes(self.next_hidden_state)


class PolicyAdapter(Protocol):
    @property
    def metadata(self) -> PolicyArtifactMetadata: ...

    def invoke(self, request: PolicyInferenceRequest) -> PolicyInferenceResponse: ...


@dataclass(frozen=True, slots=True)
class ActionSelectionProfile:
    version: str
    action_selection: str
    temperature: float


_PROFILES = {
    "argmax-v1": ActionSelectionProfile("argmax-v1", "argmax", 1.0),
    "sample-temperature-1-v1": ActionSelectionProfile(
        "sample-temperature-1-v1", "sample", 1.0
    ),
}


def resolve_inference_profile(version: str) -> ActionSelectionProfile:
    try:
        return _PROFILES[version]
    except KeyError as error:
        raise PolicyContractError(f"unsupported inference profile: {version}") from error


def validate_hidden_bytes(value: bytes) -> None:
    if type(value) is not bytes or len(value) != HIDDEN_STATE_BYTES:
        raise PolicyContractError(
            f"hidden state must contain exactly {HIDDEN_STATE_BYTES} bytes"
        )
    if not all(math.isfinite(item) for item in struct.unpack("<128f", value)):
        raise PolicyContractError("hidden state must contain only finite float32 values")


def validate_raw_logits(raw_logits: tuple[tuple[float, ...], ...]) -> None:
    if len(raw_logits) != 4 or any(len(row) != 8 for row in raw_logits):
        raise PolicyContractError("raw logits must have shape [4,8]")
    if any(
        type(value) not in {int, float} or not math.isfinite(float(value))
        for row in raw_logits
        for value in row
    ):
        raise PolicyContractError("raw logits must contain only finite numbers")


def select_masked_action(
    raw_logits: tuple[tuple[float, ...], ...],
    legal_mask: tuple[tuple[bool, ...], ...],
    profile: ActionSelectionProfile,
    *,
    game_id: str,
    round_number: int,
    turn_number: int,
    artifact_id: str,
) -> int:
    """Apply the authoritative mask and fixed profile outside the neural model."""

    validate_raw_logits(raw_logits)
    if len(legal_mask) != 4 or any(
        len(row) != 8 or any(type(value) is not bool for value in row)
        for row in legal_mask
    ):
        raise PolicyContractError("legal mask must be Boolean[4,8]")
    legal = tuple(
        index
        for index, allowed in enumerate(value for row in legal_mask for value in row)
        if allowed
    )
    if not legal:
        raise PolicyContractError("legal mask must contain at least one legal action")
    flattened = tuple(float(value) for row in raw_logits for value in row)
    if profile.action_selection == "argmax":
        return max(legal, key=lambda index: (flattened[index], -index))
    if profile.action_selection != "sample" or profile.temperature <= 0:
        raise PolicyContractError("inference profile is invalid")

    maximum = max(flattened[index] / profile.temperature for index in legal)
    weights = tuple(
        math.exp(flattened[index] / profile.temperature - maximum) for index in legal
    )
    total = sum(weights)
    seed = derive_seed(
        SERVING_SAMPLING_NAMESPACE,
        game_id,
        str(round_number),
        str(turn_number),
        artifact_id,
        profile.version,
    )
    threshold = int.from_bytes(seed, "big") / float(1 << 256) * total
    cumulative = 0.0
    for index, weight in zip(legal, weights, strict=True):
        cumulative += weight
        if threshold < cumulative:
            return index
    return legal[-1]

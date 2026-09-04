"""Policy execution contracts shared by stateful and stateless gameplay."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol
from uuid import UUID

from dracula.engine import EnginePlayer

HIDDEN_STATE_BYTES = 128 * 4


def zero_hidden_state() -> bytes:
    """Return the fixed empty recurrent state required by compatibility callers."""

    return bytes(HIDDEN_STATE_BYTES)


@dataclass(frozen=True, slots=True)
class PolicyDescriptor:
    """Immutable identity and tensor contracts for one configured controller."""

    policy_id: str = "unconfigured"
    policy_version: str = "unavailable"
    artifact_id: str = "none"
    artifact_sha256: str = "none"
    observation_schema_version: str = "dracula-policy-observation-v1"
    action_schema_version: str = "dracula-policy-action-v1"
    hidden_state_schema_version: str = "dracula-policy-hidden-v1"
    inference_profile: str = "local"


@dataclass(frozen=True, slots=True)
class PolicyTurnRequest:
    """Controller-neutral turn request carrying actor-visible decision data."""

    game_id: UUID
    policy: PolicyDescriptor
    turn_number: int
    player: EnginePlayer
    round_number: int
    turn_kind: Any
    policy_input: Any
    action_table: tuple[Any, ...]
    information_state: Any
    hidden_state: bytes


@dataclass(frozen=True, slots=True)
class PolicyTurnResult:
    """Selected action and optional legacy recurrent state from a controller."""

    action_index: int | None
    hidden_state: bytes | None


class PolicyExecutionError(Exception):
    """A retryable failure before a policy move has been committed."""


class PolicyExecutor(Protocol):
    """Synchronous controller boundary used by both gameplay services."""

    def invoke(self, request: PolicyTurnRequest) -> PolicyTurnResult: ...


class UnavailablePolicyExecutor:
    """Explicit controller used when gameplay inference is not configured."""

    def invoke(self, request: PolicyTurnRequest) -> PolicyTurnResult:
        raise PolicyExecutionError("no gameplay policy is configured")


@dataclass(frozen=True, slots=True)
class ServiceResponse:
    """Transport-neutral HTTP status and JSON-compatible response body."""

    status_code: int
    body: dict[str, Any]

"""Define the controller-neutral gameplay policy boundary.

Requests contain the acting player's visible information and legal action table.
Executors return a selected action without receiving complete engine state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol
from uuid import UUID

from dracula.engine import EnginePlayer


class PolicyContractError(ValueError):
    """A configured policy request or response violates its boundary."""


@dataclass(frozen=True, slots=True)
class PolicyDescriptor:
    """Identity pinned to a game so its opponent cannot change mid-session."""

    policy_id: str = "unconfigured"
    artifact_digest: str | None = None


@dataclass(frozen=True, slots=True)
class PolicyTurnRequest:
    """Controller-neutral turn request carrying actor-visible decision data."""

    game_id: UUID
    policy: PolicyDescriptor
    turn_number: int
    player: EnginePlayer
    round_number: int
    action_table: tuple[Any, ...]
    information_state: Any


@dataclass(frozen=True, slots=True)
class PolicyTurnResult:
    """Concrete flattened action selected by a standalone controller."""

    action_index: int | None


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

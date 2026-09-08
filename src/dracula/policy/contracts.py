"""Define the controller-neutral gameplay policy boundary.

Requests contain the acting player's visible information and legal action table.
Executors return a selected action without receiving complete engine state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from dracula.game.engine import EnginePlayer


class PolicyContractError(ValueError):
    """A configured policy request or response violates its boundary."""


@dataclass(frozen=True, slots=True)
class PolicyTurnRequest:
    """Controller-neutral turn request carrying actor-visible decision data."""

    # The stateless service uses ``role:seed``. The value affects only
    # deterministic resolution within a selected strategic group.
    game_key: str
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
    """Synchronous controller boundary used by stateless gameplay."""

    def invoke(self, request: PolicyTurnRequest) -> PolicyTurnResult: ...


class UnavailablePolicyExecutor:
    """Explicit controller used when gameplay inference is not configured."""

    def invoke(self, request: PolicyTurnRequest) -> PolicyTurnResult:
        raise PolicyExecutionError("no gameplay policy is configured")

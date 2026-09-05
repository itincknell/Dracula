"""Define the shared boundary between outer search and response selection.

The outer BGC search owns hidden-world sampling, engine transitions, and value
backup. A continuation policy sees only the simulated actor's information and
returns one move. The immutable records below carry that decision, aggregate
search evidence, and private diagnostics without coupling search to a specific
continuation algorithm.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from dracula.engine import EnginePlayer, EngineRoundResult, other_player
from dracula.search.information import SearchInformationState
from dracula.strategic_actions import StrategicActionGroup

# Both outer UCT and belief-greedy responses use this fixed scale. It preserves
# score ordering while keeping score differences comparable with UCT exploration.
ROUND_SCORE_NORMALIZER = 150.0


class SearchContractViolation(ValueError):
    """A search component returned data inconsistent with the shared contract."""


class SearchInterrupted(RuntimeError):
    """Search stopped before it could produce an atomic decision."""


@dataclass(frozen=True, slots=True)
class ContinuationStep:
    """One move from a sampled continuation retained for diagnostics.

    ``actor`` is relative to the root decision maker. Card identity and board
    position belong to a sampled private world, so this record must not cross a
    public API or policy-input boundary.
    """

    actor: str
    action_index: int
    card_id: str
    grid_index: int
    forced: bool


@dataclass(frozen=True, slots=True)
class PrincipalContinuation:
    """Best sampled line observed for the root group ultimately selected.

    This is an explanatory example of a simulation, not an additional search
    target or a claim that the sampled hidden cards were authoritative.
    """

    root_action_index: int
    terminal_value: float
    simulation_index: int
    steps: tuple[ContinuationStep, ...]


@dataclass(frozen=True, slots=True)
class StrategicGroupStatistics:
    """Visits and mean actor-relative value for one strategic action group.

    Mirrored concrete destinations share this record through their designated
    representative. ``mean_value`` is absent when no simulation evaluated the
    group, as in a forced result.
    """

    group: StrategicActionGroup
    visits: int
    mean_value: float | None


@dataclass(frozen=True, slots=True)
class ContinuationDecision:
    """The move selected by one actor-local continuation policy.

    The information fingerprint proves which visible state was evaluated, and
    the configuration digest identifies the policy used. Outer search checks
    both before accepting or caching the result. Work counts describe how the
    continuation reached its decision; they do not affect engine scoring.
    """

    information_state_fingerprint: str
    config_digest: str
    selected_group: StrategicActionGroup
    selected_action_index: int
    group_statistics: tuple[StrategicGroupStatistics, ...]
    terminal_evaluation_count: int
    model_inference_count: int


class ContinuationPolicy(Protocol):
    """Interface for choosing simulated responses inside outer BGC search.

    Implementations may use belief completions or one policy inference, but
    receive only ``SearchInformationState`` for the actor taking the move. The
    digest separates cached decisions made by different implementations or
    configurations. ``should_stop`` permits cooperative interruption.
    """

    @property
    def digest(self) -> str: ...

    def select(
        self,
        information: SearchInformationState,
        should_stop: Callable[[], bool] | None = None,
    ) -> ContinuationDecision: ...


def search_config_digest(value: object) -> str:
    """Hash a search configuration with stable key order and JSON formatting."""

    encoded = json.dumps(value, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def normalized_round_return(
    result: EngineRoundResult,
    root_player: EnginePlayer,
) -> float:
    """Convert exact round scores to the common actor-relative search scale."""

    opponent = other_player(root_player)
    return (
        result.round_scores[root_player] - result.round_scores[opponent]
    ) / ROUND_SCORE_NORMALIZER


__all__ = (
    "ContinuationStep",
    "ContinuationDecision",
    "ContinuationPolicy",
    "PrincipalContinuation",
    "ROUND_SCORE_NORMALIZER",
    "SearchContractViolation",
    "SearchInterrupted",
    "StrategicGroupStatistics",
    "normalized_round_return",
    "search_config_digest",
)

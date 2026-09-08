"""Store and update an information-set UCT search tree.

The outer search does not build a tree of complete engine states. Hidden-card
samples that look identical to the root player share one node, identified by
the root player's visible-information fingerprint. Each outgoing edge is one
strategic action group and stores its visit count and accumulated round return.

This module owns that mutable search state, UCT edge selection, and terminal
backup. Sampling hidden worlds and playing simulations belong to
``uct_simulation``; choosing the final real move belongs to
``information_set_uct``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from dracula.decision.bridge import ACTION_COUNT
from dracula.game.engine import EnginePlayer, EngineStatus, SimulationEngineState
from dracula.search.contracts import (
    ContinuationDecision,
    ContinuationStep,
    SearchContractViolation,
    normalized_round_return,
)
from dracula.decision.strategic_actions import StrategicActionGroup


@dataclass(frozen=True, slots=True)
class NodeKey:
    """Identify one root-player-visible position independently of hidden cards."""

    actor: EnginePlayer
    information_state_fingerprint: str


@dataclass(slots=True)
class Node:
    """Hold UCT statistics for every strategic action at one visible position."""

    key: NodeKey
    groups: tuple[StrategicActionGroup, ...]
    visits: int
    action_visits: list[int]
    action_value_sums: list[float]

    @classmethod
    def create(
        cls,
        key: NodeKey,
        groups: tuple[StrategicActionGroup, ...],
    ) -> Node:
        """Create an unvisited node using flattened action indexes as storage."""

        return cls(key, groups, 0, [0] * ACTION_COUNT, [0.0] * ACTION_COUNT)


@dataclass(frozen=True, slots=True)
class ObservedContinuation:
    """Retain one completed sampled line for private diagnostics."""

    root_group: StrategicActionGroup
    root_action_index: int
    terminal_value: float
    simulation_index: int
    steps: tuple[ContinuationStep, ...]


@dataclass(slots=True)
class SearchWork:
    """Collect mutable tree state and work counters for one search request."""

    nodes: dict[NodeKey, Node]
    continuation_cache: dict[str, ContinuationDecision]
    observed_continuations: list[ObservedContinuation]
    continuation_requests: int = 0
    continuation_cache_hits: int = 0
    terminal_evaluations: int = 0
    model_inferences: int = 0


@dataclass(slots=True)
class SimulationTrace:
    """Record the searched edges and private moves from one simulation."""

    path: list[tuple[Node, StrategicActionGroup]]
    steps: list[ContinuationStep]
    root_group: StrategicActionGroup | None = None
    root_action_index: int | None = None


def select_group(node: Node, exploration_constant: float) -> StrategicActionGroup:
    """Choose one outgoing edge using canonical initial coverage and UCT."""

    # Give each action one observation before comparing estimated returns.
    for group in node.groups:
        if node.action_visits[group.representative_action_index] == 0:
            return group

    log_parent = math.log(node.visits)
    best_score = -math.inf
    tied: list[StrategicActionGroup] = []
    for group in node.groups:
        representative = group.representative_action_index
        visits = node.action_visits[representative]
        mean = node.action_value_sums[representative] / visits
        score = mean + exploration_constant * math.sqrt(log_parent / visits)
        if score > best_score:
            best_score = score
            tied = [group]
        elif score == best_score:
            tied.append(group)

    # Canonical tie-breaking keeps identical requests reproducible.
    return min(tied, key=lambda group: group.representative_action_index)


def back_up_simulation(
    state: SimulationEngineState,
    root_player: EnginePlayer,
    trace: SimulationTrace,
    simulation_index: int,
    work: SearchWork,
) -> None:
    """Apply one completed round result to every searched edge in its path."""

    if (
        state.status is not EngineStatus.ROUND_COMPLETE
        or state.pending_round_result is None
        or trace.root_group is None
        or trace.root_action_index is None
        or not trace.path
    ):
        raise SearchContractViolation("simulation did not reach a backed round result")

    terminal_value = normalized_round_return(
        state.pending_round_result,
        root_player,
    )
    # One root-relative result updates every root-player choice made along the
    # sampled line, even when opponent moves occurred between those choices.
    for node, selected_group in trace.path:
        representative = selected_group.representative_action_index
        node.visits += 1
        node.action_visits[representative] += 1
        node.action_value_sums[representative] += terminal_value

    work.observed_continuations.append(
        ObservedContinuation(
            trace.root_group,
            trace.root_action_index,
            terminal_value,
            simulation_index,
            tuple(trace.steps),
        )
    )


def root_rank(node: Node, group: StrategicActionGroup) -> tuple[int, float, int]:
    """Rank root actions by visits, mean return, and canonical action order."""

    representative = group.representative_action_index
    visits = node.action_visits[representative]
    return (
        -visits,
        -node.action_value_sums[representative] / visits,
        representative,
    )


__all__ = (
    "Node",
    "NodeKey",
    "ObservedContinuation",
    "SearchWork",
    "SimulationTrace",
    "back_up_simulation",
    "root_rank",
    "select_group",
)

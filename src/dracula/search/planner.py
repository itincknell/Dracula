"""Information-set Monte Carlo planning through the current round."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable
from dataclasses import dataclass

from dracula.bridge import ACTION_COUNT, action_index_for_move, move_for_action_index
from dracula.engine import (
    EngineMove,
    EnginePlayer,
    EngineRoundResult,
    EngineStatus,
    SimulationEngineState,
    apply_simulation_move,
    legal_simulation_moves,
    other_player,
)
from dracula.randomness import Sha256CounterStream, seed_hex
from dracula.search.information import (
    SearchInformationState,
    derive_belief_sample_seed,
    derive_rollout_choice_seed,
    derive_tree_selection_seed,
    information_state_fingerprint,
    project_simulation_information_state,
    sample_determinization,
    sample_uniform_action_index,
)

SEARCH_SCHEMA_VERSION = "dracula-information-search-v1"
ROUND_SCORE_NORMALIZER = 150.0


class SearchContractViolation(ValueError):
    """Search input or an internal engine boundary is inconsistent."""


class SearchInterrupted(RuntimeError):
    """A caller stopped a search before a result was committed."""


@dataclass(frozen=True, slots=True)
class SearchConfig:
    simulation_budget: int = 500
    exploration_constant: float = math.sqrt(2.0)

    def __post_init__(self) -> None:
        if type(self.simulation_budget) is not int or self.simulation_budget < 1:
            raise SearchContractViolation("simulation budget must be a positive integer")
        if type(self.exploration_constant) not in (int, float) or not math.isfinite(
            self.exploration_constant
        ):
            raise SearchContractViolation("exploration constant must be finite")
        if self.exploration_constant < 0:
            raise SearchContractViolation("exploration constant cannot be negative")

    @property
    def digest(self) -> str:
        payload = json.dumps(
            {
                "search_schema_version": SEARCH_SCHEMA_VERSION,
                "simulation_budget": self.simulation_budget,
                "exploration_constant": float(self.exploration_constant),
            },
            separators=(",", ":"),
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class ContinuationStep:
    actor: str
    action_index: int
    card_id: str
    grid_index: int
    forced: bool


@dataclass(frozen=True, slots=True)
class PrincipalContinuation:
    root_action_index: int
    terminal_value: float
    simulation_index: int
    steps: tuple[ContinuationStep, ...]


@dataclass(frozen=True, slots=True)
class SearchResult:
    information_state_fingerprint: str
    config_digest: str
    selected_action_index: int
    action_visits: tuple[int, ...]
    mean_action_values: tuple[float | None, ...]
    simulation_count: int
    information_set_count: int
    principal_continuation: PrincipalContinuation | None

    def __post_init__(self) -> None:
        if len(self.action_visits) != ACTION_COUNT:
            raise SearchContractViolation("action visits must contain 32 entries")
        if len(self.mean_action_values) != ACTION_COUNT:
            raise SearchContractViolation("action values must contain 32 entries")


@dataclass(slots=True)
class _InformationNode:
    legal_actions: tuple[int, ...]
    visits: int
    action_visits: list[int]
    action_value_sums: list[float]

    @classmethod
    def create(cls, legal_actions: tuple[int, ...]) -> _InformationNode:
        return cls(legal_actions, 0, [0] * ACTION_COUNT, [0.0] * ACTION_COUNT)


@dataclass(frozen=True, slots=True)
class _ObservedContinuation:
    root_action_index: int
    terminal_value: float
    simulation_index: int
    steps: tuple[ContinuationStep, ...]


def _legal_action_indexes(information: SearchInformationState) -> tuple[int, ...]:
    return tuple(
        index
        for index, legal in enumerate(
            value for row in information.legal_mask for value in row
        )
        if legal
    )


def normalized_round_return(
    result: EngineRoundResult, root_player: EnginePlayer
) -> float:
    """Return the engine's exact round differential on the documented scale."""

    if not isinstance(result, EngineRoundResult):
        raise SearchContractViolation("terminal return requires an engine round result")
    root_player = EnginePlayer(root_player)
    opponent = other_player(root_player)
    value = (
        result.round_scores[root_player] - result.round_scores[opponent]
    ) / ROUND_SCORE_NORMALIZER
    if not -1.0 <= value <= 1.0:
        raise SearchContractViolation("engine round differential exceeds its score bound")
    return value


def _move_for_action(state: SimulationEngineState, action_index: int) -> EngineMove:
    actor = state.active_player
    if actor is None:
        raise SearchContractViolation("a playing simulation must have an active player")
    move = move_for_action_index(actor, action_index)
    if move not in legal_simulation_moves(state, actor):
        raise SearchContractViolation("search selected a masked engine action")
    return move


def _select_tree_action(
    node: _InformationNode,
    exploration_constant: float,
    selection_seed: bytes,
) -> int:
    # Initial coverage is canonical so every legal root action receives evidence.
    for action_index in node.legal_actions:
        if node.action_visits[action_index] == 0:
            return action_index

    log_parent = math.log(node.visits)
    best_score = -math.inf
    tied: list[int] = []
    for action_index in node.legal_actions:
        visits = node.action_visits[action_index]
        mean = node.action_value_sums[action_index] / visits
        score = mean + exploration_constant * math.sqrt(log_parent / visits)
        if score > best_score:
            best_score = score
            tied = [action_index]
        elif score == best_score:
            tied.append(action_index)
    return tied[Sha256CounterStream(selection_seed).randbelow(len(tied))]


def _uniform_action(
    state: SimulationEngineState,
    request_seed: bytes,
    simulation_index: int,
    rollout_ply: int,
) -> int:
    actor_information = project_simulation_information_state(state)
    return sample_uniform_action_index(
        actor_information,
        derive_rollout_choice_seed(request_seed, simulation_index, rollout_ply),
    )


def _continuation_step(
    root_player: EnginePlayer,
    state: SimulationEngineState,
    move: EngineMove,
    action_index: int,
    forced: bool,
) -> ContinuationStep:
    card_id = state.hands[move.player][move.hand_slot]
    if card_id is None:
        raise SearchContractViolation("continuation move refers to an empty hand slot")
    return ContinuationStep(
        actor="self" if move.player is root_player else "opponent",
        action_index=action_index,
        card_id=card_id,
        grid_index=move.global_grid_index,
        forced=forced,
    )


class InformationSetSearch:
    """Deployable planner whose public boundary contains no private engine state."""

    def __init__(self, config: SearchConfig = SearchConfig()) -> None:
        if not isinstance(config, SearchConfig):
            raise SearchContractViolation("search configuration is invalid")
        self.config = config

    def search(
        self,
        information: SearchInformationState,
        request_seed: bytes,
        should_stop: Callable[[], bool] | None = None,
    ) -> SearchResult:
        if not isinstance(information, SearchInformationState):
            raise SearchContractViolation("deployable search requires an information state")
        seed_hex(request_seed)
        root_player = information.player
        root_fingerprint = information_state_fingerprint(information)
        root_legal = _legal_action_indexes(information)
        if not root_legal:
            raise SearchContractViolation("search root has no legal actions")

        # A forced placement is an engine transition, not a planning decision.
        if len(root_legal) == 1:
            return SearchResult(
                information_state_fingerprint=root_fingerprint,
                config_digest=self.config.digest,
                selected_action_index=root_legal[0],
                action_visits=tuple(0 for _ in range(ACTION_COUNT)),
                mean_action_values=tuple(None for _ in range(ACTION_COUNT)),
                simulation_count=0,
                information_set_count=0,
                principal_continuation=None,
            )
        if self.config.simulation_budget < len(root_legal):
            raise SearchContractViolation(
                "simulation budget must visit every legal root action"
            )

        nodes: dict[str, _InformationNode] = {}
        continuations: list[_ObservedContinuation] = []
        for simulation_index in range(self.config.simulation_budget):
            self._check_interrupted(should_stop)
            sampled = sample_determinization(
                information,
                derive_belief_sample_seed(request_seed, simulation_index),
            )
            state = sampled.state
            path: list[tuple[_InformationNode, int]] = []
            steps: list[ContinuationStep] = []
            expanded = False
            root_action_index: int | None = None
            rollout_ply = 0

            while state.status is EngineStatus.PLAYING:
                self._check_interrupted(should_stop)
                actor = state.active_player
                if actor is None:
                    raise SearchContractViolation("playing simulation lost its active player")
                moves = legal_simulation_moves(state, actor)
                forced = len(moves) == 1
                if forced:
                    move = moves[0]
                    action_index = action_index_for_move(move, actor)
                elif actor is root_player and not expanded:
                    actor_information = project_simulation_information_state(state)
                    node_key = information_state_fingerprint(actor_information)
                    legal_actions = _legal_action_indexes(actor_information)
                    node = nodes.get(node_key)
                    if node is None:
                        node = _InformationNode.create(legal_actions)
                        nodes[node_key] = node
                    elif node.legal_actions != legal_actions:
                        raise SearchContractViolation(
                            "one information-set node produced inconsistent legal actions"
                        )
                    action_index = _select_tree_action(
                        node,
                        float(self.config.exploration_constant),
                        derive_tree_selection_seed(
                            request_seed, simulation_index, node_key
                        ),
                    )
                    path.append((node, action_index))
                    if node.action_visits[action_index] == 0:
                        expanded = True
                    move = _move_for_action(state, action_index)
                else:
                    action_index = _uniform_action(
                        state, request_seed, simulation_index, rollout_ply
                    )
                    move = _move_for_action(state, action_index)

                if root_action_index is None:
                    if actor is not root_player:
                        raise SearchContractViolation("root simulation began on another actor")
                    root_action_index = action_index
                steps.append(
                    _continuation_step(
                        root_player, state, move, action_index, forced
                    )
                )
                state = apply_simulation_move(state, move)
                if not isinstance(state, SimulationEngineState):
                    raise SearchContractViolation("engine discarded simulation provenance")
                rollout_ply += 1

            if (
                state.status is not EngineStatus.ROUND_COMPLETE
                or state.pending_round_result is None
            ):
                raise SearchContractViolation("round search did not reach an engine terminal")
            if root_action_index is None or not path:
                raise SearchContractViolation("a learned root simulation has no tree path")
            terminal_value = normalized_round_return(
                state.pending_round_result, root_player
            )
            for node, action_index in path:
                node.visits += 1
                node.action_visits[action_index] += 1
                node.action_value_sums[action_index] += terminal_value
            continuations.append(
                _ObservedContinuation(
                    root_action_index,
                    terminal_value,
                    simulation_index,
                    tuple(steps),
                )
            )

        root_node = nodes.get(root_fingerprint)
        if root_node is None or root_node.visits != self.config.simulation_budget:
            raise SearchContractViolation("root backup count does not match simulations")
        selected = min(
            root_legal,
            key=lambda action: (
                -root_node.action_visits[action],
                -root_node.action_value_sums[action]
                / root_node.action_visits[action],
                action,
            ),
        )
        representative = min(
            (item for item in continuations if item.root_action_index == selected),
            key=lambda item: (-item.terminal_value, item.simulation_index),
        )
        means = tuple(
            (
                root_node.action_value_sums[index] / root_node.action_visits[index]
                if root_node.action_visits[index]
                else None
            )
            for index in range(ACTION_COUNT)
        )
        return SearchResult(
            information_state_fingerprint=root_fingerprint,
            config_digest=self.config.digest,
            selected_action_index=selected,
            action_visits=tuple(root_node.action_visits),
            mean_action_values=means,
            simulation_count=root_node.visits,
            information_set_count=len(nodes),
            principal_continuation=PrincipalContinuation(
                representative.root_action_index,
                representative.terminal_value,
                representative.simulation_index,
                representative.steps,
            ),
        )

    @staticmethod
    def _check_interrupted(should_stop: Callable[[], bool] | None) -> None:
        if should_stop is not None and should_stop():
            raise SearchInterrupted("search interrupted before result commit")

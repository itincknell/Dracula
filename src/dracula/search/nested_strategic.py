"""Historical Teacher v2 search with information-safe actor-local UCT responses."""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from dracula.bridge import (
    ACTION_COUNT,
    action_index_for_move,
    build_policy_turn_context,
)
from dracula.engine import (
    EngineMove,
    EnginePlayer,
    EngineStatus,
    SimulationEngineState,
    apply_simulation_move,
    legal_moves,
)
from dracula.randomness import Sha256CounterStream, derive_seed, seed_hex
from dracula.search.information import (
    SearchInformationState,
    information_state_fingerprint,
    project_simulation_information_state,
    sample_determinization,
)
from dracula.search.planner import (
    SEARCH_SCHEMA_VERSION,
    ContinuationStep,
    InformationSetSearch,
    PrincipalContinuation,
    SearchConfig,
    SearchContractViolation,
    SearchInterrupted,
    SearchResult,
    normalized_round_return,
)

NESTED_STRATEGIC_SEARCH_SCHEMA_VERSION = (
    "dracula-strategic-information-search-v1"
)
NESTED_STRATEGIC_SEARCH_REQUEST_NAMESPACE = (
    "dracula-strategic-search-request-v1"
)
NESTED_STRATEGIC_DETERMINIZATION_NAMESPACE = (
    "dracula-strategic-search-determinization-v1"
)
NESTED_STRATEGIC_SELECTION_NAMESPACE = (
    "dracula-strategic-search-expansion-v1"
)
NESTED_STRATEGIC_RESPONSE_REQUEST_NAMESPACE = (
    "dracula-strategic-response-request-v1"
)
NESTED_STRATEGIC_SELECTION_PROFILE = (
    "max-visits-mean-value-action-index-v1"
)
NESTED_STRATEGIC_RECURSION_LIMIT = 1


@dataclass(frozen=True, slots=True)
class NestedStrategicSearchConfig:
    outer_simulation_budget: int = 32
    response_simulation_budget: int = 32
    outer_exploration_constant: float = math.sqrt(2.0)
    response_exploration_constant: float = math.sqrt(2.0)

    def __post_init__(self) -> None:
        for value, label in (
            (self.outer_simulation_budget, "outer simulation budget"),
            (self.response_simulation_budget, "response simulation budget"),
        ):
            if type(value) is not int or value < 1:
                raise SearchContractViolation(
                    f"{label} must be a positive integer"
                )
        for value, label in (
            (self.outer_exploration_constant, "outer exploration constant"),
            (
                self.response_exploration_constant,
                "response exploration constant",
            ),
        ):
            if (
                type(value) not in (int, float)
                or not math.isfinite(value)
                or value < 0
            ):
                raise SearchContractViolation(
                    f"{label} must be finite and non-negative"
                )

    @property
    def response_config(self) -> SearchConfig:
        return SearchConfig(
            self.response_simulation_budget,
            self.response_exploration_constant,
        )

    @property
    def digest(self) -> str:
        payload = json.dumps(
            {
                "outer_exploration_constant": float(
                    self.outer_exploration_constant
                ),
                "outer_simulation_budget": self.outer_simulation_budget,
                "recursion_limit": NESTED_STRATEGIC_RECURSION_LIMIT,
                "response_exploration_constant": float(
                    self.response_exploration_constant
                ),
                "response_search_schema_version": SEARCH_SCHEMA_VERSION,
                "response_simulation_budget": (
                    self.response_simulation_budget
                ),
                "search_schema_version": (
                    NESTED_STRATEGIC_SEARCH_SCHEMA_VERSION
                ),
                "selection_profile": NESTED_STRATEGIC_SELECTION_PROFILE,
            },
            separators=(",", ":"),
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class NestedStrategicNodeKey:
    actor: EnginePlayer
    information_state_fingerprint: str


@dataclass(frozen=True, slots=True)
class NestedStrategicSearchResult(SearchResult):
    elapsed_seconds: float = field(compare=False)
    response_request_count: int
    unique_response_search_count: int
    response_cache_hit_count: int
    response_simulation_count: int
    response_information_set_count: int
    total_terminal_evaluation_count: int

    def __post_init__(self) -> None:
        super(NestedStrategicSearchResult, self).__post_init__()
        if (
            not math.isfinite(self.elapsed_seconds)
            or self.elapsed_seconds < 0
        ):
            raise SearchContractViolation(
                "search latency must be finite and non-negative"
            )
        counts = (
            self.response_request_count,
            self.unique_response_search_count,
            self.response_cache_hit_count,
            self.response_simulation_count,
            self.response_information_set_count,
            self.total_terminal_evaluation_count,
        )
        if any(type(value) is not int or value < 0 for value in counts):
            raise SearchContractViolation(
                "strategic diagnostics must be non-negative"
            )
        if self.response_request_count != (
            self.unique_response_search_count
            + self.response_cache_hit_count
        ):
            raise SearchContractViolation(
                "response cache accounting is inconsistent"
            )
        if self.total_terminal_evaluation_count != (
            self.simulation_count + self.response_simulation_count
        ):
            raise SearchContractViolation(
                "terminal evaluation accounting is inconsistent"
            )

    @property
    def total_information_set_count(self) -> int:
        return (
            self.information_set_count
            + self.response_information_set_count
        )


@dataclass(slots=True)
class _InformationNode:
    key: NestedStrategicNodeKey
    legal_actions: tuple[int, ...]
    visits: int
    action_visits: list[int]
    action_value_sums: list[float]

    @classmethod
    def create(
        cls,
        key: NestedStrategicNodeKey,
        legal_actions: tuple[int, ...],
    ) -> _InformationNode:
        return cls(
            key,
            legal_actions,
            0,
            [0] * ACTION_COUNT,
            [0.0] * ACTION_COUNT,
        )


@dataclass(frozen=True, slots=True)
class _ObservedContinuation:
    root_action_index: int
    terminal_value: float
    simulation_index: int
    steps: tuple[ContinuationStep, ...]


def _validate_counter(value: int, label: str) -> str:
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return str(value)


def _legal_action_indexes(
    information: SearchInformationState,
) -> tuple[int, ...]:
    return tuple(
        index
        for index, allowed in enumerate(
            value for row in information.legal_mask for value in row
        )
        if allowed
    )


def derive_nested_strategic_search_request_seed(
    fixture_id: str,
    information: SearchInformationState,
    search_config_digest: str,
) -> bytes:
    return derive_seed(
        NESTED_STRATEGIC_SEARCH_REQUEST_NAMESPACE,
        fixture_id,
        information_state_fingerprint(information),
        information.player.value,
        str(information.round_number),
        str(information.turn_number),
        search_config_digest,
    )


def derive_nested_strategic_determinization_seed(
    request_seed: bytes,
    simulation_index: int,
) -> bytes:
    return derive_seed(
        NESTED_STRATEGIC_DETERMINIZATION_NAMESPACE,
        seed_hex(request_seed),
        _validate_counter(simulation_index, "simulation index"),
    )


def derive_nested_strategic_selection_seed(
    request_seed: bytes,
    simulation_index: int,
    node: NestedStrategicNodeKey,
) -> bytes:
    return derive_seed(
        NESTED_STRATEGIC_SELECTION_NAMESPACE,
        seed_hex(request_seed),
        _validate_counter(simulation_index, "simulation index"),
        node.actor.value,
        node.information_state_fingerprint,
    )


def derive_nested_strategic_response_seed(
    information: SearchInformationState,
    response_config_digest: str,
) -> bytes:
    """Derive an actor response seed without enclosing-world information."""

    return derive_seed(
        NESTED_STRATEGIC_RESPONSE_REQUEST_NAMESPACE,
        information_state_fingerprint(information),
        response_config_digest,
    )


def actor_relative_value(
    value: float,
    value_player: EnginePlayer,
    observer: EnginePlayer,
) -> float:
    if not math.isfinite(value) or not -1.0 <= value <= 1.0:
        raise SearchContractViolation(
            "actor value must be finite and in [-1, 1]"
        )
    return value if value_player is observer else -value


def _select_tree_action(
    node: _InformationNode,
    exploration_constant: float,
    selection_seed: bytes,
) -> int:
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


def _move_for_action(
    state: SimulationEngineState,
    action_index: int,
) -> EngineMove:
    actor = state.active_player
    if actor is None:
        raise SearchContractViolation(
            "a playing simulation must have an active player"
        )
    move = build_policy_turn_context(state, actor).action_table[action_index]
    if move is None:
        raise SearchContractViolation(
            "strategic search selected a masked action"
        )
    return move


def _continuation_step(
    root_player: EnginePlayer,
    state: SimulationEngineState,
    move: EngineMove,
    action_index: int,
    forced: bool,
) -> ContinuationStep:
    card_id = state.hands[move.player][move.hand_slot]
    if card_id is None:
        raise SearchContractViolation(
            "continuation move refers to an empty hand slot"
        )
    return ContinuationStep(
        actor="self" if move.player is root_player else "opponent",
        action_index=action_index,
        card_id=card_id,
        grid_index=move.global_grid_index,
        forced=forced,
    )


class NestedStrategicInformationSetSearch:
    """Teacher v2 outer search with isolated actor-local UCT responses."""

    def __init__(
        self,
        config: NestedStrategicSearchConfig = NestedStrategicSearchConfig(),
    ) -> None:
        if not isinstance(config, NestedStrategicSearchConfig):
            raise SearchContractViolation(
                "nested strategic search configuration is invalid"
            )
        self.config = config
        self._response_config = config.response_config
        self._response_config_digest = self._response_config.digest

    def _actor_response(
        self,
        information: SearchInformationState,
        should_stop: Callable[[], bool] | None,
    ) -> SearchResult:
        if not isinstance(information, SearchInformationState):
            raise SearchContractViolation(
                "actor response requires a player information state"
            )
        return InformationSetSearch(self._response_config).search(
            information,
            derive_nested_strategic_response_seed(
                information,
                self._response_config_digest,
            ),
            should_stop,
        )

    def search(
        self,
        information: SearchInformationState,
        request_seed: bytes,
        should_stop: Callable[[], bool] | None = None,
    ) -> NestedStrategicSearchResult:
        started = time.perf_counter()
        if not isinstance(information, SearchInformationState):
            raise SearchContractViolation(
                "strategic search requires an information state"
            )
        seed_hex(request_seed)
        root_player = information.player
        root_fingerprint = information_state_fingerprint(information)
        root_legal = _legal_action_indexes(information)
        if not root_legal:
            raise SearchContractViolation(
                "strategic search root has no legal actions"
            )

        if len(root_legal) == 1:
            return NestedStrategicSearchResult(
                information_state_fingerprint=root_fingerprint,
                config_digest=self.config.digest,
                selected_action_index=root_legal[0],
                action_visits=tuple(0 for _ in range(ACTION_COUNT)),
                mean_action_values=tuple(None for _ in range(ACTION_COUNT)),
                simulation_count=0,
                information_set_count=0,
                principal_continuation=None,
                elapsed_seconds=time.perf_counter() - started,
                response_request_count=0,
                unique_response_search_count=0,
                response_cache_hit_count=0,
                response_simulation_count=0,
                response_information_set_count=0,
                total_terminal_evaluation_count=0,
            )
        if self.config.outer_simulation_budget < len(root_legal):
            raise SearchContractViolation(
                "outer simulation budget must visit every legal root action"
            )

        nodes: dict[NestedStrategicNodeKey, _InformationNode] = {}
        response_cache: dict[
            tuple[NestedStrategicNodeKey, str],
            SearchResult,
        ] = {}
        continuations: list[_ObservedContinuation] = []
        response_requests = 0
        response_cache_hits = 0
        response_simulations = 0
        response_information_sets = 0

        for simulation_index in range(
            self.config.outer_simulation_budget
        ):
            self._check_interrupted(should_stop)
            sampled = sample_determinization(
                information,
                derive_nested_strategic_determinization_seed(
                    request_seed,
                    simulation_index,
                ),
            )
            state = sampled.state
            path: list[tuple[_InformationNode, int]] = []
            steps: list[ContinuationStep] = []
            expanded = False
            root_action_index: int | None = None

            while state.status is EngineStatus.PLAYING:
                self._check_interrupted(should_stop)
                actor = state.active_player
                if actor is None:
                    raise SearchContractViolation(
                        "playing simulation lost its active player"
                    )
                moves = legal_moves(state, actor)
                forced = len(moves) == 1
                if forced:
                    move = moves[0]
                    action_index = action_index_for_move(move, actor)
                elif actor is root_player and not expanded:
                    actor_information = (
                        project_simulation_information_state(state)
                    )
                    node_key = NestedStrategicNodeKey(
                        actor,
                        information_state_fingerprint(actor_information),
                    )
                    legal_actions = _legal_action_indexes(actor_information)
                    node = nodes.get(node_key)
                    if node is None:
                        node = _InformationNode.create(
                            node_key,
                            legal_actions,
                        )
                        nodes[node_key] = node
                    elif node.legal_actions != legal_actions:
                        raise SearchContractViolation(
                            "one strategic information node changed legal "
                            "actions"
                        )
                    action_index = _select_tree_action(
                        node,
                        float(self.config.outer_exploration_constant),
                        derive_nested_strategic_selection_seed(
                            request_seed,
                            simulation_index,
                            node_key,
                        ),
                    )
                    path.append((node, action_index))
                    if node.action_visits[action_index] == 0:
                        expanded = True
                    move = _move_for_action(state, action_index)
                else:
                    actor_information = (
                        project_simulation_information_state(state)
                    )
                    node_key = NestedStrategicNodeKey(
                        actor,
                        information_state_fingerprint(actor_information),
                    )
                    cache_key = (
                        node_key,
                        self._response_config_digest,
                    )
                    response_requests += 1
                    response = response_cache.get(cache_key)
                    if response is None:
                        response = self._actor_response(
                            actor_information,
                            should_stop,
                        )
                        if (
                            response.information_state_fingerprint
                            != node_key.information_state_fingerprint
                            or response.config_digest
                            != self._response_config_digest
                        ):
                            raise SearchContractViolation(
                                "actor response crossed its information "
                                "boundary"
                            )
                        response_cache[cache_key] = response
                        response_simulations += response.simulation_count
                        response_information_sets += (
                            response.information_set_count
                        )
                    else:
                        response_cache_hits += 1
                    action_index = response.selected_action_index
                    move = _move_for_action(state, action_index)

                if root_action_index is None:
                    if actor is not root_player:
                        raise SearchContractViolation(
                            "root simulation began on another actor"
                        )
                    root_action_index = action_index
                steps.append(
                    _continuation_step(
                        root_player,
                        state,
                        move,
                        action_index,
                        forced,
                    )
                )
                state = apply_simulation_move(state, move)

            if (
                state.status is not EngineStatus.ROUND_COMPLETE
                or state.pending_round_result is None
            ):
                raise SearchContractViolation(
                    "strategic round search did not reach an engine terminal"
                )
            if root_action_index is None or not path:
                raise SearchContractViolation(
                    "a learned strategic simulation has no root path"
                )
            root_value = normalized_round_return(
                state.pending_round_result,
                root_player,
            )
            for node, action_index in path:
                backed_value = actor_relative_value(
                    root_value,
                    root_player,
                    node.key.actor,
                )
                node.visits += 1
                node.action_visits[action_index] += 1
                node.action_value_sums[action_index] += backed_value
            continuations.append(
                _ObservedContinuation(
                    root_action_index,
                    root_value,
                    simulation_index,
                    tuple(steps),
                )
            )

        root_key = NestedStrategicNodeKey(
            root_player,
            root_fingerprint,
        )
        root_node = nodes.get(root_key)
        if (
            root_node is None
            or root_node.visits != self.config.outer_simulation_budget
        ):
            raise SearchContractViolation(
                "strategic root backup count does not match simulations"
            )
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
            (
                item
                for item in continuations
                if item.root_action_index == selected
            ),
            key=lambda item: (
                -item.terminal_value,
                item.simulation_index,
            ),
        )
        means = tuple(
            (
                root_node.action_value_sums[index]
                / root_node.action_visits[index]
                if root_node.action_visits[index]
                else None
            )
            for index in range(ACTION_COUNT)
        )
        return NestedStrategicSearchResult(
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
            elapsed_seconds=time.perf_counter() - started,
            response_request_count=response_requests,
            unique_response_search_count=len(response_cache),
            response_cache_hit_count=response_cache_hits,
            response_simulation_count=response_simulations,
            response_information_set_count=response_information_sets,
            total_terminal_evaluation_count=(
                root_node.visits + response_simulations
            ),
        )

    @staticmethod
    def _check_interrupted(
        should_stop: Callable[[], bool] | None,
    ) -> None:
        if should_stop is not None and should_stop():
            raise SearchInterrupted(
                "strategic search interrupted before result commit"
            )


__all__ = (
    "NESTED_STRATEGIC_DETERMINIZATION_NAMESPACE",
    "NESTED_STRATEGIC_RECURSION_LIMIT",
    "NESTED_STRATEGIC_RESPONSE_REQUEST_NAMESPACE",
    "NESTED_STRATEGIC_SEARCH_REQUEST_NAMESPACE",
    "NESTED_STRATEGIC_SEARCH_SCHEMA_VERSION",
    "NESTED_STRATEGIC_SELECTION_NAMESPACE",
    "NESTED_STRATEGIC_SELECTION_PROFILE",
    "NestedStrategicInformationSetSearch",
    "NestedStrategicNodeKey",
    "NestedStrategicSearchConfig",
    "NestedStrategicSearchResult",
    "actor_relative_value",
    "derive_nested_strategic_determinization_seed",
    "derive_nested_strategic_response_seed",
    "derive_nested_strategic_search_request_seed",
    "derive_nested_strategic_selection_seed",
)

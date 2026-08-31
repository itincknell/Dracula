"""Information-set UCT with actor-local belief-greedy continuations."""

from __future__ import annotations

import hashlib
import json
import math
import resource
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from dracula.bridge import ACTION_COUNT, action_index_for_move, move_for_action_index
from dracula.engine import (
    EngineMove,
    EnginePlayer,
    EngineStatus,
    SimulationEngineState,
    apply_simulation_move,
    legal_simulation_moves,
    resolve_round_scores,
    score_coffin,
)
from dracula.models import POLICY_GRID_INDICES
from dracula.randomness import Sha256CounterStream, derive_seed, seed_hex, shuffled
from dracula.search.information import (
    SearchInformationState,
    information_state_fingerprint,
    project_simulation_information_state,
    sample_determinization,
)
from dracula.search.planner import (
    ROUND_SCORE_NORMALIZER,
    ContinuationStep,
    PrincipalContinuation,
    SearchContractViolation,
    SearchInterrupted,
    SearchResult,
    normalized_round_return,
)
from dracula.search.strategic import (
    StrategicActionGroup,
    StrategicActionGroupDiagnostic,
    actor_relative_value,
    derive_strategic_destination_choice_seed,
    select_concrete_action_index,
    strategic_action_groups,
)
from dracula.search.symmetry import DESTINATION_SYMMETRY_SCHEMA_VERSION

BELIEF_GREEDY_SEARCH_SCHEMA_VERSION = "dracula-belief-greedy-search-v1"
BELIEF_GREEDY_RESPONSE_SCHEMA_VERSION = "dracula-belief-greedy-response-v1"
BELIEF_GREEDY_REQUEST_NAMESPACE = "dracula-belief-greedy-request-v1"
BELIEF_GREEDY_DETERMINIZATION_NAMESPACE = (
    "dracula-belief-greedy-determinization-v1"
)
BELIEF_GREEDY_SELECTION_NAMESPACE = "dracula-belief-greedy-selection-v1"
BELIEF_GREEDY_RESPONSE_REQUEST_NAMESPACE = (
    "dracula-belief-greedy-response-request-v1"
)
BELIEF_GREEDY_HAND_SAMPLE_NAMESPACE = "dracula-belief-greedy-hand-sample-v1"
BELIEF_GREEDY_COMPLETION_ORDER_NAMESPACE = (
    "dracula-belief-greedy-completion-order-v1"
)
BELIEF_GREEDY_SELECTION_PROFILE = "max-visits-mean-value-representative-v1"
BELIEF_GREEDY_RESPONSE_PROFILE = "max-expected-round-differential-v1"


def _canonical_digest(value: object) -> str:
    encoded = json.dumps(
        value,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class BeliefGreedySearchConfig:
    outer_simulation_budget: int = 32
    belief_completion_count: int = 8
    outer_exploration_constant: float = math.sqrt(2.0)
    destination_symmetry_enabled: bool = True

    def __post_init__(self) -> None:
        if (
            type(self.outer_simulation_budget) is not int
            or self.outer_simulation_budget < 1
        ):
            raise SearchContractViolation(
                "outer simulation budget must be a positive integer"
            )
        if (
            type(self.belief_completion_count) is not int
            or self.belief_completion_count < 1
        ):
            raise SearchContractViolation(
                "belief completion count must be a positive integer"
            )
        if (
            type(self.outer_exploration_constant) not in (int, float)
            or not math.isfinite(self.outer_exploration_constant)
            or self.outer_exploration_constant < 0
        ):
            raise SearchContractViolation(
                "outer exploration constant must be finite and non-negative"
            )
        if type(self.destination_symmetry_enabled) is not bool:
            raise SearchContractViolation(
                "destination symmetry flag must be Boolean"
            )

    @property
    def response_digest(self) -> str:
        return _canonical_digest(
            {
                "belief_completion_count": self.belief_completion_count,
                "destination_symmetry_schema_version": (
                    DESTINATION_SYMMETRY_SCHEMA_VERSION
                ),
                "response_profile": BELIEF_GREEDY_RESPONSE_PROFILE,
                "response_schema_version": (
                    BELIEF_GREEDY_RESPONSE_SCHEMA_VERSION
                ),
                "terminal_value": "exact-engine-round-differential-v1",
            }
        )

    @property
    def digest(self) -> str:
        return _canonical_digest(
            {
                "destination_symmetry_enabled": (
                    self.destination_symmetry_enabled
                ),
                "outer_exploration_constant": float(
                    self.outer_exploration_constant
                ),
                "outer_simulation_budget": self.outer_simulation_budget,
                "response_config_digest": self.response_digest,
                "search_schema_version": BELIEF_GREEDY_SEARCH_SCHEMA_VERSION,
                "selection_profile": BELIEF_GREEDY_SELECTION_PROFILE,
            }
        )


@dataclass(frozen=True, slots=True)
class BeliefGreedyNodeKey:
    actor: EnginePlayer
    information_state_fingerprint: str


@dataclass(frozen=True, slots=True)
class BeliefGreedyResponseResult:
    information_state_fingerprint: str
    config_digest: str
    selected_action_index: int
    selected_representative_action_index: int
    group_diagnostics: tuple[StrategicActionGroupDiagnostic, ...]
    belief_completion_count: int
    potential_evaluation_count: int

    def __post_init__(self) -> None:
        representatives = {
            diagnostic.group.representative_action_index
            for diagnostic in self.group_diagnostics
        }
        if self.selected_representative_action_index not in representatives:
            raise SearchContractViolation(
                "belief-greedy response selected an unknown strategic group"
            )
        selected_group = next(
            diagnostic.group
            for diagnostic in self.group_diagnostics
            if diagnostic.group.representative_action_index
            == self.selected_representative_action_index
        )
        if self.selected_action_index not in selected_group.member_action_indices:
            raise SearchContractViolation(
                "belief-greedy concrete action left its strategic group"
            )
        if (
            type(self.belief_completion_count) is not int
            or self.belief_completion_count < 1
            or self.potential_evaluation_count
            != len(self.group_diagnostics) * self.belief_completion_count
        ):
            raise SearchContractViolation(
                "belief-greedy potential accounting is inconsistent"
            )
        if any(
            diagnostic.visits != self.belief_completion_count
            or diagnostic.mean_value is None
            for diagnostic in self.group_diagnostics
        ):
            raise SearchContractViolation(
                "every belief-greedy group requires one value per completion"
            )


@dataclass(frozen=True, slots=True)
class BeliefGreedySearchResult(SearchResult):
    elapsed_seconds: float = field(compare=False)
    peak_resident_memory_bytes: int = field(compare=False)
    response_request_count: int
    unique_response_evaluation_count: int
    response_cache_hit_count: int
    response_candidate_action_count: int
    response_potential_evaluation_count: int
    selected_representative_action_index: int
    group_diagnostics: tuple[StrategicActionGroupDiagnostic, ...]

    def __post_init__(self) -> None:
        super(BeliefGreedySearchResult, self).__post_init__()
        if not math.isfinite(self.elapsed_seconds) or self.elapsed_seconds < 0:
            raise SearchContractViolation(
                "belief-greedy latency must be finite and non-negative"
            )
        if (
            type(self.peak_resident_memory_bytes) is not int
            or self.peak_resident_memory_bytes < 0
        ):
            raise SearchContractViolation(
                "belief-greedy peak memory must be non-negative"
            )
        counts = (
            self.response_request_count,
            self.unique_response_evaluation_count,
            self.response_cache_hit_count,
            self.response_candidate_action_count,
            self.response_potential_evaluation_count,
        )
        if any(type(value) is not int or value < 0 for value in counts):
            raise SearchContractViolation(
                "belief-greedy diagnostics must be non-negative integers"
            )
        if self.response_request_count != (
            self.unique_response_evaluation_count
            + self.response_cache_hit_count
        ):
            raise SearchContractViolation(
                "belief-greedy response cache accounting is inconsistent"
            )
        if self.simulation_count == 0:
            if (
                self.group_diagnostics
                or self.selected_representative_action_index
                != self.selected_action_index
            ):
                raise SearchContractViolation(
                    "a forced belief-greedy result cannot contain groups"
                )
            return
        representatives = {
            diagnostic.group.representative_action_index
            for diagnostic in self.group_diagnostics
        }
        if (
            self.selected_representative_action_index not in representatives
            or sum(
                diagnostic.visits for diagnostic in self.group_diagnostics
            )
            != self.simulation_count
        ):
            raise SearchContractViolation(
                "belief-greedy root group accounting is inconsistent"
            )


@dataclass(slots=True)
class _GroupNode:
    key: BeliefGreedyNodeKey
    groups: tuple[StrategicActionGroup, ...]
    visits: int
    action_visits: list[int]
    action_value_sums: list[float]

    @property
    def representatives(self) -> tuple[int, ...]:
        return tuple(group.representative_action_index for group in self.groups)

    @classmethod
    def create(
        cls,
        key: BeliefGreedyNodeKey,
        groups: tuple[StrategicActionGroup, ...],
    ) -> _GroupNode:
        return cls(key, groups, 0, [0] * ACTION_COUNT, [0.0] * ACTION_COUNT)


@dataclass(frozen=True, slots=True)
class _ObservedContinuation:
    root_representative_action_index: int
    root_concrete_action_index: int
    terminal_value: float
    simulation_index: int
    steps: tuple[ContinuationStep, ...]


def _counter(value: int, label: str) -> str:
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return str(value)


def derive_belief_greedy_request_seed(
    fixture_id: str,
    information: SearchInformationState,
    config_digest: str,
) -> bytes:
    return derive_seed(
        BELIEF_GREEDY_REQUEST_NAMESPACE,
        fixture_id,
        information_state_fingerprint(information),
        information.player.value,
        str(information.round_number),
        str(information.turn_number),
        config_digest,
    )


def derive_belief_greedy_determinization_seed(
    request_seed: bytes,
    simulation_index: int,
) -> bytes:
    return derive_seed(
        BELIEF_GREEDY_DETERMINIZATION_NAMESPACE,
        seed_hex(request_seed),
        _counter(simulation_index, "simulation index"),
    )


def derive_belief_greedy_selection_seed(
    request_seed: bytes,
    simulation_index: int,
    node: BeliefGreedyNodeKey,
) -> bytes:
    return derive_seed(
        BELIEF_GREEDY_SELECTION_NAMESPACE,
        seed_hex(request_seed),
        _counter(simulation_index, "simulation index"),
        node.actor.value,
        node.information_state_fingerprint,
    )


def derive_belief_greedy_response_seed(
    information: SearchInformationState,
    response_config_digest: str,
) -> bytes:
    return derive_seed(
        BELIEF_GREEDY_RESPONSE_REQUEST_NAMESPACE,
        information_state_fingerprint(information),
        response_config_digest,
    )


def _select_tree_group(
    node: _GroupNode,
    exploration_constant: float,
    selection_seed: bytes,
) -> int:
    for representative in node.representatives:
        if node.action_visits[representative] == 0:
            return representative
    log_parent = math.log(node.visits)
    best = -math.inf
    tied: list[int] = []
    for representative in node.representatives:
        visits = node.action_visits[representative]
        mean = node.action_value_sums[representative] / visits
        score = mean + exploration_constant * math.sqrt(log_parent / visits)
        if score > best:
            best = score
            tied = [representative]
        elif score == best:
            tied.append(representative)
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
    move = move_for_action_index(actor, action_index)
    if move not in legal_simulation_moves(state, actor):
        raise SearchContractViolation(
            "belief-greedy search selected a masked action"
        )
    return move


def _completed_relative_value(
    coffin: tuple[str, ...],
    actor: EnginePlayer,
) -> float:
    lines = score_coffin(coffin)
    scores = resolve_round_scores(
        tuple(line.total for line in lines.queen),
        tuple(line.total for line in lines.king),
    )
    difference = (
        scores.queen - scores.king
        if actor is EnginePlayer.QUEEN
        else scores.king - scores.queen
    )
    return difference / ROUND_SCORE_NORMALIZER


def _completion_card_order(
    cards: tuple[str, ...],
    completion_seed: bytes,
) -> tuple[str, ...]:
    return tuple(
        sorted(
            cards,
            key=lambda card_id: derive_seed(
                BELIEF_GREEDY_COMPLETION_ORDER_NAMESPACE,
                seed_hex(completion_seed),
                card_id,
            ),
        )
    )


def _hypothetical_value(
    information: SearchInformationState,
    representative_action_index: int,
    opponent_hand: tuple[str, ...],
    completion_seed: bytes,
) -> float:
    hand_slot = representative_action_index // len(POLICY_GRID_INDICES)
    destination = POLICY_GRID_INDICES[
        representative_action_index % len(POLICY_GRID_INDICES)
    ]
    card_id = information.own_hand[hand_slot]
    if card_id is None or not information.legal_mask[hand_slot][
        representative_action_index % len(POLICY_GRID_INDICES)
    ]:
        raise SearchContractViolation(
            "belief-greedy potential received an illegal representative"
        )
    coffin = list(information.coffin)
    coffin[destination] = card_id
    own_remaining = tuple(
        candidate
        for index, candidate in enumerate(information.own_hand)
        if index != hand_slot and candidate is not None
    )
    empty_indices = tuple(
        index for index, candidate in enumerate(coffin) if candidate is None
    )
    future_cards = own_remaining + opponent_hand
    if len(future_cards) != len(empty_indices):
        raise SearchContractViolation(
            "belief completion does not conserve current-round cards"
        )
    for index, future_card in zip(
        empty_indices,
        _completion_card_order(future_cards, completion_seed),
        strict=True,
    ):
        coffin[index] = future_card
    if any(candidate is None for candidate in coffin):
        raise SearchContractViolation(
            "belief completion did not fill the coffin"
        )
    return _completed_relative_value(
        tuple(coffin),  # type: ignore[arg-type]
        information.player,
    )


class BeliefGreedyResponseEvaluator:
    """Greedily rank actor actions under shared hidden-hand probabilities."""

    def __init__(self, config: BeliefGreedySearchConfig) -> None:
        if not isinstance(config, BeliefGreedySearchConfig):
            raise SearchContractViolation(
                "belief-greedy response configuration is invalid"
            )
        self.config = config

    def evaluate(
        self,
        information: SearchInformationState,
        should_stop: Callable[[], bool] | None = None,
    ) -> BeliefGreedyResponseResult:
        if not isinstance(information, SearchInformationState):
            raise SearchContractViolation(
                "belief-greedy response requires an information state"
            )
        groups = strategic_action_groups(
            information,
            self.config.destination_symmetry_enabled,
        )
        if not groups:
            raise SearchContractViolation(
                "belief-greedy response has no strategic groups"
            )
        request_seed = derive_belief_greedy_response_seed(
            information,
            self.config.response_digest,
        )
        value_sums = [0.0] * ACTION_COUNT
        visits = [0] * ACTION_COUNT
        for completion_index in range(self.config.belief_completion_count):
            _check_interrupted(should_stop)
            completion_seed = derive_seed(
                BELIEF_GREEDY_HAND_SAMPLE_NAMESPACE,
                seed_hex(request_seed),
                _counter(completion_index, "completion index"),
            )
            opponent_hand = tuple(
                sorted(
                    shuffled(information.unseen_card_ids, completion_seed)[
                        : information.opponent_remaining_count
                    ]
                )
            )
            for group in groups:
                _check_interrupted(should_stop)
                representative = group.representative_action_index
                value_sums[representative] += _hypothetical_value(
                    information,
                    representative,
                    opponent_hand,
                    completion_seed,
                )
                visits[representative] += 1
        diagnostics = tuple(
            StrategicActionGroupDiagnostic(
                group,
                visits[group.representative_action_index],
                value_sums[group.representative_action_index]
                / visits[group.representative_action_index],
            )
            for group in groups
        )
        selected_representative = min(
            (group.representative_action_index for group in groups),
            key=lambda action_index: (
                -value_sums[action_index] / visits[action_index],
                action_index,
            ),
        )
        selected_action = select_concrete_action_index(
            information,
            selected_representative,
            derive_strategic_destination_choice_seed(
                request_seed,
                "belief-greedy-response-result",
                information,
                selected_representative,
                0,
            ),
            self.config.destination_symmetry_enabled,
        )
        return BeliefGreedyResponseResult(
            information_state_fingerprint=information_state_fingerprint(
                information
            ),
            config_digest=self.config.response_digest,
            selected_action_index=selected_action,
            selected_representative_action_index=selected_representative,
            group_diagnostics=diagnostics,
            belief_completion_count=self.config.belief_completion_count,
            potential_evaluation_count=(
                len(groups) * self.config.belief_completion_count
            ),
        )


def _group_diagnostics(
    groups: tuple[StrategicActionGroup, ...],
    visits: list[int],
    value_sums: list[float],
) -> tuple[StrategicActionGroupDiagnostic, ...]:
    return tuple(
        StrategicActionGroupDiagnostic(
            group,
            visits[group.representative_action_index],
            (
                value_sums[group.representative_action_index]
                / visits[group.representative_action_index]
                if visits[group.representative_action_index]
                else None
            ),
        )
        for group in groups
    )


def _expanded_means(
    diagnostics: tuple[StrategicActionGroupDiagnostic, ...],
) -> tuple[float | None, ...]:
    result: list[float | None] = [None] * ACTION_COUNT
    for diagnostic in diagnostics:
        for member in diagnostic.group.member_action_indices:
            result[member] = diagnostic.mean_value
    return tuple(result)


def _project_visits(
    information: SearchInformationState,
    request_seed: bytes,
    diagnostics: tuple[StrategicActionGroupDiagnostic, ...],
    symmetry: bool,
) -> tuple[tuple[int, ...], dict[int, int]]:
    visits = [0] * ACTION_COUNT
    concrete: dict[int, int] = {}
    for diagnostic in diagnostics:
        representative = diagnostic.group.representative_action_index
        selected = select_concrete_action_index(
            information,
            representative,
            derive_strategic_destination_choice_seed(
                request_seed,
                "belief-greedy-root-result",
                information,
                representative,
                0,
            ),
            symmetry,
        )
        visits[selected] = diagnostic.visits
        concrete[representative] = selected
    return tuple(visits), concrete


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
            "belief-greedy continuation selected an empty hand slot"
        )
    return ContinuationStep(
        actor="self" if move.player is root_player else "opponent",
        action_index=action_index,
        card_id=card_id,
        grid_index=move.global_grid_index,
        forced=forced,
    )


def _peak_resident_memory_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _check_interrupted(
    should_stop: Callable[[], bool] | None,
) -> None:
    if should_stop is not None and should_stop():
        raise SearchInterrupted(
            "belief-greedy search interrupted before result commit"
        )


class BeliefGreedyInformationSetSearch:
    """Outer information-set UCT with fixed belief-greedy actor responses."""

    def __init__(
        self,
        config: BeliefGreedySearchConfig = BeliefGreedySearchConfig(),
    ) -> None:
        if not isinstance(config, BeliefGreedySearchConfig):
            raise SearchContractViolation(
                "belief-greedy search configuration is invalid"
            )
        self.config = config
        self._response = BeliefGreedyResponseEvaluator(config)

    def _actor_response(
        self,
        information: SearchInformationState,
        should_stop: Callable[[], bool] | None,
    ) -> BeliefGreedyResponseResult:
        return self._response.evaluate(information, should_stop)

    def search(
        self,
        information: SearchInformationState,
        request_seed: bytes,
        should_stop: Callable[[], bool] | None = None,
    ) -> BeliefGreedySearchResult:
        started = time.perf_counter()
        if not isinstance(information, SearchInformationState):
            raise SearchContractViolation(
                "belief-greedy search requires an information state"
            )
        seed_hex(request_seed)
        root_player = information.player
        root_fingerprint = information_state_fingerprint(information)
        root_groups = strategic_action_groups(
            information,
            self.config.destination_symmetry_enabled,
        )
        if not root_groups:
            raise SearchContractViolation(
                "belief-greedy search root has no strategic actions"
            )
        if len(root_groups) == 1:
            representative = root_groups[0].representative_action_index
            selected = select_concrete_action_index(
                information,
                representative,
                derive_strategic_destination_choice_seed(
                    request_seed,
                    "belief-greedy-forced-result",
                    information,
                    representative,
                    0,
                ),
                self.config.destination_symmetry_enabled,
            )
            return BeliefGreedySearchResult(
                information_state_fingerprint=root_fingerprint,
                config_digest=self.config.digest,
                selected_action_index=selected,
                action_visits=tuple(0 for _ in range(ACTION_COUNT)),
                mean_action_values=tuple(None for _ in range(ACTION_COUNT)),
                simulation_count=0,
                information_set_count=0,
                principal_continuation=None,
                elapsed_seconds=time.perf_counter() - started,
                peak_resident_memory_bytes=_peak_resident_memory_bytes(),
                response_request_count=0,
                unique_response_evaluation_count=0,
                response_cache_hit_count=0,
                response_candidate_action_count=0,
                response_potential_evaluation_count=0,
                selected_representative_action_index=selected,
                group_diagnostics=(),
            )
        if self.config.outer_simulation_budget < len(root_groups):
            raise SearchContractViolation(
                "outer simulation budget must visit every strategic root action"
            )

        nodes: dict[BeliefGreedyNodeKey, _GroupNode] = {}
        response_cache: dict[str, BeliefGreedyResponseResult] = {}
        continuations: list[_ObservedContinuation] = []
        response_requests = 0
        response_cache_hits = 0
        response_candidates = 0
        response_potentials = 0

        for simulation_index in range(self.config.outer_simulation_budget):
            _check_interrupted(should_stop)
            state = sample_determinization(
                information,
                derive_belief_greedy_determinization_seed(
                    request_seed,
                    simulation_index,
                ),
            ).state
            path: list[tuple[_GroupNode, int]] = []
            steps: list[ContinuationStep] = []
            expanded = False
            root_representative: int | None = None
            root_concrete: int | None = None

            while state.status is EngineStatus.PLAYING:
                _check_interrupted(should_stop)
                actor = state.active_player
                if actor is None:
                    raise SearchContractViolation(
                        "belief-greedy simulation lost its active player"
                    )
                moves = legal_simulation_moves(state, actor)
                forced = len(moves) == 1
                if forced:
                    move = moves[0]
                    action_index = action_index_for_move(move, actor)
                    representative = action_index
                elif actor is root_player and not expanded:
                    actor_information = project_simulation_information_state(
                        state
                    )
                    node_key = BeliefGreedyNodeKey(
                        actor,
                        information_state_fingerprint(actor_information),
                    )
                    groups = strategic_action_groups(
                        actor_information,
                        self.config.destination_symmetry_enabled,
                    )
                    node = nodes.get(node_key)
                    if node is None:
                        node = _GroupNode.create(node_key, groups)
                        nodes[node_key] = node
                    elif node.groups != groups:
                        raise SearchContractViolation(
                            "belief-greedy node changed its strategic groups"
                        )
                    representative = _select_tree_group(
                        node,
                        float(self.config.outer_exploration_constant),
                        derive_belief_greedy_selection_seed(
                            request_seed,
                            simulation_index,
                            node_key,
                        ),
                    )
                    path.append((node, representative))
                    if node.action_visits[representative] == 0:
                        expanded = True
                    action_index = select_concrete_action_index(
                        actor_information,
                        representative,
                        derive_strategic_destination_choice_seed(
                            request_seed,
                            "belief-greedy-outer-simulation",
                            actor_information,
                            representative,
                            simulation_index,
                        ),
                        self.config.destination_symmetry_enabled,
                    )
                    move = _move_for_action(state, action_index)
                else:
                    actor_information = project_simulation_information_state(
                        state
                    )
                    response_key = information_state_fingerprint(
                        actor_information
                    )
                    response_requests += 1
                    response = response_cache.get(response_key)
                    if response is None:
                        response = self._actor_response(
                            actor_information,
                            should_stop,
                        )
                        if (
                            response.information_state_fingerprint
                            != response_key
                            or response.config_digest
                            != self.config.response_digest
                        ):
                            raise SearchContractViolation(
                                "belief-greedy response crossed its "
                                "information boundary"
                            )
                        response_cache[response_key] = response
                        response_candidates += len(response.group_diagnostics)
                        response_potentials += response.potential_evaluation_count
                    else:
                        response_cache_hits += 1
                    representative = (
                        response.selected_representative_action_index
                    )
                    action_index = response.selected_action_index
                    move = _move_for_action(state, action_index)

                if root_representative is None:
                    if actor is not root_player:
                        raise SearchContractViolation(
                            "belief-greedy root simulation began on another actor"
                        )
                    root_representative = representative
                    root_concrete = action_index
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
                or root_representative is None
                or root_concrete is None
                or not path
            ):
                raise SearchContractViolation(
                    "belief-greedy simulation did not reach a backed terminal"
                )
            root_value = normalized_round_return(
                state.pending_round_result,
                root_player,
            )
            for node, representative in path:
                value = actor_relative_value(
                    root_value,
                    root_player,
                    node.key.actor,
                )
                node.visits += 1
                node.action_visits[representative] += 1
                node.action_value_sums[representative] += value
            continuations.append(
                _ObservedContinuation(
                    root_representative,
                    root_concrete,
                    root_value,
                    simulation_index,
                    tuple(steps),
                )
            )

        root_key = BeliefGreedyNodeKey(root_player, root_fingerprint)
        root_node = nodes.get(root_key)
        if (
            root_node is None
            or root_node.visits != self.config.outer_simulation_budget
        ):
            raise SearchContractViolation(
                "belief-greedy root backup count does not match simulations"
            )
        selected_representative = min(
            root_node.representatives,
            key=lambda action_index: (
                -root_node.action_visits[action_index],
                -root_node.action_value_sums[action_index]
                / root_node.action_visits[action_index],
                action_index,
            ),
        )
        observed = min(
            (
                continuation
                for continuation in continuations
                if continuation.root_representative_action_index
                == selected_representative
            ),
            key=lambda continuation: (
                -continuation.terminal_value,
                continuation.simulation_index,
            ),
        )
        diagnostics = _group_diagnostics(
            root_node.groups,
            root_node.action_visits,
            root_node.action_value_sums,
        )
        projected_visits, concrete = _project_visits(
            information,
            request_seed,
            diagnostics,
            self.config.destination_symmetry_enabled,
        )
        selected = concrete[selected_representative]
        return BeliefGreedySearchResult(
            information_state_fingerprint=root_fingerprint,
            config_digest=self.config.digest,
            selected_action_index=selected,
            action_visits=projected_visits,
            mean_action_values=_expanded_means(diagnostics),
            simulation_count=root_node.visits,
            information_set_count=len(nodes),
            principal_continuation=PrincipalContinuation(
                observed.root_concrete_action_index,
                observed.terminal_value,
                observed.simulation_index,
                observed.steps,
            ),
            elapsed_seconds=time.perf_counter() - started,
            peak_resident_memory_bytes=_peak_resident_memory_bytes(),
            response_request_count=response_requests,
            unique_response_evaluation_count=len(response_cache),
            response_cache_hit_count=response_cache_hits,
            response_candidate_action_count=response_candidates,
            response_potential_evaluation_count=response_potentials,
            selected_representative_action_index=selected_representative,
            group_diagnostics=diagnostics,
        )


__all__ = (
    "BELIEF_GREEDY_COMPLETION_ORDER_NAMESPACE",
    "BELIEF_GREEDY_DETERMINIZATION_NAMESPACE",
    "BELIEF_GREEDY_HAND_SAMPLE_NAMESPACE",
    "BELIEF_GREEDY_REQUEST_NAMESPACE",
    "BELIEF_GREEDY_RESPONSE_REQUEST_NAMESPACE",
    "BELIEF_GREEDY_RESPONSE_SCHEMA_VERSION",
    "BELIEF_GREEDY_SEARCH_SCHEMA_VERSION",
    "BELIEF_GREEDY_SELECTION_NAMESPACE",
    "BeliefGreedyInformationSetSearch",
    "BeliefGreedyNodeKey",
    "BeliefGreedyResponseEvaluator",
    "BeliefGreedyResponseResult",
    "BeliefGreedySearchConfig",
    "BeliefGreedySearchResult",
    "derive_belief_greedy_determinization_seed",
    "derive_belief_greedy_request_seed",
    "derive_belief_greedy_response_seed",
    "derive_belief_greedy_selection_seed",
)

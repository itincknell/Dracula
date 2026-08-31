"""Symmetry-aware Sam-128 dataset teacher."""

from __future__ import annotations

import hashlib
import json
import math
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
)
from dracula.randomness import Sha256CounterStream, derive_seed, seed_hex
from dracula.search.information import (
    SearchInformationState,
    information_state_fingerprint,
    project_simulation_information_state,
    sample_determinization,
)
from dracula.search.planner import (
    ContinuationStep,
    PrincipalContinuation,
    SearchContractViolation,
    SearchInterrupted,
    SearchResult,
    normalized_round_return,
)
from dracula.search.strategic import (
    STRATEGIC_DESTINATION_CHOICE_PROFILE,
    StrategicActionGroup,
    StrategicActionGroupDiagnostic,
    actor_relative_value,
    derive_strategic_destination_choice_seed,
    select_concrete_action_index,
    strategic_action_groups,
)
from dracula.search.symmetry import DESTINATION_SYMMETRY_SCHEMA_VERSION

SAM_TEACHER_SEARCH_SCHEMA_VERSION = "dracula-sam-128-search-v1"
SAM_TEACHER_RESPONSE_SCHEMA_VERSION = "dracula-sam-128-response-search-v1"
SAM_TEACHER_REQUEST_NAMESPACE = "dracula-sam-128-request-v1"
SAM_TEACHER_DETERMINIZATION_NAMESPACE = (
    "dracula-sam-128-determinization-v1"
)
SAM_TEACHER_SELECTION_NAMESPACE = "dracula-sam-128-selection-v1"
SAM_TEACHER_RESPONSE_REQUEST_NAMESPACE = (
    "dracula-sam-128-response-request-v1"
)
SAM_TEACHER_RESPONSE_DETERMINIZATION_NAMESPACE = (
    "dracula-sam-128-response-determinization-v1"
)
SAM_TEACHER_RESPONSE_SELECTION_NAMESPACE = (
    "dracula-sam-128-response-selection-v1"
)
SAM_TEACHER_RESPONSE_ROLLOUT_NAMESPACE = (
    "dracula-sam-128-response-rollout-v1"
)
SAM_TEACHER_SELECTION_PROFILE = "max-visits-mean-value-representative-v1"
SAM_TEACHER_RESPONSE_SELECTION_PROFILE = (
    "max-visits-mean-value-representative-v1"
)
SAM_TEACHER_TERMINAL_PROFILE = "exact-normalized-round-differential-v1"
SAM_TEACHER_RECURSION_LIMIT = 1


@dataclass(frozen=True, slots=True)
class SamTeacherSearchConfig:
    outer_simulation_budget: int = 128
    response_simulation_budget: int = 128
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
    def response_digest(self) -> str:
        payload = json.dumps(
            {
                "destination_choice_profile": (
                    STRATEGIC_DESTINATION_CHOICE_PROFILE
                ),
                "destination_symmetry_schema_version": (
                    DESTINATION_SYMMETRY_SCHEMA_VERSION
                ),
                "exploration_constant": float(
                    self.response_exploration_constant
                ),
                "response_schema_version": (
                    SAM_TEACHER_RESPONSE_SCHEMA_VERSION
                ),
                "selection_profile": (
                    SAM_TEACHER_RESPONSE_SELECTION_PROFILE
                ),
                "simulation_budget": self.response_simulation_budget,
                "terminal_value": SAM_TEACHER_TERMINAL_PROFILE,
            },
            separators=(",", ":"),
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @property
    def digest(self) -> str:
        payload = json.dumps(
            {
                "destination_choice_profile": (
                    STRATEGIC_DESTINATION_CHOICE_PROFILE
                ),
                "destination_symmetry_schema_version": (
                    DESTINATION_SYMMETRY_SCHEMA_VERSION
                ),
                "outer_exploration_constant": float(
                    self.outer_exploration_constant
                ),
                "outer_simulation_budget": self.outer_simulation_budget,
                "recursion_limit": SAM_TEACHER_RECURSION_LIMIT,
                "response_config_digest": self.response_digest,
                "response_simulation_budget": (
                    self.response_simulation_budget
                ),
                "search_schema_version": SAM_TEACHER_SEARCH_SCHEMA_VERSION,
                "selection_profile": SAM_TEACHER_SELECTION_PROFILE,
                "terminal_value": SAM_TEACHER_TERMINAL_PROFILE,
            },
            separators=(",", ":"),
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class SamTeacherNodeKey:
    actor: EnginePlayer
    information_state_fingerprint: str


@dataclass(frozen=True, slots=True)
class SamTeacherSearchResult(SearchResult):
    elapsed_seconds: float = field(compare=False)
    response_request_count: int
    unique_response_search_count: int
    response_cache_hit_count: int
    response_simulation_count: int
    response_information_set_count: int
    total_terminal_evaluation_count: int
    selected_representative_action_index: int
    group_diagnostics: tuple[StrategicActionGroupDiagnostic, ...]

    @property
    def selected_group(self) -> StrategicActionGroup | None:
        for diagnostic in self.group_diagnostics:
            if (
                diagnostic.group.representative_action_index
                == self.selected_representative_action_index
            ):
                return diagnostic.group
        return None

    def __post_init__(self) -> None:
        super(SamTeacherSearchResult, self).__post_init__()
        if (
            not math.isfinite(self.elapsed_seconds)
            or self.elapsed_seconds < 0
        ):
            raise SearchContractViolation(
                "Sam-128 latency must be finite and non-negative"
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
                "Sam-128 diagnostics must be non-negative integers"
            )
        if self.response_request_count != (
            self.unique_response_search_count
            + self.response_cache_hit_count
        ):
            raise SearchContractViolation(
                "Sam-128 response cache accounting is inconsistent"
            )
        if self.total_terminal_evaluation_count != (
            self.simulation_count + self.response_simulation_count
        ):
            raise SearchContractViolation(
                "Sam-128 terminal accounting is inconsistent"
            )
        if self.simulation_count == 0:
            if (
                self.group_diagnostics
                or self.selected_representative_action_index
                != self.selected_action_index
            ):
                raise SearchContractViolation(
                    "a forced Sam-128 result cannot contain a strategic group"
                )
            return

        selected_group = self.selected_group
        if (
            selected_group is None
            or self.selected_action_index
            not in selected_group.member_action_indices
            or sum(
                diagnostic.visits for diagnostic in self.group_diagnostics
            )
            != self.simulation_count
        ):
            raise SearchContractViolation(
                "Sam-128 selected group or visit accounting is inconsistent"
            )
        grouped_members = {
            action_index
            for diagnostic in self.group_diagnostics
            for action_index in diagnostic.group.member_action_indices
        }
        for diagnostic in self.group_diagnostics:
            member_visits = tuple(
                self.action_visits[action_index]
                for action_index in diagnostic.group.member_action_indices
            )
            if (
                sum(member_visits) != diagnostic.visits
                or sum(count > 0 for count in member_visits) != 1
            ):
                raise SearchContractViolation(
                    "one concrete member must carry each pooled group visit"
                )
            if any(
                self.mean_action_values[action_index]
                != diagnostic.mean_value
                for action_index in diagnostic.group.member_action_indices
            ):
                raise SearchContractViolation(
                    "mirrored Sam-128 actions must share one group value"
                )
        if any(
            visits
            for action_index, visits in enumerate(self.action_visits)
            if action_index not in grouped_members
        ):
            raise SearchContractViolation(
                "Sam-128 visits cannot leave strategic action groups"
            )


@dataclass(slots=True)
class _GroupNode:
    key: SamTeacherNodeKey
    action_groups: tuple[StrategicActionGroup, ...]
    visits: int
    action_visits: list[int]
    action_value_sums: list[float]

    @property
    def representative_actions(self) -> tuple[int, ...]:
        return tuple(
            group.representative_action_index for group in self.action_groups
        )

    @classmethod
    def create(
        cls,
        key: SamTeacherNodeKey,
        action_groups: tuple[StrategicActionGroup, ...],
    ) -> _GroupNode:
        return cls(
            key,
            action_groups,
            0,
            [0] * ACTION_COUNT,
            [0.0] * ACTION_COUNT,
        )


@dataclass(frozen=True, slots=True)
class _ObservedContinuation:
    root_representative_action_index: int
    root_concrete_action_index: int
    terminal_value: float
    simulation_index: int
    steps: tuple[ContinuationStep, ...]


@dataclass(frozen=True, slots=True)
class _ActorResponseResult:
    information_state_fingerprint: str
    config_digest: str
    selected_action_index: int
    selected_representative_action_index: int
    simulation_count: int
    information_set_count: int
    group_diagnostics: tuple[StrategicActionGroupDiagnostic, ...]


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


def sam_teacher_action_groups(
    information: SearchInformationState,
) -> tuple[StrategicActionGroup, ...]:
    """Return only the authoritative symmetry-aware strategic actions."""

    return strategic_action_groups(
        information,
        destination_symmetry_enabled=True,
    )


def derive_sam_teacher_request_seed(
    information: SearchInformationState,
    search_config_digest: str,
) -> bytes:
    return derive_seed(
        SAM_TEACHER_REQUEST_NAMESPACE,
        information_state_fingerprint(information),
        search_config_digest,
    )


def derive_sam_teacher_determinization_seed(
    request_seed: bytes,
    simulation_index: int,
) -> bytes:
    return derive_seed(
        SAM_TEACHER_DETERMINIZATION_NAMESPACE,
        seed_hex(request_seed),
        _validate_counter(simulation_index, "simulation index"),
    )


def derive_sam_teacher_selection_seed(
    request_seed: bytes,
    simulation_index: int,
    node: SamTeacherNodeKey,
) -> bytes:
    return derive_seed(
        SAM_TEACHER_SELECTION_NAMESPACE,
        seed_hex(request_seed),
        _validate_counter(simulation_index, "simulation index"),
        node.actor.value,
        node.information_state_fingerprint,
    )


def derive_sam_teacher_response_request_seed(
    information: SearchInformationState,
    response_config_digest: str,
) -> bytes:
    return derive_seed(
        SAM_TEACHER_RESPONSE_REQUEST_NAMESPACE,
        information_state_fingerprint(information),
        response_config_digest,
    )


def derive_sam_teacher_response_determinization_seed(
    response_request_seed: bytes,
    simulation_index: int,
) -> bytes:
    return derive_seed(
        SAM_TEACHER_RESPONSE_DETERMINIZATION_NAMESPACE,
        seed_hex(response_request_seed),
        _validate_counter(simulation_index, "simulation index"),
    )


def derive_sam_teacher_response_selection_seed(
    response_request_seed: bytes,
    simulation_index: int,
    node: SamTeacherNodeKey,
) -> bytes:
    return derive_seed(
        SAM_TEACHER_RESPONSE_SELECTION_NAMESPACE,
        seed_hex(response_request_seed),
        _validate_counter(simulation_index, "simulation index"),
        node.actor.value,
        node.information_state_fingerprint,
    )


def derive_sam_teacher_response_rollout_seed(
    response_request_seed: bytes,
    simulation_index: int,
    rollout_ply: int,
    information: SearchInformationState,
) -> bytes:
    return derive_seed(
        SAM_TEACHER_RESPONSE_ROLLOUT_NAMESPACE,
        seed_hex(response_request_seed),
        _validate_counter(simulation_index, "simulation index"),
        _validate_counter(rollout_ply, "rollout ply"),
        information_state_fingerprint(information),
    )


def derive_sam_teacher_destination_seed(
    request_seed: bytes,
    scope: str,
    information: SearchInformationState,
    representative_action_index: int,
    choice_index: int,
) -> bytes:
    """Use the established strategic fair-coin stream for Sam-128."""

    return derive_strategic_destination_choice_seed(
        request_seed,
        scope,
        information,
        representative_action_index,
        choice_index,
    )


def _select_tree_group(
    node: _GroupNode,
    exploration_constant: float,
    selection_seed: bytes,
) -> int:
    for representative in node.representative_actions:
        if node.action_visits[representative] == 0:
            return representative

    log_parent = math.log(node.visits)
    best_score = -math.inf
    tied: list[int] = []
    for representative in node.representative_actions:
        visits = node.action_visits[representative]
        mean = node.action_value_sums[representative] / visits
        score = mean + exploration_constant * math.sqrt(log_parent / visits)
        if score > best_score:
            best_score = score
            tied = [representative]
        elif score == best_score:
            tied.append(representative)
    return tied[Sha256CounterStream(selection_seed).randbelow(len(tied))]


def _move_for_action(
    state: SimulationEngineState,
    action_index: int,
) -> EngineMove:
    actor = state.active_player
    if actor is None:
        raise SearchContractViolation(
            "a playing Sam-128 simulation must have an active player"
        )
    move = move_for_action_index(actor, action_index)
    if move not in legal_simulation_moves(state, actor):
        raise SearchContractViolation(
            "Sam-128 selected a masked engine action"
        )
    return move


def _concrete_group_action(
    information: SearchInformationState,
    representative_action_index: int,
    request_seed: bytes,
    scope: str,
    choice_index: int,
) -> int:
    return select_concrete_action_index(
        information,
        representative_action_index,
        derive_sam_teacher_destination_seed(
            request_seed,
            scope,
            information,
            representative_action_index,
            choice_index,
        ),
        destination_symmetry_enabled=True,
    )


def _uniform_group_action(
    information: SearchInformationState,
    group_seed: bytes,
    response_request_seed: bytes,
    simulation_index: int,
    rollout_ply: int,
) -> int:
    groups = sam_teacher_action_groups(information)
    if not groups:
        raise SearchContractViolation(
            "a Sam-128 rollout has no strategic action"
        )
    group = groups[Sha256CounterStream(group_seed).randbelow(len(groups))]
    return select_concrete_action_index(
        information,
        group.representative_action_index,
        derive_sam_teacher_destination_seed(
            response_request_seed,
            "sam-128-response-rollout",
            information,
            group.representative_action_index,
            simulation_index * 8 + rollout_ply,
        ),
        destination_symmetry_enabled=True,
    )


def _group_diagnostics(
    node: _GroupNode,
) -> tuple[StrategicActionGroupDiagnostic, ...]:
    return tuple(
        StrategicActionGroupDiagnostic(
            group,
            node.action_visits[group.representative_action_index],
            (
                node.action_value_sums[group.representative_action_index]
                / node.action_visits[group.representative_action_index]
                if node.action_visits[group.representative_action_index]
                else None
            ),
        )
        for group in node.action_groups
    )


def _expanded_group_means(
    diagnostics: tuple[StrategicActionGroupDiagnostic, ...],
) -> tuple[float | None, ...]:
    values: list[float | None] = [None] * ACTION_COUNT
    for diagnostic in diagnostics:
        for action_index in diagnostic.group.member_action_indices:
            values[action_index] = diagnostic.mean_value
    return tuple(values)


def _project_group_visits(
    information: SearchInformationState,
    request_seed: bytes,
    diagnostics: tuple[StrategicActionGroupDiagnostic, ...],
    scope: str,
) -> tuple[tuple[int, ...], dict[int, int]]:
    visits = [0] * ACTION_COUNT
    concrete_by_representative: dict[int, int] = {}
    for diagnostic in diagnostics:
        representative = diagnostic.group.representative_action_index
        concrete = _concrete_group_action(
            information,
            representative,
            request_seed,
            scope,
            0,
        )
        visits[concrete] = diagnostic.visits
        concrete_by_representative[representative] = concrete
    return tuple(visits), concrete_by_representative


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
            "Sam-128 continuation refers to an empty hand slot"
        )
    return ContinuationStep(
        actor="self" if move.player is root_player else "opponent",
        action_index=action_index,
        card_id=card_id,
        grid_index=move.global_grid_index,
        forced=forced,
    )


def _check_interrupted(
    should_stop: Callable[[], bool] | None,
) -> None:
    if should_stop is not None and should_stop():
        raise SearchInterrupted(
            "Sam-128 search interrupted before result commit"
        )


class _GroupedActorInformationSetSearch:
    """One actor's symmetry-aware response UCT with uniform group rollouts."""

    def __init__(self, config: SamTeacherSearchConfig) -> None:
        self.config = config

    def search(
        self,
        information: SearchInformationState,
        should_stop: Callable[[], bool] | None = None,
    ) -> _ActorResponseResult:
        if not isinstance(information, SearchInformationState):
            raise SearchContractViolation(
                "Sam-128 response requires an information state"
            )
        root_fingerprint = information_state_fingerprint(information)
        root_player = information.player
        root_legal = _legal_action_indexes(information)
        if not root_legal:
            raise SearchContractViolation(
                "Sam-128 response has no legal actions"
            )
        if len(root_legal) == 1:
            return _ActorResponseResult(
                root_fingerprint,
                self.config.response_digest,
                root_legal[0],
                root_legal[0],
                0,
                0,
                (),
            )

        root_groups = sam_teacher_action_groups(information)
        if self.config.response_simulation_budget < len(root_groups):
            raise SearchContractViolation(
                "response simulation budget must visit every strategic action"
            )
        request_seed = derive_sam_teacher_response_request_seed(
            information,
            self.config.response_digest,
        )
        nodes: dict[SamTeacherNodeKey, _GroupNode] = {}

        for simulation_index in range(
            self.config.response_simulation_budget
        ):
            _check_interrupted(should_stop)
            state = sample_determinization(
                information,
                derive_sam_teacher_response_determinization_seed(
                    request_seed,
                    simulation_index,
                ),
            ).state
            path: list[tuple[_GroupNode, int]] = []
            expanded = False
            rollout_ply = 0

            while state.status is EngineStatus.PLAYING:
                _check_interrupted(should_stop)
                actor = state.active_player
                if actor is None:
                    raise SearchContractViolation(
                        "Sam-128 response lost its active player"
                    )
                moves = legal_simulation_moves(state, actor)
                if len(moves) == 1:
                    move = moves[0]
                elif actor is root_player and not expanded:
                    actor_information = project_simulation_information_state(
                        state
                    )
                    key = SamTeacherNodeKey(
                        actor,
                        information_state_fingerprint(actor_information),
                    )
                    groups = sam_teacher_action_groups(actor_information)
                    node = nodes.get(key)
                    if node is None:
                        node = _GroupNode.create(key, groups)
                        nodes[key] = node
                    elif node.action_groups != groups:
                        raise SearchContractViolation(
                            "one Sam-128 response node changed action groups"
                        )
                    representative = _select_tree_group(
                        node,
                        float(self.config.response_exploration_constant),
                        derive_sam_teacher_response_selection_seed(
                            request_seed,
                            simulation_index,
                            key,
                        ),
                    )
                    path.append((node, representative))
                    if node.action_visits[representative] == 0:
                        expanded = True
                    concrete = _concrete_group_action(
                        actor_information,
                        representative,
                        request_seed,
                        "sam-128-response-tree",
                        simulation_index,
                    )
                    move = _move_for_action(state, concrete)
                else:
                    actor_information = project_simulation_information_state(
                        state
                    )
                    rollout_seed = (
                        derive_sam_teacher_response_rollout_seed(
                            request_seed,
                            simulation_index,
                            rollout_ply,
                            actor_information,
                        )
                    )
                    concrete = _uniform_group_action(
                        actor_information,
                        rollout_seed,
                        request_seed,
                        simulation_index,
                        rollout_ply,
                    )
                    move = _move_for_action(state, concrete)
                state = apply_simulation_move(state, move)
                rollout_ply += 1

            if (
                state.status is not EngineStatus.ROUND_COMPLETE
                or state.pending_round_result is None
                or not path
            ):
                raise SearchContractViolation(
                    "Sam-128 response did not reach an engine terminal"
                )
            terminal_value = normalized_round_return(
                state.pending_round_result,
                root_player,
            )
            for node, representative in path:
                node.visits += 1
                node.action_visits[representative] += 1
                node.action_value_sums[representative] += (
                    actor_relative_value(
                        terminal_value,
                        root_player,
                        node.key.actor,
                    )
                )

        root_key = SamTeacherNodeKey(root_player, root_fingerprint)
        root_node = nodes.get(root_key)
        if (
            root_node is None
            or root_node.visits != self.config.response_simulation_budget
        ):
            raise SearchContractViolation(
                "Sam-128 response backup count does not match simulations"
            )
        selected_representative = min(
            root_node.representative_actions,
            key=lambda representative: (
                -root_node.action_visits[representative],
                -root_node.action_value_sums[representative]
                / root_node.action_visits[representative],
                representative,
            ),
        )
        selected = _concrete_group_action(
            information,
            selected_representative,
            request_seed,
            "sam-128-response-result",
            0,
        )
        return _ActorResponseResult(
            root_fingerprint,
            self.config.response_digest,
            selected,
            selected_representative,
            root_node.visits,
            len(nodes),
            _group_diagnostics(root_node),
        )


class SamTeacherInformationSetSearch:
    """Sam-128 outer UCT with isolated symmetry-aware actor responses."""

    def __init__(
        self,
        config: SamTeacherSearchConfig = SamTeacherSearchConfig(),
    ) -> None:
        if not isinstance(config, SamTeacherSearchConfig):
            raise SearchContractViolation(
                "Sam-128 search configuration is invalid"
            )
        self.config = config
        self._response_search = _GroupedActorInformationSetSearch(config)

    def _actor_response(
        self,
        information: SearchInformationState,
        should_stop: Callable[[], bool] | None,
    ) -> _ActorResponseResult:
        if not isinstance(information, SearchInformationState):
            raise SearchContractViolation(
                "Sam-128 actor response requires an information state"
            )
        return self._response_search.search(information, should_stop)

    def search(
        self,
        information: SearchInformationState,
        request_seed: bytes | None = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> SamTeacherSearchResult:
        started = time.perf_counter()
        if not isinstance(information, SearchInformationState):
            raise SearchContractViolation(
                "Sam-128 search requires an information state"
            )
        if request_seed is None:
            request_seed = derive_sam_teacher_request_seed(
                information,
                self.config.digest,
            )
        seed_hex(request_seed)
        root_player = information.player
        root_fingerprint = information_state_fingerprint(information)
        root_legal = _legal_action_indexes(information)
        if not root_legal:
            raise SearchContractViolation(
                "Sam-128 root has no legal actions"
            )
        if len(root_legal) == 1:
            return SamTeacherSearchResult(
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
                selected_representative_action_index=root_legal[0],
                group_diagnostics=(),
            )

        root_groups = sam_teacher_action_groups(information)
        if self.config.outer_simulation_budget < len(root_groups):
            raise SearchContractViolation(
                "outer simulation budget must visit every strategic action"
            )

        nodes: dict[SamTeacherNodeKey, _GroupNode] = {}
        response_cache: dict[
            tuple[EnginePlayer, str, str],
            _ActorResponseResult,
        ] = {}
        continuations: list[_ObservedContinuation] = []
        response_requests = 0
        response_cache_hits = 0
        response_simulations = 0
        response_information_sets = 0

        for simulation_index in range(
            self.config.outer_simulation_budget
        ):
            _check_interrupted(should_stop)
            state = sample_determinization(
                information,
                derive_sam_teacher_determinization_seed(
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
                        "Sam-128 simulation lost its active player"
                    )
                moves = legal_simulation_moves(state, actor)
                forced = len(moves) == 1
                representative: int | None = None
                if forced:
                    move = moves[0]
                    action_index = action_index_for_move(move, actor)
                elif actor is root_player and not expanded:
                    actor_information = project_simulation_information_state(
                        state
                    )
                    key = SamTeacherNodeKey(
                        actor,
                        information_state_fingerprint(actor_information),
                    )
                    groups = sam_teacher_action_groups(actor_information)
                    node = nodes.get(key)
                    if node is None:
                        node = _GroupNode.create(key, groups)
                        nodes[key] = node
                    elif node.action_groups != groups:
                        raise SearchContractViolation(
                            "one Sam-128 outer node changed action groups"
                        )
                    representative = _select_tree_group(
                        node,
                        float(self.config.outer_exploration_constant),
                        derive_sam_teacher_selection_seed(
                            request_seed,
                            simulation_index,
                            key,
                        ),
                    )
                    path.append((node, representative))
                    if node.action_visits[representative] == 0:
                        expanded = True
                    action_index = _concrete_group_action(
                        actor_information,
                        representative,
                        request_seed,
                        "sam-128-outer-tree",
                        simulation_index,
                    )
                    move = _move_for_action(state, action_index)
                else:
                    actor_information = project_simulation_information_state(
                        state
                    )
                    fingerprint = information_state_fingerprint(
                        actor_information
                    )
                    cache_key = (
                        actor,
                        fingerprint,
                        self.config.response_digest,
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
                            != fingerprint
                            or response.config_digest
                            != self.config.response_digest
                        ):
                            raise SearchContractViolation(
                                "Sam-128 response crossed its information "
                                "boundary"
                            )
                        response_cache[cache_key] = response
                        response_simulations += response.simulation_count
                        response_information_sets += (
                            response.information_set_count
                        )
                    else:
                        response_cache_hits += 1
                    representative = (
                        response.selected_representative_action_index
                    )
                    action_index = response.selected_action_index
                    move = _move_for_action(state, action_index)

                if root_representative is None:
                    if actor is not root_player or representative is None:
                        raise SearchContractViolation(
                            "Sam-128 simulation began outside its root group"
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
                    "Sam-128 outer search did not reach an engine terminal"
                )
            root_value = normalized_round_return(
                state.pending_round_result,
                root_player,
            )
            for node, representative in path:
                node.visits += 1
                node.action_visits[representative] += 1
                node.action_value_sums[representative] += (
                    actor_relative_value(
                        root_value,
                        root_player,
                        node.key.actor,
                    )
                )
            continuations.append(
                _ObservedContinuation(
                    root_representative,
                    root_concrete,
                    root_value,
                    simulation_index,
                    tuple(steps),
                )
            )

        root_key = SamTeacherNodeKey(root_player, root_fingerprint)
        root_node = nodes.get(root_key)
        if (
            root_node is None
            or root_node.visits != self.config.outer_simulation_budget
        ):
            raise SearchContractViolation(
                "Sam-128 root backup count does not match simulations"
            )
        selected_representative = min(
            root_node.representative_actions,
            key=lambda representative: (
                -root_node.action_visits[representative],
                -root_node.action_value_sums[representative]
                / root_node.action_visits[representative],
                representative,
            ),
        )
        representative_continuation = min(
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
        diagnostics = _group_diagnostics(root_node)
        projected_visits, concrete_by_representative = (
            _project_group_visits(
                information,
                request_seed,
                diagnostics,
                "sam-128-root-result",
            )
        )
        selected = concrete_by_representative[selected_representative]
        return SamTeacherSearchResult(
            information_state_fingerprint=root_fingerprint,
            config_digest=self.config.digest,
            selected_action_index=selected,
            action_visits=projected_visits,
            mean_action_values=_expanded_group_means(diagnostics),
            simulation_count=root_node.visits,
            information_set_count=len(nodes),
            principal_continuation=PrincipalContinuation(
                representative_continuation.root_concrete_action_index,
                representative_continuation.terminal_value,
                representative_continuation.simulation_index,
                representative_continuation.steps,
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
            selected_representative_action_index=selected_representative,
            group_diagnostics=diagnostics,
        )


__all__ = (
    "SAM_TEACHER_DETERMINIZATION_NAMESPACE",
    "SAM_TEACHER_RECURSION_LIMIT",
    "SAM_TEACHER_REQUEST_NAMESPACE",
    "SAM_TEACHER_RESPONSE_DETERMINIZATION_NAMESPACE",
    "SAM_TEACHER_RESPONSE_REQUEST_NAMESPACE",
    "SAM_TEACHER_RESPONSE_ROLLOUT_NAMESPACE",
    "SAM_TEACHER_RESPONSE_SCHEMA_VERSION",
    "SAM_TEACHER_RESPONSE_SELECTION_NAMESPACE",
    "SAM_TEACHER_SEARCH_SCHEMA_VERSION",
    "SAM_TEACHER_SELECTION_NAMESPACE",
    "SAM_TEACHER_SELECTION_PROFILE",
    "SamTeacherInformationSetSearch",
    "SamTeacherNodeKey",
    "SamTeacherSearchConfig",
    "SamTeacherSearchResult",
    "derive_sam_teacher_destination_seed",
    "derive_sam_teacher_determinization_seed",
    "derive_sam_teacher_request_seed",
    "derive_sam_teacher_response_determinization_seed",
    "derive_sam_teacher_response_request_seed",
    "derive_sam_teacher_response_rollout_seed",
    "derive_sam_teacher_response_selection_seed",
    "derive_sam_teacher_selection_seed",
    "sam_teacher_action_groups",
)

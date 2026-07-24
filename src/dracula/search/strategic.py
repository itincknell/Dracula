"""Teacher v2 search with information-safe shallow greedy responses."""

from __future__ import annotations

import hashlib
import json
import math
import resource
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol

from dracula.bridge import (
    ACTION_COUNT,
    HAND_SLOT_COUNT,
    action_index_for_move,
    move_for_action_index,
)
from dracula.engine import (
    EngineMove,
    EnginePlayer,
    EngineStatus,
    SimulationEngineState,
    apply_simulation_move,
    legal_simulation_moves,
)
from dracula.randomness import Sha256CounterStream, derive_seed, seed_hex
from dracula.models import POLICY_GRID_INDICES
from dracula.policy_value import (
    ACTION_SCHEMA_VERSION,
    OBSERVATION_SCHEMA_VERSION,
)
from dracula.response_distillation import (
    RESPONSE_EXAMPLE_SCHEMA_VERSION,
    RESPONSE_RANKING_SCHEMA_VERSION,
    ResponseActionGroupTarget,
    ResponseCacheIdentity,
    ResponseDistillationExample,
    ResponseExampleObserver,
)
from dracula.search.information import (
    SearchInformationState,
    information_state_fingerprint,
    policy_input_from_information_state,
    project_simulation_information_state,
    sample_determinization,
    sample_uniform_action_index,
)
from dracula.search.planner import (
    ContinuationStep,
    PrincipalContinuation,
    SearchContractViolation,
    SearchInterrupted,
    SearchResult,
    normalized_round_return,
)
from dracula.search.symmetry import (
    DESTINATION_SYMMETRY_SCHEMA_VERSION,
    destination_symmetry_groups,
)

STRATEGIC_SEARCH_SCHEMA_VERSION = "dracula-strategic-information-search-v2"
STRATEGIC_SEARCH_REQUEST_NAMESPACE = "dracula-strategic-search-request-v2"
STRATEGIC_DETERMINIZATION_NAMESPACE = (
    "dracula-strategic-search-determinization-v2"
)
STRATEGIC_SELECTION_NAMESPACE = "dracula-strategic-search-expansion-v2"
STRATEGIC_SELECTION_PROFILE = "max-visits-mean-value-action-index-v1"
STRATEGIC_DESTINATION_CHOICE_NAMESPACE = (
    "dracula-strategic-destination-choice-v1"
)
STRATEGIC_DESTINATION_CHOICE_PROFILE = (
    "derived-fair-coin-after-group-selection-v1"
)

GREEDY_RESPONSE_SCHEMA_VERSION = "dracula-shallow-greedy-response-v1"
GREEDY_RESPONSE_REQUEST_NAMESPACE = "dracula-greedy-response-request-v1"
GREEDY_RESPONSE_DETERMINIZATION_NAMESPACE = (
    "dracula-greedy-response-determinization-v1"
)
GREEDY_RESPONSE_ROLLOUT_NAMESPACE = "dracula-greedy-response-rollout-v1"
GREEDY_RESPONSE_CONTINUATION_PROFILE = "uniform-legal-to-round-end-v1"
GREEDY_RESPONSE_SELECTION_PROFILE = "max-mean-action-index-v1"
SUPPORTED_RESPONSE_COMPLETIONS = frozenset((1, 2, 4))
HYBRID_RESPONSE_SCHEMA_VERSION = "dracula-response-ranker-hybrid-v1"
HYBRID_RESPONSE_REQUEST_NAMESPACE = "dracula-response-ranker-hybrid-request-v1"
_DIGEST_CHARACTERS = frozenset("0123456789abcdef")


class StrategicResponseMode(str, Enum):
    PURE = "pure"
    STUDENT_DIRECT = "student-direct"
    STUDENT_TOP_2 = "student-top-2"
    STUDENT_TOP_3 = "student-top-3"

    @property
    def shortlist_size(self) -> int:
        return {
            StrategicResponseMode.PURE: 0,
            StrategicResponseMode.STUDENT_DIRECT: 1,
            StrategicResponseMode.STUDENT_TOP_2: 2,
            StrategicResponseMode.STUDENT_TOP_3: 3,
        }[self]


class StrategicGroupRanker(Protocol):
    """Rank strategic groups from one actor's information state only."""

    artifact_digest: str

    def rank(
        self,
        information: SearchInformationState,
        groups: tuple[StrategicActionGroup, ...],
    ) -> tuple[int, ...]: ...


@dataclass(frozen=True, slots=True)
class ShallowResponseConfig:
    completions_per_action: int = 1
    destination_symmetry_enabled: bool = True

    def __post_init__(self) -> None:
        if (
            type(self.completions_per_action) is not int
            or self.completions_per_action not in SUPPORTED_RESPONSE_COMPLETIONS
        ):
            raise SearchContractViolation(
                "response completions per action must be 1, 2, or 4"
            )
        if type(self.destination_symmetry_enabled) is not bool:
            raise SearchContractViolation(
                "destination symmetry flag must be Boolean"
            )

    @property
    def digest(self) -> str:
        payload = {
            "completions_per_action": self.completions_per_action,
            "continuation_profile": GREEDY_RESPONSE_CONTINUATION_PROFILE,
            "response_schema_version": GREEDY_RESPONSE_SCHEMA_VERSION,
            "selection_profile": GREEDY_RESPONSE_SELECTION_PROFILE,
            "terminal_value": "exact-normalized-round-differential-v1",
        }
        if self.destination_symmetry_enabled:
            payload.update(
                {
                    "destination_choice_profile": (
                        STRATEGIC_DESTINATION_CHOICE_PROFILE
                    ),
                    "destination_symmetry_schema_version": (
                        DESTINATION_SYMMETRY_SCHEMA_VERSION
                    ),
                }
            )
        encoded = json.dumps(
            payload,
            separators=(",", ":"),
            sort_keys=True,
        )
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class StrategicSearchConfig:
    outer_simulation_budget: int = 500
    response_completions_per_action: int = 1
    outer_exploration_constant: float = math.sqrt(2.0)
    destination_symmetry_enabled: bool = True
    response_mode: StrategicResponseMode | str = StrategicResponseMode.PURE
    response_ranker_artifact_digest: str | None = None

    def __post_init__(self) -> None:
        if (
            type(self.outer_simulation_budget) is not int
            or self.outer_simulation_budget < 1
        ):
            raise SearchContractViolation(
                "outer simulation budget must be a positive integer"
            )
        ShallowResponseConfig(
            self.response_completions_per_action,
            self.destination_symmetry_enabled,
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
        try:
            mode = StrategicResponseMode(self.response_mode)
        except ValueError as error:
            raise SearchContractViolation(
                "response mode must be pure, student-direct, student-top-2, "
                "or student-top-3"
            ) from error
        object.__setattr__(self, "response_mode", mode)
        digest = self.response_ranker_artifact_digest
        if mode is StrategicResponseMode.PURE:
            if digest is not None:
                raise SearchContractViolation(
                    "pure Teacher v2 cannot bind a response-ranker artifact"
                )
        elif (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in _DIGEST_CHARACTERS for character in digest)
        ):
            raise SearchContractViolation(
                "student response mode requires an artifact SHA-256 digest"
            )

    @property
    def response_config(self) -> ShallowResponseConfig:
        return ShallowResponseConfig(
            self.response_completions_per_action,
            self.destination_symmetry_enabled,
        )

    @property
    def response_shortlist_size(self) -> int:
        return StrategicResponseMode(self.response_mode).shortlist_size

    @property
    def digest(self) -> str:
        payload = {
            "outer_exploration_constant": float(
                self.outer_exploration_constant
            ),
            "outer_simulation_budget": self.outer_simulation_budget,
            "response_config_digest": self.response_config.digest,
            "search_schema_version": STRATEGIC_SEARCH_SCHEMA_VERSION,
            "selection_profile": STRATEGIC_SELECTION_PROFILE,
        }
        if self.response_mode is not StrategicResponseMode.PURE:
            payload.update(
                {
                    "hybrid_response_schema_version": (
                        HYBRID_RESPONSE_SCHEMA_VERSION
                    ),
                    "response_mode": self.response_mode.value,
                    "response_ranker_artifact_digest": (
                        self.response_ranker_artifact_digest
                    ),
                    "response_shortlist_size": self.response_shortlist_size,
                }
            )
        if self.destination_symmetry_enabled:
            payload.update(
                {
                    "destination_choice_profile": (
                        STRATEGIC_DESTINATION_CHOICE_PROFILE
                    ),
                    "destination_symmetry_schema_version": (
                        DESTINATION_SYMMETRY_SCHEMA_VERSION
                    ),
                }
            )
        encoded = json.dumps(
            payload,
            separators=(",", ":"),
            sort_keys=True,
        )
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class StrategicNodeKey:
    actor: EnginePlayer
    information_state_fingerprint: str


@dataclass(frozen=True, slots=True)
class StrategicActionGroup:
    hand_slot: int
    representative_action_index: int
    representative_grid_index: int
    member_action_indices: tuple[int, ...]
    member_grid_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        if type(self.hand_slot) is not int or not 0 <= self.hand_slot < HAND_SLOT_COUNT:
            raise SearchContractViolation("strategic group hand slot is invalid")
        if (
            type(self.representative_action_index) is not int
            or not 0 <= self.representative_action_index < ACTION_COUNT
            or self.representative_action_index // len(POLICY_GRID_INDICES)
            != self.hand_slot
        ):
            raise SearchContractViolation(
                "strategic group representative action is invalid"
            )
        if (
            type(self.representative_grid_index) is not int
            or self.representative_grid_index
            != POLICY_GRID_INDICES[
                self.representative_action_index % len(POLICY_GRID_INDICES)
            ]
        ):
            raise SearchContractViolation(
                "strategic group representative destination is invalid"
            )
        if (
            not isinstance(self.member_action_indices, tuple)
            or not self.member_action_indices
            or len(set(self.member_action_indices))
            != len(self.member_action_indices)
            or any(
                type(action_index) is not int
                or not 0 <= action_index < ACTION_COUNT
                or action_index // len(POLICY_GRID_INDICES) != self.hand_slot
                for action_index in self.member_action_indices
            )
        ):
            raise SearchContractViolation(
                "strategic group member actions are invalid"
            )
        if (
            not isinstance(self.member_grid_indices, tuple)
            or len(self.member_grid_indices) != len(self.member_action_indices)
            or tuple(
                POLICY_GRID_INDICES[
                    action_index % len(POLICY_GRID_INDICES)
                ]
                for action_index in self.member_action_indices
            )
            != self.member_grid_indices
            or self.representative_action_index not in self.member_action_indices
        ):
            raise SearchContractViolation(
                "strategic group member destinations are invalid"
            )


@dataclass(frozen=True, slots=True)
class StrategicActionGroupDiagnostic:
    group: StrategicActionGroup
    visits: int
    mean_value: float | None

    def __post_init__(self) -> None:
        if type(self.visits) is not int or self.visits < 0:
            raise SearchContractViolation(
                "strategic group visits must be non-negative"
            )
        if self.visits == 0:
            if self.mean_value is not None:
                raise SearchContractViolation(
                    "an unvisited strategic group cannot have a value"
                )
        elif (
            type(self.mean_value) not in (int, float)
            or not math.isfinite(self.mean_value)
            or not -1.0 <= self.mean_value <= 1.0
        ):
            raise SearchContractViolation(
                "a visited strategic group must have a bounded finite value"
            )


@dataclass(frozen=True, slots=True)
class GreedyResponseResult:
    information_state_fingerprint: str
    config_digest: str
    selected_action_index: int
    legal_action_indices: tuple[int, ...]
    mean_action_values: tuple[float | None, ...]
    completions_per_action: int
    candidate_action_count: int
    terminal_evaluation_count: int
    selected_representative_action_index: int
    group_diagnostics: tuple[StrategicActionGroupDiagnostic, ...]
    response_mode: StrategicResponseMode = StrategicResponseMode.PURE
    shortlisted_representative_action_indices: tuple[int, ...] = ()
    model_call_count: int = 0

    @property
    def selected_representative_grid_index(self) -> int:
        return POLICY_GRID_INDICES[
            self.selected_representative_action_index
            % len(POLICY_GRID_INDICES)
        ]

    @property
    def selected_concrete_grid_index(self) -> int:
        return POLICY_GRID_INDICES[
            self.selected_action_index % len(POLICY_GRID_INDICES)
        ]

    def __post_init__(self) -> None:
        try:
            mode = StrategicResponseMode(self.response_mode)
        except ValueError as error:
            raise SearchContractViolation("response result mode is invalid") from error
        object.__setattr__(self, "response_mode", mode)
        if len(self.mean_action_values) != ACTION_COUNT:
            raise SearchContractViolation(
                "response action values must contain 32 entries"
            )
        if self.selected_action_index not in self.legal_action_indices:
            raise SearchContractViolation("response selected an illegal action")
        representatives = tuple(
            diagnostic.group.representative_action_index
            for diagnostic in self.group_diagnostics
        )
        if (
            self.selected_representative_action_index not in representatives
            or self.selected_action_index
            not in next(
                diagnostic.group.member_action_indices
                for diagnostic in self.group_diagnostics
                if diagnostic.group.representative_action_index
                == self.selected_representative_action_index
            )
        ):
            raise SearchContractViolation(
                "response concrete action does not belong to its selected group"
            )
        if self.candidate_action_count != len(self.group_diagnostics):
            if mode is StrategicResponseMode.PURE:
                raise SearchContractViolation(
                    "pure response candidate accounting is invalid"
                )
        if self.terminal_evaluation_count != (
            self.candidate_action_count * self.completions_per_action
        ):
            raise SearchContractViolation("response terminal accounting is invalid")
        legal = set(self.legal_action_indices)
        grouped = {
            action_index
            for diagnostic in self.group_diagnostics
            for action_index in diagnostic.group.member_action_indices
        }
        if grouped != legal or len(grouped) != sum(
            len(diagnostic.group.member_action_indices)
            for diagnostic in self.group_diagnostics
        ):
            raise SearchContractViolation(
                "response groups do not partition legal actions"
            )
        representatives_set = set(representatives)
        if (
            not isinstance(self.shortlisted_representative_action_indices, tuple)
            or len(set(self.shortlisted_representative_action_indices))
            != len(self.shortlisted_representative_action_indices)
            or any(
                representative not in representatives_set
                for representative in self.shortlisted_representative_action_indices
            )
        ):
            raise SearchContractViolation("response shortlist is invalid")
        expected_shortlist_size = min(mode.shortlist_size, len(representatives))
        if (
            mode is StrategicResponseMode.PURE
            and self.shortlisted_representative_action_indices
        ) or (
            mode is not StrategicResponseMode.PURE
            and len(self.shortlisted_representative_action_indices)
            != expected_shortlist_size
        ):
            raise SearchContractViolation("response shortlist size is invalid")
        if (
            type(self.model_call_count) is not int
            or self.model_call_count
            != (0 if mode is StrategicResponseMode.PURE else 1)
        ):
            raise SearchContractViolation("response model-call accounting is invalid")
        expected_candidates = (
            len(representatives)
            if mode is StrategicResponseMode.PURE
            else 0
            if mode is StrategicResponseMode.STUDENT_DIRECT
            else len(self.shortlisted_representative_action_indices)
        )
        if self.candidate_action_count != expected_candidates:
            raise SearchContractViolation("response candidate accounting is invalid")
        if mode is StrategicResponseMode.STUDENT_DIRECT:
            if self.terminal_evaluation_count != 0:
                raise SearchContractViolation(
                    "student-direct cannot run terminal response evaluations"
                )
        elif mode in (
            StrategicResponseMode.STUDENT_TOP_2,
            StrategicResponseMode.STUDENT_TOP_3,
        ):
            if (
                self.selected_representative_action_index
                not in self.shortlisted_representative_action_indices
            ):
                raise SearchContractViolation(
                    "student response selected outside its shortlist"
                )
        values_by_group = {
            action_index: diagnostic.mean_value
            for diagnostic in self.group_diagnostics
            for action_index in diagnostic.group.member_action_indices
        }
        for index, value in enumerate(self.mean_action_values):
            if index in legal:
                expected_value = values_by_group[index]
                if expected_value is None:
                    if value is not None:
                        raise SearchContractViolation(
                            "unevaluated response groups cannot contain values"
                        )
                elif (
                    value is None
                    or not math.isfinite(value)
                    or not -1.0 <= value <= 1.0
                    or value != expected_value
                ):
                    raise SearchContractViolation(
                        "evaluated response values must be finite and bounded"
                    )
            elif value is not None:
                raise SearchContractViolation(
                    "masked response actions cannot contain a value"
                )


@dataclass(frozen=True, slots=True)
class StrategicSearchResult(SearchResult):
    elapsed_seconds: float = field(compare=False)
    peak_resident_memory_bytes: int = field(compare=False)
    response_request_count: int
    unique_response_evaluation_count: int
    response_cache_hit_count: int
    response_candidate_action_count: int
    response_terminal_evaluation_count: int
    total_terminal_evaluation_count: int
    selected_representative_action_index: int
    group_diagnostics: tuple[StrategicActionGroupDiagnostic, ...]
    response_model_call_count: int = 0

    @property
    def selected_representative_grid_index(self) -> int:
        return POLICY_GRID_INDICES[
            self.selected_representative_action_index
            % len(POLICY_GRID_INDICES)
        ]

    @property
    def selected_concrete_grid_index(self) -> int:
        return POLICY_GRID_INDICES[
            self.selected_action_index % len(POLICY_GRID_INDICES)
        ]

    def __post_init__(self) -> None:
        super(StrategicSearchResult, self).__post_init__()
        if not math.isfinite(self.elapsed_seconds) or self.elapsed_seconds < 0:
            raise SearchContractViolation(
                "search latency must be finite and non-negative"
            )
        if (
            type(self.peak_resident_memory_bytes) is not int
            or self.peak_resident_memory_bytes < 0
        ):
            raise SearchContractViolation("peak memory must be non-negative")
        counts = (
            self.response_request_count,
            self.unique_response_evaluation_count,
            self.response_cache_hit_count,
            self.response_candidate_action_count,
            self.response_terminal_evaluation_count,
            self.total_terminal_evaluation_count,
            self.response_model_call_count,
        )
        if any(type(value) is not int or value < 0 for value in counts):
            raise SearchContractViolation(
                "strategic diagnostics must be non-negative integers"
            )
        if (
            self.response_request_count
            != self.unique_response_evaluation_count
            + self.response_cache_hit_count
        ):
            raise SearchContractViolation("response cache accounting is inconsistent")
        if self.response_model_call_count > self.unique_response_evaluation_count:
            raise SearchContractViolation(
                "response model calls cannot exceed unique responses"
            )
        if self.total_terminal_evaluation_count != (
            self.simulation_count + self.response_terminal_evaluation_count
        ):
            raise SearchContractViolation(
                "terminal evaluation accounting is inconsistent"
            )
        representatives = tuple(
            diagnostic.group.representative_action_index
            for diagnostic in self.group_diagnostics
        )
        if self.simulation_count == 0:
            if self.group_diagnostics:
                raise SearchContractViolation(
                    "a forced result cannot contain strategic groups"
                )
            if (
                self.selected_representative_action_index
                != self.selected_action_index
            ):
                raise SearchContractViolation(
                    "a forced result cannot resolve a destination group"
                )
        elif (
            self.selected_representative_action_index not in representatives
            or self.selected_action_index
            not in next(
                diagnostic.group.member_action_indices
                for diagnostic in self.group_diagnostics
                if diagnostic.group.representative_action_index
                == self.selected_representative_action_index
            )
            or sum(
                diagnostic.visits for diagnostic in self.group_diagnostics
            )
            != self.simulation_count
        ):
            raise SearchContractViolation(
                "search selected action or group accounting is inconsistent"
            )
        if self.simulation_count:
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
                        "one concrete group member must carry pooled visits"
                    )
                if any(
                    self.mean_action_values[action_index]
                    != diagnostic.mean_value
                    for action_index in diagnostic.group.member_action_indices
                ):
                    raise SearchContractViolation(
                        "search group members must share one mean value"
                    )
                if (
                    diagnostic.group.representative_action_index
                    == self.selected_representative_action_index
                    and self.action_visits[self.selected_action_index]
                    != diagnostic.visits
                ):
                    raise SearchContractViolation(
                        "selected concrete action must carry its group visits"
                    )
            if any(
                visits
                for action_index, visits in enumerate(self.action_visits)
                if action_index not in grouped_members
            ):
                raise SearchContractViolation(
                    "search visits cannot leave strategic action groups"
                )


@dataclass(slots=True)
class _InformationNode:
    key: StrategicNodeKey
    action_groups: tuple[StrategicActionGroup, ...]
    visits: int
    action_visits: list[int]
    action_value_sums: list[float]

    @property
    def legal_actions(self) -> tuple[int, ...]:
        return tuple(
            group.representative_action_index for group in self.action_groups
        )

    @classmethod
    def create(
        cls,
        key: StrategicNodeKey,
        action_groups: tuple[StrategicActionGroup, ...],
    ) -> _InformationNode:
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


def _validate_counter(value: int, label: str) -> str:
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return str(value)


def _legal_action_indexes(information: SearchInformationState) -> tuple[int, ...]:
    return tuple(
        index
        for index, allowed in enumerate(
            value for row in information.legal_mask for value in row
        )
        if allowed
    )


_POLICY_POSITION_BY_GRID_INDEX = {
    grid_index: position
    for position, grid_index in enumerate(POLICY_GRID_INDICES)
}


def strategic_action_groups(
    information: SearchInformationState,
    destination_symmetry_enabled: bool = True,
) -> tuple[StrategicActionGroup, ...]:
    """Partition legal actions by hand slot and the exact destination table."""

    if not isinstance(information, SearchInformationState):
        raise SearchContractViolation(
            "strategic grouping requires a player information state"
        )
    if type(destination_symmetry_enabled) is not bool:
        raise SearchContractViolation(
            "destination symmetry flag must be Boolean"
        )
    legal = set(_legal_action_indexes(information))
    if not destination_symmetry_enabled:
        return tuple(
            StrategicActionGroup(
                hand_slot=action_index // len(POLICY_GRID_INDICES),
                representative_action_index=action_index,
                representative_grid_index=POLICY_GRID_INDICES[
                    action_index % len(POLICY_GRID_INDICES)
                ],
                member_action_indices=(action_index,),
                member_grid_indices=(
                    POLICY_GRID_INDICES[
                        action_index % len(POLICY_GRID_INDICES)
                    ],
                ),
            )
            for action_index in sorted(legal)
        )
    destination_groups = destination_symmetry_groups(information.coffin)
    groups: list[StrategicActionGroup] = []
    for hand_slot in range(HAND_SLOT_COUNT):
        if information.own_hand[hand_slot] is None:
            continue
        for destination_group in destination_groups:
            try:
                representative_position = _POLICY_POSITION_BY_GRID_INDEX[
                    destination_group.representative_grid_index
                ]
                member_positions = tuple(
                    _POLICY_POSITION_BY_GRID_INDEX[grid_index]
                    for grid_index in destination_group.member_grid_indices
                )
            except KeyError as error:
                raise SearchContractViolation(
                    "the center cannot be a strategic destination"
                ) from error
            representative_action = (
                hand_slot * len(POLICY_GRID_INDICES)
                + representative_position
            )
            member_actions = tuple(
                hand_slot * len(POLICY_GRID_INDICES) + position
                for position in member_positions
            )
            groups.append(
                StrategicActionGroup(
                    hand_slot=hand_slot,
                    representative_action_index=representative_action,
                    representative_grid_index=(
                        destination_group.representative_grid_index
                    ),
                    member_action_indices=member_actions,
                    member_grid_indices=destination_group.member_grid_indices,
                )
            )
    grouped = [
        action_index
        for group in groups
        for action_index in group.member_action_indices
    ]
    if len(set(grouped)) != len(grouped) or set(grouped) != legal:
        raise SearchContractViolation(
            "strategic action groups do not partition legal actions"
        )
    return tuple(groups)


def derive_strategic_destination_choice_seed(
    request_seed: bytes,
    scope: str,
    information: SearchInformationState,
    representative_action_index: int,
    choice_index: int,
) -> bytes:
    if not isinstance(scope, str) or not scope:
        raise ValueError("destination choice scope must be nonempty")
    if (
        type(representative_action_index) is not int
        or not 0 <= representative_action_index < ACTION_COUNT
    ):
        raise ValueError("representative action index is invalid")
    return derive_seed(
        STRATEGIC_DESTINATION_CHOICE_NAMESPACE,
        seed_hex(request_seed),
        scope,
        information_state_fingerprint(information),
        str(representative_action_index),
        _validate_counter(choice_index, "choice index"),
    )


def select_concrete_action_index(
    information: SearchInformationState,
    representative_action_index: int,
    choice_seed: bytes,
    destination_symmetry_enabled: bool = True,
) -> int:
    """Resolve a selected strategic group without changing that selection."""

    seed_hex(choice_seed)
    try:
        group = next(
            candidate
            for candidate in strategic_action_groups(
                information,
                destination_symmetry_enabled,
            )
            if candidate.representative_action_index
            == representative_action_index
        )
    except StopIteration as error:
        raise SearchContractViolation(
            "selected representative is not a legal strategic action"
        ) from error
    if len(group.member_action_indices) == 1:
        selected = group.member_action_indices[0]
    elif len(group.member_action_indices) == 2:
        selected = group.member_action_indices[
            Sha256CounterStream(choice_seed).randbelow(2)
        ]
    else:
        raise SearchContractViolation(
            "authorized destination groups must contain one or two members"
        )
    if selected not in _legal_action_indexes(information):
        raise SearchContractViolation(
            "resolved strategic destination is not currently legal"
        )
    return selected


def _group_diagnostics(
    groups: tuple[StrategicActionGroup, ...],
    visits: list[int],
    value_sums: list[float],
) -> tuple[StrategicActionGroupDiagnostic, ...]:
    return tuple(
        StrategicActionGroupDiagnostic(
            group=group,
            visits=visits[group.representative_action_index],
            mean_value=(
                value_sums[group.representative_action_index]
                / visits[group.representative_action_index]
                if visits[group.representative_action_index]
                else None
            ),
        )
        for group in groups
    )


def _expanded_group_means(
    diagnostics: tuple[StrategicActionGroupDiagnostic, ...],
) -> tuple[float | None, ...]:
    means: list[float | None] = [None] * ACTION_COUNT
    for diagnostic in diagnostics:
        for action_index in diagnostic.group.member_action_indices:
            means[action_index] = diagnostic.mean_value
    return tuple(means)


def _project_group_visits_to_concrete_actions(
    information: SearchInformationState,
    request_seed: bytes,
    diagnostics: tuple[StrategicActionGroupDiagnostic, ...],
    destination_symmetry_enabled: bool,
) -> tuple[tuple[int, ...], dict[int, int]]:
    visits = [0] * ACTION_COUNT
    concrete_by_representative: dict[int, int] = {}
    for diagnostic in diagnostics:
        representative = diagnostic.group.representative_action_index
        concrete = select_concrete_action_index(
            information,
            representative,
            derive_strategic_destination_choice_seed(
                request_seed,
                "root-result-group",
                information,
                representative,
                0,
            ),
            destination_symmetry_enabled,
        )
        visits[concrete] = diagnostic.visits
        concrete_by_representative[representative] = concrete
    return tuple(visits), concrete_by_representative


def derive_strategic_search_request_seed(
    fixture_id: str,
    information: SearchInformationState,
    search_config_digest: str,
) -> bytes:
    return derive_seed(
        STRATEGIC_SEARCH_REQUEST_NAMESPACE,
        fixture_id,
        information_state_fingerprint(information),
        information.player.value,
        str(information.round_number),
        str(information.turn_number),
        search_config_digest,
    )


def derive_strategic_determinization_seed(
    request_seed: bytes, simulation_index: int
) -> bytes:
    return derive_seed(
        STRATEGIC_DETERMINIZATION_NAMESPACE,
        seed_hex(request_seed),
        _validate_counter(simulation_index, "simulation index"),
    )


def derive_strategic_selection_seed(
    request_seed: bytes, simulation_index: int, node: StrategicNodeKey
) -> bytes:
    return derive_seed(
        STRATEGIC_SELECTION_NAMESPACE,
        seed_hex(request_seed),
        _validate_counter(simulation_index, "simulation index"),
        node.actor.value,
        node.information_state_fingerprint,
    )


def derive_greedy_response_request_seed(
    information: SearchInformationState, response_config_digest: str
) -> bytes:
    """Derive a response seed without enclosing-world information."""

    return derive_seed(
        GREEDY_RESPONSE_REQUEST_NAMESPACE,
        information_state_fingerprint(information),
        response_config_digest,
    )


def derive_hybrid_response_request_seed(
    information: SearchInformationState,
    artifact_digest: str,
    shortlist_size: int,
    response_config_digest: str,
) -> bytes:
    """Bind student choice without accepting enclosing-world identity."""

    if (
        not isinstance(artifact_digest, str)
        or len(artifact_digest) != 64
        or any(character not in _DIGEST_CHARACTERS for character in artifact_digest)
    ):
        raise ValueError("response-ranker artifact digest is invalid")
    if type(shortlist_size) is not int or shortlist_size not in {1, 2, 3}:
        raise ValueError("student shortlist size must be one, two, or three")
    return derive_seed(
        HYBRID_RESPONSE_REQUEST_NAMESPACE,
        information_state_fingerprint(information),
        artifact_digest,
        str(shortlist_size),
        response_config_digest,
    )


def derive_greedy_response_determinization_seed(
    response_request_seed: bytes, completion_index: int
) -> bytes:
    return derive_seed(
        GREEDY_RESPONSE_DETERMINIZATION_NAMESPACE,
        seed_hex(response_request_seed),
        _validate_counter(completion_index, "completion index"),
    )


def derive_greedy_response_rollout_seed(
    response_request_seed: bytes,
    completion_index: int,
    candidate_action_index: int,
    rollout_ply: int,
    acting_information: SearchInformationState,
) -> bytes:
    return derive_seed(
        GREEDY_RESPONSE_ROLLOUT_NAMESPACE,
        seed_hex(response_request_seed),
        _validate_counter(completion_index, "completion index"),
        _validate_counter(candidate_action_index, "candidate action index"),
        _validate_counter(rollout_ply, "rollout ply"),
        information_state_fingerprint(acting_information),
    )


def actor_relative_value(
    value: float,
    value_player: EnginePlayer,
    observer: EnginePlayer,
) -> float:
    """Convert one zero-sum value between player perspectives."""

    if not math.isfinite(value) or not -1.0 <= value <= 1.0:
        raise SearchContractViolation("actor value must be finite and in [-1, 1]")
    return value if EnginePlayer(value_player) is EnginePlayer(observer) else -value


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
    state: SimulationEngineState, action_index: int
) -> EngineMove:
    actor = state.active_player
    if actor is None:
        raise SearchContractViolation("a playing simulation must have an active player")
    move = move_for_action_index(actor, action_index)
    if move not in legal_simulation_moves(state, actor):
        raise SearchContractViolation("strategic search selected a masked action")
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
        raise SearchContractViolation("continuation move refers to an empty hand slot")
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


class ShallowGreedyResponseEvaluator:
    """Compare legal actions through actor-local, uniformly completed worlds."""

    def __init__(
        self, config: ShallowResponseConfig = ShallowResponseConfig()
    ) -> None:
        if not isinstance(config, ShallowResponseConfig):
            raise SearchContractViolation("response configuration is invalid")
        self.config = config

    def evaluate(
        self,
        information: SearchInformationState,
        should_stop: Callable[[], bool] | None = None,
        *,
        candidate_representative_action_indices: tuple[int, ...] | None = None,
        response_mode: StrategicResponseMode = StrategicResponseMode.PURE,
        model_call_count: int = 0,
    ) -> GreedyResponseResult:
        if not isinstance(information, SearchInformationState):
            raise SearchContractViolation(
                "response evaluation requires a player information state"
            )
        legal = _legal_action_indexes(information)
        if not legal:
            raise SearchContractViolation("response state has no legal actions")
        groups = strategic_action_groups(
            information,
            self.config.destination_symmetry_enabled,
        )
        if not groups:
            raise SearchContractViolation(
                "response state has no strategic action groups"
            )
        representatives = tuple(
            group.representative_action_index for group in groups
        )
        if candidate_representative_action_indices is None:
            candidates = representatives
        else:
            candidates = candidate_representative_action_indices
            if (
                not isinstance(candidates, tuple)
                or not candidates
                or len(set(candidates)) != len(candidates)
                or any(candidate not in representatives for candidate in candidates)
            ):
                raise SearchContractViolation(
                    "response candidate shortlist is invalid"
                )
        candidate_groups = tuple(
            group
            for group in groups
            if group.representative_action_index in candidates
        )
        request_seed = derive_greedy_response_request_seed(
            information, self.config.digest
        )
        value_sums = [0.0] * ACTION_COUNT
        visits = [0] * ACTION_COUNT

        for completion_index in range(self.config.completions_per_action):
            self._check_interrupted(should_stop)
            shared = sample_determinization(
                information,
                derive_greedy_response_determinization_seed(
                    request_seed, completion_index
                ),
            ).state
            for group in candidate_groups:
                self._check_interrupted(should_stop)
                action_index = group.representative_action_index
                state = apply_simulation_move(
                    shared, _move_for_action(shared, action_index)
                )
                rollout_ply = 0
                while state.status is EngineStatus.PLAYING:
                    self._check_interrupted(should_stop)
                    actor = state.active_player
                    if actor is None:
                        raise SearchContractViolation(
                            "response continuation lost its active player"
                        )
                    moves = legal_simulation_moves(state, actor)
                    if len(moves) == 1:
                        move = moves[0]
                    else:
                        actor_information = (
                            project_simulation_information_state(state)
                        )
                        rollout_action = sample_uniform_action_index(
                            actor_information,
                            derive_greedy_response_rollout_seed(
                                request_seed,
                                completion_index,
                                action_index,
                                rollout_ply,
                                actor_information,
                            ),
                        )
                        move = _move_for_action(state, rollout_action)
                    state = apply_simulation_move(state, move)
                    rollout_ply += 1
                if (
                    state.status is not EngineStatus.ROUND_COMPLETE
                    or state.pending_round_result is None
                ):
                    raise SearchContractViolation(
                        "response continuation did not reach a round result"
                    )
                value_sums[action_index] += normalized_round_return(
                    state.pending_round_result, information.player
                )
                visits[action_index] += 1

        diagnostics = _group_diagnostics(groups, visits, value_sums)
        means = _expanded_group_means(diagnostics)
        selected_representative = min(
            candidates,
            key=lambda index: (-float(means[index]), index),
        )
        selected = select_concrete_action_index(
            information,
            selected_representative,
            derive_strategic_destination_choice_seed(
                request_seed,
                "shallow-response-result",
                information,
                selected_representative,
                0,
            ),
            self.config.destination_symmetry_enabled,
        )
        return GreedyResponseResult(
            information_state_fingerprint=information_state_fingerprint(
                information
            ),
            config_digest=self.config.digest,
            selected_action_index=selected,
            legal_action_indices=legal,
            mean_action_values=means,
            completions_per_action=self.config.completions_per_action,
            candidate_action_count=len(candidate_groups),
            terminal_evaluation_count=(
                len(candidate_groups) * self.config.completions_per_action
            ),
            selected_representative_action_index=selected_representative,
            group_diagnostics=diagnostics,
            response_mode=response_mode,
            shortlisted_representative_action_indices=(
                ()
                if response_mode is StrategicResponseMode.PURE
                else candidates
            ),
            model_call_count=model_call_count,
        )

    @staticmethod
    def _check_interrupted(should_stop: Callable[[], bool] | None) -> None:
        if should_stop is not None and should_stop():
            raise SearchInterrupted(
                "response evaluation interrupted before result commit"
            )


def _response_distillation_example(
    information: SearchInformationState,
    response: GreedyResponseResult,
    search_config_digest: str,
) -> ResponseDistillationExample:
    """Project a completed response into the only evidence an observer may see."""

    policy_input = policy_input_from_information_state(information)
    groups = tuple(
        ResponseActionGroupTarget(
            hand_slot=diagnostic.group.hand_slot,
            representative_action_index=(
                diagnostic.group.representative_action_index
            ),
            member_action_indices=diagnostic.group.member_action_indices,
            mean_terminal_differential=float(diagnostic.mean_value),
        )
        for diagnostic in response.group_diagnostics
    )
    return ResponseDistillationExample(
        schema_version=RESPONSE_EXAMPLE_SCHEMA_VERSION,
        ranking_schema_version=RESPONSE_RANKING_SCHEMA_VERSION,
        observation_schema_version=OBSERVATION_SCHEMA_VERSION,
        action_schema_version=ACTION_SCHEMA_VERSION,
        observation=tuple(
            bool(value) for value in policy_input.observation.tolist()
        ),
        legal_mask=tuple(
            tuple(bool(value) for value in row)
            for row in policy_input.legal_mask.tolist()
        ),
        groups=groups,
        selected_representative_action_index=(
            response.selected_representative_action_index
        ),
        selected_concrete_action_index=response.selected_action_index,
        placement_number=information.turn_number,
        actor_role=information.player.value,
        actor_is_dealer=information.player is information.dealer,
        completions_per_action=response.completions_per_action,
        search_config_digest=search_config_digest,
        response_config_digest=response.config_digest,
        cache_identity=ResponseCacheIdentity(
            response.information_state_fingerprint,
            response.config_digest,
        ),
    )


class StrategicInformationSetSearch:
    """Teacher v2 outer UCT with shallow greedy continuation responses."""

    def __init__(
        self,
        config: StrategicSearchConfig = StrategicSearchConfig(),
        *,
        response_observer: ResponseExampleObserver | None = None,
        response_ranker: StrategicGroupRanker | None = None,
    ) -> None:
        if not isinstance(config, StrategicSearchConfig):
            raise SearchContractViolation("strategic search configuration is invalid")
        if response_observer is not None and not callable(response_observer):
            raise SearchContractViolation("response observer must be callable")
        if (
            response_observer is not None
            and config.response_completions_per_action != 4
        ):
            raise SearchContractViolation(
                "response capture requires four completions per action"
            )
        mode = StrategicResponseMode(config.response_mode)
        if response_observer is not None and mode is not StrategicResponseMode.PURE:
            raise SearchContractViolation(
                "response evidence capture is unavailable in hybrid modes"
            )
        if mode is StrategicResponseMode.PURE:
            if response_ranker is not None:
                raise SearchContractViolation(
                    "pure Teacher v2 cannot receive a response ranker"
                )
        elif (
            response_ranker is None
            or not hasattr(response_ranker, "rank")
            or response_ranker.artifact_digest
            != config.response_ranker_artifact_digest
        ):
            raise SearchContractViolation(
                "hybrid response mode requires its configured ranker artifact"
            )
        self.config = config
        self._response_observer = response_observer
        self._response_ranker = response_ranker
        self._response_evaluator = ShallowGreedyResponseEvaluator(
            config.response_config
        )

    def _actor_response(
        self,
        information: SearchInformationState,
        should_stop: Callable[[], bool] | None,
    ) -> GreedyResponseResult:
        if not isinstance(information, SearchInformationState):
            raise SearchContractViolation(
                "actor response requires a player information state"
            )
        mode = StrategicResponseMode(self.config.response_mode)
        if mode is StrategicResponseMode.PURE:
            return self._response_evaluator.evaluate(information, should_stop)
        return self._student_response(information, should_stop, mode)

    def _student_response(
        self,
        information: SearchInformationState,
        should_stop: Callable[[], bool] | None,
        mode: StrategicResponseMode,
    ) -> GreedyResponseResult:
        if self._response_ranker is None:
            raise SearchContractViolation("hybrid response ranker is unavailable")
        groups = strategic_action_groups(
            information,
            self.config.destination_symmetry_enabled,
        )
        ranking = self._response_ranker.rank(information, groups)
        representatives = tuple(
            group.representative_action_index for group in groups
        )
        if (
            not isinstance(ranking, tuple)
            or len(ranking) != len(representatives)
            or set(ranking) != set(representatives)
        ):
            raise SearchContractViolation(
                "response ranker must order every legal strategic group exactly"
            )
        shortlist = ranking[: min(mode.shortlist_size, len(ranking))]
        if mode in (
            StrategicResponseMode.STUDENT_TOP_2,
            StrategicResponseMode.STUDENT_TOP_3,
        ):
            return self._response_evaluator.evaluate(
                information,
                should_stop,
                candidate_representative_action_indices=shortlist,
                response_mode=mode,
                model_call_count=1,
            )
        selected_representative = shortlist[0]
        student_seed = derive_hybrid_response_request_seed(
            information,
            self._response_ranker.artifact_digest,
            mode.shortlist_size,
            self.config.response_config.digest,
        )
        selected = select_concrete_action_index(
            information,
            selected_representative,
            derive_strategic_destination_choice_seed(
                student_seed,
                "student-direct-result",
                information,
                selected_representative,
                0,
            ),
            self.config.destination_symmetry_enabled,
        )
        diagnostics = tuple(
            StrategicActionGroupDiagnostic(group, 0, None)
            for group in groups
        )
        return GreedyResponseResult(
            information_state_fingerprint=information_state_fingerprint(
                information
            ),
            config_digest=self.config.response_config.digest,
            selected_action_index=selected,
            legal_action_indices=_legal_action_indexes(information),
            mean_action_values=_expanded_group_means(diagnostics),
            completions_per_action=self.config.response_completions_per_action,
            candidate_action_count=0,
            terminal_evaluation_count=0,
            selected_representative_action_index=selected_representative,
            group_diagnostics=diagnostics,
            response_mode=mode,
            shortlisted_representative_action_indices=shortlist,
            model_call_count=1,
        )

    def search(
        self,
        information: SearchInformationState,
        request_seed: bytes,
        should_stop: Callable[[], bool] | None = None,
    ) -> StrategicSearchResult:
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
            raise SearchContractViolation("strategic search root has no legal actions")

        if len(root_legal) == 1:
            return StrategicSearchResult(
                information_state_fingerprint=root_fingerprint,
                config_digest=self.config.digest,
                selected_action_index=root_legal[0],
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
                response_terminal_evaluation_count=0,
                total_terminal_evaluation_count=0,
                selected_representative_action_index=root_legal[0],
                group_diagnostics=(),
                response_model_call_count=0,
            )
        root_groups = strategic_action_groups(
            information,
            self.config.destination_symmetry_enabled,
        )
        if self.config.outer_simulation_budget < len(root_groups):
            raise SearchContractViolation(
                "outer simulation budget must visit every strategic root action"
            )

        nodes: dict[StrategicNodeKey, _InformationNode] = {}
        response_cache: dict[
            tuple[str, str, str | None, int], GreedyResponseResult
        ] = {}
        continuations: list[_ObservedContinuation] = []
        response_requests = 0
        response_cache_hits = 0
        response_candidates = 0
        response_terminals = 0
        response_model_calls = 0

        for simulation_index in range(self.config.outer_simulation_budget):
            self._check_interrupted(should_stop)
            sampled = sample_determinization(
                information,
                derive_strategic_determinization_seed(
                    request_seed, simulation_index
                ),
            )
            state = sampled.state
            path: list[tuple[_InformationNode, int]] = []
            steps: list[ContinuationStep] = []
            expanded = False
            root_representative_action_index: int | None = None
            root_concrete_action_index: int | None = None

            while state.status is EngineStatus.PLAYING:
                self._check_interrupted(should_stop)
                actor = state.active_player
                if actor is None:
                    raise SearchContractViolation(
                        "playing simulation lost its active player"
                    )
                moves = legal_simulation_moves(state, actor)
                forced = len(moves) == 1
                if forced:
                    move = moves[0]
                    action_index = action_index_for_move(move, actor)
                elif actor is root_player and not expanded:
                    actor_information = project_simulation_information_state(state)
                    node_key = StrategicNodeKey(
                        actor,
                        information_state_fingerprint(actor_information),
                    )
                    action_groups = strategic_action_groups(
                        actor_information,
                        self.config.destination_symmetry_enabled,
                    )
                    node = nodes.get(node_key)
                    if node is None:
                        node = _InformationNode.create(node_key, action_groups)
                        nodes[node_key] = node
                    elif node.action_groups != action_groups:
                        raise SearchContractViolation(
                            "one strategic information node changed action groups"
                        )
                    representative_action_index = _select_tree_action(
                        node,
                        float(self.config.outer_exploration_constant),
                        derive_strategic_selection_seed(
                            request_seed, simulation_index, node_key
                        ),
                    )
                    path.append((node, representative_action_index))
                    if node.action_visits[representative_action_index] == 0:
                        expanded = True
                    action_index = select_concrete_action_index(
                        actor_information,
                        representative_action_index,
                        derive_strategic_destination_choice_seed(
                            request_seed,
                            "outer-simulation",
                            actor_information,
                            representative_action_index,
                            simulation_index,
                        ),
                        self.config.destination_symmetry_enabled,
                    )
                    move = _move_for_action(state, action_index)
                else:
                    actor_information = project_simulation_information_state(state)
                    cache_key = (
                        information_state_fingerprint(actor_information),
                        self.config.response_config.digest,
                        self.config.response_ranker_artifact_digest,
                        self.config.response_shortlist_size,
                    )
                    response_requests += 1
                    response = response_cache.get(cache_key)
                    if response is None:
                        response = self._actor_response(
                            actor_information, should_stop
                        )
                        if (
                            response.information_state_fingerprint
                            != cache_key[0]
                            or response.config_digest != cache_key[1]
                            or response.response_mode
                            is not self.config.response_mode
                        ):
                            raise SearchContractViolation(
                                "actor response crossed its information boundary"
                            )
                        if self._response_observer is not None:
                            self._response_observer(
                                _response_distillation_example(
                                    actor_information,
                                    response,
                                    self.config.digest,
                                )
                            )
                        response_cache[cache_key] = response
                        response_candidates += response.candidate_action_count
                        response_terminals += response.terminal_evaluation_count
                        response_model_calls += response.model_call_count
                    else:
                        response_cache_hits += 1
                    action_index = response.selected_action_index
                    move = _move_for_action(state, action_index)

                if root_representative_action_index is None:
                    if actor is not root_player:
                        raise SearchContractViolation(
                            "root simulation began on another actor"
                        )
                    root_representative_action_index = (
                        representative_action_index
                    )
                    root_concrete_action_index = action_index
                steps.append(
                    _continuation_step(
                        root_player, state, move, action_index, forced
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
            if (
                root_representative_action_index is None
                or root_concrete_action_index is None
                or not path
            ):
                raise SearchContractViolation(
                    "a learned strategic simulation has no root path"
                )
            root_value = normalized_round_return(
                state.pending_round_result, root_player
            )
            for node, action_index in path:
                backed_value = actor_relative_value(
                    root_value, root_player, node.key.actor
                )
                node.visits += 1
                node.action_visits[action_index] += 1
                node.action_value_sums[action_index] += backed_value
            continuations.append(
                _ObservedContinuation(
                    root_representative_action_index,
                    root_concrete_action_index,
                    root_value,
                    simulation_index,
                    tuple(steps),
                )
            )

        root_key = StrategicNodeKey(root_player, root_fingerprint)
        root_node = nodes.get(root_key)
        if (
            root_node is None
            or root_node.visits != self.config.outer_simulation_budget
        ):
            raise SearchContractViolation(
                "strategic root backup count does not match simulations"
            )
        selected_representative = min(
            root_node.legal_actions,
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
                if item.root_representative_action_index
                == selected_representative
            ),
            key=lambda item: (-item.terminal_value, item.simulation_index),
        )
        diagnostics = _group_diagnostics(
            root_node.action_groups,
            root_node.action_visits,
            root_node.action_value_sums,
        )
        projected_visits, concrete_by_representative = (
            _project_group_visits_to_concrete_actions(
                information,
                request_seed,
                diagnostics,
                self.config.destination_symmetry_enabled,
            )
        )
        selected = concrete_by_representative[selected_representative]
        means = _expanded_group_means(diagnostics)
        return StrategicSearchResult(
            information_state_fingerprint=root_fingerprint,
            config_digest=self.config.digest,
            selected_action_index=selected,
            action_visits=projected_visits,
            mean_action_values=means,
            simulation_count=root_node.visits,
            information_set_count=len(nodes),
            principal_continuation=PrincipalContinuation(
                representative.root_concrete_action_index,
                representative.terminal_value,
                representative.simulation_index,
                representative.steps,
            ),
            elapsed_seconds=time.perf_counter() - started,
            peak_resident_memory_bytes=_peak_resident_memory_bytes(),
            response_request_count=response_requests,
            unique_response_evaluation_count=len(response_cache),
            response_cache_hit_count=response_cache_hits,
            response_candidate_action_count=response_candidates,
            response_terminal_evaluation_count=response_terminals,
            total_terminal_evaluation_count=(
                root_node.visits + response_terminals
            ),
            selected_representative_action_index=selected_representative,
            group_diagnostics=diagnostics,
            response_model_call_count=response_model_calls,
        )

    @staticmethod
    def _check_interrupted(should_stop: Callable[[], bool] | None) -> None:
        if should_stop is not None and should_stop():
            raise SearchInterrupted(
                "strategic search interrupted before result commit"
            )


__all__ = (
    "GREEDY_RESPONSE_CONTINUATION_PROFILE",
    "GREEDY_RESPONSE_DETERMINIZATION_NAMESPACE",
    "GREEDY_RESPONSE_REQUEST_NAMESPACE",
    "GREEDY_RESPONSE_ROLLOUT_NAMESPACE",
    "GREEDY_RESPONSE_SCHEMA_VERSION",
    "GREEDY_RESPONSE_SELECTION_PROFILE",
    "HYBRID_RESPONSE_REQUEST_NAMESPACE",
    "HYBRID_RESPONSE_SCHEMA_VERSION",
    "STRATEGIC_DETERMINIZATION_NAMESPACE",
    "STRATEGIC_DESTINATION_CHOICE_NAMESPACE",
    "STRATEGIC_DESTINATION_CHOICE_PROFILE",
    "STRATEGIC_SEARCH_REQUEST_NAMESPACE",
    "STRATEGIC_SEARCH_SCHEMA_VERSION",
    "STRATEGIC_SELECTION_NAMESPACE",
    "STRATEGIC_SELECTION_PROFILE",
    "SUPPORTED_RESPONSE_COMPLETIONS",
    "GreedyResponseResult",
    "ShallowGreedyResponseEvaluator",
    "ShallowResponseConfig",
    "StrategicActionGroup",
    "StrategicActionGroupDiagnostic",
    "StrategicInformationSetSearch",
    "StrategicGroupRanker",
    "StrategicNodeKey",
    "StrategicResponseMode",
    "StrategicSearchConfig",
    "StrategicSearchResult",
    "actor_relative_value",
    "derive_greedy_response_determinization_seed",
    "derive_greedy_response_request_seed",
    "derive_greedy_response_rollout_seed",
    "derive_hybrid_response_request_seed",
    "derive_strategic_destination_choice_seed",
    "derive_strategic_determinization_seed",
    "derive_strategic_search_request_seed",
    "derive_strategic_selection_seed",
    "select_concrete_action_index",
    "strategic_action_groups",
)

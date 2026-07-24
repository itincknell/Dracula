"""Information-set search behind the local gameplay opponent boundary."""

from __future__ import annotations

import logging
import math

from dracula.api.service import (
    PolicyDescriptor,
    PolicyTurnRequest,
    PolicyTurnResult,
)
from dracula.policy_adapter import PolicyContractError
from dracula.search import (
    INFORMATION_STATE_SCHEMA_VERSION,
    SEARCH_SCHEMA_VERSION,
    STRATEGIC_SEARCH_SCHEMA_VERSION,
    STRATEGIC_SELECTION_PROFILE,
    InformationSetSearch,
    SearchConfig,
    SearchInformationState,
    StrategicResponseMode,
    StrategicInformationSetSearch,
    StrategicSearchConfig,
    ResponseRankerGroupEvaluator,
    derive_search_request_seed,
    derive_strategic_search_request_seed,
)

SEARCH_ACTION_SCHEMA_VERSION = "dracula-action-map-v1"
SEARCH_STATE_SCHEMA_VERSION = "stateless-v1"
SEARCH_SELECTION_PROFILE = "max-visits-v1"

LOGGER = logging.getLogger("uvicorn.error")


def _validate_request(
    request: PolicyTurnRequest,
    descriptor: PolicyDescriptor,
) -> SearchInformationState:
    if request.policy != descriptor:
        raise PolicyContractError("game opponent session does not match search")
    if not isinstance(request.information_state, SearchInformationState):
        raise PolicyContractError("search requires a player information state")
    information = request.information_state
    if (
        information.player is not request.player
        or information.round_number != request.round_number
        or information.turn_number != request.turn_number
    ):
        raise PolicyContractError("search information does not match the requested turn")
    return information


def _validate_selected_action(
    request: PolicyTurnRequest,
    selected_action_index: int,
) -> None:
    if (
        selected_action_index >= len(request.action_table)
        or request.action_table[selected_action_index] is None
    ):
        raise PolicyContractError("search selected an action outside service legality")


class InlineSearchExecutor:
    """Run the validated search using only the actor's information state."""

    def __init__(self, config: SearchConfig = SearchConfig()) -> None:
        self.planner = InformationSetSearch(config)

    @classmethod
    def from_values(
        cls,
        simulation_budget: int = 500,
        exploration_constant: float = math.sqrt(2.0),
    ) -> InlineSearchExecutor:
        return cls(SearchConfig(simulation_budget, exploration_constant))

    @property
    def descriptor(self) -> PolicyDescriptor:
        digest = self.planner.config.digest
        return PolicyDescriptor(
            policy_id="information-set-search",
            policy_version=SEARCH_SCHEMA_VERSION,
            artifact_id=f"sha256:{digest}",
            artifact_sha256=digest,
            observation_schema_version=INFORMATION_STATE_SCHEMA_VERSION,
            action_schema_version=SEARCH_ACTION_SCHEMA_VERSION,
            hidden_state_schema_version=SEARCH_STATE_SCHEMA_VERSION,
            inference_profile=SEARCH_SELECTION_PROFILE,
        )

    def invoke(self, request: PolicyTurnRequest) -> PolicyTurnResult:
        information = _validate_request(request, self.descriptor)

        request_seed = derive_search_request_seed(
            str(request.game_id), information, self.planner.config.digest
        )
        result = self.planner.search(information, request_seed)
        _validate_selected_action(request, result.selected_action_index)
        return PolicyTurnResult(result.selected_action_index, None)


class InlineStrategicSearchExecutor:
    """Run Teacher v2 behind the same stateless gameplay boundary as v1."""

    def __init__(
        self,
        config: StrategicSearchConfig = StrategicSearchConfig(),
        response_ranker: ResponseRankerGroupEvaluator | None = None,
    ) -> None:
        self.planner = StrategicInformationSetSearch(
            config, response_ranker=response_ranker
        )

    @classmethod
    def from_values(
        cls,
        outer_simulation_budget: int,
        response_completions_per_action: int,
        outer_exploration_constant: float = math.sqrt(2.0),
        *,
        response_mode: StrategicResponseMode | str = StrategicResponseMode.PURE,
        response_ranker_artifact: str | None = None,
    ) -> InlineStrategicSearchExecutor:
        mode = StrategicResponseMode(response_mode)
        ranker = (
            None
            if mode is StrategicResponseMode.PURE
            else ResponseRankerGroupEvaluator.from_artifact(
                response_ranker_artifact or ""
            )
        )
        return cls(
            StrategicSearchConfig(
                outer_simulation_budget=outer_simulation_budget,
                response_completions_per_action=response_completions_per_action,
                outer_exploration_constant=outer_exploration_constant,
                response_mode=mode,
                response_ranker_artifact_digest=(
                    None if ranker is None else ranker.artifact_digest
                ),
            ),
            response_ranker=ranker,
        )

    @property
    def descriptor(self) -> PolicyDescriptor:
        digest = self.planner.config.digest
        mode = StrategicResponseMode(self.planner.config.response_mode)
        return PolicyDescriptor(
            policy_id=(
                "strategic-information-set-search"
                if mode is StrategicResponseMode.PURE
                else f"strategic-information-set-search-{mode.value}"
            ),
            policy_version=STRATEGIC_SEARCH_SCHEMA_VERSION,
            artifact_id=f"sha256:{digest}",
            artifact_sha256=digest,
            observation_schema_version=INFORMATION_STATE_SCHEMA_VERSION,
            action_schema_version=SEARCH_ACTION_SCHEMA_VERSION,
            hidden_state_schema_version=SEARCH_STATE_SCHEMA_VERSION,
            inference_profile=(
                STRATEGIC_SELECTION_PROFILE
                if mode is StrategicResponseMode.PURE
                else f"{STRATEGIC_SELECTION_PROFILE}+{mode.value}"
            ),
        )

    def invoke(self, request: PolicyTurnRequest) -> PolicyTurnResult:
        information = _validate_request(request, self.descriptor)
        request_seed = derive_strategic_search_request_seed(
            str(request.game_id), information, self.planner.config.digest
        )
        result = self.planner.search(information, request_seed)
        _validate_selected_action(request, result.selected_action_index)
        LOGGER.info(
            (
                "teacher-v2 turn controller=%s config_sha256=%s "
                "game_id=%s round=%d turn=%d "
                "representative_action=%d representative_grid=%d "
                "concrete_action=%d concrete_grid=%d "
                "outer_simulations=%d response_completions=%d "
                "response_requests=%d candidate_groups=%d "
                "response_evaluations=%d cache_hits=%d "
                "model_calls=%d elapsed_seconds=%.3f"
            ),
            self.descriptor.policy_id,
            self.planner.config.digest,
            request.game_id,
            request.round_number,
            request.turn_number,
            result.selected_representative_action_index,
            result.selected_representative_grid_index,
            result.selected_action_index,
            result.selected_concrete_grid_index,
            result.simulation_count,
            self.planner.config.response_completions_per_action,
            result.response_request_count,
            result.response_candidate_action_count,
            result.response_terminal_evaluation_count,
            result.response_cache_hit_count,
            result.response_model_call_count,
            result.elapsed_seconds,
        )
        return PolicyTurnResult(result.selected_action_index, None)


__all__ = (
    "SEARCH_ACTION_SCHEMA_VERSION",
    "SEARCH_SELECTION_PROFILE",
    "SEARCH_STATE_SCHEMA_VERSION",
    "InlineSearchExecutor",
    "InlineStrategicSearchExecutor",
)

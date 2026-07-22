"""Information-set search behind the local gameplay opponent boundary."""

from __future__ import annotations

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
    InformationSetSearch,
    SearchConfig,
    SearchInformationState,
    derive_search_request_seed,
)

SEARCH_ACTION_SCHEMA_VERSION = "dracula-action-map-v1"
SEARCH_STATE_SCHEMA_VERSION = "stateless-v1"
SEARCH_SELECTION_PROFILE = "max-visits-v1"


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
        if request.policy != self.descriptor:
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

        request_seed = derive_search_request_seed(
            str(request.game_id), information, self.planner.config.digest
        )
        result = self.planner.search(information, request_seed)
        if (
            result.selected_action_index >= len(request.action_table)
            or request.action_table[result.selected_action_index] is None
        ):
            raise PolicyContractError("search selected an action outside service legality")
        return PolicyTurnResult(result.selected_action_index, None)


__all__ = (
    "SEARCH_ACTION_SCHEMA_VERSION",
    "SEARCH_SELECTION_PROFILE",
    "SEARCH_STATE_SCHEMA_VERSION",
    "InlineSearchExecutor",
)

"""Neural-guided search behind the application opponent boundary."""

from __future__ import annotations

import hashlib
from pathlib import Path

from dracula.api.service import PolicyDescriptor, PolicyTurnRequest, PolicyTurnResult
from dracula.policy_adapter import PolicyContractError
from dracula.search import (
    GUIDED_SEARCH_SCHEMA_VERSION,
    INFORMATION_STATE_SCHEMA_VERSION,
    GuidedInformationSetSearch,
    GuidedSearchConfig,
    PolicyValueModelEvaluator,
    SearchInformationState,
    derive_search_request_seed,
)

GUIDED_ACTION_SCHEMA_VERSION = "dracula-action-map-v1"
GUIDED_STATE_SCHEMA_VERSION = "stateless-v1"
GUIDED_SELECTION_PROFILE = "max-visits-v1"


class InlineGuidedSearchExecutor:
    """Run guided search from the sanitized information state in a turn request."""

    def __init__(
        self,
        artifact_path: str | Path,
        config: GuidedSearchConfig = GuidedSearchConfig(),
    ) -> None:
        path = Path(artifact_path).expanduser().resolve()
        self.artifact_path = path
        self.artifact_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        self.evaluator = PolicyValueModelEvaluator.from_artifact(path)
        self.planner = GuidedInformationSetSearch(self.evaluator, config)

    @property
    def descriptor(self) -> PolicyDescriptor:
        return PolicyDescriptor(
            policy_id="guided-information-set-search",
            policy_version=GUIDED_SEARCH_SCHEMA_VERSION,
            artifact_id=f"sha256:{self.artifact_sha256}",
            artifact_sha256=self.artifact_sha256,
            observation_schema_version=INFORMATION_STATE_SCHEMA_VERSION,
            action_schema_version=GUIDED_ACTION_SCHEMA_VERSION,
            hidden_state_schema_version=GUIDED_STATE_SCHEMA_VERSION,
            inference_profile=GUIDED_SELECTION_PROFILE,
        )

    def invoke(self, request: PolicyTurnRequest) -> PolicyTurnResult:
        if request.policy != self.descriptor:
            raise PolicyContractError("game opponent session does not match guided search")
        if not isinstance(request.information_state, SearchInformationState):
            raise PolicyContractError("guided search requires a player information state")
        information = request.information_state
        if (
            information.player is not request.player
            or information.round_number != request.round_number
            or information.turn_number != request.turn_number
        ):
            raise PolicyContractError(
                "guided-search information does not match the requested turn"
            )
        request_seed = derive_search_request_seed(
            str(request.game_id), information, self.planner.digest
        )
        result = self.planner.search(information, request_seed)
        if (
            result.selected_action_index >= len(request.action_table)
            or request.action_table[result.selected_action_index] is None
        ):
            raise PolicyContractError("guided search selected an illegal action")
        return PolicyTurnResult(result.selected_action_index, None)


__all__ = (
    "GUIDED_ACTION_SCHEMA_VERSION",
    "GUIDED_SELECTION_PROFILE",
    "GUIDED_STATE_SCHEMA_VERSION",
    "InlineGuidedSearchExecutor",
)

"""Adapt the selected standalone policy to live gameplay decisions.

The runtime loads one verified artifact, performs actor-visible masked inference,
resolves paired destinations deterministically, and returns one legal action
through the shared gameplay-policy boundary.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

from dracula.action_contract import POLICY_DESTINATION_SCOPE
from dracula.api.policy import (
    PolicyContractError,
    PolicyDescriptor,
    PolicyTurnRequest,
    PolicyTurnResult,
)
from dracula.bgc_policy import load_bgc_policy_artifact
from dracula.bgc_policy_model import BGCPolicyModel
from dracula.policy_observation import select_policy_group
from dracula.randomness import derive_seed
from dracula.search.information import (
    INFORMATION_STATE_SCHEMA_VERSION,
    SearchInformationState,
    information_state_fingerprint,
)
from dracula.strategic_actions import (
    derive_strategic_destination_choice_seed,
    select_concrete_action_index,
)

# The selected artifact's validated paired-destination behavior depends on this
# exact seed namespace.
ACTIVE_POLICY_REQUEST_NAMESPACE = "dracula-pi0-standalone-request-v1"
ACTIVE_POLICY_ACTION_SCHEMA_VERSION = "dracula-action-map-v1"
ACTIVE_POLICY_STATE_SCHEMA_VERSION = "stateless-v1"
LOGGER = logging.getLogger("uvicorn.error")


@dataclass(frozen=True, slots=True)
class ActivePolicyDecision:
    """Selected proxy, resolved legal action, and complete decision latency."""

    representative_action_index: int
    concrete_action_index: int
    latency_seconds: float


class ActivePolicyRuntime:
    """Load and run one production policy without evaluation-code dependencies."""

    def __init__(
        self,
        model: BGCPolicyModel,
        *,
        artifact_digest: str,
    ) -> None:
        self.model = model.cpu().eval().requires_grad_(False)
        self.artifact_digest = artifact_digest

    @classmethod
    def from_artifact(cls, path: str | Path) -> ActivePolicyRuntime:
        """Load the selected artifact through the strict shared verifier."""

        artifact_path = Path(path).expanduser().resolve()
        loaded = load_bgc_policy_artifact(artifact_path)
        return cls(
            loaded.model,
            artifact_digest=loaded.artifact_digest,
        )

    def decide(
        self,
        information: SearchInformationState,
        *,
        fixture_id: str,
        decision_index: int,
    ) -> ActivePolicyDecision:
        """Run one actor-visible inference and resolve its paired destination."""

        started = time.perf_counter()
        selected_group = select_policy_group(self.model, information)
        representative = selected_group.representative_action_index
        # Concrete paired placement uses a separate deterministic stream, so it
        # cannot alter which strategic group the model selected.
        request_seed = derive_seed(
            ACTIVE_POLICY_REQUEST_NAMESPACE,
            fixture_id,
            information_state_fingerprint(information),
            self.artifact_digest,
            str(decision_index),
        )
        concrete = select_concrete_action_index(
            selected_group,
            derive_strategic_destination_choice_seed(
                request_seed,
                POLICY_DESTINATION_SCOPE,
                information,
                selected_group,
                0,
            ),
        )
        return ActivePolicyDecision(
            representative_action_index=representative,
            concrete_action_index=concrete,
            latency_seconds=time.perf_counter() - started,
        )


class ActivePolicyExecutor:
    """Adapt the selected standalone policy to the gameplay policy boundary."""

    def __init__(self, artifact_path: str | Path) -> None:
        self.policy = ActivePolicyRuntime.from_artifact(artifact_path)
        self._descriptor = PolicyDescriptor(
            policy_id="standalone-bgc-policy",
            policy_version="dracula-standalone-bgc-policy-v1",
            artifact_id=f"sha256:{self.policy.artifact_digest}",
            artifact_sha256=self.policy.artifact_digest,
            observation_schema_version=INFORMATION_STATE_SCHEMA_VERSION,
            action_schema_version=ACTIVE_POLICY_ACTION_SCHEMA_VERSION,
            hidden_state_schema_version=ACTIVE_POLICY_STATE_SCHEMA_VERSION,
            inference_profile="representative-argmax-v1",
        )

    @property
    def descriptor(self) -> PolicyDescriptor:
        return self._descriptor

    def invoke(self, request: PolicyTurnRequest) -> PolicyTurnResult:
        if request.policy != self.descriptor:
            raise PolicyContractError("game opponent session does not match policy")
        if not isinstance(request.information_state, SearchInformationState):
            raise PolicyContractError("policy requires a player information state")
        information = request.information_state
        if (
            information.player is not request.player
            or information.round_number != request.round_number
            or information.turn_number != request.turn_number
        ):
            raise PolicyContractError(
                "policy information does not match the requested turn"
            )
        decision = self.policy.decide(
            information,
            fixture_id=str(request.game_id),
            decision_index=request.turn_number,
        )
        selected = decision.concrete_action_index
        if (
            selected >= len(request.action_table)
            or request.action_table[selected] is None
        ):
            raise PolicyContractError(
                "policy selected an action outside service legality"
            )
        LOGGER.info(
            (
                "standalone-policy turn artifact_sha256=%s game_id=%s "
                "round=%d turn=%d representative_action=%d concrete_action=%d "
                "decision_latency_seconds=%.6f"
            ),
            self.policy.artifact_digest,
            request.game_id,
            request.round_number,
            request.turn_number,
            decision.representative_action_index,
            decision.concrete_action_index,
            decision.latency_seconds,
        )
        return PolicyTurnResult(selected, None)


__all__ = (
    "ACTIVE_POLICY_REQUEST_NAMESPACE",
    "ActivePolicyDecision",
    "ActivePolicyExecutor",
    "ActivePolicyRuntime",
)

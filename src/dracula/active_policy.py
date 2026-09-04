"""Standalone runtime for the selected production policy artifact."""

from __future__ import annotations

import hashlib
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch

from dracula.api.policy import (
    PolicyDescriptor,
    PolicyTurnRequest,
    PolicyTurnResult,
)
from dracula.bgc_policy import (
    BGC_POLICY_CANDIDATE_STATUS,
    load_bgc_policy_artifact,
)
from dracula.bgc_policy_model import (
    BGCPolicyModel,
)
from dracula.api.policy import PolicyContractError
from dracula.randomness import derive_seed
from dracula.action_contract import (
    INFERENCE_DESTINATION_SCOPE,
    build_representative_action_projection,
    select_representative_action,
)
from dracula.search.information import (
    INFORMATION_STATE_SCHEMA_VERSION,
    SearchInformationState,
    information_state_fingerprint,
)
from dracula.policy_observation import (
    candidate_action_tensor,
    encode_policy_observation,
    engine_action_index_from_candidate,
)
from dracula.strategic_actions import (
    derive_strategic_destination_choice_seed,
    select_concrete_action_index,
    strategic_action_groups,
)

# The literal predates pi1; changing it would change deterministic paired moves.
ACTIVE_POLICY_REQUEST_NAMESPACE = "dracula-pi0-standalone-request-v1"
ACTIVE_POLICY_IDENTITY = "standalone-policy-v1"
ACTIVE_POLICY_ACTION_SCHEMA_VERSION = "dracula-action-map-v1"
ACTIVE_POLICY_STATE_SCHEMA_VERSION = "stateless-v1"
LOGGER = logging.getLogger("uvicorn.error")


@dataclass(frozen=True, slots=True)
class ActivePolicyDecision:
    """Selected proxy, resolved legal action, and measured inference latency."""

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
        artifact_identity: str,
    ) -> None:
        if not isinstance(model, BGCPolicyModel):
            raise ValueError("active policy requires BGCPolicyModel")
        if len(artifact_digest) != 64 or any(
            character not in "0123456789abcdef" for character in artifact_digest
        ):
            raise ValueError("active policy artifact digest is invalid")
        if not artifact_identity:
            raise ValueError("active policy artifact identity is invalid")
        self.model = model.cpu().eval().requires_grad_(False)
        self.artifact_digest = artifact_digest
        self.artifact_identity = artifact_identity

    @classmethod
    def from_artifact(cls, path: str | Path) -> ActivePolicyRuntime:
        """Load the selected artifact through the strict shared verifier."""

        artifact_path = Path(path).expanduser().resolve()
        loaded = load_bgc_policy_artifact(artifact_path)
        # Training artifacts retain their candidate marker; production approval
        # is the separately pinned release digest, not a mutation of the file.
        if loaded.metadata.candidate_status != BGC_POLICY_CANDIDATE_STATUS:
            raise ValueError("active policy input must be an unaccepted candidate")
        return cls(
            loaded.model,
            artifact_digest=hashlib.sha256(artifact_path.read_bytes()).hexdigest(),
            artifact_identity=loaded.metadata.state_dict_digest,
        )

    def decide(
        self,
        information: SearchInformationState,
        *,
        fixture_id: str,
        decision_index: int,
        should_stop: Callable[[], bool] | None = None,
    ) -> ActivePolicyDecision:
        """Run one actor-visible inference and resolve its paired destination."""

        if should_stop is not None and should_stop():
            raise InterruptedError("active policy inference interrupted")
        started = time.perf_counter()
        engine_legal_mask = torch.tensor(information.legal_mask, dtype=torch.bool)
        # Symmetry supplies one proxy per strategic group in engine-slot rows;
        # candidate rows are then derived from current-card membership.
        groups = strategic_action_groups(information, True)
        projection = build_representative_action_projection(
            engine_legal_mask, groups
        )
        observation = encode_policy_observation(information)
        compact_mask = candidate_action_tensor(information, projection.mask)
        with torch.inference_mode():
            logits = self.model(observation)
        compact_representative = select_representative_action(logits, compact_mask)
        if not isinstance(compact_representative, int):
            raise ValueError("single-state active policy returned batched selection")
        representative = engine_action_index_from_candidate(
            information, compact_representative
        )
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
            information,
            representative,
            derive_strategic_destination_choice_seed(
                request_seed,
                INFERENCE_DESTINATION_SCOPE,
                information,
                representative,
                0,
            ),
            True,
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

    @property
    def descriptor(self) -> PolicyDescriptor:
        return PolicyDescriptor(
            policy_id="standalone-bgc-policy",
            policy_version="dracula-standalone-bgc-policy-v1",
            artifact_id=f"sha256:{self.policy.artifact_digest}",
            artifact_sha256=self.policy.artifact_digest,
            observation_schema_version=INFORMATION_STATE_SCHEMA_VERSION,
            action_schema_version=ACTIVE_POLICY_ACTION_SCHEMA_VERSION,
            hidden_state_schema_version=ACTIVE_POLICY_STATE_SCHEMA_VERSION,
            inference_profile="representative-argmax-v1",
        )

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
        if selected >= len(request.action_table) or request.action_table[selected] is None:
            raise PolicyContractError("policy selected an action outside service legality")
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

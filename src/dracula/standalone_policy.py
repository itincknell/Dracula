"""Standalone Sam-policy inference behind the gameplay opponent boundary."""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

import torch

from dracula.api.policy import (
    PolicyDescriptor,
    PolicyTurnRequest,
    PolicyTurnResult,
)
from dracula.bridge import PolicyInput, PolicyTurnKind
from dracula.policy_adapter import PolicyContractError
from dracula.randomness import derive_seed
from dracula.sam_policy import (
    ACTION_SCHEMA_VERSION,
    INFERENCE_DESTINATION_SCOPE,
    MODEL_SCHEMA_VERSION,
    OBSERVATION_SCHEMA_VERSION,
    REPRESENTATIVE_MASK_SCHEMA_VERSION,
    LoadedSamPolicyArtifact,
    build_representative_action_projection,
    load_sam_policy_artifact,
    select_representative_action,
)
from dracula.search import (
    SearchInformationState,
    derive_strategic_destination_choice_seed,
    information_state_fingerprint,
    policy_input_from_information_state,
    select_concrete_action_index,
    strategic_action_groups,
)

STANDALONE_POLICY_CONTROLLER_VERSION = (
    "dracula-standalone-sam-policy-controller-v1"
)
STANDALONE_POLICY_STATE_SCHEMA_VERSION = "stateless-v1"
STANDALONE_POLICY_INFERENCE_PROFILE = (
    "representative-argmax-fair-coin-v1"
)
STANDALONE_POLICY_REQUEST_SEED_NAMESPACE = (
    "dracula-standalone-sam-policy-request-v1"
)

LOGGER = logging.getLogger("uvicorn.error")


@dataclass(frozen=True, slots=True)
class StandalonePolicyDecision:
    representative_action_index: int
    concrete_action_index: int
    strategic_group_count: int
    latency_seconds: float
    information_state_digest: str


def derive_standalone_policy_request_seed(
    *,
    game_id: UUID | str,
    information: SearchInformationState,
    artifact_sha256: str,
) -> bytes:
    if not isinstance(information, SearchInformationState):
        raise PolicyContractError(
            "standalone policy requires a player information state"
        )
    if (
        not isinstance(artifact_sha256, str)
        or len(artifact_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in artifact_sha256
        )
    ):
        raise PolicyContractError(
            "standalone policy artifact digest is invalid"
        )
    return derive_seed(
        STANDALONE_POLICY_REQUEST_SEED_NAMESPACE,
        str(game_id),
        information_state_fingerprint(information),
        artifact_sha256,
    )


def _policy_inputs_equal(first: PolicyInput, second: PolicyInput) -> bool:
    return torch.equal(
        first.observation.detach().cpu(),
        second.observation.detach().cpu(),
    ) and torch.equal(
        first.legal_mask.detach().cpu(),
        second.legal_mask.detach().cpu(),
    )


class StandaloneSamPolicyExecutor:
    """Run one immutable Sam classifier from player-visible information."""

    def __init__(self, artifact_path: str | Path) -> None:
        self.artifact_path = Path(artifact_path).expanduser().resolve()
        try:
            artifact_bytes = self.artifact_path.read_bytes()
        except OSError as error:
            raise PolicyContractError(
                "standalone policy artifact could not be read"
            ) from error
        self.artifact_sha256 = hashlib.sha256(artifact_bytes).hexdigest()
        self.loaded: LoadedSamPolicyArtifact = load_sam_policy_artifact(
            self.artifact_path
        )
        self.model = self.loaded.model.cpu().eval().requires_grad_(False)

    @property
    def descriptor(self) -> PolicyDescriptor:
        metadata = self.loaded.metadata
        return PolicyDescriptor(
            policy_id="standalone-sam-policy",
            policy_version=STANDALONE_POLICY_CONTROLLER_VERSION,
            # Public game events retain only this opaque model identity. The
            # verified file digest stays private to the executor and its logs.
            artifact_id=metadata.model_id,
            artifact_sha256="none",
            observation_schema_version=metadata.observation_schema_version,
            action_schema_version=metadata.action_schema_version,
            hidden_state_schema_version=(
                STANDALONE_POLICY_STATE_SCHEMA_VERSION
            ),
            inference_profile=STANDALONE_POLICY_INFERENCE_PROFILE,
        )

    def decide(
        self,
        *,
        game_id: UUID | str,
        information: SearchInformationState,
        policy_input: PolicyInput,
    ) -> StandalonePolicyDecision:
        metadata = self.loaded.metadata
        if (
            metadata.model_schema_version != MODEL_SCHEMA_VERSION
            or metadata.observation_schema_version
            != OBSERVATION_SCHEMA_VERSION
            or metadata.action_schema_version != ACTION_SCHEMA_VERSION
            or metadata.representative_mask_schema_version
            != REPRESENTATIVE_MASK_SCHEMA_VERSION
        ):
            raise PolicyContractError(
                "standalone policy artifact contracts differ"
            )
        if not isinstance(information, SearchInformationState):
            raise PolicyContractError(
                "standalone policy requires a player information state"
            )
        expected_input = policy_input_from_information_state(information)
        if (
            not isinstance(policy_input, PolicyInput)
            or not _policy_inputs_equal(policy_input, expected_input)
        ):
            raise PolicyContractError(
                "standalone policy input differs from its information state"
            )
        groups = strategic_action_groups(
            information, destination_symmetry_enabled=True
        )
        projection = build_representative_action_projection(
            policy_input.legal_mask, groups
        )
        started = time.perf_counter()
        with torch.inference_mode():
            logits = self.model(policy_input.observation)
        representative = select_representative_action(
            logits, projection.mask
        )
        if not isinstance(representative, int):
            raise PolicyContractError(
                "single-state policy returned a batched selection"
            )
        request_seed = derive_standalone_policy_request_seed(
            game_id=game_id,
            information=information,
            artifact_sha256=self.artifact_sha256,
        )
        choice_seed = derive_strategic_destination_choice_seed(
            request_seed,
            INFERENCE_DESTINATION_SCOPE,
            information,
            representative,
            0,
        )
        concrete = select_concrete_action_index(
            information,
            representative,
            choice_seed,
            destination_symmetry_enabled=True,
        )
        latency = time.perf_counter() - started
        legal = tuple(
            index
            for index, allowed in enumerate(
                value for row in information.legal_mask for value in row
            )
            if allowed
        )
        if concrete not in legal:
            raise PolicyContractError(
                "standalone policy resolved an illegal concrete action"
            )
        return StandalonePolicyDecision(
            representative,
            concrete,
            len(groups),
            latency,
            information_state_fingerprint(information),
        )

    def invoke(self, request: PolicyTurnRequest) -> PolicyTurnResult:
        if request.policy != self.descriptor:
            raise PolicyContractError(
                "game opponent session does not match standalone policy"
            )
        if not isinstance(
            request.information_state, SearchInformationState
        ):
            raise PolicyContractError(
                "standalone policy requires a player information state"
            )
        information = request.information_state
        if (
            information.player is not request.player
            or information.round_number != request.round_number
            or information.turn_number != request.turn_number
        ):
            raise PolicyContractError(
                "standalone information does not match the requested turn"
            )
        if request.turn_kind is PolicyTurnKind.FORCED_RECURRENT_TRANSITION:
            legal = tuple(
                index
                for index, move in enumerate(request.action_table)
                if move is not None
            )
            if len(legal) != 1:
                raise PolicyContractError(
                    "forced standalone turn must have one legal action"
                )
            return PolicyTurnResult(None, None)

        decision = self.decide(
            game_id=request.game_id,
            information=information,
            policy_input=request.policy_input,
        )
        if (
            decision.concrete_action_index >= len(request.action_table)
            or request.action_table[
                decision.concrete_action_index
            ]
            is None
        ):
            raise PolicyContractError(
                "standalone policy selected outside service legality"
            )
        LOGGER.info(
            (
                "standalone-sam turn controller=%s artifact_sha256=%s "
                "game_id=%s round=%d turn=%d representative_action=%d "
                "concrete_action=%d strategic_groups=%d "
                "decision_latency_seconds=%.6f"
            ),
            self.descriptor.policy_id,
            self.artifact_sha256,
            request.game_id,
            request.round_number,
            request.turn_number,
            decision.representative_action_index,
            decision.concrete_action_index,
            decision.strategic_group_count,
            decision.latency_seconds,
        )
        return PolicyTurnResult(
            decision.concrete_action_index,
            None,
        )


__all__ = (
    "STANDALONE_POLICY_CONTROLLER_VERSION",
    "STANDALONE_POLICY_INFERENCE_PROFILE",
    "STANDALONE_POLICY_REQUEST_SEED_NAMESPACE",
    "STANDALONE_POLICY_STATE_SCHEMA_VERSION",
    "StandalonePolicyDecision",
    "StandaloneSamPolicyExecutor",
    "derive_standalone_policy_request_seed",
)

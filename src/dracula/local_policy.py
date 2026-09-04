"""Validated CPU inference for one manually selected local policy archive."""

from __future__ import annotations

import hashlib
import struct
from collections.abc import Mapping
from pathlib import Path

import torch
from torch import Tensor

from dracula.api.policy import (
    PolicyDescriptor,
    PolicyTurnRequest,
    PolicyTurnResult,
)
from dracula.bridge import PolicyTurnKind
from dracula.legacy_policy_contract import (
    ACTION_MAP_VERSION,
    OBSERVATION_SCHEMA_VERSION,
    POLICY_ARCHITECTURE_VERSION,
)
from dracula.models import HIDDEN_SIZE, Policy
from dracula.policy_adapter import (
    HIDDEN_STATE_SCHEMA_VERSION,
    POLICY_ARCHIVE_FORMAT_VERSION,
    POLICY_INFERENCE_CONTRACT_VERSION,
    ActionSelectionProfile,
    PolicyAdapter,
    PolicyArtifactMetadata,
    PolicyContractError,
    PolicyInferenceRequest,
    PolicyInferenceResponse,
    resolve_inference_profile,
    select_masked_action,
    validate_hidden_bytes,
)
class PolicyArtifactError(PolicyContractError):
    """A selected policy archive cannot be used for inference."""


def hidden_bytes_to_tensor(value: bytes) -> Tensor:
    validate_hidden_bytes(value)
    return torch.tensor(struct.unpack("<128f", value), dtype=torch.float32)


def hidden_tensor_to_bytes(value: Tensor) -> bytes:
    if (
        not isinstance(value, Tensor)
        or value.device.type != "cpu"
        or value.dtype is not torch.float32
        or value.shape != (HIDDEN_SIZE,)
        or not torch.isfinite(value).all()
    ):
        raise PolicyContractError("hidden tensor must be finite CPU float32[128]")
    return struct.pack("<128f", *value.detach().tolist())


def _state_dict_fingerprint(state_dict: Mapping[str, Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state_dict.items()):
        cpu_tensor = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(cpu_tensor.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(",".join(str(size) for size in cpu_tensor.shape).encode("ascii"))
        digest.update(b"\0")
        digest.update(cpu_tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def serving_contract() -> dict[str, str | int]:
    return {
        "inference_contract_version": POLICY_INFERENCE_CONTRACT_VERSION,
        "policy_architecture_version": POLICY_ARCHITECTURE_VERSION,
        "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
        "action_schema_version": ACTION_MAP_VERSION,
        "hidden_state_schema_version": HIDDEN_STATE_SCHEMA_VERSION,
        "parameter_count": Policy.parameter_count,
        "initial_hidden_state": "little-endian-float32[128]-all-zero",
    }


class LocalTorchPolicyAdapter(PolicyAdapter):
    """Load one immutable archive and expose stateless policy inference."""

    def __init__(self, archive_path: str | Path) -> None:
        self.archive_path = Path(archive_path).expanduser().resolve()
        try:
            archive_bytes = self.archive_path.read_bytes()
            payload = torch.load(
                self.archive_path, map_location="cpu", weights_only=True
            )
        except (OSError, RuntimeError, ValueError) as error:
            raise PolicyArtifactError("policy archive could not be loaded") from error
        if not isinstance(payload, dict):
            raise PolicyArtifactError("policy archive must contain an object payload")
        if payload.get("format_version") != POLICY_ARCHIVE_FORMAT_VERSION:
            raise PolicyArtifactError("policy archive format version does not match")

        contracts = payload.get("contracts")
        expected_contracts = {
            "policy": POLICY_ARCHITECTURE_VERSION,
            "observation": OBSERVATION_SCHEMA_VERSION,
            "action_map": ACTION_MAP_VERSION,
        }
        if not isinstance(contracts, dict) or any(
            contracts.get(name) != version
            for name, version in expected_contracts.items()
        ):
            raise PolicyArtifactError("policy archive tensor contracts do not match")
        resolved_manifest = payload.get("resolved_manifest")
        if (
            not isinstance(resolved_manifest, dict)
            or resolved_manifest.get("versions") != contracts
        ):
            raise PolicyArtifactError("policy archive manifest and contracts disagree")

        declared_serving = payload.get("serving_contract")
        if declared_serving is not None and declared_serving != serving_contract():
            raise PolicyArtifactError("policy archive serving contract does not match")

        policy_value = payload.get("policy")
        if not isinstance(policy_value, dict):
            raise PolicyArtifactError("policy archive is missing policy metadata")
        policy_id = policy_value.get("policy_id")
        policy_version = policy_value.get("version")
        expected_fingerprint = policy_value.get("parameter_fingerprint")
        state_dict = policy_value.get("state_dict")
        if not all(
            isinstance(value, str) and value
            for value in (policy_id, policy_version, expected_fingerprint)
        ):
            raise PolicyArtifactError("policy identity metadata is invalid")
        if not isinstance(state_dict, dict) or not all(
            isinstance(name, str) and isinstance(tensor, Tensor)
            for name, tensor in state_dict.items()
        ):
            raise PolicyArtifactError("policy archive does not contain a state dict")

        model = Policy(seed=0).cpu()
        expected_state = model.state_dict()
        if set(state_dict) != set(expected_state):
            raise PolicyArtifactError("policy state-dict keys do not match the architecture")
        for name, expected in expected_state.items():
            actual = state_dict[name]
            if (
                actual.device.type != "cpu"
                or actual.dtype is not expected.dtype
                or actual.shape != expected.shape
            ):
                raise PolicyArtifactError(
                    f"policy state tensor does not match the architecture: {name}"
                )
            if not torch.isfinite(actual).all():
                raise PolicyArtifactError("policy archive contains non-finite weights")
        try:
            model.load_state_dict(state_dict, strict=True)
        except RuntimeError as error:
            raise PolicyArtifactError("policy state dict could not be loaded") from error
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        if parameter_count != Policy.parameter_count:
            raise PolicyArtifactError("policy parameter count does not match")
        fingerprint = _state_dict_fingerprint(model.state_dict())
        if fingerprint != expected_fingerprint:
            raise PolicyArtifactError("policy parameter fingerprint does not match")

        artifact_sha256 = hashlib.sha256(archive_bytes).hexdigest()
        self._metadata = PolicyArtifactMetadata(
            artifact_id=f"sha256:{artifact_sha256}",
            artifact_sha256=artifact_sha256,
            policy_id=policy_id,
            policy_version=policy_version,
            architecture_version=POLICY_ARCHITECTURE_VERSION,
            observation_schema_version=OBSERVATION_SCHEMA_VERSION,
            action_schema_version=ACTION_MAP_VERSION,
            hidden_state_schema_version=HIDDEN_STATE_SCHEMA_VERSION,
            parameter_count=parameter_count,
        )
        self._policy = model.eval().requires_grad_(False)

    @property
    def metadata(self) -> PolicyArtifactMetadata:
        return self._metadata

    def invoke(self, request: PolicyInferenceRequest) -> PolicyInferenceResponse:
        if not isinstance(request, PolicyInferenceRequest):
            raise PolicyContractError("adapter request must be PolicyInferenceRequest")
        if request.artifact_id != self.metadata.artifact_id:
            raise PolicyContractError("inference request names another policy artifact")
        observation = torch.tensor(request.observation, dtype=torch.bool)
        legal_mask = torch.tensor(request.legal_mask, dtype=torch.bool)
        hidden_state = hidden_bytes_to_tensor(request.hidden_state)
        with torch.inference_mode():
            raw_logits, next_hidden = self._policy(
                observation, legal_mask, hidden_state
            )
        if (
            raw_logits.device.type != "cpu"
            or raw_logits.dtype is not torch.float32
            or raw_logits.shape != (4, 8)
            or not torch.isfinite(raw_logits).all()
        ):
            raise PolicyContractError("policy returned invalid raw logits")
        return PolicyInferenceResponse(
            contract_version=POLICY_INFERENCE_CONTRACT_VERSION,
            artifact_id=self.metadata.artifact_id,
            raw_logits=tuple(
                tuple(float(value) for value in row) for row in raw_logits.tolist()
            ),
            next_hidden_state=hidden_tensor_to_bytes(next_hidden),
        )


class InlinePolicyExecutor:
    """Apply local adapter inference behind the service's executor boundary."""

    def __init__(
        self,
        adapter: PolicyAdapter,
        profile: ActionSelectionProfile,
    ) -> None:
        self.adapter = adapter
        self.profile = profile

    @classmethod
    def from_archive(
        cls, archive_path: str | Path, profile_version: str
    ) -> InlinePolicyExecutor:
        return cls(
            LocalTorchPolicyAdapter(archive_path),
            resolve_inference_profile(profile_version),
        )

    @property
    def descriptor(self) -> PolicyDescriptor:
        metadata = self.adapter.metadata
        return PolicyDescriptor(
            policy_id=metadata.policy_id,
            policy_version=metadata.policy_version,
            artifact_id=metadata.artifact_id,
            artifact_sha256=metadata.artifact_sha256,
            observation_schema_version=metadata.observation_schema_version,
            action_schema_version=metadata.action_schema_version,
            hidden_state_schema_version=metadata.hidden_state_schema_version,
            inference_profile=self.profile.version,
        )

    def invoke(self, request: PolicyTurnRequest) -> PolicyTurnResult:
        metadata = self.adapter.metadata
        expected_descriptor = self.descriptor
        if request.policy != expected_descriptor:
            raise PolicyContractError("game policy session does not match the adapter")
        if type(request.turn_number) is not int or not 1 <= request.turn_number <= 8:
            raise PolicyContractError("policy turn number must be between one and eight")
        policy_input = request.policy_input
        observation = tuple(
            bool(value) for value in policy_input.observation.detach().cpu().tolist()
        )
        legal_mask = tuple(
            tuple(bool(value) for value in row)
            for row in policy_input.legal_mask.detach().cpu().tolist()
        )
        adapter_request = PolicyInferenceRequest(
            contract_version=POLICY_INFERENCE_CONTRACT_VERSION,
            artifact_id=metadata.artifact_id,
            observation=observation,
            legal_mask=legal_mask,
            hidden_state=request.hidden_state,
        )
        response = self.adapter.invoke(adapter_request)
        if (
            response.contract_version != POLICY_INFERENCE_CONTRACT_VERSION
            or response.artifact_id != metadata.artifact_id
        ):
            raise PolicyContractError("policy adapter returned a mismatched response")
        action_index = None
        if request.turn_kind is not PolicyTurnKind.FORCED_RECURRENT_TRANSITION:
            action_index = select_masked_action(
                response.raw_logits,
                legal_mask,
                self.profile,
                game_id=str(request.game_id),
                round_number=request.round_number,
                turn_number=request.turn_number,
                artifact_id=metadata.artifact_id,
            )
        return PolicyTurnResult(action_index, response.next_hidden_state)

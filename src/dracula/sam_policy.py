"""Standalone policy classifier for symmetry-aware Sam-32 imitation."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
import torch
from torch import Tensor, nn

from dracula.action_contract import (
    ACTION_COUNT,
    COFFIN_POSITION_COUNT,
    DESTINATION_SYMMETRY_SCHEMA_VERSION,
    HAND_SLOT_COUNT,
    INFERENCE_DESTINATION_SCOPE,
    POLICY_GRID_INDICES,
    POLICY_POSITION_COUNT,
    REPRESENTATIVE_MASK_SCHEMA_VERSION,
    RepresentativeActionGroup,
    RepresentativeActionProjection,
    SamPolicyContractError,
    apply_representative_mask,
    build_representative_action_projection,
    representative_policy_probabilities,
    resolve_representative_action,
    select_representative_action,
)
from dracula.cards import CARD_COUNT, CARD_IDS
from dracula.randomness import derive_seed

MODEL_SCHEMA_VERSION = "dracula-sam-policy-v1"
ARTIFACT_SCHEMA_VERSION = "dracula-sam-policy-artifact-v1"
OBSERVATION_SCHEMA_VERSION = "dracula-observation-v1"
ACTION_SCHEMA_VERSION = "dracula-action-map-v1"
INITIALIZATION_SCHEMA_VERSION = "dracula-sam-policy-initialization-v1"
TRAINING_SNAPSHOT_SCHEMA_VERSION = "dracula-sam-policy-snapshot-v1"
OPTIMIZER_COMPATIBILITY_VERSION = "dracula-sam-policy-optimizer-v1"
MODEL_INITIALIZATION_NAMESPACE = "dracula-sam-policy-initialization-v1"
OBSERVATION_SIZE = 875
PARAMETER_COUNT = 738_569

HAND_FEATURES = HAND_SLOT_COUNT * CARD_COUNT
COFFIN_FEATURES = COFFIN_POSITION_COUNT * CARD_COUNT
STATUS_FEATURES = 3 * CARD_COUNT
CONTEXT_FEATURES = 11
HAND_START = 0
COFFIN_START = HAND_START + HAND_FEATURES
STATUS_START = COFFIN_START + COFFIN_FEATURES
CONTEXT_START = STATUS_START + STATUS_FEATURES

_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_SOURCE_REVISION = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
@dataclass(frozen=True, slots=True)
class SamPolicyArtifactMetadata:
    model_schema_version: str
    artifact_schema_version: str
    observation_schema_version: str
    action_schema_version: str
    representative_mask_schema_version: str
    destination_symmetry_schema_version: str
    initialization_schema_version: str
    training_snapshot_schema_version: str
    optimizer_compatibility_version: str
    parameter_count: int
    card_ids: tuple[str, ...]
    policy_grid_indices: tuple[int, ...]
    run_root_seed: str
    model_id: str
    initialization_ordinal: int
    initialization_seed_digest: str
    source_revision: str
    source_tree_digest: str
    training_config_digest: str
    corpus_snapshot_digest: str
    state_dict_digest: str


@dataclass(frozen=True, slots=True)
class LoadedSamPolicyArtifact:
    model: SamPolicyModel
    metadata: SamPolicyArtifactMetadata
    training_configuration: dict[str, object]


@dataclass(frozen=True, slots=True)
class _StructuredObservation:
    hand: Tensor
    coffin: Tensor
    statuses: Tensor
    context: Tensor


def derive_policy_initialization(
    run_root_seed: str,
    model_id: str,
    initialization_ordinal: int,
) -> tuple[bytes, int]:
    if not isinstance(run_root_seed, str) or not run_root_seed:
        raise SamPolicyContractError(
            "run root seed must be a nonempty string"
        )
    if not isinstance(model_id, str) or not model_id:
        raise SamPolicyContractError(
            "model ID must be a nonempty string"
        )
    if (
        type(initialization_ordinal) is not int
        or initialization_ordinal < 0
    ):
        raise SamPolicyContractError(
            "initialization ordinal must be non-negative"
        )
    try:
        digest = derive_seed(
            MODEL_INITIALIZATION_NAMESPACE,
            run_root_seed,
            model_id,
            str(initialization_ordinal),
        )
    except (TypeError, ValueError) as error:
        raise SamPolicyContractError(
            "initialization identity is invalid"
        ) from error
    return digest, int.from_bytes(digest[:8], "big", signed=False)


def _validate_observation(observation: Tensor) -> bool:
    if (
        not isinstance(observation, Tensor)
        or observation.dtype is not torch.bool
    ):
        raise SamPolicyContractError(
            "observation must have Boolean dtype"
        )
    if observation.ndim == 1:
        if observation.shape != (OBSERVATION_SIZE,):
            raise SamPolicyContractError(
                "single observation must have shape [875]"
            )
        return True
    if observation.ndim == 2:
        if (
            observation.shape[0] < 1
            or observation.shape[1] != OBSERVATION_SIZE
        ):
            raise SamPolicyContractError(
                "batched observation must have shape [B,875]"
            )
        return False
    raise SamPolicyContractError(
        "observation must be one state or one batch"
    )


def _split_observation(observation: Tensor) -> _StructuredObservation:
    batch_size = observation.shape[0]
    return _StructuredObservation(
        hand=observation[:, HAND_START:COFFIN_START].reshape(
            batch_size, HAND_SLOT_COUNT, CARD_COUNT
        ),
        coffin=observation[:, COFFIN_START:STATUS_START].reshape(
            batch_size, COFFIN_POSITION_COUNT, CARD_COUNT
        ),
        statuses=observation[:, STATUS_START:CONTEXT_START],
        context=observation[:, CONTEXT_START:],
    )


def _initialize(module: nn.Module, seed: int) -> None:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    with torch.no_grad():
        for child in module.modules():
            if isinstance(child, nn.Linear):
                nn.init.xavier_uniform_(
                    child.weight,
                    gain=1.0,
                    generator=generator,
                )
                nn.init.zeros_(child.bias)
            elif isinstance(child, nn.Embedding):
                nn.init.normal_(
                    child.weight,
                    mean=0.0,
                    std=1.0 / math.sqrt(child.embedding_dim),
                    generator=generator,
                )
            elif isinstance(child, nn.LayerNorm):
                nn.init.ones_(child.weight)
                nn.init.zeros_(child.bias)


class _SamPolicyEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.card_embedding = nn.Embedding(CARD_COUNT, 32)
        self.hand_slot_embedding = nn.Embedding(HAND_SLOT_COUNT, 8)
        self.coffin_position_embedding = nn.Embedding(
            COFFIN_POSITION_COUNT, 8
        )
        self.hand_encoder = nn.Linear(41, 64)
        self.coffin_encoder = nn.Linear(41, 64)
        self.status_encoder = nn.Linear(STATUS_FEATURES, 96)
        self.context_encoder = nn.Linear(CONTEXT_FEATURES, 16)
        self.shared_input = nn.Linear(944, 512)
        self.shared_normalization = nn.LayerNorm(512, eps=1e-5)
        self.shared_output = nn.Linear(512, 256)
        self.activation = nn.GELU(approximate="none")

    def forward(
        self, observation: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        structured = _split_observation(observation)
        dtype = self.card_embedding.weight.dtype
        batch_size = observation.shape[0]

        hand_cards = (
            structured.hand.to(dtype) @ self.card_embedding.weight
        )
        hand_slots = self.hand_slot_embedding.weight.unsqueeze(0).expand(
            batch_size, -1, -1
        )
        hand_occupied = structured.hand.any(
            dim=-1, keepdim=True
        ).to(dtype)
        hand_features = self.activation(
            self.hand_encoder(
                torch.cat(
                    (hand_cards, hand_slots, hand_occupied),
                    dim=-1,
                )
            )
        )

        coffin_cards = (
            structured.coffin.to(dtype) @ self.card_embedding.weight
        )
        coffin_positions = (
            self.coffin_position_embedding.weight.unsqueeze(0).expand(
                batch_size, -1, -1
            )
        )
        coffin_occupied = structured.coffin.any(
            dim=-1, keepdim=True
        ).to(dtype)
        coffin_features = self.activation(
            self.coffin_encoder(
                torch.cat(
                    (
                        coffin_cards,
                        coffin_positions,
                        coffin_occupied,
                    ),
                    dim=-1,
                )
            )
        )

        status_features = self.activation(
            self.status_encoder(structured.statuses.to(dtype))
        )
        context_features = self.activation(
            self.context_encoder(structured.context.to(dtype))
        )
        fused = torch.cat(
            (
                hand_features.flatten(start_dim=1),
                coffin_features.flatten(start_dim=1),
                status_features,
                context_features,
            ),
            dim=-1,
        )
        shared = self.activation(self.shared_input(fused))
        shared = self.shared_normalization(shared)
        shared = self.activation(self.shared_output(shared))
        return shared, hand_features, coffin_features


class SamPolicyModel(nn.Module):
    """Structured player-visible classifier with 32 raw action logits."""

    parameter_count = PARAMETER_COUNT

    def __init__(
        self,
        *,
        run_root_seed: str,
        model_id: str,
        initialization_ordinal: int,
    ) -> None:
        super().__init__()
        digest, seed = derive_policy_initialization(
            run_root_seed,
            model_id,
            initialization_ordinal,
        )
        global_rng_state = torch.random.get_rng_state()
        try:
            self.encoder = _SamPolicyEncoder()
            self.pair_hidden = nn.Linear(384, 256)
            self.pair_normalization = nn.LayerNorm(256, eps=1e-5)
            self.pair_output = nn.Linear(256, 1)
            self.activation = nn.GELU(approximate="none")
            self.register_buffer(
                "policy_grid_indices",
                torch.tensor(POLICY_GRID_INDICES, dtype=torch.long),
                persistent=False,
            )
            self.float()
            _initialize(self, seed)
        finally:
            torch.random.set_rng_state(global_rng_state)
        self.run_root_seed = run_root_seed
        self.model_id = model_id
        self.initialization_ordinal = initialization_ordinal
        self.initialization_seed_digest = digest.hex()
        actual_count = sum(
            parameter.numel() for parameter in self.parameters()
        )
        if actual_count != PARAMETER_COUNT:
            raise RuntimeError(
                f"Sam policy parameter count is {actual_count}, "
                f"expected {PARAMETER_COUNT}"
            )

    def forward(self, observation: Tensor) -> Tensor:
        single = _validate_observation(observation)
        first_parameter = next(self.parameters())
        if first_parameter.dtype is not torch.float32:
            raise SamPolicyContractError(
                "model parameters must have float32 dtype"
            )
        if observation.device != first_parameter.device:
            raise SamPolicyContractError(
                "model and observation must share a device"
            )
        batch = observation.unsqueeze(0) if single else observation
        shared, hand_features, coffin_features = self.encoder(batch)
        destinations = coffin_features.index_select(
            1, self.policy_grid_indices
        )
        pair_features = torch.cat(
            (
                hand_features.unsqueeze(2).expand(
                    -1, -1, POLICY_POSITION_COUNT, -1
                ),
                destinations.unsqueeze(1).expand(
                    -1, HAND_SLOT_COUNT, -1, -1
                ),
                shared[:, None, None, :].expand(
                    -1,
                    HAND_SLOT_COUNT,
                    POLICY_POSITION_COUNT,
                    -1,
                ),
            ),
            dim=-1,
        )
        pair_features = self.activation(
            self.pair_hidden(pair_features)
        )
        pair_features = self.pair_normalization(pair_features)
        logits = self.pair_output(pair_features).squeeze(-1)
        return logits.squeeze(0) if single else logits


def _state_dict_digest(state_dict: Mapping[str, Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state_dict.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(",".join(map(str, value.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _canonical_configuration(
    configuration: Mapping[str, object],
) -> tuple[dict[str, object], str]:
    if not isinstance(configuration, Mapping) or any(
        not isinstance(key, str) for key in configuration
    ):
        raise SamPolicyContractError(
            "training configuration must be a string-keyed mapping"
        )
    try:
        encoded = json.dumps(
            dict(configuration),
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        canonical = json.loads(encoded)
    except (TypeError, ValueError) as error:
        raise SamPolicyContractError(
            "training configuration must be canonical JSON"
        ) from error
    if not isinstance(canonical, dict):
        raise SamPolicyContractError(
            "training configuration must be an object"
        )
    return canonical, hashlib.sha256(encoded).hexdigest()


def _require_digest(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or _DIGEST.fullmatch(value) is None
    ):
        raise SamPolicyContractError(
            f"{label} must be a lowercase SHA-256 digest"
        )
    return value


def build_sam_policy_artifact(
    model: SamPolicyModel,
    *,
    source_revision: str,
    source_tree_digest: str,
    training_configuration: Mapping[str, object],
    corpus_snapshot_digest: str,
) -> dict[str, object]:
    if not isinstance(model, SamPolicyModel):
        raise SamPolicyContractError(
            "artifact requires a SamPolicyModel"
        )
    if (
        not isinstance(source_revision, str)
        or _SOURCE_REVISION.fullmatch(source_revision) is None
    ):
        raise SamPolicyContractError(
            "source revision must be a lowercase Git object ID"
        )
    canonical_configuration, config_digest = (
        _canonical_configuration(training_configuration)
    )
    state_dict = {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }
    if any(
        tensor.dtype is not torch.float32
        or not torch.isfinite(tensor).all()
        for tensor in state_dict.values()
    ):
        raise SamPolicyContractError(
            "artifact model parameters must be finite float32"
        )
    metadata = SamPolicyArtifactMetadata(
        model_schema_version=MODEL_SCHEMA_VERSION,
        artifact_schema_version=ARTIFACT_SCHEMA_VERSION,
        observation_schema_version=OBSERVATION_SCHEMA_VERSION,
        action_schema_version=ACTION_SCHEMA_VERSION,
        representative_mask_schema_version=(
            REPRESENTATIVE_MASK_SCHEMA_VERSION
        ),
        destination_symmetry_schema_version=(
            DESTINATION_SYMMETRY_SCHEMA_VERSION
        ),
        initialization_schema_version=(
            INITIALIZATION_SCHEMA_VERSION
        ),
        training_snapshot_schema_version=(
            TRAINING_SNAPSHOT_SCHEMA_VERSION
        ),
        optimizer_compatibility_version=(
            OPTIMIZER_COMPATIBILITY_VERSION
        ),
        parameter_count=PARAMETER_COUNT,
        card_ids=CARD_IDS,
        policy_grid_indices=POLICY_GRID_INDICES,
        run_root_seed=model.run_root_seed,
        model_id=model.model_id,
        initialization_ordinal=model.initialization_ordinal,
        initialization_seed_digest=(
            model.initialization_seed_digest
        ),
        source_revision=source_revision,
        source_tree_digest=_require_digest(
            source_tree_digest, "source tree digest"
        ),
        training_config_digest=config_digest,
        corpus_snapshot_digest=_require_digest(
            corpus_snapshot_digest, "corpus snapshot digest"
        ),
        state_dict_digest=_state_dict_digest(state_dict),
    )
    return {
        "format_version": ARTIFACT_SCHEMA_VERSION,
        "metadata": asdict(metadata),
        "training_configuration": canonical_configuration,
        "state_dict": state_dict,
    }


def save_sam_policy_artifact(
    path: str | Path,
    model: SamPolicyModel,
    *,
    source_revision: str,
    source_tree_digest: str,
    training_configuration: Mapping[str, object],
    corpus_snapshot_digest: str,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = build_sam_policy_artifact(
        model,
        source_revision=source_revision,
        source_tree_digest=source_tree_digest,
        training_configuration=training_configuration,
        corpus_snapshot_digest=corpus_snapshot_digest,
    )
    temporary = destination.with_name(
        f".{destination.name}.tmp-{os.getpid()}"
    )
    try:
        torch.save(payload, temporary)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def load_sam_policy_artifact(
    path: str | Path,
) -> LoadedSamPolicyArtifact:
    try:
        payload = torch.load(
            Path(path), map_location="cpu", weights_only=True
        )
    except (OSError, RuntimeError, ValueError) as error:
        raise SamPolicyContractError(
            "Sam policy artifact could not be loaded"
        ) from error
    if not isinstance(payload, dict) or set(payload) != {
        "format_version",
        "metadata",
        "training_configuration",
        "state_dict",
    }:
        raise SamPolicyContractError(
            "Sam policy artifact payload is invalid"
        )
    if payload["format_version"] != ARTIFACT_SCHEMA_VERSION:
        raise SamPolicyContractError(
            "Sam policy artifact format is incompatible"
        )
    metadata_value = payload["metadata"]
    expected_fields = set(
        SamPolicyArtifactMetadata.__dataclass_fields__
    )
    if (
        not isinstance(metadata_value, dict)
        or set(metadata_value) != expected_fields
    ):
        raise SamPolicyContractError(
            "Sam policy artifact metadata is invalid"
        )
    try:
        metadata = SamPolicyArtifactMetadata(**metadata_value)
    except TypeError as error:
        raise SamPolicyContractError(
            "Sam policy artifact metadata is invalid"
        ) from error

    expected_contract = {
        "model_schema_version": MODEL_SCHEMA_VERSION,
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
        "action_schema_version": ACTION_SCHEMA_VERSION,
        "representative_mask_schema_version": (
            REPRESENTATIVE_MASK_SCHEMA_VERSION
        ),
        "destination_symmetry_schema_version": (
            DESTINATION_SYMMETRY_SCHEMA_VERSION
        ),
        "initialization_schema_version": (
            INITIALIZATION_SCHEMA_VERSION
        ),
        "training_snapshot_schema_version": (
            TRAINING_SNAPSHOT_SCHEMA_VERSION
        ),
        "optimizer_compatibility_version": (
            OPTIMIZER_COMPATIBILITY_VERSION
        ),
        "parameter_count": PARAMETER_COUNT,
        "card_ids": CARD_IDS,
        "policy_grid_indices": POLICY_GRID_INDICES,
    }
    for field, expected in expected_contract.items():
        if getattr(metadata, field) != expected:
            raise SamPolicyContractError(
                f"artifact {field} is incompatible"
            )
    for field in (
        "initialization_seed_digest",
        "source_tree_digest",
        "training_config_digest",
        "corpus_snapshot_digest",
        "state_dict_digest",
    ):
        _require_digest(
            getattr(metadata, field), field.replace("_", " ")
        )
    if (
        not isinstance(metadata.source_revision, str)
        or _SOURCE_REVISION.fullmatch(metadata.source_revision) is None
    ):
        raise SamPolicyContractError(
            "artifact source revision is invalid"
        )
    expected_seed, _ = derive_policy_initialization(
        metadata.run_root_seed,
        metadata.model_id,
        metadata.initialization_ordinal,
    )
    if metadata.initialization_seed_digest != expected_seed.hex():
        raise SamPolicyContractError(
            "artifact initialization identity is inconsistent"
        )
    configuration, config_digest = _canonical_configuration(
        payload["training_configuration"]  # type: ignore[arg-type]
    )
    if config_digest != metadata.training_config_digest:
        raise SamPolicyContractError(
            "artifact training configuration digest differs"
        )

    model = SamPolicyModel(
        run_root_seed=metadata.run_root_seed,
        model_id=metadata.model_id,
        initialization_ordinal=metadata.initialization_ordinal,
    )
    state_dict = payload["state_dict"]
    expected_state = model.state_dict()
    if (
        not isinstance(state_dict, dict)
        or set(state_dict) != set(expected_state)
    ):
        raise SamPolicyContractError(
            "artifact state-dict keys are incompatible"
        )
    for name, expected in expected_state.items():
        value = state_dict[name]
        if (
            not isinstance(value, Tensor)
            or value.device.type != "cpu"
            or value.dtype is not expected.dtype
            or value.shape != expected.shape
        ):
            raise SamPolicyContractError(
                f"artifact tensor is incompatible: {name}"
            )
        if not torch.isfinite(value).all():
            raise SamPolicyContractError(
                f"artifact tensor is non-finite: {name}"
            )
    if _state_dict_digest(state_dict) != metadata.state_dict_digest:
        raise SamPolicyContractError(
            "artifact state-dict digest differs"
        )
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as error:
        raise SamPolicyContractError(
            "artifact state dict could not be loaded"
        ) from error
    return LoadedSamPolicyArtifact(
        model, metadata, configuration
    )


__all__ = (
    "ACTION_SCHEMA_VERSION",
    "ACTION_COUNT",
    "ARTIFACT_SCHEMA_VERSION",
    "DESTINATION_SYMMETRY_SCHEMA_VERSION",
    "INFERENCE_DESTINATION_SCOPE",
    "INITIALIZATION_SCHEMA_VERSION",
    "MODEL_INITIALIZATION_NAMESPACE",
    "MODEL_SCHEMA_VERSION",
    "OBSERVATION_SCHEMA_VERSION",
    "OBSERVATION_SIZE",
    "OPTIMIZER_COMPATIBILITY_VERSION",
    "PARAMETER_COUNT",
    "POLICY_GRID_INDICES",
    "REPRESENTATIVE_MASK_SCHEMA_VERSION",
    "TRAINING_SNAPSHOT_SCHEMA_VERSION",
    "LoadedSamPolicyArtifact",
    "RepresentativeActionGroup",
    "RepresentativeActionProjection",
    "SamPolicyArtifactMetadata",
    "SamPolicyContractError",
    "SamPolicyModel",
    "apply_representative_mask",
    "build_representative_action_projection",
    "build_sam_policy_artifact",
    "derive_policy_initialization",
    "load_sam_policy_artifact",
    "representative_policy_probabilities",
    "resolve_representative_action",
    "save_sam_policy_artifact",
    "select_representative_action",
)

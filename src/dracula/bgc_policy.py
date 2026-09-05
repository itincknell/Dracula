"""Create and verify standalone BGC policy artifacts.

Artifacts bind model weights to their tensor, action, symmetry, initialization,
source, dataset, and training identities. Strict checks occur once at load time
before the model enters ordinary inference.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping

import torch
from torch import Tensor

from dracula.action_contract import REPRESENTATIVE_MASK_SCHEMA_VERSION
from dracula.bgc_policy_model import (
    ACTION_SCHEMA_VERSION,
    INITIALIZATION_SCHEMA_VERSION,
    MODEL_SCHEMA_VERSION,
    OBSERVATION_SCHEMA_VERSION,
    PARAMETER_COUNT,
    POLICY_GRID_INDICES,
    BGCPolicyModel,
    BGCPolicyModelError,
    derive_policy_initialization,
)
from dracula.cards import CARD_IDS
from dracula.search.symmetry import DESTINATION_SYMMETRY_SCHEMA_VERSION

# The artifact format identifies serialization. The versions stored inside it
# separately identify model compatibility and training provenance.
BGC_POLICY_ARTIFACT_SCHEMA_VERSION = "dracula-bgc-card-policy-artifact-v2"
BGC_POLICY_SNAPSHOT_SCHEMA_VERSION = "dracula-bgc-card-policy-snapshot-v2"
BGC_POLICY_OPTIMIZER_VERSION = "dracula-bgc-card-policy-optimizer-v2"
BGC_POLICY_LOSS_SCHEMA_VERSION = "dracula-bgc-visit-distribution-loss-v1"
# Release selection pins the immutable artifact by file digest; the artifact's
# training-time candidate status is not rewritten during deployment.
BGC_POLICY_CANDIDATE_STATUS = "unaccepted"


@dataclass(frozen=True, slots=True)
class BGCPolicyArtifactMetadata:
    """Complete provenance and compatibility identity for one policy artifact."""

    model_schema_version: str
    artifact_schema_version: str
    observation_schema_version: str
    action_schema_version: str
    representative_mask_schema_version: str
    destination_symmetry_schema_version: str
    initialization_schema_version: str
    training_snapshot_schema_version: str
    optimizer_compatibility_version: str
    loss_schema_version: str
    candidate_status: str
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
    dataset_digest: str
    state_dict_digest: str


@dataclass(frozen=True, slots=True)
class LoadedBGCPolicyArtifact:
    """Verified model, metadata, canonical configuration, and file identity."""

    model: BGCPolicyModel
    metadata: BGCPolicyArtifactMetadata
    training_configuration: dict[str, object]
    artifact_digest: str


def state_dict_digest(state_dict: Mapping[str, Tensor]) -> str:
    """Hash sorted tensor names, types, shapes, and contiguous CPU bytes."""

    digest = hashlib.sha256()
    for name, tensor in sorted(state_dict.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(value.dtype).encode("ascii") + b"\0")
        digest.update(",".join(map(str, value.shape)).encode("ascii") + b"\0")
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _canonical_configuration(
    configuration: object,
) -> tuple[dict[str, object], str]:
    """Round-trip a JSON configuration and return its canonical digest."""

    if not isinstance(configuration, Mapping) or any(
        not isinstance(key, str) for key in configuration
    ):
        raise BGCPolicyModelError(
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
        raise BGCPolicyModelError(
            "training configuration must be canonical JSON"
        ) from error
    if not isinstance(canonical, dict):
        raise BGCPolicyModelError(
            "training configuration must be an object"
        )
    return canonical, hashlib.sha256(encoded).hexdigest()


def _require_digest(value: object, label: str) -> str:
    """Validate and return a lowercase SHA-256 digest."""

    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise BGCPolicyModelError(
            f"{label} must be a lowercase SHA-256 digest"
        )
    return value


def build_bgc_policy_artifact(
    model: BGCPolicyModel,
    *,
    source_revision: str,
    source_tree_digest: str,
    training_configuration: Mapping[str, object],
    corpus_snapshot_digest: str,
    dataset_digest: str,
) -> dict[str, object]:
    """Build a CPU-only policy payload after validating all bound identities."""

    if not isinstance(model, BGCPolicyModel):
        raise BGCPolicyModelError(
            "BGC policy artifact requires a BGCPolicyModel"
        )
    if (
        not isinstance(source_revision, str)
        or len(source_revision) not in (40, 64)
        or any(character not in "0123456789abcdef" for character in source_revision)
    ):
        raise BGCPolicyModelError(
            "source revision must be a lowercase Git object ID"
        )
    canonical_configuration, config_digest = _canonical_configuration(
        training_configuration
    )
    state_dict = {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }
    if any(
        tensor.dtype is not torch.float32
        or not bool(torch.isfinite(tensor).all().item())
        for tensor in state_dict.values()
    ):
        raise BGCPolicyModelError(
            "BGC policy parameters must be finite float32"
        )
    metadata = BGCPolicyArtifactMetadata(
        model_schema_version=MODEL_SCHEMA_VERSION,
        artifact_schema_version=BGC_POLICY_ARTIFACT_SCHEMA_VERSION,
        observation_schema_version=OBSERVATION_SCHEMA_VERSION,
        action_schema_version=ACTION_SCHEMA_VERSION,
        representative_mask_schema_version=REPRESENTATIVE_MASK_SCHEMA_VERSION,
        destination_symmetry_schema_version=DESTINATION_SYMMETRY_SCHEMA_VERSION,
        initialization_schema_version=INITIALIZATION_SCHEMA_VERSION,
        training_snapshot_schema_version=BGC_POLICY_SNAPSHOT_SCHEMA_VERSION,
        optimizer_compatibility_version=BGC_POLICY_OPTIMIZER_VERSION,
        loss_schema_version=BGC_POLICY_LOSS_SCHEMA_VERSION,
        candidate_status=BGC_POLICY_CANDIDATE_STATUS,
        parameter_count=PARAMETER_COUNT,
        card_ids=CARD_IDS,
        policy_grid_indices=POLICY_GRID_INDICES,
        run_root_seed=model.run_root_seed,
        model_id=model.model_id,
        initialization_ordinal=model.initialization_ordinal,
        initialization_seed_digest=model.initialization_seed_digest,
        source_revision=source_revision,
        source_tree_digest=_require_digest(
            source_tree_digest, "source tree digest"
        ),
        training_config_digest=config_digest,
        corpus_snapshot_digest=_require_digest(
            corpus_snapshot_digest, "corpus snapshot digest"
        ),
        dataset_digest=_require_digest(dataset_digest, "dataset digest"),
        state_dict_digest=state_dict_digest(state_dict),
    )
    return {
        "format_version": BGC_POLICY_ARTIFACT_SCHEMA_VERSION,
        "metadata": asdict(metadata),
        "training_configuration": canonical_configuration,
        "state_dict": state_dict,
    }


def save_bgc_policy_artifact(
    path: str | Path,
    model: BGCPolicyModel,
    *,
    source_revision: str,
    source_tree_digest: str,
    training_configuration: Mapping[str, object],
    corpus_snapshot_digest: str,
    dataset_digest: str,
) -> None:
    """Atomically save one validated standalone policy artifact."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = build_bgc_policy_artifact(
        model,
        source_revision=source_revision,
        source_tree_digest=source_tree_digest,
        training_configuration=training_configuration,
        corpus_snapshot_digest=corpus_snapshot_digest,
        dataset_digest=dataset_digest,
    )
    # Replacement occurs only after the temporary payload is flushed, so an
    # interruption cannot expose a partially written artifact at ``path``.
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        torch.save(payload, temporary)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _read_artifact_payload(path: str | Path) -> tuple[bytes, dict[str, object]]:
    """Deserialize the four-field artifact envelope without trusting its contents."""

    try:
        artifact_bytes = Path(path).read_bytes()
        payload = torch.load(
            io.BytesIO(artifact_bytes), map_location="cpu", weights_only=True
        )
    except (
        OSError,
        RuntimeError,
        ValueError,
        EOFError,
        IndexError,
        pickle.UnpicklingError,
    ) as error:
        raise BGCPolicyModelError(
            "BGC policy artifact could not be loaded"
        ) from error
    if not isinstance(payload, dict) or set(payload) != {
        "format_version",
        "metadata",
        "training_configuration",
        "state_dict",
    }:
        raise BGCPolicyModelError("BGC policy artifact payload is invalid")
    if payload["format_version"] != BGC_POLICY_ARTIFACT_SCHEMA_VERSION:
        raise BGCPolicyModelError("BGC policy artifact format is incompatible")
    return artifact_bytes, payload


def _artifact_metadata(payload: Mapping[str, object]) -> BGCPolicyArtifactMetadata:
    """Parse and verify the artifact's compatibility and provenance metadata."""

    raw_metadata = payload["metadata"]
    if (
        not isinstance(raw_metadata, dict)
        or set(raw_metadata) != set(BGCPolicyArtifactMetadata.__dataclass_fields__)
    ):
        raise BGCPolicyModelError("BGC policy artifact metadata is invalid")
    try:
        metadata = BGCPolicyArtifactMetadata(**raw_metadata)
    except TypeError as error:
        raise BGCPolicyModelError(
            "BGC policy artifact metadata is invalid"
        ) from error

    # Tensor shapes alone cannot detect changed input or action semantics.
    expected_metadata_values = {
        "model_schema_version": MODEL_SCHEMA_VERSION,
        "artifact_schema_version": BGC_POLICY_ARTIFACT_SCHEMA_VERSION,
        "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
        "action_schema_version": ACTION_SCHEMA_VERSION,
        "representative_mask_schema_version": REPRESENTATIVE_MASK_SCHEMA_VERSION,
        "destination_symmetry_schema_version": DESTINATION_SYMMETRY_SCHEMA_VERSION,
        "initialization_schema_version": INITIALIZATION_SCHEMA_VERSION,
        "training_snapshot_schema_version": BGC_POLICY_SNAPSHOT_SCHEMA_VERSION,
        "optimizer_compatibility_version": BGC_POLICY_OPTIMIZER_VERSION,
        "loss_schema_version": BGC_POLICY_LOSS_SCHEMA_VERSION,
        "candidate_status": BGC_POLICY_CANDIDATE_STATUS,
        "parameter_count": PARAMETER_COUNT,
        "card_ids": CARD_IDS,
        "policy_grid_indices": POLICY_GRID_INDICES,
    }
    for field, value in expected_metadata_values.items():
        if getattr(metadata, field) != value:
            raise BGCPolicyModelError(
                f"BGC policy artifact {field} is incompatible"
            )
    for field in (
        "initialization_seed_digest",
        "source_tree_digest",
        "training_config_digest",
        "corpus_snapshot_digest",
        "dataset_digest",
        "state_dict_digest",
    ):
        _require_digest(getattr(metadata, field), field.replace("_", " "))
    if (
        len(metadata.source_revision) not in (40, 64)
        or any(
            character not in "0123456789abcdef"
            for character in metadata.source_revision
        )
    ):
        raise BGCPolicyModelError("artifact source revision is invalid")
    expected_seed, _ = derive_policy_initialization(
        metadata.run_root_seed,
        metadata.model_id,
        metadata.initialization_ordinal,
    )
    if metadata.initialization_seed_digest != expected_seed.hex():
        raise BGCPolicyModelError(
            "artifact initialization identity is inconsistent"
        )
    return metadata


def _artifact_configuration(
    payload: Mapping[str, object],
    metadata: BGCPolicyArtifactMetadata,
) -> dict[str, object]:
    """Verify the recorded training configuration against its digest."""

    configuration, config_digest = _canonical_configuration(
        payload["training_configuration"]
    )
    if config_digest != metadata.training_config_digest:
        raise BGCPolicyModelError(
            "artifact training configuration digest differs"
        )
    return configuration


def _artifact_model(
    payload: Mapping[str, object],
    metadata: BGCPolicyArtifactMetadata,
) -> BGCPolicyModel:
    """Validate the serialized tensors and load them into the exact model shape."""

    model = BGCPolicyModel(
        run_root_seed=metadata.run_root_seed,
        model_id=metadata.model_id,
        initialization_ordinal=metadata.initialization_ordinal,
    )
    raw_state_dict = payload["state_dict"]
    expected_state = model.state_dict()
    if not isinstance(raw_state_dict, dict) or set(raw_state_dict) != set(
        expected_state
    ):
        raise BGCPolicyModelError("artifact state-dict keys are incompatible")
    state_dict: dict[str, Tensor] = {}
    for name, expected_tensor in expected_state.items():
        value = raw_state_dict[name]
        if (
            not isinstance(value, Tensor)
            or value.device.type != "cpu"
            or value.dtype is not expected_tensor.dtype
            or value.shape != expected_tensor.shape
            or not bool(torch.isfinite(value).all().item())
        ):
            raise BGCPolicyModelError(
                f"artifact tensor is incompatible: {name}"
            )
        state_dict[name] = value
    if state_dict_digest(state_dict) != metadata.state_dict_digest:
        raise BGCPolicyModelError("artifact state-dict digest differs")
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as error:
        raise BGCPolicyModelError(
            "artifact state dict could not be loaded"
        ) from error
    return model


def load_bgc_policy_artifact(path: str | Path) -> LoadedBGCPolicyArtifact:
    """Load one policy after verifying its envelope, metadata, and tensors."""

    artifact_bytes, payload = _read_artifact_payload(path)
    metadata = _artifact_metadata(payload)
    configuration = _artifact_configuration(payload, metadata)
    model = _artifact_model(payload, metadata)
    return LoadedBGCPolicyArtifact(
        model,
        metadata,
        configuration,
        hashlib.sha256(artifact_bytes).hexdigest(),
    )


__all__ = (
    "BGC_POLICY_ARTIFACT_SCHEMA_VERSION",
    "BGC_POLICY_CANDIDATE_STATUS",
    "BGC_POLICY_LOSS_SCHEMA_VERSION",
    "BGC_POLICY_OPTIMIZER_VERSION",
    "BGC_POLICY_SNAPSHOT_SCHEMA_VERSION",
    "BGCPolicyArtifactMetadata",
    "LoadedBGCPolicyArtifact",
    "build_bgc_policy_artifact",
    "load_bgc_policy_artifact",
    "save_bgc_policy_artifact",
    "state_dict_digest",
)

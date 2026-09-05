"""Save and load the standalone policy's minimal deployment artifact.

The artifact contains one format marker and the model tensors. Training
provenance belongs to run reports; deployment verifies the complete file digest
and validates tensor names, shapes, dtypes, and finite values once at load time.
"""

from __future__ import annotations

import hashlib
import io
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import torch
from torch import Tensor

from dracula.bgc_policy_model import BGCPolicyModel, BGCPolicyModelError

POLICY_ARTIFACT_FORMAT = "pi1-policy-v1"


@dataclass(frozen=True, slots=True)
class LoadedPolicyArtifact:
    """A validated inference model and the digest of its complete file."""

    model: BGCPolicyModel
    artifact_digest: str


def _model_tensors(model: BGCPolicyModel) -> dict[str, Tensor]:
    tensors = {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }
    if any(
        tensor.dtype is not torch.float32
        or not bool(torch.isfinite(tensor).all().item())
        for tensor in tensors.values()
    ):
        raise BGCPolicyModelError("policy parameters must be finite float32 tensors")
    return tensors


def save_policy_artifact(path: str | Path, model: BGCPolicyModel) -> None:
    """Atomically write one minimal CPU-only policy artifact."""

    if not isinstance(model, BGCPolicyModel):
        raise BGCPolicyModelError("policy artifact requires a BGCPolicyModel")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        torch.save(
            {"format": POLICY_ARTIFACT_FORMAT, "state_dict": _model_tensors(model)},
            temporary,
        )
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _load_payload(path: str | Path) -> tuple[bytes, Mapping[str, object]]:
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
        raise BGCPolicyModelError("policy artifact could not be loaded") from error
    if (
        not isinstance(payload, dict)
        or set(payload) != {"format", "state_dict"}
        or payload["format"] != POLICY_ARTIFACT_FORMAT
    ):
        raise BGCPolicyModelError("policy artifact format is incompatible")
    return artifact_bytes, payload


def load_policy_artifact(path: str | Path) -> LoadedPolicyArtifact:
    """Load a policy after validating every serialized tensor."""

    artifact_bytes, payload = _load_payload(path)
    model = BGCPolicyModel()
    expected = model.state_dict()
    raw = payload["state_dict"]
    if not isinstance(raw, dict) or set(raw) != set(expected):
        raise BGCPolicyModelError("policy state-dict keys are incompatible")
    tensors: dict[str, Tensor] = {}
    for name, expected_tensor in expected.items():
        value = raw[name]
        if (
            not isinstance(value, Tensor)
            or value.device.type != "cpu"
            or value.dtype is not expected_tensor.dtype
            or value.shape != expected_tensor.shape
            or not bool(torch.isfinite(value).all().item())
        ):
            raise BGCPolicyModelError(f"policy tensor is incompatible: {name}")
        tensors[name] = value
    try:
        model.load_state_dict(tensors, strict=True)
    except RuntimeError as error:
        raise BGCPolicyModelError("policy tensors could not be loaded") from error
    return LoadedPolicyArtifact(
        model,
        hashlib.sha256(artifact_bytes).hexdigest(),
    )


__all__ = (
    "LoadedPolicyArtifact",
    "POLICY_ARTIFACT_FORMAT",
    "load_policy_artifact",
    "save_policy_artifact",
)

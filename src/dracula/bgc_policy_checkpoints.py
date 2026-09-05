"""Persist and restore artifact-bound policy optimization state atomically.

Checkpoints bind model and optimizer tensors to the exact resolved training
configuration and dataset identities. A checkpoint is accepted only after all
digests match; filesystem replacement exposes either the previous complete file
or the next complete file, never a partially written state.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from dracula.bgc_policy import state_dict_digest
from dracula.bgc_policy_data import canonical_json, load_json
from dracula.bgc_policy_model import BGCPolicyModel
from dracula.bgc_policy_training_contracts import (
    BGCPolicyTrainingError,
    CHECKPOINT_FORMAT_VERSION,
    TRAINING_STATE_FORMAT_VERSION,
)

_CHECKPOINT_FIELDS = {
    "format_version",
    "kind",
    "resolved_config_digest",
    "snapshot_digest",
    "dataset_digest",
    "split_digest",
    "model_contract_digest",
    "optimizer_config_digest",
    "source_revision",
    "source_tree_digest",
    "state_dict_digest",
    "optimizer_state_digest",
    "training_state_schema_version",
    "epoch",
    "next_batch",
    "completed_epochs",
    "best_epoch",
    "best_validation_loss",
    "stale_epochs",
    "maximum_gradient_norm",
    "history",
    "model_state_dict",
    "optimizer_state_dict",
}


def atomic_bytes(path: Path, value: bytes) -> None:
    """Replace one file only after its complete contents reach the filesystem."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("wb") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json(path: Path, value: object) -> None:
    """Write canonical JSON through the shared atomic replacement path."""

    atomic_bytes(path, canonical_json(value) + b"\n")


def atomic_torch(path: Path, value: object) -> None:
    """Save one PyTorch payload without exposing an incomplete checkpoint."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        torch.save(value, temporary)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def seal_resolved_config(output: Path, resolved: dict[str, object]) -> None:
    """Create the immutable run configuration or verify the existing copy."""

    path = output / "resolved-config.json"
    if path.exists():
        if load_json(path) != resolved:
            raise BGCPolicyTrainingError("resolved training configuration differs")
    else:
        atomic_json(path, resolved)


def _nested_digest(value: object) -> str:
    """Hash nested optimizer state with explicit tensor type and shape identity."""

    digest = hashlib.sha256()

    def update(item: object) -> None:
        if isinstance(item, Tensor):
            tensor = item.detach().cpu().contiguous()
            digest.update(b"tensor\0")
            digest.update(str(tensor.dtype).encode("ascii") + b"\0")
            digest.update(",".join(map(str, tensor.shape)).encode("ascii") + b"\0")
            digest.update(tensor.numpy().tobytes(order="C"))
        elif isinstance(item, Mapping):
            digest.update(b"mapping\0")
            for key in sorted(item, key=str):
                update(str(key))
                update(item[key])
        elif isinstance(item, (tuple, list)):
            digest.update(b"sequence\0")
            for nested in item:
                update(nested)
        else:
            digest.update(canonical_json(item))
            digest.update(b"\0")

    update(value)
    return digest.hexdigest()


def checkpoint_payload(
    *,
    kind: str,
    resolved: dict[str, object],
    bundle: Any,
    model: BGCPolicyModel,
    optimizer: torch.optim.AdamW,
    epoch: int,
    next_batch: int,
    completed_epochs: int,
    best_epoch: int,
    best_validation_loss: float,
    stale_epochs: int,
    maximum_gradient_norm: float,
    history: Sequence[dict[str, object]],
) -> dict[str, object]:
    """Capture everything required for exact deterministic minibatch resume."""

    model_state = {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }
    optimizer_state = optimizer.state_dict()
    return {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "kind": kind,
        "resolved_config_digest": resolved["resolved_config_digest"],
        "snapshot_digest": bundle.snapshot.snapshot_digest,
        "dataset_digest": bundle.dataset_digest,
        "split_digest": bundle.split_digest,
        "model_contract_digest": resolved["model_contract_digest"],
        "optimizer_config_digest": resolved["optimizer_config_digest"],
        "source_revision": (
            resolved["source"]["training"]["revision"]  # type: ignore[index]
        ),
        "source_tree_digest": (
            resolved["source"]["training"]["tree_digest"]  # type: ignore[index]
        ),
        "state_dict_digest": state_dict_digest(model_state),
        "optimizer_state_digest": _nested_digest(optimizer_state),
        "training_state_schema_version": TRAINING_STATE_FORMAT_VERSION,
        "epoch": epoch,
        "next_batch": next_batch,
        "completed_epochs": completed_epochs,
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
        "stale_epochs": stale_epochs,
        "maximum_gradient_norm": maximum_gradient_norm,
        "history": list(history),
        "model_state_dict": model_state,
        "optimizer_state_dict": optimizer_state,
    }


def _expected_identity(
    resolved: dict[str, object], bundle: Any
) -> dict[str, object]:
    """Return immutable identities that a checkpoint cannot redefine."""

    return {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "resolved_config_digest": resolved["resolved_config_digest"],
        "snapshot_digest": bundle.snapshot.snapshot_digest,
        "dataset_digest": bundle.dataset_digest,
        "split_digest": bundle.split_digest,
        "model_contract_digest": resolved["model_contract_digest"],
        "optimizer_config_digest": resolved["optimizer_config_digest"],
        "source_revision": (
            resolved["source"]["training"]["revision"]  # type: ignore[index]
        ),
        "source_tree_digest": (
            resolved["source"]["training"]["tree_digest"]  # type: ignore[index]
        ),
        "training_state_schema_version": TRAINING_STATE_FORMAT_VERSION,
    }


def load_checkpoint(
    path: Path,
    *,
    resolved: dict[str, object],
    bundle: Any,
    model: BGCPolicyModel,
    optimizer: torch.optim.AdamW,
) -> dict[str, object]:
    """Verify all checkpoint identities before restoring mutable state."""

    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, ValueError) as error:
        raise BGCPolicyTrainingError(
            "training checkpoint could not be loaded"
        ) from error
    if (
        not isinstance(value, dict)
        or set(value) != _CHECKPOINT_FIELDS
        or not isinstance(value["model_state_dict"], dict)
        or not isinstance(value["optimizer_state_dict"], dict)
    ):
        raise BGCPolicyTrainingError("training checkpoint structure differs")
    if any(
        value[field] != expected
        for field, expected in _expected_identity(resolved, bundle).items()
    ):
        raise BGCPolicyTrainingError("training checkpoint identity differs")
    if (
        value["state_dict_digest"] != state_dict_digest(value["model_state_dict"])
        or value["optimizer_state_digest"]
        != _nested_digest(value["optimizer_state_dict"])
    ):
        raise BGCPolicyTrainingError("training checkpoint tensor digest differs")
    try:
        model.load_state_dict(value["model_state_dict"], strict=True)
        optimizer.load_state_dict(value["optimizer_state_dict"])
    except (RuntimeError, ValueError) as error:
        raise BGCPolicyTrainingError(
            "training checkpoint state is incompatible"
        ) from error
    return value


__all__ = (
    "atomic_bytes",
    "atomic_json",
    "atomic_torch",
    "checkpoint_payload",
    "load_checkpoint",
    "seal_resolved_config",
)

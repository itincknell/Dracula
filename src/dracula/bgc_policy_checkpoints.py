"""Persist resumable policy training state through atomic file replacement.

Checkpoints contain the resolved run configuration, dataset content identities,
model and optimizer state, and the exact minibatch cursor. No nested digest
registry is needed: the loader compares the stored values and restores tensors
through PyTorch's strict state loaders.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from dracula.bgc_policy_data import canonical_json, load_json
from dracula.bgc_policy_model import BGCPolicyModel
from dracula.bgc_policy_training_contracts import (
    BGCPolicyTrainingError,
    CHECKPOINT_FORMAT,
)

_CHECKPOINT_FIELDS = {
    "format",
    "kind",
    "configuration",
    "snapshot_digest",
    "dataset_digest",
    "split_digest",
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
    """Save one PyTorch payload without exposing a partial checkpoint."""

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
    """Create the run configuration or verify the existing immutable copy."""

    path = output / "resolved-config.json"
    if path.exists():
        if load_json(path) != resolved:
            raise BGCPolicyTrainingError("resolved training configuration differs")
    else:
        atomic_json(path, resolved)


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
    """Capture the complete state needed to continue at one minibatch boundary."""

    return {
        "format": CHECKPOINT_FORMAT,
        "kind": kind,
        "configuration": resolved,
        "snapshot_digest": bundle.snapshot.snapshot_digest,
        "dataset_digest": bundle.dataset_digest,
        "split_digest": bundle.split_digest,
        "epoch": epoch,
        "next_batch": next_batch,
        "completed_epochs": completed_epochs,
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
        "stale_epochs": stale_epochs,
        "maximum_gradient_norm": maximum_gradient_norm,
        "history": list(history),
        "model_state_dict": {
            name: tensor.detach().cpu().clone()
            for name, tensor in model.state_dict().items()
        },
        "optimizer_state_dict": optimizer.state_dict(),
    }


def load_checkpoint(
    path: Path,
    *,
    resolved: dict[str, object],
    bundle: Any,
    model: BGCPolicyModel,
    optimizer: torch.optim.AdamW,
) -> dict[str, object]:
    """Validate the run identity and restore model and optimizer state."""

    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, ValueError) as error:
        raise BGCPolicyTrainingError("training checkpoint could not be loaded") from error
    if (
        not isinstance(value, dict)
        or set(value) != _CHECKPOINT_FIELDS
        or value["format"] != CHECKPOINT_FORMAT
        or value["configuration"] != resolved
        or value["snapshot_digest"] != bundle.snapshot.snapshot_digest
        or value["dataset_digest"] != bundle.dataset_digest
        or value["split_digest"] != bundle.split_digest
        or not isinstance(value["model_state_dict"], dict)
        or not isinstance(value["optimizer_state_dict"], dict)
    ):
        raise BGCPolicyTrainingError("training checkpoint identity differs")
    try:
        model.load_state_dict(value["model_state_dict"], strict=True)
        optimizer.load_state_dict(value["optimizer_state_dict"])
    except (RuntimeError, ValueError) as error:
        raise BGCPolicyTrainingError("training checkpoint state is incompatible") from error
    return value


__all__ = (
    "atomic_bytes",
    "atomic_json",
    "atomic_torch",
    "checkpoint_payload",
    "load_checkpoint",
    "seal_resolved_config",
)

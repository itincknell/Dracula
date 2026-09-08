"""Persist resumable policy training state through atomic file replacement.

Checkpoints contain the resolved run configuration, dataset content identities,
model and optimizer state, and the exact minibatch cursor. No nested digest
registry is needed: the loader compares the stored values and restores tensors
through PyTorch's strict state loaders.
"""

from __future__ import annotations

import math
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from dracula.policy.training.data import canonical_json, load_json
from dracula.policy.model import PolicyModel
from dracula.policy.training.contracts import (
    PolicyTrainingError,
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
_CHECKPOINT_KINDS = {
    "initial",
    "latest",
    "best-validation-candidate",
    "final",
}


def _validate_progress(value: dict[str, object]) -> None:
    """Validate scalar resume state before it reaches the optimization loop."""

    integer_fields = (
        "epoch",
        "next_batch",
        "completed_epochs",
        "best_epoch",
        "stale_epochs",
    )
    if any(
        type(value[field]) is not int or value[field] < 0
        for field in integer_fields
    ):
        raise PolicyTrainingError("training checkpoint cursor is invalid")
    loss = value["best_validation_loss"]
    gradient = value["maximum_gradient_norm"]
    history = value["history"]
    if (
        value["kind"] not in _CHECKPOINT_KINDS
        or type(loss) is not float
        or math.isnan(loss)
        or type(gradient) is not float
        or not math.isfinite(gradient)
        or gradient < 0
        or not isinstance(history, list)
        or len(history) != value["completed_epochs"]
        or value["best_epoch"] > value["completed_epochs"]
    ):
        raise PolicyTrainingError("training checkpoint progress is invalid")


def _finite_tensors(value: object) -> bool:
    """Return whether every floating tensor in a nested checkpoint is finite."""

    if isinstance(value, torch.Tensor):
        return not value.is_floating_point() or bool(
            torch.isfinite(value).all().item()
        )
    if isinstance(value, dict):
        return all(_finite_tensors(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite_tensors(item) for item in value)
    return True


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
            raise PolicyTrainingError("resolved training configuration differs")
    else:
        atomic_json(path, resolved)


def checkpoint_payload(
    *,
    kind: str,
    resolved: dict[str, object],
    bundle: Any,
    model: PolicyModel,
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

    # Identity fields prevent state from one snapshot or configuration from
    # being resumed into another run.
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
        # CPU clones make checkpoint bytes independent of the training device
        # and prevent later in-memory updates from mutating the payload.
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
    model: PolicyModel,
    optimizer: torch.optim.AdamW,
) -> dict[str, object]:
    """Validate the run identity and restore model and optimizer state."""

    # Checkpoints are external binary input even when they live under the run
    # directory; loading therefore uses PyTorch's weights-only mode.
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, ValueError) as error:
        raise PolicyTrainingError("training checkpoint could not be loaded") from error
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
        raise PolicyTrainingError("training checkpoint identity differs")
    _validate_progress(value)
    # Validate every stored tensor before either mutable runtime object changes.
    if not _finite_tensors(value["model_state_dict"]) or not _finite_tensors(
        value["optimizer_state_dict"]
    ):
        raise PolicyTrainingError("training checkpoint tensors are non-finite")
    try:
        # Strict model loading catches missing or unexpected learned tensors;
        # optimizer loading restores AdamW momentum for exact continuation.
        model.load_state_dict(value["model_state_dict"], strict=True)
        optimizer.load_state_dict(value["optimizer_state_dict"])
    except (RuntimeError, ValueError) as error:
        raise PolicyTrainingError("training checkpoint state is incompatible") from error
    return value

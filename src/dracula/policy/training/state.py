"""Own resumable policy-training state and its atomic checkpoint transitions.

Optimization updates model parameters and epoch metrics; this module translates
that mutable progress into the persisted checkpoint format. A checkpoint is
accepted only after its run and dataset identities have been verified by the
checkpoint reader. Atomic replacement ensures an interrupted write can never be
mistaken for the latest resumable minibatch boundary.
"""

from __future__ import annotations

import math
import resource
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, cast

import torch

from dracula.policy.training.checkpoints import (
    atomic_json,
    atomic_torch,
    checkpoint_payload,
    load_checkpoint,
)
from dracula.policy.training.data import file_digest
from dracula.policy.model import PolicyModel
from dracula.policy.training.contracts import PolicyTrainingError


@dataclass(frozen=True, slots=True)
class TrainingPaths:
    """Name every mutable or selected checkpoint in one training run."""

    initial: Path
    latest: Path
    best: Path
    final: Path

    @classmethod
    def below(cls, output: Path) -> TrainingPaths:
        """Construct the four fixed checkpoint paths below one run directory."""

        checkpoints = output / "checkpoints"
        return cls(
            initial=checkpoints / "initial.pt",
            latest=checkpoints / "latest.pt",
            best=checkpoints / "best-validation-candidate.pt",
            final=checkpoints / "final.pt",
        )


@dataclass(slots=True)
class TrainingState:
    """Mutable cursor and validation-selection state needed for exact resume."""

    epoch: int = 0
    next_batch: int = 0
    completed_epochs: int = 0
    best_epoch: int = 0
    best_validation_loss: float = math.inf
    stale_epochs: int = 0
    maximum_gradient_norm: float = 0.0
    history: list[dict[str, object]] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class OptimizationResult:
    """Final training state plus run-level timing and stop reason."""

    state: TrainingState
    stopped_early: bool
    total_seconds: float


def peak_rss_bytes() -> int:
    """Return peak process memory in bytes on macOS and Linux."""

    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _payload(
    kind: str,
    *,
    state: TrainingState,
    resolved: dict[str, object],
    bundle: Any,
    model: PolicyModel,
    optimizer: torch.optim.AdamW,
) -> dict[str, object]:
    """Translate the in-memory cursor into the stable checkpoint schema."""

    return checkpoint_payload(
        kind=kind,
        resolved=resolved,
        bundle=bundle,
        model=model,
        optimizer=optimizer,
        epoch=state.epoch,
        next_batch=state.next_batch,
        completed_epochs=state.completed_epochs,
        best_epoch=state.best_epoch,
        best_validation_loss=state.best_validation_loss,
        stale_epochs=state.stale_epochs,
        maximum_gradient_norm=state.maximum_gradient_norm,
        history=state.history,
    )


def _restore_state(checkpoint: dict[str, object]) -> TrainingState:
    """Recover a cursor after checkpoint identity and tensors are verified."""

    return TrainingState(
        epoch=cast(int, checkpoint["epoch"]),
        next_batch=cast(int, checkpoint["next_batch"]),
        completed_epochs=cast(int, checkpoint["completed_epochs"]),
        best_epoch=cast(int, checkpoint["best_epoch"]),
        best_validation_loss=cast(float, checkpoint["best_validation_loss"]),
        stale_epochs=cast(int, checkpoint["stale_epochs"]),
        maximum_gradient_norm=cast(float, checkpoint["maximum_gradient_norm"]),
        history=list(cast(list[dict[str, object]], checkpoint["history"])),
    )


def start_or_resume(
    *,
    resume: bool,
    paths: TrainingPaths,
    resolved: dict[str, object],
    bundle: Any,
    model: PolicyModel,
    optimizer: torch.optim.AdamW,
) -> TrainingState:
    """Create the initial checkpoint pair or restore the latest sealed cursor."""

    if resume:
        checkpoint = load_checkpoint(
            paths.latest,
            resolved=resolved,
            bundle=bundle,
            model=model,
            optimizer=optimizer,
        )
        return _restore_state(checkpoint)
    if any(path.exists() for path in asdict(paths).values()):
        raise PolicyTrainingError("new run already contains checkpoints")

    state = TrainingState()
    initial = _payload(
        "initial",
        state=state,
        resolved=resolved,
        bundle=bundle,
        model=model,
        optimizer=optimizer,
    )
    atomic_torch(paths.initial, initial)
    atomic_torch(paths.latest, {**initial, "kind": "latest"})
    return state


def seal_latest(
    *,
    paths: TrainingPaths,
    state: TrainingState,
    resolved: dict[str, object],
    bundle: Any,
    model: PolicyModel,
    optimizer: torch.optim.AdamW,
) -> None:
    """Atomically persist the current minibatch cursor and optimizer state."""

    atomic_torch(
        paths.latest,
        _payload(
            "latest",
            state=state,
            resolved=resolved,
            bundle=bundle,
            model=model,
            optimizer=optimizer,
        ),
    )


def seal_completed_epoch(
    *,
    state: TrainingState,
    improved: bool,
    output: Path,
    paths: TrainingPaths,
    resolved: dict[str, object],
    bundle: Any,
    model: PolicyModel,
    optimizer: torch.optim.AdamW,
) -> None:
    """Advance to the next epoch and persist its resume and selection state."""

    # Completed epochs restart at batch zero of the next deterministic order.
    # Gradient maxima belong to the epoch just written to the metrics history.
    state.epoch += 1
    state.next_batch = 0
    state.maximum_gradient_norm = 0.0
    seal_latest(
        paths=paths,
        state=state,
        resolved=resolved,
        bundle=bundle,
        model=model,
        optimizer=optimizer,
    )
    atomic_json(
        output / "training-state.json",
        {
            "epoch": state.epoch,
            "next_batch": 0,
            "best_epoch": state.best_epoch,
            "best_validation_loss": state.best_validation_loss,
            "latest_checkpoint_digest": file_digest(paths.latest),
        },
    )
    if improved:
        best = _payload(
            "best-validation-candidate",
            state=state,
            resolved=resolved,
            bundle=bundle,
            model=model,
            optimizer=optimizer,
        )
        atomic_torch(paths.best, best)


def seal_final(
    *,
    paths: TrainingPaths,
    state: TrainingState,
    resolved: dict[str, object],
    bundle: Any,
    model: PolicyModel,
    optimizer: torch.optim.AdamW,
) -> None:
    """Write the terminal checkpoint independently of the selected best epoch."""

    atomic_torch(
        paths.final,
        _payload(
            "final",
            state=state,
            resolved=resolved,
            bundle=bundle,
            model=model,
            optimizer=optimizer,
        ),
    )

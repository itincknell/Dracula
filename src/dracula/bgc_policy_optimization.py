"""Run deterministic minibatch optimization for the standalone BGC policy.

This module owns the mutable portion of training: epoch ordering, gradient
updates, resumable progress, validation-based checkpoint selection, and early
stopping. Configuration resolution, dataset loading, artifact export, and
human-readable reports remain in :mod:`dracula.bgc_policy_training`.
"""

from __future__ import annotations

import math
import resource
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from dracula.bgc_policy_checkpoints import (
    atomic_json,
    atomic_torch,
    checkpoint_payload,
    load_checkpoint,
)
from dracula.bgc_policy_data import file_digest
from dracula.bgc_policy_metrics import (
    distributional_policy_cross_entropy,
    evaluate_bgc_policy,
)
from dracula.bgc_policy_model import BGCPolicyModel
from dracula.bgc_policy_training_contracts import (
    BATCH_SIZE,
    BGCPolicyTrainingError,
    BGCPolicyTrainingInterrupted,
    EARLY_STOP_PATIENCE,
    EPOCH_SHUFFLE_NAMESPACE,
    EpochMetrics,
    GRADIENT_CLIP_NORM,
    LATEST_CHECKPOINT_INTERVAL,
    MAXIMUM_EPOCHS,
    METRICS_FORMAT_VERSION,
    MINIMUM_EPOCHS,
    MINIMUM_IMPROVEMENT,
    TRAINING_STATE_FORMAT_VERSION,
)
from dracula.randomness import derive_seed


@dataclass(frozen=True, slots=True)
class TrainingPaths:
    """Name every mutable or selected checkpoint in one training run."""

    initial: Path
    latest: Path
    best: Path
    final: Path

    @classmethod
    def below(cls, output: Path) -> TrainingPaths:
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


def _epoch_indexes(
    dataset: Any,
    *,
    root_seed: str,
    snapshot_digest: str,
    epoch: int,
) -> Tensor:
    """Derive the complete deterministic minibatch order for one epoch."""

    seed = derive_seed(
        EPOCH_SHUFFLE_NAMESPACE,
        root_seed,
        snapshot_digest,
        dataset.split_digest,
        str(epoch),
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int.from_bytes(seed[:8], "big", signed=False))
    return torch.randperm(dataset.example_count, generator=generator)


def _batches(indexes: Tensor) -> tuple[Tensor, ...]:
    """Divide one fixed epoch order into the configured minibatches."""

    return tuple(
        indexes[start : start + BATCH_SIZE]
        for start in range(0, len(indexes), BATCH_SIZE)
    )


def _finite_parameters(model: BGCPolicyModel) -> bool:
    """Reject an update as soon as any float32 model parameter becomes invalid."""

    return all(
        parameter.dtype is torch.float32
        and bool(torch.isfinite(parameter).all().item())
        for parameter in model.parameters()
    )


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
    model: BGCPolicyModel,
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
    """Recover the cursor only after checkpoint identity and tensors are verified."""

    return TrainingState(
        epoch=checkpoint["epoch"],  # type: ignore[arg-type]
        next_batch=checkpoint["next_batch"],  # type: ignore[arg-type]
        completed_epochs=checkpoint["completed_epochs"],  # type: ignore[arg-type]
        best_epoch=checkpoint["best_epoch"],  # type: ignore[arg-type]
        best_validation_loss=(
            checkpoint["best_validation_loss"]  # type: ignore[arg-type]
        ),
        stale_epochs=checkpoint["stale_epochs"],  # type: ignore[arg-type]
        maximum_gradient_norm=(
            checkpoint["maximum_gradient_norm"]  # type: ignore[arg-type]
        ),
        history=list(checkpoint["history"]),  # type: ignore[arg-type]
    )


def _start_or_resume(
    *,
    resume: bool,
    paths: TrainingPaths,
    resolved: dict[str, object],
    bundle: Any,
    model: BGCPolicyModel,
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
        raise BGCPolicyTrainingError("new run already contains checkpoints")
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


def _train_minibatch(
    model: BGCPolicyModel,
    optimizer: torch.optim.AdamW,
    bundle: Any,
    indexes: Tensor,
    *,
    device: torch.device,
) -> float:
    """Apply one guarded distributional-policy gradient update."""

    observations, _, masks, targets, _ = bundle.training.decoded_batch(
        indexes, device=device
    )
    optimizer.zero_grad(set_to_none=True)
    loss = distributional_policy_cross_entropy(model(observations), masks, targets)
    if not bool(torch.isfinite(loss).item()):
        raise BGCPolicyTrainingError("training loss is non-finite")

    loss.backward()
    gradients = [
        parameter.grad for parameter in model.parameters() if parameter.grad is not None
    ]
    if not gradients or not all(
        bool(torch.isfinite(gradient).all().item()) for gradient in gradients
    ):
        raise BGCPolicyTrainingError("training gradients are non-finite")
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), GRADIENT_CLIP_NORM
    )
    if not bool(torch.isfinite(gradient_norm).item()):
        raise BGCPolicyTrainingError("gradient norm is non-finite")

    optimizer.step()
    if not _finite_parameters(model):
        raise BGCPolicyTrainingError("optimizer produced non-finite parameters")
    return float(gradient_norm.item())


def _train_remaining_batches(
    *,
    state: TrainingState,
    processed_batches: int,
    model: BGCPolicyModel,
    optimizer: torch.optim.AdamW,
    bundle: Any,
    resolved: dict[str, object],
    paths: TrainingPaths,
    root_seed: str,
    device: torch.device,
    interrupt_after_batches: int | None,
) -> int:
    """Finish one epoch from its current sealed minibatch boundary."""

    order = _epoch_indexes(
        bundle.training,
        root_seed=root_seed,
        snapshot_digest=bundle.snapshot.snapshot_digest,
        epoch=state.epoch,
    )
    batches = _batches(order)
    if not 0 <= state.next_batch <= len(batches):
        raise BGCPolicyTrainingError("checkpoint minibatch position is invalid")

    model.train()
    for batch_index in range(state.next_batch, len(batches)):
        gradient_norm = _train_minibatch(
            model,
            optimizer,
            bundle,
            batches[batch_index],
            device=device,
        )
        state.maximum_gradient_norm = max(
            state.maximum_gradient_norm, gradient_norm
        )
        state.next_batch = batch_index + 1
        processed_batches += 1

        interrupt_due = (
            interrupt_after_batches is not None
            and processed_batches >= interrupt_after_batches
        )
        # Periodic latest checkpoints bound lost work; interruption always seals
        # the just-completed minibatch before control returns to the caller.
        if state.next_batch % LATEST_CHECKPOINT_INTERVAL == 0 or interrupt_due:
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
        if interrupt_due:
            raise BGCPolicyTrainingInterrupted(
                "training interrupted at a sealed minibatch boundary"
            )
    return processed_batches


def _complete_epoch(
    *,
    state: TrainingState,
    epoch_started: float,
    output: Path,
    paths: TrainingPaths,
    resolved: dict[str, object],
    bundle: Any,
    model: BGCPolicyModel,
    optimizer: torch.optim.AdamW,
    device: torch.device,
) -> None:
    """Evaluate one epoch, update selection state, and seal its artifacts."""

    training_metrics = evaluate_bgc_policy(model, bundle.training, device=device)
    validation_metrics = evaluate_bgc_policy(model, bundle.validation, device=device)
    elapsed = time.perf_counter() - epoch_started
    state.completed_epochs = state.epoch + 1
    metrics = EpochMetrics(
        METRICS_FORMAT_VERSION,
        state.completed_epochs,
        training_metrics,
        validation_metrics,
        state.maximum_gradient_norm,
        elapsed,
        bundle.training.example_count / max(elapsed, 1e-12),
        peak_rss_bytes(),
    )
    state.history.append(asdict(metrics))
    atomic_json(
        output / "metrics" / f"{state.completed_epochs:04d}.json",
        state.history[-1],
    )

    validation_loss = validation_metrics.total.cross_entropy
    improved = validation_loss < state.best_validation_loss - MINIMUM_IMPROVEMENT
    if improved:
        state.best_validation_loss = validation_loss
        state.best_epoch = state.completed_epochs
        state.stale_epochs = 0
    else:
        state.stale_epochs += 1

    # A completed epoch always resumes at batch zero of the next deterministic
    # order. Gradient maxima are reported per epoch rather than across the run.
    state.epoch += 1
    state.next_batch = 0
    state.maximum_gradient_norm = 0.0
    latest = _payload(
        "latest",
        state=state,
        resolved=resolved,
        bundle=bundle,
        model=model,
        optimizer=optimizer,
    )
    atomic_torch(paths.latest, latest)
    atomic_json(
        output / "training-state.json",
        {
            "format_version": TRAINING_STATE_FORMAT_VERSION,
            "epoch": state.epoch,
            "next_batch": 0,
            "best_epoch": state.best_epoch,
            "best_validation_loss": state.best_validation_loss,
            "latest_checkpoint_digest": file_digest(paths.latest),
        },
    )
    if improved:
        atomic_torch(paths.best, {**latest, "kind": "best-validation-candidate"})

def optimize_policy(
    *,
    output: Path,
    resolved: dict[str, object],
    bundle: Any,
    model: BGCPolicyModel,
    optimizer: torch.optim.AdamW,
    root_seed: str,
    device: torch.device,
    resume: bool,
    interrupt_after_batches: int | None,
    smoke_epochs: int | None,
) -> OptimizationResult:
    """Train through early stopping or the configured epoch limit."""

    paths = TrainingPaths.below(output)
    state = _start_or_resume(
        resume=resume,
        paths=paths,
        resolved=resolved,
        bundle=bundle,
        model=model,
        optimizer=optimizer,
    )
    maximum_epochs = smoke_epochs if smoke_epochs is not None else MAXIMUM_EPOCHS
    minimum_epochs = smoke_epochs if smoke_epochs is not None else MINIMUM_EPOCHS
    processed_batches = 0
    stopped_early = False
    run_started = time.perf_counter()

    while state.epoch < maximum_epochs:
        epoch_started = time.perf_counter()
        processed_batches = _train_remaining_batches(
            state=state,
            processed_batches=processed_batches,
            model=model,
            optimizer=optimizer,
            bundle=bundle,
            resolved=resolved,
            paths=paths,
            root_seed=root_seed,
            device=device,
            interrupt_after_batches=interrupt_after_batches,
        )
        _complete_epoch(
            state=state,
            epoch_started=epoch_started,
            output=output,
            paths=paths,
            resolved=resolved,
            bundle=bundle,
            model=model,
            optimizer=optimizer,
            device=device,
        )
        if (
            state.completed_epochs >= minimum_epochs
            and state.stale_epochs >= EARLY_STOP_PATIENCE
        ):
            stopped_early = True
            break

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
    return OptimizationResult(
        state=state,
        stopped_early=stopped_early,
        total_seconds=time.perf_counter() - run_started,
    )


__all__ = (
    "OptimizationResult",
    "TrainingPaths",
    "TrainingState",
    "optimize_policy",
    "peak_rss_bytes",
)

"""Run deterministic minibatch optimization for the standalone policy.

This module owns deterministic epoch ordering, gradient updates, evaluation,
validation-based checkpoint selection, and early stopping. Resumable cursor
persistence lives in :mod:`dracula.policy.training.state`; dataset loading,
artifact export, and reporting remain outside the optimization loop.
"""

from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from dracula.policy.training.checkpoints import atomic_json
from dracula.policy.training.metrics import (
    distributional_policy_cross_entropy,
    evaluate_policy,
)
from dracula.policy.model import PolicyModel
from dracula.policy.training.contracts import (
    BATCH_SIZE,
    PolicyTrainingError,
    PolicyTrainingInterrupted,
    EARLY_STOP_PATIENCE,
    EpochMetrics,
    GRADIENT_CLIP_NORM,
    LATEST_CHECKPOINT_INTERVAL,
    MAXIMUM_EPOCHS,
    MINIMUM_EPOCHS,
    MINIMUM_IMPROVEMENT,
)
from dracula.policy.training.state import (
    OptimizationResult,
    TrainingPaths,
    TrainingState,
    peak_rss_bytes,
    seal_completed_epoch,
    seal_final,
    seal_latest,
    start_or_resume,
)
from dracula.randomness import stable_seed


def _epoch_indexes(
    dataset: Any,
    *,
    seed: int,
    snapshot_digest: str,
    epoch: int,
) -> Tensor:
    """Derive the complete deterministic minibatch order for one epoch."""

    epoch_seed = stable_seed(
        seed,
        snapshot_digest,
        dataset.split_digest,
        epoch,
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(epoch_seed)
    return torch.randperm(dataset.example_count, generator=generator)


def _batches(indexes: Tensor) -> tuple[Tensor, ...]:
    """Divide one fixed epoch order into the configured minibatches."""

    return tuple(
        indexes[start : start + BATCH_SIZE]
        for start in range(0, len(indexes), BATCH_SIZE)
    )


def _finite_parameters(model: PolicyModel) -> bool:
    """Reject an update as soon as any float32 model parameter becomes invalid."""

    return all(
        parameter.dtype is torch.float32
        and bool(torch.isfinite(parameter).all().item())
        for parameter in model.parameters()
    )


def _train_minibatch(
    model: PolicyModel,
    optimizer: torch.optim.AdamW,
    bundle: Any,
    indexes: Tensor,
    *,
    device: torch.device,
) -> float:
    """Apply one guarded distributional-policy gradient update."""

    # Only observations, representative-action masks, and visit distributions
    # participate in the objective; selected actions remain audit metadata.
    observations, _, masks, targets, _ = bundle.training.decoded_batch(
        indexes, device=device
    )
    optimizer.zero_grad(set_to_none=True)
    loss = distributional_policy_cross_entropy(model(observations), masks, targets)
    if not bool(torch.isfinite(loss).item()):
        raise PolicyTrainingError("training loss is non-finite")

    loss.backward()
    # Reject invalid gradients before clipping or AdamW can write them into
    # parameters and optimizer momentum buffers.
    gradients = [
        parameter.grad for parameter in model.parameters() if parameter.grad is not None
    ]
    if not gradients or not all(
        bool(torch.isfinite(gradient).all().item()) for gradient in gradients
    ):
        raise PolicyTrainingError("training gradients are non-finite")
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), GRADIENT_CLIP_NORM
    )
    if not bool(torch.isfinite(gradient_norm).item()):
        raise PolicyTrainingError("gradient norm is non-finite")

    optimizer.step()
    if not _finite_parameters(model):
        raise PolicyTrainingError("optimizer produced non-finite parameters")
    return float(gradient_norm.item())


def _train_remaining_batches(
    *,
    state: TrainingState,
    processed_batches: int,
    model: PolicyModel,
    optimizer: torch.optim.AdamW,
    bundle: Any,
    resolved: dict[str, object],
    paths: TrainingPaths,
    seed: int,
    device: torch.device,
    interrupt_after_batches: int | None,
) -> int:
    """Finish one epoch from its current sealed minibatch boundary."""

    # Reconstructing the order from snapshot identity, seed, and epoch makes an
    # interrupted run resume at exactly the same next minibatch.
    order = _epoch_indexes(
        bundle.training,
        seed=seed,
        snapshot_digest=bundle.snapshot.snapshot_digest,
        epoch=state.epoch,
    )
    batches = _batches(order)
    if not 0 <= state.next_batch <= len(batches):
        raise PolicyTrainingError("checkpoint minibatch position is invalid")

    model.train()
    for batch_index in range(state.next_batch, len(batches)):
        gradient_norm = _train_minibatch(
            model,
            optimizer,
            bundle,
            batches[batch_index],
            device=device,
        )
        state.maximum_gradient_norm = max(state.maximum_gradient_norm, gradient_norm)
        state.next_batch = batch_index + 1
        processed_batches += 1

        interrupt_due = (
            interrupt_after_batches is not None
            and processed_batches >= interrupt_after_batches
        )
        # Periodic latest checkpoints bound lost work; interruption always seals
        # the just-completed minibatch before control returns to the caller.
        if state.next_batch % LATEST_CHECKPOINT_INTERVAL == 0 or interrupt_due:
            seal_latest(
                paths=paths,
                state=state,
                resolved=resolved,
                bundle=bundle,
                model=model,
                optimizer=optimizer,
            )
        if interrupt_due:
            raise PolicyTrainingInterrupted(
                "training interrupted at a sealed minibatch boundary"
            )
    return processed_batches


def _evaluate_epoch(
    *,
    state: TrainingState,
    epoch_started: float,
    bundle: Any,
    model: PolicyModel,
    device: torch.device,
) -> EpochMetrics:
    """Evaluate the updated model and build one complete epoch record."""

    # Training metrics describe fit; validation metrics alone select the model.
    training_metrics = evaluate_policy(model, bundle.training, device=device)
    validation_metrics = evaluate_policy(model, bundle.validation, device=device)
    elapsed = time.perf_counter() - epoch_started
    return EpochMetrics(
        state.epoch + 1,
        training_metrics,
        validation_metrics,
        state.maximum_gradient_norm,
        elapsed,
        bundle.training.example_count / max(elapsed, 1e-12),
        peak_rss_bytes(),
    )


def _record_epoch_result(
    state: TrainingState,
    metrics: EpochMetrics,
    *,
    output: Path,
) -> bool:
    """Append metrics and update lowest-validation-loss selection state."""

    state.completed_epochs = metrics.epoch
    state.history.append(asdict(metrics))
    atomic_json(
        output / "metrics" / f"{state.completed_epochs:04d}.json",
        state.history[-1],
    )
    # Tiny numerical changes do not reset patience unless they exceed the
    # configured minimum improvement.
    validation_loss = metrics.validation.total.cross_entropy
    improved = validation_loss < state.best_validation_loss - MINIMUM_IMPROVEMENT
    if improved:
        state.best_validation_loss = validation_loss
        state.best_epoch = state.completed_epochs
        state.stale_epochs = 0
    else:
        state.stale_epochs += 1
    return improved


def _complete_epoch(
    *,
    state: TrainingState,
    epoch_started: float,
    output: Path,
    paths: TrainingPaths,
    resolved: dict[str, object],
    bundle: Any,
    model: PolicyModel,
    optimizer: torch.optim.AdamW,
    device: torch.device,
) -> None:
    """Evaluate, select, and seal one completed epoch."""

    metrics = _evaluate_epoch(
        state=state,
        epoch_started=epoch_started,
        bundle=bundle,
        model=model,
        device=device,
    )
    improved = _record_epoch_result(state, metrics, output=output)
    seal_completed_epoch(
        state=state,
        improved=improved,
        output=output,
        paths=paths,
        resolved=resolved,
        bundle=bundle,
        model=model,
        optimizer=optimizer,
    )


def _run_epoch(
    *,
    state: TrainingState,
    processed_batches: int,
    output: Path,
    paths: TrainingPaths,
    resolved: dict[str, object],
    bundle: Any,
    model: PolicyModel,
    optimizer: torch.optim.AdamW,
    seed: int,
    device: torch.device,
    interrupt_after_batches: int | None,
) -> int:
    """Train and seal the epoch currently named by the resume cursor."""

    epoch_started = time.perf_counter()
    processed_batches = _train_remaining_batches(
        state=state,
        processed_batches=processed_batches,
        model=model,
        optimizer=optimizer,
        bundle=bundle,
        resolved=resolved,
        paths=paths,
        seed=seed,
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
    return processed_batches


def _run_training_epochs(
    *,
    state: TrainingState,
    output: Path,
    paths: TrainingPaths,
    resolved: dict[str, object],
    bundle: Any,
    model: PolicyModel,
    optimizer: torch.optim.AdamW,
    seed: int,
    device: torch.device,
    interrupt_after_batches: int | None,
    smoke_epochs: int | None,
) -> bool:
    """Run complete epochs and return whether validation triggered early stop."""

    # Smoke mode shortens only the epoch limits; it exercises the same update,
    # evaluation, checkpoint, and resume path as a complete run.
    maximum_epochs = smoke_epochs if smoke_epochs is not None else MAXIMUM_EPOCHS
    minimum_epochs = smoke_epochs if smoke_epochs is not None else MINIMUM_EPOCHS
    processed_batches = 0
    while state.epoch < maximum_epochs:
        processed_batches = _run_epoch(
            state=state,
            processed_batches=processed_batches,
            output=output,
            paths=paths,
            resolved=resolved,
            bundle=bundle,
            model=model,
            optimizer=optimizer,
            seed=seed,
            device=device,
            interrupt_after_batches=interrupt_after_batches,
        )
        if (
            state.completed_epochs >= minimum_epochs
            and state.stale_epochs >= EARLY_STOP_PATIENCE
        ):
            return True
    return False


def optimize_policy(
    *,
    output: Path,
    resolved: dict[str, object],
    bundle: Any,
    model: PolicyModel,
    optimizer: torch.optim.AdamW,
    seed: int,
    device: torch.device,
    resume: bool,
    interrupt_after_batches: int | None,
    smoke_epochs: int | None,
) -> OptimizationResult:
    """Train through early stopping or the configured epoch limit."""

    paths = TrainingPaths.below(output)
    # This boundary either restores an exact sealed cursor or creates the
    # initial checkpoint before the first parameter update.
    state = start_or_resume(
        resume=resume,
        paths=paths,
        resolved=resolved,
        bundle=bundle,
        model=model,
        optimizer=optimizer,
    )
    run_started = time.perf_counter()
    stopped_early = _run_training_epochs(
        state=state,
        output=output,
        paths=paths,
        resolved=resolved,
        bundle=bundle,
        model=model,
        optimizer=optimizer,
        seed=seed,
        device=device,
        interrupt_after_batches=interrupt_after_batches,
        smoke_epochs=smoke_epochs,
    )

    # The final checkpoint records where optimization stopped; the separately
    # maintained best checkpoint remains the artifact-selection source.
    seal_final(
        paths=paths,
        state=state,
        resolved=resolved,
        bundle=bundle,
        model=model,
        optimizer=optimizer,
    )
    return OptimizationResult(
        state=state,
        stopped_early=stopped_early,
        total_seconds=time.perf_counter() - run_started,
    )

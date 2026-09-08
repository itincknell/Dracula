"""Train the standalone policy from sealed information-set UCT visit targets.

This module coordinates verified dataset loading, optimization, artifact export,
and run reports. Configuration resolution, minibatch execution, metrics,
checkpoint persistence, and command-line parsing live in focused companion
modules so each phase can be read and tested independently.
"""

from __future__ import annotations

import os
import shutil
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from dracula.policy.artifact import (
    load_policy_artifact,
    save_policy_artifact,
)
from dracula.policy.training.checkpoints import (
    load_checkpoint,
    seal_resolved_config,
)
from dracula.policy.model import (
    PolicyModel,
    PARAMETER_COUNT,
)
from dracula.policy.training.data import load_json
from dracula.policy.training.dataset import load_policy_dataset
from dracula.policy.training.metrics import (
    evaluate_policy,
)
from dracula.policy.training.optimization import (
    optimize_policy,
)
from dracula.policy.training.reporting import write_training_summary
from dracula.policy.training.config import (
    config_from_resolved,
    resolved_training_configuration,
)
from dracula.policy.training.contracts import (
    BETAS,
    PolicyTrainingConfig,
    PolicyTrainingError,
    EPSILON,
    EvaluationMetrics,
    LEARNING_RATE,
    TrainingResult,
    WEIGHT_DECAY,
)
from dracula.policy.training.state import OptimizationResult, TrainingPaths


def _training_device(name: str) -> torch.device:
    """Resolve only the two explicitly supported optimization devices."""

    if name == "cpu":
        return torch.device("cpu")
    if name == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    raise PolicyTrainingError(f"training device is unavailable: {name}")


def _optimizer(model: PolicyModel) -> torch.optim.AdamW:
    """Construct AdamW from the immutable training hyperparameters."""

    return torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        betas=BETAS,
        eps=EPSILON,
        weight_decay=WEIGHT_DECAY,
    )


def _validate_training_request(
    smoke_epochs: int | None, interrupt_after_batches: int | None
) -> None:
    """Validate invocation-only limits before creating any run artifacts."""

    if smoke_epochs is not None and (
        type(smoke_epochs) is not int or not 1 <= smoke_epochs <= 8
    ):
        raise PolicyTrainingError("smoke epochs must be between one and eight")
    if interrupt_after_batches is not None and (
        type(interrupt_after_batches) is not int or interrupt_after_batches < 1
    ):
        raise PolicyTrainingError("interrupt batch count must be positive")


def _new_model(config: PolicyTrainingConfig, device: torch.device) -> PolicyModel:
    """Construct the configured policy and verify its locked architecture size."""

    model = PolicyModel(config.run.seed)
    model = model.to(device)
    if sum(parameter.numel() for parameter in model.parameters()) != PARAMETER_COUNT:
        raise PolicyTrainingError("policy model parameter count differs")
    return model


def _export_selected_candidate(
    *,
    output: Path,
    resolved: dict[str, object],
    bundle: Any,
    best_path: Path,
) -> tuple[Path, EvaluationMetrics]:
    """Restore the best checkpoint, export it, and verify exact CPU inference."""

    # The best checkpoint, rather than the final optimization state, supplies
    # the model exported for evaluation or deployment.
    selected_model = PolicyModel()
    selected_optimizer = _optimizer(selected_model)
    load_checkpoint(
        best_path,
        resolved=resolved,
        bundle=bundle,
        model=selected_model,
        optimizer=selected_optimizer,
    )
    artifact_path = output / "artifacts" / "policy.pt"
    save_policy_artifact(artifact_path, selected_model)
    loaded = load_policy_artifact(artifact_path)
    # A small fixed probe catches export or reload changes before the much more
    # expensive complete validation pass.
    probe_indexes = torch.arange(min(8, bundle.validation.example_count))
    probe, _, _, _, _ = bundle.validation.decoded_batch(
        probe_indexes, device=torch.device("cpu")
    )
    selected_model.eval()
    loaded.model.eval()
    with torch.no_grad():
        if not torch.equal(selected_model(probe), loaded.model(probe)):
            raise PolicyTrainingError("exported inference differs from checkpoint")

    # Inference latency varies between passes, so deterministic validation compares
    # every score and count while intentionally excluding that timing field.
    selected_validation = evaluate_policy(
        selected_model, bundle.validation, device=torch.device("cpu")
    )
    repeated_validation = evaluate_policy(
        selected_model, bundle.validation, device=torch.device("cpu")
    )
    first = asdict(selected_validation)
    second = asdict(repeated_validation)
    first.pop("inference_latency_seconds")
    second.pop("inference_latency_seconds")
    if first != second:
        raise PolicyTrainingError("deterministic validation metrics differ")
    return artifact_path, selected_validation


def _prepare_training_run(
    *,
    config: PolicyTrainingConfig,
    smoke_epochs: int | None,
) -> tuple[Any, dict[str, object], Path]:
    """Verify the snapshot and create the immutable run directory structure."""

    # Dataset verification precedes output creation, so an invalid snapshot
    # cannot leave a run directory that resembles initialized training.
    bundle = load_policy_dataset(config.snapshot_path)
    resolved = resolved_training_configuration(
        config, bundle, smoke_epochs=smoke_epochs
    )
    output = config.output_path
    output.mkdir(parents=True, exist_ok=True)
    seal_resolved_config(output, resolved)
    for directory in ("checkpoints", "metrics", "artifacts", "reports"):
        (output / directory).mkdir(exist_ok=True)
    return bundle, resolved, output


def _run_optimization(
    *,
    config: PolicyTrainingConfig,
    bundle: Any,
    resolved: dict[str, object],
    output: Path,
    resume: bool,
    interrupt_after_batches: int | None,
    smoke_epochs: int | None,
) -> tuple[OptimizationResult, TrainingPaths]:
    """Construct mutable training state and run the deterministic optimizer."""

    device = _training_device(config.optimization.device)
    model = _new_model(config, device)
    optimizer = _optimizer(model)
    paths = TrainingPaths.below(output)
    result = optimize_policy(
        output=output,
        resolved=resolved,
        bundle=bundle,
        model=model,
        optimizer=optimizer,
        seed=config.run.seed,
        device=device,
        resume=resume,
        interrupt_after_batches=interrupt_after_batches,
        smoke_epochs=smoke_epochs,
    )
    return result, paths


def _finalize_training_run(
    *,
    output: Path,
    resolved: dict[str, object],
    bundle: Any,
    optimization: OptimizationResult,
    paths: TrainingPaths,
) -> TrainingResult:
    """Export the selected epoch, seal reports, and return public run paths."""

    state = optimization.state
    # Export and reporting occur only after optimization has sealed its final
    # state and preserved the independently selected best checkpoint.
    artifact_path, selected_validation = _export_selected_candidate(
        output=output,
        resolved=resolved,
        bundle=bundle,
        best_path=paths.best,
    )
    write_training_summary(
        output=output,
        resolved=resolved,
        bundle=bundle,
        paths=paths,
        artifact_path=artifact_path,
        selected_validation=selected_validation,
        completed_epochs=state.completed_epochs,
        best_epoch=state.best_epoch,
        best_validation_loss=state.best_validation_loss,
        stopped_early=optimization.stopped_early,
        history=state.history,
        total_seconds=optimization.total_seconds,
    )
    return TrainingResult(
        str(output),
        state.completed_epochs,
        state.best_epoch,
        state.best_validation_loss,
        optimization.stopped_early,
        str(paths.final),
        str(paths.best),
        str(artifact_path),
    )


def train_policy(
    config: PolicyTrainingConfig,
    *,
    resume: bool = False,
    interrupt_after_batches: int | None = None,
    smoke_epochs: int | None = None,
) -> TrainingResult:
    """Coordinate one visit-distillation training run."""

    _validate_training_request(smoke_epochs, interrupt_after_batches)
    bundle, resolved, output = _prepare_training_run(
        config=config,
        smoke_epochs=smoke_epochs,
    )
    optimization, paths = _run_optimization(
        config=config,
        bundle=bundle,
        resolved=resolved,
        output=output,
        resume=resume,
        interrupt_after_batches=interrupt_after_batches,
        smoke_epochs=smoke_epochs,
    )
    return _finalize_training_run(
        output=output,
        resolved=resolved,
        bundle=bundle,
        optimization=optimization,
        paths=paths,
    )


def validate_policy_run(output_path: str | Path) -> EvaluationMetrics:
    """Re-evaluate a run's verified best-validation checkpoint on CPU."""

    output = Path(output_path).expanduser().resolve()
    config, smoke_epochs = config_from_resolved(output)
    bundle = load_policy_dataset(config.snapshot_path)
    resolved = resolved_training_configuration(
        config, bundle, smoke_epochs=smoke_epochs
    )
    if load_json(output / "resolved-config.json") != resolved:
        raise PolicyTrainingError("run source or resolved configuration differs")
    model = PolicyModel()
    optimizer = _optimizer(model)
    load_checkpoint(
        output / "checkpoints" / "best-validation-candidate.pt",
        resolved=resolved,
        bundle=bundle,
        model=model,
        optimizer=optimizer,
    )
    return evaluate_policy(model, bundle.validation, device=torch.device("cpu"))


def export_policy_run(
    output_path: str | Path, destination: str | Path | None = None
) -> Path:
    """Verify and optionally copy an unaccepted candidate artifact atomically."""

    output = Path(output_path).expanduser().resolve()
    source = output / "artifacts" / "policy.pt"
    loaded = load_policy_artifact(source)
    if destination is None:
        return source
    target = Path(destination).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
    try:
        shutil.copyfile(source, temporary)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    verified = load_policy_artifact(target)
    if verified.artifact_digest != loaded.artifact_digest:
        raise PolicyTrainingError("copied candidate artifact digest differs")
    return target


def inspect_policy_snapshot(path: str | Path) -> dict[str, object]:
    """Verify a sealed snapshot and summarize its split and placement counts."""

    bundle = load_policy_dataset(path)
    return {
        "snapshot_digest": bundle.snapshot.snapshot_digest,
        "dataset_digest": bundle.dataset_digest,
        "split_digest": bundle.split_digest,
        "training_games": bundle.snapshot.training_game_count,
        "validation_games": bundle.snapshot.validation_game_count,
        "training_rows": bundle.training.example_count,
        "validation_rows": bundle.validation.example_count,
        "training_placement_counts": dict(
            Counter(bundle.training.placements.tolist())
        ),
        "validation_placement_counts": dict(
            Counter(bundle.validation.placements.tolist())
        ),
        "privacy_verified": True,
    }

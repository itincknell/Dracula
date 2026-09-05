"""Train the standalone policy from sealed BGC root-visit distributions.

This module coordinates verified dataset loading, optimization, artifact export,
and run reports. Configuration resolution, minibatch execution, metrics,
checkpoint persistence, and command-line parsing live in focused companion
modules so each phase can be read and tested independently.
"""

from __future__ import annotations

import os
import shutil
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from dracula.bgc_policy import (
    load_bgc_policy_artifact,
    save_bgc_policy_artifact,
)
from dracula.bgc_policy_checkpoints import (
    atomic_bytes as _atomic_bytes,
    atomic_json as _atomic_json,
    load_checkpoint as _load_checkpoint,
    seal_resolved_config as _seal_resolved_config,
)
from dracula.bgc_policy_model import (
    BGCPolicyModel,
    PARAMETER_COUNT,
)
from dracula.bgc_policy_data import (
    file_digest as _file_digest,
    load_json as _load_json,
)
from dracula.bgc_policy_metrics import (
    distributional_policy_cross_entropy,
    distributional_policy_statistics,
    evaluate_bgc_policy,
)
from dracula.bgc_policy_optimization import (
    TrainingPaths,
    optimize_policy,
    peak_rss_bytes,
)
from dracula.bgc_policy_training_config import (
    config_from_resolved as _config_from_resolved,
    load_bgc_policy_training_config,
    resolved_training_configuration as _resolved_configuration,
)
from dracula.bgc_policy_training_contracts import (
    BATCH_SIZE,
    BETAS,
    BGCPolicyTrainingConfig,
    BGCPolicyTrainingError,
    BGCPolicyTrainingInterrupted,
    DatasetSection,
    EPSILON,
    EvaluationMetrics,
    LEARNING_RATE,
    ModelSection,
    OptimizationSection,
    RunSection,
    SUMMARY_FORMAT_VERSION,
    TrainingResult,
    WEIGHT_DECAY,
)


def load_bgc_card_policy_dataset(snapshot_path: str | Path):
    """Load the sealed 659-bit corpus used by the selected standalone policy."""

    from dracula.bgc_policy_migration import load_migrated_policy_dataset

    return load_migrated_policy_dataset(snapshot_path)


def _training_device(name: str) -> torch.device:
    """Resolve only the two explicitly supported optimization devices."""

    if name == "cpu":
        return torch.device("cpu")
    if name == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    raise BGCPolicyTrainingError(f"training device is unavailable: {name}")


def _optimizer(model: BGCPolicyModel) -> torch.optim.AdamW:
    """Construct AdamW from the immutable training hyperparameters."""

    return torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        betas=BETAS,
        eps=EPSILON,
        weight_decay=WEIGHT_DECAY,
    )


def _markdown_report(
    *,
    bundle: Any,
    history: Sequence[dict[str, object]],
    best_epoch: int,
    best_loss: float,
    stopped_early: bool,
) -> str:
    lines = [
        "# BGC visit-distillation training report",
        "",
        f"- Snapshot digest: `{bundle.snapshot.snapshot_digest}`",
        f"- Dataset digest: `{bundle.dataset_digest}`",
        f"- Training examples: {bundle.training.example_count:,}",
        f"- Validation examples: {bundle.validation.example_count:,}",
        f"- Best epoch: {best_epoch}",
        f"- Best validation cross-entropy: {best_loss:.8f}",
        f"- Early stopped: {str(stopped_early).lower()}",
        "",
        "| Epoch | Train CE | Validation CE | Validation KL | Top-1 | Top-2 | Top-3 |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for entry in history:
        training = entry["training"]["total"]  # type: ignore[index]
        validation = entry["validation"]["total"]  # type: ignore[index]
        lines.append(
            f"| {entry['epoch']} | {training['cross_entropy']:.6f} | "
            f"{validation['cross_entropy']:.6f} | {validation['kl_divergence']:.6f} | "
            f"{validation['top_one']:.4f} | {validation['top_two']:.4f} | "
            f"{validation['top_three']:.4f} |"
        )
    return "\n".join(lines) + "\n"


def _validate_training_request(
    smoke_epochs: int | None, interrupt_after_batches: int | None
) -> None:
    """Validate invocation-only limits before creating any run artifacts."""

    if smoke_epochs is not None and (
        type(smoke_epochs) is not int or not 1 <= smoke_epochs <= 8
    ):
        raise BGCPolicyTrainingError("smoke epochs must be between one and eight")
    if interrupt_after_batches is not None and (
        type(interrupt_after_batches) is not int or interrupt_after_batches < 1
    ):
        raise BGCPolicyTrainingError("interrupt batch count must be positive")


def _new_model(config: BGCPolicyTrainingConfig, device: torch.device) -> BGCPolicyModel:
    """Construct the configured policy and verify its locked architecture size."""

    model = BGCPolicyModel(
        run_root_seed=config.run.root_seed,
        model_id=config.model.model_id,
        initialization_ordinal=config.model.initialization_ordinal,
    ).to(device)
    if sum(parameter.numel() for parameter in model.parameters()) != PARAMETER_COUNT:
        raise BGCPolicyTrainingError("policy model parameter count differs")
    return model


def _export_selected_candidate(
    *,
    config: BGCPolicyTrainingConfig,
    output: Path,
    resolved: dict[str, object],
    bundle: Any,
    best_path: Path,
) -> tuple[Path, Any, EvaluationMetrics]:
    """Restore the best checkpoint, export it, and verify exact CPU inference."""

    selected_model = BGCPolicyModel(
        run_root_seed=config.run.root_seed,
        model_id=config.model.model_id,
        initialization_ordinal=config.model.initialization_ordinal,
    )
    selected_optimizer = _optimizer(selected_model)
    _load_checkpoint(
        best_path,
        resolved=resolved,
        bundle=bundle,
        model=selected_model,
        optimizer=selected_optimizer,
    )
    artifact_path = output / "artifacts" / "unaccepted-candidate.pt"
    save_bgc_policy_artifact(
        artifact_path,
        selected_model,
        source_revision=(
            resolved["source"]["training"]["revision"]  # type: ignore[index]
        ),
        source_tree_digest=(
            resolved["source"]["training"]["tree_digest"]  # type: ignore[index]
        ),
        training_configuration=resolved,
        corpus_snapshot_digest=bundle.snapshot.snapshot_digest,
        dataset_digest=bundle.dataset_digest,
    )
    loaded = load_bgc_policy_artifact(artifact_path)
    probe_indexes = torch.arange(min(8, bundle.validation.example_count))
    probe, _, _, _, _ = bundle.validation.decoded_batch(
        probe_indexes, device=torch.device("cpu")
    )
    selected_model.eval()
    loaded.model.eval()
    with torch.no_grad():
        if not torch.equal(selected_model(probe), loaded.model(probe)):
            raise BGCPolicyTrainingError("exported inference differs from checkpoint")

    # Inference latency varies between passes, so deterministic validation compares
    # every score and count while intentionally excluding that timing field.
    selected_validation = evaluate_bgc_policy(
        selected_model, bundle.validation, device=torch.device("cpu")
    )
    repeated_validation = evaluate_bgc_policy(
        selected_model, bundle.validation, device=torch.device("cpu")
    )
    first = asdict(selected_validation)
    second = asdict(repeated_validation)
    first.pop("inference_latency_seconds")
    second.pop("inference_latency_seconds")
    if first != second:
        raise BGCPolicyTrainingError("deterministic validation metrics differ")
    return artifact_path, loaded, selected_validation


def _write_training_summary(
    *,
    output: Path,
    resolved: dict[str, object],
    bundle: Any,
    paths: TrainingPaths,
    artifact_path: Path,
    loaded_artifact: Any,
    selected_validation: EvaluationMetrics,
    completed_epochs: int,
    best_epoch: int,
    best_validation_loss: float,
    stopped_early: bool,
    history: Sequence[dict[str, object]],
    total_seconds: float,
) -> None:
    """Seal machine-readable and concise human-readable run summaries."""

    summary = {
        "format_version": SUMMARY_FORMAT_VERSION,
        "resolved_config_digest": resolved["resolved_config_digest"],
        "snapshot_digest": bundle.snapshot.snapshot_digest,
        "dataset_digest": bundle.dataset_digest,
        "completed_epochs": completed_epochs,
        "best_epoch": best_epoch,
        "best_validation_cross_entropy": best_validation_loss,
        "stopped_early": stopped_early,
        "history": history,
        "selected_validation": asdict(selected_validation),
        "deterministic_validation_match": True,
        "runtime": {
            "total_seconds": total_seconds,
            "peak_rss_bytes": peak_rss_bytes(),
        },
        "artifacts": {
            "initial": {
                "path": str(paths.initial),
                "digest": _file_digest(paths.initial),
            },
            "latest": {
                "path": str(paths.latest),
                "digest": _file_digest(paths.latest),
            },
            "best_validation_candidate": {
                "path": str(paths.best),
                "digest": _file_digest(paths.best),
            },
            "final": {"path": str(paths.final), "digest": _file_digest(paths.final)},
            "unaccepted_candidate": {
                "path": str(artifact_path),
                "digest": _file_digest(artifact_path),
                "state_dict_digest": loaded_artifact.metadata.state_dict_digest,
                "candidate_status": loaded_artifact.metadata.candidate_status,
            },
        },
    }
    _atomic_json(output / "metrics" / "summary.json", summary)
    _atomic_bytes(
        output / "reports" / "summary.md",
        _markdown_report(
            bundle=bundle,
            history=history,
            best_epoch=best_epoch,
            best_loss=best_validation_loss,
            stopped_early=stopped_early,
        ).encode("utf-8"),
    )


def train_bgc_policy(
    config: BGCPolicyTrainingConfig,
    *,
    resume: bool = False,
    interrupt_after_batches: int | None = None,
    smoke_epochs: int | None = None,
) -> TrainingResult:
    """Coordinate one artifact-bound visit-distillation training run."""

    _validate_training_request(smoke_epochs, interrupt_after_batches)
    bundle = load_bgc_card_policy_dataset(config.snapshot_path)
    resolved = _resolved_configuration(config, bundle, smoke_epochs=smoke_epochs)
    output = config.output_path
    output.mkdir(parents=True, exist_ok=True)
    _seal_resolved_config(output, resolved)
    for directory in ("checkpoints", "metrics", "artifacts", "reports"):
        (output / directory).mkdir(exist_ok=True)

    device = _training_device(config.optimization.device)
    model = _new_model(config, device)
    optimizer = _optimizer(model)
    paths = TrainingPaths.below(output)
    optimization = optimize_policy(
        output=output,
        resolved=resolved,
        bundle=bundle,
        model=model,
        optimizer=optimizer,
        root_seed=config.run.root_seed,
        device=device,
        resume=resume,
        interrupt_after_batches=interrupt_after_batches,
        smoke_epochs=smoke_epochs,
    )
    state = optimization.state
    artifact_path, loaded_artifact, selected_validation = (
        _export_selected_candidate(
            config=config,
            output=output,
            resolved=resolved,
            bundle=bundle,
            best_path=paths.best,
        )
    )
    _write_training_summary(
        output=output,
        resolved=resolved,
        bundle=bundle,
        paths=paths,
        artifact_path=artifact_path,
        loaded_artifact=loaded_artifact,
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


def validate_bgc_policy_run(output_path: str | Path) -> EvaluationMetrics:
    """Re-evaluate a run's verified best-validation checkpoint on CPU."""

    output = Path(output_path).expanduser().resolve()
    config, smoke_epochs = _config_from_resolved(output)
    bundle = load_bgc_card_policy_dataset(config.snapshot_path)
    resolved = _resolved_configuration(config, bundle, smoke_epochs=smoke_epochs)
    if _load_json(output / "resolved-config.json") != resolved:
        raise BGCPolicyTrainingError("run source or resolved configuration differs")
    model = BGCPolicyModel(
        run_root_seed=config.run.root_seed,
        model_id=config.model.model_id,
        initialization_ordinal=config.model.initialization_ordinal,
    )
    optimizer = _optimizer(model)
    _load_checkpoint(
        output / "checkpoints" / "best-validation-candidate.pt",
        resolved=resolved,
        bundle=bundle,
        model=model,
        optimizer=optimizer,
    )
    return evaluate_bgc_policy(model, bundle.validation, device=torch.device("cpu"))


def export_bgc_policy_run(
    output_path: str | Path, destination: str | Path | None = None
) -> Path:
    """Verify and optionally copy an unaccepted candidate artifact atomically."""

    output = Path(output_path).expanduser().resolve()
    source = output / "artifacts" / "unaccepted-candidate.pt"
    loaded = load_bgc_policy_artifact(source)
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
    verified = load_bgc_policy_artifact(target)
    if verified.metadata.state_dict_digest != loaded.metadata.state_dict_digest:
        raise BGCPolicyTrainingError("copied candidate artifact digest differs")
    return target


def inspect_bgc_policy_snapshot(path: str | Path) -> dict[str, object]:
    """Verify a sealed snapshot and summarize its split and placement counts."""

    bundle = load_bgc_card_policy_dataset(path)
    return {
        "snapshot_digest": bundle.snapshot.snapshot_digest,
        "dataset_digest": bundle.dataset_digest,
        "split_digest": bundle.split_digest,
        "training_games": sum(
            game.overlay == "training" for game in bundle.snapshot.games
        ),
        "validation_games": sum(
            game.overlay == "validation" for game in bundle.snapshot.games
        ),
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


def main(argv: Sequence[str] | None = None) -> int:
    """Delegate the public entry point to the isolated command-line adapter."""

    from dracula.bgc_policy_training_cli import main as cli_main

    return cli_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = (
    "BATCH_SIZE",
    "BGCPolicyTrainingConfig",
    "BGCPolicyTrainingError",
    "BGCPolicyTrainingInterrupted",
    "distributional_policy_cross_entropy",
    "distributional_policy_statistics",
    "evaluate_bgc_policy",
    "export_bgc_policy_run",
    "inspect_bgc_policy_snapshot",
    "load_bgc_policy_training_config",
    "main",
    "train_bgc_policy",
    "validate_bgc_policy_run",
)

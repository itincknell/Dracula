"""Write the machine-readable and human-readable policy training summaries.

Optimization produces immutable epoch history and selected artifact paths. This
module renders those trusted values after training finishes; it neither chooses
a checkpoint nor mutates model state. All files use the same atomic persistence
helpers as checkpoints so an interrupted report cannot appear complete.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from dracula.policy.training.checkpoints import atomic_bytes, atomic_json
from dracula.policy.training.data import file_digest
from dracula.policy.training.state import TrainingPaths, peak_rss_bytes
from dracula.policy.training.contracts import EvaluationMetrics


def _markdown_report(
    *,
    bundle: Any,
    history: Sequence[dict[str, object]],
    best_epoch: int,
    best_loss: float,
    stopped_early: bool,
) -> str:
    """Render the concise run summary intended for human inspection."""

    lines = [
        "# Policy visit-distillation training report",
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


def _artifact_records(paths: TrainingPaths, artifact_path: Path) -> dict[str, object]:
    """Record path and byte identity for each selected run artifact."""

    return {
        "initial": {"path": str(paths.initial), "digest": file_digest(paths.initial)},
        "latest": {"path": str(paths.latest), "digest": file_digest(paths.latest)},
        "best_validation_candidate": {
            "path": str(paths.best),
            "digest": file_digest(paths.best),
        },
        "final": {"path": str(paths.final), "digest": file_digest(paths.final)},
        "unaccepted_candidate": {
            "path": str(artifact_path),
            "digest": file_digest(artifact_path),
        },
    }


def write_training_summary(
    *,
    output: Path,
    resolved: dict[str, object],
    bundle: Any,
    paths: TrainingPaths,
    artifact_path: Path,
    selected_validation: EvaluationMetrics,
    completed_epochs: int,
    best_epoch: int,
    best_validation_loss: float,
    stopped_early: bool,
    history: Sequence[dict[str, object]],
    total_seconds: float,
) -> None:
    """Atomically seal final JSON metrics and their concise Markdown view."""

    summary = {
        "configuration": resolved,
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
        "artifacts": _artifact_records(paths, artifact_path),
    }
    atomic_json(output / "metrics" / "summary.json", summary)
    markdown = _markdown_report(
        bundle=bundle,
        history=history,
        best_epoch=best_epoch,
        best_loss=best_validation_loss,
        stopped_early=stopped_early,
    )
    atomic_bytes(output / "reports" / "summary.md", markdown.encode("utf-8"))

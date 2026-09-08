"""Load and resolve the standalone policy training configuration.

The user configuration supplies one integer seed, dataset snapshot, output
directory, and device. The resolved file records the values that affect a run
without constructing a separate digest hierarchy around them.
"""

from __future__ import annotations

import tomllib
from dataclasses import asdict
from pathlib import Path
from typing import Any

from dracula.policy.training.data import load_json
from dracula.policy.model import PARAMETER_COUNT
from dracula.policy.training.contracts import (
    BATCH_SIZE,
    BETAS,
    PolicyTrainingConfig,
    PolicyTrainingError,
    DatasetSection,
    EARLY_STOP_PATIENCE,
    EPSILON,
    GRADIENT_CLIP_NORM,
    LEARNING_RATE,
    MAXIMUM_EPOCHS,
    MINIMUM_EPOCHS,
    MINIMUM_IMPROVEMENT,
    OptimizationSection,
    RunSection,
    TRAINING_RUN_FORMAT,
    WEIGHT_DECAY,
)


def _strict_table(value: object, expected: set[str], label: str) -> dict[str, object]:
    """Require exactly the documented keys at one external TOML level."""

    if not isinstance(value, dict) or set(value) != expected:
        raise PolicyTrainingError(f"{label} fields are invalid")
    return value


def load_policy_training_config(path: str | Path) -> PolicyTrainingConfig:
    """Load the small, strict TOML configuration accepted by training."""

    try:
        value = tomllib.loads(Path(path).expanduser().resolve().read_text())
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise PolicyTrainingError("training TOML could not be read") from error
    root = _strict_table(
        value,
        {"format", "run", "dataset", "optimization"},
        "training configuration",
    )
    if root["format"] != TRAINING_RUN_FORMAT:
        raise PolicyTrainingError("training configuration format is incompatible")
    run = _strict_table(root["run"], {"run_id", "seed", "output_directory"}, "run")
    dataset = _strict_table(root["dataset"], {"snapshot_path"}, "dataset")
    optimization = _strict_table(root["optimization"], {"device"}, "optimization")
    if (
        type(run["run_id"]) is not str
        or not run["run_id"]
        or type(run["output_directory"]) is not str
        or not run["output_directory"]
        or type(run["seed"]) is not int
        or run["seed"] < 0
        or type(dataset["snapshot_path"]) is not str
        or not dataset["snapshot_path"]
        or optimization["device"] not in {"cpu", "mps"}
    ):
        raise PolicyTrainingError("training configuration values are invalid")
    return PolicyTrainingConfig(
        RunSection(run["run_id"], run["seed"], run["output_directory"]),
        DatasetSection(dataset["snapshot_path"]),
        OptimizationSection(optimization["device"]),
    )


def resolved_training_configuration(
    config: PolicyTrainingConfig,
    bundle: Any,
    *,
    smoke_epochs: int | None,
) -> dict[str, object]:
    """Record all values needed to reproduce or resume one run."""

    # Snapshot identities and fixed optimizer values are copied beside the
    # user's four choices. Resume compares this whole document before loading
    # mutable checkpoint tensors.
    return {
        "format": TRAINING_RUN_FORMAT,
        "run": asdict(config.run),
        "dataset": {
            **asdict(config.dataset),
            "snapshot_digest": bundle.snapshot.snapshot_digest,
            "dataset_digest": bundle.dataset_digest,
            "split_digest": bundle.split_digest,
            "training_examples": bundle.training.example_count,
            "validation_examples": bundle.validation.example_count,
        },
        "model": {"parameter_count": PARAMETER_COUNT},
        "optimization": {
            "name": "AdamW",
            "learning_rate": LEARNING_RATE,
            "betas": list(BETAS),
            "epsilon": EPSILON,
            "weight_decay": WEIGHT_DECAY,
            "batch_size": BATCH_SIZE,
            "global_gradient_clip": GRADIENT_CLIP_NORM,
            "minimum_epochs": MINIMUM_EPOCHS,
            "maximum_epochs": MAXIMUM_EPOCHS,
            "early_stop_patience": EARLY_STOP_PATIENCE,
            "minimum_improvement": MINIMUM_IMPROVEMENT,
            "device": config.optimization.device,
            "smoke_epochs": smoke_epochs,
        },
        "objective": "representative-masked distributional cross-entropy",
    }


def config_from_resolved(output: Path) -> tuple[PolicyTrainingConfig, int | None]:
    """Recover user-selected values from an existing resolved run file."""

    value = load_json(output / "resolved-config.json")
    if not isinstance(value, dict) or value.get("format") != TRAINING_RUN_FORMAT:
        raise PolicyTrainingError("resolved training configuration is invalid")
    try:
        config = PolicyTrainingConfig(
            RunSection(**value["run"]),
            DatasetSection(snapshot_path=value["dataset"]["snapshot_path"]),
            OptimizationSection(value["optimization"]["device"]),
        )
        smoke_epochs = value["optimization"]["smoke_epochs"]
    except (KeyError, TypeError, ValueError) as error:
        raise PolicyTrainingError("resolved training configuration is invalid") from error
    if smoke_epochs is not None and type(smoke_epochs) is not int:
        raise PolicyTrainingError("resolved smoke epoch count is invalid")
    return config, smoke_epochs

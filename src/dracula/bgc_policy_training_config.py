"""Resolve immutable configuration for standalone BGC policy training.

The TOML file names only user-selected run, dataset, model, and device values.
This module expands those values with every fixed tensor, loss, optimizer,
source, masking, and symmetry identity that must match on resume.
"""

from __future__ import annotations

import tomllib
from dataclasses import asdict
from pathlib import Path
from typing import Any

from dracula.action_contract import REPRESENTATIVE_MASK_SCHEMA_VERSION
from dracula.bgc_policy import (
    BGC_POLICY_LOSS_SCHEMA_VERSION,
    BGC_POLICY_OPTIMIZER_VERSION,
)
from dracula.bgc_policy_data import json_digest, load_json
from dracula.bgc_policy_model import (
    ACTION_SCHEMA_VERSION,
    MODEL_SCHEMA_VERSION,
    OBSERVATION_SCHEMA_VERSION,
    PARAMETER_COUNT,
)
from dracula.bgc_policy_training_contracts import (
    BATCH_SIZE,
    BETAS,
    BGCPolicyTrainingConfig,
    BGCPolicyTrainingError,
    DatasetSection,
    EARLY_STOP_PATIENCE,
    EPSILON,
    GRADIENT_CLIP_NORM,
    LEARNING_RATE,
    MAXIMUM_EPOCHS,
    MINIMUM_EPOCHS,
    MINIMUM_IMPROVEMENT,
    ModelSection,
    OptimizationSection,
    RESOLVED_CONFIG_FORMAT_VERSION,
    RunSection,
    TRAINING_CONFIG_FORMAT_VERSION,
    WEIGHT_DECAY,
)
from dracula.search.symmetry import DESTINATION_SYMMETRY_SCHEMA_VERSION
from dracula.source_identity import SOURCE_TREE_SCHEMA_VERSION, resolve_source_identity


def _strict_table(
    value: object, expected: set[str], label: str
) -> dict[str, object]:
    """Accept exactly one declared TOML table and no silent extra settings."""

    if not isinstance(value, dict) or set(value) != expected:
        raise BGCPolicyTrainingError(f"{label} fields are invalid")
    return value


def load_bgc_policy_training_config(
    path: str | Path,
) -> BGCPolicyTrainingConfig:
    """Load a strict training configuration without accepting undeclared fields."""

    config_path = Path(path).expanduser().resolve()
    try:
        value = tomllib.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise BGCPolicyTrainingError("training TOML could not be read") from error
    root = _strict_table(
        value,
        {"format_version", "run", "dataset", "model", "optimization"},
        "training configuration",
    )
    if root["format_version"] != TRAINING_CONFIG_FORMAT_VERSION:
        raise BGCPolicyTrainingError("training configuration version is incompatible")
    run = _strict_table(
        root["run"], {"run_id", "root_seed", "output_directory"}, "run"
    )
    dataset = _strict_table(root["dataset"], {"snapshot_path"}, "dataset")
    model = _strict_table(
        root["model"], {"model_id", "initialization_ordinal"}, "model"
    )
    optimization = _strict_table(root["optimization"], {"device"}, "optimization")
    values = (
        run["run_id"],
        run["root_seed"],
        run["output_directory"],
        dataset["snapshot_path"],
        model["model_id"],
        optimization["device"],
    )
    if any(type(value) is not str or not value for value in values) or (
        type(model["initialization_ordinal"]) is not int
        or model["initialization_ordinal"] < 0
    ):
        raise BGCPolicyTrainingError("training configuration value types are invalid")
    config = BGCPolicyTrainingConfig(
        RunSection(run["run_id"], run["root_seed"], run["output_directory"]),
        DatasetSection(dataset["snapshot_path"]),
        ModelSection(model["model_id"], model["initialization_ordinal"]),
        OptimizationSection(optimization["device"]),
    )
    if config.optimization.device not in {"cpu", "mps"}:
        raise BGCPolicyTrainingError("training device must be cpu or mps")
    return config


def _model_contract(config: BGCPolicyTrainingConfig) -> dict[str, object]:
    """Describe the exact tensor and action contracts bound to a run."""

    return {
        "model_id": config.model.model_id,
        "initialization_ordinal": config.model.initialization_ordinal,
        "schema_version": MODEL_SCHEMA_VERSION,
        "parameter_count": PARAMETER_COUNT,
        "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
        "action_schema_version": ACTION_SCHEMA_VERSION,
        "destination_symmetry_schema_version": DESTINATION_SYMMETRY_SCHEMA_VERSION,
        "representative_mask_schema_version": REPRESENTATIVE_MASK_SCHEMA_VERSION,
    }


def _optimizer_contract(
    config: BGCPolicyTrainingConfig, smoke_epochs: int | None
) -> dict[str, object]:
    """Describe every optimization setting that can affect learned weights."""

    return {
        "schema_version": BGC_POLICY_OPTIMIZER_VERSION,
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
    }


def resolved_training_configuration(
    config: BGCPolicyTrainingConfig,
    bundle: Any,
    *,
    smoke_epochs: int | None,
) -> dict[str, object]:
    """Bind one run to its data, code, model, loss, and optimizer identities."""

    try:
        source = resolve_source_identity()
    except Exception as error:
        raise BGCPolicyTrainingError(
            "training source identity could not be resolved"
        ) from error
    model_contract = _model_contract(config)
    optimizer_contract = _optimizer_contract(config, smoke_epochs)
    content = {
        "format_version": RESOLVED_CONFIG_FORMAT_VERSION,
        "run": asdict(config.run),
        "dataset": {
            **asdict(config.dataset),
            "snapshot_digest": bundle.snapshot.snapshot_digest,
            "dataset_digest": bundle.dataset_digest,
            "split_digest": bundle.split_digest,
            "training_examples": bundle.training.example_count,
            "validation_examples": bundle.validation.example_count,
        },
        "model": model_contract,
        "optimizer": optimizer_contract,
        "loss": {
            "schema_version": BGC_POLICY_LOSS_SCHEMA_VERSION,
            "objective": "representative-masked distributional cross-entropy",
            "illegal_action_objective": False,
            "one_hot_selected_action_objective": False,
            "value_objective": False,
        },
        "source": {
            "training": {
                "schema_version": SOURCE_TREE_SCHEMA_VERSION,
                "revision": source.revision,
                "tree_digest": source.tree_digest,
            },
            "corpus": {
                "schema_version": SOURCE_TREE_SCHEMA_VERSION,
                "revision": bundle.snapshot.source_revision,
                "tree_digest": bundle.snapshot.source_tree_digest,
            },
        },
        "masking_digest": json_digest(
            {
                "representative": REPRESENTATIVE_MASK_SCHEMA_VERSION,
                "symmetry": DESTINATION_SYMMETRY_SCHEMA_VERSION,
                "masked_logit": "negative-infinity",
            }
        ),
        "symmetry_digest": json_digest(
            {
                "schema_version": DESTINATION_SYMMETRY_SCHEMA_VERSION,
                "implementation": "authoritative-exact-pattern-table",
            }
        ),
    }
    return {
        **content,
        "model_contract_digest": json_digest(model_contract),
        "optimizer_config_digest": json_digest(optimizer_contract),
        "resolved_config_digest": json_digest(content),
    }


def config_from_resolved(
    output: Path,
) -> tuple[BGCPolicyTrainingConfig, int | None]:
    """Recover user-selected values from an immutable resolved configuration."""

    value = load_json(output / "resolved-config.json")
    if (
        not isinstance(value, dict)
        or value.get("format_version") != RESOLVED_CONFIG_FORMAT_VERSION
    ):
        raise BGCPolicyTrainingError("resolved training configuration is invalid")
    try:
        config = BGCPolicyTrainingConfig(
            RunSection(**value["run"]),
            DatasetSection(snapshot_path=value["dataset"]["snapshot_path"]),
            ModelSection(
                model_id=value["model"]["model_id"],
                initialization_ordinal=value["model"]["initialization_ordinal"],
            ),
            OptimizationSection(value["optimizer"]["device"]),
        )
        smoke_epochs = value["optimizer"]["smoke_epochs"]
    except (KeyError, TypeError, ValueError) as error:
        raise BGCPolicyTrainingError(
            "resolved configuration cannot be resumed"
        ) from error
    return config, smoke_epochs


__all__ = (
    "config_from_resolved",
    "load_bgc_policy_training_config",
    "resolved_training_configuration",
)

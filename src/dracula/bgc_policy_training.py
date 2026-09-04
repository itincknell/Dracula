"""BGC-128 visit-target snapshots and standalone policy optimization."""

from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import json
import math
import os
import resource
import shutil
import sys
import time
import tomllib
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F

from dracula.bgc_policy import (
    BGC_POLICY_LOSS_SCHEMA_VERSION,
    BGC_POLICY_MODEL_SCHEMA_VERSION,
    BGC_POLICY_OPTIMIZER_VERSION,
    load_bgc_policy_artifact,
    save_bgc_policy_artifact,
)
from dracula.randomness import derive_seed
from dracula.action_contract import (
    ACTION_COUNT,
    DESTINATION_SYMMETRY_SCHEMA_VERSION,
    HAND_SLOT_COUNT,
    POLICY_GRID_INDICES,
    POLICY_POSITION_COUNT,
    REPRESENTATIVE_MASK_SCHEMA_VERSION,
    apply_representative_mask,
)
from dracula.bgc_policy_model import (
    ACTION_SCHEMA_VERSION,
    BGCPolicyModel,
    BGCPolicyModelError,
    OBSERVATION_SCHEMA_VERSION,
    PARAMETER_COUNT,
)
from dracula.source_identity import (
    SOURCE_TREE_SCHEMA_VERSION,
    resolve_source_identity,
)

TRAINING_CONFIG_FORMAT_VERSION = "dracula-bgc-policy-training-config-v1"
RESOLVED_CONFIG_FORMAT_VERSION = "dracula-bgc-policy-resolved-config-v1"
CHECKPOINT_FORMAT_VERSION = "dracula-bgc-policy-checkpoint-v1"
TRAINING_STATE_FORMAT_VERSION = "dracula-bgc-policy-training-state-v1"
METRICS_FORMAT_VERSION = "dracula-bgc-policy-epoch-metrics-v1"
SUMMARY_FORMAT_VERSION = "dracula-bgc-policy-summary-v1"
EPOCH_SHUFFLE_NAMESPACE = "dracula-bgc-policy-epoch-shuffle-v1"

OUTER_SIMULATION_BUDGET = 128
LEARNING_RATE = 3e-4
BETAS = (0.9, 0.999)
EPSILON = 1e-8
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 256
GRADIENT_CLIP_NORM = 1.0
MINIMUM_EPOCHS = 8
MAXIMUM_EPOCHS = 50
EARLY_STOP_PATIENCE = 5
MINIMUM_IMPROVEMENT = 1e-4
LATEST_CHECKPOINT_INTERVAL = 100

_DIGEST_LENGTH = 64
_FORBIDDEN_DATA_KEYS = frozenset(
    {
        "action_values",
        "authoritative_state",
        "critic",
        "determinization",
        "determinizations",
        "engine_seed",
        "game_seed",
        "hidden_state",
        "model_data",
        "opponent_hand",
        "outer_sampled_state",
        "policy_hidden_state",
        "ppo",
        "return",
        "returns",
        "round_return",
        "score_estimate",
        "search_tree",
        "stock",
        "stock_order",
        "tree",
        "value",
        "values",
    }
)


class BGCPolicyTrainingError(ValueError):
    """A BGC corpus, snapshot, batch, or training artifact is invalid."""


class BGCPolicyTrainingInterrupted(RuntimeError):
    """Optimization stopped at a sealed minibatch boundary."""


@dataclass(frozen=True, slots=True)
class RunSection:
    run_id: str
    root_seed: str
    output_directory: str


@dataclass(frozen=True, slots=True)
class DatasetSection:
    snapshot_path: str


@dataclass(frozen=True, slots=True)
class ModelSection:
    model_id: str
    initialization_ordinal: int


@dataclass(frozen=True, slots=True)
class OptimizationSection:
    device: str = "cpu"


@dataclass(frozen=True, slots=True)
class BGCPolicyTrainingConfig:
    run: RunSection
    dataset: DatasetSection
    model: ModelSection
    optimization: OptimizationSection

    @property
    def output_path(self) -> Path:
        return Path(self.run.output_directory).expanduser().resolve()

    @property
    def snapshot_path(self) -> Path:
        return Path(self.dataset.snapshot_path).expanduser().resolve()


@dataclass(frozen=True, slots=True)
class DistributionMetrics:
    count: int
    cross_entropy: float
    kl_divergence: float
    target_entropy: float
    model_entropy: float
    top_one: float
    top_two: float
    top_three: float


@dataclass(frozen=True, slots=True)
class EvaluationMetrics:
    total: DistributionMetrics
    placements: dict[str, DistributionMetrics]
    roles: dict[str, DistributionMetrics]
    dealer_status: dict[str, DistributionMetrics]
    mean_representative_actions: float
    inference_latency_seconds: float


@dataclass(frozen=True, slots=True)
class EpochMetrics:
    format_version: str
    epoch: int
    training: EvaluationMetrics
    validation: EvaluationMetrics
    maximum_gradient_norm: float
    epoch_seconds: float
    examples_per_second: float
    peak_rss_bytes: int


@dataclass(frozen=True, slots=True)
class TrainingResult:
    output_directory: str
    completed_epochs: int
    best_epoch: int
    best_validation_cross_entropy: float
    stopped_early: bool
    final_checkpoint: str
    candidate_checkpoint: str
    exported_artifact: str


@dataclass(slots=True)
class _MetricAccumulator:
    count: int = 0
    cross_entropy: float = 0.0
    kl_divergence: float = 0.0
    target_entropy: float = 0.0
    model_entropy: float = 0.0
    top_one: int = 0
    top_two: int = 0
    top_three: int = 0

    def add(
        self,
        values: tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor],
        indexes: Tensor | None = None,
    ) -> None:
        cross_entropy, kl, target_entropy, model_entropy, top1, top2, top3 = values
        if indexes is not None:
            cross_entropy = cross_entropy.index_select(0, indexes)
            kl = kl.index_select(0, indexes)
            target_entropy = target_entropy.index_select(0, indexes)
            model_entropy = model_entropy.index_select(0, indexes)
            top1 = top1.index_select(0, indexes)
            top2 = top2.index_select(0, indexes)
            top3 = top3.index_select(0, indexes)
        self.count += int(cross_entropy.numel())
        self.cross_entropy += float(cross_entropy.sum().item())
        self.kl_divergence += float(kl.sum().item())
        self.target_entropy += float(target_entropy.sum().item())
        self.model_entropy += float(model_entropy.sum().item())
        self.top_one += int(top1.sum().item())
        self.top_two += int(top2.sum().item())
        self.top_three += int(top3.sum().item())

    def finish(self) -> DistributionMetrics:
        if self.count == 0:
            return DistributionMetrics(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        return DistributionMetrics(
            self.count,
            self.cross_entropy / self.count,
            self.kl_divergence / self.count,
            self.target_entropy / self.count,
            self.model_entropy / self.count,
            self.top_one / self.count,
            self.top_two / self.count,
            self.top_three / self.count,
        )


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise BGCPolicyTrainingError("artifact is not canonical JSON") from error


def _json_digest(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _require_digest(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != _DIGEST_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise BGCPolicyTrainingError(
            f"{label} must be a lowercase SHA-256 digest"
        )
    return value


def _load_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BGCPolicyTrainingError(f"JSON artifact could not be read: {path}") from error


def _source_document(path: Path) -> dict[str, object]:
    value = _load_json(path)
    if (
        not isinstance(value, dict)
        or set(value) != {"content", "content_digest"}
        or not isinstance(value["content"], dict)
        or value["content_digest"] != _json_digest(value["content"])
    ):
        raise BGCPolicyTrainingError(f"source artifact digest differs: {path}")
    return value


def _envelope(format_version: str, content: object) -> dict[str, object]:
    content_digest = _json_digest(content)
    unsigned = {
        "format_version": format_version,
        "content": content,
        "content_digest": content_digest,
    }
    return {**unsigned, "file_digest": _json_digest(unsigned)}


def _validate_envelope(value: object, format_version: str) -> dict[str, object]:
    if (
        not isinstance(value, dict)
        or set(value) != {"format_version", "content", "content_digest", "file_digest"}
        or value["format_version"] != format_version
        or value["content_digest"] != _json_digest(value["content"])
        or value["file_digest"] != _json_digest(
            {
                "format_version": value["format_version"],
                "content": value["content"],
                "content_digest": value["content_digest"],
            }
        )
    ):
        raise BGCPolicyTrainingError(f"{format_version} envelope digest differs")
    return value


def _atomic_bytes(path: Path, value: bytes) -> None:
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


def _atomic_json(path: Path, value: object) -> None:
    _atomic_bytes(path, _canonical_json(value) + b"\n")


def _atomic_torch(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        torch.save(value, temporary)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _assert_no_forbidden_keys(value: object) -> None:
    if isinstance(value, Mapping):
        forbidden = _FORBIDDEN_DATA_KEYS.intersection(value)
        if forbidden:
            raise BGCPolicyTrainingError(
                "training data contains forbidden fields: "
                + ", ".join(sorted(forbidden))
            )
        for nested in value.values():
            _assert_no_forbidden_keys(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            _assert_no_forbidden_keys(nested)


def _decode_packed(value: object, *, byte_count: int, bit_count: int, label: str) -> bytes:
    if not isinstance(value, str):
        raise BGCPolicyTrainingError(f"{label} must be base64 text")
    try:
        decoded = base64.b64decode(value, validate=True)
    except (ValueError, binascii.Error) as error:
        raise BGCPolicyTrainingError(f"{label} is not canonical base64") from error
    if len(decoded) != byte_count:
        raise BGCPolicyTrainingError(f"{label} has the wrong byte count")
    unused = byte_count * 8 - bit_count
    if unused and decoded[-1] & ((1 << unused) - 1):
        raise BGCPolicyTrainingError(f"{label} has nonzero padding bits")
    return decoded


def _unpack_bytes(value: bytes, bit_count: int) -> tuple[bool, ...]:
    return tuple(
        bool(value[index // 8] & (1 << (7 - index % 8)))
        for index in range(bit_count)
    )


def load_bgc_card_policy_dataset(snapshot_path: str | Path):
    """Load the sealed 659-bit corpus used by the selected standalone policy."""

    from dracula.bgc_policy_migration import load_migrated_policy_dataset

    return load_migrated_policy_dataset(snapshot_path)


def distributional_policy_statistics(
    logits: Tensor,
    representative_masks: Tensor,
    targets: Tensor,
    selected_actions: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    if (
        logits.dtype is not torch.float32
        or logits.ndim != 3
        or logits.shape[1:] != (HAND_SLOT_COUNT, POLICY_POSITION_COUNT)
        or representative_masks.dtype is not torch.bool
        or representative_masks.shape != logits.shape
        or targets.dtype is not torch.float32
        or targets.shape != (logits.shape[0], ACTION_COUNT)
        or selected_actions.dtype is not torch.long
        or selected_actions.shape != (logits.shape[0],)
        or not bool(torch.isfinite(logits).all().item())
        or not bool(torch.isfinite(targets).all().item())
    ):
        raise BGCPolicyTrainingError("distributional policy batch is malformed")
    masks = representative_masks.flatten(start_dim=1)
    if (
        bool((targets < 0).any().item())
        or not torch.allclose(
            targets.sum(dim=1),
            torch.ones(targets.shape[0], device=targets.device),
            atol=1e-7,
            rtol=0.0,
        )
        or bool((targets.masked_select(~masks) != 0).any().item())
        or not bool(masks.gather(1, selected_actions[:, None]).all().item())
    ):
        raise BGCPolicyTrainingError("visit target or audit action is invalid")
    flattened = apply_representative_mask(
        logits, representative_masks
    ).flatten(start_dim=1)
    log_probabilities = F.log_softmax(flattened, dim=1)
    probabilities = log_probabilities.exp()
    safe_log_probabilities = torch.where(
        masks, log_probabilities, torch.zeros_like(log_probabilities)
    )
    cross_entropy = -(targets * safe_log_probabilities).sum(dim=1)
    target_log = torch.where(
        targets > 0, targets.log(), torch.zeros_like(targets)
    )
    target_entropy = -(targets * target_log).sum(dim=1)
    kl_divergence = cross_entropy - target_entropy
    model_entropy = -(
        probabilities * safe_log_probabilities
    ).sum(dim=1)
    ranks = torch.argsort(flattened, dim=1, descending=True, stable=True)
    top_one = ranks[:, :1].eq(selected_actions[:, None]).any(dim=1)
    top_two = ranks[:, :2].eq(selected_actions[:, None]).any(dim=1)
    top_three = ranks[:, :3].eq(selected_actions[:, None]).any(dim=1)
    if not all(
        bool(torch.isfinite(value).all().item())
        for value in (cross_entropy, kl_divergence, target_entropy, model_entropy)
    ):
        raise BGCPolicyTrainingError("distributional policy metrics are non-finite")
    return (
        cross_entropy,
        kl_divergence,
        target_entropy,
        model_entropy,
        top_one,
        top_two,
        top_three,
    )


def distributional_policy_cross_entropy(
    logits: Tensor, representative_masks: Tensor, targets: Tensor
) -> Tensor:
    selected = targets.argmax(dim=1)
    return distributional_policy_statistics(
        logits, representative_masks, targets, selected
    )[0].mean()


def _strict_table(
    value: object, expected: set[str], label: str
) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != expected:
        raise BGCPolicyTrainingError(f"{label} fields are invalid")
    return value


def load_bgc_policy_training_config(
    path: str | Path,
) -> BGCPolicyTrainingConfig:
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


def _training_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    raise BGCPolicyTrainingError(f"training device is unavailable: {name}")


def _optimizer(model: BGCPolicyModel) -> torch.optim.AdamW:
    return torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        betas=BETAS,
        eps=EPSILON,
        weight_decay=WEIGHT_DECAY,
    )


def _resolved_configuration(
    config: BGCPolicyTrainingConfig,
    bundle: Any,
    *,
    smoke_epochs: int | None,
) -> dict[str, object]:
    try:
        source = resolve_source_identity()
    except Exception as error:
        raise BGCPolicyTrainingError("training source identity could not be resolved") from error
    model_contract = {
        "model_id": config.model.model_id,
        "initialization_ordinal": config.model.initialization_ordinal,
        "schema_version": BGC_POLICY_MODEL_SCHEMA_VERSION,
        "parameter_count": PARAMETER_COUNT,
        "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
        "action_schema_version": ACTION_SCHEMA_VERSION,
        "destination_symmetry_schema_version": DESTINATION_SYMMETRY_SCHEMA_VERSION,
        "representative_mask_schema_version": REPRESENTATIVE_MASK_SCHEMA_VERSION,
    }
    optimizer_contract = {
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
        "masking_digest": _json_digest(
            {
                "representative": REPRESENTATIVE_MASK_SCHEMA_VERSION,
                "symmetry": DESTINATION_SYMMETRY_SCHEMA_VERSION,
                "masked_logit": "negative-infinity",
            }
        ),
        "symmetry_digest": _json_digest(
            {
                "schema_version": DESTINATION_SYMMETRY_SCHEMA_VERSION,
                "implementation": "authoritative-exact-pattern-table",
            }
        ),
    }
    return {
        **content,
        "model_contract_digest": _json_digest(model_contract),
        "optimizer_config_digest": _json_digest(optimizer_contract),
        "resolved_config_digest": _json_digest(content),
    }


def _seal_resolved_config(output: Path, resolved: dict[str, object]) -> None:
    path = output / "resolved-config.json"
    if path.exists():
        if _load_json(path) != resolved:
            raise BGCPolicyTrainingError("resolved training configuration differs")
    else:
        _atomic_json(path, resolved)


def _state_dict_digest(state_dict: Mapping[str, Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state_dict.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(value.dtype).encode("ascii") + b"\0")
        digest.update(",".join(map(str, value.shape)).encode("ascii") + b"\0")
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _nested_digest(value: object) -> str:
    digest = hashlib.sha256()

    def update(item: object) -> None:
        if isinstance(item, Tensor):
            tensor = item.detach().cpu().contiguous()
            digest.update(b"tensor\0")
            digest.update(str(tensor.dtype).encode("ascii") + b"\0")
            digest.update(",".join(map(str, tensor.shape)).encode("ascii") + b"\0")
            digest.update(tensor.numpy().tobytes(order="C"))
        elif isinstance(item, Mapping):
            digest.update(b"mapping\0")
            for key in sorted(item, key=str):
                update(str(key))
                update(item[key])
        elif isinstance(item, (tuple, list)):
            digest.update(b"sequence\0")
            for nested in item:
                update(nested)
        else:
            digest.update(_canonical_json(item))
            digest.update(b"\0")

    update(value)
    return digest.hexdigest()


def _checkpoint_payload(
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
    model_state = {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }
    optimizer_state = optimizer.state_dict()
    return {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "kind": kind,
        "resolved_config_digest": resolved["resolved_config_digest"],
        "snapshot_digest": bundle.snapshot.snapshot_digest,
        "dataset_digest": bundle.dataset_digest,
        "split_digest": bundle.split_digest,
        "model_contract_digest": resolved["model_contract_digest"],
        "optimizer_config_digest": resolved["optimizer_config_digest"],
        "source_revision": resolved["source"]["training"]["revision"],  # type: ignore[index]
        "source_tree_digest": resolved["source"]["training"]["tree_digest"],  # type: ignore[index]
        "state_dict_digest": _state_dict_digest(model_state),
        "optimizer_state_digest": _nested_digest(optimizer_state),
        "training_state_schema_version": TRAINING_STATE_FORMAT_VERSION,
        "epoch": epoch,
        "next_batch": next_batch,
        "completed_epochs": completed_epochs,
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
        "stale_epochs": stale_epochs,
        "maximum_gradient_norm": maximum_gradient_norm,
        "history": list(history),
        "model_state_dict": model_state,
        "optimizer_state_dict": optimizer_state,
    }


def _load_checkpoint(
    path: Path,
    *,
    resolved: dict[str, object],
    bundle: Any,
    model: BGCPolicyModel,
    optimizer: torch.optim.AdamW,
) -> dict[str, object]:
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, ValueError) as error:
        raise BGCPolicyTrainingError("training checkpoint could not be loaded") from error
    required = {
        "format_version",
        "kind",
        "resolved_config_digest",
        "snapshot_digest",
        "dataset_digest",
        "split_digest",
        "model_contract_digest",
        "optimizer_config_digest",
        "source_revision",
        "source_tree_digest",
        "state_dict_digest",
        "optimizer_state_digest",
        "training_state_schema_version",
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
    if (
        not isinstance(value, dict)
        or set(value) != required
        or value["format_version"] != CHECKPOINT_FORMAT_VERSION
        or value["resolved_config_digest"] != resolved["resolved_config_digest"]
        or value["snapshot_digest"] != bundle.snapshot.snapshot_digest
        or value["dataset_digest"] != bundle.dataset_digest
        or value["split_digest"] != bundle.split_digest
        or value["model_contract_digest"] != resolved["model_contract_digest"]
        or value["optimizer_config_digest"] != resolved["optimizer_config_digest"]
        or value["source_revision"] != resolved["source"]["training"]["revision"]  # type: ignore[index]
        or value["source_tree_digest"] != resolved["source"]["training"]["tree_digest"]  # type: ignore[index]
        or value["training_state_schema_version"] != TRAINING_STATE_FORMAT_VERSION
        or not isinstance(value["model_state_dict"], dict)
        or not isinstance(value["optimizer_state_dict"], dict)
        or value["state_dict_digest"]
        != _state_dict_digest(value["model_state_dict"])
        or value["optimizer_state_digest"]
        != _nested_digest(value["optimizer_state_dict"])
    ):
        raise BGCPolicyTrainingError("training checkpoint identity differs")
    try:
        model.load_state_dict(value["model_state_dict"], strict=True)
        optimizer.load_state_dict(value["optimizer_state_dict"])
    except (RuntimeError, ValueError) as error:
        raise BGCPolicyTrainingError("training checkpoint state is incompatible") from error
    return value


def _epoch_indexes(
    dataset: Any,
    *,
    root_seed: str,
    snapshot_digest: str,
    epoch: int,
) -> Tensor:
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
    return tuple(
        indexes[start : start + BATCH_SIZE]
        for start in range(0, len(indexes), BATCH_SIZE)
    )


def _finite_parameters(model: BGCPolicyModel) -> bool:
    return all(
        parameter.dtype is torch.float32
        and bool(torch.isfinite(parameter).all().item())
        for parameter in model.parameters()
    )


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _metric_groups(
    accumulator: dict[str, _MetricAccumulator],
    labels: Tensor,
    values: tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor],
) -> None:
    for label in torch.unique(labels).tolist():
        indexes = torch.nonzero(labels == label, as_tuple=False).flatten()
        accumulator.setdefault(str(int(label)), _MetricAccumulator()).add(
            values, indexes
        )


def evaluate_bgc_policy(
    model: BGCPolicyModel,
    dataset: Any,
    *,
    device: torch.device,
) -> EvaluationMetrics:
    model = model.to(device)
    model.eval()
    total = _MetricAccumulator()
    placements: dict[str, _MetricAccumulator] = {}
    roles: dict[str, _MetricAccumulator] = {}
    dealer_status: dict[str, _MetricAccumulator] = {}
    representative_actions = 0
    inference_seconds = 0.0
    with torch.no_grad():
        for indexes in _batches(torch.arange(dataset.example_count)):
            observations, _, masks, targets, selected = dataset.decoded_batch(
                indexes, device=device
            )
            started = time.perf_counter()
            logits = model(observations)
            if device.type == "mps":
                torch.mps.synchronize()
            inference_seconds += time.perf_counter() - started
            values = distributional_policy_statistics(
                logits, masks, targets, selected
            )
            cpu_values = tuple(value.detach().cpu() for value in values)
            total.add(cpu_values)
            representative_actions += int(masks.sum().item())
            placement_labels = dataset.placements.index_select(0, indexes).long()
            player_labels = dataset.players.index_select(0, indexes).long()
            dealer_labels = dataset.dealers.index_select(0, indexes).long()
            _metric_groups(placements, placement_labels, cpu_values)
            _metric_groups(roles, player_labels, cpu_values)
            status_labels = player_labels.eq(dealer_labels).long()
            _metric_groups(dealer_status, status_labels, cpu_values)
    return EvaluationMetrics(
        total.finish(),
        {key: value.finish() for key, value in sorted(placements.items())},
        {
            ("queen" if key == "0" else "king"): value.finish()
            for key, value in sorted(roles.items())
        },
        {
            ("dealer" if key == "1" else "non-dealer"): value.finish()
            for key, value in sorted(dealer_status.items())
        },
        representative_actions / max(dataset.example_count, 1),
        inference_seconds / max(dataset.example_count, 1),
    )


def _metrics_dict(metrics: EpochMetrics) -> dict[str, object]:
    return asdict(metrics)


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


def train_bgc_policy(
    config: BGCPolicyTrainingConfig,
    *,
    resume: bool = False,
    interrupt_after_batches: int | None = None,
    smoke_epochs: int | None = None,
) -> TrainingResult:
    if smoke_epochs is not None and (
        type(smoke_epochs) is not int or not 1 <= smoke_epochs <= 8
    ):
        raise BGCPolicyTrainingError("smoke epochs must be between one and eight")
    if interrupt_after_batches is not None and (
        type(interrupt_after_batches) is not int or interrupt_after_batches < 1
    ):
        raise BGCPolicyTrainingError("interrupt batch count must be positive")
    bundle = load_bgc_card_policy_dataset(config.snapshot_path)
    resolved = _resolved_configuration(config, bundle, smoke_epochs=smoke_epochs)
    output = config.output_path
    output.mkdir(parents=True, exist_ok=True)
    _seal_resolved_config(output, resolved)
    for directory in ("checkpoints", "metrics", "artifacts", "reports"):
        (output / directory).mkdir(exist_ok=True)
    device = _training_device(config.optimization.device)
    model = BGCPolicyModel(
        run_root_seed=config.run.root_seed,
        model_id=config.model.model_id,
        initialization_ordinal=config.model.initialization_ordinal,
    ).to(device)
    if sum(parameter.numel() for parameter in model.parameters()) != PARAMETER_COUNT:
        raise BGCPolicyTrainingError("policy model parameter count differs")
    optimizer = _optimizer(model)
    initial_path = output / "checkpoints" / "initial.pt"
    latest_path = output / "checkpoints" / "latest.pt"
    best_path = output / "checkpoints" / "best-validation-candidate.pt"
    final_path = output / "checkpoints" / "final.pt"
    epoch = 0
    next_batch = 0
    completed_epochs = 0
    best_epoch = 0
    best_validation_loss = math.inf
    stale_epochs = 0
    maximum_gradient_norm = 0.0
    history: list[dict[str, object]] = []
    if resume:
        checkpoint = _load_checkpoint(
            latest_path,
            resolved=resolved,
            bundle=bundle,
            model=model,
            optimizer=optimizer,
        )
        epoch = checkpoint["epoch"]  # type: ignore[assignment]
        next_batch = checkpoint["next_batch"]  # type: ignore[assignment]
        completed_epochs = checkpoint["completed_epochs"]  # type: ignore[assignment]
        best_epoch = checkpoint["best_epoch"]  # type: ignore[assignment]
        best_validation_loss = checkpoint["best_validation_loss"]  # type: ignore[assignment]
        stale_epochs = checkpoint["stale_epochs"]  # type: ignore[assignment]
        maximum_gradient_norm = checkpoint["maximum_gradient_norm"]  # type: ignore[assignment]
        history = list(checkpoint["history"])  # type: ignore[arg-type]
    else:
        if any(path.exists() for path in (initial_path, latest_path, best_path, final_path)):
            raise BGCPolicyTrainingError("new run already contains checkpoints")
        initial = _checkpoint_payload(
            kind="initial",
            resolved=resolved,
            bundle=bundle,
            model=model,
            optimizer=optimizer,
            epoch=0,
            next_batch=0,
            completed_epochs=0,
            best_epoch=0,
            best_validation_loss=math.inf,
            stale_epochs=0,
            maximum_gradient_norm=0.0,
            history=(),
        )
        _atomic_torch(initial_path, initial)
        _atomic_torch(latest_path, {**initial, "kind": "latest"})
    maximum_epochs = smoke_epochs if smoke_epochs is not None else MAXIMUM_EPOCHS
    minimum_epochs = smoke_epochs if smoke_epochs is not None else MINIMUM_EPOCHS
    processed_batches = 0
    stopped_early = False
    run_started = time.perf_counter()
    while epoch < maximum_epochs:
        epoch_started = time.perf_counter()
        order = _epoch_indexes(
            bundle.training,
            root_seed=config.run.root_seed,
            snapshot_digest=bundle.snapshot.snapshot_digest,
            epoch=epoch,
        )
        batches = _batches(order)
        if not 0 <= next_batch <= len(batches):
            raise BGCPolicyTrainingError("checkpoint minibatch position is invalid")
        model.train()
        for batch_index in range(next_batch, len(batches)):
            observations, _, masks, targets, selected = bundle.training.decoded_batch(
                batches[batch_index], device=device
            )
            del selected
            optimizer.zero_grad(set_to_none=True)
            logits = model(observations)
            loss = distributional_policy_cross_entropy(logits, masks, targets)
            if not bool(torch.isfinite(loss).item()):
                raise BGCPolicyTrainingError("training loss is non-finite")
            loss.backward()
            gradients = [
                parameter.grad
                for parameter in model.parameters()
                if parameter.grad is not None
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
            maximum_gradient_norm = max(
                maximum_gradient_norm, float(gradient_norm.item())
            )
            optimizer.step()
            if not _finite_parameters(model):
                raise BGCPolicyTrainingError("optimizer produced non-finite parameters")
            next_batch = batch_index + 1
            processed_batches += 1
            if (
                next_batch % LATEST_CHECKPOINT_INTERVAL == 0
                or interrupt_after_batches is not None
                and processed_batches >= interrupt_after_batches
            ):
                _atomic_torch(
                    latest_path,
                    _checkpoint_payload(
                        kind="latest",
                        resolved=resolved,
                        bundle=bundle,
                        model=model,
                        optimizer=optimizer,
                        epoch=epoch,
                        next_batch=next_batch,
                        completed_epochs=completed_epochs,
                        best_epoch=best_epoch,
                        best_validation_loss=best_validation_loss,
                        stale_epochs=stale_epochs,
                        maximum_gradient_norm=maximum_gradient_norm,
                        history=history,
                    ),
                )
            if (
                interrupt_after_batches is not None
                and processed_batches >= interrupt_after_batches
            ):
                raise BGCPolicyTrainingInterrupted(
                    "training interrupted at a sealed minibatch boundary"
                )
        training_metrics = evaluate_bgc_policy(
            model, bundle.training, device=device
        )
        validation_metrics = evaluate_bgc_policy(
            model, bundle.validation, device=device
        )
        elapsed = time.perf_counter() - epoch_started
        completed_epochs = epoch + 1
        epoch_metrics = EpochMetrics(
            METRICS_FORMAT_VERSION,
            completed_epochs,
            training_metrics,
            validation_metrics,
            maximum_gradient_norm,
            elapsed,
            bundle.training.example_count / max(elapsed, 1e-12),
            _peak_rss_bytes(),
        )
        history.append(_metrics_dict(epoch_metrics))
        _atomic_json(
            output / "metrics" / f"{completed_epochs:04d}.json",
            history[-1],
        )
        validation_loss = validation_metrics.total.cross_entropy
        improved = validation_loss < best_validation_loss - MINIMUM_IMPROVEMENT
        if improved:
            best_validation_loss = validation_loss
            best_epoch = completed_epochs
            stale_epochs = 0
        else:
            stale_epochs += 1
        epoch += 1
        next_batch = 0
        maximum_gradient_norm = 0.0
        latest = _checkpoint_payload(
            kind="latest",
            resolved=resolved,
            bundle=bundle,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            next_batch=0,
            completed_epochs=completed_epochs,
            best_epoch=best_epoch,
            best_validation_loss=best_validation_loss,
            stale_epochs=stale_epochs,
            maximum_gradient_norm=0.0,
            history=history,
        )
        _atomic_torch(latest_path, latest)
        _atomic_json(
            output / "training-state.json",
            {
                "format_version": TRAINING_STATE_FORMAT_VERSION,
                "epoch": epoch,
                "next_batch": 0,
                "best_epoch": best_epoch,
                "best_validation_loss": best_validation_loss,
                "latest_checkpoint_digest": _file_digest(latest_path),
            },
        )
        if improved:
            _atomic_torch(
                best_path, {**latest, "kind": "best-validation-candidate"}
            )
        if (
            completed_epochs >= minimum_epochs
            and stale_epochs >= EARLY_STOP_PATIENCE
        ):
            stopped_early = True
            break
    final = _checkpoint_payload(
        kind="final",
        resolved=resolved,
        bundle=bundle,
        model=model,
        optimizer=optimizer,
        epoch=epoch,
        next_batch=0,
        completed_epochs=completed_epochs,
        best_epoch=best_epoch,
        best_validation_loss=best_validation_loss,
        stale_epochs=stale_epochs,
        maximum_gradient_norm=0.0,
        history=history,
    )
    _atomic_torch(final_path, final)
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
        source_revision=resolved["source"]["training"]["revision"],  # type: ignore[index]
        source_tree_digest=resolved["source"]["training"]["tree_digest"],  # type: ignore[index]
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
            "total_seconds": time.perf_counter() - run_started,
            "peak_rss_bytes": _peak_rss_bytes(),
        },
        "artifacts": {
            "initial": {"path": str(initial_path), "digest": _file_digest(initial_path)},
            "latest": {"path": str(latest_path), "digest": _file_digest(latest_path)},
            "best_validation_candidate": {
                "path": str(best_path),
                "digest": _file_digest(best_path),
            },
            "final": {"path": str(final_path), "digest": _file_digest(final_path)},
            "unaccepted_candidate": {
                "path": str(artifact_path),
                "digest": _file_digest(artifact_path),
                "state_dict_digest": loaded.metadata.state_dict_digest,
                "candidate_status": loaded.metadata.candidate_status,
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
    return TrainingResult(
        str(output),
        completed_epochs,
        best_epoch,
        best_validation_loss,
        stopped_early,
        str(final_path),
        str(best_path),
        str(artifact_path),
    )


def _config_from_resolved(
    output: Path,
) -> tuple[BGCPolicyTrainingConfig, int | None]:
    value = _load_json(output / "resolved-config.json")
    if not isinstance(value, dict) or value.get("format_version") != RESOLVED_CONFIG_FORMAT_VERSION:
        raise BGCPolicyTrainingError("resolved training configuration is invalid")
    try:
        config = BGCPolicyTrainingConfig(
            RunSection(**value["run"]),
            DatasetSection(
                snapshot_path=value["dataset"]["snapshot_path"]
            ),
            ModelSection(
                model_id=value["model"]["model_id"],
                initialization_ordinal=value["model"]["initialization_ordinal"],
            ),
            OptimizationSection(value["optimizer"]["device"]),
        )
        smoke_epochs = value["optimizer"]["smoke_epochs"]
    except (KeyError, TypeError, ValueError) as error:
        raise BGCPolicyTrainingError("resolved configuration cannot be resumed") from error
    return config, smoke_epochs


def validate_bgc_policy_run(output_path: str | Path) -> EvaluationMetrics:
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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="BGC-128 visit-distribution policy distillation"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser(
        "inspect", help="verify and inspect a sealed BGC snapshot"
    )
    inspect.add_argument("--snapshot", required=True)
    smoke = commands.add_parser(
        "smoke", help="run bounded synthetic-fixture optimization"
    )
    smoke.add_argument("--config", required=True)
    smoke.add_argument("--epochs", type=int, default=2)
    train = commands.add_parser("train", help="start real snapshot optimization")
    train.add_argument("--config", required=True)
    resume = commands.add_parser("resume", help="resume the latest checkpoint")
    resume.add_argument("--run", required=True)
    validate = commands.add_parser(
        "validate", help="evaluate the best unaccepted candidate"
    )
    validate.add_argument("--run", required=True)
    export = commands.add_parser(
        "export", help="verify or copy the unaccepted candidate artifact"
    )
    export.add_argument("--run", required=True)
    export.add_argument("--output")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        if arguments.command == "inspect":
            result = inspect_bgc_policy_snapshot(arguments.snapshot)
        elif arguments.command in {"train", "smoke"}:
            config = load_bgc_policy_training_config(arguments.config)
            result = asdict(
                train_bgc_policy(
                    config,
                    smoke_epochs=(arguments.epochs if arguments.command == "smoke" else None),
                )
            )
        elif arguments.command == "resume":
            output = Path(arguments.run).expanduser().resolve()
            config, smoke_epochs = _config_from_resolved(output)
            result = asdict(
                train_bgc_policy(config, resume=True, smoke_epochs=smoke_epochs)
            )
        elif arguments.command == "validate":
            result = asdict(validate_bgc_policy_run(arguments.run))
        else:
            result = {
                "artifact": str(
                    export_bgc_policy_run(arguments.run, arguments.output)
                )
            }
    except BGCPolicyTrainingInterrupted as error:
        print(json.dumps({"status": "interrupted", "message": str(error)}))
        return 75
    except (BGCPolicyTrainingError, BGCPolicyModelError) as error:
        print(str(error), file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, default=str))
    return 0


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

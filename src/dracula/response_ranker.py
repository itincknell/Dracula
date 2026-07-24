"""Supervised response ranking over sealed Teacher v2 evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import resource
import statistics
import sys
import time
import tomllib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import Tensor
from torch.nn import functional as F

from dracula.cards import CARD_IDS
from dracula.policy_value import (
    ACTION_COUNT,
    ACTION_SCHEMA_VERSION,
    MODEL_SCHEMA_VERSION,
    OBSERVATION_SCHEMA_VERSION,
    PARAMETER_COUNT,
    POLICY_GRID_INDICES,
    PolicyValueModel,
    PolicyValueOptimizationConfig,
)
from dracula.randomness import derive_pytorch_seed
from dracula.response_dataset import (
    ResponseDatasetError,
    inspect_response_dataset,
    load_response_manifest,
    load_response_shard,
)
from dracula.response_distillation import (
    RESPONSE_ACTION_GROUP_SCHEMA_VERSION,
    RESPONSE_EXAMPLE_SCHEMA_VERSION,
    RESPONSE_GROUP_LOGIT_PROFILE,
    RESPONSE_PAIR_WEIGHT_PROFILE,
    RESPONSE_RANKING_SCHEMA_VERSION,
)
from dracula.teacher import FixtureSplit

RESPONSE_RANKER_CONFIG_VERSION = "dracula-response-ranker-config-v1"
RESPONSE_RANKER_CHECKPOINT_VERSION = "dracula-response-ranker-checkpoint-v1"
RESPONSE_RANKER_ARTIFACT_VERSION = "dracula-response-ranker-artifact-v1"
RESPONSE_RANKER_METRICS_VERSION = "dracula-response-ranker-metrics-v1"
RESPONSE_RANKER_STATE_VERSION = "dracula-response-ranker-state-v1"
RESPONSE_RANKER_OPTIMIZER_VERSION = "dracula-response-ranker-adamw-v1"
RESPONSE_RANKER_VALUE_HEAD_USAGE = "frozen-unused-v1"
RESPONSE_RANKER_SHUFFLE_NAMESPACE = "dracula-response-ranker-shuffle-v1"
RESPONSE_RANKER_TRAINABLE_PARAMETER_COUNT = 331_657

_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_CHECKPOINT_NAMES = {
    "iteration-start": "iteration-start.pt",
    "epoch-latest": "epoch-latest.pt",
    "best-validation": "best-validation.pt",
    "final": "final.pt",
}
_FORBIDDEN_ARTIFACT_KEYS = frozenset(
    (
        "authoritative_state",
        "determinizations",
        "engine_seed",
        "opponent_hand",
        "search_tree",
        "stock_order",
    )
)


class ResponseRankerError(ValueError):
    """A ranker configuration, dataset, checkpoint, or artifact is invalid."""


class ResponseRankerInterrupted(RuntimeError):
    """Optimization stopped before the active epoch was committed."""


@dataclass(frozen=True, slots=True)
class RankerRunConfig:
    run_id: str
    root_seed: str
    output_directory: str
    source_revision: str


@dataclass(frozen=True, slots=True)
class RankerDatasetConfig:
    response_directory: str


@dataclass(frozen=True, slots=True)
class RankerModelConfig:
    model_id: str
    initialization_ordinal: int


@dataclass(frozen=True, slots=True)
class RankerOptimizationConfig:
    device: str = "cpu"
    batch_size: int = 256
    learning_rate: float = 3e-4
    betas: tuple[float, float] = (0.9, 0.999)
    epsilon: float = 1e-8
    weight_decay: float = 1e-4
    gradient_norm: float = 1.0
    minimum_epochs: int = 8
    maximum_epochs: int = 50
    early_stop_patience: int = 5
    minimum_improvement: float = 1e-4

    @property
    def optimizer(self) -> PolicyValueOptimizationConfig:
        return PolicyValueOptimizationConfig(
            learning_rate=self.learning_rate,
            beta1=self.betas[0],
            beta2=self.betas[1],
            epsilon=self.epsilon,
            weight_decay=self.weight_decay,
        )


@dataclass(frozen=True, slots=True)
class ResponseRankerConfig:
    run: RankerRunConfig
    dataset: RankerDatasetConfig
    model: RankerModelConfig
    optimization: RankerOptimizationConfig

    @property
    def output_path(self) -> Path:
        return Path(self.run.output_directory).expanduser().resolve()

    @property
    def response_path(self) -> Path:
        return Path(self.dataset.response_directory).expanduser().resolve()


@dataclass(frozen=True, slots=True)
class ResponseRankingDataset:
    split: FixtureSplit
    observations: Tensor
    legal_masks: Tensor
    group_member_weights: Tensor
    group_values: Tensor
    group_valid: Tensor
    group_representatives: Tensor
    selected_representatives: Tensor
    placement_numbers: Tensor
    actor_roles: tuple[str, ...]
    actor_is_dealer: Tensor
    cache_identity_digests: tuple[str, ...]
    fixture_ids: tuple[str, ...]
    manifest_digest: str

    @property
    def example_count(self) -> int:
        return int(self.observations.shape[0])


@dataclass(frozen=True, slots=True)
class ResponseRankingBundle:
    training: ResponseRankingDataset
    validation: ResponseRankingDataset
    dataset_digest: str


@dataclass(frozen=True, slots=True)
class RankerEpochMetrics:
    epoch: int
    training_ranking_loss: float
    validation_ranking_loss: float
    validation_top_group_accuracy: float
    validation_mean_regret: float
    maximum_gradient_norm: float
    epoch_seconds: float


@dataclass(frozen=True, slots=True)
class ResponseRankerTrainingResult:
    output_directory: str
    dataset_digest: str
    completed_epochs: int
    best_epoch: int
    best_validation_loss: float
    stopped_early: bool
    final_checkpoint: str
    training_seconds: float
    peak_rss_bytes: int


@dataclass(frozen=True, slots=True)
class ResponseRankerArtifactMetadata:
    model_schema_version: str
    observation_schema_version: str
    action_schema_version: str
    response_example_schema_version: str
    action_group_schema_version: str
    ranking_schema_version: str
    group_logit_profile: str
    pair_weight_profile: str
    optimizer_compatibility_version: str
    value_head_usage: str
    parameter_count: int
    trainable_parameter_count: int
    card_ids: tuple[str, ...]
    policy_grid_indices: tuple[int, ...]
    run_root_seed: str
    model_id: str
    initialization_ordinal: int
    initialization_seed_digest: str
    source_revision: str
    dataset_digest: str
    training_manifest_digest: str
    validation_manifest_digest: str
    resolved_config_digest: str
    best_epoch: int
    best_validation_loss: float
    state_dict_digest: str


@dataclass(frozen=True, slots=True)
class LoadedResponseRankerArtifact:
    model: PolicyValueModel
    metadata: ResponseRankerArtifactMetadata
    resolved_configuration: dict[str, object]


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
        raise ResponseRankerError("ranker metadata must be canonical JSON") from error


def _json_digest(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _require_digest(value: object, label: str) -> str:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise ResponseRankerError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _strict_keys(value: object, expected: set[str], label: str) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != expected:
        raise ResponseRankerError(f"{label} keys must be exactly {sorted(expected)}")
    return value


def _nonempty(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or "\0" in value:
        raise ResponseRankerError(f"{label} must be nonempty and contain no NUL")
    return value


def _positive_int(value: object, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ResponseRankerError(f"{label} must be a positive integer")
    return value


def _finite_float(value: object, label: str, *, positive: bool = False) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise ResponseRankerError(f"{label} must be a finite TOML float")
    if positive and value <= 0:
        raise ResponseRankerError(f"{label} must be positive")
    return value


def load_response_ranker_config(path: str | Path) -> ResponseRankerConfig:
    try:
        with Path(path).open("rb") as stream:
            document = tomllib.load(stream)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise ResponseRankerError("could not read response-ranker TOML") from error
    root = _strict_keys(document, {"run", "dataset", "model", "optimization"}, "configuration")
    run = _strict_keys(
        root["run"],
        {"run_id", "root_seed", "output_directory", "source_revision"},
        "run section",
    )
    dataset = _strict_keys(root["dataset"], {"response_directory"}, "dataset section")
    model = _strict_keys(
        root["model"], {"model_id", "initialization_ordinal"}, "model section"
    )
    optimization = _strict_keys(
        root["optimization"],
        {
            "device",
            "batch_size",
            "learning_rate",
            "betas",
            "epsilon",
            "weight_decay",
            "gradient_norm",
            "minimum_epochs",
            "maximum_epochs",
            "early_stop_patience",
            "minimum_improvement",
        },
        "optimization section",
    )
    if optimization["device"] != "cpu":
        raise ResponseRankerError("the bounded response-ranker experiment uses CPU")
    betas = optimization["betas"]
    if (
        not isinstance(betas, list)
        or len(betas) != 2
        or any(type(value) is not float or not math.isfinite(value) for value in betas)
    ):
        raise ResponseRankerError("optimizer betas must contain two finite floats")
    ordinal = model["initialization_ordinal"]
    if type(ordinal) is not int or ordinal < 0:
        raise ResponseRankerError("initialization ordinal must be non-negative")
    minimum_epochs = _positive_int(optimization["minimum_epochs"], "minimum epochs")
    maximum_epochs = _positive_int(optimization["maximum_epochs"], "maximum epochs")
    if minimum_epochs > maximum_epochs:
        raise ResponseRankerError("minimum epochs cannot exceed maximum epochs")
    return ResponseRankerConfig(
        RankerRunConfig(
            _nonempty(run["run_id"], "run ID"),
            _nonempty(run["root_seed"], "root seed"),
            _nonempty(run["output_directory"], "output directory"),
            _nonempty(run["source_revision"], "source revision"),
        ),
        RankerDatasetConfig(
            _nonempty(dataset["response_directory"], "response directory")
        ),
        RankerModelConfig(
            _nonempty(model["model_id"], "model ID"),
            ordinal,
        ),
        RankerOptimizationConfig(
            device="cpu",
            batch_size=_positive_int(optimization["batch_size"], "batch size"),
            learning_rate=_finite_float(
                optimization["learning_rate"], "learning rate", positive=True
            ),
            betas=(float(betas[0]), float(betas[1])),
            epsilon=_finite_float(
                optimization["epsilon"], "epsilon", positive=True
            ),
            weight_decay=_finite_float(
                optimization["weight_decay"], "weight decay"
            ),
            gradient_norm=_finite_float(
                optimization["gradient_norm"], "gradient norm", positive=True
            ),
            minimum_epochs=minimum_epochs,
            maximum_epochs=maximum_epochs,
            early_stop_patience=_positive_int(
                optimization["early_stop_patience"], "early-stop patience"
            ),
            minimum_improvement=_finite_float(
                optimization["minimum_improvement"], "minimum improvement"
            ),
        ),
    )


def _load_split(
    response_directory: Path,
    split: FixtureSplit,
    *,
    maximum_groups: int,
) -> ResponseRankingDataset:
    document = load_response_manifest(response_directory, split)
    manifest = document["manifest"]
    observations: list[Tensor] = []
    legal_masks: list[Tensor] = []
    member_weights: list[Tensor] = []
    values: list[Tensor] = []
    valid: list[Tensor] = []
    representatives: list[Tensor] = []
    selected: list[Tensor] = []
    placements: list[Tensor] = []
    dealers: list[Tensor] = []
    roles: list[str] = []
    identities: list[str] = []
    fixture_ids: list[str] = []

    for entry in manifest["shards"]:
        shard = load_response_shard(response_directory / entry["relative_path"])
        columns = shard["columns"]
        count = int(columns["observations"].shape[0])
        observations.append(columns["observations"])
        legal_masks.append(columns["legal_masks"])
        selected.append(columns["selected_representative_actions"].to(torch.int64))
        placements.append(columns["placement_numbers"].to(torch.int64))
        dealers.append(columns["actor_is_dealer"])
        roles.extend(columns["actor_roles"])
        identities.extend(columns["cache_identity_digests"])
        fixture_ids.extend([str(entry["fixture_id"])] * count)

        weights = torch.zeros((count, maximum_groups, ACTION_COUNT), dtype=torch.float32)
        group_values = torch.zeros((count, maximum_groups), dtype=torch.float32)
        group_valid = torch.zeros((count, maximum_groups), dtype=torch.bool)
        group_reps = torch.full((count, maximum_groups), -1, dtype=torch.int64)
        group_offsets = columns["group_offsets"]
        member_offsets = columns["member_offsets"]
        for row in range(count):
            group_start = int(group_offsets[row].item())
            group_end = int(group_offsets[row + 1].item())
            if group_end - group_start > maximum_groups:
                raise ResponseRankerError("response row exceeds the group bound")
            for local, group_index in enumerate(range(group_start, group_end)):
                member_start = int(member_offsets[group_index].item())
                member_end = int(member_offsets[group_index + 1].item())
                members = columns["group_member_actions"][member_start:member_end].to(
                    torch.int64
                )
                weights[row, local, members] = 1.0 / len(members)
                group_values[row, local] = float(
                    columns["group_mean_values"][group_index].item()
                )
                group_valid[row, local] = True
                group_reps[row, local] = int(
                    columns["group_representative_actions"][group_index].item()
                )
        member_weights.append(weights)
        values.append(group_values)
        valid.append(group_valid)
        representatives.append(group_reps)

    return ResponseRankingDataset(
        split=split,
        observations=torch.cat(observations),
        legal_masks=torch.cat(legal_masks),
        group_member_weights=torch.cat(member_weights),
        group_values=torch.cat(values),
        group_valid=torch.cat(valid),
        group_representatives=torch.cat(representatives),
        selected_representatives=torch.cat(selected),
        placement_numbers=torch.cat(placements),
        actor_roles=tuple(roles),
        actor_is_dealer=torch.cat(dealers),
        cache_identity_digests=tuple(identities),
        fixture_ids=tuple(fixture_ids),
        manifest_digest=str(document["manifest_digest"]),
    )


def load_response_ranking_datasets(
    response_directory: str | Path,
) -> ResponseRankingBundle:
    path = Path(response_directory).expanduser().resolve()
    try:
        inspection = inspect_response_dataset(path)
        maximum_groups = 0
        for split in (FixtureSplit.TRAINING, FixtureSplit.VALIDATION):
            manifest = load_response_manifest(path, split)["manifest"]
            for entry in manifest["shards"]:
                columns = load_response_shard(path / entry["relative_path"])["columns"]
                offsets = columns["group_offsets"]
                if len(offsets) > 1:
                    maximum_groups = max(
                        maximum_groups,
                        int(torch.max(offsets[1:] - offsets[:-1]).item()),
                    )
        training = _load_split(path, FixtureSplit.TRAINING, maximum_groups=maximum_groups)
        validation = _load_split(path, FixtureSplit.VALIDATION, maximum_groups=maximum_groups)
    except ResponseDatasetError as error:
        raise ResponseRankerError("sealed response dataset is invalid") from error
    if set(training.fixture_ids) & set(validation.fixture_ids):
        raise ResponseRankerError("training and validation fixtures overlap")
    if set(training.cache_identity_digests) & set(validation.cache_identity_digests):
        raise ResponseRankerError("training and validation response identities overlap")
    if training.example_count + validation.example_count != inspection.examples:
        raise ResponseRankerError("response dataset example count differs")
    return ResponseRankingBundle(training, validation, inspection.dataset_digest)


def freeze_unused_value_head(model: PolicyValueModel) -> None:
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(not name.startswith("value_"))
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    if trainable != RESPONSE_RANKER_TRAINABLE_PARAMETER_COUNT:
        raise ResponseRankerError(
            f"response ranker has {trainable} trainable parameters, "
            f"expected {RESPONSE_RANKER_TRAINABLE_PARAMETER_COUNT}"
        )


def build_response_ranker_optimizer(
    model: PolicyValueModel,
    config: PolicyValueOptimizationConfig,
) -> torch.optim.AdamW:
    freeze_unused_value_head(model)
    return torch.optim.AdamW(
        tuple(parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
        eps=config.epsilon,
        weight_decay=config.weight_decay,
    )


def _batch_ranking_loss(
    logits: Tensor,
    member_weights: Tensor,
    group_values: Tensor,
    group_valid: Tensor,
) -> tuple[Tensor, int, float]:
    if (
        logits.dtype is not torch.float32
        or logits.ndim != 3
        or logits.shape[1:] != (4, 8)
        or not torch.isfinite(logits).all()
    ):
        raise ResponseRankerError("ranker logits must be finite float32[B,4,8]")
    if (
        member_weights.dtype is not torch.float32
        or group_values.dtype is not torch.float32
        or group_valid.dtype is not torch.bool
        or member_weights.shape[:2] != group_values.shape
        or group_values.shape != group_valid.shape
        or member_weights.shape[0] != logits.shape[0]
        or member_weights.shape[2] != ACTION_COUNT
    ):
        raise ResponseRankerError("ranker group tensors are incompatible")
    scores = torch.einsum("bga,ba->bg", member_weights, logits.flatten(start_dim=1))
    differences = group_values.unsqueeze(2) - group_values.unsqueeze(1)
    score_differences = scores.unsqueeze(2) - scores.unsqueeze(1)
    count = group_values.shape[1]
    upper = torch.triu(
        torch.ones((count, count), dtype=torch.bool, device=logits.device),
        diagonal=1,
    )
    pair_mask = (
        group_valid.unsqueeze(2)
        & group_valid.unsqueeze(1)
        & upper.unsqueeze(0)
        & (differences != 0)
    )
    weights = differences.abs()
    contributions = weights * F.softplus(
        -differences.sign() * score_differences
    )
    row_weights = (weights * pair_mask).sum(dim=(1, 2))
    row_losses = (contributions * pair_mask).sum(dim=(1, 2)) / row_weights.clamp_min(
        torch.finfo(torch.float32).eps
    )
    row_losses = torch.where(row_weights > 0, row_losses, row_losses * 0.0)
    return (
        row_losses.mean(),
        int(pair_mask.sum().detach().cpu()),
        float(row_weights.sum().detach().cpu()),
    )


def _batch(
    dataset: ResponseRankingDataset,
    indexes: Tensor,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    return (
        dataset.observations[indexes].to(device),
        dataset.group_member_weights[indexes].to(device),
        dataset.group_values[indexes].to(device),
        dataset.group_valid[indexes].to(device),
    )


def epoch_minibatch_indices(
    config: ResponseRankerConfig,
    bundle: ResponseRankingBundle,
    epoch: int,
) -> tuple[Tensor, ...]:
    if type(epoch) is not int or epoch < 0:
        raise ResponseRankerError("epoch must be non-negative")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(
        derive_pytorch_seed(
            RESPONSE_RANKER_SHUFFLE_NAMESPACE,
            config.run.root_seed,
            bundle.dataset_digest,
            str(epoch),
        )
    )
    order = torch.randperm(bundle.training.example_count, generator=generator)
    size = config.optimization.batch_size
    return tuple(order[start : start + size] for start in range(0, len(order), size))


def _rankdata(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + end - 1) / 2.0
        for index in order[start:end]:
            ranks[index] = rank
        start = end
    return ranks


def _correlation(left: Sequence[float], right: Sequence[float]) -> float | None:
    left_ranks = _rankdata(left)
    right_ranks = _rankdata(right)
    left_mean = statistics.fmean(left_ranks)
    right_mean = statistics.fmean(right_ranks)
    numerator = sum(
        (a - left_mean) * (b - right_mean)
        for a, b in zip(left_ranks, right_ranks, strict=True)
    )
    left_scale = math.sqrt(sum((value - left_mean) ** 2 for value in left_ranks))
    right_scale = math.sqrt(sum((value - right_mean) ** 2 for value in right_ranks))
    if left_scale == 0 or right_scale == 0:
        return None
    return numerator / (left_scale * right_scale)


def _empty_metric_accumulator() -> dict[str, float]:
    return {
        "examples": 0.0,
        "top_group_correct": 0.0,
        "top_two_correct": 0.0,
        "top_three_correct": 0.0,
        "regret": 0.0,
        "pair_correct": 0.0,
        "pair_total": 0.0,
        "correlation": 0.0,
        "correlation_examples": 0.0,
    }


def _finish_metric(value: Mapping[str, float]) -> dict[str, float | int | None]:
    examples = int(value["examples"])
    if examples == 0:
        return {
            "examples": 0,
            "top_group_accuracy": None,
            "top_two_recall": None,
            "top_three_recall": None,
            "mean_teacher_value_regret": None,
            "pairwise_rank_accuracy": None,
            "rank_correlation": None,
            "rank_correlation_examples": 0,
        }
    return {
        "examples": examples,
        "top_group_accuracy": value["top_group_correct"] / examples,
        "top_two_recall": value["top_two_correct"] / examples,
        "top_three_recall": value["top_three_correct"] / examples,
        "mean_teacher_value_regret": value["regret"] / examples,
        "pairwise_rank_accuracy": (
            value["pair_correct"] / value["pair_total"]
            if value["pair_total"]
            else None
        ),
        "rank_correlation": (
            value["correlation"] / value["correlation_examples"]
            if value["correlation_examples"]
            else None
        ),
        "rank_correlation_examples": int(value["correlation_examples"]),
    }


def _metric_keys(dataset: ResponseRankingDataset, row: int) -> tuple[str, ...]:
    placement = int(dataset.placement_numbers[row].item())
    return (
        "overall",
        f"placement:{placement}",
        f"role:{dataset.actor_roles[row]}",
        f"dealer:{'dealer' if bool(dataset.actor_is_dealer[row].item()) else 'non-dealer'}",
    )


@torch.no_grad()
def evaluate_response_ranker(
    model: PolicyValueModel,
    dataset: ResponseRankingDataset,
    *,
    batch_size: int = 256,
) -> dict[str, object]:
    model.eval()
    accumulators: dict[str, dict[str, float]] = {
        "overall": _empty_metric_accumulator(),
        **{
            f"placement:{placement}": _empty_metric_accumulator()
            for placement in range(1, 8)
        },
        "role:queen": _empty_metric_accumulator(),
        "role:king": _empty_metric_accumulator(),
        "dealer:dealer": _empty_metric_accumulator(),
        "dealer:non-dealer": _empty_metric_accumulator(),
    }
    loss_total = 0.0
    examples = 0
    for start in range(0, dataset.example_count, batch_size):
        indexes = torch.arange(start, min(start + batch_size, dataset.example_count))
        observations, weights, values, valid = _batch(
            dataset, indexes, torch.device("cpu")
        )
        logits, _ = model(observations)
        loss, _, _ = _batch_ranking_loss(logits, weights, values, valid)
        if not torch.isfinite(loss):
            raise ResponseRankerError("validation ranking loss is non-finite")
        loss_total += float(loss) * len(indexes)
        examples += len(indexes)
        scores = torch.einsum(
            "bga,ba->bg", weights, logits.flatten(start_dim=1)
        )
        for local, row in enumerate(range(start, start + len(indexes))):
            mask = valid[local]
            row_values = values[local][mask].tolist()
            row_scores = scores[local][mask].tolist()
            row_reps = dataset.group_representatives[row][mask].tolist()
            teacher_rep = int(dataset.selected_representatives[row].item())
            teacher_index = row_reps.index(teacher_rep)
            model_order = sorted(
                range(len(row_reps)),
                key=lambda index: (-row_scores[index], row_reps[index]),
            )
            model_index = model_order[0]
            pair_correct = 0.0
            pair_total = 0
            for left in range(len(row_values)):
                for right in range(left + 1, len(row_values)):
                    target_difference = row_values[left] - row_values[right]
                    if target_difference == 0:
                        continue
                    score_difference = row_scores[left] - row_scores[right]
                    pair_total += 1
                    if score_difference == 0:
                        pair_correct += 0.5
                    elif (target_difference > 0) == (score_difference > 0):
                        pair_correct += 1.0
            correlation = _correlation(row_values, row_scores)
            for key in _metric_keys(dataset, row):
                accumulator = accumulators[key]
                accumulator["examples"] += 1
                accumulator["top_group_correct"] += model_index == teacher_index
                accumulator["top_two_correct"] += teacher_index in model_order[:2]
                accumulator["top_three_correct"] += teacher_index in model_order[:3]
                accumulator["regret"] += max(row_values) - row_values[model_index]
                accumulator["pair_correct"] += pair_correct
                accumulator["pair_total"] += pair_total
                if correlation is not None:
                    accumulator["correlation"] += correlation
                    accumulator["correlation_examples"] += 1
    return {
        "ranking_loss": loss_total / examples,
        "overall": _finish_metric(accumulators["overall"]),
        "placements": {
            str(placement): _finish_metric(accumulators[f"placement:{placement}"])
            for placement in range(1, 8)
        },
        "roles": {
            role: _finish_metric(accumulators[f"role:{role}"])
            for role in ("queen", "king")
        },
        "dealer_status": {
            status: _finish_metric(accumulators[f"dealer:{status}"])
            for status in ("dealer", "non-dealer")
        },
    }


def _finite_parameters_and_gradients(model: PolicyValueModel) -> bool:
    return all(
        torch.isfinite(parameter).all()
        and (parameter.grad is None or torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
    )


def _training_epoch(
    model: PolicyValueModel,
    optimizer: torch.optim.AdamW,
    dataset: ResponseRankingDataset,
    batches: Sequence[Tensor],
    gradient_norm: float,
    *,
    epoch: int,
    should_stop: Callable[[int, int], bool] | None = None,
) -> tuple[float, float]:
    model.train()
    total = 0.0
    examples = 0
    maximum_gradient = 0.0
    for batch_number, indexes in enumerate(batches):
        if should_stop is not None and should_stop(epoch, batch_number):
            raise ResponseRankerInterrupted("response-ranker optimization interrupted")
        observations, weights, values, valid = _batch(
            dataset, indexes, torch.device("cpu")
        )
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(observations)
        loss, _, _ = _batch_ranking_loss(logits, weights, values, valid)
        if not torch.isfinite(loss):
            raise ResponseRankerError("training ranking loss is non-finite")
        loss.backward()
        if not _finite_parameters_and_gradients(model):
            raise ResponseRankerError("ranker gradient is non-finite")
        if any(
            parameter.grad is not None
            for name, parameter in model.named_parameters()
            if name.startswith("value_")
        ):
            raise ResponseRankerError("unused value head received a gradient")
        clipped = torch.nn.utils.clip_grad_norm_(
            tuple(
                parameter
                for parameter in model.parameters()
                if parameter.requires_grad
            ),
            gradient_norm,
        )
        if not torch.isfinite(clipped):
            raise ResponseRankerError("ranker gradient norm is non-finite")
        maximum_gradient = max(maximum_gradient, float(clipped))
        optimizer.step()
        if not _finite_parameters_and_gradients(model):
            raise ResponseRankerError("ranker parameter is non-finite")
        total += float(loss.detach()) * len(indexes)
        examples += len(indexes)
    return total / examples, maximum_gradient


def _state_dict_digest(state_dict: Mapping[str, Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state_dict.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(",".join(map(str, value.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _cpu_copy(value: object) -> object:
    if isinstance(value, Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_copy(item) for item in value)
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
    _atomic_bytes(path, _canonical_json(value))


def _peak_rss_bytes() -> int:
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(raw if sys.platform == "darwin" else raw * 1024)


def _resolved_document(
    config: ResponseRankerConfig,
    bundle: ResponseRankingBundle,
) -> dict[str, object]:
    body: dict[str, object] = {
        "format_version": RESPONSE_RANKER_CONFIG_VERSION,
        "run": {
            **asdict(config.run),
            "output_directory": str(config.output_path),
        },
        "dataset": {
            "response_directory": str(config.response_path),
            "dataset_digest": bundle.dataset_digest,
            "training_manifest_digest": bundle.training.manifest_digest,
            "validation_manifest_digest": bundle.validation.manifest_digest,
            "training_examples": bundle.training.example_count,
            "validation_examples": bundle.validation.example_count,
        },
        "model": {
            **asdict(config.model),
            "model_schema_version": MODEL_SCHEMA_VERSION,
            "parameter_count": PARAMETER_COUNT,
            "trainable_parameter_count": RESPONSE_RANKER_TRAINABLE_PARAMETER_COUNT,
            "value_head_usage": RESPONSE_RANKER_VALUE_HEAD_USAGE,
        },
        "ranking": {
            "response_example_schema_version": RESPONSE_EXAMPLE_SCHEMA_VERSION,
            "action_group_schema_version": RESPONSE_ACTION_GROUP_SCHEMA_VERSION,
            "ranking_schema_version": RESPONSE_RANKING_SCHEMA_VERSION,
            "group_logit_profile": RESPONSE_GROUP_LOGIT_PROFILE,
            "pair_weight_profile": RESPONSE_PAIR_WEIGHT_PROFILE,
        },
        "optimization": asdict(config.optimization),
    }
    body = json.loads(_canonical_json(body))
    body["resolved_config_digest"] = _json_digest(body)
    return body


def _checkpoint_payload(
    *,
    kind: str,
    resolved: Mapping[str, object],
    bundle: ResponseRankingBundle,
    model: PolicyValueModel,
    optimizer: torch.optim.AdamW,
    completed_epoch: int,
    best_epoch: int,
    best_validation_loss: float,
    stale_epochs: int,
    history: Sequence[RankerEpochMetrics],
) -> dict[str, object]:
    return {
        "format_version": RESPONSE_RANKER_CHECKPOINT_VERSION,
        "kind": kind,
        "resolved_config_digest": resolved["resolved_config_digest"],
        "dataset_digest": bundle.dataset_digest,
        "completed_epoch": completed_epoch,
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
        "stale_epochs": stale_epochs,
        "model_state_dict": _cpu_copy(model.state_dict()),
        "optimizer_state_dict": _cpu_copy(optimizer.state_dict()),
        "history": tuple(asdict(item) for item in history),
    }


def _validate_checkpoint(
    payload: object,
    config: ResponseRankerConfig,
    resolved: Mapping[str, object],
    bundle: ResponseRankingBundle,
) -> dict[str, object]:
    expected_keys = {
        "format_version",
        "kind",
        "resolved_config_digest",
        "dataset_digest",
        "completed_epoch",
        "best_epoch",
        "best_validation_loss",
        "stale_epochs",
        "model_state_dict",
        "optimizer_state_dict",
        "history",
    }
    if not isinstance(payload, dict) or set(payload) != expected_keys:
        raise ResponseRankerError("response-ranker checkpoint payload is invalid")
    if (
        payload["format_version"] != RESPONSE_RANKER_CHECKPOINT_VERSION
        or payload["kind"] not in _CHECKPOINT_NAMES
        or payload["resolved_config_digest"] != resolved["resolved_config_digest"]
        or payload["dataset_digest"] != bundle.dataset_digest
    ):
        raise ResponseRankerError("response-ranker checkpoint identity differs")
    for key in ("completed_epoch", "best_epoch", "stale_epochs"):
        if type(payload[key]) is not int or int(payload[key]) < -1:
            raise ResponseRankerError("response-ranker checkpoint epoch is invalid")
    loss = payload["best_validation_loss"]
    if type(loss) is not float or not (math.isfinite(loss) or loss == math.inf):
        raise ResponseRankerError("response-ranker validation state is invalid")
    history = payload["history"]
    if (
        not isinstance(history, tuple)
        or len(history) != int(payload["completed_epoch"]) + 1
    ):
        raise ResponseRankerError("response-ranker checkpoint history differs")
    for item in history:
        if not isinstance(item, dict) or set(item) != set(
            RankerEpochMetrics.__dataclass_fields__
        ):
            raise ResponseRankerError("response-ranker epoch metric is invalid")
        if any(
            not math.isfinite(value)
            for key, value in item.items()
            if key != "epoch"
        ):
            raise ResponseRankerError("response-ranker epoch metric is non-finite")
    model = PolicyValueModel(
        run_root_seed=config.run.root_seed,
        model_id=config.model.model_id,
        initialization_ordinal=config.model.initialization_ordinal,
    )
    freeze_unused_value_head(model)
    expected_state = model.state_dict()
    state = payload["model_state_dict"]
    if not isinstance(state, dict) or set(state) != set(expected_state):
        raise ResponseRankerError("response-ranker model state is incompatible")
    for name, expected in expected_state.items():
        value = state[name]
        if (
            not isinstance(value, Tensor)
            or value.device.type != "cpu"
            or value.dtype is not expected.dtype
            or value.shape != expected.shape
            or not torch.isfinite(value).all()
        ):
            raise ResponseRankerError(f"invalid checkpoint tensor: {name}")
    model.load_state_dict(state, strict=True)
    optimizer = build_response_ranker_optimizer(model, config.optimization.optimizer)
    try:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    except (KeyError, RuntimeError, ValueError) as error:
        raise ResponseRankerError("ranker optimizer state is incompatible") from error
    for state_value in optimizer.state.values():
        if any(
            isinstance(item, Tensor) and not torch.isfinite(item).all()
            for item in state_value.values()
        ):
            raise ResponseRankerError("ranker optimizer state is non-finite")
    return payload


def _atomic_checkpoint(
    path: Path,
    payload: dict[str, object],
    config: ResponseRankerConfig,
    resolved: Mapping[str, object],
    bundle: ResponseRankingBundle,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        torch.save(payload, temporary)
        loaded = torch.load(temporary, map_location="cpu", weights_only=True)
        _validate_checkpoint(loaded, config, resolved, bundle)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_resolved(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ResponseRankerError("resolved ranker configuration could not be read") from error
    if (
        not isinstance(value, dict)
        or value.get("format_version") != RESPONSE_RANKER_CONFIG_VERSION
    ):
        raise ResponseRankerError("resolved ranker configuration is incompatible")
    return value


def _load_checkpoint(
    path: Path,
    config: ResponseRankerConfig,
    resolved: Mapping[str, object],
    bundle: ResponseRankingBundle,
) -> dict[str, object]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, ValueError) as error:
        raise ResponseRankerError("response-ranker checkpoint could not be loaded") from error
    return _validate_checkpoint(payload, config, resolved, bundle)


def _write_state(output: Path, resolved: Mapping[str, object], phase: str, **extra: object) -> None:
    _atomic_json(
        output / "state.json",
        {
            "format_version": RESPONSE_RANKER_STATE_VERSION,
            "resolved_config_digest": resolved["resolved_config_digest"],
            "phase": phase,
            **extra,
        },
    )


def train_response_ranker(
    config: ResponseRankerConfig,
    *,
    resume: bool = False,
    should_stop: Callable[[int, int], bool] | None = None,
    progress: Callable[[RankerEpochMetrics], None] | None = None,
) -> ResponseRankerTrainingResult:
    bundle = load_response_ranking_datasets(config.response_path)
    resolved = _resolved_document(config, bundle)
    output = config.output_path
    resolved_path = output / "resolved-config.json"
    if resume:
        if _load_resolved(resolved_path) != resolved:
            raise ResponseRankerError("resolved ranker configuration differs")
    else:
        if resolved_path.exists():
            raise ResponseRankerError("ranker output exists; use resume")
        _atomic_json(resolved_path, resolved)
    checkpoints = output / "checkpoints"
    model = PolicyValueModel(
        run_root_seed=config.run.root_seed,
        model_id=config.model.model_id,
        initialization_ordinal=config.model.initialization_ordinal,
    )
    freeze_unused_value_head(model)
    optimizer = build_response_ranker_optimizer(model, config.optimization.optimizer)
    history: list[RankerEpochMetrics] = []
    completed_epoch = -1
    best_epoch = -1
    best_validation_loss = math.inf
    stale_epochs = 0
    if resume:
        latest = checkpoints / _CHECKPOINT_NAMES["epoch-latest"]
        start = checkpoints / _CHECKPOINT_NAMES["iteration-start"]
        payload = _load_checkpoint(
            latest if latest.exists() else start, config, resolved, bundle
        )
        model.load_state_dict(payload["model_state_dict"], strict=True)
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        completed_epoch = int(payload["completed_epoch"])
        best_epoch = int(payload["best_epoch"])
        best_validation_loss = float(payload["best_validation_loss"])
        stale_epochs = int(payload["stale_epochs"])
        history = [RankerEpochMetrics(**item) for item in payload["history"]]
    else:
        untrained = evaluate_response_ranker(
            model, bundle.validation, batch_size=config.optimization.batch_size
        )
        _atomic_json(
            output / "metrics" / "untrained-control.json",
            {
                "format_version": RESPONSE_RANKER_METRICS_VERSION,
                "dataset_digest": bundle.dataset_digest,
                **untrained,
            },
        )
        start_payload = _checkpoint_payload(
            kind="iteration-start",
            resolved=resolved,
            bundle=bundle,
            model=model,
            optimizer=optimizer,
            completed_epoch=-1,
            best_epoch=-1,
            best_validation_loss=math.inf,
            stale_epochs=0,
            history=(),
        )
        _atomic_checkpoint(
            checkpoints / _CHECKPOINT_NAMES["iteration-start"],
            start_payload,
            config,
            resolved,
            bundle,
        )
    _write_state(output, resolved, "optimizing", completed_epoch=completed_epoch)
    started_training = time.perf_counter()
    stopped_early = False
    try:
        for epoch in range(completed_epoch + 1, config.optimization.maximum_epochs):
            started_epoch = time.perf_counter()
            training_loss, maximum_gradient = _training_epoch(
                model,
                optimizer,
                bundle.training,
                epoch_minibatch_indices(config, bundle, epoch),
                config.optimization.gradient_norm,
                epoch=epoch,
                should_stop=should_stop,
            )
            validation = evaluate_response_ranker(
                model,
                bundle.validation,
                batch_size=config.optimization.batch_size,
            )
            overall = validation["overall"]
            elapsed = time.perf_counter() - started_epoch
            metrics = RankerEpochMetrics(
                epoch=epoch,
                training_ranking_loss=training_loss,
                validation_ranking_loss=float(validation["ranking_loss"]),
                validation_top_group_accuracy=float(
                    overall["top_group_accuracy"]
                ),
                validation_mean_regret=float(
                    overall["mean_teacher_value_regret"]
                ),
                maximum_gradient_norm=maximum_gradient,
                epoch_seconds=elapsed,
            )
            if any(
                not math.isfinite(value)
                for key, value in asdict(metrics).items()
                if key != "epoch"
            ):
                raise ResponseRankerError("ranker epoch metric is non-finite")
            improved = (
                metrics.validation_ranking_loss
                < best_validation_loss - config.optimization.minimum_improvement
            )
            if improved:
                best_epoch = epoch
                best_validation_loss = metrics.validation_ranking_loss
                stale_epochs = 0
            else:
                stale_epochs += 1
            history.append(metrics)
            completed_epoch = epoch
            payload = _checkpoint_payload(
                kind="epoch-latest",
                resolved=resolved,
                bundle=bundle,
                model=model,
                optimizer=optimizer,
                completed_epoch=completed_epoch,
                best_epoch=best_epoch,
                best_validation_loss=best_validation_loss,
                stale_epochs=stale_epochs,
                history=history,
            )
            if improved:
                best_payload = dict(payload)
                best_payload["kind"] = "best-validation"
                _atomic_checkpoint(
                    checkpoints / _CHECKPOINT_NAMES["best-validation"],
                    best_payload,
                    config,
                    resolved,
                    bundle,
                )
            _atomic_checkpoint(
                checkpoints / _CHECKPOINT_NAMES["epoch-latest"],
                payload,
                config,
                resolved,
                bundle,
            )
            _atomic_json(
                output / "metrics" / f"{epoch:06d}.json",
                {
                    "format_version": RESPONSE_RANKER_METRICS_VERSION,
                    "dataset_digest": bundle.dataset_digest,
                    **asdict(metrics),
                },
            )
            _write_state(output, resolved, "optimizing", completed_epoch=epoch)
            if progress is not None:
                progress(metrics)
            if (
                epoch + 1 >= config.optimization.minimum_epochs
                and stale_epochs >= config.optimization.early_stop_patience
            ):
                stopped_early = epoch + 1 < config.optimization.maximum_epochs
                break
    except ResponseRankerInterrupted:
        _write_state(output, resolved, "interrupted", completed_epoch=completed_epoch)
        raise
    except Exception as error:
        _write_state(
            output,
            resolved,
            "failed",
            completed_epoch=completed_epoch,
            error=type(error).__name__,
        )
        raise
    best = _load_checkpoint(
        checkpoints / _CHECKPOINT_NAMES["best-validation"],
        config,
        resolved,
        bundle,
    )
    final = dict(best)
    final["kind"] = "final"
    _atomic_checkpoint(
        checkpoints / _CHECKPOINT_NAMES["final"], final, config, resolved, bundle
    )
    result = ResponseRankerTrainingResult(
        output_directory=str(output),
        dataset_digest=bundle.dataset_digest,
        completed_epochs=completed_epoch + 1,
        best_epoch=best_epoch,
        best_validation_loss=best_validation_loss,
        stopped_early=stopped_early,
        final_checkpoint=str(checkpoints / _CHECKPOINT_NAMES["final"]),
        training_seconds=time.perf_counter() - started_training,
        peak_rss_bytes=_peak_rss_bytes(),
    )
    _atomic_json(
        output / "metrics" / "summary.json",
        {"format_version": RESPONSE_RANKER_METRICS_VERSION, **asdict(result)},
    )
    _write_state(output, resolved, "complete", **asdict(result))
    return result


def _model_from_checkpoint(
    config: ResponseRankerConfig,
    bundle: ResponseRankingBundle,
    checkpoint: str | Path | None,
) -> tuple[PolicyValueModel, dict[str, object], dict[str, object]]:
    resolved = _load_resolved(config.output_path / "resolved-config.json")
    if resolved != _resolved_document(config, bundle):
        raise ResponseRankerError("resolved ranker configuration differs")
    path = (
        Path(checkpoint)
        if checkpoint is not None
        else config.output_path / "checkpoints" / "final.pt"
    )
    payload = _load_checkpoint(path, config, resolved, bundle)
    model = PolicyValueModel(
        run_root_seed=config.run.root_seed,
        model_id=config.model.model_id,
        initialization_ordinal=config.model.initialization_ordinal,
    )
    freeze_unused_value_head(model)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return model, payload, resolved


def validate_response_ranker(
    config: ResponseRankerConfig,
    checkpoint: str | Path | None = None,
) -> dict[str, object]:
    bundle = load_response_ranking_datasets(config.response_path)
    model, payload, _ = _model_from_checkpoint(config, bundle, checkpoint)
    return {
        "dataset_digest": bundle.dataset_digest,
        "best_epoch": int(payload["best_epoch"]),
        "training": evaluate_response_ranker(
            model, bundle.training, batch_size=config.optimization.batch_size
        ),
        "validation": evaluate_response_ranker(
            model, bundle.validation, batch_size=config.optimization.batch_size
        ),
    }


def build_response_ranker_artifact(
    model: PolicyValueModel,
    *,
    config: ResponseRankerConfig,
    resolved: Mapping[str, object],
    bundle: ResponseRankingBundle,
    best_epoch: int,
    best_validation_loss: float,
) -> dict[str, object]:
    if (
        model.run_root_seed != config.run.root_seed
        or model.model_id != config.model.model_id
        or model.initialization_ordinal != config.model.initialization_ordinal
    ):
        raise ResponseRankerError("ranker model and resolved identity differ")
    freeze_unused_value_head(model)
    state = {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }
    metadata = ResponseRankerArtifactMetadata(
        model_schema_version=MODEL_SCHEMA_VERSION,
        observation_schema_version=OBSERVATION_SCHEMA_VERSION,
        action_schema_version=ACTION_SCHEMA_VERSION,
        response_example_schema_version=RESPONSE_EXAMPLE_SCHEMA_VERSION,
        action_group_schema_version=RESPONSE_ACTION_GROUP_SCHEMA_VERSION,
        ranking_schema_version=RESPONSE_RANKING_SCHEMA_VERSION,
        group_logit_profile=RESPONSE_GROUP_LOGIT_PROFILE,
        pair_weight_profile=RESPONSE_PAIR_WEIGHT_PROFILE,
        optimizer_compatibility_version=RESPONSE_RANKER_OPTIMIZER_VERSION,
        value_head_usage=RESPONSE_RANKER_VALUE_HEAD_USAGE,
        parameter_count=PARAMETER_COUNT,
        trainable_parameter_count=RESPONSE_RANKER_TRAINABLE_PARAMETER_COUNT,
        card_ids=CARD_IDS,
        policy_grid_indices=POLICY_GRID_INDICES,
        run_root_seed=model.run_root_seed,
        model_id=model.model_id,
        initialization_ordinal=model.initialization_ordinal,
        initialization_seed_digest=model.initialization_seed_digest,
        source_revision=config.run.source_revision,
        dataset_digest=bundle.dataset_digest,
        training_manifest_digest=bundle.training.manifest_digest,
        validation_manifest_digest=bundle.validation.manifest_digest,
        resolved_config_digest=str(resolved["resolved_config_digest"]),
        best_epoch=best_epoch,
        best_validation_loss=best_validation_loss,
        state_dict_digest=_state_dict_digest(state),
    )
    return {
        "format_version": RESPONSE_RANKER_ARTIFACT_VERSION,
        "metadata": asdict(metadata),
        "resolved_configuration": dict(resolved),
        "state_dict": state,
    }


def save_response_ranker_artifact(
    destination: str | Path,
    model: PolicyValueModel,
    *,
    config: ResponseRankerConfig,
    resolved: Mapping[str, object],
    bundle: ResponseRankingBundle,
    best_epoch: int,
    best_validation_loss: float,
) -> Path:
    path = Path(destination).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_response_ranker_artifact(
        model,
        config=config,
        resolved=resolved,
        bundle=bundle,
        best_epoch=best_epoch,
        best_validation_loss=best_validation_loss,
    )
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        torch.save(payload, temporary)
        load_response_ranker_artifact(temporary)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def load_response_ranker_artifact(path: str | Path) -> LoadedResponseRankerArtifact:
    try:
        payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, ValueError) as error:
        raise ResponseRankerError("response-ranker artifact could not be loaded") from error
    if not isinstance(payload, dict) or set(payload) != {
        "format_version",
        "metadata",
        "resolved_configuration",
        "state_dict",
    }:
        raise ResponseRankerError("response-ranker artifact payload is invalid")
    if payload["format_version"] != RESPONSE_RANKER_ARTIFACT_VERSION:
        raise ResponseRankerError("response-ranker artifact format is incompatible")
    metadata_value = payload["metadata"]
    if (
        not isinstance(metadata_value, dict)
        or set(metadata_value) != set(ResponseRankerArtifactMetadata.__dataclass_fields__)
        or set(metadata_value).intersection(_FORBIDDEN_ARTIFACT_KEYS)
    ):
        raise ResponseRankerError("response-ranker artifact metadata is invalid")
    try:
        metadata = ResponseRankerArtifactMetadata(**metadata_value)
    except TypeError as error:
        raise ResponseRankerError("response-ranker metadata is incompatible") from error
    expected = {
        "model_schema_version": MODEL_SCHEMA_VERSION,
        "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
        "action_schema_version": ACTION_SCHEMA_VERSION,
        "response_example_schema_version": RESPONSE_EXAMPLE_SCHEMA_VERSION,
        "action_group_schema_version": RESPONSE_ACTION_GROUP_SCHEMA_VERSION,
        "ranking_schema_version": RESPONSE_RANKING_SCHEMA_VERSION,
        "group_logit_profile": RESPONSE_GROUP_LOGIT_PROFILE,
        "pair_weight_profile": RESPONSE_PAIR_WEIGHT_PROFILE,
        "optimizer_compatibility_version": RESPONSE_RANKER_OPTIMIZER_VERSION,
        "value_head_usage": RESPONSE_RANKER_VALUE_HEAD_USAGE,
        "parameter_count": PARAMETER_COUNT,
        "trainable_parameter_count": RESPONSE_RANKER_TRAINABLE_PARAMETER_COUNT,
        "card_ids": CARD_IDS,
        "policy_grid_indices": POLICY_GRID_INDICES,
    }
    for key, value in expected.items():
        if getattr(metadata, key) != value:
            raise ResponseRankerError(f"response-ranker artifact {key} differs")
    for field in (
        "initialization_seed_digest",
        "dataset_digest",
        "training_manifest_digest",
        "validation_manifest_digest",
        "resolved_config_digest",
        "state_dict_digest",
    ):
        _require_digest(getattr(metadata, field), field.replace("_", " "))
    if (
        not isinstance(metadata.best_epoch, int)
        or metadata.best_epoch < 0
        or not math.isfinite(metadata.best_validation_loss)
        or not metadata.source_revision
    ):
        raise ResponseRankerError("response-ranker selection metadata is invalid")
    resolved = payload["resolved_configuration"]
    if (
        not isinstance(resolved, dict)
        or set(resolved)
        != {
            "format_version",
            "run",
            "dataset",
            "model",
            "ranking",
            "optimization",
            "resolved_config_digest",
        }
        or _json_digest(
            {
                key: value
                for key, value in resolved.items()
                if key != "resolved_config_digest"
            }
        )
        != resolved.get("resolved_config_digest")
    ):
        raise ResponseRankerError("response-ranker resolved configuration differs")
    if resolved["resolved_config_digest"] != metadata.resolved_config_digest:
        raise ResponseRankerError("response-ranker configuration digest differs")
    run = resolved["run"]
    dataset = resolved["dataset"]
    model_config = resolved["model"]
    ranking = resolved["ranking"]
    if (
        not isinstance(run, dict)
        or not isinstance(dataset, dict)
        or not isinstance(model_config, dict)
        or not isinstance(ranking, dict)
        or run.get("root_seed") != metadata.run_root_seed
        or run.get("source_revision") != metadata.source_revision
        or dataset.get("dataset_digest") != metadata.dataset_digest
        or dataset.get("training_manifest_digest")
        != metadata.training_manifest_digest
        or dataset.get("validation_manifest_digest")
        != metadata.validation_manifest_digest
        or model_config.get("model_id") != metadata.model_id
        or model_config.get("initialization_ordinal")
        != metadata.initialization_ordinal
        or model_config.get("parameter_count") != metadata.parameter_count
        or model_config.get("trainable_parameter_count")
        != metadata.trainable_parameter_count
        or model_config.get("value_head_usage") != metadata.value_head_usage
        or ranking.get("response_example_schema_version")
        != metadata.response_example_schema_version
        or ranking.get("action_group_schema_version")
        != metadata.action_group_schema_version
        or ranking.get("ranking_schema_version") != metadata.ranking_schema_version
        or ranking.get("group_logit_profile") != metadata.group_logit_profile
        or ranking.get("pair_weight_profile") != metadata.pair_weight_profile
    ):
        raise ResponseRankerError("response-ranker metadata and configuration differ")
    model = PolicyValueModel(
        run_root_seed=metadata.run_root_seed,
        model_id=metadata.model_id,
        initialization_ordinal=metadata.initialization_ordinal,
    )
    if model.initialization_seed_digest != metadata.initialization_seed_digest:
        raise ResponseRankerError("response-ranker initialization identity differs")
    freeze_unused_value_head(model)
    state = payload["state_dict"]
    expected_state = model.state_dict()
    if not isinstance(state, dict) or set(state) != set(expected_state):
        raise ResponseRankerError("response-ranker state keys differ")
    for name, expected_tensor in expected_state.items():
        value = state[name]
        if (
            not isinstance(value, Tensor)
            or value.device.type != "cpu"
            or value.dtype is not expected_tensor.dtype
            or value.shape != expected_tensor.shape
            or not torch.isfinite(value).all()
        ):
            raise ResponseRankerError(f"response-ranker tensor is invalid: {name}")
    if _state_dict_digest(state) != metadata.state_dict_digest:
        raise ResponseRankerError("response-ranker state digest differs")
    model.load_state_dict(state, strict=True)
    return LoadedResponseRankerArtifact(model, metadata, resolved)


def export_response_ranker(
    config: ResponseRankerConfig,
    destination: str | Path,
    checkpoint: str | Path | None = None,
) -> Path:
    bundle = load_response_ranking_datasets(config.response_path)
    model, payload, resolved = _model_from_checkpoint(config, bundle, checkpoint)
    path = save_response_ranker_artifact(
        destination,
        model,
        config=config,
        resolved=resolved,
        bundle=bundle,
        best_epoch=int(payload["best_epoch"]),
        best_validation_loss=float(payload["best_validation_loss"]),
    )
    loaded = load_response_ranker_artifact(path)
    sample = bundle.validation.observations[: min(16, bundle.validation.example_count)]
    with torch.no_grad():
        expected_logits, _ = model(sample)
        actual_logits, _ = loaded.model(sample)
    if not torch.equal(expected_logits, actual_logits):
        raise ResponseRankerError("exported ranker inference differs exactly")
    return path


def benchmark_response_ranker_inference(
    model: PolicyValueModel,
    dataset: ResponseRankingDataset,
    *,
    batch_size: int = 256,
    warmups: int = 20,
    repetitions: int = 100,
) -> dict[str, float | int]:
    model.eval()
    single_observation = dataset.observations[0]
    single_weights = dataset.group_member_weights[0]
    batch = dataset.observations[: min(batch_size, dataset.example_count)]
    with torch.no_grad():
        for _ in range(warmups):
            logits, _ = model(single_observation)
            _ = single_weights @ logits.flatten()
            model(batch)
        single_times = []
        batch_times = []
        for _ in range(repetitions):
            started = time.perf_counter()
            logits, _ = model(single_observation)
            _ = single_weights @ logits.flatten()
            single_times.append(time.perf_counter() - started)
            started = time.perf_counter()
            model(batch)
            batch_times.append(time.perf_counter() - started)
    return {
        "single_state_median_seconds": statistics.median(single_times),
        "single_state_p95_seconds": sorted(single_times)[
            math.ceil(0.95 * len(single_times)) - 1
        ],
        "batch_size": len(batch),
        "batch_median_seconds": statistics.median(batch_times),
        "batch_per_example_median_seconds": statistics.median(batch_times)
        / len(batch),
    }


def load_config_from_run(output_directory: str | Path) -> ResponseRankerConfig:
    resolved = _load_resolved(
        Path(output_directory).expanduser().resolve() / "resolved-config.json"
    )
    run = resolved["run"]
    dataset = resolved["dataset"]
    model = resolved["model"]
    optimization = resolved["optimization"]
    return ResponseRankerConfig(
        RankerRunConfig(
            run["run_id"],
            run["root_seed"],
            run["output_directory"],
            run["source_revision"],
        ),
        RankerDatasetConfig(dataset["response_directory"]),
        RankerModelConfig(
            model["model_id"], model["initialization_ordinal"]
        ),
        RankerOptimizationConfig(
            **{
                **optimization,
                "betas": tuple(optimization["betas"]),
            }
        ),
    )


def _print_progress(metrics: RankerEpochMetrics) -> None:
    print(
        "response_ranker "
        f"epoch={metrics.epoch} "
        f"train_loss={metrics.training_ranking_loss:.6f} "
        f"validation_loss={metrics.validation_ranking_loss:.6f} "
        f"top_accuracy={metrics.validation_top_group_accuracy:.4f} "
        f"regret={metrics.validation_mean_regret:.6f} "
        f"seconds={metrics.epoch_seconds:.3f}",
        flush=True,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="dracula-response-ranker")
    subparsers = parser.add_subparsers(dest="command", required=True)
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--config", required=True)
    resume_parser = subparsers.add_parser("resume")
    resume_parser.add_argument("--output", required=True)
    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--output", required=True)
    evaluate_parser.add_argument("--checkpoint")
    export_parser = subparsers.add_parser("export")
    export_parser.add_argument("--output", required=True)
    export_parser.add_argument("--destination", required=True)
    export_parser.add_argument("--checkpoint")
    args = parser.parse_args(argv)
    if args.command == "train":
        result = train_response_ranker(
            load_response_ranker_config(args.config), progress=_print_progress
        )
        print(json.dumps(asdict(result), indent=2, sort_keys=True))
    elif args.command == "resume":
        result = train_response_ranker(
            load_config_from_run(args.output), resume=True, progress=_print_progress
        )
        print(json.dumps(asdict(result), indent=2, sort_keys=True))
    elif args.command == "evaluate":
        print(
            json.dumps(
                validate_response_ranker(
                    load_config_from_run(args.output), args.checkpoint
                ),
                indent=2,
                sort_keys=True,
            )
        )
    else:
        path = export_response_ranker(
            load_config_from_run(args.output),
            args.destination,
            args.checkpoint,
        )
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = (
    "RESPONSE_RANKER_ARTIFACT_VERSION",
    "RESPONSE_RANKER_CHECKPOINT_VERSION",
    "RESPONSE_RANKER_CONFIG_VERSION",
    "RESPONSE_RANKER_TRAINABLE_PARAMETER_COUNT",
    "LoadedResponseRankerArtifact",
    "ResponseRankerArtifactMetadata",
    "ResponseRankerConfig",
    "ResponseRankerError",
    "ResponseRankerInterrupted",
    "ResponseRankingBundle",
    "ResponseRankingDataset",
    "benchmark_response_ranker_inference",
    "build_response_ranker_artifact",
    "build_response_ranker_optimizer",
    "epoch_minibatch_indices",
    "evaluate_response_ranker",
    "export_response_ranker",
    "freeze_unused_value_head",
    "load_config_from_run",
    "load_response_ranker_artifact",
    "load_response_ranker_config",
    "load_response_ranking_datasets",
    "main",
    "save_response_ranker_artifact",
    "train_response_ranker",
    "validate_response_ranker",
)

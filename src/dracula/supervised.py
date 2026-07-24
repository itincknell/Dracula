"""Local supervised optimization over sealed search-teacher datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import resource
import statistics
import subprocess
import sys
import time
import tomllib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import Tensor

from dracula.policy_value import (
    ACTION_SCHEMA_VERSION,
    MODEL_SCHEMA_VERSION,
    OBSERVATION_SCHEMA_VERSION,
    OPTIMIZER_COMPATIBILITY_VERSION,
    PARAMETER_COUNT,
    VALUE_SCHEMA_VERSION,
    PolicyValueContractError,
    PolicyValueModel,
    PolicyValueOptimizationConfig,
    build_policy_value_optimizer,
    load_policy_value_artifact,
    policy_value_loss,
    save_policy_value_artifact,
)
from dracula.randomness import derive_pytorch_seed
from dracula.teacher import (
    FixtureSplit,
    TeacherCollectionError,
    inspect_teacher_dataset,
    load_teacher_manifest,
    load_teacher_shard,
)

SUPERVISED_CONFIG_FORMAT_VERSION = "dracula-supervised-config-v1"
SUPERVISED_CHECKPOINT_FORMAT_VERSION = "dracula-supervised-checkpoint-v1"
SUPERVISED_STATE_FORMAT_VERSION = "dracula-supervised-state-v1"
SUPERVISED_METRICS_FORMAT_VERSION = "dracula-supervised-metrics-v1"
REPLAY_SAMPLING_NAMESPACE = "dracula-replay-sampling-v1"

_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_CHECKPOINT_NAMES = {
    "iteration-start": "iteration-start.pt",
    "epoch-latest": "epoch-latest.pt",
    "best-validation": "best-validation.pt",
    "final": "final.pt",
}
_CHECKPOINT_KEYS = {
    "format_version",
    "kind",
    "resolved_config_digest",
    "dataset_digest",
    "selected_device",
    "completed_epoch",
    "best_epoch",
    "best_validation_loss",
    "stale_epochs",
    "model_state_dict",
    "optimizer_state_dict",
    "history",
}
class SupervisedTrainingError(ValueError):
    """A training configuration, dataset, or artifact violates the contract."""


class SupervisedTrainingInterrupted(RuntimeError):
    """Optimization stopped before the active epoch was committed."""


@dataclass(frozen=True, slots=True)
class RunConfig:
    run_id: str
    root_seed: str
    output_directory: str
    source_revision: str


@dataclass(frozen=True, slots=True)
class DatasetConfig:
    teacher_directory: str
    search_report_digest: str


@dataclass(frozen=True, slots=True)
class ModelConfig:
    model_id: str
    initialization_ordinal: int


@dataclass(frozen=True, slots=True)
class OptimizationConfig:
    device: str = "auto"
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
    def model_optimizer(self) -> PolicyValueOptimizationConfig:
        return PolicyValueOptimizationConfig(
            learning_rate=self.learning_rate,
            beta1=self.betas[0],
            beta2=self.betas[1],
            epsilon=self.epsilon,
            weight_decay=self.weight_decay,
        )


@dataclass(frozen=True, slots=True)
class SupervisedConfig:
    run: RunConfig
    dataset: DatasetConfig
    model: ModelConfig
    optimization: OptimizationConfig

    @property
    def output_path(self) -> Path:
        return Path(self.run.output_directory).expanduser().resolve()

    @property
    def teacher_path(self) -> Path:
        return Path(self.dataset.teacher_directory).expanduser().resolve()


@dataclass(frozen=True, slots=True)
class SupervisedDataset:
    split: FixtureSplit
    observations: Tensor
    legal_masks: Tensor
    search_policies: Tensor
    round_returns: Tensor
    fixture_ids: tuple[str, ...]
    shard_digests: tuple[str, ...]

    @property
    def example_count(self) -> int:
        return int(self.observations.shape[0])


@dataclass(frozen=True, slots=True)
class DatasetBundle:
    training: SupervisedDataset
    validation: SupervisedDataset
    dataset_digest: str


@dataclass(frozen=True, slots=True)
class EpochMetrics:
    epoch: int
    training_total_loss: float
    training_policy_cross_entropy: float
    training_value_mse: float
    validation_total_loss: float
    validation_policy_cross_entropy: float
    validation_value_mse: float
    validation_policy_accuracy: float
    validation_value_mae: float
    maximum_gradient_norm: float


@dataclass(frozen=True, slots=True)
class DeviceMeasurement:
    device: str
    available: bool
    finite: bool
    output_agreement: bool
    median_epoch_seconds: float | None
    examples_per_second: float | None
    peak_rss_bytes: int | None
    swap_growth_bytes: int | None
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class DeviceBenchmark:
    selected_device: str
    cpu: DeviceMeasurement
    mps: DeviceMeasurement


@dataclass(frozen=True, slots=True)
class TrainingResult:
    output_directory: str
    dataset_digest: str
    selected_device: str
    completed_epochs: int
    best_epoch: int
    best_validation_loss: float
    stopped_early: bool
    final_checkpoint: str


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
        raise SupervisedTrainingError("metadata must be canonical JSON") from error


def _json_digest(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _require_digest(value: object, label: str) -> str:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise SupervisedTrainingError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _strict_keys(value: object, expected: set[str], label: str) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != expected:
        raise SupervisedTrainingError(f"{label} keys must be exactly {sorted(expected)}")
    if any(not isinstance(key, str) for key in value):
        raise SupervisedTrainingError(f"{label} keys must be strings")
    return value


def _nonempty_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or "\0" in value:
        raise SupervisedTrainingError(f"{label} must be a nonempty string without NUL")
    return value


def _positive_int(value: object, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise SupervisedTrainingError(f"{label} must be a positive integer")
    return value


def _finite_float(value: object, label: str, *, positive: bool = False) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise SupervisedTrainingError(f"{label} must be a finite TOML float")
    if positive and value <= 0:
        raise SupervisedTrainingError(f"{label} must be positive")
    return value


def load_supervised_config(path: str | Path) -> SupervisedConfig:
    """Load a TOML configuration with no implicit or unknown fields."""

    try:
        with Path(path).open("rb") as stream:
            document = tomllib.load(stream)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise SupervisedTrainingError("could not read supervised TOML configuration") from error
    root = _strict_keys(document, {"run", "dataset", "model", "optimization"}, "configuration")
    run = _strict_keys(
        root["run"],
        {"run_id", "root_seed", "output_directory", "source_revision"},
        "run section",
    )
    dataset = _strict_keys(
        root["dataset"], {"teacher_directory", "search_report_digest"}, "dataset section"
    )
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
    device = optimization["device"]
    if device not in {"cpu", "mps", "auto"}:
        raise SupervisedTrainingError("optimization device must be cpu, mps, or auto")
    betas = optimization["betas"]
    if (
        not isinstance(betas, list)
        or len(betas) != 2
        or any(type(value) is not float or not math.isfinite(value) for value in betas)
    ):
        raise SupervisedTrainingError("optimization betas must contain two finite TOML floats")
    ordinal = model["initialization_ordinal"]
    if type(ordinal) is not int or ordinal < 0:
        raise SupervisedTrainingError("initialization ordinal must be a non-negative integer")
    resolved = SupervisedConfig(
        run=RunConfig(
            _nonempty_string(run["run_id"], "run ID"),
            _nonempty_string(run["root_seed"], "run root seed"),
            _nonempty_string(run["output_directory"], "output directory"),
            _nonempty_string(run["source_revision"], "source revision"),
        ),
        dataset=DatasetConfig(
            _nonempty_string(dataset["teacher_directory"], "teacher directory"),
            _require_digest(dataset["search_report_digest"], "search report digest"),
        ),
        model=ModelConfig(
            _nonempty_string(model["model_id"], "model ID"),
            ordinal,
        ),
        optimization=OptimizationConfig(
            device=device,
            batch_size=_positive_int(optimization["batch_size"], "batch size"),
            learning_rate=_finite_float(
                optimization["learning_rate"], "learning rate", positive=True
            ),
            betas=(float(betas[0]), float(betas[1])),
            epsilon=_finite_float(optimization["epsilon"], "epsilon", positive=True),
            weight_decay=_finite_float(optimization["weight_decay"], "weight decay"),
            gradient_norm=_finite_float(
                optimization["gradient_norm"], "gradient norm", positive=True
            ),
            minimum_epochs=_positive_int(optimization["minimum_epochs"], "minimum epochs"),
            maximum_epochs=_positive_int(optimization["maximum_epochs"], "maximum epochs"),
            early_stop_patience=_positive_int(
                optimization["early_stop_patience"], "early-stop patience"
            ),
            minimum_improvement=_finite_float(
                optimization["minimum_improvement"], "minimum improvement", positive=True
            ),
        ),
    )
    if resolved.optimization.minimum_epochs > resolved.optimization.maximum_epochs:
        raise SupervisedTrainingError("minimum epochs cannot exceed maximum epochs")
    try:
        resolved.optimization.model_optimizer
    except PolicyValueContractError as error:
        raise SupervisedTrainingError("optimizer values violate the model contract") from error
    return resolved


def _empty_dataset(split: FixtureSplit) -> SupervisedDataset:
    return SupervisedDataset(
        split,
        torch.empty((0, 875), dtype=torch.bool),
        torch.empty((0, 4, 8), dtype=torch.bool),
        torch.empty((0, 4, 8), dtype=torch.float32),
        torch.empty((0,), dtype=torch.float32),
        (),
        (),
    )


def _load_split(teacher_path: Path, split: FixtureSplit) -> SupervisedDataset:
    document = load_teacher_manifest(teacher_path, split)
    manifest = document["manifest"]
    assert isinstance(manifest, dict)
    datasets: list[dict[str, object]] = []
    fixture_ids: list[str] = []
    shard_digests: list[str] = []
    entries = sorted(manifest["shards"], key=lambda item: item["fixture_index"])
    for entry in entries:
        assert isinstance(entry, dict)
        shard = load_teacher_shard(teacher_path / str(entry["relative_path"]))
        datasets.append(shard["columns"])  # type: ignore[arg-type]
        fixture_ids.append(str(entry["fixture_id"]))
        shard_digests.append(str(entry["content_digest"]))
    if not datasets:
        return _empty_dataset(split)
    return SupervisedDataset(
        split=split,
        observations=torch.cat([value["observations"] for value in datasets]),  # type: ignore[list-item]
        legal_masks=torch.cat([value["legal_masks"] for value in datasets]),  # type: ignore[list-item]
        search_policies=torch.cat(
            [value["search_policies"].reshape(-1, 4, 8) for value in datasets]  # type: ignore[union-attr]
        ),
        round_returns=torch.cat([value["round_returns"] for value in datasets]),  # type: ignore[list-item]
        fixture_ids=tuple(fixture_ids),
        shard_digests=tuple(shard_digests),
    )


def load_supervised_datasets(teacher_directory: str | Path) -> DatasetBundle:
    """Validate and load only sealed training and validation manifests."""

    try:
        inspection = inspect_teacher_dataset(teacher_directory)
        teacher_path = Path(teacher_directory).expanduser().resolve()
        training = _load_split(teacher_path, FixtureSplit.TRAINING)
        validation = _load_split(teacher_path, FixtureSplit.VALIDATION)
    except TeacherCollectionError as error:
        raise SupervisedTrainingError("sealed teacher dataset is invalid") from error
    if training.example_count == 0 or validation.example_count == 0:
        raise SupervisedTrainingError("training requires nonempty sealed training and validation splits")
    if set(training.fixture_ids) & set(validation.fixture_ids):
        raise SupervisedTrainingError("training and validation fixtures overlap")
    if training.example_count + validation.example_count != inspection.examples:
        raise SupervisedTrainingError("teacher dataset example count differs")
    return DatasetBundle(training, validation, inspection.dataset_digest)


def epoch_minibatch_indices(
    config: SupervisedConfig,
    datasets: DatasetBundle,
    epoch: int,
) -> tuple[Tensor, ...]:
    if type(epoch) is not int or epoch < 0:
        raise SupervisedTrainingError("epoch must be non-negative")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(
        derive_pytorch_seed(
            REPLAY_SAMPLING_NAMESPACE,
            config.run.root_seed,
            "warm-start",
            datasets.dataset_digest,
            str(epoch),
        )
    )
    order = torch.randperm(datasets.training.example_count, generator=generator)
    batch_size = config.optimization.batch_size
    return tuple(order[start : start + batch_size] for start in range(0, len(order), batch_size))


def _peak_rss_bytes() -> int:
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(raw if sys.platform == "darwin" else raw * 1024)


def _swap_used_bytes() -> int:
    if sys.platform == "darwin":
        try:
            output = subprocess.run(
                ["sysctl", "-n", "vm.swapusage"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout
            match = re.search(r"used = ([0-9.]+)([MG])", output)
            if match:
                scale = 1024**2 if match.group(2) == "M" else 1024**3
                return int(float(match.group(1)) * scale)
        except (OSError, subprocess.SubprocessError):
            return 0
    try:
        fields = Path("/proc/meminfo").read_text(encoding="ascii").splitlines()
        values = {
            key: int(value.split()[0]) * 1024
            for key, value in (line.split(":", 1) for line in fields)
            if key in {"SwapTotal", "SwapFree"}
        }
        return values.get("SwapTotal", 0) - values.get("SwapFree", 0)
    except OSError:
        return 0


def _synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()


def _finite_parameters_and_gradients(model: PolicyValueModel) -> bool:
    return all(
        torch.isfinite(parameter).all()
        and (parameter.grad is None or torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
    )


def _batch(
    dataset: SupervisedDataset, indexes: Tensor, device: torch.device
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    return (
        dataset.observations[indexes].to(device),
        dataset.legal_masks[indexes].to(device),
        dataset.search_policies[indexes].to(device),
        dataset.round_returns[indexes].to(device),
    )


def _training_epoch(
    model: PolicyValueModel,
    optimizer: torch.optim.AdamW,
    dataset: SupervisedDataset,
    batches: Sequence[Tensor],
    device: torch.device,
    gradient_norm: float,
    *,
    epoch: int,
    should_stop: Callable[[int, int], bool] | None = None,
) -> tuple[float, float, float, float]:
    model.train()
    totals = [0.0, 0.0, 0.0]
    examples = 0
    maximum_gradient = 0.0
    for batch_number, indexes in enumerate(batches):
        if should_stop is not None and should_stop(epoch, batch_number):
            raise SupervisedTrainingInterrupted("supervised optimization interrupted")
        observations, masks, policy_targets, value_targets = _batch(dataset, indexes, device)
        optimizer.zero_grad(set_to_none=True)
        logits, values = model(observations)
        losses = policy_value_loss(logits, values, masks, policy_targets, value_targets)
        if not all(
            torch.isfinite(value).all()
            for value in (logits, values, losses.total, losses.policy_cross_entropy, losses.value_mse)
        ):
            raise SupervisedTrainingError("non-finite training output or loss")
        losses.total.backward()
        if not _finite_parameters_and_gradients(model):
            raise SupervisedTrainingError("non-finite training gradient")
        clipped = torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_norm)
        if not torch.isfinite(clipped):
            raise SupervisedTrainingError("non-finite gradient norm")
        maximum_gradient = max(maximum_gradient, float(clipped.detach().cpu()))
        optimizer.step()
        if not _finite_parameters_and_gradients(model):
            raise SupervisedTrainingError("non-finite optimized parameter")
        size = len(indexes)
        totals[0] += float(losses.total.detach().cpu()) * size
        totals[1] += float(losses.policy_cross_entropy.detach().cpu()) * size
        totals[2] += float(losses.value_mse.detach().cpu()) * size
        examples += size
    return (*(value / examples for value in totals), maximum_gradient)


@torch.no_grad()
def _validation_epoch(
    model: PolicyValueModel,
    dataset: SupervisedDataset,
    batch_size: int,
    device: torch.device,
) -> tuple[float, float, float, float, float]:
    model.eval()
    totals = [0.0, 0.0, 0.0]
    correct = 0
    absolute_error = 0.0
    examples = 0
    for start in range(0, dataset.example_count, batch_size):
        indexes = torch.arange(start, min(start + batch_size, dataset.example_count))
        observations, masks, policy_targets, value_targets = _batch(dataset, indexes, device)
        logits, values = model(observations)
        losses = policy_value_loss(logits, values, masks, policy_targets, value_targets)
        if not all(torch.isfinite(value).all() for value in (logits, values, losses.total)):
            raise SupervisedTrainingError("non-finite validation output or loss")
        size = len(indexes)
        totals[0] += float(losses.total.cpu()) * size
        totals[1] += float(losses.policy_cross_entropy.cpu()) * size
        totals[2] += float(losses.value_mse.cpu()) * size
        masked = logits.masked_fill(~masks, -torch.inf).flatten(start_dim=1)
        correct += int(
            (masked.argmax(dim=1) == policy_targets.flatten(start_dim=1).argmax(dim=1))
            .sum()
            .cpu()
        )
        absolute_error += float(torch.abs(values - value_targets).sum().cpu())
        examples += size
    return (
        totals[0] / examples,
        totals[1] / examples,
        totals[2] / examples,
        correct / examples,
        absolute_error / examples,
    )


def benchmark_optimization_devices(
    config: SupervisedConfig,
    datasets: DatasetBundle,
) -> DeviceBenchmark:
    """Measure the contract's 3-warm-up/5-timed device selection profile."""

    first_observations = datasets.validation.observations[: min(16, datasets.validation.example_count)]
    cpu_reference = PolicyValueModel(
        run_root_seed=config.run.root_seed,
        model_id=config.model.model_id,
        initialization_ordinal=config.model.initialization_ordinal,
    )
    with torch.no_grad():
        reference_logits, reference_values = cpu_reference(first_observations)

    def measure(name: str) -> DeviceMeasurement:
        if name == "mps" and not torch.backends.mps.is_available():
            return DeviceMeasurement(
                name, False, False, False, None, None, None, None, "MPS unavailable"
            )
        device = torch.device(name)
        model = PolicyValueModel(
            run_root_seed=config.run.root_seed,
            model_id=config.model.model_id,
            initialization_ordinal=config.model.initialization_ordinal,
        ).to(device)
        optimizer = build_policy_value_optimizer(model, config.optimization.model_optimizer)
        try:
            with torch.no_grad():
                logits, values = model(first_observations.to(device))
            agreement = torch.allclose(
                logits.cpu(), reference_logits, atol=1e-5, rtol=1e-4
            ) and torch.allclose(values.cpu(), reference_values, atol=1e-5, rtol=1e-4)
            batches = epoch_minibatch_indices(config, datasets, 0)
            swap_before = _swap_used_bytes()
            times: list[float] = []
            finite = True
            for repetition in range(8):
                _synchronize(device)
                started = time.perf_counter()
                _training_epoch(
                    model,
                    optimizer,
                    datasets.training,
                    batches,
                    device,
                    config.optimization.gradient_norm,
                    epoch=0,
                )
                _synchronize(device)
                elapsed = time.perf_counter() - started
                if repetition >= 3:
                    times.append(elapsed)
                finite = finite and _finite_parameters_and_gradients(model)
            median = statistics.median(times)
            return DeviceMeasurement(
                name,
                True,
                finite,
                agreement,
                median,
                datasets.training.example_count / median,
                _peak_rss_bytes(),
                max(0, _swap_used_bytes() - swap_before),
            )
        except (RuntimeError, PolicyValueContractError, SupervisedTrainingError) as error:
            return DeviceMeasurement(
                name,
                True,
                False,
                False,
                None,
                None,
                _peak_rss_bytes(),
                None,
                type(error).__name__,
            )
        finally:
            del optimizer, model
            if name == "mps":
                torch.mps.empty_cache()

    cpu = measure("cpu")
    mps = measure("mps")
    selected = "cpu"
    if (
        mps.available
        and mps.finite
        and mps.output_agreement
        and mps.median_epoch_seconds is not None
        and cpu.median_epoch_seconds is not None
        and mps.median_epoch_seconds <= cpu.median_epoch_seconds * 0.9
        and (mps.peak_rss_bytes or 0) < 6 * 1024**3
        and (mps.swap_growth_bytes or 0) <= 512 * 1024**2
    ):
        selected = "mps"
    return DeviceBenchmark(selected, cpu, mps)


def _resolved_document(
    config: SupervisedConfig,
    dataset_digest: str,
    selected_device: str,
) -> dict[str, object]:
    body = {
        "format_version": SUPERVISED_CONFIG_FORMAT_VERSION,
        "run": {
            **asdict(config.run),
            "output_directory": str(config.output_path),
        },
        "dataset": {
            **asdict(config.dataset),
            "teacher_directory": str(config.teacher_path),
            "dataset_digest": dataset_digest,
        },
        "model": asdict(config.model),
        "optimization": {
            **asdict(config.optimization),
            "betas": list(config.optimization.betas),
            "resolved_device": selected_device,
        },
        "contracts": {
            "model_schema_version": MODEL_SCHEMA_VERSION,
            "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
            "action_schema_version": ACTION_SCHEMA_VERSION,
            "value_schema_version": VALUE_SCHEMA_VERSION,
            "optimizer_compatibility_version": OPTIMIZER_COMPATIBILITY_VERSION,
            "parameter_count": PARAMETER_COUNT,
        },
    }
    identity = json.loads(json.dumps(body))
    # Physical paths are locations, not training identities. Dataset content is
    # already bound by its digest, so moving a recoverable run preserves it.
    identity["run"]["output_directory"] = "<run-output>"
    identity["dataset"]["teacher_directory"] = "<teacher-dataset>"
    return {**body, "resolved_config_digest": _json_digest(identity)}


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
    encoded = _canonical_json(value)
    json.loads(encoded)
    _atomic_bytes(path, encoded)


def _atomic_text(path: Path, value: str) -> None:
    _atomic_bytes(path, value.encode("utf-8"))


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


def _checkpoint_payload(
    *,
    kind: str,
    resolved: Mapping[str, object],
    dataset_digest: str,
    model: PolicyValueModel,
    optimizer: torch.optim.AdamW,
    completed_epoch: int,
    best_epoch: int,
    best_validation_loss: float,
    stale_epochs: int,
    history: Sequence[EpochMetrics],
) -> dict[str, object]:
    return {
        "format_version": SUPERVISED_CHECKPOINT_FORMAT_VERSION,
        "kind": kind,
        "resolved_config_digest": resolved["resolved_config_digest"],
        "dataset_digest": dataset_digest,
        "selected_device": resolved["optimization"]["resolved_device"],  # type: ignore[index]
        "completed_epoch": completed_epoch,
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
        "stale_epochs": stale_epochs,
        "model_state_dict": _cpu_copy(model.state_dict()),
        "optimizer_state_dict": _cpu_copy(optimizer.state_dict()),
        "history": tuple(asdict(item) for item in history),
    }


def _validate_checkpoint_payload(
    payload: object,
    config: SupervisedConfig,
    resolved: Mapping[str, object],
    dataset_digest: str,
) -> dict[str, object]:
    expected_resolved = _resolved_document(
        config,
        dataset_digest,
        str(resolved["optimization"]["resolved_device"]),  # type: ignore[index]
    )
    if dict(resolved) != expected_resolved:
        raise SupervisedTrainingError("resolved configuration content or digest differs")
    if not isinstance(payload, dict) or set(payload) != _CHECKPOINT_KEYS:
        raise SupervisedTrainingError("checkpoint payload is invalid")
    if payload["format_version"] != SUPERVISED_CHECKPOINT_FORMAT_VERSION:
        raise SupervisedTrainingError("checkpoint format is incompatible")
    if payload["kind"] not in _CHECKPOINT_NAMES:
        raise SupervisedTrainingError("checkpoint kind is invalid")
    if (
        payload["resolved_config_digest"] != resolved["resolved_config_digest"]
        or payload["dataset_digest"] != dataset_digest
        or payload["selected_device"] != resolved["optimization"]["resolved_device"]  # type: ignore[index]
    ):
        raise SupervisedTrainingError("checkpoint identity differs")
    for key in ("completed_epoch", "best_epoch", "stale_epochs"):
        if type(payload[key]) is not int or int(payload[key]) < -1:
            raise SupervisedTrainingError("checkpoint epoch state is invalid")
    loss = payload["best_validation_loss"]
    if type(loss) is not float or not (math.isfinite(loss) or loss == math.inf):
        raise SupervisedTrainingError("checkpoint validation state is invalid")
    history = payload["history"]
    if not isinstance(history, tuple) or len(history) != int(payload["completed_epoch"]) + 1:
        raise SupervisedTrainingError("checkpoint history differs from its epoch")
    for item in history:
        if not isinstance(item, dict) or set(item) != set(EpochMetrics.__dataclass_fields__):
            raise SupervisedTrainingError("checkpoint epoch metrics are invalid")
        try:
            metrics = EpochMetrics(**item)
        except TypeError as error:
            raise SupervisedTrainingError("checkpoint epoch metrics are incompatible") from error
        if any(not math.isfinite(value) for key, value in asdict(metrics).items() if key != "epoch"):
            raise SupervisedTrainingError("checkpoint epoch metrics are non-finite")
    model = PolicyValueModel(
        run_root_seed=config.run.root_seed,
        model_id=config.model.model_id,
        initialization_ordinal=config.model.initialization_ordinal,
    )
    state = payload["model_state_dict"]
    if not isinstance(state, dict) or set(state) != set(model.state_dict()):
        raise SupervisedTrainingError("checkpoint model state is incompatible")
    for name, expected in model.state_dict().items():
        tensor = state[name]
        if (
            not isinstance(tensor, Tensor)
            or tensor.device.type != "cpu"
            or tensor.dtype is not expected.dtype
            or tensor.shape != expected.shape
            or not torch.isfinite(tensor).all()
        ):
            raise SupervisedTrainingError(f"checkpoint model tensor is invalid: {name}")
    try:
        model.load_state_dict(state, strict=True)
        optimizer = build_policy_value_optimizer(model, config.optimization.model_optimizer)
        optimizer.load_state_dict(payload["optimizer_state_dict"])  # type: ignore[arg-type]
    except (RuntimeError, ValueError, KeyError) as error:
        raise SupervisedTrainingError("checkpoint optimizer state is incompatible") from error
    for value in optimizer.state.values():
        if any(isinstance(item, Tensor) and not torch.isfinite(item).all() for item in value.values()):
            raise SupervisedTrainingError("checkpoint optimizer state is non-finite")
    return payload


def _atomic_checkpoint(
    path: Path,
    payload: dict[str, object],
    config: SupervisedConfig,
    resolved: Mapping[str, object],
    dataset_digest: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        torch.save(payload, temporary)
        loaded = torch.load(temporary, map_location="cpu", weights_only=True)
        _validate_checkpoint_payload(loaded, config, resolved, dataset_digest)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_supervised_checkpoint(
    path: str | Path,
    config: SupervisedConfig,
    resolved: Mapping[str, object],
    dataset_digest: str,
) -> dict[str, object]:
    try:
        payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, ValueError) as error:
        raise SupervisedTrainingError("checkpoint could not be loaded") from error
    return _validate_checkpoint_payload(payload, config, resolved, dataset_digest)


def _optimizer_to(optimizer: torch.optim.AdamW, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, Tensor):
                state[key] = value.to(device)


def _write_epoch_outputs(
    output: Path,
    resolved: Mapping[str, object],
    dataset_digest: str,
    metrics: EpochMetrics,
    elapsed_seconds: float,
) -> None:
    body = {
        "format_version": SUPERVISED_METRICS_FORMAT_VERSION,
        "resolved_config_digest": resolved["resolved_config_digest"],
        "dataset_digest": dataset_digest,
        **asdict(metrics),
        "elapsed_seconds": elapsed_seconds,
    }
    _atomic_json(output / "metrics" / f"{metrics.epoch:06d}.json", body)
    _atomic_text(
        output / "reports" / f"{metrics.epoch:06d}.md",
        "\n".join(
            (
                f"# Supervised epoch {metrics.epoch}",
                "",
                f"- Dataset digest: `{dataset_digest}`",
                f"- Training loss: {metrics.training_total_loss:.6f}",
                f"- Validation loss: {metrics.validation_total_loss:.6f}",
                f"- Validation policy accuracy: {metrics.validation_policy_accuracy:.3%}",
                f"- Validation value MAE: {metrics.validation_value_mae:.6f}",
                f"- Maximum pre-clipping gradient norm: {metrics.maximum_gradient_norm:.6f}",
                f"- Epoch time: {elapsed_seconds:.3f} seconds",
                "",
            )
        ),
    )


def _write_state(output: Path, resolved: Mapping[str, object], phase: str, **extra: object) -> None:
    _atomic_json(
        output / "state.json",
        {
            "format_version": SUPERVISED_STATE_FORMAT_VERSION,
            "resolved_config_digest": resolved["resolved_config_digest"],
            "phase": phase,
            **extra,
        },
    )


def _load_resolved(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SupervisedTrainingError("resolved configuration could not be read") from error
    if not isinstance(value, dict) or value.get("format_version") != SUPERVISED_CONFIG_FORMAT_VERSION:
        raise SupervisedTrainingError("resolved configuration is incompatible")
    return value


def _prepare(
    config: SupervisedConfig,
    datasets: DatasetBundle,
    *,
    resume: bool,
    benchmark: DeviceBenchmark | None,
) -> tuple[dict[str, object], DeviceBenchmark | None]:
    output = config.output_path
    resolved_path = output / "resolved-config.json"
    if resume:
        resolved = _load_resolved(resolved_path)
        expected = _resolved_document(
            config,
            datasets.dataset_digest,
            str(resolved["optimization"]["resolved_device"]),  # type: ignore[index]
        )
        if resolved != expected:
            raise SupervisedTrainingError("resolved configuration differs from the requested run")
        return resolved, None
    if resolved_path.exists():
        raise SupervisedTrainingError("output already contains a run; use resume")
    selected = config.optimization.device
    if selected == "mps" and not torch.backends.mps.is_available():
        raise SupervisedTrainingError("MPS was requested but is unavailable")
    if selected == "auto":
        benchmark = benchmark or benchmark_optimization_devices(config, datasets)
        selected = benchmark.selected_device
    resolved = _resolved_document(config, datasets.dataset_digest, selected)
    _atomic_json(resolved_path, resolved)
    if benchmark is not None:
        _atomic_json(output / "device-benchmark.json", asdict(benchmark))
    return resolved, benchmark


def train_supervised(
    config: SupervisedConfig,
    *,
    resume: bool = False,
    should_stop: Callable[[int, int], bool] | None = None,
    benchmark: DeviceBenchmark | None = None,
    progress: Callable[[EpochMetrics], None] | None = None,
) -> TrainingResult:
    datasets = load_supervised_datasets(config.teacher_path)
    resolved, _ = _prepare(config, datasets, resume=resume, benchmark=benchmark)
    output = config.output_path
    checkpoints = output / "checkpoints"
    device = torch.device(str(resolved["optimization"]["resolved_device"]))  # type: ignore[index]
    model = PolicyValueModel(
        run_root_seed=config.run.root_seed,
        model_id=config.model.model_id,
        initialization_ordinal=config.model.initialization_ordinal,
    ).to(device)
    optimizer = build_policy_value_optimizer(model, config.optimization.model_optimizer)
    history: list[EpochMetrics] = []
    completed_epoch = -1
    best_epoch = -1
    best_validation_loss = math.inf
    stale_epochs = 0

    if resume:
        latest = checkpoints / _CHECKPOINT_NAMES["epoch-latest"]
        start = checkpoints / _CHECKPOINT_NAMES["iteration-start"]
        payload = load_supervised_checkpoint(
            latest if latest.exists() else start,
            config,
            resolved,
            datasets.dataset_digest,
        )
        model.load_state_dict(payload["model_state_dict"], strict=True)  # type: ignore[arg-type]
        optimizer.load_state_dict(payload["optimizer_state_dict"])  # type: ignore[arg-type]
        _optimizer_to(optimizer, device)
        completed_epoch = int(payload["completed_epoch"])
        best_epoch = int(payload["best_epoch"])
        best_validation_loss = float(payload["best_validation_loss"])
        stale_epochs = int(payload["stale_epochs"])
        history = [EpochMetrics(**item) for item in payload["history"]]  # type: ignore[arg-type]
    else:
        start_payload = _checkpoint_payload(
            kind="iteration-start",
            resolved=resolved,
            dataset_digest=datasets.dataset_digest,
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
            datasets.dataset_digest,
        )
    _write_state(output, resolved, "optimizing", completed_epoch=completed_epoch)

    stopped_early = False
    try:
        for epoch in range(completed_epoch + 1, config.optimization.maximum_epochs):
            started = time.perf_counter()
            train_total, train_policy, train_value, maximum_gradient = _training_epoch(
                model,
                optimizer,
                datasets.training,
                epoch_minibatch_indices(config, datasets, epoch),
                device,
                config.optimization.gradient_norm,
                epoch=epoch,
                should_stop=should_stop,
            )
            validation = _validation_epoch(
                model,
                datasets.validation,
                config.optimization.batch_size,
                device,
            )
            metrics = EpochMetrics(
                epoch,
                train_total,
                train_policy,
                train_value,
                validation[0],
                validation[1],
                validation[2],
                validation[3],
                validation[4],
                maximum_gradient,
            )
            if any(not math.isfinite(value) for key, value in asdict(metrics).items() if key != "epoch"):
                raise SupervisedTrainingError("non-finite epoch metric")
            improved = validation[0] < best_validation_loss - config.optimization.minimum_improvement
            if improved:
                best_epoch = epoch
                best_validation_loss = validation[0]
                stale_epochs = 0
            else:
                stale_epochs += 1
            history.append(metrics)
            completed_epoch = epoch
            latest_payload = _checkpoint_payload(
                kind="epoch-latest",
                resolved=resolved,
                dataset_digest=datasets.dataset_digest,
                model=model,
                optimizer=optimizer,
                completed_epoch=completed_epoch,
                best_epoch=best_epoch,
                best_validation_loss=best_validation_loss,
                stale_epochs=stale_epochs,
                history=history,
            )
            if improved:
                best_payload = dict(latest_payload)
                best_payload["kind"] = "best-validation"
                _atomic_checkpoint(
                    checkpoints / _CHECKPOINT_NAMES["best-validation"],
                    best_payload,
                    config,
                    resolved,
                    datasets.dataset_digest,
                )
            # The resume pointer moves last. If a crash separates the two
            # writes, the prior pointer deterministically replays this epoch.
            _atomic_checkpoint(
                checkpoints / _CHECKPOINT_NAMES["epoch-latest"],
                latest_payload,
                config,
                resolved,
                datasets.dataset_digest,
            )
            _write_epoch_outputs(
                output,
                resolved,
                datasets.dataset_digest,
                metrics,
                time.perf_counter() - started,
            )
            _write_state(output, resolved, "optimizing", completed_epoch=completed_epoch)
            if progress is not None:
                progress(metrics)
            if (
                completed_epoch + 1 >= config.optimization.minimum_epochs
                and stale_epochs >= config.optimization.early_stop_patience
            ):
                stopped_early = (
                    completed_epoch + 1
                    < config.optimization.maximum_epochs
                )
                break
    except SupervisedTrainingInterrupted:
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

    best = load_supervised_checkpoint(
        checkpoints / _CHECKPOINT_NAMES["best-validation"],
        config,
        resolved,
        datasets.dataset_digest,
    )
    final = dict(best)
    final["kind"] = "final"
    _atomic_checkpoint(
        checkpoints / _CHECKPOINT_NAMES["final"],
        final,
        config,
        resolved,
        datasets.dataset_digest,
    )
    result = TrainingResult(
        str(output),
        datasets.dataset_digest,
        device.type,
        completed_epoch + 1,
        best_epoch,
        best_validation_loss,
        stopped_early,
        str(checkpoints / _CHECKPOINT_NAMES["final"]),
    )
    _write_state(output, resolved, "complete", **asdict(result))
    _atomic_json(
        output / "metrics" / "summary.json",
        {"format_version": SUPERVISED_METRICS_FORMAT_VERSION, **asdict(result)},
    )
    _atomic_text(
        output / "reports" / "summary.md",
        "\n".join(
            (
                f"# Supervised run {config.run.run_id}",
                "",
                f"- Device: {result.selected_device}",
                f"- Completed epochs: {result.completed_epochs}",
                f"- Best epoch: {result.best_epoch}",
                f"- Best validation loss: {result.best_validation_loss:.6f}",
                f"- Training examples: {datasets.training.example_count}",
                f"- Validation examples: {datasets.validation.example_count}",
                "",
            )
        ),
    )
    return result


def validate_supervised(
    config: SupervisedConfig,
    checkpoint: str | Path | None = None,
) -> dict[str, float | int | str]:
    datasets = load_supervised_datasets(config.teacher_path)
    resolved = _load_resolved(config.output_path / "resolved-config.json")
    path = Path(checkpoint) if checkpoint else config.output_path / "checkpoints" / "final.pt"
    payload = load_supervised_checkpoint(path, config, resolved, datasets.dataset_digest)
    model = PolicyValueModel(
        run_root_seed=config.run.root_seed,
        model_id=config.model.model_id,
        initialization_ordinal=config.model.initialization_ordinal,
    )
    model.load_state_dict(payload["model_state_dict"], strict=True)  # type: ignore[arg-type]
    values = _validation_epoch(model, datasets.validation, config.optimization.batch_size, torch.device("cpu"))
    return {
        "checkpoint": str(path.resolve()),
        "examples": datasets.validation.example_count,
        "total_loss": values[0],
        "policy_cross_entropy": values[1],
        "value_mse": values[2],
        "policy_accuracy": values[3],
        "value_mae": values[4],
    }


def export_supervised(
    config: SupervisedConfig,
    destination: str | Path,
    checkpoint: str | Path | None = None,
) -> Path:
    datasets = load_supervised_datasets(config.teacher_path)
    resolved = _load_resolved(config.output_path / "resolved-config.json")
    path = Path(checkpoint) if checkpoint else config.output_path / "checkpoints" / "final.pt"
    payload = load_supervised_checkpoint(path, config, resolved, datasets.dataset_digest)
    model = PolicyValueModel(
        run_root_seed=config.run.root_seed,
        model_id=config.model.model_id,
        initialization_ordinal=config.model.initialization_ordinal,
    )
    model.load_state_dict(payload["model_state_dict"], strict=True)  # type: ignore[arg-type]
    output = Path(destination).expanduser().resolve()
    save_policy_value_artifact(
        output,
        model,
        source_revision=config.run.source_revision,
        training_configuration=resolved,
        dataset_digest=datasets.dataset_digest,
        search_report_digest=config.dataset.search_report_digest,
    )
    load_policy_value_artifact(output)
    return output


def load_config_from_run(output_directory: str | Path) -> SupervisedConfig:
    resolved = _load_resolved(Path(output_directory).expanduser().resolve() / "resolved-config.json")
    run = resolved.get("run")
    dataset = resolved.get("dataset")
    model = resolved.get("model")
    optimization = resolved.get("optimization")
    if not all(isinstance(value, dict) for value in (run, dataset, model, optimization)):
        raise SupervisedTrainingError("resolved configuration sections are invalid")
    assert isinstance(run, dict) and isinstance(dataset, dict)
    assert isinstance(model, dict) and isinstance(optimization, dict)
    try:
        config = SupervisedConfig(
            RunConfig(
                str(run["run_id"]), str(run["root_seed"]), str(run["output_directory"]), str(run["source_revision"])
            ),
            DatasetConfig(str(dataset["teacher_directory"]), str(dataset["search_report_digest"])),
            ModelConfig(str(model["model_id"]), int(model["initialization_ordinal"])),
            OptimizationConfig(
                device=str(optimization["device"]),
                batch_size=int(optimization["batch_size"]),
                learning_rate=float(optimization["learning_rate"]),
                betas=tuple(float(value) for value in optimization["betas"]),  # type: ignore[arg-type]
                epsilon=float(optimization["epsilon"]),
                weight_decay=float(optimization["weight_decay"]),
                gradient_norm=float(optimization["gradient_norm"]),
                minimum_epochs=int(optimization["minimum_epochs"]),
                maximum_epochs=int(optimization["maximum_epochs"]),
                early_stop_patience=int(optimization["early_stop_patience"]),
                minimum_improvement=float(optimization["minimum_improvement"]),
            ),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise SupervisedTrainingError("resolved configuration is incomplete") from error
    expected = _resolved_document(
        config,
        _require_digest(dataset.get("dataset_digest"), "dataset digest"),
        str(optimization.get("resolved_device")),
    )
    if resolved != expected:
        raise SupervisedTrainingError("resolved configuration content or digest differs")
    return config


def _print_progress(metrics: EpochMetrics) -> None:
    print(
        "supervised_training "
        f"epoch={metrics.epoch} "
        f"train_loss={metrics.training_total_loss:.6f} "
        f"validation_loss={metrics.validation_total_loss:.6f}",
        flush=True,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="dracula-supervised")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("smoke", "train"):
        command = subparsers.add_parser(name)
        command.add_argument("--config", required=True)
    resume = subparsers.add_parser("resume")
    resume.add_argument("--output", required=True)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--output", required=True)
    validate.add_argument("--checkpoint")
    export = subparsers.add_parser("export")
    export.add_argument("--output", required=True)
    export.add_argument("--destination", required=True)
    export.add_argument("--checkpoint")
    args = parser.parse_args(argv)
    if args.command in {"smoke", "train"}:
        result = train_supervised(load_supervised_config(args.config), progress=_print_progress)
        print(json.dumps(asdict(result), indent=2, sort_keys=True))
    elif args.command == "resume":
        config = load_config_from_run(args.output)
        result = train_supervised(config, resume=True, progress=_print_progress)
        print(json.dumps(asdict(result), indent=2, sort_keys=True))
    elif args.command == "validate":
        print(
            json.dumps(
                validate_supervised(load_config_from_run(args.output), args.checkpoint),
                indent=2,
                sort_keys=True,
            )
        )
    else:
        destination = export_supervised(
            load_config_from_run(args.output), args.destination, args.checkpoint
        )
        print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = (
    "DatasetBundle",
    "DatasetConfig",
    "DeviceBenchmark",
    "DeviceMeasurement",
    "EpochMetrics",
    "ModelConfig",
    "OptimizationConfig",
    "RunConfig",
    "SupervisedConfig",
    "SupervisedDataset",
    "SupervisedTrainingError",
    "SupervisedTrainingInterrupted",
    "TrainingResult",
    "benchmark_optimization_devices",
    "epoch_minibatch_indices",
    "export_supervised",
    "load_config_from_run",
    "load_supervised_checkpoint",
    "load_supervised_config",
    "load_supervised_datasets",
    "main",
    "train_supervised",
    "validate_supervised",
)

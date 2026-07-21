"""TOML configuration resolution for reproducible local training runs."""

from __future__ import annotations

import json
import math
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from dracula.cards import CARD_SCHEMA_VERSION
from dracula.collection import COLLECTION_FORMAT_VERSION
from dracula.engine import ENGINE_VERSION, RULES_VERSION
from dracula.models import Critic, Policy
from dracula.randomness import RANDOMNESS_SCHEMA_VERSION

TRAINING_FORMAT_VERSION = "dracula-training-v1"
CHECKPOINT_FORMAT_VERSION = "dracula-checkpoint-v1"
EVALUATION_FORMAT_VERSION = "dracula-evaluation-v1"
OBSERVATION_SCHEMA_VERSION = "dracula-observation-v1"
ACTION_MAP_VERSION = "dracula-action-map-v1"
POLICY_ARCHITECTURE_VERSION = "dracula-policy-v1"
CRITIC_ARCHITECTURE_VERSION = "dracula-critic-v1"


class TrainingConfigurationError(ValueError):
    """A training configuration cannot be resolved to one immutable manifest."""


@dataclass(frozen=True, slots=True)
class RunSettings:
    run_id: str
    root_seed: str
    iteration_count: int
    output_directory: str
    profile: str
    retain_collections: bool


@dataclass(frozen=True, slots=True)
class PopulationSettings:
    policy_ids: tuple[str, ...]
    policy_checkpoints: tuple[tuple[str, str], ...]
    critic_checkpoint: str | None


@dataclass(frozen=True, slots=True)
class FixtureSettings:
    collection_lane_roots: tuple[str, ...]
    collection_generations: int
    held_out_lane_roots: tuple[str, ...]
    held_out_generations: int
    next_collection_counter: int
    next_evaluation_counter: int


@dataclass(frozen=True, slots=True)
class ComputeSettings:
    collection_device: str
    optimization_device: str
    trajectory_batch_size: int
    numerical_atol: float
    numerical_rtol: float


@dataclass(frozen=True, slots=True)
class TrainingSettings:
    actor_learning_rate: float
    critic_learning_rate: float
    adam_beta1: float
    adam_beta2: float
    adam_epsilon: float
    gradient_norm_limit: float
    ppo_clip: float
    approximate_kl_limit: float
    illegal_probability_threshold: float
    burn_in_entropy_coefficient: float
    trained_entropy_coefficient: float
    critic_validation_improvement: float
    critic_validation_consecutive_windows: int
    critic_minimum_collected_rounds: int
    actor_ramp_collected_rounds: int
    actor_weight_override: float | None


@dataclass(frozen=True, slots=True)
class VersionSettings:
    rules: str = RULES_VERSION
    cards: str = CARD_SCHEMA_VERSION
    engine: str = ENGINE_VERSION
    randomness: str = RANDOMNESS_SCHEMA_VERSION
    observation: str = OBSERVATION_SCHEMA_VERSION
    action_map: str = ACTION_MAP_VERSION
    policy: str = POLICY_ARCHITECTURE_VERSION
    critic: str = CRITIC_ARCHITECTURE_VERSION
    collection: str = COLLECTION_FORMAT_VERSION
    checkpoint: str = CHECKPOINT_FORMAT_VERSION
    evaluation: str = EVALUATION_FORMAT_VERSION
    training: str = TRAINING_FORMAT_VERSION


@dataclass(frozen=True, slots=True)
class ResolvedTrainingConfig:
    run: RunSettings
    population: PopulationSettings
    fixtures: FixtureSettings
    compute: ComputeSettings
    training: TrainingSettings
    versions: VersionSettings = VersionSettings()

    @property
    def run_directory(self) -> Path:
        return Path(self.run.output_directory) / self.run.run_id

    def manifest(self) -> dict[str, object]:
        value = asdict(self)
        value["population"]["policy_checkpoints"] = dict(  # type: ignore[index]
            self.population.policy_checkpoints
        )
        return value

    @classmethod
    def from_manifest(cls, value: object) -> ResolvedTrainingConfig:
        if not isinstance(value, dict):
            raise TrainingConfigurationError("resolved manifest must be an object")
        try:
            run = RunSettings(**value["run"])
            population_data = dict(value["population"])
            population_data["policy_ids"] = tuple(population_data["policy_ids"])
            checkpoints = population_data.get("policy_checkpoints", {})
            population_data["policy_checkpoints"] = tuple(sorted(checkpoints.items()))
            population = PopulationSettings(**population_data)
            fixtures_data = dict(value["fixtures"])
            fixtures_data["collection_lane_roots"] = tuple(
                fixtures_data["collection_lane_roots"]
            )
            fixtures_data["held_out_lane_roots"] = tuple(
                fixtures_data["held_out_lane_roots"]
            )
            fixtures = FixtureSettings(**fixtures_data)
            compute = ComputeSettings(**value["compute"])
            training = TrainingSettings(**value["training"])
            versions = VersionSettings(**value["versions"])
        except (KeyError, TypeError, ValueError, AttributeError) as error:
            raise TrainingConfigurationError("resolved manifest is invalid") from error
        resolved = cls(run, population, fixtures, compute, training, versions)
        _validate_resolved(resolved)
        return resolved


def load_training_config(path: str | Path) -> ResolvedTrainingConfig:
    config_path = Path(path)
    try:
        with config_path.open("rb") as source:
            raw = tomllib.load(source)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise TrainingConfigurationError("training TOML could not be read") from error
    if not isinstance(raw, dict):
        raise TrainingConfigurationError("training TOML must contain tables")
    return _resolve(raw, base_directory=config_path.parent.resolve())


def canonical_manifest_bytes(config: ResolvedTrainingConfig) -> bytes:
    return json.dumps(
        config.manifest(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _resolve(raw: dict[str, Any], *, base_directory: Path) -> ResolvedTrainingConfig:
    _reject_unknown_tables(raw)
    run_data = _table(raw, "run")
    profile = run_data.get("profile", "full")
    if profile not in {"full", "smoke"}:
        raise TrainingConfigurationError("run profile must be 'full' or 'smoke'")
    smoke = profile == "smoke"

    run_id = run_data.get("run_id")
    root_seed = run_data.get("root_seed")
    _identifier(run_id, "run ID")
    _identifier(root_seed, "root seed")
    output_value = run_data.get("output_directory", "runs")
    if not isinstance(output_value, str) or not output_value:
        raise TrainingConfigurationError("output directory must be a nonempty path")
    output_path = Path(output_value).expanduser()
    if not output_path.is_absolute():
        output_path = base_directory / output_path
    run = RunSettings(
        run_id=run_id,
        root_seed=root_seed,
        iteration_count=_positive_int(run_data.get("iteration_count", 1), "iteration count"),
        output_directory=str(output_path.resolve()),
        profile=profile,
        retain_collections=_boolean(
            run_data.get("retain_collections", False), "retain collections"
        ),
    )

    population_data = _table(raw, "population")
    default_policy_count = 2 if smoke else 5
    policy_ids = tuple(
        population_data.get(
            "policy_ids", [f"policy-{index}" for index in range(default_policy_count)]
        )
    )
    if len(policy_ids) != default_policy_count or len(set(policy_ids)) != len(policy_ids):
        raise TrainingConfigurationError(
            f"{profile} profile requires {default_policy_count} distinct policies"
        )
    for policy_id in policy_ids:
        _identifier(policy_id, "policy ID")
    checkpoint_data = population_data.get("policy_checkpoints", {})
    if not isinstance(checkpoint_data, dict) or not set(checkpoint_data).issubset(policy_ids):
        raise TrainingConfigurationError("policy checkpoints must be keyed by active policy ID")
    policy_checkpoints = tuple(
        sorted(
            (policy_id, _resolved_optional_path(value, base_directory, "policy checkpoint"))
            for policy_id, value in checkpoint_data.items()
        )
    )
    critic_checkpoint_value = population_data.get("critic_checkpoint")
    population = PopulationSettings(
        policy_ids=policy_ids,
        policy_checkpoints=policy_checkpoints,
        critic_checkpoint=(
            None
            if critic_checkpoint_value is None
            else _resolved_optional_path(
                critic_checkpoint_value, base_directory, "critic checkpoint"
            )
        ),
    )

    fixture_data = _table(raw, "fixtures")
    collection_count = 1 if smoke else 12
    held_out_count = 1 if smoke else 12
    collection_roots = tuple(
        fixture_data.get(
            "collection_lane_roots",
            [f"collection-lane-{index:02d}" for index in range(collection_count)],
        )
    )
    held_out_roots = tuple(
        fixture_data.get(
            "held_out_lane_roots",
            [f"held-out-lane-{index:02d}" for index in range(held_out_count)],
        )
    )
    requested_collection_count = fixture_data.get(
        "collection_lane_count", collection_count
    )
    requested_held_out_count = fixture_data.get("held_out_lane_count", held_out_count)
    if requested_collection_count != len(collection_roots):
        raise TrainingConfigurationError(
            "collection lane count does not match configured roots"
        )
    if requested_held_out_count != len(held_out_roots):
        raise TrainingConfigurationError("held-out lane count does not match configured roots")
    if len(set(collection_roots)) != len(collection_roots) or len(set(held_out_roots)) != len(
        held_out_roots
    ):
        raise TrainingConfigurationError("lane roots must be distinct within each fixture set")
    if set(collection_roots) & set(held_out_roots):
        raise TrainingConfigurationError("collection and held-out lane roots must be disjoint")
    for root in (*collection_roots, *held_out_roots):
        _identifier(root, "lane root")
    fixtures = FixtureSettings(
        collection_lane_roots=collection_roots,
        collection_generations=_positive_int(
            fixture_data.get("collection_generations", 1 if smoke else 4),
            "collection generations",
        ),
        held_out_lane_roots=held_out_roots,
        held_out_generations=_positive_int(
            fixture_data.get("held_out_generations", 1), "held-out generations"
        ),
        next_collection_counter=_nonnegative_int(
            fixture_data.get("next_collection_counter", 0),
            "next collection counter",
        ),
        next_evaluation_counter=_nonnegative_int(
            fixture_data.get("next_evaluation_counter", 0),
            "next evaluation counter",
        ),
    )

    compute_data = _table(raw, "compute")
    collection_requested = compute_data.get("collection_device", "cpu")
    if collection_requested not in {"cpu", "auto"}:
        raise TrainingConfigurationError("collection device must resolve to CPU")
    optimization_requested = compute_data.get(
        "optimization_device", "cpu"
    )
    optimization_device = _resolve_optimization_device(optimization_requested)
    compute = ComputeSettings(
        collection_device="cpu",
        optimization_device=optimization_device,
        trajectory_batch_size=_bounded_int(
            compute_data.get("trajectory_batch_size", 1 if smoke else 16),
            "trajectory batch size",
            1,
            64,
        ),
        numerical_atol=_positive_float(
            compute_data.get("numerical_atol", 2e-4), "numerical absolute tolerance"
        ),
        numerical_rtol=_positive_float(
            compute_data.get("numerical_rtol", 2e-4), "numerical relative tolerance"
        ),
    )

    training_data = _table(raw, "training")
    override = training_data.get("actor_weight_override", 1.0 if smoke else None)
    if override is not None:
        override = _unit_float(override, "actor weight override")
        if not smoke:
            raise TrainingConfigurationError(
                "actor weight override is available only in the smoke profile"
            )
    training = TrainingSettings(
        actor_learning_rate=_nonnegative_float(
            training_data.get("actor_learning_rate", 3e-4), "actor learning rate"
        ),
        critic_learning_rate=_nonnegative_float(
            training_data.get("critic_learning_rate", 1e-3), "critic learning rate"
        ),
        adam_beta1=_fraction_float(training_data.get("adam_beta1", 0.9), "Adam beta1"),
        adam_beta2=_fraction_float(
            training_data.get("adam_beta2", 0.999), "Adam beta2"
        ),
        adam_epsilon=_positive_float(
            training_data.get("adam_epsilon", 1e-8), "Adam epsilon"
        ),
        gradient_norm_limit=_positive_float(
            training_data.get("gradient_norm_limit", 0.5), "gradient norm limit"
        ),
        ppo_clip=_fraction_float(training_data.get("ppo_clip", 0.2), "PPO clip"),
        approximate_kl_limit=_nonnegative_float(
            training_data.get("approximate_kl_limit", 0.015),
            "approximate KL limit",
        ),
        illegal_probability_threshold=_nonnegative_float(
            training_data.get("illegal_probability_threshold", 0.001),
            "illegal probability threshold",
        ),
        burn_in_entropy_coefficient=_nonnegative_float(
            training_data.get("burn_in_entropy_coefficient", 0.005),
            "burn-in entropy coefficient",
        ),
        trained_entropy_coefficient=_nonnegative_float(
            training_data.get("trained_entropy_coefficient", 0.001),
            "trained entropy coefficient",
        ),
        critic_validation_improvement=_unit_float(
            training_data.get("critic_validation_improvement", 0.05),
            "critic validation improvement",
        ),
        critic_validation_consecutive_windows=_positive_int(
            training_data.get("critic_validation_consecutive_windows", 3),
            "critic validation consecutive windows",
        ),
        critic_minimum_collected_rounds=_positive_int(
            training_data.get("critic_minimum_collected_rounds", 10_000),
            "critic minimum collected rounds",
        ),
        actor_ramp_collected_rounds=_positive_int(
            training_data.get("actor_ramp_collected_rounds", 10_000),
            "actor ramp collected rounds",
        ),
        actor_weight_override=override,
    )
    resolved = ResolvedTrainingConfig(run, population, fixtures, compute, training)
    _validate_resolved(resolved)
    return resolved


def _validate_resolved(config: ResolvedTrainingConfig) -> None:
    if config.versions != VersionSettings():
        raise TrainingConfigurationError("resolved contract versions do not match runtime")
    if config.compute.collection_device != "cpu":
        raise TrainingConfigurationError("collection must run on CPU")
    if config.compute.optimization_device not in {"cpu", "mps"}:
        raise TrainingConfigurationError("optimization device must be CPU or MPS")
    if config.run.profile == "smoke" and (
        len(config.population.policy_ids),
        len(config.fixtures.collection_lane_roots),
        config.fixtures.collection_generations,
        len(config.fixtures.held_out_lane_roots),
        config.fixtures.held_out_generations,
        config.compute.trajectory_batch_size,
    ) != (2, 1, 1, 1, 1, 1):
        raise TrainingConfigurationError(
            "smoke profile dimensions do not match their fixed contract"
        )


def _resolve_optimization_device(value: object) -> str:
    if value == "cpu":
        return "cpu"
    if value == "mps":
        if not _mps_operation_probe():
            raise TrainingConfigurationError("MPS was requested but its operation probe failed")
        return "mps"
    if value == "auto":
        return "mps" if _mps_operation_probe() else "cpu"
    raise TrainingConfigurationError("optimization device must be cpu, mps, or auto")


def _mps_operation_probe() -> bool:
    if not torch.backends.mps.is_available():
        return False
    try:
        policy = Policy(seed=0).to("mps")
        critic = Critic(seed=0).to("mps")
        observation = torch.zeros(875, dtype=torch.bool, device="mps")
        mask = torch.zeros(4, 8, dtype=torch.bool, device="mps")
        mask[0, 0] = True
        logits, hidden = policy(observation, mask, policy.initial_hidden())
        value = critic(observation)
        (logits.mean() + hidden.mean() + value).backward()
        return all(
            parameter.grad is not None and torch.isfinite(parameter.grad).all()
            for parameter in (*policy.parameters(), *critic.parameters())
        )
    except (RuntimeError, NotImplementedError):
        return False


def _reject_unknown_tables(raw: dict[str, Any]) -> None:
    unknown = set(raw) - {"run", "population", "fixtures", "compute", "training"}
    if unknown:
        raise TrainingConfigurationError(
            f"unknown top-level configuration tables: {', '.join(sorted(unknown))}"
        )


def _table(raw: dict[str, Any], name: str) -> dict[str, Any]:
    value = raw.get(name, {})
    if not isinstance(value, dict):
        raise TrainingConfigurationError(f"{name} must be a TOML table")
    return value


def _identifier(value: object, label: str) -> None:
    if not isinstance(value, str) or not value or "\0" in value:
        raise TrainingConfigurationError(f"{label} must be nonempty text without NUL")


def _positive_int(value: object, label: str) -> int:
    if type(value) is not int or value < 1:
        raise TrainingConfigurationError(f"{label} must be a positive integer")
    return value


def _nonnegative_int(value: object, label: str) -> int:
    if type(value) is not int or value < 0:
        raise TrainingConfigurationError(f"{label} must be a non-negative integer")
    return value


def _bounded_int(value: object, label: str, lower: int, upper: int) -> int:
    if type(value) is not int or not lower <= value <= upper:
        raise TrainingConfigurationError(f"{label} must be between {lower} and {upper}")
    return value


def _number(value: object, label: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TrainingConfigurationError(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise TrainingConfigurationError(f"{label} must be finite")
    return number


def _positive_float(value: object, label: str) -> float:
    number = _number(value, label)
    if number <= 0:
        raise TrainingConfigurationError(f"{label} must be positive")
    return number


def _nonnegative_float(value: object, label: str) -> float:
    number = _number(value, label)
    if number < 0:
        raise TrainingConfigurationError(f"{label} must be non-negative")
    return number


def _unit_float(value: object, label: str) -> float:
    number = _number(value, label)
    if not 0 <= number <= 1:
        raise TrainingConfigurationError(f"{label} must be between zero and one")
    return number


def _fraction_float(value: object, label: str) -> float:
    number = _number(value, label)
    if not 0 < number < 1:
        raise TrainingConfigurationError(f"{label} must be between zero and one")
    return number


def _boolean(value: object, label: str) -> bool:
    if type(value) is not bool:
        raise TrainingConfigurationError(f"{label} must be Boolean")
    return value


def _resolved_optional_path(value: object, base_directory: Path, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise TrainingConfigurationError(f"{label} must be a nonempty path")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_directory / path
    return str(path.resolve())

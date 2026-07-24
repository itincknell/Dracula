"""Recoverable single-model expert iteration over guided-search self-play."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import resource
import statistics
import sys
import time
import tomllib
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import Tensor

from dracula.bridge import ACTION_COUNT, PolicyTurnKind, build_policy_turn_context
from dracula.engine import (
    EnginePlayer,
    EngineStatus,
    advance_after_round,
    apply_move,
    create_game,
    other_player,
)
from dracula.policy_value import (
    ACTION_SCHEMA_VERSION,
    MODEL_SCHEMA_VERSION,
    OBSERVATION_SCHEMA_VERSION,
    PARAMETER_COUNT,
    VALUE_SCHEMA_VERSION,
    PolicyValueModel,
    PolicyValueOptimizationConfig,
    build_policy_value_optimizer,
    load_policy_value_artifact,
    save_policy_value_artifact,
)
from dracula.randomness import Sha256CounterStream, derive_seed, seed_hex
from dracula.search import (
    GUIDED_SEARCH_SCHEMA_VERSION,
    INFORMATION_STATE_SCHEMA_VERSION,
    GuidedInformationSetSearch,
    GuidedSearchConfig,
    PolicyValueModelEvaluator,
    SearchInterrupted,
    derive_search_request_seed,
    information_state_fingerprint,
    information_state_from_engine,
    policy_input_from_information_state,
)
from dracula.supervised import (
    DatasetBundle,
    DeviceBenchmark,
    OptimizationConfig,
    SupervisedConfig,
    SupervisedDataset,
    _cpu_copy,
    _optimizer_to,
    _training_epoch,
    _validation_epoch,
    benchmark_optimization_devices,
    load_supervised_datasets,
)
from dracula.teacher import FixtureSplit

EXPERT_CONFIG_VERSION = "dracula-expert-config-v1"
EXPERT_STATE_VERSION = "dracula-expert-state-v1"
EXPERT_SHARD_VERSION = "dracula-expert-shard-v1"
EXPERT_MANIFEST_VERSION = "dracula-expert-manifest-v1"
EXPERT_CHECKPOINT_VERSION = "dracula-expert-checkpoint-v1"
EXPERT_ACCEPTED_VERSION = "dracula-expert-accepted-v1"
EXPERT_METRICS_VERSION = "dracula-expert-metrics-v1"
EXPERT_FIXTURE_NAMESPACE = "dracula-search-fixture-v1"
EXPERT_FIXTURE_ID_NAMESPACE = "dracula-expert-fixture-id-v1"
EXPERT_ACTION_NAMESPACE = "dracula-self-play-action-v1"
EXPERT_REPLAY_NAMESPACE = "dracula-replay-sampling-v1"
MAX_WORKERS = 4
ROWS_PER_GAME = 42

_FORBIDDEN_KEYS = {
    "authoritative_state",
    "determinization",
    "engine_seed",
    "game_seed",
    "hands",
    "opponent_hand",
    "policy_hidden",
    "search_tree",
    "stock",
}


class ExpertIterationError(ValueError):
    """An expert-iteration contract or artifact is invalid."""


class ExpertIterationInterrupted(RuntimeError):
    """A phase stopped before its commit point."""


@dataclass(frozen=True, slots=True)
class ExpertRunConfig:
    run_id: str
    root_seed: str
    output_directory: str
    source_revision: str
    maximum_iterations: int


@dataclass(frozen=True, slots=True)
class AcceptedConfig:
    artifact_path: str


@dataclass(frozen=True, slots=True)
class ExpertTeacherConfig:
    directory: str
    search_report_digest: str


@dataclass(frozen=True, slots=True)
class ExpertCollectionConfig:
    training_games: int = 108
    validation_games: int = 12
    workers: int = 4
    simulation_budget: int = 100
    puct_constant: float = 1.5
    early_placement_count: int = 4

    @property
    def search_config(self) -> GuidedSearchConfig:
        return GuidedSearchConfig(
            simulation_budget=self.simulation_budget,
            puct_constant=self.puct_constant,
        )


@dataclass(frozen=True, slots=True)
class ReplayConfig:
    accepted_iterations: int = 5
    examples_per_epoch: int = 9_072


@dataclass(frozen=True, slots=True)
class ExpertEvaluationConfig:
    candidate_budget: int = 100
    random_pairs: int = 12
    ppo_pairs: int = 12
    search_pairs: int = 60
    prior_pairs: int = 60
    search_control_budget: int = 500
    bootstrap_samples: int = 2_000
    workers: int = 4
    strategic_repetitions: int = 1


@dataclass(frozen=True, slots=True)
class ExpertControlsConfig:
    ppo_archive: str


@dataclass(frozen=True, slots=True)
class ExpertConfig:
    run: ExpertRunConfig
    accepted: AcceptedConfig
    teacher: ExpertTeacherConfig
    collection: ExpertCollectionConfig
    replay: ReplayConfig
    optimization: OptimizationConfig
    evaluation: ExpertEvaluationConfig
    controls: ExpertControlsConfig

    @property
    def output_path(self) -> Path:
        return Path(self.run.output_directory).expanduser().resolve()

    @property
    def accepted_artifact_path(self) -> Path:
        return Path(self.accepted.artifact_path).expanduser().resolve()

    @property
    def teacher_path(self) -> Path:
        return Path(self.teacher.directory).expanduser().resolve()


@dataclass(frozen=True, slots=True)
class ExpertFixture:
    split: FixtureSplit
    index: int
    fixture_id: str
    game_seed: str


@dataclass(frozen=True, slots=True)
class ExpertCollectionResult:
    iteration: int
    training_games: int
    validation_games: int
    examples: int
    simulations: int
    wall_seconds: float
    examples_per_hour: float
    p50_decision_seconds: float
    p95_decision_seconds: float
    peak_rss_bytes: int
    dataset_digest: str


@dataclass(frozen=True, slots=True)
class ExpertEpochMetrics:
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
    elapsed_seconds: float


@dataclass(frozen=True, slots=True)
class ExpertTrainingResult:
    iteration: int
    completed_epochs: int
    best_epoch: int
    best_validation_loss: float
    candidate_checkpoint: str
    candidate_artifact: str
    replay_digest: str


@dataclass(frozen=True, slots=True)
class ExpertIterationResult:
    iteration: int
    phase: str
    collection: ExpertCollectionResult
    training: ExpertTrainingResult
    evaluation_path: str
    eligible: bool


@dataclass(frozen=True, slots=True)
class _PendingDecision:
    observation: Tensor
    legal_mask: Tensor
    visits: Tensor
    selected_action: int
    information_digest: str
    player: EnginePlayer
    dealer: EnginePlayer
    round_number: int
    placement_number: int


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
        raise ExpertIterationError("metadata is not canonical JSON") from error


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _atomic_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, value: object) -> None:
    _atomic_bytes(path, _canonical_json(value))


def _atomic_text(path: Path, value: str) -> None:
    _atomic_bytes(path, value.encode("utf-8"))


def _atomic_torch(path: Path, payload: object, validator: Callable[[object], object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        torch.save(payload, temporary)
        loaded = torch.load(temporary, map_location="cpu", weights_only=True)
        validator(loaded)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ExpertIterationError(f"cannot read JSON artifact: {path}") from error


def _strict(value: object, keys: set[str], label: str) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != keys:
        raise ExpertIterationError(f"{label} keys must be exactly {sorted(keys)}")
    return value


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or "\0" in value:
        raise ExpertIterationError(f"{label} must be nonempty and contain no NUL")
    return value


def _sha256_digest(value: object, label: str) -> str:
    digest = _string(value, label)
    if len(digest) != 64:
        raise ExpertIterationError(f"{label} must be a lowercase SHA-256 digest")
    try:
        int(digest, 16)
    except ValueError as error:
        raise ExpertIterationError(
            f"{label} must be a lowercase SHA-256 digest"
        ) from error
    if digest != digest.lower():
        raise ExpertIterationError(f"{label} must be a lowercase SHA-256 digest")
    return digest


def _integer(value: object, label: str, *, minimum: int = 1) -> int:
    if type(value) is not int or value < minimum:
        raise ExpertIterationError(f"{label} must be an integer at least {minimum}")
    return value


def _number(value: object, label: str, *, minimum: float = 0.0) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or value < minimum:
        raise ExpertIterationError(f"{label} must be finite and at least {minimum}")
    return float(value)


def load_expert_config(path: str | Path) -> ExpertConfig:
    try:
        with Path(path).open("rb") as stream:
            document = tomllib.load(stream)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise ExpertIterationError("cannot read expert TOML configuration") from error
    root = _strict(
        document,
        {"run", "accepted", "teacher", "collection", "replay", "optimization", "evaluation", "controls"},
        "configuration",
    )
    run = _strict(root["run"], {"run_id", "root_seed", "output_directory", "source_revision", "maximum_iterations"}, "run")
    accepted = _strict(root["accepted"], {"artifact_path"}, "accepted")
    teacher = _strict(root["teacher"], {"directory", "search_report_digest"}, "teacher")
    collection = _strict(root["collection"], {"training_games", "validation_games", "workers", "simulation_budget", "puct_constant", "early_placement_count"}, "collection")
    replay = _strict(root["replay"], {"accepted_iterations", "examples_per_epoch"}, "replay")
    optimization = _strict(root["optimization"], {"device", "batch_size", "learning_rate", "betas", "epsilon", "weight_decay", "gradient_norm", "minimum_epochs", "maximum_epochs", "early_stop_patience", "minimum_improvement"}, "optimization")
    evaluation = _strict(root["evaluation"], {"candidate_budget", "random_pairs", "ppo_pairs", "search_pairs", "prior_pairs", "search_control_budget", "bootstrap_samples", "workers", "strategic_repetitions"}, "evaluation")
    controls = _strict(root["controls"], {"ppo_archive"}, "controls")
    betas = optimization["betas"]
    if not isinstance(betas, list) or len(betas) != 2:
        raise ExpertIterationError("optimizer betas must contain two numbers")
    config = ExpertConfig(
        ExpertRunConfig(
            _string(run["run_id"], "run ID"),
            _string(run["root_seed"], "root seed"),
            _string(run["output_directory"], "output directory"),
            _string(run["source_revision"], "source revision"),
            _integer(run["maximum_iterations"], "maximum iterations"),
        ),
        AcceptedConfig(_string(accepted["artifact_path"], "accepted artifact")),
        ExpertTeacherConfig(
            _string(teacher["directory"], "teacher directory"),
            _sha256_digest(teacher["search_report_digest"], "search report digest"),
        ),
        ExpertCollectionConfig(
            _integer(collection["training_games"], "training games"),
            _integer(collection["validation_games"], "validation games"),
            _integer(collection["workers"], "workers"),
            _integer(collection["simulation_budget"], "simulation budget"),
            _number(collection["puct_constant"], "PUCT constant"),
            _integer(collection["early_placement_count"], "early placement count", minimum=0),
        ),
        ReplayConfig(
            _integer(replay["accepted_iterations"], "accepted replay iterations"),
            _integer(replay["examples_per_epoch"], "examples per epoch"),
        ),
        OptimizationConfig(
            device=_string(optimization["device"], "optimization device"),
            batch_size=_integer(optimization["batch_size"], "batch size"),
            learning_rate=_number(optimization["learning_rate"], "learning rate", minimum=1e-20),
            betas=(float(betas[0]), float(betas[1])),
            epsilon=_number(optimization["epsilon"], "epsilon", minimum=1e-20),
            weight_decay=_number(optimization["weight_decay"], "weight decay"),
            gradient_norm=_number(optimization["gradient_norm"], "gradient norm", minimum=1e-20),
            minimum_epochs=_integer(optimization["minimum_epochs"], "minimum epochs"),
            maximum_epochs=_integer(optimization["maximum_epochs"], "maximum epochs"),
            early_stop_patience=_integer(optimization["early_stop_patience"], "early stop patience"),
            minimum_improvement=_number(optimization["minimum_improvement"], "minimum improvement", minimum=1e-20),
        ),
        ExpertEvaluationConfig(**{
            key: _integer(evaluation[key], key.replace("_", " "))
            for key in ExpertEvaluationConfig.__dataclass_fields__
        }),
        ExpertControlsConfig(_string(controls["ppo_archive"], "PPO archive")),
    )
    if config.collection.workers > MAX_WORKERS or config.evaluation.workers > MAX_WORKERS:
        raise ExpertIterationError(f"worker counts cannot exceed {MAX_WORKERS}")
    if not 0 <= config.collection.early_placement_count <= 7:
        raise ExpertIterationError("early placement count must be between zero and seven")
    if config.optimization.device not in {"cpu", "mps", "auto"}:
        raise ExpertIterationError("optimization device must be cpu, mps, or auto")
    if config.optimization.minimum_epochs > config.optimization.maximum_epochs:
        raise ExpertIterationError("minimum epochs cannot exceed maximum epochs")
    config.collection.search_config
    config.optimization.model_optimizer
    return config


def _resolved_config(config: ExpertConfig, selected_device: str) -> dict[str, object]:
    body = {
        "format_version": EXPERT_CONFIG_VERSION,
        "run": {**asdict(config.run), "output_directory": str(config.output_path)},
        "accepted": {"artifact_path": str(config.accepted_artifact_path)},
        "teacher": {**asdict(config.teacher), "directory": str(config.teacher_path)},
        "collection": asdict(config.collection),
        "replay": asdict(config.replay),
        "optimization": {**asdict(config.optimization), "betas": list(config.optimization.betas), "resolved_device": selected_device},
        "evaluation": asdict(config.evaluation),
        "controls": {"ppo_archive": str(Path(config.controls.ppo_archive).expanduser().resolve())},
        "contracts": {
            "model_schema_version": MODEL_SCHEMA_VERSION,
            "information_state_schema_version": INFORMATION_STATE_SCHEMA_VERSION,
            "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
            "action_schema_version": ACTION_SCHEMA_VERSION,
            "value_schema_version": VALUE_SCHEMA_VERSION,
            "search_schema_version": GUIDED_SEARCH_SCHEMA_VERSION,
            "parameter_count": PARAMETER_COUNT,
        },
    }
    identity = json.loads(json.dumps(body))
    identity["run"]["output_directory"] = "<run-output>"
    identity["accepted"]["artifact_path"] = "<accepted-artifact>"
    identity["teacher"]["directory"] = "<teacher-dataset>"
    identity["controls"]["ppo_archive"] = "<ppo-control>"
    return {**body, "resolved_config_digest": _digest(identity)}


def _fixture(config: ExpertConfig, iteration: int, split: FixtureSplit, index: int) -> ExpertFixture:
    components = (config.run.root_seed, "expert", split.value, str(iteration), str(index))
    return ExpertFixture(
        split,
        index,
        seed_hex(derive_seed(EXPERT_FIXTURE_ID_NAMESPACE, *components)),
        seed_hex(derive_seed(EXPERT_FIXTURE_NAMESPACE, *components)),
    )


def _fixture_schedule(config: ExpertConfig, iteration: int) -> tuple[ExpertFixture, ...]:
    return tuple(
        _fixture(config, iteration, split, index)
        for split, count in (
            (FixtureSplit.TRAINING, config.collection.training_games),
            (FixtureSplit.VALIDATION, config.collection.validation_games),
        )
        for index in range(count)
    )


def _action_from_visits(
    visits: Sequence[int],
    *,
    config: ExpertConfig,
    iteration: int,
    fixture: ExpertFixture,
    round_number: int,
    placement: int,
    maximum_action: int,
) -> int:
    if placement > config.collection.early_placement_count:
        return maximum_action
    total = sum(visits)
    stream = Sha256CounterStream(
        derive_seed(
            EXPERT_ACTION_NAMESPACE,
            config.run.root_seed,
            str(iteration),
            fixture.fixture_id,
            str(round_number),
            str(placement),
        )
    )
    target = stream.randbelow(total)
    cumulative = 0
    for index, count in enumerate(visits):
        cumulative += count
        if target < cumulative:
            return index
    raise ExpertIterationError("visit distribution did not select an action")


def _empty_columns() -> dict[str, list[object]]:
    return {
        "observations": [],
        "legal_masks": [],
        "search_visits": [],
        "search_policies": [],
        "selected_action_indices": [],
        "round_returns": [],
        "information_state_digests": [],
        "fixture_ids": [],
        "splits": [],
        "players": [],
        "dealers": [],
        "round_numbers": [],
        "placement_numbers": [],
    }


def _append_round(columns: dict[str, list[object]], pending: Sequence[_PendingDecision], result) -> None:
    queen = (result.round_scores[EnginePlayer.QUEEN] - result.round_scores[EnginePlayer.KING]) / 150.0
    returns = {EnginePlayer.QUEEN: queen, EnginePlayer.KING: -queen}
    for row in pending:
        columns["observations"].append(row.observation)
        columns["legal_masks"].append(row.legal_mask)
        columns["search_visits"].append(row.visits)
        columns["search_policies"].append(row.visits.float() / float(row.visits.sum()))
        columns["selected_action_indices"].append(row.selected_action)
        columns["round_returns"].append(returns[row.player])
        columns["information_state_digests"].append(row.information_digest)
        columns["players"].append(row.player.value)
        columns["dealers"].append(row.dealer.value)
        columns["round_numbers"].append(row.round_number)
        columns["placement_numbers"].append(row.placement_number)


def _seal_columns(columns: dict[str, list[object]], fixture: ExpertFixture) -> dict[str, object]:
    count = len(columns["observations"])
    return {
        "observations": torch.stack(columns["observations"]),
        "legal_masks": torch.stack(columns["legal_masks"]),
        "search_visits": torch.stack(columns["search_visits"]),
        "search_policies": torch.stack(columns["search_policies"]),
        "selected_action_indices": torch.tensor(columns["selected_action_indices"], dtype=torch.int64),
        "round_returns": torch.tensor(columns["round_returns"], dtype=torch.float32),
        "information_state_digests": tuple(columns["information_state_digests"]),
        "fixture_ids": tuple(fixture.fixture_id for _ in range(count)),
        "splits": tuple(fixture.split.value for _ in range(count)),
        "players": tuple(columns["players"]),
        "dealers": tuple(columns["dealers"]),
        "round_numbers": torch.tensor(columns["round_numbers"], dtype=torch.int64),
        "placement_numbers": torch.tensor(columns["placement_numbers"], dtype=torch.int64),
    }


def _tensor_digest(metadata: Mapping[str, object], columns: Mapping[str, object]) -> str:
    digest = hashlib.sha256(_canonical_json(dict(metadata)))
    for name in sorted(columns):
        digest.update(name.encode("utf-8") + b"\0")
        value = columns[name]
        if isinstance(value, Tensor):
            tensor = value.detach().cpu().contiguous()
            digest.update(str(tensor.dtype).encode("ascii") + b"\0")
            digest.update(",".join(map(str, tensor.shape)).encode("ascii") + b"\0")
            digest.update(tensor.numpy().tobytes(order="C"))
        else:
            digest.update(_canonical_json(value))
    return digest.hexdigest()


def _validate_expert_shard(payload: object) -> dict[str, object]:
    if not isinstance(payload, dict) or set(payload) != {"format_version", "metadata", "columns", "content_digest"}:
        raise ExpertIterationError("expert shard shape is invalid")
    if payload["format_version"] != EXPERT_SHARD_VERSION:
        raise ExpertIterationError("expert shard version is incompatible")
    metadata = payload["metadata"]
    columns = payload["columns"]
    if not isinstance(metadata, dict) or not isinstance(columns, dict):
        raise ExpertIterationError("expert shard sections are invalid")
    if set(columns) != set(_empty_columns()) or any(key in _FORBIDDEN_KEYS for key in (*metadata, *columns)):
        raise ExpertIterationError("expert shard columns are invalid or private")
    expected_versions = {
        "model_schema_version": MODEL_SCHEMA_VERSION,
        "information_state_schema_version": INFORMATION_STATE_SCHEMA_VERSION,
        "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
        "action_schema_version": ACTION_SCHEMA_VERSION,
        "value_schema_version": VALUE_SCHEMA_VERSION,
        "search_schema_version": GUIDED_SEARCH_SCHEMA_VERSION,
    }
    if any(metadata.get(key) != value for key, value in expected_versions.items()):
        raise ExpertIterationError("expert shard contract version differs")
    observations = columns["observations"]
    masks = columns["legal_masks"]
    visits = columns["search_visits"]
    policies = columns["search_policies"]
    selected = columns["selected_action_indices"]
    returns = columns["round_returns"]
    rounds = columns["round_numbers"]
    placements = columns["placement_numbers"]
    expected = (
        (observations, torch.bool, (ROWS_PER_GAME, 875)),
        (masks, torch.bool, (ROWS_PER_GAME, 4, 8)),
        (visits, torch.int64, (ROWS_PER_GAME, ACTION_COUNT)),
        (policies, torch.float32, (ROWS_PER_GAME, ACTION_COUNT)),
        (selected, torch.int64, (ROWS_PER_GAME,)),
        (returns, torch.float32, (ROWS_PER_GAME,)),
        (rounds, torch.int64, (ROWS_PER_GAME,)),
        (placements, torch.int64, (ROWS_PER_GAME,)),
    )
    for tensor, dtype, shape in expected:
        if not isinstance(tensor, Tensor) or tensor.device.type != "cpu" or tensor.dtype is not dtype or tuple(tensor.shape) != shape:
            raise ExpertIterationError("expert shard tensor contract differs")
    strings = ("information_state_digests", "fixture_ids", "splits", "players", "dealers")
    if any(not isinstance(columns[name], tuple) or len(columns[name]) != ROWS_PER_GAME for name in strings):
        raise ExpertIterationError("expert shard identifier columns differ")
    flat_masks = masks.flatten(start_dim=1)
    if not torch.isfinite(policies).all() or not torch.isfinite(returns).all():
        raise ExpertIterationError("expert shard has non-finite targets")
    if torch.any(visits < 0) or torch.any(visits[~flat_masks] != 0) or torch.any(policies[~flat_masks] != 0):
        raise ExpertIterationError("expert shard assigns visits to illegal actions")
    totals = visits.sum(dim=1)
    if torch.any(totals <= 0) or not torch.allclose(policies, visits.float() / totals[:, None].float(), atol=0, rtol=0):
        raise ExpertIterationError("expert shard policy does not normalize visits")
    if torch.any(selected < 0) or torch.any(selected >= ACTION_COUNT) or not torch.all(flat_masks.gather(1, selected[:, None]).squeeze(1)):
        raise ExpertIterationError("expert shard selected action is illegal")
    if torch.any(returns < -1) or torch.any(returns > 1):
        raise ExpertIterationError("expert shard return is out of range")
    if rounds.tolist() != [round_number for round_number in range(1, 7) for _ in range(7)] or placements.tolist() != list(range(1, 8)) * 6:
        raise ExpertIterationError("expert shard row order differs")
    for round_number in range(1, 7):
        indexes = [index for index, value in enumerate(rounds.tolist()) if value == round_number]
        queen = next(index for index in indexes if columns["players"][index] == "queen")
        king = next(index for index in indexes if columns["players"][index] == "king")
        if returns[queen].item() != -returns[king].item():
            raise ExpertIterationError("expert round targets are not player-relative opposites")
    if payload["content_digest"] != _tensor_digest(metadata, columns):
        raise ExpertIterationError("expert shard digest differs")
    return payload


def load_expert_shard(path: str | Path) -> dict[str, object]:
    try:
        return _validate_expert_shard(torch.load(Path(path), map_location="cpu", weights_only=True))
    except (OSError, RuntimeError, ValueError, EOFError, pickle.UnpicklingError) as error:
        if isinstance(error, ExpertIterationError):
            raise
        raise ExpertIterationError("expert shard cannot be loaded") from error


def _collect_game(
    config: ExpertConfig,
    iteration: int,
    fixture: ExpertFixture,
    artifact_path: str,
    should_stop: Callable[[], bool] | None = None,
) -> dict[str, object]:
    evaluator = PolicyValueModelEvaluator.from_artifact(artifact_path)
    planner = GuidedInformationSetSearch(evaluator, config.collection.search_config)
    state = create_game(fixture.game_seed)
    columns = _empty_columns()
    pending: list[_PendingDecision] = []
    latencies: list[float] = []
    model_evaluations = 0
    while state.status is not EngineStatus.GAME_COMPLETE:
        while state.status is EngineStatus.PLAYING:
            if should_stop is not None and should_stop():
                raise ExpertIterationInterrupted("expert collection interrupted")
            actor = state.active_player
            if actor is None:
                raise ExpertIterationError("playing collection state lost its actor")
            context = build_policy_turn_context(state, actor)
            if context.kind is PolicyTurnKind.FORCED_RECURRENT_TRANSITION:
                if context.forced_move is None:
                    raise ExpertIterationError("forced collection turn has no move")
                move = context.forced_move
            else:
                information = information_state_from_engine(state, actor)
                policy_input = policy_input_from_information_state(information)
                placement = len(state.current_round_moves) + 1
                request_seed = derive_search_request_seed(fixture.fixture_id, information, planner.digest)
                started = time.perf_counter()
                try:
                    search = planner.search(information, request_seed, should_stop)
                except SearchInterrupted as error:
                    raise ExpertIterationInterrupted(str(error)) from error
                latencies.append(time.perf_counter() - started)
                model_evaluations += search.model_evaluation_count
                selected = _action_from_visits(
                    search.action_visits,
                    config=config,
                    iteration=iteration,
                    fixture=fixture,
                    round_number=state.round_number,
                    placement=placement,
                    maximum_action=search.selected_action_index,
                )
                pending.append(
                    _PendingDecision(
                        policy_input.observation.detach().cpu().clone(),
                        policy_input.legal_mask.detach().cpu().clone(),
                        torch.tensor(search.action_visits, dtype=torch.int64),
                        selected,
                        information_state_fingerprint(information),
                        actor,
                        state.dealer,
                        state.round_number,
                        placement,
                    )
                )
                move = context.action_table[selected]
                if move is None:
                    raise ExpertIterationError("guided self-play selected a masked action")
            state = apply_move(state, move).state
        result = state.pending_round_result
        if result is None:
            raise ExpertIterationError("completed self-play round has no result")
        _append_round(columns, pending, result)
        pending.clear()
        state = advance_after_round(state)
    sealed = _seal_columns(columns, fixture)
    metadata = {
        "model_schema_version": MODEL_SCHEMA_VERSION,
        "information_state_schema_version": INFORMATION_STATE_SCHEMA_VERSION,
        "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
        "action_schema_version": ACTION_SCHEMA_VERSION,
        "value_schema_version": VALUE_SCHEMA_VERSION,
        "search_schema_version": GUIDED_SEARCH_SCHEMA_VERSION,
        "search_config_digest": config.collection.search_config.digest,
        "planner_digest": planner.digest,
        "evaluator_digest": evaluator.digest,
        "iteration": iteration,
        "fixture_id": fixture.fixture_id,
        "fixture_index": fixture.index,
        "split": fixture.split.value,
    }
    content_digest = _tensor_digest(metadata, sealed)
    payload = {
        "format_version": EXPERT_SHARD_VERSION,
        "metadata": metadata,
        "columns": sealed,
        "content_digest": content_digest,
    }
    return {
        "payload": payload,
        "summary": {
            "fixture_id": fixture.fixture_id,
            "fixture_index": fixture.index,
            "split": fixture.split.value,
            "content_digest": content_digest,
            "examples": ROWS_PER_GAME,
            "simulations": config.collection.simulation_budget * ROWS_PER_GAME,
            "latencies": latencies,
            "model_evaluations": model_evaluations,
            "peak_rss_bytes": _peak_rss_bytes(),
        },
    }


def _collect_game_worker(args: tuple[ExpertConfig, int, ExpertFixture, str]) -> dict[str, object]:
    return _collect_game(*args)


def _peak_rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def _percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[max(0, min(len(ordered) - 1, math.ceil(quantile * len(ordered)) - 1))]


def _collection_directory(config: ExpertConfig, iteration: int) -> Path:
    return config.output_path / "iterations" / f"{iteration:06d}" / "collection"


def _manifest_path(config: ExpertConfig, iteration: int, split: FixtureSplit) -> Path:
    return _collection_directory(config, iteration) / f"{split.value}-manifest.json"


def _shard_path(config: ExpertConfig, iteration: int, fixture: ExpertFixture) -> Path:
    return _collection_directory(config, iteration) / "games" / fixture.split.value / f"{fixture.fixture_id}.pt"


def _existing_summary(config: ExpertConfig, iteration: int, fixture: ExpertFixture) -> dict[str, object] | None:
    path = _shard_path(config, iteration, fixture)
    if not path.is_file():
        return None
    payload = load_expert_shard(path)
    metadata = payload["metadata"]
    if metadata["fixture_id"] != fixture.fixture_id or metadata["iteration"] != iteration:
        raise ExpertIterationError("existing expert shard belongs to another fixture")
    metrics_path = path.with_suffix(".metrics.json")
    metrics = _read_json(metrics_path)
    if not isinstance(metrics, dict):
        raise ExpertIterationError("expert shard metrics are invalid")
    return {
        "fixture_id": fixture.fixture_id,
        "fixture_index": fixture.index,
        "split": fixture.split.value,
        "content_digest": payload["content_digest"],
        "examples": ROWS_PER_GAME,
        "simulations": config.collection.simulation_budget * ROWS_PER_GAME,
        **metrics,
    }


def _write_collected(config: ExpertConfig, iteration: int, fixture: ExpertFixture, result: Mapping[str, object]) -> dict[str, object]:
    path = _shard_path(config, iteration, fixture)
    payload = result["payload"]
    summary = dict(result["summary"])  # type: ignore[arg-type]
    _atomic_torch(path, payload, _validate_expert_shard)
    metrics = {
        "latencies": summary.pop("latencies"),
        "model_evaluations": summary.pop("model_evaluations"),
        "peak_rss_bytes": summary.pop("peak_rss_bytes"),
    }
    _atomic_json(path.with_suffix(".metrics.json"), metrics)
    return {**summary, **metrics}


def _seal_manifests(config: ExpertConfig, iteration: int, summaries: Sequence[Mapping[str, object]]) -> str:
    digests = []
    for split in (FixtureSplit.TRAINING, FixtureSplit.VALIDATION):
        rows = [
            {
                "fixture_id": row["fixture_id"],
                "fixture_index": row["fixture_index"],
                "relative_path": str(
                    _shard_path(
                        config,
                        iteration,
                        next(item for item in _fixture_schedule(config, iteration) if item.fixture_id == row["fixture_id"]),
                    ).relative_to(_collection_directory(config, iteration))
                ),
                "content_digest": row["content_digest"],
                "examples": row["examples"],
                "simulations": row["simulations"],
            }
            for row in sorted(summaries, key=lambda item: int(item["fixture_index"]))
            if row["split"] == split.value
        ]
        body = {
            "format_version": EXPERT_MANIFEST_VERSION,
            "iteration": iteration,
            "split": split.value,
            "search_config_digest": config.collection.search_config.digest,
            "fixture_count": len(rows),
            "example_count": sum(int(row["examples"]) for row in rows),
            "shards": rows,
        }
        document = {"manifest": body, "manifest_digest": _digest(body)}
        _atomic_json(_manifest_path(config, iteration, split), document)
        digests.append(document["manifest_digest"])
    return _digest(digests)


def collect_expert_iteration(
    config: ExpertConfig,
    iteration: int,
    artifact_path: str | Path,
    *,
    should_stop: Callable[[str, int], bool] | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> ExpertCollectionResult:
    metrics_path = _collection_directory(config, iteration) / "metrics.json"
    if (
        metrics_path.is_file()
        and all(
            _manifest_path(config, iteration, split).is_file()
            for split in (FixtureSplit.TRAINING, FixtureSplit.VALIDATION)
        )
    ):
        value = _read_json(metrics_path)
        if not isinstance(value, dict) or value.get("format_version") != EXPERT_METRICS_VERSION:
            raise ExpertIterationError("expert collection metrics are invalid")
        return ExpertCollectionResult(
            **{key: value[key] for key in ExpertCollectionResult.__dataclass_fields__}
        )
    started = time.perf_counter()
    fixtures = _fixture_schedule(config, iteration)
    summaries: list[dict[str, object]] = []
    missing: list[ExpertFixture] = []
    for fixture in fixtures:
        existing = _existing_summary(config, iteration, fixture)
        if existing is None:
            missing.append(fixture)
        else:
            summaries.append(existing)
    stop = None if should_stop is None else lambda: should_stop("collection", len(summaries))
    try:
        if config.collection.workers == 1 or should_stop is not None:
            for fixture in missing:
                result = _collect_game(config, iteration, fixture, str(artifact_path), stop)
                summaries.append(_write_collected(config, iteration, fixture, result))
                if progress:
                    progress(len(summaries), len(fixtures))
        else:
            with ProcessPoolExecutor(max_workers=config.collection.workers) as pool:
                futures = {
                    pool.submit(_collect_game_worker, (config, iteration, fixture, str(artifact_path))): fixture
                    for fixture in missing
                }
                for future in as_completed(futures):
                    fixture = futures[future]
                    summaries.append(_write_collected(config, iteration, fixture, future.result()))
                    if progress:
                        progress(len(summaries), len(fixtures))
    except (KeyboardInterrupt, ExpertIterationInterrupted, SearchInterrupted) as error:
        raise ExpertIterationInterrupted("expert collection interrupted") from error
    dataset_digest = _seal_manifests(config, iteration, summaries)
    latencies = [float(value) for row in summaries for value in row["latencies"]]  # type: ignore[union-attr]
    wall = time.perf_counter() - started
    result = ExpertCollectionResult(
        iteration,
        sum(row["split"] == FixtureSplit.TRAINING.value for row in summaries),
        sum(row["split"] == FixtureSplit.VALIDATION.value for row in summaries),
        len(summaries) * ROWS_PER_GAME,
        sum(int(row["simulations"]) for row in summaries),
        wall,
        len(summaries) * ROWS_PER_GAME / wall * 3600 if wall else 0.0,
        statistics.median(latencies) if latencies else 0.0,
        _percentile(latencies, 0.95),
        max((int(row["peak_rss_bytes"]) for row in summaries), default=0),
        dataset_digest,
    )
    _atomic_json(_collection_directory(config, iteration) / "metrics.json", {"format_version": EXPERT_METRICS_VERSION, **asdict(result)})
    return result


def load_expert_manifest(directory: str | Path, split: FixtureSplit) -> dict[str, object]:
    path = Path(directory).expanduser().resolve() / f"{FixtureSplit(split).value}-manifest.json"
    document = _read_json(path)
    if not isinstance(document, dict) or set(document) != {"manifest", "manifest_digest"}:
        raise ExpertIterationError("expert manifest document is invalid")
    manifest = document["manifest"]
    if not isinstance(manifest, dict) or document["manifest_digest"] != _digest(manifest) or manifest.get("format_version") != EXPERT_MANIFEST_VERSION:
        raise ExpertIterationError("expert manifest digest or version differs")
    if manifest.get("split") != FixtureSplit(split).value:
        raise ExpertIterationError("expert manifest split differs")
    for row in manifest.get("shards", []):
        shard = load_expert_shard(Path(directory) / str(row["relative_path"]))
        if shard["content_digest"] != row["content_digest"]:
            raise ExpertIterationError("expert manifest shard digest differs")
    return document


def _expert_dataset(directory: Path, split: FixtureSplit) -> SupervisedDataset:
    document = load_expert_manifest(directory, split)
    manifest = document["manifest"]
    assert isinstance(manifest, dict)
    payloads = [
        load_expert_shard(directory / str(row["relative_path"]))
        for row in sorted(manifest["shards"], key=lambda item: item["fixture_index"])
    ]
    columns = [payload["columns"] for payload in payloads]
    return SupervisedDataset(
        split,
        torch.cat([row["observations"] for row in columns]),  # type: ignore[list-item]
        torch.cat([row["legal_masks"] for row in columns]),  # type: ignore[list-item]
        torch.cat([row["search_policies"].reshape(-1, 4, 8) for row in columns]),  # type: ignore[union-attr]
        torch.cat([row["round_returns"] for row in columns]),  # type: ignore[list-item]
        tuple(str(row["metadata"]["fixture_id"]) for row in payloads),  # type: ignore[index]
        tuple(str(row["content_digest"]) for row in payloads),
    )


def _concat_datasets(split: FixtureSplit, datasets: Sequence[SupervisedDataset]) -> SupervisedDataset:
    if not datasets:
        raise ExpertIterationError("replay requires at least one dataset source")
    return SupervisedDataset(
        split,
        torch.cat([value.observations for value in datasets]),
        torch.cat([value.legal_masks for value in datasets]),
        torch.cat([value.search_policies for value in datasets]),
        torch.cat([value.round_returns for value in datasets]),
        tuple(item for value in datasets for item in value.fixture_ids),
        tuple(item for value in datasets for item in value.shard_digests),
    )


def _accepted_pointer(config: ExpertConfig) -> dict[str, object]:
    value = _read_json(config.output_path / "accepted.json")
    if not isinstance(value, dict) or value.get("format_version") != EXPERT_ACCEPTED_VERSION:
        raise ExpertIterationError("accepted pointer is incompatible")
    return value


def _replay_sources(config: ExpertConfig, iteration: int) -> tuple[tuple[SupervisedDataset, ...], SupervisedDataset, str]:
    teacher = load_supervised_datasets(config.teacher_path)
    pointer = _accepted_pointer(config)
    accepted_iterations = [int(value) for value in pointer.get("accepted_iterations", [])]
    retained = accepted_iterations[-config.replay.accepted_iterations :]
    directories = [
        _collection_directory(config, value)
        for value in (*retained, iteration)
    ]
    expert_training = [_expert_dataset(path, FixtureSplit.TRAINING) for path in directories]
    expert_validation = [_expert_dataset(path, FixtureSplit.VALIDATION) for path in directories]
    training_sources = (teacher.training, *expert_training)
    validation = _concat_datasets(FixtureSplit.VALIDATION, (teacher.validation, *expert_validation))
    fixture_sets = [set(source.fixture_ids) for source in (*training_sources, validation)]
    for index, values in enumerate(fixture_sets):
        if any(values & later for later in fixture_sets[index + 1 :]):
            raise ExpertIterationError("replay fixture IDs overlap")
    replay_digest = _digest(
        {
            "training_source_shards": [list(source.shard_digests) for source in training_sources],
            "validation_shards": list(validation.shard_digests),
        }
    )
    return training_sources, validation, replay_digest


def replay_epoch_dataset(
    config: ExpertConfig,
    iteration: int,
    epoch: int,
    sources: Sequence[SupervisedDataset],
) -> SupervisedDataset:
    stream = Sha256CounterStream(
        derive_seed(
            EXPERT_REPLAY_NAMESPACE,
            config.run.root_seed,
            str(iteration),
            str(epoch),
        )
    )
    picks = [
        (source_index := stream.randbelow(len(sources)), stream.randbelow(sources[source_index].example_count))
        for _ in range(config.replay.examples_per_epoch)
    ]
    return SupervisedDataset(
        FixtureSplit.TRAINING,
        torch.stack([sources[source].observations[row] for source, row in picks]),
        torch.stack([sources[source].legal_masks[row] for source, row in picks]),
        torch.stack([sources[source].search_policies[row] for source, row in picks]),
        torch.stack([sources[source].round_returns[row] for source, row in picks]),
        tuple(sources[source].fixture_ids[0] for source, _ in picks),
        tuple(sources[source].shard_digests[0] for source, _ in picks),
    )


def _checkpoint_payload(
    *,
    kind: str,
    config_digest: str,
    replay_digest: str,
    iteration: int,
    device: str,
    model: PolicyValueModel,
    optimizer: torch.optim.AdamW,
    completed_epoch: int,
    best_epoch: int,
    best_loss: float,
    stale_epochs: int,
    history: Sequence[ExpertEpochMetrics],
) -> dict[str, object]:
    return {
        "format_version": EXPERT_CHECKPOINT_VERSION,
        "kind": kind,
        "config_digest": config_digest,
        "replay_digest": replay_digest,
        "iteration": iteration,
        "device": device,
        "model_state_dict": _cpu_copy(model.state_dict()),
        "optimizer_state_dict": _cpu_copy(optimizer.state_dict()),
        "completed_epoch": completed_epoch,
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "stale_epochs": stale_epochs,
        "history": tuple(asdict(value) for value in history),
    }


def _validate_checkpoint(payload: object) -> dict[str, object]:
    required = {"format_version", "kind", "config_digest", "replay_digest", "iteration", "device", "model_state_dict", "optimizer_state_dict", "completed_epoch", "best_epoch", "best_validation_loss", "stale_epochs", "history"}
    if not isinstance(payload, dict) or set(payload) != required or payload["format_version"] != EXPERT_CHECKPOINT_VERSION:
        raise ExpertIterationError("expert checkpoint shape or version differs")
    model = PolicyValueModel(
        run_root_seed="checkpoint-validation",
        model_id="validation",
        initialization_ordinal=0,
    )
    state = payload["model_state_dict"]
    if not isinstance(state, dict) or set(state) != set(model.state_dict()):
        raise ExpertIterationError("expert checkpoint model keys differ")
    for name, expected in model.state_dict().items():
        value = state[name]
        if not isinstance(value, Tensor) or value.device.type != "cpu" or value.shape != expected.shape or value.dtype != expected.dtype or not torch.isfinite(value).all():
            raise ExpertIterationError(f"expert checkpoint tensor is invalid: {name}")
    model.load_state_dict(state, strict=True)
    optimizer = build_policy_value_optimizer(model)
    try:
        optimizer.load_state_dict(payload["optimizer_state_dict"])  # type: ignore[arg-type]
    except (ValueError, KeyError) as error:
        raise ExpertIterationError("expert checkpoint optimizer differs") from error
    if any(not torch.isfinite(item).all() for state_value in optimizer.state.values() for item in state_value.values() if isinstance(item, Tensor)):
        raise ExpertIterationError("expert checkpoint optimizer is non-finite")
    return payload


def load_expert_checkpoint(path: str | Path) -> dict[str, object]:
    try:
        return _validate_checkpoint(torch.load(Path(path), map_location="cpu", weights_only=True))
    except (OSError, RuntimeError, ValueError, EOFError, pickle.UnpicklingError) as error:
        if isinstance(error, ExpertIterationError):
            raise
        raise ExpertIterationError("expert checkpoint cannot be loaded") from error


def _load_accepted_model_optimizer(config: ExpertConfig, device: torch.device) -> tuple[PolicyValueModel, torch.optim.AdamW]:
    pointer = _accepted_pointer(config)
    checkpoint_path = pointer.get("checkpoint")
    if checkpoint_path:
        payload = load_expert_checkpoint(str(checkpoint_path))
        model = PolicyValueModel(
            run_root_seed="accepted", model_id="accepted", initialization_ordinal=0
        ).to(device)
        model.load_state_dict(payload["model_state_dict"], strict=True)  # type: ignore[arg-type]
        optimizer = build_policy_value_optimizer(model, config.optimization.model_optimizer)
        optimizer.load_state_dict(payload["optimizer_state_dict"])  # type: ignore[arg-type]
        _optimizer_to(optimizer, device)
        return model, optimizer
    loaded = load_policy_value_artifact(str(pointer["artifact"]))
    model = loaded.model.to(device)
    return model, build_policy_value_optimizer(model, config.optimization.model_optimizer)


def _benchmark_device(
    config: ExpertConfig,
    sources: Sequence[SupervisedDataset],
    validation: SupervisedDataset,
) -> DeviceBenchmark:
    benchmark_dataset = DatasetBundle(
        _concat_datasets(FixtureSplit.TRAINING, sources),
        validation,
        _digest([list(source.shard_digests) for source in sources]),
    )
    shim = SupervisedConfig(
        run=type("Run", (), {"root_seed": config.run.root_seed})(),  # type: ignore[arg-type]
        dataset=type("Dataset", (), {})(),  # type: ignore[arg-type]
        model=type("Model", (), {"model_id": "expert-benchmark", "initialization_ordinal": 0})(),  # type: ignore[arg-type]
        optimization=config.optimization,
    )
    return benchmark_optimization_devices(shim, benchmark_dataset)


def _select_device(config: ExpertConfig) -> str:
    resolved = _read_json(config.output_path / "resolved-config.json")
    if not isinstance(resolved, dict) or not isinstance(resolved.get("optimization"), dict):
        raise ExpertIterationError("resolved expert device is unavailable")
    selected = str(resolved["optimization"].get("resolved_device"))
    if selected not in {"cpu", "mps"}:
        raise ExpertIterationError("resolved expert device is invalid")
    if selected == "mps" and not torch.backends.mps.is_available():
        raise ExpertIterationError("resolved MPS device is unavailable")
    return selected


def optimize_expert_candidate(
    config: ExpertConfig,
    iteration: int,
    *,
    resume: bool = False,
    should_stop: Callable[[str, int], bool] | None = None,
    progress: Callable[[ExpertEpochMetrics], None] | None = None,
) -> ExpertTrainingResult:
    sources, validation, replay_digest = _replay_sources(config, iteration)
    iteration_dir = config.output_path / "iterations" / f"{iteration:06d}"
    checkpoints = iteration_dir / "checkpoints"
    latest_path = checkpoints / "epoch-latest.pt"
    best_path = checkpoints / "best-validation.pt"
    selected_device = _select_device(config)
    device = torch.device(selected_device)
    model, optimizer = _load_accepted_model_optimizer(config, device)
    completed_epoch = -1
    best_epoch = -1
    best_loss = math.inf
    stale = 0
    history: list[ExpertEpochMetrics] = []
    resolved = _read_json(config.output_path / "resolved-config.json")
    assert isinstance(resolved, dict)
    config_digest = str(resolved["resolved_config_digest"])
    if resume and latest_path.is_file():
        payload = load_expert_checkpoint(latest_path)
        if payload["config_digest"] != config_digest or payload["replay_digest"] != replay_digest or payload["iteration"] != iteration:
            raise ExpertIterationError("resume checkpoint identity differs")
        model.load_state_dict(payload["model_state_dict"], strict=True)  # type: ignore[arg-type]
        optimizer.load_state_dict(payload["optimizer_state_dict"])  # type: ignore[arg-type]
        _optimizer_to(optimizer, device)
        completed_epoch = int(payload["completed_epoch"])
        best_epoch = int(payload["best_epoch"])
        best_loss = float(payload["best_validation_loss"])
        stale = int(payload["stale_epochs"])
        history = [ExpertEpochMetrics(**value) for value in payload["history"]]  # type: ignore[arg-type]
    elif not (checkpoints / "iteration-start.pt").is_file():
        start = _checkpoint_payload(
            kind="iteration-start", config_digest=config_digest, replay_digest=replay_digest,
            iteration=iteration, device=selected_device, model=model, optimizer=optimizer,
            completed_epoch=-1, best_epoch=-1, best_loss=math.inf, stale_epochs=0, history=(),
        )
        _atomic_torch(checkpoints / "iteration-start.pt", start, _validate_checkpoint)

    for epoch in range(completed_epoch + 1, config.optimization.maximum_epochs):
        if should_stop is not None and should_stop("optimization", epoch):
            raise ExpertIterationInterrupted("expert optimization interrupted")
        started = time.perf_counter()
        dataset = replay_epoch_dataset(config, iteration, epoch, sources)
        batches = tuple(
            torch.arange(start, min(start + config.optimization.batch_size, dataset.example_count))
            for start in range(0, dataset.example_count, config.optimization.batch_size)
        )
        train = _training_epoch(
            model, optimizer, dataset, batches, device, config.optimization.gradient_norm,
            epoch=epoch,
        )
        valid = _validation_epoch(model, validation, config.optimization.batch_size, device)
        metrics = ExpertEpochMetrics(
            epoch, train[0], train[1], train[2], valid[0], valid[1], valid[2], valid[3], valid[4], train[3], time.perf_counter() - started,
        )
        if any(not math.isfinite(value) for key, value in asdict(metrics).items() if key != "epoch"):
            raise ExpertIterationError("expert optimization produced non-finite metrics")
        improved = valid[0] < best_loss - config.optimization.minimum_improvement
        if improved:
            best_loss = valid[0]
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
        history.append(metrics)
        payload = _checkpoint_payload(
            kind="epoch-latest", config_digest=config_digest, replay_digest=replay_digest,
            iteration=iteration, device=selected_device, model=model, optimizer=optimizer,
            completed_epoch=epoch, best_epoch=best_epoch, best_loss=best_loss,
            stale_epochs=stale, history=history,
        )
        if improved:
            best = dict(payload)
            best["kind"] = "best-validation"
            _atomic_torch(best_path, best, _validate_checkpoint)
        _atomic_torch(latest_path, payload, _validate_checkpoint)
        _atomic_json(iteration_dir / "metrics" / f"{epoch:06d}.json", {"format_version": EXPERT_METRICS_VERSION, **asdict(metrics)})
        if progress:
            progress(metrics)
        if epoch + 1 >= config.optimization.minimum_epochs and stale >= config.optimization.early_stop_patience:
            break
    best = load_expert_checkpoint(best_path)
    candidate_checkpoint = checkpoints / "candidate.pt"
    candidate = dict(best)
    candidate["kind"] = "candidate"
    _atomic_torch(candidate_checkpoint, candidate, _validate_checkpoint)
    model.load_state_dict(candidate["model_state_dict"], strict=True)  # type: ignore[arg-type]
    candidate_artifact = iteration_dir / "candidate-model.pt"
    save_policy_value_artifact(
        candidate_artifact,
        model.to("cpu"),
        source_revision=config.run.source_revision,
        training_configuration={
            "expert_config_digest": config_digest,
            "iteration": iteration,
            "replay_digest": replay_digest,
            "best_epoch": int(candidate["best_epoch"]),
        },
        dataset_digest=replay_digest,
        search_report_digest=config.teacher.search_report_digest,
    )
    result = ExpertTrainingResult(
        iteration,
        len(history),
        int(candidate["best_epoch"]),
        float(candidate["best_validation_loss"]),
        str(candidate_checkpoint),
        str(candidate_artifact),
        replay_digest,
    )
    _atomic_json(iteration_dir / "training-result.json", {"format_version": EXPERT_METRICS_VERSION, **asdict(result)})
    return result


def _write_state(config: ExpertConfig, phase: str, iteration: int, **extra: object) -> None:
    resolved = _read_json(config.output_path / "resolved-config.json")
    assert isinstance(resolved, dict)
    _atomic_json(
        config.output_path / "state.json",
        {
            "format_version": EXPERT_STATE_VERSION,
            "resolved_config_digest": resolved["resolved_config_digest"],
            "phase": phase,
            "iteration": iteration,
            **extra,
        },
    )


def _prepare_run(config: ExpertConfig) -> None:
    config.output_path.mkdir(parents=True, exist_ok=True)
    resolved_path = config.output_path / "resolved-config.json"
    if resolved_path.exists():
        resolved = _read_json(resolved_path)
        if not isinstance(resolved, dict):
            raise ExpertIterationError("resolved expert configuration is invalid")
        selected = str(resolved.get("optimization", {}).get("resolved_device"))  # type: ignore[union-attr]
        expected = _resolved_config(config, selected)
        if resolved != expected:
            raise ExpertIterationError("resolved expert configuration differs")
        if not (config.output_path / "accepted.json").is_file():
            _write_initial_accepted_pointer(config)
        if not (config.output_path / "state.json").is_file():
            _write_state(config, "ready", 0)
        return
    load_policy_value_artifact(config.accepted_artifact_path)
    datasets = load_supervised_datasets(config.teacher_path)
    requested = config.optimization.device
    benchmark = None
    if requested == "mps" and not torch.backends.mps.is_available():
        raise ExpertIterationError("MPS was requested but is unavailable")
    if requested == "auto":
        benchmark = _benchmark_device(
            config, (datasets.training,), datasets.validation
        )
        selected = benchmark.selected_device
    else:
        selected = requested
    _atomic_json(resolved_path, _resolved_config(config, selected))
    if benchmark is not None:
        _atomic_json(config.output_path / "device-benchmark.json", asdict(benchmark))
    _write_initial_accepted_pointer(config)
    _write_state(config, "ready", 0)


def _write_initial_accepted_pointer(config: ExpertConfig) -> None:
    _atomic_json(
        config.output_path / "accepted.json",
        {
            "format_version": EXPERT_ACCEPTED_VERSION,
            "accepted_version": 0,
            "artifact": str(config.accepted_artifact_path),
            "checkpoint": None,
            "accepted_iterations": [],
            "history": [
                {
                    "version": 0,
                    "artifact": str(config.accepted_artifact_path),
                    "checkpoint": None,
                }
            ],
        },
    )


def load_expert_config_from_run(output: str | Path) -> ExpertConfig:
    resolved = _read_json(Path(output).expanduser().resolve() / "resolved-config.json")
    if not isinstance(resolved, dict) or resolved.get("format_version") != EXPERT_CONFIG_VERSION:
        raise ExpertIterationError("resolved expert configuration is incompatible")
    # Reconstruct directly so resume never depends on the original TOML path.
    run = resolved["run"]
    return ExpertConfig(
        ExpertRunConfig(str(run["run_id"]), str(run["root_seed"]), str(run["output_directory"]), str(run["source_revision"]), int(run["maximum_iterations"])),
        AcceptedConfig(str(resolved["accepted"]["artifact_path"])),
        ExpertTeacherConfig(str(resolved["teacher"]["directory"]), str(resolved["teacher"]["search_report_digest"])),
        ExpertCollectionConfig(**resolved["collection"]),
        ReplayConfig(**resolved["replay"]),
        OptimizationConfig(**{key: (tuple(value) if key == "betas" else value) for key, value in resolved["optimization"].items() if key != "resolved_device"}),
        ExpertEvaluationConfig(**resolved["evaluation"]),
        ExpertControlsConfig(str(resolved["controls"]["ppo_archive"])),
    )


def run_expert_iteration(
    config: ExpertConfig,
    *,
    resume: bool = False,
    should_stop: Callable[[str, int], bool] | None = None,
    progress: Callable[[str], None] | None = None,
) -> ExpertIterationResult:
    _prepare_run(config)
    state = _read_json(config.output_path / "state.json")
    if not isinstance(state, dict):
        raise ExpertIterationError("expert run state is invalid")
    if state["phase"] == "awaiting-decision":
        raise ExpertIterationError("candidate requires accept or reject before another cycle")
    pointer = _accepted_pointer(config)
    prior_phase = str(state["phase"])
    if resume and prior_phase in {"collecting", "optimizing", "evaluating"}:
        iteration = int(state["iteration"])
    elif prior_phase == "ready":
        iteration = 0
    else:
        iteration = int(state["iteration"]) + 1
    if iteration >= config.run.maximum_iterations:
        raise ExpertIterationError("expert run reached its configured iteration limit")
    accepted_artifact = str(pointer["artifact"])
    _write_state(config, "collecting", iteration)
    if progress:
        progress(f"expert iteration={iteration} phase=collection")
    collection = collect_expert_iteration(
        config,
        iteration,
        accepted_artifact,
        should_stop=should_stop,
        progress=(None if progress is None else lambda done, total: progress(f"expert collection={done}/{total}")),
    )
    _write_state(config, "optimizing", iteration, collection=asdict(collection))
    if progress:
        progress(f"expert iteration={iteration} phase=optimization")
    training_path = config.output_path / "iterations" / f"{iteration:06d}" / "training-result.json"
    if resume and prior_phase == "evaluating" and training_path.is_file():
        training_value = _read_json(training_path)
        if not isinstance(training_value, dict):
            raise ExpertIterationError("training result is invalid")
        training = ExpertTrainingResult(
            **{
                key: training_value[key]
                for key in ExpertTrainingResult.__dataclass_fields__
            }
        )
    else:
        training = optimize_expert_candidate(
            config,
            iteration,
            resume=resume and prior_phase == "optimizing",
            should_stop=should_stop,
            progress=(None if progress is None else lambda value: progress(f"expert epoch={value.epoch} validation_loss={value.validation_total_loss:.6f}")),
        )
    _write_state(config, "evaluating", iteration, collection=asdict(collection), training=asdict(training))
    if should_stop is not None and should_stop("evaluation", 0):
        raise ExpertIterationInterrupted("expert evaluation interrupted")
    if progress:
        progress(f"expert iteration={iteration} phase=evaluation")
    from dracula.expert_evaluation import evaluate_expert_candidate

    evaluation = evaluate_expert_candidate(
        config,
        iteration,
        resume=resume and prior_phase == "evaluating",
        progress=progress,
        should_stop=should_stop,
    )
    result = ExpertIterationResult(
        iteration,
        "awaiting-decision",
        collection,
        training,
        str(config.output_path / "iterations" / f"{iteration:06d}" / "evaluation" / "evaluation.json"),
        bool(evaluation["eligible"]),
    )
    result_state = asdict(result)
    result_state.pop("iteration")
    result_state.pop("phase")
    _write_state(config, "awaiting-decision", iteration, **result_state)
    return result


def accept_candidate(config: ExpertConfig) -> dict[str, object]:
    state = _read_json(config.output_path / "state.json")
    if not isinstance(state, dict) or state.get("phase") != "awaiting-decision":
        raise ExpertIterationError("no candidate is awaiting a decision")
    iteration = int(state["iteration"])
    training = _read_json(config.output_path / "iterations" / f"{iteration:06d}" / "training-result.json")
    pointer = _accepted_pointer(config)
    accepted_iterations = [*pointer.get("accepted_iterations", []), iteration]
    updated = {
        "format_version": EXPERT_ACCEPTED_VERSION,
        "accepted_version": int(pointer["accepted_version"]) + 1,
        "artifact": str(training["candidate_artifact"]),
        "checkpoint": str(training["candidate_checkpoint"]),
        "accepted_iterations": accepted_iterations,
        "history": [
            *pointer.get("history", []),
            {
                "version": int(pointer["accepted_version"]) + 1,
                "artifact": str(training["candidate_artifact"]),
                "checkpoint": str(training["candidate_checkpoint"]),
                "iteration": iteration,
            },
        ],
    }
    _atomic_json(config.output_path / "accepted.json", updated)
    _write_state(config, "accepted", iteration, accepted_version=updated["accepted_version"])
    return updated


def reject_candidate(config: ExpertConfig) -> dict[str, object]:
    state = _read_json(config.output_path / "state.json")
    if not isinstance(state, dict) or state.get("phase") != "awaiting-decision":
        raise ExpertIterationError("no candidate is awaiting a decision")
    iteration = int(state["iteration"])
    pointer = _accepted_pointer(config)
    _write_state(config, "rejected", iteration, accepted_version=pointer["accepted_version"])
    return pointer


def export_accepted(config: ExpertConfig, destination: str | Path) -> Path:
    pointer = _accepted_pointer(config)
    source = Path(str(pointer["artifact"]))
    load_policy_value_artifact(source)
    destination_path = Path(destination).expanduser().resolve()
    _atomic_bytes(destination_path, source.read_bytes())
    load_policy_value_artifact(destination_path)
    return destination_path


def _print_progress(message: str) -> None:
    print(message, flush=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="dracula-expert")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("smoke", "run"):
        command = commands.add_parser(name)
        command.add_argument("--config", required=True)
    for name in ("resume", "evaluate", "accept", "reject"):
        command = commands.add_parser(name)
        command.add_argument("--output", required=True)
    export = commands.add_parser("export")
    export.add_argument("--output", required=True)
    export.add_argument("--destination", required=True)
    args = parser.parse_args(argv)
    if args.command in {"smoke", "run"}:
        result = run_expert_iteration(load_expert_config(args.config), progress=_print_progress)
        print(json.dumps(asdict(result), indent=2, sort_keys=True))
    elif args.command == "resume":
        result = run_expert_iteration(load_expert_config_from_run(args.output), resume=True, progress=_print_progress)
        print(json.dumps(asdict(result), indent=2, sort_keys=True))
    elif args.command == "evaluate":
        config = load_expert_config_from_run(args.output)
        state = _read_json(config.output_path / "state.json")
        from dracula.expert_evaluation import evaluate_expert_candidate

        print(json.dumps(evaluate_expert_candidate(config, int(state["iteration"]), resume=True, progress=_print_progress), indent=2, sort_keys=True))  # type: ignore[index]
    elif args.command == "accept":
        print(json.dumps(accept_candidate(load_expert_config_from_run(args.output)), indent=2, sort_keys=True))
    elif args.command == "reject":
        print(json.dumps(reject_candidate(load_expert_config_from_run(args.output)), indent=2, sort_keys=True))
    else:
        print(export_accepted(load_expert_config_from_run(args.output), args.destination))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = (
    "ExpertCollectionResult",
    "ExpertConfig",
    "ExpertEpochMetrics",
    "ExpertIterationError",
    "ExpertIterationInterrupted",
    "ExpertIterationResult",
    "ExpertTrainingResult",
    "accept_candidate",
    "collect_expert_iteration",
    "export_accepted",
    "load_expert_checkpoint",
    "load_expert_config",
    "load_expert_config_from_run",
    "load_expert_manifest",
    "load_expert_shard",
    "optimize_expert_candidate",
    "reject_candidate",
    "replay_epoch_dataset",
    "run_expert_iteration",
)

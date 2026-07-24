"""Deterministic search-teacher collection and sealed dataset artifacts."""

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
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from enum import StrEnum
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
    OBSERVATION_SCHEMA_VERSION,
    VALUE_SCHEMA_VERSION,
)
from dracula.randomness import Sha256CounterStream, derive_seed, seed_hex
from dracula.search import (
    GREEDY_RESPONSE_SCHEMA_VERSION,
    INFORMATION_STATE_SCHEMA_VERSION,
    STRATEGIC_SEARCH_SCHEMA_VERSION,
    SearchInformationState,
    StrategicInformationSetSearch,
    StrategicSearchConfig,
    StrategicSearchResult,
    SearchInterrupted,
    derive_strategic_search_request_seed,
    information_state_fingerprint,
    information_state_from_engine,
    policy_input_from_information_state,
)

TEACHER_DATASET_SCHEMA_VERSION = "dracula-search-teacher-dataset-v2"
TEACHER_SHARD_FORMAT_VERSION = "dracula-search-teacher-shard-v2"
TEACHER_MANIFEST_FORMAT_VERSION = "dracula-search-teacher-manifest-v2"
TEACHER_CACHE_FORMAT_VERSION = "dracula-search-teacher-cache-v2"
TEACHER_STATE_FORMAT_VERSION = "dracula-search-teacher-state-v2"
TEACHER_CONFIG_FORMAT_VERSION = "dracula-search-teacher-config-v2"
TEACHER_SPLITS_FORMAT_VERSION = "dracula-search-teacher-splits-v2"

LEGACY_TEACHER_DATASET_SCHEMA_VERSION = "dracula-search-teacher-dataset-v1"
LEGACY_TEACHER_SHARD_FORMAT_VERSION = "dracula-search-teacher-shard-v1"
LEGACY_TEACHER_MANIFEST_FORMAT_VERSION = "dracula-search-teacher-manifest-v1"
LEGACY_TEACHER_CACHE_FORMAT_VERSION = "dracula-search-teacher-cache-v1"
LEGACY_TEACHER_CONFIG_FORMAT_VERSION = "dracula-search-teacher-config-v1"
LEGACY_SEARCH_SCHEMA_VERSION = "dracula-information-search-v1"

APPROVED_TEACHER_PROFILE = "shallow-response-teacher-v2-32x4"
TEACHER_APPROVAL_IDENTITY = "manual-browser-approval-2026-07-23"
TEACHER_VALIDATION_REPORT_DIGEST = (
    "53574e4132686152fd9e9f2d962a6c31f0302ba7a285fe1779f07bee42c5ce77"
)
APPROVED_OUTER_SIMULATION_BUDGET = 32
APPROVED_RESPONSE_COMPLETIONS_PER_ACTION = 4

FIXTURE_SEED_NAMESPACE = "dracula-search-fixture-v1"
FIXTURE_ID_NAMESPACE = "dracula-search-fixture-id-v1"
ACTION_SELECTION_NAMESPACE = "dracula-self-play-action-v2"
MAX_WORKERS = 6
DEFAULT_COLLECTION_WORKERS = 4
EXAMPLES_PER_GAME = 42
EXAMPLES_PER_PLAYER = 21
_STOP_REQUEST_FILENAME = ".stop-requested"

_DIGEST_LENGTH = hashlib.sha256().digest_size * 2
_FORBIDDEN_ARTIFACT_KEYS = {
    "authoritative_state",
    "determination",
    "determinization",
    "engine_seed",
    "game_seed",
    "hands",
    "opponent_hand",
    "principal_continuation",
    "sampled_stock",
    "search_tree",
    "stock",
    "tree",
}


class TeacherCollectionError(ValueError):
    """Teacher configuration, collection, or artifact is invalid."""


class TeacherCollectionInterrupted(RuntimeError):
    """Collection stopped without committing a partial game shard."""


class FixtureSplit(StrEnum):
    TRAINING = "training"
    VALIDATION = "validation"
    ABSOLUTE = "absolute"


@dataclass(frozen=True, slots=True)
class TeacherCollectionConfig:
    run_id: str
    root_seed: str
    output_directory: str
    simulation_budget: int = APPROVED_OUTER_SIMULATION_BUDGET
    response_completions_per_action: int = (
        APPROVED_RESPONSE_COMPLETIONS_PER_ACTION
    )
    exploration_constant: float = math.sqrt(2.0)
    training_games: int = 216
    validation_games: int = 24
    absolute_fixtures: int = 60
    workers: int = DEFAULT_COLLECTION_WORKERS
    early_placement_count: int = 4

    def __post_init__(self) -> None:
        if not isinstance(self.run_id, str) or not self.run_id:
            raise TeacherCollectionError("run ID must be nonempty")
        if not isinstance(self.root_seed, str) or not self.root_seed:
            raise TeacherCollectionError("root seed must be nonempty")
        if not isinstance(self.output_directory, str) or not self.output_directory:
            raise TeacherCollectionError("output directory must be nonempty")
        try:
            StrategicSearchConfig(
                outer_simulation_budget=self.simulation_budget,
                response_completions_per_action=(
                    self.response_completions_per_action
                ),
                outer_exploration_constant=self.exploration_constant,
            )
        except ValueError as error:
            raise TeacherCollectionError("search configuration is invalid") from error
        if (
            self.simulation_budget != APPROVED_OUTER_SIMULATION_BUDGET
            or self.response_completions_per_action
            != APPROVED_RESPONSE_COMPLETIONS_PER_ACTION
            or self.exploration_constant != math.sqrt(2.0)
        ):
            raise TeacherCollectionError(
                "new teacher collection is locked to the approved 32x4 profile"
            )
        counts = (self.training_games, self.validation_games, self.absolute_fixtures)
        if any(type(value) is not int or value < 0 for value in counts):
            raise TeacherCollectionError("fixture counts must be non-negative integers")
        if self.training_games + self.validation_games < 1:
            raise TeacherCollectionError("collection requires at least one dataset game")
        if type(self.workers) is not int or not 1 <= self.workers <= MAX_WORKERS:
            raise TeacherCollectionError(f"workers must be between 1 and {MAX_WORKERS}")
        if type(self.early_placement_count) is not int or not 0 <= self.early_placement_count <= 7:
            raise TeacherCollectionError("early placement count must be between zero and seven")

    @property
    def output_path(self) -> Path:
        return Path(self.output_directory).expanduser().resolve()

    @property
    def search_config(self) -> StrategicSearchConfig:
        return StrategicSearchConfig(
            outer_simulation_budget=self.simulation_budget,
            response_completions_per_action=(
                self.response_completions_per_action
            ),
            outer_exploration_constant=self.exploration_constant,
        )

    @property
    def teacher_profile_digest(self) -> str:
        return _json_digest(
            {
                "approval_identity": TEACHER_APPROVAL_IDENTITY,
                "response_config_digest": self.search_config.response_config.digest,
                "response_schema_version": GREEDY_RESPONSE_SCHEMA_VERSION,
                "search_config_digest": self.search_config.digest,
                "search_schema_version": STRATEGIC_SEARCH_SCHEMA_VERSION,
                "teacher_profile": APPROVED_TEACHER_PROFILE,
                "validation_report_digest": TEACHER_VALIDATION_REPORT_DIGEST,
            }
        )

    @property
    def digest(self) -> str:
        values = _resolved_config_data(self)
        del values["output_directory"]
        return _json_digest(values)


@dataclass(frozen=True, slots=True)
class TeacherFixture:
    split: FixtureSplit
    index: int
    fixture_id: str
    _game_seed: str


@dataclass(frozen=True, slots=True)
class TeacherShardSummary:
    split: FixtureSplit
    fixture_index: int
    fixture_id: str
    relative_path: str
    content_digest: str
    examples: int
    simulations: int
    search_seconds: float
    cache_hits: int
    peak_rss_bytes: int


@dataclass(frozen=True, slots=True)
class TeacherCollectionMetrics:
    examples: int
    simulations: int
    search_seconds: float
    wall_seconds: float
    examples_per_hour: float
    simulations_per_second: float
    decision_latency_p50_seconds: float
    decision_latency_p95_seconds: float
    peak_rss_bytes: int
    cache_hits: int
    failed_decisions: int

    def __post_init__(self) -> None:
        counts = (
            self.examples,
            self.simulations,
            self.peak_rss_bytes,
            self.cache_hits,
            self.failed_decisions,
        )
        if any(type(value) is not int or value < 0 for value in counts):
            raise TeacherCollectionError("collection metric counts must be non-negative")
        measurements = (
            self.search_seconds,
            self.wall_seconds,
            self.examples_per_hour,
            self.simulations_per_second,
            self.decision_latency_p50_seconds,
            self.decision_latency_p95_seconds,
        )
        if any(
            type(value) not in (int, float)
            or not math.isfinite(value)
            or value < 0
            for value in measurements
        ):
            raise TeacherCollectionError("collection measurements must be finite")


@dataclass(frozen=True, slots=True)
class TeacherDatasetInspection:
    config_digest: str
    training_games: int
    validation_games: int
    examples: int
    simulations: int
    dataset_digest: str
    metrics: TeacherCollectionMetrics


@dataclass(frozen=True, slots=True)
class _PendingDecision:
    observation: Tensor
    legal_mask: Tensor
    search_visits: Tensor
    search_policy: Tensor
    selected_action_index: int
    information_state_digest: str
    fixture_id: str
    split: FixtureSplit
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
        raise TeacherCollectionError("artifact metadata must be canonical JSON") from error


def _json_digest(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _require_digest(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != _DIGEST_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise TeacherCollectionError(f"{label} must be a lowercase SHA-256 digest")
    return value


def derive_teacher_fixture(
    root_seed: str,
    split: FixtureSplit,
    index: int,
) -> TeacherFixture:
    if not isinstance(root_seed, str) or not root_seed:
        raise TeacherCollectionError("root seed must be nonempty")
    try:
        split = FixtureSplit(split)
    except (TypeError, ValueError) as error:
        raise TeacherCollectionError("fixture split is invalid") from error
    if type(index) is not int or index < 0:
        raise TeacherCollectionError("fixture index must be non-negative")
    phase = "evaluation" if split is FixtureSplit.ABSOLUTE else "teacher"
    components = (root_seed, phase, split.value, str(index))
    fixture_id = seed_hex(derive_seed(FIXTURE_ID_NAMESPACE, *components))
    game_seed = seed_hex(derive_seed(FIXTURE_SEED_NAMESPACE, *components))
    return TeacherFixture(split, index, fixture_id, game_seed)


def fixture_schedule(config: TeacherCollectionConfig) -> tuple[TeacherFixture, ...]:
    return tuple(
        derive_teacher_fixture(config.root_seed, split, index)
        for split, count in (
            (FixtureSplit.TRAINING, config.training_games),
            (FixtureSplit.VALIDATION, config.validation_games),
            (FixtureSplit.ABSOLUTE, config.absolute_fixtures),
        )
        for index in range(count)
    )


def _resolved_config_data(config: TeacherCollectionConfig) -> dict[str, object]:
    values = asdict(config)
    values["output_directory"] = str(config.output_path)
    values["format_version"] = TEACHER_CONFIG_FORMAT_VERSION
    values["teacher_profile"] = APPROVED_TEACHER_PROFILE
    values["teacher_profile_digest"] = config.teacher_profile_digest
    values["approval_identity"] = TEACHER_APPROVAL_IDENTITY
    values["validation_report_digest"] = TEACHER_VALIDATION_REPORT_DIGEST
    values["search_schema_version"] = STRATEGIC_SEARCH_SCHEMA_VERSION
    values["response_schema_version"] = GREEDY_RESPONSE_SCHEMA_VERSION
    values["search_config_digest"] = config.search_config.digest
    values["response_config_digest"] = config.search_config.response_config.digest
    values["information_state_schema_version"] = INFORMATION_STATE_SCHEMA_VERSION
    values["observation_schema_version"] = OBSERVATION_SCHEMA_VERSION
    values["action_schema_version"] = ACTION_SCHEMA_VERSION
    values["value_schema_version"] = VALUE_SCHEMA_VERSION
    values["teacher_dataset_schema_version"] = TEACHER_DATASET_SCHEMA_VERSION
    return values


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


def _load_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise TeacherCollectionError(f"could not read JSON artifact: {path}") from error


def _peak_rss_bytes() -> int:
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(raw if sys.platform == "darwin" else raw * 1024)


def _configure_collection_worker() -> None:
    """Keep one game worker from creating another CPU thread pool."""

    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = "1"
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError as error:
            raise TeacherCollectionError(
                "PyTorch inter-op threads were initialized before collection"
            ) from error
    if torch.get_num_threads() != 1 or torch.get_num_interop_threads() != 1:
        raise TeacherCollectionError(
            "teacher collection workers require one PyTorch thread"
        )


def _select_action_from_visits(
    visits: tuple[int, ...],
    *,
    config: TeacherCollectionConfig,
    fixture: TeacherFixture,
    round_number: int,
    placement_number: int,
    maximum_visit_action: int,
) -> int:
    if placement_number > config.early_placement_count:
        return maximum_visit_action
    total = sum(visits)
    if total != config.simulation_budget:
        raise TeacherCollectionError("search visits do not equal the simulation budget")
    seed = derive_seed(
        ACTION_SELECTION_NAMESPACE,
        config.root_seed,
        "teacher",
        fixture.fixture_id,
        str(round_number),
        str(placement_number),
    )
    threshold = Sha256CounterStream(seed).randbelow(total)
    cumulative = 0
    for action_index, count in enumerate(visits):
        cumulative += count
        if threshold < cumulative:
            return action_index
    raise TeacherCollectionError("visit distribution could not select an action")


def _decision_cache_path(
    output: Path,
    fixture: TeacherFixture,
    round_number: int,
    placement_number: int,
    information_digest: str,
) -> Path:
    return (
        output
        / "teacher"
        / "cache"
        / fixture.split.value
        / fixture.fixture_id
        / f"r{round_number:02d}-p{placement_number:02d}-{information_digest[:16]}.json"
    )


def _run_teacher_search(
    information: SearchInformationState,
    fixture: TeacherFixture,
    config: TeacherCollectionConfig,
    should_stop: Callable[[], bool] | None,
) -> StrategicSearchResult:
    request_seed = derive_strategic_search_request_seed(
        fixture.fixture_id,
        information,
        config.search_config.digest,
    )
    return StrategicInformationSetSearch(config.search_config).search(
        information,
        request_seed,
        should_stop,
    )


def _validate_decision_cache(
    value: object,
    *,
    fixture: TeacherFixture,
    config: TeacherCollectionConfig,
    information_digest: str,
    legal_mask: Tensor,
    round_number: int,
    placement_number: int,
) -> tuple[tuple[int, ...], int, float]:
    if not isinstance(value, dict) or set(value) != {
        "format_version",
        "teacher_profile",
        "teacher_profile_digest",
        "approval_identity",
        "validation_report_digest",
        "fixture_id",
        "split",
        "round_number",
        "placement_number",
        "information_state_digest",
        "search_schema_version",
        "response_schema_version",
        "search_config_digest",
        "response_config_digest",
        "action_visits",
        "selected_action_index",
        "search_seconds",
    }:
        raise TeacherCollectionError("decision cache has an invalid shape")
    expected = {
        "format_version": TEACHER_CACHE_FORMAT_VERSION,
        "teacher_profile": APPROVED_TEACHER_PROFILE,
        "teacher_profile_digest": config.teacher_profile_digest,
        "approval_identity": TEACHER_APPROVAL_IDENTITY,
        "validation_report_digest": TEACHER_VALIDATION_REPORT_DIGEST,
        "fixture_id": fixture.fixture_id,
        "split": fixture.split.value,
        "round_number": round_number,
        "placement_number": placement_number,
        "information_state_digest": information_digest,
        "search_schema_version": STRATEGIC_SEARCH_SCHEMA_VERSION,
        "response_schema_version": GREEDY_RESPONSE_SCHEMA_VERSION,
        "search_config_digest": config.search_config.digest,
        "response_config_digest": config.search_config.response_config.digest,
    }
    if any(value[key] != expected_value for key, expected_value in expected.items()):
        raise TeacherCollectionError("decision cache does not match the request")
    visits_value = value["action_visits"]
    if (
        not isinstance(visits_value, list)
        or len(visits_value) != ACTION_COUNT
        or any(type(count) is not int or count < 0 for count in visits_value)
        or sum(visits_value) != config.simulation_budget
    ):
        raise TeacherCollectionError("decision cache visits are invalid")
    visits = tuple(visits_value)
    mask = tuple(bool(item) for item in legal_mask.flatten().tolist())
    if any(count and not mask[index] for index, count in enumerate(visits)):
        raise TeacherCollectionError("decision cache assigns visits to an illegal action")
    selected = value["selected_action_index"]
    if type(selected) is not int or not 0 <= selected < ACTION_COUNT or not mask[selected]:
        raise TeacherCollectionError("decision cache selected an illegal action")
    search_seconds = value["search_seconds"]
    if (
        type(search_seconds) not in (int, float)
        or not math.isfinite(search_seconds)
        or search_seconds < 0
    ):
        raise TeacherCollectionError("decision cache search time is invalid")
    return visits, selected, float(search_seconds)


def _teacher_decision(
    state,
    fixture: TeacherFixture,
    config: TeacherCollectionConfig,
    should_stop: Callable[[], bool] | None,
) -> tuple[_PendingDecision, int, float, bool]:
    actor = state.active_player
    if actor is None:
        raise TeacherCollectionError("teacher decision requires an active player")
    context = build_policy_turn_context(state, actor)
    if context.kind is PolicyTurnKind.FORCED_RECURRENT_TRANSITION:
        raise TeacherCollectionError("forced placement reached the learned decision path")
    information = information_state_from_engine(state)
    policy_input = policy_input_from_information_state(information)
    information_digest = information_state_fingerprint(information)
    round_number = state.round_number
    placement_number = len(state.current_round_moves) + 1
    cache_path = _decision_cache_path(
        config.output_path,
        fixture,
        round_number,
        placement_number,
        information_digest,
    )
    cache_hit = cache_path.exists()
    elapsed = 0.0
    if cache_hit:
        visits, selected_action, elapsed = _validate_decision_cache(
            _load_json(cache_path),
            fixture=fixture,
            config=config,
            information_digest=information_digest,
            legal_mask=policy_input.legal_mask,
            round_number=round_number,
            placement_number=placement_number,
        )
    else:
        if should_stop is not None and should_stop():
            raise TeacherCollectionInterrupted("collection stopped before a decision")
        started = time.perf_counter()
        try:
            result = _run_teacher_search(
                information,
                fixture,
                config,
                should_stop,
            )
        except SearchInterrupted as error:
            raise TeacherCollectionInterrupted(str(error)) from error
        elapsed = time.perf_counter() - started
        if result.information_state_fingerprint != information_digest:
            raise TeacherCollectionError("search returned another information state")
        if result.config_digest != config.search_config.digest:
            raise TeacherCollectionError("search returned another configuration")
        if result.simulation_count != config.simulation_budget:
            raise TeacherCollectionError("search simulation count differs")
        visits = result.action_visits
        selected_action = _select_action_from_visits(
            visits,
            config=config,
            fixture=fixture,
            round_number=round_number,
            placement_number=placement_number,
            maximum_visit_action=result.selected_action_index,
        )
        _atomic_json(
            cache_path,
            {
                "format_version": TEACHER_CACHE_FORMAT_VERSION,
                "teacher_profile": APPROVED_TEACHER_PROFILE,
                "teacher_profile_digest": config.teacher_profile_digest,
                "approval_identity": TEACHER_APPROVAL_IDENTITY,
                "validation_report_digest": TEACHER_VALIDATION_REPORT_DIGEST,
                "fixture_id": fixture.fixture_id,
                "split": fixture.split.value,
                "round_number": round_number,
                "placement_number": placement_number,
                "information_state_digest": information_digest,
                "search_schema_version": STRATEGIC_SEARCH_SCHEMA_VERSION,
                "response_schema_version": GREEDY_RESPONSE_SCHEMA_VERSION,
                "search_config_digest": config.search_config.digest,
                "response_config_digest": (
                    config.search_config.response_config.digest
                ),
                "action_visits": list(visits),
                "selected_action_index": selected_action,
                "search_seconds": elapsed,
            },
        )
    visits_tensor = torch.tensor(visits, dtype=torch.int64)
    search_policy = visits_tensor.to(torch.float32) / float(config.simulation_budget)
    return (
        _PendingDecision(
            observation=policy_input.observation.detach().cpu().clone(),
            legal_mask=policy_input.legal_mask.detach().cpu().clone(),
            search_visits=visits_tensor,
            search_policy=search_policy,
            selected_action_index=selected_action,
            information_state_digest=information_digest,
            fixture_id=fixture.fixture_id,
            split=fixture.split,
            player=actor,
            dealer=state.dealer,
            round_number=round_number,
            placement_number=placement_number,
        ),
        config.simulation_budget,
        elapsed,
        cache_hit,
    )


def _empty_shard_columns() -> dict[str, list[object]]:
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


def _append_completed_round(
    columns: dict[str, list[object]],
    pending: Sequence[_PendingDecision],
    round_result,
) -> None:
    queen_return = (
        round_result.round_scores[EnginePlayer.QUEEN]
        - round_result.round_scores[EnginePlayer.KING]
    ) / 150.0
    returns = {
        EnginePlayer.QUEEN: queen_return,
        EnginePlayer.KING: -queen_return,
    }
    for decision in pending:
        columns["observations"].append(decision.observation)
        columns["legal_masks"].append(decision.legal_mask)
        columns["search_visits"].append(decision.search_visits)
        columns["search_policies"].append(decision.search_policy)
        columns["selected_action_indices"].append(decision.selected_action_index)
        columns["round_returns"].append(returns[decision.player])
        columns["information_state_digests"].append(
            decision.information_state_digest
        )
        columns["fixture_ids"].append(decision.fixture_id)
        columns["splits"].append(decision.split.value)
        columns["players"].append(decision.player.value)
        columns["dealers"].append(decision.dealer.value)
        columns["round_numbers"].append(decision.round_number)
        columns["placement_numbers"].append(decision.placement_number)


def _tensor_columns(columns: dict[str, list[object]]) -> dict[str, object]:
    return {
        "observations": torch.stack(columns["observations"]),
        "legal_masks": torch.stack(columns["legal_masks"]),
        "search_visits": torch.stack(columns["search_visits"]),
        "search_policies": torch.stack(columns["search_policies"]),
        "selected_action_indices": torch.tensor(
            columns["selected_action_indices"], dtype=torch.int64
        ),
        "round_returns": torch.tensor(columns["round_returns"], dtype=torch.float32),
        "information_state_digests": tuple(columns["information_state_digests"]),
        "fixture_ids": tuple(columns["fixture_ids"]),
        "splits": tuple(columns["splits"]),
        "players": tuple(columns["players"]),
        "dealers": tuple(columns["dealers"]),
        "round_numbers": torch.tensor(columns["round_numbers"], dtype=torch.int64),
        "placement_numbers": torch.tensor(
            columns["placement_numbers"], dtype=torch.int64
        ),
    }


def _update_digest_with_tensor(digest, name: str, tensor: Tensor) -> None:
    value = tensor.detach().cpu().contiguous()
    digest.update(name.encode("utf-8"))
    digest.update(b"\0")
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(",".join(map(str, value.shape)).encode("ascii"))
    digest.update(b"\0")
    digest.update(value.numpy().tobytes(order="C"))


def _shard_content_digest(metadata: Mapping[str, object], columns: Mapping[str, object]) -> str:
    digest = hashlib.sha256(_canonical_json(dict(metadata)))
    for name in sorted(columns):
        value = columns[name]
        if isinstance(value, Tensor):
            _update_digest_with_tensor(digest, name, value)
        else:
            digest.update(name.encode("utf-8"))
            digest.update(b"\0")
            digest.update(_canonical_json(value))
    return digest.hexdigest()


def _shard_metadata(
    fixture: TeacherFixture,
    config: TeacherCollectionConfig,
) -> dict[str, object]:
    return {
        "dataset_schema_version": TEACHER_DATASET_SCHEMA_VERSION,
        "teacher_profile": APPROVED_TEACHER_PROFILE,
        "teacher_profile_digest": config.teacher_profile_digest,
        "approval_identity": TEACHER_APPROVAL_IDENTITY,
        "validation_report_digest": TEACHER_VALIDATION_REPORT_DIGEST,
        "information_state_schema_version": INFORMATION_STATE_SCHEMA_VERSION,
        "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
        "action_schema_version": ACTION_SCHEMA_VERSION,
        "value_schema_version": VALUE_SCHEMA_VERSION,
        "search_schema_version": STRATEGIC_SEARCH_SCHEMA_VERSION,
        "response_schema_version": GREEDY_RESPONSE_SCHEMA_VERSION,
        "search_config_digest": config.search_config.digest,
        "response_config_digest": config.search_config.response_config.digest,
        "collection_config_digest": config.digest,
        "fixture_id": fixture.fixture_id,
        "split": fixture.split.value,
        "fixture_index": fixture.index,
    }


def _validate_shard_payload(payload: object) -> dict[str, object]:
    if not isinstance(payload, dict) or set(payload) != {
        "format_version",
        "metadata",
        "columns",
        "content_digest",
    }:
        raise TeacherCollectionError("teacher shard payload is invalid")
    format_version = payload["format_version"]
    if format_version not in {
        TEACHER_SHARD_FORMAT_VERSION,
        LEGACY_TEACHER_SHARD_FORMAT_VERSION,
    }:
        raise TeacherCollectionError("teacher shard format is incompatible")
    metadata = payload["metadata"]
    columns = payload["columns"]
    if not isinstance(metadata, dict) or not isinstance(columns, dict):
        raise TeacherCollectionError("teacher shard sections are invalid")
    if set(columns) != set(_empty_shard_columns()):
        raise TeacherCollectionError("teacher shard columns are invalid")
    if any(key in _FORBIDDEN_ARTIFACT_KEYS for key in (*metadata, *columns)):
        raise TeacherCollectionError("teacher shard contains a private field")
    if format_version == TEACHER_SHARD_FORMAT_VERSION:
        expected_metadata_keys = {
            "dataset_schema_version",
            "teacher_profile",
            "teacher_profile_digest",
            "approval_identity",
            "validation_report_digest",
            "information_state_schema_version",
            "observation_schema_version",
            "action_schema_version",
            "value_schema_version",
            "search_schema_version",
            "response_schema_version",
            "search_config_digest",
            "response_config_digest",
            "collection_config_digest",
            "fixture_id",
            "split",
            "fixture_index",
        }
        expected_versions = {
            "dataset_schema_version": TEACHER_DATASET_SCHEMA_VERSION,
            "teacher_profile": APPROVED_TEACHER_PROFILE,
            "approval_identity": TEACHER_APPROVAL_IDENTITY,
            "validation_report_digest": TEACHER_VALIDATION_REPORT_DIGEST,
            "information_state_schema_version": INFORMATION_STATE_SCHEMA_VERSION,
            "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
            "action_schema_version": ACTION_SCHEMA_VERSION,
            "value_schema_version": VALUE_SCHEMA_VERSION,
            "search_schema_version": STRATEGIC_SEARCH_SCHEMA_VERSION,
            "response_schema_version": GREEDY_RESPONSE_SCHEMA_VERSION,
        }
    else:
        expected_metadata_keys = {
            "dataset_schema_version",
            "information_state_schema_version",
            "observation_schema_version",
            "action_schema_version",
            "value_schema_version",
            "search_schema_version",
            "search_config_digest",
            "collection_config_digest",
            "fixture_id",
            "split",
            "fixture_index",
        }
        expected_versions = {
            "dataset_schema_version": LEGACY_TEACHER_DATASET_SCHEMA_VERSION,
            "information_state_schema_version": INFORMATION_STATE_SCHEMA_VERSION,
            "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
            "action_schema_version": ACTION_SCHEMA_VERSION,
            "value_schema_version": VALUE_SCHEMA_VERSION,
            "search_schema_version": LEGACY_SEARCH_SCHEMA_VERSION,
        }
    if set(metadata) != expected_metadata_keys:
        raise TeacherCollectionError("teacher shard metadata shape is incompatible")
    if any(metadata.get(key) != value for key, value in expected_versions.items()):
        raise TeacherCollectionError("teacher shard contract version is incompatible")
    if format_version == TEACHER_SHARD_FORMAT_VERSION:
        for key in (
            "teacher_profile_digest",
            "validation_report_digest",
            "response_config_digest",
        ):
            _require_digest(metadata.get(key), key.replace("_", " "))
    for key in (
        "search_config_digest",
        "collection_config_digest",
        "fixture_id",
    ):
        _require_digest(metadata.get(key), key.replace("_", " "))
    try:
        split = FixtureSplit(metadata.get("split"))
    except (TypeError, ValueError) as error:
        raise TeacherCollectionError("teacher shard split is invalid") from error
    if split is FixtureSplit.ABSOLUTE:
        raise TeacherCollectionError("absolute fixtures cannot enter a teacher shard")
    if type(metadata.get("fixture_index")) is not int or metadata["fixture_index"] < 0:
        raise TeacherCollectionError("teacher shard fixture index is invalid")

    observations = columns["observations"]
    legal_masks = columns["legal_masks"]
    visits = columns["search_visits"]
    policies = columns["search_policies"]
    selected = columns["selected_action_indices"]
    returns = columns["round_returns"]
    round_numbers = columns["round_numbers"]
    placement_numbers = columns["placement_numbers"]
    tensors = (
        observations,
        legal_masks,
        visits,
        policies,
        selected,
        returns,
        round_numbers,
        placement_numbers,
    )
    if any(not isinstance(value, Tensor) or value.device.type != "cpu" for value in tensors):
        raise TeacherCollectionError("teacher shard tensors must be on CPU")
    row_count = observations.shape[0]
    expected_shapes = (
        (observations, torch.bool, (row_count, 875)),
        (legal_masks, torch.bool, (row_count, 4, 8)),
        (visits, torch.int64, (row_count, ACTION_COUNT)),
        (policies, torch.float32, (row_count, ACTION_COUNT)),
        (selected, torch.int64, (row_count,)),
        (returns, torch.float32, (row_count,)),
        (round_numbers, torch.int64, (row_count,)),
        (placement_numbers, torch.int64, (row_count,)),
    )
    for tensor, dtype, shape in expected_shapes:
        if tensor.dtype is not dtype or tensor.shape != shape:
            raise TeacherCollectionError("teacher shard tensor shape or dtype differs")
    string_columns = (
        "information_state_digests",
        "fixture_ids",
        "splits",
        "players",
        "dealers",
    )
    if any(
        not isinstance(columns[name], tuple) or len(columns[name]) != row_count
        for name in string_columns
    ):
        raise TeacherCollectionError("teacher shard identifier columns differ")
    if row_count != EXAMPLES_PER_GAME:
        raise TeacherCollectionError("a full teacher game must contain 42 examples")
    if not torch.isfinite(policies).all() or not torch.isfinite(returns).all():
        raise TeacherCollectionError("teacher shard contains non-finite targets")
    if torch.any(visits < 0):
        raise TeacherCollectionError("teacher shard contains negative visits")
    simulation_budget = visits.sum(dim=1)
    if not torch.all(simulation_budget == simulation_budget[0]):
        raise TeacherCollectionError("teacher shard search budgets differ")
    if not torch.allclose(
        policies,
        visits.to(torch.float32) / simulation_budget[:, None].to(torch.float32),
        atol=0.0,
        rtol=0.0,
    ):
        raise TeacherCollectionError("teacher search policies do not match visits")
    flattened_masks = legal_masks.flatten(start_dim=1)
    if torch.any(visits[~flattened_masks] != 0) or torch.any(policies[~flattened_masks] != 0):
        raise TeacherCollectionError("teacher targets assign mass to illegal actions")
    if torch.any(selected < 0) or torch.any(selected >= ACTION_COUNT):
        raise TeacherCollectionError("teacher selected action is out of range")
    if not torch.all(flattened_masks.gather(1, selected[:, None]).squeeze(1)):
        raise TeacherCollectionError("teacher selected action is illegal")
    if not torch.all(visits.gather(1, selected[:, None]).squeeze(1) > 0):
        raise TeacherCollectionError("teacher selected action has no search visit")
    if torch.any(returns < -1) or torch.any(returns > 1):
        raise TeacherCollectionError("teacher return is outside [-1,1]")
    if torch.any(round_numbers < 1) or torch.any(round_numbers > 6):
        raise TeacherCollectionError("teacher round number is invalid")
    if torch.any(placement_numbers < 1) or torch.any(placement_numbers > 7):
        raise TeacherCollectionError("forced placement entered the teacher rows")
    if any(value != metadata["fixture_id"] for value in columns["fixture_ids"]):
        raise TeacherCollectionError("teacher rows name another fixture")
    if any(value != split.value for value in columns["splits"]):
        raise TeacherCollectionError("teacher rows name another split")
    if any(value not in {player.value for player in EnginePlayer} for value in columns["players"]):
        raise TeacherCollectionError("teacher row player is invalid")
    if any(value not in {player.value for player in EnginePlayer} for value in columns["dealers"]):
        raise TeacherCollectionError("teacher row dealer is invalid")
    for value in columns["information_state_digests"]:
        _require_digest(value, "information state digest")
    expected_rounds = [round_number for round_number in range(1, 7) for _ in range(7)]
    expected_placements = list(range(1, 8)) * 6
    if (
        round_numbers.tolist() != expected_rounds
        or placement_numbers.tolist() != expected_placements
    ):
        raise TeacherCollectionError("teacher rows are not in canonical game order")
    initial_dealer = EnginePlayer(columns["dealers"][0])
    for round_number in range(1, 7):
        expected_dealer = (
            initial_dealer if round_number % 2 == 1 else other_player(initial_dealer)
        )
        round_indexes = [
            index
            for index, value in enumerate(round_numbers.tolist())
            if value == round_number
        ]
        if any(columns["dealers"][index] != expected_dealer.value for index in round_indexes):
            raise TeacherCollectionError("teacher dealer sequence differs")
        expected_non_dealer = other_player(expected_dealer)
        expected_players = [
            expected_non_dealer.value if placement % 2 == 1 else expected_dealer.value
            for placement in range(1, 8)
        ]
        if [columns["players"][index] for index in round_indexes] != expected_players:
            raise TeacherCollectionError("teacher player sequence differs")
        for player in EnginePlayer:
            indexes = [
                index
                for index, (row_round, row_player) in enumerate(
                    zip(round_numbers.tolist(), columns["players"], strict=True)
                )
                if row_round == round_number and row_player == player.value
            ]
            expected_count = (
                3 if columns["dealers"][indexes[0]] == player.value else 4
            )
            if len(indexes) != expected_count:
                raise TeacherCollectionError("teacher role balance or forced exclusion differs")
            values = returns[indexes]
            if not torch.all(values == values[0]):
                raise TeacherCollectionError("one player round has inconsistent returns")
        queen_index = next(
            index
            for index, (row_round, player) in enumerate(
                zip(round_numbers.tolist(), columns["players"], strict=True)
            )
            if row_round == round_number and player == EnginePlayer.QUEEN.value
        )
        king_index = next(
            index
            for index, (row_round, player) in enumerate(
                zip(round_numbers.tolist(), columns["players"], strict=True)
            )
            if row_round == round_number and player == EnginePlayer.KING.value
        )
        if returns[queen_index].item() != -returns[king_index].item():
            raise TeacherCollectionError("player-relative round returns are not opposite")
    expected_digest = _shard_content_digest(metadata, columns)
    if payload["content_digest"] != expected_digest:
        raise TeacherCollectionError("teacher shard content digest differs")
    return payload


def load_teacher_shard(path: str | Path) -> dict[str, object]:
    try:
        payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, ValueError, EOFError, pickle.UnpicklingError) as error:
        raise TeacherCollectionError("teacher shard could not be loaded") from error
    return _validate_shard_payload(payload)


def _atomic_shard(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        torch.save(payload, temporary)
        _validate_shard_payload(
            torch.load(temporary, map_location="cpu", weights_only=True)
        )
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _shard_path(output: Path, fixture: TeacherFixture) -> Path:
    return output / "teacher" / "games" / fixture.split.value / f"{fixture.fixture_id}.pt"


def _collect_fixture(
    fixture: TeacherFixture,
    config: TeacherCollectionConfig,
    should_stop: Callable[[], bool] | None = None,
) -> TeacherShardSummary:
    if fixture.split is FixtureSplit.ABSOLUTE:
        raise TeacherCollectionError("absolute fixtures cannot produce teacher rows")
    state = create_game(fixture._game_seed)
    columns = _empty_shard_columns()
    pending: list[_PendingDecision] = []
    simulations = 0
    search_seconds = 0.0
    cache_hits = 0
    latencies: list[float] = []
    while state.status is not EngineStatus.GAME_COMPLETE:
        while state.status is EngineStatus.PLAYING:
            if should_stop is not None and should_stop():
                raise TeacherCollectionInterrupted("collection stopped before game seal")
            context = build_policy_turn_context(state, state.active_player)
            if context.kind is PolicyTurnKind.FORCED_RECURRENT_TRANSITION:
                move = context.forced_move
                if move is None:
                    raise TeacherCollectionError("forced transition has no engine move")
            else:
                decision, count, elapsed, cache_hit = _teacher_decision(
                    state, fixture, config, should_stop
                )
                pending.append(decision)
                simulations += count
                search_seconds += elapsed
                cache_hits += int(cache_hit)
                latencies.append(elapsed)
                move = context.action_table[decision.selected_action_index]
                if move is None:
                    raise TeacherCollectionError("teacher selected a masked action")
            state = apply_move(state, move).state
        round_result = state.pending_round_result
        if round_result is None:
            raise TeacherCollectionError("completed round has no score result")
        _append_completed_round(columns, pending, round_result)
        pending.clear()
        state = advance_after_round(state)
    sealed_columns = _tensor_columns(columns)
    metadata = _shard_metadata(fixture, config)
    content_digest = _shard_content_digest(metadata, sealed_columns)
    payload = {
        "format_version": TEACHER_SHARD_FORMAT_VERSION,
        "metadata": metadata,
        "columns": sealed_columns,
        "content_digest": content_digest,
    }
    destination = _shard_path(config.output_path, fixture)
    summary = TeacherShardSummary(
        split=fixture.split,
        fixture_index=fixture.index,
        fixture_id=fixture.fixture_id,
        relative_path=str(destination.relative_to(config.output_path)),
        content_digest=content_digest,
        examples=sealed_columns["observations"].shape[0],
        simulations=simulations,
        search_seconds=search_seconds,
        cache_hits=cache_hits,
        peak_rss_bytes=_peak_rss_bytes(),
    )
    # Latencies are process-local metrics and never enter the deterministic shard.
    _atomic_json(
        destination.with_suffix(".metrics.json"),
        {
            "decision_latencies_seconds": latencies,
            "search_seconds": search_seconds,
            "simulations": simulations,
            "cache_hits": cache_hits,
            "peak_rss_bytes": summary.peak_rss_bytes,
            "torch_num_threads": torch.get_num_threads(),
            "torch_num_interop_threads": torch.get_num_interop_threads(),
        },
    )
    _atomic_shard(destination, payload)
    cache_directory = (
        destination.parents[2]
        / "cache"
        / fixture.split.value
        / fixture.fixture_id
    )
    if cache_directory.exists():
        for item in cache_directory.iterdir():
            item.unlink()
        cache_directory.rmdir()
    return summary


def _collect_fixture_worker(
    fixture: TeacherFixture,
    config: TeacherCollectionConfig,
) -> TeacherShardSummary:
    stop_path = (
        config.output_path / "teacher" / _STOP_REQUEST_FILENAME
    )
    return _collect_fixture(fixture, config, stop_path.exists)


def _manifest_document(
    split: FixtureSplit,
    config: TeacherCollectionConfig,
    summaries: Sequence[TeacherShardSummary],
) -> dict[str, object]:
    shards = [
        {
            "fixture_index": item.fixture_index,
            "fixture_id": item.fixture_id,
            "relative_path": item.relative_path,
            "content_digest": item.content_digest,
            "examples": item.examples,
            "simulations": item.simulations,
        }
        for item in sorted(summaries, key=lambda item: item.fixture_index)
    ]
    body = {
        "format_version": TEACHER_MANIFEST_FORMAT_VERSION,
        "dataset_schema_version": TEACHER_DATASET_SCHEMA_VERSION,
        "teacher_profile": APPROVED_TEACHER_PROFILE,
        "teacher_profile_digest": config.teacher_profile_digest,
        "approval_identity": TEACHER_APPROVAL_IDENTITY,
        "validation_report_digest": TEACHER_VALIDATION_REPORT_DIGEST,
        "search_schema_version": STRATEGIC_SEARCH_SCHEMA_VERSION,
        "response_schema_version": GREEDY_RESPONSE_SCHEMA_VERSION,
        "split": split.value,
        "collection_config_digest": config.digest,
        "search_config_digest": config.search_config.digest,
        "response_config_digest": config.search_config.response_config.digest,
        "fixture_count": len(shards),
        "example_count": sum(item["examples"] for item in shards),
        "simulation_count": sum(item["simulations"] for item in shards),
        "shards": shards,
    }
    return {"manifest": body, "manifest_digest": _json_digest(body)}


def load_teacher_manifest(
    output_directory: str | Path,
    split: FixtureSplit,
) -> dict[str, object]:
    output = Path(output_directory).expanduser().resolve()
    document = _load_json(output / "teacher" / f"{FixtureSplit(split).value}-manifest.json")
    if not isinstance(document, dict) or set(document) != {"manifest", "manifest_digest"}:
        raise TeacherCollectionError("teacher manifest document is invalid")
    manifest = document["manifest"]
    if not isinstance(manifest, dict) or document["manifest_digest"] != _json_digest(manifest):
        raise TeacherCollectionError("teacher manifest digest differs")
    manifest_format = manifest.get("format_version")
    if manifest_format not in {
        TEACHER_MANIFEST_FORMAT_VERSION,
        LEGACY_TEACHER_MANIFEST_FORMAT_VERSION,
    }:
        raise TeacherCollectionError("teacher manifest format is incompatible")
    if manifest_format == TEACHER_MANIFEST_FORMAT_VERSION:
        expected_manifest_keys = {
            "format_version",
            "dataset_schema_version",
            "teacher_profile",
            "teacher_profile_digest",
            "approval_identity",
            "validation_report_digest",
            "search_schema_version",
            "response_schema_version",
            "split",
            "collection_config_digest",
            "search_config_digest",
            "response_config_digest",
            "fixture_count",
            "example_count",
            "simulation_count",
            "shards",
        }
        expected_manifest_values = {
            "dataset_schema_version": TEACHER_DATASET_SCHEMA_VERSION,
            "teacher_profile": APPROVED_TEACHER_PROFILE,
            "approval_identity": TEACHER_APPROVAL_IDENTITY,
            "validation_report_digest": TEACHER_VALIDATION_REPORT_DIGEST,
            "search_schema_version": STRATEGIC_SEARCH_SCHEMA_VERSION,
            "response_schema_version": GREEDY_RESPONSE_SCHEMA_VERSION,
        }
        expected_shard_format = TEACHER_SHARD_FORMAT_VERSION
        binding_keys = (
            "dataset_schema_version",
            "teacher_profile",
            "teacher_profile_digest",
            "approval_identity",
            "validation_report_digest",
            "search_schema_version",
            "response_schema_version",
            "collection_config_digest",
            "search_config_digest",
            "response_config_digest",
        )
    else:
        expected_manifest_keys = {
            "format_version",
            "dataset_schema_version",
            "split",
            "collection_config_digest",
            "search_config_digest",
            "fixture_count",
            "example_count",
            "simulation_count",
            "shards",
        }
        expected_manifest_values = {
            "dataset_schema_version": LEGACY_TEACHER_DATASET_SCHEMA_VERSION,
        }
        expected_shard_format = LEGACY_TEACHER_SHARD_FORMAT_VERSION
        binding_keys = (
            "dataset_schema_version",
            "collection_config_digest",
            "search_config_digest",
        )
    if set(manifest) != expected_manifest_keys:
        raise TeacherCollectionError("teacher manifest shape is incompatible")
    if any(
        manifest.get(key) != value
        for key, value in expected_manifest_values.items()
    ):
        raise TeacherCollectionError("teacher manifest contract is incompatible")
    for key in binding_keys:
        if key.endswith("_digest"):
            _require_digest(manifest.get(key), key.replace("_", " "))
    if manifest.get("split") != FixtureSplit(split).value:
        raise TeacherCollectionError("teacher manifest split differs")
    shards = manifest.get("shards")
    if not isinstance(shards, list) or len(shards) != manifest.get("fixture_count"):
        raise TeacherCollectionError("teacher manifest shard list differs")
    examples = 0
    simulations = 0
    fixture_ids: set[str] = set()
    fixture_indexes: set[int] = set()
    for entry in shards:
        if not isinstance(entry, dict) or set(entry) != {
            "fixture_index",
            "fixture_id",
            "relative_path",
            "content_digest",
            "examples",
            "simulations",
        }:
            raise TeacherCollectionError("teacher manifest shard entry is invalid")
        fixture_id = _require_digest(entry["fixture_id"], "fixture ID")
        content_digest = _require_digest(entry["content_digest"], "content digest")
        fixture_index = entry["fixture_index"]
        relative_path = entry["relative_path"]
        if (
            type(fixture_index) is not int
            or fixture_index < 0
            or not isinstance(relative_path, str)
        ):
            raise TeacherCollectionError("teacher manifest fixture identity is invalid")
        relative = Path(relative_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise TeacherCollectionError("teacher manifest path escapes the run")
        shard = load_teacher_shard(output / relative)
        if shard["format_version"] != expected_shard_format:
            raise TeacherCollectionError(
                "teacher manifest mixes artifact contract versions"
            )
        if shard["content_digest"] != entry.get("content_digest"):
            raise TeacherCollectionError("teacher manifest names another shard digest")
        metadata = shard["metadata"]
        if any(metadata.get(key) != manifest.get(key) for key in binding_keys):
            raise TeacherCollectionError(
                "teacher manifest and shard contract bindings differ"
            )
        if (
            metadata["fixture_id"] != fixture_id
            or metadata["fixture_index"] != fixture_index
            or metadata["split"] != FixtureSplit(split).value
            or shard["content_digest"] != content_digest
        ):
            raise TeacherCollectionError("teacher manifest and shard identities differ")
        if fixture_id in fixture_ids or fixture_index in fixture_indexes:
            raise TeacherCollectionError("teacher manifest repeats a fixture")
        fixture_ids.add(fixture_id)
        fixture_indexes.add(fixture_index)
        columns = shard["columns"]
        shard_examples = columns["observations"].shape[0]
        shard_simulations = int(columns["search_visits"].sum().item())
        if entry["examples"] != shard_examples or entry["simulations"] != shard_simulations:
            raise TeacherCollectionError("teacher manifest shard totals differ")
        examples += shard_examples
        simulations += shard_simulations
    if fixture_indexes != set(range(len(shards))):
        raise TeacherCollectionError("teacher manifest fixture indexes are not contiguous")
    if (
        examples != manifest.get("example_count")
        or simulations != manifest.get("simulation_count")
    ):
        raise TeacherCollectionError("teacher manifest totals differ")
    return document


def _load_shard_summary(
    fixture: TeacherFixture,
    config: TeacherCollectionConfig,
) -> TeacherShardSummary | None:
    path = _shard_path(config.output_path, fixture)
    if not path.exists():
        return None
    payload = load_teacher_shard(path)
    metadata = payload["metadata"]
    if (
        metadata["fixture_id"] != fixture.fixture_id
        or metadata["split"] != fixture.split.value
        or metadata["fixture_index"] != fixture.index
        or metadata["collection_config_digest"] != config.digest
    ):
        raise TeacherCollectionError("existing teacher shard belongs to another fixture")
    metrics_path = path.with_suffix(".metrics.json")
    metrics = _load_json(metrics_path) if metrics_path.exists() else {}
    columns = payload["columns"]
    cache_directory = (
        path.parents[2] / "cache" / fixture.split.value / fixture.fixture_id
    )
    if cache_directory.exists():
        for item in cache_directory.iterdir():
            item.unlink()
        cache_directory.rmdir()
    return TeacherShardSummary(
        split=fixture.split,
        fixture_index=fixture.index,
        fixture_id=fixture.fixture_id,
        relative_path=str(path.relative_to(config.output_path)),
        content_digest=payload["content_digest"],
        examples=columns["observations"].shape[0],
        simulations=int(columns["search_visits"].sum().item()),
        search_seconds=float(metrics.get("search_seconds", 0.0)),
        cache_hits=int(metrics.get("cache_hits", 0)),
        peak_rss_bytes=int(metrics.get("peak_rss_bytes", 0)),
    )


def _write_fixture_splits(config: TeacherCollectionConfig) -> None:
    groups = {
        split.value: [
            fixture.fixture_id
            for fixture in fixture_schedule(config)
            if fixture.split is split
        ]
        for split in FixtureSplit
    }
    all_ids = [value for values in groups.values() for value in values]
    if len(all_ids) != len(set(all_ids)):
        raise TeacherCollectionError("fixture splits overlap")
    body = {
        "format_version": TEACHER_SPLITS_FORMAT_VERSION,
        "collection_config_digest": config.digest,
        "fixtures": groups,
    }
    _atomic_json(
        config.output_path / "teacher" / "fixture-splits.json",
        {"splits": body, "splits_digest": _json_digest(body)},
    )


def _percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def _aggregate_metrics(
    config: TeacherCollectionConfig,
    summaries: Sequence[TeacherShardSummary],
    wall_seconds: float,
    failed_decisions: int,
) -> TeacherCollectionMetrics:
    latencies: list[float] = []
    for summary in summaries:
        path = config.output_path / summary.relative_path
        metrics = _load_json(path.with_suffix(".metrics.json"))
        values = metrics.get("decision_latencies_seconds", [])
        if isinstance(values, list):
            latencies.extend(float(value) for value in values)
    examples = sum(item.examples for item in summaries)
    simulations = sum(item.simulations for item in summaries)
    search_seconds = sum(item.search_seconds for item in summaries)
    return TeacherCollectionMetrics(
        examples=examples,
        simulations=simulations,
        search_seconds=search_seconds,
        wall_seconds=wall_seconds,
        examples_per_hour=examples / wall_seconds * 3600 if wall_seconds else 0.0,
        simulations_per_second=simulations / search_seconds if search_seconds else 0.0,
        decision_latency_p50_seconds=statistics.median(latencies) if latencies else 0.0,
        decision_latency_p95_seconds=_percentile(latencies, 0.95),
        peak_rss_bytes=max((item.peak_rss_bytes for item in summaries), default=0),
        cache_hits=sum(item.cache_hits for item in summaries),
        failed_decisions=failed_decisions,
    )


def _write_state(config: TeacherCollectionConfig, phase: str, **extra: object) -> None:
    _atomic_json(
        config.output_path / "state.json",
        {
            "format_version": TEACHER_STATE_FORMAT_VERSION,
            "collection_config_digest": config.digest,
            "phase": phase,
            **extra,
        },
    )


def _prepare_run(config: TeacherCollectionConfig, resume: bool) -> None:
    output = config.output_path
    resolved_path = output / "resolved-config.json"
    resolved = _resolved_config_data(config)
    if resolved_path.exists():
        if _load_json(resolved_path) != resolved:
            raise TeacherCollectionError("resolved teacher configuration differs")
        if not resume:
            raise TeacherCollectionError("output already exists; use resume")
    else:
        if resume:
            raise TeacherCollectionError("cannot resume without a resolved configuration")
        _atomic_json(resolved_path, resolved)
    (output / "teacher" / _STOP_REQUEST_FILENAME).unlink(missing_ok=True)
    _write_fixture_splits(config)


def _collect_parallel_fixtures(
    fixtures: Sequence[TeacherFixture],
    config: TeacherCollectionConfig,
    summaries: list[TeacherShardSummary],
    total: int,
    progress: Callable[[TeacherShardSummary, int, int], None] | None,
) -> None:
    stop_path = config.output_path / "teacher" / _STOP_REQUEST_FILENAME
    executor = ProcessPoolExecutor(
        max_workers=config.workers,
        initializer=_configure_collection_worker,
    )
    futures = {
        executor.submit(_collect_fixture_worker, fixture, config): fixture
        for fixture in fixtures
    }
    try:
        for future in as_completed(futures):
            summary = future.result()
            summaries.append(summary)
            if progress is not None:
                progress(summary, len(summaries), total)
    except BaseException:
        _atomic_bytes(stop_path, b"stop\n")
        for future in futures:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)
        raise
    else:
        executor.shutdown(wait=True)
    finally:
        stop_path.unlink(missing_ok=True)


def collect_teacher_dataset(
    config: TeacherCollectionConfig,
    *,
    resume: bool = False,
    should_stop: Callable[[], bool] | None = None,
    progress: Callable[[TeacherShardSummary, int, int], None] | None = None,
) -> TeacherDatasetInspection:
    _prepare_run(config, resume)
    started = time.perf_counter()
    _write_state(config, "collecting")
    fixtures = tuple(
        fixture
        for fixture in fixture_schedule(config)
        if fixture.split is not FixtureSplit.ABSOLUTE
    )
    summaries: list[TeacherShardSummary] = []
    remaining: list[TeacherFixture] = []
    for fixture in fixtures:
        existing = _load_shard_summary(fixture, config)
        if existing is None:
            remaining.append(fixture)
        else:
            summaries.append(existing)
    try:
        if config.workers == 1 or should_stop is not None:
            _configure_collection_worker()
            for fixture in remaining:
                summary = _collect_fixture(fixture, config, should_stop)
                summaries.append(summary)
                if progress is not None:
                    progress(summary, len(summaries), len(fixtures))
        elif remaining:
            _collect_parallel_fixtures(
                remaining,
                config,
                summaries,
                len(fixtures),
                progress,
            )
    except (KeyboardInterrupt, TeacherCollectionInterrupted):
        metrics = _aggregate_metrics(
            config, summaries, time.perf_counter() - started, 0
        )
        _atomic_json(config.output_path / "teacher" / "metrics.json", asdict(metrics))
        _write_state(config, "interrupted")
        raise TeacherCollectionInterrupted("teacher collection interrupted")
    except Exception as error:
        metrics = _aggregate_metrics(
            config, summaries, time.perf_counter() - started, 1
        )
        _atomic_json(config.output_path / "teacher" / "metrics.json", asdict(metrics))
        _write_state(config, "failed", failed_decisions=1, error=type(error).__name__)
        raise

    summaries.sort(key=lambda item: (item.split.value, item.fixture_index))
    for split in (FixtureSplit.TRAINING, FixtureSplit.VALIDATION):
        document = _manifest_document(
            split,
            config,
            tuple(item for item in summaries if item.split is split),
        )
        _atomic_json(
            config.output_path / "teacher" / f"{split.value}-manifest.json",
            document,
        )
    metrics = _aggregate_metrics(config, summaries, time.perf_counter() - started, 0)
    _atomic_json(config.output_path / "teacher" / "metrics.json", asdict(metrics))
    manifests = tuple(
        load_teacher_manifest(config.output_path, split)
        for split in (FixtureSplit.TRAINING, FixtureSplit.VALIDATION)
    )
    dataset_digest = _json_digest(
        [document["manifest_digest"] for document in manifests]
    )
    _write_state(
        config,
        "complete",
        dataset_digest=dataset_digest,
        examples=metrics.examples,
        simulations=metrics.simulations,
    )
    return TeacherDatasetInspection(
        config_digest=config.digest,
        training_games=config.training_games,
        validation_games=config.validation_games,
        examples=metrics.examples,
        simulations=metrics.simulations,
        dataset_digest=dataset_digest,
        metrics=metrics,
    )


def load_teacher_config(output_directory: str | Path) -> TeacherCollectionConfig:
    value = _load_json(Path(output_directory).expanduser().resolve() / "resolved-config.json")
    if (
        not isinstance(value, dict)
        or value.get("format_version") != TEACHER_CONFIG_FORMAT_VERSION
    ):
        if (
            isinstance(value, dict)
            and value.get("format_version")
            == LEGACY_TEACHER_CONFIG_FORMAT_VERSION
        ):
            raise TeacherCollectionError(
                "legacy v1 teacher runs are inspect-only and cannot resume collection"
            )
        raise TeacherCollectionError("resolved teacher configuration is invalid")
    fields = set(TeacherCollectionConfig.__dataclass_fields__)
    try:
        config = TeacherCollectionConfig(**{key: value[key] for key in fields})
    except (KeyError, TypeError) as error:
        raise TeacherCollectionError("resolved teacher configuration is incomplete") from error
    if _resolved_config_data(config) != value:
        raise TeacherCollectionError("resolved teacher configuration has unknown fields")
    return config


def inspect_teacher_dataset(output_directory: str | Path) -> TeacherDatasetInspection:
    output = Path(output_directory).expanduser().resolve()
    resolved = _load_json(output / "resolved-config.json")
    if not isinstance(resolved, dict):
        raise TeacherCollectionError("resolved teacher configuration is invalid")
    resolved_format = resolved.get("format_version")
    if resolved_format == TEACHER_CONFIG_FORMAT_VERSION:
        config = load_teacher_config(output)
        config_digest = config.digest
    elif resolved_format == LEGACY_TEACHER_CONFIG_FORMAT_VERSION:
        if resolved.get("search_schema_version") != LEGACY_SEARCH_SCHEMA_VERSION:
            raise TeacherCollectionError(
                "legacy teacher search contract is incompatible"
            )
        if (
            resolved.get("teacher_dataset_schema_version")
            != LEGACY_TEACHER_DATASET_SCHEMA_VERSION
        ):
            raise TeacherCollectionError(
                "legacy teacher dataset contract is incompatible"
            )
        configured_output = resolved.get("output_directory")
        if (
            not isinstance(configured_output, str)
            or Path(configured_output).expanduser().resolve() != output
        ):
            raise TeacherCollectionError(
                "legacy teacher output directory differs"
            )
        config = None
        config_digest = ""
    else:
        raise TeacherCollectionError("resolved teacher configuration is invalid")
    documents = tuple(
        load_teacher_manifest(output, split)
        for split in (FixtureSplit.TRAINING, FixtureSplit.VALIDATION)
    )
    manifests = [document["manifest"] for document in documents]
    contract_keys = (
        "format_version",
        "dataset_schema_version",
        "teacher_profile",
        "teacher_profile_digest",
        "approval_identity",
        "validation_report_digest",
        "search_schema_version",
        "response_schema_version",
        "collection_config_digest",
        "search_config_digest",
        "response_config_digest",
    )
    bindings = tuple(
        tuple(manifest.get(key) for key in contract_keys)
        for manifest in manifests
    )
    if bindings[0] != bindings[1]:
        raise TeacherCollectionError(
            "training and validation manifests mix teacher contracts"
        )
    manifest_config_digest = manifests[0].get("collection_config_digest")
    _require_digest(manifest_config_digest, "collection config digest")
    if config is not None and manifest_config_digest != config_digest:
        raise TeacherCollectionError(
            "teacher manifests name another resolved configuration"
        )
    config_digest = str(manifest_config_digest)
    metrics_value = _load_json(output / "teacher" / "metrics.json")
    if not isinstance(metrics_value, dict):
        raise TeacherCollectionError("teacher metrics are invalid")
    try:
        metrics = TeacherCollectionMetrics(**metrics_value)
    except TypeError as error:
        raise TeacherCollectionError("teacher metrics are incompatible") from error
    return TeacherDatasetInspection(
        config_digest=config_digest,
        training_games=int(manifests[0]["fixture_count"]),
        validation_games=int(manifests[1]["fixture_count"]),
        examples=sum(int(value["example_count"]) for value in manifests),
        simulations=sum(int(value["simulation_count"]) for value in manifests),
        dataset_digest=_json_digest(
            [document["manifest_digest"] for document in documents]
        ),
        metrics=metrics,
    )


def _print_inspection(inspection: TeacherDatasetInspection) -> None:
    print(
        json.dumps(
            {
                **asdict(inspection),
                "metrics": asdict(inspection.metrics),
            },
            indent=2,
            sort_keys=True,
        )
    )


def _print_progress(
    summary: TeacherShardSummary,
    completed: int,
    total: int,
) -> None:
    print(
        "teacher_collection "
        f"completed={completed}/{total} "
        f"split={summary.split.value} "
        f"fixture={summary.fixture_index} "
        f"examples={summary.examples} "
        f"simulations={summary.simulations}",
        flush=True,
    )


def _collection_parser(subparsers, name: str, *, smoke: bool) -> None:
    parser = subparsers.add_parser(name)
    parser.add_argument("--output", required=True)
    parser.add_argument("--run-id", default="teacher-smoke" if smoke else "teacher-001")
    parser.add_argument(
        "--root-seed",
        default="dracula-teacher-smoke-v2" if smoke else None,
        required=not smoke,
    )
    parser.add_argument(
        "--simulation-budget",
        type=int,
        default=APPROVED_OUTER_SIMULATION_BUDGET,
        help="fixed at the approved value of 32",
    )
    parser.add_argument(
        "--exploration-constant",
        type=float,
        default=math.sqrt(2.0),
        help="fixed at sqrt(2) by the approved teacher profile",
    )
    parser.add_argument("--training-games", type=int, default=1 if smoke else 216)
    parser.add_argument("--validation-games", type=int, default=0 if smoke else 24)
    parser.add_argument("--absolute-fixtures", type=int, default=2 if smoke else 60)
    parser.add_argument(
        "--workers",
        type=int,
        default=1 if smoke else DEFAULT_COLLECTION_WORKERS,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="dracula-teacher")
    subparsers = parser.add_subparsers(dest="command", required=True)
    _collection_parser(subparsers, "smoke", smoke=True)
    _collection_parser(subparsers, "full", smoke=False)
    resume_parser = subparsers.add_parser("resume")
    resume_parser.add_argument("--output", required=True)
    inspect_parser = subparsers.add_parser("inspect")
    inspect_parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "resume":
            inspection = collect_teacher_dataset(
                load_teacher_config(args.output),
                resume=True,
                progress=_print_progress,
            )
        elif args.command == "inspect":
            inspection = inspect_teacher_dataset(args.output)
        else:
            config = TeacherCollectionConfig(
                run_id=args.run_id,
                root_seed=args.root_seed,
                output_directory=args.output,
                simulation_budget=args.simulation_budget,
                response_completions_per_action=(
                    APPROVED_RESPONSE_COMPLETIONS_PER_ACTION
                ),
                exploration_constant=args.exploration_constant,
                training_games=args.training_games,
                validation_games=args.validation_games,
                absolute_fixtures=args.absolute_fixtures,
                workers=args.workers,
            )
            inspection = collect_teacher_dataset(
                config, progress=_print_progress
            )
    except TeacherCollectionInterrupted:
        print("teacher_collection interrupted", file=sys.stderr)
        return 130
    _print_inspection(inspection)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = (
    "ACTION_SELECTION_NAMESPACE",
    "APPROVED_OUTER_SIMULATION_BUDGET",
    "APPROVED_RESPONSE_COMPLETIONS_PER_ACTION",
    "APPROVED_TEACHER_PROFILE",
    "DEFAULT_COLLECTION_WORKERS",
    "EXAMPLES_PER_GAME",
    "EXAMPLES_PER_PLAYER",
    "FIXTURE_ID_NAMESPACE",
    "FIXTURE_SEED_NAMESPACE",
    "FixtureSplit",
    "TeacherCollectionConfig",
    "TeacherCollectionError",
    "TeacherCollectionInterrupted",
    "TeacherCollectionMetrics",
    "TeacherDatasetInspection",
    "TeacherFixture",
    "TEACHER_APPROVAL_IDENTITY",
    "TEACHER_VALIDATION_REPORT_DIGEST",
    "collect_teacher_dataset",
    "derive_teacher_fixture",
    "fixture_schedule",
    "inspect_teacher_dataset",
    "load_teacher_config",
    "load_teacher_manifest",
    "load_teacher_shard",
    "main",
)

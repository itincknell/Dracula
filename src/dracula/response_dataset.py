"""Deterministic game-sharded response-ranking dataset collection."""

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
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import Tensor

from dracula.bridge import PolicyTurnKind, build_policy_turn_context
from dracula.engine import EngineStatus, advance_after_round, apply_move, create_game
from dracula.policy_value import ACTION_SCHEMA_VERSION, OBSERVATION_SCHEMA_VERSION
from dracula.randomness import Sha256CounterStream, derive_seed
from dracula.response_distillation import (
    RESPONSE_ACTION_GROUP_SCHEMA_VERSION,
    RESPONSE_EXAMPLE_SCHEMA_VERSION,
    RESPONSE_GROUP_LOGIT_PROFILE,
    RESPONSE_PAIR_WEIGHT_PROFILE,
    RESPONSE_RANKING_SCHEMA_VERSION,
    ResponseCacheIdentity,
    ResponseDistillationExample,
)
from dracula.search import (
    DESTINATION_SYMMETRY_SCHEMA_VERSION,
    GREEDY_RESPONSE_SCHEMA_VERSION,
    STRATEGIC_SEARCH_SCHEMA_VERSION,
    StrategicInformationSetSearch,
    StrategicSearchConfig,
    derive_strategic_search_request_seed,
    information_state_from_engine,
)
from dracula.teacher import (
    ACTION_SELECTION_NAMESPACE,
    APPROVED_OUTER_SIMULATION_BUDGET,
    APPROVED_RESPONSE_COMPLETIONS_PER_ACTION,
    APPROVED_TEACHER_PROFILE,
    DEFAULT_COLLECTION_WORKERS,
    FixtureSplit,
    TEACHER_APPROVAL_IDENTITY,
    TEACHER_VALIDATION_REPORT_DIGEST,
    TeacherFixture,
    derive_teacher_fixture,
)

RESPONSE_DATASET_SCHEMA_VERSION = "dracula-response-ranking-dataset-v1"
RESPONSE_SHARD_FORMAT_VERSION = "dracula-response-ranking-shard-v1"
RESPONSE_MANIFEST_FORMAT_VERSION = "dracula-response-ranking-manifest-v1"
RESPONSE_CONFIG_FORMAT_VERSION = "dracula-response-ranking-config-v1"
RESPONSE_STATE_FORMAT_VERSION = "dracula-response-ranking-state-v1"
RESPONSE_SPLITS_FORMAT_VERSION = "dracula-response-ranking-splits-v1"
RESPONSE_COLLECTION_PROFILE = "teacher-v2-response-ranking-32x4-v1"

_STOP_REQUEST_FILENAME = ".stop-requested"
_DIGEST_LENGTH = hashlib.sha256().digest_size * 2
_FORBIDDEN_KEYS = frozenset(
    (
        "authoritative_state",
        "determinization",
        "determinizations",
        "engine_seed",
        "game_seed",
        "hands",
        "model",
        "model_data",
        "opponent_hand",
        "outer_sampled_state",
        "policy_hidden_state",
        "search_tree",
        "stock",
        "stock_order",
        "tree",
    )
)


class ResponseDatasetError(ValueError):
    """Response collection configuration or artifacts violate the contract."""


class ResponseDatasetInterrupted(RuntimeError):
    """Collection stopped without committing a partial game shard."""


@dataclass(frozen=True, slots=True)
class ResponseDatasetConfig:
    run_id: str
    root_seed: str
    output_directory: str
    training_games: int = 4
    validation_games: int = 2
    workers: int = DEFAULT_COLLECTION_WORKERS
    outer_simulation_budget: int = APPROVED_OUTER_SIMULATION_BUDGET
    response_completions_per_action: int = (
        APPROVED_RESPONSE_COMPLETIONS_PER_ACTION
    )
    outer_exploration_constant: float = math.sqrt(2.0)
    early_placement_count: int = 4

    def __post_init__(self) -> None:
        if not isinstance(self.run_id, str) or not self.run_id:
            raise ResponseDatasetError("run ID must be nonempty")
        if not isinstance(self.root_seed, str) or not self.root_seed:
            raise ResponseDatasetError("root seed must be nonempty")
        if not isinstance(self.output_directory, str) or not self.output_directory:
            raise ResponseDatasetError("output directory must be nonempty")
        if (
            self.outer_simulation_budget != APPROVED_OUTER_SIMULATION_BUDGET
            or self.response_completions_per_action
            != APPROVED_RESPONSE_COMPLETIONS_PER_ACTION
            or self.outer_exploration_constant != math.sqrt(2.0)
        ):
            raise ResponseDatasetError(
                "response collection is locked to symmetry-aware Teacher v2 32x4"
            )
        if (
            type(self.training_games) is not int
            or self.training_games < 1
            or type(self.validation_games) is not int
            or self.validation_games < 1
        ):
            raise ResponseDatasetError(
                "response collection requires positive training and validation games"
            )
        if type(self.workers) is not int or not 1 <= self.workers <= 6:
            raise ResponseDatasetError("workers must be between one and six")
        if (
            type(self.early_placement_count) is not int
            or not 0 <= self.early_placement_count <= 7
        ):
            raise ResponseDatasetError(
                "early placement count must be between zero and seven"
            )
        _ = self.search_config

    @property
    def output_path(self) -> Path:
        return Path(self.output_directory).expanduser().resolve()

    @property
    def search_config(self) -> StrategicSearchConfig:
        return StrategicSearchConfig(
            outer_simulation_budget=self.outer_simulation_budget,
            response_completions_per_action=(
                self.response_completions_per_action
            ),
            outer_exploration_constant=self.outer_exploration_constant,
            destination_symmetry_enabled=True,
        )

    @property
    def digest(self) -> str:
        values = _resolved_config(self)
        del values["output_directory"]
        return _json_digest(values)


@dataclass(frozen=True, slots=True)
class ResponseShardSummary:
    split: FixtureSplit
    fixture_index: int
    fixture_id: str
    relative_path: str
    content_digest: str
    examples: int
    response_requests: int
    observer_emissions: int
    duplicates_removed: int
    search_seconds: float
    peak_rss_bytes: int


@dataclass(frozen=True, slots=True)
class ResponseDatasetInspection:
    config_digest: str
    training_games: int
    validation_games: int
    examples: int
    response_requests: int
    observer_emissions: int
    duplicates_removed: int
    dataset_digest: str
    wall_seconds: float
    search_seconds: float
    peak_rss_bytes: int
    disk_bytes: int


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
        raise ResponseDatasetError("metadata must be canonical JSON") from error


def _json_digest(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _require_digest(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != _DIGEST_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ResponseDatasetError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _contract_bindings(config: ResponseDatasetConfig) -> dict[str, object]:
    return {
        "dataset_schema_version": RESPONSE_DATASET_SCHEMA_VERSION,
        "collection_profile": RESPONSE_COLLECTION_PROFILE,
        "teacher_profile": APPROVED_TEACHER_PROFILE,
        "approval_identity": TEACHER_APPROVAL_IDENTITY,
        "validation_report_digest": TEACHER_VALIDATION_REPORT_DIGEST,
        "search_schema_version": STRATEGIC_SEARCH_SCHEMA_VERSION,
        "response_schema_version": GREEDY_RESPONSE_SCHEMA_VERSION,
        "symmetry_schema_version": DESTINATION_SYMMETRY_SCHEMA_VERSION,
        "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
        "action_schema_version": ACTION_SCHEMA_VERSION,
        "response_example_schema_version": RESPONSE_EXAMPLE_SCHEMA_VERSION,
        "action_group_schema_version": RESPONSE_ACTION_GROUP_SCHEMA_VERSION,
        "ranking_schema_version": RESPONSE_RANKING_SCHEMA_VERSION,
        "group_logit_profile": RESPONSE_GROUP_LOGIT_PROFILE,
        "pair_weight_profile": RESPONSE_PAIR_WEIGHT_PROFILE,
        "search_config_digest": config.search_config.digest,
        "response_config_digest": config.search_config.response_config.digest,
    }


def _resolved_config(config: ResponseDatasetConfig) -> dict[str, object]:
    values = asdict(config)
    values["output_directory"] = str(config.output_path)
    values["format_version"] = RESPONSE_CONFIG_FORMAT_VERSION
    values.update(_contract_bindings(config))
    return values


def fixture_schedule(
    config: ResponseDatasetConfig,
) -> tuple[TeacherFixture, ...]:
    return tuple(
        derive_teacher_fixture(config.root_seed, split, index)
        for split, count in (
            (FixtureSplit.TRAINING, config.training_games),
            (FixtureSplit.VALIDATION, config.validation_games),
        )
        for index in range(count)
    )


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
        raise ResponseDatasetError(f"could not read JSON artifact: {path}") from error


def _peak_rss_bytes() -> int:
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(raw if sys.platform == "darwin" else raw * 1024)


def _configure_worker() -> None:
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
            raise ResponseDatasetError(
                "worker inter-op threads were initialized before collection"
            ) from error


def _actual_action(
    visits: tuple[int, ...],
    selected_action_index: int,
    *,
    fixture: TeacherFixture,
    config: ResponseDatasetConfig,
    round_number: int,
    placement_number: int,
) -> int:
    if placement_number > config.early_placement_count:
        return selected_action_index
    if sum(visits) != config.outer_simulation_budget:
        raise ResponseDatasetError("root visits do not match the outer budget")
    seed = derive_seed(
        ACTION_SELECTION_NAMESPACE,
        config.root_seed,
        "teacher",
        fixture.fixture_id,
        str(round_number),
        str(placement_number),
    )
    threshold = Sha256CounterStream(seed).randbelow(sum(visits))
    cumulative = 0
    for action_index, count in enumerate(visits):
        cumulative += count
        if threshold < cumulative:
            return action_index
    raise ResponseDatasetError("root visits could not select an action")


def _empty_columns() -> dict[str, list[object]]:
    return {
        "observations": [],
        "legal_masks": [],
        "group_offsets": [0],
        "group_hand_slots": [],
        "group_representative_actions": [],
        "group_mean_values": [],
        "member_offsets": [0],
        "group_member_actions": [],
        "selected_representative_actions": [],
        "selected_concrete_actions": [],
        "placement_numbers": [],
        "actor_roles": [],
        "actor_is_dealer": [],
        "information_state_digests": [],
        "cache_identity_digests": [],
    }


def _append_example(
    columns: dict[str, list[object]],
    example: ResponseDistillationExample,
) -> None:
    columns["observations"].append(example.observation)
    columns["legal_masks"].append(example.legal_mask)
    for group in example.groups:
        columns["group_hand_slots"].append(group.hand_slot)
        columns["group_representative_actions"].append(
            group.representative_action_index
        )
        columns["group_mean_values"].append(group.mean_terminal_differential)
        columns["group_member_actions"].extend(group.member_action_indices)
        columns["member_offsets"].append(
            len(columns["group_member_actions"])
        )
    columns["group_offsets"].append(len(columns["group_hand_slots"]))
    columns["selected_representative_actions"].append(
        example.selected_representative_action_index
    )
    columns["selected_concrete_actions"].append(
        example.selected_concrete_action_index
    )
    columns["placement_numbers"].append(example.placement_number)
    columns["actor_roles"].append(example.actor_role)
    columns["actor_is_dealer"].append(example.actor_is_dealer)
    columns["information_state_digests"].append(
        example.cache_identity.information_state_fingerprint
    )
    columns["cache_identity_digests"].append(example.cache_identity.digest)


def _tensor_columns(columns: dict[str, list[object]]) -> dict[str, object]:
    example_count = len(columns["actor_roles"])
    observations = torch.tensor(columns["observations"], dtype=torch.bool)
    legal_masks = torch.tensor(columns["legal_masks"], dtype=torch.bool)
    if example_count == 0:
        observations = torch.empty((0, 875), dtype=torch.bool)
        legal_masks = torch.empty((0, 4, 8), dtype=torch.bool)
    return {
        "observations": observations,
        "legal_masks": legal_masks,
        "group_offsets": torch.tensor(columns["group_offsets"], dtype=torch.int64),
        "group_hand_slots": torch.tensor(
            columns["group_hand_slots"], dtype=torch.int8
        ),
        "group_representative_actions": torch.tensor(
            columns["group_representative_actions"], dtype=torch.int8
        ),
        "group_mean_values": torch.tensor(
            columns["group_mean_values"], dtype=torch.float64
        ),
        "member_offsets": torch.tensor(
            columns["member_offsets"], dtype=torch.int64
        ),
        "group_member_actions": torch.tensor(
            columns["group_member_actions"], dtype=torch.int8
        ),
        "selected_representative_actions": torch.tensor(
            columns["selected_representative_actions"], dtype=torch.int8
        ),
        "selected_concrete_actions": torch.tensor(
            columns["selected_concrete_actions"], dtype=torch.int8
        ),
        "placement_numbers": torch.tensor(
            columns["placement_numbers"], dtype=torch.int8
        ),
        "actor_roles": tuple(columns["actor_roles"]),
        "actor_is_dealer": torch.tensor(
            columns["actor_is_dealer"], dtype=torch.bool
        ),
        "information_state_digests": tuple(
            columns["information_state_digests"]
        ),
        "cache_identity_digests": tuple(columns["cache_identity_digests"]),
    }


def _update_tensor_digest(digest, name: str, tensor: Tensor) -> None:
    value = tensor.detach().cpu().contiguous()
    digest.update(name.encode("utf-8"))
    digest.update(b"\0")
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(",".join(map(str, value.shape)).encode("ascii"))
    digest.update(b"\0")
    digest.update(value.numpy().tobytes(order="C"))


def _content_digest(
    metadata: Mapping[str, object],
    columns: Mapping[str, object],
) -> str:
    digest = hashlib.sha256(_canonical_json(metadata))
    for name in sorted(columns):
        value = columns[name]
        if isinstance(value, Tensor):
            _update_tensor_digest(digest, name, value)
        else:
            digest.update(name.encode("utf-8"))
            digest.update(b"\0")
            digest.update(_canonical_json(value))
    return digest.hexdigest()


def _shard_path(
    output: Path, fixture: TeacherFixture
) -> Path:
    return (
        output
        / "response-distillation"
        / "games"
        / fixture.split.value
        / f"{fixture.fixture_id}.pt"
    )


def _metadata(
    fixture: TeacherFixture,
    config: ResponseDatasetConfig,
) -> dict[str, object]:
    return {
        **_contract_bindings(config),
        "collection_config_digest": config.digest,
        "fixture_id": fixture.fixture_id,
        "fixture_index": fixture.index,
        "split": fixture.split.value,
    }


def _validate_columns(columns: object) -> dict[str, object]:
    expected = set(_empty_columns())
    if not isinstance(columns, dict) or set(columns) != expected:
        raise ResponseDatasetError("response shard columns are invalid")
    tensors = (
        "observations",
        "legal_masks",
        "group_offsets",
        "group_hand_slots",
        "group_representative_actions",
        "group_mean_values",
        "member_offsets",
        "group_member_actions",
        "selected_representative_actions",
        "selected_concrete_actions",
        "placement_numbers",
        "actor_is_dealer",
    )
    if any(
        not isinstance(columns[name], Tensor)
        or columns[name].device.type != "cpu"
        for name in tensors
    ):
        raise ResponseDatasetError("response shard tensors must be on CPU")
    observations = columns["observations"]
    masks = columns["legal_masks"]
    count = observations.shape[0]
    if observations.dtype is not torch.bool or observations.shape != (count, 875):
        raise ResponseDatasetError("response observations must be bool[N,875]")
    if masks.dtype is not torch.bool or masks.shape != (count, 4, 8):
        raise ResponseDatasetError("response masks must be bool[N,4,8]")
    per_example_names = (
        "selected_representative_actions",
        "selected_concrete_actions",
        "placement_numbers",
        "actor_is_dealer",
    )
    if any(columns[name].shape != (count,) for name in per_example_names):
        raise ResponseDatasetError("response per-example columns differ")
    roles = columns["actor_roles"]
    information_digests = columns["information_state_digests"]
    cache_digests = columns["cache_identity_digests"]
    if (
        not isinstance(roles, tuple)
        or len(roles) != count
        or any(role not in {"queen", "king"} for role in roles)
        or not isinstance(information_digests, tuple)
        or len(information_digests) != count
        or not isinstance(cache_digests, tuple)
        or len(cache_digests) != count
    ):
        raise ResponseDatasetError("response identifier columns differ")
    if len(set(cache_digests)) != count:
        raise ResponseDatasetError("response shard contains duplicate cache rows")
    for value in (*information_digests, *cache_digests):
        _require_digest(value, "response identity")
    if count and (
        torch.any(columns["placement_numbers"] < 1)
        or torch.any(columns["placement_numbers"] > 7)
    ):
        raise ResponseDatasetError("forced or invalid placements entered the shard")
    group_offsets = columns["group_offsets"]
    member_offsets = columns["member_offsets"]
    group_count = columns["group_hand_slots"].numel()
    member_count = columns["group_member_actions"].numel()
    if (
        group_offsets.dtype is not torch.int64
        or group_offsets.shape != (count + 1,)
        or group_offsets[0].item() != 0
        or group_offsets[-1].item() != group_count
        or torch.any(group_offsets[1:] < group_offsets[:-1])
    ):
        raise ResponseDatasetError("response group offsets are invalid")
    if (
        member_offsets.dtype is not torch.int64
        or member_offsets.shape != (group_count + 1,)
        or member_offsets[0].item() != 0
        or member_offsets[-1].item() != member_count
        or torch.any(member_offsets[1:] < member_offsets[:-1])
    ):
        raise ResponseDatasetError("response member offsets are invalid")
    if (
        columns["group_mean_values"].dtype is not torch.float64
        or columns["group_mean_values"].shape != (group_count,)
        or not torch.isfinite(columns["group_mean_values"]).all()
        or torch.any(columns["group_mean_values"] < -1)
        or torch.any(columns["group_mean_values"] > 1)
    ):
        raise ResponseDatasetError("response group values are invalid")
    if (
        columns["group_hand_slots"].dtype is not torch.int8
        or columns["group_representative_actions"].dtype is not torch.int8
        or columns["group_member_actions"].dtype is not torch.int8
        or columns["selected_representative_actions"].dtype is not torch.int8
        or columns["selected_concrete_actions"].dtype is not torch.int8
        or columns["placement_numbers"].dtype is not torch.int8
        or columns["actor_is_dealer"].dtype is not torch.bool
    ):
        raise ResponseDatasetError("response compact-column dtypes are invalid")
    for row in range(count):
        group_start = int(group_offsets[row].item())
        group_end = int(group_offsets[row + 1].item())
        representatives = columns["group_representative_actions"][
            group_start:group_end
        ].to(torch.int64)
        values = columns["group_mean_values"][group_start:group_end]
        if group_start == group_end:
            raise ResponseDatasetError("response row has no strategic groups")
        expected_selected = min(
            range(group_end - group_start),
            key=lambda offset: (
                -float(values[offset].item()),
                int(representatives[offset].item()),
            ),
        )
        selected_rep = int(
            columns["selected_representative_actions"][row].item()
        )
        if selected_rep != int(representatives[expected_selected].item()):
            raise ResponseDatasetError(
                "response selected group is not the canonical maximum"
            )
        grouped_members: set[int] = set()
        selected_members: set[int] | None = None
        for group_index in range(group_start, group_end):
            member_start = int(member_offsets[group_index].item())
            member_end = int(member_offsets[group_index + 1].item())
            members = {
                int(value)
                for value in columns["group_member_actions"][
                    member_start:member_end
                ].tolist()
            }
            if not members or grouped_members.intersection(members):
                raise ResponseDatasetError(
                    "response groups do not partition concrete actions"
                )
            grouped_members.update(members)
            hand_slot = int(columns["group_hand_slots"][group_index].item())
            representative = int(
                columns["group_representative_actions"][group_index].item()
            )
            if (
                not 0 <= hand_slot < 4
                or representative not in members
                or representative // 8 != hand_slot
                or any(member // 8 != hand_slot for member in members)
            ):
                raise ResponseDatasetError(
                    "response action group identity is invalid"
                )
            if int(
                columns["group_representative_actions"][group_index].item()
            ) == selected_rep:
                selected_members = members
        legal = {
            index
            for index, allowed in enumerate(masks[row].flatten().tolist())
            if allowed
        }
        if grouped_members != legal:
            raise ResponseDatasetError(
                "response groups do not partition the legal mask"
            )
        if (
            selected_members is None
            or int(columns["selected_concrete_actions"][row].item())
            not in selected_members
        ):
            raise ResponseDatasetError(
                "response concrete action is outside its selected group"
            )
    return columns


def _validate_payload(payload: object) -> dict[str, object]:
    if (
        not isinstance(payload, dict)
        or set(payload)
        != {"format_version", "metadata", "columns", "content_digest"}
        or payload["format_version"] != RESPONSE_SHARD_FORMAT_VERSION
        or not isinstance(payload["metadata"], dict)
    ):
        raise ResponseDatasetError("response shard payload is incompatible")
    metadata = payload["metadata"]
    if set(metadata).intersection(_FORBIDDEN_KEYS):
        raise ResponseDatasetError("response shard metadata contains a private field")
    expected_metadata = {
        *set(_contract_bindings(
            ResponseDatasetConfig(
                "_shape",
                "_shape",
                ".",
                1,
                1,
                1,
            )
        )),
        "collection_config_digest",
        "fixture_id",
        "fixture_index",
        "split",
    }
    if set(metadata) != expected_metadata:
        raise ResponseDatasetError("response shard metadata shape is invalid")
    for key in (
        "validation_report_digest",
        "search_config_digest",
        "response_config_digest",
        "collection_config_digest",
        "fixture_id",
    ):
        _require_digest(metadata[key], key.replace("_", " "))
    if metadata["split"] not in {
        FixtureSplit.TRAINING.value,
        FixtureSplit.VALIDATION.value,
    }:
        raise ResponseDatasetError("response shard split is invalid")
    columns = _validate_columns(payload["columns"])
    for information_digest, cache_digest in zip(
        columns["information_state_digests"],
        columns["cache_identity_digests"],
        strict=True,
    ):
        if (
            ResponseCacheIdentity(
                information_digest,
                str(metadata["response_config_digest"]),
            ).digest
            != cache_digest
        ):
            raise ResponseDatasetError("response cache identity digest differs")
    expected_digest = _content_digest(metadata, columns)
    if payload["content_digest"] != expected_digest:
        raise ResponseDatasetError("response shard content digest differs")
    return payload


def load_response_shard(path: str | Path) -> dict[str, object]:
    try:
        payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    except (
        OSError,
        RuntimeError,
        ValueError,
        EOFError,
        pickle.UnpicklingError,
    ) as error:
        raise ResponseDatasetError("response shard could not be loaded") from error
    return _validate_payload(payload)


def _atomic_shard(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        torch.save(payload, temporary)
        _validate_payload(
            torch.load(temporary, map_location="cpu", weights_only=True)
        )
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _collect_fixture(
    fixture: TeacherFixture,
    config: ResponseDatasetConfig,
    should_stop: Callable[[], bool] | None = None,
) -> ResponseShardSummary:
    state = create_game(fixture._game_seed)
    examples: dict[str, ResponseDistillationExample] = {}
    observer_emissions = 0
    response_requests = 0
    search_seconds = 0.0

    def observe(example: ResponseDistillationExample) -> None:
        nonlocal observer_emissions
        observer_emissions += 1
        previous = examples.get(example.cache_identity.digest)
        if previous is not None and previous != example:
            raise ResponseDatasetError(
                "one response cache identity produced different evidence"
            )
        examples.setdefault(example.cache_identity.digest, example)

    while state.status is not EngineStatus.GAME_COMPLETE:
        while state.status is EngineStatus.PLAYING:
            if should_stop is not None and should_stop():
                raise ResponseDatasetInterrupted(
                    "collection stopped before game seal"
                )
            actor = state.active_player
            context = build_policy_turn_context(state, actor)
            if context.kind is PolicyTurnKind.FORCED_RECURRENT_TRANSITION:
                move = context.forced_move
                if move is None:
                    raise ResponseDatasetError("forced placement has no move")
            else:
                information = information_state_from_engine(state)
                request_seed = derive_strategic_search_request_seed(
                    fixture.fixture_id,
                    information,
                    config.search_config.digest,
                )
                started = time.perf_counter()
                result = StrategicInformationSetSearch(
                    config.search_config,
                    response_observer=observe,
                ).search(information, request_seed, should_stop)
                search_seconds += time.perf_counter() - started
                response_requests += result.response_request_count
                action_index = _actual_action(
                    result.action_visits,
                    result.selected_action_index,
                    fixture=fixture,
                    config=config,
                    round_number=state.round_number,
                    placement_number=len(state.current_round_moves) + 1,
                )
                move = context.action_table[action_index]
                if move is None:
                    raise ResponseDatasetError(
                        "Teacher v2 selected a masked root action"
                    )
            state = apply_move(state, move).state
        state = advance_after_round(state)

    columns = _empty_columns()
    for example in examples.values():
        _append_example(columns, example)
    sealed_columns = _tensor_columns(columns)
    metadata = _metadata(fixture, config)
    content_digest = _content_digest(metadata, sealed_columns)
    payload = {
        "format_version": RESPONSE_SHARD_FORMAT_VERSION,
        "metadata": metadata,
        "columns": sealed_columns,
        "content_digest": content_digest,
    }
    destination = _shard_path(config.output_path, fixture)
    summary = ResponseShardSummary(
        split=fixture.split,
        fixture_index=fixture.index,
        fixture_id=fixture.fixture_id,
        relative_path=str(destination.relative_to(config.output_path)),
        content_digest=content_digest,
        examples=len(examples),
        response_requests=response_requests,
        observer_emissions=observer_emissions,
        duplicates_removed=response_requests - len(examples),
        search_seconds=search_seconds,
        peak_rss_bytes=_peak_rss_bytes(),
    )
    _atomic_json(destination.with_suffix(".metrics.json"), asdict(summary))
    _atomic_shard(destination, payload)
    return summary


def _collect_worker(
    fixture: TeacherFixture,
    config: ResponseDatasetConfig,
) -> ResponseShardSummary:
    stop_path = (
        config.output_path
        / "response-distillation"
        / _STOP_REQUEST_FILENAME
    )
    return _collect_fixture(fixture, config, stop_path.exists)


def _summary_from_shard(
    fixture: TeacherFixture,
    config: ResponseDatasetConfig,
) -> ResponseShardSummary | None:
    path = _shard_path(config.output_path, fixture)
    if not path.exists():
        return None
    shard = load_response_shard(path)
    metadata = shard["metadata"]
    if (
        metadata["fixture_id"] != fixture.fixture_id
        or metadata["fixture_index"] != fixture.index
        or metadata["split"] != fixture.split.value
        or metadata["collection_config_digest"] != config.digest
    ):
        raise ResponseDatasetError(
            "existing response shard belongs to another fixture"
        )
    metrics = _load_json(path.with_suffix(".metrics.json"))
    if not isinstance(metrics, dict):
        raise ResponseDatasetError("response shard metrics are invalid")
    try:
        return ResponseShardSummary(
            split=FixtureSplit(metrics["split"]),
            fixture_index=int(metrics["fixture_index"]),
            fixture_id=str(metrics["fixture_id"]),
            relative_path=str(metrics["relative_path"]),
            content_digest=str(metrics["content_digest"]),
            examples=int(metrics["examples"]),
            response_requests=int(metrics["response_requests"]),
            observer_emissions=int(metrics["observer_emissions"]),
            duplicates_removed=int(metrics["duplicates_removed"]),
            search_seconds=float(metrics["search_seconds"]),
            peak_rss_bytes=int(metrics["peak_rss_bytes"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ResponseDatasetError(
            "response shard metrics are incompatible"
        ) from error


def _manifest(
    split: FixtureSplit,
    config: ResponseDatasetConfig,
    summaries: Sequence[ResponseShardSummary],
) -> dict[str, object]:
    shards = [
        {
            "fixture_index": summary.fixture_index,
            "fixture_id": summary.fixture_id,
            "relative_path": summary.relative_path,
            "content_digest": summary.content_digest,
            "examples": summary.examples,
        }
        for summary in sorted(summaries, key=lambda item: item.fixture_index)
    ]
    body = {
        "format_version": RESPONSE_MANIFEST_FORMAT_VERSION,
        **_contract_bindings(config),
        "collection_config_digest": config.digest,
        "split": split.value,
        "fixture_count": len(shards),
        "example_count": sum(item["examples"] for item in shards),
        "shards": shards,
    }
    return {"manifest": body, "manifest_digest": _json_digest(body)}


def load_response_manifest(
    output_directory: str | Path,
    split: FixtureSplit,
) -> dict[str, object]:
    output = Path(output_directory).expanduser().resolve()
    path = (
        output
        / "response-distillation"
        / f"{FixtureSplit(split).value}-manifest.json"
    )
    document = _load_json(path)
    if (
        not isinstance(document, dict)
        or set(document) != {"manifest", "manifest_digest"}
        or not isinstance(document["manifest"], dict)
        or document["manifest_digest"]
        != _json_digest(document["manifest"])
    ):
        raise ResponseDatasetError("response manifest document is invalid")
    manifest = document["manifest"]
    if (
        manifest.get("format_version") != RESPONSE_MANIFEST_FORMAT_VERSION
        or manifest.get("dataset_schema_version")
        != RESPONSE_DATASET_SCHEMA_VERSION
        or manifest.get("split") != FixtureSplit(split).value
        or not isinstance(manifest.get("shards"), list)
        or manifest.get("fixture_count") != len(manifest["shards"])
    ):
        raise ResponseDatasetError("response manifest is incompatible")
    seen_fixtures: set[str] = set()
    seen_indexes: set[int] = set()
    seen_cache_identities: set[str] = set()
    examples = 0
    for entry in manifest["shards"]:
        if not isinstance(entry, dict) or set(entry) != {
            "fixture_index",
            "fixture_id",
            "relative_path",
            "content_digest",
            "examples",
        }:
            raise ResponseDatasetError("response manifest shard entry is invalid")
        relative = Path(entry["relative_path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ResponseDatasetError("response manifest path escapes the run")
        shard = load_response_shard(output / relative)
        metadata = shard["metadata"]
        if any(
            metadata.get(key) != manifest.get(key)
            for key in (
                *_contract_bindings(
                    ResponseDatasetConfig("_shape", "_shape", ".", 1, 1, 1)
                ),
                "collection_config_digest",
                "split",
            )
        ):
            raise ResponseDatasetError(
                "response manifest and shard contract bindings differ"
            )
        if (
            metadata["fixture_id"] != entry["fixture_id"]
            or metadata["fixture_index"] != entry["fixture_index"]
            or shard["content_digest"] != entry["content_digest"]
        ):
            raise ResponseDatasetError(
                "response manifest and shard identities differ"
            )
        cache_identities = shard["columns"]["cache_identity_digests"]
        overlap = seen_cache_identities.intersection(cache_identities)
        if overlap:
            raise ResponseDatasetError(
                "response dataset repeats a cache identity across shards"
            )
        seen_cache_identities.update(cache_identities)
        fixture_id = _require_digest(entry["fixture_id"], "fixture ID")
        fixture_index = entry["fixture_index"]
        if fixture_id in seen_fixtures or fixture_index in seen_indexes:
            raise ResponseDatasetError("response manifest repeats a fixture")
        seen_fixtures.add(fixture_id)
        seen_indexes.add(fixture_index)
        examples += len(cache_identities)
        if entry["examples"] != len(cache_identities):
            raise ResponseDatasetError("response manifest example count differs")
    if seen_indexes != set(range(len(manifest["shards"]))):
        raise ResponseDatasetError("response fixture indexes are not contiguous")
    if examples != manifest.get("example_count"):
        raise ResponseDatasetError("response manifest total differs")
    return document


def _write_splits(config: ResponseDatasetConfig) -> None:
    fixtures = fixture_schedule(config)
    groups = {
        split.value: [
            fixture.fixture_id
            for fixture in fixtures
            if fixture.split is split
        ]
        for split in (FixtureSplit.TRAINING, FixtureSplit.VALIDATION)
    }
    all_ids = [item for values in groups.values() for item in values]
    if len(all_ids) != len(set(all_ids)):
        raise ResponseDatasetError("training and validation fixtures overlap")
    body = {
        "format_version": RESPONSE_SPLITS_FORMAT_VERSION,
        "collection_config_digest": config.digest,
        "fixtures": groups,
    }
    _atomic_json(
        config.output_path
        / "response-distillation"
        / "fixture-splits.json",
        {"splits": body, "splits_digest": _json_digest(body)},
    )


def _prepare(config: ResponseDatasetConfig, resume: bool) -> None:
    resolved_path = config.output_path / "resolved-config.json"
    resolved = _resolved_config(config)
    if resolved_path.exists():
        if _load_json(resolved_path) != resolved:
            raise ResponseDatasetError(
                "resolved response configuration differs"
            )
        if not resume:
            raise ResponseDatasetError("output exists; use resume")
    else:
        if resume:
            raise ResponseDatasetError(
                "cannot resume without a resolved configuration"
            )
        _atomic_json(resolved_path, resolved)
    (
        config.output_path
        / "response-distillation"
        / _STOP_REQUEST_FILENAME
    ).unlink(missing_ok=True)
    _write_splits(config)


def _write_state(
    config: ResponseDatasetConfig,
    phase: str,
    **extra: object,
) -> None:
    _atomic_json(
        config.output_path / "state.json",
        {
            "format_version": RESPONSE_STATE_FORMAT_VERSION,
            "collection_config_digest": config.digest,
            "phase": phase,
            **extra,
        },
    )


def _directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _inspection(
    config: ResponseDatasetConfig,
    summaries: Sequence[ResponseShardSummary],
    wall_seconds: float,
) -> ResponseDatasetInspection:
    documents = tuple(
        load_response_manifest(config.output_path, split)
        for split in (FixtureSplit.TRAINING, FixtureSplit.VALIDATION)
    )
    training = documents[0]["manifest"]
    validation = documents[1]["manifest"]
    train_ids = {entry["fixture_id"] for entry in training["shards"]}
    validation_ids = {entry["fixture_id"] for entry in validation["shards"]}
    if not train_ids.isdisjoint(validation_ids):
        raise ResponseDatasetError("training and validation fixtures overlap")
    cache_identities: set[str] = set()
    for document in documents:
        for entry in document["manifest"]["shards"]:
            shard = load_response_shard(
                config.output_path / entry["relative_path"]
            )
            values = set(shard["columns"]["cache_identity_digests"])
            if cache_identities.intersection(values):
                raise ResponseDatasetError(
                    "response cache identity crosses dataset shards"
                )
            cache_identities.update(values)
    return ResponseDatasetInspection(
        config_digest=config.digest,
        training_games=int(training["fixture_count"]),
        validation_games=int(validation["fixture_count"]),
        examples=sum(int(doc["manifest"]["example_count"]) for doc in documents),
        response_requests=sum(item.response_requests for item in summaries),
        observer_emissions=sum(item.observer_emissions for item in summaries),
        duplicates_removed=sum(item.duplicates_removed for item in summaries),
        dataset_digest=_json_digest(
            [document["manifest_digest"] for document in documents]
        ),
        wall_seconds=wall_seconds,
        search_seconds=sum(item.search_seconds for item in summaries),
        peak_rss_bytes=max(
            (item.peak_rss_bytes for item in summaries), default=0
        ),
        disk_bytes=_directory_size(
            config.output_path / "response-distillation"
        ),
    )


def collect_response_dataset(
    config: ResponseDatasetConfig,
    *,
    resume: bool = False,
    should_stop: Callable[[], bool] | None = None,
    progress: Callable[[ResponseShardSummary, int, int], None] | None = None,
) -> ResponseDatasetInspection:
    _prepare(config, resume)
    _write_state(config, "collecting")
    started = time.perf_counter()
    fixtures = fixture_schedule(config)
    summaries: list[ResponseShardSummary] = []
    remaining: list[TeacherFixture] = []
    for fixture in fixtures:
        summary = _summary_from_shard(fixture, config)
        if summary is None:
            remaining.append(fixture)
        else:
            summaries.append(summary)
    try:
        if config.workers == 1 or should_stop is not None:
            _configure_worker()
            for fixture in remaining:
                summary = _collect_fixture(fixture, config, should_stop)
                summaries.append(summary)
                if progress is not None:
                    progress(summary, len(summaries), len(fixtures))
        elif remaining:
            stop_path = (
                config.output_path
                / "response-distillation"
                / _STOP_REQUEST_FILENAME
            )
            executor = ProcessPoolExecutor(
                max_workers=config.workers,
                initializer=_configure_worker,
            )
            futures = {
                executor.submit(_collect_worker, fixture, config): fixture
                for fixture in remaining
            }
            try:
                for future in as_completed(futures):
                    summary = future.result()
                    summaries.append(summary)
                    if progress is not None:
                        progress(summary, len(summaries), len(fixtures))
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
    except (KeyboardInterrupt, ResponseDatasetInterrupted):
        _write_state(config, "interrupted")
        raise ResponseDatasetInterrupted("response collection interrupted")
    except Exception as error:
        _write_state(config, "failed", error=type(error).__name__)
        raise

    summaries.sort(key=lambda item: (item.split.value, item.fixture_index))
    for split in (FixtureSplit.TRAINING, FixtureSplit.VALIDATION):
        document = _manifest(
            split,
            config,
            tuple(item for item in summaries if item.split is split),
        )
        _atomic_json(
            config.output_path
            / "response-distillation"
            / f"{split.value}-manifest.json",
            document,
        )
    inspection = _inspection(
        config, summaries, time.perf_counter() - started
    )
    _atomic_json(
        config.output_path / "response-distillation" / "metrics.json",
        asdict(inspection),
    )
    _write_state(
        config,
        "complete",
        dataset_digest=inspection.dataset_digest,
        examples=inspection.examples,
    )
    return inspection


def load_response_config(
    output_directory: str | Path,
) -> ResponseDatasetConfig:
    output = Path(output_directory).expanduser().resolve()
    value = _load_json(output / "resolved-config.json")
    if (
        not isinstance(value, dict)
        or value.get("format_version") != RESPONSE_CONFIG_FORMAT_VERSION
    ):
        raise ResponseDatasetError("resolved response configuration is invalid")
    fields = set(ResponseDatasetConfig.__dataclass_fields__)
    try:
        config = ResponseDatasetConfig(
            **{key: value[key] for key in fields}
        )
    except (KeyError, TypeError) as error:
        raise ResponseDatasetError(
            "resolved response configuration is incomplete"
        ) from error
    if _resolved_config(config) != value:
        raise ResponseDatasetError(
            "resolved response configuration has unknown fields"
        )
    return config


def inspect_response_dataset(
    output_directory: str | Path,
) -> ResponseDatasetInspection:
    config = load_response_config(output_directory)
    fixtures = fixture_schedule(config)
    summaries = []
    for fixture in fixtures:
        summary = _summary_from_shard(fixture, config)
        if summary is None:
            raise ResponseDatasetError("response dataset is incomplete")
        summaries.append(summary)
    metrics = _load_json(
        config.output_path / "response-distillation" / "metrics.json"
    )
    wall_seconds = (
        float(metrics["wall_seconds"])
        if isinstance(metrics, dict) and "wall_seconds" in metrics
        else 0.0
    )
    inspection = _inspection(config, summaries, wall_seconds)
    if (
        isinstance(metrics, dict)
        and metrics.get("dataset_digest") != inspection.dataset_digest
    ):
        raise ResponseDatasetError("response metrics name another dataset")
    return inspection


def response_dataset_statistics(
    output_directory: str | Path,
) -> dict[str, object]:
    config = load_response_config(output_directory)
    placement_counts: Counter[int] = Counter()
    group_counts: Counter[int] = Counter()
    selected_counts: Counter[int] = Counter()
    spreads: list[float] = []
    gaps: list[float] = []
    placement_spreads: dict[int, list[float]] = {}
    placement_gaps: dict[int, list[float]] = {}
    placement_groups: dict[int, list[int]] = {}
    role_dealer_counts: Counter[str] = Counter()
    for fixture in fixture_schedule(config):
        shard = load_response_shard(_shard_path(config.output_path, fixture))
        columns = shard["columns"]
        offsets = columns["group_offsets"]
        for row, placement in enumerate(columns["placement_numbers"].tolist()):
            start = int(offsets[row].item())
            end = int(offsets[row + 1].item())
            values = sorted(
                (
                    float(value)
                    for value in columns["group_mean_values"][start:end].tolist()
                ),
                reverse=True,
            )
            placement_counts[int(placement)] += 1
            group_count = end - start
            group_counts[group_count] += 1
            selected_counts[
                int(columns["selected_representative_actions"][row].item())
            ] += 1
            spread = values[0] - values[-1]
            gap = values[0] - values[1] if len(values) > 1 else 0.0
            spreads.append(spread)
            gaps.append(gap)
            placement_spreads.setdefault(int(placement), []).append(spread)
            placement_gaps.setdefault(int(placement), []).append(gap)
            placement_groups.setdefault(int(placement), []).append(group_count)
            role = columns["actor_roles"][row]
            dealer = bool(columns["actor_is_dealer"][row].item())
            role_dealer_counts[f"{role}:{'dealer' if dealer else 'non-dealer'}"] += 1
    inspection = inspect_response_dataset(config.output_path)
    return {
        "dataset_digest": inspection.dataset_digest,
        "examples": inspection.examples,
        "examples_by_placement": dict(sorted(placement_counts.items())),
        "placement_statistics": {
            placement: {
                "examples": placement_counts[placement],
                "mean_group_count": statistics.fmean(
                    placement_groups[placement]
                ),
                "score_spread": _summary(placement_spreads[placement]),
                "top_two_score_gap": _summary(placement_gaps[placement]),
            }
            for placement in sorted(placement_counts)
        },
        "group_count_frequency": dict(sorted(group_counts.items())),
        "score_spread": _summary(spreads),
        "top_two_score_gap": _summary(gaps),
        "selected_group_frequency": dict(sorted(selected_counts.items())),
        "role_dealer_frequency": dict(sorted(role_dealer_counts.items())),
        "response_requests": inspection.response_requests,
        "observer_emissions": inspection.observer_emissions,
        "duplicates_removed": inspection.duplicates_removed,
        "duplicate_rate": (
            inspection.duplicates_removed / inspection.response_requests
            if inspection.response_requests
            else 0.0
        ),
        "wall_seconds": inspection.wall_seconds,
        "search_seconds": inspection.search_seconds,
        "peak_rss_bytes": inspection.peak_rss_bytes,
        "disk_bytes": inspection.disk_bytes,
    }


def _summary(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "median": 0.0, "p95": 0.0}
    ordered = sorted(values)
    p95_index = min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p95": ordered[p95_index],
    }


def _print_progress(
    summary: ResponseShardSummary,
    complete: int,
    total: int,
) -> None:
    print(
        "response_dataset "
        f"completed={complete}/{total} "
        f"split={summary.split.value} "
        f"fixture={summary.fixture_index} "
        f"examples={summary.examples} "
        f"requests={summary.response_requests}",
        flush=True,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="dracula-response-dataset")
    subparsers = parser.add_subparsers(dest="command", required=True)
    collect_parser = subparsers.add_parser("collect")
    collect_parser.add_argument("--output", required=True)
    collect_parser.add_argument("--run-id", required=True)
    collect_parser.add_argument("--root-seed", required=True)
    collect_parser.add_argument("--training-games", type=int, default=4)
    collect_parser.add_argument("--validation-games", type=int, default=2)
    collect_parser.add_argument("--workers", type=int, default=4)
    resume_parser = subparsers.add_parser("resume")
    resume_parser.add_argument("--output", required=True)
    inspect_parser = subparsers.add_parser("inspect")
    inspect_parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "collect":
            inspection = collect_response_dataset(
                ResponseDatasetConfig(
                    run_id=args.run_id,
                    root_seed=args.root_seed,
                    output_directory=args.output,
                    training_games=args.training_games,
                    validation_games=args.validation_games,
                    workers=args.workers,
                ),
                progress=_print_progress,
            )
        elif args.command == "resume":
            config = load_response_config(args.output)
            inspection = collect_response_dataset(
                config, resume=True, progress=_print_progress
            )
        else:
            inspection = inspect_response_dataset(args.output)
    except ResponseDatasetInterrupted:
        print("response_dataset interrupted", file=sys.stderr)
        return 130
    print(json.dumps(asdict(inspection), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = (
    "RESPONSE_COLLECTION_PROFILE",
    "RESPONSE_CONFIG_FORMAT_VERSION",
    "RESPONSE_DATASET_SCHEMA_VERSION",
    "RESPONSE_MANIFEST_FORMAT_VERSION",
    "RESPONSE_SHARD_FORMAT_VERSION",
    "ResponseDatasetConfig",
    "ResponseDatasetError",
    "ResponseDatasetInspection",
    "ResponseDatasetInterrupted",
    "collect_response_dataset",
    "fixture_schedule",
    "inspect_response_dataset",
    "load_response_config",
    "load_response_manifest",
    "load_response_shard",
    "main",
    "response_dataset_statistics",
)

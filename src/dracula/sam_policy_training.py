"""Committed-corpus loading and standalone Sam-policy imitation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import resource
import shutil
import subprocess
import sys
import time
import tomllib
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor

from dracula.engine import EnginePlayer
from dracula.randomness import derive_seed
from dracula.sam_miner import (
    SAM_MINER_CORPUS_MANIFEST_VERSION,
    SAM_MINER_DECK_MANIFEST_VERSION,
    SAM_MINER_ROUND_MANIFEST_VERSION,
    SAM_MINER_SHARD_FORMAT_VERSION,
    SAM_MINER_SOURCE_TREE_SCHEMA_VERSION,
    DeckSplit,
    SamMinerConfig,
    SamMinerError,
    SamMinerRow,
    load_config as load_miner_config,
    load_subtree_shard,
    resolve_source_identity,
    unpack_legal_mask,
    unpack_observation,
    verify_corpus,
)
from dracula.sam_policy import (
    ACTION_COUNT,
    ACTION_SCHEMA_VERSION,
    COFFIN_START,
    HAND_SLOT_COUNT,
    MODEL_SCHEMA_VERSION,
    OBSERVATION_SCHEMA_VERSION,
    OBSERVATION_SIZE,
    OPTIMIZER_COMPATIBILITY_VERSION,
    PARAMETER_COUNT,
    POLICY_GRID_INDICES,
    POLICY_POSITION_COUNT,
    REPRESENTATIVE_MASK_SCHEMA_VERSION,
    STATUS_START,
    SamPolicyContractError,
    SamPolicyModel,
    apply_representative_mask,
    build_representative_action_projection,
    load_sam_policy_artifact,
    save_sam_policy_artifact,
)
from dracula.search.symmetry import destination_symmetry_groups

SNAPSHOT_FORMAT_VERSION = "dracula-sam-policy-snapshot-v1"
TRAINING_CONFIG_FORMAT_VERSION = "dracula-sam-policy-training-config-v1"
RESOLVED_CONFIG_FORMAT_VERSION = (
    "dracula-sam-policy-resolved-config-v1"
)
CHECKPOINT_FORMAT_VERSION = "dracula-sam-policy-checkpoint-v1"
METRICS_FORMAT_VERSION = "dracula-sam-policy-epoch-metrics-v1"
SUMMARY_FORMAT_VERSION = "dracula-sam-policy-summary-v1"
DATASET_LOADER_SCHEMA_VERSION = "dracula-sam-policy-loader-v1"
SPLIT_SCHEMA_VERSION = "dracula-sam-policy-overlay-split-v1"
TRAINING_STATE_SCHEMA_VERSION = "dracula-sam-policy-training-state-v1"
EPOCH_SHUFFLE_NAMESPACE = "dracula-sam-policy-epoch-shuffle-v1"

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
_SNAPSHOT_OVERLAYS = ("training", "validation")
_FORBIDDEN_DATA_KEYS = frozenset(
    {
        "action_values",
        "authoritative_state",
        "determinization",
        "determinizations",
        "engine_seed",
        "game_seed",
        "model",
        "model_data",
        "opponent_hand",
        "outer_sampled_state",
        "policy_hidden_state",
        "return",
        "returns",
        "root_visits",
        "round_return",
        "score_estimate",
        "search_tree",
        "stock",
        "stock_order",
        "tree",
        "value",
        "values",
        "visits",
    }
)


class SamPolicyTrainingError(ValueError):
    """A snapshot, row, configuration, or training artifact is invalid."""


class SamPolicyTrainingInterrupted(RuntimeError):
    """Training stopped at a sealed minibatch boundary."""


@dataclass(frozen=True, slots=True)
class SnapshotDeck:
    ordinal: int
    original_split: str
    fixture_index: int
    fixture_id: str
    relative_path: str
    content_digest: str
    file_digest: str
    row_count: int
    terminal_leaf_count: int
    overlay: Literal["training", "validation"]


@dataclass(frozen=True, slots=True)
class CommittedCorpusSnapshot:
    path: Path
    corpus_directory: Path
    snapshot_digest: str
    source_corpus_manifest_digest: str
    source_collection_config_digest: str
    source_revision: str
    source_tree_digest: str
    highest_included_ordinal: int
    decks: tuple[SnapshotDeck, ...]
    training_row_count: int
    validation_row_count: int
    placement_counts: tuple[tuple[str, tuple[tuple[int, int], ...]], ...]


@dataclass(frozen=True, slots=True)
class SamPolicyDataset:
    overlay: Literal["training", "validation"]
    observations_packed: Tensor
    representative_masks_packed: Tensor
    targets: Tensor
    placements: Tensor
    players: Tensor
    dealers: Tensor
    deck_ordinals: Tensor
    fixture_ids: tuple[str, ...]
    dataset_digest: str
    split_digest: str

    @property
    def example_count(self) -> int:
        return int(self.targets.shape[0])

    def decoded_batch(
        self, indexes: Tensor, *, device: torch.device
    ) -> tuple[Tensor, Tensor, Tensor]:
        if (
            indexes.dtype is not torch.long
            or indexes.ndim != 1
            or indexes.device.type != "cpu"
        ):
            raise SamPolicyTrainingError(
                "dataset indexes must be a one-dimensional CPU long tensor"
            )
        observations = _unpack_tensor_bits(
            self.observations_packed.index_select(0, indexes),
            OBSERVATION_SIZE,
        ).to(device=device)
        masks = _unpack_tensor_bits(
            self.representative_masks_packed.index_select(0, indexes),
            ACTION_COUNT,
        ).reshape(-1, HAND_SLOT_COUNT, POLICY_POSITION_COUNT)
        targets = self.targets.index_select(0, indexes)
        return (
            observations.to(device=device),
            masks.to(device=device),
            targets.to(device=device),
        )


@dataclass(frozen=True, slots=True)
class SamPolicyDatasetBundle:
    snapshot: CommittedCorpusSnapshot
    training: SamPolicyDataset
    validation: SamPolicyDataset
    dataset_digest: str
    split_digest: str


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
    minimum_epochs: int = MINIMUM_EPOCHS
    maximum_epochs: int = MAXIMUM_EPOCHS
    early_stop_patience: int = EARLY_STOP_PATIENCE
    minimum_improvement: float = MINIMUM_IMPROVEMENT


@dataclass(frozen=True, slots=True)
class SamPolicyTrainingConfig:
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
class GroupMetrics:
    count: int
    cross_entropy: float
    top_one: float
    top_two: float
    top_three: float


@dataclass(frozen=True, slots=True)
class EvaluationMetrics:
    total: GroupMetrics
    placements: dict[str, GroupMetrics]
    roles: dict[str, GroupMetrics]
    dealer_status: dict[str, GroupMetrics]
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
    exported_artifact: str


@dataclass(frozen=True, slots=True)
class _StoredStrategicGroup:
    hand_slot: int
    representative_action_index: int
    representative_grid_index: int
    member_action_indices: tuple[int, ...]
    member_grid_indices: tuple[int, ...]


@dataclass(slots=True)
class _MetricAccumulator:
    loss: float = 0.0
    count: int = 0
    top_one: int = 0
    top_two: int = 0
    top_three: int = 0

    def add(
        self,
        losses: Tensor,
        top_one: Tensor,
        top_two: Tensor,
        top_three: Tensor,
        indexes: Tensor | None = None,
    ) -> None:
        if indexes is not None:
            losses = losses.index_select(0, indexes)
            top_one = top_one.index_select(0, indexes)
            top_two = top_two.index_select(0, indexes)
            top_three = top_three.index_select(0, indexes)
        self.loss += float(losses.sum().item())
        self.count += int(losses.numel())
        self.top_one += int(top_one.sum().item())
        self.top_two += int(top_two.sum().item())
        self.top_three += int(top_three.sum().item())

    def finish(self) -> GroupMetrics:
        if self.count < 1:
            return GroupMetrics(0, 0.0, 0.0, 0.0, 0.0)
        return GroupMetrics(
            self.count,
            self.loss / self.count,
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
        raise SamPolicyTrainingError(
            "artifact is not canonical JSON"
        ) from error


def _json_digest(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _require_digest(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != _DIGEST_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise SamPolicyTrainingError(
            f"{label} must be a lowercase SHA-256 digest"
        )
    return value


def _load_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SamPolicyTrainingError(
            f"JSON artifact could not be read: {path}"
        ) from error


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
        or set(value)
        != {"format_version", "content", "content_digest", "file_digest"}
        or value["format_version"] != format_version
        or value["content_digest"] != _json_digest(value["content"])
        or value["file_digest"]
        != _json_digest(
            {
                "format_version": value["format_version"],
                "content": value["content"],
                "content_digest": value["content_digest"],
            }
        )
    ):
        raise SamPolicyTrainingError(
            f"{format_version} envelope digest differs"
        )
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
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
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
            raise SamPolicyTrainingError(
                "training data contains forbidden fields: "
                + ", ".join(sorted(forbidden))
            )
        for nested in value.values():
            _assert_no_forbidden_keys(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            _assert_no_forbidden_keys(nested)


def _continuous_ordinal(
    config: SamMinerConfig, split: DeckSplit, fixture_index: int
) -> int:
    cycle = (
        (DeckSplit.TRAINING,) * config.training_decks
        + (DeckSplit.VALIDATION,) * config.validation_decks
        + (DeckSplit.TEST,) * config.test_decks
    )
    positions = [
        index for index, item in enumerate(cycle) if item is split
    ]
    if not positions:
        raise SamPolicyTrainingError(
            "snapshot deck uses a disabled corpus split"
        )
    cycle_number, split_offset = divmod(
        fixture_index, len(positions)
    )
    return cycle_number * len(cycle) + positions[split_offset]


def _corpus_manifest(
    corpus_directory: Path,
) -> tuple[SamMinerConfig, dict[str, object]]:
    try:
        config = load_miner_config(corpus_directory)
        verify_corpus(corpus_directory)
    except SamMinerError as error:
        raise SamPolicyTrainingError(
            "source corpus verification failed"
        ) from error
    if not config.continuous:
        raise SamPolicyTrainingError(
            "snapshot source must be a continuous committed corpus"
        )
    document = _validate_envelope(
        _load_json(corpus_directory / "corpus-manifest.json"),
        SAM_MINER_CORPUS_MANIFEST_VERSION,
    )
    content = document["content"]
    if (
        not isinstance(content, dict)
        or content.get("collection_config_digest") != config.digest
        or not isinstance(content.get("decks"), list)
    ):
        raise SamPolicyTrainingError(
            "source corpus manifest identity differs"
        )
    return config, document


def _snapshot_decks(
    config: SamMinerConfig, document: dict[str, object]
) -> tuple[SnapshotDeck, ...]:
    content = document["content"]
    assert isinstance(content, dict)
    entries: list[tuple[int, dict[str, object]]] = []
    for value in content["decks"]:  # type: ignore[index]
        if not isinstance(value, dict):
            raise SamPolicyTrainingError(
                "source corpus deck entry is malformed"
            )
        split = DeckSplit(str(value["split"]))
        ordinal = _continuous_ordinal(
            config, split, int(value["fixture_index"])
        )
        entries.append((ordinal, value))
    entries.sort(key=lambda item: item[0])
    if [ordinal for ordinal, _ in entries] != list(range(len(entries))):
        raise SamPolicyTrainingError(
            "snapshot requires a gap-free committed deck prefix"
        )
    if len(entries) < 3:
        raise SamPolicyTrainingError(
            "snapshot requires at least three committed decks"
        )
    validation_ordinals = {len(entries) - 2, len(entries) - 1}
    result = []
    for ordinal, value in entries:
        relative = Path(str(value["relative_path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise SamPolicyTrainingError(
                "source deck path escapes the corpus"
            )
        result.append(
            SnapshotDeck(
                ordinal=ordinal,
                original_split=str(value["split"]),
                fixture_index=int(value["fixture_index"]),
                fixture_id=_require_digest(
                    value["fixture_id"], "fixture ID"
                ),
                relative_path=relative.as_posix(),
                content_digest=_require_digest(
                    value["content_digest"], "deck content digest"
                ),
                file_digest=_require_digest(
                    value["file_digest"], "deck file digest"
                ),
                row_count=int(value["row_count"]),
                terminal_leaf_count=int(
                    value["terminal_leaf_count"]
                ),
                overlay=(
                    "validation"
                    if ordinal in validation_ordinals
                    else "training"
                ),
            )
        )
    return tuple(result)


def _iter_rows_for_deck(
    corpus_directory: Path,
    config: SamMinerConfig,
    deck: SnapshotDeck,
) -> Iterable[SamMinerRow]:
    deck_path = corpus_directory / deck.relative_path
    deck_document = _validate_envelope(
        _load_json(deck_path), SAM_MINER_DECK_MANIFEST_VERSION
    )
    if (
        deck_document["content_digest"] != deck.content_digest
        or deck_document["file_digest"] != deck.file_digest
    ):
        raise SamPolicyTrainingError(
            "snapshot deck manifest digest differs"
        )
    content = deck_document["content"]
    if (
        not isinstance(content, dict)
        or content.get("collection_config_digest") != config.digest
        or content.get("fixture_id") != deck.fixture_id
        or content.get("fixture_index") != deck.fixture_index
        or content.get("split") != deck.original_split
        or content.get("row_count") != deck.row_count
        or not isinstance(content.get("rounds"), list)
    ):
        raise SamPolicyTrainingError(
            "snapshot deck manifest identity differs"
        )
    deck_row_count = 0
    round_numbers: list[int] = []
    for round_entry in content["rounds"]:
        if not isinstance(round_entry, dict):
            raise SamPolicyTrainingError(
                "snapshot round reference is malformed"
            )
        round_relative = Path(str(round_entry["relative_path"]))
        if round_relative.is_absolute() or ".." in round_relative.parts:
            raise SamPolicyTrainingError(
                "snapshot round path escapes the corpus"
            )
        round_document = _validate_envelope(
            _load_json(corpus_directory / round_relative),
            SAM_MINER_ROUND_MANIFEST_VERSION,
        )
        if (
            round_document["content_digest"]
            != round_entry["content_digest"]
            or round_document["file_digest"] != round_entry["file_digest"]
        ):
            raise SamPolicyTrainingError(
                "snapshot round manifest digest differs"
            )
        round_content = round_document["content"]
        if (
            not isinstance(round_content, dict)
            or round_content.get("collection_config_digest") != config.digest
            or round_content.get("fixture_id") != deck.fixture_id
            or not isinstance(round_content.get("shards"), list)
        ):
            raise SamPolicyTrainingError(
                "snapshot round manifest identity differs"
            )
        round_number = int(round_content["round_number"])
        round_numbers.append(round_number)
        round_rows = 0
        paths: list[str] = []
        for shard_entry in round_content["shards"]:
            if not isinstance(shard_entry, dict):
                raise SamPolicyTrainingError(
                    "snapshot shard reference is malformed"
                )
            relative = Path(str(shard_entry["relative_path"]))
            if relative.is_absolute() or ".." in relative.parts:
                raise SamPolicyTrainingError(
                    "snapshot shard path escapes the corpus"
                )
            try:
                shard = load_subtree_shard(
                    corpus_directory / relative, config
                )
            except SamMinerError as error:
                raise SamPolicyTrainingError(
                    "snapshot shard verification failed"
                ) from error
            if (
                shard["format_version"] != SAM_MINER_SHARD_FORMAT_VERSION
                or shard["content_digest"]
                != shard_entry["content_digest"]
                or shard["file_digest"] != shard_entry["file_digest"]
            ):
                raise SamPolicyTrainingError(
                    "snapshot shard digest differs"
                )
            shard_content = shard["content"]
            assert isinstance(shard_content, dict)
            _assert_no_forbidden_keys(shard_content)
            raw_rows = shard_content["rows"]
            if (
                not isinstance(raw_rows, list)
                or len(raw_rows) != int(shard_entry["row_count"])
            ):
                raise SamPolicyTrainingError(
                    "snapshot shard row count differs"
                )
            for raw_row in raw_rows:
                row = SamMinerRow.from_dict(raw_row)
                if (
                    row.deck_fixture_id != deck.fixture_id
                    or row.round_number != round_number
                ):
                    raise SamPolicyTrainingError(
                        "snapshot shard row identity differs"
                    )
                round_rows += 1
                deck_row_count += 1
                yield row
            paths.append(relative.as_posix())
        if paths != sorted(paths):
            raise SamPolicyTrainingError(
                "snapshot shard order is not canonical"
            )
        if round_rows != int(round_content["row_count"]):
            raise SamPolicyTrainingError(
                "snapshot round row count differs"
            )
    if (
        round_numbers != list(range(1, 7))
        or deck_row_count != deck.row_count
    ):
        raise SamPolicyTrainingError(
            "snapshot deck row hierarchy differs"
        )


def _read_rows_for_deck(
    corpus_directory: Path,
    config: SamMinerConfig,
    deck: SnapshotDeck,
) -> tuple[SamMinerRow, ...]:
    return tuple(_iter_rows_for_deck(corpus_directory, config, deck))


def _expected_groups(
    observation: tuple[bool, ...],
    legal_mask: tuple[tuple[bool, ...], ...],
) -> tuple[_StoredStrategicGroup, ...]:
    coffin_bits = observation[COFFIN_START:STATUS_START]
    coffin = tuple(
        object()
        if any(
            coffin_bits[
                grid_index * 54 : (grid_index + 1) * 54
            ]
        )
        else None
        for grid_index in range(9)
    )
    destination_groups = destination_symmetry_groups(coffin)
    grid_to_policy = {
        grid_index: position
        for position, grid_index in enumerate(POLICY_GRID_INDICES)
    }
    legal_slots = tuple(
        slot for slot, row in enumerate(legal_mask) if any(row)
    )
    groups = []
    for slot in legal_slots:
        legal_positions = {
            position
            for position, allowed in enumerate(legal_mask[slot])
            if allowed
        }
        for destination_group in destination_groups:
            member_positions = tuple(
                grid_to_policy[grid]
                for grid in destination_group.member_grid_indices
            )
            if not set(member_positions).issubset(legal_positions):
                raise SamPolicyTrainingError(
                    "authoritative groups disagree with engine legality"
                )
            representative_position = grid_to_policy[
                destination_group.representative_grid_index
            ]
            groups.append(
                _StoredStrategicGroup(
                    hand_slot=slot,
                    representative_action_index=(
                        slot * POLICY_POSITION_COUNT
                        + representative_position
                    ),
                    representative_grid_index=(
                        destination_group.representative_grid_index
                    ),
                    member_action_indices=tuple(
                        slot * POLICY_POSITION_COUNT + position
                        for position in member_positions
                    ),
                    member_grid_indices=(
                        destination_group.member_grid_indices
                    ),
                )
            )
    groups.sort(key=lambda group: group.representative_action_index)
    return tuple(groups)


def _pack_tensor_bits(bits: Tensor) -> Tensor:
    if bits.dtype is not torch.bool or bits.ndim != 2:
        raise SamPolicyTrainingError(
            "packed input must be a two-dimensional Boolean tensor"
        )
    padded = ((bits.shape[1] + 7) // 8) * 8
    if padded != bits.shape[1]:
        bits = F.pad(bits, (0, padded - bits.shape[1]))
    weights = torch.tensor(
        (128, 64, 32, 16, 8, 4, 2, 1), dtype=torch.int16
    )
    return (
        bits.reshape(bits.shape[0], -1, 8).to(torch.int16)
        .mul(weights)
        .sum(dim=2)
        .to(torch.uint8)
    )


def _unpack_tensor_bits(packed: Tensor, bit_count: int) -> Tensor:
    if packed.dtype is not torch.uint8 or packed.ndim != 2:
        raise SamPolicyTrainingError(
            "packed dataset tensor must be two-dimensional uint8"
        )
    shifts = torch.arange(7, -1, -1, dtype=torch.uint8)
    bits = ((packed.unsqueeze(-1) >> shifts) & 1).to(torch.bool)
    return bits.reshape(packed.shape[0], -1)[:, :bit_count]


def _project_row(
    row: SamMinerRow,
) -> tuple[bytes, bytes, int]:
    observation = unpack_observation(row)
    legal_mask_tuple = unpack_legal_mask(row)
    expected_groups = _expected_groups(observation, legal_mask_tuple)
    expected_members = tuple(
        group.member_action_indices for group in expected_groups
    )
    if expected_members != row.strategic_groups:
        raise SamPolicyTrainingError(
            "sealed strategic groups differ from authoritative symmetry"
        )
    legal_actions = {
        slot * POLICY_POSITION_COUNT + position
        for slot, mask_row in enumerate(legal_mask_tuple)
        for position, allowed in enumerate(mask_row)
        if allowed
    }
    grouped_actions = {
        action
        for group in expected_groups
        for action in group.member_action_indices
    }
    if grouped_actions != legal_actions:
        raise SamPolicyTrainingError(
            "representative groups do not partition engine legality"
        )
    if (
        row.teacher_group_index >= len(expected_groups)
        or expected_groups[
            row.teacher_group_index
        ].representative_action_index
        != row.teacher_group_representative
    ):
        raise SamPolicyTrainingError(
            "Sam group target differs from its designated representative"
        )
    target = row.teacher_group_representative
    representatives = {
        group.representative_action_index for group in expected_groups
    }
    if target not in representatives:
        raise SamPolicyTrainingError(
            "Sam target is absent from the representative mask"
        )
    mask = bytearray(4)
    for representative in representatives:
        mask[representative // 8] |= 1 << (7 - representative % 8)
    observation_bytes = bytes(row.observation_packed)
    return observation_bytes, bytes(mask), target


def _scan_snapshot_counts(
    corpus_directory: Path,
    config: SamMinerConfig,
    decks: Sequence[SnapshotDeck],
) -> dict[str, Counter[int]]:
    counts = {overlay: Counter() for overlay in _SNAPSHOT_OVERLAYS}
    for deck in decks:
        for row in _iter_rows_for_deck(corpus_directory, config, deck):
            _project_row(row)
            counts[deck.overlay][row.placement_number] += 1
    return counts


def create_committed_corpus_snapshot(
    corpus_directory: str | Path,
    output_path: str | Path,
) -> CommittedCorpusSnapshot:
    corpus = Path(corpus_directory).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    config, corpus_document = _corpus_manifest(corpus)
    decks = _snapshot_decks(config, corpus_document)
    counts = _scan_snapshot_counts(corpus, config, decks)
    content = {
        "snapshot_schema_version": SNAPSHOT_FORMAT_VERSION,
        "dataset_loader_schema_version": DATASET_LOADER_SCHEMA_VERSION,
        "source_corpus_directory": str(corpus),
        "source_corpus_manifest_digest": corpus_document["content_digest"],
        "source_collection_config_digest": config.digest,
        "source_tree_schema_version": config.source_tree_schema_version,
        "source_revision": config.source_revision,
        "source_tree_digest": config.source_tree_digest,
        "highest_included_ordinal": len(decks) - 1,
        "split_schema_version": SPLIT_SCHEMA_VERSION,
        "split_rule": "highest-two-validation-earlier-training-v1",
        "decks": [asdict(deck) for deck in decks],
        "training_row_count": sum(
            deck.row_count for deck in decks if deck.overlay == "training"
        ),
        "validation_row_count": sum(
            deck.row_count
            for deck in decks
            if deck.overlay == "validation"
        ),
        "placement_counts": {
            overlay: {
                str(placement): counts[overlay][placement]
                for placement in range(1, 8)
            }
            for overlay in _SNAPSHOT_OVERLAYS
        },
    }
    if (
        content["source_tree_schema_version"]
        != SAM_MINER_SOURCE_TREE_SCHEMA_VERSION
        or content["source_revision"] is None
        or content["source_tree_digest"] is None
    ):
        raise SamPolicyTrainingError(
            "snapshot source corpus lacks a sealed source identity"
        )
    document = _envelope(SNAPSHOT_FORMAT_VERSION, content)
    if output.exists():
        existing = _validate_envelope(
            _load_json(output), SNAPSHOT_FORMAT_VERSION
        )
        if existing != document:
            raise SamPolicyTrainingError(
                "snapshot path already contains different content"
            )
    else:
        _atomic_json(output, document)
    return load_committed_corpus_snapshot(output)


def load_committed_corpus_snapshot(
    path: str | Path,
) -> CommittedCorpusSnapshot:
    snapshot_path = Path(path).expanduser().resolve()
    document = _validate_envelope(
        _load_json(snapshot_path), SNAPSHOT_FORMAT_VERSION
    )
    content = document["content"]
    required = {
        "snapshot_schema_version",
        "dataset_loader_schema_version",
        "source_corpus_directory",
        "source_corpus_manifest_digest",
        "source_collection_config_digest",
        "source_tree_schema_version",
        "source_revision",
        "source_tree_digest",
        "highest_included_ordinal",
        "split_schema_version",
        "split_rule",
        "decks",
        "training_row_count",
        "validation_row_count",
        "placement_counts",
    }
    if (
        not isinstance(content, dict)
        or set(content) != required
        or content["snapshot_schema_version"] != SNAPSHOT_FORMAT_VERSION
        or content["dataset_loader_schema_version"]
        != DATASET_LOADER_SCHEMA_VERSION
        or content["source_tree_schema_version"]
        != SAM_MINER_SOURCE_TREE_SCHEMA_VERSION
        or content["split_schema_version"] != SPLIT_SCHEMA_VERSION
        or content["split_rule"]
        != "highest-two-validation-earlier-training-v1"
        or not isinstance(content["decks"], list)
    ):
        raise SamPolicyTrainingError(
            "snapshot contract is incompatible"
        )
    corpus = Path(str(content["source_corpus_directory"])).resolve()
    config, current_corpus = _corpus_manifest(corpus)
    if (
        config.digest != content["source_collection_config_digest"]
        or config.source_revision != content["source_revision"]
        or config.source_tree_digest != content["source_tree_digest"]
    ):
        raise SamPolicyTrainingError(
            "snapshot source configuration differs"
        )
    current_content = current_corpus["content"]
    assert isinstance(current_content, dict)
    current_by_fixture = {
        entry["fixture_id"]: entry
        for entry in current_content["decks"]  # type: ignore[index]
        if isinstance(entry, dict)
    }
    try:
        decks = tuple(
            SnapshotDeck(**deck_value)
            for deck_value in content["decks"]
        )
    except (TypeError, ValueError) as error:
        raise SamPolicyTrainingError(
            "snapshot deck entries are malformed"
        ) from error
    if (
        len(decks) < 3
        or tuple(deck.ordinal for deck in decks)
        != tuple(range(len(decks)))
        or content["highest_included_ordinal"] != len(decks) - 1
        or tuple(deck.overlay for deck in decks[-2:])
        != ("validation", "validation")
        or any(deck.overlay != "training" for deck in decks[:-2])
    ):
        raise SamPolicyTrainingError(
            "snapshot overlay assignment differs"
        )
    for deck in decks:
        current = current_by_fixture.get(deck.fixture_id)
        expected = {
            "split": deck.original_split,
            "fixture_index": deck.fixture_index,
            "fixture_id": deck.fixture_id,
            "relative_path": deck.relative_path,
            "content_digest": deck.content_digest,
            "file_digest": deck.file_digest,
            "row_count": deck.row_count,
            "terminal_leaf_count": deck.terminal_leaf_count,
        }
        if current != expected:
            raise SamPolicyTrainingError(
                "snapshot deck is absent or changed in the source corpus"
            )
    placement_value = content["placement_counts"]
    if not isinstance(placement_value, dict):
        raise SamPolicyTrainingError(
            "snapshot placement counts are malformed"
        )
    placement_counts = tuple(
        (
            overlay,
            tuple(
                (placement, int(placement_value[overlay][str(placement)]))
                for placement in range(1, 8)
            ),
        )
        for overlay in _SNAPSHOT_OVERLAYS
    )
    return CommittedCorpusSnapshot(
        path=snapshot_path,
        corpus_directory=corpus,
        snapshot_digest=str(document["content_digest"]),
        source_corpus_manifest_digest=_require_digest(
            content["source_corpus_manifest_digest"],
            "source corpus manifest digest",
        ),
        source_collection_config_digest=_require_digest(
            content["source_collection_config_digest"],
            "source collection configuration digest",
        ),
        source_revision=str(content["source_revision"]),
        source_tree_digest=_require_digest(
            content["source_tree_digest"], "source tree digest"
        ),
        highest_included_ordinal=int(content["highest_included_ordinal"]),
        decks=decks,
        training_row_count=int(content["training_row_count"]),
        validation_row_count=int(content["validation_row_count"]),
        placement_counts=placement_counts,
    )


def _empty_dataset_tensors(
    count: int,
) -> dict[str, Tensor]:
    return {
        "observations": torch.empty((count, 110), dtype=torch.uint8),
        "masks": torch.empty((count, 4), dtype=torch.uint8),
        "targets": torch.empty(count, dtype=torch.long),
        "placements": torch.empty(count, dtype=torch.uint8),
        "players": torch.empty(count, dtype=torch.uint8),
        "dealers": torch.empty(count, dtype=torch.uint8),
        "ordinals": torch.empty(count, dtype=torch.int32),
    }


def _dataset_from_tensors(
    overlay: Literal["training", "validation"],
    tensors: Mapping[str, Tensor],
    row_identity_digest: str,
    decks: Sequence[SnapshotDeck],
) -> SamPolicyDataset:
    observations = tensors["observations"]
    masks = tensors["masks"]
    targets = tensors["targets"]
    placements = tensors["placements"]
    players = tensors["players"]
    dealers = tensors["dealers"]
    ordinals = tensors["ordinals"]
    fixture_ids = tuple(
        deck.fixture_id for deck in decks if deck.overlay == overlay
    )
    deck_bindings = [
        {
            "ordinal": deck.ordinal,
            "fixture_id": deck.fixture_id,
            "content_digest": deck.content_digest,
            "row_count": deck.row_count,
        }
        for deck in decks
        if deck.overlay == overlay
    ]
    split_digest = _json_digest(
        {
            "split_schema_version": SPLIT_SCHEMA_VERSION,
            "overlay": overlay,
            "decks": deck_bindings,
        }
    )
    dataset_digest = _json_digest(
        {
            "dataset_loader_schema_version": DATASET_LOADER_SCHEMA_VERSION,
            "overlay": overlay,
            "split_digest": split_digest,
            "example_count": int(targets.shape[0]),
            "row_identity_digest": row_identity_digest,
        }
    )
    return SamPolicyDataset(
        overlay=overlay,
        observations_packed=observations,
        representative_masks_packed=masks,
        targets=targets,
        placements=placements,
        players=players,
        dealers=dealers,
        deck_ordinals=ordinals,
        fixture_ids=fixture_ids,
        dataset_digest=dataset_digest,
        split_digest=split_digest,
    )


def load_snapshot_dataset(
    snapshot_path: str | Path,
) -> SamPolicyDatasetBundle:
    snapshot = load_committed_corpus_snapshot(snapshot_path)
    config = load_miner_config(snapshot.corpus_directory)
    tensors = {
        "training": _empty_dataset_tensors(snapshot.training_row_count),
        "validation": _empty_dataset_tensors(snapshot.validation_row_count),
    }
    cursors = {"training": 0, "validation": 0}
    counts = {overlay: Counter() for overlay in _SNAPSHOT_OVERLAYS}
    row_digests = {
        overlay: hashlib.sha256(
            (
                f"{DATASET_LOADER_SCHEMA_VERSION}:"
                f"{overlay}:row-identities-v1\0"
            ).encode("ascii")
        )
        for overlay in _SNAPSHOT_OVERLAYS
    }
    for deck in snapshot.decks:
        overlay = deck.overlay
        values = tensors[overlay]
        for row in _iter_rows_for_deck(
            snapshot.corpus_directory, config, deck
        ):
            observation, mask, target = _project_row(row)
            index = cursors[overlay]
            if index >= values["targets"].shape[0]:
                raise SamPolicyTrainingError(
                    "snapshot contains more rows than its sealed total"
                )
            values["observations"][index] = torch.tensor(
                tuple(observation), dtype=torch.uint8
            )
            values["masks"][index] = torch.tensor(
                tuple(mask), dtype=torch.uint8
            )
            values["targets"][index] = target
            values["placements"][index] = row.placement_number
            values["players"][index] = (
                0 if row.player is EnginePlayer.QUEEN else 1
            )
            values["dealers"][index] = (
                0 if row.dealer is EnginePlayer.QUEEN else 1
            )
            values["ordinals"][index] = deck.ordinal
            row_digests[overlay].update(
                bytes.fromhex(row.branch_path_digest)
            )
            row_digests[overlay].update(target.to_bytes(1, "big"))
            counts[overlay][row.placement_number] += 1
            cursors[overlay] += 1
    if (
        cursors["training"] != snapshot.training_row_count
        or cursors["validation"] != snapshot.validation_row_count
    ):
        raise SamPolicyTrainingError(
            "snapshot contains fewer rows than its sealed total"
        )
    expected_counts = {
        overlay: dict(placement_counts)
        for overlay, placement_counts in snapshot.placement_counts
    }
    if any(
        dict(counts[overlay]) != {
            placement: count
            for placement, count in expected_counts[overlay].items()
            if count
        }
        for overlay in _SNAPSHOT_OVERLAYS
    ):
        raise SamPolicyTrainingError(
            "snapshot placement totals differ from verified rows"
        )
    training = _dataset_from_tensors(
        "training",
        tensors["training"],
        row_digests["training"].hexdigest(),
        snapshot.decks,
    )
    validation = _dataset_from_tensors(
        "validation",
        tensors["validation"],
        row_digests["validation"].hexdigest(),
        snapshot.decks,
    )
    if (
        training.example_count != snapshot.training_row_count
        or validation.example_count != snapshot.validation_row_count
        or set(training.deck_ordinals.tolist()).intersection(
            validation.deck_ordinals.tolist()
        )
        or set(training.fixture_ids).intersection(validation.fixture_ids)
    ):
        raise SamPolicyTrainingError(
            "snapshot overlay isolation differs"
        )
    split_digest = _json_digest(
        {
            "training": training.split_digest,
            "validation": validation.split_digest,
        }
    )
    dataset_digest = _json_digest(
        {
            "snapshot_digest": snapshot.snapshot_digest,
            "training": training.dataset_digest,
            "validation": validation.dataset_digest,
        }
    )
    return SamPolicyDatasetBundle(
        snapshot,
        training,
        validation,
        dataset_digest,
        split_digest,
    )


def _strict_table(
    value: object, expected: set[str], label: str
) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != expected:
        raise SamPolicyTrainingError(
            f"{label} fields are invalid"
        )
    return value


def load_training_config(path: str | Path) -> SamPolicyTrainingConfig:
    config_path = Path(path).expanduser().resolve()
    try:
        value = tomllib.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise SamPolicyTrainingError(
            "training TOML could not be read"
        ) from error
    root = _strict_table(
        value,
        {"format_version", "run", "dataset", "model", "optimization"},
        "training configuration",
    )
    if root["format_version"] != TRAINING_CONFIG_FORMAT_VERSION:
        raise SamPolicyTrainingError(
            "training configuration version is incompatible"
        )
    run = _strict_table(
        root["run"],
        {"run_id", "root_seed", "output_directory"},
        "run",
    )
    dataset = _strict_table(
        root["dataset"], {"snapshot_path"}, "dataset"
    )
    model = _strict_table(
        root["model"],
        {"model_id", "initialization_ordinal"},
        "model",
    )
    optimization = _strict_table(
        root["optimization"],
        {
            "device",
            "minimum_epochs",
            "maximum_epochs",
            "early_stop_patience",
            "minimum_improvement",
        },
        "optimization",
    )
    typed_strings = (
        run["run_id"],
        run["root_seed"],
        run["output_directory"],
        dataset["snapshot_path"],
        model["model_id"],
        optimization["device"],
    )
    typed_integers = (
        model["initialization_ordinal"],
        optimization["minimum_epochs"],
        optimization["maximum_epochs"],
        optimization["early_stop_patience"],
    )
    minimum_improvement = optimization["minimum_improvement"]
    if (
        any(type(value) is not str for value in typed_strings)
        or any(type(value) is not int for value in typed_integers)
        or type(minimum_improvement) not in (int, float)
    ):
        raise SamPolicyTrainingError(
            "training configuration value types are invalid"
        )
    try:
        result = SamPolicyTrainingConfig(
            RunSection(
                str(run["run_id"]),
                str(run["root_seed"]),
                str(run["output_directory"]),
            ),
            DatasetSection(str(dataset["snapshot_path"])),
            ModelSection(
                str(model["model_id"]),
                int(model["initialization_ordinal"]),
            ),
            OptimizationSection(
                str(optimization["device"]),
                int(optimization["minimum_epochs"]),
                int(optimization["maximum_epochs"]),
                int(optimization["early_stop_patience"]),
                float(optimization["minimum_improvement"]),
            ),
        )
    except (TypeError, ValueError) as error:
        raise SamPolicyTrainingError(
            "training configuration values are malformed"
        ) from error
    _validate_training_config(result)
    return result


def _validate_training_config(config: SamPolicyTrainingConfig) -> None:
    strings = (
        config.run.run_id,
        config.run.root_seed,
        config.run.output_directory,
        config.dataset.snapshot_path,
        config.model.model_id,
    )
    if any(not isinstance(value, str) or not value for value in strings):
        raise SamPolicyTrainingError(
            "training strings must be nonempty"
        )
    if (
        type(config.model.initialization_ordinal) is not int
        or config.model.initialization_ordinal < 0
        or config.optimization.device not in {"cpu", "mps"}
        or type(config.optimization.minimum_epochs) is not int
        or type(config.optimization.maximum_epochs) is not int
        or not 1
        <= config.optimization.minimum_epochs
        <= config.optimization.maximum_epochs
        or type(config.optimization.early_stop_patience) is not int
        or config.optimization.early_stop_patience < 1
        or not math.isfinite(config.optimization.minimum_improvement)
        or config.optimization.minimum_improvement <= 0
    ):
        raise SamPolicyTrainingError(
            "training optimization values are invalid"
        )


def _device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    raise SamPolicyTrainingError(
        f"training device is unavailable: {name}"
    )


def _optimizer(model: SamPolicyModel) -> torch.optim.AdamW:
    return torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        betas=BETAS,
        eps=EPSILON,
        weight_decay=WEIGHT_DECAY,
    )


def _resolved_configuration(
    config: SamPolicyTrainingConfig,
    bundle: SamPolicyDatasetBundle,
) -> dict[str, object]:
    try:
        training_source = resolve_source_identity()
    except SamMinerError as error:
        raise SamPolicyTrainingError(
            "training source identity could not be resolved"
        ) from error
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
        "model": {
            **asdict(config.model),
            "schema_version": MODEL_SCHEMA_VERSION,
            "parameter_count": PARAMETER_COUNT,
            "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
            "action_schema_version": ACTION_SCHEMA_VERSION,
            "representative_mask_schema_version": (
                REPRESENTATIVE_MASK_SCHEMA_VERSION
            ),
        },
        "optimizer": {
            "compatibility_version": OPTIMIZER_COMPATIBILITY_VERSION,
            "name": "AdamW",
            "learning_rate": LEARNING_RATE,
            "betas": list(BETAS),
            "epsilon": EPSILON,
            "weight_decay": WEIGHT_DECAY,
            "batch_size": BATCH_SIZE,
            "global_gradient_clip": GRADIENT_CLIP_NORM,
            **asdict(config.optimization),
        },
        "source": {
            "training": {
                "schema_version": training_source.schema_version,
                "revision": training_source.revision,
                "tree_digest": training_source.tree_digest,
            },
            "corpus": {
                "schema_version": SAM_MINER_SOURCE_TREE_SCHEMA_VERSION,
                "revision": bundle.snapshot.source_revision,
                "tree_digest": bundle.snapshot.source_tree_digest,
            },
        },
    }
    return {
        **content,
        "model_contract_digest": _json_digest(content["model"]),
        "optimizer_config_digest": _json_digest(content["optimizer"]),
        "resolved_config_digest": _json_digest(content),
    }


def _seal_resolved_config(
    output: Path, resolved: dict[str, object]
) -> None:
    path = output / "resolved-config.json"
    if path.exists():
        if _load_json(path) != resolved:
            raise SamPolicyTrainingError(
                "resolved training configuration differs"
            )
        return
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


def _digest_nested(value: object) -> str:
    digest = hashlib.sha256()

    def update(item: object) -> None:
        if isinstance(item, Tensor):
            tensor = item.detach().cpu().contiguous()
            digest.update(b"tensor\0")
            digest.update(str(tensor.dtype).encode("ascii") + b"\0")
            digest.update(
                ",".join(map(str, tensor.shape)).encode("ascii") + b"\0"
            )
            digest.update(tensor.numpy().tobytes(order="C"))
        elif isinstance(item, Mapping):
            digest.update(b"mapping\0")
            for key in sorted(item, key=lambda value: str(value)):
                update(str(key))
                update(item[key])
        elif isinstance(item, (list, tuple)):
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
    bundle: SamPolicyDatasetBundle,
    model: SamPolicyModel,
    optimizer: torch.optim.AdamW,
    epoch: int,
    next_batch: int,
    maximum_gradient_norm: float,
    completed_epochs: int,
    best_epoch: int,
    best_validation_loss: float,
    stale_epochs: int,
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
        "dataset_digest": bundle.dataset_digest,
        "split_digest": bundle.split_digest,
        "snapshot_digest": bundle.snapshot.snapshot_digest,
        "model_contract_digest": resolved["model_contract_digest"],
        "optimizer_config_digest": resolved["optimizer_config_digest"],
        "source_revision": resolved["source"]["training"]["revision"],
        "source_tree_digest": resolved["source"]["training"]["tree_digest"],
        "corpus_source_revision": bundle.snapshot.source_revision,
        "corpus_source_tree_digest": bundle.snapshot.source_tree_digest,
        "state_dict_digest": _state_dict_digest(model_state),
        "optimizer_state_digest": _digest_nested(optimizer_state),
        "training_state_schema_version": TRAINING_STATE_SCHEMA_VERSION,
        "epoch": epoch,
        "next_batch": next_batch,
        "maximum_gradient_norm": maximum_gradient_norm,
        "completed_epochs": completed_epochs,
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
        "stale_epochs": stale_epochs,
        "history": list(history),
        "model_state_dict": model_state,
        "optimizer_state_dict": optimizer_state,
    }


def _save_checkpoint(path: Path, payload: dict[str, object]) -> None:
    _atomic_torch(path, payload)


def _load_checkpoint(
    path: Path,
    *,
    resolved: dict[str, object],
    bundle: SamPolicyDatasetBundle,
    model: SamPolicyModel,
    optimizer: torch.optim.AdamW,
) -> dict[str, object]:
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, ValueError) as error:
        raise SamPolicyTrainingError(
            "training checkpoint could not be loaded"
        ) from error
    required = {
        "format_version",
        "kind",
        "resolved_config_digest",
        "dataset_digest",
        "split_digest",
        "snapshot_digest",
        "model_contract_digest",
        "optimizer_config_digest",
        "source_revision",
        "source_tree_digest",
        "corpus_source_revision",
        "corpus_source_tree_digest",
        "state_dict_digest",
        "optimizer_state_digest",
        "training_state_schema_version",
        "epoch",
        "next_batch",
        "maximum_gradient_norm",
        "completed_epochs",
        "best_epoch",
        "best_validation_loss",
        "stale_epochs",
        "history",
        "model_state_dict",
        "optimizer_state_dict",
    }
    if (
        not isinstance(value, dict)
        or set(value) != required
        or value["format_version"] != CHECKPOINT_FORMAT_VERSION
        or value["resolved_config_digest"]
        != resolved["resolved_config_digest"]
        or value["dataset_digest"] != bundle.dataset_digest
        or value["split_digest"] != bundle.split_digest
        or value["snapshot_digest"] != bundle.snapshot.snapshot_digest
        or value["model_contract_digest"]
        != resolved["model_contract_digest"]
        or value["optimizer_config_digest"]
        != resolved["optimizer_config_digest"]
        or value["source_revision"]
        != resolved["source"]["training"]["revision"]
        or value["source_tree_digest"]
        != resolved["source"]["training"]["tree_digest"]
        or value["corpus_source_revision"]
        != bundle.snapshot.source_revision
        or value["corpus_source_tree_digest"]
        != bundle.snapshot.source_tree_digest
        or value["training_state_schema_version"]
        != TRAINING_STATE_SCHEMA_VERSION
        or not isinstance(value["model_state_dict"], dict)
        or not isinstance(value["optimizer_state_dict"], dict)
    ):
        raise SamPolicyTrainingError(
            "training checkpoint identity differs"
        )
    if (
        _state_dict_digest(value["model_state_dict"])
        != value["state_dict_digest"]
        or _digest_nested(value["optimizer_state_dict"])
        != value["optimizer_state_digest"]
    ):
        raise SamPolicyTrainingError(
            "training checkpoint state digest differs"
        )
    try:
        model.load_state_dict(value["model_state_dict"], strict=True)
        optimizer.load_state_dict(value["optimizer_state_dict"])
    except (RuntimeError, ValueError) as error:
        raise SamPolicyTrainingError(
            "training checkpoint state is incompatible"
        ) from error
    return value


def _epoch_indexes(
    dataset: SamPolicyDataset,
    *,
    root_seed: str,
    snapshot_digest: str,
    epoch: int,
) -> Tensor:
    digest = derive_seed(
        EPOCH_SHUFFLE_NAMESPACE,
        root_seed,
        snapshot_digest,
        dataset.split_digest,
        str(epoch),
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(
        int.from_bytes(digest[:8], "big", signed=False)
    )
    return torch.randperm(
        dataset.example_count, generator=generator, dtype=torch.long
    )


def _batches(indexes: Tensor) -> tuple[Tensor, ...]:
    return tuple(
        indexes[start : start + BATCH_SIZE]
        for start in range(0, len(indexes), BATCH_SIZE)
    )


def _finite_parameters(model: SamPolicyModel) -> bool:
    return all(
        parameter.dtype is torch.float32
        and bool(torch.isfinite(parameter).all().item())
        for parameter in model.parameters()
    )


def _masked_losses_and_hits(
    logits: Tensor, masks: Tensor, targets: Tensor
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    if (
        logits.dtype is not torch.float32
        or logits.shape != masks.shape
        or masks.dtype is not torch.bool
        or targets.dtype is not torch.long
        or not bool(torch.isfinite(logits).all().item())
    ):
        raise SamPolicyTrainingError(
            "model output or training batch is malformed"
        )
    flattened_masks = masks.flatten(start_dim=1)
    if not bool(
        flattened_masks.gather(1, targets[:, None]).all().item()
    ):
        raise SamPolicyTrainingError(
            "training target is absent from its representative mask"
        )
    flattened = apply_representative_mask(logits, masks).flatten(start_dim=1)
    losses = F.cross_entropy(flattened, targets, reduction="none")
    if not bool(torch.isfinite(losses).all().item()):
        raise SamPolicyTrainingError("training loss is non-finite")
    ranks = torch.argsort(flattened, dim=1, descending=True, stable=True)
    return (
        losses,
        ranks[:, :1].eq(targets[:, None]).any(dim=1),
        ranks[:, :2].eq(targets[:, None]).any(dim=1),
        ranks[:, :3].eq(targets[:, None]).any(dim=1),
    )


def _peak_rss_bytes() -> int:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(usage if sys.platform == "darwin" else usage * 1024)


def _swap_used_bytes() -> int | None:
    """Read system swap without making training depend on platform metrics."""

    if sys.platform == "darwin":
        try:
            completed = subprocess.run(
                ["/usr/sbin/sysctl", "-n", "vm.swapusage"],
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        match = re.search(r"\bused = ([0-9.]+)([KMG])\b", completed.stdout)
        if match is None:
            return None
        scale = {"K": 1024, "M": 1024**2, "G": 1024**3}[match.group(2)]
        return int(float(match.group(1)) * scale)
    try:
        fields = {}
        for line in Path("/proc/meminfo").read_text(
            encoding="utf-8"
        ).splitlines():
            name, raw = line.split(":", 1)
            fields[name] = int(raw.strip().split()[0]) * 1024
        return fields["SwapTotal"] - fields["SwapFree"]
    except (KeyError, OSError, ValueError):
        return None


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _progress_event(event: str, **fields: object) -> None:
    print(
        json.dumps(
            {
                "event": event,
                "time_unix": time.time(),
                **fields,
            },
            allow_nan=False,
            sort_keys=True,
        ),
        flush=True,
    )


def _synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()


def evaluate_model(
    model: SamPolicyModel,
    dataset: SamPolicyDataset,
    *,
    device: torch.device,
) -> EvaluationMetrics:
    model.eval()
    total = _MetricAccumulator()
    placements = {
        str(value): _MetricAccumulator() for value in range(1, 8)
    }
    roles = {
        EnginePlayer.QUEEN.value: _MetricAccumulator(),
        EnginePlayer.KING.value: _MetricAccumulator(),
    }
    dealer_status = {
        "dealer": _MetricAccumulator(),
        "non-dealer": _MetricAccumulator(),
    }
    representative_count = 0
    inference_seconds = 0.0
    with torch.no_grad():
        for indexes in _batches(
            torch.arange(dataset.example_count, dtype=torch.long)
        ):
            observations, masks, targets = dataset.decoded_batch(
                indexes, device=device
            )
            _synchronize(device)
            started = time.perf_counter()
            logits = model(observations)
            _synchronize(device)
            inference_seconds += time.perf_counter() - started
            losses, top1, top2, top3 = _masked_losses_and_hits(
                logits, masks, targets
            )
            losses = losses.cpu()
            top1, top2, top3 = top1.cpu(), top2.cpu(), top3.cpu()
            total.add(losses, top1, top2, top3)
            representative_count += int(masks.sum().item())
            batch_placements = dataset.placements.index_select(0, indexes)
            batch_players = dataset.players.index_select(0, indexes)
            batch_dealers = dataset.dealers.index_select(0, indexes)
            for placement in range(1, 8):
                selected = torch.nonzero(
                    batch_placements == placement, as_tuple=False
                ).flatten()
                placements[str(placement)].add(
                    losses, top1, top2, top3, selected
                )
            for player_index, label in enumerate(
                (EnginePlayer.QUEEN.value, EnginePlayer.KING.value)
            ):
                selected = torch.nonzero(
                    batch_players == player_index, as_tuple=False
                ).flatten()
                roles[label].add(losses, top1, top2, top3, selected)
            dealer_selected = torch.nonzero(
                batch_players == batch_dealers, as_tuple=False
            ).flatten()
            nondealer_selected = torch.nonzero(
                batch_players != batch_dealers, as_tuple=False
            ).flatten()
            dealer_status["dealer"].add(
                losses, top1, top2, top3, dealer_selected
            )
            dealer_status["non-dealer"].add(
                losses, top1, top2, top3, nondealer_selected
            )
    return EvaluationMetrics(
        total=total.finish(),
        placements={
            key: accumulator.finish()
            for key, accumulator in placements.items()
        },
        roles={
            key: accumulator.finish() for key, accumulator in roles.items()
        },
        dealer_status={
            key: accumulator.finish()
            for key, accumulator in dealer_status.items()
        },
        mean_representative_actions=(
            representative_count / dataset.example_count
        ),
        inference_latency_seconds=(
            inference_seconds / dataset.example_count
        ),
    )


def _metrics_dict(metrics: EpochMetrics) -> dict[str, object]:
    return asdict(metrics)


def _load_config_from_resolved(path: Path) -> SamPolicyTrainingConfig:
    value = _load_json(path / "resolved-config.json")
    if not isinstance(value, dict):
        raise SamPolicyTrainingError(
            "resolved training configuration is malformed"
        )
    try:
        return SamPolicyTrainingConfig(
            RunSection(**value["run"]),  # type: ignore[arg-type]
            DatasetSection(
                snapshot_path=value["dataset"]["snapshot_path"]  # type: ignore[index]
            ),
            ModelSection(
                model_id=value["model"]["model_id"],  # type: ignore[index]
                initialization_ordinal=value["model"][  # type: ignore[index]
                    "initialization_ordinal"
                ],
            ),
            OptimizationSection(
                device=value["optimizer"]["device"],  # type: ignore[index]
                minimum_epochs=value["optimizer"][  # type: ignore[index]
                    "minimum_epochs"
                ],
                maximum_epochs=value["optimizer"][  # type: ignore[index]
                    "maximum_epochs"
                ],
                early_stop_patience=value["optimizer"][  # type: ignore[index]
                    "early_stop_patience"
                ],
                minimum_improvement=value["optimizer"][  # type: ignore[index]
                    "minimum_improvement"
                ],
            ),
        )
    except (KeyError, TypeError) as error:
        raise SamPolicyTrainingError(
            "resolved training configuration is malformed"
        ) from error


def _markdown_report(
    *,
    resolved: dict[str, object],
    bundle: SamPolicyDatasetBundle,
    history: Sequence[dict[str, object]],
    best_epoch: int,
    best_validation_loss: float,
    stopped_early: bool,
) -> str:
    lines = [
        "# Standalone Sam policy training",
        "",
        f"- Snapshot: `{bundle.snapshot.snapshot_digest}`",
        f"- Dataset: `{bundle.dataset_digest}`",
        f"- Training examples: {bundle.training.example_count:,}",
        f"- Validation examples: {bundle.validation.example_count:,}",
        f"- Model parameters: {PARAMETER_COUNT:,}",
        f"- Best epoch: {best_epoch}",
        f"- Best validation cross-entropy: {best_validation_loss:.8f}",
        f"- Early stop: {'yes' if stopped_early else 'no'}",
        "",
        "## Epochs",
        "",
        "| Epoch | Train CE | Validation CE | Top-1 | Top-2 | Top-3 |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for entry in history:
        training = entry["training"]["total"]  # type: ignore[index]
        validation = entry["validation"]["total"]  # type: ignore[index]
        lines.append(
            f"| {entry['epoch']} | {training['cross_entropy']:.6f} "
            f"| {validation['cross_entropy']:.6f} "
            f"| {validation['top_one']:.4f} "
            f"| {validation['top_two']:.4f} "
            f"| {validation['top_three']:.4f} |"
        )
    lines.extend(
        [
            "",
            "The run contains policy-classification metrics only. "
            "No value target or value metric is present.",
            "",
            f"Resolved configuration: `{resolved['resolved_config_digest']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def train_sam_policy(
    config: SamPolicyTrainingConfig,
    *,
    resume: bool = False,
    interrupt_after_batches: int | None = None,
) -> TrainingResult:
    _validate_training_config(config)
    run_started = time.perf_counter()
    initial_swap = _swap_used_bytes()
    _progress_event(
        "dataset_load_started",
        run_id=config.run.run_id,
        snapshot=str(config.snapshot_path),
    )
    bundle = load_snapshot_dataset(config.snapshot_path)
    _progress_event(
        "dataset_loaded",
        snapshot_digest=bundle.snapshot.snapshot_digest,
        dataset_digest=bundle.dataset_digest,
        training_examples=bundle.training.example_count,
        validation_examples=bundle.validation.example_count,
        elapsed_seconds=time.perf_counter() - run_started,
    )
    resolved = _resolved_configuration(config, bundle)
    output = config.output_path
    output.mkdir(parents=True, exist_ok=True)
    _seal_resolved_config(output, resolved)
    snapshot_copy = output / "corpus-snapshot.json"
    source_snapshot = _load_json(bundle.snapshot.path)
    if snapshot_copy.exists():
        if _load_json(snapshot_copy) != source_snapshot:
            raise SamPolicyTrainingError(
                "run snapshot copy differs"
            )
    else:
        _atomic_json(snapshot_copy, source_snapshot)

    device = _device(config.optimization.device)
    model = SamPolicyModel(
        run_root_seed=config.run.root_seed,
        model_id=config.model.model_id,
        initialization_ordinal=config.model.initialization_ordinal,
    ).to(device)
    optimizer = _optimizer(model)
    checkpoints = output / "checkpoints"
    initial_path = checkpoints / "initial.pt"
    latest_path = checkpoints / "latest.pt"
    best_path = checkpoints / "best-validation.pt"
    final_path = checkpoints / "final.pt"
    history: list[dict[str, object]] = []
    epoch = 0
    next_batch = 0
    maximum_gradient_norm = 0.0
    completed_epochs = 0
    best_epoch = -1
    best_validation_loss = math.inf
    stale_epochs = 0

    if resume:
        checkpoint = _load_checkpoint(
            latest_path,
            resolved=resolved,
            bundle=bundle,
            model=model,
            optimizer=optimizer,
        )
        epoch = int(checkpoint["epoch"])
        next_batch = int(checkpoint["next_batch"])
        maximum_gradient_norm = float(
            checkpoint["maximum_gradient_norm"]
        )
        completed_epochs = int(checkpoint["completed_epochs"])
        best_epoch = int(checkpoint["best_epoch"])
        best_validation_loss = float(
            checkpoint["best_validation_loss"]
        )
        stale_epochs = int(checkpoint["stale_epochs"])
        history = list(checkpoint["history"])  # type: ignore[arg-type]
        for state in optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, Tensor):
                    state[key] = value.to(device)
    else:
        if latest_path.exists() or final_path.exists():
            raise SamPolicyTrainingError(
                "training run already contains checkpoints"
            )
        initial = _checkpoint_payload(
            kind="initial",
            resolved=resolved,
            bundle=bundle,
            model=model,
            optimizer=optimizer,
            epoch=0,
            next_batch=0,
            maximum_gradient_norm=0.0,
            completed_epochs=0,
            best_epoch=-1,
            best_validation_loss=math.inf,
            stale_epochs=0,
            history=(),
        )
        _save_checkpoint(initial_path, initial)
        _save_checkpoint(latest_path, {**initial, "kind": "latest"})

    untrained_model = SamPolicyModel(
        run_root_seed=config.run.root_seed,
        model_id=config.model.model_id,
        initialization_ordinal=config.model.initialization_ordinal,
    ).to(device)
    untrained_validation = evaluate_model(
        untrained_model, bundle.validation, device=device
    )
    del untrained_model
    _progress_event(
        "untrained_validation_completed",
        cross_entropy=untrained_validation.total.cross_entropy,
        top_one=untrained_validation.total.top_one,
        top_two=untrained_validation.total.top_two,
        top_three=untrained_validation.total.top_three,
    )

    processed_batches = 0
    stopped_early = False
    while epoch < config.optimization.maximum_epochs:
        started = time.perf_counter()
        order = _epoch_indexes(
            bundle.training,
            root_seed=config.run.root_seed,
            snapshot_digest=bundle.snapshot.snapshot_digest,
            epoch=epoch,
        )
        batches = _batches(order)
        if not 0 <= next_batch <= len(batches):
            raise SamPolicyTrainingError(
                "checkpoint minibatch position is invalid"
            )
        model.train()
        for batch_index in range(next_batch, len(batches)):
            observations, masks, targets = (
                bundle.training.decoded_batch(
                    batches[batch_index], device=device
                )
            )
            optimizer.zero_grad(set_to_none=True)
            logits = model(observations)
            losses, _, _, _ = _masked_losses_and_hits(
                logits, masks, targets
            )
            loss = losses.mean()
            if not bool(torch.isfinite(loss).item()):
                raise SamPolicyTrainingError(
                    "training loss is non-finite"
                )
            loss.backward()
            gradients = [
                parameter.grad
                for parameter in model.parameters()
                if parameter.grad is not None
            ]
            if not gradients or not all(
                bool(torch.isfinite(gradient).all().item())
                for gradient in gradients
            ):
                raise SamPolicyTrainingError(
                    "training gradient is absent or non-finite"
                )
            norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), GRADIENT_CLIP_NORM
            )
            if not bool(torch.isfinite(norm).item()):
                raise SamPolicyTrainingError(
                    "gradient norm is non-finite"
                )
            maximum_gradient_norm = max(
                maximum_gradient_norm, float(norm.item())
            )
            optimizer.step()
            if not _finite_parameters(model):
                raise SamPolicyTrainingError(
                    "optimizer produced non-finite model parameters"
                )
            next_batch = batch_index + 1
            processed_batches += 1
            payload = None
            if (
                next_batch % LATEST_CHECKPOINT_INTERVAL == 0
                or interrupt_after_batches is not None
                and processed_batches >= interrupt_after_batches
            ):
                payload = _checkpoint_payload(
                    kind="latest",
                    resolved=resolved,
                    bundle=bundle,
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    next_batch=next_batch,
                    maximum_gradient_norm=maximum_gradient_norm,
                    completed_epochs=completed_epochs,
                    best_epoch=best_epoch,
                    best_validation_loss=best_validation_loss,
                    stale_epochs=stale_epochs,
                    history=history,
                )
                _save_checkpoint(latest_path, payload)
            if (
                interrupt_after_batches is not None
                and processed_batches >= interrupt_after_batches
            ):
                raise SamPolicyTrainingInterrupted(
                    "training interrupted at a sealed minibatch boundary"
                )

        training_metrics = evaluate_model(
            model, bundle.training, device=device
        )
        validation_metrics = evaluate_model(
            model, bundle.validation, device=device
        )
        completed_epochs = epoch + 1
        elapsed = time.perf_counter() - started
        epoch_metrics = EpochMetrics(
            METRICS_FORMAT_VERSION,
            epoch + 1,
            training_metrics,
            validation_metrics,
            maximum_gradient_norm,
            elapsed,
            bundle.training.example_count / max(elapsed, 1e-12),
            _peak_rss_bytes(),
        )
        metrics_value = _metrics_dict(epoch_metrics)
        history.append(metrics_value)
        _atomic_json(
            output / "metrics" / f"{epoch + 1:04d}.json",
            metrics_value,
        )
        validation_loss = validation_metrics.total.cross_entropy
        improved = (
            validation_loss
            < best_validation_loss
            - config.optimization.minimum_improvement
        )
        if improved:
            best_validation_loss = validation_loss
            best_epoch = epoch + 1
            stale_epochs = 0
        else:
            stale_epochs += 1
        epoch += 1
        next_batch = 0
        maximum_gradient_norm = 0.0
        payload = _checkpoint_payload(
            kind="latest",
            resolved=resolved,
            bundle=bundle,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            next_batch=0,
            maximum_gradient_norm=0.0,
            completed_epochs=completed_epochs,
            best_epoch=best_epoch,
            best_validation_loss=best_validation_loss,
            stale_epochs=stale_epochs,
            history=history,
        )
        _save_checkpoint(latest_path, payload)
        if improved:
            _save_checkpoint(
                best_path, {**payload, "kind": "best-validation"}
            )
        _progress_event(
            "epoch_completed",
            epoch=epoch,
            training_cross_entropy=(
                training_metrics.total.cross_entropy
            ),
            validation_cross_entropy=validation_loss,
            validation_top_one=validation_metrics.total.top_one,
            validation_top_two=validation_metrics.total.top_two,
            validation_top_three=validation_metrics.total.top_three,
            maximum_gradient_norm=(
                epoch_metrics.maximum_gradient_norm
            ),
            epoch_seconds=elapsed,
            examples_per_second=epoch_metrics.examples_per_second,
            best_epoch=best_epoch,
            best_validation_cross_entropy=best_validation_loss,
            stale_epochs=stale_epochs,
            improved=improved,
        )
        if (
            completed_epochs >= config.optimization.minimum_epochs
            and stale_epochs >= config.optimization.early_stop_patience
        ):
            stopped_early = True
            break

    final_payload = _checkpoint_payload(
        kind="final",
        resolved=resolved,
        bundle=bundle,
        model=model,
        optimizer=optimizer,
        epoch=epoch,
        next_batch=0,
        maximum_gradient_norm=0.0,
        completed_epochs=completed_epochs,
        best_epoch=best_epoch,
        best_validation_loss=best_validation_loss,
        stale_epochs=stale_epochs,
        history=history,
    )
    _save_checkpoint(final_path, final_payload)
    best_model = SamPolicyModel(
        run_root_seed=config.run.root_seed,
        model_id=config.model.model_id,
        initialization_ordinal=config.model.initialization_ordinal,
    )
    best_optimizer = _optimizer(best_model)
    _load_checkpoint(
        best_path,
        resolved=resolved,
        bundle=bundle,
        model=best_model,
        optimizer=best_optimizer,
    )
    selected_validation = evaluate_model(
        best_model, bundle.validation, device=torch.device("cpu")
    )
    repeated_validation = evaluate_model(
        best_model, bundle.validation, device=torch.device("cpu")
    )
    selected_comparable = asdict(selected_validation)
    repeated_comparable = asdict(repeated_validation)
    selected_comparable.pop("inference_latency_seconds")
    repeated_comparable.pop("inference_latency_seconds")
    deterministic_validation_match = (
        selected_comparable == repeated_comparable
    )
    if not deterministic_validation_match:
        raise SamPolicyTrainingError(
            "repeated deterministic validation metrics differ"
        )

    checkpoint_verification: dict[str, object] = {}
    for label, checkpoint_path in (
        ("initial", initial_path),
        ("latest", latest_path),
        ("best-validation", best_path),
        ("final", final_path),
    ):
        verification_model = SamPolicyModel(
            run_root_seed=config.run.root_seed,
            model_id=config.model.model_id,
            initialization_ordinal=config.model.initialization_ordinal,
        )
        verification_optimizer = _optimizer(verification_model)
        checkpoint = _load_checkpoint(
            checkpoint_path,
            resolved=resolved,
            bundle=bundle,
            model=verification_model,
            optimizer=verification_optimizer,
        )
        checkpoint_verification[label] = {
            "path": str(checkpoint_path),
            "file_digest": _file_digest(checkpoint_path),
            "file_bytes": checkpoint_path.stat().st_size,
            "kind": checkpoint["kind"],
            "epoch": checkpoint["epoch"],
            "next_batch": checkpoint["next_batch"],
            "state_dict_digest": checkpoint["state_dict_digest"],
            "optimizer_state_digest": checkpoint[
                "optimizer_state_digest"
            ],
        }

    artifact_path = output / "artifacts" / "policy.pt"
    save_sam_policy_artifact(
        artifact_path,
        best_model,
        source_revision=resolved["source"]["training"]["revision"],
        source_tree_digest=resolved["source"]["training"]["tree_digest"],
        training_configuration=resolved,
        corpus_snapshot_digest=bundle.snapshot.snapshot_digest,
    )
    loaded = load_sam_policy_artifact(artifact_path)
    probe_indexes = torch.arange(
        min(8, bundle.validation.example_count), dtype=torch.long
    )
    probe, _, _ = bundle.validation.decoded_batch(
        probe_indexes, device=torch.device("cpu")
    )
    with torch.no_grad():
        if not torch.equal(best_model(probe), loaded.model(probe)):
            raise SamPolicyTrainingError(
                "exported policy inference differs"
            )
    final_swap = _swap_used_bytes()
    total_seconds = time.perf_counter() - run_started
    summary = {
        "format_version": SUMMARY_FORMAT_VERSION,
        "resolved_config_digest": resolved["resolved_config_digest"],
        "dataset_digest": bundle.dataset_digest,
        "snapshot_digest": bundle.snapshot.snapshot_digest,
        "completed_epochs": completed_epochs,
        "best_epoch": best_epoch,
        "best_validation_cross_entropy": best_validation_loss,
        "stopped_early": stopped_early,
        "history": history,
        "untrained_validation": asdict(untrained_validation),
        "selected_validation": asdict(selected_validation),
        "repeated_validation": asdict(repeated_validation),
        "deterministic_validation_match": (
            deterministic_validation_match
        ),
        "checkpoint_verification": checkpoint_verification,
        "artifact": {
            "path": str(artifact_path),
            "file_digest": _file_digest(artifact_path),
            "file_bytes": artifact_path.stat().st_size,
            "state_dict_digest": loaded.metadata.state_dict_digest,
            "inference_matches_selected_checkpoint": True,
        },
        "runtime": {
            "total_seconds": total_seconds,
            "peak_rss_bytes": _peak_rss_bytes(),
            "initial_swap_used_bytes": initial_swap,
            "final_swap_used_bytes": final_swap,
            "swap_growth_bytes": (
                None
                if initial_swap is None or final_swap is None
                else final_swap - initial_swap
            ),
        },
    }
    _atomic_json(output / "metrics" / "summary.json", summary)
    _atomic_bytes(
        output / "reports" / "summary.md",
        _markdown_report(
            resolved=resolved,
            bundle=bundle,
            history=history,
            best_epoch=best_epoch,
            best_validation_loss=best_validation_loss,
            stopped_early=stopped_early,
        ).encode("utf-8"),
    )
    _progress_event(
        "training_completed",
        completed_epochs=completed_epochs,
        best_epoch=best_epoch,
        best_validation_cross_entropy=best_validation_loss,
        stopped_early=stopped_early,
        artifact=str(artifact_path),
        artifact_state_dict_digest=loaded.metadata.state_dict_digest,
        total_seconds=total_seconds,
    )
    return TrainingResult(
        str(output),
        completed_epochs,
        best_epoch,
        best_validation_loss,
        stopped_early,
        str(final_path),
        str(artifact_path),
    )


def validate_run(output_directory: str | Path) -> EvaluationMetrics:
    output = Path(output_directory).expanduser().resolve()
    config = _load_config_from_resolved(output)
    bundle = load_snapshot_dataset(config.snapshot_path)
    resolved = _resolved_configuration(config, bundle)
    model = SamPolicyModel(
        run_root_seed=config.run.root_seed,
        model_id=config.model.model_id,
        initialization_ordinal=config.model.initialization_ordinal,
    )
    optimizer = _optimizer(model)
    _load_checkpoint(
        output / "checkpoints" / "best-validation.pt",
        resolved=resolved,
        bundle=bundle,
        model=model,
        optimizer=optimizer,
    )
    return evaluate_model(
        model,
        bundle.validation,
        device=torch.device("cpu"),
    )


def export_run(
    output_directory: str | Path,
    destination: str | Path | None = None,
) -> Path:
    output = Path(output_directory).expanduser().resolve()
    source = output / "artifacts" / "policy.pt"
    loaded = load_sam_policy_artifact(source)
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
    verified = load_sam_policy_artifact(target)
    if verified.metadata.state_dict_digest != loaded.metadata.state_dict_digest:
        raise SamPolicyTrainingError(
            "copied export state digest differs"
        )
    return target


def inspect_snapshot(path: str | Path) -> dict[str, object]:
    bundle = load_snapshot_dataset(path)
    return {
        "snapshot_digest": bundle.snapshot.snapshot_digest,
        "dataset_digest": bundle.dataset_digest,
        "split_digest": bundle.split_digest,
        "highest_included_ordinal": (
            bundle.snapshot.highest_included_ordinal
        ),
        "training_decks": len(bundle.training.fixture_ids),
        "validation_decks": len(bundle.validation.fixture_ids),
        "training_examples": bundle.training.example_count,
        "validation_examples": bundle.validation.example_count,
        "training_placement_counts": dict(
            Counter(bundle.training.placements.tolist())
        ),
        "validation_placement_counts": dict(
            Counter(bundle.validation.placements.tolist())
        ),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Standalone Sam-32 policy imitation"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    snapshot = subparsers.add_parser(
        "snapshot", help="seal a committed-corpus prefix"
    )
    snapshot.add_argument("--corpus", required=True)
    snapshot.add_argument("--output", required=True)
    inspect = subparsers.add_parser(
        "inspect", help="verify and inspect a sealed snapshot"
    )
    inspect.add_argument("--snapshot", required=True)
    smoke = subparsers.add_parser(
        "smoke", help="run a bounded two-minibatch training smoke"
    )
    smoke.add_argument("--config", required=True)
    train = subparsers.add_parser("train", help="start policy training")
    train.add_argument("--config", required=True)
    resume = subparsers.add_parser(
        "resume", help="resume from the latest checkpoint"
    )
    resume.add_argument("--run", required=True)
    validate = subparsers.add_parser(
        "validate", help="evaluate the best validation checkpoint"
    )
    validate.add_argument("--run", required=True)
    export = subparsers.add_parser(
        "export", help="verify or copy the selected policy artifact"
    )
    export.add_argument("--run", required=True)
    export.add_argument("--output")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        if arguments.command == "snapshot":
            result: object = asdict(
                create_committed_corpus_snapshot(
                    arguments.corpus, arguments.output
                )
            )
            result["path"] = str(result["path"])  # type: ignore[index]
            result["corpus_directory"] = str(  # type: ignore[index]
                result["corpus_directory"]  # type: ignore[index]
            )
        elif arguments.command == "inspect":
            result = inspect_snapshot(arguments.snapshot)
        elif arguments.command in {"train", "smoke"}:
            config = load_training_config(arguments.config)
            if arguments.command == "smoke":
                config = replace(
                    config,
                    optimization=replace(
                        config.optimization,
                        minimum_epochs=1,
                        maximum_epochs=1,
                        early_stop_patience=1,
                    ),
                )
            result = asdict(
                train_sam_policy(config)
            )
        elif arguments.command == "resume":
            run = Path(arguments.run).expanduser().resolve()
            result = asdict(
                train_sam_policy(
                    _load_config_from_resolved(run), resume=True
                )
            )
        elif arguments.command == "validate":
            result = asdict(validate_run(arguments.run))
        else:
            result = {
                "artifact": str(
                    export_run(arguments.run, arguments.output)
                )
            }
    except SamPolicyTrainingInterrupted as error:
        print(
            json.dumps(
                {"status": "interrupted", "message": str(error)},
                sort_keys=True,
            )
        )
        return 75
    except (
        SamPolicyTrainingError,
        SamPolicyContractError,
        SamMinerError,
    ) as error:
        print(str(error), file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = (
    "BATCH_SIZE",
    "CHECKPOINT_FORMAT_VERSION",
    "CommittedCorpusSnapshot",
    "DATASET_LOADER_SCHEMA_VERSION",
    "DatasetSection",
    "EpochMetrics",
    "EvaluationMetrics",
    "GroupMetrics",
    "OptimizationSection",
    "RESOLVED_CONFIG_FORMAT_VERSION",
    "RunSection",
    "SNAPSHOT_FORMAT_VERSION",
    "SamPolicyDataset",
    "SamPolicyDatasetBundle",
    "SamPolicyTrainingConfig",
    "SamPolicyTrainingError",
    "SamPolicyTrainingInterrupted",
    "TrainingResult",
    "create_committed_corpus_snapshot",
    "evaluate_model",
    "export_run",
    "inspect_snapshot",
    "load_committed_corpus_snapshot",
    "load_snapshot_dataset",
    "load_training_config",
    "main",
    "train_sam_policy",
    "validate_run",
)

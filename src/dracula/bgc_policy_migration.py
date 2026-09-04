"""Verified reader for the sealed 659-bit card-set training corpus."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from dracula.bgc_policy_model import (
    ACTION_COUNT,
    ACTION_SCHEMA_VERSION,
    HAND_CANDIDATE_COUNT,
    OBSERVATION_SCHEMA_VERSION,
    POLICY_POSITION_COUNT,
    OBSERVATION_SIZE,
)
from dracula.bgc_policy_training import (
    BGCPolicyTrainingError,
    OUTER_SIMULATION_BUDGET,
    _assert_no_forbidden_keys,
    _decode_packed,
    _load_json,
    _unpack_bytes,
)

MIGRATION_SCHEMA_VERSION = "dracula-bgc-card-set-migration-v1"
ROW_SCHEMA_VERSION = "dracula-bgc-card-set-row-v1"
GAME_SCHEMA_VERSION = "dracula-bgc-card-set-game-v1"
CORPUS_SCHEMA_VERSION = "dracula-bgc-card-set-corpus-v1"
LOADER_SCHEMA_VERSION = "dracula-bgc-card-set-loader-v1"


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pack(values: tuple[bool, ...]) -> bytes:
    packed = bytearray((len(values) + 7) // 8)
    for index, value in enumerate(values):
        if value:
            packed[index // 8] |= 1 << (7 - index % 8)
    return bytes(packed)


@dataclass(frozen=True, slots=True)
class MigratedPolicyDataset:
    overlay: Literal["training", "validation"]
    observations_packed: Tensor
    engine_masks_packed: Tensor
    representative_masks_packed: Tensor
    visit_counts: Tensor
    selected_targets: Tensor
    placements: Tensor
    players: Tensor
    dealers: Tensor
    game_ordinals: Tensor
    fixture_ids: tuple[str, ...]
    dataset_digest: str
    split_digest: str

    @property
    def example_count(self) -> int:
        return int(self.visit_counts.shape[0])

    def decoded_batch(self, indexes: Tensor, *, device: torch.device):
        shifts = torch.arange(7, -1, -1, dtype=torch.uint8)
        def unpack(source: Tensor, bits: int) -> Tensor:
            packed = source.index_select(0, indexes)
            return (((packed.unsqueeze(-1) >> shifts) & 1).to(torch.bool)).flatten(start_dim=1)[:, :bits]
        observations = unpack(self.observations_packed, OBSERVATION_SIZE).to(device)
        def unpack_masks(source: Tensor) -> Tensor:
            return unpack(source, ACTION_COUNT).to(device).reshape(
                -1, HAND_CANDIDATE_COUNT, POLICY_POSITION_COUNT
            )
        engine = unpack_masks(self.engine_masks_packed)
        representative = unpack_masks(self.representative_masks_packed)
        visits = self.visit_counts.index_select(0, indexes).to(device=device, dtype=torch.float32)
        targets = visits / float(OUTER_SIMULATION_BUDGET)
        selected = self.selected_targets.index_select(0, indexes).to(device=device)
        return observations, engine, representative, targets, selected


@dataclass(frozen=True, slots=True)
class MigratedSnapshotGame:
    overlay: Literal["training", "validation"]


@dataclass(frozen=True, slots=True)
class MigratedSnapshotIdentity:
    snapshot_digest: str
    source_revision: str
    source_tree_digest: str
    games: tuple[MigratedSnapshotGame, ...]


@dataclass(frozen=True, slots=True)
class MigratedPolicyDatasetBundle:
    path: Path
    snapshot: MigratedSnapshotIdentity
    source_snapshot_digest: str
    training: MigratedPolicyDataset
    validation: MigratedPolicyDataset
    dataset_digest: str
    split_digest: str


def _load_migrated_split(root: Path, entries: list[dict[str, object]], overlay: str, manifest_digest: str) -> MigratedPolicyDataset:
    selected_entries = [entry for entry in entries if entry["overlay"] == overlay]
    total_count = sum(int(entry["row_count"]) for entry in selected_entries)
    observations = torch.empty((total_count, 83), dtype=torch.uint8)
    engine_masks = torch.empty((total_count, 4), dtype=torch.uint8)
    representative_masks = torch.empty((total_count, 4), dtype=torch.uint8)
    visits = torch.zeros((total_count, ACTION_COUNT), dtype=torch.uint8)
    selected = torch.empty(total_count, dtype=torch.long)
    placements = torch.empty(total_count, dtype=torch.uint8)
    players = torch.empty(total_count, dtype=torch.uint8)
    dealers = torch.empty(total_count, dtype=torch.uint8)
    ordinals = torch.empty(total_count, dtype=torch.int64)
    fixtures: set[str] = set()
    ordinal_values: set[int] = set()
    row_digest = hashlib.sha256()
    row_index = 0
    for entry in selected_entries:
        path = root / str(entry["path"])
        if _file_digest(path) != entry["file_digest"]:
            raise BGCPolicyTrainingError("migrated game file digest changed")
        document = _load_json(path)
        content = document.get("content")
        if not isinstance(content, dict) or _digest(content) != document.get("content_digest"):
            raise BGCPolicyTrainingError("migrated game content digest changed")
        _assert_no_forbidden_keys(content)
        for raw in content["rows"]:  # type: ignore[index]
            if not isinstance(raw, dict) or raw.get("row_schema_version") != ROW_SCHEMA_VERSION:
                raise BGCPolicyTrainingError("migrated row schema differs")
            observation = _decode_packed(raw["observation_packed"], byte_count=83, bit_count=659, label="observation")
            legal = _decode_packed(raw["legal_mask_packed"], byte_count=4, bit_count=32, label="legal mask")
            groups = raw["strategic_groups"]
            representatives = raw["strategic_group_representatives"]
            group_visits = raw["strategic_group_visits"]
            if not isinstance(groups, list) or not isinstance(representatives, list) or not isinstance(group_visits, list):
                raise BGCPolicyTrainingError("migrated groups are malformed")
            dense = [0] * ACTION_COUNT
            representative_actions = []
            members = set()
            for group, representative, visit_count in zip(groups, representatives, group_visits, strict=True):
                if not isinstance(group, list) or type(representative) is not int or type(visit_count) is not int:
                    raise BGCPolicyTrainingError("migrated group entry is malformed")
                representative_actions.append(representative)
                dense[representative] = visit_count
                members.update(group)
            legal_bits = _unpack_bytes(legal, ACTION_COUNT)
            if members != {index for index, value in enumerate(legal_bits) if value} or sum(dense) != 128:
                raise BGCPolicyTrainingError("migrated legality or visits differ")
            representative_mask = _pack(tuple(index in set(representative_actions) for index in range(ACTION_COUNT)))
            observations[row_index] = torch.tensor(tuple(observation), dtype=torch.uint8)
            engine_masks[row_index] = torch.tensor(tuple(legal), dtype=torch.uint8)
            representative_masks[row_index] = torch.tensor(
                tuple(representative_mask), dtype=torch.uint8
            )
            visits[row_index] = torch.tensor(dense, dtype=torch.uint8)
            selected[row_index] = int(raw["selected_group_representative"])
            placements[row_index] = int(raw["placement_number"])
            players[row_index] = 0 if raw["player"] == "queen" else 1
            dealers[row_index] = 0 if raw["dealer"] == "queen" else 1
            ordinals[row_index] = int(entry["ordinal"])
            fixtures.add(str(entry["fixture_id"]))
            ordinal_values.add(int(entry["ordinal"]))
            row_digest.update(_canonical(raw))
            row_digest.update(b"\0")
            row_index += 1
    if row_index != total_count:
        raise BGCPolicyTrainingError(
            f"migrated {overlay} split row count differs: {row_index} != {total_count}"
        )
    split_digest = _digest({"fixtures": sorted(fixtures), "ordinals": sorted(ordinal_values), "overlay": overlay})
    dataset_digest = _digest({"manifest": manifest_digest, "overlay": overlay, "rows_digest": row_digest.hexdigest(), "split": split_digest})
    return MigratedPolicyDataset(
        overlay,  # type: ignore[arg-type]
        observations,
        engine_masks,
        representative_masks,
        visits,
        selected,
        placements,
        players,
        dealers,
        ordinals,
        tuple(sorted(fixtures)),
        dataset_digest,
        split_digest,
    )


def load_migrated_policy_dataset(path: str | Path) -> MigratedPolicyDatasetBundle:
    root = Path(path).expanduser().resolve()
    if root.is_file():
        root = root.parent
    document = _load_json(root / "corpus-manifest.json")
    content = document.get("content")
    if not isinstance(content, dict) or _digest(content) != document.get("content_digest"):
        raise BGCPolicyTrainingError("migrated corpus manifest digest changed")
    if (
        content.get("corpus_schema_version") != CORPUS_SCHEMA_VERSION
        or content.get("observation_schema_version") != OBSERVATION_SCHEMA_VERSION
        or content.get("action_schema_version") != ACTION_SCHEMA_VERSION
        or type(content.get("game_count")) is not int
        or int(content["game_count"]) < 1
        or content.get("row_count") != int(content["game_count"]) * 42
    ):
        raise BGCPolicyTrainingError("migrated corpus contract differs")
    entries = content.get("games")
    if not isinstance(entries, list) or len(entries) != content["game_count"]:
        raise BGCPolicyTrainingError("migrated corpus games differ")
    training = _load_migrated_split(root, entries, "training", document["content_digest"])
    validation = _load_migrated_split(root, entries, "validation", document["content_digest"])
    source_document = _load_json(Path(str(content["source_snapshot_path"])))
    source_content = source_document.get("content")
    if (
        not isinstance(source_content, dict)
        or source_document.get("content_digest") != content["source_snapshot_digest"]
    ):
        raise BGCPolicyTrainingError("source snapshot identity changed")
    identity = MigratedSnapshotIdentity(
        document["content_digest"],
        str(source_content["source_revision"]),
        str(source_content["source_tree_digest"]),
        tuple(MigratedSnapshotGame(entry["overlay"]) for entry in entries),  # type: ignore[arg-type]
    )
    split_digest = _digest({"training": training.split_digest, "validation": validation.split_digest})
    dataset_digest = _digest({"training": training.dataset_digest, "validation": validation.dataset_digest})
    return MigratedPolicyDatasetBundle(
        root,
        identity,
        content["source_snapshot_digest"],
        training,
        validation,
        dataset_digest,
        split_digest,
    )


__all__ = (
    "CORPUS_SCHEMA_VERSION",
    "MigratedPolicyDataset",
    "MigratedPolicyDatasetBundle",
    "load_migrated_policy_dataset",
)

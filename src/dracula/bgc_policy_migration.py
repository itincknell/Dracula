"""Physical D0 migration from stable hand slots to card-set candidates."""

from __future__ import annotations

import base64
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from torch import Tensor

from dracula.bgc_policy_model import (
    ACTION_COUNT,
    ACTION_SCHEMA_VERSION,
    HAND_CANDIDATE_COUNT,
    OBSERVATION_SCHEMA_VERSION,
    OBSERVATION_SIZE,
    POLICY_POSITION_COUNT,
    compact_action_tensor_from_legacy,
    compact_observation_from_legacy,
)
from dracula.bgc_policy_training import (
    BGCCommittedCorpusSnapshot,
    BGCPolicyTrainingError,
    OUTER_SIMULATION_BUDGET,
    _decode_packed,
    _load_json,
    _unpack_bytes,
    load_bgc_policy_snapshot,
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


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("wb") as handle:
            handle.write(_canonical(value) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _pack(values: tuple[bool, ...]) -> bytes:
    packed = bytearray((len(values) + 7) // 8)
    for index, value in enumerate(values):
        if value:
            packed[index // 8] |= 1 << (7 - index % 8)
    return bytes(packed)


def _tensor_from_bits(value: bytes, bit_count: int) -> Tensor:
    return torch.tensor(_unpack_bytes(value, bit_count), dtype=torch.bool)


def _slot_to_candidate(observation: Tensor) -> dict[int, int]:
    hand = observation[: 4 * 54].reshape(4, 54)
    mapping: dict[int, int] = {}
    candidate = 0
    for slot in range(4):
        if bool(hand[slot].any().item()):
            mapping[slot] = candidate
            candidate += 1
    if not 1 <= candidate <= 4:
        raise BGCPolicyTrainingError("legacy row has no current hand card")
    return mapping


def _map_action(action: int, mapping: dict[int, int]) -> int:
    if type(action) is not int or not 0 <= action < ACTION_COUNT:
        raise BGCPolicyTrainingError("legacy action index is invalid")
    slot, destination = divmod(action, POLICY_POSITION_COUNT)
    if slot not in mapping:
        raise BGCPolicyTrainingError("legacy action names an empty hand slot")
    return mapping[slot] * POLICY_POSITION_COUNT + destination


def _migrate_row(raw: dict[str, object]) -> dict[str, object]:
    legacy_packed = _decode_packed(
        raw["observation_packed"], byte_count=110, bit_count=875, label="observation"
    )
    legacy_observation = _tensor_from_bits(legacy_packed, 875)
    compact_observation = compact_observation_from_legacy(legacy_observation)
    mapping = _slot_to_candidate(legacy_observation)
    legal_packed = _decode_packed(
        raw["legal_mask_packed"], byte_count=4, bit_count=32, label="legal mask"
    )
    legal = _tensor_from_bits(legal_packed, 32).reshape(4, 8)
    compact_legal = compact_action_tensor_from_legacy(legacy_observation, legal)

    raw_groups = raw["strategic_groups"]
    raw_representatives = raw["strategic_group_representatives"]
    raw_visits = raw["strategic_group_visits"]
    if not all(isinstance(value, list) for value in (raw_groups, raw_representatives, raw_visits)):
        raise BGCPolicyTrainingError("legacy strategic groups are malformed")
    translated = []
    for members, representative, visits in zip(
        raw_groups, raw_representatives, raw_visits, strict=True  # type: ignore[arg-type]
    ):
        if not isinstance(members, list) or type(representative) is not int or type(visits) is not int:
            raise BGCPolicyTrainingError("legacy strategic group entry is malformed")
        translated.append(
            (
                _map_action(representative, mapping),
                [_map_action(member, mapping) for member in members],
                visits,
            )
        )
    translated.sort(key=lambda item: item[0])
    if sum(item[2] for item in translated) != OUTER_SIMULATION_BUDGET:
        raise BGCPolicyTrainingError("migrated visits do not sum to 128")
    selected = _map_action(raw["selected_group_representative"], mapping)  # type: ignore[arg-type]
    representatives = [item[0] for item in translated]
    if selected not in representatives:
        raise BGCPolicyTrainingError("migrated selected action is absent")
    legal_actions = {index for index, allowed in enumerate(compact_legal.flatten()) if bool(allowed)}
    group_actions = {member for _, members, _ in translated for member in members}
    if legal_actions != group_actions:
        raise BGCPolicyTrainingError("migrated groups do not partition legal actions")

    migrated = {
        key: raw[key]
        for key in (
            "dealer",
            "fixture_id",
            "information_state_fingerprint",
            "placement_number",
            "player",
            "round_number",
            "search_config_digest",
            "trajectory_profile",
        )
    }
    migrated.update(
        {
            "action_schema_version": ACTION_SCHEMA_VERSION,
            "legal_mask_packed": base64.b64encode(
                _pack(tuple(bool(value) for value in compact_legal.flatten().tolist()))
            ).decode("ascii"),
            "observation_packed": base64.b64encode(
                _pack(tuple(bool(value) for value in compact_observation.tolist()))
            ).decode("ascii"),
            "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
            "row_schema_version": ROW_SCHEMA_VERSION,
            "selected_group_representative": selected,
            "strategic_group_representatives": representatives,
            "strategic_group_visits": [item[2] for item in translated],
            "strategic_groups": [item[1] for item in translated],
        }
    )
    return migrated


def migrate_snapshot(source_snapshot: str | Path, output_directory: str | Path) -> dict[str, object]:
    snapshot = load_bgc_policy_snapshot(source_snapshot)
    output = Path(output_directory).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise BGCPolicyTrainingError("migrated corpus output must be empty")
    output.mkdir(parents=True, exist_ok=True)
    games: list[dict[str, object]] = []
    placement_counts = {overlay: {str(value): 0 for value in range(1, 8)} for overlay in ("training", "validation")}
    try:
        for game in snapshot.games:
            source_path = snapshot.corpus_directory / game.relative_path
            source_document = _load_json(source_path)
            content = source_document.get("content")
            if not isinstance(content, dict) or content.get("ordinal") != game.ordinal:
                raise BGCPolicyTrainingError("source game content differs during migration")
            raw_rows = content.get("rows")
            if not isinstance(raw_rows, list) or len(raw_rows) != 42:
                raise BGCPolicyTrainingError("source game rows differ during migration")
            rows = [_migrate_row(row) for row in raw_rows if isinstance(row, dict)]
            if len(rows) != 42:
                raise BGCPolicyTrainingError("source game contains a malformed row")
            migrated_content = {
                "fixture_id": game.fixture_id,
                "game_schema_version": GAME_SCHEMA_VERSION,
                "ordinal": game.ordinal,
                "overlay": game.overlay,
                "rows": rows,
                "source_content_digest": game.content_digest,
                "trajectory_profile": game.trajectory_profile,
            }
            document = {"content": migrated_content, "content_digest": _digest(migrated_content)}
            relative = f"games/{game.ordinal:06d}.json"
            destination = output / relative
            _atomic_json(destination, document)
            games.append(
                {
                    "content_digest": document["content_digest"],
                    "file_digest": _file_digest(destination),
                    "fixture_id": game.fixture_id,
                    "ordinal": game.ordinal,
                    "overlay": game.overlay,
                    "path": relative,
                    "row_count": 42,
                    "trajectory_profile": game.trajectory_profile,
                }
            )
            for row in rows:
                placement_counts[game.overlay][str(row["placement_number"])] += 1
        content = {
            "action_schema_version": ACTION_SCHEMA_VERSION,
            "corpus_schema_version": CORPUS_SCHEMA_VERSION,
            "game_count": len(games),
            "games": games,
            "loader_schema_version": LOADER_SCHEMA_VERSION,
            "migration_schema_version": MIGRATION_SCHEMA_VERSION,
            "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
            "placement_counts": placement_counts,
            "row_count": len(games) * 42,
            "source_snapshot_digest": snapshot.snapshot_digest,
            "source_snapshot_path": str(snapshot.path),
        }
        manifest = {"content": content, "content_digest": _digest(content)}
        _atomic_json(output / "corpus-manifest.json", manifest)
        return manifest
    except BaseException:
        # An incomplete output is never accepted because it has no root manifest.
        raise


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
    "MIGRATION_SCHEMA_VERSION",
    "MigratedPolicyDataset",
    "MigratedPolicyDatasetBundle",
    "load_migrated_policy_dataset",
    "migrate_snapshot",
)

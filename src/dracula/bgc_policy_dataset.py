"""Read and verify the sealed 659-bit policy training corpus.

This training utility validates sealed deck and row manifests,
decodes player-visible observations, and exposes immutable examples to the
current trainer without accepting earlier row formats.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch

from dracula.bridge import ACTION_COUNT, POLICY_POSITION_COUNT
from dracula.bgc_policy_model import (
    HAND_CANDIDATE_COUNT,
    OBSERVATION_SIZE,
)
from dracula.bgc_policy_data import (
    assert_no_forbidden_keys as _assert_no_forbidden_keys,
    canonical_json as _canonical,
    decode_packed as _decode_packed,
    file_digest as _file_digest,
    json_digest as _digest,
    load_json as _load_json,
    unpack_bytes as _unpack_bytes,
)
from dracula.bgc_policy_training_contracts import (
    BGCPolicyTrainingError,
    OUTER_SIMULATION_BUDGET,
)

DATASET_FORMAT = "dracula-bgc-card-set-corpus-v1"


def _pack(values: tuple[bool, ...]) -> bytes:
    """Pack Boolean fields most-significant-bit first, matching the sealed corpus."""

    packed = bytearray((len(values) + 7) // 8)
    for index, value in enumerate(values):
        if value:
            packed[index // 8] |= 1 << (7 - index % 8)
    return bytes(packed)


@dataclass(frozen=True, slots=True)
class PolicyDataset:
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
        """Decode selected packed rows and transfer model inputs to one device."""

        shifts = torch.arange(7, -1, -1, dtype=torch.uint8)

        def unpack(source: Tensor, bits: int) -> Tensor:
            packed = source.index_select(0, indexes)
            unpacked = ((packed.unsqueeze(-1) >> shifts) & 1).to(torch.bool)
            return unpacked.flatten(start_dim=1)[:, :bits]

        observations = unpack(self.observations_packed, OBSERVATION_SIZE).to(device)

        def unpack_masks(source: Tensor) -> Tensor:
            return unpack(source, ACTION_COUNT).to(device).reshape(
                -1, HAND_CANDIDATE_COUNT, POLICY_POSITION_COUNT
            )

        engine = unpack_masks(self.engine_masks_packed)
        representative = unpack_masks(self.representative_masks_packed)
        visits = self.visit_counts.index_select(0, indexes).to(
            device=device, dtype=torch.float32
        )
        targets = visits / float(OUTER_SIMULATION_BUDGET)
        selected = self.selected_targets.index_select(0, indexes).to(device=device)
        return observations, engine, representative, targets, selected


@dataclass(frozen=True, slots=True)
class SnapshotGame:
    overlay: Literal["training", "validation"]


@dataclass(frozen=True, slots=True)
class SnapshotIdentity:
    snapshot_digest: str
    games: tuple[SnapshotGame, ...]


@dataclass(frozen=True, slots=True)
class PolicyDatasetBundle:
    path: Path
    snapshot: SnapshotIdentity
    training: PolicyDataset
    validation: PolicyDataset
    dataset_digest: str
    split_digest: str


@dataclass(frozen=True, slots=True)
class _DecodedRow:
    """Validated row values in the compact forms stored by the tensor dataset."""

    observation: bytes
    legal_mask: bytes
    representative_mask: bytes
    visits: tuple[int, ...]
    selected_action: int
    placement: int
    player: int
    dealer: int


def _load_game_rows(
    root: Path, entry: dict[str, object], overlay: str
) -> list[object]:
    """Verify one sealed game and return its still-untrusted row documents."""

    path = (root / str(entry["path"])).resolve()
    if not path.is_relative_to(root):
        raise BGCPolicyTrainingError("game path escapes the corpus")
    if _file_digest(path) != entry["file_digest"]:
        raise BGCPolicyTrainingError("game file digest changed")
    document = _load_json(path)
    if not isinstance(document, dict):
        raise BGCPolicyTrainingError("game document is malformed")
    content = document.get("content")
    if (
        not isinstance(content, dict)
        or _digest(content) != document.get("content_digest")
        or document.get("content_digest") != entry.get("content_digest")
        or content.get("fixture_id") != entry.get("fixture_id")
        or content.get("ordinal") != entry.get("ordinal")
        or content.get("overlay") != overlay
        or content.get("trajectory_profile") != entry.get("trajectory_profile")
    ):
        raise BGCPolicyTrainingError("game content digest changed")
    _assert_no_forbidden_keys(content)
    rows = content.get("rows")
    if not isinstance(rows, list) or len(rows) != entry["row_count"]:
        raise BGCPolicyTrainingError("game row count differs")
    return rows


def _project_group_visits(
    raw: dict[str, object], legal_mask: bytes
) -> tuple[tuple[int, ...], bytes, int]:
    """Project strategic-group visits onto their representative action indexes."""

    groups = raw["strategic_groups"]
    representatives = raw["strategic_group_representatives"]
    group_visits = raw["strategic_group_visits"]
    if (
        not isinstance(groups, list)
        or not isinstance(representatives, list)
        or not isinstance(group_visits, list)
        or not groups
        or len(groups) != len(representatives)
        or len(groups) != len(group_visits)
    ):
        raise BGCPolicyTrainingError("strategic groups are malformed")

    dense = [0] * ACTION_COUNT
    representative_actions: set[int] = set()
    members: set[int] = set()
    for group, representative, visit_count in zip(
        groups, representatives, group_visits
    ):
        if (
            not isinstance(group, list)
            or type(representative) is not int
            or type(visit_count) is not int
            or not group
            or any(type(member) is not int for member in group)
            or len(group) != len(set(group))
            or not all(0 <= member < ACTION_COUNT for member in group)
            or representative not in group
            or representative in representative_actions
            or not 0 <= visit_count <= OUTER_SIMULATION_BUDGET
            or members.intersection(group)
        ):
            raise BGCPolicyTrainingError("strategic group entry is malformed")
        representative_actions.add(representative)
        dense[representative] = visit_count
        members.update(group)

    legal_actions = {
        index
        for index, value in enumerate(_unpack_bytes(legal_mask, ACTION_COUNT))
        if value
    }
    if members != legal_actions or sum(dense) != OUTER_SIMULATION_BUDGET:
        raise BGCPolicyTrainingError("legality or visits differ")
    selected_action = raw.get("selected_group_representative")
    if (
        type(selected_action) is not int
        or selected_action not in representative_actions
    ):
        raise BGCPolicyTrainingError("selected group is invalid")
    representative_mask = _pack(
        tuple(index in representative_actions for index in range(ACTION_COUNT))
    )
    return tuple(dense), representative_mask, selected_action


def _decode_row(raw: object, entry: dict[str, object]) -> _DecodedRow:
    """Validate one external row before it enters the trusted tensor dataset."""

    if (
        not isinstance(raw, dict)
        or raw.get("fixture_id") != entry["fixture_id"]
        or raw.get("trajectory_profile") != entry["trajectory_profile"]
        or raw.get("player") not in {"queen", "king"}
        or raw.get("dealer") not in {"queen", "king"}
        or type(raw.get("round_number")) is not int
        or not 1 <= raw["round_number"] <= 6
        or type(raw.get("placement_number")) is not int
        or not 1 <= raw["placement_number"] <= 7
    ):
        raise BGCPolicyTrainingError("dataset row is incompatible")
    observation = _decode_packed(
        raw["observation_packed"],
        byte_count=83,
        bit_count=659,
        label="observation",
    )
    legal_mask = _decode_packed(
        raw["legal_mask_packed"],
        byte_count=4,
        bit_count=32,
        label="legal mask",
    )
    visits, representative_mask, selected_action = _project_group_visits(
        raw, legal_mask
    )
    return _DecodedRow(
        observation=observation,
        legal_mask=legal_mask,
        representative_mask=representative_mask,
        visits=visits,
        selected_action=selected_action,
        placement=raw["placement_number"],
        player=0 if raw["player"] == "queen" else 1,
        dealer=0 if raw["dealer"] == "queen" else 1,
    )


def _load_dataset_split(
    root: Path,
    entries: list[dict[str, object]],
    overlay: str,
    manifest_digest: str,
) -> PolicyDataset:
    """Verify and decode one complete deck-level split from sealed game files."""

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
        for raw in _load_game_rows(root, entry, overlay):
            row = _decode_row(raw, entry)
            observations[row_index] = torch.tensor(
                tuple(row.observation), dtype=torch.uint8
            )
            engine_masks[row_index] = torch.tensor(
                tuple(row.legal_mask), dtype=torch.uint8
            )
            representative_masks[row_index] = torch.tensor(
                tuple(row.representative_mask), dtype=torch.uint8
            )
            visits[row_index] = torch.tensor(row.visits, dtype=torch.uint8)
            selected[row_index] = row.selected_action
            placements[row_index] = row.placement
            players[row_index] = row.player
            dealers[row_index] = row.dealer
            ordinals[row_index] = int(entry["ordinal"])
            fixtures.add(str(entry["fixture_id"]))
            ordinal_values.add(int(entry["ordinal"]))
            # The digest covers the original canonical row, not its in-memory
            # tensor representation, so loader changes cannot alter identity.
            row_digest.update(_canonical(raw))
            row_digest.update(b"\0")
            row_index += 1
    if row_index != total_count:
        raise BGCPolicyTrainingError(
            f"{overlay} split row count differs: {row_index} != {total_count}"
        )
    split_digest = _digest(
        {
            "fixtures": sorted(fixtures),
            "ordinals": sorted(ordinal_values),
            "overlay": overlay,
        }
    )
    dataset_digest = _digest(
        {
            "manifest": manifest_digest,
            "overlay": overlay,
            "rows_digest": row_digest.hexdigest(),
            "split": split_digest,
        }
    )
    return PolicyDataset(
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


def _load_corpus_manifest(
    root: Path,
) -> tuple[dict[str, object], list[dict[str, object]], str]:
    """Verify the root manifest and its canonical, gap-free game index."""

    document = _load_json(root / "corpus-manifest.json")
    if not isinstance(document, dict):
        raise BGCPolicyTrainingError("corpus manifest is malformed")
    content = document.get("content")
    if (
        not isinstance(content, dict)
        or _digest(content) != document.get("content_digest")
    ):
        raise BGCPolicyTrainingError("corpus manifest digest changed")
    if (
        content.get("corpus_schema_version") != DATASET_FORMAT
        or type(content.get("game_count")) is not int
        or int(content["game_count"]) < 1
        or content.get("row_count") != int(content["game_count"]) * 42
    ):
        raise BGCPolicyTrainingError("corpus format differs")
    entries = content.get("games")
    if not isinstance(entries, list) or len(entries) != content["game_count"]:
        raise BGCPolicyTrainingError("corpus game index differs")
    entry_fields = {
        "content_digest",
        "file_digest",
        "fixture_id",
        "ordinal",
        "overlay",
        "path",
        "row_count",
        "trajectory_profile",
    }
    for ordinal, entry in enumerate(entries):
        if (
            not isinstance(entry, dict)
            or set(entry) != entry_fields
            or entry.get("ordinal") != ordinal
            or entry.get("overlay") not in {"training", "validation"}
            or entry.get("row_count") != 42
            or any(
                not isinstance(entry.get(field), str) or not entry[field]
                for field in (
                    "content_digest",
                    "file_digest",
                    "fixture_id",
                    "path",
                    "trajectory_profile",
                )
            )
        ):
            raise BGCPolicyTrainingError("corpus game entry differs")
    return content, entries, document["content_digest"]  # type: ignore[return-value]


def _snapshot_identity(
    entries: list[dict[str, object]],
    snapshot_digest: str,
) -> SnapshotIdentity:
    """Expose the sealed corpus identity and its deck-level split."""

    return SnapshotIdentity(
        snapshot_digest,
        tuple(
            SnapshotGame(entry["overlay"])  # type: ignore[arg-type]
            for entry in entries
        ),
    )


def load_policy_dataset(path: str | Path) -> PolicyDatasetBundle:
    """Load the sealed policy corpus after verifying its manifests and rows."""

    root = Path(path).expanduser().resolve()
    if root.is_file():
        root = root.parent
    content, entries, manifest_digest = _load_corpus_manifest(root)
    training = _load_dataset_split(
        root, entries, "training", manifest_digest
    )
    validation = _load_dataset_split(
        root, entries, "validation", manifest_digest
    )
    identity = _snapshot_identity(entries, manifest_digest)
    split_digest = _digest(
        {"training": training.split_digest, "validation": validation.split_digest}
    )
    dataset_digest = _digest(
        {"training": training.dataset_digest, "validation": validation.dataset_digest}
    )
    return PolicyDatasetBundle(
        root,
        identity,
        training,
        validation,
        dataset_digest,
        split_digest,
    )


__all__ = (
    "DATASET_FORMAT",
    "PolicyDataset",
    "PolicyDatasetBundle",
    "load_policy_dataset",
)

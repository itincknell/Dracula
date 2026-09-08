"""Read and verify the sealed 659-bit policy training corpus.

This training utility validates sealed deck and row manifests,
decodes player-visible observations, and exposes immutable examples to the
current trainer without accepting earlier row formats.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

import torch

from dracula.decision.bridge import ACTION_COUNT, POLICY_POSITION_COUNT
from dracula.policy.model import (
    HAND_CANDIDATE_COUNT,
    OBSERVATION_SIZE,
)
from dracula.policy.training.data import (
    assert_no_forbidden_keys,
    canonical_json,
    file_digest,
    json_digest,
    load_json,
)
from dracula.policy.training.rows import DecodedPolicyRow, decode_policy_row
from dracula.policy.training.contracts import (
    PolicyTrainingError,
    OUTER_SIMULATION_BUDGET,
)

DATASET_FORMAT = "dracula-bgc-card-set-corpus-v1"
DatasetOverlay = Literal["training", "validation"]


@dataclass(frozen=True, slots=True)
class PolicyDataset:
    """One verified split stored as dense metadata and compact packed inputs.

    Rows remain packed in memory because the complete corpus is large. A batch
    decodes only its selected 83-byte observations and two four-byte action
    masks. Visit counts stay as uint8 because each count is at most 128.
    """

    overlay: DatasetOverlay
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
        """Return the number of rows shared by every tensor in this split."""

        return int(self.visit_counts.shape[0])

    def decoded_batch(self, indexes: Tensor, *, device: torch.device):
        """Decode selected rows into model and loss tensors on one device.

        Returns Boolean observations ``[B,659]``, Boolean engine and
        representative masks ``[B,4,8]``, float visit probabilities ``[B,32]``,
        and integer selected-action audit labels ``[B]``.
        """

        shifts = torch.arange(7, -1, -1, dtype=torch.uint8)

        def unpack(source: Tensor, bits: int) -> Tensor:
            """Expand packed rows and discard byte-alignment padding bits."""

            # Each byte expands most-significant bit first. Flattening then
            # slicing removes the five unused padding bits from observations.
            packed = source.index_select(0, indexes)
            unpacked = ((packed.unsqueeze(-1) >> shifts) & 1).to(torch.bool)
            return unpacked.flatten(start_dim=1)[:, :bits]

        observations = unpack(self.observations_packed, OBSERVATION_SIZE).to(device)

        def unpack_masks(source: Tensor) -> Tensor:
            """Restore flattened 32-bit masks to four card-by-position rows."""

            return unpack(source, ACTION_COUNT).to(device).reshape(
                -1, HAND_CANDIDATE_COUNT, POLICY_POSITION_COUNT
            )

        engine = unpack_masks(self.engine_masks_packed)
        representative = unpack_masks(self.representative_masks_packed)
        visits = self.visit_counts.index_select(0, indexes).to(
            device=device, dtype=torch.float32
        )
        # Every verified visit row sums to 128, so division creates a complete
        # probability distribution without renormalizing malformed input.
        targets = visits / float(OUTER_SIMULATION_BUDGET)
        selected = self.selected_targets.index_select(0, indexes).to(device=device)
        return observations, engine, representative, targets, selected


@dataclass(frozen=True, slots=True)
class SnapshotIdentity:
    """Manifest digest and game counts needed by checkpoints and reports."""

    snapshot_digest: str
    training_game_count: int
    validation_game_count: int


@dataclass(frozen=True, slots=True)
class PolicyDatasetBundle:
    """The verified training and validation splits from one sealed snapshot."""

    path: Path
    snapshot: SnapshotIdentity
    training: PolicyDataset
    validation: PolicyDataset
    dataset_digest: str
    split_digest: str


def _load_game_rows(
    root: Path, entry: dict[str, object], overlay: str
) -> list[object]:
    """Verify one sealed game and return its still-untrusted row documents."""

    # Resolve before reading so a manifest cannot escape its corpus directory.
    path = (root / str(entry["path"])).resolve()
    if not path.is_relative_to(root):
        raise PolicyTrainingError("game path escapes the corpus")
    if file_digest(path) != entry["file_digest"]:
        raise PolicyTrainingError("game file digest changed")
    document = load_json(path)
    if not isinstance(document, dict):
        raise PolicyTrainingError("game document is malformed")
    content = document.get("content")
    # The game file, its embedded content digest, and its root-manifest entry
    # must all describe the same sealed fixture.
    if (
        not isinstance(content, dict)
        or json_digest(content) != document.get("content_digest")
        or document.get("content_digest") != entry.get("content_digest")
        or content.get("fixture_id") != entry.get("fixture_id")
        or content.get("ordinal") != entry.get("ordinal")
        or content.get("overlay") != overlay
        or content.get("trajectory_profile") != entry.get("trajectory_profile")
    ):
        raise PolicyTrainingError("game content digest changed")
    assert_no_forbidden_keys(content)
    rows = content.get("rows")
    if not isinstance(rows, list) or len(rows) != entry["row_count"]:
        raise PolicyTrainingError("game row count differs")
    return rows


@dataclass(slots=True)
class _SplitBuilder:
    """Accumulate verified rows into the compact tensors used by training."""

    observations: Tensor
    engine_masks: Tensor
    representative_masks: Tensor
    visits: Tensor
    selected: Tensor
    placements: Tensor
    players: Tensor
    dealers: Tensor
    ordinals: Tensor
    fixtures: set[str] = field(default_factory=set)
    ordinal_values: set[int] = field(default_factory=set)
    row_digest: Any = field(default_factory=hashlib.sha256)
    row_index: int = 0

    @classmethod
    def allocate(cls, row_count: int) -> _SplitBuilder:
        """Allocate fixed-size tensors after manifests establish the row count."""

        return cls(
            observations=torch.empty((row_count, 83), dtype=torch.uint8),
            engine_masks=torch.empty((row_count, 4), dtype=torch.uint8),
            representative_masks=torch.empty((row_count, 4), dtype=torch.uint8),
            visits=torch.zeros((row_count, ACTION_COUNT), dtype=torch.uint8),
            selected=torch.empty(row_count, dtype=torch.long),
            placements=torch.empty(row_count, dtype=torch.uint8),
            players=torch.empty(row_count, dtype=torch.uint8),
            dealers=torch.empty(row_count, dtype=torch.uint8),
            ordinals=torch.empty(row_count, dtype=torch.int64),
        )

    def add(
        self,
        row: DecodedPolicyRow,
        *,
        raw: object,
        entry: dict[str, object],
    ) -> None:
        """Store one trusted row and extend its source-content identity."""

        index = self.row_index
        self.observations[index] = torch.tensor(
            tuple(row.observation), dtype=torch.uint8
        )
        self.engine_masks[index] = torch.tensor(
            tuple(row.legal_mask), dtype=torch.uint8
        )
        self.representative_masks[index] = torch.tensor(
            tuple(row.representative_mask), dtype=torch.uint8
        )
        self.visits[index] = torch.tensor(row.visits, dtype=torch.uint8)
        self.selected[index] = row.selected_action
        self.placements[index] = row.placement
        self.players[index] = row.player
        self.dealers[index] = row.dealer
        self.ordinals[index] = int(entry["ordinal"])
        self.fixtures.add(str(entry["fixture_id"]))
        self.ordinal_values.add(int(entry["ordinal"]))
        # Identity follows the source row bytes, not an implementation-specific
        # tensor layout, so batching changes cannot redefine a sealed dataset.
        self.row_digest.update(canonical_json(raw))
        self.row_digest.update(b"\0")
        self.row_index += 1


def _finish_dataset_split(
    builder: _SplitBuilder,
    *,
    overlay: DatasetOverlay,
    manifest_digest: str,
) -> PolicyDataset:
    """Bind populated split tensors to their fixture and source-row identity."""

    # Split identity depends on game membership, while dataset identity also
    # binds the exact source rows and their root manifest.
    split_digest = json_digest(
        {
            "fixtures": sorted(builder.fixtures),
            "ordinals": sorted(builder.ordinal_values),
            "overlay": overlay,
        }
    )
    dataset_digest = json_digest(
        {
            "manifest": manifest_digest,
            "overlay": overlay,
            "rows_digest": builder.row_digest.hexdigest(),
            "split": split_digest,
        }
    )
    return PolicyDataset(
        overlay,
        builder.observations,
        builder.engine_masks,
        builder.representative_masks,
        builder.visits,
        builder.selected,
        builder.placements,
        builder.players,
        builder.dealers,
        builder.ordinals,
        tuple(sorted(builder.fixtures)),
        dataset_digest,
        split_digest,
    )


def _load_dataset_split(
    root: Path,
    entries: list[dict[str, object]],
    overlay: DatasetOverlay,
    manifest_digest: str,
) -> PolicyDataset:
    """Verify and decode one complete deck-level split from sealed game files."""

    selected_entries = [entry for entry in entries if entry["overlay"] == overlay]
    # Manifests establish the exact allocation size before any row is decoded.
    total_count = sum(int(entry["row_count"]) for entry in selected_entries)
    builder = _SplitBuilder.allocate(total_count)
    for entry in selected_entries:
        for raw in _load_game_rows(root, entry, overlay):
            row = decode_policy_row(
                raw,
                fixture_id=entry["fixture_id"],
                trajectory_profile=entry["trajectory_profile"],
            )
            builder.add(row, raw=raw, entry=entry)
    if builder.row_index != total_count:
        raise PolicyTrainingError(
            f"{overlay} split row count differs: {builder.row_index} != {total_count}"
        )
    return _finish_dataset_split(
        builder,
        overlay=overlay,
        manifest_digest=manifest_digest,
    )


def _validate_manifest_entries(content: dict[str, object]) -> list[dict[str, object]]:
    """Validate the canonical, gap-free game index inside a root manifest."""

    entries = content.get("games")
    if not isinstance(entries, list) or len(entries) != content["game_count"]:
        raise PolicyTrainingError("corpus game index differs")
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
    # List position is the canonical game ordinal; gaps or reordering therefore
    # fail before individual game files are opened.
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
            raise PolicyTrainingError("corpus game entry differs")
    return entries


def _load_corpus_manifest(
    root: Path,
) -> tuple[list[dict[str, object]], str]:
    """Verify the root manifest and return its trusted game index."""

    document = load_json(root / "corpus-manifest.json")
    if not isinstance(document, dict):
        raise PolicyTrainingError("corpus manifest is malformed")
    content = document.get("content")
    if (
        not isinstance(content, dict)
        or json_digest(content) != document.get("content_digest")
    ):
        raise PolicyTrainingError("corpus manifest digest changed")
    if (
        content.get("corpus_schema_version") != DATASET_FORMAT
        or type(content.get("game_count")) is not int
        or int(content["game_count"]) < 1
        or content.get("row_count") != int(content["game_count"]) * 42
    ):
        raise PolicyTrainingError("corpus format differs")
    assert_no_forbidden_keys(content)
    entries = _validate_manifest_entries(content)
    return entries, cast(str, document["content_digest"])


def _snapshot_identity(
    entries: list[dict[str, object]],
    snapshot_digest: str,
) -> SnapshotIdentity:
    """Expose the sealed corpus identity and its deck-level split."""

    return SnapshotIdentity(
        snapshot_digest=snapshot_digest,
        training_game_count=sum(entry["overlay"] == "training" for entry in entries),
        validation_game_count=sum(
            entry["overlay"] == "validation" for entry in entries
        ),
    )


def _validate_split_isolation(
    training: PolicyDataset,
    validation: PolicyDataset,
) -> None:
    """Require nonempty, fixture-disjoint deck-level training overlays."""

    if training.example_count == 0 or validation.example_count == 0:
        raise PolicyTrainingError("training and validation splits must be nonempty")
    if set(training.fixture_ids).intersection(validation.fixture_ids):
        raise PolicyTrainingError("training and validation fixtures overlap")
    training_ordinals = set(training.game_ordinals.tolist())
    validation_ordinals = set(validation.game_ordinals.tolist())
    if training_ordinals.intersection(validation_ordinals):
        raise PolicyTrainingError("training and validation games overlap")


def load_policy_dataset(path: str | Path) -> PolicyDatasetBundle:
    """Load the sealed policy corpus after verifying its manifests and rows."""

    root = Path(path).expanduser().resolve()
    if root.is_file():
        root = root.parent
    entries, manifest_digest = _load_corpus_manifest(root)
    training = _load_dataset_split(
        root, entries, "training", manifest_digest
    )
    validation = _load_dataset_split(
        root, entries, "validation", manifest_digest
    )
    # Whole-game isolation is checked after both independently sealed overlays
    # are decoded; no game or fixture may influence both optimization and selection.
    _validate_split_isolation(training, validation)
    identity = _snapshot_identity(entries, manifest_digest)
    split_digest = json_digest(
        {"training": training.split_digest, "validation": validation.split_digest}
    )
    dataset_digest = json_digest(
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

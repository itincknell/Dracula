"""Deterministic branched corpus mining and sealed Sam-32 artifacts."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import multiprocessing
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import Protocol

from dracula.bridge import (
    ACTION_COUNT,
    PolicyTurnKind,
    build_policy_turn_context,
    move_for_action_index,
)
from dracula.engine import (
    EnginePlayer,
    EngineState,
    EngineStatus,
    advance_after_round,
    apply_move,
    create_game,
    state_fingerprint,
)
from dracula.policy_value import (
    ACTION_SCHEMA_VERSION,
    OBSERVATION_SCHEMA_VERSION,
)
from dracula.randomness import Sha256CounterStream, derive_seed, seed_hex
from dracula.search import (
    DESTINATION_SYMMETRY_SCHEMA_VERSION,
    SAM_TEACHER_SEARCH_SCHEMA_VERSION,
    SearchContractViolation,
    SamTeacherSearchConfig,
    SamTeacherSearchResult,
    SamTeacherInformationSetSearch,
    SearchInterrupted,
    SearchInformationState,
    StrategicActionGroup,
    derive_sam_teacher_destination_seed,
    derive_sam_teacher_request_seed,
    information_state_fingerprint,
    information_state_from_engine,
    policy_input_from_information_state,
    sam_teacher_action_groups,
    select_concrete_action_index,
)

SAM_MINER_DATASET_SCHEMA_VERSION = "dracula-sam-miner-dataset-v2"
SAM_MINER_ROW_SCHEMA_VERSION = "dracula-sam-miner-row-v2"
SAM_MINER_BRANCH_SCHEMA_VERSION = "dracula-sam-miner-branch-v2"
SAM_MINER_CACHE_FORMAT_VERSION = "dracula-sam-miner-cache-v2"
SAM_MINER_SHARD_FORMAT_VERSION = "dracula-sam-miner-subtree-shard-v2"
SAM_MINER_ROUND_MANIFEST_VERSION = "dracula-sam-miner-round-manifest-v2"
SAM_MINER_DECK_MANIFEST_VERSION = "dracula-sam-miner-deck-manifest-v2"
SAM_MINER_CORPUS_MANIFEST_VERSION = "dracula-sam-miner-corpus-manifest-v2"
SAM_MINER_CONFIG_FORMAT_VERSION = "dracula-sam-miner-config-v4"
SAM_MINER_SOURCE_CONFIG_FORMAT_VERSION = "dracula-sam-miner-config-v3"
SAM_MINER_LEGACY_CONFIG_FORMAT_VERSION = "dracula-sam-miner-config-v2"
SAM_MINER_SPLITS_FORMAT_VERSION = "dracula-sam-miner-splits-v2"
SAM_MINER_STATE_FORMAT_VERSION = "dracula-sam-miner-state-v2"
SAM_MINER_SMOKE_FORMAT_VERSION = "dracula-sam-miner-prefix-smoke-v1"
SAM_MINER_SOURCE_TREE_SCHEMA_VERSION = "dracula-source-tree-v1"
SAM_MINER_CONTROLLER_PROFILE = "sam-32-symmetry-nested-uct-v1"
SAM_MINER_STUB_PROFILE = "sam-128-deterministic-test-stub-v1"
SAM_MINER_REDUCED_TEST_PROFILE = "sam-128-reduced-test-budget-v1"

SAM_MINER_DECK_NAMESPACE = "dracula-sam-miner-deck-v1"
SAM_MINER_ALTERNATIVES_NAMESPACE = "dracula-sam-miner-alternatives-v1"
SAM_MINER_DESTINATION_NAMESPACE = "dracula-sam-miner-destination-v1"
SAM_MINER_STUB_NAMESPACE = "dracula-sam-miner-test-stub-v1"

DEFAULT_CHILD_COUNT = 4
DEFAULT_WORKER_COUNT = 1
MAX_WORKER_COUNT = 8
DEFAULT_TEACHER_SIMULATION_BUDGET = 32
DEFAULT_MINIMUM_FREE_DISK_BYTES = 1 << 30
DEFAULT_INFLIGHT_DISK_RESERVE_PER_WORKER_BYTES = 256 << 20
OBSERVATION_BIT_COUNT = 875
LEGAL_MASK_BIT_COUNT = 32
PACKED_OBSERVATION_BYTES = 110
PACKED_LEGAL_MASK_BYTES = 4
LEARNED_PLACEMENTS = 7
ROUNDS_PER_DECK = 6

_DIGEST_LENGTH = hashlib.sha256().digest_size * 2
_FORBIDDEN_KEYS = frozenset(
    {
        "action_values",
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
        "root_visits",
        "round_return",
        "search_tree",
        "stock",
        "stock_order",
        "tree",
        "visits",
    }
)


class SamMinerError(ValueError):
    """Configuration or artifact data violates the Sam-miner contract."""


class SamMinerInterrupted(RuntimeError):
    """Mining stopped without committing a partial subtree."""


class SamMinerDiskFloorReached(SamMinerInterrupted):
    """Collection stopped before consuming its configured free-disk reserve."""


class DeckSplit(StrEnum):
    TRAINING = "training"
    VALIDATION = "validation"
    TEST = "test"


@dataclass(frozen=True, slots=True)
class SamMinerTeacherSelection:
    selected_representative_action_index: int
    selected_concrete_action_index: int


class SamMinerTeacher(Protocol):
    schema_version: str
    configuration_digest: str
    controller_profile: str

    def select(
        self,
        information: SearchInformationState,
        should_stop: Callable[[], bool] | None = None,
    ) -> SamMinerTeacherSelection: ...


@dataclass(frozen=True, slots=True)
class DeterministicSamTeacherStub:
    """Cheap structural stand-in for the production Sam teacher."""

    schema_version: str = SAM_TEACHER_SEARCH_SCHEMA_VERSION
    configuration_digest: str = hashlib.sha256(
        SAM_MINER_STUB_PROFILE.encode("utf-8")
    ).hexdigest()
    controller_profile: str = SAM_MINER_STUB_PROFILE

    def select(
        self,
        information: SearchInformationState,
        should_stop: Callable[[], bool] | None = None,
    ) -> SamMinerTeacherSelection:
        if should_stop is not None and should_stop():
            raise SamMinerInterrupted("teacher stub was interrupted")
        groups = sam_teacher_action_groups(information)
        if not groups:
            raise SamMinerError("teacher stub received no strategic actions")
        seed = derive_seed(
            SAM_MINER_STUB_NAMESPACE,
            information_state_fingerprint(information),
            self.configuration_digest,
        )
        selected = groups[
            Sha256CounterStream(seed).randbelow(len(groups))
        ]
        request_seed = derive_sam_teacher_request_seed(
            information,
            self.configuration_digest,
        )
        concrete = select_concrete_action_index(
            information,
            selected.representative_action_index,
            derive_sam_teacher_destination_seed(
                request_seed,
                "sam-miner-stub-result",
                information,
                selected.representative_action_index,
                0,
            ),
        )
        return SamMinerTeacherSelection(
            selected.representative_action_index,
            concrete,
        )


class Sam128MinerTeacher:
    """Strict adapter from the production nested result to one group label."""

    schema_version = SAM_TEACHER_SEARCH_SCHEMA_VERSION

    def __init__(
        self,
        search_config: SamTeacherSearchConfig = SamTeacherSearchConfig(
            DEFAULT_TEACHER_SIMULATION_BUDGET,
            DEFAULT_TEACHER_SIMULATION_BUDGET,
        ),
        *,
        controller_profile: str = SAM_MINER_CONTROLLER_PROFILE,
    ) -> None:
        if not isinstance(search_config, SamTeacherSearchConfig):
            raise SamMinerError("Sam teacher configuration is invalid")
        if not isinstance(controller_profile, str) or not controller_profile:
            raise SamMinerError("Sam controller profile must be nonempty")
        if (
            controller_profile == SAM_MINER_CONTROLLER_PROFILE
            and search_config
            != SamTeacherSearchConfig(
                DEFAULT_TEACHER_SIMULATION_BUDGET,
                DEFAULT_TEACHER_SIMULATION_BUDGET,
            )
        ):
            raise SamMinerError(
                "production Sam is locked to 32 outer and 32 response "
                "simulations"
            )
        self.search_config = search_config
        self.configuration_digest = search_config.digest
        self.controller_profile = controller_profile
        self._planner = SamTeacherInformationSetSearch(search_config)
        self.query_count = 0

    def select(
        self,
        information: SearchInformationState,
        should_stop: Callable[[], bool] | None = None,
    ) -> SamMinerTeacherSelection:
        if not isinstance(information, SearchInformationState):
            raise SamMinerError(
                "Sam teacher requires a player information state"
            )
        request_seed = derive_sam_teacher_request_seed(
            information,
            self.configuration_digest,
        )
        self.query_count += 1
        try:
            result = self._planner.search(
                information,
                request_seed,
                should_stop,
            )
        except SearchInterrupted as error:
            raise SamMinerInterrupted(str(error)) from error
        except SearchContractViolation as error:
            raise SamMinerError(
                f"Sam search contract failed: {error}"
            ) from error
        return validate_sam128_result(
            information,
            self.search_config,
            result,
        )


def validate_sam128_result(
    information: SearchInformationState,
    search_config: SamTeacherSearchConfig,
    result: SamTeacherSearchResult,
) -> SamMinerTeacherSelection:
    """Reject any result that cannot be sealed as a direct Sam label."""

    if not isinstance(result, SamTeacherSearchResult):
        raise SamMinerError("Sam returned an incompatible result type")
    fingerprint = information_state_fingerprint(information)
    groups = sam_teacher_action_groups(information)
    legal_actions = {
        index
        for index, allowed in enumerate(
            value for row in information.legal_mask for value in row
        )
        if allowed
    }
    if (
        result.information_state_fingerprint != fingerprint
        or result.config_digest != search_config.digest
        or result.simulation_count
        != search_config.outer_simulation_budget
        or result.information_set_count < 1
        or result.principal_continuation is None
    ):
        raise SamMinerError(
            "Sam result identity or outer accounting differs"
        )
    diagnostics = result.group_diagnostics
    if (
        tuple(diagnostic.group for diagnostic in diagnostics) != groups
        or sum(diagnostic.visits for diagnostic in diagnostics)
        != search_config.outer_simulation_budget
    ):
        raise SamMinerError(
            "Sam diagnostics do not match strategic groups or budget"
        )
    for diagnostic in diagnostics:
        if (
            type(diagnostic.visits) is not int
            or diagnostic.visits < 1
            or diagnostic.mean_value is None
            or not math.isfinite(diagnostic.mean_value)
            or not -1.0 <= diagnostic.mean_value <= 1.0
        ):
            raise SamMinerError(
                "Sam diagnostics contain malformed visits or values"
            )
    selected_diagnostic = min(
        diagnostics,
        key=lambda diagnostic: (
            -diagnostic.visits,
            -float(diagnostic.mean_value),
            diagnostic.group.representative_action_index,
        ),
    )
    selected_group = selected_diagnostic.group
    if (
        result.selected_representative_action_index
        != selected_group.representative_action_index
        or result.selected_action_index
        not in selected_group.member_action_indices
        or result.selected_action_index not in legal_actions
    ):
        raise SamMinerError(
            "Sam selected an illegal or non-maximal strategic group"
        )
    diagnostic_members = {
        member
        for diagnostic in diagnostics
        for member in diagnostic.group.member_action_indices
    }
    if diagnostic_members != legal_actions:
        raise SamMinerError(
            "Sam diagnostics do not partition legal actions"
        )
    for diagnostic in diagnostics:
        for member in diagnostic.group.member_action_indices:
            value = result.mean_action_values[member]
            if (
                value is None
                or not math.isfinite(value)
                or value != diagnostic.mean_value
            ):
                raise SamMinerError(
                    "Sam expanded action values are malformed"
                )
        if sum(
            result.action_visits[member]
            for member in diagnostic.group.member_action_indices
        ) != diagnostic.visits:
            raise SamMinerError(
                "Sam expanded action visits are malformed"
            )
    if any(
        result.action_visits[index] != 0
        or result.mean_action_values[index] is not None
        for index in range(ACTION_COUNT)
        if index not in legal_actions
    ):
        raise SamMinerError(
            "Sam diagnostics contain illegal-action evidence"
        )
    return SamMinerTeacherSelection(
        selected_group.representative_action_index,
        result.selected_action_index,
    )


@dataclass(frozen=True, slots=True)
class SamMinerConfig:
    run_id: str
    root_seed: str
    output_directory: str
    training_decks: int = 1
    validation_decks: int = 1
    test_decks: int = 1
    child_count: int = DEFAULT_CHILD_COUNT
    workers: int = DEFAULT_WORKER_COUNT
    teacher_schema_version: str = SAM_TEACHER_SEARCH_SCHEMA_VERSION
    teacher_configuration_digest: str = SamTeacherSearchConfig(
        DEFAULT_TEACHER_SIMULATION_BUDGET,
        DEFAULT_TEACHER_SIMULATION_BUDGET,
    ).digest
    teacher_controller_profile: str = SAM_MINER_CONTROLLER_PROFILE
    teacher_outer_simulation_budget: int = DEFAULT_TEACHER_SIMULATION_BUDGET
    teacher_response_simulation_budget: int = (
        DEFAULT_TEACHER_SIMULATION_BUDGET
    )
    teacher_outer_exploration_constant: float = math.sqrt(2.0)
    teacher_response_exploration_constant: float = math.sqrt(2.0)
    continuous: bool = False
    minimum_free_disk_bytes: int = 0
    inflight_disk_reserve_per_worker_bytes: int = (
        DEFAULT_INFLIGHT_DISK_RESERVE_PER_WORKER_BYTES
    )
    source_tree_schema_version: str | None = None
    source_revision: str | None = None
    source_tree_digest: str | None = None

    def __post_init__(self) -> None:
        for value, label in (
            (self.run_id, "run ID"),
            (self.root_seed, "root seed"),
            (self.output_directory, "output directory"),
            (self.teacher_schema_version, "teacher schema version"),
            (self.teacher_controller_profile, "teacher controller profile"),
        ):
            if not isinstance(value, str) or not value:
                raise SamMinerError(f"{label} must be nonempty")
        _require_digest(
            self.teacher_configuration_digest,
            "teacher configuration digest",
        )
        source_identity = (
            self.source_tree_schema_version,
            self.source_revision,
            self.source_tree_digest,
        )
        if any(value is not None for value in source_identity):
            if not all(value is not None for value in source_identity):
                raise SamMinerError(
                    "source identity fields must be provided together"
                )
            if (
                self.source_tree_schema_version
                != SAM_MINER_SOURCE_TREE_SCHEMA_VERSION
            ):
                raise SamMinerError("source tree schema version differs")
            if (
                not isinstance(self.source_revision, str)
                or len(self.source_revision) not in (40, 64)
                or any(
                    character not in "0123456789abcdef"
                    for character in self.source_revision
                )
            ):
                raise SamMinerError(
                    "source revision must be a lowercase Git object ID"
                )
            _require_digest(self.source_tree_digest, "source tree digest")
        for value, label in (
            (self.training_decks, "training deck count"),
            (self.validation_decks, "validation deck count"),
            (self.test_decks, "test deck count"),
        ):
            if type(value) is not int or value < 0:
                raise SamMinerError(f"{label} must be non-negative")
        if not self.training_decks + self.validation_decks + self.test_decks:
            raise SamMinerError("at least one deck fixture is required")
        if type(self.child_count) is not int or self.child_count < 1:
            raise SamMinerError("child count must be a positive integer")
        if type(self.continuous) is not bool:
            raise SamMinerError("continuous must be boolean")
        if (
            type(self.minimum_free_disk_bytes) is not int
            or self.minimum_free_disk_bytes < 0
        ):
            raise SamMinerError(
                "minimum free disk bytes must be a non-negative integer"
            )
        if (
            type(self.inflight_disk_reserve_per_worker_bytes) is not int
            or self.inflight_disk_reserve_per_worker_bytes < 0
        ):
            raise SamMinerError(
                "in-flight disk reserve must be a non-negative integer"
            )
        if self.continuous and self.minimum_free_disk_bytes < 1:
            raise SamMinerError(
                "continuous collection requires a positive free-disk floor"
            )
        if (
            type(self.workers) is not int
            or not 1 <= self.workers <= MAX_WORKER_COUNT
        ):
            raise SamMinerError(
                f"worker count must be between 1 and {MAX_WORKER_COUNT}"
            )
        expected_config = SamTeacherSearchConfig(
            DEFAULT_TEACHER_SIMULATION_BUDGET,
            DEFAULT_TEACHER_SIMULATION_BUDGET,
        )
        expected = expected_config.digest
        if (
            self.teacher_controller_profile == SAM_MINER_CONTROLLER_PROFILE
            and (
                self.teacher_search_config != expected_config
                or self.teacher_configuration_digest != expected
                or self.teacher_schema_version
                != SAM_TEACHER_SEARCH_SCHEMA_VERSION
            )
        ):
            raise SamMinerError(
                "production Sam configuration must be exact 32x32"
            )

    @property
    def teacher_search_config(self) -> SamTeacherSearchConfig:
        return SamTeacherSearchConfig(
            outer_simulation_budget=self.teacher_outer_simulation_budget,
            response_simulation_budget=self.teacher_response_simulation_budget,
            outer_exploration_constant=(
                self.teacher_outer_exploration_constant
            ),
            response_exploration_constant=(
                self.teacher_response_exploration_constant
            ),
        )

    @property
    def output_path(self) -> Path:
        return Path(self.output_directory).expanduser().resolve()

    @property
    def branch_configuration_digest(self) -> str:
        return _json_digest(
            {
                "branch_schema_version": SAM_MINER_BRANCH_SCHEMA_VERSION,
                "child_count": self.child_count,
                "placement_seven_profile": (
                    "all-two-legal-actions-v1"
                ),
                "teacher_child_concrete_profile": (
                    "sam-128-root-result-fair-coin-v1"
                ),
            }
        )

    @property
    def digest(self) -> str:
        values = _resolved_config(self)
        del values["output_directory"]
        # Scheduling cannot change the identity of the mined examples.
        del values["workers"]
        del values["minimum_free_disk_bytes"]
        del values["inflight_disk_reserve_per_worker_bytes"]
        return _json_digest(values)


@dataclass(frozen=True, slots=True)
class DeckFixture:
    split: DeckSplit
    index: int
    fixture_id: str
    _engine_seed: str


@dataclass(frozen=True, slots=True)
class SamMinerRow:
    dataset_schema_version: str
    dataset_schema_digest: str
    collection_configuration_digest: str
    teacher_schema_version: str
    teacher_schema_digest: str
    teacher_configuration_digest: str
    branch_schema_version: str
    branch_schema_digest: str
    branch_configuration_digest: str
    observation_schema_version: str
    observation_schema_digest: str
    action_schema_version: str
    action_schema_digest: str
    symmetry_schema_version: str
    symmetry_schema_digest: str
    information_state_fingerprint: str
    observation_packed: bytes
    legal_mask_packed: bytes
    strategic_groups: tuple[tuple[int, ...], ...]
    teacher_group_index: int
    teacher_group_representative: int
    round_number: int
    placement_number: int
    player: EnginePlayer
    dealer: EnginePlayer
    deck_fixture_id: str
    branch_path_digest: str

    def __post_init__(self) -> None:
        _validate_row(self)

    def to_dict(self) -> dict[str, object]:
        return {
            "dataset_schema_version": self.dataset_schema_version,
            "dataset_schema_digest": self.dataset_schema_digest,
            "collection_configuration_digest": (
                self.collection_configuration_digest
            ),
            "teacher_schema_version": self.teacher_schema_version,
            "teacher_schema_digest": self.teacher_schema_digest,
            "teacher_configuration_digest": (
                self.teacher_configuration_digest
            ),
            "branch_schema_version": self.branch_schema_version,
            "branch_schema_digest": self.branch_schema_digest,
            "branch_configuration_digest": (
                self.branch_configuration_digest
            ),
            "observation_schema_version": self.observation_schema_version,
            "observation_schema_digest": self.observation_schema_digest,
            "action_schema_version": self.action_schema_version,
            "action_schema_digest": self.action_schema_digest,
            "symmetry_schema_version": self.symmetry_schema_version,
            "symmetry_schema_digest": self.symmetry_schema_digest,
            "information_state_fingerprint": (
                self.information_state_fingerprint
            ),
            "observation_packed": base64.b64encode(
                self.observation_packed
            ).decode("ascii"),
            "legal_mask_packed": base64.b64encode(
                self.legal_mask_packed
            ).decode("ascii"),
            "strategic_groups": [
                list(group) for group in self.strategic_groups
            ],
            "teacher_group_index": self.teacher_group_index,
            "teacher_group_representative": (
                self.teacher_group_representative
            ),
            "round_number": self.round_number,
            "placement_number": self.placement_number,
            "player": self.player.value,
            "dealer": self.dealer.value,
            "deck_fixture_id": self.deck_fixture_id,
            "branch_path_digest": self.branch_path_digest,
        }

    @classmethod
    def from_dict(cls, value: object) -> SamMinerRow:
        if not isinstance(value, dict) or set(value) != _ROW_FIELDS:
            raise SamMinerError("Sam-miner row fields are invalid")
        _assert_private_fields_absent(value)
        try:
            return cls(
                dataset_schema_version=str(
                    value["dataset_schema_version"]
                ),
                dataset_schema_digest=str(value["dataset_schema_digest"]),
                collection_configuration_digest=str(
                    value["collection_configuration_digest"]
                ),
                teacher_schema_version=str(
                    value["teacher_schema_version"]
                ),
                teacher_schema_digest=str(value["teacher_schema_digest"]),
                teacher_configuration_digest=str(
                    value["teacher_configuration_digest"]
                ),
                branch_schema_version=str(
                    value["branch_schema_version"]
                ),
                branch_schema_digest=str(value["branch_schema_digest"]),
                branch_configuration_digest=str(
                    value["branch_configuration_digest"]
                ),
                observation_schema_version=str(
                    value["observation_schema_version"]
                ),
                observation_schema_digest=str(
                    value["observation_schema_digest"]
                ),
                action_schema_version=str(value["action_schema_version"]),
                action_schema_digest=str(value["action_schema_digest"]),
                symmetry_schema_version=str(
                    value["symmetry_schema_version"]
                ),
                symmetry_schema_digest=str(
                    value["symmetry_schema_digest"]
                ),
                information_state_fingerprint=str(
                    value["information_state_fingerprint"]
                ),
                observation_packed=base64.b64decode(
                    str(value["observation_packed"]), validate=True
                ),
                legal_mask_packed=base64.b64decode(
                    str(value["legal_mask_packed"]), validate=True
                ),
                strategic_groups=tuple(
                    tuple(int(member) for member in group)
                    for group in value["strategic_groups"]  # type: ignore[union-attr]
                ),
                teacher_group_index=int(value["teacher_group_index"]),
                teacher_group_representative=int(
                    value["teacher_group_representative"]
                ),
                round_number=int(value["round_number"]),
                placement_number=int(value["placement_number"]),
                player=EnginePlayer(str(value["player"])),
                dealer=EnginePlayer(str(value["dealer"])),
                deck_fixture_id=str(value["deck_fixture_id"]),
                branch_path_digest=str(value["branch_path_digest"]),
            )
        except (
            KeyError,
            TypeError,
            ValueError,
            base64.binascii.Error,
        ) as error:
            raise SamMinerError("Sam-miner row is malformed") from error


@dataclass(frozen=True, slots=True)
class BranchNodeResult:
    rows: tuple[SamMinerRow, ...]
    terminal_leaf_count: int


@dataclass(frozen=True, slots=True)
class RoundTreeResult:
    rows: tuple[SamMinerRow, ...]
    terminal_leaf_count: int
    trunk_terminal_state: EngineState


@dataclass(frozen=True, slots=True)
class DeckTreeResult:
    rows: tuple[SamMinerRow, ...]
    terminal_leaf_count: int
    terminal_state: EngineState


@dataclass(frozen=True, slots=True)
class TeacherCacheResult:
    groups: tuple[tuple[int, ...], ...]
    selected_group_index: int
    selected_representative_action_index: int
    selected_concrete_action_index: int
    cache_key: str
    cache_hit: bool


@dataclass(frozen=True, slots=True)
class ShardSummary:
    relative_path: str
    content_digest: str
    file_digest: str
    row_count: int
    placement_counts: tuple[tuple[int, int], ...]
    terminal_leaf_count: int


@dataclass(frozen=True, slots=True)
class CorpusInspection:
    config_digest: str
    corpus_manifest_digest: str
    deck_count: int
    round_count: int
    shard_count: int
    row_count: int
    terminal_leaf_count: int
    cache_entry_count: int
    disk_bytes: int


@dataclass(frozen=True, slots=True)
class BranchPrefixResult:
    rows: tuple[SamMinerRow, ...]
    frontier_path_digests: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PrefixSmokeInspection:
    content_digest: str
    file_digest: str
    row_count: int
    frontier_count: int
    maximum_placement: int
    elapsed_seconds: float
    relative_path: str


@dataclass(frozen=True, slots=True)
class DeckWorkerResult:
    fixture: DeckFixture
    staging_directory: str
    deck_content_digest: str
    deck_file_digest: str
    torch_thread_count: int
    torch_interop_thread_count: int


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
        raise SamMinerError("value is not canonical JSON") from error


def _json_digest(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


@dataclass(frozen=True, slots=True)
class SamMinerSourceIdentity:
    schema_version: str
    revision: str
    tree_digest: str


def resolve_source_identity() -> SamMinerSourceIdentity:
    """Identify the Git base and exact collection-relevant working tree."""

    repository = Path(__file__).resolve().parents[2]
    try:
        revision = subprocess.run(
            ("git", "rev-parse", "HEAD"),
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise SamMinerError(
            "Sam collection requires a Git source revision"
        ) from error
    if (
        len(revision) not in (40, 64)
        or any(character not in "0123456789abcdef" for character in revision)
    ):
        raise SamMinerError("Git returned an invalid source revision")

    paths = [repository / "NORTHSTARS", repository / "pyproject.toml"]
    paths.extend(sorted((repository / "src").rglob("*.py")))
    paths.extend(sorted((repository / "docs").rglob("*.md")))
    entries: list[dict[str, str]] = []
    for path in paths:
        if not path.is_file():
            raise SamMinerError(
                f"source-tree input is missing: {path.relative_to(repository)}"
            )
        entries.append(
            {
                "path": path.relative_to(repository).as_posix(),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    tree_digest = _json_digest(
        {
            "schema_version": SAM_MINER_SOURCE_TREE_SCHEMA_VERSION,
            "files": entries,
        }
    )
    return SamMinerSourceIdentity(
        SAM_MINER_SOURCE_TREE_SCHEMA_VERSION,
        revision,
        tree_digest,
    )


def _require_digest(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != _DIGEST_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise SamMinerError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _schema_digest(version: str, **shape: object) -> str:
    return _json_digest({"schema_version": version, **shape})


DATASET_SCHEMA_DIGEST = _schema_digest(
    SAM_MINER_DATASET_SCHEMA_VERSION,
    row_schema_version=SAM_MINER_ROW_SCHEMA_VERSION,
)
BRANCH_SCHEMA_DIGEST = _schema_digest(
    SAM_MINER_BRANCH_SCHEMA_VERSION,
    child_zero="teacher",
    alternatives="uniform-without-replacement",
    terminal="round-complete",
)
OBSERVATION_SCHEMA_DIGEST = _schema_digest(
    OBSERVATION_SCHEMA_VERSION,
    dtype="bool",
    bit_count=OBSERVATION_BIT_COUNT,
    packing="most-significant-bit-first",
)
ACTION_SCHEMA_DIGEST = _schema_digest(
    ACTION_SCHEMA_VERSION,
    action_count=ACTION_COUNT,
    group_member_count=(1, 2),
)
SYMMETRY_SCHEMA_DIGEST = _schema_digest(
    DESTINATION_SYMMETRY_SCHEMA_VERSION,
    scope="authoritative-table-only",
)
TEACHER_SCHEMA_DIGEST = _schema_digest(
    SAM_TEACHER_SEARCH_SCHEMA_VERSION,
    controller_profile=SAM_MINER_CONTROLLER_PROFILE,
)


def _contract_bindings(config: SamMinerConfig) -> dict[str, object]:
    return {
        "dataset_schema_version": SAM_MINER_DATASET_SCHEMA_VERSION,
        "dataset_schema_digest": DATASET_SCHEMA_DIGEST,
        "teacher_schema_version": config.teacher_schema_version,
        "teacher_schema_digest": _schema_digest(
            config.teacher_schema_version,
            controller_profile=config.teacher_controller_profile,
        ),
        "teacher_configuration_digest": (
            config.teacher_configuration_digest
        ),
        "teacher_controller_profile": config.teacher_controller_profile,
        "branch_schema_version": SAM_MINER_BRANCH_SCHEMA_VERSION,
        "branch_schema_digest": BRANCH_SCHEMA_DIGEST,
        "branch_configuration_digest": (
            config.branch_configuration_digest
        ),
        "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
        "observation_schema_digest": OBSERVATION_SCHEMA_DIGEST,
        "action_schema_version": ACTION_SCHEMA_VERSION,
        "action_schema_digest": ACTION_SCHEMA_DIGEST,
        "symmetry_schema_version": DESTINATION_SYMMETRY_SCHEMA_VERSION,
        "symmetry_schema_digest": SYMMETRY_SCHEMA_DIGEST,
    }


def _resolved_config(config: SamMinerConfig) -> dict[str, object]:
    value = asdict(config)
    value["output_directory"] = str(config.output_path)
    if config.source_revision is None:
        value.pop("source_tree_schema_version")
        value.pop("source_revision")
        value.pop("source_tree_digest")
        value["format_version"] = SAM_MINER_LEGACY_CONFIG_FORMAT_VERSION
    else:
        value["format_version"] = SAM_MINER_CONFIG_FORMAT_VERSION
    value.update(_contract_bindings(config))
    return value


def _verify_source_identity(
    config: SamMinerConfig, *, required: bool = False
) -> None:
    """Reject execution when sealed collection source no longer matches."""

    if config.source_revision is None:
        if required:
            raise SamMinerError(
                "collection configuration must seal a source revision "
                "and source tree digest"
            )
        return
    current = resolve_source_identity()
    if (
        current.schema_version != config.source_tree_schema_version
        or current.revision != config.source_revision
        or current.tree_digest != config.source_tree_digest
    ):
        raise SamMinerError(
            "current source revision or source tree differs from "
            "the resolved collection configuration"
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
        raise SamMinerError(f"could not read JSON artifact: {path}") from error


def _envelope(format_version: str, content: object) -> dict[str, object]:
    content_digest = _json_digest(content)
    unsigned = {
        "format_version": format_version,
        "content": content,
        "content_digest": content_digest,
    }
    return {**unsigned, "file_digest": _json_digest(unsigned)}


def _validate_envelope(
    value: object,
    format_version: str,
) -> dict[str, object]:
    if (
        not isinstance(value, dict)
        or set(value)
        != {
            "format_version",
            "content",
            "content_digest",
            "file_digest",
        }
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
        raise SamMinerError("artifact envelope digest or format differs")
    if not isinstance(value["content"], dict):
        raise SamMinerError("artifact content must be an object")
    return value


def _write_envelope(
    path: Path,
    format_version: str,
    content: object,
) -> dict[str, object]:
    document = _envelope(format_version, content)
    _atomic_json(path, document)
    return _validate_envelope(_load_json(path), format_version)


def _pack_bools(values: Sequence[bool], expected_count: int) -> bytes:
    if len(values) != expected_count or any(type(value) is not bool for value in values):
        raise SamMinerError(f"expected exactly {expected_count} boolean values")
    packed = bytearray((expected_count + 7) // 8)
    for index, value in enumerate(values):
        if value:
            packed[index // 8] |= 1 << (7 - index % 8)
    return bytes(packed)


def _unpack_bools(value: bytes, bit_count: int) -> tuple[bool, ...]:
    if type(value) is not bytes or len(value) != (bit_count + 7) // 8:
        raise SamMinerError("packed boolean byte count differs")
    if bit_count % 8:
        unused_mask = (1 << (8 - bit_count % 8)) - 1
        if value[-1] & unused_mask:
            raise SamMinerError("unused packed observation bits must be zero")
    return tuple(
        bool(value[index // 8] & (1 << (7 - index % 8)))
        for index in range(bit_count)
    )


def unpack_observation(row: SamMinerRow) -> tuple[bool, ...]:
    return _unpack_bools(row.observation_packed, OBSERVATION_BIT_COUNT)


def unpack_legal_mask(row: SamMinerRow) -> tuple[tuple[bool, ...], ...]:
    values = _unpack_bools(row.legal_mask_packed, LEGAL_MASK_BIT_COUNT)
    return tuple(
        tuple(values[start : start + 8])
        for start in range(0, LEGAL_MASK_BIT_COUNT, 8)
    )


def _groups_as_members(
    groups: Sequence[StrategicActionGroup],
) -> tuple[tuple[int, ...], ...]:
    return tuple(group.member_action_indices for group in groups)


def _validate_group_partition(
    groups: tuple[tuple[int, ...], ...],
    legal_mask: tuple[tuple[bool, ...], ...],
) -> None:
    if not groups:
        raise SamMinerError("row has no strategic groups")
    flattened = [member for group in groups for member in group]
    legal = {
        index
        for index, allowed in enumerate(
            value for mask_row in legal_mask for value in mask_row
        )
        if allowed
    }
    if (
        any(
            not isinstance(group, tuple)
            or len(group) not in (1, 2)
            or tuple(sorted(group)) != group
            for group in groups
        )
        or len(flattened) != len(set(flattened))
        or set(flattened) != legal
        or any(not 0 <= member < ACTION_COUNT for member in flattened)
    ):
        raise SamMinerError(
            "strategic groups do not canonically partition legal actions"
        )


def _make_row(
    information: SearchInformationState,
    groups: tuple[StrategicActionGroup, ...],
    teacher_group_index: int,
    *,
    config: SamMinerConfig,
    fixture_id: str,
    branch_path_digest: str,
) -> SamMinerRow:
    policy_input = policy_input_from_information_state(information)
    observation = tuple(
        bool(value) for value in policy_input.observation.tolist()
    )
    legal_mask = tuple(
        tuple(bool(value) for value in row)
        for row in policy_input.legal_mask.tolist()
    )
    group = groups[teacher_group_index]
    bindings = _contract_bindings(config)
    return SamMinerRow(
        dataset_schema_version=str(bindings["dataset_schema_version"]),
        dataset_schema_digest=str(bindings["dataset_schema_digest"]),
        collection_configuration_digest=config.digest,
        teacher_schema_version=str(bindings["teacher_schema_version"]),
        teacher_schema_digest=str(bindings["teacher_schema_digest"]),
        teacher_configuration_digest=str(
            bindings["teacher_configuration_digest"]
        ),
        branch_schema_version=str(bindings["branch_schema_version"]),
        branch_schema_digest=str(bindings["branch_schema_digest"]),
        branch_configuration_digest=str(
            bindings["branch_configuration_digest"]
        ),
        observation_schema_version=str(
            bindings["observation_schema_version"]
        ),
        observation_schema_digest=str(
            bindings["observation_schema_digest"]
        ),
        action_schema_version=str(bindings["action_schema_version"]),
        action_schema_digest=str(bindings["action_schema_digest"]),
        symmetry_schema_version=str(bindings["symmetry_schema_version"]),
        symmetry_schema_digest=str(bindings["symmetry_schema_digest"]),
        information_state_fingerprint=information_state_fingerprint(
            information
        ),
        observation_packed=_pack_bools(
            observation, OBSERVATION_BIT_COUNT
        ),
        legal_mask_packed=_pack_bools(
            tuple(value for row in legal_mask for value in row),
            LEGAL_MASK_BIT_COUNT,
        ),
        strategic_groups=_groups_as_members(groups),
        teacher_group_index=teacher_group_index,
        teacher_group_representative=(
            group.representative_action_index
        ),
        round_number=information.round_number,
        placement_number=len(information.current_round_moves) + 1,
        player=information.player,
        dealer=information.dealer,
        deck_fixture_id=fixture_id,
        branch_path_digest=branch_path_digest,
    )


def _validate_row(row: SamMinerRow) -> None:
    expected = {
        "dataset_schema_version": SAM_MINER_DATASET_SCHEMA_VERSION,
        "dataset_schema_digest": DATASET_SCHEMA_DIGEST,
        "branch_schema_version": SAM_MINER_BRANCH_SCHEMA_VERSION,
        "branch_schema_digest": BRANCH_SCHEMA_DIGEST,
        "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
        "observation_schema_digest": OBSERVATION_SCHEMA_DIGEST,
        "action_schema_version": ACTION_SCHEMA_VERSION,
        "action_schema_digest": ACTION_SCHEMA_DIGEST,
        "symmetry_schema_version": DESTINATION_SYMMETRY_SCHEMA_VERSION,
        "symmetry_schema_digest": SYMMETRY_SCHEMA_DIGEST,
    }
    for name, value in expected.items():
        if getattr(row, name) != value:
            raise SamMinerError(f"row {name} is incompatible")
    for name in (
        "teacher_schema_digest",
        "teacher_configuration_digest",
        "collection_configuration_digest",
        "branch_configuration_digest",
        "information_state_fingerprint",
        "deck_fixture_id",
        "branch_path_digest",
    ):
        _require_digest(getattr(row, name), name.replace("_", " "))
    if len(row.observation_packed) != PACKED_OBSERVATION_BYTES:
        raise SamMinerError("packed observation must contain 110 bytes")
    if len(row.legal_mask_packed) != PACKED_LEGAL_MASK_BYTES:
        raise SamMinerError("packed legal mask must contain four bytes")
    _unpack_bools(row.observation_packed, OBSERVATION_BIT_COUNT)
    mask = unpack_legal_mask(row)
    _validate_group_partition(row.strategic_groups, mask)
    if (
        type(row.teacher_group_index) is not int
        or not 0 <= row.teacher_group_index < len(row.strategic_groups)
        or row.teacher_group_representative
        not in row.strategic_groups[row.teacher_group_index]
    ):
        raise SamMinerError("teacher group label is invalid")
    if (
        type(row.round_number) is not int
        or not 1 <= row.round_number <= ROUNDS_PER_DECK
        or type(row.placement_number) is not int
        or not 1 <= row.placement_number <= LEARNED_PLACEMENTS
        or not isinstance(row.player, EnginePlayer)
        or not isinstance(row.dealer, EnginePlayer)
    ):
        raise SamMinerError("row lifecycle fields are invalid")


_ROW_FIELDS = frozenset(
    {
        "dataset_schema_version",
        "dataset_schema_digest",
        "collection_configuration_digest",
        "teacher_schema_version",
        "teacher_schema_digest",
        "teacher_configuration_digest",
        "branch_schema_version",
        "branch_schema_digest",
        "branch_configuration_digest",
        "observation_schema_version",
        "observation_schema_digest",
        "action_schema_version",
        "action_schema_digest",
        "symmetry_schema_version",
        "symmetry_schema_digest",
        "information_state_fingerprint",
        "observation_packed",
        "legal_mask_packed",
        "strategic_groups",
        "teacher_group_index",
        "teacher_group_representative",
        "round_number",
        "placement_number",
        "player",
        "dealer",
        "deck_fixture_id",
        "branch_path_digest",
    }
)


def _assert_private_fields_absent(value: object) -> None:
    if isinstance(value, Mapping):
        forbidden = _FORBIDDEN_KEYS.intersection(value)
        if forbidden:
            raise SamMinerError(
                "artifact contains forbidden private fields: "
                + ", ".join(sorted(forbidden))
            )
        for nested in value.values():
            _assert_private_fields_absent(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            _assert_private_fields_absent(nested)


def derive_deck_fixture(
    root_seed: str,
    split: DeckSplit,
    index: int,
) -> DeckFixture:
    split = DeckSplit(split)
    if not isinstance(root_seed, str) or not root_seed:
        raise SamMinerError("root seed must be nonempty")
    if type(index) is not int or index < 0:
        raise SamMinerError("deck index must be non-negative")
    seed = derive_seed(
        SAM_MINER_DECK_NAMESPACE,
        root_seed,
        split.value,
        str(index),
    )
    fixture_id = _json_digest(
        {
            "namespace": SAM_MINER_DECK_NAMESPACE,
            "split": split.value,
            "index": index,
            "seed_digest": seed_hex(seed),
        }
    )
    return DeckFixture(split, index, fixture_id, seed_hex(seed))


def fixture_schedule(config: SamMinerConfig) -> tuple[DeckFixture, ...]:
    if config.continuous:
        raise SamMinerError(
            "continuous collection has no finite fixture schedule"
        )
    return tuple(
        derive_deck_fixture(config.root_seed, split, index)
        for split, count in (
            (DeckSplit.TRAINING, config.training_decks),
            (DeckSplit.VALIDATION, config.validation_decks),
            (DeckSplit.TEST, config.test_decks),
        )
        for index in range(count)
    )


def _split_cycle(config: SamMinerConfig) -> tuple[DeckSplit, ...]:
    return tuple(
        split
        for split, count in (
            (DeckSplit.TRAINING, config.training_decks),
            (DeckSplit.VALIDATION, config.validation_decks),
            (DeckSplit.TEST, config.test_decks),
        )
        for _ in range(count)
    )


def continuous_deck_fixture(
    config: SamMinerConfig, ordinal: int
) -> DeckFixture:
    """Derive one fixture from the infinite, versioned split cycle."""

    if not config.continuous:
        raise SamMinerError("continuous fixture requires continuous mode")
    if type(ordinal) is not int or ordinal < 0:
        raise SamMinerError("continuous deck ordinal must be non-negative")
    cycle = _split_cycle(config)
    position = ordinal % len(cycle)
    cycle_number = ordinal // len(cycle)
    split = cycle[position]
    split_offset = sum(1 for item in cycle[:position] if item is split)
    split_count = sum(1 for item in cycle if item is split)
    return derive_deck_fixture(
        config.root_seed,
        split,
        cycle_number * split_count + split_offset,
    )


def _continuous_fixture_ordinal(
    config: SamMinerConfig, fixture: DeckFixture
) -> int:
    cycle = _split_cycle(config)
    positions = [
        position for position, split in enumerate(cycle) if split is fixture.split
    ]
    split_count = len(positions)
    cycle_number, split_offset = divmod(fixture.index, split_count)
    return cycle_number * len(cycle) + positions[split_offset]


def _root_path_digest(fixture_id: str, round_number: int) -> str:
    return _json_digest(
        {
            "branch_schema_version": SAM_MINER_BRANCH_SCHEMA_VERSION,
            "deck_fixture_id": fixture_id,
            "round_number": round_number,
            "root": True,
        }
    )


def _child_path_digest(
    *,
    round_number: int,
    placement_number: int,
    actor: EnginePlayer,
    parent_path_digest: str,
    child_ordinal: int,
    strategic_representative: int,
    concrete_action_index: int,
) -> str:
    return _json_digest(
        {
            "branch_schema_version": SAM_MINER_BRANCH_SCHEMA_VERSION,
            "round_number": round_number,
            "placement_number": placement_number,
            "actor": actor.value,
            "parent_path_digest": parent_path_digest,
            "child_ordinal": child_ordinal,
            "strategic_representative": strategic_representative,
            "concrete_action_index": concrete_action_index,
        }
    )


def _cache_key(
    information_fingerprint: str,
    config: SamMinerConfig,
) -> str:
    return _json_digest(
        {
            "information_state_fingerprint": information_fingerprint,
            "teacher_schema_version": config.teacher_schema_version,
            "teacher_configuration_digest": (
                config.teacher_configuration_digest
            ),
            "symmetry_schema_version": (
                DESTINATION_SYMMETRY_SCHEMA_VERSION
            ),
            "action_schema_version": ACTION_SCHEMA_VERSION,
        }
    )


def _cache_path(config: SamMinerConfig, key: str) -> Path:
    return config.output_path / "cache" / key[:2] / f"{key}.json"


def _validate_cache_content(
    content: object,
    *,
    config: SamMinerConfig,
    information_fingerprint: str,
    groups: tuple[StrategicActionGroup, ...],
) -> dict[str, object]:
    if not isinstance(content, dict) or set(content) != {
        "cache_key",
        "controller_profile",
        "teacher_schema_version",
        "teacher_configuration_digest",
        "symmetry_schema_version",
        "action_schema_version",
        "information_state_fingerprint",
        "strategic_groups",
        "selected_group_index",
        "selected_representative_action_index",
        "selected_concrete_action_index",
    }:
        raise SamMinerError("teacher cache fields are incompatible")
    _assert_private_fields_absent(content)
    expected_key = _cache_key(information_fingerprint, config)
    expected_groups = [list(group) for group in _groups_as_members(groups)]
    if (
        content["cache_key"] != expected_key
        or content["controller_profile"]
        != config.teacher_controller_profile
        or content["teacher_schema_version"]
        != config.teacher_schema_version
        or content["teacher_configuration_digest"]
        != config.teacher_configuration_digest
        or content["symmetry_schema_version"]
        != DESTINATION_SYMMETRY_SCHEMA_VERSION
        or content["action_schema_version"] != ACTION_SCHEMA_VERSION
        or content["information_state_fingerprint"]
        != information_fingerprint
        or content["strategic_groups"] != expected_groups
    ):
        raise SamMinerError("teacher cache identity or group partition differs")
    index = content["selected_group_index"]
    if (
        type(index) is not int
        or not 0 <= index < len(groups)
        or content["selected_representative_action_index"]
        != groups[index].representative_action_index
        or content["selected_concrete_action_index"]
        not in groups[index].member_action_indices
    ):
        raise SamMinerError("teacher cache selected group is invalid")
    return content


def _validate_cache_artifact(
    path: Path,
    config: SamMinerConfig,
) -> dict[str, object]:
    """Validate a sealed cache entry without reconstructing its engine state."""

    document = _validate_envelope(
        _load_json(path), SAM_MINER_CACHE_FORMAT_VERSION
    )
    content = document["content"]
    if not isinstance(content, dict):
        raise SamMinerError("teacher cache content is invalid")
    fingerprint = _require_digest(
        content.get("information_state_fingerprint"),
        "information-state fingerprint",
    )
    raw_groups = content.get("strategic_groups")
    if (
        not isinstance(raw_groups, list)
        or not raw_groups
        or any(
            not isinstance(group, list)
            or not 1 <= len(group) <= 2
            or any(type(member) is not int for member in group)
            or group != sorted(group)
            or len(group) != len(set(group))
            or any(not 0 <= member < ACTION_COUNT for member in group)
            for group in raw_groups
        )
    ):
        raise SamMinerError("teacher cache strategic groups are invalid")
    groups = tuple(tuple(group) for group in raw_groups)
    flattened = tuple(member for group in groups for member in group)
    if len(flattened) != len(set(flattened)):
        raise SamMinerError("teacher cache strategic groups overlap")
    expected_key = _cache_key(fingerprint, config)
    index = content.get("selected_group_index")
    if (
        set(content)
        != {
            "cache_key",
            "controller_profile",
            "teacher_schema_version",
            "teacher_configuration_digest",
            "symmetry_schema_version",
            "action_schema_version",
            "information_state_fingerprint",
            "strategic_groups",
            "selected_group_index",
            "selected_representative_action_index",
            "selected_concrete_action_index",
        }
        or content.get("cache_key") != expected_key
        or path.stem != expected_key
        or path.parent.name != expected_key[:2]
        or content.get("controller_profile")
        != config.teacher_controller_profile
        or content.get("teacher_schema_version")
        != config.teacher_schema_version
        or content.get("teacher_configuration_digest")
        != config.teacher_configuration_digest
        or content.get("symmetry_schema_version")
        != DESTINATION_SYMMETRY_SCHEMA_VERSION
        or content.get("action_schema_version") != ACTION_SCHEMA_VERSION
        or type(index) is not int
        or not 0 <= index < len(groups)
        or content.get("selected_representative_action_index")
        not in groups[index]
        or content.get("selected_concrete_action_index")
        not in groups[index]
    ):
        raise SamMinerError("teacher cache artifact identity differs")
    _assert_private_fields_absent(content)
    return document


def resolve_teacher(
    information: SearchInformationState,
    groups: tuple[StrategicActionGroup, ...],
    teacher: SamMinerTeacher,
    config: SamMinerConfig,
    should_stop: Callable[[], bool] | None = None,
    *,
    use_cache: bool = True,
) -> TeacherCacheResult:
    if (
        teacher.schema_version != config.teacher_schema_version
        or teacher.configuration_digest
        != config.teacher_configuration_digest
        or teacher.controller_profile != config.teacher_controller_profile
    ):
        raise SamMinerError("injected teacher does not match resolved configuration")
    fingerprint = information_state_fingerprint(information)
    key = _cache_key(fingerprint, config)
    path = _cache_path(config, key)
    if use_cache and path.exists():
        document = _validate_envelope(
            _load_json(path), SAM_MINER_CACHE_FORMAT_VERSION
        )
        content = _validate_cache_content(
            document["content"],
            config=config,
            information_fingerprint=fingerprint,
            groups=groups,
        )
        return TeacherCacheResult(
            groups=_groups_as_members(groups),
            selected_group_index=int(content["selected_group_index"]),
            selected_representative_action_index=int(
                content["selected_representative_action_index"]
            ),
            selected_concrete_action_index=int(
                content["selected_concrete_action_index"]
            ),
            cache_key=key,
            cache_hit=True,
        )

    selection = teacher.select(information, should_stop)
    if not isinstance(selection, SamMinerTeacherSelection):
        raise SamMinerError("teacher returned an incompatible selection")
    try:
        selected_index = next(
            index
            for index, group in enumerate(groups)
            if group.representative_action_index
            == selection.selected_representative_action_index
        )
    except StopIteration as error:
        raise SamMinerError("teacher selected an illegal strategic group") from error
    if (
        selection.selected_concrete_action_index
        not in groups[selected_index].member_action_indices
        or not information.legal_mask[
            selection.selected_concrete_action_index // 8
        ][selection.selected_concrete_action_index % 8]
    ):
        raise SamMinerError("teacher selected an illegal concrete action")
    content = {
        "cache_key": key,
        "controller_profile": config.teacher_controller_profile,
        "teacher_schema_version": config.teacher_schema_version,
        "teacher_configuration_digest": (
            config.teacher_configuration_digest
        ),
        "symmetry_schema_version": DESTINATION_SYMMETRY_SCHEMA_VERSION,
        "action_schema_version": ACTION_SCHEMA_VERSION,
        "information_state_fingerprint": fingerprint,
        "strategic_groups": [
            list(group) for group in _groups_as_members(groups)
        ],
        "selected_group_index": selected_index,
        "selected_representative_action_index": (
            selection.selected_representative_action_index
        ),
        "selected_concrete_action_index": (
            selection.selected_concrete_action_index
        ),
    }
    if use_cache:
        _write_envelope(path, SAM_MINER_CACHE_FORMAT_VERSION, content)
    return TeacherCacheResult(
        groups=_groups_as_members(groups),
        selected_group_index=selected_index,
        selected_representative_action_index=(
            selection.selected_representative_action_index
        ),
        selected_concrete_action_index=(
            selection.selected_concrete_action_index
        ),
        cache_key=key,
        cache_hit=False,
    )


def _alternative_groups(
    groups: tuple[StrategicActionGroup, ...],
    selected_index: int,
    *,
    information: SearchInformationState,
    config: SamMinerConfig,
    fixture_id: str,
    parent_path_digest: str,
) -> tuple[StrategicActionGroup, ...]:
    placement = len(information.current_round_moves) + 1
    available = [
        group for index, group in enumerate(groups) if index != selected_index
    ]
    desired = min(config.child_count - 1, len(available))
    if placement < LEARNED_PLACEMENTS and len(groups) < config.child_count:
        raise SamMinerError(
            "fewer strategic actions than configured before placement seven"
        )
    if placement == LEARNED_PLACEMENTS and len(groups) != 2:
        raise SamMinerError(
            "placement seven must contain exactly two legal strategic actions"
        )
    seed = derive_seed(
        SAM_MINER_ALTERNATIVES_NAMESPACE,
        config.root_seed,
        fixture_id,
        str(information.round_number),
        str(placement),
        parent_path_digest,
        information_state_fingerprint(information),
        str(config.child_count),
    )
    stream = Sha256CounterStream(seed)
    for index in range(desired):
        selected = index + stream.randbelow(len(available) - index)
        available[index], available[selected] = (
            available[selected],
            available[index],
        )
    return tuple(available[:desired])


def _concrete_action(
    information: SearchInformationState,
    representative: int,
    *,
    config: SamMinerConfig,
    parent_path_digest: str,
    child_ordinal: int,
) -> int:
    placement = len(information.current_round_moves) + 1
    seed = derive_seed(
        SAM_MINER_DESTINATION_NAMESPACE,
        config.branch_configuration_digest,
        information_state_fingerprint(information),
        str(representative),
        str(information.round_number),
        str(placement),
        parent_path_digest,
        str(child_ordinal),
    )
    return select_concrete_action_index(
        information,
        representative,
        seed,
        destination_symmetry_enabled=True,
    )


def _apply_action(
    state: EngineState,
    information: SearchInformationState,
    action_index: int,
) -> EngineState:
    context = build_policy_turn_context(state, information.player)
    if not 0 <= action_index < ACTION_COUNT:
        raise SamMinerError("branch action index is out of range")
    move = context.action_table[action_index]
    if move is None or move != move_for_action_index(
        information.player, action_index
    ):
        raise SamMinerError("branch selected a masked action")
    return apply_move(state, move).state


def _forced_continuation(state: EngineState) -> EngineState:
    if state.active_player is None:
        raise SamMinerError("playing state has no active player")
    context = build_policy_turn_context(state, state.active_player)
    if (
        context.kind is not PolicyTurnKind.FORCED_RECURRENT_TRANSITION
        or context.forced_move is None
    ):
        raise SamMinerError("expected the forced eighth placement")
    return apply_move(state, context.forced_move).state


def mine_branch_subtree(
    state: EngineState,
    *,
    fixture_id: str,
    path_digest: str,
    teacher: SamMinerTeacher,
    config: SamMinerConfig,
    should_stop: Callable[[], bool] | None = None,
    use_teacher_cache: bool = True,
) -> BranchNodeResult:
    """Mine one immutable subtree rooted at an arbitrary playing state."""

    if should_stop is not None and should_stop():
        raise SamMinerInterrupted("subtree mining interrupted before seal")
    if state.status is EngineStatus.ROUND_COMPLETE:
        return BranchNodeResult((), 1)
    if state.status is not EngineStatus.PLAYING or state.active_player is None:
        raise SamMinerError("subtree must remain within one playing round")
    context = build_policy_turn_context(state, state.active_player)
    if context.kind is PolicyTurnKind.FORCED_RECURRENT_TRANSITION:
        terminal = _forced_continuation(state)
        if terminal.status is not EngineStatus.ROUND_COMPLETE:
            raise SamMinerError("forced eighth placement did not end the round")
        return BranchNodeResult((), 1)

    information = information_state_from_engine(state)
    groups = sam_teacher_action_groups(information)
    cached = resolve_teacher(
        information,
        groups,
        teacher,
        config,
        should_stop,
        use_cache=use_teacher_cache,
    )
    row = _make_row(
        information,
        groups,
        cached.selected_group_index,
        config=config,
        fixture_id=fixture_id,
        branch_path_digest=path_digest,
    )
    alternatives = _alternative_groups(
        groups,
        cached.selected_group_index,
        information=information,
        config=config,
        fixture_id=fixture_id,
        parent_path_digest=path_digest,
    )
    selected_groups = (
        groups[cached.selected_group_index],
        *alternatives,
    )
    rows: list[SamMinerRow] = [row]
    leaves = 0
    for ordinal, group in enumerate(selected_groups):
        concrete = (
            cached.selected_concrete_action_index
            if ordinal == 0
            else _concrete_action(
                information,
                group.representative_action_index,
                config=config,
                parent_path_digest=path_digest,
                child_ordinal=ordinal,
            )
        )
        child_path = _child_path_digest(
            round_number=information.round_number,
            placement_number=row.placement_number,
            actor=information.player,
            parent_path_digest=path_digest,
            child_ordinal=ordinal,
            strategic_representative=(
                group.representative_action_index
            ),
            concrete_action_index=concrete,
        )
        child_state = _apply_action(state, information, concrete)
        child = mine_branch_subtree(
            child_state,
            fixture_id=fixture_id,
            path_digest=child_path,
            teacher=teacher,
            config=config,
            should_stop=should_stop,
            use_teacher_cache=use_teacher_cache,
        )
        rows.extend(child.rows)
        leaves += child.terminal_leaf_count
    return BranchNodeResult(tuple(rows), leaves)


def mine_branch_prefix(
    state: EngineState,
    *,
    fixture_id: str,
    path_digest: str,
    maximum_placement: int,
    teacher: SamMinerTeacher,
    config: SamMinerConfig,
    should_stop: Callable[[], bool] | None = None,
    use_teacher_cache: bool = True,
) -> BranchPrefixResult:
    """Mine the exact full-run prefix without permitting round completion."""

    if (
        type(maximum_placement) is not int
        or not 1 <= maximum_placement <= 6
    ):
        raise SamMinerError(
            "smoke maximum placement must be between one and six"
        )
    if should_stop is not None and should_stop():
        raise SamMinerInterrupted("prefix mining interrupted before seal")
    if (
        state.status is not EngineStatus.PLAYING
        or state.active_player is None
    ):
        raise SamMinerError("prefix must remain in a playing round")
    placement = len(state.current_round_moves) + 1
    if placement > maximum_placement:
        return BranchPrefixResult((), (path_digest,))
    context = build_policy_turn_context(state, state.active_player)
    if context.kind is PolicyTurnKind.FORCED_RECURRENT_TRANSITION:
        raise SamMinerError("prefix smoke cannot reach a forced placement")

    information = information_state_from_engine(state)
    groups = sam_teacher_action_groups(information)
    cached = resolve_teacher(
        information,
        groups,
        teacher,
        config,
        should_stop,
        use_cache=use_teacher_cache,
    )
    row = _make_row(
        information,
        groups,
        cached.selected_group_index,
        config=config,
        fixture_id=fixture_id,
        branch_path_digest=path_digest,
    )
    alternatives = _alternative_groups(
        groups,
        cached.selected_group_index,
        information=information,
        config=config,
        fixture_id=fixture_id,
        parent_path_digest=path_digest,
    )
    selected_groups = (
        groups[cached.selected_group_index],
        *alternatives,
    )
    rows: list[SamMinerRow] = [row]
    frontier: list[str] = []
    for ordinal, group in enumerate(selected_groups):
        concrete = (
            cached.selected_concrete_action_index
            if ordinal == 0
            else _concrete_action(
                information,
                group.representative_action_index,
                config=config,
                parent_path_digest=path_digest,
                child_ordinal=ordinal,
            )
        )
        child_path = _child_path_digest(
            round_number=information.round_number,
            placement_number=placement,
            actor=information.player,
            parent_path_digest=path_digest,
            child_ordinal=ordinal,
            strategic_representative=(
                group.representative_action_index
            ),
            concrete_action_index=concrete,
        )
        child_state = _apply_action(state, information, concrete)
        if child_state.status is not EngineStatus.PLAYING:
            raise SamMinerError("prefix smoke completed a round")
        if placement == maximum_placement:
            frontier.append(child_path)
            continue
        child = mine_branch_prefix(
            child_state,
            fixture_id=fixture_id,
            path_digest=child_path,
            maximum_placement=maximum_placement,
            teacher=teacher,
            config=config,
            should_stop=should_stop,
            use_teacher_cache=use_teacher_cache,
        )
        rows.extend(child.rows)
        frontier.extend(child.frontier_path_digests)
    return BranchPrefixResult(tuple(rows), tuple(frontier))


def _teacher_children(
    state: EngineState,
    *,
    fixture_id: str,
    path_digest: str,
    teacher: SamMinerTeacher,
    config: SamMinerConfig,
    should_stop: Callable[[], bool] | None = None,
    use_teacher_cache: bool = True,
) -> tuple[
    SearchInformationState,
    tuple[StrategicActionGroup, ...],
    TeacherCacheResult,
    tuple[StrategicActionGroup, ...],
]:
    information = information_state_from_engine(state)
    groups = sam_teacher_action_groups(information)
    cached = resolve_teacher(
        information,
        groups,
        teacher,
        config,
        should_stop,
        use_cache=use_teacher_cache,
    )
    alternatives = _alternative_groups(
        groups,
        cached.selected_group_index,
        information=information,
        config=config,
        fixture_id=fixture_id,
        parent_path_digest=path_digest,
    )
    return (
        information,
        groups,
        cached,
        (groups[cached.selected_group_index], *alternatives),
    )


def play_teacher_trunk(
    state: EngineState,
    *,
    fixture_id: str,
    root_path_digest: str,
    teacher: SamMinerTeacher,
    config: SamMinerConfig,
    should_stop: Callable[[], bool] | None = None,
    use_teacher_cache: bool = True,
) -> EngineState:
    path_digest = root_path_digest
    current = state
    while current.status is EngineStatus.PLAYING:
        if current.active_player is None:
            raise SamMinerError("trunk playing state has no active player")
        context = build_policy_turn_context(
            current, current.active_player
        )
        if context.kind is PolicyTurnKind.FORCED_RECURRENT_TRANSITION:
            current = _forced_continuation(current)
            continue
        information, groups, cached, _ = _teacher_children(
            current,
            fixture_id=fixture_id,
            path_digest=path_digest,
            teacher=teacher,
            config=config,
            should_stop=should_stop,
            use_teacher_cache=use_teacher_cache,
        )
        group = groups[cached.selected_group_index]
        concrete = cached.selected_concrete_action_index
        path_digest = _child_path_digest(
            round_number=information.round_number,
            placement_number=len(information.current_round_moves) + 1,
            actor=information.player,
            parent_path_digest=path_digest,
            child_ordinal=0,
            strategic_representative=(
                group.representative_action_index
            ),
            concrete_action_index=concrete,
        )
        current = _apply_action(current, information, concrete)
    if current.status is not EngineStatus.ROUND_COMPLETE:
        raise SamMinerError("teacher trunk did not finish its round")
    return current


def mine_round_tree(
    state: EngineState,
    *,
    fixture_id: str,
    teacher: SamMinerTeacher,
    config: SamMinerConfig,
    should_stop: Callable[[], bool] | None = None,
    use_teacher_cache: bool = True,
) -> RoundTreeResult:
    if state.status is not EngineStatus.PLAYING:
        raise SamMinerError("round tree must start in a playing round")
    root_path = _root_path_digest(fixture_id, state.round_number)
    result = mine_branch_subtree(
        state,
        fixture_id=fixture_id,
        path_digest=root_path,
        teacher=teacher,
        config=config,
        should_stop=should_stop,
        use_teacher_cache=use_teacher_cache,
    )
    trunk = play_teacher_trunk(
        state,
        fixture_id=fixture_id,
        root_path_digest=root_path,
        teacher=teacher,
        config=config,
        should_stop=should_stop,
        use_teacher_cache=use_teacher_cache,
    )
    return RoundTreeResult(result.rows, result.terminal_leaf_count, trunk)


def mine_deck_tree(
    state: EngineState,
    *,
    fixture_id: str,
    teacher: SamMinerTeacher,
    config: SamMinerConfig,
    should_stop: Callable[[], bool] | None = None,
    use_teacher_cache: bool = True,
) -> DeckTreeResult:
    current = state
    rows: list[SamMinerRow] = []
    leaves = 0
    while current.status is not EngineStatus.GAME_COMPLETE:
        result = mine_round_tree(
            current,
            fixture_id=fixture_id,
            teacher=teacher,
            config=config,
            should_stop=should_stop,
            use_teacher_cache=use_teacher_cache,
        )
        rows.extend(result.rows)
        leaves += result.terminal_leaf_count
        current = advance_after_round(result.trunk_terminal_state)
    return DeckTreeResult(tuple(rows), leaves, current)


def _shard_content(
    *,
    config: SamMinerConfig,
    fixture: DeckFixture,
    round_number: int,
    shard_id: str,
    rows: Sequence[SamMinerRow],
    terminal_leaf_count: int,
) -> dict[str, object]:
    ordered = sorted(
        rows, key=lambda row: (row.placement_number, row.branch_path_digest)
    )
    placement_counts = Counter(row.placement_number for row in ordered)
    content = {
        **_contract_bindings(config),
        "collection_config_digest": config.digest,
        "fixture_id": fixture.fixture_id,
        "fixture_index": fixture.index,
        "split": fixture.split.value,
        "round_number": round_number,
        "shard_id": shard_id,
        "row_count": len(ordered),
        "placement_counts": {
            str(key): placement_counts[key] for key in sorted(placement_counts)
        },
        "terminal_leaf_count": terminal_leaf_count,
        "rows": [row.to_dict() for row in ordered],
    }
    _assert_private_fields_absent(content)
    return content


def _validate_shard_content(
    content: object,
    config: SamMinerConfig | None = None,
) -> dict[str, object]:
    required = {
        *_contract_bindings(
            config
            if config is not None
            else SamMinerConfig(
                "_shape",
                "_shape",
                ".",
                1,
                0,
                0,
                teacher_configuration_digest=(
                    hashlib.sha256(b"_shape").hexdigest()
                ),
                teacher_controller_profile="_shape",
            )
        ),
        "collection_config_digest",
        "fixture_id",
        "fixture_index",
        "split",
        "round_number",
        "shard_id",
        "row_count",
        "placement_counts",
        "terminal_leaf_count",
        "rows",
    }
    if not isinstance(content, dict) or set(content) != required:
        raise SamMinerError("subtree shard fields are invalid")
    _assert_private_fields_absent(content)
    if config is not None:
        expected = _contract_bindings(config)
        if any(content[key] != value for key, value in expected.items()):
            raise SamMinerError("subtree shard contract bindings differ")
        if content["collection_config_digest"] != config.digest:
            raise SamMinerError("subtree shard configuration differs")
    for name in (
        "collection_config_digest",
        "fixture_id",
    ):
        _require_digest(content[name], name.replace("_", " "))
    if content["split"] not in {split.value for split in DeckSplit}:
        raise SamMinerError("subtree shard split is invalid")
    rows_value = content["rows"]
    if not isinstance(rows_value, list):
        raise SamMinerError("subtree shard rows must be a list")
    rows = tuple(SamMinerRow.from_dict(row) for row in rows_value)
    if (
        content["row_count"] != len(rows)
        or rows
        != tuple(
            sorted(
                rows,
                key=lambda row: (
                    row.placement_number,
                    row.branch_path_digest,
                ),
            )
        )
        or any(row.deck_fixture_id != content["fixture_id"] for row in rows)
        or any(row.round_number != content["round_number"] for row in rows)
        or (
            config is not None
            and any(
                row.collection_configuration_digest != config.digest
                or row.teacher_configuration_digest
                != config.teacher_configuration_digest
                or row.branch_configuration_digest
                != config.branch_configuration_digest
                for row in rows
            )
        )
    ):
        raise SamMinerError("subtree shard row ordering or identity differs")
    counts = Counter(row.placement_number for row in rows)
    if content["placement_counts"] != {
        str(key): counts[key] for key in sorted(counts)
    }:
        raise SamMinerError("subtree shard placement counts differ")
    if (
        type(content["terminal_leaf_count"]) is not int
        or content["terminal_leaf_count"] < 0
    ):
        raise SamMinerError("subtree leaf count is invalid")
    return content


def write_subtree_shard(
    path: Path,
    *,
    config: SamMinerConfig,
    fixture: DeckFixture,
    round_number: int,
    shard_id: str,
    rows: Sequence[SamMinerRow],
    terminal_leaf_count: int,
) -> ShardSummary:
    content = _shard_content(
        config=config,
        fixture=fixture,
        round_number=round_number,
        shard_id=shard_id,
        rows=rows,
        terminal_leaf_count=terminal_leaf_count,
    )
    document = _write_envelope(
        path, SAM_MINER_SHARD_FORMAT_VERSION, content
    )
    _validate_shard_content(document["content"], config)
    counts = Counter(row.placement_number for row in rows)
    return ShardSummary(
        relative_path=str(path.relative_to(config.output_path)),
        content_digest=str(document["content_digest"]),
        file_digest=str(document["file_digest"]),
        row_count=len(rows),
        placement_counts=tuple(sorted(counts.items())),
        terminal_leaf_count=terminal_leaf_count,
    )


def load_subtree_shard(
    path: str | Path,
    config: SamMinerConfig | None = None,
) -> dict[str, object]:
    document = _validate_envelope(
        _load_json(Path(path)), SAM_MINER_SHARD_FORMAT_VERSION
    )
    _validate_shard_content(document["content"], config)
    return document


def _deck_directory(
    config: SamMinerConfig,
    fixture: DeckFixture,
) -> Path:
    return (
        config.output_path
        / "decks"
        / fixture.split.value
        / fixture.fixture_id
    )


def _round_directory(
    config: SamMinerConfig,
    fixture: DeckFixture,
    round_number: int,
) -> Path:
    return _deck_directory(config, fixture) / f"round-{round_number:02d}"


def _summary_from_shard(
    path: Path,
    config: SamMinerConfig,
) -> ShardSummary:
    document = load_subtree_shard(path, config)
    content = document["content"]
    assert isinstance(content, dict)
    counts = tuple(
        (int(key), int(value))
        for key, value in content["placement_counts"].items()  # type: ignore[union-attr]
    )
    return ShardSummary(
        relative_path=str(path.relative_to(config.output_path)),
        content_digest=str(document["content_digest"]),
        file_digest=str(document["file_digest"]),
        row_count=int(content["row_count"]),
        placement_counts=counts,
        terminal_leaf_count=int(content["terminal_leaf_count"]),
    )


def _mine_round_artifacts(
    state: EngineState,
    *,
    fixture: DeckFixture,
    teacher: SamMinerTeacher,
    config: SamMinerConfig,
    should_stop: Callable[[], bool] | None = None,
) -> tuple[dict[str, object], EngineState]:
    round_number = state.round_number
    directory = _round_directory(config, fixture, round_number)
    manifest_path = directory / "round-manifest.json"
    root_path_digest = _root_path_digest(
        fixture.fixture_id, round_number
    )
    if manifest_path.exists():
        manifest = _load_round_manifest(
            manifest_path, config, fixture, round_number
        )
        trunk = play_teacher_trunk(
            state,
            fixture_id=fixture.fixture_id,
            root_path_digest=root_path_digest,
            teacher=teacher,
            config=config,
            should_stop=should_stop,
        )
        content = manifest["content"]
        assert isinstance(content, dict)
        if content["trunk_terminal_fingerprint"] != state_fingerprint(trunk):
            raise SamMinerError("sealed round trunk fingerprint differs")
        return manifest, trunk

    if should_stop is not None and should_stop():
        raise SamMinerInterrupted("round mining interrupted before root seal")
    information, groups, cached, selected_groups = _teacher_children(
        state,
        fixture_id=fixture.fixture_id,
        path_digest=root_path_digest,
        teacher=teacher,
        config=config,
        should_stop=should_stop,
    )
    root_row = _make_row(
        information,
        groups,
        cached.selected_group_index,
        config=config,
        fixture_id=fixture.fixture_id,
        branch_path_digest=root_path_digest,
    )
    root_shard = directory / "root.json"
    if root_shard.exists():
        root_summary = _summary_from_shard(root_shard, config)
    else:
        root_summary = write_subtree_shard(
            root_shard,
            config=config,
            fixture=fixture,
            round_number=round_number,
            shard_id="root",
            rows=(root_row,),
            terminal_leaf_count=0,
        )

    subtree_summaries: list[ShardSummary] = []
    for ordinal, group in enumerate(selected_groups):
        if should_stop is not None and should_stop():
            raise SamMinerInterrupted(
                "round mining interrupted at sealed subtree boundary"
            )
        concrete = (
            cached.selected_concrete_action_index
            if ordinal == 0
            else _concrete_action(
                information,
                group.representative_action_index,
                config=config,
                parent_path_digest=root_path_digest,
                child_ordinal=ordinal,
            )
        )
        child_path = _child_path_digest(
            round_number=round_number,
            placement_number=1,
            actor=information.player,
            parent_path_digest=root_path_digest,
            child_ordinal=ordinal,
            strategic_representative=(
                group.representative_action_index
            ),
            concrete_action_index=concrete,
        )
        path = directory / f"subtree-{ordinal:02d}.json"
        if path.exists():
            summary = _summary_from_shard(path, config)
        else:
            child_state = _apply_action(state, information, concrete)
            result = mine_branch_subtree(
                child_state,
                fixture_id=fixture.fixture_id,
                path_digest=child_path,
                teacher=teacher,
                config=config,
                should_stop=should_stop,
            )
            summary = write_subtree_shard(
                path,
                config=config,
                fixture=fixture,
                round_number=round_number,
                shard_id=f"subtree-{ordinal:02d}",
                rows=result.rows,
                terminal_leaf_count=result.terminal_leaf_count,
            )
        subtree_summaries.append(summary)

    trunk = play_teacher_trunk(
        state,
        fixture_id=fixture.fixture_id,
        root_path_digest=root_path_digest,
        teacher=teacher,
        config=config,
        should_stop=should_stop,
    )
    summaries = (root_summary, *subtree_summaries)
    manifest_content = {
        **_contract_bindings(config),
        "collection_config_digest": config.digest,
        "fixture_id": fixture.fixture_id,
        "fixture_index": fixture.index,
        "split": fixture.split.value,
        "round_number": round_number,
        "trunk_terminal_fingerprint": state_fingerprint(trunk),
        "row_count": sum(summary.row_count for summary in summaries),
        "terminal_leaf_count": sum(
            summary.terminal_leaf_count for summary in subtree_summaries
        ),
        "shards": [
            {
                "relative_path": summary.relative_path,
                "content_digest": summary.content_digest,
                "file_digest": summary.file_digest,
                "row_count": summary.row_count,
                "terminal_leaf_count": summary.terminal_leaf_count,
            }
            for summary in sorted(
                summaries, key=lambda item: item.relative_path
            )
        ],
    }
    manifest = _write_envelope(
        manifest_path,
        SAM_MINER_ROUND_MANIFEST_VERSION,
        manifest_content,
    )
    _load_round_manifest(
        manifest_path, config, fixture, round_number
    )
    return manifest, trunk


def _load_round_manifest(
    path: Path,
    config: SamMinerConfig,
    fixture: DeckFixture,
    round_number: int,
) -> dict[str, object]:
    document = _validate_envelope(
        _load_json(path), SAM_MINER_ROUND_MANIFEST_VERSION
    )
    content = document["content"]
    if (
        not isinstance(content, dict)
        or any(
            content.get(key) != value
            for key, value in _contract_bindings(config).items()
        )
        or content.get("collection_config_digest") != config.digest
        or content.get("fixture_id") != fixture.fixture_id
        or content.get("fixture_index") != fixture.index
        or content.get("split") != fixture.split.value
        or content.get("round_number") != round_number
        or not isinstance(content.get("shards"), list)
    ):
        raise SamMinerError("round manifest identity differs")
    _require_digest(
        content.get("trunk_terminal_fingerprint"),
        "trunk terminal fingerprint",
    )
    rows = 0
    leaves = 0
    relative_paths: list[str] = []
    for entry in content["shards"]:
        if not isinstance(entry, dict) or set(entry) != {
            "relative_path",
            "content_digest",
            "file_digest",
            "row_count",
            "terminal_leaf_count",
        }:
            raise SamMinerError("round manifest shard entry is invalid")
        relative = Path(str(entry["relative_path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise SamMinerError("round manifest shard path escapes the corpus")
        shard = load_subtree_shard(config.output_path / relative, config)
        if (
            shard["content_digest"] != entry["content_digest"]
            or shard["file_digest"] != entry["file_digest"]
            or shard["content"]["row_count"] != entry["row_count"]  # type: ignore[index]
            or shard["content"]["terminal_leaf_count"]  # type: ignore[index]
            != entry["terminal_leaf_count"]
        ):
            raise SamMinerError("round manifest shard digest differs")
        relative_paths.append(str(relative))
        rows += int(entry["row_count"])
        leaves += int(entry["terminal_leaf_count"])
    if relative_paths != sorted(relative_paths):
        raise SamMinerError("round manifest shards are not canonical")
    if content.get("row_count") != rows or content.get(
        "terminal_leaf_count"
    ) != leaves:
        raise SamMinerError("round manifest totals differ")
    return document


def _mine_deck_artifacts(
    fixture: DeckFixture,
    *,
    teacher: SamMinerTeacher,
    config: SamMinerConfig,
    should_stop: Callable[[], bool] | None = None,
) -> dict[str, object]:
    directory = _deck_directory(config, fixture)
    manifest_path = directory / "deck-manifest.json"
    if manifest_path.exists():
        return _load_deck_manifest(manifest_path, config, fixture)

    state = create_game(fixture._engine_seed)
    rounds: list[dict[str, object]] = []
    while state.status is not EngineStatus.GAME_COMPLETE:
        document, terminal_round = _mine_round_artifacts(
            state,
            fixture=fixture,
            teacher=teacher,
            config=config,
            should_stop=should_stop,
        )
        rounds.append(document)
        state = advance_after_round(terminal_round)
    entries = [
        {
            "round_number": int(document["content"]["round_number"]),  # type: ignore[index]
            "relative_path": str(
                (
                    directory
                    / f"round-{int(document['content']['round_number']):02d}"  # type: ignore[index]
                    / "round-manifest.json"
                ).relative_to(config.output_path)
            ),
            "content_digest": str(document["content_digest"]),
            "file_digest": str(document["file_digest"]),
            "row_count": int(document["content"]["row_count"]),  # type: ignore[index]
            "terminal_leaf_count": int(
                document["content"]["terminal_leaf_count"]  # type: ignore[index]
            ),
        }
        for document in rounds
    ]
    content = {
        **_contract_bindings(config),
        "collection_config_digest": config.digest,
        "fixture_id": fixture.fixture_id,
        "fixture_index": fixture.index,
        "split": fixture.split.value,
        "round_count": len(entries),
        "row_count": sum(entry["row_count"] for entry in entries),
        "terminal_leaf_count": sum(
            entry["terminal_leaf_count"] for entry in entries
        ),
        "terminal_state_fingerprint": state_fingerprint(state),
        "rounds": entries,
    }
    _write_envelope(
        manifest_path, SAM_MINER_DECK_MANIFEST_VERSION, content
    )
    return _load_deck_manifest(manifest_path, config, fixture)


def _load_deck_manifest(
    path: Path,
    config: SamMinerConfig,
    fixture: DeckFixture,
) -> dict[str, object]:
    document = _validate_envelope(
        _load_json(path), SAM_MINER_DECK_MANIFEST_VERSION
    )
    content = document["content"]
    if (
        not isinstance(content, dict)
        or any(
            content.get(key) != value
            for key, value in _contract_bindings(config).items()
        )
        or content.get("collection_config_digest") != config.digest
        or content.get("fixture_id") != fixture.fixture_id
        or content.get("fixture_index") != fixture.index
        or content.get("split") != fixture.split.value
        or content.get("round_count") != ROUNDS_PER_DECK
        or not isinstance(content.get("rounds"), list)
    ):
        raise SamMinerError("deck manifest identity differs")
    rows = 0
    leaves = 0
    numbers: list[int] = []
    for entry in content["rounds"]:
        if not isinstance(entry, dict):
            raise SamMinerError("deck round entry is invalid")
        number = int(entry["round_number"])
        relative = Path(str(entry["relative_path"]))
        round_document = _load_round_manifest(
            config.output_path / relative, config, fixture, number
        )
        if (
            round_document["content_digest"] != entry["content_digest"]
            or round_document["file_digest"] != entry["file_digest"]
            or round_document["content"]["row_count"] != entry["row_count"]  # type: ignore[index]
            or round_document["content"]["terminal_leaf_count"]  # type: ignore[index]
            != entry["terminal_leaf_count"]
        ):
            raise SamMinerError("deck round manifest digest differs")
        numbers.append(number)
        rows += int(entry["row_count"])
        leaves += int(entry["terminal_leaf_count"])
    if numbers != list(range(1, ROUNDS_PER_DECK + 1)):
        raise SamMinerError("deck round manifests are not canonical")
    if content.get("row_count") != rows or content.get(
        "terminal_leaf_count"
    ) != leaves:
        raise SamMinerError("deck manifest totals differ")
    return document


def _worker_output_path(
    config: SamMinerConfig,
    fixture: DeckFixture,
) -> Path:
    return (
        config.output_path
        / ".workers"
        / f"{fixture.split.value}-{fixture.index:06d}-{fixture.fixture_id}"
    )


def _worker_config(
    config: SamMinerConfig,
    fixture: DeckFixture,
) -> SamMinerConfig:
    return replace(
        config,
        output_directory=str(_worker_output_path(config, fixture)),
        workers=1,
    )


def _configure_deck_worker() -> None:
    """Prevent one deck process from creating another CPU thread pool."""

    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = "1"
    import torch

    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError as error:
            raise SamMinerError(
                "worker inter-op threads initialized before deck mining"
            ) from error
    if torch.get_num_threads() != 1 or torch.get_num_interop_threads() != 1:
        raise SamMinerError("deck workers require one PyTorch thread")


def _mine_deck_worker(
    fixture: DeckFixture,
    config: SamMinerConfig,
    stop_path: str,
) -> DeckWorkerResult:
    worker_config = _worker_config(config, fixture)
    stop = Path(stop_path)
    document = _mine_deck_artifacts(
        fixture,
        teacher=_teacher_for_config(worker_config),
        config=worker_config,
        should_stop=stop.exists,
    )
    import torch

    return DeckWorkerResult(
        fixture=fixture,
        staging_directory=str(worker_config.output_path),
        deck_content_digest=str(document["content_digest"]),
        deck_file_digest=str(document["file_digest"]),
        torch_thread_count=torch.get_num_threads(),
        torch_interop_thread_count=torch.get_num_interop_threads(),
    )


def _merge_worker_cache(
    worker_config: SamMinerConfig,
    config: SamMinerConfig,
) -> None:
    source_root = worker_config.output_path / "cache"
    if not source_root.exists():
        return
    for source in sorted(source_root.rglob("*.json")):
        source_document = _validate_cache_artifact(source, worker_config)
        relative = source.relative_to(source_root)
        destination = config.output_path / "cache" / relative
        if destination.exists():
            destination_document = _validate_cache_artifact(
                destination, config
            )
            if destination_document != source_document:
                raise SamMinerError(
                    "worker caches disagree for one information state"
                )
            continue
        _atomic_bytes(destination, source.read_bytes())
        if _validate_cache_artifact(destination, config) != source_document:
            raise SamMinerError("merged teacher cache bytes differ")


def _merge_worker_deck(
    result: DeckWorkerResult,
    config: SamMinerConfig,
) -> dict[str, object]:
    if (
        result.torch_thread_count != 1
        or result.torch_interop_thread_count != 1
    ):
        raise SamMinerError("deck worker used more than one PyTorch thread")
    fixture = result.fixture
    worker_config = _worker_config(config, fixture)
    if Path(result.staging_directory) != worker_config.output_path:
        raise SamMinerError("deck worker staging identity differs")
    source = _deck_directory(worker_config, fixture)
    source_manifest = source / "deck-manifest.json"
    document = _load_deck_manifest(
        source_manifest, worker_config, fixture
    )
    if (
        document["content_digest"] != result.deck_content_digest
        or document["file_digest"] != result.deck_file_digest
    ):
        raise SamMinerError("deck worker result digest differs")

    # Cache entries are individually sealed. The deck directory is renamed
    # only after every source artifact and cache entry has been verified.
    _merge_worker_cache(worker_config, config)
    destination = _deck_directory(config, fixture)
    destination_manifest = destination / "deck-manifest.json"
    if destination.exists():
        existing = _load_deck_manifest(
            destination_manifest, config, fixture
        )
        if (
            existing["content_digest"] != document["content_digest"]
            or existing["file_digest"] != document["file_digest"]
        ):
            raise SamMinerError(
                "committed deck differs from completed worker deck"
            )
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(source, destination)
        existing = _load_deck_manifest(
            destination_manifest, config, fixture
        )
        if (
            existing["content_digest"] != document["content_digest"]
            or existing["file_digest"] != document["file_digest"]
        ):
            raise SamMinerError("committed worker deck bytes differ")
    shutil.rmtree(worker_config.output_path, ignore_errors=True)
    return existing


def _write_splits(config: SamMinerConfig) -> None:
    if config.continuous:
        groups: dict[str, object] = {
            "mode": "continuous",
            "cycle": [split.value for split in _split_cycle(config)],
        }
    else:
        fixtures = fixture_schedule(config)
        finite_groups = {
            split.value: [
                fixture.fixture_id
                for fixture in fixtures
                if fixture.split is split
            ]
            for split in DeckSplit
        }
        flattened = [
            fixture for values in finite_groups.values() for fixture in values
        ]
        if len(flattened) != len(set(flattened)):
            raise SamMinerError("deck fixture splits overlap")
        groups = {"fixtures": finite_groups}
    content = {
        "format_version": SAM_MINER_SPLITS_FORMAT_VERSION,
        "collection_config_digest": config.digest,
        **groups,
    }
    _write_envelope(
        config.output_path / "fixture-splits.json",
        SAM_MINER_SPLITS_FORMAT_VERSION,
        content,
    )


def initialize_corpus(config: SamMinerConfig) -> None:
    _verify_source_identity(config, required=True)
    resolved_path = config.output_path / "resolved-config.json"
    if resolved_path.exists():
        raise SamMinerError("output exists; use resume")
    _atomic_json(resolved_path, _resolved_config(config))
    _write_splits(config)
    _write_collection_state(
        config,
        "initialized",
        next_deck_ordinal=(0 if config.continuous else None),
    )


def load_config(output_directory: str | Path) -> SamMinerConfig:
    output = Path(output_directory).expanduser().resolve()
    value = _load_json(output / "resolved-config.json")
    if not isinstance(value, dict):
        raise SamMinerError("resolved configuration is invalid")
    try:
        config = SamMinerConfig(
            run_id=str(value["run_id"]),
            root_seed=str(value["root_seed"]),
            output_directory=str(value["output_directory"]),
            training_decks=int(value["training_decks"]),
            validation_decks=int(value["validation_decks"]),
            test_decks=int(value["test_decks"]),
            child_count=int(value["child_count"]),
            workers=int(value.get("workers", DEFAULT_WORKER_COUNT)),
            teacher_schema_version=str(value["teacher_schema_version"]),
            teacher_configuration_digest=str(
                value["teacher_configuration_digest"]
            ),
            teacher_controller_profile=str(
                value["teacher_controller_profile"]
            ),
            teacher_outer_simulation_budget=int(
                value["teacher_outer_simulation_budget"]
            ),
            teacher_response_simulation_budget=int(
                value["teacher_response_simulation_budget"]
            ),
            teacher_outer_exploration_constant=float(
                value["teacher_outer_exploration_constant"]
            ),
            teacher_response_exploration_constant=float(
                value["teacher_response_exploration_constant"]
            ),
            continuous=bool(value.get("continuous", False)),
            minimum_free_disk_bytes=int(
                value.get("minimum_free_disk_bytes", 0)
            ),
            inflight_disk_reserve_per_worker_bytes=int(
                value.get(
                    "inflight_disk_reserve_per_worker_bytes",
                    DEFAULT_INFLIGHT_DISK_RESERVE_PER_WORKER_BYTES,
                )
            ),
            source_tree_schema_version=(
                str(value["source_tree_schema_version"])
                if "source_tree_schema_version" in value
                else None
            ),
            source_revision=(
                str(value["source_revision"])
                if "source_revision" in value
                else None
            ),
            source_tree_digest=(
                str(value["source_tree_digest"])
                if "source_tree_digest" in value
                else None
            ),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise SamMinerError("resolved configuration is malformed") from error
    expected = _resolved_config(config)
    if value.get("format_version") in {
        SAM_MINER_SOURCE_CONFIG_FORMAT_VERSION,
        SAM_MINER_LEGACY_CONFIG_FORMAT_VERSION,
    }:
        expected.pop("continuous")
        expected.pop("minimum_free_disk_bytes")
        expected.pop("inflight_disk_reserve_per_worker_bytes")
        expected["format_version"] = value["format_version"]
    if "workers" not in value:
        # Parallelism was added without changing any corpus contract digest.
        expected.pop("workers")
    if config.output_path != output or expected != value:
        raise SamMinerError("resolved configuration digest or path differs")
    return config


def _write_corpus_manifest(
    config: SamMinerConfig,
    deck_documents: Sequence[dict[str, object]],
) -> dict[str, object]:
    entries = sorted(
        (
            {
                "split": str(document["content"]["split"]),  # type: ignore[index]
                "fixture_index": int(
                    document["content"]["fixture_index"]  # type: ignore[index]
                ),
                "fixture_id": str(document["content"]["fixture_id"]),  # type: ignore[index]
                "relative_path": str(
                    (
                        config.output_path
                        / "decks"
                        / str(document["content"]["split"])  # type: ignore[index]
                        / str(document["content"]["fixture_id"])  # type: ignore[index]
                        / "deck-manifest.json"
                    ).relative_to(config.output_path)
                ),
                "content_digest": str(document["content_digest"]),
                "file_digest": str(document["file_digest"]),
                "row_count": int(document["content"]["row_count"]),  # type: ignore[index]
                "terminal_leaf_count": int(
                    document["content"]["terminal_leaf_count"]  # type: ignore[index]
                ),
            }
            for document in deck_documents
        ),
        key=lambda entry: (
            list(DeckSplit).index(DeckSplit(entry["split"])),
            entry["fixture_index"],
        ),
    )
    content = {
        **_contract_bindings(config),
        "collection_config_digest": config.digest,
        "deck_count": len(entries),
        "row_count": sum(entry["row_count"] for entry in entries),
        "terminal_leaf_count": sum(
            entry["terminal_leaf_count"] for entry in entries
        ),
        "decks": entries,
    }
    return _write_envelope(
        config.output_path / "corpus-manifest.json",
        SAM_MINER_CORPUS_MANIFEST_VERSION,
        content,
    )


def _parallel_deck_documents(
    config: SamMinerConfig,
    *,
    should_stop: Callable[[], bool] | None,
    fixtures: Sequence[DeckFixture] | None = None,
) -> list[dict[str, object]]:
    fixtures = (
        tuple(fixtures)
        if fixtures is not None
        else fixture_schedule(config)
    )
    documents: dict[str, dict[str, object]] = {}
    remaining: list[DeckFixture] = []
    for fixture in fixtures:
        manifest = _deck_directory(config, fixture) / "deck-manifest.json"
        if manifest.exists():
            documents[fixture.fixture_id] = _load_deck_manifest(
                manifest, config, fixture
            )
        else:
            remaining.append(fixture)
    if not remaining:
        return [documents[fixture.fixture_id] for fixture in fixtures]

    workers_root = config.output_path / ".workers"
    stop_path = workers_root / "stop-requested"
    stop_path.unlink(missing_ok=True)
    workers_root.mkdir(parents=True, exist_ok=True)
    executor = ProcessPoolExecutor(
        max_workers=config.workers,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_configure_deck_worker,
    )
    futures = {
        executor.submit(
            _mine_deck_worker,
            fixture,
            config,
            str(stop_path),
        ): fixture
        for fixture in remaining
    }
    try:
        while futures:
            if should_stop is not None and should_stop():
                raise SamMinerInterrupted(
                    "corpus mining interrupted while deck workers were active"
                )
            completed, _ = wait(
                tuple(futures),
                timeout=0.01,
                return_when=FIRST_COMPLETED,
            )
            for future in sorted(
                completed,
                key=lambda item: (
                    list(DeckSplit).index(futures[item].split),
                    futures[item].index,
                ),
            ):
                fixture = futures.pop(future)
                try:
                    result = future.result()
                except SamMinerInterrupted:
                    raise
                except Exception as error:
                    raise SamMinerError(
                        f"deck worker failed for {fixture.fixture_id}"
                    ) from error
                if result.fixture != fixture:
                    raise SamMinerError("deck worker returned another fixture")
                document = _merge_worker_deck(
                    result, config
                )
                documents[fixture.fixture_id] = document
                print(
                    json.dumps(
                        {
                            "event": "deck_sealed",
                            "fixture_id": fixture.fixture_id,
                            "fixture_index": fixture.index,
                            "split": fixture.split.value,
                            "row_count": int(
                                document["content"]["row_count"]  # type: ignore[index]
                            ),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
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
    try:
        workers_root.rmdir()
    except OSError:
        pass
    return [documents[fixture.fixture_id] for fixture in fixtures]


def _committed_deck_documents(
    config: SamMinerConfig,
) -> list[dict[str, object]]:
    documents: list[dict[str, object]] = []
    decks_root = config.output_path / "decks"
    if not decks_root.exists():
        return documents
    for path in sorted(decks_root.glob("*/*/deck-manifest.json")):
        value = _load_json(path)
        if not isinstance(value, dict) or not isinstance(
            value.get("content"), dict
        ):
            raise SamMinerError("committed deck manifest is malformed")
        content = value["content"]
        assert isinstance(content, dict)
        fixture = derive_deck_fixture(
            config.root_seed,
            DeckSplit(str(content.get("split"))),
            int(content.get("fixture_index")),
        )
        documents.append(_load_deck_manifest(path, config, fixture))
    return documents


def _continuous_next_ordinal(
    config: SamMinerConfig,
    documents: Sequence[dict[str, object]],
) -> int:
    ordinals = sorted(
        _continuous_fixture_ordinal(
            config,
            derive_deck_fixture(
                config.root_seed,
                DeckSplit(str(document["content"]["split"])),  # type: ignore[index]
                int(document["content"]["fixture_index"]),  # type: ignore[index]
            ),
        )
        for document in documents
    )
    if ordinals != list(range(len(ordinals))):
        raise SamMinerError("continuous corpus contains a deck-sequence gap")
    return len(ordinals)


def _continuous_committed_prefix(
    config: SamMinerConfig,
    documents: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    """Return only the sealed, gap-free prefix eligible for the manifest."""

    by_ordinal: dict[int, dict[str, object]] = {}
    for document in documents:
        content = document["content"]
        assert isinstance(content, dict)
        fixture = derive_deck_fixture(
            config.root_seed,
            DeckSplit(str(content["split"])),
            int(content["fixture_index"]),
        )
        ordinal = _continuous_fixture_ordinal(config, fixture)
        if ordinal in by_ordinal:
            raise SamMinerError("continuous corpus repeats a deck ordinal")
        by_ordinal[ordinal] = document
    prefix: list[dict[str, object]] = []
    ordinal = 0
    while ordinal in by_ordinal:
        prefix.append(by_ordinal[ordinal])
        ordinal += 1
    return prefix


def _write_collection_state(
    config: SamMinerConfig,
    phase: str,
    *,
    next_deck_ordinal: int | None = None,
    free_disk_bytes: int | None = None,
) -> None:
    content: dict[str, object] = {
        "format_version": SAM_MINER_STATE_FORMAT_VERSION,
        "collection_config_digest": config.digest,
        "phase": phase,
    }
    if next_deck_ordinal is not None:
        content["next_deck_ordinal"] = next_deck_ordinal
    if free_disk_bytes is not None:
        content["free_disk_bytes"] = free_disk_bytes
    _atomic_json(config.output_path / "state.json", content)


def _free_disk_bytes(config: SamMinerConfig) -> int:
    return shutil.disk_usage(config.output_path).free


def _continuous_disk_guard_reached(
    config: SamMinerConfig, *, before_batch: bool
) -> bool:
    required = config.minimum_free_disk_bytes
    if before_batch:
        required += (
            config.workers
            * config.inflight_disk_reserve_per_worker_bytes
        )
    return _free_disk_bytes(config) <= required


def _mine_continuous_corpus(
    config: SamMinerConfig,
    teacher: SamMinerTeacher,
    *,
    should_stop: Callable[[], bool] | None,
) -> CorpusInspection:
    documents = _continuous_committed_prefix(
        config, _committed_deck_documents(config)
    )
    next_ordinal = _continuous_next_ordinal(config, documents)
    if documents:
        _write_corpus_manifest(config, documents)

    while True:
        if should_stop is not None and should_stop():
            if not documents:
                _write_corpus_manifest(config, documents)
            _write_collection_state(
                config, "stopped", next_deck_ordinal=next_ordinal
            )
            return inspect_corpus(config.output_path)
        if _continuous_disk_guard_reached(config, before_batch=True):
            if not documents:
                _write_corpus_manifest(config, documents)
            _write_collection_state(
                config,
                "disk-floor",
                next_deck_ordinal=next_ordinal,
                free_disk_bytes=_free_disk_bytes(config),
            )
            return inspect_corpus(config.output_path)

        fixtures = tuple(
            continuous_deck_fixture(config, ordinal)
            for ordinal in range(
                next_ordinal, next_ordinal + config.workers
            )
        )

        def stop_batch() -> bool:
            return (
                (should_stop is not None and should_stop())
                or _continuous_disk_guard_reached(
                    config, before_batch=False
                )
            )

        try:
            if config.workers == 1:
                batch = [
                    _mine_deck_artifacts(
                        fixtures[0],
                        teacher=teacher,
                        config=config,
                        should_stop=stop_batch,
                    )
                ]
            else:
                batch = _parallel_deck_documents(
                    config,
                    should_stop=stop_batch,
                    fixtures=fixtures,
                )
        except SamMinerInterrupted as error:
            disk_floor = _continuous_disk_guard_reached(
                config, before_batch=False
            )
            documents = _continuous_committed_prefix(
                config, _committed_deck_documents(config)
            )
            next_ordinal = _continuous_next_ordinal(config, documents)
            _write_corpus_manifest(config, documents)
            _write_collection_state(
                config,
                "disk-floor" if disk_floor else "interrupted",
                next_deck_ordinal=next_ordinal,
                free_disk_bytes=(
                    _free_disk_bytes(config) if disk_floor else None
                ),
            )
            if disk_floor:
                return inspect_corpus(config.output_path)
            raise error

        documents.extend(batch)
        next_ordinal = _continuous_next_ordinal(config, documents)
        _write_corpus_manifest(config, documents)
        _write_collection_state(
            config, "collecting", next_deck_ordinal=next_ordinal
        )
        print(
            json.dumps(
                {
                    "event": "continuous_batch_sealed",
                    "deck_count": len(documents),
                    "next_deck_ordinal": next_ordinal,
                    "row_count": sum(
                        int(document["content"]["row_count"])  # type: ignore[index]
                        for document in documents
                    ),
                    "free_disk_bytes": _free_disk_bytes(config),
                },
                sort_keys=True,
            ),
            flush=True,
        )


def mine_corpus(
    config: SamMinerConfig,
    teacher: SamMinerTeacher,
    *,
    resume: bool = False,
    should_stop: Callable[[], bool] | None = None,
) -> CorpusInspection:
    _verify_source_identity(config, required=True)
    if resume:
        if load_config(config.output_path) != config:
            raise SamMinerError("resume configuration differs")
    else:
        initialize_corpus(config)
    if (
        teacher.schema_version != config.teacher_schema_version
        or teacher.configuration_digest
        != config.teacher_configuration_digest
        or teacher.controller_profile != config.teacher_controller_profile
    ):
        raise SamMinerError("injected teacher differs from resolved config")
    try:
        if config.continuous:
            return _mine_continuous_corpus(
                config, teacher, should_stop=should_stop
            )
        if config.workers == 1:
            documents: list[dict[str, object]] = []
            for fixture in fixture_schedule(config):
                if should_stop is not None and should_stop():
                    raise SamMinerInterrupted(
                        "corpus mining interrupted between decks"
                    )
                documents.append(
                    _mine_deck_artifacts(
                        fixture,
                        teacher=teacher,
                        config=config,
                        should_stop=should_stop,
                    )
                )
        else:
            # Validate before spawning; workers reconstruct only the two
            # versioned teacher implementations accepted by configuration.
            _teacher_for_config(config)
            documents = _parallel_deck_documents(
                config,
                should_stop=should_stop,
            )
    except (KeyboardInterrupt, SamMinerInterrupted):
        _write_collection_state(config, "interrupted")
        raise SamMinerInterrupted("corpus mining interrupted")
    _write_corpus_manifest(config, documents)
    _write_collection_state(config, "complete")
    return inspect_corpus(config.output_path)


def _load_corpus_manifest(
    config: SamMinerConfig,
) -> dict[str, object]:
    document = _validate_envelope(
        _load_json(config.output_path / "corpus-manifest.json"),
        SAM_MINER_CORPUS_MANIFEST_VERSION,
    )
    content = document["content"]
    if (
        not isinstance(content, dict)
        or any(
            content.get(key) != value
            for key, value in _contract_bindings(config).items()
        )
        or content.get("collection_config_digest") != config.digest
        or not isinstance(content.get("decks"), list)
    ):
        raise SamMinerError("corpus manifest identity differs")
    seen: set[str] = set()
    rows = 0
    leaves = 0
    canonical: list[tuple[int, int]] = []
    continuous_ordinals: list[int] = []
    for entry in content["decks"]:
        if not isinstance(entry, dict):
            raise SamMinerError("corpus deck entry is invalid")
        fixture = derive_deck_fixture(
            config.root_seed,
            DeckSplit(str(entry["split"])),
            int(entry["fixture_index"]),
        )
        if fixture.fixture_id != entry["fixture_id"]:
            raise SamMinerError("corpus fixture derivation differs")
        relative = Path(str(entry["relative_path"]))
        deck = _load_deck_manifest(
            config.output_path / relative, config, fixture
        )
        if (
            deck["content_digest"] != entry["content_digest"]
            or deck["file_digest"] != entry["file_digest"]
            or deck["content"]["row_count"] != entry["row_count"]  # type: ignore[index]
            or deck["content"]["terminal_leaf_count"]  # type: ignore[index]
            != entry["terminal_leaf_count"]
        ):
            raise SamMinerError("corpus deck digest differs")
        if fixture.fixture_id in seen:
            raise SamMinerError("corpus repeats a deck fixture")
        seen.add(fixture.fixture_id)
        canonical.append((list(DeckSplit).index(fixture.split), fixture.index))
        if config.continuous:
            continuous_ordinals.append(
                _continuous_fixture_ordinal(config, fixture)
            )
        rows += int(entry["row_count"])
        leaves += int(entry["terminal_leaf_count"])
    if canonical != sorted(canonical):
        raise SamMinerError("corpus deck manifests are not canonical")
    if config.continuous and sorted(continuous_ordinals) != list(
        range(len(continuous_ordinals))
    ):
        raise SamMinerError("continuous corpus deck sequence differs")
    if (
        content.get("deck_count") != len(seen)
        or content.get("row_count") != rows
        or content.get("terminal_leaf_count") != leaves
    ):
        raise SamMinerError("corpus totals differ")
    return document


def inspect_corpus(output_directory: str | Path) -> CorpusInspection:
    config = load_config(output_directory)
    document = _load_corpus_manifest(config)
    content = document["content"]
    assert isinstance(content, dict)
    cache_count = sum(1 for _ in (config.output_path / "cache").rglob("*.json"))
    disk_bytes = sum(
        path.stat().st_size
        for path in config.output_path.rglob("*")
        if path.is_file()
    )
    return CorpusInspection(
        config_digest=config.digest,
        corpus_manifest_digest=str(document["content_digest"]),
        deck_count=int(content["deck_count"]),
        round_count=int(content["deck_count"]) * ROUNDS_PER_DECK,
        shard_count=int(content["deck_count"])
        * ROUNDS_PER_DECK
        * (config.child_count + 1),
        row_count=int(content["row_count"]),
        terminal_leaf_count=int(content["terminal_leaf_count"]),
        cache_entry_count=cache_count,
        disk_bytes=disk_bytes,
    )


def verify_corpus(output_directory: str | Path) -> CorpusInspection:
    config = load_config(output_directory)
    splits = _validate_envelope(
        _load_json(config.output_path / "fixture-splits.json"),
        SAM_MINER_SPLITS_FORMAT_VERSION,
    )
    split_content = splits["content"]
    if (
        not isinstance(split_content, dict)
        or split_content.get("collection_config_digest") != config.digest
    ):
        raise SamMinerError("fixture split artifact differs")
    if config.continuous:
        if (
            split_content.get("mode") != "continuous"
            or split_content.get("cycle")
            != [split.value for split in _split_cycle(config)]
            or set(split_content)
            != {
                "format_version",
                "collection_config_digest",
                "mode",
                "cycle",
            }
        ):
            raise SamMinerError("continuous fixture split cycle differs")
    else:
        expected = {
            split.value: [
                fixture.fixture_id
                for fixture in fixture_schedule(config)
                if fixture.split is split
            ]
            for split in DeckSplit
        }
        if split_content.get("fixtures") != expected:
            raise SamMinerError("fixture split roots differ")
    _assert_private_fields_absent(_load_corpus_manifest(config))
    return inspect_corpus(output_directory)


def summarize_corpus(output_directory: str | Path) -> dict[str, object]:
    inspection = verify_corpus(output_directory)
    return asdict(inspection)


def load_prefix_smoke(
    path: str | Path,
    config: SamMinerConfig,
) -> dict[str, object]:
    document = _validate_envelope(
        _load_json(Path(path)), SAM_MINER_SMOKE_FORMAT_VERSION
    )
    content = document["content"]
    required = {
        *_contract_bindings(config),
        "collection_config_digest",
        "fixture_id",
        "fixture_index",
        "split",
        "maximum_placement",
        "row_count",
        "frontier_count",
        "frontier_path_digests",
        "rows",
    }
    if (
        not isinstance(content, dict)
        or set(content) != required
        or any(
            content.get(key) != value
            for key, value in _contract_bindings(config).items()
        )
        or content.get("collection_config_digest") != config.digest
        or content.get("split") not in {
            split.value for split in DeckSplit
        }
        or type(content.get("maximum_placement")) is not int
        or not 1 <= int(content["maximum_placement"]) <= 6
        or not isinstance(content.get("rows"), list)
        or not isinstance(content.get("frontier_path_digests"), list)
    ):
        raise SamMinerError("prefix smoke artifact is incompatible")
    _assert_private_fields_absent(content)
    rows = tuple(SamMinerRow.from_dict(row) for row in content["rows"])
    frontier = tuple(content["frontier_path_digests"])
    if (
        content["row_count"] != len(rows)
        or content["frontier_count"] != len(frontier)
        or any(
            row.deck_fixture_id != content["fixture_id"]
            or row.placement_number > content["maximum_placement"]
            or row.collection_configuration_digest != config.digest
            or row.teacher_configuration_digest
            != config.teacher_configuration_digest
            or row.branch_configuration_digest
            != config.branch_configuration_digest
            for row in rows
        )
        or any(
            _require_digest(value, "frontier path digest") != value
            for value in frontier
        )
        or rows
        != tuple(
            sorted(
                rows,
                key=lambda row: (
                    row.placement_number,
                    row.branch_path_digest,
                ),
            )
        )
    ):
        raise SamMinerError("prefix smoke rows or frontier differ")
    return document


def mine_real_teacher_smoke(
    config: SamMinerConfig,
    teacher: SamMinerTeacher,
    *,
    maximum_placement: int = 1,
    should_stop: Callable[[], bool] | None = None,
    use_teacher_cache: bool = True,
) -> PrefixSmokeInspection:
    """Seal a bounded production prefix without reaching round completion."""

    _verify_source_identity(config, required=True)
    if config.teacher_controller_profile != SAM_MINER_CONTROLLER_PROFILE:
        raise SamMinerError("real-teacher smoke requires production Sam-32")
    if load_config(config.output_path) != config:
        raise SamMinerError("smoke configuration differs from resolved config")
    fixture = (
        continuous_deck_fixture(config, 0)
        if config.continuous
        else fixture_schedule(config)[0]
    )
    state = create_game(fixture._engine_seed)
    root_path = _root_path_digest(fixture.fixture_id, 1)
    started = time.perf_counter()
    result = mine_branch_prefix(
        state,
        fixture_id=fixture.fixture_id,
        path_digest=root_path,
        maximum_placement=maximum_placement,
        teacher=teacher,
        config=config,
        should_stop=should_stop,
        use_teacher_cache=use_teacher_cache,
    )
    elapsed = time.perf_counter() - started
    rows = tuple(
        sorted(
            result.rows,
            key=lambda row: (
                row.placement_number,
                row.branch_path_digest,
            ),
        )
    )
    content = {
        **_contract_bindings(config),
        "collection_config_digest": config.digest,
        "fixture_id": fixture.fixture_id,
        "fixture_index": fixture.index,
        "split": fixture.split.value,
        "maximum_placement": maximum_placement,
        "row_count": len(rows),
        "frontier_count": len(result.frontier_path_digests),
        "frontier_path_digests": sorted(
            result.frontier_path_digests
        ),
        "rows": [row.to_dict() for row in rows],
    }
    _assert_private_fields_absent(content)
    destination = (
        config.output_path
        / "smoke"
        / f"prefix-placement-{maximum_placement:02d}.json"
    )
    document = _write_envelope(
        destination,
        SAM_MINER_SMOKE_FORMAT_VERSION,
        content,
    )
    load_prefix_smoke(destination, config)
    return PrefixSmokeInspection(
        content_digest=str(document["content_digest"]),
        file_digest=str(document["file_digest"]),
        row_count=len(rows),
        frontier_count=len(result.frontier_path_digests),
        maximum_placement=maximum_placement,
        elapsed_seconds=elapsed,
        relative_path=str(destination.relative_to(config.output_path)),
    )


def _stub_config(
    *,
    run_id: str,
    root_seed: str,
    output_directory: str,
    training_decks: int,
    validation_decks: int,
    test_decks: int,
    child_count: int,
    workers: int = DEFAULT_WORKER_COUNT,
    continuous: bool = False,
    minimum_free_disk_bytes: int = 0,
) -> SamMinerConfig:
    stub = DeterministicSamTeacherStub()
    source = resolve_source_identity()
    return SamMinerConfig(
        run_id=run_id,
        root_seed=root_seed,
        output_directory=output_directory,
        training_decks=training_decks,
        validation_decks=validation_decks,
        test_decks=test_decks,
        child_count=child_count,
        workers=workers,
        continuous=continuous,
        minimum_free_disk_bytes=minimum_free_disk_bytes,
        teacher_schema_version=stub.schema_version,
        teacher_configuration_digest=stub.configuration_digest,
        teacher_controller_profile=stub.controller_profile,
        source_tree_schema_version=source.schema_version,
        source_revision=source.revision,
        source_tree_digest=source.tree_digest,
    )


def _production_config(
    *,
    run_id: str,
    root_seed: str,
    output_directory: str,
    training_decks: int,
    validation_decks: int,
    test_decks: int,
    child_count: int,
    workers: int = DEFAULT_WORKER_COUNT,
    continuous: bool = False,
    minimum_free_disk_bytes: int = 0,
) -> SamMinerConfig:
    search_config = SamTeacherSearchConfig(
        DEFAULT_TEACHER_SIMULATION_BUDGET,
        DEFAULT_TEACHER_SIMULATION_BUDGET,
    )
    source = resolve_source_identity()
    return SamMinerConfig(
        run_id=run_id,
        root_seed=root_seed,
        output_directory=output_directory,
        training_decks=training_decks,
        validation_decks=validation_decks,
        test_decks=test_decks,
        child_count=child_count,
        workers=workers,
        continuous=continuous,
        minimum_free_disk_bytes=minimum_free_disk_bytes,
        teacher_schema_version=SAM_TEACHER_SEARCH_SCHEMA_VERSION,
        teacher_configuration_digest=search_config.digest,
        teacher_controller_profile=SAM_MINER_CONTROLLER_PROFILE,
        teacher_outer_simulation_budget=(
            search_config.outer_simulation_budget
        ),
        teacher_response_simulation_budget=(
            search_config.response_simulation_budget
        ),
        teacher_outer_exploration_constant=(
            search_config.outer_exploration_constant
        ),
        teacher_response_exploration_constant=(
            search_config.response_exploration_constant
        ),
        source_tree_schema_version=source.schema_version,
        source_revision=source.revision,
        source_tree_digest=source.tree_digest,
    )


def _teacher_for_config(config: SamMinerConfig) -> SamMinerTeacher:
    _verify_source_identity(config, required=True)
    if config.teacher_controller_profile == SAM_MINER_CONTROLLER_PROFILE:
        return Sam128MinerTeacher(config.teacher_search_config)
    if config.teacher_controller_profile == SAM_MINER_STUB_PROFILE:
        return DeterministicSamTeacherStub()
    raise SamMinerError(
        "resolved teacher profile has no executable implementation"
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="dracula-sam-miner")
    subparsers = parser.add_subparsers(dest="command", required=True)
    initialize = subparsers.add_parser("initialize")
    initialize.add_argument("--output", required=True)
    initialize.add_argument("--run-id", required=True)
    initialize.add_argument("--root-seed", required=True)
    initialize.add_argument("--training-decks", type=int, default=1)
    initialize.add_argument("--validation-decks", type=int, default=1)
    initialize.add_argument("--test-decks", type=int, default=1)
    initialize.add_argument("--child-count", type=int, default=4)
    initialize.add_argument(
        "--workers",
        type=int,
        choices=range(1, MAX_WORKER_COUNT + 1),
        default=DEFAULT_WORKER_COUNT,
    )
    initialize.add_argument(
        "--teacher-profile",
        choices=("sam-32", "deterministic-stub"),
        default="sam-32",
    )
    initialize.add_argument("--continuous", action="store_true")
    initialize.add_argument(
        "--minimum-free-disk-gib",
        type=float,
        default=1.0,
    )
    smoke = subparsers.add_parser("smoke")
    smoke.add_argument("--output", required=True)
    smoke.add_argument("--maximum-placement", type=int, default=1)
    for command in ("mine", "resume", "inspect", "verify", "summarize"):
        subparsers.add_parser(command).add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "initialize":
            factory = (
                _production_config
                if args.teacher_profile == "sam-32"
                else _stub_config
            )
            minimum_free_disk_bytes = (
                int(args.minimum_free_disk_gib * (1 << 30))
                if args.continuous
                else 0
            )
            if args.continuous and args.minimum_free_disk_gib <= 0:
                raise SamMinerError(
                    "minimum free disk GiB must be positive"
                )
            config = factory(
                run_id=args.run_id,
                root_seed=args.root_seed,
                output_directory=args.output,
                training_decks=args.training_decks,
                validation_decks=args.validation_decks,
                test_decks=args.test_decks,
                child_count=args.child_count,
                workers=args.workers,
                continuous=args.continuous,
                minimum_free_disk_bytes=minimum_free_disk_bytes,
            )
            initialize_corpus(config)
            result: object = _resolved_config(config)
        elif args.command == "smoke":
            config = load_config(args.output)
            result = asdict(
                mine_real_teacher_smoke(
                    config,
                    _teacher_for_config(config),
                    maximum_placement=args.maximum_placement,
                )
            )
        elif args.command in {"mine", "resume"}:
            config = load_config(args.output)
            result = mine_corpus(
                config,
                _teacher_for_config(config),
                resume=True,
            )
            result = asdict(result)
        elif args.command == "inspect":
            result = asdict(inspect_corpus(args.output))
        elif args.command == "verify":
            result = asdict(verify_corpus(args.output))
        else:
            result = summarize_corpus(args.output)
    except SamMinerInterrupted:
        print("sam_miner interrupted", file=sys.stderr)
        return 130
    except SamMinerError as error:
        print(f"sam_miner error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = (
    "ACTION_SCHEMA_DIGEST",
    "BRANCH_SCHEMA_DIGEST",
    "BranchPrefixResult",
    "CorpusInspection",
    "DATASET_SCHEMA_DIGEST",
    "DeckFixture",
    "DeckSplit",
    "DeterministicSamTeacherStub",
    "OBSERVATION_SCHEMA_DIGEST",
    "PrefixSmokeInspection",
    "SAM_MINER_BRANCH_SCHEMA_VERSION",
    "SAM_MINER_CACHE_FORMAT_VERSION",
    "SAM_MINER_CONTROLLER_PROFILE",
    "SAM_MINER_CORPUS_MANIFEST_VERSION",
    "SAM_MINER_DATASET_SCHEMA_VERSION",
    "SAM_MINER_DECK_MANIFEST_VERSION",
    "SAM_MINER_REDUCED_TEST_PROFILE",
    "SAM_MINER_ROW_SCHEMA_VERSION",
    "SAM_MINER_ROUND_MANIFEST_VERSION",
    "SAM_MINER_SMOKE_FORMAT_VERSION",
    "SAM_MINER_SHARD_FORMAT_VERSION",
    "SAM_MINER_SOURCE_TREE_SCHEMA_VERSION",
    "SAM_MINER_STUB_PROFILE",
    "SYMMETRY_SCHEMA_DIGEST",
    "Sam128MinerTeacher",
    "SamMinerConfig",
    "SamMinerError",
    "SamMinerInterrupted",
    "SamMinerRow",
    "SamMinerSourceIdentity",
    "SamMinerTeacher",
    "SamMinerTeacherSelection",
    "derive_deck_fixture",
    "fixture_schedule",
    "initialize_corpus",
    "inspect_corpus",
    "load_config",
    "load_prefix_smoke",
    "load_subtree_shard",
    "main",
    "mine_branch_subtree",
    "mine_branch_prefix",
    "mine_corpus",
    "mine_deck_tree",
    "mine_round_tree",
    "mine_real_teacher_smoke",
    "play_teacher_trunk",
    "resolve_teacher",
    "resolve_source_identity",
    "summarize_corpus",
    "unpack_legal_mask",
    "unpack_observation",
    "verify_corpus",
    "validate_sam128_result",
    "write_subtree_shard",
)

"""Turn-balanced dataset mining with the belief-greedy search teacher."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import resource
import shutil
import signal
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import torch

from dracula.bridge import ACTION_COUNT, move_for_action_index
from dracula.engine import (
    EnginePlayer,
    EngineState,
    EngineStatus,
    advance_after_round,
    apply_move,
    create_game,
    legal_moves,
)
from dracula.randomness import Sha256CounterStream, derive_seed, seed_hex
from dracula.search import (
    BeliefGreedyInformationSetSearch,
    BeliefGreedySearchConfig,
    derive_belief_greedy_request_seed,
    information_state_fingerprint,
    information_state_from_engine,
    policy_input_from_information_state,
    select_concrete_action_index,
    strategic_action_groups,
)

BELIEF_GREEDY_MINER_SCHEMA_VERSION = "dracula-belief-greedy-miner-v2"
BELIEF_GREEDY_MINER_ROW_VERSION = "dracula-belief-greedy-balanced-row-v2"
BELIEF_GREEDY_MINER_GAME_VERSION = "dracula-belief-greedy-balanced-game-v2"
BELIEF_GREEDY_MINER_MANIFEST_VERSION = (
    "dracula-belief-greedy-balanced-manifest-v2"
)
BELIEF_GREEDY_CONTINUOUS_CONFIG_VERSION = (
    "dracula-belief-greedy-continuous-config-v2"
)
BELIEF_GREEDY_CONTINUOUS_MANIFEST_VERSION = (
    "dracula-belief-greedy-continuous-manifest-v2"
)
BELIEF_GREEDY_SOURCE_TREE_VERSION = "dracula-source-tree-v1"
BELIEF_GREEDY_MINER_DECK_NAMESPACE = "dracula-belief-greedy-miner-deck-v1"
BELIEF_GREEDY_MINER_FIXTURE_NAMESPACE = (
    "dracula-belief-greedy-miner-fixture-v1"
)
BELIEF_GREEDY_MINER_TRAJECTORY_NAMESPACE = (
    "dracula-belief-greedy-miner-trajectory-v1"
)
BELIEF_GREEDY_MINER_ALTERNATIVE_NAMESPACE = (
    "dracula-belief-greedy-miner-alternative-v1"
)
BELIEF_GREEDY_MINER_DESTINATION_NAMESPACE = (
    "dracula-belief-greedy-miner-destination-v1"
)

OBSERVATION_BIT_COUNT = 875
LEGAL_MASK_BIT_COUNT = ACTION_COUNT
LEARNED_PLACEMENTS = 7
ROUNDS_PER_GAME = 6
ROWS_PER_PLACEMENT_PER_GAME = ROUNDS_PER_GAME
ROWS_PER_GAME = LEARNED_PLACEMENTS * ROUNDS_PER_GAME


class BeliefGreedyMinerError(ValueError):
    """A balanced-miner input or artifact violates its contract."""


@dataclass(frozen=True, slots=True)
class BeliefGreedySourceIdentity:
    revision: str
    tree_digest: str


def resolve_source_identity() -> BeliefGreedySourceIdentity:
    """Bind collection to the Git base and exact imported source inputs."""

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
        raise BeliefGreedyMinerError(
            "belief-greedy collection requires a Git revision"
        ) from error
    if (
        len(revision) not in (40, 64)
        or any(character not in "0123456789abcdef" for character in revision)
    ):
        raise BeliefGreedyMinerError("Git returned an invalid revision")
    paths = [repository / "NORTHSTARS", repository / "pyproject.toml"]
    paths.extend(sorted((repository / "src").rglob("*.py")))
    paths.extend(sorted((repository / "docs").rglob("*.md")))
    entries: list[dict[str, str]] = []
    for path in paths:
        if not path.is_file():
            raise BeliefGreedyMinerError(
                f"source input is missing: {path.relative_to(repository)}"
            )
        entries.append(
            {
                "path": path.relative_to(repository).as_posix(),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    return BeliefGreedySourceIdentity(
        revision=revision,
        tree_digest=_digest(
            {
                "schema_version": BELIEF_GREEDY_SOURCE_TREE_VERSION,
                "files": entries,
            }
        ),
    )


class TrajectoryProfile(StrEnum):
    TEACHER = "teacher"
    ONE_DEVIATION = "one-deviation"
    TWO_DEVIATIONS = "two-deviations"
    MIXED = "mixed"
    ALTERNATIVE = "alternative"


TRAJECTORY_PROFILE_CYCLE = tuple(TrajectoryProfile)


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _pack_bools(values: tuple[bool, ...], expected_count: int) -> bytes:
    if len(values) != expected_count:
        raise BeliefGreedyMinerError(
            f"expected {expected_count} Boolean values, received {len(values)}"
        )
    packed = bytearray((expected_count + 7) // 8)
    for index, value in enumerate(values):
        if type(value) is not bool:
            raise BeliefGreedyMinerError("packed values must be Boolean")
        if value:
            packed[index // 8] |= 1 << (7 - index % 8)
    return bytes(packed)


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    encoded = _canonical_bytes(value) + b"\n"
    with temporary.open("wb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


@dataclass(frozen=True, slots=True)
class BalancedMinerConfig:
    root_seed: str
    outer_simulation_budget: int
    belief_completion_count: int = 8
    worker_count: int = 4

    def __post_init__(self) -> None:
        if not isinstance(self.root_seed, str) or not self.root_seed:
            raise BeliefGreedyMinerError("root seed must be nonempty")
        for label, value in (
            ("outer simulation budget", self.outer_simulation_budget),
            ("belief completion count", self.belief_completion_count),
            ("worker count", self.worker_count),
        ):
            if type(value) is not int or value < 1:
                raise BeliefGreedyMinerError(
                    f"{label} must be a positive integer"
                )
        if self.worker_count > 8:
            raise BeliefGreedyMinerError("worker count cannot exceed eight")

    @property
    def search_config(self) -> BeliefGreedySearchConfig:
        return BeliefGreedySearchConfig(
            outer_simulation_budget=self.outer_simulation_budget,
            belief_completion_count=self.belief_completion_count,
        )

    @property
    def content_digest(self) -> str:
        return _digest(
            {
                "miner_schema_version": BELIEF_GREEDY_MINER_SCHEMA_VERSION,
                "root_seed": self.root_seed,
                "search_config_digest": self.search_config.digest,
                "trajectory_profiles": [
                    profile.value for profile in TRAJECTORY_PROFILE_CYCLE
                ],
            }
        )

    @property
    def runtime_digest(self) -> str:
        return _digest(
            {
                "content_digest": self.content_digest,
                "worker_count": self.worker_count,
            }
        )


@dataclass(frozen=True, slots=True)
class BalancedMinerRow:
    information_state_fingerprint: str
    observation_packed: bytes
    legal_mask_packed: bytes
    strategic_groups: tuple[tuple[int, ...], ...]
    strategic_group_representatives: tuple[int, ...]
    strategic_group_visits: tuple[int, ...]
    selected_group_representative: int
    round_number: int
    placement_number: int
    player: EnginePlayer
    dealer: EnginePlayer
    fixture_id: str
    trajectory_profile: TrajectoryProfile
    search_config_digest: str

    def canonical_data(self) -> dict[str, object]:
        return {
            "dealer": self.dealer.value,
            "fixture_id": self.fixture_id,
            "information_state_fingerprint": (
                self.information_state_fingerprint
            ),
            "legal_mask_packed": base64.b64encode(
                self.legal_mask_packed
            ).decode("ascii"),
            "observation_packed": base64.b64encode(
                self.observation_packed
            ).decode("ascii"),
            "placement_number": self.placement_number,
            "player": self.player.value,
            "round_number": self.round_number,
            "row_schema_version": BELIEF_GREEDY_MINER_ROW_VERSION,
            "search_config_digest": self.search_config_digest,
            "selected_group_representative": (
                self.selected_group_representative
            ),
            "strategic_group_representatives": list(
                self.strategic_group_representatives
            ),
            "strategic_group_visits": list(self.strategic_group_visits),
            "strategic_groups": [
                list(group) for group in self.strategic_groups
            ],
            "trajectory_profile": self.trajectory_profile.value,
        }


@dataclass(frozen=True, slots=True)
class MinedBalancedGame:
    ordinal: int
    fixture_id: str
    trajectory_profile: TrajectoryProfile
    rows: tuple[BalancedMinerRow, ...]
    content_digest: str
    elapsed_seconds: float
    search_seconds: float
    response_request_count: int
    response_potential_evaluation_count: int
    peak_rss_bytes: int

    def artifact_data(self) -> dict[str, object]:
        content = {
            "fixture_id": self.fixture_id,
            "game_schema_version": BELIEF_GREEDY_MINER_GAME_VERSION,
            "ordinal": self.ordinal,
            "rows": [row.canonical_data() for row in self.rows],
            "trajectory_profile": self.trajectory_profile.value,
        }
        if _digest(content) != self.content_digest:
            raise BeliefGreedyMinerError("game content digest changed")
        return {
            "content": content,
            "content_digest": self.content_digest,
        }


def _deck_seed(config: BalancedMinerConfig, ordinal: int) -> str:
    return seed_hex(
        derive_seed(
            BELIEF_GREEDY_MINER_DECK_NAMESPACE,
            config.root_seed,
            str(ordinal),
            config.content_digest,
        )
    )


def _fixture_id(config: BalancedMinerConfig, ordinal: int) -> str:
    return seed_hex(
        derive_seed(
            BELIEF_GREEDY_MINER_FIXTURE_NAMESPACE,
            config.root_seed,
            str(ordinal),
            config.content_digest,
        )
    )


def _profile(ordinal: int) -> TrajectoryProfile:
    return TRAJECTORY_PROFILE_CYCLE[ordinal % len(TRAJECTORY_PROFILE_CYCLE)]


def _use_teacher_action(
    profile: TrajectoryProfile,
    ordinal: int,
    round_number: int,
    placement_number: int,
    information_fingerprint: str,
) -> bool:
    if profile is TrajectoryProfile.TEACHER:
        return True
    first_deviation = 1 + ((ordinal + round_number) % LEARNED_PLACEMENTS)
    if profile is TrajectoryProfile.ONE_DEVIATION:
        return placement_number != first_deviation
    second_deviation = 1 + (
        (ordinal + round_number + 3) % LEARNED_PLACEMENTS
    )
    if profile is TrajectoryProfile.TWO_DEVIATIONS:
        return placement_number not in {first_deviation, second_deviation}
    if profile is TrajectoryProfile.ALTERNATIVE:
        return False
    stream = Sha256CounterStream(
        derive_seed(
            BELIEF_GREEDY_MINER_TRAJECTORY_NAMESPACE,
            str(ordinal),
            str(round_number),
            str(placement_number),
            information_fingerprint,
        )
    )
    return stream.randbelow(2) == 0


def _alternative_representative(
    representatives: tuple[int, ...],
    teacher_representative: int,
    *,
    ordinal: int,
    round_number: int,
    placement_number: int,
    information_fingerprint: str,
) -> int:
    alternatives = tuple(
        representative
        for representative in representatives
        if representative != teacher_representative
    )
    if not alternatives:
        return teacher_representative
    stream = Sha256CounterStream(
        derive_seed(
            BELIEF_GREEDY_MINER_ALTERNATIVE_NAMESPACE,
            str(ordinal),
            str(round_number),
            str(placement_number),
            information_fingerprint,
        )
    )
    return alternatives[stream.randbelow(len(alternatives))]


def _row(
    information,
    groups,
    group_visits: tuple[int, ...],
    selected_representative: int,
    fixture_id: str,
    profile: TrajectoryProfile,
    search_digest: str,
) -> BalancedMinerRow:
    policy_input = policy_input_from_information_state(information)
    observation = tuple(
        bool(value) for value in policy_input.observation.tolist()
    )
    legal_mask = tuple(
        bool(value) for value in policy_input.legal_mask.flatten().tolist()
    )
    return BalancedMinerRow(
        information_state_fingerprint=information_state_fingerprint(
            information
        ),
        observation_packed=_pack_bools(
            observation,
            OBSERVATION_BIT_COUNT,
        ),
        legal_mask_packed=_pack_bools(
            legal_mask,
            LEGAL_MASK_BIT_COUNT,
        ),
        strategic_groups=tuple(
            group.member_action_indices for group in groups
        ),
        strategic_group_representatives=tuple(
            group.representative_action_index for group in groups
        ),
        strategic_group_visits=group_visits,
        selected_group_representative=selected_representative,
        round_number=information.round_number,
        placement_number=information.turn_number,
        player=information.player,
        dealer=information.dealer,
        fixture_id=fixture_id,
        trajectory_profile=profile,
        search_config_digest=search_digest,
    )


def _legal_move_for_action(
    state: EngineState,
    player: EnginePlayer,
    action_index: int,
):
    move = move_for_action_index(player, action_index)
    if move not in legal_moves(state, player):
        raise BeliefGreedyMinerError("trajectory selected an illegal action")
    return move


def mine_balanced_game(
    config: BalancedMinerConfig,
    ordinal: int,
) -> MinedBalancedGame:
    if type(ordinal) is not int or ordinal < 0:
        raise BeliefGreedyMinerError("game ordinal must be non-negative")
    torch.set_num_threads(1)
    started = time.perf_counter()
    search_seconds = 0.0
    response_requests = 0
    response_potentials = 0
    fixture_id = _fixture_id(config, ordinal)
    profile = _profile(ordinal)
    state = create_game(_deck_seed(config, ordinal))
    planner = BeliefGreedyInformationSetSearch(config.search_config)
    rows: list[BalancedMinerRow] = []

    while state.status is not EngineStatus.GAME_COMPLETE:
        if state.status is EngineStatus.ROUND_COMPLETE:
            state = advance_after_round(state)
            continue
        if state.active_player is None:
            raise BeliefGreedyMinerError("playing state has no active player")
        placement = len(state.current_round_moves) + 1
        if placement == LEARNED_PLACEMENTS + 1:
            moves = legal_moves(state, state.active_player)
            if len(moves) != 1:
                raise BeliefGreedyMinerError(
                    "the eighth placement must be forced"
                )
            state = apply_move(state, moves[0]).state
            continue

        information = information_state_from_engine(state)
        groups = strategic_action_groups(information, True)
        request = derive_belief_greedy_request_seed(
            fixture_id,
            information,
            config.search_config.digest,
        )
        result = planner.search(information, request)
        search_seconds += result.elapsed_seconds
        response_requests += result.response_request_count
        response_potentials += result.response_potential_evaluation_count
        representative = result.selected_representative_action_index
        representatives = tuple(
            group.representative_action_index for group in groups
        )
        diagnostic_by_representative = {
            diagnostic.group.representative_action_index: diagnostic
            for diagnostic in result.group_diagnostics
        }
        if set(diagnostic_by_representative) != set(representatives):
            raise BeliefGreedyMinerError(
                "teacher visit diagnostics do not match strategic groups"
            )
        group_visits = tuple(
            diagnostic_by_representative[representative].visits
            for representative in representatives
        )
        if (
            sum(group_visits) != config.outer_simulation_budget
            or any(type(visits) is not int or visits < 1 for visits in group_visits)
        ):
            raise BeliefGreedyMinerError(
                "teacher visits do not sum to the outer search budget"
            )
        if representative not in representatives:
            raise BeliefGreedyMinerError(
                "teacher selected an unknown strategic group"
            )
        rows.append(
            _row(
                information,
                groups,
                group_visits,
                representative,
                fixture_id,
                profile,
                config.search_config.digest,
            )
        )
        use_teacher = _use_teacher_action(
            profile,
            ordinal,
            state.round_number,
            placement,
            information_state_fingerprint(information),
        )
        if use_teacher:
            action_index = result.selected_action_index
        else:
            trajectory_representative = _alternative_representative(
                representatives,
                representative,
                ordinal=ordinal,
                round_number=state.round_number,
                placement_number=placement,
                information_fingerprint=information_state_fingerprint(
                    information
                ),
            )
            action_index = select_concrete_action_index(
                information,
                trajectory_representative,
                derive_seed(
                    BELIEF_GREEDY_MINER_DESTINATION_NAMESPACE,
                    str(ordinal),
                    str(state.round_number),
                    str(placement),
                    information_state_fingerprint(information),
                    str(trajectory_representative),
                ),
                True,
            )
        state = apply_move(
            state,
            _legal_move_for_action(
                state,
                information.player,
                action_index,
            ),
        ).state

    if len(rows) != ROWS_PER_GAME:
        raise BeliefGreedyMinerError(
            f"complete game produced {len(rows)} rows, expected {ROWS_PER_GAME}"
        )
    placement_counts = {
        placement: sum(row.placement_number == placement for row in rows)
        for placement in range(1, LEARNED_PLACEMENTS + 1)
    }
    if any(count != ROWS_PER_PLACEMENT_PER_GAME for count in placement_counts.values()):
        raise BeliefGreedyMinerError("game rows are not placement-balanced")
    content = {
        "fixture_id": fixture_id,
        "game_schema_version": BELIEF_GREEDY_MINER_GAME_VERSION,
        "ordinal": ordinal,
        "rows": [row.canonical_data() for row in rows],
        "trajectory_profile": profile.value,
    }
    return MinedBalancedGame(
        ordinal=ordinal,
        fixture_id=fixture_id,
        trajectory_profile=profile,
        rows=tuple(rows),
        content_digest=_digest(content),
        elapsed_seconds=time.perf_counter() - started,
        search_seconds=search_seconds,
        response_request_count=response_requests,
        response_potential_evaluation_count=response_potentials,
        peak_rss_bytes=_peak_rss_bytes(),
    )


def _worker(payload: tuple[BalancedMinerConfig, int]) -> MinedBalancedGame:
    config, ordinal = payload
    return mine_balanced_game(config, ordinal)


def mine_balanced_games(
    config: BalancedMinerConfig,
    ordinals: tuple[int, ...],
    output: Path | None = None,
) -> tuple[MinedBalancedGame, ...]:
    if len(ordinals) != len(set(ordinals)) or any(
        type(ordinal) is not int or ordinal < 0 for ordinal in ordinals
    ):
        raise BeliefGreedyMinerError(
            "game ordinals must be distinct non-negative integers"
        )
    if not ordinals:
        return ()
    with ProcessPoolExecutor(max_workers=config.worker_count) as executor:
        games = tuple(
            executor.map(
                _worker,
                ((config, ordinal) for ordinal in ordinals),
            )
        )
    ordered = tuple(sorted(games, key=lambda game: game.ordinal))
    if output is not None:
        game_dir = output / "games"
        for game in ordered:
            _atomic_json(
                game_dir / f"{game.ordinal:06d}.json",
                game.artifact_data(),
            )
        manifest_content = {
            "configuration_content_digest": config.content_digest,
            "game_digests": [game.content_digest for game in ordered],
            "game_ordinals": [game.ordinal for game in ordered],
            "manifest_schema_version": (
                BELIEF_GREEDY_MINER_MANIFEST_VERSION
            ),
            "row_count": sum(len(game.rows) for game in ordered),
        }
        _atomic_json(
            output / "manifest.json",
            {
                "content": manifest_content,
                "content_digest": _digest(manifest_content),
            },
        )
    return ordered


def _continuous_resolved_document(
    config: BalancedMinerConfig,
    *,
    minimum_free_disk_bytes: int,
    source: BeliefGreedySourceIdentity,
) -> dict[str, object]:
    content = {
        "belief_completion_count": config.belief_completion_count,
        "collector_schema_version": (
            BELIEF_GREEDY_CONTINUOUS_CONFIG_VERSION
        ),
        "configuration_content_digest": config.content_digest,
        "configuration_runtime_digest": config.runtime_digest,
        "minimum_free_disk_bytes": minimum_free_disk_bytes,
        "outer_simulation_budget": config.outer_simulation_budget,
        "root_seed": config.root_seed,
        "source_revision": source.revision,
        "source_tree_digest": source.tree_digest,
        "source_tree_schema_version": BELIEF_GREEDY_SOURCE_TREE_VERSION,
        "trajectory_profiles": [
            profile.value for profile in TRAJECTORY_PROFILE_CYCLE
        ],
        "worker_count": config.worker_count,
    }
    return {"content": content, "content_digest": _digest(content)}


def _read_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BeliefGreedyMinerError(f"cannot read {path}") from error


def _verified_document(path: Path) -> dict[str, object]:
    document = _read_json(path)
    if not isinstance(document, dict):
        raise BeliefGreedyMinerError(f"{path} is not a JSON object")
    content = document.get("content")
    digest = document.get("content_digest")
    if not isinstance(content, dict) or digest != _digest(content):
        raise BeliefGreedyMinerError(f"{path} failed digest verification")
    return document


def _verify_game_document(
    path: Path,
    *,
    expected_ordinal: int,
    expected_config_digest: str,
    expected_simulation_budget: int,
) -> dict[str, object]:
    document = _verified_document(path)
    content = document["content"]
    assert isinstance(content, dict)
    if content.get("game_schema_version") != BELIEF_GREEDY_MINER_GAME_VERSION:
        raise BeliefGreedyMinerError(f"{path} uses an incompatible game schema")
    if content.get("ordinal") != expected_ordinal:
        raise BeliefGreedyMinerError(f"{path} has the wrong game ordinal")
    rows = content.get("rows")
    if not isinstance(rows, list) or len(rows) != ROWS_PER_GAME:
        raise BeliefGreedyMinerError(f"{path} does not contain 42 rows")
    placements = {placement: 0 for placement in range(1, 8)}
    forbidden = {
        "action_values",
        "determinization",
        "engine_seed",
        "opponent_hand",
        "root_visits",
        "search_tree",
        "stock_order",
    }
    for row in rows:
        if not isinstance(row, dict):
            raise BeliefGreedyMinerError(f"{path} contains a malformed row")
        if forbidden.intersection(row):
            raise BeliefGreedyMinerError(f"{path} contains private search data")
        placement = row.get("placement_number")
        if placement not in placements:
            raise BeliefGreedyMinerError(f"{path} has an invalid placement")
        placements[placement] += 1
        if row.get("search_config_digest") != expected_config_digest:
            raise BeliefGreedyMinerError(f"{path} has configuration drift")
        representatives = row.get("strategic_group_representatives")
        visits = row.get("strategic_group_visits")
        selected = row.get("selected_group_representative")
        if (
            row.get("row_schema_version") != BELIEF_GREEDY_MINER_ROW_VERSION
            or not isinstance(representatives, list)
            or not isinstance(visits, list)
            or len(visits) != len(representatives)
            or any(type(value) is not int or value < 1 for value in visits)
            or sum(visits) != expected_simulation_budget
            or selected not in representatives
            or visits[representatives.index(selected)] != max(visits)
        ):
            raise BeliefGreedyMinerError(
                f"{path} has an invalid root-visit target"
            )
    if any(count != ROWS_PER_PLACEMENT_PER_GAME for count in placements.values()):
        raise BeliefGreedyMinerError(f"{path} is not placement-balanced")
    return document


def _manifest_document(
    config: BalancedMinerConfig,
    game_entries: list[dict[str, object]],
) -> dict[str, object]:
    content = {
        "configuration_content_digest": config.content_digest,
        "game_count": len(game_entries),
        "games": game_entries,
        "manifest_schema_version": (
            BELIEF_GREEDY_CONTINUOUS_MANIFEST_VERSION
        ),
        "placement_row_counts": {
            str(placement): len(game_entries) * ROWS_PER_PLACEMENT_PER_GAME
            for placement in range(1, LEARNED_PLACEMENTS + 1)
        },
        "row_count": len(game_entries) * ROWS_PER_GAME,
    }
    return {"content": content, "content_digest": _digest(content)}


def _load_continuous_entries(
    output: Path,
    config: BalancedMinerConfig,
) -> list[dict[str, object]]:
    manifest_path = output / "corpus-manifest.json"
    if not manifest_path.exists():
        return []
    document = _verified_document(manifest_path)
    content = document["content"]
    assert isinstance(content, dict)
    if (
        content.get("manifest_schema_version")
        != BELIEF_GREEDY_CONTINUOUS_MANIFEST_VERSION
        or content.get("configuration_content_digest")
        != config.content_digest
    ):
        raise BeliefGreedyMinerError("continuous manifest is incompatible")
    entries = content.get("games")
    if not isinstance(entries, list):
        raise BeliefGreedyMinerError("continuous manifest has no game list")
    verified: list[dict[str, object]] = []
    for ordinal, entry in enumerate(entries):
        if not isinstance(entry, dict) or entry.get("ordinal") != ordinal:
            raise BeliefGreedyMinerError("continuous game ordinals are not gap-free")
        relative_path = entry.get("path")
        if not isinstance(relative_path, str):
            raise BeliefGreedyMinerError("continuous game path is malformed")
        game = _verify_game_document(
            output / relative_path,
            expected_ordinal=ordinal,
            expected_config_digest=config.search_config.digest,
            expected_simulation_budget=config.outer_simulation_budget,
        )
        if game.get("content_digest") != entry.get("content_digest"):
            raise BeliefGreedyMinerError("manifest game digest changed")
        verified.append(entry)
    if content.get("game_count") != len(verified) or content.get(
        "row_count"
    ) != len(verified) * ROWS_PER_GAME:
        raise BeliefGreedyMinerError("continuous manifest totals are wrong")
    return verified


def inspect_continuous_corpus(output: Path) -> dict[str, object]:
    resolved = _verified_document(output / "resolved-config.json")
    manifest = _verified_document(output / "corpus-manifest.json")
    content = manifest["content"]
    assert isinstance(content, dict)
    return {
        "corpus_manifest_digest": manifest["content_digest"],
        "free_disk_bytes": shutil.disk_usage(output).free,
        "game_count": content["game_count"],
        "placement_row_counts": content["placement_row_counts"],
        "resolved_config_digest": resolved["content_digest"],
        "row_count": content["row_count"],
    }


def run_continuous_collection(
    config: BalancedMinerConfig,
    *,
    output: Path,
    minimum_free_disk_bytes: int,
    max_games: int | None = None,
) -> dict[str, object]:
    """Mine complete games until interrupted, capped, or disk-guarded."""

    if minimum_free_disk_bytes < 1:
        raise BeliefGreedyMinerError("continuous collection needs a disk floor")
    if max_games is not None and max_games < 1:
        raise BeliefGreedyMinerError("maximum games must be positive")
    source = resolve_source_identity()
    expected_resolved = _continuous_resolved_document(
        config,
        minimum_free_disk_bytes=minimum_free_disk_bytes,
        source=source,
    )
    output.mkdir(parents=True, exist_ok=True)
    resolved_path = output / "resolved-config.json"
    if resolved_path.exists():
        if _verified_document(resolved_path) != expected_resolved:
            raise BeliefGreedyMinerError(
                "resume source or resolved configuration differs"
            )
    else:
        if any(output.iterdir()):
            raise BeliefGreedyMinerError(
                "new continuous output directory is not empty"
            )
        _atomic_json(resolved_path, expected_resolved)
        _atomic_json(output / "corpus-manifest.json", _manifest_document(config, []))

    entries = _load_continuous_entries(output, config)
    stop_requested = False

    def request_stop(_signum, _frame) -> None:
        nonlocal stop_requested
        stop_requested = True

    previous_sigint = signal.signal(signal.SIGINT, request_stop)
    started = time.perf_counter()
    initial_games = len(entries)
    print(
        json.dumps(
            {
                "event": "collector_started" if not entries else "collector_resumed",
                "game_count": len(entries),
                "outer_simulation_budget": config.outer_simulation_budget,
                "belief_completion_count": config.belief_completion_count,
                "row_count": len(entries) * ROWS_PER_GAME,
                "source_tree_digest": source.tree_digest,
                "worker_count": config.worker_count,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    stop_reason = "interrupted"
    try:
        while not stop_requested:
            if max_games is not None and len(entries) >= max_games:
                stop_reason = "maximum-games"
                break
            free_bytes = shutil.disk_usage(output).free
            if free_bytes <= minimum_free_disk_bytes:
                stop_reason = "disk-floor"
                break
            remaining = (
                config.worker_count
                if max_games is None
                else min(config.worker_count, max_games - len(entries))
            )
            ordinals = tuple(range(len(entries), len(entries) + remaining))
            batch_started = time.perf_counter()
            games = mine_balanced_games(config, ordinals)
            batch_seconds = time.perf_counter() - batch_started
            for game in games:
                relative = f"games/{game.ordinal:06d}.json"
                path = output / relative
                _atomic_json(path, game.artifact_data())
                _verify_game_document(
                    path,
                    expected_ordinal=game.ordinal,
                    expected_config_digest=config.search_config.digest,
                    expected_simulation_budget=(
                        config.outer_simulation_budget
                    ),
                )
                entries.append(
                    {
                        "content_digest": game.content_digest,
                        "fixture_id": game.fixture_id,
                        "ordinal": game.ordinal,
                        "path": relative,
                        "row_count": len(game.rows),
                        "trajectory_profile": game.trajectory_profile.value,
                    }
                )
            _atomic_json(
                output / "corpus-manifest.json",
                _manifest_document(config, entries),
            )
            session_rows = (len(entries) - initial_games) * ROWS_PER_GAME
            session_seconds = time.perf_counter() - started
            print(
                json.dumps(
                    {
                        "batch_seconds": batch_seconds,
                        "event": "continuous_batch_sealed",
                        "free_disk_bytes": shutil.disk_usage(output).free,
                        "game_count": len(entries),
                        "placement_row_count": len(entries)
                        * ROWS_PER_PLACEMENT_PER_GAME,
                        "projected_24h_rows": (
                            session_rows / session_seconds * 86_400
                            if session_seconds > 0
                            else 0.0
                        ),
                        "row_count": len(entries) * ROWS_PER_GAME,
                        "rows_per_second": (
                            session_rows / session_seconds
                            if session_seconds > 0
                            else 0.0
                        ),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        if stop_requested:
            stop_reason = "interrupted"
    finally:
        signal.signal(signal.SIGINT, previous_sigint)
    state = {
        "event": "collector_stopped",
        "free_disk_bytes": shutil.disk_usage(output).free,
        "game_count": len(entries),
        "reason": stop_reason,
        "row_count": len(entries) * ROWS_PER_GAME,
    }
    _atomic_json(output / "state.json", state)
    print(json.dumps(state, sort_keys=True), flush=True)
    return inspect_continuous_corpus(output)


def _summary(
    config: BalancedMinerConfig,
    games: tuple[MinedBalancedGame, ...],
    wall_seconds: float,
) -> dict[str, object]:
    row_count = sum(len(game.rows) for game in games)
    rows_per_second = row_count / wall_seconds
    projected_total = rows_per_second * 86_400
    return {
        "belief_completion_count": config.belief_completion_count,
        "configuration_content_digest": config.content_digest,
        "game_count": len(games),
        "mean_game_seconds": sum(game.elapsed_seconds for game in games)
        / len(games),
        "outer_simulation_budget": config.outer_simulation_budget,
        "peak_worker_rss_bytes": max(game.peak_rss_bytes for game in games),
        "profile_counts": {
            profile.value: sum(
                game.trajectory_profile is profile for game in games
            )
            for profile in TRAJECTORY_PROFILE_CYCLE
        },
        "projected_24h_rows_by_placement": {
            str(placement): projected_total / LEARNED_PLACEMENTS
            for placement in range(1, LEARNED_PLACEMENTS + 1)
        },
        "projected_24h_total_rows": projected_total,
        "response_potential_evaluation_count": sum(
            game.response_potential_evaluation_count for game in games
        ),
        "response_request_count": sum(
            game.response_request_count for game in games
        ),
        "row_count": row_count,
        "rows_per_second": rows_per_second,
        "search_seconds": sum(game.search_seconds for game in games),
        "wall_seconds": wall_seconds,
        "worker_count": config.worker_count,
    }


def run_benchmark(
    *,
    output: Path,
    root_seed: str,
    game_count: int,
    worker_count: int,
    belief_completion_count: int,
) -> dict[str, object]:
    if game_count < len(TRAJECTORY_PROFILE_CYCLE):
        raise BeliefGreedyMinerError(
            "benchmark requires at least five games to cover every profile"
        )
    output.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, object]] = []
    for budget in (32, 128):
        config = BalancedMinerConfig(
            root_seed=root_seed,
            outer_simulation_budget=budget,
            belief_completion_count=belief_completion_count,
            worker_count=worker_count,
        )
        started = time.perf_counter()
        games = mine_balanced_games(
            config,
            tuple(range(game_count)),
            output / f"outer-{budget}",
        )
        summaries.append(
            _summary(config, games, time.perf_counter() - started)
        )
    result = {
        "benchmark_schema_version": BELIEF_GREEDY_MINER_SCHEMA_VERSION,
        "root_seed": root_seed,
        "summaries": summaries,
    }
    _atomic_json(output / "benchmark.json", result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dracula-belief-greedy-miner")
    subparsers = parser.add_subparsers(dest="command", required=True)
    mine = subparsers.add_parser("mine")
    benchmark = subparsers.add_parser("benchmark")
    continuous = subparsers.add_parser("continuous")
    inspect = subparsers.add_parser("inspect")
    for command in (mine, benchmark):
        command.add_argument("--output", required=True, type=Path)
        command.add_argument("--root-seed", required=True)
        command.add_argument("--games", type=int, default=10)
        command.add_argument("--workers", type=int, default=4)
        command.add_argument("--belief-completions", type=int, default=8)
    mine.add_argument("--outer-simulations", type=int, choices=(32, 128), default=32)
    continuous.add_argument("--output", required=True, type=Path)
    continuous.add_argument("--root-seed", required=True)
    continuous.add_argument("--workers", type=int, default=4)
    continuous.add_argument("--belief-completions", type=int, default=8)
    continuous.add_argument(
        "--outer-simulations", type=int, choices=(32, 128), default=128
    )
    continuous.add_argument("--minimum-free-disk-gib", type=float, default=1.0)
    continuous.add_argument("--max-games", type=int)
    inspect.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "inspect":
        result = inspect_continuous_corpus(args.output)
    elif args.command == "continuous":
        config = BalancedMinerConfig(
            root_seed=args.root_seed,
            outer_simulation_budget=args.outer_simulations,
            belief_completion_count=args.belief_completions,
            worker_count=args.workers,
        )
        result = run_continuous_collection(
            config,
            output=args.output,
            minimum_free_disk_bytes=int(args.minimum_free_disk_gib * (1 << 30)),
            max_games=args.max_games,
        )
    elif args.command == "benchmark":
        result = run_benchmark(
            output=args.output,
            root_seed=args.root_seed,
            game_count=args.games,
            worker_count=args.workers,
            belief_completion_count=args.belief_completions,
        )
    else:
        config = BalancedMinerConfig(
            root_seed=args.root_seed,
            outer_simulation_budget=args.outer_simulations,
            belief_completion_count=args.belief_completions,
            worker_count=args.workers,
        )
        started = time.perf_counter()
        games = mine_balanced_games(
            config,
            tuple(range(args.games)),
            args.output,
        )
        result = _summary(config, games, time.perf_counter() - started)
        _atomic_json(args.output / "summary.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = (
    "BELIEF_GREEDY_MINER_SCHEMA_VERSION",
    "BalancedMinerConfig",
    "BalancedMinerRow",
    "BeliefGreedySourceIdentity",
    "BeliefGreedyMinerError",
    "MinedBalancedGame",
    "ROWS_PER_GAME",
    "ROWS_PER_PLACEMENT_PER_GAME",
    "TRAJECTORY_PROFILE_CYCLE",
    "TrajectoryProfile",
    "main",
    "inspect_continuous_corpus",
    "mine_balanced_game",
    "mine_balanced_games",
    "run_benchmark",
    "run_continuous_collection",
    "resolve_source_identity",
)

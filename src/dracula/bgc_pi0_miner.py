"""Separately versioned balanced D1 collection with accepted pi0 responses."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import resource
import shutil
import signal
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import torch

from dracula.accepted_pi0 import (
    ACCEPTED_PI0_BGC_SEARCH_SCHEMA_VERSION,
    ACCEPTED_PI0_RESPONSE_CACHE_SCHEMA_VERSION,
    ACCEPTED_PI0_RESPONSE_SCHEMA_VERSION,
    AcceptedPi0BGCSearch,
    AcceptedPi0ContinuationAdapter,
)
from dracula.belief_greedy_miner import (
    BELIEF_GREEDY_SOURCE_TREE_VERSION,
    LEARNED_PLACEMENTS,
    LEGAL_MASK_BIT_COUNT,
    ROWS_PER_GAME,
    ROWS_PER_PLACEMENT_PER_GAME,
    TRAJECTORY_PROFILE_CYCLE,
    BeliefGreedySourceIdentity,
    TrajectoryProfile,
    resolve_source_identity,
)
from dracula.bgc_policy_evaluation import load_accepted_pi0_bundle
from dracula.bgc_policy_evaluation import AcceptedPi0Bundle
from dracula.bgc_policy import load_bgc_policy_artifact
from dracula.bgc_policy_model import (
    ACTION_SCHEMA_VERSION,
    OBSERVATION_SCHEMA_VERSION,
    OBSERVATION_SIZE,
    compact_action_index_from_legacy,
    compact_action_tensor_from_legacy,
    compact_observation_from_legacy,
)
from dracula.bridge import move_for_action_index
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
    BeliefGreedySearchConfig,
    derive_belief_greedy_request_seed,
    information_state_fingerprint,
    information_state_from_engine,
    policy_input_from_information_state,
    select_concrete_action_index,
    strategic_action_groups,
)

D1_COLLECTOR_SCHEMA_VERSION = "dracula-bgc-pi0-d1-collector-v2"
D1_DATASET_SCHEMA_VERSION = "dracula-bgc-pi0-d1-dataset-v2"
D1_ROW_SCHEMA_VERSION = "dracula-bgc-pi0-d1-row-v2"
D1_GAME_SCHEMA_VERSION = "dracula-bgc-pi0-d1-game-v2"
D1_MANIFEST_SCHEMA_VERSION = "dracula-bgc-pi0-d1-manifest-v2"
D1_RESOLVED_CONFIG_SCHEMA_VERSION = "dracula-bgc-pi0-d1-config-v2"
D1_REPORT_SCHEMA_VERSION = "dracula-bgc-pi0-d1-report-v1"
D1_SEARCH_SCHEMA_VERSION = ACCEPTED_PI0_BGC_SEARCH_SCHEMA_VERSION
D1_RESPONSE_SCHEMA_VERSION = ACCEPTED_PI0_RESPONSE_SCHEMA_VERSION
D1_CACHE_SCHEMA_VERSION = ACCEPTED_PI0_RESPONSE_CACHE_SCHEMA_VERSION
D1_DECK_NAMESPACE = "dracula-bgc-pi0-d1-deck-v1"
D1_FIXTURE_NAMESPACE = "dracula-bgc-pi0-d1-fixture-v1"
D1_TRAJECTORY_NAMESPACE = "dracula-bgc-pi0-d1-trajectory-v1"
D1_ALTERNATIVE_NAMESPACE = "dracula-bgc-pi0-d1-alternative-v1"
D1_DESTINATION_NAMESPACE = "dracula-bgc-pi0-d1-destination-v1"
D1_OUTER_SIMULATIONS = 128
D1_BASE_BELIEF_COMPLETIONS = 8
D1_USER_SELECTION_SCHEMA_VERSION = "dracula-d1-user-selected-pi0-v1"
D1_USER_SELECTION_FILENAME = "d1-selection-manifest.json"
D1_USER_SELECTED_CHECKPOINT = "selected-pi0.pt"


class D1MinerError(ValueError):
    """A D1 configuration, accepted policy, or artifact is invalid."""


def _load_d1_pi0_bundle(path: str | Path) -> AcceptedPi0Bundle:
    """Load strict acceptance evidence or an explicit, digest-bound D1 selection."""
    directory = Path(path).expanduser().resolve()
    selection_path = directory / D1_USER_SELECTION_FILENAME
    if not selection_path.exists():
        return load_accepted_pi0_bundle(directory)
    document = _verified_document(selection_path)
    content = document["content"]
    assert isinstance(content, dict)
    required = {
        "artifact_digest",
        "controller_evidence_content_digest",
        "controller_evidence_file_digest",
        "controller_evidence_path",
        "selection_basis",
        "selection_schema_version",
        "source_revision",
        "source_tree_digest",
    }
    if set(content) != required:
        raise D1MinerError("D1 user-selection manifest is incompatible")
    if (
        content["selection_schema_version"] != D1_USER_SELECTION_SCHEMA_VERSION
        or content["selection_basis"] != "explicit-user-d1-authorization-v1"
    ):
        raise D1MinerError("D1 user-selection basis differs")
    checkpoint_path = directory / D1_USER_SELECTED_CHECKPOINT
    checkpoint_digest = _file_digest(checkpoint_path)
    if checkpoint_digest != content["artifact_digest"]:
        raise D1MinerError("D1 selected checkpoint digest differs")
    evidence_path = Path(str(content["controller_evidence_path"])).expanduser().resolve()
    if _file_digest(evidence_path) != content["controller_evidence_file_digest"]:
        raise D1MinerError("D1 controller evidence file digest differs")
    evidence = _verified_controller_evidence(evidence_path)
    evidence_content = evidence["content"]
    if (
        evidence["content_digest"] != content["controller_evidence_content_digest"]
        or evidence_content.get("comparison_id") != "pi0-bgc-vs-base-bgc"
        or evidence_content.get("artifact_digest") != checkpoint_digest
        or evidence_content.get("complete") is not True
        or not isinstance(evidence_content.get("completed_deck_count"), int)
        or evidence_content["completed_deck_count"] < 1
    ):
        raise D1MinerError("D1 controller evidence is incompatible")
    source = resolve_source_identity()
    if (
        content["source_revision"] != source.revision
        or content["source_tree_digest"] != source.tree_digest
    ):
        raise D1MinerError("D1 selected source identity differs")
    try:
        artifact = load_bgc_policy_artifact(checkpoint_path)
    except Exception as error:
        raise D1MinerError("D1 selected pi0 artifact is invalid") from error
    return AcceptedPi0Bundle(
        directory=directory,
        checkpoint_path=checkpoint_path,
        checkpoint_digest=checkpoint_digest,
        acceptance_content_digest=document["content_digest"],
        acceptance_file_digest=_file_digest(selection_path),
        artifact=artifact,
    )


def create_d1_user_selection(
    *, candidate: Path, controller_evidence: Path, output: Path
) -> dict[str, object]:
    """Seal the user's explicit D1 selection without claiming protocol acceptance."""
    candidate = candidate.expanduser().resolve()
    controller_evidence = controller_evidence.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise D1MinerError("D1 selection output directory is not empty")
    artifact_digest = _file_digest(candidate)
    try:
        load_bgc_policy_artifact(candidate)
    except Exception as error:
        raise D1MinerError("D1 selected pi0 artifact is invalid") from error
    evidence = _verified_controller_evidence(controller_evidence)
    evidence_content = evidence["content"]
    if (
        evidence_content.get("comparison_id") != "pi0-bgc-vs-base-bgc"
        or evidence_content.get("artifact_digest") != artifact_digest
        or evidence_content.get("complete") is not True
        or not isinstance(evidence_content.get("completed_deck_count"), int)
        or evidence_content["completed_deck_count"] < 1
    ):
        raise D1MinerError("D1 selection requires completed matching controller evidence")
    source = resolve_source_identity()
    content = {
        "artifact_digest": artifact_digest,
        "controller_evidence_content_digest": evidence["content_digest"],
        "controller_evidence_file_digest": _file_digest(controller_evidence),
        "controller_evidence_path": str(controller_evidence),
        "selection_basis": "explicit-user-d1-authorization-v1",
        "selection_schema_version": D1_USER_SELECTION_SCHEMA_VERSION,
        "source_revision": source.revision,
        "source_tree_digest": source.tree_digest,
    }
    document = {"content": content, "content_digest": _digest(content)}
    output.mkdir(parents=True, exist_ok=True)
    temporary = output / f".{D1_USER_SELECTED_CHECKPOINT}.{os.getpid()}.partial"
    try:
        shutil.copyfile(candidate, temporary)
        if _file_digest(temporary) != artifact_digest:
            raise D1MinerError("D1 selected checkpoint copy differs")
        os.replace(temporary, output / D1_USER_SELECTED_CHECKPOINT)
        _atomic_json(output / D1_USER_SELECTION_FILENAME, document)
        bundle = _load_d1_pi0_bundle(output)
    except Exception:
        temporary.unlink(missing_ok=True)
        shutil.rmtree(output, ignore_errors=True)
        raise
    return {
        "artifact_digest": bundle.checkpoint_digest,
        "controller_evidence_content_digest": evidence["content_digest"],
        "selection_manifest_digest": bundle.acceptance_content_digest,
        "selection_path": str(output.resolve()),
    }


def _canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise D1MinerError("D1 artifact is not canonical JSON") from error


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    try:
        with temporary.open("wb") as stream:
            stream.write(_canonical_bytes(value) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise D1MinerError(f"cannot read D1 artifact: {path}") from error


def _verified_document(path: Path) -> dict[str, object]:
    document = _read_json(path)
    if (
        not isinstance(document, dict)
        or set(document) != {"content", "content_digest"}
        or not isinstance(document["content"], dict)
        or document["content_digest"] != _digest(document["content"])
    ):
        raise D1MinerError(f"D1 artifact failed digest verification: {path}")
    return document


def _verified_controller_evidence(path: Path) -> dict[str, object]:
    document = _read_json(path)
    if (
        not isinstance(document, dict)
        or set(document)
        != {"format_version", "content", "content_digest", "file_digest"}
        or document["format_version"] != "dracula-pi0-controller-progress-v1"
        or not isinstance(document["content"], dict)
        or document["content_digest"] != _digest(document["content"])
        or document["file_digest"]
        != _digest(
            {
                "format_version": document["format_version"],
                "content": document["content"],
                "content_digest": document["content_digest"],
            }
        )
    ):
        raise D1MinerError("D1 controller evidence failed digest verification")
    return document


def _pack_bools(values: tuple[bool, ...], expected: int) -> bytes:
    if len(values) != expected or any(type(value) is not bool for value in values):
        raise D1MinerError("D1 packed Boolean shape differs")
    packed = bytearray((expected + 7) // 8)
    for index, value in enumerate(values):
        if value:
            packed[index // 8] |= 1 << (7 - index % 8)
    return bytes(packed)


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


@dataclass(frozen=True, slots=True)
class D1MinerConfig:
    root_seed: str
    accepted_pi0_directory: str
    accepted_checkpoint_digest: str
    acceptance_manifest_digest: str
    controller_identity_digest: str
    worker_count: int = 4

    @classmethod
    def resolve(
        cls,
        *,
        root_seed: str,
        accepted_pi0_directory: str | Path,
        worker_count: int = 4,
    ) -> D1MinerConfig:
        try:
            bundle = _load_d1_pi0_bundle(accepted_pi0_directory)
        except Exception as error:
            raise D1MinerError("D1 requires a valid accepted pi0 bundle") from error
        adapter = AcceptedPi0ContinuationAdapter(bundle)
        controller_digest = AcceptedPi0BGCSearch(
            adapter,
            BeliefGreedySearchConfig(
                outer_simulation_budget=D1_OUTER_SIMULATIONS,
                belief_completion_count=D1_BASE_BELIEF_COMPLETIONS,
            ),
        ).controller_digest
        return cls(
            root_seed=root_seed,
            accepted_pi0_directory=str(bundle.directory),
            accepted_checkpoint_digest=bundle.checkpoint_digest,
            acceptance_manifest_digest=bundle.acceptance_content_digest,
            controller_identity_digest=controller_digest,
            worker_count=worker_count,
        )

    def __post_init__(self) -> None:
        if not isinstance(self.root_seed, str) or not self.root_seed:
            raise D1MinerError("D1 root seed must be nonempty")
        if type(self.worker_count) is not int or not 1 <= self.worker_count <= 8:
            raise D1MinerError("D1 worker count must be between one and eight")
        try:
            bundle = _load_d1_pi0_bundle(self.accepted_pi0_directory)
        except Exception as error:
            raise D1MinerError("D1 requires a valid accepted pi0 bundle") from error
        if (
            bundle.checkpoint_digest != self.accepted_checkpoint_digest
            or bundle.acceptance_content_digest != self.acceptance_manifest_digest
        ):
            raise D1MinerError("D1 accepted pi0 identity differs")
        adapter = AcceptedPi0ContinuationAdapter(bundle)
        actual_controller_digest = AcceptedPi0BGCSearch(
            adapter,
            BeliefGreedySearchConfig(
                outer_simulation_budget=D1_OUTER_SIMULATIONS,
                belief_completion_count=D1_BASE_BELIEF_COMPLETIONS,
            ),
        ).controller_digest
        if actual_controller_digest != self.controller_identity_digest:
            raise D1MinerError("D1 controller identity differs")

    @property
    def search_config(self) -> BeliefGreedySearchConfig:
        return BeliefGreedySearchConfig(
            outer_simulation_budget=D1_OUTER_SIMULATIONS,
            belief_completion_count=D1_BASE_BELIEF_COMPLETIONS,
        )

    @property
    def controller_digest(self) -> str:
        return self.controller_identity_digest

    @property
    def content_digest(self) -> str:
        return _digest(
            {
                "acceptance_manifest_digest": self.acceptance_manifest_digest,
                "accepted_checkpoint_digest": self.accepted_checkpoint_digest,
                "cache_schema_version": D1_CACHE_SCHEMA_VERSION,
                "collector_schema_version": D1_COLLECTOR_SCHEMA_VERSION,
                "controller_digest": self.controller_digest,
                "dataset_schema_version": D1_DATASET_SCHEMA_VERSION,
                "game_schema_version": D1_GAME_SCHEMA_VERSION,
                "manifest_schema_version": D1_MANIFEST_SCHEMA_VERSION,
                "response_schema_version": D1_RESPONSE_SCHEMA_VERSION,
                "root_seed": self.root_seed,
                "row_schema_version": D1_ROW_SCHEMA_VERSION,
                "search_config_digest": self.search_config.digest,
                "search_schema_version": D1_SEARCH_SCHEMA_VERSION,
                "trajectory_profiles": [
                    profile.value for profile in TRAJECTORY_PROFILE_CYCLE
                ],
            }
        )

    @property
    def runtime_digest(self) -> str:
        return _digest(
            {"content_digest": self.content_digest, "worker_count": self.worker_count}
        )


@dataclass(frozen=True, slots=True)
class D1Row:
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
    controller_digest: str
    accepted_checkpoint_digest: str
    acceptance_manifest_digest: str

    def canonical_data(self) -> dict[str, object]:
        return {
            "acceptance_manifest_digest": self.acceptance_manifest_digest,
            "accepted_checkpoint_digest": self.accepted_checkpoint_digest,
            "cache_schema_version": D1_CACHE_SCHEMA_VERSION,
            "controller_digest": self.controller_digest,
            "dataset_schema_version": D1_DATASET_SCHEMA_VERSION,
            "action_schema_version": ACTION_SCHEMA_VERSION,
            "dealer": self.dealer.value,
            "fixture_id": self.fixture_id,
            "information_state_fingerprint": self.information_state_fingerprint,
            "legal_mask_packed": base64.b64encode(self.legal_mask_packed).decode(
                "ascii"
            ),
            "observation_packed": base64.b64encode(self.observation_packed).decode(
                "ascii"
            ),
            "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
            "placement_number": self.placement_number,
            "player": self.player.value,
            "response_schema_version": D1_RESPONSE_SCHEMA_VERSION,
            "round_number": self.round_number,
            "row_schema_version": D1_ROW_SCHEMA_VERSION,
            "search_config_digest": self.search_config_digest,
            "search_schema_version": D1_SEARCH_SCHEMA_VERSION,
            "selected_group_representative": self.selected_group_representative,
            "strategic_group_representatives": list(
                self.strategic_group_representatives
            ),
            "strategic_group_visits": list(self.strategic_group_visits),
            "strategic_groups": [list(group) for group in self.strategic_groups],
            "trajectory_profile": self.trajectory_profile.value,
        }


@dataclass(frozen=True, slots=True)
class MinedD1Game:
    ordinal: int
    fixture_id: str
    trajectory_profile: TrajectoryProfile
    rows: tuple[D1Row, ...]
    content_digest: str
    elapsed_seconds: float
    search_seconds: float
    response_request_count: int
    response_model_inference_count: int
    peak_rss_bytes: int

    def artifact_data(self, config: D1MinerConfig) -> dict[str, object]:
        content = {
            "acceptance_manifest_digest": config.acceptance_manifest_digest,
            "accepted_checkpoint_digest": config.accepted_checkpoint_digest,
            "cache_schema_version": D1_CACHE_SCHEMA_VERSION,
            "configuration_content_digest": config.content_digest,
            "controller_digest": config.controller_digest,
            "dataset_schema_version": D1_DATASET_SCHEMA_VERSION,
            "fixture_id": self.fixture_id,
            "game_schema_version": D1_GAME_SCHEMA_VERSION,
            "ordinal": self.ordinal,
            "response_schema_version": D1_RESPONSE_SCHEMA_VERSION,
            "rows": [row.canonical_data() for row in self.rows],
            "search_schema_version": D1_SEARCH_SCHEMA_VERSION,
            "trajectory_profile": self.trajectory_profile.value,
        }
        if _digest(content) != self.content_digest:
            raise D1MinerError("D1 game content digest changed")
        return {"content": content, "content_digest": self.content_digest}


def _deck_seed(config: D1MinerConfig, ordinal: int) -> str:
    return seed_hex(
        derive_seed(D1_DECK_NAMESPACE, config.root_seed, str(ordinal), config.content_digest)
    )


def _fixture_id(config: D1MinerConfig, ordinal: int) -> str:
    return seed_hex(
        derive_seed(
            D1_FIXTURE_NAMESPACE, config.root_seed, str(ordinal), config.content_digest
        )
    )


def _profile(ordinal: int) -> TrajectoryProfile:
    return TRAJECTORY_PROFILE_CYCLE[ordinal % len(TRAJECTORY_PROFILE_CYCLE)]


def _use_teacher_action(
    profile: TrajectoryProfile,
    ordinal: int,
    round_number: int,
    placement_number: int,
    fingerprint: str,
) -> bool:
    if profile is TrajectoryProfile.TEACHER:
        return True
    first = 1 + ((ordinal + round_number) % LEARNED_PLACEMENTS)
    if profile is TrajectoryProfile.ONE_DEVIATION:
        return placement_number != first
    second = 1 + ((ordinal + round_number + 3) % LEARNED_PLACEMENTS)
    if profile is TrajectoryProfile.TWO_DEVIATIONS:
        return placement_number not in {first, second}
    if profile is TrajectoryProfile.ALTERNATIVE:
        return False
    stream = Sha256CounterStream(
        derive_seed(
            D1_TRAJECTORY_NAMESPACE,
            str(ordinal),
            str(round_number),
            str(placement_number),
            fingerprint,
        )
    )
    return stream.randbelow(2) == 0


def _alternative_representative(
    representatives: tuple[int, ...],
    selected: int,
    *,
    ordinal: int,
    round_number: int,
    placement_number: int,
    fingerprint: str,
) -> int:
    alternatives = tuple(value for value in representatives if value != selected)
    if not alternatives:
        return selected
    stream = Sha256CounterStream(
        derive_seed(
            D1_ALTERNATIVE_NAMESPACE,
            str(ordinal),
            str(round_number),
            str(placement_number),
            fingerprint,
        )
    )
    return alternatives[stream.randbelow(len(alternatives))]


def _row(
    information,
    groups,
    visits: tuple[int, ...],
    selected: int,
    fixture_id: str,
    profile: TrajectoryProfile,
    config: D1MinerConfig,
) -> D1Row:
    policy_input = policy_input_from_information_state(information)
    observation = compact_observation_from_legacy(policy_input.observation)
    legal_mask = compact_action_tensor_from_legacy(
        policy_input.observation, policy_input.legal_mask
    )
    translated = sorted(
        (
            compact_action_index_from_legacy(
                policy_input.observation, group.representative_action_index
            ),
            tuple(
                compact_action_index_from_legacy(policy_input.observation, member)
                for member in group.member_action_indices
            ),
            visit_count,
        )
        for group, visit_count in zip(groups, visits, strict=True)
    )
    compact_selected = compact_action_index_from_legacy(
        policy_input.observation, selected
    )
    return D1Row(
        information_state_fingerprint=information_state_fingerprint(information),
        observation_packed=_pack_bools(
            tuple(bool(value) for value in observation.tolist()),
            OBSERVATION_SIZE,
        ),
        legal_mask_packed=_pack_bools(
            tuple(bool(value) for value in legal_mask.flatten().tolist()),
            LEGAL_MASK_BIT_COUNT,
        ),
        strategic_groups=tuple(value[1] for value in translated),
        strategic_group_representatives=tuple(value[0] for value in translated),
        strategic_group_visits=tuple(value[2] for value in translated),
        selected_group_representative=compact_selected,
        round_number=information.round_number,
        placement_number=information.turn_number,
        player=information.player,
        dealer=information.dealer,
        fixture_id=fixture_id,
        trajectory_profile=profile,
        search_config_digest=config.search_config.digest,
        controller_digest=config.controller_digest,
        accepted_checkpoint_digest=config.accepted_checkpoint_digest,
        acceptance_manifest_digest=config.acceptance_manifest_digest,
    )


def _legal_move(state: EngineState, player: EnginePlayer, action_index: int):
    move = move_for_action_index(player, action_index)
    if move not in legal_moves(state, player):
        raise D1MinerError("D1 trajectory selected an illegal action")
    return move


def mine_d1_game(config: D1MinerConfig, ordinal: int) -> MinedD1Game:
    if type(ordinal) is not int or ordinal < 0:
        raise D1MinerError("D1 ordinal must be non-negative")
    torch.set_num_threads(1)
    bundle = _load_d1_pi0_bundle(config.accepted_pi0_directory)
    if (
        bundle.checkpoint_digest != config.accepted_checkpoint_digest
        or bundle.acceptance_content_digest != config.acceptance_manifest_digest
    ):
        raise D1MinerError("accepted pi0 changed before D1 game")
    adapter = AcceptedPi0ContinuationAdapter(bundle)
    planner = AcceptedPi0BGCSearch(adapter, config.search_config)
    started = time.perf_counter()
    search_seconds = 0.0
    response_requests = 0
    response_inferences = 0
    fixture_id = _fixture_id(config, ordinal)
    profile = _profile(ordinal)
    state = create_game(_deck_seed(config, ordinal))
    rows: list[D1Row] = []
    while state.status is not EngineStatus.GAME_COMPLETE:
        if state.status is EngineStatus.ROUND_COMPLETE:
            state = advance_after_round(state)
            continue
        if state.active_player is None:
            raise D1MinerError("D1 playing state has no active player")
        placement = len(state.current_round_moves) + 1
        if placement == LEARNED_PLACEMENTS + 1:
            moves = legal_moves(state, state.active_player)
            if len(moves) != 1:
                raise D1MinerError("D1 eighth placement is not forced")
            state = apply_move(state, moves[0]).state
            continue
        information = information_state_from_engine(state)
        groups = strategic_action_groups(information, True)
        result = planner.search(
            information,
            derive_belief_greedy_request_seed(
                fixture_id, information, config.search_config.digest
            ),
        )
        if result.config_digest != config.search_config.digest:
            raise D1MinerError("D1 root search configuration drifted")
        search_seconds += result.elapsed_seconds
        response_requests += result.response_request_count
        response_inferences += result.unique_response_evaluation_count
        representatives = tuple(
            group.representative_action_index for group in groups
        )
        diagnostics = {
            item.group.representative_action_index: item
            for item in result.group_diagnostics
        }
        visits = tuple(diagnostics[value].visits for value in representatives)
        selected = result.selected_representative_action_index
        if (
            set(diagnostics) != set(representatives)
            or sum(visits) != D1_OUTER_SIMULATIONS
            or any(type(value) is not int or value < 1 for value in visits)
            or selected not in representatives
            or visits[representatives.index(selected)] != max(visits)
        ):
            raise D1MinerError("D1 root visit target is invalid")
        rows.append(_row(information, groups, visits, selected, fixture_id, profile, config))
        fingerprint = information_state_fingerprint(information)
        if _use_teacher_action(
            profile, ordinal, state.round_number, placement, fingerprint
        ):
            action_index = result.selected_action_index
        else:
            trajectory_representative = _alternative_representative(
                representatives,
                selected,
                ordinal=ordinal,
                round_number=state.round_number,
                placement_number=placement,
                fingerprint=fingerprint,
            )
            action_index = select_concrete_action_index(
                information,
                trajectory_representative,
                derive_seed(
                    D1_DESTINATION_NAMESPACE,
                    str(ordinal),
                    str(state.round_number),
                    str(placement),
                    fingerprint,
                    str(trajectory_representative),
                ),
                True,
            )
        state = apply_move(
            state, _legal_move(state, information.player, action_index)
        ).state
    if len(rows) != ROWS_PER_GAME or any(
        sum(row.placement_number == placement for row in rows)
        != ROWS_PER_PLACEMENT_PER_GAME
        for placement in range(1, LEARNED_PLACEMENTS + 1)
    ):
        raise D1MinerError("D1 complete game is not placement-balanced")
    content = {
        "acceptance_manifest_digest": config.acceptance_manifest_digest,
        "accepted_checkpoint_digest": config.accepted_checkpoint_digest,
        "cache_schema_version": D1_CACHE_SCHEMA_VERSION,
        "configuration_content_digest": config.content_digest,
        "controller_digest": config.controller_digest,
        "dataset_schema_version": D1_DATASET_SCHEMA_VERSION,
        "fixture_id": fixture_id,
        "game_schema_version": D1_GAME_SCHEMA_VERSION,
        "ordinal": ordinal,
        "response_schema_version": D1_RESPONSE_SCHEMA_VERSION,
        "rows": [row.canonical_data() for row in rows],
        "search_schema_version": D1_SEARCH_SCHEMA_VERSION,
        "trajectory_profile": profile.value,
    }
    return MinedD1Game(
        ordinal=ordinal,
        fixture_id=fixture_id,
        trajectory_profile=profile,
        rows=tuple(rows),
        content_digest=_digest(content),
        elapsed_seconds=time.perf_counter() - started,
        search_seconds=search_seconds,
        response_request_count=response_requests,
        response_model_inference_count=response_inferences,
        peak_rss_bytes=_peak_rss_bytes(),
    )


def _worker(payload: tuple[D1MinerConfig, int]) -> MinedD1Game:
    config, ordinal = payload
    return mine_d1_game(config, ordinal)


def mine_d1_games(
    config: D1MinerConfig, ordinals: tuple[int, ...]
) -> tuple[MinedD1Game, ...]:
    if not ordinals:
        return ()
    if len(ordinals) != len(set(ordinals)) or any(
        type(ordinal) is not int or ordinal < 0 for ordinal in ordinals
    ):
        raise D1MinerError("D1 ordinals must be distinct non-negative integers")
    with ProcessPoolExecutor(max_workers=config.worker_count) as executor:
        games = tuple(
            executor.map(_worker, ((config, ordinal) for ordinal in ordinals))
        )
    return tuple(sorted(games, key=lambda game: game.ordinal))


def _resolved_document(
    config: D1MinerConfig,
    *,
    minimum_free_disk_bytes: int,
    source: BeliefGreedySourceIdentity,
) -> dict[str, object]:
    content = {
        "acceptance_manifest_digest": config.acceptance_manifest_digest,
        "accepted_checkpoint_digest": config.accepted_checkpoint_digest,
        "accepted_pi0_directory": config.accepted_pi0_directory,
        "base_belief_completion_count": D1_BASE_BELIEF_COMPLETIONS,
        "cache_schema_version": D1_CACHE_SCHEMA_VERSION,
        "collector_schema_version": D1_RESOLVED_CONFIG_SCHEMA_VERSION,
        "configuration_content_digest": config.content_digest,
        "configuration_runtime_digest": config.runtime_digest,
        "controller_digest": config.controller_digest,
        "dataset_schema_version": D1_DATASET_SCHEMA_VERSION,
        "game_schema_version": D1_GAME_SCHEMA_VERSION,
        "manifest_schema_version": D1_MANIFEST_SCHEMA_VERSION,
        "minimum_free_disk_bytes": minimum_free_disk_bytes,
        "outer_simulation_budget": D1_OUTER_SIMULATIONS,
        "report_schema_version": D1_REPORT_SCHEMA_VERSION,
        "response_schema_version": D1_RESPONSE_SCHEMA_VERSION,
        "root_seed": config.root_seed,
        "row_schema_version": D1_ROW_SCHEMA_VERSION,
        "search_config_digest": config.search_config.digest,
        "search_schema_version": D1_SEARCH_SCHEMA_VERSION,
        "source_revision": source.revision,
        "source_tree_digest": source.tree_digest,
        "source_tree_schema_version": BELIEF_GREEDY_SOURCE_TREE_VERSION,
        "trajectory_profiles": [
            profile.value for profile in TRAJECTORY_PROFILE_CYCLE
        ],
        "worker_count": config.worker_count,
    }
    return {"content": content, "content_digest": _digest(content)}


def _manifest_document(
    config: D1MinerConfig, entries: list[dict[str, object]]
) -> dict[str, object]:
    content = {
        "acceptance_manifest_digest": config.acceptance_manifest_digest,
        "accepted_checkpoint_digest": config.accepted_checkpoint_digest,
        "cache_schema_version": D1_CACHE_SCHEMA_VERSION,
        "configuration_content_digest": config.content_digest,
        "controller_digest": config.controller_digest,
        "dataset_schema_version": D1_DATASET_SCHEMA_VERSION,
        "game_count": len(entries),
        "games": entries,
        "manifest_schema_version": D1_MANIFEST_SCHEMA_VERSION,
        "placement_row_counts": {
            str(placement): len(entries) * ROWS_PER_PLACEMENT_PER_GAME
            for placement in range(1, LEARNED_PLACEMENTS + 1)
        },
        "response_schema_version": D1_RESPONSE_SCHEMA_VERSION,
        "row_count": len(entries) * ROWS_PER_GAME,
        "search_schema_version": D1_SEARCH_SCHEMA_VERSION,
    }
    return {"content": content, "content_digest": _digest(content)}


def initialize_d1_corpus(
    config: D1MinerConfig,
    *,
    output: Path,
    minimum_free_disk_bytes: int,
) -> dict[str, object]:
    if minimum_free_disk_bytes < 1:
        raise D1MinerError("D1 requires a positive disk guard")
    _load_d1_pi0_bundle(config.accepted_pi0_directory)
    source = resolve_source_identity()
    resolved = _resolved_document(
        config, minimum_free_disk_bytes=minimum_free_disk_bytes, source=source
    )
    if output.exists() and any(output.iterdir()):
        raise D1MinerError("new D1 output directory is not empty")
    output.mkdir(parents=True, exist_ok=True)
    try:
        _atomic_json(output / "resolved-config.json", resolved)
        _atomic_json(output / "corpus-manifest.json", _manifest_document(config, []))
    except Exception:
        shutil.rmtree(output, ignore_errors=True)
        raise
    return inspect_d1_corpus(output)


def _verify_game(
    path: Path,
    *,
    config: D1MinerConfig,
    expected_ordinal: int,
) -> dict[str, object]:
    document = _verified_document(path)
    content = document["content"]
    assert isinstance(content, dict)
    required_identity = {
        "acceptance_manifest_digest": config.acceptance_manifest_digest,
        "accepted_checkpoint_digest": config.accepted_checkpoint_digest,
        "cache_schema_version": D1_CACHE_SCHEMA_VERSION,
        "configuration_content_digest": config.content_digest,
        "controller_digest": config.controller_digest,
        "dataset_schema_version": D1_DATASET_SCHEMA_VERSION,
        "game_schema_version": D1_GAME_SCHEMA_VERSION,
        "response_schema_version": D1_RESPONSE_SCHEMA_VERSION,
        "search_schema_version": D1_SEARCH_SCHEMA_VERSION,
    }
    if content.get("ordinal") != expected_ordinal or any(
        content.get(key) != value for key, value in required_identity.items()
    ):
        raise D1MinerError("D1 game identity differs")
    rows = content.get("rows")
    if not isinstance(rows, list) or len(rows) != ROWS_PER_GAME:
        raise D1MinerError("D1 game does not contain 42 rows")
    placements = {placement: 0 for placement in range(1, 8)}
    forbidden = {
        "action_values",
        "authoritative_state",
        "determinization",
        "engine_seed",
        "model_state",
        "opponent_hand",
        "root_search_tree",
        "search_tree",
        "stock_order",
    }
    for row in rows:
        if not isinstance(row, dict) or forbidden.intersection(row):
            raise D1MinerError("D1 row contains private or malformed data")
        row_identity = {
            key: value
            for key, value in required_identity.items()
            if key not in {"configuration_content_digest", "game_schema_version"}
        }
        if any(row.get(key) != value for key, value in row_identity.items()):
            raise D1MinerError("D1 row identity differs")
        placement = row.get("placement_number")
        if placement not in placements:
            raise D1MinerError("D1 row placement is invalid")
        placements[placement] += 1
        representatives = row.get("strategic_group_representatives")
        visits = row.get("strategic_group_visits")
        selected = row.get("selected_group_representative")
        if (
            row.get("row_schema_version") != D1_ROW_SCHEMA_VERSION
            or row.get("search_config_digest") != config.search_config.digest
            or not isinstance(representatives, list)
            or not isinstance(visits, list)
            or len(representatives) != len(visits)
            or any(type(value) is not int or value < 1 for value in visits)
            or sum(visits) != D1_OUTER_SIMULATIONS
            or selected not in representatives
            or visits[representatives.index(selected)] != max(visits)
        ):
            raise D1MinerError("D1 row root visits are invalid")
    if any(count != ROWS_PER_PLACEMENT_PER_GAME for count in placements.values()):
        raise D1MinerError("D1 rows are not placement-balanced")
    return document


def _load_entries(output: Path, config: D1MinerConfig) -> list[dict[str, object]]:
    manifest = _verified_document(output / "corpus-manifest.json")
    content = manifest["content"]
    assert isinstance(content, dict)
    if (
        content.get("manifest_schema_version") != D1_MANIFEST_SCHEMA_VERSION
        or content.get("configuration_content_digest") != config.content_digest
        or content.get("accepted_checkpoint_digest")
        != config.accepted_checkpoint_digest
        or content.get("acceptance_manifest_digest")
        != config.acceptance_manifest_digest
    ):
        raise D1MinerError("D1 manifest identity differs")
    entries = content.get("games")
    if not isinstance(entries, list):
        raise D1MinerError("D1 manifest games are malformed")
    verified = []
    for ordinal, entry in enumerate(entries):
        if not isinstance(entry, dict) or entry.get("ordinal") != ordinal:
            raise D1MinerError("D1 game ordinals are not gap-free")
        relative = entry.get("path")
        if not isinstance(relative, str):
            raise D1MinerError("D1 game path is malformed")
        game = _verify_game(output / relative, config=config, expected_ordinal=ordinal)
        if game["content_digest"] != entry.get("content_digest"):
            raise D1MinerError("D1 game digest differs from manifest")
        verified.append(entry)
    if (
        content.get("game_count") != len(verified)
        or content.get("row_count") != len(verified) * ROWS_PER_GAME
    ):
        raise D1MinerError("D1 manifest totals differ")
    return verified


def inspect_d1_corpus(output: Path) -> dict[str, object]:
    resolved = _verified_document(output / "resolved-config.json")
    manifest = _verified_document(output / "corpus-manifest.json")
    content = manifest["content"]
    assert isinstance(content, dict)
    return {
        "accepted_checkpoint_digest": content["accepted_checkpoint_digest"],
        "acceptance_manifest_digest": content["acceptance_manifest_digest"],
        "corpus_manifest_digest": manifest["content_digest"],
        "free_disk_bytes": shutil.disk_usage(output).free,
        "game_count": content["game_count"],
        "placement_row_counts": content["placement_row_counts"],
        "resolved_config_digest": resolved["content_digest"],
        "row_count": content["row_count"],
    }


def _config_from_resolved(output: Path, accepted_pi0: str | Path) -> tuple[D1MinerConfig, int]:
    resolved = _verified_document(output / "resolved-config.json")
    content = resolved["content"]
    assert isinstance(content, dict)
    if content.get("collector_schema_version") != D1_RESOLVED_CONFIG_SCHEMA_VERSION:
        raise D1MinerError("resolved artifact is not a D1 configuration")
    config = D1MinerConfig.resolve(
        root_seed=content["root_seed"],
        accepted_pi0_directory=accepted_pi0,
        worker_count=content["worker_count"],
    )
    source = resolve_source_identity()
    expected = _resolved_document(
        config,
        minimum_free_disk_bytes=content["minimum_free_disk_bytes"],
        source=source,
    )
    if expected != resolved:
        raise D1MinerError("D1 resume source or configuration differs")
    return config, content["minimum_free_disk_bytes"]


def commit_d1_games(
    output: Path,
    config: D1MinerConfig,
    entries: list[dict[str, object]],
    games: tuple[MinedD1Game, ...],
) -> None:
    ordered = tuple(sorted(games, key=lambda game: game.ordinal))
    expected = tuple(range(len(entries), len(entries) + len(ordered)))
    if (
        len({game.ordinal for game in ordered}) != len(ordered)
        or tuple(game.ordinal for game in ordered) != expected
    ):
        raise D1MinerError("D1 commit games are not the next canonical batch")
    staged_entries = list(entries)
    for game in ordered:
        relative = f"games/{game.ordinal:06d}.json"
        path = output / relative
        _atomic_json(path, game.artifact_data(config))
        verified = _verify_game(path, config=config, expected_ordinal=game.ordinal)
        staged_entries.append(
            {
                "content_digest": verified["content_digest"],
                "fixture_id": game.fixture_id,
                "ordinal": game.ordinal,
                "path": relative,
                "row_count": ROWS_PER_GAME,
                "trajectory_profile": game.trajectory_profile.value,
            }
        )
    _atomic_json(output / "corpus-manifest.json", _manifest_document(config, staged_entries))
    entries[:] = staged_entries


def run_d1_collection(
    config: D1MinerConfig,
    *,
    output: Path,
    minimum_free_disk_bytes: int,
    max_games: int | None = None,
) -> dict[str, object]:
    source = resolve_source_identity()
    expected_resolved = _resolved_document(
        config, minimum_free_disk_bytes=minimum_free_disk_bytes, source=source
    )
    if not output.exists():
        initialize_d1_corpus(
            config, output=output, minimum_free_disk_bytes=minimum_free_disk_bytes
        )
    elif _verified_document(output / "resolved-config.json") != expected_resolved:
        raise D1MinerError("D1 resume source or configuration differs")
    entries = _load_entries(output, config)
    stop_requested = False

    def request_stop(_signum, _frame) -> None:
        nonlocal stop_requested
        stop_requested = True

    previous = signal.signal(signal.SIGINT, request_stop)
    stop_reason = "interrupted"
    try:
        while not stop_requested:
            if max_games is not None and len(entries) >= max_games:
                stop_reason = "maximum-games"
                break
            if shutil.disk_usage(output).free <= minimum_free_disk_bytes:
                stop_reason = "disk-floor"
                break
            remaining = (
                config.worker_count
                if max_games is None
                else min(config.worker_count, max_games - len(entries))
            )
            ordinals = tuple(range(len(entries), len(entries) + remaining))
            games = mine_d1_games(config, ordinals)
            # A SIGINT received while workers finish commits the complete batch
            # and then exits. No partially produced game is visible.
            commit_d1_games(output, config, entries, games)
            print(
                json.dumps(
                    {
                        "event": "d1_batch_sealed",
                        "game_count": len(entries),
                        "row_count": len(entries) * ROWS_PER_GAME,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    finally:
        signal.signal(signal.SIGINT, previous)
    state = {
        "acceptance_manifest_digest": config.acceptance_manifest_digest,
        "accepted_checkpoint_digest": config.accepted_checkpoint_digest,
        "event": "d1_collector_stopped",
        "game_count": len(entries),
        "reason": stop_reason,
        "report_schema_version": D1_REPORT_SCHEMA_VERSION,
        "row_count": len(entries) * ROWS_PER_GAME,
    }
    _atomic_json(
        output / "state.json",
        {"content": state, "content_digest": _digest(state)},
    )
    return inspect_d1_corpus(output)


def verify_d1_corpus(output: Path, accepted_pi0: str | Path) -> dict[str, object]:
    config, _minimum = _config_from_resolved(output, accepted_pi0)
    entries = _load_entries(output, config)
    result = inspect_d1_corpus(output)
    result["verified_game_count"] = len(entries)
    result["controller_digest"] = config.controller_digest
    return result


def _common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--accepted-pi0", required=True, type=Path)
    parser.add_argument("--root-seed", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--minimum-free-disk-gib", type=float, default=1.0)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dracula-bgc-pi0-miner")
    commands = parser.add_subparsers(dest="command", required=True)
    initialize = commands.add_parser("initialize")
    continuous = commands.add_parser("continuous")
    inspect = commands.add_parser("inspect")
    verify = commands.add_parser("verify")
    select = commands.add_parser(
        "select", help="seal an explicit user-selected pi0 bundle for D1"
    )
    _common_arguments(initialize)
    _common_arguments(continuous)
    continuous.add_argument("--max-games", type=int)
    inspect.add_argument("--output", required=True, type=Path)
    verify.add_argument("--output", required=True, type=Path)
    verify.add_argument("--accepted-pi0", required=True, type=Path)
    select.add_argument("--candidate", required=True, type=Path)
    select.add_argument("--controller-evidence", required=True, type=Path)
    select.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        if arguments.command == "select":
            result = create_d1_user_selection(
                candidate=arguments.candidate,
                controller_evidence=arguments.controller_evidence,
                output=arguments.output,
            )
        elif arguments.command == "inspect":
            result = inspect_d1_corpus(arguments.output)
        elif arguments.command == "verify":
            result = verify_d1_corpus(arguments.output, arguments.accepted_pi0)
        else:
            config = D1MinerConfig.resolve(
                root_seed=arguments.root_seed,
                accepted_pi0_directory=arguments.accepted_pi0,
                worker_count=arguments.workers,
            )
            minimum = int(arguments.minimum_free_disk_gib * (1 << 30))
            if arguments.command == "initialize":
                result = initialize_d1_corpus(
                    config, output=arguments.output, minimum_free_disk_bytes=minimum
                )
            else:
                result = run_d1_collection(
                    config,
                    output=arguments.output,
                    minimum_free_disk_bytes=minimum,
                    max_games=arguments.max_games,
                )
    except D1MinerError as error:
        print(str(error), file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = (
    "D1_CACHE_SCHEMA_VERSION",
    "D1_COLLECTOR_SCHEMA_VERSION",
    "D1_DATASET_SCHEMA_VERSION",
    "D1_GAME_SCHEMA_VERSION",
    "D1_MANIFEST_SCHEMA_VERSION",
    "D1MinerConfig",
    "D1MinerError",
    "D1Row",
    "MinedD1Game",
    "commit_d1_games",
    "create_d1_user_selection",
    "initialize_d1_corpus",
    "inspect_d1_corpus",
    "main",
    "mine_d1_game",
    "mine_d1_games",
    "run_d1_collection",
    "verify_d1_corpus",
)

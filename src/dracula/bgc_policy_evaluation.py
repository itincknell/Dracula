"""Deterministic standalone and prospective-controller evaluation for pi0."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import resource
import shutil
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Protocol

import torch

from dracula.belief_greedy_miner import resolve_source_identity
from dracula.bgc_policy import (
    BGC_POLICY_CANDIDATE_STATUS,
    LoadedBGCPolicyArtifact,
    load_bgc_policy_artifact,
)
from dracula.bridge import move_for_action_index
from dracula.engine import (
    EnginePlayer,
    EngineStatus,
    advance_after_round,
    apply_move,
    create_game,
    derive_game_outcome,
    legal_moves,
    other_player,
)
from dracula.randomness import Sha256CounterStream, derive_seed
from dracula.sam_policy import (
    DESTINATION_SYMMETRY_SCHEMA_VERSION,
    INFERENCE_DESTINATION_SCOPE,
    REPRESENTATIVE_MASK_SCHEMA_VERSION,
    build_representative_action_projection,
    select_representative_action,
)
from dracula.bgc_policy_model import (
    ACTION_SCHEMA_VERSION,
    BGCPolicyModel,
    OBSERVATION_SCHEMA_VERSION,
    compact_action_tensor_from_legacy,
    compact_observation_from_legacy,
    legacy_action_index_from_compact,
)
from dracula.search import (
    BeliefGreedyInformationSetSearch,
    BeliefGreedyResponseEvaluator,
    BeliefGreedyResponseResult,
    BeliefGreedySearchConfig,
    SearchInformationState,
    SearchInterrupted,
    StrategicActionGroupDiagnostic,
    derive_belief_greedy_request_seed,
    derive_strategic_destination_choice_seed,
    information_state_fingerprint,
    information_state_from_engine,
    policy_input_from_information_state,
    select_concrete_action_index,
    strategic_action_groups,
)

EVALUATION_FIXTURE_SCHEMA_VERSION = "dracula-pi0-evaluation-fixtures-v1"
EVALUATION_REPORT_SCHEMA_VERSION = "dracula-pi0-comparison-report-v1"
VALIDATION_EVIDENCE_SCHEMA_VERSION = "dracula-pi0-validation-evidence-v1"
ACCEPTANCE_SCHEMA_VERSION = "dracula-pi0-acceptance-v2"
EVALUATION_PROTOCOL_VERSION = "dracula-pi0-evaluation-protocol-v1"
THROUGHPUT_REPORT_SCHEMA_VERSION = "dracula-pi0-bgc-throughput-v1"
INCREMENTAL_CONTROLLER_SCHEMA_VERSION = "dracula-pi0-controller-progress-v1"
THROUGHPUT_DECK_NAMESPACE = "dracula-pi0-bgc-throughput-deck-v1"
EVALUATION_DECK_NAMESPACE = "dracula-pi0-evaluation-deck-v1"
EVALUATION_FIXTURE_NAMESPACE = "dracula-pi0-evaluation-fixture-v1"
RANDOM_LEGAL_NAMESPACE = "dracula-pi0-random-legal-v1"
ONE_PLY_REQUEST_NAMESPACE = "dracula-pi0-one-ply-request-v1"
STANDALONE_POLICY_NAMESPACE = "dracula-pi0-standalone-request-v1"
POLICY_RESPONSE_NAMESPACE = "dracula-pi0-bgc-response-v1"
PAIRED_BOOTSTRAP_NAMESPACE = "dracula-pi0-paired-bootstrap-v1"

PRIMARY_DECK_COUNT = 60
EXTENSION_DECK_COUNT = 120
TOTAL_EXTENDED_DECK_COUNT = 180
BOOTSTRAP_SAMPLE_COUNT = 2_000
BELIEF_COMPLETION_COUNT = 8
OUTER_SIMULATION_BUDGET = 128

_DIGEST_LENGTH = 64
_COMPARISONS = (
    "standalone-random-legal",
    "standalone-one-ply-belief-greedy",
    "pi0-bgc-vs-base-bgc",
)
_FORBIDDEN_REPORT_KEYS = frozenset(
    {
        "authoritative_state",
        "determinization",
        "engine_seed",
        "game_seed",
        "hidden_state",
        "logits",
        "model_state",
        "opponent_hand",
        "policy_mask",
        "search_tree",
        "stock_order",
    }
)


class Pi0EvaluationError(ValueError):
    """Evaluation fixtures, decisions, evidence, or acceptance are invalid."""


class Pi0EvaluationInterrupted(RuntimeError):
    """Evaluation stopped before an atomic report was committed."""


@dataclass(frozen=True, slots=True)
class EvaluationProtocol:
    primary_deck_count: int = PRIMARY_DECK_COUNT
    extension_deck_count: int = EXTENSION_DECK_COUNT
    bootstrap_samples: int = BOOTSTRAP_SAMPLE_COUNT
    outer_simulation_budget: int = OUTER_SIMULATION_BUDGET
    belief_completion_count: int = BELIEF_COMPLETION_COUNT

    def __post_init__(self) -> None:
        values = (
            self.primary_deck_count,
            self.extension_deck_count,
            self.bootstrap_samples,
            self.outer_simulation_budget,
            self.belief_completion_count,
        )
        if any(type(value) is not int or value < 1 for value in values):
            raise Pi0EvaluationError("evaluation protocol values must be positive")

    @property
    def total_deck_count(self) -> int:
        return self.primary_deck_count + self.extension_deck_count

    @property
    def digest(self) -> str:
        return _json_digest(
            {
                "belief_completion_count": self.belief_completion_count,
                "bootstrap_samples": self.bootstrap_samples,
                "extension_deck_count": self.extension_deck_count,
                "fixture_schema_version": EVALUATION_FIXTURE_SCHEMA_VERSION,
                "outer_simulation_budget": self.outer_simulation_budget,
                "primary_deck_count": self.primary_deck_count,
                "protocol_version": EVALUATION_PROTOCOL_VERSION,
            }
        )


PRODUCTION_EVALUATION_PROTOCOL = EvaluationProtocol()


@dataclass(frozen=True, slots=True)
class EvaluationFixture:
    ordinal: int
    fixture_id: str
    deck_seed: str
    phase: Literal["primary", "extension"]


@dataclass(frozen=True, slots=True)
class EvaluationFixtureManifest:
    path: Path
    digest: str
    root_seed: str
    protocol: EvaluationProtocol
    fixtures: tuple[EvaluationFixture, ...]


@dataclass(frozen=True, slots=True)
class OpponentDecision:
    representative_action_index: int
    concrete_action_index: int
    latency_seconds: float
    peak_resident_memory_bytes: int


class EvaluationOpponent(Protocol):
    @property
    def identity(self) -> str: ...

    @property
    def digest(self) -> str: ...

    def decide(
        self,
        information: SearchInformationState,
        *,
        fixture_id: str,
        decision_index: int,
        should_stop: Callable[[], bool] | None = None,
    ) -> OpponentDecision: ...


@dataclass(frozen=True, slots=True)
class RoundEvaluationRecord:
    round_number: int
    subject_is_dealer: bool
    subject_score: int
    opponent_score: int


@dataclass(frozen=True, slots=True)
class GameEvaluationRecord:
    fixture_id: str
    subject_role: str
    action_trace_digest: str
    subject_won: bool
    tied: bool
    subject_score: int
    opponent_score: int
    rounds: tuple[RoundEvaluationRecord, ...]
    subject_decision_count: int
    subject_latency_seconds: float
    subject_maximum_latency_seconds: float
    subject_peak_resident_memory_bytes: int
    illegal_action_count: int


@dataclass(frozen=True, slots=True)
class AcceptanceDecision:
    status: Literal["passed", "failed", "inconclusive"]
    reasons: tuple[str, ...]
    accepted_directory: str | None


@dataclass(frozen=True, slots=True)
class AcceptedPi0Bundle:
    directory: Path
    checkpoint_path: Path
    checkpoint_digest: str
    acceptance_content_digest: str
    acceptance_file_digest: str
    artifact: LoadedBGCPolicyArtifact


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
        raise Pi0EvaluationError("evaluation artifact is not canonical JSON") from error


def _json_digest(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _require_digest(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != _DIGEST_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Pi0EvaluationError(f"{label} must be a lowercase SHA-256 digest")
    return value


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
        or set(value) != {"format_version", "content", "content_digest", "file_digest"}
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
        raise Pi0EvaluationError(f"{format_version} envelope digest differs")
    return value


def _load_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Pi0EvaluationError(f"evaluation artifact could not be read: {path}") from error


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
    _atomic_bytes(path, _canonical_json(value) + b"\n")


def _assert_report_privacy(value: object) -> None:
    if isinstance(value, Mapping):
        forbidden = _FORBIDDEN_REPORT_KEYS.intersection(value)
        if forbidden:
            raise Pi0EvaluationError(
                "evaluation report contains private fields: "
                + ", ".join(sorted(forbidden))
            )
        for nested in value.values():
            _assert_report_privacy(nested)
    elif isinstance(value, (tuple, list)):
        for nested in value:
            _assert_report_privacy(nested)


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _check_stop(should_stop: Callable[[], bool] | None) -> None:
    if should_stop is not None and should_stop():
        raise Pi0EvaluationInterrupted("evaluation interrupted before report commit")


def _legal_action_indexes(information: SearchInformationState) -> tuple[int, ...]:
    return tuple(
        index
        for index, allowed in enumerate(
            value for row in information.legal_mask for value in row
        )
        if allowed
    )


def _decision(
    information: SearchInformationState,
    representative: int,
    concrete: int,
    *,
    started: float,
) -> OpponentDecision:
    groups = strategic_action_groups(information, True)
    selected = next(
        (
            group
            for group in groups
            if group.representative_action_index == representative
        ),
        None,
    )
    if selected is None or concrete not in selected.member_action_indices:
        raise Pi0EvaluationError("opponent selected outside its strategic group")
    if concrete not in _legal_action_indexes(information):
        raise Pi0EvaluationError("opponent selected an illegal concrete action")
    return OpponentDecision(
        representative,
        concrete,
        time.perf_counter() - started,
        _peak_rss_bytes(),
    )


class RandomLegalOpponent:
    @property
    def identity(self) -> str:
        return "random-legal-strategic-groups-v1"

    @property
    def digest(self) -> str:
        return _json_digest(
            {"identity": self.identity, "namespace": RANDOM_LEGAL_NAMESPACE}
        )

    def decide(
        self,
        information: SearchInformationState,
        *,
        fixture_id: str,
        decision_index: int,
        should_stop: Callable[[], bool] | None = None,
    ) -> OpponentDecision:
        _check_stop(should_stop)
        started = time.perf_counter()
        groups = strategic_action_groups(information, True)
        request_seed = derive_seed(
            RANDOM_LEGAL_NAMESPACE,
            fixture_id,
            information_state_fingerprint(information),
            str(decision_index),
        )
        selected = groups[Sha256CounterStream(request_seed).randbelow(len(groups))]
        representative = selected.representative_action_index
        concrete = select_concrete_action_index(
            information,
            representative,
            derive_strategic_destination_choice_seed(
                request_seed,
                "pi0-random-legal-result-v1",
                information,
                representative,
                0,
            ),
            True,
        )
        return _decision(
            information, representative, concrete, started=started
        )


class OnePlyBeliefGreedyOpponent:
    def __init__(self, belief_completion_count: int = BELIEF_COMPLETION_COUNT) -> None:
        self.config = BeliefGreedySearchConfig(
            outer_simulation_budget=OUTER_SIMULATION_BUDGET,
            belief_completion_count=belief_completion_count,
        )
        self.evaluator = BeliefGreedyResponseEvaluator(self.config)

    @property
    def identity(self) -> str:
        return "one-ply-belief-greedy-v1"

    @property
    def digest(self) -> str:
        return _json_digest(
            {
                "identity": self.identity,
                "response_digest": self.config.response_digest,
            }
        )

    def decide(
        self,
        information: SearchInformationState,
        *,
        fixture_id: str,
        decision_index: int,
        should_stop: Callable[[], bool] | None = None,
    ) -> OpponentDecision:
        del fixture_id, decision_index
        _check_stop(should_stop)
        started = time.perf_counter()
        result = self.evaluator.evaluate(information, should_stop)
        return _decision(
            information,
            result.selected_representative_action_index,
            result.selected_action_index,
            started=started,
        )


class StandalonePi0Opponent:
    def __init__(
        self,
        model: BGCPolicyModel,
        *,
        artifact_digest: str,
        artifact_identity: str,
    ) -> None:
        if not isinstance(model, BGCPolicyModel):
            raise Pi0EvaluationError("pi0 opponent requires BGCPolicyModel")
        self.model = model.cpu().eval().requires_grad_(False)
        self.artifact_digest = _require_digest(
            artifact_digest, "pi0 artifact digest"
        )
        if not isinstance(artifact_identity, str) or not artifact_identity:
            raise Pi0EvaluationError("pi0 artifact identity is invalid")
        self.artifact_identity = artifact_identity

    @classmethod
    def from_artifact(cls, path: str | Path) -> StandalonePi0Opponent:
        artifact_path = Path(path).expanduser().resolve()
        loaded = load_bgc_policy_artifact(artifact_path)
        if loaded.metadata.candidate_status != BGC_POLICY_CANDIDATE_STATUS:
            raise Pi0EvaluationError("pi0 input must be an unaccepted candidate")
        return cls(
            loaded.model,
            artifact_digest=_file_digest(artifact_path),
            artifact_identity=loaded.metadata.state_dict_digest,
        )

    @property
    def identity(self) -> str:
        return "standalone-pi0-v1"

    @property
    def digest(self) -> str:
        return _json_digest(
            {
                "artifact_digest": self.artifact_digest,
                "artifact_identity": self.artifact_identity,
                "controller": self.identity,
            }
        )

    def decide(
        self,
        information: SearchInformationState,
        *,
        fixture_id: str,
        decision_index: int,
        should_stop: Callable[[], bool] | None = None,
    ) -> OpponentDecision:
        _check_stop(should_stop)
        started = time.perf_counter()
        policy_input = policy_input_from_information_state(information)
        groups = strategic_action_groups(information, True)
        projection = build_representative_action_projection(
            policy_input.legal_mask, groups
        )
        observation = compact_observation_from_legacy(policy_input.observation)
        compact_mask = compact_action_tensor_from_legacy(
            policy_input.observation, projection.mask
        )
        with torch.inference_mode():
            logits = self.model(observation)
        compact_representative = select_representative_action(logits, compact_mask)
        if not isinstance(compact_representative, int):
            raise Pi0EvaluationError("single-state pi0 returned batched selection")
        representative = legacy_action_index_from_compact(
            policy_input.observation, compact_representative
        )
        request_seed = derive_seed(
            STANDALONE_POLICY_NAMESPACE,
            fixture_id,
            information_state_fingerprint(information),
            self.artifact_digest,
            str(decision_index),
        )
        concrete = select_concrete_action_index(
            information,
            representative,
            derive_strategic_destination_choice_seed(
                request_seed,
                INFERENCE_DESTINATION_SCOPE,
                information,
                representative,
                0,
            ),
            True,
        )
        return _decision(
            information, representative, concrete, started=started
        )


class _Pi0ContinuationResponseEvaluator:
    """Select one continuation from only the simulated actor's information."""

    def __init__(
        self,
        policy: StandalonePi0Opponent,
        search_config: BeliefGreedySearchConfig,
    ) -> None:
        self.policy = policy
        self.search_config = search_config

    @property
    def digest(self) -> str:
        return _json_digest(
            {
                "artifact_digest": self.policy.artifact_digest,
                "controller": "pi0-continuation-response-v1",
                "response_namespace": POLICY_RESPONSE_NAMESPACE,
                "search_response_digest": self.search_config.response_digest,
            }
        )

    def evaluate(
        self,
        information: SearchInformationState,
        should_stop: Callable[[], bool] | None = None,
    ) -> BeliefGreedyResponseResult:
        _check_stop(should_stop)
        groups = strategic_action_groups(information, True)
        decision = self.policy.decide(
            information,
            fixture_id=_json_digest(
                {
                    "information_state": information_state_fingerprint(information),
                    "namespace": POLICY_RESPONSE_NAMESPACE,
                    "policy_digest": self.policy.digest,
                }
            ),
            decision_index=0,
            should_stop=should_stop,
        )
        diagnostics = tuple(
            StrategicActionGroupDiagnostic(group, 1, 0.0) for group in groups
        )
        # The outer implementation checks the response digest to prevent a
        # response from crossing an information-state cache boundary. The
        # evaluation report separately binds this policy evaluator's digest.
        return BeliefGreedyResponseResult(
            information_state_fingerprint(information),
            self.search_config.response_digest,
            decision.concrete_action_index,
            decision.representative_action_index,
            diagnostics,
            1,
            len(groups),
        )


class Pi0ContinuationBeliefGreedySearch(BeliefGreedyInformationSetSearch):
    """The unchanged outer BGC search with pi0 continuation choices."""

    def __init__(
        self,
        policy: StandalonePi0Opponent,
        config: BeliefGreedySearchConfig,
    ) -> None:
        super().__init__(config)
        self.policy_response = _Pi0ContinuationResponseEvaluator(policy, config)

    @property
    def controller_digest(self) -> str:
        return _json_digest(
            {
                "outer_search_digest": self.config.digest,
                "policy_response_digest": self.policy_response.digest,
                "schema": "dracula-pi0-bgc-controller-v1",
            }
        )

    def _actor_response(
        self,
        information: SearchInformationState,
        should_stop: Callable[[], bool] | None,
    ) -> BeliefGreedyResponseResult:
        return self.policy_response.evaluate(information, should_stop)


class BGCSearchOpponent:
    def __init__(
        self,
        planner: BeliefGreedyInformationSetSearch,
        *,
        identity: str,
        controller_digest: str,
    ) -> None:
        self.planner = planner
        self._identity = identity
        self._digest = _require_digest(controller_digest, "BGC controller digest")

    @classmethod
    def base(cls, config: BeliefGreedySearchConfig) -> BGCSearchOpponent:
        planner = BeliefGreedyInformationSetSearch(config)
        return cls(
            planner,
            identity="base-bgc-one-ply-response-v1",
            controller_digest=config.digest,
        )

    @classmethod
    def pi0(
        cls,
        policy: StandalonePi0Opponent,
        config: BeliefGreedySearchConfig,
    ) -> BGCSearchOpponent:
        planner = Pi0ContinuationBeliefGreedySearch(policy, config)
        return cls(
            planner,
            identity="pi0-bgc-response-v1",
            controller_digest=planner.controller_digest,
        )

    @property
    def identity(self) -> str:
        return self._identity

    @property
    def digest(self) -> str:
        return self._digest

    def decide(
        self,
        information: SearchInformationState,
        *,
        fixture_id: str,
        decision_index: int,
        should_stop: Callable[[], bool] | None = None,
    ) -> OpponentDecision:
        del decision_index
        _check_stop(should_stop)
        started = time.perf_counter()
        # Both base and candidate use the base search digest here. This keeps
        # outer determinizations and selection seeds paired exactly.
        request_seed = derive_belief_greedy_request_seed(
            fixture_id, information, self.planner.config.digest
        )
        try:
            result = self.planner.search(information, request_seed, should_stop)
        except SearchInterrupted as error:
            raise Pi0EvaluationInterrupted(str(error)) from error
        return _decision(
            information,
            result.selected_representative_action_index,
            result.selected_action_index,
            started=started,
        )


def create_evaluation_fixture_manifest(
    output_path: str | Path,
    *,
    root_seed: str,
    protocol: EvaluationProtocol = PRODUCTION_EVALUATION_PROTOCOL,
) -> EvaluationFixtureManifest:
    if not isinstance(root_seed, str) or not root_seed:
        raise Pi0EvaluationError("evaluation root seed must be nonempty")
    fixtures = []
    for ordinal in range(protocol.total_deck_count):
        deck_seed = derive_seed(
            EVALUATION_DECK_NAMESPACE,
            root_seed,
            str(ordinal),
            protocol.digest,
        ).hex()
        fixture_id = derive_seed(
            EVALUATION_FIXTURE_NAMESPACE,
            root_seed,
            str(ordinal),
            hashlib.sha256(deck_seed.encode("ascii")).hexdigest(),
            protocol.digest,
        ).hex()
        fixtures.append(
            {
                "deck_seed": deck_seed,
                "fixture_id": fixture_id,
                "ordinal": ordinal,
                "phase": (
                    "primary"
                    if ordinal < protocol.primary_deck_count
                    else "extension"
                ),
            }
        )
    content = {
        "deck_namespace": EVALUATION_DECK_NAMESPACE,
        "fixture_namespace": EVALUATION_FIXTURE_NAMESPACE,
        "fixtures": fixtures,
        "protocol": asdict(protocol),
        "protocol_digest": protocol.digest,
        "root_seed": root_seed,
    }
    document = _envelope(EVALUATION_FIXTURE_SCHEMA_VERSION, content)
    path = Path(output_path).expanduser().resolve()
    if path.exists():
        if _load_json(path) != document:
            raise Pi0EvaluationError("existing evaluation fixture manifest differs")
    else:
        _atomic_json(path, document)
    return load_evaluation_fixture_manifest(path, protocol=protocol)


def load_evaluation_fixture_manifest(
    path: str | Path,
    *,
    protocol: EvaluationProtocol = PRODUCTION_EVALUATION_PROTOCOL,
) -> EvaluationFixtureManifest:
    manifest_path = Path(path).expanduser().resolve()
    document = _validate_envelope(
        _load_json(manifest_path), EVALUATION_FIXTURE_SCHEMA_VERSION
    )
    content = document["content"]
    if (
        not isinstance(content, dict)
        or set(content)
        != {
            "deck_namespace",
            "fixture_namespace",
            "fixtures",
            "protocol",
            "protocol_digest",
            "root_seed",
        }
        or content["deck_namespace"] != EVALUATION_DECK_NAMESPACE
        or content["fixture_namespace"] != EVALUATION_FIXTURE_NAMESPACE
        or content["protocol"] != asdict(protocol)
        or content["protocol_digest"] != protocol.digest
        or not isinstance(content["root_seed"], str)
        or not content["root_seed"]
        or not isinstance(content["fixtures"], list)
        or len(content["fixtures"]) != protocol.total_deck_count
    ):
        raise Pi0EvaluationError("evaluation fixture manifest is incompatible")
    fixtures: list[EvaluationFixture] = []
    for ordinal, raw in enumerate(content["fixtures"]):
        expected_phase = (
            "primary" if ordinal < protocol.primary_deck_count else "extension"
        )
        if (
            not isinstance(raw, dict)
            or set(raw) != {"deck_seed", "fixture_id", "ordinal", "phase"}
            or raw["ordinal"] != ordinal
            or raw["phase"] != expected_phase
        ):
            raise Pi0EvaluationError("evaluation fixture ordering differs")
        deck_seed = derive_seed(
            EVALUATION_DECK_NAMESPACE,
            content["root_seed"],
            str(ordinal),
            protocol.digest,
        ).hex()
        fixture_id = derive_seed(
            EVALUATION_FIXTURE_NAMESPACE,
            content["root_seed"],
            str(ordinal),
            hashlib.sha256(deck_seed.encode("ascii")).hexdigest(),
            protocol.digest,
        ).hex()
        if raw["deck_seed"] != deck_seed or raw["fixture_id"] != fixture_id:
            raise Pi0EvaluationError("evaluation fixture seed derivation differs")
        fixtures.append(
            EvaluationFixture(ordinal, fixture_id, deck_seed, expected_phase)
        )
    if len({fixture.deck_seed for fixture in fixtures}) != len(fixtures):
        raise Pi0EvaluationError("evaluation deck seeds are not distinct")
    return EvaluationFixtureManifest(
        manifest_path,
        document["content_digest"],
        content["root_seed"],
        protocol,
        tuple(fixtures),
    )


def _play_game(
    fixture: EvaluationFixture,
    *,
    subject: EvaluationOpponent,
    control: EvaluationOpponent,
    subject_role: EnginePlayer,
    should_stop: Callable[[], bool] | None,
) -> GameEvaluationRecord:
    _check_stop(should_stop)
    state = create_game(fixture.deck_seed)
    opponents = {
        subject_role: subject,
        other_player(subject_role): control,
    }
    rounds: list[RoundEvaluationRecord] = []
    action_trace: list[dict[str, object]] = []
    subject_latencies: list[float] = []
    subject_peak_memory: list[int] = []
    decision_index = 0
    while state.status is not EngineStatus.GAME_COMPLETE:
        while state.status is EngineStatus.PLAYING:
            _check_stop(should_stop)
            actor = state.active_player
            if actor is None:
                raise Pi0EvaluationError("playing evaluation has no active player")
            moves = legal_moves(state, actor)
            if len(moves) == 1:
                action_trace.append(
                    {
                        "actor": actor.value,
                        "global_grid_index": moves[0].global_grid_index,
                        "hand_slot": moves[0].hand_slot,
                        "forced": True,
                    }
                )
                state = apply_move(state, moves[0]).state
                continue
            information = information_state_from_engine(state)
            decision = opponents[actor].decide(
                information,
                fixture_id=fixture.fixture_id,
                decision_index=decision_index,
                should_stop=should_stop,
            )
            decision_index += 1
            move = move_for_action_index(actor, decision.concrete_action_index)
            if move not in moves:
                raise Pi0EvaluationError("evaluation opponent returned an illegal move")
            if actor is subject_role:
                subject_latencies.append(decision.latency_seconds)
                subject_peak_memory.append(decision.peak_resident_memory_bytes)
            action_trace.append(
                {
                    "actor": actor.value,
                    "global_grid_index": move.global_grid_index,
                    "hand_slot": move.hand_slot,
                    "forced": False,
                }
            )
            state = apply_move(state, move).state
        result = state.pending_round_result
        if result is None:
            raise Pi0EvaluationError("completed evaluation round has no score")
        opponent_role = other_player(subject_role)
        rounds.append(
            RoundEvaluationRecord(
                result.round_number,
                result.dealer is subject_role,
                result.round_scores[subject_role],
                result.round_scores[opponent_role],
            )
        )
        state = advance_after_round(state)
    outcome = derive_game_outcome(state)
    opponent_role = other_player(subject_role)
    return GameEvaluationRecord(
        fixture.fixture_id,
        subject_role.value,
        _json_digest(action_trace),
        outcome.winner is subject_role,
        outcome.winner is None,
        state.total_scores[subject_role],
        state.total_scores[opponent_role],
        tuple(rounds),
        len(subject_latencies),
        sum(subject_latencies),
        max(subject_latencies, default=0.0),
        max(subject_peak_memory, default=0),
        0,
    )


def _percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        raise Pi0EvaluationError("percentile requires values")
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(quantile * len(ordered)) - 1)
    return float(ordered[max(index, 0)])


def _paired_vectors(records: Sequence[GameEvaluationRecord]) -> tuple[float, ...]:
    blocks: dict[str, list[GameEvaluationRecord]] = {}
    for record in records:
        blocks.setdefault(record.fixture_id, []).append(record)
    vectors = []
    for fixture_id, games in sorted(blocks.items()):
        if (
            len(games) != 2
            or {game.subject_role for game in games} != {"queen", "king"}
        ):
            raise Pi0EvaluationError(
                f"evaluation fixture is not role paired: {fixture_id}"
            )
        vectors.append(
            sum(game.subject_score - game.opponent_score for game in games) / 2.0
        )
    return tuple(vectors)


def _deterministic_result_digest(
    records: Sequence[GameEvaluationRecord],
) -> str:
    return _json_digest(
        [
            {
                "action_trace_digest": record.action_trace_digest,
                "fixture_id": record.fixture_id,
                "illegal_action_count": record.illegal_action_count,
                "opponent_score": record.opponent_score,
                "rounds": [asdict(item) for item in record.rounds],
                "subject_role": record.subject_role,
                "subject_score": record.subject_score,
                "subject_won": record.subject_won,
                "tied": record.tied,
            }
            for record in records
        ]
    )


def paired_confidence_interval(
    records: Sequence[GameEvaluationRecord],
    *,
    comparison_id: str,
    fixture_manifest_digest: str,
    bootstrap_samples: int,
) -> tuple[float, float]:
    vectors = _paired_vectors(records)
    if type(bootstrap_samples) is not int or bootstrap_samples < 1:
        raise Pi0EvaluationError("bootstrap sample count must be positive")
    records_digest = _json_digest([asdict(record) for record in records])
    stream = Sha256CounterStream(
        derive_seed(
            PAIRED_BOOTSTRAP_NAMESPACE,
            comparison_id,
            fixture_manifest_digest,
            records_digest,
            str(bootstrap_samples),
        )
    )
    estimates = [
        sum(vectors[stream.randbelow(len(vectors))] for _ in vectors)
        / len(vectors)
        for _ in range(bootstrap_samples)
    ]
    return _percentile(estimates, 0.025), _percentile(estimates, 0.975)


def _split_metrics(records: Sequence[GameEvaluationRecord]) -> dict[str, object]:
    wins = sum(record.subject_won for record in records)
    return {
        "games": len(records),
        "wins": wins,
        "win_rate": wins / len(records) if records else None,
        "ties": sum(record.tied for record in records),
        "mean_score_differential": (
            sum(record.subject_score - record.opponent_score for record in records)
            / len(records)
            if records
            else None
        ),
    }


def _comparison_metrics(
    records: Sequence[GameEvaluationRecord],
    *,
    comparison_id: str,
    fixture_manifest_digest: str,
    bootstrap_samples: int,
) -> dict[str, object]:
    vectors = _paired_vectors(records)
    interval = paired_confidence_interval(
        records,
        comparison_id=comparison_id,
        fixture_manifest_digest=fixture_manifest_digest,
        bootstrap_samples=bootstrap_samples,
    )
    rounds = [round_record for record in records for round_record in record.rounds]
    round_wins = sum(
        round_record.subject_score > round_record.opponent_score
        for round_record in rounds
    )
    latency_values = [
        record.subject_latency_seconds / record.subject_decision_count
        for record in records
        if record.subject_decision_count
    ]
    maximum_latencies = [record.subject_maximum_latency_seconds for record in records]
    role_splits = {
        role: _split_metrics(
            [record for record in records if record.subject_role == role]
        )
        for role in ("queen", "king")
    }
    dealer_splits = {}
    for label, dealer in (("dealer", True), ("non-dealer", False)):
        selected = [round_record for round_record in rounds if round_record.subject_is_dealer is dealer]
        selected_wins = sum(
            round_record.subject_score > round_record.opponent_score
            for round_record in selected
        )
        dealer_splits[label] = {
            "rounds": len(selected),
            "wins": selected_wins,
            "win_rate": selected_wins / len(selected) if selected else None,
            "ties": sum(
                round_record.subject_score == round_record.opponent_score
                for round_record in selected
            ),
            "mean_score_differential": (
                sum(
                    round_record.subject_score - round_record.opponent_score
                    for round_record in selected
                )
                / len(selected)
                if selected
                else None
            ),
        }
    return {
        "deck_pairs": len(vectors),
        "games": _split_metrics(records),
        "rounds": {
            "count": len(rounds),
            "wins": round_wins,
            "win_rate": round_wins / len(rounds),
            "ties": sum(
                round_record.subject_score == round_record.opponent_score
                for round_record in rounds
            ),
            "mean_score_differential": sum(
                round_record.subject_score - round_record.opponent_score
                for round_record in rounds
            )
            / len(rounds),
        },
        "paired_score_differential": sum(vectors) / len(vectors),
        "paired_score_differential_ci_95": list(interval),
        "role_splits": role_splits,
        "dealer_splits": dealer_splits,
        "latency_seconds": {
            "mean_per_decision": (
                sum(latency_values) / len(latency_values)
                if latency_values
                else 0.0
            ),
            "p50_game_mean": _percentile(latency_values, 0.5) if latency_values else 0.0,
            "p95_game_mean": _percentile(latency_values, 0.95) if latency_values else 0.0,
            "maximum_decision": max(maximum_latencies, default=0.0),
        },
        "peak_resident_memory_bytes": max(
            (record.subject_peak_resident_memory_bytes for record in records),
            default=0,
        ),
        "illegal_action_count": sum(record.illegal_action_count for record in records),
    }


def _needs_extension(metrics: Mapping[str, object]) -> bool:
    point = float(metrics["paired_score_differential"])
    interval = metrics["paired_score_differential_ci_95"]
    if not isinstance(interval, list) or len(interval) != 2:
        raise Pi0EvaluationError("paired interval is malformed")
    return point > 0 and float(interval[0]) <= 0 <= float(interval[1])


def run_paired_comparison(
    *,
    comparison_id: str,
    comparison_type: Literal["standalone", "controller"],
    subject: EvaluationOpponent,
    control: EvaluationOpponent,
    fixture_manifest: EvaluationFixtureManifest,
    artifact_digest: str,
    output_path: str | Path,
    should_stop: Callable[[], bool] | None = None,
) -> Path:
    if comparison_id not in _COMPARISONS:
        raise Pi0EvaluationError("comparison identity is not contractual")
    _require_digest(artifact_digest, "candidate artifact digest")
    protocol = fixture_manifest.protocol
    records: list[GameEvaluationRecord] = []

    def add_fixture(fixture: EvaluationFixture) -> None:
        for role in EnginePlayer:
            _check_stop(should_stop)
            records.append(
                _play_game(
                    fixture,
                    subject=subject,
                    control=control,
                    subject_role=role,
                    should_stop=should_stop,
                )
            )

    for fixture in fixture_manifest.fixtures[: protocol.primary_deck_count]:
        add_fixture(fixture)
    primary_metrics = _comparison_metrics(
        records,
        comparison_id=comparison_id,
        fixture_manifest_digest=fixture_manifest.digest,
        bootstrap_samples=protocol.bootstrap_samples,
    )
    extended = _needs_extension(primary_metrics)
    if extended:
        for fixture in fixture_manifest.fixtures[protocol.primary_deck_count :]:
            add_fixture(fixture)
    metrics = _comparison_metrics(
        records,
        comparison_id=comparison_id,
        fixture_manifest_digest=fixture_manifest.digest,
        bootstrap_samples=protocol.bootstrap_samples,
    )
    del metrics
    return _seal_comparison_records(
        comparison_id=comparison_id,
        comparison_type=comparison_type,
        records=records,
        subject_identity=subject.identity,
        subject_digest=subject.digest,
        control_identity=control.identity,
        control_digest=control.digest,
        fixture_manifest=fixture_manifest,
        artifact_digest=artifact_digest,
        output_path=output_path,
        extended=extended,
    )


def _incremental_controller_document(
    *,
    records: Sequence[GameEvaluationRecord],
    subject: EvaluationOpponent,
    control: EvaluationOpponent,
    fixture_manifest: EvaluationFixtureManifest,
    artifact_digest: str,
    target_deck_count: int,
) -> dict[str, object]:
    completed_deck_count = len(records) // 2
    metrics = _comparison_metrics(
        records,
        comparison_id="pi0-bgc-vs-base-bgc",
        fixture_manifest_digest=fixture_manifest.digest,
        bootstrap_samples=fixture_manifest.protocol.bootstrap_samples,
    )
    source = resolve_source_identity()
    content = {
        "artifact_digest": _require_digest(
            artifact_digest, "candidate artifact digest"
        ),
        "bootstrap_namespace": PAIRED_BOOTSTRAP_NAMESPACE,
        "comparison_id": "pi0-bgc-vs-base-bgc",
        "complete": completed_deck_count == target_deck_count,
        "completed_deck_count": completed_deck_count,
        "control": {
            "digest": _require_digest(control.digest, "control digest"),
            "identity": control.identity,
        },
        "deterministic_result_digest": _deterministic_result_digest(records),
        "fixture_manifest_digest": fixture_manifest.digest,
        "games": [asdict(record) for record in records],
        "metrics": metrics,
        "protocol": asdict(fixture_manifest.protocol),
        "protocol_digest": fixture_manifest.protocol.digest,
        "source_revision": source.revision,
        "source_tree_digest": source.tree_digest,
        "subject": {
            "digest": _require_digest(subject.digest, "subject digest"),
            "identity": subject.identity,
        },
        "target_deck_count": target_deck_count,
    }
    _assert_report_privacy(content)
    return _envelope(INCREMENTAL_CONTROLLER_SCHEMA_VERSION, content)


def _load_incremental_controller_records(
    path: Path,
    *,
    subject: EvaluationOpponent,
    control: EvaluationOpponent,
    fixture_manifest: EvaluationFixtureManifest,
    artifact_digest: str,
    target_deck_count: int,
) -> tuple[GameEvaluationRecord, ...]:
    document = _validate_envelope(
        _load_json(path), INCREMENTAL_CONTROLLER_SCHEMA_VERSION
    )
    content = document["content"]
    if not isinstance(content, dict):
        raise Pi0EvaluationError("incremental controller content is malformed")
    required = {
        "artifact_digest",
        "bootstrap_namespace",
        "comparison_id",
        "complete",
        "completed_deck_count",
        "control",
        "deterministic_result_digest",
        "fixture_manifest_digest",
        "games",
        "metrics",
        "protocol",
        "protocol_digest",
        "source_revision",
        "source_tree_digest",
        "subject",
        "target_deck_count",
    }
    expected_subject = {"digest": subject.digest, "identity": subject.identity}
    expected_control = {"digest": control.digest, "identity": control.identity}
    if (
        set(content) != required
        or content["artifact_digest"] != artifact_digest
        or content["bootstrap_namespace"] != PAIRED_BOOTSTRAP_NAMESPACE
        or content["comparison_id"] != "pi0-bgc-vs-base-bgc"
        or content["fixture_manifest_digest"] != fixture_manifest.digest
        or content["protocol"] != asdict(fixture_manifest.protocol)
        or content["protocol_digest"] != fixture_manifest.protocol.digest
        or content["subject"] != expected_subject
        or content["control"] != expected_control
        or content["target_deck_count"] != target_deck_count
        or not isinstance(content["games"], list)
    ):
        raise Pi0EvaluationError("incremental controller identity is incompatible")
    records = tuple(_record_from_json(value) for value in content["games"])
    if len(records) % 2:
        raise Pi0EvaluationError("incremental controller contains an unpaired game")
    completed = len(records) // 2
    expected_fixture_ids = tuple(
        fixture.fixture_id
        for fixture in fixture_manifest.fixtures[:completed]
        for _ in range(2)
    )
    if (
        type(content["completed_deck_count"]) is not int
        or content["completed_deck_count"] != completed
        or completed > target_deck_count
        or type(content["complete"]) is not bool
        or content["complete"] != (completed == target_deck_count)
        or tuple(record.fixture_id for record in records) != expected_fixture_ids
        or content["deterministic_result_digest"]
        != _deterministic_result_digest(records)
    ):
        raise Pi0EvaluationError("incremental controller coverage differs")
    for offset in range(0, len(records), 2):
        if {records[offset].subject_role, records[offset + 1].subject_role} != {
            "queen",
            "king",
        }:
            raise Pi0EvaluationError("incremental controller role pair differs")
    metrics = _comparison_metrics(
        records,
        comparison_id="pi0-bgc-vs-base-bgc",
        fixture_manifest_digest=fixture_manifest.digest,
        bootstrap_samples=fixture_manifest.protocol.bootstrap_samples,
    )
    if content["metrics"] != metrics:
        raise Pi0EvaluationError("incremental controller metrics differ")
    return records


def run_incremental_controller_comparison(
    *,
    subject: EvaluationOpponent,
    control: EvaluationOpponent,
    fixture_manifest: EvaluationFixtureManifest,
    artifact_digest: str,
    output_path: str | Path,
    target_deck_count: int,
    should_stop: Callable[[], bool] | None = None,
    progress_callback: Callable[[Mapping[str, object]], None] | None = None,
) -> Path:
    if (
        type(target_deck_count) is not int
        or target_deck_count < 1
        or target_deck_count > fixture_manifest.protocol.primary_deck_count
    ):
        raise Pi0EvaluationError(
            "incremental controller deck count must fit the primary fixtures"
        )
    _require_digest(artifact_digest, "candidate artifact digest")
    path = Path(output_path).expanduser().resolve()
    records = list(
        _load_incremental_controller_records(
            path,
            subject=subject,
            control=control,
            fixture_manifest=fixture_manifest,
            artifact_digest=artifact_digest,
            target_deck_count=target_deck_count,
        )
        if path.exists()
        else ()
    )
    for fixture in fixture_manifest.fixtures[
        len(records) // 2 : target_deck_count
    ]:
        pair: list[GameEvaluationRecord] = []
        for role in EnginePlayer:
            _check_stop(should_stop)
            pair.append(
                _play_game(
                    fixture,
                    subject=subject,
                    control=control,
                    subject_role=role,
                    should_stop=should_stop,
                )
            )
        records.extend(pair)
        document = _incremental_controller_document(
            records=records,
            subject=subject,
            control=control,
            fixture_manifest=fixture_manifest,
            artifact_digest=artifact_digest,
            target_deck_count=target_deck_count,
        )
        _atomic_json(path, document)
        metrics = document["content"]["metrics"]
        update = {
            "completed_deck_count": len(records) // 2,
            "game_score_differential": metrics["games"][
                "mean_score_differential"
            ],
            "paired_score_differential": metrics["paired_score_differential"],
            "paired_score_differential_ci_95": metrics[
                "paired_score_differential_ci_95"
            ],
            "queen_score_differential": metrics["role_splits"]["queen"][
                "mean_score_differential"
            ],
            "king_score_differential": metrics["role_splits"]["king"][
                "mean_score_differential"
            ],
            "target_deck_count": target_deck_count,
        }
        if progress_callback is not None:
            progress_callback(update)
    return path


def _seal_comparison_records(
    *,
    comparison_id: str,
    comparison_type: Literal["standalone", "controller"],
    records: Sequence[GameEvaluationRecord],
    subject_identity: str,
    subject_digest: str,
    control_identity: str,
    control_digest: str,
    fixture_manifest: EvaluationFixtureManifest,
    artifact_digest: str,
    output_path: str | Path,
    extended: bool,
) -> Path:
    protocol = fixture_manifest.protocol
    deck_count = len(records) // 2
    expected_count = (
        protocol.total_deck_count if extended else protocol.primary_deck_count
    )
    if deck_count != expected_count:
        raise Pi0EvaluationError("comparison record count differs from extension state")
    metrics = _comparison_metrics(
        records,
        comparison_id=comparison_id,
        fixture_manifest_digest=fixture_manifest.digest,
        bootstrap_samples=protocol.bootstrap_samples,
    )
    if not extended and _needs_extension(metrics):
        raise Pi0EvaluationError("comparison requires the contractual extension")
    source = resolve_source_identity()
    content = {
        "artifact_digest": _require_digest(
            artifact_digest, "candidate artifact digest"
        ),
        "bootstrap_namespace": PAIRED_BOOTSTRAP_NAMESPACE,
        "comparison_id": comparison_id,
        "comparison_type": comparison_type,
        "control": {
            "digest": _require_digest(control_digest, "control digest"),
            "identity": control_identity,
        },
        "deck_count": deck_count,
        "deterministic_result_digest": _deterministic_result_digest(records),
        "extended": extended,
        "fixture_manifest_digest": fixture_manifest.digest,
        "games": [asdict(record) for record in records],
        "metrics": metrics,
        "protocol": asdict(protocol),
        "protocol_digest": protocol.digest,
        "source_revision": source.revision,
        "source_tree_digest": source.tree_digest,
        "subject": {
            "digest": _require_digest(subject_digest, "subject digest"),
            "identity": subject_identity,
        },
    }
    _assert_report_privacy(content)
    document = _envelope(EVALUATION_REPORT_SCHEMA_VERSION, content)
    path = Path(output_path).expanduser().resolve()
    if path.exists():
        if _load_json(path) != document:
            raise Pi0EvaluationError("existing comparison report differs")
    else:
        _atomic_json(path, document)
    load_comparison_report(
        path,
        fixture_manifest=fixture_manifest,
        artifact_digest=artifact_digest,
    )
    return path


def _record_from_json(value: object) -> GameEvaluationRecord:
    if not isinstance(value, dict) or set(value) != {
        "fixture_id",
        "subject_role",
        "action_trace_digest",
        "subject_won",
        "tied",
        "subject_score",
        "opponent_score",
        "rounds",
        "subject_decision_count",
        "subject_latency_seconds",
        "subject_maximum_latency_seconds",
        "subject_peak_resident_memory_bytes",
        "illegal_action_count",
    }:
        raise Pi0EvaluationError("comparison game record is malformed")
    rounds_raw = value["rounds"]
    if not isinstance(rounds_raw, list) or len(rounds_raw) != 6:
        raise Pi0EvaluationError("comparison game must contain six rounds")
    try:
        rounds = tuple(RoundEvaluationRecord(**item) for item in rounds_raw)
        record = GameEvaluationRecord(
            fixture_id=value["fixture_id"],
            subject_role=value["subject_role"],
            action_trace_digest=value["action_trace_digest"],
            subject_won=value["subject_won"],
            tied=value["tied"],
            subject_score=value["subject_score"],
            opponent_score=value["opponent_score"],
            rounds=rounds,
            subject_decision_count=value["subject_decision_count"],
            subject_latency_seconds=value["subject_latency_seconds"],
            subject_maximum_latency_seconds=value[
                "subject_maximum_latency_seconds"
            ],
            subject_peak_resident_memory_bytes=value[
                "subject_peak_resident_memory_bytes"
            ],
            illegal_action_count=value["illegal_action_count"],
        )
    except (TypeError, ValueError) as error:
        raise Pi0EvaluationError("comparison game record types are invalid") from error
    numeric_integers = (
        record.subject_score,
        record.opponent_score,
        record.subject_decision_count,
        record.subject_peak_resident_memory_bytes,
        record.illegal_action_count,
    )
    if (
        not isinstance(record.fixture_id, str)
        or not record.fixture_id
        or _require_digest(record.action_trace_digest, "action trace")
        != record.action_trace_digest
        or record.subject_role not in {"queen", "king"}
        or type(record.subject_won) is not bool
        or type(record.tied) is not bool
        or record.subject_won and record.tied
        or any(type(number) is not int or number < 0 for number in numeric_integers)
        or type(record.subject_latency_seconds) not in (int, float)
        or type(record.subject_maximum_latency_seconds) not in (int, float)
        or not math.isfinite(record.subject_latency_seconds)
        or not math.isfinite(record.subject_maximum_latency_seconds)
        or record.subject_latency_seconds < 0
        or record.subject_maximum_latency_seconds < 0
        or tuple(round_record.round_number for round_record in record.rounds)
        != tuple(range(1, 7))
    ):
        raise Pi0EvaluationError("comparison game record values are invalid")
    for round_record in record.rounds:
        if (
            type(round_record.subject_is_dealer) is not bool
            or type(round_record.subject_score) is not int
            or type(round_record.opponent_score) is not int
            or round_record.subject_score < 0
            or round_record.opponent_score < 0
        ):
            raise Pi0EvaluationError("comparison round record is invalid")
    if (
        record.subject_score
        != sum(round_record.subject_score for round_record in record.rounds)
        or record.opponent_score
        != sum(round_record.opponent_score for round_record in record.rounds)
        or record.subject_won != (record.subject_score > record.opponent_score)
        or record.tied != (record.subject_score == record.opponent_score)
    ):
        raise Pi0EvaluationError("comparison game result contradicts round evidence")
    return record


def load_comparison_report(
    path: str | Path,
    *,
    fixture_manifest: EvaluationFixtureManifest,
    artifact_digest: str,
) -> dict[str, object]:
    report_path = Path(path).expanduser().resolve()
    document = _validate_envelope(
        _load_json(report_path), EVALUATION_REPORT_SCHEMA_VERSION
    )
    content = document["content"]
    required = {
        "artifact_digest",
        "bootstrap_namespace",
        "comparison_id",
        "comparison_type",
        "control",
        "deck_count",
        "deterministic_result_digest",
        "extended",
        "fixture_manifest_digest",
        "games",
        "metrics",
        "protocol",
        "protocol_digest",
        "source_revision",
        "source_tree_digest",
        "subject",
    }
    if (
        not isinstance(content, dict)
        or set(content) != required
        or content["artifact_digest"] != artifact_digest
        or content["bootstrap_namespace"] != PAIRED_BOOTSTRAP_NAMESPACE
        or content["comparison_id"] not in _COMPARISONS
        or content["comparison_type"] not in {"standalone", "controller"}
        or content["fixture_manifest_digest"] != fixture_manifest.digest
        or content["protocol"] != asdict(fixture_manifest.protocol)
        or content["protocol_digest"] != fixture_manifest.protocol.digest
        or type(content["extended"]) is not bool
        or not isinstance(content["games"], list)
    ):
        raise Pi0EvaluationError("comparison report identity is incompatible")
    for identity in (content["subject"], content["control"]):
        if (
            not isinstance(identity, dict)
            or set(identity) != {"digest", "identity"}
            or not isinstance(identity["identity"], str)
            or not identity["identity"]
        ):
            raise Pi0EvaluationError("comparison controller identity is malformed")
        _require_digest(identity["digest"], "comparison controller digest")
    _require_digest(content["source_tree_digest"], "evaluation source tree")
    source_revision = content["source_revision"]
    if (
        not isinstance(source_revision, str)
        or len(source_revision) not in (40, 64)
        or any(character not in "0123456789abcdef" for character in source_revision)
    ):
        raise Pi0EvaluationError("evaluation source revision is invalid")
    records = tuple(_record_from_json(value) for value in content["games"])
    if content["deterministic_result_digest"] != _deterministic_result_digest(records):
        raise Pi0EvaluationError("comparison deterministic result digest differs")
    expected_decks = (
        fixture_manifest.protocol.total_deck_count
        if content["extended"]
        else fixture_manifest.protocol.primary_deck_count
    )
    expected_fixture_ids = {
        fixture.fixture_id
        for fixture in fixture_manifest.fixtures[:expected_decks]
    }
    if (
        content["deck_count"] != expected_decks
        or len(records) != expected_decks * 2
        or {record.fixture_id for record in records} != expected_fixture_ids
    ):
        raise Pi0EvaluationError("comparison report fixture coverage differs")
    recomputed = _comparison_metrics(
        records,
        comparison_id=content["comparison_id"],
        fixture_manifest_digest=fixture_manifest.digest,
        bootstrap_samples=fixture_manifest.protocol.bootstrap_samples,
    )
    if recomputed != content["metrics"]:
        raise Pi0EvaluationError("comparison metrics differ from game evidence")
    if content["extended"] is False and _needs_extension(recomputed):
        raise Pi0EvaluationError("comparison omitted its required extension")
    _assert_report_privacy(content)
    return {
        "path": str(report_path),
        "content_digest": document["content_digest"],
        "file_digest": _file_digest(report_path),
        "content": content,
    }


def _comparison_condition(
    comparison_id: str, metrics: Mapping[str, object]
) -> tuple[Literal["passed", "failed", "inconclusive"], tuple[str, ...]]:
    point = float(metrics["paired_score_differential"])
    interval = metrics["paired_score_differential_ci_95"]
    role_splits = metrics["role_splits"]
    if (
        not isinstance(interval, list)
        or len(interval) != 2
        or not isinstance(role_splits, dict)
    ):
        raise Pi0EvaluationError("comparison acceptance metrics are malformed")
    lower = float(interval[0])
    queen = float(role_splits["queen"]["mean_score_differential"])
    king = float(role_splits["king"]["mean_score_differential"])
    controller = comparison_id == "pi0-bgc-vs-base-bgc"
    required_role_minimum = 0.0 if controller else math.nextafter(0.0, math.inf)
    failures = []
    inconclusive = []
    if point < 0:
        failures.append("paired point estimate is negative")
    elif point == 0:
        inconclusive.append("paired point estimate is zero")
    elif lower <= 0:
        inconclusive.append("paired confidence interval includes zero")
    if queen < required_role_minimum:
        if queen < 0:
            failures.append("Queen point estimate is below its requirement")
        else:
            inconclusive.append("Queen point estimate is not strictly positive")
    if king < required_role_minimum:
        if king < 0:
            failures.append("King point estimate is below its requirement")
        else:
            inconclusive.append("King point estimate is not strictly positive")
    if failures:
        return "failed", tuple(failures)
    if inconclusive:
        return "inconclusive", tuple(inconclusive)
    return "passed", ()


def _expected_controller_identities(
    candidate: LoadedBGCPolicyArtifact,
    artifact_digest: str,
    protocol: EvaluationProtocol,
) -> dict[str, tuple[str, str, str, str]]:
    policy = StandalonePi0Opponent(
        candidate.model,
        artifact_digest=artifact_digest,
        artifact_identity=candidate.metadata.state_dict_digest,
    )
    random = RandomLegalOpponent()
    one_ply = OnePlyBeliefGreedyOpponent(protocol.belief_completion_count)
    search_config = BeliefGreedySearchConfig(
        outer_simulation_budget=protocol.outer_simulation_budget,
        belief_completion_count=protocol.belief_completion_count,
    )
    base = BGCSearchOpponent.base(search_config)
    candidate_controller = BGCSearchOpponent.pi0(policy, search_config)
    return {
        "standalone-random-legal": (
            policy.identity,
            policy.digest,
            random.identity,
            random.digest,
        ),
        "standalone-one-ply-belief-greedy": (
            policy.identity,
            policy.digest,
            one_ply.identity,
            one_ply.digest,
        ),
        "pi0-bgc-vs-base-bgc": (
            candidate_controller.identity,
            candidate_controller.digest,
            base.identity,
            base.digest,
        ),
    }


def build_validation_evidence(
    *,
    candidate_artifact_path: str | Path,
    fixture_manifest: EvaluationFixtureManifest,
    comparison_paths: Mapping[str, str | Path],
    output_path: str | Path,
) -> AcceptanceDecision:
    candidate_path = Path(candidate_artifact_path).expanduser().resolve()
    candidate = load_bgc_policy_artifact(candidate_path)
    artifact_digest = _file_digest(candidate_path)
    if candidate.metadata.candidate_status != BGC_POLICY_CANDIDATE_STATUS:
        raise Pi0EvaluationError("validation requires an unaccepted candidate")
    if set(comparison_paths) != set(_COMPARISONS):
        raise Pi0EvaluationError("validation requires all three comparisons")
    expected = _expected_controller_identities(
        candidate, artifact_digest, fixture_manifest.protocol
    )
    source = resolve_source_identity()
    reports = {}
    conditions = {}
    reasons: list[str] = []
    statuses = []
    for comparison_id in _COMPARISONS:
        report = load_comparison_report(
            comparison_paths[comparison_id],
            fixture_manifest=fixture_manifest,
            artifact_digest=artifact_digest,
        )
        content = report["content"]
        assert isinstance(content, dict)
        if content["comparison_id"] != comparison_id:
            raise Pi0EvaluationError("comparison path has the wrong identity")
        subject_identity, subject_digest, control_identity, control_digest = expected[
            comparison_id
        ]
        if content["subject"] != {
            "identity": subject_identity,
            "digest": subject_digest,
        } or content["control"] != {
            "identity": control_identity,
            "digest": control_digest,
        }:
            raise Pi0EvaluationError("comparison controller evidence is stale or forged")
        if (
            content["source_revision"] != source.revision
            or content["source_tree_digest"] != source.tree_digest
        ):
            raise Pi0EvaluationError("comparison source evidence is stale")
        metrics = content["metrics"]
        assert isinstance(metrics, dict)
        if metrics["illegal_action_count"] != 0:
            raise Pi0EvaluationError("comparison contains an illegal action")
        status, comparison_reasons = _comparison_condition(comparison_id, metrics)
        statuses.append(status)
        reasons.extend(
            f"{comparison_id}: {reason}" for reason in comparison_reasons
        )
        conditions[comparison_id] = {
            "status": status,
            "reasons": list(comparison_reasons),
        }
        reports[comparison_id] = {
            "content_digest": report["content_digest"],
            "file_digest": report["file_digest"],
            "path": str(Path(comparison_paths[comparison_id]).expanduser().resolve()),
            "fixture_manifest_digest": content["fixture_manifest_digest"],
            "metrics_digest": _json_digest(metrics),
        }
    if "failed" in statuses:
        overall: Literal["passed", "failed", "inconclusive"] = "failed"
    elif "inconclusive" in statuses:
        overall = "inconclusive"
    else:
        overall = "passed"
    evidence_content = {
        "artifact": {
            "candidate_status": candidate.metadata.candidate_status,
            "dataset_digest": candidate.metadata.dataset_digest,
            "file_digest": artifact_digest,
            "snapshot_digest": candidate.metadata.corpus_snapshot_digest,
            "state_dict_digest": candidate.metadata.state_dict_digest,
            "training_config_digest": candidate.metadata.training_config_digest,
            "training_source_revision": candidate.metadata.source_revision,
            "training_source_tree_digest": candidate.metadata.source_tree_digest,
        },
        "comparisons": reports,
        "conditions": conditions,
        "fixture_manifest_digest": fixture_manifest.digest,
        "overall_status": overall,
        "protocol": asdict(fixture_manifest.protocol),
        "protocol_digest": fixture_manifest.protocol.digest,
        "source_revision": source.revision,
        "source_tree_digest": source.tree_digest,
    }
    _assert_report_privacy(evidence_content)
    document = _envelope(VALIDATION_EVIDENCE_SCHEMA_VERSION, evidence_content)
    output = Path(output_path).expanduser().resolve()
    if output.exists():
        if _load_json(output) != document:
            raise Pi0EvaluationError("existing validation evidence differs")
    else:
        _atomic_json(output, document)
    return AcceptanceDecision(overall, tuple(reasons), None)


def load_validation_evidence(
    path: str | Path,
    *,
    candidate_artifact_path: str | Path,
    fixture_manifest: EvaluationFixtureManifest,
) -> dict[str, object]:
    evidence_path = Path(path).expanduser().resolve()
    document = _validate_envelope(
        _load_json(evidence_path), VALIDATION_EVIDENCE_SCHEMA_VERSION
    )
    content = document["content"]
    required = {
        "artifact",
        "comparisons",
        "conditions",
        "fixture_manifest_digest",
        "overall_status",
        "protocol",
        "protocol_digest",
        "source_revision",
        "source_tree_digest",
    }
    if (
        not isinstance(content, dict)
        or set(content) != required
        or content["fixture_manifest_digest"] != fixture_manifest.digest
        or content["protocol"] != asdict(fixture_manifest.protocol)
        or content["protocol_digest"] != fixture_manifest.protocol.digest
        or content["overall_status"] not in {"passed", "failed", "inconclusive"}
        or not isinstance(content["artifact"], dict)
        or not isinstance(content["comparisons"], dict)
        or set(content["comparisons"]) != set(_COMPARISONS)
        or not isinstance(content["conditions"], dict)
        or set(content["conditions"]) != set(_COMPARISONS)
    ):
        raise Pi0EvaluationError("validation evidence is incompatible")
    for comparison_id in _COMPARISONS:
        recorded = content["comparisons"][comparison_id]
        condition = content["conditions"][comparison_id]
        if (
            not isinstance(recorded, dict)
            or set(recorded)
            != {
                "content_digest",
                "file_digest",
                "fixture_manifest_digest",
                "metrics_digest",
                "path",
            }
            or not isinstance(recorded["path"], str)
            or not recorded["path"]
            or any(
                not isinstance(recorded[key], str)
                or len(recorded[key]) != _DIGEST_LENGTH
                for key in (
                    "content_digest",
                    "file_digest",
                    "fixture_manifest_digest",
                    "metrics_digest",
                )
            )
            or not isinstance(condition, dict)
            or set(condition) != {"reasons", "status"}
            or condition["status"] not in {"passed", "failed", "inconclusive"}
            or not isinstance(condition["reasons"], list)
            or any(not isinstance(reason, str) for reason in condition["reasons"])
        ):
            raise Pi0EvaluationError("validation evidence entries are malformed")
    candidate_path = Path(candidate_artifact_path).expanduser().resolve()
    candidate = load_bgc_policy_artifact(candidate_path)
    artifact = content["artifact"]
    expected_artifact = {
        "candidate_status": candidate.metadata.candidate_status,
        "dataset_digest": candidate.metadata.dataset_digest,
        "file_digest": _file_digest(candidate_path),
        "snapshot_digest": candidate.metadata.corpus_snapshot_digest,
        "state_dict_digest": candidate.metadata.state_dict_digest,
        "training_config_digest": candidate.metadata.training_config_digest,
        "training_source_revision": candidate.metadata.source_revision,
        "training_source_tree_digest": candidate.metadata.source_tree_digest,
    }
    if artifact != expected_artifact:
        raise Pi0EvaluationError("validation evidence belongs to another candidate")
    source = resolve_source_identity()
    if (
        content["source_revision"] != source.revision
        or content["source_tree_digest"] != source.tree_digest
    ):
        raise Pi0EvaluationError("validation evidence source is stale")
    expected_statuses = []
    for comparison_id in _COMPARISONS:
        report_path = content["comparisons"][comparison_id]["path"]
        report = load_comparison_report(
            report_path,
            fixture_manifest=fixture_manifest,
            artifact_digest=expected_artifact["file_digest"],
        )
        recorded = content["comparisons"][comparison_id]
        if (
            report["content_digest"] != recorded["content_digest"]
            or report["file_digest"] != recorded["file_digest"]
        ):
            raise Pi0EvaluationError("validation comparison digest changed")
        report_content = report["content"]
        assert isinstance(report_content, dict)
        metrics = report_content["metrics"]
        assert isinstance(metrics, dict)
        if _json_digest(metrics) != recorded["metrics_digest"]:
            raise Pi0EvaluationError("validation comparison metrics changed")
        status, reasons = _comparison_condition(comparison_id, metrics)
        if content["conditions"][comparison_id] != {
            "status": status,
            "reasons": list(reasons),
        }:
            raise Pi0EvaluationError("validation condition was forged")
        expected_statuses.append(status)
    expected_overall = (
        "failed"
        if "failed" in expected_statuses
        else "inconclusive"
        if "inconclusive" in expected_statuses
        else "passed"
    )
    if content["overall_status"] != expected_overall:
        raise Pi0EvaluationError("validation overall result was forged")
    _assert_report_privacy(content)
    return {
        "path": str(evidence_path),
        "content": content,
        "content_digest": document["content_digest"],
        "file_digest": _file_digest(evidence_path),
    }


def accept_pi0_candidate(
    *,
    candidate_artifact_path: str | Path,
    validation_evidence_path: str | Path,
    fixture_manifest: EvaluationFixtureManifest,
    output_directory: str | Path,
    require_production_protocol: bool = True,
    should_stop: Callable[[], bool] | None = None,
) -> AcceptanceDecision:
    _check_stop(should_stop)
    evidence = load_validation_evidence(
        validation_evidence_path,
        candidate_artifact_path=candidate_artifact_path,
        fixture_manifest=fixture_manifest,
    )
    content = evidence["content"]
    assert isinstance(content, dict)
    status = content["overall_status"]
    reasons = tuple(
        f"{comparison}: {reason}"
        for comparison, result in content["conditions"].items()
        for reason in result["reasons"]
    )
    if status != "passed":
        return AcceptanceDecision(status, reasons, None)
    if require_production_protocol and fixture_manifest.protocol != PRODUCTION_EVALUATION_PROTOCOL:
        raise Pi0EvaluationError("acceptance requires the production 60/180 protocol")
    _check_stop(should_stop)
    candidate_path = Path(candidate_artifact_path).expanduser().resolve()
    candidate = load_bgc_policy_artifact(candidate_path)
    output = Path(output_directory).expanduser().resolve()
    if output.exists():
        raise Pi0EvaluationError("accepted checkpoint directory already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{output.name}.tmp-{os.getpid()}"
    if staging.exists():
        shutil.rmtree(staging)
    try:
        staging.mkdir()
        checkpoint = staging / "accepted-pi0.pt"
        shutil.copyfile(candidate_path, checkpoint)
        with checkpoint.open("rb") as stream:
            os.fsync(stream.fileno())
        checkpoint_digest = _file_digest(checkpoint)
        _check_stop(should_stop)
        evidence_copy = staging / "validation-evidence.json"
        shutil.copyfile(Path(validation_evidence_path).expanduser().resolve(), evidence_copy)
        if _file_digest(evidence_copy) != evidence["file_digest"]:
            raise Pi0EvaluationError("validation evidence changed during acceptance")
        fixture_copy = staging / "evaluation-fixtures.json"
        shutil.copyfile(fixture_manifest.path, fixture_copy)
        if _file_digest(fixture_copy) != _file_digest(fixture_manifest.path):
            raise Pi0EvaluationError("evaluation fixtures changed during acceptance")
        comparison_copies: dict[str, dict[str, str]] = {}
        comparisons_directory = staging / "comparisons"
        comparisons_directory.mkdir()
        comparisons = content["comparisons"]
        assert isinstance(comparisons, dict)
        for comparison_id in _COMPARISONS:
            recorded = comparisons[comparison_id]
            source_report = Path(recorded["path"])
            destination = comparisons_directory / f"{comparison_id}.json"
            shutil.copyfile(source_report, destination)
            if _file_digest(destination) != recorded["file_digest"]:
                raise Pi0EvaluationError("comparison changed during acceptance")
            comparison_copies[comparison_id] = {
                "content_digest": recorded["content_digest"],
                "file_digest": recorded["file_digest"],
                "metrics_digest": recorded["metrics_digest"],
                "path": f"comparisons/{comparison_id}.json",
            }
        _check_stop(should_stop)
        source = resolve_source_identity()
        acceptance_content = {
            "accepted_checkpoint_digest": checkpoint_digest,
            "action_schema_digest": _json_digest(
                {"schema_version": ACTION_SCHEMA_VERSION}
            ),
            "action_schema_version": ACTION_SCHEMA_VERSION,
            "candidate_artifact_digest": _file_digest(candidate_path),
            "candidate_state_dict_digest": candidate.metadata.state_dict_digest,
            "comparison_reports": comparison_copies,
            "dataset_digest": candidate.metadata.dataset_digest,
            "evaluation_fixture_manifest_digest": fixture_manifest.digest,
            "evaluation_fixture_manifest_file_digest": _file_digest(fixture_copy),
            "evaluation_report_digest": evidence["content_digest"],
            "evaluation_report_file_digest": evidence["file_digest"],
            "evaluation_result": "passed",
            "loss_schema_version": candidate.metadata.loss_schema_version,
            "observation_schema_digest": _json_digest(
                {"schema_version": OBSERVATION_SCHEMA_VERSION}
            ),
            "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
            "model_schema_version": candidate.metadata.model_schema_version,
            "optimizer_compatibility_version": (
                candidate.metadata.optimizer_compatibility_version
            ),
            "protocol_digest": fixture_manifest.protocol.digest,
            "representative_mask_schema_digest": _json_digest(
                {"schema_version": REPRESENTATIVE_MASK_SCHEMA_VERSION}
            ),
            "representative_mask_schema_version": (
                REPRESENTATIVE_MASK_SCHEMA_VERSION
            ),
            "snapshot_digest": candidate.metadata.corpus_snapshot_digest,
            "source_revision": source.revision,
            "source_tree_digest": source.tree_digest,
            "training_config_digest": candidate.metadata.training_config_digest,
            "training_source_revision": candidate.metadata.source_revision,
            "training_source_tree_digest": candidate.metadata.source_tree_digest,
            "destination_symmetry_schema_digest": _json_digest(
                {"schema_version": DESTINATION_SYMMETRY_SCHEMA_VERSION}
            ),
            "destination_symmetry_schema_version": (
                DESTINATION_SYMMETRY_SCHEMA_VERSION
            ),
        }
        manifest = _envelope(ACCEPTANCE_SCHEMA_VERSION, acceptance_content)
        _atomic_json(staging / "acceptance-manifest.json", manifest)
        _validate_envelope(
            _load_json(staging / "acceptance-manifest.json"),
            ACCEPTANCE_SCHEMA_VERSION,
        )
        _check_stop(should_stop)
        directory_handle = os.open(staging, os.O_RDONLY)
        try:
            os.fsync(directory_handle)
        finally:
            os.close(directory_handle)
        _check_stop(should_stop)
        os.replace(staging, output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return AcceptanceDecision("passed", (), str(output))


def load_accepted_pi0_bundle(path: str | Path) -> AcceptedPi0Bundle:
    directory = Path(path).expanduser().resolve()
    manifest_path = directory / "acceptance-manifest.json"
    document = _validate_envelope(
        _load_json(manifest_path), ACCEPTANCE_SCHEMA_VERSION
    )
    content = document["content"]
    required = {
        "accepted_checkpoint_digest",
        "action_schema_digest",
        "action_schema_version",
        "candidate_artifact_digest",
        "candidate_state_dict_digest",
        "comparison_reports",
        "dataset_digest",
        "destination_symmetry_schema_digest",
        "destination_symmetry_schema_version",
        "evaluation_fixture_manifest_digest",
        "evaluation_fixture_manifest_file_digest",
        "evaluation_report_digest",
        "evaluation_report_file_digest",
        "evaluation_result",
        "loss_schema_version",
        "model_schema_version",
        "observation_schema_digest",
        "observation_schema_version",
        "optimizer_compatibility_version",
        "protocol_digest",
        "representative_mask_schema_digest",
        "representative_mask_schema_version",
        "snapshot_digest",
        "source_revision",
        "source_tree_digest",
        "training_config_digest",
        "training_source_revision",
        "training_source_tree_digest",
    }
    if not isinstance(content, dict) or set(content) != required:
        raise Pi0EvaluationError("accepted pi0 manifest is incompatible")
    expected_contracts = {
        "action_schema_version": ACTION_SCHEMA_VERSION,
        "action_schema_digest": _json_digest(
            {"schema_version": ACTION_SCHEMA_VERSION}
        ),
        "destination_symmetry_schema_version": (
            DESTINATION_SYMMETRY_SCHEMA_VERSION
        ),
        "destination_symmetry_schema_digest": _json_digest(
            {"schema_version": DESTINATION_SYMMETRY_SCHEMA_VERSION}
        ),
        "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
        "observation_schema_digest": _json_digest(
            {"schema_version": OBSERVATION_SCHEMA_VERSION}
        ),
        "representative_mask_schema_version": REPRESENTATIVE_MASK_SCHEMA_VERSION,
        "representative_mask_schema_digest": _json_digest(
            {"schema_version": REPRESENTATIVE_MASK_SCHEMA_VERSION}
        ),
    }
    if any(content[key] != value for key, value in expected_contracts.items()):
        raise Pi0EvaluationError("accepted pi0 action or masking contract differs")
    if content["evaluation_result"] != "passed":
        raise Pi0EvaluationError("accepted pi0 evaluation did not pass")
    checkpoint_path = directory / "accepted-pi0.pt"
    checkpoint_digest = _file_digest(checkpoint_path)
    if (
        content["accepted_checkpoint_digest"] != checkpoint_digest
        or content["candidate_artifact_digest"] != checkpoint_digest
    ):
        raise Pi0EvaluationError("accepted pi0 checkpoint digest differs")
    artifact = load_bgc_policy_artifact(checkpoint_path)
    metadata = artifact.metadata
    if (
        metadata.state_dict_digest != content["candidate_state_dict_digest"]
        or metadata.dataset_digest != content["dataset_digest"]
        or metadata.corpus_snapshot_digest != content["snapshot_digest"]
        or metadata.training_config_digest != content["training_config_digest"]
        or metadata.source_revision != content["training_source_revision"]
        or metadata.source_tree_digest != content["training_source_tree_digest"]
        or metadata.loss_schema_version != content["loss_schema_version"]
        or metadata.model_schema_version != content["model_schema_version"]
        or metadata.optimizer_compatibility_version
        != content["optimizer_compatibility_version"]
        or metadata.action_schema_version != content["action_schema_version"]
        or metadata.observation_schema_version
        != content["observation_schema_version"]
        or metadata.representative_mask_schema_version
        != content["representative_mask_schema_version"]
        or metadata.destination_symmetry_schema_version
        != content["destination_symmetry_schema_version"]
    ):
        raise Pi0EvaluationError("accepted pi0 model or training identity differs")

    fixture_path = directory / "evaluation-fixtures.json"
    if _file_digest(fixture_path) != content["evaluation_fixture_manifest_file_digest"]:
        raise Pi0EvaluationError("accepted pi0 fixture file digest differs")
    fixture_manifest = load_evaluation_fixture_manifest(
        fixture_path, protocol=PRODUCTION_EVALUATION_PROTOCOL
    )
    if (
        fixture_manifest.digest != content["evaluation_fixture_manifest_digest"]
        or fixture_manifest.protocol.digest != content["protocol_digest"]
    ):
        raise Pi0EvaluationError("accepted pi0 fixture identity differs")

    evidence_path = directory / "validation-evidence.json"
    if _file_digest(evidence_path) != content["evaluation_report_file_digest"]:
        raise Pi0EvaluationError("accepted pi0 evidence file digest differs")
    evidence_document = _validate_envelope(
        _load_json(evidence_path), VALIDATION_EVIDENCE_SCHEMA_VERSION
    )
    evidence = evidence_document["content"]
    if (
        evidence_document["content_digest"] != content["evaluation_report_digest"]
        or not isinstance(evidence, dict)
        or evidence.get("overall_status") != "passed"
        or evidence.get("fixture_manifest_digest") != fixture_manifest.digest
        or evidence.get("protocol_digest") != fixture_manifest.protocol.digest
        or evidence.get("source_revision") != content["source_revision"]
        or evidence.get("source_tree_digest") != content["source_tree_digest"]
    ):
        raise Pi0EvaluationError("accepted pi0 validation evidence differs")
    artifact_evidence = evidence.get("artifact")
    if (
        not isinstance(artifact_evidence, dict)
        or artifact_evidence.get("file_digest") != checkpoint_digest
        or artifact_evidence.get("state_dict_digest")
        != metadata.state_dict_digest
        or artifact_evidence.get("dataset_digest") != metadata.dataset_digest
        or artifact_evidence.get("snapshot_digest")
        != metadata.corpus_snapshot_digest
        or artifact_evidence.get("training_config_digest")
        != metadata.training_config_digest
    ):
        raise Pi0EvaluationError("accepted pi0 artifact evidence differs")
    recorded_reports = content["comparison_reports"]
    evidence_reports = evidence.get("comparisons")
    evidence_conditions = evidence.get("conditions")
    if (
        not isinstance(recorded_reports, dict)
        or set(recorded_reports) != set(_COMPARISONS)
        or not isinstance(evidence_reports, dict)
        or set(evidence_reports) != set(_COMPARISONS)
        or not isinstance(evidence_conditions, dict)
        or set(evidence_conditions) != set(_COMPARISONS)
    ):
        raise Pi0EvaluationError("accepted pi0 comparison set differs")
    for comparison_id in _COMPARISONS:
        condition = evidence_conditions[comparison_id]
        if not isinstance(condition, dict) or condition.get("status") != "passed":
            raise Pi0EvaluationError("accepted pi0 comparison did not pass")
        copied = recorded_reports[comparison_id]
        source_record = evidence_reports[comparison_id]
        if (
            not isinstance(copied, dict)
            or set(copied)
            != {"content_digest", "file_digest", "metrics_digest", "path"}
            or not isinstance(source_record, dict)
            or any(
                copied[key] != source_record[key]
                for key in ("content_digest", "file_digest", "metrics_digest")
            )
        ):
            raise Pi0EvaluationError("accepted pi0 comparison identity differs")
        report_path = directory / copied["path"]
        report = load_comparison_report(
            report_path,
            fixture_manifest=fixture_manifest,
            artifact_digest=checkpoint_digest,
        )
        report_content = report["content"]
        assert isinstance(report_content, dict)
        if (
            report["content_digest"] != copied["content_digest"]
            or report["file_digest"] != copied["file_digest"]
            or _json_digest(report_content["metrics"]) != copied["metrics_digest"]
            or report_content["source_revision"] != content["source_revision"]
            or report_content["source_tree_digest"] != content["source_tree_digest"]
        ):
            raise Pi0EvaluationError("accepted pi0 comparison evidence differs")
    return AcceptedPi0Bundle(
        directory=directory,
        checkpoint_path=checkpoint_path,
        checkpoint_digest=checkpoint_digest,
        acceptance_content_digest=document["content_digest"],
        acceptance_file_digest=_file_digest(manifest_path),
        artifact=artifact,
    )


def _load_production_manifest(path: str | Path) -> EvaluationFixtureManifest:
    return load_evaluation_fixture_manifest(
        path, protocol=PRODUCTION_EVALUATION_PROTOCOL
    )


def _run_standalone_cli(arguments: argparse.Namespace) -> None:
    fixtures = _load_production_manifest(arguments.fixtures)
    policy = StandalonePi0Opponent.from_artifact(arguments.candidate)
    control: EvaluationOpponent
    comparison_id: str
    if arguments.control == "random":
        control = RandomLegalOpponent()
        comparison_id = "standalone-random-legal"
    else:
        control = OnePlyBeliefGreedyOpponent(
            fixtures.protocol.belief_completion_count
        )
        comparison_id = "standalone-one-ply-belief-greedy"
    result = run_paired_comparison(
        comparison_id=comparison_id,
        comparison_type="standalone",
        subject=policy,
        control=control,
        fixture_manifest=fixtures,
        artifact_digest=policy.artifact_digest,
        output_path=arguments.output,
    )
    print(json.dumps({"comparison": comparison_id, "report": str(result)}))


def _run_controller_cli(arguments: argparse.Namespace) -> None:
    fixtures = _load_production_manifest(arguments.fixtures)
    policy = StandalonePi0Opponent.from_artifact(arguments.candidate)
    config = BeliefGreedySearchConfig(
        outer_simulation_budget=fixtures.protocol.outer_simulation_budget,
        belief_completion_count=fixtures.protocol.belief_completion_count,
    )
    candidate = BGCSearchOpponent.pi0(policy, config)
    base = BGCSearchOpponent.base(config)
    result = run_paired_comparison(
        comparison_id="pi0-bgc-vs-base-bgc",
        comparison_type="controller",
        subject=candidate,
        control=base,
        fixture_manifest=fixtures,
        artifact_digest=policy.artifact_digest,
        output_path=arguments.output,
    )
    print(
        json.dumps(
            {"comparison": "pi0-bgc-vs-base-bgc", "report": str(result)}
        )
    )


def _run_incremental_controller_cli(arguments: argparse.Namespace) -> None:
    fixtures = _load_production_manifest(arguments.fixtures)
    policy = StandalonePi0Opponent.from_artifact(arguments.candidate)
    config = BeliefGreedySearchConfig(
        outer_simulation_budget=fixtures.protocol.outer_simulation_budget,
        belief_completion_count=fixtures.protocol.belief_completion_count,
    )
    candidate = BGCSearchOpponent.pi0(policy, config)
    base = BGCSearchOpponent.base(config)

    def emit(update: Mapping[str, object]) -> None:
        print(json.dumps(update, sort_keys=True), flush=True)

    result = run_incremental_controller_comparison(
        subject=candidate,
        control=base,
        fixture_manifest=fixtures,
        artifact_digest=policy.artifact_digest,
        output_path=arguments.output,
        target_deck_count=arguments.decks,
        progress_callback=emit,
    )
    print(json.dumps({"complete": True, "report": str(result)}), flush=True)


def _throughput_game(
    request: tuple[str, str, int],
) -> dict[str, object]:
    artifact_path, root_seed, ordinal = request
    torch.set_num_threads(1)
    policy = StandalonePi0Opponent.from_artifact(artifact_path)
    config = BeliefGreedySearchConfig(
        outer_simulation_budget=128,
        belief_completion_count=8,
    )
    opponent = BGCSearchOpponent.pi0(policy, config)
    deck_seed = derive_seed(
        THROUGHPUT_DECK_NAMESPACE, root_seed, str(ordinal)
    ).hex()
    fixture_id = _json_digest(
        {"deck_seed": deck_seed, "namespace": THROUGHPUT_DECK_NAMESPACE}
    )
    state = create_game(deck_seed)
    latencies: list[float] = []
    decision_index = 0
    started = time.perf_counter()
    while state.status is not EngineStatus.GAME_COMPLETE:
        while state.status is EngineStatus.PLAYING:
            actor = state.active_player
            if actor is None:
                raise Pi0EvaluationError("throughput game has no active player")
            moves = legal_moves(state, actor)
            if len(moves) == 1:
                state = apply_move(state, moves[0]).state
                continue
            decision = opponent.decide(
                information_state_from_engine(state),
                fixture_id=fixture_id,
                decision_index=decision_index,
            )
            decision_index += 1
            move = move_for_action_index(actor, decision.concrete_action_index)
            if move not in moves:
                raise Pi0EvaluationError("throughput controller selected an illegal move")
            latencies.append(decision.latency_seconds)
            state = apply_move(state, move).state
        state = advance_after_round(state)
    if decision_index != 42 or len(latencies) != 42:
        raise Pi0EvaluationError("throughput game did not contain 42 learned decisions")
    return {
        "decision_count": decision_index,
        "elapsed_seconds": time.perf_counter() - started,
        "maximum_decision_seconds": max(latencies),
        "mean_decision_seconds": sum(latencies) / len(latencies),
        "ordinal": ordinal,
        "peak_rss_bytes": _peak_rss_bytes(),
        "search_seconds": sum(latencies),
    }


def run_pi0_bgc_throughput_benchmark(
    *,
    candidate_path: str | Path,
    output_path: str | Path,
    root_seed: str,
    game_count: int,
    worker_count: int,
) -> Path:
    if (
        not isinstance(root_seed, str)
        or not root_seed
        or type(game_count) is not int
        or type(worker_count) is not int
        or worker_count < 1
        or game_count < worker_count
    ):
        raise Pi0EvaluationError(
            "throughput requires a nonempty seed and at least one game per worker"
        )
    candidate = Path(candidate_path).expanduser().resolve()
    policy = StandalonePi0Opponent.from_artifact(candidate)
    requests = tuple((str(candidate), root_seed, ordinal) for ordinal in range(game_count))
    started = time.perf_counter()
    if worker_count == 1:
        games = tuple(_throughput_game(request) for request in requests)
    else:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            games = tuple(executor.map(_throughput_game, requests))
    wall_seconds = time.perf_counter() - started
    row_count = sum(int(game["decision_count"]) for game in games)
    rows_per_second = row_count / wall_seconds
    content = {
        "artifact_digest": policy.artifact_digest,
        "controller_digest": BGCSearchOpponent.pi0(
            policy,
            BeliefGreedySearchConfig(
                outer_simulation_budget=128,
                belief_completion_count=8,
            ),
        ).digest,
        "game_count": game_count,
        "games": list(games),
        "outer_simulations": 128,
        "response_policy": "pi0-argmax-v1",
        "root_seed": root_seed,
        "row_count": row_count,
        "rows_per_second": rows_per_second,
        "estimated_rows_per_24_hours": rows_per_second * 86_400.0,
        "wall_seconds": wall_seconds,
        "worker_count": worker_count,
    }
    _assert_report_privacy(content)
    destination = Path(output_path).expanduser().resolve()
    _atomic_json(destination, _envelope(THROUGHPUT_REPORT_SCHEMA_VERSION, content))
    return destination


def _build_evidence_cli(arguments: argparse.Namespace) -> None:
    fixtures = _load_production_manifest(arguments.fixtures)
    decision = build_validation_evidence(
        candidate_artifact_path=arguments.candidate,
        fixture_manifest=fixtures,
        comparison_paths={
            "standalone-random-legal": arguments.random_report,
            "standalone-one-ply-belief-greedy": arguments.one_ply_report,
            "pi0-bgc-vs-base-bgc": arguments.controller_report,
        },
        output_path=arguments.output,
    )
    print(json.dumps(asdict(decision), sort_keys=True))


def _accept_cli(arguments: argparse.Namespace) -> None:
    fixtures = _load_production_manifest(arguments.fixtures)
    decision = accept_pi0_candidate(
        candidate_artifact_path=arguments.candidate,
        validation_evidence_path=arguments.evidence,
        fixture_manifest=fixtures,
        output_directory=arguments.output,
    )
    print(json.dumps(asdict(decision), sort_keys=True))


def _inspect_cli(arguments: argparse.Namespace) -> None:
    fixtures = _load_production_manifest(arguments.fixtures)
    if arguments.kind == "comparison":
        candidate_digest = _file_digest(Path(arguments.candidate).expanduser().resolve())
        result = load_comparison_report(
            arguments.path,
            fixture_manifest=fixtures,
            artifact_digest=candidate_digest,
        )
    else:
        result = load_validation_evidence(
            arguments.path,
            candidate_artifact_path=arguments.candidate,
            fixture_manifest=fixtures,
        )
    print(
        json.dumps(
            {
                "content_digest": result["content_digest"],
                "file_digest": result["file_digest"],
                "path": result["path"],
            },
            sort_keys=True,
        )
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dracula-bgc-policy-evaluation",
        description="Evaluate and explicitly accept an unaccepted pi0 candidate.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    fixtures = commands.add_parser("fixtures", help="seal the production fixtures")
    fixtures.add_argument("--root-seed", required=True)
    fixtures.add_argument("--output", required=True, type=Path)

    standalone = commands.add_parser(
        "standalone", help="compare pi0 with a standalone control"
    )
    standalone.add_argument("--candidate", required=True, type=Path)
    standalone.add_argument("--fixtures", required=True, type=Path)
    standalone.add_argument("--control", choices=("random", "one-ply"), required=True)
    standalone.add_argument("--output", required=True, type=Path)

    controller = commands.add_parser(
        "controller", help="compare pi0-BGC-128 with base BGC-128"
    )
    controller.add_argument("--candidate", required=True, type=Path)
    controller.add_argument("--fixtures", required=True, type=Path)
    controller.add_argument("--output", required=True, type=Path)

    incremental_controller = commands.add_parser(
        "controller-progress",
        help="compare the controllers and seal cumulative metrics after every deck pair",
    )
    incremental_controller.add_argument("--candidate", required=True, type=Path)
    incremental_controller.add_argument("--fixtures", required=True, type=Path)
    incremental_controller.add_argument("--output", required=True, type=Path)
    incremental_controller.add_argument("--decks", type=int, default=12)

    benchmark = commands.add_parser(
        "benchmark", help="measure four-worker pi0-BGC collection throughput"
    )
    benchmark.add_argument("--candidate", required=True, type=Path)
    benchmark.add_argument("--output", required=True, type=Path)
    benchmark.add_argument("--root-seed", required=True)
    benchmark.add_argument("--games", type=int, default=8)
    benchmark.add_argument("--workers", type=int, default=4)

    evidence = commands.add_parser(
        "evidence", help="verify all reports and seal validation evidence"
    )
    evidence.add_argument("--candidate", required=True, type=Path)
    evidence.add_argument("--fixtures", required=True, type=Path)
    evidence.add_argument("--random-report", required=True, type=Path)
    evidence.add_argument("--one-ply-report", required=True, type=Path)
    evidence.add_argument("--controller-report", required=True, type=Path)
    evidence.add_argument("--output", required=True, type=Path)

    accept = commands.add_parser(
        "accept", help="atomically accept only completely passing evidence"
    )
    accept.add_argument("--candidate", required=True, type=Path)
    accept.add_argument("--fixtures", required=True, type=Path)
    accept.add_argument("--evidence", required=True, type=Path)
    accept.add_argument("--output", required=True, type=Path)

    inspect = commands.add_parser("inspect", help="verify a report or evidence")
    inspect.add_argument("kind", choices=("comparison", "evidence"))
    inspect.add_argument("--path", required=True, type=Path)
    inspect.add_argument("--candidate", required=True, type=Path)
    inspect.add_argument("--fixtures", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "fixtures":
        result = create_evaluation_fixture_manifest(
            arguments.output,
            root_seed=arguments.root_seed,
            protocol=PRODUCTION_EVALUATION_PROTOCOL,
        )
        print(json.dumps({"digest": result.digest, "path": str(result.path)}))
    elif arguments.command == "standalone":
        _run_standalone_cli(arguments)
    elif arguments.command == "controller":
        _run_controller_cli(arguments)
    elif arguments.command == "controller-progress":
        _run_incremental_controller_cli(arguments)
    elif arguments.command == "benchmark":
        result = run_pi0_bgc_throughput_benchmark(
            candidate_path=arguments.candidate,
            output_path=arguments.output,
            root_seed=arguments.root_seed,
            game_count=arguments.games,
            worker_count=arguments.workers,
        )
        print(json.dumps({"benchmark": str(result)}))
    elif arguments.command == "evidence":
        _build_evidence_cli(arguments)
    elif arguments.command == "accept":
        _accept_cli(arguments)
    else:
        _inspect_cli(arguments)
    return 0


__all__ = (
    "AcceptanceDecision",
    "AcceptedPi0Bundle",
    "BGCSearchOpponent",
    "EvaluationFixture",
    "EvaluationFixtureManifest",
    "EvaluationProtocol",
    "INCREMENTAL_CONTROLLER_SCHEMA_VERSION",
    "GameEvaluationRecord",
    "OnePlyBeliefGreedyOpponent",
    "OpponentDecision",
    "PRODUCTION_EVALUATION_PROTOCOL",
    "Pi0ContinuationBeliefGreedySearch",
    "Pi0EvaluationError",
    "Pi0EvaluationInterrupted",
    "RandomLegalOpponent",
    "RoundEvaluationRecord",
    "StandalonePi0Opponent",
    "accept_pi0_candidate",
    "build_validation_evidence",
    "create_evaluation_fixture_manifest",
    "load_comparison_report",
    "load_accepted_pi0_bundle",
    "load_evaluation_fixture_manifest",
    "load_validation_evidence",
    "main",
    "paired_confidence_interval",
    "run_paired_comparison",
    "run_pi0_bgc_throughput_benchmark",
)


if __name__ == "__main__":
    raise SystemExit(main())

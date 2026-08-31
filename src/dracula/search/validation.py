"""Reproducible strategic fixtures and absolute-control search evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import resource
import statistics
import sys
import tempfile
import time
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

from dracula.bridge import PolicyTurnKind, build_policy_turn_context
from dracula.engine import (
    EnginePlayer,
    EngineState,
    EngineStatus,
    advance_after_round,
    apply_move,
    create_game,
    derive_game_outcome,
    legal_moves,
    other_player,
)
from dracula.local_policy import LocalTorchPolicyAdapter
from dracula.policy_adapter import (
    POLICY_INFERENCE_CONTRACT_VERSION,
    PolicyInferenceRequest,
    resolve_inference_profile,
    select_masked_action,
)
from dracula.randomness import Sha256CounterStream, derive_seed
from dracula.search.diagnostic import solve_perfect_information_round
from dracula.search.information import (
    derive_search_request_seed,
    information_state_from_engine,
    sample_determinization,
)
from dracula.search.planner import InformationSetSearch, SearchConfig
from dracula.search.strategic import (
    StrategicInformationSetSearch,
    StrategicSearchConfig,
    derive_strategic_search_request_seed,
)
from dracula.search.strategic_fixtures import (
    STRATEGIC_FIXTURE_SCHEMA_VERSION,
    STRATEGIC_FIXTURES,
    FixtureEvidence,
    StrategicFixture,
    fixture_action_features,
    replay_fixture,
    validate_fixture_semantics,
)

SEARCH_VALIDATION_FORMAT_VERSION = "dracula-search-validation-v1"
VALIDATION_GAME_NAMESPACE = "dracula-search-validation-game-v1"
VALIDATION_RANDOM_NAMESPACE = "dracula-search-validation-random-v1"
VALIDATION_BOOTSTRAP_NAMESPACE = "dracula-search-validation-bootstrap-v1"
VALIDATION_EXACT_WORLD_NAMESPACE = "dracula-search-validation-exact-world-v1"
DEFAULT_POLICY_ARCHIVE = Path(
    "runs/training-004/archives/policy-2-policy-2-v20.pt"
)
MINIMUM_CONTROL_GAME_PAIRS = 12


class SearchValidationError(RuntimeError):
    """The validation configuration or evidence is incomplete."""


@dataclass(frozen=True, slots=True)
class ValidationConfig:
    budgets: tuple[int, ...] = (100, 500, 2_000)
    selected_budget: int = 500
    sensitivity_repetitions: int = 3
    game_pairs: int = MINIMUM_CONTROL_GAME_PAIRS
    workers: int = 4
    bootstrap_samples: int = 2_000

    def __post_init__(self) -> None:
        if (
            not self.budgets
            or tuple(sorted(set(self.budgets))) != self.budgets
            or any(type(value) is not int or value < 1 for value in self.budgets)
        ):
            raise ValueError("validation budgets must be distinct positive integers")
        if self.selected_budget not in self.budgets:
            raise ValueError("selected budget must be one of the benchmark budgets")
        for value, label in (
            (self.sensitivity_repetitions, "sensitivity repetitions"),
            (self.game_pairs, "game pairs"),
            (self.workers, "workers"),
            (self.bootstrap_samples, "bootstrap samples"),
        ):
            if type(value) is not int or value < 1:
                raise ValueError(f"{label} must be a positive integer")


@dataclass(frozen=True, slots=True)
class FixtureSearchRecord:
    fixture_id: str
    budget: int
    repetition: int
    selected_action_index: int
    passed: bool
    simulations: int
    latency_seconds: float
    information_sets: int
    action_statistics: tuple[tuple[int, int, float | None], ...]


@dataclass(frozen=True, slots=True)
class RoundComparison:
    round_number: int
    subject_dealer: bool
    subject_score: int
    opponent_score: int


@dataclass(frozen=True, slots=True)
class DecisionTiming:
    controller: str
    budget: int | None
    round_number: int
    accepted_moves: int
    simulations: int
    latency_seconds: float


@dataclass(frozen=True, slots=True)
class GameComparisonRecord:
    comparison: str
    game_seed: str
    subject_role: EnginePlayer
    subject_won: bool
    tied: bool
    rounds: tuple[RoundComparison, ...]
    decisions: tuple[DecisionTiming, ...]


@dataclass(frozen=True, slots=True)
class _ControllerSpec:
    kind: str
    budget: int | None = None
    response_completions: int | None = None
    archive_path: str | None = None


class _Controller:
    def __init__(self, spec: _ControllerSpec, game_id: str) -> None:
        self.spec = spec
        self.game_id = game_id
        self.planner = (
            InformationSetSearch(SearchConfig(spec.budget))
            if spec.kind == "search" and spec.budget is not None
            else None
        )
        self.strategic_planner = (
            StrategicInformationSetSearch(
                StrategicSearchConfig(spec.budget, spec.response_completions)
            )
            if spec.kind == "strategic"
            and spec.budget is not None
            and spec.response_completions is not None
            else None
        )
        self.adapter = (
            LocalTorchPolicyAdapter(spec.archive_path)
            if spec.kind == "ppo" and spec.archive_path is not None
            else None
        )
        self.profile = resolve_inference_profile("argmax-v1")
        self.hidden = bytes(512)

    def select(self, state: EngineState) -> tuple[object, DecisionTiming | None]:
        actor = state.active_player
        if actor is None:
            raise SearchValidationError("controller received a state without an actor")
        moves = legal_moves(state, actor)
        context = build_policy_turn_context(state, actor)
        if self.spec.kind == "search":
            if len(moves) == 1:
                return moves[0], None
            if self.planner is None or self.spec.budget is None:
                raise SearchValidationError("search controller is not configured")
            information = information_state_from_engine(state)
            request_seed = derive_search_request_seed(
                (
                    f"{self.game_id}:{actor.value}:{state.round_number}:"
                    f"{len(state.current_round_moves)}"
                ),
                information,
                self.planner.config.digest,
            )
            started = time.perf_counter()
            result = self.planner.search(information, request_seed)
            elapsed = time.perf_counter() - started
            move = context.action_table[result.selected_action_index]
            if move is None:
                raise SearchValidationError("search selected a masked action")
            return move, DecisionTiming(
                "search",
                self.spec.budget,
                state.round_number,
                len(state.current_round_moves),
                result.simulation_count,
                elapsed,
            )
        if self.spec.kind == "strategic":
            if len(moves) == 1:
                return moves[0], None
            if (
                self.strategic_planner is None
                or self.spec.budget is None
                or self.spec.response_completions is None
            ):
                raise SearchValidationError("strategic controller is not configured")
            information = information_state_from_engine(state)
            request_seed = derive_strategic_search_request_seed(
                (
                    f"{self.game_id}:{actor.value}:{state.round_number}:"
                    f"{len(state.current_round_moves)}"
                ),
                information,
                self.strategic_planner.config.digest,
            )
            started = time.perf_counter()
            result = self.strategic_planner.search(information, request_seed)
            elapsed = time.perf_counter() - started
            move = context.action_table[result.selected_action_index]
            if move is None:
                raise SearchValidationError("strategic search selected a masked action")
            return move, DecisionTiming(
                "strategic",
                self.spec.budget,
                state.round_number,
                len(state.current_round_moves),
                result.total_terminal_evaluation_count,
                elapsed,
            )
        if self.spec.kind == "random":
            if len(moves) == 1:
                return moves[0], None
            seed = derive_seed(
                VALIDATION_RANDOM_NAMESPACE,
                self.game_id,
                actor.value,
                str(state.round_number),
                str(len(state.current_round_moves)),
            )
            return moves[Sha256CounterStream(seed).randbelow(len(moves))], None
        if self.adapter is None:
            raise SearchValidationError("PPO controller is not configured")
        observation = tuple(
            bool(value) for value in context.input.observation.detach().cpu().tolist()
        )
        legal_mask = tuple(
            tuple(bool(value) for value in row)
            for row in context.input.legal_mask.detach().cpu().tolist()
        )
        response = self.adapter.invoke(
            PolicyInferenceRequest(
                POLICY_INFERENCE_CONTRACT_VERSION,
                self.adapter.metadata.artifact_id,
                observation,
                legal_mask,
                self.hidden,
            )
        )
        self.hidden = response.next_hidden_state
        if context.kind is PolicyTurnKind.FORCED_RECURRENT_TRANSITION:
            return moves[0], None
        action_index = select_masked_action(
            response.raw_logits,
            legal_mask,
            self.profile,
            game_id=self.game_id,
            round_number=state.round_number,
            turn_number=len(state.current_round_moves) + 1,
            artifact_id=self.adapter.metadata.artifact_id,
        )
        move = context.action_table[action_index]
        if move is None:
            raise SearchValidationError("PPO control selected a masked action")
        return move, None


def _fixture_by_id(fixture_id: str) -> StrategicFixture:
    try:
        return next(fixture for fixture in STRATEGIC_FIXTURES if fixture.fixture_id == fixture_id)
    except StopIteration as error:
        raise SearchValidationError(f"unknown strategic fixture: {fixture_id}") from error


def _run_fixture_search(task: tuple[str, int, int]) -> FixtureSearchRecord:
    fixture_id, budget, repetition = task
    fixture = _fixture_by_id(fixture_id)
    state = replay_fixture(fixture)
    information = information_state_from_engine(state)
    config = SearchConfig(budget)
    request_seed = derive_search_request_seed(
        f"{fixture.fixture_id}:sensitivity:{repetition}", information, config.digest
    )
    started = time.perf_counter()
    result = InformationSetSearch(config).search(information, request_seed)
    latency = time.perf_counter() - started
    statistics_rows = tuple(
        (index, result.action_visits[index], result.mean_action_values[index])
        for index in range(len(result.action_visits))
        if result.action_visits[index] or result.mean_action_values[index] is not None
    )
    return FixtureSearchRecord(
        fixture.fixture_id,
        budget,
        repetition,
        result.selected_action_index,
        result.selected_action_index in fixture.expected_action_indices,
        result.simulation_count,
        latency,
        result.information_set_count,
        statistics_rows,
    )


def _play_game_task(
    task: tuple[str, str, EnginePlayer, _ControllerSpec, _ControllerSpec]
) -> GameComparisonRecord:
    comparison, game_seed, subject_role, subject_spec, opponent_spec = task
    identities = {
        subject_role: _Controller(subject_spec, f"{comparison}:{game_seed}:subject"),
        other_player(subject_role): _Controller(
            opponent_spec, f"{comparison}:{game_seed}:opponent"
        ),
    }
    state = create_game(game_seed)
    rounds: list[RoundComparison] = []
    timings: list[DecisionTiming] = []
    while state.status is not EngineStatus.GAME_COMPLETE:
        while state.status is EngineStatus.PLAYING:
            actor = state.active_player
            if actor is None:
                raise SearchValidationError("playing comparison lost its actor")
            move, timing = identities[actor].select(state)
            if timing is not None:
                timings.append(timing)
            state = apply_move(state, move).state  # type: ignore[arg-type]
        result = state.pending_round_result
        if result is None:
            raise SearchValidationError("comparison round has no engine score")
        rounds.append(
            RoundComparison(
                state.round_number,
                subject_role is state.dealer,
                result.round_scores[subject_role],
                result.round_scores[other_player(subject_role)],
            )
        )
        state = advance_after_round(state)
    outcome = derive_game_outcome(state)
    return GameComparisonRecord(
        comparison,
        game_seed,
        subject_role,
        outcome.winner is subject_role,
        outcome.winner is None,
        tuple(rounds),
        tuple(timings),
    )


def _game_seeds(pair_count: int) -> tuple[str, ...]:
    return tuple(
        derive_seed(VALIDATION_GAME_NAMESPACE, f"pair-{index:03d}").hex()
        for index in range(pair_count)
    )


def _percentile(values: Sequence[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(quantile * len(ordered)) - 1)
    return ordered[max(index, 0)]


def _split_game_metrics(records: Sequence[GameComparisonRecord]) -> dict[str, object]:
    games = len(records)
    wins = sum(record.subject_won for record in records)
    ties = sum(record.tied for record in records)
    return {
        "games": games,
        "wins": wins,
        "ties": ties,
        "victory_percentage": (wins + 0.5 * ties) / games if games else None,
    }


def _bootstrap_game_advantage(
    comparison: str,
    records: Sequence[GameComparisonRecord],
    samples: int,
) -> tuple[float, float]:
    blocks: dict[str, list[float]] = {}
    for record in records:
        score = 0.5 if record.tied else float(record.subject_won)
        blocks.setdefault(record.game_seed, []).append(2.0 * score - 1.0)
    vectors = tuple(sum(values) / len(values) for _, values in sorted(blocks.items()))
    if not vectors:
        raise SearchValidationError("bootstrap requires game results")
    stream = Sha256CounterStream(
        derive_seed(
            VALIDATION_BOOTSTRAP_NAMESPACE,
            comparison,
            str(samples),
            *(f"{value:.17g}" for value in vectors),
        )
    )
    estimates = []
    for _ in range(samples):
        estimates.append(
            sum(vectors[stream.randbelow(len(vectors))] for _ in vectors)
            / len(vectors)
        )
    return (
        float(_percentile(estimates, 0.025)),
        float(_percentile(estimates, 0.975)),
    )


def _comparison_metrics(
    comparison: str,
    records: Sequence[GameComparisonRecord],
    bootstrap_samples: int,
) -> dict[str, object]:
    relevant = [record for record in records if record.comparison == comparison]
    rounds = [round_result for record in relevant for round_result in record.rounds]
    decisions = [decision for record in relevant for decision in record.decisions]
    round_scores = [
        1.0
        if result.subject_score > result.opponent_score
        else 0.5
        if result.subject_score == result.opponent_score
        else 0.0
        for result in rounds
    ]
    latencies = [decision.latency_seconds for decision in decisions]
    simulations = sum(decision.simulations for decision in decisions)
    return {
        "overall": _split_game_metrics(relevant),
        "queen": _split_game_metrics(
            [record for record in relevant if record.subject_role is EnginePlayer.QUEEN]
        ),
        "king": _split_game_metrics(
            [record for record in relevant if record.subject_role is EnginePlayer.KING]
        ),
        "rounds": {
            "count": len(rounds),
            "victory_percentage": statistics.mean(round_scores) if round_scores else None,
            "mean_score_differential": (
                statistics.mean(
                    result.subject_score - result.opponent_score for result in rounds
                )
                if rounds
                else None
            ),
            "dealer_victory_percentage": statistics.mean(
                score for score, result in zip(round_scores, rounds, strict=True) if result.subject_dealer
            ),
            "non_dealer_victory_percentage": statistics.mean(
                score for score, result in zip(round_scores, rounds, strict=True) if not result.subject_dealer
            ),
        },
        "paired_game_advantage_ci_95": list(
            _bootstrap_game_advantage(comparison, relevant, bootstrap_samples)
        ),
        "decision_latency_seconds": {
            "count": len(latencies),
            "mean": statistics.mean(latencies) if latencies else None,
            "p50": _percentile(latencies, 0.5),
            "p90": _percentile(latencies, 0.9),
            "p95": _percentile(latencies, 0.95),
            "maximum": max(latencies) if latencies else None,
        },
        "simulations_per_second": (
            simulations / sum(latencies) if latencies and sum(latencies) else None
        ),
    }


def _fixture_metrics(records: Sequence[FixtureSearchRecord]) -> dict[str, object]:
    result: dict[str, object] = {}
    for budget in sorted({record.budget for record in records}):
        rows = [record for record in records if record.budget == budget]
        fixture_rows = []
        for fixture in STRATEGIC_FIXTURES:
            if fixture.evidence is FixtureEvidence.FORCED:
                continue
            samples = [row for row in rows if row.fixture_id == fixture.fixture_id]
            fixture_rows.append(
                {
                    "fixture_id": fixture.fixture_id,
                    "passes": sum(row.passed for row in samples),
                    "samples": len(samples),
                    "pass_rate": (
                        sum(row.passed for row in samples) / len(samples)
                        if samples
                        else None
                    ),
                    "selected_actions": [row.selected_action_index for row in samples],
                }
            )
        latencies = [row.latency_seconds for row in rows if row.simulations]
        simulations = sum(row.simulations for row in rows)
        result[str(budget)] = {
            "passes": sum(row.passed for row in rows),
            "samples": len(rows),
            "pass_rate": sum(row.passed for row in rows) / len(rows),
            "fixture_pass_rate": sum(
                all(row.passed for row in rows if row.fixture_id == fixture.fixture_id)
                for fixture in STRATEGIC_FIXTURES
                if fixture.evidence is not FixtureEvidence.FORCED
            )
            / sum(
                fixture.evidence is not FixtureEvidence.FORCED
                for fixture in STRATEGIC_FIXTURES
            ),
            "stable_selected_action_rate": sum(
                len(
                    {
                        row.selected_action_index
                        for row in rows
                        if row.fixture_id == fixture.fixture_id
                    }
                )
                == 1
                for fixture in STRATEGIC_FIXTURES
                if fixture.evidence is not FixtureEvidence.FORCED
            )
            / sum(
                fixture.evidence is not FixtureEvidence.FORCED
                for fixture in STRATEGIC_FIXTURES
            ),
            "mean_latency_seconds": statistics.mean(latencies) if latencies else None,
            "decision_latency_seconds": {
                "count": len(latencies),
                "p50": _percentile(latencies, 0.50),
                "p90": _percentile(latencies, 0.90),
                "p95": _percentile(latencies, 0.95),
                "maximum": max(latencies) if latencies else None,
            },
            "p95_latency_seconds": _percentile(latencies, 0.95),
            "simulations_per_second": simulations / sum(latencies),
            "fixtures": fixture_rows,
        }
    return result


def _exact_fixture_evidence() -> list[dict[str, object]]:
    rows = []
    for fixture in STRATEGIC_FIXTURES:
        validate_fixture_semantics(fixture)
        if fixture.evidence is not FixtureEvidence.EXHAUSTIVE:
            continue
        state = replay_fixture(fixture)
        information = information_state_from_engine(state)
        sampled = sample_determinization(
            information,
            derive_seed(VALIDATION_EXACT_WORLD_NAMESPACE, fixture.fixture_id),
        )
        result = solve_perfect_information_round(
            sampled.state, diagnostic_only=True, maximum_states=100_000
        )
        values = sorted(
            (
                (float(value), index)
                for index, value in enumerate(result.action_values)
                if value is not None
            ),
            reverse=True,
        )
        margin = values[0][0] - values[1][0] if len(values) > 1 else None
        if result.selected_action_index not in fixture.expected_action_indices:
            raise SearchValidationError(
                f"exhaustive fixture drifted: {fixture.fixture_id}"
            )
        rows.append(
            {
                "fixture_id": fixture.fixture_id,
                "selected_action_index": result.selected_action_index,
                "value": result.action_values[result.selected_action_index],
                "margin": margin,
                "evaluated_states": result.evaluated_states,
                "action_values": [
                    {"action_index": index, "value": value} for value, index in values
                ],
            }
        )
    return rows


def _fixture_catalog() -> list[dict[str, object]]:
    rows = []
    for fixture in STRATEGIC_FIXTURES:
        state = replay_fixture(fixture)
        features = fixture_action_features(fixture)
        rows.append(
            {
                "fixture_id": fixture.fixture_id,
                "title": fixture.title,
                "behaviors": [behavior.value for behavior in fixture.behaviors],
                "evidence": fixture.evidence.value,
                "evidence_value": fixture.evidence_value,
                "evidence_margin": fixture.evidence_margin,
                "rationale": fixture.rationale,
                "round_number": state.round_number,
                "accepted_moves": fixture.accepted_moves,
                "stage": fixture.stage.value,
                "player": fixture.player.value,
                "dealer": fixture.dealer,
                "own_hand": list(state.hands[fixture.player]),
                "coffin": list(state.coffin),
                "expected_actions": [
                    {
                        "action_index": index,
                        "card_id": features[index].card_id,
                        "grid_index": features[index].move.global_grid_index,
                    }
                    for index in fixture.expected_action_indices
                ],
            }
        )
    return rows


def _peak_memory_task(budget: int) -> dict[str, object]:
    fixture = STRATEGIC_FIXTURES[0]
    state = replay_fixture(fixture)
    information = information_state_from_engine(state)
    config = SearchConfig(budget)
    seed = derive_search_request_seed(
        f"memory:{budget}", information, config.digest
    )
    divisor = 1024.0 * 1024.0 if sys.platform == "darwin" else 1024.0
    baseline = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / divisor
    result = InformationSetSearch(config).search(information, seed)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / divisor
    return {
        "budget": budget,
        "simulations": result.simulation_count,
        "baseline_rss_mib": baseline,
        "peak_rss_mib": peak,
        "incremental_peak_mib": max(0.0, peak - baseline),
    }


def _atomic_write(path: Path, data: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _fixture_cache_path(output_directory: Path) -> Path:
    return output_directory / "fixture-records.json"


def _game_cache_path(output_directory: Path) -> Path:
    return output_directory / "game-records.json"


def _write_fixture_cache(
    output_directory: Path,
    config: ValidationConfig,
    records: Sequence[FixtureSearchRecord],
) -> None:
    payload = {
        "format_version": SEARCH_VALIDATION_FORMAT_VERSION,
        "fixture_schema_version": STRATEGIC_FIXTURE_SCHEMA_VERSION,
        "budgets": list(config.budgets),
        "sensitivity_repetitions": config.sensitivity_repetitions,
        "records": [asdict(record) for record in records],
    }
    _atomic_write(
        _fixture_cache_path(output_directory),
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
    )


def _read_fixture_cache(
    output_directory: Path, config: ValidationConfig
) -> list[FixtureSearchRecord] | None:
    path = _fixture_cache_path(output_directory)
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("format_version") != SEARCH_VALIDATION_FORMAT_VERSION
        or payload.get("fixture_schema_version") != STRATEGIC_FIXTURE_SCHEMA_VERSION
        or payload.get("budgets") != list(config.budgets)
        or payload.get("sensitivity_repetitions") != config.sensitivity_repetitions
    ):
        return None
    records = []
    for row in payload.get("records", []):
        records.append(
            FixtureSearchRecord(
                fixture_id=str(row["fixture_id"]),
                budget=int(row["budget"]),
                repetition=int(row["repetition"]),
                selected_action_index=int(row["selected_action_index"]),
                passed=bool(row["passed"]),
                simulations=int(row["simulations"]),
                latency_seconds=float(row["latency_seconds"]),
                information_sets=int(row["information_sets"]),
                action_statistics=tuple(
                    (int(item[0]), int(item[1]), None if item[2] is None else float(item[2]))
                    for item in row["action_statistics"]
                ),
            )
        )
    expected_count = (
        sum(
            fixture.evidence is not FixtureEvidence.FORCED
            for fixture in STRATEGIC_FIXTURES
        )
        * len(config.budgets)
        * config.sensitivity_repetitions
    )
    return records if len(records) == expected_count else None


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _game_record_key(record: GameComparisonRecord) -> tuple[str, str, EnginePlayer]:
    return record.comparison, record.game_seed, record.subject_role


def _write_game_cache(
    output_directory: Path,
    config: ValidationConfig,
    archive_sha256: str,
    records: Sequence[GameComparisonRecord],
) -> None:
    payload = {
        "format_version": SEARCH_VALIDATION_FORMAT_VERSION,
        "selected_budget": config.selected_budget,
        "lower_budget": config.budgets[0],
        "policy_archive_sha256": archive_sha256,
        "records": [
            asdict(record)
            for record in sorted(
                records,
                key=lambda item: (
                    item.comparison,
                    item.game_seed,
                    item.subject_role.value,
                ),
            )
        ],
    }
    _atomic_write(
        _game_cache_path(output_directory),
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
    )


def _read_game_cache(
    output_directory: Path,
    config: ValidationConfig,
    archive_sha256: str,
) -> list[GameComparisonRecord]:
    path = _game_cache_path(output_directory)
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            payload.get("format_version") != SEARCH_VALIDATION_FORMAT_VERSION
            or payload.get("selected_budget") != config.selected_budget
            or payload.get("lower_budget") != config.budgets[0]
            or payload.get("policy_archive_sha256") != archive_sha256
        ):
            return []
        rows = payload.get("records", [])
    else:
        # A completed pre-cache report is itself sealed evidence and can seed resume.
        report_path = output_directory / "search-validation.json"
        if not report_path.is_file():
            return []
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report_config = report.get("configuration", {})
        if (
            report.get("format_version") != SEARCH_VALIDATION_FORMAT_VERSION
            or report_config.get("selected_budget") != config.selected_budget
            or report_config.get("budgets", [None])[0] != config.budgets[0]
            or report.get("policy_control", {}).get("sha256") != archive_sha256
        ):
            return []
        rows = report.get("raw_games", [])
    records = []
    for row in rows:
        records.append(
            GameComparisonRecord(
                comparison=str(row["comparison"]),
                game_seed=str(row["game_seed"]),
                subject_role=EnginePlayer(row["subject_role"]),
                subject_won=bool(row["subject_won"]),
                tied=bool(row["tied"]),
                rounds=tuple(
                    RoundComparison(
                        round_number=int(item["round_number"]),
                        subject_dealer=bool(item["subject_dealer"]),
                        subject_score=int(item["subject_score"]),
                        opponent_score=int(item["opponent_score"]),
                    )
                    for item in row["rounds"]
                ),
                decisions=tuple(
                    DecisionTiming(
                        controller=str(item["controller"]),
                        budget=(
                            None if item["budget"] is None else int(item["budget"])
                        ),
                        round_number=int(item["round_number"]),
                        accepted_moves=int(item["accepted_moves"]),
                        simulations=int(item["simulations"]),
                        latency_seconds=float(item["latency_seconds"]),
                    )
                    for item in row["decisions"]
                ),
            )
        )
    return records


def run_validation(
    *,
    output_directory: Path,
    archive_path: Path = DEFAULT_POLICY_ARCHIVE,
    config: ValidationConfig = ValidationConfig(),
    progress: bool = True,
) -> dict[str, object]:
    output_directory = output_directory.expanduser().resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    archive_path = archive_path.expanduser().resolve()
    if not archive_path.is_file():
        raise SearchValidationError(f"PPO archive is unavailable: {archive_path}")
    LocalTorchPolicyAdapter(archive_path)
    archive_sha256 = _sha256(archive_path)
    for fixture in STRATEGIC_FIXTURES:
        validate_fixture_semantics(fixture)

    fixture_tasks = [
        (fixture.fixture_id, budget, repetition)
        for fixture in STRATEGIC_FIXTURES
        if fixture.evidence is not FixtureEvidence.FORCED
        for budget in config.budgets
        for repetition in range(config.sensitivity_repetitions)
    ]
    fixture_records = _read_fixture_cache(output_directory, config)
    if fixture_records is None:
        fixture_records = []
        with ProcessPoolExecutor(max_workers=config.workers) as pool:
            futures = [pool.submit(_run_fixture_search, task) for task in fixture_tasks]
            for completed, future in enumerate(as_completed(futures), start=1):
                fixture_records.append(future.result())
                if progress:
                    print(f"fixture {completed}/{len(futures)}", flush=True)
        _write_fixture_cache(output_directory, config, fixture_records)
    elif progress:
        print(f"fixture cache reused ({len(fixture_records)} decisions)", flush=True)

    subject = _ControllerSpec("search", config.selected_budget)
    controls = (
        ("random-legal", _ControllerSpec("random")),
        ("policy-2-v20", _ControllerSpec("ppo", archive_path=str(archive_path))),
        (
            f"search-{config.budgets[0]}",
            _ControllerSpec("search", config.budgets[0]),
        ),
    )
    game_tasks = []
    for control_name, control in controls:
        comparison = f"search-{config.selected_budget}-vs-{control_name}"
        for game_seed in _game_seeds(config.game_pairs):
            for role in EnginePlayer:
                game_tasks.append((comparison, game_seed, role, subject, control))
    requested_game_keys = {
        (comparison, game_seed, subject_role)
        for comparison, game_seed, subject_role, _, _ in game_tasks
    }
    game_records = [
        record
        for record in _read_game_cache(output_directory, config, archive_sha256)
        if _game_record_key(record) in requested_game_keys
    ]
    completed_keys = {_game_record_key(record) for record in game_records}
    missing_game_tasks = [
        task
        for task in game_tasks
        if (task[0], task[1], task[2]) not in completed_keys
    ]
    if progress and game_records:
        print(
            f"game cache reused ({len(game_records)}/{len(game_tasks)} games)",
            flush=True,
        )
    with ProcessPoolExecutor(max_workers=config.workers) as pool:
        futures = [pool.submit(_play_game_task, task) for task in missing_game_tasks]
        for completed, future in enumerate(as_completed(futures), start=1):
            game_records.append(future.result())
            _write_game_cache(
                output_directory, config, archive_sha256, game_records
            )
            if progress:
                print(
                    f"game {len(completed_keys) + completed}/{len(game_tasks)}",
                    flush=True,
                )

    with ProcessPoolExecutor(max_workers=min(config.workers, len(config.budgets))) as pool:
        memory = list(pool.map(_peak_memory_task, config.budgets))
    exact = _exact_fixture_evidence()
    comparisons = {
        comparison: _comparison_metrics(
            comparison, game_records, config.bootstrap_samples
        )
        for comparison in sorted({record.comparison for record in game_records})
    }
    fixture_metrics = _fixture_metrics(fixture_records)
    exact_fixture_ids = {
        fixture.fixture_id
        for fixture in STRATEGIC_FIXTURES
        if fixture.evidence is FixtureEvidence.EXHAUSTIVE
    }
    upper_bound_agreement = {
        str(budget): {
            "decisions": len(rows),
            "agreement_rate": sum(row.passed for row in rows) / len(rows),
        }
        for budget in config.budgets
        for rows in (
            [
                record
                for record in fixture_records
                if record.budget == budget
                and record.fixture_id in exact_fixture_ids
            ],
        )
    }
    selected_fixtures = fixture_metrics[str(config.selected_budget)]
    control_passes = all(
        float(metrics["paired_game_advantage_ci_95"][0]) > 0  # type: ignore[index]
        for name, metrics in comparisons.items()
        if not name.endswith(f"search-{config.budgets[0]}")
    )
    gate_passed = (
        float(selected_fixtures["fixture_pass_rate"]) == 1.0  # type: ignore[index]
        and control_passes
        and config.game_pairs >= MINIMUM_CONTROL_GAME_PAIRS
    )
    document: dict[str, object] = {
        "format_version": SEARCH_VALIDATION_FORMAT_VERSION,
        "fixture_schema_version": STRATEGIC_FIXTURE_SCHEMA_VERSION,
        "configuration": asdict(config),
        "sample_design": {
            "independent_deck_pairs_per_control": config.game_pairs,
            "role_balanced_games_per_control": 2 * config.game_pairs,
            "minimum_gate_pairs": MINIMUM_CONTROL_GAME_PAIRS,
            "pairing_unit": "deck seed with Queen and King assignments reversed",
            "stopping_rule": "fixed sample; no interim stopping",
        },
        "policy_control": {
            "path": str(archive_path),
            "sha256": archive_sha256,
            "profile": "argmax-v1",
        },
        "gate_passed": gate_passed,
        "fixture_catalog": _fixture_catalog(),
        "strategic_fixtures": fixture_metrics,
        "exact_upper_bound": exact,
        "perfect_information_agreement": upper_bound_agreement,
        "comparisons": comparisons,
        "memory": sorted(memory, key=lambda row: int(row["budget"])),
        "failed_fixture_evidence": [
            asdict(record) for record in fixture_records if not record.passed
        ],
        "raw_games": [asdict(record) for record in game_records],
    }
    _atomic_write(
        output_directory / "search-validation.json",
        json.dumps(document, indent=2, sort_keys=True, default=str) + "\n",
    )
    _atomic_write(
        output_directory / "search-validation.md", _markdown_report(document)
    )
    return document


def _pct(value: object) -> str:
    return "n/a" if value is None else f"{100 * float(value):.1f}%"


def _markdown_report(document: dict[str, object]) -> str:
    config = document["configuration"]
    fixtures = document["strategic_fixtures"]
    comparisons = document["comparisons"]
    lines = [
        "# Search validation",
        "",
        f"Gate: **{'passed' if document['gate_passed'] else 'failed'}**",
        "",
        (
            f"Controls use {document['sample_design']['independent_deck_pairs_per_control']} "  # type: ignore[index]
            "independent decks with both role assignments "
            f"({document['sample_design']['role_balanced_games_per_control']} games per control)."  # type: ignore[index]
        ),
        "",
        "## Fixture catalog",
        "",
        "| Fixture | Actor | Stage | Behaviors | Expected placement | Evidence |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for fixture in document["fixture_catalog"]:  # type: ignore[union-attr]
        placements = ", ".join(
            f"{item['card_id']} to {item['grid_index']}"
            for item in fixture["expected_actions"]
        )
        lines.append(
            f"| {fixture['fixture_id']} | {fixture['player']} "
            f"({'dealer' if fixture['dealer'] else 'non-dealer'}) | "
            f"{fixture['stage']} | {', '.join(fixture['behaviors']) or 'forced'} | "
            f"{placements} | {fixture['evidence']} |"
        )
    lines.extend(
        (
        "",
        "## Strategic fixtures",
        "",
        "| Budget | Decision pass | Fixture pass | p50 latency | p95 latency | Maximum | Simulations/sec |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        )
    )
    for budget in config["budgets"]:  # type: ignore[index]
        row = fixtures[str(budget)]  # type: ignore[index]
        timing = row["decision_latency_seconds"]
        lines.append(
            f"| {budget} | {_pct(row['pass_rate'])} | {_pct(row['fixture_pass_rate'])} "
            f"| {float(timing['p50']):.3f}s | {float(timing['p95']):.3f}s "
            f"| {float(timing['maximum']):.3f}s "
            f"| {float(row['simulations_per_second']):.1f} |"
        )
    lines.extend(
        (
            "",
            "## Hidden-sample sensitivity",
            "",
            "The request seed changes the determinization and rollout streams. The "
            "decision pass rate above measures whether each sampled run retained an "
            "evidence-backed action. Exact selected-action stability was:",
            "",
            "| Budget | Fixtures with one selected action across seeds |",
            "| ---: | ---: |",
        )
    )
    for budget in config["budgets"]:  # type: ignore[index]
        row = fixtures[str(budget)]  # type: ignore[index]
        lines.append(f"| {budget} | {_pct(row['stable_selected_action_rate'])} |")
    lines.extend(
        (
            "",
            "## Exhaustive upper-bound evidence",
            "",
            "| Fixture | Action | Value | Margin | States |",
            "| --- | ---: | ---: | ---: | ---: |",
        )
    )
    for row in document["exact_upper_bound"]:  # type: ignore[union-attr]
        lines.append(
            f"| {row['fixture_id']} | {row['selected_action_index']} "
            f"| {float(row['value']):.5f} | {float(row['margin']):.5f} "
            f"| {row['evaluated_states']} |"
        )
    lines.extend(
        (
            "",
            "| Budget | Agreement with exact action |",
            "| ---: | ---: |",
        )
    )
    for budget in config["budgets"]:  # type: ignore[index]
        agreement = document["perfect_information_agreement"][str(budget)]  # type: ignore[index]
        lines.append(
            f"| {budget} | {_pct(agreement['agreement_rate'])} "
            f"({agreement['decisions']} decisions) |"
        )
    lines.extend(
        (
            "",
            "## Absolute controls",
            "",
            "| Comparison | Games | Game win | Round win | Mean round differential | Paired advantage 95% CI |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        )
    )
    for name, row in comparisons.items():  # type: ignore[union-attr]
        overall = row["overall"]
        rounds = row["rounds"]
        interval = row["paired_game_advantage_ci_95"]
        lines.append(
            f"| {name} | {overall['games']} | {_pct(overall['victory_percentage'])} "
            f"| {_pct(rounds['victory_percentage'])} "
            f"| {float(rounds['mean_score_differential']):.2f} "
            f"| [{float(interval[0]):.3f}, {float(interval[1]):.3f}] |"
        )
    lines.extend(
        (
            "",
            "| Comparison | Queen game win | King game win | Dealer round win | Non-dealer round win |",
            "| --- | ---: | ---: | ---: | ---: |",
        )
    )
    for name, row in comparisons.items():  # type: ignore[union-attr]
        lines.append(
            f"| {name} | {_pct(row['queen']['victory_percentage'])} "
            f"| {_pct(row['king']['victory_percentage'])} "
            f"| {_pct(row['rounds']['dealer_victory_percentage'])} "
            f"| {_pct(row['rounds']['non_dealer_victory_percentage'])} |"
        )
    failed = document["failed_fixture_evidence"]
    lines.extend(("", "## Failed fixture decisions", ""))
    if not failed:
        lines.append("None.")
    else:
        lines.extend(
            (
                "| Fixture | Budget | Seed repetition | Selected | Action visits and values |",
                "| --- | ---: | ---: | ---: | --- |",
            )
        )
        for row in failed:  # type: ignore[assignment]
            distribution = ", ".join(
                f"{action}:{visits}/{float(value):.3f}"
                for action, visits, value in row["action_statistics"]
                if value is not None
            )
            lines.append(
                f"| {row['fixture_id']} | {row['budget']} | {row['repetition']} "
                f"| {row['selected_action_index']} | {distribution} |"
            )
    lines.extend(("", "## Peak memory", ""))
    lines.append("| Budget | Process peak | Incremental peak |")
    lines.append("| ---: | ---: | ---: |")
    for row in document["memory"]:  # type: ignore[union-attr]
        lines.append(
            f"| {row['budget']} | {float(row['peak_rss_mib']):.1f} MiB "
            f"| {float(row['incremental_peak_mib']):.1f} MiB |"
        )
    lines.extend(
        (
            "",
            "## Scope",
            "",
            "The exact diagnostic covers reduced round-six positions where the unseen "
            "pool fixes the opponent hand. It is not available to live search. Full-game "
            "comparisons use only player information. This report establishes the search "
            "gate; it does not validate a neural teacher or deployment profile.",
            "",
        )
    )
    return "\n".join(lines)


def main(arguments: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate the search-only opponent")
    parser.add_argument(
        "--output-directory", type=Path, default=Path("runs/search-validation-001")
    )
    parser.add_argument("--policy-archive", type=Path, default=DEFAULT_POLICY_ARCHIVE)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--sensitivity-repetitions", type=int, default=3)
    parser.add_argument(
        "--game-pairs", type=int, default=MINIMUM_CONTROL_GAME_PAIRS
    )
    parsed = parser.parse_args(arguments)
    document = run_validation(
        output_directory=parsed.output_directory,
        archive_path=parsed.policy_archive,
        config=ValidationConfig(
            workers=parsed.workers,
            sensitivity_repetitions=parsed.sensitivity_repetitions,
            game_pairs=parsed.game_pairs,
        ),
    )
    print(f"gate={'passed' if document['gate_passed'] else 'failed'}")
    return 0 if document["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

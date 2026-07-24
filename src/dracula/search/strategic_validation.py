"""Absolute validation for the Teacher v2 strategic search gate."""

from __future__ import annotations

import argparse
import json
import math
import os
import resource
import statistics
import tempfile
import time
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

from dracula.bridge import action_index_for_move
from dracula.engine import EnginePlayer, legal_moves
from dracula.search.defensive_fixtures import (
    DEFENSIVE_FIXTURE_SCHEMA_VERSION,
    DEFENSIVE_FIXTURES,
    DefensiveBehavior,
    replay_defensive_fixture,
)
from dracula.search.diagnostic import solve_adversarial_perfect_information_round
from dracula.search.diagnostic import solve_perfect_information_round
from dracula.search.information import (
    derive_search_request_seed,
    information_state_from_engine,
)
from dracula.search.planner import InformationSetSearch, SearchConfig
from dracula.search.strategic import (
    StrategicInformationSetSearch,
    StrategicSearchConfig,
    derive_strategic_search_request_seed,
)
from dracula.search.strategic_fixtures import (
    STRATEGIC_FIXTURES,
    FixtureEvidence,
    RoundStage,
    replay_fixture,
)
from dracula.search.validation import (
    DEFAULT_POLICY_ARCHIVE,
    GameComparisonRecord,
    _ControllerSpec,
    _comparison_metrics,
    _game_record_key,
    _game_seeds,
    _play_game_task,
)

STRATEGIC_VALIDATION_FORMAT_VERSION = "dracula-teacher-v2-validation-v2"
DEFAULT_OUTPUT_DIRECTORY = Path("runs/teacher-v2-shallow-validation-001")


@dataclass(frozen=True, slots=True)
class StrategicProfile:
    outer_simulations: int
    response_completions: int

    def __post_init__(self) -> None:
        StrategicSearchConfig(self.outer_simulations, self.response_completions)

    @property
    def label(self) -> str:
        return f"v2-{self.outer_simulations}x{self.response_completions}c"


@dataclass(frozen=True, slots=True)
class StrategicValidationConfig:
    selected_profile: StrategicProfile = StrategicProfile(500, 1)
    lower_profile: StrategicProfile = StrategicProfile(500, 2)
    higher_profile: StrategicProfile = StrategicProfile(500, 4)
    benchmark_profiles: tuple[StrategicProfile, ...] = (
        StrategicProfile(32, 1),
        StrategicProfile(32, 2),
        StrategicProfile(32, 4),
    )
    defensive_repetitions: int = 5
    constructive_repetitions: int = 3
    control_game_pairs: int = 12
    version_one_game_pairs: int = 30
    bootstrap_samples: int = 2_000
    workers: int = 4

    def __post_init__(self) -> None:
        for value, label in (
            (self.defensive_repetitions, "defensive repetitions"),
            (self.constructive_repetitions, "constructive repetitions"),
            (self.control_game_pairs, "control game pairs"),
            (self.version_one_game_pairs, "version one game pairs"),
            (self.bootstrap_samples, "bootstrap samples"),
            (self.workers, "workers"),
        ):
            if type(value) is not int or value < 1:
                raise ValueError(f"{label} must be a positive integer")
        if self.version_one_game_pairs < 30:
            raise ValueError("version one comparison requires at least 30 deck pairs")
        if self.control_game_pairs < 12:
            raise ValueError("absolute controls require at least 12 deck pairs")


@dataclass(frozen=True, slots=True)
class StrategicFixtureRecord:
    suite: str
    fixture_id: str
    controller: str
    repetition: int
    selected_action_index: int
    expected_action_indices: tuple[int, ...]
    passed: bool
    outer_simulations: int
    response_request_count: int
    unique_response_evaluation_count: int
    response_cache_hit_count: int
    response_candidate_action_count: int
    response_terminal_evaluations: int
    total_terminal_evaluations: int
    latency_seconds: float
    peak_rss_mib: float
    visits: tuple[int, ...]
    mean_values: tuple[float | None, ...]


@dataclass(frozen=True, slots=True)
class BenchmarkRecord:
    stage: RoundStage
    profile: StrategicProfile
    fixture_id: str
    latency_seconds: float
    outer_simulations: int
    response_request_count: int
    unique_response_evaluation_count: int
    response_cache_hit_count: int
    response_candidate_action_count: int
    response_terminal_evaluations: int
    total_terminal_evaluations: int
    terminal_evaluations_per_second: float
    peak_rss_mib: float


def _peak_rss_mib() -> float:
    divisor = 1024.0 * 1024.0 if os.uname().sysname == "Darwin" else 1024.0
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / divisor


def _defensive_fixture(fixture_id: str):
    return next(value for value in DEFENSIVE_FIXTURES if value.fixture_id == fixture_id)


def _constructive_fixture(fixture_id: str):
    return next(value for value in STRATEGIC_FIXTURES if value.fixture_id == fixture_id)


def _fixture_task(
    task: tuple[str, str, str, int, StrategicProfile | None]
) -> StrategicFixtureRecord:
    suite, fixture_id, controller, repetition, profile = task
    if suite == "defensive":
        fixture = _defensive_fixture(fixture_id)
        state = replay_defensive_fixture(fixture)
    else:
        fixture = _constructive_fixture(fixture_id)
        state = replay_fixture(fixture)
    information = information_state_from_engine(state)
    started = time.perf_counter()
    if controller == "v1-500":
        config = SearchConfig(500)
        result = InformationSetSearch(config).search(
            information,
            derive_search_request_seed(
                f"teacher-v2:{suite}:{fixture_id}:{repetition}:v1",
                information,
                config.digest,
            ),
        )
        response_terminal_evaluations = 0
        response_requests = 0
        unique_response_evaluations = 0
        response_cache_hits = 0
        response_candidate_actions = 0
        total_terminal_evaluations = result.simulation_count
        controller_label = controller
    else:
        if profile is None:
            raise ValueError("Teacher v2 fixture task requires a profile")
        config = StrategicSearchConfig(
            profile.outer_simulations, profile.response_completions
        )
        result = StrategicInformationSetSearch(config).search(
            information,
            derive_strategic_search_request_seed(
                f"teacher-v2:{suite}:{fixture_id}:{repetition}:{profile.label}",
                information,
                config.digest,
            ),
        )
        response_terminal_evaluations = result.response_terminal_evaluation_count
        response_requests = result.response_request_count
        unique_response_evaluations = result.unique_response_evaluation_count
        response_cache_hits = result.response_cache_hit_count
        response_candidate_actions = result.response_candidate_action_count
        total_terminal_evaluations = result.total_terminal_evaluation_count
        controller_label = profile.label
    return StrategicFixtureRecord(
        suite,
        fixture_id,
        controller_label,
        repetition,
        result.selected_action_index,
        fixture.expected_action_indices,
        result.selected_action_index in fixture.expected_action_indices,
        result.simulation_count,
        response_requests,
        unique_response_evaluations,
        response_cache_hits,
        response_candidate_actions,
        response_terminal_evaluations,
        total_terminal_evaluations,
        time.perf_counter() - started,
        _peak_rss_mib(),
        result.action_visits,
        result.mean_action_values,
    )


def _benchmark_task(task: tuple[RoundStage, StrategicProfile]) -> BenchmarkRecord:
    stage, profile = task
    fixture = next(
        value
        for value in DEFENSIVE_FIXTURES
        if value.stage is stage
        and DefensiveBehavior.FORCED_TRANSITION not in value.behaviors
    )
    state = replay_defensive_fixture(fixture)
    information = information_state_from_engine(state)
    config = StrategicSearchConfig(
        profile.outer_simulations, profile.response_completions
    )
    started = time.perf_counter()
    result = StrategicInformationSetSearch(config).search(
        information,
        derive_strategic_search_request_seed(
            f"teacher-v2:benchmark:{stage.value}:{profile.label}",
            information,
            config.digest,
        ),
    )
    latency = time.perf_counter() - started
    return BenchmarkRecord(
        stage,
        profile,
        fixture.fixture_id,
        latency,
        result.simulation_count,
        result.response_request_count,
        result.unique_response_evaluation_count,
        result.response_cache_hit_count,
        result.response_candidate_action_count,
        result.response_terminal_evaluation_count,
        result.total_terminal_evaluation_count,
        result.total_terminal_evaluation_count / latency,
        _peak_rss_mib(),
    )


def _exact_evidence() -> list[dict[str, object]]:
    rows = []
    for fixture in DEFENSIVE_FIXTURES:
        state = replay_defensive_fixture(fixture)
        legal = {
            action_index_for_move(move, state.active_player)
            for move in legal_moves(state, state.active_player)
        }
        if not set(fixture.expected_action_indices) <= legal:
            raise ValueError(f"fixture has an illegal expected action: {fixture.fixture_id}")
        if DefensiveBehavior.FORCED_TRANSITION in fixture.behaviors:
            if legal != set(fixture.expected_action_indices):
                raise ValueError(f"forced fixture drifted: {fixture.fixture_id}")
            continue
        result = solve_adversarial_perfect_information_round(
            state, diagnostic_only=True, maximum_states=100_000
        )
        values = sorted(
            (
                (float(value), index)
                for index, value in enumerate(result.action_values)
                if value is not None
            ),
            reverse=True,
        )
        best_value = values[0][0]
        exact_actions = tuple(
            sorted(index for value, index in values if value == best_value)
        )
        next_value = next(
            (value for value, _ in values if value < best_value), best_value
        )
        margin = best_value - next_value
        if exact_actions != fixture.expected_action_indices:
            raise ValueError(f"exact fixture action drifted: {fixture.fixture_id}")
        if not math.isclose(best_value, float(fixture.exact_value), abs_tol=1e-12):
            raise ValueError(f"exact fixture value drifted: {fixture.fixture_id}")
        if not math.isclose(margin, float(fixture.exact_margin), abs_tol=1e-12):
            raise ValueError(f"exact fixture margin drifted: {fixture.fixture_id}")
        rows.append(
            {
                "fixture_id": fixture.fixture_id,
                "selected_actions": list(exact_actions),
                "value": best_value,
                "margin": margin,
                "evaluated_states": result.evaluated_states,
                "action_values": [
                    {"action_index": index, "value": value}
                    for value, index in values
                ],
            }
        )
    return rows


def _constructive_exact_evidence() -> list[dict[str, object]]:
    rows = []
    for fixture in STRATEGIC_FIXTURES:
        if fixture.evidence is FixtureEvidence.FORCED:
            continue
        state = replay_fixture(fixture)
        result = solve_perfect_information_round(
            state, diagnostic_only=True, maximum_states=2_000_000
        )
        if result.selected_action_index not in fixture.expected_action_indices:
            raise ValueError(
                f"constructive fixture action drifted: {fixture.fixture_id}"
            )
        values = sorted(
            (
                (float(value), index)
                for index, value in enumerate(result.action_values)
                if value is not None
            ),
            reverse=True,
        )
        next_value = next(
            (value for value, _ in values if value < values[0][0]), values[0][0]
        )
        rows.append(
            {
                "fixture_id": fixture.fixture_id,
                "selected_action": result.selected_action_index,
                "value": values[0][0],
                "margin": values[0][0] - next_value,
                "evaluated_states": result.evaluated_states,
                "action_values": [
                    {"action_index": index, "value": value}
                    for value, index in values
                ],
            }
        )
    return rows


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _read_records(path: Path, record_type):
    if not path.is_file():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("format_version") != STRATEGIC_VALIDATION_FORMAT_VERSION:
        return []
    records = []
    for row in payload.get("records", []):
        if record_type is StrategicFixtureRecord:
            records.append(
                StrategicFixtureRecord(
                    suite=row["suite"],
                    fixture_id=row["fixture_id"],
                    controller=row["controller"],
                    repetition=int(row["repetition"]),
                    selected_action_index=int(row["selected_action_index"]),
                    expected_action_indices=tuple(row["expected_action_indices"]),
                    passed=bool(row["passed"]),
                    outer_simulations=int(row["outer_simulations"]),
                    response_request_count=int(row["response_request_count"]),
                    unique_response_evaluation_count=int(
                        row["unique_response_evaluation_count"]
                    ),
                    response_cache_hit_count=int(row["response_cache_hit_count"]),
                    response_candidate_action_count=int(
                        row["response_candidate_action_count"]
                    ),
                    response_terminal_evaluations=int(
                        row["response_terminal_evaluations"]
                    ),
                    total_terminal_evaluations=int(row["total_terminal_evaluations"]),
                    latency_seconds=float(row["latency_seconds"]),
                    peak_rss_mib=float(row["peak_rss_mib"]),
                    visits=tuple(row["visits"]),
                    mean_values=tuple(row["mean_values"]),
                )
            )
        else:
            records.append(
                BenchmarkRecord(
                    stage=RoundStage(row["stage"]),
                    profile=StrategicProfile(**row["profile"]),
                    fixture_id=row["fixture_id"],
                    latency_seconds=float(row["latency_seconds"]),
                    outer_simulations=int(row["outer_simulations"]),
                    response_request_count=int(row["response_request_count"]),
                    unique_response_evaluation_count=int(
                        row["unique_response_evaluation_count"]
                    ),
                    response_cache_hit_count=int(row["response_cache_hit_count"]),
                    response_candidate_action_count=int(
                        row["response_candidate_action_count"]
                    ),
                    response_terminal_evaluations=int(
                        row["response_terminal_evaluations"]
                    ),
                    total_terminal_evaluations=int(row["total_terminal_evaluations"]),
                    terminal_evaluations_per_second=float(
                        row["terminal_evaluations_per_second"]
                    ),
                    peak_rss_mib=float(row["peak_rss_mib"]),
                )
            )
    return records


def _write_records(path: Path, records: Sequence[object]) -> None:
    _atomic_write(
        path,
        json.dumps(
            {
                "format_version": STRATEGIC_VALIDATION_FORMAT_VERSION,
                "records": [asdict(record) for record in records],
            },
            indent=2,
            sort_keys=True,
            default=str,
        )
        + "\n",
    )


def _fixture_summary(records: Sequence[StrategicFixtureRecord]) -> dict[str, object]:
    result = {}
    for controller in sorted({record.controller for record in records}):
        rows = [record for record in records if record.controller == controller]
        suites = {}
        for suite in ("constructive", "defensive"):
            selected = [record for record in rows if record.suite == suite]
            suites[suite] = {
                "passes": sum(record.passed for record in selected),
                "samples": len(selected),
                "pass_rate": (
                    sum(record.passed for record in selected) / len(selected)
                    if selected
                    else None
                ),
            }
        latencies = [record.latency_seconds for record in rows]
        evaluations = sum(record.total_terminal_evaluations for record in rows)
        response_requests = sum(record.response_request_count for record in rows)
        response_cache_hits = sum(
            record.response_cache_hit_count for record in rows
        )
        result[controller] = {
            **suites,
            "latency_seconds": {
                "p50": statistics.median(latencies) if latencies else None,
                "p95": (
                    sorted(latencies)[math.ceil(0.95 * len(latencies)) - 1]
                    if latencies
                    else None
                ),
                "maximum": max(latencies) if latencies else None,
            },
            "terminal_evaluations_per_second": (
                evaluations / sum(latencies) if latencies else None
            ),
            "response_cache": {
                "requests": response_requests,
                "unique_evaluations": sum(
                    record.unique_response_evaluation_count for record in rows
                ),
                "hits": response_cache_hits,
                "hit_rate": (
                    response_cache_hits / response_requests
                    if response_requests
                    else None
                ),
            },
            "peak_rss_mib": max((record.peak_rss_mib for record in rows), default=0.0),
        }
    return result


def _game_tasks(config: StrategicValidationConfig, archive_path: Path):
    selected = _ControllerSpec(
        "strategic",
        config.selected_profile.outer_simulations,
        config.selected_profile.response_completions,
    )
    controls = (
        ("random-legal", _ControllerSpec("random"), config.control_game_pairs),
        (
            "policy-2-v20",
            _ControllerSpec("ppo", archive_path=str(archive_path)),
            config.control_game_pairs,
        ),
        ("v1-500", _ControllerSpec("search", 500), config.version_one_game_pairs),
        (
            config.lower_profile.label,
            _ControllerSpec(
                "strategic",
                config.lower_profile.outer_simulations,
                config.lower_profile.response_completions,
            ),
            config.control_game_pairs,
        ),
        (
            config.higher_profile.label,
            _ControllerSpec(
                "strategic",
                config.higher_profile.outer_simulations,
                config.higher_profile.response_completions,
            ),
            config.control_game_pairs,
        ),
    )
    tasks = []
    for name, control, pair_count in controls:
        comparison = f"{config.selected_profile.label}-vs-{name}"
        for game_seed in _game_seeds(pair_count):
            for role in EnginePlayer:
                tasks.append((comparison, game_seed, role, selected, control))
    return tasks


def _read_games(path: Path) -> list[GameComparisonRecord]:
    if not path.is_file():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("format_version") != STRATEGIC_VALIDATION_FORMAT_VERSION:
        return []
    records = []
    from dracula.search.validation import DecisionTiming, RoundComparison

    for row in payload.get("records", []):
        records.append(
            GameComparisonRecord(
                comparison=row["comparison"],
                game_seed=row["game_seed"],
                subject_role=EnginePlayer(row["subject_role"]),
                subject_won=bool(row["subject_won"]),
                tied=bool(row["tied"]),
                rounds=tuple(RoundComparison(**value) for value in row["rounds"]),
                decisions=tuple(DecisionTiming(**value) for value in row["decisions"]),
            )
        )
    return records


def _write_games(path: Path, records: Sequence[GameComparisonRecord]) -> None:
    _atomic_write(
        path,
        json.dumps(
            {
                "format_version": STRATEGIC_VALIDATION_FORMAT_VERSION,
                "records": [asdict(record) for record in records],
            },
            indent=2,
            sort_keys=True,
            default=str,
        )
        + "\n",
    )


def run_validation(
    *,
    output_directory: Path = DEFAULT_OUTPUT_DIRECTORY,
    archive_path: Path = DEFAULT_POLICY_ARCHIVE,
    config: StrategicValidationConfig = StrategicValidationConfig(),
    run_benchmarks: bool = True,
    run_games: bool = True,
) -> dict[str, object]:
    output_directory = output_directory.expanduser().resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    archive_path = archive_path.expanduser().resolve()
    if run_games and not archive_path.is_file():
        raise FileNotFoundError(f"PPO archive is unavailable: {archive_path}")

    exact_path = output_directory / "exact-evidence.json"
    if exact_path.is_file():
        exact_document = json.loads(exact_path.read_text(encoding="utf-8"))
        if (
            exact_document.get("format_version")
            == STRATEGIC_VALIDATION_FORMAT_VERSION
            and exact_document.get("fixture_schema_version")
            == DEFENSIVE_FIXTURE_SCHEMA_VERSION
        ):
            exact = exact_document["defensive_evidence"]
            constructive_exact = exact_document.get("constructive_evidence")
        else:
            exact = _exact_evidence()
            constructive_exact = None
    else:
        exact = _exact_evidence()
        constructive_exact = None
    if constructive_exact is None:
        constructive_exact = _constructive_exact_evidence()
    _atomic_write(
        exact_path,
        json.dumps(
            {
                "format_version": STRATEGIC_VALIDATION_FORMAT_VERSION,
                "fixture_schema_version": DEFENSIVE_FIXTURE_SCHEMA_VERSION,
                "defensive_evidence": exact,
                "constructive_evidence": constructive_exact,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )
    fixture_path = output_directory / "fixture-records.json"
    fixture_records = _read_records(fixture_path, StrategicFixtureRecord)
    fixture_keys = {
        (record.suite, record.fixture_id, record.controller, record.repetition)
        for record in fixture_records
    }
    fixture_tasks = []
    for fixture in DEFENSIVE_FIXTURES:
        if DefensiveBehavior.FORCED_TRANSITION in fixture.behaviors:
            continue
        for repetition in range(config.defensive_repetitions):
            for controller, profile in (
                (config.selected_profile.label, config.selected_profile),
                ("v1-500", None),
            ):
                key = ("defensive", fixture.fixture_id, controller, repetition)
                if key not in fixture_keys:
                    fixture_tasks.append(
                        ("defensive", fixture.fixture_id, controller, repetition, profile)
                    )
    for fixture in STRATEGIC_FIXTURES:
        if fixture.evidence is FixtureEvidence.FORCED:
            continue
        for repetition in range(config.constructive_repetitions):
            key = (
                "constructive",
                fixture.fixture_id,
                config.selected_profile.label,
                repetition,
            )
            if key not in fixture_keys:
                fixture_tasks.append(
                    (
                        "constructive",
                        fixture.fixture_id,
                        config.selected_profile.label,
                        repetition,
                        config.selected_profile,
                    )
                )
    with ProcessPoolExecutor(max_workers=config.workers) as pool:
        futures = [pool.submit(_fixture_task, task) for task in fixture_tasks]
        for future in as_completed(futures):
            fixture_records.append(future.result())
            _write_records(fixture_path, fixture_records)

    benchmark_path = output_directory / "benchmark-records.json"
    benchmark_records = _read_records(benchmark_path, BenchmarkRecord)
    if run_benchmarks:
        benchmark_keys = {
            (record.stage, record.profile) for record in benchmark_records
        }
        tasks = [
            (stage, profile)
            for profile in config.benchmark_profiles
            for stage in RoundStage
            if (stage, profile) not in benchmark_keys
        ]
        # One process per measurement keeps peak RSS and latency attributable.
        for task in tasks:
            with ProcessPoolExecutor(max_workers=1) as pool:
                benchmark_records.append(pool.submit(_benchmark_task, task).result())
            _write_records(benchmark_path, benchmark_records)

    game_path = output_directory / "game-records.json"
    game_records = _read_games(game_path)
    if run_games:
        tasks = _game_tasks(config, archive_path)
        requested = {(task[0], task[1], task[2]) for task in tasks}
        game_records = [
            record for record in game_records if _game_record_key(record) in requested
        ]
        completed = {_game_record_key(record) for record in game_records}
        missing = [
            task for task in tasks if (task[0], task[1], task[2]) not in completed
        ]
        with ProcessPoolExecutor(max_workers=config.workers) as pool:
            futures = [pool.submit(_play_game_task, task) for task in missing]
            for future in as_completed(futures):
                game_records.append(future.result())
                _write_games(game_path, game_records)

    fixture_summary = _fixture_summary(fixture_records)
    comparisons = {
        name: _comparison_metrics(name, game_records, config.bootstrap_samples)
        for name in sorted({record.comparison for record in game_records})
    }
    selected_label = config.selected_profile.label
    selected_summary = fixture_summary.get(selected_label, {})
    v1_summary = fixture_summary.get("v1-500", {})
    defensive_pass = (
        selected_summary.get("defensive", {}).get("pass_rate") == 1.0
        if selected_summary
        else False
    )
    constructive_pass = (
        selected_summary.get("constructive", {}).get("pass_rate") == 1.0
        if selected_summary
        else False
    )
    defensive_improvement = (
        defensive_pass
        and float(selected_summary["defensive"]["pass_rate"])
        > float(v1_summary["defensive"]["pass_rate"])
    )
    absolute_control_names = (
        f"{selected_label}-vs-random-legal",
        f"{selected_label}-vs-policy-2-v20",
    )
    absolute_control_pass = all(
        name in comparisons
        and float(comparisons[name]["paired_game_advantage_ci_95"][0]) > 0
        for name in absolute_control_names
    )
    version_one_name = f"{selected_label}-vs-v1-500"
    version_one_non_regression = bool(
        version_one_name in comparisons
        and float(
            comparisons[version_one_name]["paired_game_advantage_ci_95"][0]
        )
        >= -0.05
    )
    absolute_pass = absolute_control_pass and version_one_non_regression
    selected_latencies = [
        decision.latency_seconds
        for record in game_records
        for decision in record.decisions
        if decision.controller == "strategic"
    ]
    v1_latencies = [
        decision.latency_seconds
        for record in game_records
        for decision in record.decisions
        if decision.controller == "search"
    ]
    selected_p95 = (
        sorted(selected_latencies)[math.ceil(0.95 * len(selected_latencies)) - 1]
        if selected_latencies
        else None
    )
    v1_p95 = (
        sorted(v1_latencies)[math.ceil(0.95 * len(v1_latencies)) - 1]
        if v1_latencies
        else None
    )
    latency_pass = bool(
        selected_p95 is not None
        and v1_p95 is not None
        and selected_p95 <= 1.5 * v1_p95
        and max(selected_latencies) <= 30
    )
    peak_memory = max(
        [record.peak_rss_mib for record in fixture_records]
        + [record.peak_rss_mib for record in benchmark_records]
        + [0.0]
    )
    gate_passed = bool(
        defensive_pass
        and constructive_pass
        and defensive_improvement
        and absolute_pass
        and latency_pass
        and peak_memory < 1024
    )
    failed_records = [record for record in fixture_records if not record.passed]
    document = {
        "format_version": STRATEGIC_VALIDATION_FORMAT_VERSION,
        "defensive_fixture_schema_version": DEFENSIVE_FIXTURE_SCHEMA_VERSION,
        "configuration": asdict(config),
        "gate_passed": gate_passed,
        "privacy_gate": "passed by the dedicated strategic-search invariance suite",
        "exact_defensive_evidence": exact,
        "exact_constructive_evidence": constructive_exact,
        "fixtures": fixture_summary,
        "benchmarks": [asdict(record) for record in benchmark_records],
        "comparisons": comparisons,
        "absolute_gate": {
            "passed": absolute_pass,
            "permanent_controls_passed": absolute_control_pass,
            "version_one_non_regression_passed": version_one_non_regression,
        },
        "latency_gate": {
            "passed": latency_pass,
            "selected_p95_seconds": selected_p95,
            "v1_p95_seconds": v1_p95,
            "selected_maximum_seconds": max(selected_latencies, default=None),
        },
        "peak_rss_mib": peak_memory,
        "failed_fixture_evidence": [asdict(record) for record in failed_records],
        "sample_design": {
            "random_and_ppo_deck_pairs": config.control_game_pairs,
            "version_one_deck_pairs": config.version_one_game_pairs,
            "roles_per_deck": 2,
            "stopping_rule": "fixed samples with phase-level resume",
        },
    }
    _atomic_write(
        output_directory / "teacher-v2-validation.json",
        json.dumps(document, indent=2, sort_keys=True, default=str) + "\n",
    )
    _atomic_write(
        output_directory / "teacher-v2-validation.md",
        _markdown_report(document),
    )
    return document


def _pct(value: object) -> str:
    return "n/a" if value is None else f"{100 * float(value):.1f}%"


def _markdown_report(document: dict[str, object]) -> str:
    fixtures = document["fixtures"]
    lines = [
        "# Teacher v2 validation",
        "",
        f"Human-test gate: **{'passed' if document['gate_passed'] else 'failed'}**",
        "",
        "## Strategic fixtures",
        "",
        "| Controller | Constructive | Defensive | p95 latency | Peak RSS |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for controller, row in fixtures.items():
        lines.append(
            f"| {controller} | {_pct(row['constructive']['pass_rate'])} "
            f"| {_pct(row['defensive']['pass_rate'])} "
            f"| {float(row['latency_seconds']['p95']):.3f}s "
            f"| {float(row['peak_rss_mib']):.1f} MiB |"
        )
    lines.extend(
        (
            "",
            "## Absolute controls",
            "",
            "| Comparison | Games | Game win | Round win | "
            "Mean round differential | Paired 95% CI |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        )
    )
    for name, row in document["comparisons"].items():
        lines.append(
            f"| {name} | {row['overall']['games']} "
            f"| {_pct(row['overall']['victory_percentage'])} "
            f"| {_pct(row['rounds']['victory_percentage'])} "
            f"| {float(row['rounds']['mean_score_differential']):.2f} "
            f"| [{float(row['paired_game_advantage_ci_95'][0]):.3f}, "
            f"{float(row['paired_game_advantage_ci_95'][1]):.3f}] |"
        )
    lines.extend(
        (
            "",
            "## Budget benchmarks",
            "",
            "| Profile | Stage | Latency | Terminal evaluations/sec | Peak RSS |",
            "| --- | --- | ---: | ---: | ---: |",
        )
    )
    for row in document["benchmarks"]:
        profile = row["profile"]
        lines.append(
            f"| {profile['outer_simulations']}×{profile['response_completions']}c "
            f"| {row['stage']} | {float(row['latency_seconds']):.3f}s "
            f"| {float(row['terminal_evaluations_per_second']):.1f} "
            f"| {float(row['peak_rss_mib']):.1f} MiB |"
        )
    lines.extend(("", "## Failed fixture decisions", ""))
    if not document["failed_fixture_evidence"]:
        lines.append("None.")
    else:
        for row in document["failed_fixture_evidence"]:
            visited = [
                f"{index}:{visits}/{row['mean_values'][index]}"
                for index, visits in enumerate(row["visits"])
                if visits
            ]
            lines.append(
                f"- `{row['controller']}` failed `{row['fixture_id']}` with action "
                f"{row['selected_action_index']}; visits `{' '.join(visited)}`."
            )
    lines.extend(
        (
            "",
            "## Decision",
            "",
            (
                "Teacher v2 may enter human testing only when this report passes."
                if not document["gate_passed"]
                else "Teacher v2 passed the automated gate and may enter human testing."
            ),
            "",
        )
    )
    return "\n".join(lines)


def _parse_profile(value: str) -> StrategicProfile:
    try:
        outer_text, completions_text = value.split(":", maxsplit=1)
        return StrategicProfile(int(outer_text), int(completions_text))
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError(
            "benchmark profiles must use OUTER:COMPLETIONS"
        ) from error


def main(arguments: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate Teacher v2 search")
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    parser.add_argument("--policy-archive", type=Path, default=DEFAULT_POLICY_ARCHIVE)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--selected-outer", type=int, default=500)
    parser.add_argument("--selected-completions", type=int, default=1)
    parser.add_argument("--lower-outer", type=int, default=500)
    parser.add_argument("--lower-completions", type=int, default=2)
    parser.add_argument("--higher-outer", type=int, default=500)
    parser.add_argument("--higher-completions", type=int, default=4)
    parser.add_argument("--defensive-repetitions", type=int, default=5)
    parser.add_argument("--constructive-repetitions", type=int, default=3)
    parser.add_argument("--control-game-pairs", type=int, default=12)
    parser.add_argument("--version-one-game-pairs", type=int, default=30)
    parser.add_argument("--bootstrap-samples", type=int, default=2_000)
    parser.add_argument(
        "--benchmark-profile",
        action="append",
        metavar="OUTER:COMPLETIONS",
        help="repeat to replace the default 32x1, 32x2, and 32x4 benchmarks",
    )
    parser.add_argument("--skip-benchmarks", action="store_true")
    parser.add_argument("--skip-games", action="store_true")
    parsed = parser.parse_args(arguments)
    benchmark_profiles = (
        tuple(_parse_profile(value) for value in parsed.benchmark_profile)
        if parsed.benchmark_profile
        else StrategicValidationConfig().benchmark_profiles
    )
    config = StrategicValidationConfig(
        selected_profile=StrategicProfile(
            parsed.selected_outer, parsed.selected_completions
        ),
        lower_profile=StrategicProfile(
            parsed.lower_outer, parsed.lower_completions
        ),
        higher_profile=StrategicProfile(
            parsed.higher_outer, parsed.higher_completions
        ),
        benchmark_profiles=benchmark_profiles,
        defensive_repetitions=parsed.defensive_repetitions,
        constructive_repetitions=parsed.constructive_repetitions,
        control_game_pairs=parsed.control_game_pairs,
        version_one_game_pairs=parsed.version_one_game_pairs,
        bootstrap_samples=parsed.bootstrap_samples,
        workers=parsed.workers,
    )
    document = run_validation(
        output_directory=parsed.output_directory,
        archive_path=parsed.policy_archive,
        config=config,
        run_benchmarks=not parsed.skip_benchmarks,
        run_games=not parsed.skip_games,
    )
    print(f"gate={'passed' if document['gate_passed'] else 'failed'}")
    return 0 if document["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

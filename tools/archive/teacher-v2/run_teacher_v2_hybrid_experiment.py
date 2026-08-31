"""Run the paired competence and efficiency experiment for hybrid Teacher v2."""

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
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import torch

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
from dracula.randomness import derive_seed
from dracula.search import (
    ResponseRankerGroupEvaluator,
    StrategicInformationSetSearch,
    StrategicResponseMode,
    StrategicSearchConfig,
    information_state_from_engine,
)
from dracula.search.defensive_fixtures import (
    DEFENSIVE_FIXTURES,
    replay_defensive_fixture,
)
from dracula.search.signal_measurement import build_signal_fixtures
from dracula.search.strategic_fixtures import (
    STRATEGIC_FIXTURES,
    replay_fixture,
)
from dracula.search.validation import VALIDATION_GAME_NAMESPACE

EXPERIMENT_SCHEMA_VERSION = "dracula-teacher-v2-hybrid-experiment-v1"
EXPERIMENT_REQUEST_NAMESPACE = "dracula-teacher-v2-hybrid-experiment-request-v1"
EXPERIMENT_GAME_NAMESPACE = "dracula-teacher-v2-hybrid-experiment-game-v1"
OUTER_SIMULATIONS = 32
RESPONSE_COMPLETIONS = 4
PLACEMENT_SEED_COUNT = 5
GAME_SEED_COUNT = 12

_MODE: StrategicResponseMode | None = None
_PLANNER: StrategicInformationSetSearch | None = None
_RANKER: "_TimedRanker | None" = None


class _TimedRanker:
    """Measure only information encoding, model inference, and group ranking."""

    def __init__(self, delegate: ResponseRankerGroupEvaluator) -> None:
        self.delegate = delegate
        self.artifact_digest = delegate.artifact_digest
        self.call_count = 0
        self.elapsed_seconds = 0.0

    def rank(self, information, groups):
        started = time.perf_counter()
        ranking = self.delegate.rank(information, groups)
        self.elapsed_seconds += time.perf_counter() - started
        self.call_count += 1
        return ranking


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _atomic_json(path: Path, payload: object) -> str:
    content = (
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _mode_config(
    mode: StrategicResponseMode,
    artifact_digest: str | None,
) -> StrategicSearchConfig:
    return StrategicSearchConfig(
        outer_simulation_budget=OUTER_SIMULATIONS,
        response_completions_per_action=RESPONSE_COMPLETIONS,
        response_mode=mode,
        response_ranker_artifact_digest=(
            None if mode is StrategicResponseMode.PURE else artifact_digest
        ),
    )


def _initialize_worker(mode_value: str, artifact_path: str) -> None:
    global _MODE, _PLANNER, _RANKER
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    _MODE = StrategicResponseMode(mode_value)
    _RANKER = (
        None
        if _MODE is StrategicResponseMode.PURE
        else _TimedRanker(
            ResponseRankerGroupEvaluator.from_artifact(artifact_path)
        )
    )
    config = _mode_config(
        _MODE, None if _RANKER is None else _RANKER.artifact_digest
    )
    _PLANNER = StrategicInformationSetSearch(
        config, response_ranker=_RANKER
    )


def _diagnostics(result) -> list[dict[str, object]]:
    return [
        {
            "representative_action_index": (
                diagnostic.group.representative_action_index
            ),
            "member_action_indices": list(
                diagnostic.group.member_action_indices
            ),
            "visits": diagnostic.visits,
            "mean_value": diagnostic.mean_value,
        }
        for diagnostic in result.group_diagnostics
    ]


def _run_search(information, request_seed: bytes) -> dict[str, object]:
    if _MODE is None or _PLANNER is None:
        raise RuntimeError("hybrid experiment worker is not initialized")
    before_calls = 0 if _RANKER is None else _RANKER.call_count
    before_model_seconds = (
        0.0 if _RANKER is None else _RANKER.elapsed_seconds
    )
    result = _PLANNER.search(information, request_seed)
    model_calls = (
        0 if _RANKER is None else _RANKER.call_count - before_calls
    )
    model_seconds = (
        0.0
        if _RANKER is None
        else _RANKER.elapsed_seconds - before_model_seconds
    )
    if model_calls != result.response_model_call_count:
        raise RuntimeError("measured and reported model calls disagree")
    return {
        "mode": _MODE.value,
        "selected_action_index": result.selected_action_index,
        "selected_representative_action_index": (
            result.selected_representative_action_index
        ),
        "action_visits": list(result.action_visits),
        "mean_action_values": list(result.mean_action_values),
        "group_diagnostics": _diagnostics(result),
        "outer_simulations": result.simulation_count,
        "response_requests": result.response_request_count,
        "unique_response_evaluations": (
            result.unique_response_evaluation_count
        ),
        "response_cache_hits": result.response_cache_hit_count,
        "response_candidate_actions": (
            result.response_candidate_action_count
        ),
        "response_terminal_evaluations": (
            result.response_terminal_evaluation_count
        ),
        "total_terminal_evaluations": (
            result.total_terminal_evaluation_count
        ),
        "model_calls": model_calls,
        "model_inference_seconds": model_seconds,
        "latency_seconds": result.elapsed_seconds,
        "peak_rss_bytes": _peak_rss_bytes(),
    }


def _fixture_task(task: tuple[str, str]) -> dict[str, object]:
    suite, fixture_id = task
    if suite == "constructive":
        fixture = next(
            value
            for value in STRATEGIC_FIXTURES
            if value.fixture_id == fixture_id
        )
        state = replay_fixture(fixture)
    else:
        fixture = next(
            value
            for value in DEFENSIVE_FIXTURES
            if value.fixture_id == fixture_id
        )
        state = replay_defensive_fixture(fixture)
    information = information_state_from_engine(state)
    record = _run_search(
        information,
        derive_seed(
            EXPERIMENT_REQUEST_NAMESPACE,
            "fixture",
            suite,
            fixture_id,
        ),
    )
    selected = int(record["selected_action_index"])
    representative = int(record["selected_representative_action_index"])
    expected = set(fixture.expected_action_indices)
    selected_group = next(
        (
            diagnostic
            for diagnostic in record["group_diagnostics"]
            if diagnostic["representative_action_index"] == representative
        ),
        None,
    )
    members = (
        {selected}
        if selected_group is None
        else set(selected_group["member_action_indices"])
    )
    return {
        "kind": "fixture",
        "key": f"{suite}:{fixture_id}",
        "suite": suite,
        "fixture_id": fixture_id,
        "title": fixture.title,
        "placement_number": len(state.current_round_moves) + 1,
        "actor": information.player.value,
        "dealer": information.dealer.value,
        "actor_is_dealer": information.player is information.dealer,
        "expected_action_indices": sorted(expected),
        "exact_action_pass": selected in expected,
        "strategic_group_pass": bool(expected & members),
        **record,
    }


def _placement_task(task: tuple[str, int]) -> dict[str, object]:
    fixture_id, seed_index = task
    fixture = next(
        value
        for value in build_signal_fixtures()
        if value.fixture_id == fixture_id
    )
    record = _run_search(
        fixture.information,
        derive_seed(
            EXPERIMENT_REQUEST_NAMESPACE,
            "placement",
            fixture.fixture_id,
            str(seed_index),
        ),
    )
    return {
        "kind": "placement",
        "key": f"{fixture.fixture_id}:{seed_index}",
        "fixture_id": fixture.fixture_id,
        "placement_number": fixture.placement_number,
        "category": fixture.category,
        "pattern": fixture.pattern,
        "seed_index": seed_index,
        "actor": fixture.information.player.value,
        "dealer": fixture.information.dealer.value,
        "actor_is_dealer": (
            fixture.information.player is fixture.information.dealer
        ),
        **record,
    }


def _select_move(
    state,
    mode: StrategicResponseMode,
    controller_label: str,
    game_seed: str,
) -> tuple[object, dict[str, object] | None]:
    moves = legal_moves(state, state.active_player)
    if len(moves) == 1:
        return moves[0], None
    if _PLANNER is None or _MODE is None:
        raise RuntimeError("game worker is not initialized")
    if mode is _MODE:
        planner = _PLANNER
        ranker = _RANKER
    elif mode is StrategicResponseMode.PURE:
        planner = StrategicInformationSetSearch(
            _mode_config(StrategicResponseMode.PURE, None)
        )
        ranker = None
    else:
        raise RuntimeError("game worker cannot select an unconfigured mode")
    information = information_state_from_engine(state)
    request_seed = derive_seed(
        EXPERIMENT_GAME_NAMESPACE,
        game_seed,
        state.active_player.value,
        str(state.round_number),
        str(len(state.current_round_moves) + 1),
    )
    before_calls = 0 if ranker is None else ranker.call_count
    before_seconds = 0.0 if ranker is None else ranker.elapsed_seconds
    result = planner.search(information, request_seed)
    model_calls = 0 if ranker is None else ranker.call_count - before_calls
    model_seconds = (
        0.0 if ranker is None else ranker.elapsed_seconds - before_seconds
    )
    from dracula.bridge import build_policy_turn_context

    context = build_policy_turn_context(state, state.active_player)
    selected = context.action_table[result.selected_action_index]
    if selected is None:
        raise RuntimeError("experiment search selected a masked action")
    return selected, {
        "controller": controller_label,
        "mode": mode.value,
        "round_number": state.round_number,
        "placement_number": len(state.current_round_moves) + 1,
        "actor": state.active_player.value,
        "actor_is_dealer": state.active_player is state.dealer,
        "selected_action_index": result.selected_action_index,
        "selected_representative_action_index": (
            result.selected_representative_action_index
        ),
        "latency_seconds": result.elapsed_seconds,
        "response_terminal_evaluations": (
            result.response_terminal_evaluation_count
        ),
        "total_terminal_evaluations": result.total_terminal_evaluation_count,
        "model_calls": model_calls,
        "model_inference_seconds": model_seconds,
    }


def _game_task(task: tuple[int, str, str]) -> dict[str, object]:
    game_index, game_seed, subject_role_value = task
    if _MODE is None or _MODE is StrategicResponseMode.PURE:
        raise RuntimeError("game phase requires one hybrid subject")
    subject_role = EnginePlayer(subject_role_value)
    controls = {
        subject_role: (_MODE, "subject"),
        other_player(subject_role): (StrategicResponseMode.PURE, "teacher"),
    }
    state = create_game(game_seed)
    initial_dealer = state.dealer
    decisions = []
    rounds = []
    while state.status is not EngineStatus.GAME_COMPLETE:
        while state.status is EngineStatus.PLAYING:
            actor = state.active_player
            if actor is None:
                raise RuntimeError("game comparison lost its active player")
            mode, label = controls[actor]
            move, decision = _select_move(
                state, mode, label, game_seed
            )
            if decision is not None:
                decisions.append(decision)
            state = apply_move(state, move).state
        result = state.pending_round_result
        if result is None:
            raise RuntimeError("game comparison completed a round without score")
        rounds.append(
            {
                "round_number": state.round_number,
                "subject_is_dealer": subject_role is state.dealer,
                "subject_score": result.round_scores[subject_role],
                "teacher_score": result.round_scores[
                    other_player(subject_role)
                ],
            }
        )
        state = advance_after_round(state)
    outcome = derive_game_outcome(state)
    return {
        "kind": "game",
        "key": f"pair-{game_index:03d}:{subject_role.value}",
        "comparison": f"{_MODE.value}-vs-pure",
        "game_seed_id": hashlib.sha256(game_seed.encode("ascii")).hexdigest(),
        "subject_role": subject_role.value,
        "initial_dealer": initial_dealer.value,
        "subject_won": outcome.winner is subject_role,
        "tied": outcome.winner is None,
        "rounds": rounds,
        "decisions": decisions,
        "peak_rss_bytes": _peak_rss_bytes(),
    }


def _run_phase(
    *,
    mode: StrategicResponseMode,
    artifact_path: Path,
    tasks: list[tuple[str, tuple[Any, ...]]],
    workers: int,
    progress,
) -> list[dict[str, object]]:
    functions = {
        "fixture": _fixture_task,
        "placement": _placement_task,
        "game": _game_task,
    }
    records = []
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_initialize_worker,
        initargs=(mode.value, str(artifact_path)),
    ) as pool:
        futures = {
            pool.submit(functions[kind], task): (kind, task)
            for kind, task in tasks
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            records.append(future.result())
            kind, task = futures[future]
            task_label = (
                (task[0], task[2]) if kind == "game" else task
            )
            progress(
                f"{mode.value} {kind} {completed}/{len(futures)} "
                f"{task_label}"
            )
    return sorted(records, key=lambda row: (str(row["kind"]), str(row["key"])))


def _percentile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[
        max(0, min(len(ordered) - 1, math.ceil(len(ordered) * probability) - 1))
    ]


def _mode_metrics(records: list[dict[str, object]]) -> dict[str, object]:
    fixed_decisions = [
        record for record in records if record["kind"] != "game"
    ]
    game_decisions = [
        decision
        for record in records
        if record["kind"] == "game"
        for decision in record["decisions"]
        if decision["controller"] == "subject"
    ]
    fixed_latencies = [
        float(record["latency_seconds"]) for record in fixed_decisions
    ]
    game_latencies = [
        float(record["latency_seconds"]) for record in game_decisions
    ]
    fixtures = [record for record in records if record["kind"] == "fixture"]
    games = [record for record in records if record["kind"] == "game"]
    rounds = [round_ for game in games for round_ in game["rounds"]]
    round_scores = [
        1.0
        if row["subject_score"] > row["teacher_score"]
        else 0.5
        if row["subject_score"] == row["teacher_score"]
        else 0.0
        for row in rounds
    ]
    return {
        "decision_count": len(fixed_decisions),
        "gameplay_decision_count": len(game_decisions),
        "fixtures": {
            suite: {
                "count": len(selected := [
                    record for record in fixtures if record["suite"] == suite
                ]),
                "exact_action_passes": sum(
                    bool(record["exact_action_pass"]) for record in selected
                ),
                "strategic_group_passes": sum(
                    bool(record["strategic_group_pass"]) for record in selected
                ),
            }
            for suite in ("constructive", "defensive")
        },
        "games": {
            "count": len(games),
            "wins": sum(bool(game["subject_won"]) for game in games),
            "ties": sum(bool(game["tied"]) for game in games),
            "victory_percentage": (
                statistics.fmean(
                    0.5 if game["tied"] else float(game["subject_won"])
                    for game in games
                )
                if games
                else None
            ),
            "queen": _game_split(games, role="queen"),
            "king": _game_split(games, role="king"),
        },
        "rounds": {
            "count": len(rounds),
            "victory_percentage": (
                statistics.fmean(round_scores) if round_scores else None
            ),
            "mean_score_differential": (
                statistics.fmean(
                    int(row["subject_score"]) - int(row["teacher_score"])
                    for row in rounds
                )
                if rounds
                else None
            ),
            "dealer": _round_split(rounds, True),
            "non_dealer": _round_split(rounds, False),
            "queen": _role_round_split(games, EnginePlayer.QUEEN),
            "king": _role_round_split(games, EnginePlayer.KING),
        },
        "latency_seconds": {
            "p50": _percentile(fixed_latencies, 0.50),
            "p95": _percentile(fixed_latencies, 0.95),
            "maximum": max(fixed_latencies, default=None),
        },
        "gameplay_latency_seconds": {
            "p50": _percentile(game_latencies, 0.50),
            "p95": _percentile(game_latencies, 0.95),
            "maximum": max(game_latencies, default=None),
        },
        "response_terminal_evaluations": sum(
            int(record["response_terminal_evaluations"])
            for record in fixed_decisions
        ),
        "gameplay_response_terminal_evaluations": sum(
            int(record["response_terminal_evaluations"])
            for record in game_decisions
        ),
        "model_inference": _inference_metrics(fixed_decisions),
        "gameplay_model_inference": _inference_metrics(game_decisions),
        "peak_rss_bytes": max(
            (
                int(record["peak_rss_bytes"])
                for record in records
                if "peak_rss_bytes" in record
            ),
            default=0,
        ),
    }


def _game_split(
    games: list[dict[str, object]], *, role: str
) -> dict[str, object]:
    selected = [game for game in games if game["subject_role"] == role]
    return {
        "games": len(selected),
        "wins": sum(bool(game["subject_won"]) for game in selected),
        "ties": sum(bool(game["tied"]) for game in selected),
        "victory_percentage": (
            statistics.fmean(
                0.5 if game["tied"] else float(game["subject_won"])
                for game in selected
            )
            if selected
            else None
        ),
    }


def _round_split(
    rounds: list[dict[str, object]], dealer: bool
) -> dict[str, object]:
    selected = [
        row for row in rounds if bool(row["subject_is_dealer"]) is dealer
    ]
    scores = [
        1.0
        if row["subject_score"] > row["teacher_score"]
        else 0.5
        if row["subject_score"] == row["teacher_score"]
        else 0.0
        for row in selected
    ]
    return {
        "rounds": len(selected),
        "victory_percentage": statistics.fmean(scores) if scores else None,
        "mean_score_differential": (
            statistics.fmean(
                int(row["subject_score"]) - int(row["teacher_score"])
                for row in selected
            )
            if selected
            else None
        ),
    }


def _role_round_split(
    games: list[dict[str, object]], role: EnginePlayer
) -> dict[str, object]:
    rounds = [
        round_
        for game in games
        if game["subject_role"] == role.value
        for round_ in game["rounds"]
    ]
    scores = [
        1.0
        if row["subject_score"] > row["teacher_score"]
        else 0.5
        if row["subject_score"] == row["teacher_score"]
        else 0.0
        for row in rounds
    ]
    return {
        "rounds": len(rounds),
        "victory_percentage": statistics.fmean(scores) if scores else None,
        "mean_score_differential": (
            statistics.fmean(
                int(row["subject_score"]) - int(row["teacher_score"])
                for row in rounds
            )
            if rounds
            else None
        ),
    }


def _inference_metrics(
    records: list[dict[str, object]],
) -> dict[str, object]:
    calls = sum(int(record["model_calls"]) for record in records)
    elapsed = sum(
        float(record["model_inference_seconds"]) for record in records
    )
    return {
        "calls": calls,
        "seconds": elapsed,
        "per_call_milliseconds": (
            1_000.0 * elapsed / calls if calls else None
        ),
    }


def _paired_analysis(
    records_by_mode: dict[str, list[dict[str, object]]],
) -> dict[str, object]:
    pure = {
        (record["kind"], record["key"]): record
        for record in records_by_mode[StrategicResponseMode.PURE.value]
        if record["kind"] != "game"
    }
    comparisons = {}
    for mode in (
        value
        for value in StrategicResponseMode
        if value is not StrategicResponseMode.PURE
        and value.value in records_by_mode
    ):
        hybrid_records = [
            record
            for record in records_by_mode.get(mode.value, [])
            if record["kind"] != "game"
        ]
        paired = []
        disagreements = []
        for hybrid in hybrid_records:
            teacher = pure[(hybrid["kind"], hybrid["key"])]
            teacher_values = {
                int(row["representative_action_index"]): row["mean_value"]
                for row in teacher["group_diagnostics"]
            }
            hybrid_values = {
                int(row["representative_action_index"]): row["mean_value"]
                for row in hybrid["group_diagnostics"]
            }
            teacher_action = int(
                teacher["selected_representative_action_index"]
            )
            hybrid_action = int(
                hybrid["selected_representative_action_index"]
            )
            if not teacher_values:
                agreement = teacher_action == hybrid_action
                paired.append(
                    {
                        "kind": hybrid["kind"],
                        "key": hybrid["key"],
                        "placement_number": hybrid["placement_number"],
                        "agreement": agreement,
                        "teacher_value_regret": 0.0,
                    }
                )
                if not agreement:
                    raise RuntimeError(
                        "forced controller results disagree on the only action"
                    )
                continue
            available_teacher_values = [
                float(value)
                for value in teacher_values.values()
                if value is not None
            ]
            hybrid_teacher_value = teacher_values[hybrid_action]
            teacher_regret = (
                max(available_teacher_values) - float(hybrid_teacher_value)
                if hybrid_teacher_value is not None
                else None
            )
            agreement = teacher_action == hybrid_action
            row = {
                "kind": hybrid["kind"],
                "key": hybrid["key"],
                "placement_number": hybrid["placement_number"],
                "agreement": agreement,
                "teacher_value_regret": teacher_regret,
            }
            paired.append(row)
            if not agreement:
                disagreements.append(
                    {
                        **row,
                        "suite": hybrid.get("suite"),
                        "fixture_id": hybrid.get("fixture_id"),
                        "seed_index": hybrid.get("seed_index"),
                        "actor": hybrid["actor"],
                        "dealer": hybrid["dealer"],
                        "teacher_selected_representative": teacher_action,
                        "hybrid_selected_representative": hybrid_action,
                        "teacher_selected_concrete": teacher[
                            "selected_action_index"
                        ],
                        "hybrid_selected_concrete": hybrid[
                            "selected_action_index"
                        ],
                        "teacher_value_of_teacher_action": teacher_values[
                            teacher_action
                        ],
                        "teacher_value_of_hybrid_action": teacher_values[
                            hybrid_action
                        ],
                        "hybrid_value_of_teacher_action": hybrid_values[
                            teacher_action
                        ],
                        "hybrid_value_of_hybrid_action": hybrid_values[
                            hybrid_action
                        ],
                        "teacher_groups": teacher["group_diagnostics"],
                        "hybrid_groups": hybrid["group_diagnostics"],
                    }
                )
        comparisons[mode.value] = {
            "paired_decisions": len(paired),
            "strategic_group_agreement": (
                sum(bool(row["agreement"]) for row in paired) / len(paired)
            ),
            "mean_teacher_value_regret": statistics.fmean(
                float(row["teacher_value_regret"])
                for row in paired
                if row["teacher_value_regret"] is not None
            ),
            "by_placement": {
                str(placement): {
                    "decisions": len(selected := [
                        row
                        for row in paired
                        if row["placement_number"] == placement
                    ]),
                    "agreement": (
                        sum(bool(row["agreement"]) for row in selected)
                        / len(selected)
                    ),
                    "mean_teacher_value_regret": statistics.fmean(
                        float(row["teacher_value_regret"])
                        for row in selected
                        if row["teacher_value_regret"] is not None
                    ),
                }
                for placement in range(1, 8)
            },
            "disagreement_count": len(disagreements),
            "disagreements": disagreements,
        }
    return comparisons


def _fixture_tasks() -> list[tuple[str, tuple[Any, ...]]]:
    return [
        ("fixture", ("constructive", fixture.fixture_id))
        for fixture in STRATEGIC_FIXTURES
    ] + [
        ("fixture", ("defensive", fixture.fixture_id))
        for fixture in DEFENSIVE_FIXTURES
    ]


def _placement_tasks() -> list[tuple[str, tuple[Any, ...]]]:
    return [
        ("placement", (fixture.fixture_id, seed_index))
        for fixture in build_signal_fixtures()
        for seed_index in range(PLACEMENT_SEED_COUNT)
    ]


def _game_tasks() -> list[tuple[str, tuple[Any, ...]]]:
    seeds = tuple(
        derive_seed(VALIDATION_GAME_NAMESPACE, f"pair-{index:03d}").hex()
        for index in range(GAME_SEED_COUNT)
    )
    return [
        ("game", (game_index, game_seed, role.value))
        for game_index, game_seed in enumerate(seeds)
        for role in EnginePlayer
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--prior-results",
        action="append",
        default=[],
        type=Path,
        help="Merge sealed records from an earlier experiment phase.",
    )
    parser.add_argument(
        "--phases",
        nargs="+",
        choices=("decisions", "games", "summarize", "all"),
        default=("all",),
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=tuple(mode.value for mode in StrategicResponseMode),
        default=tuple(mode.value for mode in StrategicResponseMode),
    )
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("worker count must be positive")
    artifact = args.artifact.expanduser().resolve()
    if not artifact.is_file():
        raise FileNotFoundError(f"response ranker is unavailable: {artifact}")
    phases = (
        {"decisions", "games"}
        if "all" in args.phases
        else set(args.phases)
    )
    modes = tuple(StrategicResponseMode(mode) for mode in args.modes)
    records_by_mode: dict[str, list[dict[str, object]]] = defaultdict(list)
    for prior_path in args.prior_results:
        prior = json.loads(
            prior_path.expanduser().resolve().read_text(encoding="utf-8")
        )
        if prior.get("format_version") != EXPERIMENT_SCHEMA_VERSION:
            raise ValueError("prior hybrid experiment schema is incompatible")
        if (
            prior.get("configuration", {}).get(
                "response_ranker_artifact_digest"
            )
            != hashlib.sha256(artifact.read_bytes()).hexdigest()
        ):
            raise ValueError("prior hybrid experiment used another ranker")
        for mode, records in prior.get("records", {}).items():
            records_by_mode[mode].extend(records)
    started = time.perf_counter()

    def progress(message: str) -> None:
        print(message, flush=True)

    if "decisions" in phases:
        tasks = _fixture_tasks() + _placement_tasks()
        for mode in modes:
            records_by_mode[mode.value].extend(
                _run_phase(
                    mode=mode,
                    artifact_path=artifact,
                    tasks=tasks,
                    workers=args.workers,
                    progress=progress,
                )
            )
    if "games" in phases:
        tasks = _game_tasks()
        for mode in modes:
            if mode is StrategicResponseMode.PURE:
                continue
            records_by_mode[mode.value].extend(
                _run_phase(
                    mode=mode,
                    artifact_path=artifact,
                    tasks=tasks,
                    workers=args.workers,
                    progress=progress,
                )
            )
    for mode, records in records_by_mode.items():
        identities = [(record["kind"], record["key"]) for record in records]
        if len(identities) != len(set(identities)):
            raise ValueError(f"duplicate experiment record for {mode}")
    artifact_digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    document = {
        "format_version": EXPERIMENT_SCHEMA_VERSION,
        "configuration": {
            "outer_simulations": OUTER_SIMULATIONS,
            "response_completions": RESPONSE_COMPLETIONS,
            "placement_request_seeds": PLACEMENT_SEED_COUNT,
            "game_seed_count": GAME_SEED_COUNT,
            "game_roles_per_seed": 2,
            "workers": args.workers,
            "response_ranker_artifact_digest": artifact_digest,
            "modes": [mode.value for mode in modes],
            "phases": sorted(phases),
        },
        "wall_seconds": time.perf_counter() - started,
        "metrics": {
            mode: _mode_metrics(records)
            for mode, records in sorted(records_by_mode.items())
        },
        "paired_decisions": (
            _paired_analysis(records_by_mode)
            if StrategicResponseMode.PURE.value in records_by_mode
            and StrategicResponseMode.STUDENT_DIRECT.value in records_by_mode
            and StrategicResponseMode.STUDENT_TOP_2.value in records_by_mode
            else {}
        ),
        "records": dict(sorted(records_by_mode.items())),
    }
    digest = _atomic_json(args.output, document)
    print(f"sealed {args.output} sha256={digest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

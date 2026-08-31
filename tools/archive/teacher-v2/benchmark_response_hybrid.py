"""Benchmark pure and response-ranker Teacher v2 modes on fixed states."""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import resource
import statistics
import sys
from pathlib import Path
from time import perf_counter

import torch

from dracula.engine import apply_move, create_game, legal_moves
from dracula.randomness import derive_seed
from dracula.search import (
    ResponseRankerGroupEvaluator,
    StrategicInformationSetSearch,
    StrategicResponseMode,
    StrategicSearchConfig,
    information_state_from_engine,
)

BENCHMARK_SCHEMA_VERSION = "dracula-response-hybrid-benchmark-v1"
BENCHMARK_SEED_NAMESPACE = "dracula-response-hybrid-benchmark-v1"
PLACEMENTS = ((1, 0), (4, 3), (6, 5))
FIXTURE_COUNT = 3


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(len(ordered) * fraction) - 1)]


def _information(fixture_index: int, move_count: int):
    state = create_game(f"response-hybrid-benchmark-{fixture_index}")
    for _ in range(move_count):
        state = apply_move(
            state, legal_moves(state, state.active_player)[0]
        ).state
    return information_state_from_engine(state)


def _worker(
    mode_value: str,
    artifact_path: str,
    queue,
) -> None:
    torch.set_num_threads(1)
    mode = StrategicResponseMode(mode_value)
    ranker = (
        None
        if mode is StrategicResponseMode.PURE
        else ResponseRankerGroupEvaluator.from_artifact(artifact_path)
    )
    config = StrategicSearchConfig(
        outer_simulation_budget=32,
        response_completions_per_action=4,
        response_mode=mode,
        response_ranker_artifact_digest=(
            None if ranker is None else ranker.artifact_digest
        ),
    )
    records = []
    for placement, move_count in PLACEMENTS:
        for fixture_index in range(FIXTURE_COUNT):
            information = _information(fixture_index, move_count)
            request_seed = derive_seed(
                BENCHMARK_SEED_NAMESPACE,
                str(placement),
                str(fixture_index),
            )
            planner = StrategicInformationSetSearch(
                config, response_ranker=ranker
            )
            started = perf_counter()
            result = planner.search(information, request_seed)
            records.append(
                {
                    "placement": placement,
                    "fixture_index": fixture_index,
                    "elapsed_seconds": perf_counter() - started,
                    "selected_action_index": result.selected_action_index,
                    "selected_representative_action_index": (
                        result.selected_representative_action_index
                    ),
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
                    "response_model_calls": result.response_model_call_count,
                    "outer_terminal_evaluations": result.simulation_count,
                }
            )
    latencies = [record["elapsed_seconds"] for record in records]
    queue.put(
        {
            "mode": mode.value,
            "configuration_digest": config.digest,
            "artifact_digest": (
                None if ranker is None else ranker.artifact_digest
            ),
            "decisions": len(records),
            "latency_median_seconds": statistics.median(latencies),
            "latency_p95_seconds": _percentile(latencies, 0.95),
            "latency_max_seconds": max(latencies),
            "peak_rss_bytes": _peak_rss_bytes(),
            "totals": {
                key: sum(record[key] for record in records)
                for key in (
                    "response_requests",
                    "unique_response_evaluations",
                    "response_cache_hits",
                    "response_candidate_actions",
                    "response_terminal_evaluations",
                    "response_model_calls",
                    "outer_terminal_evaluations",
                )
            },
            "by_placement": {
                str(placement): {
                    "decisions": sum(
                        record["placement"] == placement for record in records
                    ),
                    "latency_median_seconds": statistics.median(
                        record["elapsed_seconds"]
                        for record in records
                        if record["placement"] == placement
                    ),
                    "response_terminal_evaluations": sum(
                        record["response_terminal_evaluations"]
                        for record in records
                        if record["placement"] == placement
                    ),
                    "response_model_calls": sum(
                        record["response_model_calls"]
                        for record in records
                        if record["placement"] == placement
                    ),
                }
                for placement, _ in PLACEMENTS
            },
            "records": records,
        }
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    context = multiprocessing.get_context("spawn")
    results = []
    for mode in StrategicResponseMode:
        queue = context.Queue()
        process = context.Process(
            target=_worker,
            args=(mode.value, args.artifact, queue),
        )
        process.start()
        result = queue.get()
        process.join()
        if process.exitcode != 0:
            raise RuntimeError(f"{mode.value} benchmark process failed")
        results.append(result)
    pure_terminals = next(
        result["totals"]["response_terminal_evaluations"]
        for result in results
        if result["mode"] == StrategicResponseMode.PURE.value
    )
    for result in results:
        result["terminal_evaluation_savings_fraction"] = (
            1.0
            - result["totals"]["response_terminal_evaluations"]
            / pure_terminals
        )
    document = {
        "format_version": BENCHMARK_SCHEMA_VERSION,
        "outer_simulations": 32,
        "response_completions_per_action": 4,
        "fixtures_per_placement": FIXTURE_COUNT,
        "placements": [placement for placement, _ in PLACEMENTS],
        "results": results,
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(document, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

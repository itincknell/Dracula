"""Measure Teacher v2 root signal before and after destination reduction."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import tempfile
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Callable, Protocol

from dracula.bridge import global_grid_index
from dracula.engine import (
    EngineMove,
    EnginePlayer,
    apply_move,
    create_game,
    legal_moves,
)
from dracula.models import POLICY_GRID_INDICES
from dracula.randomness import derive_seed
from dracula.search.information import (
    SearchInformationState,
    information_state_from_engine,
)
from dracula.search.strategic import (
    GREEDY_RESPONSE_CONTINUATION_PROFILE,
    GREEDY_RESPONSE_SCHEMA_VERSION,
    GREEDY_RESPONSE_SELECTION_PROFILE,
    STRATEGIC_SEARCH_SCHEMA_VERSION,
    STRATEGIC_SELECTION_PROFILE,
    StrategicActionGroup,
    StrategicInformationSetSearch,
    StrategicSearchConfig,
    strategic_action_groups,
)

SIGNAL_MEASUREMENT_SCHEMA_VERSION = (
    "dracula-teacher-v2-symmetry-signal-v1"
)
SIGNAL_REQUEST_NAMESPACE = "dracula-teacher-v2-signal-request-v1"
BASELINE_PROFILE = "teacher-v2-32x4-no-destination-symmetry"
REDUCED_PROFILE = "teacher-v2-32x4-authoritative-destination-symmetry"
REQUEST_SEED_COUNT = 5

_ROLE_SEEDS = {
    EnginePlayer.QUEEN: "search-role-0",
    EnginePlayer.KING: "engine-contract-fixture-1",
}


class SignalMeasurementError(ValueError):
    """A fixed fixture or computed measurement violates its contract."""


class _SearchResult(Protocol):
    information_state_fingerprint: str
    config_digest: str
    action_visits: tuple[int, ...]
    mean_action_values: tuple[float | None, ...]
    selected_action_index: int
    simulation_count: int
    information_set_count: int
    principal_continuation: object
    response_request_count: int
    unique_response_evaluation_count: int
    response_cache_hit_count: int
    response_candidate_action_count: int
    response_terminal_evaluation_count: int
    total_terminal_evaluation_count: int
    selected_representative_action_index: int
    group_diagnostics: tuple[object, ...]
    elapsed_seconds: float
    peak_resident_memory_bytes: int


@dataclass(frozen=True, slots=True)
class SignalFixture:
    fixture_id: str
    placement_number: int
    category: str
    pattern: str
    information: SearchInformationState

    def __post_init__(self) -> None:
        if not self.fixture_id or not self.category or not self.pattern:
            raise SignalMeasurementError("fixture labels must be nonempty")
        if (
            type(self.placement_number) is not int
            or not 1 <= self.placement_number <= 7
            or self.information.turn_number != self.placement_number
        ):
            raise SignalMeasurementError(
                "fixture placement does not match its information state"
            )


@dataclass(frozen=True, slots=True)
class AggregatedRootSignal:
    groups: tuple[StrategicActionGroup, ...]
    group_visits: tuple[int, ...]
    group_mean_values: tuple[float, ...]
    selected_group_index: int

    def __post_init__(self) -> None:
        if (
            not self.groups
            or len(self.group_visits) != len(self.groups)
            or len(self.group_mean_values) != len(self.groups)
            or not 0 <= self.selected_group_index < len(self.groups)
        ):
            raise SignalMeasurementError(
                "aggregated root signal has inconsistent dimensions"
            )
        if any(type(value) is not int or value < 0 for value in self.group_visits):
            raise SignalMeasurementError("aggregated visits are invalid")
        if any(
            not math.isfinite(value) or not -1.0 <= value <= 1.0
            for value in self.group_mean_values
        ):
            raise SignalMeasurementError("aggregated values are invalid")


def _apply_relative_destination(state, perspective, position: int):
    actor = state.active_player
    hand_slot = next(
        index
        for index, card_id in enumerate(state.hands[actor])
        if card_id is not None
    )
    move = EngineMove(
        actor,
        hand_slot,
        global_grid_index(perspective, position - 1),
    )
    if move not in legal_moves(state, actor):
        raise SignalMeasurementError(
            "fixed fixture requested an illegal relative destination"
        )
    return apply_move(state, move).state


def _state_for_fixture(
    *,
    initial_dealer: EnginePlayer,
    placement_number: int,
    prior_relative_destinations: tuple[int, ...],
):
    state = create_game(_ROLE_SEEDS[initial_dealer])
    if state.dealer is not initial_dealer:
        raise SignalMeasurementError("role fixture seed has the wrong dealer")
    perspective = (
        state.dealer
        if placement_number % 2 == 0
        else state.active_player
    )
    for position in prior_relative_destinations:
        state = _apply_relative_destination(state, perspective, position)
    if len(state.current_round_moves) != placement_number - 1:
        raise SignalMeasurementError(
            "fixed fixture must specify every prior placement"
        )
    return state


def build_signal_fixtures() -> tuple[SignalFixture, ...]:
    """Create all fixed role-balanced states without retaining engine seeds."""

    specs: list[tuple[int, str, str, tuple[int, ...]]] = [
        (1, "placement-1-opening", "center-only", ()),
    ]
    specs.extend(
        (
            2,
            "placement-2-adjacent",
            f"center-plus-{position}",
            (position,),
        )
        for position in (2, 8, 4, 6)
    )
    specs.extend(
        (
            (3, "placement-3-line", "horizontal-line", (4, 6)),
            (3, "placement-3-line", "vertical-line", (2, 8)),
            (
                3,
                "placement-3-no-symmetry",
                "corner-growth-control",
                (2, 4),
            ),
        )
    )
    no_symmetry_sequence = (2, 4, 1, 7, 8, 3)
    specs.extend(
        (
            placement,
            f"placement-{placement}-no-symmetry",
            "continued-corner-growth",
            no_symmetry_sequence[: placement - 1],
        )
        for placement in range(4, 8)
    )

    fixtures = []
    for placement, category, pattern, prior_destinations in specs:
        for dealer in EnginePlayer:
            state = _state_for_fixture(
                initial_dealer=dealer,
                placement_number=placement,
                prior_relative_destinations=prior_destinations,
            )
            information = information_state_from_engine(state)
            fixtures.append(
                SignalFixture(
                    fixture_id=(
                        f"p{placement}-{pattern}-"
                        f"actor-{information.player.value}-"
                        f"dealer-{dealer.value}"
                    ),
                    placement_number=placement,
                    category=category,
                    pattern=pattern,
                    information=information,
                )
            )
    _validate_fixture_matrix(tuple(fixtures))
    return tuple(fixtures)


def _validate_fixture_matrix(fixtures: tuple[SignalFixture, ...]) -> None:
    counts = Counter(fixture.category for fixture in fixtures)
    expected = {
        "placement-1-opening": 2,
        "placement-2-adjacent": 8,
        "placement-3-line": 4,
        "placement-3-no-symmetry": 2,
        **{
            f"placement-{placement}-no-symmetry": 2
            for placement in range(4, 8)
        },
    }
    if counts != Counter(expected):
        raise SignalMeasurementError("fixture category matrix is incomplete")
    for placement in range(1, 8):
        selected = [
            fixture
            for fixture in fixtures
            if fixture.placement_number == placement
        ]
        if {fixture.information.player for fixture in selected} != set(
            EnginePlayer
        ):
            raise SignalMeasurementError(
                "each placement must contain both acting roles"
            )
        if {fixture.information.dealer for fixture in selected} != set(
            EnginePlayer
        ):
            raise SignalMeasurementError(
                "each placement must contain both dealer assignments"
            )


def aggregate_result_to_groups(
    information: SearchInformationState,
    result: _SearchResult,
) -> AggregatedRootSignal:
    """Project concrete baseline evidence into authoritative action groups."""

    groups = strategic_action_groups(information, True)
    group_visits = []
    group_values = []
    selected_group = None
    for group_index, group in enumerate(groups):
        visits = sum(
            result.action_visits[index]
            for index in group.member_action_indices
        )
        weighted_value = sum(
            result.action_visits[index] * float(result.mean_action_values[index])
            for index in group.member_action_indices
            if result.action_visits[index]
            and result.mean_action_values[index] is not None
        )
        if visits < 1:
            raise SignalMeasurementError(
                "every strategic group must receive initial coverage"
            )
        group_visits.append(visits)
        group_values.append(weighted_value / visits)
        if result.selected_action_index in group.member_action_indices:
            selected_group = group_index
    if sum(group_visits) != result.simulation_count:
        raise SignalMeasurementError(
            "aggregated group visits differ from the search budget"
        )
    if selected_group is None:
        raise SignalMeasurementError(
            "selected concrete action has no authoritative group"
        )
    return AggregatedRootSignal(
        groups=groups,
        group_visits=tuple(group_visits),
        group_mean_values=tuple(group_values),
        selected_group_index=selected_group,
    )


def _entropy(visits: tuple[int, ...]) -> float:
    total = sum(visits)
    return -sum(
        (count / total) * math.log(count / total)
        for count in visits
        if count
    )


def _normalized_entropy(visits: tuple[int, ...]) -> float:
    return _entropy(visits) / math.log(len(visits)) if len(visits) > 1 else 0.0


def _average_ranks(values: tuple[float, ...]) -> tuple[float, ...]:
    ordered = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(ordered):
        end = start + 1
        while (
            end < len(ordered)
            and values[ordered[end]] == values[ordered[start]]
        ):
            end += 1
        average = (start + end - 1) / 2.0
        for ordinal in range(start, end):
            ranks[ordered[ordinal]] = average
        start = end
    return tuple(ranks)


def _rank_correlation(
    first: tuple[float, ...],
    second: tuple[float, ...],
) -> float:
    if len(first) != len(second) or not first:
        raise SignalMeasurementError("rank vectors must have equal dimensions")
    first_ranks = _average_ranks(first)
    second_ranks = _average_ranks(second)
    first_mean = statistics.fmean(first_ranks)
    second_mean = statistics.fmean(second_ranks)
    numerator = sum(
        (left - first_mean) * (right - second_mean)
        for left, right in zip(first_ranks, second_ranks, strict=True)
    )
    first_scale = sum((value - first_mean) ** 2 for value in first_ranks)
    second_scale = sum((value - second_mean) ** 2 for value in second_ranks)
    if first_scale == 0.0 or second_scale == 0.0:
        return 1.0 if first_ranks == second_ranks else 0.0
    return numerator / math.sqrt(first_scale * second_scale)


def _mean_rank_agreement(
    value_vectors: tuple[tuple[float, ...], ...],
) -> float:
    correlations = [
        _rank_correlation(first, second)
        for first, second in combinations(value_vectors, 2)
    ]
    return statistics.fmean(correlations) if correlations else 1.0


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    index = max(
        0,
        min(len(ordered) - 1, math.ceil(probability * len(ordered)) - 1),
    )
    return ordered[index]


def _request_seed(fixture: SignalFixture, seed_index: int) -> bytes:
    return derive_seed(
        SIGNAL_REQUEST_NAMESPACE,
        fixture.fixture_id,
        str(seed_index),
    )


def _run_record(
    fixture: SignalFixture,
    seed_index: int,
    *,
    mode: str,
    config: StrategicSearchConfig,
) -> dict[str, object]:
    result = StrategicInformationSetSearch(config).search(
        fixture.information,
        _request_seed(fixture, seed_index),
    )
    aggregated = aggregate_result_to_groups(fixture.information, result)
    selected_group = aggregated.groups[aggregated.selected_group_index]
    concrete_grid = POLICY_GRID_INDICES[
        result.selected_action_index % len(POLICY_GRID_INDICES)
    ]
    return {
        "fixture_id": fixture.fixture_id,
        "placement_number": fixture.placement_number,
        "category": fixture.category,
        "pattern": fixture.pattern,
        "actor": fixture.information.player.value,
        "dealer": fixture.information.dealer.value,
        "seed_index": seed_index,
        "mode": mode,
        "information_state_fingerprint": (
            result.information_state_fingerprint
        ),
        "search_config_digest": result.config_digest,
        "concrete_legal_action_count": len(
            [
                value
                for row in fixture.information.legal_mask
                for value in row
                if value
            ]
        ),
        "strategic_action_group_count": len(aggregated.groups),
        "group_representative_actions": [
            group.representative_action_index for group in aggregated.groups
        ],
        "group_member_actions": [
            list(group.member_action_indices) for group in aggregated.groups
        ],
        "group_visits": list(aggregated.group_visits),
        "group_mean_values": list(aggregated.group_mean_values),
        "action_visits": list(result.action_visits),
        "mean_action_values": list(result.mean_action_values),
        "selected_group_index": aggregated.selected_group_index,
        "selected_representative_action": (
            selected_group.representative_action_index
        ),
        "selected_concrete_action": result.selected_action_index,
        "selected_concrete_grid_index": concrete_grid,
        "selected_group_grid_members": list(
            selected_group.member_grid_indices
        ),
        "response_requests": result.response_request_count,
        "unique_response_evaluations": (
            result.unique_response_evaluation_count
        ),
        "response_cache_hits": result.response_cache_hit_count,
        "candidate_actions_evaluated": (
            result.response_candidate_action_count
        ),
        "terminal_evaluations": result.response_terminal_evaluation_count,
        "total_terminal_evaluations": (
            result.total_terminal_evaluation_count
        ),
        "information_set_count": result.information_set_count,
        "deterministic_diagnostics_digest": (
            deterministic_result_digest(result)
        ),
        "latency_seconds": result.elapsed_seconds,
        "peak_memory_bytes": result.peak_resident_memory_bytes,
    }


def deterministic_result_digest(result: _SearchResult) -> str:
    """Hash all reproducible search evidence, excluding timing and RSS."""

    principal = result.principal_continuation
    payload = {
        "information_state_fingerprint": (
            result.information_state_fingerprint
        ),
        "config_digest": result.config_digest,
        "selected_action_index": result.selected_action_index,
        "action_visits": result.action_visits,
        "mean_action_values": result.mean_action_values,
        "simulation_count": result.simulation_count,
        "information_set_count": result.information_set_count,
        "principal_continuation": (
            None
            if principal is None
            else {
                "root_action_index": principal.root_action_index,
                "terminal_value": principal.terminal_value,
                "simulation_index": principal.simulation_index,
                "steps": [
                    {
                        "actor": step.actor,
                        "action_index": step.action_index,
                        "card_id": step.card_id,
                        "grid_index": step.grid_index,
                        "forced": step.forced,
                    }
                    for step in principal.steps
                ],
            }
        ),
        "response_request_count": result.response_request_count,
        "unique_response_evaluation_count": (
            result.unique_response_evaluation_count
        ),
        "response_cache_hit_count": result.response_cache_hit_count,
        "response_candidate_action_count": (
            result.response_candidate_action_count
        ),
        "response_terminal_evaluation_count": (
            result.response_terminal_evaluation_count
        ),
        "total_terminal_evaluation_count": (
            result.total_terminal_evaluation_count
        ),
        "selected_representative_action_index": (
            result.selected_representative_action_index
        ),
        "group_diagnostics": [
            {
                "hand_slot": diagnostic.group.hand_slot,
                "representative_action_index": (
                    diagnostic.group.representative_action_index
                ),
                "representative_grid_index": (
                    diagnostic.group.representative_grid_index
                ),
                "member_action_indices": (
                    diagnostic.group.member_action_indices
                ),
                "member_grid_indices": diagnostic.group.member_grid_indices,
                "visits": diagnostic.visits,
                "mean_value": diagnostic.mean_value,
            }
            for diagnostic in result.group_diagnostics
        ],
    }
    encoded = json.dumps(
        payload,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _configure_measurement_worker() -> None:
    import torch

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def _run_fixture_records(
    fixture: SignalFixture,
) -> list[dict[str, object]]:
    configs = measurement_configs()
    return [
        _run_record(
            fixture,
            seed_index,
            mode=mode,
            config=config,
        )
        for seed_index in range(REQUEST_SEED_COUNT)
        for mode, config in configs.items()
    ]


def measurement_configs() -> dict[str, StrategicSearchConfig]:
    return {
        "baseline": StrategicSearchConfig(
            outer_simulation_budget=32,
            response_completions_per_action=4,
            destination_symmetry_enabled=False,
        ),
        "reduced": StrategicSearchConfig(
            outer_simulation_budget=32,
            response_completions_per_action=4,
            destination_symmetry_enabled=True,
        ),
    }


def _config_data(config: StrategicSearchConfig) -> dict[str, object]:
    return {
        "search_schema": STRATEGIC_SEARCH_SCHEMA_VERSION,
        "search_config_digest": config.digest,
        "outer_simulation_budget": config.outer_simulation_budget,
        "outer_exploration_constant": (
            float(config.outer_exploration_constant)
        ),
        "response_schema": GREEDY_RESPONSE_SCHEMA_VERSION,
        "response_config_digest": config.response_config.digest,
        "response_completions_per_action": (
            config.response_completions_per_action
        ),
        "destination_symmetry_enabled": (
            config.destination_symmetry_enabled
        ),
        "terminal_value": "exact-normalized-round-differential-v1",
        "root_selection": STRATEGIC_SELECTION_PROFILE,
        "response_selection": GREEDY_RESPONSE_SELECTION_PROFILE,
        "response_continuation": GREEDY_RESPONSE_CONTINUATION_PROFILE,
    }


def _state_metric(records: list[dict[str, object]], budget: int) -> dict[str, object]:
    selected = Counter(
        int(record["selected_representative_action"])
        for record in records
    )
    visit_vectors = tuple(
        tuple(int(value) for value in record["group_visits"])
        for record in records
    )
    value_vectors = tuple(
        tuple(float(value) for value in record["group_mean_values"])
        for record in records
    )
    maximum_shares = []
    margins = []
    entropies = []
    for visits in visit_vectors:
        ordered = sorted(visits, reverse=True)
        maximum_shares.append(ordered[0] / budget)
        margins.append(
            (ordered[0] - ordered[1]) / budget
            if len(ordered) > 1
            else 1.0
        )
        entropies.append(_normalized_entropy(visits))
    mode = str(records[0]["mode"])
    concrete_count = int(records[0]["concrete_legal_action_count"])
    group_count = int(records[0]["strategic_action_group_count"])
    initial_coverage = (
        concrete_count / budget
        if mode == "baseline"
        else group_count / budget
    )
    return {
        "fixture_id": records[0]["fixture_id"],
        "mode": mode,
        "concrete_legal_action_count": concrete_count,
        "strategic_action_group_count": group_count,
        "initial_action_coverage_share": initial_coverage,
        "normalized_visit_entropy": statistics.fmean(entropies),
        "maximum_group_visit_share": statistics.fmean(maximum_shares),
        "top_two_group_visit_margin": statistics.fmean(margins),
        "selected_group_agreement": max(selected.values()) / len(records),
        "mean_action_value_rank_agreement": _mean_rank_agreement(
            value_vectors
        ),
    }


def _pair_frequencies(records: list[dict[str, object]]) -> dict[str, object]:
    counts: dict[str, Counter[int]] = defaultdict(Counter)
    for record in records:
        members = tuple(
            int(value) for value in record["selected_group_grid_members"]
        )
        if len(members) != 2:
            continue
        key = "-".join(str(index + 1) for index in members)
        counts[key][int(record["selected_concrete_grid_index"]) + 1] += 1
    return {
        key: {
            "count": sum(counter.values()),
            "positions": {
                str(position): count for position, count in sorted(counter.items())
            },
            "frequencies": {
                str(position): count / sum(counter.values())
                for position, count in sorted(counter.items())
            },
        }
        for key, counter in sorted(counts.items())
    }


def _summarize(
    fixtures: tuple[SignalFixture, ...],
    records: list[dict[str, object]],
    wall_seconds: float,
    workers: int,
) -> dict[str, object]:
    state_records: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for record in records:
        state_records[
            (str(record["fixture_id"]), str(record["mode"]))
        ].append(record)
    state_metrics = [
        _state_metric(values, 32)
        for _key, values in sorted(state_records.items())
    ]
    categories = []
    category_names = dict.fromkeys(
        fixture.category for fixture in fixtures
    )
    for category in category_names:
        selected_records = [
            record for record in records if record["category"] == category
        ]
        selected_states = [
            metric
            for metric in state_metrics
            if any(
                record["fixture_id"] == metric["fixture_id"]
                and record["category"] == category
                for record in selected_records
            )
        ]
        by_mode = {}
        for mode in ("baseline", "reduced"):
            mode_records = [
                record
                for record in selected_records
                if record["mode"] == mode
            ]
            mode_states = [
                metric for metric in selected_states if metric["mode"] == mode
            ]
            latencies = [
                float(record["latency_seconds"]) for record in mode_records
            ]
            by_mode[mode] = {
                "state_count": len(mode_states),
                **{
                    key: statistics.fmean(
                        float(state[key]) for state in mode_states
                    )
                    for key in (
                        "concrete_legal_action_count",
                        "strategic_action_group_count",
                        "initial_action_coverage_share",
                        "normalized_visit_entropy",
                        "maximum_group_visit_share",
                        "top_two_group_visit_margin",
                        "selected_group_agreement",
                        "mean_action_value_rank_agreement",
                    )
                },
                "response_requests": statistics.fmean(
                    int(record["response_requests"])
                    for record in mode_records
                ),
                "response_cache_hits": statistics.fmean(
                    int(record["response_cache_hits"])
                    for record in mode_records
                ),
                "candidate_actions_evaluated": statistics.fmean(
                    int(record["candidate_actions_evaluated"])
                    for record in mode_records
                ),
                "terminal_evaluations": statistics.fmean(
                    int(record["terminal_evaluations"])
                    for record in mode_records
                ),
                "latency_mean_seconds": statistics.fmean(latencies),
                "latency_p95_seconds": _percentile(latencies, 0.95),
                "peak_memory_bytes": max(
                    int(record["peak_memory_bytes"])
                    for record in mode_records
                ),
                "selected_pair_frequencies": _pair_frequencies(mode_records),
            }
        paired = []
        indexed = {
            (
                str(record["fixture_id"]),
                int(record["seed_index"]),
                str(record["mode"]),
            ): record
            for record in selected_records
        }
        for fixture_id, seed_index in {
            (key[0], key[1]) for key in indexed
        }:
            paired.append(
                indexed[(fixture_id, seed_index, "baseline")][
                    "selected_representative_action"
                ]
                == indexed[(fixture_id, seed_index, "reduced")][
                    "selected_representative_action"
                ]
            )
        categories.append(
            {
                "category": category,
                "placement_number": int(
                    selected_records[0]["placement_number"]
                ),
                "modes": by_mode,
                "cross_mode_selected_group_agreement": (
                    sum(paired) / len(paired)
                ),
            }
        )
    return {
        "schema_version": SIGNAL_MEASUREMENT_SCHEMA_VERSION,
        "request_seed_namespace": SIGNAL_REQUEST_NAMESPACE,
        "request_seeds_per_state": REQUEST_SEED_COUNT,
        "fixture_count": len(fixtures),
        "search_count": len(records),
        "wall_seconds": wall_seconds,
        "workers": workers,
        "profiles": {
            "baseline": BASELINE_PROFILE,
            "reduced": REDUCED_PROFILE,
        },
        "configurations": {
            mode: _config_data(config)
            for mode, config in measurement_configs().items()
        },
        "categories": categories,
        "states": state_metrics,
        "runs": records,
    }


def run_signal_measurement(
    progress: Callable[[str], None] | None = None,
    *,
    workers: int = 4,
) -> dict[str, object]:
    if type(workers) is not int or workers < 1:
        raise SignalMeasurementError("worker count must be positive")
    fixtures = build_signal_fixtures()
    records: list[dict[str, object]] = []
    started = time.perf_counter()
    if workers == 1:
        fixture_results = map(_run_fixture_records, fixtures)
        executor = None
    else:
        executor = ProcessPoolExecutor(
            max_workers=workers,
            initializer=_configure_measurement_worker,
        )
        fixture_results = executor.map(_run_fixture_records, fixtures)
    try:
        for fixture_index, (fixture, fixture_records) in enumerate(
            zip(fixtures, fixture_results, strict=True),
            start=1,
        ):
            records.extend(fixture_records)
            if progress is not None:
                progress(
                    f"completed {fixture_index}/{len(fixtures)} "
                    f"{fixture.fixture_id}"
                )
    finally:
        if executor is not None:
            executor.shutdown(cancel_futures=True)
    return _summarize(
        fixtures,
        records,
        time.perf_counter() - started,
        workers,
    )


def _percentage(value: object) -> str:
    return f"{100.0 * float(value):.1f}%"


def render_markdown(result: dict[str, object]) -> str:
    categories = result["categories"]
    category_by_name = {
        category["category"]: category for category in categories
    }
    opening = category_by_name["placement-1-opening"]["modes"]
    adjacent = category_by_name["placement-2-adjacent"]["modes"]
    line = category_by_name["placement-3-line"]["modes"]
    lines = [
        "# Teacher v2 symmetry signal measurement",
        "",
        "## Method",
        "",
        (
            f"{result['fixture_count']} deterministic role-balanced states "
            f"were measured with {result['request_seeds_per_state']} request "
            "seeds under the frozen 32x4 baseline and symmetry-reduced 32x4 "
            "Teacher v2."
        ),
        "",
        (
            "Baseline concrete visits and visit-weighted action values were "
            "projected into the authoritative destination groups before "
            "comparison. Initial coverage remains the cost paid by each "
            "planner: concrete actions for baseline and strategic groups for "
            "the reduced planner."
        ),
        "",
        "## Findings",
        "",
        (
            "The reduction halves initial root coverage at placement 1 and "
            "the conditional placement-3 line states, and reduces placement-2 "
            "coverage from "
            f"{_percentage(adjacent['baseline']['initial_action_coverage_share'])} "
            "to "
            f"{_percentage(adjacent['reduced']['initial_action_coverage_share'])}."
        ),
        "",
        (
            "That saved work does not produce a clearly concentrated "
            "32-visit target. Reduced-search normalized entropy remains "
            f"{_percentage(opening['reduced']['normalized_visit_entropy'])} "
            "at placement 1, "
            f"{_percentage(adjacent['reduced']['normalized_visit_entropy'])} "
            "at placement 2, and "
            f"{_percentage(line['reduced']['normalized_visit_entropy'])} "
            "in the conditional placement-3 line states. Mean top-two margins "
            "remain at or below "
            f"{_percentage(max(opening['reduced']['top_two_group_visit_margin'], adjacent['reduced']['top_two_group_visit_margin'], line['reduced']['top_two_group_visit_margin']))}."
        ),
        "",
        (
            "Selected-group agreement improves in those three categories, "
            "while action-value rank agreement improves at placement 1 and "
            "the conditional placement-3 line states but not placement 2. "
            "The signal result is therefore mixed: symmetry reduces redundant "
            "computation and modestly improves some stability measures, but "
            "does not make the 32-visit distribution a strong policy target."
        ),
        "",
        (
            "Mean latency falls from "
            f"{opening['baseline']['latency_mean_seconds']:.3f}s to "
            f"{opening['reduced']['latency_mean_seconds']:.3f}s at placement 1, "
            f"from {adjacent['baseline']['latency_mean_seconds']:.3f}s to "
            f"{adjacent['reduced']['latency_mean_seconds']:.3f}s at placement 2, "
            f"and from {line['baseline']['latency_mean_seconds']:.3f}s to "
            f"{line['reduced']['latency_mean_seconds']:.3f}s in conditional "
            "placement-3 line states."
        ),
        "",
        (
            "Paired-position frequencies below are descriptive; five request "
            "seeds per state are sufficient for signal comparison, not a "
            "statistical test of coin fairness."
        ),
        "",
        "## Root signal by placement",
        "",
        (
            "| Placement | State | Mode | Concrete | Groups | Coverage | "
            "Entropy | Max share | Top-two | Selected agreement | "
            "Value-rank agreement |"
        ),
        "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for category in categories:
        for mode in ("baseline", "reduced"):
            values = category["modes"][mode]
            lines.append(
                "| {placement} | {category} | {mode} | {concrete:.1f} | "
                "{groups:.1f} | {coverage} | {entropy} | {maximum} | "
                "{margin} | {selected} | {rank:.3f} |".format(
                    placement=category["placement_number"],
                    category=category["category"],
                    mode=mode,
                    concrete=values["concrete_legal_action_count"],
                    groups=values["strategic_action_group_count"],
                    coverage=_percentage(
                        values["initial_action_coverage_share"]
                    ),
                    entropy=_percentage(
                        values["normalized_visit_entropy"]
                    ),
                    maximum=_percentage(
                        values["maximum_group_visit_share"]
                    ),
                    margin=_percentage(
                        values["top_two_group_visit_margin"]
                    ),
                    selected=_percentage(
                        values["selected_group_agreement"]
                    ),
                    rank=values["mean_action_value_rank_agreement"],
                )
            )
    lines.extend(
        [
            "",
            "Value-rank agreement is mean pairwise Spearman correlation across "
            "the five request seeds.",
            "",
            "## Search work by placement",
            "",
            (
                "| Placement | State | Mode | Response requests | Cache hits | "
                "Candidates | Terminals | Mean latency | p95 latency | Peak RSS |"
            ),
            "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for category in categories:
        for mode in ("baseline", "reduced"):
            values = category["modes"][mode]
            lines.append(
                "| {placement} | {category} | {mode} | {requests:.1f} | "
                "{hits:.1f} | {candidates:.1f} | {terminals:.1f} | "
                "{mean:.3f}s | {p95:.3f}s | {rss:.1f} MiB |".format(
                    placement=category["placement_number"],
                    category=category["category"],
                    mode=mode,
                    requests=values["response_requests"],
                    hits=values["response_cache_hits"],
                    candidates=values["candidate_actions_evaluated"],
                    terminals=values["terminal_evaluations"],
                    mean=values["latency_mean_seconds"],
                    p95=values["latency_p95_seconds"],
                    rss=values["peak_memory_bytes"] / (1024 * 1024),
                )
            )
    lines.extend(
        [
            "",
            "## Paired-position selection",
            "",
        ]
    )
    for category in categories:
        lines.append(f"### {category['category']}")
        lines.append("")
        for mode in ("baseline", "reduced"):
            frequencies = category["modes"][mode][
                "selected_pair_frequencies"
            ]
            lines.append(f"- {mode}: `{json.dumps(frequencies, sort_keys=True)}`")
        lines.append(
            "- Cross-mode selected-group agreement: "
            f"{_percentage(category['cross_mode_selected_group_agreement'])}"
        )
        lines.append("")
    lines.extend(
        [
            "## Runtime",
            "",
            f"- Searches: {result['search_count']}",
            f"- Worker processes: {result['workers']}",
            f"- Wall time: {float(result['wall_seconds']):.1f} seconds",
            "",
        ]
    )
    return "\n".join(lines)


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json",
        type=Path,
        default=Path(".local/teacher-v2-symmetry-signal.json"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/history/teacher-v2/teacher-v2-symmetry-signal.md"),
    )
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    result = run_signal_measurement(
        lambda message: print(message, flush=True),
        workers=args.workers,
    )
    _atomic_text(
        args.json,
        json.dumps(result, indent=2, sort_keys=True) + "\n",
    )
    _atomic_text(args.report, render_markdown(result))
    print(json.dumps({"json": str(args.json), "report": str(args.report)}))
    return 0


__all__ = (
    "AggregatedRootSignal",
    "BASELINE_PROFILE",
    "REDUCED_PROFILE",
    "REQUEST_SEED_COUNT",
    "SIGNAL_MEASUREMENT_SCHEMA_VERSION",
    "SIGNAL_REQUEST_NAMESPACE",
    "SignalFixture",
    "SignalMeasurementError",
    "aggregate_result_to_groups",
    "build_signal_fixtures",
    "deterministic_result_digest",
    "main",
    "measurement_configs",
    "render_markdown",
    "run_signal_measurement",
)


if __name__ == "__main__":
    raise SystemExit(main())

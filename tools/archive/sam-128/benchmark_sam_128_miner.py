"""Resumable benchmark for the exact Sam-128 branched dataset miner."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import resource
import statistics
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

from dracula.engine import (
    EnginePlayer,
    apply_move,
    create_game,
    legal_moves,
)
from dracula.search import (
    SamTeacherInformationSetSearch,
    SamTeacherSearchConfig,
    derive_sam_teacher_request_seed,
    information_state_fingerprint,
    information_state_from_engine,
    sam_teacher_action_groups,
)
from dracula.search.signal_measurement import (
    SignalFixture,
    build_signal_fixtures,
)
from dracula.sam_miner import (
    SAM_MINER_CONTROLLER_PROFILE,
    SAM_TEACHER_SEARCH_SCHEMA_VERSION,
    DeterministicSamTeacherStub,
    SamMinerConfig,
    SamMinerError,
    SamMinerTeacherSelection,
    fixture_schedule,
    initialize_corpus,
    mine_branch_prefix,
    mine_corpus,
    resolve_teacher,
    validate_sam128_result,
)
import dracula.sam_miner as miner


BENCHMARK_SCHEMA_VERSION = "dracula-sam-128-miner-benchmark-v1"
BENCHMARK_ROOT_SEED = "sam-128-miner-benchmark-2026-07-24-v1"
_SWAP_PATTERN = re.compile(r"used = ([0-9.]+)([MG])")
_MEMORY_FREE_PATTERN = re.compile(r"free percentage: ([0-9]+)%")


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


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
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ).encode("utf-8") + b"\n"
    json.loads(encoded)
    _atomic_bytes(path, encoded)


def _write_sealed(path: Path, content: object) -> dict[str, object]:
    document = {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "content_digest": _digest(content),
        "content": content,
    }
    _atomic_json(path, document)
    return document


def _load_sealed(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(value, dict)
        or set(value) != {"schema_version", "content_digest", "content"}
        or value["schema_version"] != BENCHMARK_SCHEMA_VERSION
        or value["content_digest"] != _digest(value["content"])
    ):
        raise RuntimeError(f"invalid benchmark artifact: {path}")
    return value


def _swap_used_bytes() -> int | None:
    try:
        output = subprocess.run(
            ["sysctl", "-n", "vm.swapusage"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    match = _SWAP_PATTERN.search(output)
    if match is None:
        return None
    scale = 1024**2 if match.group(2) == "M" else 1024**3
    return int(float(match.group(1)) * scale)


def _memory_free_percent() -> int | None:
    try:
        output = subprocess.run(
            ["memory_pressure", "-Q"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    match = _MEMORY_FREE_PATTERN.search(output)
    return int(match.group(1)) if match is not None else None


def _thermal_status() -> str:
    try:
        return subprocess.run(
            ["pmset", "-g", "therm"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unavailable"


def _process_tree_sample(root_pid: int) -> tuple[float, int, int]:
    try:
        output = subprocess.run(
            ["ps", "-axo", "pid=,ppid=,%cpu=,rss="],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return 0.0, 0, 0
    rows: list[tuple[int, int, float, int]] = []
    for line in output.splitlines():
        fields = line.split()
        if len(fields) != 4:
            continue
        try:
            rows.append(
                (
                    int(fields[0]),
                    int(fields[1]),
                    float(fields[2]),
                    int(fields[3]) * 1024,
                )
            )
        except ValueError:
            continue
    members = {root_pid}
    changed = True
    while changed:
        changed = False
        for pid, parent, _, _ in rows:
            if parent in members and pid not in members:
                members.add(pid)
                changed = True
    selected = [row for row in rows if row[0] in members]
    return (
        sum(row[2] for row in selected),
        sum(row[3] for row in selected),
        max(
            (row[3] for row in selected if row[0] != root_pid),
            default=0,
        ),
    )


def _peak_rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def _allocated_disk_bytes(path: Path) -> int:
    return sum(
        item.stat().st_blocks * 512
        for item in path.rglob("*")
        if item.is_file()
    )


def _percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("percentile requires values")
    ordered = sorted(values)
    index = max(
        0,
        min(len(ordered) - 1, math.ceil(probability * len(ordered)) - 1),
    )
    return ordered[index]


def _result_digest(result: object) -> str:
    diagnostics = getattr(result, "group_diagnostics")
    principal = getattr(result, "principal_continuation")
    content = {
        "information_state_fingerprint": getattr(
            result, "information_state_fingerprint"
        ),
        "config_digest": getattr(result, "config_digest"),
        "selected_action_index": getattr(result, "selected_action_index"),
        "selected_representative_action_index": getattr(
            result, "selected_representative_action_index"
        ),
        "action_visits": getattr(result, "action_visits"),
        "mean_action_values": getattr(result, "mean_action_values"),
        "simulation_count": getattr(result, "simulation_count"),
        "information_set_count": getattr(result, "information_set_count"),
        "response_request_count": getattr(result, "response_request_count"),
        "unique_response_search_count": getattr(
            result, "unique_response_search_count"
        ),
        "response_cache_hit_count": getattr(
            result, "response_cache_hit_count"
        ),
        "response_simulation_count": getattr(
            result, "response_simulation_count"
        ),
        "response_information_set_count": getattr(
            result, "response_information_set_count"
        ),
        "total_terminal_evaluation_count": getattr(
            result, "total_terminal_evaluation_count"
        ),
        "group_diagnostics": [
            {
                "representative": item.group.representative_action_index,
                "members": item.group.member_action_indices,
                "visits": item.visits,
                "mean_value": item.mean_value,
            }
            for item in diagnostics
        ],
        "principal": (
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
    }
    return _digest(content)


@dataclass(slots=True)
class InstrumentedSam128:
    schema_version: str = SAM_TEACHER_SEARCH_SCHEMA_VERSION
    configuration_digest: str = SamTeacherSearchConfig().digest
    controller_profile: str = SAM_MINER_CONTROLLER_PROFILE
    planner: SamTeacherInformationSetSearch = field(init=False)
    records: list[dict[str, object]] = field(init=False)

    def __post_init__(self) -> None:
        self.planner = SamTeacherInformationSetSearch(
            SamTeacherSearchConfig()
        )
        self.records: list[dict[str, object]] = []

    def select(
        self,
        information,
        should_stop: Callable[[], bool] | None = None,
    ) -> SamMinerTeacherSelection:
        request = derive_sam_teacher_request_seed(
            information, self.configuration_digest
        )
        started = time.perf_counter()
        result = self.planner.search(information, request, should_stop)
        wall = time.perf_counter() - started
        self.records.append(
            _search_record("miner-query", information, result, wall)
        )
        return validate_sam128_result(
            information, SamTeacherSearchConfig(), result
        )


def _search_record(
    fixture_id: str,
    information,
    result,
    wall_seconds: float,
) -> dict[str, object]:
    return {
        "fixture_id": fixture_id,
        "placement_number": information.turn_number,
        "actor": information.player.value,
        "dealer": information.dealer.value,
        "information_state_fingerprint": information_state_fingerprint(
            information
        ),
        "strategic_group_count": len(sam_teacher_action_groups(information)),
        "selected_action_index": result.selected_action_index,
        "selected_representative_action_index": (
            result.selected_representative_action_index
        ),
        "teacher_seconds": result.elapsed_seconds,
        "wall_seconds": wall_seconds,
        "outer_simulations": result.simulation_count,
        "response_requests": result.response_request_count,
        "unique_response_searches": result.unique_response_search_count,
        "response_cache_hits": result.response_cache_hit_count,
        "response_simulations": result.response_simulation_count,
        "terminal_evaluations": result.total_terminal_evaluation_count,
        "information_sets": result.information_set_count,
        "response_information_sets": result.response_information_set_count,
        "result_digest": _result_digest(result),
        "peak_rss_bytes": _peak_rss_bytes(),
    }


def _fixture_map() -> dict[str, SignalFixture]:
    return {fixture.fixture_id: fixture for fixture in build_signal_fixtures()}


def _selected_fixtures() -> tuple[SignalFixture, ...]:
    fixtures = build_signal_fixtures()
    selected: list[SignalFixture] = []
    for placement in range(1, 8):
        candidates = [
            fixture
            for fixture in fixtures
            if fixture.placement_number == placement
            and (
                placement < 3
                or "no-symmetry" in fixture.category
            )
        ]
        selected.append(sorted(candidates, key=lambda item: item.fixture_id)[0])
    symmetry_patterns = (
        "center-only",
        "center-plus-2",
        "center-plus-8",
        "center-plus-4",
        "center-plus-6",
        "horizontal-line",
        "vertical-line",
    )
    for pattern in symmetry_patterns:
        candidate = sorted(
            (
                fixture
                for fixture in fixtures
                if fixture.pattern == pattern
            ),
            key=lambda item: item.fixture_id,
        )[0]
        if candidate.fixture_id not in {
            fixture.fixture_id for fixture in selected
        }:
            selected.append(candidate)
    return tuple(selected)


def _configure_worker() -> None:
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = "1"
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    if torch.get_num_threads() != 1 or torch.get_num_interop_threads() != 1:
        raise RuntimeError("benchmark workers require one PyTorch thread")


def _run_search_fixture(fixture_id: str) -> dict[str, object]:
    fixture = _fixture_map()[fixture_id]
    config = SamTeacherSearchConfig()
    planner = SamTeacherInformationSetSearch(config)
    request = derive_sam_teacher_request_seed(
        fixture.information, config.digest
    )
    started = time.perf_counter()
    result = planner.search(fixture.information, request)
    return _search_record(
        fixture.fixture_id,
        fixture.information,
        result,
        time.perf_counter() - started,
    )


def run_teacher_measurements(output: Path) -> dict[str, object]:
    destination = output / "teacher-measurements.json"
    if destination.exists():
        return _load_sealed(destination)
    records: list[dict[str, object]] = []
    for fixture in _selected_fixtures():
        query_path = (
            output / "teacher-queries" / f"{fixture.fixture_id}.json"
        )
        if query_path.exists():
            record = _load_sealed(query_path)["content"]
        else:
            print(
                f"teacher fixture={fixture.fixture_id} "
                f"placement={fixture.placement_number}",
                flush=True,
            )
            record = _run_search_fixture(fixture.fixture_id)
            _write_sealed(query_path, record)
        records.append(record)
    repeat_records = []
    placement_fixtures = {
        placement: next(
            fixture
            for fixture in _selected_fixtures()
            if fixture.placement_number == placement
        )
        for placement in range(1, 8)
    }
    for placement, fixture in placement_fixtures.items():
        repeat_path = (
            output
            / "teacher-queries"
            / f"{fixture.fixture_id}-repeat.json"
        )
        if repeat_path.exists():
            repeated = _load_sealed(repeat_path)["content"]
        else:
            print(
                f"repro fixture={fixture.fixture_id} placement={placement}",
                flush=True,
            )
            repeated = _run_search_fixture(fixture.fixture_id)
        original = next(
            record
            for record in records
            if record["fixture_id"] == fixture.fixture_id
        )
        repeated["matches_original"] = (
            repeated["result_digest"] == original["result_digest"]
        )
        if not repeated["matches_original"]:
            raise RuntimeError("Sam-128 repeated fixture changed result")
        repeat_records.append(repeated)
        if not repeat_path.exists():
            _write_sealed(repeat_path, repeated)

    cache_config = miner._production_config(
        run_id="sam-128-benchmark-cache",
        root_seed=BENCHMARK_ROOT_SEED,
        output_directory=str(output / "cache-probe"),
        training_decks=1,
        validation_decks=0,
        test_decks=0,
        child_count=4,
        workers=1,
    )
    if (cache_config.output_path / "resolved-config.json").exists():
        cache_config = miner.load_config(cache_config.output_path)
    else:
        initialize_corpus(cache_config)
    cache_fixture = placement_fixtures[4]
    groups = sam_teacher_action_groups(cache_fixture.information)
    teacher = InstrumentedSam128()
    cold_started = time.perf_counter()
    cold = resolve_teacher(
        cache_fixture.information,
        groups,
        teacher,
        cache_config,
    )
    cold_seconds = time.perf_counter() - cold_started
    warm_started = time.perf_counter()
    warm = resolve_teacher(
        cache_fixture.information,
        groups,
        teacher,
        cache_config,
    )
    warm_seconds = time.perf_counter() - warm_started
    if (
        cold.cache_hit
        or not warm.cache_hit
        or cold.selected_group_index != warm.selected_group_index
    ):
        raise RuntimeError("cold/warm cache probe is invalid")
    content = {
        "search_config": asdict(SamTeacherSearchConfig()),
        "search_config_digest": SamTeacherSearchConfig().digest,
        "records": records,
        "repeat_records": repeat_records,
        "cache_probe": {
            "fixture_id": cache_fixture.fixture_id,
            "cold_seconds": cold_seconds,
            "warm_seconds": warm_seconds,
            "cache_misses": 1,
            "cache_hits": 1,
            "teacher_queries": len(teacher.records),
            "teacher_record": teacher.records[0],
        },
    }
    return _write_sealed(destination, content)


def run_prefix_measurements(output: Path) -> dict[str, object]:
    destination = output / "prefix-measurements.json"
    if destination.exists():
        return _load_sealed(destination)
    prefix_root = output / "prefix-run"
    config = miner._production_config(
        run_id="sam-128-prefix-benchmark",
        root_seed=BENCHMARK_ROOT_SEED,
        output_directory=str(prefix_root),
        training_decks=1,
        validation_decks=0,
        test_decks=0,
        child_count=4,
        workers=1,
    )
    initialize_corpus(config)
    fixture = fixture_schedule(config)[0]
    state = create_game(fixture._engine_seed)
    root_path = miner._root_path_digest(fixture.fixture_id, 1)
    levels: list[dict[str, object]] = []
    for depth in range(1, 5):
        print(f"prefix depth={depth}", flush=True)
        teacher = InstrumentedSam128()
        cache_before = len(tuple((prefix_root / "cache").rglob("*.json")))
        started = time.perf_counter()
        result = mine_branch_prefix(
            state,
            fixture_id=fixture.fixture_id,
            path_digest=root_path,
            maximum_placement=depth,
            teacher=teacher,
            config=config,
        )
        wall = time.perf_counter() - started
        cache_after = len(tuple((prefix_root / "cache").rglob("*.json")))
        expected_examples = sum(4**index for index in range(depth))
        expected_frontier = 4**depth
        if (
            len(result.rows) != expected_examples
            or len(result.frontier_path_digests) != expected_frontier
        ):
            raise RuntimeError("branch-prefix structural counts differ")
        level = {
            "depth": depth,
            "examples": len(result.rows),
            "frontier_branches": len(result.frontier_path_digests),
            "terminal_leaves": 0,
            "wall_seconds": wall,
            "examples_per_hour": len(result.rows) / wall * 3600.0,
            "teacher_queries": len(teacher.records),
            "cache_entries_before": cache_before,
            "cache_entries_after": cache_after,
            "cache_hits": len(result.rows) - len(teacher.records),
            "cache_misses": len(teacher.records),
            "terminal_evaluations": sum(
                int(record["terminal_evaluations"])
                for record in teacher.records
            ),
            "teacher_records": teacher.records,
            "rows_digest": _digest(
                [row.to_dict() for row in result.rows]
            ),
            "frontier_digest": _digest(result.frontier_path_digests),
            "peak_rss_bytes": _peak_rss_bytes(),
        }
        levels.append(level)
        _write_sealed(
            output / "prefix-levels" / f"depth-{depth}.json", level
        )
    return _write_sealed(
        destination,
        {
            "fixture_id": fixture.fixture_id,
            "configuration_digest": config.digest,
            "levels": levels,
        },
    )


def run_structural_measurement(output: Path) -> dict[str, object]:
    destination = output / "structural-measurement.json"
    if destination.exists():
        document = _load_sealed(destination)
        content = document["content"]
        if "allocated_disk_bytes" not in content:
            corpus = output / "structural-stub-corpus"
            content["allocated_disk_bytes"] = _allocated_disk_bytes(corpus)
            content["allocated_disk_bytes_per_row"] = (
                content["allocated_disk_bytes"] / content["examples"]
            )
            return _write_sealed(destination, content)
        return document
    corpus = output / "structural-stub-corpus"
    stub = DeterministicSamTeacherStub()
    config = SamMinerConfig(
        run_id="sam-128-structural-benchmark",
        root_seed=BENCHMARK_ROOT_SEED,
        output_directory=str(corpus),
        training_decks=1,
        validation_decks=0,
        test_decks=0,
        child_count=4,
        workers=1,
        teacher_schema_version=stub.schema_version,
        teacher_configuration_digest=stub.configuration_digest,
        teacher_controller_profile=stub.controller_profile,
    )
    started = time.perf_counter()
    inspection = mine_corpus(config, stub)
    wall = time.perf_counter() - started
    shard_paths = sorted(corpus.glob("decks/*/*/round-*/*.json"))
    shard_paths = [
        path
        for path in shard_paths
        if path.name == "root.json" or path.name.startswith("subtree-")
    ]
    shard_bytes = sum(path.stat().st_size for path in shard_paths)
    content = {
        "configuration_digest": config.digest,
        "inspection": asdict(inspection),
        "wall_seconds": wall,
        "examples_per_hour": inspection.row_count / wall * 3600.0,
        "teacher_queries": inspection.cache_entry_count,
        "cache_hits": inspection.row_count - inspection.cache_entry_count,
        "cache_misses": inspection.cache_entry_count,
        "examples": inspection.row_count,
        "terminal_leaves": inspection.terminal_leaf_count,
        "disk_bytes": inspection.disk_bytes,
        "disk_bytes_per_row": inspection.disk_bytes / inspection.row_count,
        "allocated_disk_bytes": _allocated_disk_bytes(corpus),
        "allocated_disk_bytes_per_row": (
            _allocated_disk_bytes(corpus) / inspection.row_count
        ),
        "sealed_shard_count": len(shard_paths),
        "sealed_shard_bytes": shard_bytes,
        "mean_bytes_per_sealed_shard": shard_bytes / len(shard_paths),
        "peak_rss_bytes": _peak_rss_bytes(),
    }
    return _write_sealed(destination, content)


def _worker_fixture_ids() -> tuple[str, ...]:
    fixtures = build_signal_fixtures()
    return tuple(
        sorted(
            (
                fixture
                for fixture in fixtures
                if fixture.placement_number == placement
                and (
                    placement < 3
                    or "no-symmetry" in fixture.category
                )
            ),
            key=lambda item: item.fixture_id,
        )[0].fixture_id
        for placement in range(2, 8)
    )


def _run_worker_count(
    output: Path,
    workers: int,
) -> dict[str, object]:
    destination = output / "worker-counts" / f"workers-{workers}.json"
    if destination.exists():
        return _load_sealed(destination)
    fixture_ids = _worker_fixture_ids()
    swap_start = _swap_used_bytes()
    swap_peak = swap_start
    minimum_free = _memory_free_percent()
    cpu_samples: list[float] = []
    total_rss_peak = 0
    worker_rss_peak = 0
    thermal_before = _thermal_status()
    started = time.perf_counter()
    completion_times: list[float] = []
    records: list[dict[str, object]] = []
    executor = ProcessPoolExecutor(
        max_workers=workers,
        initializer=_configure_worker,
    )
    futures = {
        executor.submit(_run_search_fixture, fixture_id): fixture_id
        for fixture_id in fixture_ids
    }
    while futures:
        completed, _ = wait(
            tuple(futures),
            timeout=1.0,
            return_when=FIRST_COMPLETED,
        )
        cpu, total_rss, process_rss = _process_tree_sample(os.getpid())
        cpu_samples.append(cpu)
        total_rss_peak = max(total_rss_peak, total_rss)
        worker_rss_peak = max(worker_rss_peak, process_rss)
        swap = _swap_used_bytes()
        if swap is not None:
            swap_peak = max(swap_peak or swap, swap)
        free = _memory_free_percent()
        if free is not None:
            minimum_free = min(
                minimum_free if minimum_free is not None else free,
                free,
            )
        for future in completed:
            fixture_id = futures.pop(future)
            record = future.result()
            if record["fixture_id"] != fixture_id:
                raise RuntimeError("worker returned another fixture")
            records.append(record)
            completion_times.append(time.perf_counter() - started)
            print(
                f"workers={workers} complete={len(records)}/{len(fixture_ids)} "
                f"fixture={fixture_id}",
                flush=True,
            )
    executor.shutdown(wait=True)
    wall = time.perf_counter() - started
    swap_end = _swap_used_bytes()
    records.sort(key=lambda item: fixture_ids.index(str(item["fixture_id"])))
    halfway = max(1, len(completion_times) // 2)
    first_rate = halfway / completion_times[halfway - 1]
    remaining = len(completion_times) - halfway
    last_rate = (
        remaining / (completion_times[-1] - completion_times[halfway - 1])
        if remaining and completion_times[-1] > completion_times[halfway - 1]
        else first_rate
    )
    content = {
        "workers": workers,
        "fixture_ids": fixture_ids,
        "query_count": len(records),
        "wall_seconds": wall,
        "queries_per_hour": len(records) / wall * 3600.0,
        "terminal_evaluations": sum(
            int(record["terminal_evaluations"]) for record in records
        ),
        "mean_aggregate_cpu_percent": (
            statistics.fmean(cpu_samples) if cpu_samples else None
        ),
        "mean_machine_cpu_percent": (
            statistics.fmean(cpu_samples) / 8.0
            if cpu_samples
            else None
        ),
        "peak_worker_rss_bytes": worker_rss_peak,
        "peak_total_rss_bytes": total_rss_peak,
        "swap_start_bytes": swap_start,
        "swap_peak_bytes": swap_peak,
        "swap_end_bytes": swap_end,
        "swap_peak_growth_bytes": (
            max(0, (swap_peak or 0) - swap_start)
            if swap_start is not None and swap_peak is not None
            else None
        ),
        "swap_end_growth_bytes": (
            swap_end - swap_start
            if swap_start is not None and swap_end is not None
            else None
        ),
        "minimum_memory_free_percent": minimum_free,
        "thermal_status_before": thermal_before,
        "thermal_status_after": _thermal_status(),
        # Completion-order rates are retained as workload observations. The
        # fixtures have different placement costs, so the report must not
        # interpret this ratio as an isolated thermal measurement.
        "first_half_completion_rate": first_rate,
        "last_half_completion_rate": last_rate,
        "completion_rate_ratio": last_rate / first_rate,
        "records": records,
    }
    return _write_sealed(destination, content)


def run_worker_measurements(output: Path) -> dict[str, object]:
    destination = output / "worker-measurements.json"
    if destination.exists():
        return _load_sealed(destination)
    documents = [
        _run_worker_count(output, workers)
        for workers in (1, 2, 4, 6)
    ]
    contents = [document["content"] for document in documents]
    expected = contents[0]["fixture_ids"]
    reference = {
        record["fixture_id"]: record["result_digest"]
        for record in contents[0]["records"]
    }
    for content in contents[1:]:
        if content["fixture_ids"] != expected or {
            record["fixture_id"]: record["result_digest"]
            for record in content["records"]
        } != reference:
            raise RuntimeError("worker counts changed deterministic search")
    return _write_sealed(
        destination,
        {
            "fixture_ids": expected,
            "deterministic_result_digests": reference,
            "results": contents,
        },
    )


def _summary(output: Path) -> dict[str, object]:
    teacher = _load_sealed(output / "teacher-measurements.json")["content"]
    prefixes = _load_sealed(output / "prefix-measurements.json")["content"]
    structural = _load_sealed(
        output / "structural-measurement.json"
    )["content"]
    workers = _load_sealed(output / "worker-measurements.json")["content"]
    teacher_records = [
        *teacher["records"],
        *teacher["repeat_records"],
    ]
    latencies = [
        float(record["wall_seconds"]) for record in teacher_records
    ]
    by_placement = {
        str(placement): [
            float(record["wall_seconds"])
            for record in teacher_records
            if int(record["placement_number"]) == placement
        ]
        for placement in range(1, 8)
    }
    best_worker = max(
        workers["results"],
        key=lambda item: float(item["queries_per_hour"]),
    )
    single_worker = next(
        item for item in workers["results"] if item["workers"] == 1
    )
    speedup = float(best_worker["queries_per_hour"]) / float(
        single_worker["queries_per_hour"]
    )
    placement_examples = (1, 4, 16, 64, 256, 1024, 4096)
    weighted_round_seconds = sum(
        placement_examples[placement - 1]
        * statistics.fmean(by_placement[str(placement)])
        for placement in range(1, 8)
    )
    parallel_round_seconds = weighted_round_seconds / speedup
    bytes_per_row = float(structural["disk_bytes_per_row"])
    allocated_bytes_per_row = float(
        structural["allocated_disk_bytes_per_row"]
    )
    repeat_ratios = []
    for repeated in teacher["repeat_records"]:
        original = next(
            record
            for record in teacher["records"]
            if record["fixture_id"] == repeated["fixture_id"]
        )
        repeat_ratios.append(
            float(repeated["wall_seconds"]) / float(original["wall_seconds"])
        )
    worker_projections = []
    for worker_result in workers["results"]:
        worker_speedup = float(worker_result["queries_per_hour"]) / float(
            single_worker["queries_per_hour"]
        )
        projected_round = weighted_round_seconds / worker_speedup
        worker_projections.append(
            {
                "workers": worker_result["workers"],
                "measured_speedup": worker_speedup,
                "examples_per_hour": 5_461 / projected_round * 3600.0,
                "one_million_examples_seconds": (
                    projected_round / 5_461 * 1_000_000
                ),
                "one_round_seconds": projected_round,
                "one_deck_seconds": projected_round * 6,
            }
        )
    content = {
        "benchmark_schema_version": BENCHMARK_SCHEMA_VERSION,
        "root_seed": BENCHMARK_ROOT_SEED,
        "teacher_config_digest": SamTeacherSearchConfig().digest,
        "teacher_latency": {
            "count": len(latencies),
            "mean_seconds": statistics.fmean(latencies),
            "p50_seconds": _percentile(latencies, 0.50),
            "p95_seconds": _percentile(latencies, 0.95),
            "maximum_seconds": max(latencies),
            "by_placement_seconds": {
                placement: {
                    "count": len(values),
                    "mean": statistics.fmean(values),
                    "minimum": min(values),
                    "maximum": max(values),
                }
                for placement, values in by_placement.items()
            },
        },
        "cache_probe": teacher["cache_probe"],
        "prefix_levels": prefixes["levels"],
        "structural": structural,
        "worker_results": workers["results"],
        "selected_projection_workers": best_worker["workers"],
        "measured_worker_speedup": speedup,
        "worker_projections": worker_projections,
        "sustained_repeat_timing": {
            "matched_fixture_ratios": repeat_ratios,
            "median_repeat_to_initial_ratio": statistics.median(
                repeat_ratios
            ),
            "maximum_repeat_to_initial_ratio": max(repeat_ratios),
            "minimum_repeat_to_initial_ratio": min(repeat_ratios),
            "interpretation": (
                "same-fixture deterministic reruns after the initial suite; "
                "used as the thermal-throughput drift observation"
            ),
        },
        "projections": {
            "one_million_examples_seconds": (
                parallel_round_seconds / 5_461 * 1_000_000
            ),
            "one_million_examples_logical_disk_bytes": (
                1_000_000 * bytes_per_row
            ),
            "one_million_examples_allocated_disk_bytes": (
                1_000_000 * allocated_bytes_per_row
            ),
            "one_round_seconds": parallel_round_seconds,
            "one_deck_seconds": parallel_round_seconds * 6,
            "one_round_logical_disk_bytes": 5_461 * bytes_per_row,
            "one_round_allocated_disk_bytes": (
                5_461 * allocated_bytes_per_row
            ),
            "one_deck_logical_disk_bytes": 32_766 * bytes_per_row,
            "one_deck_allocated_disk_bytes": (
                32_766 * allocated_bytes_per_row
            ),
            "projection_method": (
                "placement-weighted measured query latency divided by "
                "measured best worker throughput speedup"
            ),
        },
        "reproducible": all(
            bool(record["matches_original"])
            for record in teacher["repeat_records"]
        ),
    }
    return _write_sealed(output / "summary.json", content)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--phase",
        choices=("teacher", "prefix", "structural", "workers", "summary", "all"),
        default="all",
    )
    args = parser.parse_args(argv)
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    prior = {
        "opponent_mode": "search-v2-nested",
        "outer_simulations": 32,
        "response_simulations": 32,
        "exploration_constant": math.sqrt(2.0),
        "database": ".local/dracula.sqlite3",
        "api_port": 8000,
        "frontend_port": 4173,
    }
    prior_path = output / "prior-preview.json"
    if not prior_path.exists():
        _write_sealed(prior_path, prior)
    phases = (
        ("teacher", run_teacher_measurements),
        ("prefix", run_prefix_measurements),
        ("structural", run_structural_measurement),
        ("workers", run_worker_measurements),
    )
    if args.phase == "all":
        for name, function in phases:
            print(f"phase={name}", flush=True)
            function(output)
        _summary(output)
    elif args.phase == "summary":
        _summary(output)
    else:
        dict(phases)[args.phase](output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Benchmark complete-game Teacher v2 collection worker counts."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

import torch

from dracula.teacher import FixtureSplit, load_teacher_manifest, load_teacher_shard

_SWAP_PATTERN = re.compile(r"used = ([0-9.]+)([MG])")
_MEMORY_FREE_PATTERN = re.compile(r"free percentage: ([0-9]+)%")


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
        max((row[3] for row in selected), default=0),
    )


def _update_tensor_digest(digest, name: str, value: torch.Tensor) -> None:
    tensor = value.detach().cpu().contiguous()
    digest.update(name.encode("utf-8"))
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(str(tuple(tensor.shape)).encode("ascii"))
    digest.update(tensor.numpy().tobytes(order="C"))


def _trajectory_digest(output: Path) -> str:
    digest = hashlib.sha256()
    for split in (FixtureSplit.TRAINING, FixtureSplit.VALIDATION):
        manifest = load_teacher_manifest(output, split)["manifest"]
        for entry in manifest["shards"]:
            shard = load_teacher_shard(output / entry["relative_path"])
            for name, value in sorted(shard["columns"].items()):
                if isinstance(value, torch.Tensor):
                    _update_tensor_digest(digest, name, value)
                else:
                    digest.update(name.encode("utf-8"))
                    digest.update(
                        json.dumps(
                            value,
                            separators=(",", ":"),
                            sort_keys=True,
                        ).encode("utf-8")
                    )
    return digest.hexdigest()


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


def _run_one(
    *,
    output: Path,
    root_seed: str,
    games: int,
    workers: int,
    sample_interval: float,
) -> dict[str, object]:
    if output.exists():
        raise RuntimeError(f"cold-cache output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    log_path = output.parent / f"workers-{workers}.log"
    command = [
        sys.executable,
        "-m",
        "dracula.teacher",
        "full",
        "--output",
        str(output),
        "--run-id",
        f"teacher-v2-parallelism-{workers}",
        "--root-seed",
        root_seed,
        "--training-games",
        str(games),
        "--validation-games",
        "0",
        "--absolute-fixtures",
        "0",
        "--workers",
        str(workers),
    ]
    swap_start = _swap_used_bytes()
    swap_peak = swap_start
    free_percent_min = _memory_free_percent()
    cpu_samples: list[float] = []
    peak_total_rss = 0
    peak_process_rss = 0
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=Path.cwd(),
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            while process.poll() is None:
                cpu, total_rss, process_rss = _process_tree_sample(process.pid)
                cpu_samples.append(cpu)
                peak_total_rss = max(peak_total_rss, total_rss)
                peak_process_rss = max(peak_process_rss, process_rss)
                swap = _swap_used_bytes()
                if swap is not None:
                    swap_peak = max(swap_peak or swap, swap)
                free_percent = _memory_free_percent()
                if free_percent is not None:
                    free_percent_min = min(
                        free_percent_min
                        if free_percent_min is not None
                        else free_percent,
                        free_percent,
                    )
                time.sleep(sample_interval)
        except KeyboardInterrupt:
            process.send_signal(signal.SIGINT)
            process.wait()
            raise
        return_code = process.wait()
    wall_seconds = time.perf_counter() - started
    if return_code != 0:
        raise RuntimeError(
            f"teacher benchmark failed for {workers} workers; see {log_path}"
        )
    metrics = json.loads(
        (output / "teacher" / "metrics.json").read_text(encoding="utf-8")
    )
    worker_metrics = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(output.glob("teacher/games/**/*.metrics.json"))
    ]
    swap_end = _swap_used_bytes()
    return {
        "workers": workers,
        "games": games,
        "examples": metrics["examples"],
        "wall_seconds": wall_seconds,
        "collector_wall_seconds": metrics["wall_seconds"],
        "aggregate_search_seconds": metrics["search_seconds"],
        "examples_per_hour": metrics["examples_per_hour"],
        "simulations_per_second": metrics["simulations_per_second"],
        "mean_aggregate_cpu_percent": (
            sum(cpu_samples) / len(cpu_samples) if cpu_samples else None
        ),
        "mean_machine_cpu_percent": (
            sum(cpu_samples) / len(cpu_samples) / 8.0
            if cpu_samples
            else None
        ),
        "peak_process_rss_bytes": peak_process_rss,
        "peak_total_rss_bytes": peak_total_rss,
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
        "minimum_memory_free_percent": free_percent_min,
        "maximum_worker_reported_rss_bytes": max(
            (item["peak_rss_bytes"] for item in worker_metrics),
            default=0,
        ),
        "torch_thread_counts": sorted(
            {
                (
                    item["torch_num_threads"],
                    item["torch_num_interop_threads"],
                )
                for item in worker_metrics
            }
        ),
        "trajectory_digest": _trajectory_digest(output),
        "thermal_status_after": _thermal_status(),
        "log_path": str(log_path),
        "output_path": str(output),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--root-seed", default="teacher-v2-parallelism-v1")
    parser.add_argument("--games", type=int, default=6)
    parser.add_argument(
        "--workers",
        type=int,
        nargs="+",
        default=(1, 2, 4, 6),
    )
    parser.add_argument("--sample-interval", type=float, default=0.25)
    args = parser.parse_args()
    output_root = Path(args.output_root).expanduser().resolve()
    results = [
        _run_one(
            output=output_root / f"workers-{workers}",
            root_seed=args.root_seed,
            games=args.games,
            workers=workers,
            sample_interval=args.sample_interval,
        )
        for workers in args.workers
    ]
    trajectory_digests = {item["trajectory_digest"] for item in results}
    if len(trajectory_digests) != 1:
        raise RuntimeError("worker counts produced different trajectories")
    document = {
        "benchmark": {
            "games": args.games,
            "root_seed": args.root_seed,
            "worker_counts": args.workers,
            "sample_interval_seconds": args.sample_interval,
        },
        "results": results,
    }
    destination = output_root / "results.json"
    destination.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(document, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Exercise the production Lambda image through its public HTTP boundary.

The validator runs the image twice. The first process measures warm gameplay,
cache hits, policy inference, logging, and filesystem behavior. The second
disables the replay cache and reconstructs saved histories after a process
restart. Equality between those results demonstrates that the cache and warm
process memory improve speed without becoming required game state.

This tool operates only on a locally built Docker image. It does not contact
AWS, mutate infrastructure, or make Bedrock requests.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from build_lambda_context import SELECTED_POLICY_SHA256
from lambda_validation_gameplay import (
    COMPLETE_GAME_FIXTURES,
    FORBIDDEN_RESPONSE_KEYS,
    ContainerValidationError,
    all_keys,
    next_command,
    play_complete_game,
    request_json,
    wait_for_health,
    validate_frontend,
)
POLICY_LATENCY_PATTERN = re.compile(r"decision_latency_seconds=([0-9.]+)")


@dataclass(frozen=True)
class PrimaryMeasurements:
    """Measurements and replay fixtures retained from the warm-cache process."""

    initialization: float
    readiness: float
    warm_health: list[float]
    games: list[dict[str, Any]]
    request_latencies: list[float]
    request_count: int
    first_checkpoints: dict[int, dict[str, Any]]
    warm_replay: list[float]
    memory: list[int]
    logs: str
    log_lines: int
    log_bytes: int


@dataclass(frozen=True)
class RestartMeasurements:
    """Measurements from a fresh process with replay caching disabled."""

    initialization: float
    readiness: float
    replay_ms_by_history_length: dict[str, float]
    complete_game_replay: float
    memory: int
    logs: str


def _run(*arguments: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    """Run Docker synchronously while retaining output for validation errors."""

    return subprocess.run(
        arguments,
        check=check,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _memory_bytes(container: str) -> int:
    """Convert Docker's human-readable current memory use into bytes."""

    output = _run(
        "docker", "stats", "--no-stream", "--format", "{{.MemUsage}}", container
    ).stdout.strip()
    raw = output.split("/", 1)[0].strip()
    match = re.fullmatch(r"([0-9.]+)([KMG]iB|B)", raw)
    if match is None:
        raise ContainerValidationError(f"unrecognized Docker memory value: {raw}")
    multiplier = {"B": 1, "KiB": 1024, "MiB": 1024**2, "GiB": 1024**3}[
        match.group(2)
    ]
    return round(float(match.group(1)) * multiplier)


def _start_container(
    image: str,
    name: str,
    port: int,
    environment: Mapping[str, str] | None = None,
) -> float:
    """Start a read-only local container and return its start timestamp."""

    _run("docker", "rm", "--force", name, check=False)
    started = time.perf_counter()
    # Lambda permits writes only below /tmp. The local run reproduces that
    # filesystem rule instead of giving the application a writable image.
    arguments = [
        "docker",
        "run",
        "--detach",
        "--name",
        name,
        "--read-only",
        "--tmpfs",
        "/tmp:rw,noexec,nosuid,size=512m",
        "--publish",
        f"127.0.0.1:{port}:8080",
    ]
    for key, value in sorted((environment or {}).items()):
        arguments.extend(("--env", f"{key}={value}"))
    result = _run(*arguments, image)
    if not result.stdout.strip():
        raise ContainerValidationError("Docker did not return a container ID")
    return started


def _stop_container(name: str) -> str:
    """Capture complete logs before removing the disposable container."""

    logs = _run("docker", "logs", name, check=False).stdout
    _run("docker", "rm", "--force", name, check=False)
    return logs


def _validate_read_only_and_artifact(container: str) -> None:
    """Verify the embedded policy and Lambda-style writable-filesystem boundary."""

    digest = _run(
        "docker",
        "exec",
        container,
        "sha256sum",
        "/opt/dracula/artifacts/policy.pt",
    ).stdout.split()[0]
    if digest != SELECTED_POLICY_SHA256:
        raise ContainerValidationError("container pi1 digest differs")
    write_check = _run(
        "docker",
        "exec",
        container,
        "sh",
        "-c",
        "! touch /var/task/root-write-test && touch /tmp/tmp-write-test",
        check=False,
    )
    if write_check.returncode != 0:
        raise ContainerValidationError("read-only root or writable /tmp check failed")


def _percentile(values: list[float], fraction: float) -> float:
    """Return a nearest-rank sample percentile for concise local measurements."""

    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))
    return ordered[index]


def _validate_json_logs(logs: str) -> tuple[int, int]:
    """Require one JSON object per nonempty CloudWatch-bound output line."""

    encoded_bytes = len(logs.encode("utf-8"))
    lines = [line for line in logs.splitlines() if line.strip()]
    for line in lines:
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as error:
            raise ContainerValidationError("container emitted a non-JSON log line") from error
        if not isinstance(payload, dict):
            raise ContainerValidationError("container log line is not a JSON object")
    return len(lines), encoded_bytes


def _validate_health_payload(health: dict[str, Any]) -> None:
    """Confirm policy availability and deliberately disabled local narration."""

    expected = {
        "narration_enabled": False,
        "opponent_configured": True,
        "status": "ok",
    }
    if health != expected:
        raise ContainerValidationError(f"unexpected health response: {health}")


def _measure_warm_health(base_url: str) -> list[float]:
    """Measure a small stable sample after process readiness."""

    latencies: list[float] = []
    for _ in range(20):
        _payload, elapsed, status = request_json(base_url, "GET", "/health")
        if status != 200:
            raise ContainerValidationError("warm health request failed")
        latencies.append(elapsed)
    return latencies


def _exercise_game_fixtures(
    base_url: str, container: str
) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]], list[float], int, list[int]]:
    """Play both roles and dealer assignments while sampling memory and privacy."""

    games: list[dict[str, Any]] = []
    first_checkpoints: dict[int, dict[str, Any]] = {}
    latencies: list[float] = []
    request_count = 0
    memory: list[int] = []
    for index, (role, seed, dealer) in enumerate(COMPLETE_GAME_FIXTURES):
        response, checkpoints, game_latencies, game_requests = play_complete_game(
            base_url, human_role=role, seed=seed, expected_initial_dealer=dealer
        )
        if FORBIDDEN_RESPONSE_KEYS & all_keys(response):
            raise ContainerValidationError("public response contains forbidden fields")
        if SELECTED_POLICY_SHA256 in json.dumps(response, sort_keys=True):
            raise ContainerValidationError("public response exposes the model digest")
        games.append(response)
        latencies.extend(game_latencies)
        request_count += game_requests
        memory.append(_memory_bytes(container))
        # A single game's complete checkpoint set is enough for restart tests;
        # the other fixtures exist to cover roles and dealers.
        if index == 0:
            first_checkpoints = checkpoints
    return games, first_checkpoints, latencies, request_count, memory


def _validate_retry_and_history(base_url: str, opening: dict[str, Any]) -> None:
    """Distinguish an identical retry from a duplicated command in history."""

    first_command = next_command(opening)
    duplicate_a, _elapsed, status = request_json(
        base_url, "POST", "/games/command", first_command
    )
    duplicate_b, _elapsed, repeated_status = request_json(
        base_url, "POST", "/games/command", first_command
    )
    if status != 200 or repeated_status != 200 or duplicate_a != duplicate_b:
        raise ContainerValidationError("duplicate stateless request is not deterministic")
    malformed_history = {
        "seed": duplicate_a["envelope"]["seed"],
        "history": duplicate_a["envelope"]["history"] + [first_command["command"]],
    }
    invalid, _elapsed, invalid_status = request_json(
        base_url, "POST", "/games/resume", {"envelope": malformed_history}, allow_error=True
    )
    if invalid_status != 422 or invalid.get("code") != "invalid_history":
        raise ContainerValidationError("duplicated history was not rejected")


def _validate_disabled_narration(base_url: str, opening: dict[str, Any]) -> None:
    """Confirm narration failure remains an explicit non-gameplay result."""

    narration, _elapsed, status = request_json(
        base_url,
        "POST",
        "/narration",
        {"envelope": opening["envelope"], "cue_type": "opening"},
    )
    expected = {"cue_type": "opening", "status": "unavailable", "text": None}
    if status != 200 or narration != expected:
        raise ContainerValidationError("disabled narration returned an unexpected result")


def _measure_warm_replay(
    base_url: str, completed_response: dict[str, Any]
) -> list[float]:
    """Measure repeated reconstruction of one complete cached game."""

    latencies: list[float] = []
    for _ in range(10):
        replayed, elapsed, status = request_json(
            base_url,
            "POST",
            "/games/resume",
            {"envelope": completed_response["envelope"]},
        )
        if status != 200 or replayed != completed_response:
            raise ContainerValidationError("warm replay differs from authoritative result")
        latencies.append(elapsed)
    return latencies


def _run_primary_container(
    image: str, container: str, port: int, base_url: str
) -> PrimaryMeasurements:
    """Execute the warm-process validation phase and capture its logs."""

    # The first process validates ordinary warm operation and collects replay
    # fixtures that remain meaningful after that process is destroyed.
    started = _start_container(image, container, port)
    health, readiness = wait_for_health(base_url)
    initialization = time.perf_counter() - started
    _validate_health_payload(health)
    validate_frontend(
        base_url.removesuffix("/Dracula/api"),
        Path(__file__).resolve().parents[1] / "frontend/dist",
    )
    _validate_read_only_and_artifact(container)
    memory = [_memory_bytes(container)]
    warm_health = _measure_warm_health(base_url)
    games, checkpoints, gameplay, requests, game_memory = _exercise_game_fixtures(
        base_url, container
    )
    memory.extend(game_memory)
    _validate_retry_and_history(base_url, checkpoints[1])
    _validate_disabled_narration(base_url, checkpoints[1])
    warm_replay = _measure_warm_replay(base_url, games[0])
    memory.append(_memory_bytes(container))
    logs = _stop_container(container)
    log_lines, log_bytes = _validate_json_logs(logs)
    return PrimaryMeasurements(
        initialization, readiness, warm_health, games, gameplay, requests,
        checkpoints, warm_replay, memory, logs, log_lines, log_bytes
    )


def _run_cache_disabled_container(
    image: str,
    container: str,
    port: int,
    base_url: str,
    checkpoints: dict[int, dict[str, Any]],
) -> RestartMeasurements:
    """Replay retained histories in a fresh process with no cache assistance."""

    # Disabling the optimization proves that reconstructed results do not rely
    # on state retained by the first process or its in-memory replay cache.
    started = _start_container(
        image, container, port, {"DRACULA_REPLAY_CACHE_ENTRIES": "0"}
    )
    _health, readiness = wait_for_health(base_url)
    initialization = time.perf_counter() - started
    replay_ms: dict[str, float] = {}
    complete_game_replay: float | None = None
    for history_length, checkpoint in sorted(checkpoints.items()):
        replayed, elapsed, status = request_json(
            base_url, "POST", "/games/resume", {"envelope": checkpoint["envelope"]}
        )
        if status != 200 or replayed != checkpoint:
            raise ContainerValidationError(
                f"cache-disabled replay differs at history length {history_length}"
            )
        replay_ms[str(history_length)] = elapsed * 1000.0
        if history_length == 31:
            complete_game_replay = elapsed
    if complete_game_replay is None:
        raise ContainerValidationError("complete-game replay checkpoint is absent")
    memory = _memory_bytes(container)
    logs = _stop_container(container)
    return RestartMeasurements(
        initialization, readiness, replay_ms, complete_game_replay, memory, logs
    )


def _policy_latencies(logs: str) -> list[float]:
    """Extract policy decision timings emitted by the gameplay service."""

    latencies = [
        float(match.group(1)) for match in POLICY_LATENCY_PATTERN.finditer(logs)
    ]
    if not latencies:
        raise ContainerValidationError("container logs contain no pi1 inference timings")
    return latencies


def _validation_result(
    image: str, primary: PrimaryMeasurements, restart: RestartMeasurements
) -> dict[str, object]:
    """Combine both process runs into the machine-readable validation record."""

    all_logs = primary.logs + restart.logs
    policy = _policy_latencies(all_logs)
    log_lines, log_bytes = _validate_json_logs(all_logs)
    # Seeds are deliberately client-visible but must not enter production logs.
    for _role, seed, _dealer in COMPLETE_GAME_FIXTURES:
        if seed in all_logs:
            raise ContainerValidationError("container logs expose a game seed")
    image_data = json.loads(
        _run("docker", "image", "inspect", image, "--format", "{{json .}}").stdout
    )
    # Preserve raw restart timings alongside aggregates so later release review
    # can distinguish long-history replay cost from ordinary warm requests.
    return {
        "artifact_sha256": SELECTED_POLICY_SHA256,
        "cache_disabled_replay_ms_by_history_length": restart.replay_ms_by_history_length,
        "cold_replay_ms": restart.complete_game_replay * 1000.0,
        "complete_game_count": len(primary.games),
        "frontend_bytes_verified": True,
        "complete_game_requests": primary.request_count,
        "container_architecture": image_data["Architecture"],
        "image": image,
        "image_size_bytes": image_data["Size"],
        "initialization_seconds": [primary.initialization, restart.initialization],
        "log_bytes": log_bytes,
        "log_line_count": log_lines,
        "narration_mode": "disabled",
        "peak_memory_bytes": max(*primary.memory, restart.memory),
        "primary_log_bytes": primary.log_bytes,
        "primary_log_bytes_per_complete_game_upper_bound": primary.log_bytes / len(primary.games),
        "primary_log_line_count": primary.log_lines,
        "pi1_inference_count": len(policy),
        "pi1_inference_ms_mean": statistics.fmean(policy) * 1000.0,
        "pi1_inference_ms_p95": _percentile(policy, 0.95) * 1000.0,
        "readiness_seconds": [primary.readiness, restart.readiness],
        "warm_gameplay_ms_mean": statistics.fmean(primary.request_latencies) * 1000.0,
        "warm_gameplay_ms_p50": _percentile(primary.request_latencies, 0.50) * 1000.0,
        "warm_gameplay_ms_p95": _percentile(primary.request_latencies, 0.95) * 1000.0,
        "warm_health_ms_mean": statistics.fmean(primary.warm_health) * 1000.0,
        "warm_health_ms_p95": _percentile(primary.warm_health, 0.95) * 1000.0,
        "warm_replay_ms_mean": statistics.fmean(primary.warm_replay) * 1000.0,
        "warm_replay_ms_p95": _percentile(primary.warm_replay, 0.95) * 1000.0,
    }


def _write_results(output: Path, result: dict[str, object], logs: str) -> None:
    """Write local measurement evidence; these files are ignored build output."""

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    output.with_suffix(".log").write_text(logs)


def validate(image: str, port: int, output: Path) -> dict[str, object]:
    """Run both validation phases and always remove their Docker container."""

    container = "dracula-lambda-validation"
    base_url = f"http://127.0.0.1:{port}/Dracula/api"
    try:
        primary = _run_primary_container(image, container, port, base_url)
        restart = _run_cache_disabled_container(
            image, container, port, base_url, primary.first_checkpoints
        )
        result = _validation_result(image, primary, restart)
        _write_results(output, result, primary.logs + restart.logs)
        return result
    finally:
        _run("docker", "rm", "--force", container, check=False)


def _parser() -> argparse.ArgumentParser:
    """Define the local image, host port, and ignored measurement output."""

    parser = argparse.ArgumentParser(
        description="Validate the built stateless Lambda container locally."
    )
    parser.add_argument("--image", default="dracula-api:local")
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument(
        "--output", type=Path, default=Path("build/lambda-validation.json")
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run container validation and print its machine-readable result."""

    arguments = _parser().parse_args(argv)
    if not 1024 <= arguments.port <= 65535:
        raise SystemExit("--port must be between 1024 and 65535")
    result = validate(arguments.image, arguments.port, arguments.output)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

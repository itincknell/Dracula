"""Exercise the production Lambda image through its public HTTP boundary.

The validator measures startup and request behavior while checking stateless
replay, policy loading, fake narration, privacy, and read-only operation.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from build_lambda_context import PI1_SHA256


FORBIDDEN_RESPONSE_KEYS = frozenset(
    {
        "authoritative_state",
        "determination",
        "engine_seed",
        "logits",
        "model_state",
        "opponent_hand",
        "policy_input",
        "policy_mask",
        "search_tree",
        "stock_order",
        "tensor",
    }
)
POLICY_LATENCY_PATTERN = re.compile(r"decision_latency_seconds=([0-9.]+)")
COMPLETE_GAME_FIXTURES = (
    ("queen", "stateless-queen-6", "queen"),
    ("queen", "stateless-king-0", "king"),
    ("king", "stateless-queen-6", "queen"),
    ("king", "stateless-king-0", "king"),
)


class ContainerValidationError(RuntimeError):
    pass


def _run(*arguments: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        arguments,
        check=check,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _request(
    base_url: str,
    method: str,
    path: str,
    body: dict[str, Any] | None = None,
    *,
    allow_error: bool = False,
) -> tuple[dict[str, Any], float, int]:
    encoded = None if body is None else json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        base_url + path,
        data=encoded,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            raw = response.read()
            status = response.status
    except urllib.error.HTTPError as error:
        if not allow_error:
            raise ContainerValidationError(
                f"{method} {path} failed ({error.code}): {error.read().decode()}"
            ) from error
        raw = error.read()
        status = error.code
    elapsed = time.perf_counter() - started
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ContainerValidationError(f"{method} {path} returned non-object JSON")
    return payload, elapsed, status


def _wait_for_health(base_url: str) -> tuple[dict[str, Any], float]:
    started = time.perf_counter()
    deadline = started + 45.0
    while time.perf_counter() < deadline:
        try:
            health, _elapsed, status = _request(base_url, "GET", "/health")
            if status == 200:
                return health, time.perf_counter() - started
        except (OSError, ContainerValidationError):
            pass
        time.sleep(0.1)
    raise ContainerValidationError("container did not become healthy within 45 seconds")


def _all_keys(value: Any) -> set[str]:
    if isinstance(value, dict):
        nested = set().union(*(_all_keys(item) for item in value.values()))
        return set(value) | nested
    if isinstance(value, list):
        return set().union(*(_all_keys(item) for item in value)) if value else set()
    return set()


def _next_command(response: dict[str, Any]) -> dict[str, Any]:
    game = response["game"]
    phase = game["phase"]["kind"]
    if phase == "human_turn":
        move = game["legal_moves"][0]
        command = {
            "type": "place",
            "hand_slot": move["hand_slot"],
            "position": move["position"],
        }
    elif phase == "scoring":
        command = {"type": "advance_round"}
    else:
        raise ContainerValidationError(f"unexpected game phase: {phase}")
    return {"envelope": response["envelope"], "command": command}


def _memory_bytes(container: str) -> int:
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
    _run("docker", "rm", "--force", name, check=False)
    started = time.perf_counter()
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
    logs = _run("docker", "logs", name, check=False).stdout
    _run("docker", "rm", "--force", name, check=False)
    return logs


def _validate_read_only_and_artifact(container: str) -> None:
    digest = _run(
        "docker",
        "exec",
        container,
        "sha256sum",
        "/opt/dracula/artifacts/pi1.pt",
    ).stdout.split()[0]
    if digest != PI1_SHA256:
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
    persistence_check = _run(
        "docker",
        "exec",
        container,
        "sh",
        "-c",
        'test -z "$DRACULA_DATABASE_PATH" && '
        '! find /var/task /opt/dracula -type f -name "*.sqlite3" -print -quit | grep -q .',
        check=False,
    )
    if persistence_check.returncode != 0:
        raise ContainerValidationError("production image has a database dependency")


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))
    return ordered[index]


def _play_complete_game(
    base_url: str,
    *,
    human_role: str,
    seed: str,
    expected_initial_dealer: str,
) -> tuple[dict[str, Any], dict[int, dict[str, Any]], list[float], int]:
    response, elapsed, status = _request(
        base_url,
        "POST",
        "/games",
        {"human_role": human_role, "seed": seed},
    )
    if status != 201:
        raise ContainerValidationError("game creation did not return 201")
    if response["game"]["dealer"] != expected_initial_dealer:
        raise ContainerValidationError("initial dealer differs from fixture")
    requests = 1
    latencies = [elapsed]
    checkpoints = {len(response["envelope"]["history"]): response}
    while response["game"]["phase"]["kind"] != "game_complete":
        response, elapsed, status = _request(
            base_url, "POST", "/games/command", _next_command(response)
        )
        if status != 200:
            raise ContainerValidationError("game command did not return 200")
        requests += 1
        latencies.append(elapsed)
        checkpoints[len(response["envelope"]["history"])] = response
    if len(response["game"]["completed_rounds"]) != 6:
        raise ContainerValidationError("container game did not complete six rounds")
    if len(response["envelope"]["history"]) != 31:
        raise ContainerValidationError("complete game history does not contain 31 commands")
    return response, checkpoints, latencies, requests


def _validate_json_logs(logs: str) -> tuple[int, int]:
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


def validate(image: str, port: int, output: Path) -> dict[str, object]:
    container = "dracula-lambda-validation"
    base_url = f"http://127.0.0.1:{port}"
    all_logs = ""
    memory: list[int] = []
    try:
        started = _start_container(image, container, port)
        health, readiness = _wait_for_health(base_url)
        initialization = time.perf_counter() - started
        if health != {
            "gameplay_mode": "stateless",
            "narration_configured": False,
            "narration_enabled": False,
            "opponent_configured": True,
            "status": "ok",
        }:
            raise ContainerValidationError(f"unexpected health response: {health}")
        _validate_read_only_and_artifact(container)
        memory.append(_memory_bytes(container))

        warm_health: list[float] = []
        for _ in range(20):
            _payload, elapsed, status = _request(base_url, "GET", "/health")
            if status != 200:
                raise ContainerValidationError("warm health request failed")
            warm_health.append(elapsed)

        games: list[dict[str, Any]] = []
        request_latencies: list[float] = []
        requests = 0
        first_checkpoints: dict[int, dict[str, Any]] = {}
        for index, (role, seed, dealer) in enumerate(COMPLETE_GAME_FIXTURES):
            response, checkpoints, latencies, game_requests = _play_complete_game(
                base_url,
                human_role=role,
                seed=seed,
                expected_initial_dealer=dealer,
            )
            games.append(response)
            request_latencies.extend(latencies)
            requests += game_requests
            if index == 0:
                first_checkpoints = checkpoints
            memory.append(_memory_bytes(container))
            if FORBIDDEN_RESPONSE_KEYS & _all_keys(response):
                raise ContainerValidationError("public response contains forbidden fields")
            if PI1_SHA256 in json.dumps(response, sort_keys=True):
                raise ContainerValidationError("public response exposes the model digest")

        response = games[0]
        opening = first_checkpoints[1]
        first_command = _next_command(opening)
        duplicate_a, _elapsed, status = _request(
            base_url, "POST", "/games/command", first_command
        )
        duplicate_b, _elapsed, repeated_status = _request(
            base_url, "POST", "/games/command", first_command
        )
        if status != 200 or repeated_status != 200 or duplicate_a != duplicate_b:
            raise ContainerValidationError("duplicate stateless request is not deterministic")
        malformed_history = {
            "seed": duplicate_a["envelope"]["seed"],
            "history": duplicate_a["envelope"]["history"] + [first_command["command"]],
        }
        invalid, _elapsed, invalid_status = _request(
            base_url,
            "POST",
            "/games/resume",
            {"envelope": malformed_history},
            allow_error=True,
        )
        if invalid_status != 422 or invalid.get("code") != "invalid_history":
            raise ContainerValidationError("duplicated history was not rejected")

        narration, _elapsed, narration_status = _request(
            base_url,
            "POST",
            "/narration",
            {"envelope": opening["envelope"], "cue_type": "opening"},
        )
        if narration_status != 200 or narration != {
            "cue_type": "opening",
            "status": "unavailable",
            "text": None,
        }:
            raise ContainerValidationError("disabled narration returned an unexpected result")

        envelope = response["envelope"]
        warm_replay: list[float] = []
        for _ in range(10):
            replayed, elapsed, status = _request(
                base_url, "POST", "/games/resume", {"envelope": envelope}
            )
            if status != 200 or replayed != response:
                raise ContainerValidationError("warm replay differs from authoritative result")
            warm_replay.append(elapsed)
        memory.append(_memory_bytes(container))
        first_logs = _stop_container(container)
        primary_log_lines, primary_log_bytes = _validate_json_logs(first_logs)
        all_logs += first_logs

        started = _start_container(
            image,
            container,
            port,
            {"DRACULA_REPLAY_CACHE_ENTRIES": "0"},
        )
        _health, second_readiness = _wait_for_health(base_url)
        second_initialization = time.perf_counter() - started
        history_lengths = sorted(first_checkpoints)
        cold_replay_by_history: dict[str, float] = {}
        cold_replay = 0.0
        for history_length in history_lengths:
            checkpoint = first_checkpoints[history_length]
            cold_replayed, elapsed, status = _request(
                base_url,
                "POST",
                "/games/resume",
                {"envelope": checkpoint["envelope"]},
            )
            if status != 200 or cold_replayed != checkpoint:
                raise ContainerValidationError(
                    f"cache-disabled replay differs at history length {history_length}"
                )
            cold_replay_by_history[str(history_length)] = elapsed * 1000.0
            if history_length == 31:
                cold_replay = elapsed
        memory.append(_memory_bytes(container))
        all_logs += _stop_container(container)

        policy_latencies = [
            float(match.group(1))
            for match in POLICY_LATENCY_PATTERN.finditer(all_logs)
        ]
        if not policy_latencies:
            raise ContainerValidationError("container logs contain no pi1 inference timings")
        image_size = int(
            _run(
                "docker", "image", "inspect", image, "--format", "{{.Size}}"
            ).stdout.strip()
        )
        log_lines, log_bytes = _validate_json_logs(all_logs)
        for _role, seed, _dealer in COMPLETE_GAME_FIXTURES:
            if seed in all_logs:
                raise ContainerValidationError("container logs expose a game seed")
        result: dict[str, object] = {
            "artifact_sha256": PI1_SHA256,
            "cache_disabled_replay_ms_by_history_length": cold_replay_by_history,
            "cold_replay_ms": cold_replay * 1000.0,
            "complete_game_count": len(games),
            "complete_game_requests": requests,
            "container_architecture": _run(
                "docker", "image", "inspect", image, "--format", "{{.Architecture}}"
            ).stdout.strip(),
            "image": image,
            "image_size_bytes": image_size,
            "initialization_seconds": [initialization, second_initialization],
            "log_bytes": log_bytes,
            "log_line_count": log_lines,
            "narration_mode": "disabled",
            "peak_memory_bytes": max(memory),
            "primary_log_bytes": primary_log_bytes,
            "primary_log_bytes_per_complete_game_upper_bound": (
                primary_log_bytes / len(games)
            ),
            "primary_log_line_count": primary_log_lines,
            "pi1_inference_count": len(policy_latencies),
            "pi1_inference_ms_mean": statistics.fmean(policy_latencies) * 1000.0,
            "pi1_inference_ms_p95": _percentile(policy_latencies, 0.95) * 1000.0,
            "readiness_seconds": [readiness, second_readiness],
            "warm_gameplay_ms_mean": statistics.fmean(request_latencies) * 1000.0,
            "warm_gameplay_ms_p50": _percentile(request_latencies, 0.50) * 1000.0,
            "warm_gameplay_ms_p95": _percentile(request_latencies, 0.95) * 1000.0,
            "warm_health_ms_mean": statistics.fmean(warm_health) * 1000.0,
            "warm_health_ms_p95": _percentile(warm_health, 0.95) * 1000.0,
            "warm_replay_ms_mean": statistics.fmean(warm_replay) * 1000.0,
            "warm_replay_ms_p95": _percentile(warm_replay, 0.95) * 1000.0,
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        output.with_suffix(".log").write_text(all_logs)
        return result
    finally:
        _run("docker", "rm", "--force", container, check=False)


def _parser() -> argparse.ArgumentParser:
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
    arguments = _parser().parse_args(argv)
    if not 1024 <= arguments.port <= 65535:
        raise SystemExit("--port must be between 1024 and 65535")
    result = validate(arguments.image, arguments.port, arguments.output)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

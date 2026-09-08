"""Exercise stateless gameplay through a running container's HTTP boundary.

This module owns request parsing and complete-game traversal for the Lambda
validator. Docker lifecycle, filesystem checks, logs, and performance reporting
remain in ``validate_lambda_container.py``.
"""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
import time
import urllib.error
import urllib.request
from typing import Any


FORBIDDEN_RESPONSE_KEYS = frozenset(
    {
        "authoritative_state",
        "determination",
        "determinization",
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
COMPLETE_GAME_FIXTURES = (
    ("queen", "stateless-queen-6", "queen"),
    ("queen", "stateless-king-1", "king"),
    ("king", "stateless-queen-6", "queen"),
    ("king", "stateless-king-1", "king"),
)


def validate_frontend(origin: str, distribution: Path) -> dict[str, int]:
    """Compare served assets to the build, including binary transport through AWS."""

    files = sorted(path for path in distribution.rglob("*") if path.is_file())
    largest = max(files, key=lambda path: path.stat().st_size)
    # Buffered Lambda responses have a 6 MB limit, including the base64 envelope.
    # A 4 MB file ceiling leaves room for that expansion and response headers.
    if largest.stat().st_size > 4_000_000:
        raise ContainerValidationError(f"asset exceeds buffered response budget: {largest.name}")
    # The distribution is small: check every card, portrait, favicon, and font,
    # not just a representative file that could miss a binary transport problem.
    for path in files:
        relative = path.relative_to(distribution).as_posix()
        url = origin + "/Dracula/" + ("" if relative == "index.html" else relative)
        with urllib.request.urlopen(url, timeout=30) as response:
            served = response.read()
            if response.status != 200 or not response.headers.get("Content-Type"):
                raise ContainerValidationError(f"frontend asset response is invalid: {relative}")
        if hashlib.sha256(served).digest() != hashlib.sha256(path.read_bytes()).digest():
            raise ContainerValidationError(f"frontend asset differs: {relative}")
    return {"files_checked": len(files), "largest_asset_bytes": largest.stat().st_size}


class ContainerValidationError(RuntimeError):
    """The running container violated its public gameplay contract."""


def request_json(
    base_url: str,
    method: str,
    path: str,
    body: dict[str, Any] | None = None,
    *,
    allow_error: bool = False,
) -> tuple[dict[str, Any], float, int]:
    """Issue one measured JSON request and return object, latency, and status.

    Error responses are raised by default. Tests that intentionally exercise a
    public error pass ``allow_error=True`` so the same JSON decoding and shape
    checks apply to both success and failure bodies.
    """

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
        # urllib raises for non-2xx responses even when the caller deliberately
        # requested an invalid-history or other public error fixture.
        if not allow_error:
            raise ContainerValidationError(
                f"{method} {path} failed ({error.code}): {error.read().decode()}"
            ) from error
        raw = error.read()
        status = error.code
    elapsed = time.perf_counter() - started
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ContainerValidationError(
            f"{method} {path} returned malformed JSON"
        ) from error
    if not isinstance(payload, dict):
        raise ContainerValidationError(f"{method} {path} returned non-object JSON")
    return payload, elapsed, status


def wait_for_health(base_url: str) -> tuple[dict[str, Any], float]:
    """Poll until the container accepts traffic or the cold-start limit expires."""

    started = time.perf_counter()
    deadline = started + 45.0
    while time.perf_counter() < deadline:
        try:
            health, _elapsed, status = request_json(base_url, "GET", "/health")
            if status == 200:
                return health, time.perf_counter() - started
        except (OSError, ContainerValidationError):
            pass
        time.sleep(0.1)
    raise ContainerValidationError("container did not become healthy within 45 seconds")


def all_keys(value: Any) -> set[str]:
    """Collect nested response keys for the private-field leak check."""

    if isinstance(value, dict):
        nested = set().union(*(all_keys(item) for item in value.values()))
        return set(value) | nested
    if isinstance(value, list):
        return set().union(*(all_keys(item) for item in value)) if value else set()
    return set()


def next_command(response: dict[str, Any]) -> dict[str, Any]:
    """Choose the first legal human move or advance a finished round."""

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


def play_complete_game(
    base_url: str,
    *,
    human_role: str,
    seed: str,
    expected_initial_dealer: str,
) -> tuple[dict[str, Any], dict[int, dict[str, Any]], list[float], int]:
    """Play six rounds while retaining replay checkpoints at every history size.

    The checkpoints let the container validator restart the process and replay
    short, medium, and complete histories through the public API.
    """

    # Start with a known dealer so the four fixtures jointly cover both human
    # roles and both opening-player arrangements.
    response, elapsed, status = request_json(
        base_url, "POST", "/games", {"human_role": human_role, "seed": seed}
    )
    if status != 201:
        raise ContainerValidationError("game creation did not return 201")
    if response["game"]["dealer"] != expected_initial_dealer:
        raise ContainerValidationError("initial dealer differs from fixture")
    requests = 1
    latencies = [elapsed]
    checkpoints = {len(response["envelope"]["history"]): response}
    # Choosing the first advertised legal move is sufficient here: this test
    # exercises transport and replay, not policy quality.
    while response["game"]["phase"]["kind"] != "game_complete":
        response, elapsed, status = request_json(
            base_url, "POST", "/games/command", next_command(response)
        )
        if status != 200:
            raise ContainerValidationError("game command did not return 200")
        requests += 1
        latencies.append(elapsed)
        # One checkpoint per history length supports cold replay at every stage.
        checkpoints[len(response["envelope"]["history"])] = response
    if len(response["game"]["completed_rounds"]) != 6:
        raise ContainerValidationError("container game did not complete six rounds")
    if len(response["envelope"]["history"]) != 31:
        raise ContainerValidationError("complete game history does not contain 31 commands")
    return response, checkpoints, latencies, requests

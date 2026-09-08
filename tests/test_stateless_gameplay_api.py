"""Exercise production stateless gameplay from request through public response.

Tests cover complete games, replay after reload, malformed histories, duplicate
commands, cache equivalence and eviction, policy determinism, and privacy.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from dracula.api.development import create_app
from dracula.policy.contracts import PolicyTurnResult
from dracula.api.stateless.contracts import RecoveryEnvelope, StatelessHumanGameView
from dracula.game.engine import EnginePlayer, EngineStatus, initial_dealer
from dracula.decision.information import SearchInformationState

pytestmark = pytest.mark.filterwarnings(
    "ignore:Using `httpx` with `starlette.testclient` is deprecated"
)


class FirstLegalStatelessPolicy:
    def __init__(self) -> None:
        self.requests: list[Any] = []

    def invoke(self, request: Any) -> PolicyTurnResult:
        assert isinstance(request.information_state, SearchInformationState)
        assert request.information_state.player is request.player
        assert not hasattr(request.information_state, "seed")
        assert not hasattr(request.information_state, "stock")
        assert not hasattr(request.information_state, "opponent_hand")
        self.requests.append(request)
        action = next(
            index for index, move in enumerate(request.action_table) if move is not None
        )
        return PolicyTurnResult(action)


class IllegalStatelessPolicy:
    def invoke(self, request: Any) -> PolicyTurnResult:
        action = next(
            index for index, move in enumerate(request.action_table) if move is None
        )
        return PolicyTurnResult(action)


def _app(
    policy: Any | None = None,
    *,
    cache_entries: int = 256,
) -> Any:
    return create_app(
        replay_cache_entries=cache_entries,
        policy_executor=policy or FirstLegalStatelessPolicy(),
        narration_enabled=False,
    )


def _start(
    client: TestClient,
    *,
    role: str,
    seed: str,
) -> dict[str, Any]:
    response = client.post("/games", json={"human_role": role, "seed": seed})
    assert response.status_code == 201, response.json()
    return response.json()


def _next_request(response: dict[str, Any]) -> dict[str, Any]:
    game = response["game"]
    if game["phase"]["kind"] == "human_turn":
        legal = game["legal_moves"][0]
        command = {
            "type": "place",
            "hand_slot": legal["hand_slot"],
            "position": legal["position"],
        }
    elif game["phase"]["kind"] == "scoring":
        command = {"type": "advance_round"}
    else:
        raise AssertionError(f"unexpected stateless phase: {game['phase']}")
    return {"envelope": response["envelope"], "command": command}


def _all_keys(value: Any) -> set[str]:
    if isinstance(value, dict):
        nested = set().union(*(_all_keys(item) for item in value.values()))
        return set(value) | nested
    if isinstance(value, list):
        return set().union(*(_all_keys(item) for item in value)) if value else set()
    return set()


@pytest.mark.parametrize(
    ("human_role", "seed", "expected_initial_dealer"),
    (
        ("queen", "stateless-queen-6", EnginePlayer.QUEEN),
        ("king", "stateless-king-1", EnginePlayer.KING),
    ),
)
def test_complete_games_reconstruct_on_cache_miss_at_every_lifecycle_state(
    human_role: str,
    seed: str,
    expected_initial_dealer: EnginePlayer,
) -> None:
    assert initial_dealer(seed) is expected_initial_dealer
    policy = FirstLegalStatelessPolicy()
    application = _app(policy)
    with TestClient(application) as client:
        response = _start(client, role=human_role, seed=seed)
        scoring_rounds: list[int] = []
        while response["game"]["phase"]["kind"] != "game_complete":
            application.state.stateless_gameplay_service.cache.clear()
            resumed = client.post(
                "/games/resume",
                json={"envelope": response["envelope"]},
            )
            assert resumed.status_code == 200, resumed.json()
            assert resumed.json() == response

            mutation = client.post("/games/command", json=_next_request(response))
            assert mutation.status_code == 200, mutation.json()
            response = mutation.json()
            if response["game"]["phase"]["kind"] == "scoring":
                scoring_rounds.append(response["game"]["round_number"])

    assert scoring_rounds == [1, 2, 3, 4, 5, 6]
    assert response["game"]["status"] == "game_complete"
    assert len(response["game"]["completed_rounds"]) == 6
    assert len(response["envelope"]["history"]) == 31
    assert sum(
        command["type"] == "place" for command in response["envelope"]["history"]
    ) == 24
    assert sum(
        command["type"] == "advance_round"
        for command in response["envelope"]["history"]
    ) == 6


def test_identical_request_cache_hit_cache_miss_and_new_process_match() -> None:
    body: dict[str, Any]
    first_policy = FirstLegalStatelessPolicy()
    first_app = _app(first_policy)
    with TestClient(first_app) as client:
        started = _start(client, role="queen", seed="stateless-idempotency")
        body = _next_request(started)
        first = client.post("/games/command", json=body)
        repeated = client.post("/games/command", json=body)
        assert first.status_code == repeated.status_code == 200
        assert first.json() == repeated.json()
        first_app.state.stateless_gameplay_service.cache.clear()
        cold = client.post("/games/command", json=body)
        assert cold.status_code == 200
        assert cold.json() == first.json()

    second_policy = FirstLegalStatelessPolicy()
    with TestClient(_app(second_policy, cache_entries=0)) as client:
        restarted = client.post("/games/command", json=body)
    assert restarted.status_code == 200
    assert restarted.json() == first.json()
    # The visible role and seed directly identify reproducible paired choices.
    assert first_policy.requests[0].game_key == "queen:stateless-idempotency"


def test_cache_is_canonical_bounded_and_ordinarily_evicts() -> None:
    application = _app(cache_entries=1)
    with TestClient(application) as client:
        first = _start(client, role="queen", seed="cache-one")
        second = _start(client, role="king", seed="cache-two")

    first_envelope = RecoveryEnvelope.model_validate(first["envelope"])
    second_envelope = RecoveryEnvelope.model_validate(second["envelope"])
    # Capacity one leaves the newest envelope present and evicts the older one.
    cache = application.state.stateless_gameplay_service.cache
    assert cache.get(first_envelope) is None
    assert cache.get(second_envelope) is not None


def test_stateless_view_rejects_an_unknown_phase() -> None:
    with TestClient(_app()) as client:
        game = _start(client, role="queen", seed="stateless-phase")["game"]
    game["phase"] = {"kind": "waiting"}
    with pytest.raises(ValidationError):
        StatelessHumanGameView.model_validate(game)


def test_history_order_duplicates_and_illegal_divergence_are_rejected() -> None:
    with TestClient(_app()) as client:
        started = _start(client, role="queen", seed="invalid-history")
        premature = client.post(
            "/games/command",
            json={
                "envelope": started["envelope"],
                "command": {"type": "advance_round"},
            },
        )
        assert premature.status_code == 409
        assert premature.json()["code"] == "wrong_phase"

        request = _next_request(started)
        accepted = client.post("/games/command", json=request)
        assert accepted.status_code == 200
        duplicated_history = {
            "seed": accepted.json()["envelope"]["seed"],
            "history": accepted.json()["envelope"]["history"]
            + [request["command"]],
        }
        duplicate = client.post(
            "/games/resume", json={"envelope": duplicated_history}
        )
        assert duplicate.status_code == 422
        assert duplicate.json()["code"] == "invalid_history"

        illegal = client.post(
            "/games/command",
            json={
                "envelope": started["envelope"],
                "command": {"type": "place", "hand_slot": 0, "position": 4},
            },
        )
        assert illegal.status_code == 422
        assert illegal.json()["code"] == "invalid_command"

        reordered = client.post(
            "/games/resume",
            json={
                "envelope": {
                    "seed": "invalid-history",
                    "history": [
                        {"type": "select_role", "human_role": "queen"},
                        {"type": "advance_round"},
                        request["command"],
                    ],
                }
            },
        )
        assert reordered.status_code == 422
        assert reordered.json()["code"] == "invalid_history"


def test_valid_older_envelope_intentionally_forms_a_deterministic_branch() -> None:
    with TestClient(_app()) as client:
        started = _start(client, role="queen", seed="stateless-branch")
        legal = started["game"]["legal_moves"]
        assert len(legal) >= 2
        requests = [
            {
                "envelope": started["envelope"],
                "command": {
                    "type": "place",
                    "hand_slot": move["hand_slot"],
                    "position": move["position"],
                },
            }
            for move in legal[:2]
        ]
        branches = [client.post("/games/command", json=item) for item in requests]
    assert all(response.status_code == 200 for response in branches)
    assert branches[0].json()["envelope"] != branches[1].json()["envelope"]


def test_forced_opponent_placements_bypass_policy_inference() -> None:
    policy = FirstLegalStatelessPolicy()
    with TestClient(_app(policy)) as client:
        response = _start(client, role="queen", seed="forced-bypass")
        while response["game"]["phase"]["kind"] != "game_complete":
            mutation = client.post("/games/command", json=_next_request(response))
            assert mutation.status_code == 200, mutation.json()
            response = mutation.json()

    # Each actor makes four placements per round. The dealer's fourth placement
    # is uniquely forced, so the opponent needs 3 inferences in dealer rounds and
    # 4 in non-dealer rounds: 3 * 3 + 3 * 4 = 21.
    assert len(policy.requests) == 21


def test_public_response_contains_only_envelope_and_human_projection() -> None:
    policy = FirstLegalStatelessPolicy()
    application = _app(policy)
    with TestClient(application) as client:
        response = _start(client, role="king", seed="stateless-privacy")
    payload_text = json.dumps(response, sort_keys=True)
    keys = _all_keys(response)
    assert set(response) == {"envelope", "game"}
    assert set(response["envelope"]) == {"seed", "history"}
    assert not {
        "game_id",
        "instance_id",
        "stock",
        "opponent_hand",
        "engine_state",
        "state_fingerprint",
        "policy",
        "policy_id",
        "policy_version",
        "artifact_id",
        "artifact_sha256",
        "logits",
        "tensor",
        "legal_mask",
        "action_table",
        "hidden_state",
        "search_tree",
    } & keys
    assert all(
        "hand_slot" not in move
        for move in response["game"]["current_round_moves"]
    )

    replayed = application.state.stateless_gameplay_service.replay(
        RecoveryEnvelope.model_validate(response["envelope"])
    )
    opponent = EnginePlayer(response["game"]["opponent_role"])
    for card_id in replayed.state.hands[opponent]:
        if card_id is not None:
            assert f'"{card_id}"' not in payload_text
    for card_id in replayed.state.stock:
        assert f'"{card_id}"' not in payload_text


def test_invalid_policy_output_commits_no_command_or_cache_entry() -> None:
    application = _app(IllegalStatelessPolicy())
    with TestClient(application) as client:
        response = client.post(
            "/games", json={"human_role": "queen", "seed": "policy-failure"}
        )
        if response.status_code == 201:
            body = _next_request(response.json())
            failed_envelope = RecoveryEnvelope.model_validate(
                {
                    "seed": body["envelope"]["seed"],
                    "history": body["envelope"]["history"] + [body["command"]],
                }
            )
            failed = client.post("/games/command", json=body)
            assert failed.status_code == 503
            assert failed.json()["code"] == "dependency_unavailable"
            cache = application.state.stateless_gameplay_service.cache
            assert cache.get(failed_envelope) is None
        else:
            assert response.status_code == 503
            failed_envelope = RecoveryEnvelope.model_validate(
                {
                    "seed": "policy-failure",
                    "history": [
                        {"type": "select_role", "human_role": "queen"},
                    ],
                }
            )
            cache = application.state.stateless_gameplay_service.cache
            assert cache.get(failed_envelope) is None


def test_development_app_exposes_only_stateless_routes() -> None:
    application = _app()
    assert not hasattr(application.state, "game_repository")
    with TestClient(application) as client:
        health = client.get("/health")
        missing_local_route = client.get("/games/00000000-0000-0000-0000-000000000000")
        malformed = client.post(
            "/games",
            json={"human_role": "queen", "seed": "ok", "unknown": True},
        )
    assert health.json() == {
        "status": "ok",
        "opponent_configured": True,
        "narration_enabled": False,
    }
    assert missing_local_route.status_code == 404
    assert malformed.status_code == 422
    assert malformed.json()["code"] == "validation_error"


def test_stateless_configuration_rejects_bad_cache_bounds() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        create_app(replay_cache_entries=-1)


def test_production_entrypoint_cannot_fall_back_to_local_persistence() -> None:
    from dracula.api.production import app

    client = TestClient(app)
    assert client.get("/Dracula/api/health").json()["status"] == "ok"
    # The old session/event routes are not revived by mounting the web app.
    assert client.get("/Dracula/api/sessions").status_code == 404

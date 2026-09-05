"""Exercise local SQLite-backed gameplay transactions and public projections.

The suite covers lifecycle commands, retries, repository conflicts, recorded
sessions, policy calls, and exclusion of private engine state from responses.
"""

from __future__ import annotations

import struct
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient

from dracula.api.app import create_app
from dracula.api.policy import PolicyDescriptor, PolicyTurnResult
from dracula.api.repository import (
    GameRepository,
    InMemoryGameRepository,
    SQLiteGameRepository,
)
from dracula.engine import EnginePlayer, EngineStatus, legal_moves

pytestmark = pytest.mark.filterwarnings(
    "ignore:Using `httpx` with `starlette.testclient` is deprecated"
)


class FirstLegalPolicy:
    """Test-only policy with observable recurrent-state advancement."""

    def __init__(self, repository: GameRepository | None = None) -> None:
        self.repository = repository
        self.pending_was_observed = False
        self.last_policy: PolicyDescriptor | None = None

    def invoke(self, request: Any) -> PolicyTurnResult:
        self.last_policy = request.policy
        if self.repository is not None:
            claimed = self.repository.load(request.game_id)
            self.pending_was_observed = (
                claimed.phase.kind == "opponent_turn"
                and claimed.phase.status == "pending"
                and claimed.engine_state.status is EngineStatus.PLAYING
            )
        action_index = next(
            index
            for index, move in enumerate(request.action_table)
            if move is not None
        )
        values = list(struct.unpack("<128f", request.hidden_state))
        values[0] += 1.0
        return PolicyTurnResult(action_index, struct.pack("<128f", *values))


class InvalidPolicy:
    def invoke(self, request: Any) -> PolicyTurnResult:
        values = list(struct.unpack("<128f", request.hidden_state))
        values[0] = 99.0
        illegal_index = next(
            index
            for index, move in enumerate(request.action_table)
            if move is None
        )
        return PolicyTurnResult(illegal_index, struct.pack("<128f", *values))


@pytest.fixture(params=("memory", "sqlite"))
def repository(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[GameRepository]:
    if request.param == "memory":
        yield InMemoryGameRepository()
        return
    value = SQLiteGameRepository(tmp_path / "games.sqlite3")
    try:
        yield value
    finally:
        value.close()


def _client(
    repository: GameRepository,
    policy: Any | None = None,
    policy_descriptor: PolicyDescriptor | None = None,
) -> TestClient:
    return TestClient(
        create_app(
            repository=repository,
            policy_executor=policy or FirstLegalPolicy(repository),
            policy_descriptor=policy_descriptor,
            narration_enabled=False,
        )
    )


def _start(
    client: TestClient,
    *,
    role: str = "queen",
    seed: str = "gameplay-api-fixture",
    request_id: UUID | None = None,
) -> dict[str, Any]:
    response = client.post(
        "/games",
        json={
            "human_role": role,
            "seed": seed,
            "request_id": str(request_id or uuid4()),
        },
    )
    assert response.status_code == 201, response.json()
    return response.json()


def _mutate_once(client: TestClient, view: dict[str, Any]) -> dict[str, Any]:
    base = f"/games/{view['game_id']}"
    body = {"expected_version": view["version"], "request_id": str(uuid4())}
    if view["phase"]["kind"] == "human_turn":
        body["move_id"] = view["legal_moves"][0]["move_id"]
        response = client.post(f"{base}/moves", json=body)
    elif view["phase"]["kind"] == "opponent_turn":
        response = client.post(f"{base}/opponent-turn", json=body)
        if response.status_code == 202:
            pending = response.json()
            assert pending["phase"]["status"] == "pending"
            response = client.post(f"{base}/opponent-turn", json=body)
    elif view["phase"]["kind"] == "scoring":
        response = client.post(
            f"{base}/rounds/{view['round_number']}/advance", json=body
        )
    else:
        raise AssertionError(f"cannot mutate phase {view['phase']}")
    assert response.status_code == 200, response.json()
    return response.json()


def _reach_human_turn(client: TestClient, view: dict[str, Any]) -> dict[str, Any]:
    while view["phase"]["kind"] != "human_turn":
        view = _mutate_once(client, view)
    return view


def test_local_seed_override_is_reproducible_and_remains_private(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DRACULA_LOCAL_GAME_SEED", "local-browser-fixture-v1")
    repository = InMemoryGameRepository()
    with _client(repository) as client:
        first = client.post(
            "/games",
            json={"human_role": "queen", "request_id": str(uuid4())},
        )
        second = client.post(
            "/games",
            json={"human_role": "queen", "request_id": str(uuid4())},
        )
    assert first.status_code == second.status_code == 201
    first_session = repository.load(UUID(first.json()["game_id"]))
    second_session = repository.load(UUID(second.json()["game_id"]))
    assert first_session.engine_state == second_session.engine_state
    assert "local-browser-fixture-v1" not in first.text
    assert "seed" not in first.json()


@pytest.mark.parametrize("human_role", ("queen", "king"))
def test_creation_projects_only_human_information_and_exact_engine_legality(
    repository: GameRepository, human_role: str
) -> None:
    with _client(repository) as client:
        view = _reach_human_turn(
            client, _start(client, role=human_role, seed=f"role-{human_role}")
        )

    session = repository.load(UUID(view["game_id"]))
    engine_legal = {
        (
            session.engine_state.hands[session.human_role][move.hand_slot],
            move.hand_slot,
            move.global_grid_index,
        )
        for move in legal_moves(session.engine_state, session.human_role)
    }
    api_legal = {
        (move["card_id"], move["hand_slot"], move["position"])
        for move in view["legal_moves"]
    }
    assert view["human_role"] == human_role
    assert api_legal == engine_legal
    assert len({move["move_id"] for move in view["legal_moves"]}) == len(api_legal)
    # Move tokens carry no readable card, slot, position, game, or version data.
    assert all(
        len(move["move_id"]) == 43
        and ":" not in move["move_id"]
        for move in view["legal_moves"]
    )

    def keys(value: Any) -> set[str]:
        if isinstance(value, dict):
            return set(value) | set().union(*(keys(item) for item in value.values()))
        if isinstance(value, list):
            return set().union(*(keys(item) for item in value)) if value else set()
        return set()

    public_keys = keys(view)
    assert not {
        "stock",
        "seed",
        "opponent_hand",
        "policy_hidden_state",
        "hidden_state",
        "model_tensor",
    } & public_keys


def test_idempotency_versions_turns_and_expired_moves_are_rejected_stably(
    repository: GameRepository,
) -> None:
    with _client(repository) as client:
        view = _reach_human_turn(client, _start(client, seed="idempotency"))
        game_id = view["game_id"]
        first = view["legal_moves"][0]

        if view["version"] > 0:
            stale = client.post(
                f"/games/{game_id}/moves",
                json={
                    "move_id": first["move_id"],
                    "expected_version": view["version"] - 1,
                    "request_id": str(uuid4()),
                },
            )
            assert stale.status_code == 409
            assert stale.json()["code"] == "stale_version"

        wrong_turn = client.post(
            f"/games/{game_id}/opponent-turn",
            json={"expected_version": view["version"], "request_id": str(uuid4())},
        )
        assert wrong_turn.status_code == 409
        assert wrong_turn.json()["code"] == "wrong_turn"

        premature_advance = client.post(
            f"/games/{game_id}/rounds/{view['round_number']}/advance",
            json={"expected_version": view["version"], "request_id": str(uuid4())},
        )
        assert premature_advance.status_code == 409
        assert premature_advance.json()["code"] == "wrong_phase"

        request_id = uuid4()
        accepted_payload = {
            "move_id": first["move_id"],
            "expected_version": view["version"],
            "request_id": str(request_id),
        }
        accepted = client.post(f"/games/{game_id}/moves", json=accepted_payload)
        assert accepted.status_code == 200
        repeated = client.post(f"/games/{game_id}/moves", json=accepted_payload)
        assert repeated.status_code == 200
        assert repeated.json() == accepted.json()
        assert repository.load(UUID(game_id)).version == accepted.json()["version"]

        stale = client.post(
            f"/games/{game_id}/opponent-turn",
            json={
                "expected_version": view["version"],
                "request_id": str(uuid4()),
            },
        )
        assert stale.status_code == 409
        assert stale.json()["code"] == "stale_version"

        conflicting = client.post(
            f"/games/{game_id}/moves",
            json={**accepted_payload, "move_id": "different"},
        )
        assert conflicting.status_code == 409
        assert conflicting.json()["code"] == "request_id_conflict"

        after_opponent = _mutate_once(client, accepted.json())
        expired = client.post(
            f"/games/{game_id}/moves",
            json={
                "move_id": first["move_id"],
                "expected_version": after_opponent["version"],
                "request_id": str(uuid4()),
            },
        )
        assert expired.status_code == 422
        assert expired.json()["code"] == "invalid_move"

        malformed = client.post("/games", json={"human_role": "vampire"})
        assert malformed.status_code == 422
        assert malformed.json()["code"] == "validation_error"


def test_policy_claim_is_visible_and_failed_output_commits_neither_move_nor_hidden(
    repository: GameRepository,
) -> None:
    observer = FirstLegalPolicy(repository)
    with _client(repository, observer) as client:
        view = _start(client, seed="policy-claim")
        if view["phase"]["kind"] == "human_turn":
            view = _mutate_once(client, view)
        assert view["phase"]["kind"] == "opponent_turn"
        request_id = str(uuid4())
        request_body = {
            "expected_version": view["version"],
            "request_id": request_id,
        }
        successful = client.post(
            f"/games/{view['game_id']}/opponent-turn",
            json=request_body,
        )
        assert successful.status_code == 202
        assert successful.json()["phase"]["job_id"] == request_id
        reloaded_pending = client.get(f"/games/{view['game_id']}")
        assert reloaded_pending.status_code == 200
        assert reloaded_pending.json() == successful.json()
        successful = client.post(
            f"/games/{view['game_id']}/opponent-turn",
            json=request_body,
        )
        assert successful.status_code == 200
        assert observer.pending_was_observed

    failing_repository = InMemoryGameRepository()
    with _client(failing_repository, InvalidPolicy()) as client:
        view = _start(client, seed="policy-atomic")
        if view["phase"]["kind"] == "human_turn":
            view = _mutate_once(client, view)
        before = failing_repository.load(UUID(view["game_id"]))
        request_body = {
            "expected_version": view["version"],
            "request_id": str(uuid4()),
        }
        response = client.post(
            f"/games/{view['game_id']}/opponent-turn",
            json=request_body,
        )
        assert response.status_code == 202
        claimed = failing_repository.load(UUID(view["game_id"]))
        assert claimed.version == before.version
        assert claimed.engine_state == before.engine_state
        assert claimed.policy_session.hidden_state == before.policy_session.hidden_state
        response = client.post(
            f"/games/{view['game_id']}/opponent-turn", json=request_body
        )
        after = failing_repository.load(UUID(view["game_id"]))

    assert response.status_code == 503
    assert response.json()["code"] == "dependency_unavailable"
    assert response.json()["retryable"] is True
    assert after.version == before.version
    assert after.engine_state == before.engine_state
    assert after.policy_session.hidden_state == before.policy_session.hidden_state
    assert after.phase.kind == "opponent_turn" and after.phase.status == "failed"


def test_complete_seeded_game_is_resumable_and_scoring_matches_engine(
    repository: GameRepository,
) -> None:
    policy = FirstLegalPolicy(repository)
    with _client(repository, policy) as client:
        view = _start(client, seed="six-round-api-replay")
        scoring_views: list[dict[str, Any]] = []
        while view["phase"]["kind"] != "game_complete":
            view = _mutate_once(client, view)
            reloaded = client.get(f"/games/{view['game_id']}")
            assert reloaded.status_code == 200
            assert reloaded.json() == view
            if view["phase"]["kind"] == "scoring":
                scoring_views.append(view)

        event_response = client.get(
            f"/games/{view['game_id']}/events", params={"after_sequence": 2}
        )

    session = repository.load(UUID(view["game_id"]))
    assert session.engine_state.status is EngineStatus.GAME_COMPLETE
    assert len(view["completed_rounds"]) == 6
    assert [len(round_value["moves"]) for round_value in view["completed_rounds"]] == [
        8
    ] * 6
    assert len(scoring_views) == 6
    assert struct.unpack("<f", session.policy_session.hidden_state[:4])[0] == 24.0
    assert event_response.status_code == 200
    assert event_response.json()["latest_sequence"] == len(view["events"])
    assert [event["sequence"] for event in view["events"]] == list(
        range(1, len(view["events"]) + 1)
    )
    assert sum(event["event_type"] == "move_accepted" for event in view["events"]) == 48
    assert view["narration_enabled"] is False
    assert all("narrat" not in event["event_type"] for event in view["events"])

    # The API preserves engine line order (dealer orientation first) and emits
    # values already calculated by the engine rather than recomputing rules in UI code.
    for scoring_view in scoring_views:
        internal = repository.load(UUID(view["game_id"])).engine_state.completed_rounds[
            scoring_view["round_number"] - 1
        ]
        record = scoring_view["pending_round_result"]
        assert record is not None
        expected_lines = tuple(
            line
            for player in (internal.dealer, EnginePlayer.KING if internal.dealer is EnginePlayer.QUEEN else EnginePlayer.QUEEN)
            for line in internal.line_scores[player]
        )
        assert [line["total"] for line in record["line_scores"]] == [
            line.total for line in expected_lines
        ]
        line_steps = [
            step for step in record["scoring_sequence"] if step["kind"] == "score_line"
        ]
        assert [
            (
                step["details"]["value_1"],
                step["details"]["value_2"],
                step["details"]["value_3"],
            )
            for step in line_steps
        ] == [line.values for line in expected_lines]
        for player_offset in (0, 3):
            player_steps = line_steps[player_offset : player_offset + 3]
            ranked = sorted(
                range(3),
                key=lambda index: (-player_steps[index]["line"]["total"], index),
            )
            assert [player_steps[index]["details"]["rank"] for index in ranked] == [1, 2, 3]

    already_advanced = TestClient(
        create_app(repository=repository, policy_executor=policy, narration_enabled=False)
    ).post(
        f"/games/{view['game_id']}/rounds/6/advance",
        json={"expected_version": view["version"], "request_id": str(uuid4())},
    )
    assert already_advanced.status_code == 409
    assert already_advanced.json()["code"] == "already_advanced"


def test_not_found_and_create_idempotency_have_stable_bodies(
    repository: GameRepository,
) -> None:
    with _client(repository) as client:
        missing = client.get(f"/games/{uuid4()}")
        assert missing.status_code == 404
        assert missing.json()["code"] == "not_found"

        request_id = uuid4()
        first = _start(client, request_id=request_id, seed="create-idempotency")
        repeated = client.post(
            "/games",
            json={
                "human_role": "queen",
                "seed": "create-idempotency",
                "request_id": str(request_id),
            },
        )
        assert repeated.status_code == 201
        assert repeated.json() == first
        conflict = client.post(
            "/games",
            json={
                "human_role": "king",
                "seed": "create-idempotency",
                "request_id": str(request_id),
            },
        )
        assert conflict.status_code == 409
        assert conflict.json()["code"] == "request_id_conflict"


def test_unconfigured_live_policy_fails_without_playing_a_substitute_move() -> None:
    repository = InMemoryGameRepository()
    with TestClient(
        create_app(repository=repository, narration_enabled=False)
    ) as client:
        view = _start(client, seed="no-policy-fallback")
        if view["phase"]["kind"] == "human_turn":
            response = client.post(
                f"/games/{view['game_id']}/moves",
                json={
                    "move_id": view["legal_moves"][0]["move_id"],
                    "expected_version": view["version"],
                    "request_id": str(uuid4()),
                },
            )
            assert response.status_code == 200
            view = response.json()
        before = repository.load(UUID(view["game_id"]))
        request = {
            "expected_version": view["version"],
            "request_id": str(uuid4()),
        }
        assert client.post(
            f"/games/{view['game_id']}/opponent-turn", json=request
        ).status_code == 202
        failed = client.post(
            f"/games/{view['game_id']}/opponent-turn", json=request
        )
        after = repository.load(UUID(view["game_id"]))

    assert failed.status_code == 503
    assert after.version == before.version
    assert after.engine_state == before.engine_state
    assert after.policy_session.hidden_state == before.policy_session.hidden_state


def test_sqlite_reopen_resumes_a_claimed_policy_turn(tmp_path: Path) -> None:
    database = tmp_path / "resume.sqlite3"
    first_repository = SQLiteGameRepository(database)
    original_descriptor = PolicyDescriptor(
        policy_id="persisted-policy", policy_version="version-7", artifact_id="archive-7"
    )
    try:
        with _client(
            first_repository,
            FirstLegalPolicy(first_repository),
            original_descriptor,
        ) as client:
            view = _start(client, seed="sqlite-process-resume")
            if view["phase"]["kind"] == "human_turn":
                response = client.post(
                    f"/games/{view['game_id']}/moves",
                    json={
                        "move_id": view["legal_moves"][0]["move_id"],
                        "expected_version": view["version"],
                        "request_id": str(uuid4()),
                    },
                )
                assert response.status_code == 200
                view = response.json()
            request_body = {
                "expected_version": view["version"],
                "request_id": str(uuid4()),
            }
            pending = client.post(
                f"/games/{view['game_id']}/opponent-turn", json=request_body
            )
            assert pending.status_code == 202
            pending_view = pending.json()
    finally:
        first_repository.close()

    second_repository = SQLiteGameRepository(database)
    resumed_policy = FirstLegalPolicy(second_repository)
    try:
        with _client(
            second_repository,
            resumed_policy,
            PolicyDescriptor(
                policy_id="new-default", policy_version="version-8", artifact_id="archive-8"
            ),
        ) as client:
            reloaded = client.get(f"/games/{view['game_id']}")
            assert reloaded.status_code == 200
            assert reloaded.json() == pending_view
            completed = client.post(
                f"/games/{view['game_id']}/opponent-turn", json=request_body
            )
            assert completed.status_code == 200
            assert completed.json()["version"] == view["version"] + 1
            assert resumed_policy.last_policy == original_descriptor
    finally:
        second_repository.close()

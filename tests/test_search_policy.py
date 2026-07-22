"""Local gameplay integration for the information-set search opponent."""

from __future__ import annotations

from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from dracula.api.app import create_app
from dracula.api.repository import InMemoryGameRepository
from dracula.api.service import PolicyTurnRequest
from dracula.api.session import HIDDEN_STATE_BYTES
from dracula.bridge import build_policy_turn_context
from dracula.engine import EnginePlayer, create_game
from dracula.search import information_state_from_engine
from dracula.search_policy import InlineSearchExecutor


def _request(executor: InlineSearchExecutor) -> PolicyTurnRequest:
    state = create_game("search-gameplay-adapter")
    player = state.active_player
    assert player is not None
    context = build_policy_turn_context(state, player)
    return PolicyTurnRequest(
        game_id=uuid4(),
        policy=executor.descriptor,
        turn_number=1,
        player=player,
        round_number=state.round_number,
        turn_kind=context.kind,
        policy_input=context.input,
        action_table=context.action_table,
        information_state=information_state_from_engine(state, player),
        hidden_state=bytes(HIDDEN_STATE_BYTES),
    )


# The executor boundary contains player-visible projections, never EngineState.
def test_search_executor_is_deterministic_legal_and_stateless() -> None:
    executor = InlineSearchExecutor.from_values(32)
    request = _request(executor)

    assert not hasattr(request, "context")
    assert not hasattr(request, "engine_state")
    assert not hasattr(request, "seed")
    first = executor.invoke(request)
    repeated = executor.invoke(request)

    assert first == repeated
    assert first.action_index is not None
    assert request.action_table[first.action_index] is not None
    assert first.hidden_state is None


# FastAPI supplies the sanitized information state and persists the accepted move.
def test_search_executor_completes_an_api_opponent_turn() -> None:
    executor = InlineSearchExecutor.from_values(32)
    repository = InMemoryGameRepository()
    app = create_app(
        repository=repository,
        policy_executor=executor,
        policy_descriptor=executor.descriptor,
        narration_enabled=False,
    )
    with TestClient(app) as client:
        for index in range(100):
            response = client.post(
                "/games",
                json={
                    "human_role": EnginePlayer.QUEEN.value,
                    "seed": f"search-api-{index}",
                    "request_id": str(uuid4()),
                },
            )
            view = response.json()
            if view["phase"]["kind"] == "opponent_turn":
                break
        else:
            raise AssertionError("could not create an opponent-first fixture")

        body = {
            "expected_version": view["version"],
            "request_id": str(uuid4()),
        }
        claimed = client.post(
            f"/games/{view['game_id']}/opponent-turn", json=body
        )
        assert claimed.status_code == 202
        completed = client.post(
            f"/games/{view['game_id']}/opponent-turn", json=body
        )

    assert completed.status_code == 200
    payload = completed.json()
    assert payload["version"] == view["version"] + 1
    assert len(payload["current_round_moves"]) == 1
    serialized = completed.text
    assert "unseen_card_ids" not in serialized
    assert "information_state" not in serialized
    assert "action_visits" not in serialized


# Environment selection changes only the injected opponent, not the HTTP contract.
def test_app_selects_and_validates_search_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DRACULA_OPPONENT_MODE", "search")
    monkeypatch.setenv("DRACULA_SEARCH_SIMULATIONS", "32")
    monkeypatch.setenv("DRACULA_SEARCH_EXPLORATION", "1.25")
    app = create_app(repository=InMemoryGameRepository(), narration_enabled=False)
    descriptor = app.state.gameplay_service.policy_descriptor

    assert descriptor.policy_id == "information-set-search"
    assert descriptor.inference_profile == "max-visits-v1"

    monkeypatch.setenv("DRACULA_SEARCH_SIMULATIONS", "0")
    with pytest.raises(ValueError, match="positive integer"):
        create_app(repository=InMemoryGameRepository(), narration_enabled=False)

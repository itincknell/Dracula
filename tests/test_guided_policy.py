"""Application-boundary tests for the neural-guided search controller."""

from __future__ import annotations

import hashlib
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from dracula.api.app import create_app
from dracula.api.repository import InMemoryGameRepository
from dracula.api.service import PolicyTurnRequest
from dracula.api.session import HIDDEN_STATE_BYTES
from dracula.bridge import build_policy_turn_context
from dracula.engine import EnginePlayer, create_game
from dracula.guided_policy import InlineGuidedSearchExecutor
from dracula.policy_value import PolicyValueModel, save_policy_value_artifact
from dracula.search import GuidedSearchConfig, information_state_from_engine


@pytest.fixture
def policy_value_artifact(tmp_path):
    model = PolicyValueModel(
        run_root_seed="guided-adapter-test",
        model_id="guided-adapter",
        initialization_ordinal=0,
    )
    path = tmp_path / "policy-value.pt"
    save_policy_value_artifact(
        path,
        model,
        source_revision="test",
        training_configuration={"purpose": "guided adapter test"},
        dataset_digest=hashlib.sha256(b"dataset").hexdigest(),
        search_report_digest=hashlib.sha256(b"search").hexdigest(),
    )
    return path


def _request(executor: InlineGuidedSearchExecutor) -> PolicyTurnRequest:
    state = create_game("guided-gameplay-adapter")
    player = state.active_player
    assert player is not None
    context = build_policy_turn_context(state, player)
    return PolicyTurnRequest(
        game_id=uuid4(),
        policy=executor.descriptor,
        turn_number=1,
        player=player,
        round_number=1,
        turn_kind=context.kind,
        policy_input=context.input,
        action_table=context.action_table,
        information_state=information_state_from_engine(state, player),
        hidden_state=bytes(HIDDEN_STATE_BYTES),
    )


# The live adapter accepts only the same sanitized information-state boundary as search.
def test_guided_executor_is_deterministic_legal_and_stateless(policy_value_artifact) -> None:
    executor = InlineGuidedSearchExecutor(
        policy_value_artifact, GuidedSearchConfig(simulation_budget=20)
    )
    request = _request(executor)

    first = executor.invoke(request)
    repeated = executor.invoke(request)

    assert first == repeated
    assert first.action_index is not None
    assert request.action_table[first.action_index] is not None
    assert first.hidden_state is None
    assert str(policy_value_artifact) not in repr(executor.descriptor)


# FastAPI must fail explicitly when guided mode has no compatible artifact.
def test_app_guided_mode_is_explicit_and_keeps_private_model_data_out_of_responses(
    monkeypatch: pytest.MonkeyPatch, policy_value_artifact
) -> None:
    monkeypatch.setenv("DRACULA_OPPONENT_MODE", "guided")
    monkeypatch.setenv("DRACULA_POLICY_VALUE_ARTIFACT", str(policy_value_artifact))
    monkeypatch.setenv("DRACULA_GUIDED_SIMULATIONS", "20")
    app = create_app(repository=InMemoryGameRepository(), narration_enabled=False)
    assert app.state.gameplay_service.policy_descriptor.policy_id == "guided-information-set-search"

    with TestClient(app) as client:
        response = client.post(
            "/games",
            json={
                "human_role": EnginePlayer.QUEEN.value,
                "seed": "guided-api-privacy",
                "request_id": str(uuid4()),
            },
        )
    assert response.status_code == 201
    serialized = response.text
    assert str(policy_value_artifact) not in serialized
    assert "policy_logits" not in serialized
    assert "search_tree" not in serialized
    assert "unseen_card_ids" not in serialized

    monkeypatch.delenv("DRACULA_POLICY_VALUE_ARTIFACT")
    with pytest.raises(ValueError, match="DRACULA_POLICY_VALUE_ARTIFACT"):
        create_app(repository=InMemoryGameRepository(), narration_enabled=False)

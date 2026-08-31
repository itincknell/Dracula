"""Gameplay-boundary tests for the standalone Sam policy controller."""

from __future__ import annotations

from pathlib import Path
from uuid import UUID, uuid4

import pytest
import torch
from fastapi.testclient import TestClient

from dracula.api.app import create_app
from dracula.api.repository import InMemoryGameRepository
from dracula.api.service import PolicyTurnRequest
from dracula.api.session import HIDDEN_STATE_BYTES
from dracula.bridge import PolicyTurnKind, build_policy_turn_context
from dracula.engine import EnginePlayer, EngineStatus, other_player
from dracula.sam_policy import (
    SamPolicyModel,
    load_sam_policy_artifact,
    save_sam_policy_artifact,
)
from dracula.search import information_state_from_engine
from dracula.standalone_policy import StandaloneSamPolicyExecutor


@pytest.fixture
def policy_artifact(tmp_path: Path) -> Path:
    path = tmp_path / "standalone-policy.pt"
    save_sam_policy_artifact(
        path,
        SamPolicyModel(
            run_root_seed="standalone-policy-test",
            model_id="standalone-policy-test",
            initialization_ordinal=0,
        ),
        source_revision="1" * 40,
        source_tree_digest="2" * 64,
        training_configuration={"test": True},
        corpus_snapshot_digest="3" * 64,
    )
    return path


def _request_for_session(
    executor: StandaloneSamPolicyExecutor,
    repository: InMemoryGameRepository,
    game_id: UUID,
) -> PolicyTurnRequest:
    session = repository.load(game_id)
    player = other_player(session.human_role)
    context = build_policy_turn_context(session.engine_state, player)
    return PolicyTurnRequest(
        game_id=game_id,
        policy=executor.descriptor,
        turn_number=len(session.engine_state.current_round_moves) + 1,
        player=player,
        round_number=session.engine_state.round_number,
        turn_kind=context.kind,
        policy_input=context.input,
        action_table=context.action_table,
        information_state=information_state_from_engine(
            session.engine_state, player
        ),
        hidden_state=bytes(HIDDEN_STATE_BYTES),
    )


# The exported file is strict, finite, and produces the exact direct-model
# tensor used by the gameplay executor.
def test_artifact_direct_and_exported_inference_are_exact(
    policy_artifact: Path,
) -> None:
    first = load_sam_policy_artifact(policy_artifact)
    second = load_sam_policy_artifact(policy_artifact)
    observation = torch.zeros(875, dtype=torch.bool)

    with torch.inference_mode():
        direct = first.model(observation)
        exported = second.model(observation)

    assert torch.equal(direct, exported)
    assert direct.shape == (4, 8)
    assert direct.dtype is torch.float32
    assert torch.isfinite(direct).all()


# Selecting the standalone mode is explicit, validates the artifact at startup,
# and does not alias the nested-search comparison controller.
def test_environment_selects_standalone_policy(
    monkeypatch: pytest.MonkeyPatch,
    policy_artifact: Path,
) -> None:
    monkeypatch.setenv("DRACULA_OPPONENT_MODE", "sam-policy")
    monkeypatch.setenv(
        "DRACULA_SAM_POLICY_ARTIFACT", str(policy_artifact)
    )
    app = create_app(
        repository=InMemoryGameRepository(), narration_enabled=False
    )
    executor = app.state.gameplay_service.policy_executor
    descriptor = app.state.gameplay_service.policy_descriptor

    assert isinstance(executor, StandaloneSamPolicyExecutor)
    assert descriptor.policy_id == "standalone-sam-policy"
    assert descriptor.inference_profile == (
        "representative-argmax-fair-coin-v1"
    )
    assert descriptor.artifact_id == "standalone-policy-test"
    assert descriptor.artifact_sha256 == "none"

    monkeypatch.delenv("DRACULA_SAM_POLICY_ARTIFACT")
    with pytest.raises(ValueError, match="must be a nonempty path"):
        create_app(
            repository=InMemoryGameRepository(),
            narration_enabled=False,
        )


# A claimed turn is immutable while inference runs. Repeating the accepted
# request returns the original response and cannot execute a second move.
def test_standalone_policy_retry_is_idempotent_and_private(
    policy_artifact: Path,
) -> None:
    executor = StandaloneSamPolicyExecutor(policy_artifact)
    repository = InMemoryGameRepository()
    app = create_app(
        repository=repository,
        policy_executor=executor,
        policy_descriptor=executor.descriptor,
        narration_enabled=False,
    )
    with TestClient(app) as client:
        for ordinal in range(100):
            response = client.post(
                "/games",
                json={
                    "human_role": "queen",
                    "seed": f"standalone-opponent-first-{ordinal}",
                    "request_id": str(uuid4()),
                },
            )
            view = response.json()
            if view["phase"]["kind"] == "opponent_turn":
                break
        else:
            raise AssertionError("no opponent-first test fixture found")

        request = {
            "expected_version": view["version"],
            "request_id": str(uuid4()),
        }
        claimed = client.post(
            f"/games/{view['game_id']}/opponent-turn", json=request
        )
        assert claimed.status_code == 202
        before_completion = repository.load(UUID(view["game_id"]))
        completed = client.post(
            f"/games/{view['game_id']}/opponent-turn", json=request
        )
        repeated = client.post(
            f"/games/{view['game_id']}/opponent-turn", json=request
        )

    assert completed.status_code == 200
    assert repeated.status_code == 200
    assert repeated.json() == completed.json()
    accepted = repository.load(UUID(view["game_id"]))
    assert accepted.version == before_completion.version + 1
    assert len(accepted.engine_state.current_round_moves) == 1

    public = completed.text.lower()
    forbidden = (
        str(policy_artifact.resolve()).lower(),
        executor.artifact_sha256,
        "raw_logits",
        "legal_mask",
        "representative_mask",
        "information_state",
        "hidden_state_bytes",
        "opponent_hand",
        "stock_order",
        "engine_seed",
        "state_dict",
    )
    assert all(value not in public for value in forbidden)


def _play_complete_api_game(
    policy_artifact: Path, human_role: EnginePlayer
) -> tuple[dict[str, object], int, int]:
    executor = StandaloneSamPolicyExecutor(policy_artifact)
    repository = InMemoryGameRepository()
    app = create_app(
        repository=repository,
        policy_executor=executor,
        policy_descriptor=executor.descriptor,
        narration_enabled=False,
    )
    opponent_turns = 0
    forced_turns = 0
    with TestClient(app) as client:
        response = client.post(
            "/games",
            json={
                "human_role": human_role.value,
                "seed": f"standalone-complete-{human_role.value}",
                "request_id": str(uuid4()),
            },
        )
        assert response.status_code == 201
        view = response.json()
        while view["phase"]["kind"] != "game_complete":
            game_id = view["game_id"]
            phase = view["phase"]["kind"]
            if phase == "human_turn":
                response = client.post(
                    f"/games/{game_id}/moves",
                    json={
                        "move_id": view["legal_moves"][0]["move_id"],
                        "expected_version": view["version"],
                        "request_id": str(uuid4()),
                    },
                )
            elif phase == "opponent_turn":
                request = _request_for_session(
                    executor, repository, UUID(game_id)
                )
                if (
                    request.turn_kind
                    is PolicyTurnKind.FORCED_RECURRENT_TRANSITION
                ):
                    forced_turns += 1
                body = {
                    "expected_version": view["version"],
                    "request_id": str(uuid4()),
                }
                claimed = client.post(
                    f"/games/{game_id}/opponent-turn", json=body
                )
                assert claimed.status_code == 202
                response = client.post(
                    f"/games/{game_id}/opponent-turn", json=body
                )
                opponent_turns += 1
            else:
                response = client.post(
                    f"/games/{game_id}/rounds/{view['round_number']}/advance",
                    json={
                        "expected_version": view["version"],
                        "request_id": str(uuid4()),
                    },
                )
            assert response.status_code == 200, response.json()
            view = response.json()

    session = repository.load(UUID(view["game_id"]))
    assert session.engine_state.status is EngineStatus.GAME_COMPLETE
    return view, opponent_turns, forced_turns


@pytest.mark.parametrize("human_role", tuple(EnginePlayer))
def test_complete_seeded_api_game_is_legal_for_both_roles(
    policy_artifact: Path,
    human_role: EnginePlayer,
) -> None:
    view, opponent_turns, forced_turns = _play_complete_api_game(
        policy_artifact, human_role
    )

    assert view["status"] == "game_complete"
    assert len(view["completed_rounds"]) == 6
    assert opponent_turns == 24
    assert forced_turns == 3

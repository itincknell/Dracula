"""Verify the minimal π1 artifact and its gameplay adapter."""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path
import pytest
import torch

import dracula.policy.runtime as active_policy_module
from dracula.policy.runtime import ActivePolicyExecutor, ActivePolicyRuntime
from dracula.policy.contracts import (
    PolicyTurnRequest,
    PolicyTurnResult,
)
from dracula.policy.artifact import load_policy_artifact, save_policy_artifact
from dracula.policy.model import (
    PolicyModel,
    PolicyModelError,
)
from dracula.decision.bridge import build_policy_turn_context, move_for_action_index
from dracula.game.engine import (
    EnginePlayer,
    EngineStatus,
    advance_after_round,
    apply_move,
    create_game,
    legal_moves,
)
from dracula.policy.observation import encode_policy_observation
from dracula.decision.information import information_state_from_engine
from dracula.decision.strategic_actions import strategic_action_groups

ROOT = Path(__file__).resolve().parents[1]
SELECTED_ARTIFACT = ROOT / "runs/bgc-policy-pi1-001/artifacts/pi1-policy.pt"


@pytest.fixture()
def policy_artifact(tmp_path: Path) -> tuple[Path, PolicyModel]:
    model = PolicyModel(404)
    path = tmp_path / "policy.pt"
    save_policy_artifact(path, model)
    return path, model


@pytest.mark.parametrize("content", (b"", b"truncated"), ids=("empty", "truncated"))
def test_unreadable_artifacts_are_rejected(tmp_path: Path, content: bytes) -> None:
    path = tmp_path / "bad.pt"
    path.write_bytes(content)
    with pytest.raises(PolicyModelError, match="could not be loaded"):
        load_policy_artifact(path)


def test_wrong_format_and_tensor_keys_are_rejected(tmp_path: Path) -> None:
    wrong_format = tmp_path / "wrong-format.pt"
    torch.save({"format": "something-else", "state_dict": {}}, wrong_format)
    with pytest.raises(PolicyModelError, match="format is incompatible"):
        load_policy_artifact(wrong_format)

    missing_key = tmp_path / "missing-key.pt"
    state = PolicyModel().state_dict()
    state.pop(next(iter(state)))
    torch.save({"format": "pi1-policy-v1", "state_dict": state}, missing_key)
    with pytest.raises(PolicyModelError, match="keys are incompatible"):
        load_policy_artifact(missing_key)


@pytest.mark.parametrize("kind", ("non-finite", "wrong-shape"))
def test_incompatible_tensors_are_rejected(
    policy_artifact: tuple[Path, PolicyModel], kind: str
) -> None:
    path, _ = policy_artifact
    payload = torch.load(path, map_location="cpu", weights_only=True)
    tensor = payload["state_dict"]["card_embedding.weight"].clone()
    if kind == "non-finite":
        tensor.flatten()[0] = torch.nan
    else:
        tensor = tensor.flatten()
    payload["state_dict"]["card_embedding.weight"] = tensor
    modified = path.with_name("modified.pt")
    torch.save(payload, modified)
    with pytest.raises(PolicyModelError, match="tensor is incompatible"):
        load_policy_artifact(modified)


def test_save_load_preserves_logits_and_loads_only_once(
    policy_artifact: tuple[Path, PolicyModel],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path, original = policy_artifact
    loaded = load_policy_artifact(path)
    information = information_state_from_engine(create_game("artifact-round-trip"))
    observation = encode_policy_observation(information)
    with torch.inference_mode():
        assert torch.equal(original(observation), loaded.model(observation))

    load_count = 0
    real_loader = active_policy_module.load_policy_artifact

    def counted_loader(artifact_path: Path):
        nonlocal load_count
        load_count += 1
        return real_loader(artifact_path)

    monkeypatch.setattr(active_policy_module, "load_policy_artifact", counted_loader)
    runtime = ActivePolicyRuntime.from_artifact(path)
    first = runtime.decide(information, fixture_id="round-trip", decision_index=1)
    second = runtime.decide(information, fixture_id="round-trip", decision_index=1)
    assert load_count == 1
    assert first.representative_action_index == second.representative_action_index
    assert first.concrete_action_index == second.concrete_action_index


def test_executor_accepts_an_actor_visible_gameplay_request(
    policy_artifact: tuple[Path, PolicyModel],
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level("INFO", logger=active_policy_module.LOGGER.name)
    path, _ = policy_artifact
    executor = ActivePolicyExecutor(path)
    state = create_game("unbound-stateless-artifact")
    information = information_state_from_engine(state)
    context = build_policy_turn_context(state, information.player)

    result = executor.invoke(
        PolicyTurnRequest(
            game_key="queen:artifact-gameplay-request",
            turn_number=information.turn_number,
            player=information.player,
            round_number=information.round_number,
            action_table=context.action_table,
            information_state=information,
        )
    )

    assert result.action_index is not None
    assert context.action_table[result.action_index] is not None
    assert "decision_latency_seconds=" in caplog.text
    assert "artifact-gameplay-request" not in caplog.text


@pytest.mark.skipif(not SELECTED_ARTIFACT.is_file(), reason="local pi1 artifact absent")
def test_selected_artifact_executes_one_actor_visible_decision() -> None:
    executor = ActivePolicyExecutor(SELECTED_ARTIFACT)
    state = create_game("selected-artifact-executor")
    assert state.active_player is not None
    context = build_policy_turn_context(state, state.active_player)
    information = information_state_from_engine(state)
    result = executor.invoke(
        PolicyTurnRequest(
            game_key="selected-artifact-game",
            turn_number=information.turn_number,
            player=information.player,
            round_number=information.round_number,
            action_table=context.action_table,
            information_state=information,
        )
    )
    assert isinstance(result, PolicyTurnResult)
    assert result.action_index is not None
    assert context.action_table[result.action_index] is not None
    assert {field.name for field in fields(result)} == {"action_index"}


@pytest.mark.skipif(not SELECTED_ARTIFACT.is_file(), reason="local pi1 artifact absent")
@pytest.mark.parametrize("seed", ("stateless-queen-6", "stateless-king-0"))
def test_selected_artifact_is_legal_through_complete_games(seed: str) -> None:
    runtime = ActivePolicyRuntime.from_artifact(SELECTED_ARTIFACT)
    state = create_game(seed)
    players_by_placement = {placement: set() for placement in range(1, 8)}
    while state.status is not EngineStatus.GAME_COMPLETE:
        if state.status is EngineStatus.ROUND_COMPLETE:
            state = advance_after_round(state)
            continue
        assert state.active_player is not None
        placement = len(state.current_round_moves) + 1
        legal = legal_moves(state, state.active_player)
        if placement == 8:
            state = apply_move(state, legal[0]).state
            continue
        information = information_state_from_engine(state)
        decision = runtime.decide(
            information,
            fixture_id=seed,
            decision_index=(state.round_number - 1) * 7 + placement,
        )
        group = next(
            group
            for group in strategic_action_groups(information)
            if group.representative_action_index
            == decision.representative_action_index
        )
        assert decision.concrete_action_index in group.member_action_indices
        move = move_for_action_index(state.active_player, decision.concrete_action_index)
        assert move in legal
        players_by_placement[placement].add(state.active_player)
        state = apply_move(state, move).state
    assert all(players == set(EnginePlayer) for players in players_by_placement.values())

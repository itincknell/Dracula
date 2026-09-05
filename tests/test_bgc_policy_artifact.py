"""Verify strict policy artifact loading and runtime ownership.

Malformed envelopes, metadata drift, tensor corruption, and digest mismatches
must fail at load time; verified models must not be reloaded during inference.
"""

from __future__ import annotations

import copy
from dataclasses import fields
from pathlib import Path
from uuid import UUID

import pytest
import torch

import dracula.active_policy as active_policy_module
from dracula.active_policy import ActivePolicyExecutor, ActivePolicyRuntime
from dracula.api.policy import PolicyTurnRequest, PolicyTurnResult, zero_hidden_state
from dracula.bgc_policy import (
    load_bgc_policy_artifact,
    save_bgc_policy_artifact,
)
from dracula.bgc_policy_model import BGCPolicyModel, BGCPolicyModelError
from dracula.bridge import build_policy_turn_context, move_for_action_index
from dracula.engine import (
    EnginePlayer,
    EngineStatus,
    advance_after_round,
    apply_move,
    create_game,
    legal_moves,
)
from dracula.policy_observation import encode_policy_observation
from dracula.search.information import information_state_from_engine
from dracula.strategic_actions import strategic_action_groups

ROOT = Path(__file__).resolve().parents[1]
SELECTED_ARTIFACT = (
    ROOT / "runs/bgc-policy-pi1-001/artifacts/unaccepted-candidate.pt"
)


@pytest.fixture()
def policy_artifact(tmp_path: Path) -> tuple[Path, BGCPolicyModel]:
    model = BGCPolicyModel(
        run_root_seed="policy-artifact-test",
        model_id="pi1-test",
        initialization_ordinal=0,
    )
    path = tmp_path / "policy.pt"
    save_bgc_policy_artifact(
        path,
        model,
        source_revision="a" * 40,
        source_tree_digest="b" * 64,
        training_configuration={"run": {"id": "policy-artifact-test"}},
        corpus_snapshot_digest="c" * 64,
        dataset_digest="d" * 64,
    )
    return path, model


def _payload(path: Path) -> dict[str, object]:
    loaded = torch.load(path, map_location="cpu", weights_only=True)
    assert isinstance(loaded, dict)
    return copy.deepcopy(loaded)


def _write_payload(path: Path, payload: object) -> Path:
    destination = path.with_name("modified.pt")
    torch.save(payload, destination)
    return destination


@pytest.mark.parametrize("content", (b"", b"truncated"), ids=("empty", "truncated"))
def test_truncated_artifacts_are_rejected_by_the_artifact_boundary(
    tmp_path: Path, content: bytes
) -> None:
    path = tmp_path / "truncated.pt"
    path.write_bytes(content)
    with pytest.raises(BGCPolicyModelError, match="could not be loaded"):
        load_bgc_policy_artifact(path)


def test_malformed_loaded_payload_is_rejected(policy_artifact) -> None:
    path, _model = policy_artifact
    malformed = _write_payload(path, ["not", "a", "policy", "artifact"])
    with pytest.raises(BGCPolicyModelError, match="payload is invalid"):
        load_bgc_policy_artifact(malformed)


@pytest.mark.parametrize(
    ("kind", "message"),
    (
        ("metadata", "observation_schema_version is incompatible"),
        ("configuration", "training configuration digest differs"),
        ("state-dict", "state-dict digest differs"),
    ),
)
def test_artifact_identity_mismatches_are_rejected(
    policy_artifact,
    kind: str,
    message: str,
) -> None:
    path, _model = policy_artifact
    payload = _payload(path)
    metadata = payload["metadata"]
    state_dict = payload["state_dict"]
    assert isinstance(metadata, dict)
    assert isinstance(state_dict, dict)
    if kind == "metadata":
        metadata["observation_schema_version"] = "incompatible-observation"
    elif kind == "configuration":
        payload["training_configuration"] = {"run": {"id": "changed"}}
    else:
        tensor = state_dict["card_embedding.weight"].clone()
        tensor.flatten()[0] += 1.0
        state_dict["card_embedding.weight"] = tensor

    with pytest.raises(BGCPolicyModelError, match=message):
        load_bgc_policy_artifact(_write_payload(path, payload))


@pytest.mark.parametrize("kind", ("non-finite", "wrong-shape"))
def test_incompatible_artifact_tensors_are_rejected(
    policy_artifact,
    kind: str,
) -> None:
    path, _model = policy_artifact
    payload = _payload(path)
    state_dict = payload["state_dict"]
    assert isinstance(state_dict, dict)
    tensor = state_dict["card_embedding.weight"].clone()
    if kind == "non-finite":
        tensor.flatten()[0] = torch.nan
    else:
        tensor = tensor.flatten()
    state_dict["card_embedding.weight"] = tensor

    with pytest.raises(BGCPolicyModelError, match="artifact tensor is incompatible"):
        load_bgc_policy_artifact(_write_payload(path, payload))


def test_save_load_preserves_logits_actions_and_verifies_only_at_load(
    policy_artifact,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path, original_model = policy_artifact
    loaded = load_bgc_policy_artifact(path)
    information = information_state_from_engine(create_game("artifact-round-trip"))
    observation = encode_policy_observation(information)
    with torch.inference_mode():
        assert torch.equal(original_model(observation), loaded.model(observation))

    direct_runtime = ActivePolicyRuntime(
        original_model,
        artifact_digest=loaded.artifact_digest,
    )
    direct = direct_runtime.decide(
        information,
        fixture_id="artifact-round-trip",
        decision_index=1,
    )

    load_count = 0
    real_loader = active_policy_module.load_bgc_policy_artifact

    def counted_loader(artifact_path):
        nonlocal load_count
        load_count += 1
        return real_loader(artifact_path)

    monkeypatch.setattr(
        active_policy_module,
        "load_bgc_policy_artifact",
        counted_loader,
    )
    runtime = ActivePolicyRuntime.from_artifact(path)
    first = runtime.decide(
        information,
        fixture_id="artifact-round-trip",
        decision_index=1,
    )
    second = runtime.decide(
        information,
        fixture_id="artifact-round-trip",
        decision_index=1,
    )
    assert load_count == 1
    assert (first.representative_action_index, first.concrete_action_index) == (
        direct.representative_action_index,
        direct.concrete_action_index,
    )
    assert (second.representative_action_index, second.concrete_action_index) == (
        first.representative_action_index,
        first.concrete_action_index,
    )


@pytest.mark.skipif(not SELECTED_ARTIFACT.is_file(), reason="local pi1 artifact absent")
def test_selected_artifact_executes_one_private_decision_into_a_minimal_result() -> None:
    executor = ActivePolicyExecutor(SELECTED_ARTIFACT)
    state = create_game("selected-artifact-executor")
    assert state.active_player is not None
    context = build_policy_turn_context(state, state.active_player)
    information = information_state_from_engine(state)
    result = executor.invoke(
        PolicyTurnRequest(
            game_id=UUID("00000000-0000-0000-0000-000000000001"),
            policy=executor.descriptor,
            turn_number=information.turn_number,
            player=information.player,
            round_number=information.round_number,
            turn_kind=context.kind,
            policy_input=None,
            action_table=context.action_table,
            information_state=information,
            hidden_state=zero_hidden_state(),
        )
    )
    assert isinstance(result, PolicyTurnResult)
    assert result.action_index is not None
    assert context.action_table[result.action_index] is not None
    assert result.hidden_state is None
    assert {field.name for field in fields(result)} == {"action_index", "hidden_state"}


@pytest.mark.skipif(not SELECTED_ARTIFACT.is_file(), reason="local pi1 artifact absent")
@pytest.mark.parametrize("seed", ("stateless-queen-6", "stateless-king-0"))
def test_selected_artifact_is_legal_at_every_learned_placement_for_both_roles(
    seed: str,
) -> None:
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
        groups = strategic_action_groups(information)
        selected_group = next(
            group
            for group in groups
            if group.representative_action_index
            == decision.representative_action_index
        )
        assert decision.concrete_action_index in selected_group.member_action_indices
        move = move_for_action_index(
            state.active_player,
            decision.concrete_action_index,
        )
        assert move in legal
        players_by_placement[placement].add(state.active_player)
        state = apply_move(state, move).state

    assert all(players == set(EnginePlayer) for players in players_by_placement.values())

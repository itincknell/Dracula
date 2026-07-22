"""Archived-policy loading, stateless inference, and API integration tests."""

from __future__ import annotations

import math
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable
from uuid import UUID, uuid4

import pytest
import torch
from fastapi.testclient import TestClient

from dracula.api.app import create_app
from dracula.api.repository import InMemoryGameRepository
from dracula.api.service import PolicyTurnRequest
from dracula.api.session import HIDDEN_STATE_BYTES
from dracula.bridge import (
    PolicyTurnKind,
    apply_policy_action,
    build_policy_turn_context,
)
from dracula.collection import module_fingerprint
from dracula.engine import EnginePlayer, EngineStatus, create_game, other_player
from dracula.local_policy import (
    InlinePolicyExecutor,
    LocalTorchPolicyAdapter,
    PolicyArtifactError,
    hidden_bytes_to_tensor,
    hidden_tensor_to_bytes,
    serving_contract,
)
from dracula.models import Policy
from dracula.policy_adapter import (
    HIDDEN_STATE_SCHEMA_VERSION,
    POLICY_ARCHIVE_FORMAT_VERSION,
    POLICY_INFERENCE_CONTRACT_VERSION,
    PolicyContractError,
    PolicyInferenceRequest,
    resolve_inference_profile,
    select_masked_action,
)
from dracula.training_config import VersionSettings


def _archive_payload(seed: int = 41) -> dict[str, Any]:
    policy = Policy(seed=seed).cpu().eval()
    state_dict = {
        name: tensor.detach().cpu().clone()
        for name, tensor in policy.state_dict().items()
    }
    contracts = asdict(VersionSettings())
    return {
        "format_version": POLICY_ARCHIVE_FORMAT_VERSION,
        "run_id": "candidate-test-run",
        "policy": {
            "policy_id": "candidate-policy",
            "version": "candidate-policy-v7",
            "parameter_fingerprint": module_fingerprint(policy),
            "state_dict": state_dict,
        },
        "contracts": contracts,
        "serving_contract": serving_contract(),
        "resolved_manifest": {"versions": contracts},
        "resolved_manifest_hash": "0" * 64,
        "comparison_report": None,
    }


@pytest.fixture
def policy_archive(tmp_path: Path) -> Path:
    path = tmp_path / "candidate.pt"
    torch.save(_archive_payload(), path)
    return path


def _adapter_request(
    adapter: LocalTorchPolicyAdapter,
    *,
    seed: str = "adapter-forward",
    hidden_state: bytes | None = None,
) -> tuple[Any, PolicyInferenceRequest]:
    state = create_game(seed)
    assert state.active_player is not None
    context = build_policy_turn_context(state, state.active_player)
    request = PolicyInferenceRequest(
        contract_version=POLICY_INFERENCE_CONTRACT_VERSION,
        artifact_id=adapter.metadata.artifact_id,
        observation=tuple(
            bool(value) for value in context.input.observation.tolist()
        ),
        legal_mask=tuple(
            tuple(bool(value) for value in row)
            for row in context.input.legal_mask.tolist()
        ),
        hidden_state=hidden_state or bytes(HIDDEN_STATE_BYTES),
    )
    return context, request


def test_adapter_matches_direct_policy_forward_and_hidden_bytes_round_trip(
    policy_archive: Path,
) -> None:
    adapter = LocalTorchPolicyAdapter(policy_archive)
    context, request = _adapter_request(adapter)
    response = adapter.invoke(request)

    payload = torch.load(policy_archive, map_location="cpu", weights_only=True)
    direct = Policy(seed=0).cpu().eval()
    direct.load_state_dict(payload["policy"]["state_dict"], strict=True)
    with torch.inference_mode():
        expected_logits, expected_hidden = direct(
            context.input.observation,
            context.input.legal_mask,
            hidden_bytes_to_tensor(request.hidden_state),
        )

    assert torch.equal(torch.tensor(response.raw_logits), expected_logits)
    assert response.next_hidden_state == hidden_tensor_to_bytes(expected_hidden)
    values = torch.linspace(-2.0, 2.0, 128, dtype=torch.float32)
    assert torch.equal(hidden_bytes_to_tensor(hidden_tensor_to_bytes(values)), values)
    assert adapter.metadata.parameter_count == 443_145
    assert adapter.metadata.hidden_state_schema_version == HIDDEN_STATE_SCHEMA_VERSION


def _write_mutated_archive(
    tmp_path: Path,
    name: str,
    mutate: Callable[[dict[str, Any]], None],
) -> Path:
    payload = _archive_payload()
    mutate(payload)
    path = tmp_path / f"{name}.pt"
    torch.save(payload, path)
    return path


@pytest.mark.parametrize(
    ("name", "mutate"),
    [
        ("format", lambda value: value.update(format_version="wrong-v1")),
        (
            "architecture",
            lambda value: value["contracts"].update(policy="wrong-policy-v1"),
        ),
        (
            "observation",
            lambda value: value["contracts"].update(observation="wrong-observation-v1"),
        ),
        (
            "action",
            lambda value: value["contracts"].update(action_map="wrong-action-v1"),
        ),
        (
            "hidden",
            lambda value: value["serving_contract"].update(
                hidden_state_schema_version="wrong-hidden-v1"
            ),
        ),
        (
            "parameter-count",
            lambda value: value["serving_contract"].update(parameter_count=1),
        ),
        (
            "missing-state-key",
            lambda value: value["policy"]["state_dict"].pop(
                "pair_output.bias"
            ),
        ),
        (
            "wrong-shape",
            lambda value: value["policy"]["state_dict"].update(
                {"pair_output.bias": torch.zeros(2)}
            ),
        ),
        (
            "non-finite-weight",
            lambda value: value["policy"]["state_dict"][
                "pair_output.bias"
            ].fill_(math.nan),
        ),
        (
            "fingerprint",
            lambda value: value["policy"].update(parameter_fingerprint="0" * 64),
        ),
    ],
)
def test_invalid_policy_archives_are_rejected(
    tmp_path: Path,
    name: str,
    mutate: Callable[[dict[str, Any]], None],
) -> None:
    path = _write_mutated_archive(tmp_path, name, mutate)
    with pytest.raises(PolicyArtifactError):
        LocalTorchPolicyAdapter(path)


def test_invalid_artifact_path_and_non_finite_model_output_are_rejected(
    tmp_path: Path, policy_archive: Path
) -> None:
    with pytest.raises(PolicyArtifactError):
        LocalTorchPolicyAdapter(tmp_path / "missing.pt")

    adapter = LocalTorchPolicyAdapter(policy_archive)
    _, request = _adapter_request(adapter)
    with torch.no_grad():
        adapter._policy.pair_output.bias.fill_(math.nan)
    with pytest.raises(PolicyContractError, match="raw logits"):
        adapter.invoke(request)


def test_action_selection_hard_masks_illegal_logits_and_sampling_is_repeatable() -> None:
    logits = tuple(
        tuple(1_000.0 if (row, column) == (0, 0) else float(row * 8 + column)
              for column in range(8))
        for row in range(4)
    )
    mask = tuple(
        tuple((row, column) in {(2, 3), (3, 7)} for column in range(8))
        for row in range(4)
    )
    arguments = {
        "game_id": "00000000-0000-4000-8000-000000000001",
        "round_number": 2,
        "turn_number": 5,
        "artifact_id": "sha256:" + "1" * 64,
    }
    assert select_masked_action(
        logits, mask, resolve_inference_profile("argmax-v1"), **arguments
    ) == 31
    sampled = select_masked_action(
        logits,
        mask,
        resolve_inference_profile("sample-temperature-1-v1"),
        **arguments,
    )
    assert sampled in {19, 31}
    assert sampled == select_masked_action(
        logits,
        mask,
        resolve_inference_profile("sample-temperature-1-v1"),
        **arguments,
    )


def _state_with_active_player(player: EnginePlayer) -> Any:
    for index in range(100):
        state = create_game(f"policy-orientation-{player.value}-{index}")
        if state.active_player is player:
            return state
    raise AssertionError("could not find a deterministic orientation fixture")


@pytest.mark.parametrize("policy_player", tuple(EnginePlayer))
def test_queen_and_king_policy_actions_map_to_the_correct_global_move(
    policy_archive: Path, policy_player: EnginePlayer
) -> None:
    adapter = LocalTorchPolicyAdapter(policy_archive)
    executor = InlinePolicyExecutor(adapter, resolve_inference_profile("argmax-v1"))
    state = _state_with_active_player(policy_player)
    context = build_policy_turn_context(state, policy_player)
    request = PolicyTurnRequest(
        game_id=uuid4(),
        policy=executor.descriptor,
        turn_number=1,
        context=context,
        hidden_state=bytes(HIDDEN_STATE_BYTES),
    )

    first = executor.invoke(request)
    second = executor.invoke(request)
    assert first == second
    transition = apply_policy_action(state, context, first.action_index)
    assert first.action_index is not None
    expected = context.action_table[first.action_index]
    assert expected is not None
    assert transition.move == expected
    assert transition.played_move.player is policy_player


def _play_complete_game(
    archive_path: Path, *, human_role: str = "queen"
) -> tuple[dict[str, Any], InMemoryGameRepository, int, int]:
    adapter = LocalTorchPolicyAdapter(archive_path)
    executor = InlinePolicyExecutor(adapter, resolve_inference_profile("argmax-v1"))
    repository = InMemoryGameRepository()
    app = create_app(
        repository=repository,
        policy_executor=executor,
        policy_descriptor=executor.descriptor,
        narration_enabled=False,
    )
    opponent_updates = 0
    forced_updates = 0
    with TestClient(app) as client:
        response = client.post(
            "/games",
            json={
                "human_role": human_role,
                "seed": "real-candidate-complete-game",
                "request_id": str(uuid4()),
            },
        )
        assert response.status_code == 201
        view = response.json()
        initial = repository.load(UUID(view["game_id"]))
        assert initial.policy_session.hidden_state == bytes(HIDDEN_STATE_BYTES)

        while view["phase"]["kind"] != "game_complete":
            game_id = view["game_id"]
            if view["phase"]["kind"] == "human_turn":
                response = client.post(
                    f"/games/{game_id}/moves",
                    json={
                        "move_id": view["legal_moves"][0]["move_id"],
                        "expected_version": view["version"],
                        "request_id": str(uuid4()),
                    },
                )
            elif view["phase"]["kind"] == "opponent_turn":
                before = repository.load(UUID(game_id))
                policy_player = other_player(before.human_role)
                context = build_policy_turn_context(before.engine_state, policy_player)
                if context.kind is PolicyTurnKind.FORCED_RECURRENT_TRANSITION:
                    forced_updates += 1
                request_body = {
                    "expected_version": view["version"],
                    "request_id": str(uuid4()),
                }
                claimed = client.post(
                    f"/games/{game_id}/opponent-turn", json=request_body
                )
                assert claimed.status_code == 202
                claimed_session = repository.load(UUID(game_id))
                assert claimed_session.engine_state == before.engine_state
                assert (
                    claimed_session.policy_session.hidden_state
                    == before.policy_session.hidden_state
                )
                response = client.post(
                    f"/games/{game_id}/opponent-turn", json=request_body
                )
                assert response.status_code == 200
                accepted = repository.load(UUID(game_id))
                assert accepted.version == before.version + 1
                assert (
                    accepted.policy_session.hidden_state
                    != before.policy_session.hidden_state
                )
                repeated = client.post(
                    f"/games/{game_id}/opponent-turn", json=request_body
                )
                assert repeated.status_code == 200
                assert repeated.json() == response.json()
                assert (
                    repository.load(UUID(game_id)).policy_session.hidden_state
                    == accepted.policy_session.hidden_state
                )
                opponent_updates += 1
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
    return view, repository, opponent_updates, forced_updates


def test_complete_api_game_runs_against_a_real_training_archive() -> None:
    candidates = [
        Path("runs/training-004/archives/policy-2-policy-2-v20.pt"),
        *sorted(Path("runs/training-003/archives").glob("policy-*-v40.pt")),
    ]
    candidates = [candidate for candidate in candidates if candidate.exists()]
    if not candidates:
        pytest.skip("local training archive is not present")
    view, repository, opponent_updates, forced_updates = _play_complete_game(
        candidates[0]
    )

    assert view["status"] == "game_complete"
    assert len(view["completed_rounds"]) == 6
    assert opponent_updates == 24
    assert forced_updates == 3
    session = repository.load(UUID(view["game_id"]))
    assert session.engine_state.status is EngineStatus.GAME_COMPLETE
    serialized = str(view)
    assert str(candidates[0].resolve()) not in serialized
    assert "raw_logits" not in serialized
    assert "next_hidden_state" not in serialized
    assert "policy_state" not in serialized


def test_environment_selects_candidate_and_rejects_invalid_configuration(
    monkeypatch: pytest.MonkeyPatch, policy_archive: Path, tmp_path: Path
) -> None:
    monkeypatch.setenv("DRACULA_POLICY_ARCHIVE", str(policy_archive))
    monkeypatch.setenv("DRACULA_POLICY_INFERENCE_PROFILE", "argmax-v1")
    configured = create_app(repository=InMemoryGameRepository())
    assert configured.state.gameplay_service.policy_descriptor.policy_id == "candidate-policy"
    assert (
        configured.state.gameplay_service.policy_descriptor.inference_profile
        == "argmax-v1"
    )

    monkeypatch.setenv("DRACULA_POLICY_ARCHIVE", str(tmp_path / "missing.pt"))
    with pytest.raises(PolicyArtifactError):
        create_app(repository=InMemoryGameRepository())

    monkeypatch.setenv("DRACULA_POLICY_ARCHIVE", str(policy_archive))
    monkeypatch.setenv("DRACULA_POLICY_INFERENCE_PROFILE", "unknown-profile-v1")
    with pytest.raises(PolicyContractError):
        create_app(repository=InMemoryGameRepository())

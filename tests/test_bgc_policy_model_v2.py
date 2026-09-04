"""Direct card-set observation and compact candidate policy invariants."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import torch

from dracula.active_policy import ActivePolicyRuntime
from dracula.bgc_policy_model import (
    BGCPolicyModel,
    OBSERVATION_SIZE,
    PARAMETER_COUNT,
    candidate_card_indices,
)
from dracula.engine import apply_move, create_game, legal_moves
from dracula.policy_observation import (
    candidate_action_index_from_engine,
    candidate_action_tensor,
    encode_policy_observation,
    engine_action_index_from_candidate,
)
from dracula.search.information import information_state_from_engine


ROOT = Path(__file__).resolve().parents[1]
SELECTED_ARTIFACT = (
    ROOT / "runs/bgc-policy-pi1-001/artifacts/unaccepted-candidate.pt"
)


def _information():
    state = create_game("bgc-card-policy-v2-fixture")
    return information_state_from_engine(state)


def test_direct_card_set_shapes_and_exact_parameter_count() -> None:
    observation = encode_policy_observation(_information())
    model = BGCPolicyModel(
        run_root_seed="bgc-card-model-test",
        model_id="pi1",
        initialization_ordinal=0,
    )
    candidates, present = candidate_card_indices(observation)
    assert observation.shape == (OBSERVATION_SIZE,) == (659,)
    assert candidates.shape == present.shape == (4,)
    assert present.all()
    assert model(observation).shape == (4, 8)
    assert sum(parameter.numel() for parameter in model.parameters()) == PARAMETER_COUNT
    assert PARAMETER_COUNT == 754_601


def test_direct_observations_match_frozen_pre_cleanup_fixtures() -> None:
    expected = (
        "880dbad90758315bec5da197a6932f9ae949f06294c73f829283c6c5f668fe4a",
        "7f97a139f89a3fb85b1dda482f0f3f65b844c70222cf06c73162329effb2e639",
        "6df85b298e1d0b82205761dd1352b9b3615da510d6cdd71931a8b84f64ecc591",
        "118a287940067fa0845cb39692ebec4632e52a51c242088310b86b77e8c2421d",
        "6506b744e76c1e71a2aa7b77b94bbd550bec1accbcfa053e915ff8e6b6605a22",
        "4c3e073d0102ccaa1890fbb803238f1b5608dee2fc12d471fdfbba458c595b75",
        "ce37a0d1e5a8a76748d74eb22ef5c98b9d37a849bd40b8c91ec7255b3b76cdf2",
    )
    state = create_game("pi1-direct-queen")
    actual = []
    for _ in range(7):
        observation = encode_policy_observation(information_state_from_engine(state))
        actual.append(hashlib.sha256(observation.numpy().tobytes()).hexdigest())
        state = apply_move(state, legal_moves(state, state.active_player)[0]).state
    assert tuple(actual) == expected


def test_model_initialization_is_deterministic_and_has_no_slot_embeddings() -> None:
    torch.manual_seed(17)
    global_state = torch.random.get_rng_state().clone()
    first = BGCPolicyModel(
        run_root_seed="bgc-card-model-test", model_id="pi1", initialization_ordinal=0
    )
    second = BGCPolicyModel(
        run_root_seed="bgc-card-model-test", model_id="pi1", initialization_ordinal=0
    )
    assert all(
        torch.equal(left, right)
        for left, right in zip(first.parameters(), second.parameters(), strict=True)
    )
    assert torch.equal(torch.random.get_rng_state(), global_state)
    assert not any("slot" in name for name, _ in first.named_parameters())


def test_batch_equivalence_gradients_and_action_round_trip() -> None:
    information = _information()
    observation = encode_policy_observation(information)
    model = BGCPolicyModel(
        run_root_seed="bgc-card-model-batch-test",
        model_id="pi1",
        initialization_ordinal=0,
    )
    individual = model(observation)
    batched = model(torch.stack((observation, observation)))
    assert torch.allclose(individual, batched[0], atol=1e-6, rtol=1e-6)
    assert torch.equal(batched[0], batched[1])
    batched.square().mean().backward()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
    legal = torch.tensor(information.legal_mask, dtype=torch.bool)
    compact = candidate_action_tensor(information, legal)
    for compact_index in torch.nonzero(
        compact.flatten(), as_tuple=False
    ).flatten().tolist():
        engine_index = engine_action_index_from_candidate(information, compact_index)
        assert candidate_action_index_from_engine(information, engine_index) == compact_index


@pytest.mark.skipif(not SELECTED_ARTIFACT.is_file(), reason="local pi1 artifact absent")
def test_selected_artifact_matches_pre_cleanup_logits_and_actions() -> None:
    expected = (
        ("48724f8ae407964c2db1686e7c30d717cc21772042d9457eae4055a2ac38ab82", 17, 22),
        ("5f17692aaca65b3d9e3bd96e53e7caf04335bc78e58d997c37f1808d0d9f85e2", 22, 22),
        ("b0dbf34005820e0381fd489431a265971d285995f2cb681d9392b50b4219fe26", 29, 29),
        ("e908f1b3e1e257ac387ae773f22c9a650dc499201f5c8333dc7a464062c101ac", 22, 22),
        ("973304b77219ed992a627450641ce55cdb91049f03b6c5f8e5b3871cef3cc380", 18, 18),
        ("75fe4716c73cbaa03fa24feb0c01c4b0eb3b55f7312b25057e78d4fdf92b5784", 22, 22),
        ("c5cefaa98d891a134deae5e7ddd9357c66b4f0c2a7c75ae25acf743aa1f0dcd9", 31, 31),
    )
    runtime = ActivePolicyRuntime.from_artifact(SELECTED_ARTIFACT)
    state = create_game("pi1-direct-queen")
    for placement, (logits_digest, representative, concrete) in enumerate(
        expected, start=1
    ):
        information = information_state_from_engine(state)
        with torch.inference_mode():
            logits = runtime.model(encode_policy_observation(information))
        decision = runtime.decide(
            information,
            fixture_id="pi1-direct-queen",
            decision_index=placement,
        )
        assert hashlib.sha256(logits.numpy().tobytes()).hexdigest() == logits_digest
        assert decision.representative_action_index == representative
        assert decision.concrete_action_index == concrete
        state = apply_move(state, legal_moves(state, state.active_player)[0]).state

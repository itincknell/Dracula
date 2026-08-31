"""Card-set observation and compact candidate policy invariants."""

from __future__ import annotations

import torch

from dracula.bgc_policy_model import (
    BGCPolicyModel,
    OBSERVATION_SIZE,
    PARAMETER_COUNT,
    candidate_card_indices,
    compact_action_index_from_legacy,
    compact_action_tensor_from_legacy,
    compact_observation_from_legacy,
    legacy_action_index_from_compact,
)
from dracula.bridge import build_policy_turn_context
from dracula.engine import EnginePlayer, create_game


def _legacy():
    state = create_game("bgc-card-policy-v2-fixture")
    assert state.active_player is not None
    return build_policy_turn_context(state, state.active_player).input


def test_card_set_shapes_and_exact_parameter_count() -> None:
    legacy = _legacy()
    observation = compact_observation_from_legacy(legacy.observation)
    model = BGCPolicyModel(
        run_root_seed="bgc-card-model-test",
        model_id="pi0",
        initialization_ordinal=0,
    )
    candidates, present = candidate_card_indices(observation)
    assert observation.shape == (OBSERVATION_SIZE,) == (659,)
    assert candidates.shape == present.shape == (4,)
    assert present.all()
    assert model(observation).shape == (4, 8)
    assert sum(parameter.numel() for parameter in model.parameters()) == PARAMETER_COUNT
    assert PARAMETER_COUNT == 754_601


def test_legacy_slot_permutations_have_identical_card_set_inputs_and_actions() -> None:
    legacy = _legacy()
    permutation = torch.tensor((2, 0, 3, 1))
    permuted_observation = legacy.observation.clone()
    permuted_observation[:216] = legacy.observation[:216].reshape(4, 54).index_select(
        0, permutation
    ).flatten()
    permuted_mask = legacy.legal_mask.index_select(0, permutation)
    assert torch.equal(
        compact_observation_from_legacy(legacy.observation),
        compact_observation_from_legacy(permuted_observation),
    )
    assert torch.equal(
        compact_action_tensor_from_legacy(legacy.observation, legacy.legal_mask),
        compact_action_tensor_from_legacy(permuted_observation, permuted_mask),
    )


def test_model_initialization_is_deterministic_and_does_not_use_slot_embeddings() -> None:
    torch.manual_seed(17)
    global_state = torch.random.get_rng_state().clone()
    first = BGCPolicyModel(
        run_root_seed="bgc-card-model-test",
        model_id="pi0",
        initialization_ordinal=0,
    )
    second = BGCPolicyModel(
        run_root_seed="bgc-card-model-test",
        model_id="pi0",
        initialization_ordinal=0,
    )
    assert all(
        torch.equal(left, right)
        for left, right in zip(first.parameters(), second.parameters(), strict=True)
    )
    assert torch.equal(torch.random.get_rng_state(), global_state)
    assert not any("slot" in name for name, _ in first.named_parameters())


def test_batch_equivalence_gradients_and_action_round_trip() -> None:
    legacy = _legacy()
    observation = compact_observation_from_legacy(legacy.observation)
    model = BGCPolicyModel(
        run_root_seed="bgc-card-model-batch-test",
        model_id="pi0",
        initialization_ordinal=0,
    )
    individual = model(observation)
    batched = model(torch.stack((observation, observation)))
    assert torch.allclose(individual, batched[0], atol=1e-6, rtol=1e-6)
    assert torch.equal(batched[0], batched[1])
    batched.square().mean().backward()
    assert all(
        parameter.grad is not None
        and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
    for legacy_index in torch.nonzero(
        legacy.legal_mask.flatten(), as_tuple=False
    ).flatten().tolist():
        compact_index = compact_action_index_from_legacy(
            legacy.observation, legacy_index
        )
        assert (
            legacy_action_index_from_compact(
                legacy.observation, compact_index
            )
            == legacy_index
        )

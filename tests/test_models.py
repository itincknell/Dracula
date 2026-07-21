"""Mechanical acceptance tests for neural-model architecture version 1."""

from __future__ import annotations

import math

import pytest
import torch
from torch import Tensor, nn

from dracula.models import (
    CARD_COUNT,
    COFFIN_START,
    CONTEXT_START,
    Critic,
    HIDDEN_SIZE,
    OBSERVATION_SIZE,
    Policy,
    STATUS_START,
)
from dracula.randomness import derive_pytorch_seed

MODEL_INITIALIZATION_NAMESPACE = "dracula-model-initialization-v1"


def _observations(batch_size: int = 1) -> Tensor:
    """Build valid Boolean observations without depending on the future bridge."""

    observations = torch.zeros(batch_size, OBSERVATION_SIZE, dtype=torch.bool)
    for batch_index in range(batch_size):
        cards = [(batch_index * 11 + offset) % CARD_COUNT for offset in range(13)]
        hand_cards = cards[:4]
        coffin_cards = cards[4:13]

        for slot, card in enumerate(hand_cards):
            observations[batch_index, slot * CARD_COUNT + card] = True
        for position, card in enumerate(coffin_cards):
            observations[batch_index, COFFIN_START + position * CARD_COUNT + card] = True

        # Status rows form the required played/in-hand/hidden partition.
        observations[batch_index, STATUS_START + 2 * CARD_COUNT : CONTEXT_START] = True
        for card in coffin_cards:
            observations[batch_index, STATUS_START + 2 * CARD_COUNT + card] = False
            observations[batch_index, STATUS_START + card] = True
        for card in hand_cards:
            observations[batch_index, STATUS_START + 2 * CARD_COUNT + card] = False
            observations[batch_index, STATUS_START + CARD_COUNT + card] = True

        observations[batch_index, CONTEXT_START + batch_index % 6] = True
        observations[batch_index, CONTEXT_START + 6 + batch_index % 4] = True
        observations[batch_index, CONTEXT_START + 10] = bool(batch_index % 2)
    return observations


def _legal_masks(batch_size: int = 1) -> Tensor:
    masks = torch.zeros(batch_size, 4, 8, dtype=torch.bool)
    for batch_index in range(batch_size):
        for hand_slot in range(4):
            masks[batch_index, hand_slot, (2 * hand_slot + batch_index) % 8] = True
            masks[batch_index, hand_slot, (2 * hand_slot + batch_index + 1) % 8] = True
    return masks


def _parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


@pytest.fixture
def policy() -> Policy:
    return Policy(
        seed=derive_pytorch_seed(
            MODEL_INITIALIZATION_NAMESPACE, "test-root", "policy", "policy-0", "0"
        )
    )


@pytest.fixture
def critic() -> Critic:
    return Critic(
        seed=derive_pytorch_seed(
            MODEL_INITIALIZATION_NAMESPACE, "test-root", "critic", "critic-0", "0"
        )
    )


def test_documented_parameter_counts(policy: Policy, critic: Critic) -> None:
    assert _parameter_count(policy) == 443_145
    assert _parameter_count(critic) == 143_273


def test_single_and_batch_shapes(policy: Policy, critic: Critic) -> None:
    observation = _observations()[0]
    legal_mask = _legal_masks()[0]
    hidden = policy.initial_hidden()

    logits, next_hidden = policy(observation, legal_mask, hidden)
    value = critic(observation)
    assert observation.shape == (875,)
    assert legal_mask.shape == (4, 8)
    assert logits.shape == (4, 8)
    assert next_hidden.shape == (128,)
    assert value.shape == ()

    observations = _observations(3)
    legal_masks = _legal_masks(3)
    hidden_batch = policy.initial_hidden(3)
    logits_batch, next_hidden_batch = policy(observations, legal_masks, hidden_batch)
    values = critic(observations)
    assert observations.shape == (3, 875)
    assert legal_masks.shape == (3, 4, 8)
    assert logits_batch.shape == (3, 4, 8)
    assert next_hidden_batch.shape == (3, 128)
    assert values.shape == (3,)


def test_initial_hidden_is_an_all_zero_game_boundary(policy: Policy) -> None:
    assert policy.initial_hidden().shape == (HIDDEN_SIZE,)
    assert policy.initial_hidden(5).shape == (5, HIDDEN_SIZE)
    assert torch.count_nonzero(policy.initial_hidden()) == 0
    assert torch.count_nonzero(policy.initial_hidden(5)) == 0


def test_initialization_is_deterministic_from_a_derived_seed() -> None:
    seed = derive_pytorch_seed(
        MODEL_INITIALIZATION_NAMESPACE, "run-root", "policy", "policy-3", "0"
    )
    first = Policy(seed=seed)
    second = Policy(seed=seed)
    other = Policy(
        seed=derive_pytorch_seed(
            MODEL_INITIALIZATION_NAMESPACE, "run-root", "policy", "policy-4", "0"
        )
    )

    for first_parameter, second_parameter in zip(first.parameters(), second.parameters(), strict=True):
        assert torch.equal(first_parameter, second_parameter)
    assert any(
        not torch.equal(first_parameter, other_parameter)
        for first_parameter, other_parameter in zip(first.parameters(), other.parameters(), strict=True)
    )

    critic_seed = derive_pytorch_seed(
        MODEL_INITIALIZATION_NAMESPACE, "run-root", "critic", "shared", "0"
    )
    first_critic = Critic(seed=critic_seed)
    second_critic = Critic(seed=critic_seed)
    for first_parameter, second_parameter in zip(
        first_critic.parameters(), second_critic.parameters(), strict=True
    ):
        assert torch.equal(first_parameter, second_parameter)


def test_seeded_initialization_does_not_consume_the_global_random_stream() -> None:
    torch.manual_seed(718)
    expected_next_value = torch.rand(1)
    torch.manual_seed(718)
    Policy(
        seed=derive_pytorch_seed(
            MODEL_INITIALIZATION_NAMESPACE, "run-root", "policy", "policy-0", "0"
        )
    )
    Critic(
        seed=derive_pytorch_seed(
            MODEL_INITIALIZATION_NAMESPACE, "run-root", "critic", "shared", "0"
        )
    )
    assert torch.equal(torch.rand(1), expected_next_value)


@pytest.mark.parametrize("model_type", [Policy, Critic])
def test_exact_initialization_families(model_type: type[Policy] | type[Critic]) -> None:
    model_kind = "policy" if model_type is Policy else "critic"
    model = model_type(
        seed=derive_pytorch_seed(
            MODEL_INITIALIZATION_NAMESPACE,
            "test-root",
            model_kind,
            f"{model_kind}-initialization-rules",
            "0",
        )
    )

    for module in model.modules():
        if isinstance(module, nn.Linear):
            fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(module.weight)
            xavier_bound = math.sqrt(6.0 / (fan_in + fan_out))
            assert torch.max(torch.abs(module.weight)) <= xavier_bound
            assert torch.count_nonzero(module.bias) == 0
        elif isinstance(module, nn.Embedding):
            expected_std = 1.0 / math.sqrt(module.embedding_dim)
            # Sampling is stochastic, so check scale rather than exact sample moments.
            assert module.weight.std().item() == pytest.approx(expected_std, rel=0.35)
            assert abs(module.weight.mean().item()) < expected_std * 0.5
        elif isinstance(module, nn.LayerNorm):
            assert torch.equal(module.weight, torch.ones_like(module.weight))
            assert torch.count_nonzero(module.bias) == 0

    recurrent_modules = [module for module in model.modules() if isinstance(module, nn.GRUCell)]
    for recurrent in recurrent_modules:
        fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(recurrent.weight_ih)
        xavier_bound = math.sqrt(6.0 / (fan_in + fan_out))
        assert torch.max(torch.abs(recurrent.weight_ih)) <= xavier_bound
        assert torch.count_nonzero(recurrent.bias_ih) == 0
        assert torch.count_nonzero(recurrent.bias_hh) == 0
        identity = torch.eye(HIDDEN_SIZE)
        for gate in recurrent.weight_hh.chunk(3, dim=0):
            assert torch.allclose(gate @ gate.T, identity, atol=1e-5, rtol=1e-5)


def test_batch_and_individual_forward_are_equivalent(policy: Policy, critic: Critic) -> None:
    observations = _observations(4)
    masks = _legal_masks(4)
    hidden = torch.linspace(-0.5, 0.5, 4 * HIDDEN_SIZE).reshape(4, HIDDEN_SIZE)

    batch_logits, batch_hidden = policy(observations, masks, hidden)
    batch_values = critic(observations)
    for index in range(4):
        single_logits, single_hidden = policy(observations[index], masks[index], hidden[index])
        single_value = critic(observations[index])
        # Batched GEMM can accumulate in a different order than a single row.
        assert torch.allclose(single_logits, batch_logits[index], atol=5e-6, rtol=1e-5)
        assert torch.allclose(single_hidden, batch_hidden[index], atol=5e-6, rtol=1e-5)
        assert torch.allclose(single_value, batch_values[index], atol=5e-6, rtol=1e-5)


def _transpose_coffin(observation: Tensor) -> Tensor:
    transposed = observation.clone()
    coffin = observation[COFFIN_START:STATUS_START].reshape(3, 3, CARD_COUNT)
    transposed[COFFIN_START:STATUS_START] = coffin.transpose(0, 1).reshape(-1)
    return transposed


def test_queen_and_king_equivalence_after_player_relative_normalization(
    policy: Policy, critic: Critic
) -> None:
    queen_normalized = _observations()[0]
    # An equivalent King's authoritative board is transposed. The bridge applies
    # the self-inverse transpose, so the model boundary receives the same view.
    king_authoritative = _transpose_coffin(queen_normalized)
    king_normalized = _transpose_coffin(king_authoritative)
    assert torch.equal(queen_normalized, king_normalized)

    mask = _legal_masks()[0]
    queen_logits, queen_hidden = policy(queen_normalized, mask, policy.initial_hidden())
    king_logits, king_hidden = policy(king_normalized, mask, policy.initial_hidden())
    assert torch.equal(queen_logits, king_logits)
    assert torch.equal(queen_hidden, king_hidden)
    assert torch.equal(critic(queen_normalized), critic(king_normalized))


def test_legal_mask_enters_recurrent_core_and_every_pair(policy: Policy) -> None:
    observation = _observations(2)
    legal_mask = _legal_masks(2)
    captured: dict[str, Tensor] = {}

    recurrent_hook = policy.recurrent.register_forward_pre_hook(
        lambda _module, inputs: captured.__setitem__("recurrent", inputs[0].detach().clone())
    )
    pair_hook = policy.pair_input.register_forward_pre_hook(
        lambda _module, inputs: captured.__setitem__("pair", inputs[0].detach().clone())
    )
    try:
        raw_logits, _ = policy(observation, legal_mask, policy.initial_hidden(2))
    finally:
        recurrent_hook.remove()
        pair_hook.remove()

    # The final 32 GRU inputs are the flattened legal mask.
    assert torch.equal(captured["recurrent"][:, -32:], legal_mask.flatten(start_dim=1).float())
    # The final feature for each shared pair-head invocation is that pair's bit.
    assert captured["pair"].shape == (2, 4, 8, 257)
    assert torch.equal(captured["pair"][..., -1], legal_mask.float())
    # Hard masking belongs outside the model; illegal positions still have raw finite logits.
    assert torch.isfinite(raw_logits).all()
    assert torch.isfinite(raw_logits[~legal_mask]).all()


def test_forward_and_gradients_are_finite(policy: Policy, critic: Critic) -> None:
    observations = _observations(3)
    masks = _legal_masks(3)
    hidden = policy.initial_hidden(3).requires_grad_()

    logits, next_hidden = policy(observations, masks, hidden)
    values = critic(observations)
    loss = logits.square().mean() + next_hidden.square().mean() + values.square().mean()
    loss.backward()

    assert torch.isfinite(logits).all()
    assert torch.isfinite(next_hidden).all()
    assert torch.isfinite(values).all()
    assert hidden.grad is not None and torch.isfinite(hidden.grad).all()
    for parameter in (*policy.parameters(), *critic.parameters()):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_hidden_state_continues_across_all_24_steps(policy: Policy) -> None:
    observations = _observations(24)
    masks = _legal_masks(24)

    hidden = policy.initial_hidden()
    for step in range(24):
        _, hidden = policy(observations[step], masks[step], hidden)
    uninterrupted = hidden

    hidden = policy.initial_hidden()
    for step in range(12):
        _, hidden = policy(observations[step], masks[step], hidden)
    retained_at_round_boundary = hidden
    for step in range(12, 24):
        _, hidden = policy(observations[step], masks[step], hidden)
    assert torch.allclose(hidden, uninterrupted, atol=1e-6, rtol=1e-5)

    reset_hidden = policy.initial_hidden()
    for step in range(12, 24):
        _, reset_hidden = policy(observations[step], masks[step], reset_hidden)
    assert not torch.allclose(retained_at_round_boundary, policy.initial_hidden())
    assert not torch.allclose(reset_hidden, uninterrupted)


def test_forced_transitions_update_hidden_but_have_no_actor_loss(policy: Policy) -> None:
    observations = _observations(24)
    masks = _legal_masks(24)
    forced_steps = (3, 11, 19)
    for step in forced_steps:
        masks[step].zero_()
        masks[step, step % 4, step % 8] = True

    hidden = policy.initial_hidden()
    logits_by_step = []
    hidden_before_forced = []
    hidden_after_forced = []
    for step in range(24):
        if step in forced_steps:
            hidden_before_forced.append(hidden)
        logits, hidden = policy(observations[step], masks[step], hidden)
        logits_by_step.append(logits)
        if step in forced_steps:
            hidden_after_forced.append(hidden)

    for before, after in zip(hidden_before_forced, hidden_after_forced, strict=True):
        assert not torch.allclose(before, after)

    logits_sequence = torch.stack(logits_by_step)
    logits_sequence.retain_grad()
    actor_loss_mask = torch.ones(24, dtype=torch.bool)
    actor_loss_mask[list(forced_steps)] = False
    actor_loss = logits_sequence[actor_loss_mask].square().mean()
    actor_loss.backward()

    # Forced logits are absent from actor loss even though their GRU transitions
    # remain in the graph and can carry later-step gradients through hidden state.
    assert logits_sequence.grad is not None
    assert torch.count_nonzero(logits_sequence.grad[~actor_loss_mask]) == 0


def test_actor_and_critic_parameters_and_gradients_are_independent(
    policy: Policy, critic: Critic
) -> None:
    policy_storage = {parameter.data_ptr() for parameter in policy.parameters()}
    critic_storage = {parameter.data_ptr() for parameter in critic.parameters()}
    assert policy_storage.isdisjoint(critic_storage)

    observations = _observations(2)
    masks = _legal_masks(2)
    logits, next_hidden = policy(observations, masks, policy.initial_hidden(2))
    (logits.mean() + next_hidden.mean()).backward()
    assert all(parameter.grad is not None for parameter in policy.parameters())
    assert all(parameter.grad is None for parameter in critic.parameters())

    policy.zero_grad(set_to_none=True)
    critic(observations).mean().backward()
    assert all(parameter.grad is None for parameter in policy.parameters())
    assert all(parameter.grad is not None for parameter in critic.parameters())


def test_cpu_forward_and_backward(policy: Policy, critic: Critic) -> None:
    assert next(policy.parameters()).device.type == "cpu"
    observations = _observations(2)
    masks = _legal_masks(2)
    logits, hidden = policy(observations, masks, policy.initial_hidden(2))
    values = critic(observations)
    (logits.mean() + hidden.mean() + values.mean()).backward()
    assert torch.isfinite(logits).all()
    assert torch.isfinite(hidden).all()
    assert torch.isfinite(values).all()


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS is unavailable")
def test_mps_matches_cpu_with_numerical_tolerances() -> None:
    policy_seed = derive_pytorch_seed(
        MODEL_INITIALIZATION_NAMESPACE, "test-root", "policy", "device-policy", "0"
    )
    critic_seed = derive_pytorch_seed(
        MODEL_INITIALIZATION_NAMESPACE, "test-root", "critic", "device-critic", "0"
    )
    cpu_policy = Policy(seed=policy_seed)
    cpu_critic = Critic(seed=critic_seed)
    mps_policy = Policy(seed=policy_seed).to("mps")
    mps_critic = Critic(seed=critic_seed).to("mps")

    observations = _observations(2)
    masks = _legal_masks(2)
    cpu_logits, cpu_hidden = cpu_policy(observations, masks, cpu_policy.initial_hidden(2))
    cpu_values = cpu_critic(observations)

    mps_observations = observations.to("mps")
    mps_masks = masks.to("mps")
    mps_logits, mps_hidden = mps_policy(
        mps_observations, mps_masks, mps_policy.initial_hidden(2)
    )
    mps_values = mps_critic(mps_observations)
    (mps_logits.mean() + mps_hidden.mean() + mps_values.mean()).backward()

    # CPU and Metal use different kernels; agreement is numerical, not bitwise.
    assert torch.allclose(mps_logits.cpu(), cpu_logits, atol=2e-4, rtol=2e-4)
    assert torch.allclose(mps_hidden.cpu(), cpu_hidden, atol=2e-4, rtol=2e-4)
    assert torch.allclose(mps_values.cpu(), cpu_values, atol=2e-4, rtol=2e-4)
    for parameter in (*mps_policy.parameters(), *mps_critic.parameters()):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()

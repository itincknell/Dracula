"""Numerical and transactional tests for critic and recurrent PPO optimization."""

from __future__ import annotations

import copy
from dataclasses import replace

import pytest
import torch

from dracula.collection import (
    ActorBucket,
    PolicyVersion,
    build_smoke_schedule,
    collect_schedule,
    module_fingerprint,
    normalized_round_return,
)
from dracula.models import Critic, Policy
from dracula.optimization import (
    NumericalOptimizationError,
    OptimizationConfig,
    actor_weight,
    critic_mse,
    illegal_probability_loss,
    optimize_collection,
    ppo_clipped_surrogate,
    prepare_actor_bucket,
    replay_current_policy,
)
from dracula.randomness import derive_pytorch_seed

INITIALIZATION_NAMESPACE = "dracula-model-initialization-v1"
RUN_ROOT = "optimization-test-root"
POLICY_A = PolicyVersion("policy-0", "policy-0-v0")
POLICY_B = PolicyVersion("policy-1", "policy-1-v0")


def _seed(model_kind: str, stable_id: str) -> int:
    return derive_pytorch_seed(
        INITIALIZATION_NAMESPACE,
        "optimization-model-root",
        model_kind,
        stable_id,
        "0",
    )


@pytest.fixture(scope="module")
def smoke_setup():
    policies = {
        POLICY_A: Policy(seed=_seed("policy", POLICY_A.policy_id)),
        POLICY_B: Policy(seed=_seed("policy", POLICY_B.policy_id)),
    }
    critic = Critic(seed=_seed("critic", "shared"))
    collection = collect_schedule(
        run_root_seed=RUN_ROOT,
        schedule=build_smoke_schedule(
            POLICY_A, POLICY_B, "optimization-smoke-lane"
        ),
        policies=policies,
        critic=critic,
        critic_version="critic-v0",
    )
    return policies, critic, collection


def test_round_return_and_critic_mse_use_the_documented_fixed_scale() -> None:
    assert normalized_round_return(150, 0) == 1.0
    assert normalized_round_return(0, 150) == -1.0
    assert normalized_round_return(75, 30) == 0.3
    predictions = torch.tensor([0.5, -0.5])
    targets = torch.tensor([1.0, 0.0])
    assert torch.equal(critic_mse(predictions, targets), torch.tensor(0.25))


def test_actor_weight_burn_in_ramp_and_entropy_boundary() -> None:
    config = OptimizationConfig()
    assert actor_weight(9_999, 3, config) == 0.0
    assert actor_weight(10_000, 2, config) == 0.0
    assert actor_weight(15_000, 3, config) == 0.5
    assert actor_weight(20_000, 3, config) == 1.0


# Normalization spans the learner's complete bucket rather than each trajectory.
def test_advantages_are_normalized_once_across_the_full_actor_bucket(
    smoke_setup,
) -> None:
    _, _, collection = smoke_setup
    original = collection.actor_buckets[0].trajectories[0]
    shifted = replace(
        original,
        transitions=tuple(
            replace(
                transition,
                critic_value=(
                    transition.critic_value + 0.25
                    if transition.critic_value is not None
                    else None
                ),
            )
            for transition in original.transitions
        ),
    )
    prepared = prepare_actor_bucket(
        ActorBucket(original.learner, (original, shifted))
    )
    normalized = torch.cat(
        [trajectory.normalized_advantages for trajectory in prepared.trajectories]
    )
    assert normalized.mean().abs().item() < 1e-6
    assert torch.allclose(normalized.std(unbiased=False), torch.tensor(1.0))
    assert all(
        trajectory.normalized_advantages.mean().abs().item() > 0.05
        for trajectory in prepared.trajectories
    )


# Replaying unchanged behavior weights must reproduce the saved action density exactly.
def test_unchanged_weights_produce_probability_ratio_one(smoke_setup) -> None:
    policies, _, collection = smoke_setup
    bucket = prepare_actor_bucket(collection.actor_buckets[0])
    policy = copy.deepcopy(policies[bucket.learner])
    replay = replay_current_policy(policy, bucket.trajectories[0], device="cpu")
    old_logs = bucket.trajectories[0].old_log_probabilities
    surrogate = ppo_clipped_surrogate(
        replay.learned_log_probabilities,
        old_logs,
        bucket.trajectories[0].normalized_advantages,
    )
    assert torch.equal(surrogate.probability_ratios, torch.ones(21))


def test_positive_and_negative_advantages_exercise_both_clipped_branches() -> None:
    ratios = torch.tensor([1.5, 0.5, 1.1, 0.9])
    old_logs = torch.zeros(4)
    new_logs = ratios.log().requires_grad_()
    advantages = torch.tensor([1.0, -1.0, 1.0, -1.0])
    surrogate = ppo_clipped_surrogate(new_logs, old_logs, advantages)
    assert surrogate.clipped_branch_selected.tolist() == [True, True, False, False]
    assert torch.allclose(
        surrogate.clipped_objective, torch.tensor([1.2, -0.8, 1.1, -0.9])
    )
    surrogate.loss.backward()
    assert torch.isfinite(new_logs.grad).all()


# The threshold creates a true zero-cost region without disconnecting higher penalties.
def test_illegal_loss_is_zero_below_threshold_and_differentiable_above() -> None:
    legal_mask = torch.ones(1, 4, 8, dtype=torch.bool)
    legal_mask[0, 3, 7] = False
    below_logits = torch.zeros(1, 4, 8, requires_grad=True)
    with torch.no_grad():
        below_logits[0, 3, 7] = -20.0
    below_loss, below_probability, _ = illegal_probability_loss(
        below_logits, legal_mask
    )
    assert below_probability.item() < 0.001
    assert below_loss.item() == 0.0
    below_loss.backward()
    assert torch.count_nonzero(below_logits.grad) == 0

    above_logits = torch.zeros(1, 4, 8, requires_grad=True)
    above_loss, above_probability, _ = illegal_probability_loss(
        above_logits, legal_mask
    )
    assert above_probability.item() > 0.001
    above_loss.backward()
    assert above_logits.grad[0, 3, 7].abs().item() > 0
    assert torch.isfinite(above_logits.grad).all()


# The pre-mask auxiliary objective must reach illegal logits before action masking.
def test_illegal_logits_receive_pre_mask_gradients() -> None:
    logits = torch.zeros(2, 4, 8, requires_grad=True)
    legal_mask = torch.zeros(2, 4, 8, dtype=torch.bool)
    legal_mask[:, 0, 0] = True
    loss, _, _ = illegal_probability_loss(logits, legal_mask)
    loss.backward()
    assert torch.count_nonzero(logits.grad[~legal_mask]) > 0
    assert torch.count_nonzero(logits.grad[legal_mask]) > 0


# Forced rows remain in recurrence but are absent from every sampled loss vector.
def test_forced_steps_contribute_no_loss_terms(smoke_setup) -> None:
    policies, _, collection = smoke_setup
    prepared = prepare_actor_bucket(collection.actor_buckets[0]).trajectories[0]
    policy = copy.deepcopy(policies[prepared.trajectory.learner])
    replay = replay_current_policy(policy, prepared, device="cpu")
    forced_indices = tuple(
        index
        for index, transition in enumerate(prepared.trajectory.transitions)
        if not transition.actor_loss_mask
    )
    assert len(forced_indices) == 3
    assert len(replay.learned_step_indices) == 21
    assert set(forced_indices).isdisjoint(replay.learned_step_indices)
    assert prepared.old_critic_values.shape == (21,)
    for index in forced_indices:
        replay.raw_logits[index].retain_grad()
    loss = (
        -replay.learned_log_probabilities.mean()
        - replay.learned_entropies.mean()
        + replay.learned_illegal_penalties.mean()
    )
    loss.backward()
    assert all(replay.raw_logits[index].grad is None for index in forced_indices)


# A forced recurrent update must still influence the next learned state and its loss.
def test_forced_step_transmits_recurrent_gradient_to_later_learned_step(
    smoke_setup,
) -> None:
    policies, _, collection = smoke_setup
    prepared = prepare_actor_bucket(collection.actor_buckets[0]).trajectories[0]
    policy = copy.deepcopy(policies[prepared.trajectory.learner])
    replay = replay_current_policy(
        policy, prepared, device="cpu", retain_hidden_gradients=True
    )
    forced_index = next(
        index
        for index, transition in enumerate(prepared.trajectory.transitions[:-1])
        if not transition.actor_loss_mask
    )
    learned_position = replay.learned_step_indices.index(forced_index + 1)
    (-replay.learned_log_probabilities[learned_position]).backward()
    gradient = replay.hidden_states[forced_index].grad
    assert gradient is not None
    assert torch.linalg.vector_norm(gradient).item() > 0


# Retaining hidden state across rounds lets a late objective train the early recurrence.
def test_later_round_loss_backpropagates_through_earlier_recurrent_chain(
    smoke_setup,
) -> None:
    policies, _, collection = smoke_setup
    prepared = prepare_actor_bucket(collection.actor_buckets[0]).trajectories[0]
    policy = copy.deepcopy(policies[prepared.trajectory.learner])
    replay = replay_current_policy(
        policy, prepared, device="cpu", retain_hidden_gradients=True
    )
    (-replay.learned_log_probabilities[-1]).backward()
    first_hidden_gradient = replay.hidden_states[0].grad
    assert first_hidden_gradient is not None
    assert torch.linalg.vector_norm(first_hidden_gradient).item() > 0


# The full smoke update is transactional, uses four passes, and changes both learners.
def test_smoke_optimization_changes_copies_and_preserves_source_models(
    smoke_setup,
) -> None:
    policies, critic, collection = smoke_setup
    source_policy_fingerprints = {
        identity: module_fingerprint(policy) for identity, policy in policies.items()
    }
    source_critic_fingerprint = module_fingerprint(critic)
    prepared_digests = {
        bucket.learner: prepare_actor_bucket(bucket).fixed_data_digest
        for bucket in collection.actor_buckets
    }
    result = optimize_collection(
        collection=collection,
        policies=policies,
        critic=critic,
        iteration_index=0,
        collected_rounds=0,
        critic_validation_windows=0,
        config=OptimizationConfig(
            actor_weight_override=1.0,
            trajectory_minibatch_size=1,
            approximate_kl_limit=100.0,
        ),
    )
    assert result.actor_weight == 1.0
    assert result.entropy_coefficient == 0.001
    assert len(result.critic_pass_losses) == 4
    assert all(report.passes_completed == 4 for report in result.policy_reports)
    assert all(not report.early_stopped for report in result.policy_reports)
    assert all(
        report.fixed_data_digest == prepared_digests[report.learner]
        for report in result.policy_reports
    )
    assert all(
        module_fingerprint(result.policies[identity])
        != source_policy_fingerprints[identity]
        for identity in policies
    )
    assert module_fingerprint(result.critic) != source_critic_fingerprint
    assert all(
        module_fingerprint(policy) == source_policy_fingerprints[identity]
        for identity, policy in policies.items()
    )
    assert module_fingerprint(critic) == source_critic_fingerprint
    # Completed learners and Adam state return to CPU before the next learner runs.
    assert all(
        parameter.device.type == "cpu"
        for policy in result.policies.values()
        for parameter in policy.parameters()
    )
    assert all(parameter.device.type == "cpu" for parameter in result.critic.parameters())
    optimizer_tensors = [
        value
        for state in (*result.policy_optimizer_states.values(), result.critic_optimizer_state)
        for value in _nested_tensors(state)
    ]
    assert optimizer_tensors
    assert all(tensor.device.type == "cpu" for tensor in optimizer_tensors)


# Actor and critic graphs are disjoint even though one iteration updates both copies.
def test_actor_and_critic_gradients_remain_independent(smoke_setup) -> None:
    policies, critic, collection = smoke_setup
    prepared = prepare_actor_bucket(collection.actor_buckets[0]).trajectories[0]
    policy = copy.deepcopy(policies[prepared.trajectory.learner])
    value_model = copy.deepcopy(critic)
    replay = replay_current_policy(policy, prepared, device="cpu")
    (-replay.learned_log_probabilities.mean()).backward()
    assert any(parameter.grad is not None for parameter in policy.parameters())
    assert all(parameter.grad is None for parameter in value_model.parameters())

    policy.zero_grad(set_to_none=True)
    rows = collection.critic_rows[:21]
    observations = torch.stack([row.observation for row in rows]).clone()
    targets = torch.tensor([row.round_return for row in rows])
    critic_mse(value_model(observations), targets).backward()
    assert any(parameter.grad is not None for parameter in value_model.parameters())
    assert all(parameter.grad is None for parameter in policy.parameters())


# A reached KL boundary halts subsequent actor passes while critic regression continues.
def test_kl_early_stopping_prevents_later_ppo_passes(smoke_setup) -> None:
    policies, critic, collection = smoke_setup
    result = optimize_collection(
        collection=collection,
        policies=policies,
        critic=critic,
        iteration_index=1,
        collected_rounds=0,
        critic_validation_windows=0,
        config=OptimizationConfig(
            actor_learning_rate=0.0,
            critic_learning_rate=0.0,
            actor_weight_override=1.0,
            trajectory_minibatch_size=1,
            approximate_kl_limit=0.0,
        ),
    )
    assert all(report.passes_completed == 1 for report in result.policy_reports)
    assert all(report.early_stopped for report in result.policy_reports)
    assert all(report.approximate_kl == (0.0,) for report in result.policy_reports)
    assert len(result.critic_pass_losses) == 4


@pytest.mark.parametrize("failure_kind", ["output", "gradient"])
def test_nonfinite_update_aborts_without_committing_source_weights(
    smoke_setup, monkeypatch, failure_kind
) -> None:
    policies, critic, collection = smoke_setup
    before = {
        identity: module_fingerprint(policy) for identity, policy in policies.items()
    }
    critic_before = module_fingerprint(critic)

    if failure_kind == "output":
        original_forward = Policy.forward

        def nonfinite_forward(self, observation, legal_mask, hidden_state):
            logits, next_hidden = original_forward(
                self, observation, legal_mask, hidden_state
            )
            return logits * float("nan"), next_hidden

        monkeypatch.setattr(Policy, "forward", nonfinite_forward)
    else:
        monkeypatch.setattr(
            torch.nn.utils,
            "clip_grad_norm_",
            lambda *args, **kwargs: torch.tensor(float("inf")),
        )

    with pytest.raises(NumericalOptimizationError):
        optimize_collection(
            collection=collection,
            policies=policies,
            critic=critic,
            iteration_index=2,
            collected_rounds=0,
            critic_validation_windows=0,
            config=OptimizationConfig(
                actor_weight_override=1.0,
                trajectory_minibatch_size=1,
            ),
        )
    assert all(
        module_fingerprint(policy) == before[identity]
        for identity, policy in policies.items()
    )
    assert module_fingerprint(critic) == critic_before


def test_nonfinite_critic_input_is_rejected_before_loss() -> None:
    with pytest.raises(NumericalOptimizationError):
        critic_mse(torch.tensor([float("inf")]), torch.tensor([0.0]))


def test_finite_inputs_that_overflow_a_loss_are_rejected() -> None:
    with pytest.raises(NumericalOptimizationError):
        critic_mse(torch.tensor([3.0e38]), torch.tensor([0.0]))


def _nested_tensors(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _nested_tensors(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _nested_tensors(item)

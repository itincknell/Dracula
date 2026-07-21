"""Critic regression and recurrent PPO optimization over sealed collection data."""

from __future__ import annotations

import copy
import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from dracula.collection import (
    ActorBucket,
    CollectionData,
    CriticRow,
    PlayerTrajectory,
    PolicyVersion,
    module_fingerprint,
)
from dracula.models import Critic, Policy
from dracula.randomness import derive_pytorch_seed

TRAJECTORY_SHUFFLE_NAMESPACE = "dracula-trajectory-shuffle-v1"
CRITIC_SHUFFLE_NAMESPACE = "dracula-critic-shuffle-v1"


class OptimizationContractViolation(ValueError):
    """Optimization inputs do not satisfy the sealed collection contract."""


class NumericalOptimizationError(RuntimeError):
    """A non-finite value made an optimization transaction unsafe to commit."""


@dataclass(frozen=True, slots=True)
class OptimizationConfig:
    actor_learning_rate: float = 3e-4
    critic_learning_rate: float = 1e-3
    adam_betas: tuple[float, float] = (0.9, 0.999)
    adam_epsilon: float = 1e-8
    gradient_norm_limit: float = 0.5
    optimization_passes: int = 4
    trajectory_minibatch_size: int = 16
    maximum_trajectory_minibatch_size: int = 64
    ppo_clip: float = 0.2
    approximate_kl_limit: float = 0.015
    illegal_probability_threshold: float = 0.001
    burn_in_entropy_coefficient: float = 0.005
    trained_entropy_coefficient: float = 0.001
    minimum_collected_rounds: int = 10_000
    actor_ramp_rounds: int = 10_000
    required_critic_validation_windows: int = 3
    actor_weight_override: float | None = None
    actor_weight_start: float = 0.0
    actor_weight_end: float = 1.0
    actor_requires_critic_validation: bool = True
    device: str = "cpu"

    def __post_init__(self) -> None:
        finite_nonnegative = (
            (self.actor_learning_rate, "actor learning rate"),
            (self.critic_learning_rate, "critic learning rate"),
            (self.approximate_kl_limit, "approximate KL limit"),
            (self.illegal_probability_threshold, "illegal probability threshold"),
            (self.burn_in_entropy_coefficient, "burn-in entropy coefficient"),
            (self.trained_entropy_coefficient, "trained entropy coefficient"),
        )
        for value, label in finite_nonnegative:
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise OptimizationContractViolation(f"{label} must be finite and non-negative")
        if (
            not isinstance(self.adam_epsilon, (int, float))
            or not math.isfinite(self.adam_epsilon)
            or self.adam_epsilon <= 0
        ):
            raise OptimizationContractViolation("Adam epsilon must be finite and positive")
        if (
            not isinstance(self.gradient_norm_limit, (int, float))
            or not math.isfinite(self.gradient_norm_limit)
            or self.gradient_norm_limit <= 0
        ):
            raise OptimizationContractViolation(
                "gradient norm limit must be finite and positive"
            )
        if (
            not isinstance(self.ppo_clip, (int, float))
            or not math.isfinite(self.ppo_clip)
            or not 0 < self.ppo_clip < 1
        ):
            raise OptimizationContractViolation("PPO clip must be between zero and one")
        if not (
            isinstance(self.adam_betas, tuple)
            and len(self.adam_betas) == 2
            and all(
                isinstance(value, (int, float))
                and math.isfinite(value)
                and 0 <= value < 1
                for value in self.adam_betas
            )
        ):
            raise OptimizationContractViolation("Adam betas must be a pair in [0, 1)")
        if type(self.optimization_passes) is not int or self.optimization_passes != 4:
            raise OptimizationContractViolation("optimization requires exactly four passes")
        if (
            type(self.maximum_trajectory_minibatch_size) is not int
            or not 1 <= self.maximum_trajectory_minibatch_size <= 64
        ):
            raise OptimizationContractViolation(
                "maximum trajectory minibatch size must be between one and 64"
            )
        if (
            type(self.trajectory_minibatch_size) is not int
            or self.trajectory_minibatch_size < 1
            or self.trajectory_minibatch_size > self.maximum_trajectory_minibatch_size
        ):
            raise OptimizationContractViolation(
                "trajectory minibatch size must be within the configured maximum"
            )
        for value, label in (
            (self.minimum_collected_rounds, "minimum collected rounds"),
            (self.actor_ramp_rounds, "actor ramp rounds"),
            (
                self.required_critic_validation_windows,
                "required critic validation windows",
            ),
        ):
            if type(value) is not int or value < 1:
                raise OptimizationContractViolation(f"{label} must be positive")
        if self.actor_weight_override is not None and (
            not isinstance(self.actor_weight_override, (int, float))
            or not math.isfinite(self.actor_weight_override)
            or not 0 <= self.actor_weight_override <= 1
        ):
            raise OptimizationContractViolation("actor weight override must be in [0, 1]")
        for value, label in (
            (self.actor_weight_start, "actor starting weight"),
            (self.actor_weight_end, "actor ending weight"),
        ):
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
                or not 0 <= value <= 1
            ):
                raise OptimizationContractViolation(f"{label} must be in [0, 1]")
        if self.actor_weight_end < self.actor_weight_start:
            raise OptimizationContractViolation(
                "actor ending weight cannot be below its starting weight"
            )
        if type(self.actor_requires_critic_validation) is not bool:
            raise OptimizationContractViolation(
                "actor critic-validation requirement must be Boolean"
            )
        try:
            torch.device(self.device)
        except (RuntimeError, TypeError) as error:
            raise OptimizationContractViolation("optimization device is invalid") from error


@dataclass(frozen=True, slots=True)
class PpoSurrogate:
    loss: Tensor
    probability_ratios: Tensor
    unclipped_objective: Tensor
    clipped_objective: Tensor
    clipped_branch_selected: Tensor


@dataclass(frozen=True, slots=True)
class PreparedTrajectory:
    trajectory: PlayerTrajectory
    old_log_probabilities: Tensor
    old_critic_values: Tensor
    round_returns: Tensor
    advantages: Tensor
    normalized_advantages: Tensor


@dataclass(frozen=True, slots=True)
class PreparedActorBucket:
    learner: PolicyVersion
    trajectories: tuple[PreparedTrajectory, ...]
    fixed_data_digest: str

    @property
    def learned_row_count(self) -> int:
        return sum(item.old_log_probabilities.numel() for item in self.trajectories)


@dataclass(frozen=True, slots=True)
class ReplayedTrajectory:
    raw_logits: tuple[Tensor, ...]
    hidden_states: tuple[Tensor, ...]
    learned_step_indices: tuple[int, ...]
    learned_log_probabilities: Tensor
    learned_entropies: Tensor
    learned_illegal_probabilities: Tensor
    learned_illegal_penalties: Tensor


@dataclass(frozen=True, slots=True)
class PolicyPassMetrics:
    policy_loss: float
    ppo_loss: float
    legal_entropy: float
    mean_illegal_probability: float
    illegal_loss: float
    gradient_norm: float


@dataclass(frozen=True, slots=True)
class PolicyOptimizationReport:
    learner: PolicyVersion
    passes_completed: int
    approximate_kl: tuple[float, ...]
    early_stopped: bool
    fixed_data_digest: str
    pass_metrics: tuple[PolicyPassMetrics, ...]


@dataclass(frozen=True, slots=True)
class OptimizationResult:
    policies: Mapping[PolicyVersion, Policy]
    critic: Critic
    policy_reports: tuple[PolicyOptimizationReport, ...]
    critic_pass_losses: tuple[float, ...]
    critic_pass_gradient_norms: tuple[float, ...]
    policy_optimizer_states: Mapping[PolicyVersion, dict[str, object]]
    critic_optimizer_state: dict[str, object]
    actor_weight: float
    entropy_coefficient: float
    peak_device_memory_bytes: int | None


def actor_weight(
    collected_rounds: int,
    critic_validation_windows: int,
    config: OptimizationConfig,
) -> float:
    """Return the configured burn-in and linear-ramp policy coefficient."""

    if type(collected_rounds) is not int or collected_rounds < 0:
        raise OptimizationContractViolation("collected rounds must be non-negative")
    if type(critic_validation_windows) is not int or critic_validation_windows < 0:
        raise OptimizationContractViolation(
            "critic validation window count must be non-negative"
        )
    if config.actor_weight_override is not None:
        return float(config.actor_weight_override)
    if collected_rounds < config.minimum_collected_rounds:
        return 0.0
    if (
        config.actor_requires_critic_validation
        and critic_validation_windows < config.required_critic_validation_windows
    ):
        return 0.0
    progress = min(
        1.0,
        max(
            0.0,
            (collected_rounds - config.minimum_collected_rounds)
            / config.actor_ramp_rounds,
        ),
    )
    return config.actor_weight_start + progress * (
        config.actor_weight_end - config.actor_weight_start
    )


def entropy_coefficient(weight: float, config: OptimizationConfig) -> float:
    if not math.isfinite(weight) or not 0 <= weight <= 1:
        raise OptimizationContractViolation("actor weight must be in [0, 1]")
    if config.actor_weight_override is not None:
        return (
            config.trained_entropy_coefficient
            if weight >= 1.0
            else config.burn_in_entropy_coefficient
        )
    width = config.actor_weight_end - config.actor_weight_start
    if width == 0:
        progress = 1.0 if weight >= config.actor_weight_end else 0.0
    else:
        progress = min(
            1.0, max(0.0, (weight - config.actor_weight_start) / width)
        )
    return config.burn_in_entropy_coefficient + progress * (
        config.trained_entropy_coefficient - config.burn_in_entropy_coefficient
    )


def critic_mse(predictions: Tensor, round_returns: Tensor) -> Tensor:
    if predictions.shape != round_returns.shape or predictions.numel() == 0:
        raise OptimizationContractViolation(
            "critic predictions and returns must have the same nonempty shape"
        )
    _require_finite_tensor(predictions, "critic predictions")
    _require_finite_tensor(round_returns, "critic returns")
    loss = torch.mean(torch.square(predictions - round_returns))
    _require_finite_tensor(loss, "critic loss")
    return loss


def normalize_advantages(advantages: Tensor) -> Tensor:
    if advantages.ndim != 1 or advantages.numel() == 0:
        raise OptimizationContractViolation("advantages must be a nonempty vector")
    _require_finite_tensor(advantages, "advantages")
    normalized = (advantages - advantages.mean()) / (
        advantages.std(unbiased=False) + 1e-8
    )
    _require_finite_tensor(normalized, "normalized advantages")
    return normalized


def ppo_clipped_surrogate(
    new_log_probabilities: Tensor,
    old_log_probabilities: Tensor,
    normalized_advantages: Tensor,
    *,
    clip: float = 0.2,
) -> PpoSurrogate:
    if not (
        new_log_probabilities.shape
        == old_log_probabilities.shape
        == normalized_advantages.shape
    ) or new_log_probabilities.numel() == 0:
        raise OptimizationContractViolation("PPO inputs must have one nonempty shape")
    for tensor, label in (
        (new_log_probabilities, "new log probabilities"),
        (old_log_probabilities, "old log probabilities"),
        (normalized_advantages, "normalized advantages"),
    ):
        _require_finite_tensor(tensor, label)
    if not math.isfinite(clip) or not 0 < clip < 1:
        raise OptimizationContractViolation("PPO clip must be between zero and one")
    ratios = torch.exp(new_log_probabilities - old_log_probabilities)
    _require_finite_tensor(ratios, "PPO probability ratios")
    unclipped = ratios * normalized_advantages
    clipped = torch.clamp(ratios, 1.0 - clip, 1.0 + clip) * normalized_advantages
    _require_finite_tensor(unclipped, "unclipped PPO objective")
    _require_finite_tensor(clipped, "clipped PPO objective")
    selected = clipped < unclipped
    loss = -torch.minimum(unclipped, clipped).mean()
    _require_finite_tensor(loss, "PPO loss")
    return PpoSurrogate(
        loss=loss,
        probability_ratios=ratios,
        unclipped_objective=unclipped,
        clipped_objective=clipped,
        clipped_branch_selected=selected,
    )


def legal_action_entropy(raw_logits: Tensor, legal_mask: Tensor) -> Tensor:
    flattened_logits, flattened_mask = _flatten_actions(raw_logits, legal_mask)
    masked_logits = flattened_logits.masked_fill(~flattened_mask, float("-inf"))
    log_probabilities = torch.log_softmax(masked_logits, dim=-1)
    probabilities = torch.softmax(masked_logits, dim=-1)
    safe_log_probabilities = log_probabilities.masked_fill(~flattened_mask, 0.0)
    terms = probabilities * safe_log_probabilities
    entropies = -terms.sum(dim=-1)
    _require_finite_tensor(entropies, "legal-action entropy")
    return entropies


def illegal_probability_loss(
    raw_logits: Tensor,
    legal_mask: Tensor,
    *,
    threshold: float = 0.001,
) -> tuple[Tensor, Tensor, Tensor]:
    if not math.isfinite(threshold) or threshold < 0:
        raise OptimizationContractViolation(
            "illegal probability threshold must be finite and non-negative"
        )
    flattened_logits, flattened_mask = _flatten_actions(raw_logits, legal_mask)
    probabilities = torch.softmax(flattened_logits, dim=-1)
    illegal_probabilities = (probabilities * (~flattened_mask).to(probabilities.dtype)).sum(
        dim=-1
    )
    penalties = torch.relu(illegal_probabilities - threshold).square()
    _require_finite_tensor(illegal_probabilities, "illegal probabilities")
    _require_finite_tensor(penalties, "illegal probability penalties")
    return penalties.mean(), illegal_probabilities, penalties


def approximate_kl(
    old_log_probabilities: Tensor, new_log_probabilities: Tensor
) -> Tensor:
    if old_log_probabilities.shape != new_log_probabilities.shape:
        raise OptimizationContractViolation("KL log probabilities must share a shape")
    _require_finite_tensor(old_log_probabilities, "old KL log probabilities")
    _require_finite_tensor(new_log_probabilities, "new KL log probabilities")
    value = (old_log_probabilities - new_log_probabilities).mean()
    _require_finite_tensor(value, "approximate KL")
    return value


def prepare_actor_bucket(bucket: ActorBucket) -> PreparedActorBucket:
    old_log_probabilities: list[float] = []
    old_critic_values: list[float] = []
    round_returns: list[float] = []
    learned_counts: list[int] = []
    for trajectory in bucket.trajectories:
        count = 0
        for transition in trajectory.transitions:
            if not transition.actor_loss_mask:
                continue
            if (
                transition.action_log_probability is None
                or transition.critic_value is None
                or transition.round_return is None
            ):
                raise OptimizationContractViolation(
                    "learned transition is missing fixed behavior data"
                )
            old_log_probabilities.append(transition.action_log_probability)
            old_critic_values.append(transition.critic_value)
            round_returns.append(transition.round_return)
            count += 1
        learned_counts.append(count)
    old_logs = torch.tensor(old_log_probabilities, dtype=torch.float32)
    old_values = torch.tensor(old_critic_values, dtype=torch.float32)
    returns = torch.tensor(round_returns, dtype=torch.float32)
    advantages = returns - old_values
    normalized = normalize_advantages(advantages)
    prepared: list[PreparedTrajectory] = []
    start = 0
    for trajectory, count in zip(bucket.trajectories, learned_counts, strict=True):
        end = start + count
        prepared.append(
            PreparedTrajectory(
                trajectory=trajectory,
                old_log_probabilities=old_logs[start:end].clone(),
                old_critic_values=old_values[start:end].clone(),
                round_returns=returns[start:end].clone(),
                advantages=advantages[start:end].clone(),
                normalized_advantages=normalized[start:end].clone(),
            )
        )
        start = end
    result = PreparedActorBucket(
        learner=bucket.learner,
        trajectories=tuple(prepared),
        fixed_data_digest="",
    )
    return PreparedActorBucket(
        learner=result.learner,
        trajectories=result.trajectories,
        fixed_data_digest=_prepared_data_digest(result.trajectories),
    )


def replay_current_policy(
    policy: Policy,
    prepared: PreparedTrajectory,
    *,
    device: torch.device | str,
    retain_hidden_gradients: bool = False,
    illegal_probability_threshold: float = 0.001,
) -> ReplayedTrajectory:
    """Rebuild one current-policy recurrent chain from its game-boundary zero state."""

    target_device = torch.device(device)
    hidden = policy.initial_hidden(device=target_device)
    raw_logits: list[Tensor] = []
    hidden_states: list[Tensor] = []
    learned_indices: list[int] = []
    learned_logs: list[Tensor] = []
    learned_entropies: list[Tensor] = []
    illegal_probabilities: list[Tensor] = []
    illegal_penalties: list[Tensor] = []

    for step_index, transition in enumerate(prepared.trajectory.transitions):
        # Collection runs under inference mode. Cloning makes ordinary tensors that
        # autograd may safely retain while reconstructing the current recurrent graph.
        observation = transition.policy_input.observation.to(target_device).clone()
        legal_mask = transition.policy_input.legal_mask.to(target_device).clone()
        logits, hidden = policy(observation, legal_mask, hidden)
        _require_finite_tensor(logits, "policy logits")
        _require_finite_tensor(hidden, "policy hidden state")
        if retain_hidden_gradients and hidden.requires_grad:
            hidden.retain_grad()
        raw_logits.append(logits)
        hidden_states.append(hidden)
        if not transition.actor_loss_mask:
            continue
        flattened_logits = logits.flatten()
        flattened_mask = legal_mask.flatten()
        masked_logits = flattened_logits.masked_fill(~flattened_mask, float("-inf"))
        log_probability = torch.log_softmax(masked_logits, dim=0)[
            transition.action_index
        ]
        entropy = legal_action_entropy(logits.unsqueeze(0), legal_mask.unsqueeze(0))[0]
        _, illegal_probability, penalty = illegal_probability_loss(
            logits.unsqueeze(0),
            legal_mask.unsqueeze(0),
            threshold=illegal_probability_threshold,
        )
        learned_indices.append(step_index)
        learned_logs.append(log_probability)
        learned_entropies.append(entropy)
        illegal_probabilities.append(illegal_probability[0])
        illegal_penalties.append(penalty[0])

    if len(learned_indices) != prepared.old_log_probabilities.numel():
        raise OptimizationContractViolation("replay learned-row count changed")
    return ReplayedTrajectory(
        raw_logits=tuple(raw_logits),
        hidden_states=tuple(hidden_states),
        learned_step_indices=tuple(learned_indices),
        learned_log_probabilities=torch.stack(learned_logs),
        learned_entropies=torch.stack(learned_entropies),
        learned_illegal_probabilities=torch.stack(illegal_probabilities),
        learned_illegal_penalties=torch.stack(illegal_penalties),
    )


def optimize_collection(
    *,
    collection: CollectionData,
    policies: Mapping[PolicyVersion, Policy],
    critic: Critic,
    iteration_index: int,
    collected_rounds: int,
    critic_validation_windows: int,
    config: OptimizationConfig | None = None,
    policy_optimizer_states: Mapping[PolicyVersion, dict[str, object]] | None = None,
    critic_optimizer_state: dict[str, object] | None = None,
) -> OptimizationResult:
    """Run one transactional four-pass update without mutating source models."""

    selected_config = config or OptimizationConfig()
    if type(iteration_index) is not int or iteration_index < 0:
        raise OptimizationContractViolation("iteration index must be non-negative")
    _validate_behavior_models(collection, policies, critic)
    weight = actor_weight(collected_rounds, critic_validation_windows, selected_config)
    entropy = entropy_coefficient(weight, selected_config)
    device = torch.device(selected_config.device)

    # Caller-owned models are the commit boundary. All numerical work happens on copies.
    updated_policies = {
        identity: copy.deepcopy(policies[identity]).cpu().train()
        for identity in collection.schedule.policies
    }
    updated_critic = copy.deepcopy(critic).cpu().train()
    for model in (*updated_policies.values(), updated_critic):
        model.requires_grad_(True)

    prepared_buckets = {
        bucket.learner: prepare_actor_bucket(bucket) for bucket in collection.actor_buckets
    }
    pass_metrics: dict[PolicyVersion, list[PolicyPassMetrics]] = {
        identity: [] for identity in collection.schedule.policies
    }
    kl_values: dict[PolicyVersion, list[float]] = {
        identity: [] for identity in collection.schedule.policies
    }
    halted: set[PolicyVersion] = set()
    optimizer_states: dict[PolicyVersion, dict[str, object]] = {
        identity: copy.deepcopy(state)
        for identity, state in (policy_optimizer_states or {}).items()
    }
    if not set(optimizer_states).issubset(collection.schedule.policies):
        raise OptimizationContractViolation(
            "policy optimizer states do not match the active population"
        )
    saved_critic_optimizer_state = copy.deepcopy(critic_optimizer_state)
    critic_losses: list[float] = []
    critic_gradient_norms: list[float] = []
    peak_device_memory_bytes = _device_memory_bytes(device)

    for pass_index in range(selected_config.optimization_passes):
        for identity in collection.schedule.policies:
            if identity in halted:
                continue
            policy = updated_policies[identity].to(device).train()
            prepared = prepared_buckets[identity]
            _assert_prepared_unchanged(prepared)
            # Adam state is serialized between actors so only one policy optimizer is live.
            policy_optimizer = _adam(
                policy, selected_config.actor_learning_rate, selected_config
            )
            if identity in optimizer_states:
                policy_optimizer.load_state_dict(optimizer_states[identity])
            metrics = _optimize_policy_pass(
                policy=policy,
                prepared=prepared,
                optimizer=policy_optimizer,
                run_root_seed=collection.run_root_seed,
                iteration_index=iteration_index,
                pass_index=pass_index,
                actor_weight_value=weight,
                entropy_coefficient_value=entropy,
                config=selected_config,
                device=device,
            )
            optimizer_states[identity] = _cpu_copy(policy_optimizer.state_dict())
            del policy_optimizer
            pass_metrics[identity].append(metrics)
            kl = _bucket_approximate_kl(policy, prepared, device=device)
            kl_number = float(kl.detach().cpu().item())
            kl_values[identity].append(kl_number)
            if kl_number >= selected_config.approximate_kl_limit:
                halted.add(identity)
            _assert_prepared_unchanged(prepared)
            peak_device_memory_bytes = _maximum_optional(
                peak_device_memory_bytes, _device_memory_bytes(device)
            )
            policy.zero_grad(set_to_none=True)
            policy.cpu()

        updated_critic.to(device).train()
        critic_optimizer = _adam(
            updated_critic, selected_config.critic_learning_rate, selected_config
        )
        if saved_critic_optimizer_state is not None:
            critic_optimizer.load_state_dict(saved_critic_optimizer_state)
        critic_loss, critic_gradient_norm = _optimize_critic_pass(
            critic=updated_critic,
            rows=collection.critic_rows,
            optimizer=critic_optimizer,
            run_root_seed=collection.run_root_seed,
            iteration_index=iteration_index,
            critic_version=collection.critic_version,
            pass_index=pass_index,
            config=selected_config,
            device=device,
        )
        saved_critic_optimizer_state = _cpu_copy(critic_optimizer.state_dict())
        del critic_optimizer
        critic_losses.append(critic_loss)
        critic_gradient_norms.append(critic_gradient_norm)
        peak_device_memory_bytes = _maximum_optional(
            peak_device_memory_bytes, _device_memory_bytes(device)
        )
        updated_critic.zero_grad(set_to_none=True)
        updated_critic.cpu()

    reports = tuple(
        PolicyOptimizationReport(
            learner=identity,
            passes_completed=len(pass_metrics[identity]),
            approximate_kl=tuple(kl_values[identity]),
            early_stopped=identity in halted,
            fixed_data_digest=prepared_buckets[identity].fixed_data_digest,
            pass_metrics=tuple(pass_metrics[identity]),
        )
        for identity in collection.schedule.policies
    )
    return OptimizationResult(
        policies=updated_policies,
        critic=updated_critic,
        policy_reports=reports,
        critic_pass_losses=tuple(critic_losses),
        critic_pass_gradient_norms=tuple(critic_gradient_norms),
        policy_optimizer_states=optimizer_states,
        critic_optimizer_state=saved_critic_optimizer_state or {},
        actor_weight=weight,
        entropy_coefficient=entropy,
        peak_device_memory_bytes=peak_device_memory_bytes,
    )


def _optimize_policy_pass(
    *,
    policy: Policy,
    prepared: PreparedActorBucket,
    optimizer: torch.optim.Optimizer,
    run_root_seed: str,
    iteration_index: int,
    pass_index: int,
    actor_weight_value: float,
    entropy_coefficient_value: float,
    config: OptimizationConfig,
    device: torch.device,
) -> PolicyPassMetrics:
    order = _shuffled_indices(
        len(prepared.trajectories),
        TRAJECTORY_SHUFFLE_NAMESPACE,
        run_root_seed,
        str(iteration_index),
        prepared.learner.version,
        str(pass_index),
    )
    weighted_totals = torch.zeros(5, dtype=torch.float64)
    gradient_norms: list[float] = []
    total_rows = 0
    for start in range(0, len(order), config.trajectory_minibatch_size):
        selected = [
            prepared.trajectories[index]
            for index in order[start : start + config.trajectory_minibatch_size]
        ]
        optimizer.zero_grad(set_to_none=True)
        replays = [
            replay_current_policy(
                policy,
                item,
                device=device,
                illegal_probability_threshold=config.illegal_probability_threshold,
            )
            for item in selected
        ]
        new_logs = torch.cat([item.learned_log_probabilities for item in replays])
        old_logs = torch.cat([item.old_log_probabilities for item in selected]).to(device)
        normalized = torch.cat([item.normalized_advantages for item in selected]).to(
            device
        )
        entropies = torch.cat([item.learned_entropies for item in replays])
        illegal_penalties = torch.cat(
            [item.learned_illegal_penalties for item in replays]
        )
        illegal_probabilities = torch.cat(
            [item.learned_illegal_probabilities for item in replays]
        )
        ppo = ppo_clipped_surrogate(
            new_logs, old_logs, normalized, clip=config.ppo_clip
        )
        mean_entropy = entropies.mean()
        illegal_loss = illegal_penalties.mean()
        total_loss = (
            actor_weight_value * ppo.loss
            - entropy_coefficient_value * mean_entropy
            + illegal_loss
        )
        _require_finite_tensor(total_loss, "policy loss")
        total_loss.backward()
        gradient_norms.append(
            _clip_and_validate_gradients(policy, config.gradient_norm_limit)
        )
        optimizer.step()
        _validate_module_finite(policy, "policy")
        row_count = new_logs.numel()
        weighted_totals += torch.tensor(
            [
                float(total_loss.detach().cpu()),
                float(ppo.loss.detach().cpu()),
                float(mean_entropy.detach().cpu()),
                float(illegal_probabilities.mean().detach().cpu()),
                float(illegal_loss.detach().cpu()),
            ],
            dtype=torch.float64,
        ) * row_count
        total_rows += row_count
    means = weighted_totals / total_rows
    return PolicyPassMetrics(
        *(float(value) for value in means),
        sum(gradient_norms) / len(gradient_norms),
    )


def _optimize_critic_pass(
    *,
    critic: Critic,
    rows: Sequence[CriticRow],
    optimizer: torch.optim.Optimizer,
    run_root_seed: str,
    iteration_index: int,
    critic_version: str,
    pass_index: int,
    config: OptimizationConfig,
    device: torch.device,
) -> tuple[float, float]:
    groups = _critic_trajectory_groups(rows)
    order = _shuffled_indices(
        len(groups),
        CRITIC_SHUFFLE_NAMESPACE,
        run_root_seed,
        str(iteration_index),
        critic_version,
        str(pass_index),
    )
    weighted_loss = 0.0
    total_rows = 0
    gradient_norms: list[float] = []
    for start in range(0, len(order), config.trajectory_minibatch_size):
        batch_groups = [
            groups[index] for index in order[start : start + config.trajectory_minibatch_size]
        ]
        batch_rows = [row for group in batch_groups for row in group]
        observations = torch.stack([row.observation for row in batch_rows]).to(device)
        returns = torch.tensor(
            [row.round_return for row in batch_rows], dtype=torch.float32, device=device
        )
        optimizer.zero_grad(set_to_none=True)
        predictions = critic(observations)
        _require_finite_tensor(predictions, "critic outputs")
        loss = critic_mse(predictions, returns)
        _require_finite_tensor(loss, "critic loss")
        loss.backward()
        gradient_norms.append(
            _clip_and_validate_gradients(critic, config.gradient_norm_limit)
        )
        optimizer.step()
        _validate_module_finite(critic, "critic")
        weighted_loss += float(loss.detach().cpu()) * len(batch_rows)
        total_rows += len(batch_rows)
    return weighted_loss / total_rows, sum(gradient_norms) / len(gradient_norms)


def _bucket_approximate_kl(
    policy: Policy, prepared: PreparedActorBucket, *, device: torch.device
) -> Tensor:
    old_logs: list[Tensor] = []
    new_logs: list[Tensor] = []
    with torch.no_grad():
        for item in prepared.trajectories:
            replay = replay_current_policy(policy, item, device=device)
            old_logs.append(item.old_log_probabilities.to(device))
            new_logs.append(replay.learned_log_probabilities)
    return approximate_kl(torch.cat(old_logs), torch.cat(new_logs))


def _flatten_actions(raw_logits: Tensor, legal_mask: Tensor) -> tuple[Tensor, Tensor]:
    if raw_logits.shape != legal_mask.shape or raw_logits.shape[-2:] != (4, 8):
        raise OptimizationContractViolation(
            "raw logits and legal mask must share a trailing [4, 8] shape"
        )
    if legal_mask.dtype is not torch.bool:
        raise OptimizationContractViolation("legal mask must have Boolean dtype")
    _require_finite_tensor(raw_logits, "raw policy logits")
    flattened_logits = raw_logits.reshape(*raw_logits.shape[:-2], 32)
    flattened_mask = legal_mask.reshape(*legal_mask.shape[:-2], 32)
    if not flattened_mask.any(dim=-1).all():
        raise OptimizationContractViolation("every policy row must have a legal action")
    return flattened_logits, flattened_mask


def _validate_behavior_models(
    collection: CollectionData,
    policies: Mapping[PolicyVersion, Policy],
    critic: Critic,
) -> None:
    if set(policies) != set(collection.schedule.policies):
        raise OptimizationContractViolation(
            "policy models do not match collected behavior versions"
        )
    records = {record.identity: record for record in collection.behavior_policies}
    for identity in collection.schedule.policies:
        policy = policies[identity]
        if not isinstance(policy, Policy):
            raise OptimizationContractViolation("actor update requires Policy modules")
        if module_fingerprint(policy) != records[identity].parameter_fingerprint:
            raise OptimizationContractViolation(
                f"policy parameters do not match behavior evidence for {identity.policy_id}"
            )
    if not isinstance(critic, Critic):
        raise OptimizationContractViolation("critic update requires a Critic module")
    if module_fingerprint(critic) != collection.critic_parameter_fingerprint:
        raise OptimizationContractViolation("critic parameters do not match V_old evidence")


def _adam(
    model: nn.Module, learning_rate: float, config: OptimizationConfig
) -> torch.optim.Adam:
    return torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        betas=config.adam_betas,
        eps=config.adam_epsilon,
    )


def _cpu_copy(value):
    if isinstance(value, Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_copy(item) for item in value)
    return copy.deepcopy(value)


def _device_memory_bytes(device: torch.device) -> int | None:
    if device.type != "mps":
        return None
    try:
        return int(torch.mps.driver_allocated_memory())
    except (AttributeError, RuntimeError):
        return None


def _maximum_optional(first: int | None, second: int | None) -> int | None:
    values = [value for value in (first, second) if value is not None]
    return max(values) if values else None


def _shuffled_indices(count: int, namespace: str, *components: str) -> list[int]:
    if count < 1:
        raise OptimizationContractViolation("optimization bucket cannot be empty")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(derive_pytorch_seed(namespace, *components))
    return torch.randperm(count, generator=generator).tolist()


def _critic_trajectory_groups(rows: Sequence[CriticRow]) -> tuple[tuple[CriticRow, ...], ...]:
    grouped: dict[tuple[str, PolicyVersion, object], list[CriticRow]] = {}
    for row in rows:
        grouped.setdefault((row.fixture_id, row.learner, row.player), []).append(row)
    result = tuple(tuple(values) for values in grouped.values())
    if not result or any(len(group) != 21 for group in result):
        raise OptimizationContractViolation(
            "critic rows must form complete 21-row player trajectories"
        )
    return result


def _prepared_data_digest(trajectories: Sequence[PreparedTrajectory]) -> str:
    digest = hashlib.sha256()
    for item in trajectories:
        digest.update(item.trajectory.fixture_id.encode("ascii"))
        digest.update(b"\0")
        for tensor in (
            item.old_log_probabilities,
            item.old_critic_values,
            item.round_returns,
            item.advantages,
            item.normalized_advantages,
        ):
            cpu = tensor.detach().cpu().contiguous()
            digest.update(cpu.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _assert_prepared_unchanged(prepared: PreparedActorBucket) -> None:
    if _prepared_data_digest(prepared.trajectories) != prepared.fixed_data_digest:
        raise OptimizationContractViolation(
            "fixed behavior advantages or critic estimates changed during update"
        )


def _require_finite_tensor(tensor: Tensor, label: str) -> None:
    if not torch.isfinite(tensor).all():
        raise NumericalOptimizationError(f"{label} contains NaN or infinity")


def _clip_and_validate_gradients(model: nn.Module, limit: float) -> float:
    parameters = [parameter for parameter in model.parameters() if parameter.grad is not None]
    if not parameters:
        raise NumericalOptimizationError("optimization produced no gradients")
    for parameter in parameters:
        _require_finite_tensor(parameter.grad, "gradient")
    try:
        norm = torch.nn.utils.clip_grad_norm_(
            parameters, max_norm=limit, error_if_nonfinite=True
        )
    except RuntimeError as error:
        raise NumericalOptimizationError("gradient norm is NaN or infinite") from error
    _require_finite_tensor(torch.as_tensor(norm), "gradient norm")
    for parameter in parameters:
        _require_finite_tensor(parameter.grad, "clipped gradient")
    return float(torch.as_tensor(norm).detach().cpu().item())


def _validate_module_finite(model: nn.Module, label: str) -> None:
    for tensor in model.state_dict().values():
        _require_finite_tensor(tensor, f"{label} parameter")

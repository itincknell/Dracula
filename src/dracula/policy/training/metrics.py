"""Compute distributional loss and evaluation slices for standalone policies.

Training and validation share the same representative-masked probability
calculation. This module owns that mathematics and its metric aggregation; it
does not load datasets, mutate model parameters, select checkpoints, or write
artifacts.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, NamedTuple

import torch
from torch import Tensor
from torch.nn import functional as F

from dracula.decision.action_mask import apply_representative_mask
from dracula.policy.model import PolicyModel
from dracula.policy.training.contracts import (
    BATCH_SIZE,
    PolicyTrainingError,
    DistributionMetrics,
    EvaluationMetrics,
)
from dracula.decision.bridge import ACTION_COUNT, HAND_SLOT_COUNT, POLICY_POSITION_COUNT


class _PolicyStatistics(NamedTuple):
    """Per-example values shared by loss evaluation and metric aggregation."""

    cross_entropy: Tensor
    kl_divergence: Tensor
    target_entropy: Tensor
    model_entropy: Tensor
    top_one: Tensor
    top_two: Tensor
    top_three: Tensor


@dataclass(slots=True)
class _MetricAccumulator:
    """Accumulate per-example tensors before producing one averaged metric set."""

    count: int = 0
    cross_entropy: float = 0.0
    kl_divergence: float = 0.0
    target_entropy: float = 0.0
    model_entropy: float = 0.0
    top_one: int = 0
    top_two: int = 0
    top_three: int = 0

    def add(
        self,
        values: _PolicyStatistics,
        indexes: Tensor | None = None,
    ) -> None:
        """Add all examples or one indexed subgroup from a calculated batch."""

        cross_entropy, kl, target_entropy, model_entropy, top1, top2, top3 = values
        if indexes is not None:
            cross_entropy = cross_entropy.index_select(0, indexes)
            kl = kl.index_select(0, indexes)
            target_entropy = target_entropy.index_select(0, indexes)
            model_entropy = model_entropy.index_select(0, indexes)
            top1 = top1.index_select(0, indexes)
            top2 = top2.index_select(0, indexes)
            top3 = top3.index_select(0, indexes)
        self.count += int(cross_entropy.numel())
        self.cross_entropy += float(cross_entropy.sum().item())
        self.kl_divergence += float(kl.sum().item())
        self.target_entropy += float(target_entropy.sum().item())
        self.model_entropy += float(model_entropy.sum().item())
        self.top_one += int(top1.sum().item())
        self.top_two += int(top2.sum().item())
        self.top_three += int(top3.sum().item())

    def finish(self) -> DistributionMetrics:
        """Return averages, using an explicit empty result for absent slices."""

        if self.count == 0:
            return DistributionMetrics(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        return DistributionMetrics(
            self.count,
            self.cross_entropy / self.count,
            self.kl_divergence / self.count,
            self.target_entropy / self.count,
            self.model_entropy / self.count,
            self.top_one / self.count,
            self.top_two / self.count,
            self.top_three / self.count,
        )


def _flat_mask_for_batch(
    logits: Tensor,
    representative_masks: Tensor,
    targets: Tensor,
    selected_actions: Tensor,
) -> Tensor:
    """Check dynamic tensor compatibility and return the flattened mask."""

    if (
        logits.dtype is not torch.float32
        or logits.ndim != 3
        or logits.shape[1:] != (HAND_SLOT_COUNT, POLICY_POSITION_COUNT)
        or representative_masks.dtype is not torch.bool
        or representative_masks.shape != logits.shape
        or targets.dtype is not torch.float32
        or targets.shape != (logits.shape[0], ACTION_COUNT)
        or selected_actions.dtype is not torch.long
        or selected_actions.shape != (logits.shape[0],)
        or not bool(torch.isfinite(logits).all().item())
        or not bool(torch.isfinite(targets).all().item())
    ):
        raise PolicyTrainingError("distributional policy batch is malformed")
    # Visit sums, representative membership, and selected-action legality were
    # established once when the immutable rows entered the dataset.
    return representative_masks.flatten(start_dim=1)


def distributional_policy_statistics(
    logits: Tensor,
    representative_masks: Tensor,
    targets: Tensor,
    selected_actions: Tensor,
) -> _PolicyStatistics:
    """Compute per-row loss, entropy, divergence, and selected-action agreement."""

    masks = _flat_mask_for_batch(
        logits, representative_masks, targets, selected_actions
    )

    # Masking precedes log-softmax, so illegal and non-representative actions
    # receive exactly zero probability and no loss gradient.
    flattened = apply_representative_mask(
        logits, representative_masks
    ).flatten(start_dim=1)
    log_probabilities = F.log_softmax(flattened, dim=1)
    probabilities = log_probabilities.exp()
    # Masked positions contain -infinity after log-softmax. Replace those
    # entries with zero only for entropy multiplication, where 0 * -inf would
    # otherwise produce NaN despite the action having zero target/probability.
    safe_log_probabilities = torch.where(
        masks, log_probabilities, torch.zeros_like(log_probabilities)
    )
    # Cross-entropy measures how well the model reproduces all 128 visit shares.
    # Subtracting the target's own entropy gives KL divergence without a second
    # log-softmax pass.
    cross_entropy = -(targets * safe_log_probabilities).sum(dim=1)
    target_log = torch.where(targets > 0, targets.log(), torch.zeros_like(targets))
    target_entropy = -(targets * target_log).sum(dim=1)
    kl_divergence = cross_entropy - target_entropy
    model_entropy = -(probabilities * safe_log_probabilities).sum(dim=1)

    # Stable sorting makes equal-logit agreement metrics use canonical action order.
    ranks = torch.argsort(flattened, dim=1, descending=True, stable=True)
    top_one = ranks[:, :1].eq(selected_actions[:, None]).any(dim=1)
    top_two = ranks[:, :2].eq(selected_actions[:, None]).any(dim=1)
    top_three = ranks[:, :3].eq(selected_actions[:, None]).any(dim=1)
    if not all(
        bool(torch.isfinite(value).all().item())
        for value in (cross_entropy, kl_divergence, target_entropy, model_entropy)
    ):
        raise PolicyTrainingError("distributional policy metrics are non-finite")
    return _PolicyStatistics(
        cross_entropy,
        kl_divergence,
        target_entropy,
        model_entropy,
        top_one,
        top_two,
        top_three,
    )


def distributional_policy_cross_entropy(
    logits: Tensor,
    representative_masks: Tensor,
    targets: Tensor,
) -> Tensor:
    """Return mean visit-distribution cross-entropy over representative actions."""

    selected = targets.argmax(dim=1)
    return distributional_policy_statistics(
        logits, representative_masks, targets, selected
    )[0].mean()


def _metric_groups(
    accumulators: dict[str, _MetricAccumulator],
    labels: Tensor,
    values: _PolicyStatistics,
) -> None:
    """Add one batch to each distinct categorical evaluation slice."""

    for label in torch.unique(labels).tolist():
        indexes = torch.nonzero(labels == label, as_tuple=False).flatten()
        accumulators.setdefault(str(int(label)), _MetricAccumulator()).add(
            values, indexes
        )


@dataclass(slots=True)
class _EvaluationAccumulator:
    """Collect overall and categorical metrics across evaluation batches."""

    total: _MetricAccumulator = field(default_factory=_MetricAccumulator)
    placements: dict[str, _MetricAccumulator] = field(default_factory=dict)
    roles: dict[str, _MetricAccumulator] = field(default_factory=dict)
    dealer_status: dict[str, _MetricAccumulator] = field(default_factory=dict)
    representative_actions: int = 0
    inference_seconds: float = 0.0

    def add(
        self,
        dataset: Any,
        indexes: Tensor,
        masks: Tensor,
        values: _PolicyStatistics,
        inference_seconds: float,
    ) -> None:
        """Add one device-independent batch and its categorical labels."""

        cpu_values = _PolicyStatistics(*(value.detach().cpu() for value in values))
        self.total.add(cpu_values)
        self.representative_actions += int(masks.sum().item())
        placements = dataset.placements.index_select(0, indexes).long()
        players = dataset.players.index_select(0, indexes).long()
        dealers = dataset.dealers.index_select(0, indexes).long()
        _metric_groups(self.placements, placements, cpu_values)
        _metric_groups(self.roles, players, cpu_values)
        _metric_groups(self.dealer_status, players.eq(dealers).long(), cpu_values)
        self.inference_seconds += inference_seconds

    def finish(self, example_count: int) -> EvaluationMetrics:
        """Convert accumulated sums into the public evaluation result."""

        denominator = max(example_count, 1)
        return EvaluationMetrics(
            self.total.finish(),
            {key: value.finish() for key, value in sorted(self.placements.items())},
            {
                ("queen" if key == "0" else "king"): value.finish()
                for key, value in sorted(self.roles.items())
            },
            {
                ("dealer" if key == "1" else "non-dealer"): value.finish()
                for key, value in sorted(self.dealer_status.items())
            },
            self.representative_actions / denominator,
            self.inference_seconds / denominator,
        )


def _evaluate_batch(
    model: PolicyModel,
    dataset: Any,
    indexes: Tensor,
    *,
    device: torch.device,
) -> tuple[Tensor, _PolicyStatistics, float]:
    """Run one decoded batch and return device-independent timing inputs."""

    observations, _, masks, targets, selected = dataset.decoded_batch(
        indexes, device=device
    )
    started = time.perf_counter()
    logits = model(observations)
    if device.type == "mps":
        torch.mps.synchronize()
    elapsed = time.perf_counter() - started
    values = distributional_policy_statistics(logits, masks, targets, selected)
    return masks, values, elapsed


def evaluate_policy(
    model: PolicyModel,
    dataset: Any,
    *,
    device: torch.device,
) -> EvaluationMetrics:
    """Evaluate one immutable dataset split without changing model parameters."""

    model = model.to(device)
    model.eval()
    result = _EvaluationAccumulator()
    with torch.no_grad():
        indexes = torch.arange(dataset.example_count)
        for start in range(0, len(indexes), BATCH_SIZE):
            batch_indexes = indexes[start : start + BATCH_SIZE]
            masks, values, elapsed = _evaluate_batch(
                model,
                dataset,
                batch_indexes,
                device=device,
            )
            result.add(dataset, batch_indexes, masks, values, elapsed)
    return result.finish(dataset.example_count)

"""Compute the distributional loss and evaluation slices for BGC policies.

Training and validation share the same representative-masked probability
calculation. This module owns that mathematics and its metric aggregation; it
does not load datasets, mutate model parameters, select checkpoints, or write
artifacts.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F

from dracula.action_contract import apply_representative_mask
from dracula.bgc_policy_model import BGCPolicyModel
from dracula.bgc_policy_training_contracts import (
    BATCH_SIZE,
    BGCPolicyTrainingError,
    DistributionMetrics,
    EvaluationMetrics,
)
from dracula.bridge import ACTION_COUNT, HAND_SLOT_COUNT, POLICY_POSITION_COUNT


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
        values: tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor],
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


def _validated_flat_mask(
    logits: Tensor,
    representative_masks: Tensor,
    targets: Tensor,
    selected_actions: Tensor,
) -> Tensor:
    """Validate one external dataset/model batch and return its flat legal mask."""

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
        raise BGCPolicyTrainingError("distributional policy batch is malformed")
    masks = representative_masks.flatten(start_dim=1)
    if (
        bool((targets < 0).any().item())
        or not torch.allclose(
            targets.sum(dim=1),
            torch.ones(targets.shape[0], device=targets.device),
            atol=1e-7,
            rtol=0.0,
        )
        or bool((targets.masked_select(~masks) != 0).any().item())
        or not bool(masks.gather(1, selected_actions[:, None]).all().item())
    ):
        raise BGCPolicyTrainingError("visit target or audit action is invalid")
    return masks


def distributional_policy_statistics(
    logits: Tensor,
    representative_masks: Tensor,
    targets: Tensor,
    selected_actions: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Compute per-row loss, entropy, divergence, and selected-action agreement."""

    masks = _validated_flat_mask(
        logits, representative_masks, targets, selected_actions
    )

    # Masking precedes log-softmax, so illegal and non-representative actions
    # receive exactly zero probability and no loss gradient.
    flattened = apply_representative_mask(
        logits, representative_masks
    ).flatten(start_dim=1)
    log_probabilities = F.log_softmax(flattened, dim=1)
    probabilities = log_probabilities.exp()
    safe_log_probabilities = torch.where(
        masks, log_probabilities, torch.zeros_like(log_probabilities)
    )
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
        raise BGCPolicyTrainingError("distributional policy metrics are non-finite")
    return (
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
    values: tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor],
) -> None:
    """Add one batch to each distinct categorical evaluation slice."""

    for label in torch.unique(labels).tolist():
        indexes = torch.nonzero(labels == label, as_tuple=False).flatten()
        accumulators.setdefault(str(int(label)), _MetricAccumulator()).add(
            values, indexes
        )


def evaluate_bgc_policy(
    model: BGCPolicyModel,
    dataset: Any,
    *,
    device: torch.device,
) -> EvaluationMetrics:
    """Evaluate one immutable dataset split without changing model parameters."""

    model = model.to(device)
    model.eval()
    total = _MetricAccumulator()
    placements: dict[str, _MetricAccumulator] = {}
    roles: dict[str, _MetricAccumulator] = {}
    dealer_status: dict[str, _MetricAccumulator] = {}
    representative_actions = 0
    inference_seconds = 0.0
    with torch.no_grad():
        indexes = torch.arange(dataset.example_count)
        for start in range(0, len(indexes), BATCH_SIZE):
            batch_indexes = indexes[start : start + BATCH_SIZE]
            observations, _, masks, targets, selected = dataset.decoded_batch(
                batch_indexes, device=device
            )
            started = time.perf_counter()
            logits = model(observations)
            if device.type == "mps":
                torch.mps.synchronize()
            inference_seconds += time.perf_counter() - started

            values = distributional_policy_statistics(
                logits, masks, targets, selected
            )
            cpu_values = tuple(value.detach().cpu() for value in values)
            total.add(cpu_values)
            representative_actions += int(masks.sum().item())
            placement_labels = dataset.placements.index_select(
                0, batch_indexes
            ).long()
            player_labels = dataset.players.index_select(0, batch_indexes).long()
            dealer_labels = dataset.dealers.index_select(0, batch_indexes).long()
            _metric_groups(placements, placement_labels, cpu_values)
            _metric_groups(roles, player_labels, cpu_values)
            _metric_groups(
                dealer_status,
                player_labels.eq(dealer_labels).long(),
                cpu_values,
            )

    return EvaluationMetrics(
        total.finish(),
        {key: value.finish() for key, value in sorted(placements.items())},
        {
            ("queen" if key == "0" else "king"): value.finish()
            for key, value in sorted(roles.items())
        },
        {
            ("dealer" if key == "1" else "non-dealer"): value.finish()
            for key, value in sorted(dealer_status.items())
        },
        representative_actions / max(dataset.example_count, 1),
        inference_seconds / max(dataset.example_count, 1),
    )


__all__ = (
    "distributional_policy_cross_entropy",
    "distributional_policy_statistics",
    "evaluate_bgc_policy",
)

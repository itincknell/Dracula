"""Run the bounded committed-corpus Sam-policy smoke experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import resource
import subprocess
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import torch
from torch import Tensor

import dracula.sam_policy_training as training
from dracula.randomness import derive_seed
from dracula.sam_miner import resolve_source_identity
from dracula.sam_policy import (
    PARAMETER_COUNT,
    SamPolicyModel,
    load_sam_policy_artifact,
    save_sam_policy_artifact,
)

SMOKE_SCHEMA_VERSION = "dracula-sam-policy-training-smoke-v1"
SMOKE_SHUFFLE_NAMESPACE = "dracula-sam-policy-smoke-shuffle-v1"
SMOKE_EXAMPLES_PER_STRATUM = 256
SMOKE_EPOCHS = 3
BENCHMARK_BATCHES = 40
BENCHMARK_WARMUP_BATCHES = 3


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _progress(path: Path, message: str) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(message.rstrip() + "\n")
        stream.flush()


def _rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys_platform() == "darwin" else value * 1024)


def sys_platform() -> str:
    import sys

    return sys.platform


def _swap_used_bytes() -> int | None:
    try:
        output = subprocess.check_output(
            ["sysctl", "-n", "vm.swapusage"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    match = re.search(r"used = ([0-9.]+)([MG])", output)
    if match is None:
        return None
    scale = 1 << (20 if match.group(2) == "M" else 30)
    return int(float(match.group(1)) * scale)


def _stratified_indexes(
    dataset: training.SamPolicyDataset,
    *,
    root_seed: str,
    snapshot_digest: str,
) -> Tensor:
    available = {
        (
            int(dataset.placements[index].item()),
            int(dataset.players[index].item()),
            int(
                dataset.players[index].item()
                == dataset.dealers[index].item()
            ),
        )
        for index in range(dataset.example_count)
    }
    order = training._epoch_indexes(
        dataset,
        root_seed=root_seed,
        snapshot_digest=snapshot_digest,
        epoch=0,
    )
    selected: list[int] = []
    counts: Counter[tuple[int, int, int]] = Counter()
    for index in order.tolist():
        placement = int(dataset.placements[index].item())
        player = int(dataset.players[index].item())
        dealer_status = int(
            dataset.players[index].item()
            == dataset.dealers[index].item()
        )
        key = (placement, player, dealer_status)
        if counts[key] < SMOKE_EXAMPLES_PER_STRATUM:
            selected.append(index)
            counts[key] += 1
    if (
        set(counts) != available
        or {key[0] for key in available} != set(range(1, 8))
        or {key[1] for key in available} != {0, 1}
        or {key[2] for key in available} != {0, 1}
        or any(counts[key] < 1 for key in available)
    ):
        raise RuntimeError("smoke subset does not span every required stratum")
    return torch.tensor(selected, dtype=torch.long)


def _subset_batches(
    indexes: Tensor,
    *,
    root_seed: str,
    snapshot_digest: str,
    epoch: int,
) -> tuple[Tensor, ...]:
    seed = derive_seed(
        SMOKE_SHUFFLE_NAMESPACE,
        root_seed,
        snapshot_digest,
        str(epoch),
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int.from_bytes(seed[:8], "big"))
    order = torch.randperm(len(indexes), generator=generator)
    shuffled = indexes.index_select(0, order)
    return training._batches(shuffled)


def _subset_loss(
    model: SamPolicyModel,
    dataset: training.SamPolicyDataset,
    indexes: Tensor,
    *,
    device: torch.device,
) -> float:
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for batch in training._batches(indexes):
            observations, masks, targets = dataset.decoded_batch(
                batch, device=device
            )
            losses, _, _, _ = training._masked_losses_and_hits(
                model(observations), masks, targets
            )
            total += float(losses.sum().item())
            count += int(losses.numel())
    return total / count


def _update_batch(
    model: SamPolicyModel,
    optimizer: torch.optim.AdamW,
    dataset: training.SamPolicyDataset,
    indexes: Tensor,
    *,
    device: torch.device,
) -> float:
    model.train()
    observations, masks, targets = dataset.decoded_batch(
        indexes, device=device
    )
    optimizer.zero_grad(set_to_none=True)
    losses, _, _, _ = training._masked_losses_and_hits(
        model(observations), masks, targets
    )
    loss = losses.mean()
    loss.backward()
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    if not gradients or not all(
        bool(torch.isfinite(gradient).all().item())
        for gradient in gradients
    ):
        raise RuntimeError("smoke gradient is absent or non-finite")
    norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), training.GRADIENT_CLIP_NORM
    )
    if not bool(torch.isfinite(norm).item()):
        raise RuntimeError("smoke gradient norm is non-finite")
    optimizer.step()
    if not training._finite_parameters(model):
        raise RuntimeError("smoke optimizer produced non-finite parameters")
    return float(norm.item())


def _new_model(
    device: torch.device, *, model_id: str
) -> tuple[SamPolicyModel, torch.optim.AdamW]:
    model = SamPolicyModel(
        run_root_seed="sam-policy-training-smoke-001-root",
        model_id=model_id,
        initialization_ordinal=0,
    ).to(device)
    return model, training._optimizer(model)


def _component_delta(
    before: dict[str, Tensor],
    after: dict[str, Tensor],
    prefixes: tuple[str, ...],
) -> float:
    return math.sqrt(
        sum(
            float(
                (
                    after[name].detach().cpu() - before[name]
                ).square().sum().item()
            )
            for name in before
            if name.startswith(prefixes)
        )
    )


def _train_control(
    dataset: training.SamPolicyDataset,
    indexes: Tensor,
    *,
    snapshot_digest: str,
) -> tuple[
    SamPolicyModel,
    torch.optim.AdamW,
    list[float],
    float,
    dict[str, float],
    dict[str, Tensor],
]:
    device = torch.device("cpu")
    model, optimizer = _new_model(device, model_id="smoke-resume-proof")
    initial_state = {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }
    losses = [_subset_loss(model, dataset, indexes, device=device)]
    maximum_gradient_norm = 0.0
    for epoch in range(SMOKE_EPOCHS):
        for batch in _subset_batches(
            indexes,
            root_seed="sam-policy-training-smoke-001-root",
            snapshot_digest=snapshot_digest,
            epoch=epoch,
        ):
            maximum_gradient_norm = max(
                maximum_gradient_norm,
                _update_batch(
                    model,
                    optimizer,
                    dataset,
                    batch,
                    device=device,
                ),
            )
        losses.append(
            _subset_loss(model, dataset, indexes, device=device)
        )
    final_state = {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }
    deltas = {
        "embeddings": _component_delta(
            initial_state,
            final_state,
            (
                "encoder.card_embedding",
                "encoder.hand_slot_embedding",
                "encoder.coffin_position_embedding",
            ),
        ),
        "shared_body": _component_delta(
            initial_state,
            final_state,
            (
                "encoder.shared_input",
                "encoder.shared_normalization",
                "encoder.shared_output",
            ),
        ),
        "pair_head": _component_delta(
            initial_state,
            final_state,
            (
                "pair_hidden",
                "pair_normalization",
                "pair_output",
            ),
        ),
    }
    return (
        model,
        optimizer,
        losses,
        maximum_gradient_norm,
        deltas,
        final_state,
    )


def _resume_proof(
    dataset: training.SamPolicyDataset,
    indexes: Tensor,
    *,
    snapshot_digest: str,
    run_directory: Path,
    control_state: dict[str, Tensor],
    control_optimizer: torch.optim.AdamW,
) -> dict[str, object]:
    device = torch.device("cpu")
    model, optimizer = _new_model(device, model_id="smoke-resume-proof")
    checkpoint_path = run_directory / "checkpoints" / "interrupted.pt"
    interrupt_after = 5
    processed = 0
    interrupted_epoch = 0
    interrupted_batch = 0
    for epoch in range(SMOKE_EPOCHS):
        batches = _subset_batches(
            indexes,
            root_seed="sam-policy-training-smoke-001-root",
            snapshot_digest=snapshot_digest,
            epoch=epoch,
        )
        for batch_index, batch in enumerate(batches):
            _update_batch(
                model,
                optimizer,
                dataset,
                batch,
                device=device,
            )
            processed += 1
            if processed == interrupt_after:
                interrupted_epoch = epoch
                interrupted_batch = batch_index + 1
                training._atomic_torch(
                    checkpoint_path,
                    {
                        "format_version": (
                            "dracula-sam-policy-smoke-interruption-v1"
                        ),
                        "snapshot_digest": snapshot_digest,
                        "epoch": interrupted_epoch,
                        "next_batch": interrupted_batch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                    },
                )
                break
        if processed == interrupt_after:
            break

    resumed_model, resumed_optimizer = _new_model(
        device, model_id="smoke-resume-proof"
    )
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=True
    )
    resumed_model.load_state_dict(checkpoint["model_state_dict"])
    resumed_optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    for epoch in range(interrupted_epoch, SMOKE_EPOCHS):
        batches = _subset_batches(
            indexes,
            root_seed="sam-policy-training-smoke-001-root",
            snapshot_digest=snapshot_digest,
            epoch=epoch,
        )
        start = interrupted_batch if epoch == interrupted_epoch else 0
        for batch in batches[start:]:
            _update_batch(
                resumed_model,
                resumed_optimizer,
                dataset,
                batch,
                device=device,
            )
    resumed_state = {
        name: tensor.detach().cpu()
        for name, tensor in resumed_model.state_dict().items()
    }
    model_exact = all(
        torch.equal(control_state[name], resumed_state[name])
        for name in control_state
    )
    optimizer_exact = (
        training._digest_nested(control_optimizer.state_dict())
        == training._digest_nested(resumed_optimizer.state_dict())
    )
    if not model_exact or not optimizer_exact:
        raise RuntimeError("CPU interruption resume is not exact")
    training._atomic_torch(
        run_directory / "checkpoints" / "resumed-final.pt",
        {
            "format_version": "dracula-sam-policy-smoke-final-v1",
            "snapshot_digest": snapshot_digest,
            "model_state_dict": resumed_state,
            "optimizer_state_dict": resumed_optimizer.state_dict(),
        },
    )
    return {
        "interrupt_after_batches": interrupt_after,
        "interrupted_epoch": interrupted_epoch,
        "next_batch": interrupted_batch,
        "model_state_exact": model_exact,
        "optimizer_state_exact": optimizer_exact,
        "checkpoint": str(checkpoint_path),
    }


def _benchmark_batches(indexes: Tensor) -> tuple[Tensor, ...]:
    base = training._batches(indexes)
    needed = BENCHMARK_WARMUP_BATCHES + BENCHMARK_BATCHES
    return tuple(base[index % len(base)] for index in range(needed))


def _benchmark_device(
    device_name: str,
    dataset: training.SamPolicyDataset,
    indexes: Tensor,
) -> tuple[dict[str, object], dict[str, Tensor] | None]:
    if device_name == "mps" and not torch.backends.mps.is_available():
        return (
            {
                "device": "mps",
                "available": False,
                "finite": False,
                "reason": "MPS is unavailable",
            },
            None,
        )
    device = torch.device(device_name)
    model, optimizer = _new_model(
        device, model_id="smoke-device-benchmark"
    )
    batches = _benchmark_batches(indexes)
    swap_before = _swap_used_bytes()
    peak_before = _rss_bytes()
    elapsed = 0.0
    examples = 0
    for index, batch in enumerate(batches):
        if device.type == "mps":
            torch.mps.synchronize()
        started = time.perf_counter()
        _update_batch(
            model, optimizer, dataset, batch, device=device
        )
        if device.type == "mps":
            torch.mps.synchronize()
        if index >= BENCHMARK_WARMUP_BATCHES:
            elapsed += time.perf_counter() - started
            examples += len(batch)
    swap_after = _swap_used_bytes()
    state = {
        name: tensor.detach().cpu()
        for name, tensor in model.state_dict().items()
    }
    return (
        {
            "device": device_name,
            "available": True,
            "finite": training._finite_parameters(model),
            "measured_batches": BENCHMARK_BATCHES,
            "measured_examples": examples,
            "wall_seconds": elapsed,
            "examples_per_second": examples / elapsed,
            "projected_full_training_epoch_seconds": (
                dataset.example_count / (examples / elapsed)
            ),
            "peak_rss_bytes": max(peak_before, _rss_bytes()),
            "swap_growth_bytes": (
                None
                if swap_before is None or swap_after is None
                else swap_after - swap_before
            ),
        },
        state,
    )


def _max_state_difference(
    first: dict[str, Tensor], second: dict[str, Tensor]
) -> float:
    return max(
        float((first[name] - second[name]).abs().max().item())
        for name in first
    )


def _dataset_statistics(
    bundle: training.SamPolicyDatasetBundle,
) -> dict[str, object]:
    combined = (bundle.training, bundle.validation)
    target_counts: Counter[int] = Counter()
    representative_counts: Counter[int] = Counter()
    observation_targets: dict[bytes, int] = {}
    duplicate_rows = 0
    conflicting_rows = 0
    conflicting_observations: set[bytes] = set()
    role_counts: Counter[str] = Counter()
    dealer_counts: Counter[str] = Counter()
    for dataset in combined:
        decoded_masks = training._unpack_tensor_bits(
            dataset.representative_masks_packed, 32
        )
        representative_counts.update(
            int(value) for value in decoded_masks.sum(dim=1).tolist()
        )
        target_counts.update(int(value) for value in dataset.targets.tolist())
        for player, dealer in zip(
            dataset.players.tolist(),
            dataset.dealers.tolist(),
            strict=True,
        ):
            role_counts["Queen" if player == 0 else "King"] += 1
            dealer_counts[
                "dealer" if player == dealer else "non-dealer"
            ] += 1
        for packed, target in zip(
            dataset.observations_packed,
            dataset.targets.tolist(),
            strict=True,
        ):
            key = bytes(packed.tolist())
            existing = observation_targets.get(key)
            if existing is None:
                observation_targets[key] = 1 << int(target)
            else:
                duplicate_rows += 1
                updated = existing | (1 << int(target))
                if updated != existing:
                    conflicting_rows += 1
                    conflicting_observations.add(key)
                observation_targets[key] = updated
    referenced_bytes = 0
    for deck in bundle.snapshot.decks:
        deck_root = (
            bundle.snapshot.corpus_directory
            / deck.relative_path
        ).parent
        referenced_bytes += sum(
            path.stat().st_size
            for path in deck_root.rglob("*")
            if path.is_file()
        )
    return {
        "training_decks": len(bundle.training.fixture_ids),
        "validation_decks": len(bundle.validation.fixture_ids),
        "original_split_decks": dict(
            Counter(deck.original_split for deck in bundle.snapshot.decks)
        ),
        "training_rows": bundle.training.example_count,
        "validation_rows": bundle.validation.example_count,
        "placement_counts": {
            overlay: dict(counts)
            for overlay, counts in bundle.snapshot.placement_counts
        },
        "role_counts": dict(role_counts),
        "dealer_status_counts": dict(dealer_counts),
        "representative_action_counts": dict(
            sorted(representative_counts.items())
        ),
        "target_frequencies": dict(sorted(target_counts.items())),
        "unique_observations": len(observation_targets),
        "duplicate_observation_rows": duplicate_rows,
        "conflicting_duplicate_rows": conflicting_rows,
        "conflicting_observations": len(conflicting_observations),
        "snapshot_manifest_bytes": bundle.snapshot.path.stat().st_size,
        "referenced_committed_deck_bytes": referenced_bytes,
        "privacy_verification": "passed",
        "schema_verification": "passed",
    }


def _group_metrics_table(
    title: str, values: dict[str, training.GroupMetrics]
) -> list[str]:
    lines = [
        f"### {title}",
        "",
        "| Group | Rows | CE | Top-1 | Top-2 | Top-3 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key, metric in values.items():
        lines.append(
            f"| {key} | {metric.count:,} | {metric.cross_entropy:.5f} "
            f"| {metric.top_one:.4f} | {metric.top_two:.4f} "
            f"| {metric.top_three:.4f} |"
        )
    return lines


def _report(
    result: dict[str, object],
) -> str:
    corpus = result["corpus"]
    snapshot = result["snapshot"]
    smoke = result["smoke"]
    benchmark = result["benchmark"]
    validation = smoke["validation"]
    lines = [
        "# Sam policy training smoke 001",
        "",
        "## Frozen corpus",
        "",
        f"- Committed decks: {corpus['deck_count']:,}",
        f"- Committed rows: {corpus['row_count']:,}",
        f"- Terminal leaves: {corpus['terminal_leaf_count']:,}",
        f"- Corpus content digest: `{corpus['content_digest']}`",
        f"- Snapshot digest: `{snapshot['snapshot_digest']}`",
        f"- Training decks/rows: {snapshot['training_decks']:,} / "
        f"{snapshot['training_rows']:,}",
        f"- Validation decks/rows: {snapshot['validation_decks']:,} / "
        f"{snapshot['validation_rows']:,}",
        f"- Original corpus split labels: "
        f"`{snapshot['original_split_decks']}`",
        f"- Snapshot load: {snapshot['load_seconds']:.2f} seconds "
        f"({snapshot['load_rows_per_second']:.0f} rows/s)",
        f"- Referenced deck bytes: "
        f"{snapshot['referenced_committed_deck_bytes']:,}",
        "",
        "### Placement rows",
        "",
        "| Placement | Training | Validation |",
        "| ---: | ---: | ---: |",
    ]
    training_placements = snapshot["placement_counts"]["training"]
    validation_placements = snapshot["placement_counts"]["validation"]
    for placement in range(1, 8):
        lines.append(
            f"| {placement} | {training_placements[placement]:,} "
            f"| {validation_placements[placement]:,} |"
        )
    lines.extend(
        [
            "",
            f"- Queen/King rows: {snapshot['role_counts']['Queen']:,} / "
            f"{snapshot['role_counts']['King']:,}",
            f"- Dealer/non-dealer rows: "
            f"{snapshot['dealer_status_counts']['dealer']:,} / "
            f"{snapshot['dealer_status_counts']['non-dealer']:,}",
            f"- Unique observations: {snapshot['unique_observations']:,}",
            f"- Duplicate observation rows: "
            f"{snapshot['duplicate_observation_rows']:,}",
            f"- Conflicting observations: "
            f"{snapshot['conflicting_observations']:,}",
            f"- Conflicting duplicate rows: "
            f"{snapshot['conflicting_duplicate_rows']:,}",
            f"- Representative-action-count distribution: "
            f"`{snapshot['representative_action_counts']}`",
            f"- Target frequencies by action index: "
            f"`{snapshot['target_frequencies']}`",
            "- Privacy verification: passed",
            "- Schema and digest verification: passed",
            "",
            "## Bounded optimization",
            "",
            f"- Stratified training rows: {smoke['subset_rows']:,}",
            f"- Training cross-entropy: "
            f"`{' → '.join(f'{value:.5f}' for value in smoke['losses'])}`",
            f"- Maximum pre-clip gradient norm: "
            f"{smoke['maximum_gradient_norm']:.5f}",
            f"- Embedding update L2: {smoke['component_deltas']['embeddings']:.6f}",
            f"- Shared-body update L2: "
            f"{smoke['component_deltas']['shared_body']:.6f}",
            f"- Pair-head update L2: "
            f"{smoke['component_deltas']['pair_head']:.6f}",
            f"- Value parameters: {smoke['value_parameter_count']}",
            f"- Exact CPU resume: "
            f"{smoke['resume']['model_state_exact']} model / "
            f"{smoke['resume']['optimizer_state_exact']} optimizer",
            f"- Exported inference exact: "
            f"{smoke['exported_inference_exact']}",
            f"- Validation CE/top-1/top-2/top-3: "
            f"{validation['total']['cross_entropy']:.5f} / "
            f"{validation['total']['top_one']:.4f} / "
            f"{validation['total']['top_two']:.4f} / "
            f"{validation['total']['top_three']:.4f}",
            "",
        ]
    )
    placement_metrics = {
        key: training.GroupMetrics(**value)
        for key, value in validation["placements"].items()
    }
    role_metrics = {
        key: training.GroupMetrics(**value)
        for key, value in validation["roles"].items()
    }
    dealer_metrics = {
        key: training.GroupMetrics(**value)
        for key, value in validation["dealer_status"].items()
    }
    lines.extend(_group_metrics_table("Validation by placement", placement_metrics))
    lines.extend([""])
    lines.extend(_group_metrics_table("Validation by role", role_metrics))
    lines.extend([""])
    lines.extend(_group_metrics_table("Validation by dealer status", dealer_metrics))
    lines.extend(
        [
            "",
            "## Device benchmark",
            "",
            "| Device | Finite | Examples/s | Full epoch | Peak RSS | Swap growth |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for device in ("cpu", "mps"):
        values = benchmark[device]
        if not values["available"]:
            lines.append(f"| {device.upper()} | unavailable | — | — | — | — |")
        else:
            lines.append(
                f"| {device.upper()} | {values['finite']} "
                f"| {values['examples_per_second']:.1f} "
                f"| {values['projected_full_training_epoch_seconds']:.1f}s "
                f"| {values['peak_rss_bytes']:,} "
                f"| {values['swap_growth_bytes']} |"
            )
    lines.extend(
        [
            "",
            f"- CPU/MPS maximum parameter difference after identical updates: "
            f"{benchmark['maximum_parameter_difference']}",
            f"- Selected-device projected epoch including train/validation "
            f"metrics: "
            f"{benchmark['projected_selected_epoch_with_metrics_seconds']:.1f}s",
            f"- Selected full-run device: **{benchmark['selected_device']}**",
            "",
            "The full snapshot optimization was not started.",
            "",
        ]
    )
    return "\n".join(lines)


def run(
    snapshot_path: Path,
    run_directory: Path,
    report_path: Path,
    progress_path: Path,
) -> dict[str, object]:
    source_identity = json.loads(
        (snapshot_path.parent / "source-corpus-identity.json").read_text(
            encoding="utf-8"
        )
    )
    record_digest = source_identity.pop("record_digest")
    if _digest(source_identity) != record_digest:
        raise RuntimeError("snapshot source-identity digest differs")
    corpus_identity = {
        "deck_count": source_identity["deck_count"],
        "row_count": source_identity["row_count"],
        "terminal_leaf_count": source_identity["terminal_leaf_count"],
        "configuration_digest": source_identity[
            "configuration_digest"
        ],
        "teacher_digest": source_identity[
            "teacher_configuration_digest"
        ],
        "teacher_schema_digest": source_identity[
            "teacher_schema_digest"
        ],
        "source_revision": source_identity["source_revision"],
        "source_tree_digest": source_identity["source_tree_digest"],
        "content_digest": source_identity["content_digest"],
        "file_digest": source_identity["file_digest"],
        "record_digest": record_digest,
    }
    started = time.perf_counter()
    bundle = training.load_snapshot_dataset(snapshot_path)
    load_seconds = time.perf_counter() - started
    statistics = _dataset_statistics(bundle)
    statistics.update(
        {
            "snapshot_digest": bundle.snapshot.snapshot_digest,
            "dataset_digest": bundle.dataset_digest,
            "split_digest": bundle.split_digest,
            "load_seconds": load_seconds,
            "load_rows_per_second": (
                (
                    bundle.training.example_count
                    + bundle.validation.example_count
                )
                / load_seconds
            ),
            "peak_rss_bytes": _rss_bytes(),
        }
    )
    _progress(
        progress_path,
        f"phase=inspect status=complete rows="
        f"{bundle.training.example_count + bundle.validation.example_count} "
        f"seconds={load_seconds:.2f}",
    )
    subset = _stratified_indexes(
        bundle.training,
        root_seed="sam-policy-training-smoke-001-root",
        snapshot_digest=bundle.snapshot.snapshot_digest,
    )
    _progress(
        progress_path,
        f"phase=optimization status=running subset_rows={len(subset)}",
    )
    (
        model,
        optimizer,
        losses,
        maximum_gradient_norm,
        component_deltas,
        control_state,
    ) = _train_control(
        bundle.training,
        subset,
        snapshot_digest=bundle.snapshot.snapshot_digest,
    )
    if not losses[-1] < losses[0]:
        raise RuntimeError("bounded smoke did not reduce training loss")
    resume = _resume_proof(
        bundle.training,
        subset,
        snapshot_digest=bundle.snapshot.snapshot_digest,
        run_directory=run_directory,
        control_state=control_state,
        control_optimizer=optimizer,
    )
    validation = training.evaluate_model(
        model, bundle.validation, device=torch.device("cpu")
    )
    source = resolve_source_identity()
    artifact = run_directory / "artifacts" / "policy.pt"
    smoke_configuration = {
        "format_version": SMOKE_SCHEMA_VERSION,
        "snapshot_digest": bundle.snapshot.snapshot_digest,
        "dataset_digest": bundle.dataset_digest,
        "subset_digest": hashlib.sha256(
            subset.numpy().tobytes()
        ).hexdigest(),
        "subset_rows": len(subset),
        "epochs": SMOKE_EPOCHS,
        "optimizer": {
            "name": "AdamW",
            "learning_rate": training.LEARNING_RATE,
            "betas": list(training.BETAS),
            "epsilon": training.EPSILON,
            "weight_decay": training.WEIGHT_DECAY,
            "batch_size": training.BATCH_SIZE,
            "gradient_clip": training.GRADIENT_CLIP_NORM,
        },
    }
    save_sam_policy_artifact(
        artifact,
        model,
        source_revision=source.revision,
        source_tree_digest=source.tree_digest,
        training_configuration=smoke_configuration,
        corpus_snapshot_digest=bundle.snapshot.snapshot_digest,
    )
    loaded = load_sam_policy_artifact(artifact)
    probe_indexes = torch.arange(256, dtype=torch.long)
    probe, _, _ = bundle.validation.decoded_batch(
        probe_indexes, device=torch.device("cpu")
    )
    with torch.no_grad():
        inference_exact = torch.equal(model(probe), loaded.model(probe))
    if not inference_exact:
        raise RuntimeError("exported smoke inference differs")
    value_parameters = [
        name for name, _ in model.named_parameters() if "value" in name
    ]
    if value_parameters:
        raise RuntimeError("standalone smoke model contains value parameters")
    smoke = {
        "subset_rows": len(subset),
        "subset_digest": smoke_configuration["subset_digest"],
        "losses": losses,
        "maximum_gradient_norm": maximum_gradient_norm,
        "component_deltas": component_deltas,
        "parameter_count": sum(
            parameter.numel() for parameter in model.parameters()
        ),
        "value_parameter_count": len(value_parameters),
        "resume": resume,
        "artifact": str(artifact),
        "artifact_state_digest": loaded.metadata.state_dict_digest,
        "exported_inference_exact": inference_exact,
        "validation": asdict(validation),
    }
    if smoke["parameter_count"] != PARAMETER_COUNT:
        raise RuntimeError("smoke model parameter count differs")
    _progress(
        progress_path,
        f"phase=optimization status=complete initial_loss={losses[0]:.6f} "
        f"final_loss={losses[-1]:.6f}",
    )
    _progress(progress_path, "phase=benchmark status=running")
    cpu, cpu_state = _benchmark_device("cpu", bundle.training, subset)
    mps, mps_state = _benchmark_device("mps", bundle.training, subset)
    difference = None
    if cpu_state is not None and mps_state is not None:
        difference = _max_state_difference(cpu_state, mps_state)
    selected = "cpu"
    if (
        mps.get("available")
        and mps.get("finite")
        and float(mps["wall_seconds"]) < float(cpu["wall_seconds"])
    ):
        selected = "mps"
    benchmark = {
        "cpu": cpu,
        "mps": mps,
        "maximum_parameter_difference": difference,
        "selected_device": selected,
        "projected_selected_epoch_with_metrics_seconds": (
            float((mps if selected == "mps" else cpu)[
                "projected_full_training_epoch_seconds"
            ])
            + validation.inference_latency_seconds
            * (
                bundle.training.example_count
                + bundle.validation.example_count
            )
        ),
    }
    result = {
        "format_version": SMOKE_SCHEMA_VERSION,
        "corpus": corpus_identity,
        "snapshot": statistics,
        "smoke": smoke,
        "benchmark": benchmark,
    }
    result["result_digest"] = _digest(result)
    run_directory.mkdir(parents=True, exist_ok=True)
    (run_directory / "results.json").write_bytes(
        _canonical_json(result) + b"\n"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(_report(result), encoding="utf-8")
    _progress(
        progress_path,
        f"phase=benchmark status=complete selected_device={selected}",
    )
    _progress(
        progress_path,
        f"stage=complete result_digest={result['result_digest']}",
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--snapshot",
        default="runs/sam-policy-snapshot-001/corpus-snapshot.json",
    )
    parser.add_argument(
        "--run", default="runs/sam-policy-training-smoke-001"
    )
    parser.add_argument(
        "--report",
        default="reports/active/sam-policy-training-smoke-001.md",
    )
    parser.add_argument(
        "--progress", default="output-sam-policy-training-smoke-001"
    )
    arguments = parser.parse_args()
    result = run(
        Path(arguments.snapshot).resolve(),
        Path(arguments.run).resolve(),
        Path(arguments.report).resolve(),
        Path(arguments.progress).resolve(),
    )
    print(
        json.dumps(
            {
                "snapshot_digest": result["snapshot"]["snapshot_digest"],
                "initial_loss": result["smoke"]["losses"][0],
                "final_loss": result["smoke"]["losses"][-1],
                "selected_device": result["benchmark"]["selected_device"],
                "result_digest": result["result_digest"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

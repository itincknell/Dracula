"""Validate supervised training against the sealed Teacher v2 smoke corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, replace
from pathlib import Path

import torch

from dracula.policy_value import (
    PolicyValueModel,
    load_policy_value_artifact,
)
from dracula.supervised import (
    SupervisedTrainingInterrupted,
    benchmark_optimization_devices,
    epoch_minibatch_indices,
    export_supervised,
    load_config_from_run,
    load_supervised_checkpoint,
    load_supervised_config,
    load_supervised_datasets,
    train_supervised,
    validate_supervised,
)
from dracula.teacher import (
    APPROVED_TEACHER_PROFILE,
    TEACHER_DATASET_SCHEMA_VERSION,
    FixtureSplit,
    load_teacher_manifest,
)

_RETIRED_TERMS = (
    "behavior_log",
    "critic",
    "hidden_state",
    "policy_population",
    "ppo",
    "recurrent",
)


def _nested_equal(left: object, right: object) -> bool:
    if isinstance(left, torch.Tensor):
        return (
            isinstance(right, torch.Tensor)
            and left.dtype is right.dtype
            and left.shape == right.shape
            and torch.equal(left, right)
        )
    if isinstance(left, dict):
        return (
            isinstance(right, dict)
            and left.keys() == right.keys()
            and all(_nested_equal(left[key], right[key]) for key in left)
        )
    if isinstance(left, (list, tuple)):
        return (
            isinstance(right, type(left))
            and len(left) == len(right)
            and all(
                _nested_equal(first, second)
                for first, second in zip(left, right, strict=True)
            )
        )
    return left == right


def _keys(value: object):
    if isinstance(value, dict):
        for key, nested in value.items():
            yield str(key).lower()
            yield from _keys(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            yield from _keys(nested)


def _checkpoint(run_config, name: str, datasets, resolved):
    return load_supervised_checkpoint(
        run_config.output_path / "checkpoints" / name,
        run_config,
        resolved,
        datasets.dataset_digest,
    )


def _order_digest(indexes: tuple[torch.Tensor, ...]) -> str:
    order = torch.cat(indexes).contiguous()
    return hashlib.sha256(order.numpy().tobytes(order="C")).hexdigest()


def _parameter_change(initial, trained, name: str) -> dict[str, float | int]:
    first = initial["model_state_dict"][name]
    second = trained["model_state_dict"][name]
    difference = torch.abs(second - first)
    return {
        "changed_elements": int(torch.count_nonzero(difference).item()),
        "maximum_absolute_change": float(difference.max().item()),
    }


def _validate_epoch_artifacts(
    output: Path,
    dataset_digest: str,
    completed_epochs: int,
) -> dict[str, object]:
    elapsed: list[float] = []
    metrics: list[dict[str, object]] = []
    finite_fields = (
        "training_total_loss",
        "training_policy_cross_entropy",
        "training_value_mse",
        "validation_total_loss",
        "validation_policy_cross_entropy",
        "validation_value_mse",
        "validation_policy_accuracy",
        "validation_value_mae",
        "maximum_gradient_norm",
        "elapsed_seconds",
    )
    for epoch in range(completed_epochs):
        path = output / "metrics" / f"{epoch:06d}.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("dataset_digest") != dataset_digest:
            raise AssertionError("per-epoch metric does not bind the dataset")
        if value.get("epoch") != epoch:
            raise AssertionError("per-epoch metric index differs")
        for field in finite_fields:
            item = value.get(field)
            if (
                type(item) not in (int, float)
                or not math.isfinite(float(item))
            ):
                raise AssertionError(f"non-finite epoch field: {field}")
        if not 0 <= float(value["validation_policy_accuracy"]) <= 1:
            raise AssertionError("policy accuracy is outside [0,1]")
        report = (
            output / "reports" / f"{epoch:06d}.md"
        ).read_text(encoding="utf-8")
        if dataset_digest not in report:
            raise AssertionError("per-epoch report does not bind the dataset")
        elapsed.append(float(value["elapsed_seconds"]))
        metrics.append(value)
    return {
        "epochs": metrics,
        "mean_epoch_seconds": sum(elapsed) / len(elapsed),
        "minimum_epoch_seconds": min(elapsed),
        "maximum_epoch_seconds": max(elapsed),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--resume-output", required=True, type=Path)
    parser.add_argument("--export", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    config = load_supervised_config(args.config)
    datasets = load_supervised_datasets(config.teacher_path)
    expected_digest = (
        "b231051c1ee7474bb197cb692baa7ae0803d736720a25799eb80b7399216a9cc"
    )
    if datasets.dataset_digest != expected_digest:
        raise AssertionError("trainer loaded another dataset")
    if (datasets.training.example_count, datasets.validation.example_count) != (
        336,
        84,
    ):
        raise AssertionError("smoke split sizes differ")
    if set(datasets.training.fixture_ids) & set(
        datasets.validation.fixture_ids
    ):
        raise AssertionError("training and validation fixtures overlap")
    for split in (FixtureSplit.TRAINING, FixtureSplit.VALIDATION):
        manifest = load_teacher_manifest(config.teacher_path, split)["manifest"]
        if (
            manifest["dataset_schema_version"]
            != TEACHER_DATASET_SCHEMA_VERSION
            or manifest["teacher_profile"] != APPROVED_TEACHER_PROFILE
        ):
            raise AssertionError("trainer input is not Teacher v2")

    clean_config = load_config_from_run(config.output_path)
    clean_resolved = json.loads(
        (clean_config.output_path / "resolved-config.json").read_text(
            encoding="utf-8"
        )
    )
    if clean_resolved["dataset"]["dataset_digest"] != expected_digest:
        raise AssertionError("resolved run does not bind the dataset")
    clean_initial = _checkpoint(
        clean_config, "iteration-start.pt", datasets, clean_resolved
    )
    clean_latest = _checkpoint(
        clean_config, "epoch-latest.pt", datasets, clean_resolved
    )
    clean_best = _checkpoint(
        clean_config, "best-validation.pt", datasets, clean_resolved
    )
    clean_final = _checkpoint(
        clean_config, "final.pt", datasets, clean_resolved
    )
    for checkpoint in (
        clean_initial,
        clean_latest,
        clean_best,
        clean_final,
    ):
        if checkpoint["dataset_digest"] != expected_digest:
            raise AssertionError("checkpoint does not bind the dataset")
    if clean_initial["completed_epoch"] != -1:
        raise AssertionError("iteration-start is not the initial state")
    if not _nested_equal(
        clean_best["model_state_dict"], clean_final["model_state_dict"]
    ):
        raise AssertionError("final checkpoint is not best-validation")

    epoch_artifacts = _validate_epoch_artifacts(
        clean_config.output_path,
        expected_digest,
        int(clean_latest["completed_epoch"]) + 1,
    )

    resume_config = replace(
        config,
        run=replace(
            config.run,
            output_directory=str(args.resume_output.resolve()),
        ),
    )
    try:
        train_supervised(
            resume_config,
            should_stop=lambda epoch, batch: epoch == 5 and batch == 0,
        )
    except SupervisedTrainingInterrupted:
        pass
    else:
        raise AssertionError("controlled interruption did not occur")
    interrupted_state = json.loads(
        (resume_config.output_path / "state.json").read_text(encoding="utf-8")
    )
    if (
        interrupted_state.get("phase") != "interrupted"
        or interrupted_state.get("completed_epoch") != 4
    ):
        raise AssertionError("interruption did not preserve the epoch boundary")
    if list(resume_config.output_path.rglob("*.tmp-*")):
        raise AssertionError("interrupted run retained a partial artifact")

    resumed_config = load_config_from_run(resume_config.output_path)
    resumed_result = train_supervised(resumed_config, resume=True)
    resumed_resolved = json.loads(
        (resumed_config.output_path / "resolved-config.json").read_text(
            encoding="utf-8"
        )
    )
    resumed_latest = _checkpoint(
        resumed_config, "epoch-latest.pt", datasets, resumed_resolved
    )
    resumed_best = _checkpoint(
        resumed_config, "best-validation.pt", datasets, resumed_resolved
    )
    resumed_final = _checkpoint(
        resumed_config, "final.pt", datasets, resumed_resolved
    )
    if not _nested_equal(clean_latest, resumed_latest):
        raise AssertionError("resumed latest state differs from clean CPU state")
    if not _nested_equal(clean_best, resumed_best):
        raise AssertionError("resumed best state differs from clean CPU state")
    if not _nested_equal(clean_final, resumed_final):
        raise AssertionError("resumed final state differs from clean CPU state")

    batch_digests = [
        _order_digest(epoch_minibatch_indices(config, datasets, epoch))
        for epoch in range(config.optimization.maximum_epochs)
    ]
    if len(batch_digests) != len(set(batch_digests)):
        raise AssertionError("epoch minibatch orders unexpectedly repeat")

    changes = {
        "shared_body": _parameter_change(
            clean_initial,
            clean_latest,
            "encoder.shared_input.weight",
        ),
        "policy_head": _parameter_change(
            clean_initial,
            clean_latest,
            "policy_pair_output.weight",
        ),
        "value_head": _parameter_change(
            clean_initial,
            clean_latest,
            "value_output.weight",
        ),
    }
    if any(item["changed_elements"] == 0 for item in changes.values()):
        raise AssertionError("one model component did not update")

    export_supervised(
        clean_config,
        args.export,
        clean_config.output_path
        / "checkpoints"
        / "best-validation.pt",
    )
    artifact = load_policy_value_artifact(args.export)
    if artifact.metadata.dataset_digest != expected_digest:
        raise AssertionError("export does not bind the dataset")
    direct = PolicyValueModel(
        run_root_seed=clean_config.run.root_seed,
        model_id=clean_config.model.model_id,
        initialization_ordinal=clean_config.model.initialization_ordinal,
    )
    direct.load_state_dict(clean_best["model_state_dict"], strict=True)
    observations = datasets.validation.observations[:16]
    direct_outputs = direct(observations)
    artifact_outputs = artifact.model(observations)
    if not (
        torch.equal(direct_outputs[0], artifact_outputs[0])
        and torch.equal(direct_outputs[1], artifact_outputs[1])
    ):
        raise AssertionError("export inference differs from best-validation")

    retired_keys = set()
    for value in (
        clean_resolved,
        clean_initial,
        clean_latest,
        clean_best,
        clean_final,
        artifact.training_configuration,
    ):
        retired_keys.update(
            key
            for key in _keys(value)
            if any(term in key for term in _RETIRED_TERMS)
        )
    if retired_keys:
        raise AssertionError(f"retired metadata entered training: {retired_keys}")

    benchmark = benchmark_optimization_devices(config, datasets)
    validation = validate_supervised(clean_config)
    result = {
        "dataset": {
            "digest": datasets.dataset_digest,
            "training_examples": datasets.training.example_count,
            "validation_examples": datasets.validation.example_count,
            "training_fixture_count": len(datasets.training.fixture_ids),
            "validation_fixture_count": len(
                datasets.validation.fixture_ids
            ),
            "fixture_overlap": 0,
            "teacher_profile": APPROVED_TEACHER_PROFILE,
        },
        "clean_run": {
            "completed_epochs": int(clean_latest["completed_epoch"]) + 1,
            "best_epoch": int(clean_best["best_epoch"]),
            "best_validation_loss": float(
                clean_best["best_validation_loss"]
            ),
            "first_epoch": epoch_artifacts["epochs"][0],
            "last_epoch": epoch_artifacts["epochs"][-1],
            "epoch_timing": {
                key: value
                for key, value in epoch_artifacts.items()
                if key != "epochs"
            },
            "validation": validation,
        },
        "parameter_updates": changes,
        "resume": {
            "interrupted_after_completed_epoch": 4,
            "resumed_completed_epochs": resumed_result.completed_epochs,
            "latest_checkpoint_exact_match": True,
            "best_checkpoint_exact_match": True,
            "final_checkpoint_exact_match": True,
            "minibatch_order_digests": batch_digests,
        },
        "artifact_binding": {
            "iteration_start": expected_digest,
            "per_epoch_metrics": expected_digest,
            "epoch_latest": expected_digest,
            "best_validation": expected_digest,
            "final": expected_digest,
            "export": artifact.metadata.dataset_digest,
            "export_state_dict_digest": (
                artifact.metadata.state_dict_digest
            ),
            "export_inference_exact_match": True,
        },
        "retired_metadata_keys": sorted(retired_keys),
        "device_benchmark": asdict(benchmark),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

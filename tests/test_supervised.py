"""Supervised search-distillation training invariants."""

from __future__ import annotations

import hashlib
import inspect
import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

import dracula.supervised as supervised
from dracula.policy_value import (
    PolicyValueModel,
    load_policy_value_artifact,
    policy_value_loss,
)
from dracula.supervised import (
    DatasetConfig,
    ModelConfig,
    OptimizationConfig,
    RunConfig,
    SupervisedConfig,
    SupervisedTrainingError,
    SupervisedTrainingInterrupted,
    epoch_minibatch_indices,
    export_supervised,
    load_config_from_run,
    load_supervised_checkpoint,
    load_supervised_config,
    load_supervised_datasets,
    train_supervised,
)
from dracula.teacher import TeacherCollectionConfig, collect_teacher_dataset


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


@pytest.fixture(scope="module")
def sealed_teacher(tmp_path_factory) -> Path:
    output = tmp_path_factory.mktemp("supervised-teacher")
    collect_teacher_dataset(
        TeacherCollectionConfig(
            run_id="supervised-tests",
            root_seed="supervised-tests-root",
            output_directory=str(output),
            simulation_budget=32,
            training_games=1,
            validation_games=1,
            absolute_fixtures=1,
            workers=2,
        )
    )
    return output


def _config(
    output: Path,
    teacher: Path,
    *,
    device: str = "cpu",
    maximum_epochs: int = 3,
) -> SupervisedConfig:
    return SupervisedConfig(
        RunConfig("supervised-test", "supervised-run-root", str(output), "test-revision"),
        DatasetConfig(str(teacher), _digest("search-report")),
        ModelConfig("supervised-model", 0),
        OptimizationConfig(
            device=device,
            batch_size=16,
            learning_rate=3e-4,
            betas=(0.9, 0.999),
            epsilon=1e-8,
            weight_decay=1e-4,
            gradient_norm=1.0,
            minimum_epochs=maximum_epochs,
            maximum_epochs=maximum_epochs,
            early_stop_patience=2,
            minimum_improvement=1e-4,
        ),
    )


def _checkpoint(config: SupervisedConfig, name: str = "final.pt") -> dict[str, object]:
    datasets = load_supervised_datasets(config.teacher_path)
    resolved = json.loads((config.output_path / "resolved-config.json").read_text())
    return load_supervised_checkpoint(
        config.output_path / "checkpoints" / name,
        config,
        resolved,
        datasets.dataset_digest,
    )


def _assert_nested_equal(left: object, right: object) -> None:
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert isinstance(right, dict) and left.keys() == right.keys()
        for key in left:
            _assert_nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert isinstance(right, type(left)) and len(left) == len(right)
        for first, second in zip(left, right, strict=True):
            _assert_nested_equal(first, second)
    else:
        assert left == right


# Strict parsing prevents a misspelled optimization control from silently using a default.
def test_toml_configuration_is_strict_and_resolves_exact_values(tmp_path, sealed_teacher) -> None:
    path = tmp_path / "training.toml"
    path.write_text(
        f'''[run]
run_id = "toml-test"
root_seed = "toml-root"
output_directory = "{tmp_path / 'run'}"
source_revision = "revision"

[dataset]
teacher_directory = "{sealed_teacher}"
search_report_digest = "{_digest('report')}"

[model]
model_id = "model"
initialization_ordinal = 2

[optimization]
device = "cpu"
batch_size = 32
learning_rate = 0.0003
betas = [0.9, 0.999]
epsilon = 0.00000001
weight_decay = 0.0001
gradient_norm = 1.0
minimum_epochs = 2
maximum_epochs = 4
early_stop_patience = 2
minimum_improvement = 0.0001
''',
        encoding="utf-8",
    )
    config = load_supervised_config(path)
    assert config.optimization.batch_size == 32
    assert config.model.initialization_ordinal == 2
    path.write_text(path.read_text().replace('batch_size = 32', 'batch_size = 32\nunknown = 1'))
    with pytest.raises(SupervisedTrainingError, match="keys must be exactly"):
        load_supervised_config(path)


# The optimizer must be capable of fitting a fixed search target rather than only running.
def test_repeated_fixed_batch_steps_reduce_total_loss(sealed_teacher) -> None:
    datasets = load_supervised_datasets(sealed_teacher)
    model = PolicyValueModel(
        run_root_seed="learnable-root", model_id="learnable", initialization_ordinal=0
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    indexes = torch.arange(8)
    batch = supervised._batch(datasets.training, indexes, torch.device("cpu"))
    with torch.no_grad():
        logits, values = model(batch[0])
        initial = float(policy_value_loss(logits, values, batch[1], batch[2], batch[3]).total)
    for _ in range(20):
        supervised._training_epoch(
            model, optimizer, datasets.training, (indexes,), torch.device("cpu"), 1.0, epoch=0
        )
    with torch.no_grad():
        logits, values = model(batch[0])
        final = float(policy_value_loss(logits, values, batch[1], batch[2], batch[3]).total)
    assert final < initial


# Both supervised objectives must train their heads and the common representation.
def test_policy_value_and_shared_body_receive_gradients(sealed_teacher) -> None:
    datasets = load_supervised_datasets(sealed_teacher)
    model = PolicyValueModel(run_root_seed="gradient-root", model_id="gradient", initialization_ordinal=0)
    batch = supervised._batch(datasets.training, torch.arange(12), torch.device("cpu"))
    logits, values = model(batch[0])
    policy_value_loss(logits, values, batch[1], batch[2], batch[3]).total.backward()
    assert model.encoder.shared_input.weight.grad is not None
    assert torch.count_nonzero(model.encoder.shared_input.weight.grad) > 0
    assert model.policy_pair_output.weight.grad is not None
    assert torch.count_nonzero(model.policy_pair_output.weight.grad) > 0
    assert model.value_output.weight.grad is not None
    assert torch.count_nonzero(model.value_output.weight.grad) > 0


# A CPU run is a bit-reproducible reference, including model and optimizer state.
def test_fixed_seed_reproduces_complete_cpu_training(tmp_path, sealed_teacher) -> None:
    first = _config(tmp_path / "first", sealed_teacher, maximum_epochs=2)
    second = replace(first, run=replace(first.run, output_directory=str(tmp_path / "second")))
    first_result = train_supervised(first)
    second_result = train_supervised(second)
    assert not first_result.stopped_early
    assert not second_result.stopped_early
    assert first_result.best_validation_loss == second_result.best_validation_loss
    _assert_nested_equal(_checkpoint(first), _checkpoint(second))


# Minibatches are a permutation of training rows and cannot draw a validation fixture.
def test_validation_rows_never_enter_training_batches(tmp_path, sealed_teacher) -> None:
    config = _config(tmp_path / "batching", sealed_teacher)
    datasets = load_supervised_datasets(sealed_teacher)
    indexes = torch.cat(epoch_minibatch_indices(config, datasets, 0))
    assert torch.equal(torch.sort(indexes).values, torch.arange(datasets.training.example_count))
    assert set(datasets.training.fixture_ids).isdisjoint(datasets.validation.fixture_ids)


# An epoch boundary contains every state required to reproduce the uninterrupted next epoch.
def test_resume_reproduces_minibatches_model_and_optimizer(tmp_path, sealed_teacher) -> None:
    clean = _config(tmp_path / "clean", sealed_teacher, maximum_epochs=3)
    interrupted = replace(clean, run=replace(clean.run, output_directory=str(tmp_path / "resumed")))
    train_supervised(clean)
    with pytest.raises(SupervisedTrainingInterrupted):
        train_supervised(interrupted, should_stop=lambda epoch, batch: epoch == 1 and batch == 0)
    resumed_config = load_config_from_run(interrupted.output_path)
    train_supervised(resumed_config, resume=True)
    _assert_nested_equal(_checkpoint(clean), _checkpoint(resumed_config))


# Work inside an incomplete epoch must never replace a validated checkpoint.
def test_interrupted_and_nonfinite_epochs_commit_no_partial_checkpoint(
    tmp_path, sealed_teacher, monkeypatch
) -> None:
    interrupted = _config(tmp_path / "partial", sealed_teacher, maximum_epochs=1)
    with pytest.raises(SupervisedTrainingInterrupted):
        train_supervised(interrupted, should_stop=lambda epoch, batch: batch == 1)
    names = {path.name for path in (interrupted.output_path / "checkpoints").iterdir()}
    assert names == {"iteration-start.pt"}
    assert not list(interrupted.output_path.rglob("*.tmp-*"))

    nonfinite = _config(tmp_path / "nonfinite", sealed_teacher, maximum_epochs=1)
    original = supervised.policy_value_loss

    def nan_loss(*args, **kwargs):
        result = original(*args, **kwargs)
        return replace(result, total=result.total * torch.tensor(float("nan")))

    monkeypatch.setattr(supervised, "policy_value_loss", nan_loss)
    with pytest.raises(SupervisedTrainingError, match="non-finite"):
        train_supervised(nonfinite)
    names = {path.name for path in (nonfinite.output_path / "checkpoints").iterdir()}
    assert names == {"iteration-start.pt"}
    assert not list(nonfinite.output_path.rglob("*.tmp-*"))


# MPS is numerically comparable, while CPU remains the exact reference.
@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_cpu_and_mps_forward_and_training_agree_within_tolerance(tmp_path, sealed_teacher) -> None:
    datasets = load_supervised_datasets(sealed_teacher)
    cpu = PolicyValueModel(run_root_seed="device-root", model_id="device", initialization_ordinal=0)
    mps = PolicyValueModel(run_root_seed="device-root", model_id="device", initialization_ordinal=0).to("mps")
    inputs = datasets.validation.observations[:8]
    cpu_outputs = cpu(inputs)
    mps_outputs = mps(inputs.to("mps"))
    assert torch.allclose(cpu_outputs[0], mps_outputs[0].cpu(), atol=1e-5, rtol=1e-4)
    assert torch.allclose(cpu_outputs[1], mps_outputs[1].cpu(), atol=1e-5, rtol=1e-4)

    cpu_config = _config(tmp_path / "cpu", sealed_teacher, device="cpu", maximum_epochs=1)
    mps_config = _config(tmp_path / "mps", sealed_teacher, device="mps", maximum_epochs=1)
    cpu_result = train_supervised(cpu_config)
    mps_result = train_supervised(mps_config)
    assert mps_result.best_validation_loss == pytest.approx(
        cpu_result.best_validation_loss, rel=1e-3, abs=1e-4
    )


# Export is a deployment artifact, but it must preserve direct model inference exactly.
def test_exported_artifact_reproduces_checkpoint_inference(tmp_path, sealed_teacher) -> None:
    config = _config(tmp_path / "export-run", sealed_teacher, maximum_epochs=1)
    train_supervised(config)
    datasets = load_supervised_datasets(sealed_teacher)
    epoch_metrics = json.loads(
        (config.output_path / "metrics" / "000000.json").read_text()
    )
    assert epoch_metrics["dataset_digest"] == datasets.dataset_digest
    assert datasets.dataset_digest in (
        config.output_path / "reports" / "000000.md"
    ).read_text()
    destination = export_supervised(config, tmp_path / "candidate.pt")
    loaded = load_policy_value_artifact(destination)
    checkpoint = _checkpoint(config)
    direct = PolicyValueModel(
        run_root_seed=config.run.root_seed,
        model_id=config.model.model_id,
        initialization_ordinal=config.model.initialization_ordinal,
    )
    direct.load_state_dict(checkpoint["model_state_dict"], strict=True)
    observations = datasets.validation.observations[:5]
    direct_outputs = direct(observations)
    artifact_outputs = loaded.model(observations)
    assert torch.equal(direct_outputs[0], artifact_outputs[0])
    assert torch.equal(direct_outputs[1], artifact_outputs[1])


# The active trainer consumes only sealed targets and the shared model contract.
def test_active_trainer_has_no_retired_training_dependencies(tmp_path, sealed_teacher) -> None:
    config = _config(tmp_path / "metadata", sealed_teacher, maximum_epochs=1)
    train_supervised(config)
    payload = _checkpoint(config)

    def keys(value):
        if isinstance(value, dict):
            for key, nested in value.items():
                yield str(key).lower()
                yield from keys(nested)
        elif isinstance(value, (list, tuple)):
            for nested in value:
                yield from keys(nested)

    retired = ("ppo", "behavior_log", "hidden_state", "critic", "population", "recurrent")
    assert not any(term in key for key in keys(payload) for term in retired)
    source = inspect.getsource(supervised)
    assert "from dracula.training" not in source
    assert "from dracula.optimization" not in source
    assert "from dracula.collection" not in source


# The five commands share the same validated run artifacts rather than separate code paths.
def test_cli_smoke_validate_export_and_resume(tmp_path, sealed_teacher, capsys) -> None:
    config_path = tmp_path / "cli.toml"
    output = tmp_path / "cli-run"
    destination = tmp_path / "cli-export.pt"
    config_path.write_text(
        f'''[run]
run_id = "cli-smoke"
root_seed = "cli-smoke-root"
output_directory = "{output}"
source_revision = "test-revision"

[dataset]
teacher_directory = "{sealed_teacher}"
search_report_digest = "{_digest('cli-report')}"

[model]
model_id = "cli-model"
initialization_ordinal = 0

[optimization]
device = "cpu"
batch_size = 42
learning_rate = 0.0003
betas = [0.9, 0.999]
epsilon = 0.00000001
weight_decay = 0.0001
gradient_norm = 1.0
minimum_epochs = 1
maximum_epochs = 1
early_stop_patience = 1
minimum_improvement = 0.0001
''',
        encoding="utf-8",
    )
    assert supervised.main(["smoke", "--config", str(config_path)]) == 0
    assert supervised.main(["validate", "--output", str(output)]) == 0
    assert (
        supervised.main(
            ["export", "--output", str(output), "--destination", str(destination)]
        )
        == 0
    )
    assert supervised.main(["resume", "--output", str(output)]) == 0
    assert destination.exists()
    assert "validation_loss" in capsys.readouterr().out

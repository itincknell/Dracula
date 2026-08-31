"""Committed-corpus and standalone Sam-policy training invariants."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

import dracula.sam_miner as miner
import dracula.sam_policy_training as training
from dracula.sam_policy import (
    PARAMETER_COUNT,
    SamPolicyContractError,
    load_sam_policy_artifact,
)


@pytest.fixture(scope="module")
def sealed_snapshot(
    tmp_path_factory: pytest.TempPathFactory,
) -> Path:
    root = tmp_path_factory.mktemp("sam-policy-corpus")
    corpus = root / "corpus"
    teacher = miner.DeterministicSamTeacherStub()
    source = miner.resolve_source_identity()
    config = miner.SamMinerConfig(
        run_id="sam-policy-loader-fixture",
        root_seed="sam-policy-loader-fixture-root",
        output_directory=str(corpus),
        training_decks=1,
        validation_decks=1,
        test_decks=1,
        child_count=1,
        workers=1,
        continuous=True,
        minimum_free_disk_bytes=1,
        teacher_schema_version=teacher.schema_version,
        teacher_configuration_digest=teacher.configuration_digest,
        teacher_controller_profile=teacher.controller_profile,
        source_tree_schema_version=source.schema_version,
        source_revision=source.revision,
        source_tree_digest=source.tree_digest,
    )

    def stop_after_three_decks() -> bool:
        return (
            len(tuple(corpus.glob("decks/*/*/deck-manifest.json")))
            >= 3
        )

    result = miner.mine_corpus(
        config,
        teacher,
        should_stop=stop_after_three_decks,
    )
    assert result.deck_count == 3
    snapshot = training.create_committed_corpus_snapshot(
        corpus, root / "snapshot.json"
    )
    return snapshot.path


def _config(
    snapshot: Path,
    output: Path,
    *,
    maximum_epochs: int = 1,
) -> training.SamPolicyTrainingConfig:
    return training.SamPolicyTrainingConfig(
        training.RunSection(
            run_id="sam-policy-training-test",
            root_seed="sam-policy-training-test-root",
            output_directory=str(output),
        ),
        training.DatasetSection(snapshot_path=str(snapshot)),
        training.ModelSection(
            model_id="sam-policy-training-test-model",
            initialization_ordinal=0,
        ),
        training.OptimizationSection(
            device="cpu",
            minimum_epochs=1,
            maximum_epochs=maximum_epochs,
            early_stop_patience=1,
            minimum_improvement=1e-4,
        ),
    )


def _all_keys(value: object) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, dict):
        keys.update(str(key) for key in value)
        for nested in value.values():
            keys.update(_all_keys(nested))
    elif isinstance(value, (list, tuple)):
        for nested in value:
            keys.update(_all_keys(nested))
    return keys


def test_snapshot_seals_complete_decks_and_overlay_isolation(
    sealed_snapshot: Path,
) -> None:
    """Branches from one deck can never cross the experiment overlay."""

    snapshot = training.load_committed_corpus_snapshot(sealed_snapshot)
    bundle = training.load_snapshot_dataset(sealed_snapshot)

    assert [deck.ordinal for deck in snapshot.decks] == [0, 1, 2]
    assert [deck.overlay for deck in snapshot.decks] == [
        "training",
        "validation",
        "validation",
    ]
    assert bundle.training.example_count == 42
    assert bundle.validation.example_count == 84
    assert set(bundle.training.fixture_ids).isdisjoint(
        bundle.validation.fixture_ids
    )
    assert set(bundle.training.deck_ordinals.tolist()).isdisjoint(
        bundle.validation.deck_ordinals.tolist()
    )


def test_rows_become_exact_proxy_targets(
    sealed_snapshot: Path,
) -> None:
    """The loader re-derives proxy legality instead of trusting a label mask."""

    bundle = training.load_snapshot_dataset(sealed_snapshot)
    indexes = torch.arange(
        bundle.training.example_count, dtype=torch.long
    )
    observations, masks, targets = bundle.training.decoded_batch(
        indexes, device=torch.device("cpu")
    )

    assert observations.shape == (42, 875)
    assert observations.dtype is torch.bool
    assert masks.shape == (42, 4, 8)
    assert masks.dtype is torch.bool
    assert targets.shape == (42,)
    assert masks.flatten(start_dim=1).gather(
        1, targets[:, None]
    ).all()
    assert torch.equal(
        bundle.training.placements,
        torch.tensor(list(range(1, 8)) * 6, dtype=torch.uint8),
    )


def test_non_proxy_group_member_cannot_become_a_target(
    sealed_snapshot: Path,
) -> None:
    """A concrete mirror member is not interchangeable with its proxy label."""

    snapshot = training.load_committed_corpus_snapshot(sealed_snapshot)
    config = miner.load_config(snapshot.corpus_directory)
    rows = training._read_rows_for_deck(
        snapshot.corpus_directory, config, snapshot.decks[0]
    )
    paired = next(
        row
        for row in rows
        if len(row.strategic_groups[row.teacher_group_index]) == 2
    )
    non_proxy = next(
        member
        for member in paired.strategic_groups[paired.teacher_group_index]
        if member != paired.teacher_group_representative
    )
    malformed = replace(
        paired, teacher_group_representative=non_proxy
    )

    with pytest.raises(
        training.SamPolicyTrainingError,
        match="designated representative",
    ):
        training._project_row(malformed)


def test_later_corpus_growth_cannot_change_a_sealed_snapshot(
    sealed_snapshot: Path,
) -> None:
    """The moving corpus may grow while the referenced three-deck prefix stays fixed."""

    before = sealed_snapshot.read_bytes()
    snapshot = training.load_committed_corpus_snapshot(sealed_snapshot)
    config = miner.load_config(snapshot.corpus_directory)

    def stop_after_four_decks() -> bool:
        return (
            len(
                tuple(
                    snapshot.corpus_directory.glob(
                        "decks/*/*/deck-manifest.json"
                    )
                )
            )
            >= 4
        )

    miner.mine_corpus(
        config,
        miner.DeterministicSamTeacherStub(),
        resume=True,
        should_stop=stop_after_four_decks,
    )
    bundle = training.load_snapshot_dataset(sealed_snapshot)

    assert sealed_snapshot.read_bytes() == before
    assert miner.inspect_corpus(snapshot.corpus_directory).deck_count == 4
    assert bundle.training.example_count == 42
    assert bundle.validation.example_count == 84


def test_snapshot_rejects_historical_or_tampered_formats(
    sealed_snapshot: Path,
    tmp_path: Path,
) -> None:
    """Only the dedicated committed-corpus snapshot can enter this loader."""

    value = json.loads(sealed_snapshot.read_text(encoding="utf-8"))
    value["format_version"] = "dracula-teacher-manifest-v1"
    incompatible = tmp_path / "historical.json"
    incompatible.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(
        training.SamPolicyTrainingError, match="envelope digest differs"
    ):
        training.load_committed_corpus_snapshot(incompatible)


def test_strict_toml_rejects_unknown_training_fields(
    sealed_snapshot: Path,
    tmp_path: Path,
) -> None:
    """Unreviewed optimizer knobs cannot silently enter a resolved run."""

    path = tmp_path / "invalid.toml"
    path.write_text(
        f"""
format_version = "{training.TRAINING_CONFIG_FORMAT_VERSION}"
[run]
run_id = "invalid"
root_seed = "invalid"
output_directory = "{tmp_path / 'run'}"
[dataset]
snapshot_path = "{sealed_snapshot}"
[model]
model_id = "invalid"
initialization_ordinal = 0
[optimization]
device = "cpu"
minimum_epochs = 1
maximum_epochs = 1
early_stop_patience = 1
minimum_improvement = 0.0001
value_loss_weight = 1.0
""",
        encoding="utf-8",
    )
    with pytest.raises(
        training.SamPolicyTrainingError,
        match="optimization fields are invalid",
    ):
        training.load_training_config(path)


def test_epoch_minibatches_are_deterministic_and_natural(
    sealed_snapshot: Path,
) -> None:
    """Each epoch permutes every natural row exactly once without resampling."""

    bundle = training.load_snapshot_dataset(sealed_snapshot)
    first = training._epoch_indexes(
        bundle.training,
        root_seed="minibatch-root",
        snapshot_digest=bundle.snapshot.snapshot_digest,
        epoch=0,
    )
    repeated = training._epoch_indexes(
        bundle.training,
        root_seed="minibatch-root",
        snapshot_digest=bundle.snapshot.snapshot_digest,
        epoch=0,
    )
    later = training._epoch_indexes(
        bundle.training,
        root_seed="minibatch-root",
        snapshot_digest=bundle.snapshot.snapshot_digest,
        epoch=1,
    )

    assert torch.equal(first, repeated)
    assert sorted(first.tolist()) == list(range(42))
    assert not torch.equal(first, later)


def test_training_resume_reproduces_exact_cpu_state(
    sealed_snapshot: Path,
    tmp_path: Path,
) -> None:
    """A sealed minibatch resume must reach the uninterrupted model state."""

    direct_config = _config(sealed_snapshot, tmp_path / "direct")
    direct = training.train_sam_policy(direct_config)

    resumed_config = _config(sealed_snapshot, tmp_path / "resumed")
    with pytest.raises(training.SamPolicyTrainingInterrupted):
        training.train_sam_policy(
            resumed_config, interrupt_after_batches=1
        )
    resumed = training.train_sam_policy(resumed_config, resume=True)

    direct_artifact = load_sam_policy_artifact(
        direct.exported_artifact
    )
    resumed_artifact = load_sam_policy_artifact(
        resumed.exported_artifact
    )
    assert direct_artifact.metadata.parameter_count == PARAMETER_COUNT
    assert (
        direct_artifact.metadata.state_dict_digest
        == resumed_artifact.metadata.state_dict_digest
    )
    direct_checkpoint = torch.load(
        direct.final_checkpoint, map_location="cpu", weights_only=True
    )
    resumed_checkpoint = torch.load(
        resumed.final_checkpoint, map_location="cpu", weights_only=True
    )
    assert (
        direct_checkpoint["optimizer_state_digest"]
        == resumed_checkpoint["optimizer_state_digest"]
    )
    assert not tuple((tmp_path / "resumed").rglob("*.tmp-*"))


def test_metrics_and_export_are_policy_only(
    sealed_snapshot: Path,
    tmp_path: Path,
) -> None:
    """The active path cannot quietly restore a value objective or output."""

    config = _config(sealed_snapshot, tmp_path / "policy-only")
    result = training.train_sam_policy(config)
    summary = json.loads(
        (config.output_path / "metrics" / "summary.json").read_text(
            encoding="utf-8"
        )
    )
    keys = _all_keys(summary)
    loaded = load_sam_policy_artifact(result.exported_artifact)
    observations = torch.zeros((2, 875), dtype=torch.bool)

    assert not any("value" in key.lower() for key in keys)
    assert loaded.model(observations).shape == (2, 4, 8)
    assert not hasattr(loaded.model, "value_head")
    assert training.validate_run(config.output_path).total.count == 84


def test_atomic_checkpoint_failure_never_replaces_committed_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed temporary write cannot replace the last committed checkpoint."""

    path = tmp_path / "checkpoint.pt"
    training._atomic_torch(path, {"version": 1})
    original = path.read_bytes()

    def fail_save(_value: object, temporary: Path) -> None:
        temporary.write_bytes(b"partial")
        raise RuntimeError("simulated write failure")

    monkeypatch.setattr(torch, "save", fail_save)
    with pytest.raises(RuntimeError, match="simulated"):
        training._atomic_torch(path, {"version": 2})

    assert path.read_bytes() == original
    assert not tuple(tmp_path.glob("*.tmp-*"))


def test_export_rejects_historical_model_format(tmp_path: Path) -> None:
    """A policy/value or recurrent artifact cannot load as this classifier."""

    path = tmp_path / "historical.pt"
    torch.save(
        {
            "format_version": "dracula-policy-value-artifact-v1",
            "state_dict": {},
        },
        path,
    )
    with pytest.raises(SamPolicyContractError, match="payload is invalid"):
        load_sam_policy_artifact(path)

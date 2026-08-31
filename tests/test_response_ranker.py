"""Response-ranker learning, recovery, artifact, and reporting invariants."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch

import dracula.response_ranker as ranker
from dracula.policy_value import PolicyValueModel
from dracula.response_distillation import (
    ResponseActionGroupTarget,
    ResponseCacheIdentity,
    ResponseDistillationExample,
    response_pairwise_ranking_loss,
)
from dracula.teacher import FixtureSplit


@pytest.fixture(scope="module")
def sealed_bundle() -> ranker.ResponseRankingBundle:
    return _synthetic_bundle()


def _model() -> PolicyValueModel:
    model = PolicyValueModel(
        run_root_seed="response-ranker-test",
        model_id="response-ranker-test",
        initialization_ordinal=0,
    )
    ranker.freeze_unused_value_head(model)
    return model


# Ranker tests use a self-contained sealed-data projection so their privacy and
# split invariants do not depend on archived local experiment artifacts.
def test_sealed_dataset_has_exact_split_and_privacy_shape(
    sealed_bundle: ranker.ResponseRankingBundle,
) -> None:
    assert sealed_bundle.dataset_digest == "6" * 64
    assert sealed_bundle.training.example_count == 64
    assert sealed_bundle.validation.example_count == 24
    assert sealed_bundle.training.observations.shape == (64, 875)
    assert sealed_bundle.training.legal_masks.shape == (64, 4, 8)
    assert not set(sealed_bundle.training.fixture_ids).intersection(
        sealed_bundle.validation.fixture_ids
    )
    assert not set(sealed_bundle.training.cache_identity_digests).intersection(
        sealed_bundle.validation.cache_identity_digests
    )
    assert 1 not in sealed_bundle.training.placement_numbers.tolist()
    assert set(sealed_bundle.training.placement_numbers.tolist()) == set(range(2, 8))


def _example_from_row(
    dataset: ranker.ResponseRankingDataset, row: int
) -> ResponseDistillationExample:
    valid = dataset.group_valid[row]
    groups = []
    for index in torch.nonzero(valid).flatten().tolist():
        members = tuple(
            torch.nonzero(dataset.group_member_weights[row, index])
            .flatten()
            .tolist()
        )
        groups.append(
            ResponseActionGroupTarget(
                members[0] // 8,
                int(dataset.group_representatives[row, index]),
                members,
                float(dataset.group_values[row, index]),
            )
        )
    response_digest = "1" * 64
    return ResponseDistillationExample(
        schema_version="dracula-response-ranking-example-v1",
        ranking_schema_version="dracula-pairwise-response-ranking-v1",
        observation_schema_version="dracula-observation-v1",
        action_schema_version="dracula-action-map-v1",
        observation=tuple(bool(value) for value in dataset.observations[row]),
        legal_mask=tuple(
            tuple(bool(value) for value in values)
            for values in dataset.legal_masks[row]
        ),
        groups=tuple(groups),
        selected_representative_action_index=int(
            dataset.selected_representatives[row]
        ),
        selected_concrete_action_index=groups[
            [
                group.representative_action_index for group in groups
            ].index(int(dataset.selected_representatives[row]))
        ].member_action_indices[0],
        placement_number=int(dataset.placement_numbers[row]),
        actor_role=dataset.actor_roles[row],
        actor_is_dealer=bool(dataset.actor_is_dealer[row]),
        completions_per_action=4,
        search_config_digest="2" * 64,
        response_config_digest=response_digest,
        cache_identity=ResponseCacheIdentity("3" * 64, response_digest),
    )


# Dense minibatch padding is an optimization only; its objective must remain
# exactly the pairwise loss locked at the response-evaluation boundary.
def test_dense_batch_loss_matches_the_contract(
    sealed_bundle: ranker.ResponseRankingBundle,
) -> None:
    dataset = sealed_bundle.training
    example = _example_from_row(dataset, 0)
    logits = torch.linspace(-1.0, 1.0, 32, dtype=torch.float32).reshape(4, 8)
    scalar = response_pairwise_ranking_loss(logits, example)
    dense, pairs, weight = ranker._batch_ranking_loss(
        logits.unsqueeze(0),
        dataset.group_member_weights[:1],
        dataset.group_values[:1],
        dataset.group_valid[:1],
    )
    assert torch.allclose(dense, scalar.total, atol=1e-7, rtol=0)
    assert pairs == scalar.contributing_pair_count
    assert weight == pytest.approx(scalar.total_pair_weight, abs=1e-6)


# The response experiment reuses the model body and policy head but cannot
# silently train the unrelated value head.
def test_one_update_is_finite_and_leaves_value_head_exactly_frozen(
    sealed_bundle: ranker.ResponseRankingBundle,
) -> None:
    model = _model()
    optimizer = ranker.build_response_ranker_optimizer(
        model, ranker.PolicyValueOptimizationConfig()
    )
    before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }
    loss, gradient = ranker._training_epoch(
        model,
        optimizer,
        sealed_bundle.training,
        (torch.arange(64),),
        1.0,
        epoch=0,
    )
    assert loss > 0 and torch.isfinite(torch.tensor(loss))
    assert gradient > 0 and torch.isfinite(torch.tensor(gradient))
    assert any(
        not torch.equal(before[name], parameter)
        for name, parameter in model.named_parameters()
        if not name.startswith("value_")
    )
    assert all(
        torch.equal(before[name], parameter)
        and parameter.grad is None
        and not parameter.requires_grad
        for name, parameter in model.named_parameters()
        if name.startswith("value_")
    )


def _synthetic_dataset(
    split: FixtureSplit, count: int, fixture_prefix: str
) -> ranker.ResponseRankingDataset:
    generator = torch.Generator().manual_seed(917 + int(split is FixtureSplit.VALIDATION))
    observations = torch.rand((count, 875), generator=generator) > 0.82
    legal_masks = torch.zeros((count, 4, 8), dtype=torch.bool)
    legal_masks[:, 0, :3] = True
    weights = torch.zeros((count, 3, 32), dtype=torch.float32)
    weights[:, 0, 0] = 1
    weights[:, 1, 1] = 1
    weights[:, 2, 2] = 1
    values = torch.empty((count, 3), dtype=torch.float32)
    for row in range(count):
        values[row] = (
            torch.tensor([0.7, 0.1, -0.4])
            if bool(observations[row, row % 875])
            else torch.tensor([-0.2, 0.8, 0.0])
        )
    selected = values.argmax(dim=1)
    return ranker.ResponseRankingDataset(
        split=split,
        observations=observations,
        legal_masks=legal_masks,
        group_member_weights=weights,
        group_values=values,
        group_valid=torch.ones((count, 3), dtype=torch.bool),
        group_representatives=torch.tensor([[0, 1, 2]] * count),
        selected_representatives=selected,
        placement_numbers=torch.tensor(
            [2 + row % 6 for row in range(count)], dtype=torch.int64
        ),
        actor_roles=tuple("queen" if row % 2 == 0 else "king" for row in range(count)),
        actor_is_dealer=torch.tensor([row % 3 == 0 for row in range(count)]),
        cache_identity_digests=tuple(
            f"{row + (0 if split is FixtureSplit.TRAINING else 1000):064x}"
            for row in range(count)
        ),
        fixture_ids=tuple(f"{fixture_prefix}-{row // 4}" for row in range(count)),
        manifest_digest=("4" if split is FixtureSplit.TRAINING else "5") * 64,
    )


def _synthetic_bundle() -> ranker.ResponseRankingBundle:
    return ranker.ResponseRankingBundle(
        _synthetic_dataset(FixtureSplit.TRAINING, 64, "train"),
        _synthetic_dataset(FixtureSplit.VALIDATION, 24, "validation"),
        "6" * 64,
    )


def _config(output: Path) -> ranker.ResponseRankerConfig:
    return ranker.ResponseRankerConfig(
        ranker.RankerRunConfig(
            "response-ranker-recovery",
            "response-ranker-recovery",
            str(output),
            "test-revision",
        ),
        ranker.RankerDatasetConfig("synthetic"),
        ranker.RankerModelConfig("response-ranker-recovery", 0),
        ranker.RankerOptimizationConfig(
            batch_size=4,
            minimum_epochs=3,
            maximum_epochs=3,
            early_stop_patience=3,
        ),
    )


# A partial epoch is never committed. Resume reloads the last epoch boundary,
# reconstructs the same shuffle, and reaches byte-identical CPU weights.
def test_interrupted_cpu_resume_is_exact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = _synthetic_bundle()
    monkeypatch.setattr(ranker, "load_response_ranking_datasets", lambda _path: bundle)
    complete_config = _config(tmp_path / "complete")
    interrupted_config = _config(tmp_path / "interrupted")
    ranker.train_response_ranker(complete_config)
    with pytest.raises(ranker.ResponseRankerInterrupted):
        ranker.train_response_ranker(
            interrupted_config,
            should_stop=lambda epoch, batch: epoch == 1 and batch == 1,
        )
    ranker.train_response_ranker(interrupted_config, resume=True)
    complete = torch.load(
        complete_config.output_path / "checkpoints" / "final.pt",
        map_location="cpu",
        weights_only=True,
    )
    resumed = torch.load(
        interrupted_config.output_path / "checkpoints" / "final.pt",
        map_location="cpu",
        weights_only=True,
    )
    assert complete["model_state_dict"].keys() == resumed["model_state_dict"].keys()
    assert all(
        torch.equal(complete["model_state_dict"][name], resumed["model_state_dict"][name])
        for name in complete["model_state_dict"]
    )
    for complete_epoch, resumed_epoch in zip(
        complete["history"], resumed["history"], strict=True
    ):
        assert {
            key: value
            for key, value in complete_epoch.items()
            if key != "epoch_seconds"
        } == {
            key: value
            for key, value in resumed_epoch.items()
            if key != "epoch_seconds"
        }


# The deployable file is a distinct response-ranker artifact and must reproduce
# checkpoint inference exactly without carrying training examples or game data.
def test_artifact_is_compatible_private_and_exact(
    tmp_path: Path,
    sealed_bundle: ranker.ResponseRankingBundle,
) -> None:
    config = replace(
        ranker.load_response_ranker_config(
            "configs/archive/teacher-v2/teacher-v2-response-ranker.toml"
        ),
        run=replace(
            ranker.load_response_ranker_config(
                "configs/archive/teacher-v2/teacher-v2-response-ranker.toml"
            ).run,
            output_directory=str(tmp_path / "artifact-run"),
        ),
    )
    resolved = ranker._resolved_document(config, sealed_bundle)
    model = PolicyValueModel(
        run_root_seed=config.run.root_seed,
        model_id=config.model.model_id,
        initialization_ordinal=config.model.initialization_ordinal,
    )
    ranker.freeze_unused_value_head(model)
    path = ranker.save_response_ranker_artifact(
        tmp_path / "ranker.pt",
        model,
        config=config,
        resolved=resolved,
        bundle=sealed_bundle,
        best_epoch=3,
        best_validation_loss=0.5,
    )
    loaded = ranker.load_response_ranker_artifact(path)
    observations = sealed_bundle.validation.observations[:8]
    with torch.no_grad():
        expected, _ = model(observations)
        actual, _ = loaded.model(observations)
    assert torch.equal(expected, actual)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    serialized_keys = repr(payload.keys()) + repr(payload["metadata"].keys())
    assert all(
        private not in serialized_keys
        for private in (
            "authoritative_state",
            "opponent_hand",
            "stock_order",
            "search_tree",
        )
    )
    payload["metadata"]["ranking_schema_version"] = "wrong"
    bad = tmp_path / "bad.pt"
    torch.save(payload, bad)
    with pytest.raises(ranker.ResponseRankerError):
        ranker.load_response_ranker_artifact(bad)


# Placement one is genuinely absent at this observer boundary; reporting it as
# an empty split avoids silently relabeling placement-two evidence.
def test_metrics_report_every_requested_split(
    sealed_bundle: ranker.ResponseRankingBundle,
) -> None:
    metrics = ranker.evaluate_response_ranker(_model(), sealed_bundle.validation)
    assert set(metrics["placements"]) == {str(value) for value in range(1, 8)}
    assert metrics["placements"]["1"]["examples"] == 0
    assert metrics["placements"]["1"]["top_group_accuracy"] is None
    assert metrics["placements"]["7"]["examples"] > 0
    assert set(metrics["roles"]) == {"queen", "king"}
    assert set(metrics["dealer_status"]) == {"dealer", "non-dealer"}

"""BGC-128 visit-target snapshot and policy-distillation invariants."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

import dracula.belief_greedy_miner as miner
import dracula.bgc_policy_training as training
from dracula.bgc_policy import load_bgc_policy_artifact
from dracula.bridge import move_for_action_index
from dracula.engine import (
    EngineStatus,
    advance_after_round,
    apply_move,
    create_game,
    legal_moves,
)
from dracula.search import (
    information_state_from_engine,
    select_concrete_action_index,
    strategic_action_groups,
)
from dracula.randomness import derive_seed
from dracula.sam_policy import COFFIN_START


def _digest(value: object) -> str:
    encoded = json.dumps(value, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, separators=(",", ":"), sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _synthetic_game(config: miner.BalancedMinerConfig, ordinal: int):
    fixture = miner._fixture_id(config, ordinal)
    profile = miner.TRAJECTORY_PROFILE_CYCLE[
        ordinal % len(miner.TRAJECTORY_PROFILE_CYCLE)
    ]
    state = create_game(miner._deck_seed(config, ordinal))
    rows = []
    while state.status is not EngineStatus.GAME_COMPLETE:
        if state.status is EngineStatus.ROUND_COMPLETE:
            state = advance_after_round(state)
            continue
        assert state.active_player is not None
        placement = len(state.current_round_moves) + 1
        if placement == 8:
            moves = legal_moves(state, state.active_player)
            assert len(moves) == 1
            state = apply_move(state, moves[0]).state
            continue
        information = information_state_from_engine(state)
        groups = strategic_action_groups(information, True)
        selected_index = (
            ordinal + state.round_number + placement
        ) % len(groups)
        selected = groups[selected_index].representative_action_index
        visits = [1] * len(groups)
        visits[selected_index] += 128 - len(groups)
        rows.append(
            miner._row(
                information,
                groups,
                tuple(visits),
                selected,
                fixture,
                profile,
                config.search_config.digest,
            )
        )
        concrete = select_concrete_action_index(
            information,
            selected,
            derive_seed(
                "bgc-v2-synthetic-concrete-v1",
                str(ordinal),
                str(state.round_number),
                str(placement),
            ),
            True,
        )
        move = move_for_action_index(information.player, concrete)
        assert move in legal_moves(state, information.player)
        state = apply_move(state, move).state
    assert len(rows) == 42
    content = {
        "fixture_id": fixture,
        "game_schema_version": miner.BELIEF_GREEDY_MINER_GAME_VERSION,
        "ordinal": ordinal,
        "rows": [row.canonical_data() for row in rows],
        "trajectory_profile": profile.value,
    }
    return content, _digest(content)


@pytest.fixture(scope="module")
def synthetic_corpus(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("bgc-v2-corpus")
    corpus = root / "corpus"
    config = miner.BalancedMinerConfig(
        root_seed="bgc-v2-synthetic",
        outer_simulation_budget=128,
        belief_completion_count=8,
        worker_count=1,
    )
    source = miner.resolve_source_identity()
    _write(
        corpus / "resolved-config.json",
        miner._continuous_resolved_document(
            config, minimum_free_disk_bytes=1, source=source
        ),
    )
    entries = []
    for ordinal in range(10):
        content, content_digest = _synthetic_game(config, ordinal)
        relative = f"games/{ordinal:06d}.json"
        _write(
            corpus / relative,
            {"content": content, "content_digest": content_digest},
        )
        entries.append(
            {
                "content_digest": content_digest,
                "fixture_id": content["fixture_id"],
                "ordinal": ordinal,
                "path": relative,
                "row_count": 42,
                "trajectory_profile": content["trajectory_profile"],
            }
        )
    _write(corpus / "corpus-manifest.json", miner._manifest_document(config, entries))
    return corpus


@pytest.fixture(scope="module")
def synthetic_snapshot(
    synthetic_corpus: Path, tmp_path_factory: pytest.TempPathFactory
) -> Path:
    output = tmp_path_factory.mktemp("bgc-v2-snapshot")
    snapshot = training.create_bgc_policy_snapshot(
        synthetic_corpus,
        output,
        specification=training.SnapshotSpecification(10, 1, 1),
    )
    return snapshot.path


@pytest.fixture(scope="module")
def synthetic_card_set_corpus(
    synthetic_snapshot: Path, tmp_path_factory: pytest.TempPathFactory
) -> Path:
    from dracula.bgc_policy_migration import migrate_snapshot

    output = tmp_path_factory.mktemp("bgc-card-set-corpus")
    migrate_snapshot(synthetic_snapshot, output)
    return output


def _config(snapshot: Path, output: Path) -> training.BGCPolicyTrainingConfig:
    return training.BGCPolicyTrainingConfig(
        training.RunSection("bgc-smoke", "bgc-smoke-root", str(output)),
        training.DatasetSection(str(snapshot)),
        training.ModelSection("pi0-smoke", 0),
        training.OptimizationSection("cpu"),
    )


def test_eligibility_and_snapshot_refuse_an_incomplete_d0(
    synthetic_corpus: Path, tmp_path: Path
) -> None:
    status = training.inspect_corpus_eligibility(synthetic_corpus)
    assert status["committed_games"] == 10
    assert status["committed_rows"] == 420
    assert status["complete_five_game_blocks"] == 2
    assert status["digest_valid"] is True
    assert status["eligible"] is False
    assert status["remaining_games"] == 11_895
    assert status["remaining_rows"] == 499_590
    with pytest.raises(training.BGCPolicyTrainingError, match="11,905"):
        training.create_bgc_policy_snapshot(synthetic_corpus, tmp_path / "refused")


def test_snapshot_and_loader_preserve_deck_isolation_and_exact_visits(
    synthetic_snapshot: Path,
) -> None:
    bundle = training.load_bgc_policy_dataset(synthetic_snapshot)
    assert bundle.training.example_count == 210
    assert bundle.validation.example_count == 210
    assert set(bundle.training.fixture_ids).isdisjoint(bundle.validation.fixture_ids)
    assert set(bundle.training.game_ordinals.tolist()).isdisjoint(
        bundle.validation.game_ordinals.tolist()
    )
    indexes = torch.arange(bundle.training.example_count)
    observations, engine_masks, masks, targets, selected = (
        bundle.training.decoded_batch(indexes, device=torch.device("cpu"))
    )
    assert observations.shape == (210, 875)
    assert engine_masks.shape == masks.shape == (210, 4, 8)
    assert targets.shape == (210, 32)
    assert torch.equal(targets.sum(dim=1), torch.ones(210))
    assert torch.count_nonzero(targets.masked_select(~masks.flatten(start_dim=1))) == 0
    assert masks.flatten(start_dim=1).gather(1, selected[:, None]).all()
    assert torch.all(masks <= engine_masks)
    assert Counter(bundle.training.placements.tolist()) == {
        placement: 30 for placement in range(1, 8)
    }
    all_observations = torch.cat(
        (
            bundle.training.decoded_batch(indexes, device=torch.device("cpu"))[0],
            bundle.validation.decoded_batch(indexes, device=torch.device("cpu"))[0],
        )
    )
    occupied_patterns = {
        tuple(
            position + 1
            for position in range(9)
            if observation[
                COFFIN_START + position * 54 : COFFIN_START + (position + 1) * 54
            ].any()
        )
        for observation in all_observations
    }
    assert {
        (5,),
        (2, 5),
        (5, 8),
        (4, 5),
        (5, 6),
        (4, 5, 6),
        (2, 5, 8),
    }.issubset(occupied_patterns)


def test_distributional_loss_masks_probability_and_gradient() -> None:
    logits = torch.randn(2, 4, 8, requires_grad=True)
    masks = torch.zeros(2, 4, 8, dtype=torch.bool)
    masks[:, 0, :3] = True
    targets = torch.zeros(2, 32)
    targets[:, :3] = torch.tensor([0.25, 0.5, 0.25])
    loss = training.distributional_policy_cross_entropy(logits, masks, targets)
    loss.backward()
    probabilities = torch.softmax(
        torch.where(masks, logits.detach(), -torch.inf).flatten(start_dim=1), dim=1
    )
    assert torch.count_nonzero(probabilities.masked_select(~masks.flatten(1))) == 0
    assert torch.count_nonzero(logits.grad.masked_select(~masks)) == 0
    assert "illegal_action_objective" not in training.distributional_policy_cross_entropy.__code__.co_varnames


def test_row_projection_rejects_historical_privacy_and_visit_failures(
    synthetic_corpus: Path,
) -> None:
    game = json.loads((synthetic_corpus / "games/000000.json").read_text())
    row = game["content"]["rows"][0]
    config = miner.BalancedMinerConfig("bgc-v2-synthetic", 128, 8, 1)
    arguments = {
        "expected_fixture": row["fixture_id"],
        "expected_profile": row["trajectory_profile"],
        "expected_search_digest": config.search_config.digest,
    }
    old = dict(row, row_schema_version="dracula-belief-greedy-row-v1")
    with pytest.raises(training.BGCPolicyTrainingError, match="version-two"):
        training._project_raw_row(old, **arguments)
    private = dict(row, opponent_hand=["AC"])
    with pytest.raises(training.BGCPolicyTrainingError):
        training._project_raw_row(private, **arguments)
    malformed = dict(row, strategic_group_visits=list(row["strategic_group_visits"]))
    malformed["strategic_group_visits"][0] += 1
    with pytest.raises(training.BGCPolicyTrainingError, match="sum to 128"):
        training._project_raw_row(malformed, **arguments)


def test_deterministic_minibatches(synthetic_snapshot: Path) -> None:
    bundle = training.load_bgc_policy_dataset(synthetic_snapshot)
    first = training._epoch_indexes(
        bundle.training,
        root_seed="shuffle",
        snapshot_digest=bundle.snapshot.snapshot_digest,
        epoch=3,
    )
    second = training._epoch_indexes(
        bundle.training,
        root_seed="shuffle",
        snapshot_digest=bundle.snapshot.snapshot_digest,
        epoch=3,
    )
    different = training._epoch_indexes(
        bundle.training,
        root_seed="shuffle",
        snapshot_digest=bundle.snapshot.snapshot_digest,
        epoch=4,
    )
    assert torch.equal(first, second)
    assert not torch.equal(first, different)


def test_smoke_training_resume_export_and_atomic_artifacts(
    synthetic_card_set_corpus: Path, tmp_path: Path
) -> None:
    direct = tmp_path / "direct"
    resumed = tmp_path / "resumed"
    direct_result = training.train_bgc_policy(
        _config(synthetic_card_set_corpus, direct), smoke_epochs=2
    )
    with pytest.raises(training.BGCPolicyTrainingInterrupted):
        training.train_bgc_policy(
            _config(synthetic_card_set_corpus, resumed),
            smoke_epochs=2,
            interrupt_after_batches=1,
        )
    resumed_result = training.train_bgc_policy(
        _config(synthetic_card_set_corpus, resumed), resume=True, smoke_epochs=2
    )
    direct_final = torch.load(direct_result.final_checkpoint, weights_only=True)
    resumed_final = torch.load(resumed_result.final_checkpoint, weights_only=True)
    assert direct_final["state_dict_digest"] == resumed_final["state_dict_digest"]
    assert direct_final["optimizer_state_digest"] == resumed_final["optimizer_state_digest"]
    summary = json.loads((direct / "metrics/summary.json").read_text())
    losses = [epoch["training"]["total"]["cross_entropy"] for epoch in summary["history"]]
    assert losses[-1] < losses[0]
    assert summary["best_epoch"] in {1, 2}
    artifact = load_bgc_policy_artifact(direct_result.exported_artifact)
    assert artifact.metadata.candidate_status == "unaccepted"
    assert artifact.metadata.parameter_count == 754_601
    initial = torch.load(direct / "checkpoints/initial.pt", weights_only=True)
    for prefixes in (
        ("card_embedding", "card_encoder", "hand_set_encoder"),
        ("shared_input", "shared_output"),
        ("pair_hidden", "pair_output"),
    ):
        assert any(
            not torch.equal(
                initial["model_state_dict"][name],
                direct_final["model_state_dict"][name],
            )
            for name in initial["model_state_dict"]
            if name.startswith(prefixes)
        )
    assert not list(direct.rglob("*.tmp-*"))


def test_validation_preserves_the_sealed_relative_output_identity(
    synthetic_card_set_corpus: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A relative configured run path must remain valid after canonical lookup."""

    monkeypatch.chdir(tmp_path)
    config = training.BGCPolicyTrainingConfig(
        training.RunSection("relative-smoke", "relative-root", "relative-run"),
        training.DatasetSection(str(synthetic_card_set_corpus)),
        training.ModelSection("relative-pi0", 0),
        training.OptimizationSection("cpu"),
    )
    training.train_bgc_policy(config, smoke_epochs=1)
    metrics = training.validate_bgc_policy_run(tmp_path / "relative-run")
    assert metrics.total.count == 210


def test_training_config_is_strict(synthetic_snapshot: Path, tmp_path: Path) -> None:
    path = tmp_path / "invalid.toml"
    path.write_text(
        f'''format_version = "{training.TRAINING_CONFIG_FORMAT_VERSION}"
[run]
run_id = "bad"
root_seed = "bad"
output_directory = "{tmp_path / 'run'}"
[dataset]
snapshot_path = "{synthetic_snapshot}"
[model]
model_id = "pi0"
initialization_ordinal = 0
[optimization]
device = "cpu"
illegal_action_penalty = 1.0
''',
        encoding="utf-8",
    )
    with pytest.raises(training.BGCPolicyTrainingError, match="optimization fields"):
        training.load_bgc_policy_training_config(path)

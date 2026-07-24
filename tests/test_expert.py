"""Recovery and artifact invariants for single-model expert iteration."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from dracula.expert import (
    EXPERT_STATE_VERSION,
    ExpertIterationError,
    ExpertIterationInterrupted,
    accept_candidate,
    collect_expert_iteration,
    export_accepted,
    load_expert_config,
    load_expert_manifest,
    load_expert_shard,
    reject_candidate,
    replay_epoch_dataset,
    run_expert_iteration,
)
from dracula.policy_value import PolicyValueModel, load_policy_value_artifact, save_policy_value_artifact
from dracula.search import information_state_fingerprint
from dracula.supervised import load_supervised_datasets
from dracula.teacher import (
    FixtureSplit,
    TeacherCollectionConfig,
    collect_teacher_dataset,
)


@pytest.fixture(scope="module")
def expert_foundation(tmp_path_factory):
    root = tmp_path_factory.mktemp("expert-foundation")
    teacher = root / "teacher"
    collect_teacher_dataset(
        TeacherCollectionConfig(
            run_id="expert-test-teacher",
            root_seed="expert-test-teacher-root",
            output_directory=str(teacher),
            simulation_budget=32,
            training_games=1,
            validation_games=1,
            absolute_fixtures=0,
            workers=1,
        )
    )
    artifact = root / "accepted.pt"
    save_policy_value_artifact(
        artifact,
        PolicyValueModel(
            run_root_seed="expert-test-model",
            model_id="expert-test",
            initialization_ordinal=0,
        ),
        source_revision="test",
        training_configuration={"purpose": "expert test"},
        dataset_digest=hashlib.sha256(b"dataset").hexdigest(),
        search_report_digest=hashlib.sha256(b"search").hexdigest(),
    )
    return root, teacher, artifact


def _write_config(path: Path, output: Path, teacher: Path, artifact: Path) -> Path:
    ppo = Path("runs/training-004/archives/policy-2-policy-2-v20.pt").resolve()
    path.write_text(
        f'''[run]
run_id = "expert-test"
root_seed = "expert-test-root"
output_directory = "{output}"
source_revision = "test"
maximum_iterations = 2

[accepted]
artifact_path = "{artifact}"

[teacher]
directory = "{teacher}"
search_report_digest = "{hashlib.sha256(b'search').hexdigest()}"

[collection]
training_games = 1
validation_games = 1
workers = 1
simulation_budget = 20
puct_constant = 1.5
early_placement_count = 4

[replay]
accepted_iterations = 5
examples_per_epoch = 84

[optimization]
device = "cpu"
batch_size = 42
learning_rate = 0.0003
betas = [0.9, 0.999]
epsilon = 1e-8
weight_decay = 0.0001
gradient_norm = 1.0
minimum_epochs = 2
maximum_epochs = 2
early_stop_patience = 1
minimum_improvement = 0.0001

[evaluation]
candidate_budget = 20
random_pairs = 1
ppo_pairs = 1
search_pairs = 1
prior_pairs = 1
search_control_budget = 20
bootstrap_samples = 20
workers = 1
strategic_repetitions = 1

[controls]
ppo_archive = "{ppo}"
''',
        encoding="utf-8",
    )
    return path


# The teacher evidence digest binds an expert run to the validated search contract.
def test_configuration_rejects_noncanonical_search_report_digest(
    tmp_path: Path, expert_foundation
) -> None:
    _, teacher, artifact = expert_foundation
    path = _write_config(tmp_path / "invalid.toml", tmp_path / "run", teacher, artifact)
    path.write_text(
        path.read_text(encoding="utf-8").replace(
            hashlib.sha256(b"search").hexdigest(), "not-a-digest"
        ),
        encoding="utf-8",
    )
    with pytest.raises(ExpertIterationError, match="lowercase SHA-256"):
        load_expert_config(path)


@pytest.fixture
def fast_guided(monkeypatch: pytest.MonkeyPatch):
    def search(planner, information, _seed, should_stop=None):
        if should_stop is not None and should_stop():
            raise ExpertIterationInterrupted("stopped")
        legal = [
            index
            for index, allowed in enumerate(value for row in information.legal_mask for value in row)
            if allowed
        ]
        visits = [0] * 32
        visits[legal[0]] = planner.config.simulation_budget
        return SimpleNamespace(
            information_state_fingerprint=information_state_fingerprint(information),
            action_visits=tuple(visits),
            selected_action_index=legal[0],
            model_evaluation_count=1,
        )

    monkeypatch.setattr("dracula.search.guided.GuidedInformationSetSearch.search", search)


# Guided shards retain only model-visible tensors, exact visit targets, and round returns.
def test_collection_replay_is_deterministic_bounded_and_private(
    tmp_path, expert_foundation, fast_guided
) -> None:
    _, teacher, artifact = expert_foundation
    config = load_expert_config(
        _write_config(tmp_path / "config.toml", tmp_path / "run", teacher, artifact)
    )
    # Preparing a run creates the immutable accepted pointer used by collection replay.
    from dracula.expert import _prepare_run, _replay_sources

    _prepare_run(config)
    result = collect_expert_iteration(config, 0, artifact)
    assert result.training_games == result.validation_games == 1
    assert result.examples == 84

    manifest = load_expert_manifest(
        config.output_path / "iterations/000000/collection", FixtureSplit.TRAINING
    )["manifest"]
    shard = load_expert_shard(
        config.output_path / "iterations/000000/collection" / manifest["shards"][0]["relative_path"]
    )
    columns = shard["columns"]
    assert columns["observations"].shape == (42, 875)
    assert torch.allclose(
        columns["search_policies"],
        columns["search_visits"].float() / columns["search_visits"].sum(dim=1, keepdim=True),
        atol=0,
        rtol=0,
    )
    serialized_keys = json.dumps([*shard["metadata"], *columns])
    for forbidden in ("stock", "game_seed", "opponent_hand", "search_tree", "determinization"):
        assert forbidden not in serialized_keys

    sources, _, _ = _replay_sources(config, 0)
    first = replay_epoch_dataset(config, 0, 3, sources)
    repeated = replay_epoch_dataset(config, 0, 3, sources)
    assert first.example_count == 84
    assert torch.equal(first.observations, repeated.observations)
    assert torch.equal(first.search_policies, repeated.search_policies)


# Each phase leaves a stable resume point; acceptance is an explicit pointer update.
def test_complete_cycle_resumes_each_phase_and_requires_manual_decision(
    tmp_path, expert_foundation, fast_guided, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, teacher, artifact = expert_foundation
    config = load_expert_config(
        _write_config(tmp_path / "config.toml", tmp_path / "run", teacher, artifact)
    )

    def fake_evaluation(config, iteration, *, should_stop=None, **_kwargs):
        if should_stop is not None and should_stop("evaluation", 0):
            raise ExpertIterationInterrupted("evaluation stopped")
        output = config.output_path / "iterations" / f"{iteration:06d}" / "evaluation"
        output.mkdir(parents=True, exist_ok=True)
        document = {"eligible": False, "iteration": iteration}
        (output / "evaluation.json").write_text(json.dumps(document), encoding="utf-8")
        return document

    monkeypatch.setattr("dracula.expert_evaluation.evaluate_expert_candidate", fake_evaluation)

    with pytest.raises(ExpertIterationInterrupted):
        run_expert_iteration(
            config, should_stop=lambda phase, index: phase == "collection" and index == 0
        )
    assert json.loads((config.output_path / "state.json").read_text())["phase"] == "collecting"

    with pytest.raises(ExpertIterationInterrupted):
        run_expert_iteration(
            config,
            resume=True,
            should_stop=lambda phase, index: phase == "optimization" and index == 0,
        )
    assert json.loads((config.output_path / "state.json").read_text())["phase"] == "optimizing"

    with pytest.raises(ExpertIterationInterrupted):
        run_expert_iteration(
            config,
            resume=True,
            should_stop=lambda phase, index: phase == "evaluation" and index == 0,
        )
    assert json.loads((config.output_path / "state.json").read_text())["phase"] == "evaluating"

    result = run_expert_iteration(config, resume=True)
    assert result.phase == "awaiting-decision"
    assert not result.eligible
    with pytest.raises(ExpertIterationError, match="accept or reject"):
        run_expert_iteration(config, resume=True)

    clean_config = load_expert_config(
        _write_config(
            tmp_path / "clean-config.toml",
            tmp_path / "clean-run",
            teacher,
            artifact,
        )
    )
    clean = run_expert_iteration(clean_config)
    resumed_state = torch.load(
        result.training.candidate_checkpoint, map_location="cpu", weights_only=True
    )["model_state_dict"]
    clean_state = torch.load(
        clean.training.candidate_checkpoint, map_location="cpu", weights_only=True
    )["model_state_dict"]
    assert all(
        torch.equal(resumed_state[name], clean_state[name]) for name in resumed_state
    )
    assert not list(config.output_path.rglob("*.tmp-*"))

    accepted = accept_candidate(config)
    assert accepted["accepted_version"] == 1
    assert accepted["accepted_iterations"] == [0]
    exported = export_accepted(config, tmp_path / "accepted-export.pt")
    load_policy_value_artifact(exported)


# Rejection preserves the accepted model and excludes the rejected collection from replay.
def test_rejection_preserves_the_current_pointer(
    tmp_path, expert_foundation, fast_guided, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, teacher, artifact = expert_foundation
    config = load_expert_config(
        _write_config(tmp_path / "config.toml", tmp_path / "run", teacher, artifact)
    )

    def fake_evaluation(config, iteration, **_kwargs):
        output = config.output_path / "iterations" / f"{iteration:06d}" / "evaluation"
        output.mkdir(parents=True, exist_ok=True)
        (output / "evaluation.json").write_text("{}", encoding="utf-8")
        return {"eligible": False}

    monkeypatch.setattr("dracula.expert_evaluation.evaluate_expert_candidate", fake_evaluation)
    run_expert_iteration(config)
    before = json.loads((config.output_path / "accepted.json").read_text())
    after = reject_candidate(config)
    assert after == before
    assert after["accepted_iterations"] == []
    state = json.loads((config.output_path / "state.json").read_text())
    assert state["format_version"] == EXPERT_STATE_VERSION
    assert state["phase"] == "rejected"

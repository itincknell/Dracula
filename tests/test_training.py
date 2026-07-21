"""Configuration, evaluation, checkpoint, and phase-recovery contract tests."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest
import torch

from dracula.collection import PolicyVersion, load_collection_artifact, module_fingerprint
from dracula.evaluation import build_evaluation_schedule
from dracula.models import Policy
from dracula.randomness import derive_pytorch_seed
from dracula.training import (
    MODEL_INITIALIZATION_NAMESPACE,
    SimulatedPhaseInterruption,
    TrainingPhase,
    TrainingSuite,
    TrainingSuiteError,
    _atomic_bytes,
    model_states_close,
)
from dracula.training_config import (
    ResolvedTrainingConfig,
    TrainingConfigurationError,
    canonical_manifest_bytes,
    load_training_config,
)


def _config_file(
    tmp_path: Path,
    run_id: str,
    *,
    iteration_count: int = 1,
    retain_collections: bool = True,
) -> Path:
    path = tmp_path / f"{run_id}.toml"
    path.write_text(
        f"""
[run]
run_id = "{run_id}"
root_seed = "recovery-shared-root"
iteration_count = {iteration_count}
output_directory = "{tmp_path.as_posix()}/runs"
profile = "smoke"
retain_collections = {str(retain_collections).lower()}

[population]
policy_ids = ["policy-0", "policy-1"]

[fixtures]
collection_lane_count = 1
collection_generations = 1
collection_lane_roots = ["recovery-collection-lane"]
held_out_lane_count = 1
held_out_generations = 1
held_out_lane_roots = ["recovery-held-out-lane"]

[compute]
collection_device = "cpu"
optimization_device = "cpu"
trajectory_batch_size = 1

[training]
actor_weight_override = 1.0
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return path


def _full_defaults_file(tmp_path: Path) -> Path:
    path = tmp_path / "full.toml"
    path.write_text(
        f"""
[run]
run_id = "full-defaults"
root_seed = "full-default-root"
output_directory = "{tmp_path.as_posix()}/runs"
profile = "full"
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return path


def _active_fingerprints(suite: TrainingSuite) -> tuple[dict[str, str], str]:
    snapshot = suite._active_population()
    policies = {
        identity.policy_id: module_fingerprint(policy)
        for identity, policy in snapshot.policies.items()
    }
    return policies, module_fingerprint(snapshot.critic)


def _collection_hash(suite: TrainingSuite) -> str:
    reference = suite.state["artifacts"]["collection"]
    artifact = load_collection_artifact(suite.run_directory / reference["path"])
    return artifact.content_hash


def _nested_equal(first, second) -> bool:
    if isinstance(first, torch.Tensor) and isinstance(second, torch.Tensor):
        return torch.equal(first, second)
    if isinstance(first, dict) and isinstance(second, dict):
        return first.keys() == second.keys() and all(
            _nested_equal(first[key], second[key]) for key in first
        )
    if isinstance(first, (list, tuple)) and isinstance(second, type(first)):
        return len(first) == len(second) and all(
            _nested_equal(left, right) for left, right in zip(first, second, strict=True)
        )
    return first == second


# Resolution fixes the full population and fixture defaults in an immutable manifest.
def test_full_defaults_and_smoke_profile_resolve_exact_dimensions(tmp_path) -> None:
    full_path = _full_defaults_file(tmp_path)
    full = load_training_config(full_path)
    assert len(full.population.policy_ids) == 5
    assert len(full.fixtures.collection_lane_roots) == 12
    assert full.fixtures.collection_generations == 4
    assert len(full.fixtures.held_out_lane_roots) == 12
    assert full.fixtures.held_out_generations == 1
    assert full.compute.trajectory_batch_size == 16
    assert full.compute.collection_device == "cpu"
    assert full.compute.optimization_device == "cpu"
    assert full.training.actor_requires_critic_validation is False
    full_path.write_text(
        full_path.read_text(encoding="utf-8")
        + "\n[training]\nactor_weight_override = 0.1\n",
        encoding="utf-8",
    )
    assert load_training_config(full_path).training.actor_weight_override == 0.1
    with pytest.raises(FrozenInstanceError):
        full.run.run_id = "changed"  # type: ignore[misc]

    smoke = load_training_config(_config_file(tmp_path, "smoke-dimensions"))
    assert len(smoke.population.policy_ids) == 2
    assert len(smoke.fixtures.collection_lane_roots) == 1
    assert smoke.fixtures.collection_generations == 1
    assert len(smoke.fixtures.held_out_lane_roots) == 1
    assert smoke.compute.trajectory_batch_size == 1
    assert smoke.training.actor_weight_override == 1.0
    assert ResolvedTrainingConfig.from_manifest(smoke.manifest()) == smoke
    legacy_manifest = smoke.manifest()
    for field in (
        "actor_weight_start",
        "actor_weight_end",
        "actor_requires_critic_validation",
    ):
        del legacy_manifest["training"][field]
    legacy = ResolvedTrainingConfig.from_manifest(legacy_manifest)
    assert legacy.training.actor_weight_start == 0.0
    assert legacy.training.actor_weight_end == 1.0
    assert legacy.training.actor_requires_critic_validation is True
    assert canonical_manifest_bytes(smoke) == canonical_manifest_bytes(smoke)
    suite = TrainingSuite.create(smoke)
    manifest_before = (suite.run_directory / "resolved-config.json").read_bytes()
    with pytest.raises(TrainingSuiteError, match="already contains"):
        TrainingSuite.create(smoke)
    assert (suite.run_directory / "resolved-config.json").read_bytes() == manifest_before


def test_collection_is_cpu_only_and_lane_roots_must_be_disjoint(tmp_path) -> None:
    path = _config_file(tmp_path, "invalid-config")
    text = path.read_text().replace(
        'collection_device = "cpu"', 'collection_device = "mps"'
    )
    path.write_text(text)
    with pytest.raises(TrainingConfigurationError, match="collection device"):
        load_training_config(path)

    path = _config_file(tmp_path, "overlap-config")
    text = path.read_text().replace(
        'held_out_lane_roots = ["recovery-held-out-lane"]',
        'held_out_lane_roots = ["recovery-collection-lane"]',
    )
    path.write_text(text)
    with pytest.raises(TrainingConfigurationError, match="disjoint"):
        load_training_config(path)


# Twelve held-out lanes give every pair balanced roles and 5,040 validation rows.
def test_default_held_out_schedule_has_balanced_roles_and_exact_rows() -> None:
    identities = tuple(
        PolicyVersion(f"policy-{index}", f"policy-{index}-v1")
        for index in range(5)
    )
    schedule = build_evaluation_schedule(
        identities, tuple(f"held-out-{index:02d}" for index in range(12))
    )
    assert len(schedule.fixtures) == 120
    assert len(schedule.fixtures) * 42 == 5_040
    for first_index in range(5):
        for second_index in range(first_index + 1, 5):
            first, second = identities[first_index], identities[second_index]
            fixtures = [
                fixture
                for fixture in schedule.fixtures
                if (fixture.policy_a, fixture.policy_b) == (first, second)
            ]
            assert sum(fixture.queen == first for fixture in fixtures) == 6
            assert sum(fixture.queen == second for fixture in fixtures) == 6


# Held-out critic rows are evaluated after optimization and never enlarge its row pool.
def test_smoke_iteration_keeps_validation_rows_out_of_training_and_limits_reports(
    tmp_path, capsys,
) -> None:
    suite = TrainingSuite.create(
        load_training_config(_config_file(tmp_path, "report-contract"))
    )
    suite.run()
    progress = capsys.readouterr().out
    assert "phase=collection status=running games=1/1" in progress
    assert "phase=optimization status=complete" in progress
    assert "phase=evaluation status=complete" in progress
    assert "run=report-contract status=complete" in progress
    metrics_path = suite.run_directory / suite.state["artifacts"]["metrics"]["path"]
    report_path = suite.run_directory / suite.state["artifacts"]["report"]["path"]
    metrics = json.loads(metrics_path.read_bytes())
    report = report_path.read_text()
    assert metrics["runtime"]["optimization_learned_rows"] == 42
    assert metrics["evaluation"]["critic_validation"]["row_count"] == 42
    assert set(metrics) == {
        "format_version",
        "run_id",
        "iteration_index",
        "contracts",
        "compute",
        "population",
        "training",
        "evaluation",
        "runtime",
        "validation_failures",
    }
    serialized = json.dumps(metrics)
    assert "policy_hidden" not in serialized
    assert "fixture_results" not in serialized
    assert "action_index" not in serialized
    assert "_swap_used_bytes_at_start" not in serialized
    assert metrics["runtime"]["peak_process_memory_bytes"] > 0
    assert metrics["runtime"]["peak_mps_allocation_bytes"] is None
    assert metrics["runtime"]["swap_growth_bytes"] is None or (
        metrics["runtime"]["swap_growth_bytes"] >= 0
    )
    assert "Policy comparison" in report
    assert "Training diagnostics" in report
    assert "Peak process memory" in report
    assert "Swap growth" in report

    evaluation = suite.evaluate_current()
    for policy in evaluation.policy_metrics:
        for split in (policy.overall, policy.queen, policy.king):
            if split.games:
                assert split.victory_percentage == (
                    split.wins + 0.5 * split.ties
                ) / split.games
    queen_returns = evaluation.fixture_results[0].queen_round_returns
    expected_zero_mse = sum(value * value for value in queen_returns) / 6
    assert evaluation.critic_validation.zero_predictor_mse == pytest.approx(
        expected_zero_mse, rel=1e-6, abs=1e-7
    )


@pytest.fixture(scope="module")
def uninterrupted_reference(tmp_path_factory):
    root = tmp_path_factory.mktemp("training-reference")
    suite = TrainingSuite.create(
        load_training_config(_config_file(root, "uninterrupted-reference"))
    )
    suite.run()
    return {
        "collection_hash": _collection_hash(suite),
        "fingerprints": _active_fingerprints(suite),
        "versions": dict(suite.state["policy_versions"]),
        "critic_version": suite.state["critic_version"],
    }


@pytest.mark.parametrize(
    ("interruption", "expected_phase", "preserved_artifact"),
    [
        ("collection", TrainingPhase.ITERATION_START, "iteration_start"),
        ("optimization", TrainingPhase.COLLECTION_SEALED, "collection"),
        ("evaluation", TrainingPhase.POST_UPDATE, "post_update"),
    ],
)
def test_phase_interruption_resumes_from_the_last_complete_boundary(
    tmp_path,
    uninterrupted_reference,
    interruption,
    expected_phase,
    preserved_artifact,
) -> None:
    suite = TrainingSuite.create(
        load_training_config(_config_file(tmp_path, f"resume-{interruption}"))
    )
    with pytest.raises(SimulatedPhaseInterruption):
        suite.run_iteration(interrupt_after_work=interruption)
    assert suite.state["phase"] == expected_phase.value
    preserved = dict(suite.state["artifacts"][preserved_artifact])

    # Unreferenced debris cannot advance state or replace a committed artifact.
    partial = suite.run_directory / ".uncommitted.partial"
    partial.write_bytes(b"partial")
    resumed = TrainingSuite.open(suite.run_directory)
    resumed.resume()
    assert resumed.state["phase"] == TrainingPhase.ITERATION_COMPLETE.value
    assert resumed.state["artifacts"][preserved_artifact] == preserved
    assert dict(resumed.state["policy_versions"]) == uninterrupted_reference["versions"]
    assert resumed.state["critic_version"] == uninterrupted_reference["critic_version"]
    assert _collection_hash(resumed) == uninterrupted_reference["collection_hash"]
    assert _active_fingerprints(resumed) == uninterrupted_reference["fingerprints"]
    assert partial.read_bytes() == b"partial"


# Atomic validation failure leaves an existing committed file and state untouched.
def test_partial_json_never_replaces_committed_state(tmp_path) -> None:
    destination = tmp_path / "state.json"
    destination.write_bytes(b'{"phase":"committed"}')
    with pytest.raises(json.JSONDecodeError):
        _atomic_bytes(destination, b'{"phase":', validate_json=True)
    assert destination.read_bytes() == b'{"phase":"committed"}'
    assert not list(tmp_path.glob(".state.json.*"))


# A successful validation window affects actor weight only in the following iteration.
def test_actor_burn_in_uses_round_count_and_prior_consecutive_windows(
    tmp_path, monkeypatch
) -> None:
    import dracula.training as training_module

    config = load_training_config(
        _config_file(tmp_path, "burn-in-next-iteration", iteration_count=2)
    )
    config = replace(
        config,
        training=replace(
            config.training,
            actor_weight_override=None,
            critic_minimum_collected_rounds=6,
            actor_ramp_collected_rounds=6,
            critic_validation_consecutive_windows=1,
            actor_requires_critic_validation=True,
        ),
    )
    original_evaluate = training_module.evaluate_population

    def passing_evaluation(**kwargs):
        result = original_evaluate(**kwargs)
        return replace(
            result,
            critic_validation=replace(result.critic_validation, passed=True),
        )

    monkeypatch.setattr(training_module, "evaluate_population", passing_evaluation)
    suite = TrainingSuite.create(config)
    suite.run_iteration()
    first_metrics = json.loads(
        (suite.run_directory / suite.state["artifacts"]["metrics"]["path"]).read_bytes()
    )
    assert first_metrics["training"]["actor_weight"] == 0.0
    assert suite.state["critic_validation_streak"] == 1

    suite.run_iteration()
    second_metrics = json.loads(
        (suite.run_directory / suite.state["artifacts"]["metrics"]["path"]).read_bytes()
    )
    assert second_metrics["training"]["actor_weight"] == 1.0


# Replacement uses the next derived initialization seed and leaves the critic untouched.
def test_manual_archive_and_random_replacement_preserve_shared_critic(tmp_path) -> None:
    config = load_training_config(_config_file(tmp_path, "manual-population"))
    suite = TrainingSuite.create(config)
    suite.run()
    before = suite._active_population()
    critic_fingerprint = module_fingerprint(before.critic)
    critic_optimizer_state = before.critic_optimizer_state
    other_identity = next(
        identity for identity in before.policies if identity.policy_id == "policy-1"
    )
    other_fingerprint = module_fingerprint(before.policies[other_identity])

    archive_path = suite.archive_policy("policy-0")
    archive = torch.load(archive_path, map_location="cpu", weights_only=True)
    assert archive["format_version"] == "dracula-policy-archive-v1"
    assert archive["comparison_report"]["path"].endswith("reports/000000.md")
    assert archive["comparison_report"]["markdown"].startswith("# Training iteration")
    assert archive["resolved_manifest"] == config.manifest()
    assert "critic" not in archive

    replacement_identity = suite.replace_policy("policy-0")
    assert replacement_identity.version == "policy-0-v2"
    after = TrainingSuite.open(suite.run_directory)._active_population()
    assert module_fingerprint(after.critic) == critic_fingerprint
    assert _nested_equal(after.critic_optimizer_state, critic_optimizer_state)
    retained_identity = next(
        identity for identity in after.policies if identity.policy_id == "policy-1"
    )
    assert module_fingerprint(after.policies[retained_identity]) == other_fingerprint
    assert replacement_identity not in after.policy_optimizer_states

    expected_seed = derive_pytorch_seed(
        MODEL_INITIALIZATION_NAMESPACE,
        config.run.root_seed,
        "policy",
        "policy-0",
        "1",
    )
    assert module_fingerprint(after.policies[replacement_identity]) == module_fingerprint(
        Policy(seed=expected_seed)
    )


# Cross-device recovery compares neural state numerically, never by raw bytes.
def test_model_state_comparison_uses_manifest_tolerances(tmp_path) -> None:
    config = load_training_config(_config_file(tmp_path, "numerical-tolerances"))
    first = Policy(seed=31)
    close = Policy(seed=31)
    with torch.no_grad():
        next(close.parameters()).add_(config.compute.numerical_atol / 2)
    assert model_states_close(
        first,
        close,
        atol=config.compute.numerical_atol,
        rtol=config.compute.numerical_rtol,
    )
    far = Policy(seed=31)
    with torch.no_grad():
        next(far.parameters()).add_(0.01)
    assert not model_states_close(
        first,
        far,
        atol=config.compute.numerical_atol,
        rtol=config.compute.numerical_rtol,
    )

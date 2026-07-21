"""Phase-checkpointed local self-play training, recovery, and reporting."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import resource
import subprocess
import tempfile
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import torch

from dracula.collection import (
    PolicyVersion,
    build_collection_schedule,
    collect_schedule,
    load_collection_artifact,
    module_fingerprint,
    save_collection_artifact,
    seal_collection,
)
from dracula.evaluation import (
    EvaluationResult,
    build_evaluation_schedule,
    evaluate_population,
)
from dracula.models import Critic, Policy
from dracula.optimization import (
    OptimizationConfig,
    OptimizationResult,
    optimize_collection,
)
from dracula.randomness import derive_pytorch_seed
from dracula.training_config import (
    CHECKPOINT_FORMAT_VERSION,
    ResolvedTrainingConfig,
    TRAINING_FORMAT_VERSION,
    TrainingConfigurationError,
    canonical_manifest_bytes,
    load_training_config,
)

MODEL_INITIALIZATION_NAMESPACE = "dracula-model-initialization-v1"
RUN_STATE_FORMAT_VERSION = "dracula-run-state-v1"
METRICS_FORMAT_VERSION = "dracula-metrics-v1"
ARCHIVE_FORMAT_VERSION = "dracula-policy-archive-v1"


class TrainingPhase(StrEnum):
    READY = "ready"
    ITERATION_START = "iteration_start"
    COLLECTION_SEALED = "collection_sealed"
    POST_UPDATE = "post_update"
    ITERATION_COMPLETE = "iteration_complete"


class TrainingSuiteError(RuntimeError):
    """The local training run or one of its committed artifacts is invalid."""


class SimulatedPhaseInterruption(RuntimeError):
    """Test hook representing process loss after phase work but before commit."""


@dataclass(slots=True)
class PopulationSnapshot:
    policies: dict[PolicyVersion, Policy]
    critic: Critic
    policy_optimizer_states: dict[PolicyVersion, dict[str, object]]
    critic_optimizer_state: dict[str, object] | None
    policy_initialization_ordinals: dict[str, int]
    policy_version_ordinals: dict[str, int]
    critic_version: str
    critic_version_ordinal: int
    collected_training_rounds: int
    critic_validation_streak: int
    next_collection_counter: int
    next_evaluation_counter: int
    optimization_diagnostics: dict[str, object] | None = None


def create_training_run(config_path: str | Path) -> TrainingSuite:
    return TrainingSuite.create(load_training_config(config_path))


class TrainingSuite:
    def __init__(
        self,
        config: ResolvedTrainingConfig,
        state: dict[str, object],
    ) -> None:
        self.config = config
        self.run_directory = config.run_directory
        self.state = state

    @classmethod
    def create(cls, config: ResolvedTrainingConfig) -> TrainingSuite:
        run_directory = config.run_directory
        run_directory.mkdir(parents=True, exist_ok=True)
        manifest_path = run_directory / "resolved-config.json"
        manifest_bytes = canonical_manifest_bytes(config)
        if manifest_path.exists():
            raise TrainingSuiteError("run directory already contains a resolved manifest")
        _atomic_bytes(manifest_path, manifest_bytes, validate_json=True)
        state = {
            "format_version": RUN_STATE_FORMAT_VERSION,
            "run_id": config.run.run_id,
            "manifest_hash": _sha256(manifest_bytes),
            "phase": TrainingPhase.READY.value,
            "iteration_index": -1,
            "artifacts": {},
            "policy_versions": {},
            "critic_version": "critic-v0",
            "collected_training_rounds": 0,
            "critic_validation_streak": 0,
            "next_collection_counter": config.fixtures.next_collection_counter,
            "next_evaluation_counter": config.fixtures.next_evaluation_counter,
            "phase_metrics": {},
        }
        suite = cls(config, state)
        suite._commit_state(state)
        return suite

    @classmethod
    def open(cls, run_directory: str | Path) -> TrainingSuite:
        root = Path(run_directory).resolve()
        try:
            manifest_bytes = (root / "resolved-config.json").read_bytes()
            manifest = json.loads(manifest_bytes)
            state = json.loads((root / "state.json").read_bytes())
        except (OSError, json.JSONDecodeError) as error:
            raise TrainingSuiteError("run manifest or state could not be loaded") from error
        config = ResolvedTrainingConfig.from_manifest(manifest)
        if config.run_directory != root:
            raise TrainingSuiteError("resolved run directory does not match its location")
        _validate_run_state(state, config, _sha256(manifest_bytes))
        return cls(config, state)

    def run(self) -> dict[str, object]:
        while self._completed_iteration_count() < self.config.run.iteration_count:
            self.run_iteration()
        return self.state

    def resume(self) -> dict[str, object]:
        return self.run()

    def run_iteration(
        self, *, interrupt_after_work: str | None = None
    ) -> dict[str, object]:
        if interrupt_after_work not in {None, "collection", "optimization", "evaluation"}:
            raise ValueError("unknown interruption phase")
        phase = TrainingPhase(self.state["phase"])
        if phase in {TrainingPhase.READY, TrainingPhase.ITERATION_COMPLETE}:
            self._begin_iteration()
            phase = TrainingPhase.ITERATION_START
        if phase is TrainingPhase.ITERATION_START:
            self._collect(interrupt_after_work == "collection")
            phase = TrainingPhase.COLLECTION_SEALED
        if phase is TrainingPhase.COLLECTION_SEALED:
            self._optimize(interrupt_after_work == "optimization")
            phase = TrainingPhase.POST_UPDATE
        if phase is TrainingPhase.POST_UPDATE:
            self._evaluate_and_commit(interrupt_after_work == "evaluation")
        return self.state

    def evaluate_current(self) -> EvaluationResult:
        snapshot = self._active_population()
        identities = tuple(sorted(snapshot.policies))
        schedule = build_evaluation_schedule(
            identities,
            self.config.fixtures.held_out_lane_roots,
            generation_count=self.config.fixtures.held_out_generations,
            start_game_counter=int(self.state["next_evaluation_counter"]),
        )
        return evaluate_population(
            run_root_seed=self.config.run.root_seed,
            schedule=schedule,
            policies=snapshot.policies,
            critic=snapshot.critic,
            required_improvement=self.config.training.critic_validation_improvement,
        )

    def archive_policy(
        self, policy_id: str, destination: str | Path | None = None
    ) -> Path:
        snapshot = self._active_population()
        identity = _identity_for_id(snapshot.policies, policy_id)
        report_reference = _artifact(self.state, "report")
        comparison_report = None
        if report_reference is not None:
            report_path = self.run_directory / report_reference["path"]
            _validate_reference(report_path, report_reference)
            comparison_report = {
                **report_reference,
                "markdown": report_path.read_text(encoding="utf-8"),
            }
        payload = {
            "format_version": ARCHIVE_FORMAT_VERSION,
            "run_id": self.config.run.run_id,
            "policy": {
                "policy_id": identity.policy_id,
                "version": identity.version,
                "parameter_fingerprint": module_fingerprint(snapshot.policies[identity]),
                "state_dict": _cpu_state_dict(snapshot.policies[identity]),
            },
            "contracts": self.config.manifest()["versions"],
            "resolved_manifest": self.config.manifest(),
            "resolved_manifest_hash": self.state["manifest_hash"],
            "comparison_report": comparison_report,
        }
        archive_path = (
            Path(destination)
            if destination is not None
            else self.run_directory / "archives" / f"{policy_id}-{identity.version}.pt"
        )
        _atomic_torch_payload(archive_path, payload)
        return archive_path

    def replace_policy(self, policy_id: str) -> PolicyVersion:
        if TrainingPhase(self.state["phase"]) not in {
            TrainingPhase.READY,
            TrainingPhase.ITERATION_COMPLETE,
        }:
            raise TrainingSuiteError("policy replacement is allowed only between iterations")
        snapshot = self._active_population()
        old_identity = _identity_for_id(snapshot.policies, policy_id)
        initialization_ordinal = snapshot.policy_initialization_ordinals[policy_id] + 1
        version_ordinal = snapshot.policy_version_ordinals[policy_id] + 1
        new_identity = PolicyVersion(policy_id, f"{policy_id}-v{version_ordinal}")
        seed = derive_pytorch_seed(
            MODEL_INITIALIZATION_NAMESPACE,
            self.config.run.root_seed,
            "policy",
            policy_id,
            str(initialization_ordinal),
        )
        new_policy = Policy(seed=seed).cpu().eval()
        policies = dict(snapshot.policies)
        del policies[old_identity]
        policies[new_identity] = new_policy
        optimizer_states = dict(snapshot.policy_optimizer_states)
        optimizer_states.pop(old_identity, None)
        snapshot.policies = dict(sorted(policies.items()))
        snapshot.policy_optimizer_states = optimizer_states
        snapshot.policy_initialization_ordinals[policy_id] = initialization_ordinal
        snapshot.policy_version_ordinals[policy_id] = version_ordinal
        replacement_path = (
            self.run_directory
            / "checkpoints"
            / "replacements"
            / f"{policy_id}-v{version_ordinal}.pt"
        )
        reference = self._save_checkpoint(snapshot, replacement_path)
        state = copy.deepcopy(self.state)
        artifacts = dict(state.get("artifacts", {}))
        artifacts["latest_checkpoint"] = reference
        state["artifacts"] = artifacts
        state["policy_versions"] = _policy_versions(snapshot)
        self._commit_state(state)
        return new_identity

    def _begin_iteration(self) -> None:
        snapshot = self._active_population()
        iteration_index = int(self.state["iteration_index"]) + 1
        if iteration_index >= self.config.run.iteration_count:
            raise TrainingSuiteError("configured iteration count is already complete")
        checkpoint_path = (
            self.run_directory
            / "checkpoints"
            / f"{iteration_index:06d}"
            / "iteration-start.pt"
        )
        reference = self._save_checkpoint(snapshot, checkpoint_path)
        state = copy.deepcopy(self.state)
        state.update(
            {
                "phase": TrainingPhase.ITERATION_START.value,
                "iteration_index": iteration_index,
                "policy_versions": _policy_versions(snapshot),
                "critic_version": snapshot.critic_version,
                "collected_training_rounds": snapshot.collected_training_rounds,
                "critic_validation_streak": snapshot.critic_validation_streak,
                "next_collection_counter": snapshot.next_collection_counter,
                "next_evaluation_counter": snapshot.next_evaluation_counter,
                "phase_metrics": {
                    "_swap_used_bytes_at_start": _swap_used_bytes(),
                },
                "artifacts": {
                    "iteration_start": reference,
                    "latest_checkpoint": reference,
                },
            }
        )
        self._commit_state(state)

    def _collect(self, interrupt: bool) -> None:
        snapshot = self._load_referenced_checkpoint("iteration_start")
        identities = tuple(sorted(snapshot.policies))
        schedule = build_collection_schedule(
            identities,
            self.config.fixtures.collection_lane_roots,
            generation_count=self.config.fixtures.collection_generations,
            start_game_counter=snapshot.next_collection_counter,
        )
        started = time.perf_counter()
        collection = collect_schedule(
            run_root_seed=self.config.run.root_seed,
            schedule=schedule,
            policies=snapshot.policies,
            critic=snapshot.critic,
            critic_version=snapshot.critic_version,
        )
        duration = time.perf_counter() - started
        if interrupt:
            raise SimulatedPhaseInterruption("collection interrupted before sealing")
        artifact = seal_collection(collection)
        collection_path = (
            self.run_directory
            / "collections"
            / f"{int(self.state['iteration_index']):06d}.pt"
        )
        save_collection_artifact(artifact, collection_path)
        reference = _reference(self.run_directory, collection_path)
        state = copy.deepcopy(self.state)
        artifacts = dict(state["artifacts"])
        artifacts["collection"] = reference
        state["artifacts"] = artifacts
        state["phase"] = TrainingPhase.COLLECTION_SEALED.value
        phase_metrics = dict(state["phase_metrics"])
        phase_metrics.update(
            {
                "collection_seconds": duration,
                "collection_games": schedule.counts.engine_games,
                "collection_games_per_second": schedule.counts.engine_games
                / max(duration, 1e-12),
            }
        )
        state["phase_metrics"] = phase_metrics
        self._commit_state(state)

    def _optimize(self, interrupt: bool) -> None:
        snapshot = self._load_referenced_checkpoint("iteration_start")
        collection_reference = _required_artifact(self.state, "collection")
        collection_path = self.run_directory / collection_reference["path"]
        _validate_reference(collection_path, collection_reference)
        collection = load_collection_artifact(collection_path).collection
        collected_rounds = (
            snapshot.collected_training_rounds
            + collection.schedule.counts.engine_games * 6
        )
        started = time.perf_counter()
        result = optimize_collection(
            collection=collection,
            policies=snapshot.policies,
            critic=snapshot.critic,
            iteration_index=int(self.state["iteration_index"]),
            collected_rounds=collected_rounds,
            critic_validation_windows=snapshot.critic_validation_streak,
            config=_optimization_config(self.config),
            policy_optimizer_states=snapshot.policy_optimizer_states,
            critic_optimizer_state=snapshot.critic_optimizer_state,
        )
        duration = time.perf_counter() - started
        if interrupt:
            raise SimulatedPhaseInterruption("optimization interrupted before checkpoint")
        updated = _updated_snapshot(
            snapshot,
            result,
            collected_rounds=collected_rounds,
            next_collection_counter=(
                snapshot.next_collection_counter
                + self.config.fixtures.collection_generations
            ),
        )
        checkpoint_path = (
            self.run_directory
            / "checkpoints"
            / f"{int(self.state['iteration_index']):06d}"
            / "post-update.pt"
        )
        reference = self._save_checkpoint(updated, checkpoint_path)
        state = copy.deepcopy(self.state)
        artifacts = dict(state["artifacts"])
        artifacts["post_update"] = reference
        artifacts["latest_checkpoint"] = reference
        state.update(
            {
                "phase": TrainingPhase.POST_UPDATE.value,
                "artifacts": artifacts,
                "policy_versions": _policy_versions(updated),
                "critic_version": updated.critic_version,
                "collected_training_rounds": updated.collected_training_rounds,
                "next_collection_counter": updated.next_collection_counter,
            }
        )
        phase_metrics = dict(state["phase_metrics"])
        phase_metrics.update(
            {
                "optimization_seconds": duration,
                "optimization_learned_rows": collection.schedule.counts.learned_rows,
                "optimization_rows_per_second": (
                    collection.schedule.counts.learned_rows / max(duration, 1e-12)
                ),
            }
        )
        state["phase_metrics"] = phase_metrics
        self._commit_state(state)

    def _evaluate_and_commit(self, interrupt: bool) -> None:
        snapshot = self._load_referenced_checkpoint("post_update")
        schedule = build_evaluation_schedule(
            tuple(sorted(snapshot.policies)),
            self.config.fixtures.held_out_lane_roots,
            generation_count=self.config.fixtures.held_out_generations,
            start_game_counter=snapshot.next_evaluation_counter,
        )
        started = time.perf_counter()
        evaluation = evaluate_population(
            run_root_seed=self.config.run.root_seed,
            schedule=schedule,
            policies=snapshot.policies,
            critic=snapshot.critic,
            required_improvement=self.config.training.critic_validation_improvement,
        )
        duration = time.perf_counter() - started
        if interrupt:
            raise SimulatedPhaseInterruption("evaluation interrupted before report commit")
        streak = (
            snapshot.critic_validation_streak + 1
            if evaluation.critic_validation.passed
            else 0
        )
        next_evaluation_counter = (
            snapshot.next_evaluation_counter
            + self.config.fixtures.held_out_generations
        )
        phase_metrics = dict(self.state["phase_metrics"])
        phase_metrics.update(
            {
                "evaluation_seconds": duration,
                "evaluation_games": len(schedule.fixtures),
                "evaluation_games_per_second": len(schedule.fixtures) / max(duration, 1e-12),
            }
        )
        metrics = _iteration_metrics(
            config=self.config,
            iteration_index=int(self.state["iteration_index"]),
            snapshot=snapshot,
            evaluation=evaluation,
            phase_metrics=phase_metrics,
            critic_validation_streak=streak,
        )
        metrics_path = (
            self.run_directory / "metrics" / f"{int(self.state['iteration_index']):06d}.json"
        )
        report_path = (
            self.run_directory / "reports" / f"{int(self.state['iteration_index']):06d}.md"
        )
        _atomic_bytes(metrics_path, _canonical_json(metrics), validate_json=True)
        _atomic_bytes(report_path, _markdown_report(metrics).encode("utf-8"))
        metrics_reference = _reference(self.run_directory, metrics_path)
        report_reference = _reference(self.run_directory, report_path)
        state = copy.deepcopy(self.state)
        artifacts = dict(state["artifacts"])
        artifacts["metrics"] = metrics_reference
        artifacts["report"] = report_reference
        if not self.config.run.retain_collections:
            collection_reference = artifacts.pop("collection", None)
            if collection_reference is not None:
                (self.run_directory / collection_reference["path"]).unlink(missing_ok=True)
        state.update(
            {
                "phase": TrainingPhase.ITERATION_COMPLETE.value,
                "artifacts": artifacts,
                "critic_validation_streak": streak,
                "next_evaluation_counter": next_evaluation_counter,
                "phase_metrics": phase_metrics,
            }
        )
        self._commit_state(state)

    def _active_population(self) -> PopulationSnapshot:
        phase = TrainingPhase(self.state["phase"])
        if phase is TrainingPhase.READY:
            return _initial_population(self.config)
        key = (
            "iteration_start"
            if phase in {TrainingPhase.ITERATION_START, TrainingPhase.COLLECTION_SEALED}
            else "latest_checkpoint"
        )
        snapshot = self._load_referenced_checkpoint(key)
        if phase is TrainingPhase.ITERATION_COMPLETE:
            snapshot.collected_training_rounds = int(self.state["collected_training_rounds"])
            snapshot.critic_validation_streak = int(self.state["critic_validation_streak"])
            snapshot.next_collection_counter = int(self.state["next_collection_counter"])
            snapshot.next_evaluation_counter = int(self.state["next_evaluation_counter"])
        return snapshot

    def _load_referenced_checkpoint(self, key: str) -> PopulationSnapshot:
        reference = _required_artifact(self.state, key)
        path = self.run_directory / reference["path"]
        _validate_reference(path, reference)
        return _load_checkpoint(path, self.config, str(self.state["manifest_hash"]))

    def _save_checkpoint(
        self, snapshot: PopulationSnapshot, path: Path
    ) -> dict[str, str]:
        payload = _checkpoint_payload(
            snapshot,
            self.config,
            str(self.state["manifest_hash"]),
        )
        _atomic_torch_payload(path, payload)
        return _reference(self.run_directory, path)

    def _commit_state(self, state: dict[str, object]) -> None:
        try:
            manifest_hash = _sha256(
                (self.run_directory / "resolved-config.json").read_bytes()
            )
        except OSError as error:
            raise TrainingSuiteError("resolved manifest is missing") from error
        _validate_run_state(state, self.config, manifest_hash)
        _atomic_bytes(
            self.run_directory / "state.json",
            _canonical_json(state),
            validate_json=True,
        )
        self.state = state

    def _completed_iteration_count(self) -> int:
        if TrainingPhase(self.state["phase"]) is TrainingPhase.ITERATION_COMPLETE:
            return int(self.state["iteration_index"]) + 1
        return max(0, int(self.state["iteration_index"]))


def _initial_population(config: ResolvedTrainingConfig) -> PopulationSnapshot:
    checkpoint_paths = dict(config.population.policy_checkpoints)
    policies: dict[PolicyVersion, Policy] = {}
    for policy_id in config.population.policy_ids:
        seed = derive_pytorch_seed(
            MODEL_INITIALIZATION_NAMESPACE,
            config.run.root_seed,
            "policy",
            policy_id,
            "0",
        )
        model = Policy(seed=seed)
        if policy_id in checkpoint_paths:
            model.load_state_dict(_external_state_dict(checkpoint_paths[policy_id], "policy"))
        policies[PolicyVersion(policy_id, f"{policy_id}-v0")] = model.cpu().eval()
    critic_seed = derive_pytorch_seed(
        MODEL_INITIALIZATION_NAMESPACE,
        config.run.root_seed,
        "critic",
        "shared",
        "0",
    )
    critic = Critic(seed=critic_seed)
    if config.population.critic_checkpoint is not None:
        critic.load_state_dict(
            _external_state_dict(config.population.critic_checkpoint, "critic")
        )
    return PopulationSnapshot(
        policies=dict(sorted(policies.items())),
        critic=critic.cpu().eval(),
        policy_optimizer_states={},
        critic_optimizer_state=None,
        policy_initialization_ordinals={
            policy_id: 0 for policy_id in config.population.policy_ids
        },
        policy_version_ordinals={policy_id: 0 for policy_id in config.population.policy_ids},
        critic_version="critic-v0",
        critic_version_ordinal=0,
        collected_training_rounds=0,
        critic_validation_streak=0,
        next_collection_counter=config.fixtures.next_collection_counter,
        next_evaluation_counter=config.fixtures.next_evaluation_counter,
    )


def _updated_snapshot(
    original: PopulationSnapshot,
    result: OptimizationResult,
    *,
    collected_rounds: int,
    next_collection_counter: int,
) -> PopulationSnapshot:
    policies: dict[PolicyVersion, Policy] = {}
    optimizer_states: dict[PolicyVersion, dict[str, object]] = {}
    version_ordinals = dict(original.policy_version_ordinals)
    for old_identity in sorted(original.policies):
        ordinal = version_ordinals[old_identity.policy_id] + 1
        version_ordinals[old_identity.policy_id] = ordinal
        new_identity = PolicyVersion(
            old_identity.policy_id, f"{old_identity.policy_id}-v{ordinal}"
        )
        model = copy.deepcopy(result.policies[old_identity]).cpu().eval()
        model.zero_grad(set_to_none=True)
        policies[new_identity] = model
        optimizer_states[new_identity] = _to_cpu(
            result.policy_optimizer_states[old_identity]
        )
    critic = copy.deepcopy(result.critic).cpu().eval()
    critic.zero_grad(set_to_none=True)
    diagnostics = {
        "actor_weight": result.actor_weight,
        "entropy_coefficient": result.entropy_coefficient,
        "policies": [
            {
                "policy_id": report.learner.policy_id,
                "passes_completed": report.passes_completed,
                "approximate_kl": list(report.approximate_kl),
                "early_stopped": report.early_stopped,
                "passes": [asdict(metrics) for metrics in report.pass_metrics],
            }
            for report in result.policy_reports
        ],
        "critic_pass_losses": list(result.critic_pass_losses),
        "critic_pass_gradient_norms": list(result.critic_pass_gradient_norms),
        "peak_optimization_device_memory_bytes": result.peak_device_memory_bytes,
    }
    return PopulationSnapshot(
        policies=dict(sorted(policies.items())),
        critic=critic,
        policy_optimizer_states=optimizer_states,
        critic_optimizer_state=_to_cpu(result.critic_optimizer_state),
        policy_initialization_ordinals=dict(original.policy_initialization_ordinals),
        policy_version_ordinals=version_ordinals,
        critic_version=f"critic-v{original.critic_version_ordinal + 1}",
        critic_version_ordinal=original.critic_version_ordinal + 1,
        collected_training_rounds=collected_rounds,
        critic_validation_streak=original.critic_validation_streak,
        next_collection_counter=next_collection_counter,
        next_evaluation_counter=original.next_evaluation_counter,
        optimization_diagnostics=diagnostics,
    )


def _checkpoint_payload(
    snapshot: PopulationSnapshot,
    config: ResolvedTrainingConfig,
    manifest_hash: str,
) -> dict[str, object]:
    _validate_population(snapshot)
    return {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "training_format_version": TRAINING_FORMAT_VERSION,
        "manifest_hash": manifest_hash,
        "manifest": config.manifest(),
        "policies": [
            {
                "policy_id": identity.policy_id,
                "version": identity.version,
                "initialization_ordinal": snapshot.policy_initialization_ordinals[
                    identity.policy_id
                ],
                "version_ordinal": snapshot.policy_version_ordinals[identity.policy_id],
                "state_dict": _cpu_state_dict(snapshot.policies[identity]),
                "optimizer_state": _to_cpu(
                    snapshot.policy_optimizer_states.get(identity)
                ),
            }
            for identity in sorted(snapshot.policies)
        ],
        "critic": {
            "version": snapshot.critic_version,
            "version_ordinal": snapshot.critic_version_ordinal,
            "state_dict": _cpu_state_dict(snapshot.critic),
            "optimizer_state": _to_cpu(snapshot.critic_optimizer_state),
        },
        "counters": {
            "collected_training_rounds": snapshot.collected_training_rounds,
            "critic_validation_streak": snapshot.critic_validation_streak,
            "next_collection_counter": snapshot.next_collection_counter,
            "next_evaluation_counter": snapshot.next_evaluation_counter,
        },
        "optimization_diagnostics": snapshot.optimization_diagnostics,
    }


def _load_checkpoint(
    path: Path,
    config: ResolvedTrainingConfig,
    manifest_hash: str,
) -> PopulationSnapshot:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, ValueError) as error:
        raise TrainingSuiteError("checkpoint could not be loaded") from error
    if (
        not isinstance(payload, dict)
        or payload.get("format_version") != CHECKPOINT_FORMAT_VERSION
        or payload.get("training_format_version") != TRAINING_FORMAT_VERSION
        or payload.get("manifest_hash") != manifest_hash
        or payload.get("manifest") != config.manifest()
    ):
        raise TrainingSuiteError("checkpoint contract or manifest does not match")
    policy_entries = payload.get("policies")
    critic_entry = payload.get("critic")
    counters = payload.get("counters")
    if (
        not isinstance(policy_entries, list)
        or not isinstance(critic_entry, dict)
        or not isinstance(counters, dict)
    ):
        raise TrainingSuiteError("checkpoint population is malformed")
    policies: dict[PolicyVersion, Policy] = {}
    optimizer_states: dict[PolicyVersion, dict[str, object]] = {}
    initialization_ordinals: dict[str, int] = {}
    version_ordinals: dict[str, int] = {}
    for entry in policy_entries:
        if not isinstance(entry, dict):
            raise TrainingSuiteError("checkpoint policy entry is malformed")
        identity = PolicyVersion(entry.get("policy_id"), entry.get("version"))
        model = Policy(seed=0)
        model.load_state_dict(entry.get("state_dict"))
        policies[identity] = model.cpu().eval()
        initialization_ordinals[identity.policy_id] = int(entry["initialization_ordinal"])
        version_ordinals[identity.policy_id] = int(entry["version_ordinal"])
        optimizer_state = entry.get("optimizer_state")
        if optimizer_state is not None:
            optimizer_states[identity] = optimizer_state
    critic = Critic(seed=0)
    critic.load_state_dict(critic_entry.get("state_dict"))
    snapshot = PopulationSnapshot(
        policies=dict(sorted(policies.items())),
        critic=critic.cpu().eval(),
        policy_optimizer_states=optimizer_states,
        critic_optimizer_state=critic_entry.get("optimizer_state"),
        policy_initialization_ordinals=initialization_ordinals,
        policy_version_ordinals=version_ordinals,
        critic_version=str(critic_entry.get("version")),
        critic_version_ordinal=int(critic_entry.get("version_ordinal")),
        collected_training_rounds=int(counters["collected_training_rounds"]),
        critic_validation_streak=int(counters["critic_validation_streak"]),
        next_collection_counter=int(counters["next_collection_counter"]),
        next_evaluation_counter=int(counters["next_evaluation_counter"]),
        optimization_diagnostics=payload.get("optimization_diagnostics"),
    )
    _validate_population(snapshot)
    return snapshot


def _validate_population(snapshot: PopulationSnapshot) -> None:
    if not snapshot.policies or len({identity.policy_id for identity in snapshot.policies}) != len(
        snapshot.policies
    ):
        raise TrainingSuiteError("population policy IDs must be nonempty and unique")
    for identity, policy in snapshot.policies.items():
        if not isinstance(policy, Policy):
            raise TrainingSuiteError("checkpoint contains an invalid policy")
        _validate_finite_module(policy)
        if identity.policy_id not in snapshot.policy_initialization_ordinals:
            raise TrainingSuiteError("policy initialization ordinal is missing")
        if identity.policy_id not in snapshot.policy_version_ordinals:
            raise TrainingSuiteError("policy version ordinal is missing")
    if not isinstance(snapshot.critic, Critic):
        raise TrainingSuiteError("checkpoint contains an invalid critic")
    _validate_finite_module(snapshot.critic)
    for value in (
        snapshot.collected_training_rounds,
        snapshot.critic_validation_streak,
        snapshot.next_collection_counter,
        snapshot.next_evaluation_counter,
    ):
        if type(value) is not int or value < 0:
            raise TrainingSuiteError("checkpoint counters must be non-negative integers")


def _validate_finite_module(module: torch.nn.Module) -> None:
    if any(not torch.isfinite(tensor).all() for tensor in module.state_dict().values()):
        raise TrainingSuiteError("checkpoint model contains NaN or infinity")


def _optimization_config(config: ResolvedTrainingConfig) -> OptimizationConfig:
    training = config.training
    return OptimizationConfig(
        actor_learning_rate=training.actor_learning_rate,
        critic_learning_rate=training.critic_learning_rate,
        adam_betas=(training.adam_beta1, training.adam_beta2),
        adam_epsilon=training.adam_epsilon,
        gradient_norm_limit=training.gradient_norm_limit,
        trajectory_minibatch_size=config.compute.trajectory_batch_size,
        ppo_clip=training.ppo_clip,
        approximate_kl_limit=training.approximate_kl_limit,
        illegal_probability_threshold=training.illegal_probability_threshold,
        burn_in_entropy_coefficient=training.burn_in_entropy_coefficient,
        trained_entropy_coefficient=training.trained_entropy_coefficient,
        minimum_collected_rounds=training.critic_minimum_collected_rounds,
        actor_ramp_rounds=training.actor_ramp_collected_rounds,
        required_critic_validation_windows=training.critic_validation_consecutive_windows,
        actor_weight_override=training.actor_weight_override,
        device=config.compute.optimization_device,
    )


def _iteration_metrics(
    *,
    config: ResolvedTrainingConfig,
    iteration_index: int,
    snapshot: PopulationSnapshot,
    evaluation: EvaluationResult,
    phase_metrics: Mapping[str, object],
    critic_validation_streak: int,
) -> dict[str, object]:
    diagnostics = snapshot.optimization_diagnostics or {}
    policy_diagnostics: dict[str, object] = {}
    for item in diagnostics.get("policies", []):  # type: ignore[union-attr]
        passes = item["passes"]
        policy_diagnostics[item["policy_id"]] = {
            "passes_completed": item["passes_completed"],
            "policy_loss": _mean(pass_["policy_loss"] for pass_ in passes),
            "entropy": _mean(pass_["legal_entropy"] for pass_ in passes),
            "approximate_kl": _mean(item["approximate_kl"]),
            "illegal_probability": _mean(
                pass_["mean_illegal_probability"] for pass_ in passes
            ),
            "gradient_norm": _mean(pass_["gradient_norm"] for pass_ in passes),
        }
    public_phase_metrics = {
        key: value for key, value in phase_metrics.items() if not key.startswith("_")
    }
    swap_start = phase_metrics.get("_swap_used_bytes_at_start")
    swap_end = _swap_used_bytes()
    swap_growth = (
        max(0, swap_end - swap_start)
        if isinstance(swap_start, int) and isinstance(swap_end, int)
        else None
    )
    return {
        "format_version": METRICS_FORMAT_VERSION,
        "run_id": config.run.run_id,
        "iteration_index": iteration_index,
        "contracts": config.manifest()["versions"],
        "compute": asdict(config.compute),
        "population": {
            "policies": [
                {
                    "policy_id": metrics.policy.policy_id,
                    "version": metrics.policy.version,
                    "overall": asdict(metrics.overall),
                    "queen": asdict(metrics.queen),
                    "king": asdict(metrics.king),
                }
                for metrics in evaluation.policy_metrics
            ],
            "critic_version": snapshot.critic_version,
        },
        "training": {
            "actor_weight": diagnostics.get("actor_weight"),
            "entropy_coefficient": diagnostics.get("entropy_coefficient"),
            "policies": policy_diagnostics,
            "critic_mse": _mean(diagnostics.get("critic_pass_losses", [])),
            "critic_gradient_norm": _mean(
                diagnostics.get("critic_pass_gradient_norms", [])
            ),
        },
        "evaluation": {
            "games": len(evaluation.fixture_results),
            "critic_validation": asdict(evaluation.critic_validation),
            "critic_validation_streak": critic_validation_streak,
        },
        "runtime": {
            **public_phase_metrics,
            "peak_process_memory_bytes": _peak_process_memory_bytes(),
            "peak_mps_allocation_bytes": diagnostics.get(
                "peak_optimization_device_memory_bytes"
            ),
            "swap_growth_bytes": swap_growth,
        },
        "validation_failures": [],
    }


def _markdown_report(metrics: Mapping[str, object]) -> str:
    population = metrics["population"]
    evaluation = metrics["evaluation"]
    training = metrics["training"]
    runtime = metrics["runtime"]
    lines = [
        f"# Training iteration {metrics['iteration_index']}",
        "",
        f"Run: `{metrics['run_id']}`  ",
        f"Optimization device: `{metrics['compute']['optimization_device']}`  ",
        f"Critic: `{population['critic_version']}`",
        "",
        "## Policy comparison",
        "",
        "| Policy | Overall | Queen | King | Mean round return |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for policy in population["policies"]:
        lines.append(
            "| {version} | {overall} | {queen} | {king} | {mean_return} |".format(
                version=policy["version"],
                overall=_percentage(policy["overall"]["victory_percentage"]),
                queen=_percentage(policy["queen"]["victory_percentage"]),
                king=_percentage(policy["king"]["victory_percentage"]),
                mean_return=_number_text(policy["overall"]["mean_round_return"]),
            )
        )
    validation = evaluation["critic_validation"]
    lines.extend(
        [
            "",
            "## Training diagnostics",
            "",
            f"Actor weight: `{training['actor_weight']}`  ",
            f"Critic training MSE: `{_number_text(training['critic_mse'])}`  ",
            f"Held-out critic MSE: `{_number_text(validation['critic_mse'])}`  ",
            f"Zero-predictor MSE: `{_number_text(validation['zero_predictor_mse'])}`  ",
            f"Critic validation passed: `{'yes' if validation['passed'] else 'no'}`",
            "",
            "| Policy | Loss | Entropy | KL | Illegal probability | Gradient norm |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for policy_id, diagnostics in training["policies"].items():
        lines.append(
            "| {policy} | {loss} | {entropy} | {kl} | {illegal} | {gradient} |".format(
                policy=policy_id,
                loss=_number_text(diagnostics["policy_loss"]),
                entropy=_number_text(diagnostics["entropy"]),
                kl=_number_text(diagnostics["approximate_kl"]),
                illegal=_number_text(diagnostics["illegal_probability"]),
                gradient=_number_text(diagnostics["gradient_norm"]),
            )
        )
    lines.extend(
        [
            "",
            "## Runtime",
            "",
            f"Collection: `{_number_text(runtime['collection_seconds'])} s`  ",
            f"Optimization: `{_number_text(runtime['optimization_seconds'])} s`  ",
            f"Evaluation: `{_number_text(runtime['evaluation_seconds'])} s`  ",
            f"Peak process memory: `{_byte_text(runtime['peak_process_memory_bytes'])}`  ",
            f"Peak MPS allocation: `{_byte_text(runtime['peak_mps_allocation_bytes'])}`  ",
            f"Swap growth: `{_byte_text(runtime['swap_growth_bytes'])}`",
            "",
        ]
    )
    return "\n".join(lines)


def _validate_run_state(
    state: object,
    config: ResolvedTrainingConfig,
    manifest_hash: str,
) -> None:
    if not isinstance(state, dict):
        raise TrainingSuiteError("run state must be an object")
    if (
        state.get("format_version") != RUN_STATE_FORMAT_VERSION
        or state.get("run_id") != config.run.run_id
        or state.get("manifest_hash") != manifest_hash
    ):
        raise TrainingSuiteError("run state does not match the resolved manifest")
    try:
        TrainingPhase(state.get("phase"))
    except (TypeError, ValueError) as error:
        raise TrainingSuiteError("run state phase is invalid") from error
    if type(state.get("iteration_index")) is not int or int(state["iteration_index"]) < -1:
        raise TrainingSuiteError("run state iteration index is invalid")
    for field in (
        "collected_training_rounds",
        "critic_validation_streak",
        "next_collection_counter",
        "next_evaluation_counter",
    ):
        if type(state.get(field)) is not int or int(state[field]) < 0:
            raise TrainingSuiteError(f"run state {field} is invalid")
    if not isinstance(state.get("artifacts"), dict):
        raise TrainingSuiteError("run state artifacts must be an object")


def _atomic_bytes(path: Path, payload: bytes, *, validate_json: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as temporary:
            temporary_path = temporary.name
            temporary.write(payload)
            temporary.flush()
            os.fsync(temporary.fileno())
        if validate_json:
            json.loads(Path(temporary_path).read_bytes())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and os.path.exists(temporary_path):
            os.unlink(temporary_path)


def _atomic_torch_payload(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as temporary:
            temporary_path = temporary.name
            torch.save(payload, temporary)
            temporary.flush()
            os.fsync(temporary.fileno())
        loaded = torch.load(temporary_path, map_location="cpu", weights_only=True)
        if not isinstance(loaded, dict):
            raise TrainingSuiteError("temporary checkpoint validation failed")
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and os.path.exists(temporary_path):
            os.unlink(temporary_path)


def _reference(run_directory: Path, path: Path) -> dict[str, str]:
    return {
        "path": str(path.resolve().relative_to(run_directory.resolve())),
        "sha256": _sha256(path.read_bytes()),
    }


def _validate_reference(path: Path, reference: Mapping[str, object]) -> None:
    if not path.is_file() or _sha256(path.read_bytes()) != reference.get("sha256"):
        raise TrainingSuiteError("committed artifact is missing or changed")


def _artifact(state: Mapping[str, object], key: str) -> dict[str, str] | None:
    artifacts = state.get("artifacts")
    value = artifacts.get(key) if isinstance(artifacts, dict) else None
    return value if isinstance(value, dict) else None


def _required_artifact(state: Mapping[str, object], key: str) -> dict[str, str]:
    value = _artifact(state, key)
    if value is None or not isinstance(value.get("path"), str) or not isinstance(
        value.get("sha256"), str
    ):
        raise TrainingSuiteError(f"run state is missing the {key} artifact")
    return value


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _cpu_state_dict(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().clone() for name, tensor in module.state_dict().items()
    }


def _to_cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_cpu(item) for item in value)
    return copy.deepcopy(value)


def _policy_versions(snapshot: PopulationSnapshot) -> dict[str, str]:
    return {identity.policy_id: identity.version for identity in sorted(snapshot.policies)}


def _identity_for_id(
    policies: Mapping[PolicyVersion, Policy], policy_id: str
) -> PolicyVersion:
    matches = [identity for identity in policies if identity.policy_id == policy_id]
    if len(matches) != 1:
        raise TrainingSuiteError(f"active policy not found: {policy_id}")
    return matches[0]


def _external_state_dict(path: str, kind: str) -> Mapping[str, torch.Tensor]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, ValueError) as error:
        raise TrainingSuiteError(f"{kind} checkpoint could not be loaded") from error
    if isinstance(payload, dict):
        if "state_dict" in payload:
            payload = payload["state_dict"]
        elif kind == "policy" and isinstance(payload.get("policy"), dict):
            payload = payload["policy"].get("state_dict")
        elif kind == "critic" and isinstance(payload.get("critic"), dict):
            payload = payload["critic"].get("state_dict")
    if not isinstance(payload, dict) or not all(
        isinstance(name, str) and isinstance(tensor, torch.Tensor)
        for name, tensor in payload.items()
    ):
        raise TrainingSuiteError(f"{kind} checkpoint does not contain a state dict")
    return payload


def _mean(values) -> float | None:
    numbers = list(values)
    return sum(numbers) / len(numbers) if numbers else None


def _percentage(value: float | None) -> str:
    return "—" if value is None else f"{100.0 * value:.1f}%"


def _number_text(value: float | None) -> str:
    return "—" if value is None else f"{value:.6g}"


def _byte_text(value: int | None) -> str:
    if value is None:
        return "unavailable"
    return f"{value / (1024 * 1024):.1f} MiB"


def _peak_process_memory_bytes() -> int:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(usage if os.uname().sysname == "Darwin" else usage * 1024)


def _swap_used_bytes() -> int | None:
    if os.uname().sysname == "Darwin":
        completed = subprocess.run(
            ["sysctl", "-n", "vm.swapusage"],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            return None
        match = re.search(r"\bused\s*=\s*([0-9.]+)([KMG])", completed.stdout)
        if match is None:
            return None
        scale = {"K": 1024, "M": 1024**2, "G": 1024**3}[match.group(2)]
        return int(float(match.group(1)) * scale)
    try:
        values = {}
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            key, value = line.split(":", 1)
            if key in {"SwapTotal", "SwapFree"}:
                values[key] = int(value.strip().split()[0]) * 1024
        return values["SwapTotal"] - values["SwapFree"]
    except (OSError, KeyError, ValueError):
        return None


def model_states_close(
    first: torch.nn.Module,
    second: torch.nn.Module,
    *,
    atol: float,
    rtol: float,
) -> bool:
    """Compare CPU/MPS recovery results with the manifest's numerical tolerances."""

    first_state = first.state_dict()
    second_state = second.state_dict()
    return first_state.keys() == second_state.keys() and all(
        torch.allclose(
            first_state[name].detach().cpu(),
            second_state[name].detach().cpu(),
            atol=atol,
            rtol=rtol,
        )
        for name in first_state
    )

"""Define immutable configuration, metric, and result types for policy training.

Keeping these records separate from optimization lets the corpus reader,
checkpoint layer, evaluator, and CLI share one vocabulary without importing the
training loop. The constants below are the persisted training contract used by
resolved configurations and resumable checkpoints.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

TRAINING_CONFIG_FORMAT_VERSION = "dracula-bgc-policy-training-config-v1"
RESOLVED_CONFIG_FORMAT_VERSION = "dracula-bgc-policy-resolved-config-v1"
CHECKPOINT_FORMAT_VERSION = "dracula-bgc-policy-checkpoint-v1"
TRAINING_STATE_FORMAT_VERSION = "dracula-bgc-policy-training-state-v1"
METRICS_FORMAT_VERSION = "dracula-bgc-policy-epoch-metrics-v1"
SUMMARY_FORMAT_VERSION = "dracula-bgc-policy-summary-v1"
EPOCH_SHUFFLE_NAMESPACE = "dracula-bgc-policy-epoch-shuffle-v1"

OUTER_SIMULATION_BUDGET = 128
LEARNING_RATE = 3e-4
BETAS = (0.9, 0.999)
EPSILON = 1e-8
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 256
GRADIENT_CLIP_NORM = 1.0
MINIMUM_EPOCHS = 8
MAXIMUM_EPOCHS = 50
EARLY_STOP_PATIENCE = 5
MINIMUM_IMPROVEMENT = 1e-4
LATEST_CHECKPOINT_INTERVAL = 100


class BGCPolicyTrainingError(ValueError):
    """A corpus, snapshot, batch, configuration, or artifact is invalid."""


class BGCPolicyTrainingInterrupted(RuntimeError):
    """Optimization stopped after sealing its latest minibatch boundary."""


@dataclass(frozen=True, slots=True)
class RunSection:
    """Identity and output location of one training run."""

    run_id: str
    root_seed: str
    output_directory: str


@dataclass(frozen=True, slots=True)
class DatasetSection:
    """Immutable corpus snapshot consumed by training and validation."""

    snapshot_path: str


@dataclass(frozen=True, slots=True)
class ModelSection:
    """Model identity used for deterministic parameter initialization."""

    model_id: str
    initialization_ordinal: int


@dataclass(frozen=True, slots=True)
class OptimizationSection:
    """Execution device; optimizer mathematics are fixed by this contract."""

    device: str = "cpu"


@dataclass(frozen=True, slots=True)
class BGCPolicyTrainingConfig:
    """Typed form of the strict user-supplied TOML configuration."""

    run: RunSection
    dataset: DatasetSection
    model: ModelSection
    optimization: OptimizationSection

    @property
    def output_path(self) -> Path:
        return Path(self.run.output_directory).expanduser().resolve()

    @property
    def snapshot_path(self) -> Path:
        return Path(self.dataset.snapshot_path).expanduser().resolve()


@dataclass(frozen=True, slots=True)
class DistributionMetrics:
    """Averaged distribution loss, entropy, and teacher-action agreement."""

    count: int
    cross_entropy: float
    kl_divergence: float
    target_entropy: float
    model_entropy: float
    top_one: float
    top_two: float
    top_three: float


@dataclass(frozen=True, slots=True)
class EvaluationMetrics:
    """Overall and role/placement/dealer slices for one dataset split."""

    total: DistributionMetrics
    placements: dict[str, DistributionMetrics]
    roles: dict[str, DistributionMetrics]
    dealer_status: dict[str, DistributionMetrics]
    mean_representative_actions: float
    inference_latency_seconds: float


@dataclass(frozen=True, slots=True)
class EpochMetrics:
    """Training and validation measurements sealed after one complete epoch."""

    format_version: str
    epoch: int
    training: EvaluationMetrics
    validation: EvaluationMetrics
    maximum_gradient_norm: float
    epoch_seconds: float
    examples_per_second: float
    peak_rss_bytes: int


@dataclass(frozen=True, slots=True)
class TrainingResult:
    """Paths and checkpoint-selection outcome returned after training."""

    output_directory: str
    completed_epochs: int
    best_epoch: int
    best_validation_cross_entropy: float
    stopped_early: bool
    final_checkpoint: str
    candidate_checkpoint: str
    exported_artifact: str


__all__ = (
    "BATCH_SIZE",
    "BETAS",
    "BGCPolicyTrainingConfig",
    "BGCPolicyTrainingError",
    "BGCPolicyTrainingInterrupted",
    "CHECKPOINT_FORMAT_VERSION",
    "DatasetSection",
    "DistributionMetrics",
    "EARLY_STOP_PATIENCE",
    "EPOCH_SHUFFLE_NAMESPACE",
    "EPSILON",
    "EpochMetrics",
    "EvaluationMetrics",
    "GRADIENT_CLIP_NORM",
    "LATEST_CHECKPOINT_INTERVAL",
    "LEARNING_RATE",
    "MAXIMUM_EPOCHS",
    "METRICS_FORMAT_VERSION",
    "MINIMUM_EPOCHS",
    "MINIMUM_IMPROVEMENT",
    "ModelSection",
    "OUTER_SIMULATION_BUDGET",
    "OptimizationSection",
    "RESOLVED_CONFIG_FORMAT_VERSION",
    "RunSection",
    "SUMMARY_FORMAT_VERSION",
    "TRAINING_CONFIG_FORMAT_VERSION",
    "TRAINING_STATE_FORMAT_VERSION",
    "TrainingResult",
    "WEIGHT_DECAY",
)

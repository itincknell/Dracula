"""Feed-forward policy/value model for search-guided play."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import Tensor, nn

from dracula.cards import CARD_COUNT, CARD_IDS
from dracula.randomness import derive_seed

MODEL_SCHEMA_VERSION = "dracula-policy-value-v1"
OBSERVATION_SCHEMA_VERSION = "dracula-observation-v1"
ACTION_SCHEMA_VERSION = "dracula-action-map-v1"
VALUE_SCHEMA_VERSION = "dracula-round-value-v1"
ARTIFACT_FORMAT_VERSION = "dracula-policy-value-artifact-v1"
OPTIMIZER_COMPATIBILITY_VERSION = "dracula-policy-value-optimizer-v1"
MODEL_INITIALIZATION_NAMESPACE = "dracula-model-initialization-v2"

OBSERVATION_SIZE = 875
HAND_SLOT_COUNT = 4
COFFIN_POSITION_COUNT = 9
POLICY_POSITION_COUNT = 8
ACTION_COUNT = HAND_SLOT_COUNT * POLICY_POSITION_COUNT
POLICY_GRID_INDICES = (0, 1, 2, 3, 5, 6, 7, 8)
PARAMETER_COUNT = 339_978

HAND_FEATURES = HAND_SLOT_COUNT * CARD_COUNT
COFFIN_FEATURES = COFFIN_POSITION_COUNT * CARD_COUNT
STATUS_FEATURES = 3 * CARD_COUNT
CONTEXT_FEATURES = 11
HAND_START = 0
COFFIN_START = HAND_START + HAND_FEATURES
STATUS_START = COFFIN_START + COFFIN_FEATURES
CONTEXT_START = STATUS_START + STATUS_FEATURES

_DIGEST = re.compile(r"^[0-9a-f]{64}$")


class PolicyValueContractError(ValueError):
    """A model input, target, or artifact violates the neural contract."""


@dataclass(frozen=True, slots=True)
class PolicyValueLoss:
    total: Tensor
    policy_cross_entropy: Tensor
    value_mse: Tensor


@dataclass(frozen=True, slots=True)
class PolicyValueOptimizationConfig:
    learning_rate: float = 3e-4
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    weight_decay: float = 1e-4

    def __post_init__(self) -> None:
        values = (
            self.learning_rate,
            self.beta1,
            self.beta2,
            self.epsilon,
            self.weight_decay,
        )
        if any(type(value) is not float or not math.isfinite(value) for value in values):
            raise PolicyValueContractError("optimizer values must be finite floats")
        if self.learning_rate <= 0 or self.epsilon <= 0 or self.weight_decay < 0:
            raise PolicyValueContractError("optimizer scale values are invalid")
        if not 0 <= self.beta1 < 1 or not 0 <= self.beta2 < 1:
            raise PolicyValueContractError("optimizer betas must be in [0, 1)")


@dataclass(frozen=True, slots=True)
class PolicyValueArtifactMetadata:
    model_schema_version: str
    observation_schema_version: str
    action_schema_version: str
    value_schema_version: str
    optimizer_compatibility_version: str
    parameter_count: int
    card_ids: tuple[str, ...]
    policy_grid_indices: tuple[int, ...]
    run_root_seed: str
    model_id: str
    initialization_ordinal: int
    initialization_seed_digest: str
    source_revision: str
    training_config_digest: str
    dataset_digest: str
    search_report_digest: str
    state_dict_digest: str


@dataclass(frozen=True, slots=True)
class LoadedPolicyValueArtifact:
    model: PolicyValueModel
    metadata: PolicyValueArtifactMetadata
    training_configuration: dict[str, object]


@dataclass(frozen=True, slots=True)
class _StructuredObservation:
    hand: Tensor
    coffin: Tensor
    statuses: Tensor
    context: Tensor


def derive_model_initialization(
    run_root_seed: str,
    model_id: str,
    initialization_ordinal: int,
) -> tuple[bytes, int]:
    if not isinstance(run_root_seed, str) or not run_root_seed:
        raise PolicyValueContractError("run root seed must be a nonempty string")
    if not isinstance(model_id, str) or not model_id:
        raise PolicyValueContractError("model ID must be a nonempty string")
    if type(initialization_ordinal) is not int or initialization_ordinal < 0:
        raise PolicyValueContractError("initialization ordinal must be non-negative")
    try:
        digest = derive_seed(
            MODEL_INITIALIZATION_NAMESPACE,
            run_root_seed,
            model_id,
            str(initialization_ordinal),
        )
    except (TypeError, ValueError) as error:
        raise PolicyValueContractError("initialization identity is invalid") from error
    return digest, int.from_bytes(digest[:8], "big", signed=False)


def _validate_observation(observation: Tensor) -> bool:
    if not isinstance(observation, Tensor) or observation.dtype is not torch.bool:
        raise PolicyValueContractError("observation must have Boolean dtype")
    if observation.ndim == 1:
        if observation.shape != (OBSERVATION_SIZE,):
            raise PolicyValueContractError("single observation must have shape [875]")
        return True
    if observation.ndim == 2:
        if observation.shape[0] < 1 or observation.shape[1] != OBSERVATION_SIZE:
            raise PolicyValueContractError("batched observation must have shape [B,875]")
        return False
    raise PolicyValueContractError("observation must be one state or one batch")


def _split_observation(observation: Tensor) -> _StructuredObservation:
    batch_size = observation.shape[0]
    return _StructuredObservation(
        hand=observation[:, HAND_START:COFFIN_START].reshape(
            batch_size, HAND_SLOT_COUNT, CARD_COUNT
        ),
        coffin=observation[:, COFFIN_START:STATUS_START].reshape(
            batch_size, COFFIN_POSITION_COUNT, CARD_COUNT
        ),
        statuses=observation[:, STATUS_START:CONTEXT_START],
        context=observation[:, CONTEXT_START:],
    )


def _initialize(module: nn.Module, seed: int) -> None:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    with torch.no_grad():
        for child in module.modules():
            if isinstance(child, nn.Linear):
                nn.init.xavier_uniform_(child.weight, gain=1.0, generator=generator)
                nn.init.zeros_(child.bias)
            elif isinstance(child, nn.Embedding):
                nn.init.normal_(
                    child.weight,
                    mean=0.0,
                    std=1.0 / math.sqrt(child.embedding_dim),
                    generator=generator,
                )
            elif isinstance(child, nn.LayerNorm):
                nn.init.ones_(child.weight)
                nn.init.zeros_(child.bias)


class _StructuredEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.card_embedding = nn.Embedding(CARD_COUNT, 32)
        self.hand_slot_embedding = nn.Embedding(HAND_SLOT_COUNT, 8)
        self.coffin_position_embedding = nn.Embedding(COFFIN_POSITION_COUNT, 8)
        self.hand_encoder = nn.Linear(41, 64)
        self.coffin_encoder = nn.Linear(41, 64)
        self.status_encoder = nn.Linear(STATUS_FEATURES, 96)
        self.context_encoder = nn.Linear(CONTEXT_FEATURES, 16)
        self.shared_input = nn.Linear(944, 256)
        self.shared_normalization = nn.LayerNorm(256)
        self.shared_output = nn.Linear(256, 128)
        self.activation = nn.GELU()

    def forward(self, observation: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        structured = _split_observation(observation)
        dtype = self.card_embedding.weight.dtype
        batch_size = observation.shape[0]

        hand_cards = structured.hand.to(dtype) @ self.card_embedding.weight
        hand_slots = self.hand_slot_embedding.weight.unsqueeze(0).expand(
            batch_size, -1, -1
        )
        hand_occupied = structured.hand.any(dim=-1, keepdim=True).to(dtype)
        hand_features = self.activation(
            self.hand_encoder(torch.cat((hand_cards, hand_slots, hand_occupied), dim=-1))
        )

        coffin_cards = structured.coffin.to(dtype) @ self.card_embedding.weight
        coffin_positions = self.coffin_position_embedding.weight.unsqueeze(0).expand(
            batch_size, -1, -1
        )
        coffin_occupied = structured.coffin.any(dim=-1, keepdim=True).to(dtype)
        coffin_features = self.activation(
            self.coffin_encoder(
                torch.cat(
                    (coffin_cards, coffin_positions, coffin_occupied), dim=-1
                )
            )
        )

        status_features = self.activation(
            self.status_encoder(structured.statuses.to(dtype))
        )
        context_features = self.activation(
            self.context_encoder(structured.context.to(dtype))
        )
        fused = torch.cat(
            (
                hand_features.flatten(start_dim=1),
                coffin_features.flatten(start_dim=1),
                status_features,
                context_features,
            ),
            dim=-1,
        )
        shared = self.activation(self.shared_input(fused))
        shared = self.shared_normalization(shared)
        shared = self.activation(self.shared_output(shared))
        return shared, hand_features, coffin_features


class PolicyValueModel(nn.Module):
    """Shared representation with raw policy logits and bounded round value."""

    parameter_count = PARAMETER_COUNT

    def __init__(
        self,
        *,
        run_root_seed: str,
        model_id: str,
        initialization_ordinal: int,
    ) -> None:
        super().__init__()
        digest, seed = derive_model_initialization(
            run_root_seed, model_id, initialization_ordinal
        )
        global_rng_state = torch.random.get_rng_state()
        try:
            self.encoder = _StructuredEncoder()
            self.policy_pair_input = nn.Linear(256, 128)
            self.policy_pair_normalization = nn.LayerNorm(128)
            self.policy_pair_output = nn.Linear(128, 1)
            self.value_hidden = nn.Linear(128, 64)
            self.value_output = nn.Linear(64, 1)
            self.activation = nn.GELU()
            self.register_buffer(
                "policy_grid_indices",
                torch.tensor(POLICY_GRID_INDICES, dtype=torch.long),
                persistent=False,
            )
            _initialize(self, seed)
        finally:
            torch.random.set_rng_state(global_rng_state)
        self.run_root_seed = run_root_seed
        self.model_id = model_id
        self.initialization_ordinal = initialization_ordinal
        self.initialization_seed_digest = digest.hex()
        actual_count = sum(parameter.numel() for parameter in self.parameters())
        if actual_count != PARAMETER_COUNT:
            raise RuntimeError(
                f"policy/value parameter count is {actual_count}, expected {PARAMETER_COUNT}"
            )

    def forward(self, observation: Tensor) -> tuple[Tensor, Tensor]:
        single = _validate_observation(observation)
        first_parameter = next(self.parameters())
        if first_parameter.dtype is not torch.float32:
            raise PolicyValueContractError("model parameters must have float32 dtype")
        if observation.device != first_parameter.device:
            raise PolicyValueContractError("model and observation must share a device")
        observation_batch = observation.unsqueeze(0) if single else observation
        shared, hand_features, coffin_features = self.encoder(observation_batch)
        destination_features = coffin_features.index_select(
            1, self.policy_grid_indices
        )
        pair_features = torch.cat(
            (
                hand_features.unsqueeze(2).expand(
                    -1, -1, POLICY_POSITION_COUNT, -1
                ),
                destination_features.unsqueeze(1).expand(
                    -1, HAND_SLOT_COUNT, -1, -1
                ),
                shared[:, None, None, :].expand(
                    -1, HAND_SLOT_COUNT, POLICY_POSITION_COUNT, -1
                ),
            ),
            dim=-1,
        )
        policy_features = self.activation(self.policy_pair_input(pair_features))
        policy_features = self.policy_pair_normalization(policy_features)
        logits = self.policy_pair_output(policy_features).squeeze(-1)
        value = torch.tanh(
            self.value_output(self.activation(self.value_hidden(shared)))
        ).squeeze(-1)
        if single:
            return logits.squeeze(0), value.squeeze(0)
        return logits, value


def _validate_mask(logits: Tensor, legal_mask: Tensor) -> tuple[Tensor, bool]:
    if not isinstance(logits, Tensor) or logits.dtype is not torch.float32:
        raise PolicyValueContractError("policy logits must have float32 dtype")
    if not torch.isfinite(logits).all():
        raise PolicyValueContractError("policy logits must be finite before masking")
    if not isinstance(legal_mask, Tensor) or legal_mask.dtype is not torch.bool:
        raise PolicyValueContractError("legal mask must have Boolean dtype")
    single = logits.ndim == 2
    expected = (
        (HAND_SLOT_COUNT, POLICY_POSITION_COUNT)
        if single
        else (logits.shape[0], HAND_SLOT_COUNT, POLICY_POSITION_COUNT)
        if logits.ndim == 3
        else None
    )
    if expected is None or logits.shape != expected or legal_mask.shape != expected:
        raise PolicyValueContractError("logits and legal mask must have shape [4,8] or [B,4,8]")
    if logits.device != legal_mask.device:
        raise PolicyValueContractError("logits and legal mask must share a device")
    mask_batch = legal_mask.unsqueeze(0) if single else legal_mask
    if not mask_batch.flatten(start_dim=1).any(dim=1).all():
        raise PolicyValueContractError("every state must contain a legal action")
    return mask_batch, single


def apply_legal_mask(logits: Tensor, legal_mask: Tensor) -> Tensor:
    """Apply authoritative legality without changing the neural model."""

    _, _ = _validate_mask(logits, legal_mask)
    return logits.masked_fill(~legal_mask, -torch.inf)


def legal_policy_probabilities(logits: Tensor, legal_mask: Tensor) -> Tensor:
    masked = apply_legal_mask(logits, legal_mask)
    if logits.ndim == 2:
        return torch.softmax(masked.reshape(ACTION_COUNT), dim=0).reshape(logits.shape)
    return torch.softmax(masked.flatten(start_dim=1), dim=1).reshape(logits.shape)


def policy_value_loss(
    logits: Tensor,
    values: Tensor,
    legal_mask: Tensor,
    policy_targets: Tensor,
    value_targets: Tensor,
) -> PolicyValueLoss:
    mask_batch, single = _validate_mask(logits, legal_mask)
    logits_batch = logits.unsqueeze(0) if single else logits
    values_batch = values.unsqueeze(0) if values.ndim == 0 else values
    targets_batch = policy_targets.unsqueeze(0) if single else policy_targets
    value_targets_batch = value_targets.unsqueeze(0) if value_targets.ndim == 0 else value_targets
    batch_size = logits_batch.shape[0]

    if policy_targets.dtype is not torch.float32 or targets_batch.shape != logits_batch.shape:
        raise PolicyValueContractError("policy targets must be float32 with the logits shape")
    if values.dtype is not torch.float32 or values_batch.shape != (batch_size,):
        raise PolicyValueContractError("values must be float32 with shape [B]")
    if value_targets.dtype is not torch.float32 or value_targets_batch.shape != (batch_size,):
        raise PolicyValueContractError("value targets must be float32 with shape [B]")
    tensors = (values_batch, targets_batch, value_targets_batch)
    if any(tensor.device != logits.device for tensor in tensors):
        raise PolicyValueContractError("model outputs, masks, and targets must share a device")
    if any(not torch.isfinite(tensor).all() for tensor in tensors):
        raise PolicyValueContractError("outputs and targets must be finite")
    if torch.any(values_batch < -1) or torch.any(values_batch > 1):
        raise PolicyValueContractError("model values must be in [-1,1]")
    if torch.any(targets_batch < 0):
        raise PolicyValueContractError("policy targets cannot be negative")
    if torch.count_nonzero(targets_batch[~mask_batch]) != 0:
        raise PolicyValueContractError("illegal actions must have zero target probability")
    target_mass = targets_batch.sum(dim=(1, 2))
    if not torch.allclose(
        target_mass, torch.ones_like(target_mass), atol=1e-6, rtol=0.0
    ):
        raise PolicyValueContractError("policy targets must sum to one")
    if torch.any(value_targets_batch < -1) or torch.any(value_targets_batch > 1):
        raise PolicyValueContractError("value targets must be in [-1,1]")

    masked_logits = logits_batch.masked_fill(~mask_batch, -torch.inf)
    log_probabilities = torch.log_softmax(masked_logits.flatten(start_dim=1), dim=1)
    safe_log_probabilities = torch.where(
        mask_batch.flatten(start_dim=1),
        log_probabilities,
        torch.zeros_like(log_probabilities),
    )
    policy_loss = -(
        targets_batch.flatten(start_dim=1) * safe_log_probabilities
    ).sum(dim=1).mean()
    value_loss = torch.mean((values_batch - value_targets_batch) ** 2)
    return PolicyValueLoss(policy_loss + value_loss, policy_loss, value_loss)


def build_policy_value_optimizer(
    model: PolicyValueModel,
    config: PolicyValueOptimizationConfig = PolicyValueOptimizationConfig(),
) -> torch.optim.AdamW:
    if not isinstance(model, PolicyValueModel):
        raise PolicyValueContractError("optimizer requires a PolicyValueModel")
    if not isinstance(config, PolicyValueOptimizationConfig):
        raise PolicyValueContractError("optimizer configuration is invalid")
    return torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
        eps=config.epsilon,
        weight_decay=config.weight_decay,
    )


def _state_dict_digest(state_dict: Mapping[str, Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state_dict.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(",".join(map(str, value.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _canonical_configuration(configuration: Mapping[str, object]) -> tuple[dict[str, object], str]:
    if not isinstance(configuration, Mapping) or any(
        not isinstance(key, str) for key in configuration
    ):
        raise PolicyValueContractError("training configuration must be a string-keyed mapping")
    try:
        encoded = json.dumps(
            dict(configuration),
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        canonical = json.loads(encoded)
    except (TypeError, ValueError) as error:
        raise PolicyValueContractError("training configuration must be canonical JSON") from error
    if not isinstance(canonical, dict):
        raise PolicyValueContractError("training configuration must be an object")
    return canonical, hashlib.sha256(encoded).hexdigest()


def _require_digest(value: object, label: str) -> str:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise PolicyValueContractError(f"{label} must be a lowercase SHA-256 digest")
    return value


def build_policy_value_artifact(
    model: PolicyValueModel,
    *,
    source_revision: str,
    training_configuration: Mapping[str, object],
    dataset_digest: str,
    search_report_digest: str,
) -> dict[str, object]:
    if not isinstance(model, PolicyValueModel):
        raise PolicyValueContractError("artifact requires a PolicyValueModel")
    if not isinstance(source_revision, str) or not source_revision:
        raise PolicyValueContractError("source revision must be nonempty")
    canonical_configuration, config_digest = _canonical_configuration(
        training_configuration
    )
    state_dict = {
        name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()
    }
    metadata = PolicyValueArtifactMetadata(
        model_schema_version=MODEL_SCHEMA_VERSION,
        observation_schema_version=OBSERVATION_SCHEMA_VERSION,
        action_schema_version=ACTION_SCHEMA_VERSION,
        value_schema_version=VALUE_SCHEMA_VERSION,
        optimizer_compatibility_version=OPTIMIZER_COMPATIBILITY_VERSION,
        parameter_count=PARAMETER_COUNT,
        card_ids=CARD_IDS,
        policy_grid_indices=POLICY_GRID_INDICES,
        run_root_seed=model.run_root_seed,
        model_id=model.model_id,
        initialization_ordinal=model.initialization_ordinal,
        initialization_seed_digest=model.initialization_seed_digest,
        source_revision=source_revision,
        training_config_digest=config_digest,
        dataset_digest=_require_digest(dataset_digest, "dataset digest"),
        search_report_digest=_require_digest(
            search_report_digest, "search report digest"
        ),
        state_dict_digest=_state_dict_digest(state_dict),
    )
    return {
        "format_version": ARTIFACT_FORMAT_VERSION,
        "metadata": asdict(metadata),
        "training_configuration": canonical_configuration,
        "state_dict": state_dict,
    }


def save_policy_value_artifact(
    path: str | Path,
    model: PolicyValueModel,
    *,
    source_revision: str,
    training_configuration: Mapping[str, object],
    dataset_digest: str,
    search_report_digest: str,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = build_policy_value_artifact(
        model,
        source_revision=source_revision,
        training_configuration=training_configuration,
        dataset_digest=dataset_digest,
        search_report_digest=search_report_digest,
    )
    temporary = destination.with_name(f".{destination.name}.tmp")
    try:
        torch.save(payload, temporary)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def load_policy_value_artifact(path: str | Path) -> LoadedPolicyValueArtifact:
    try:
        payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, ValueError) as error:
        raise PolicyValueContractError("policy/value artifact could not be loaded") from error
    if not isinstance(payload, dict) or set(payload) != {
        "format_version",
        "metadata",
        "training_configuration",
        "state_dict",
    }:
        raise PolicyValueContractError("policy/value artifact payload is invalid")
    if payload["format_version"] != ARTIFACT_FORMAT_VERSION:
        raise PolicyValueContractError("policy/value artifact format is incompatible")
    metadata_value = payload["metadata"]
    expected_fields = set(PolicyValueArtifactMetadata.__dataclass_fields__)
    if not isinstance(metadata_value, dict) or set(metadata_value) != expected_fields:
        raise PolicyValueContractError("policy/value artifact metadata is invalid")
    try:
        metadata = PolicyValueArtifactMetadata(**metadata_value)
    except TypeError as error:
        raise PolicyValueContractError("policy/value artifact metadata is invalid") from error
    expected_contract = {
        "model_schema_version": MODEL_SCHEMA_VERSION,
        "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
        "action_schema_version": ACTION_SCHEMA_VERSION,
        "value_schema_version": VALUE_SCHEMA_VERSION,
        "optimizer_compatibility_version": OPTIMIZER_COMPATIBILITY_VERSION,
        "parameter_count": PARAMETER_COUNT,
        "card_ids": CARD_IDS,
        "policy_grid_indices": POLICY_GRID_INDICES,
    }
    for field, expected in expected_contract.items():
        if getattr(metadata, field) != expected:
            raise PolicyValueContractError(f"artifact {field} is incompatible")
    for field in (
        "initialization_seed_digest",
        "training_config_digest",
        "dataset_digest",
        "search_report_digest",
        "state_dict_digest",
    ):
        _require_digest(getattr(metadata, field), field.replace("_", " "))
    if not isinstance(metadata.source_revision, str) or not metadata.source_revision:
        raise PolicyValueContractError("artifact source revision is invalid")
    expected_seed_digest, _ = derive_model_initialization(
        metadata.run_root_seed,
        metadata.model_id,
        metadata.initialization_ordinal,
    )
    if metadata.initialization_seed_digest != expected_seed_digest.hex():
        raise PolicyValueContractError("artifact initialization identity is inconsistent")
    configuration, config_digest = _canonical_configuration(
        payload["training_configuration"]  # type: ignore[arg-type]
    )
    if config_digest != metadata.training_config_digest:
        raise PolicyValueContractError("artifact training configuration digest differs")

    model = PolicyValueModel(
        run_root_seed=metadata.run_root_seed,
        model_id=metadata.model_id,
        initialization_ordinal=metadata.initialization_ordinal,
    )
    state_dict = payload["state_dict"]
    expected_state = model.state_dict()
    if not isinstance(state_dict, dict) or set(state_dict) != set(expected_state):
        raise PolicyValueContractError("artifact state-dict keys are incompatible")
    for name, expected in expected_state.items():
        value = state_dict[name]
        if (
            not isinstance(value, Tensor)
            or value.device.type != "cpu"
            or value.dtype is not expected.dtype
            or value.shape != expected.shape
        ):
            raise PolicyValueContractError(f"artifact tensor is incompatible: {name}")
        if not torch.isfinite(value).all():
            raise PolicyValueContractError(f"artifact tensor is non-finite: {name}")
    if _state_dict_digest(state_dict) != metadata.state_dict_digest:
        raise PolicyValueContractError("artifact state-dict digest differs")
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as error:
        raise PolicyValueContractError("artifact state dict could not be loaded") from error
    return LoadedPolicyValueArtifact(model, metadata, configuration)


__all__ = (
    "ACTION_SCHEMA_VERSION",
    "ARTIFACT_FORMAT_VERSION",
    "MODEL_INITIALIZATION_NAMESPACE",
    "MODEL_SCHEMA_VERSION",
    "OBSERVATION_SCHEMA_VERSION",
    "OPTIMIZER_COMPATIBILITY_VERSION",
    "PARAMETER_COUNT",
    "POLICY_GRID_INDICES",
    "VALUE_SCHEMA_VERSION",
    "LoadedPolicyValueArtifact",
    "PolicyValueArtifactMetadata",
    "PolicyValueContractError",
    "PolicyValueLoss",
    "PolicyValueModel",
    "PolicyValueOptimizationConfig",
    "apply_legal_mask",
    "build_policy_value_artifact",
    "build_policy_value_optimizer",
    "derive_model_initialization",
    "legal_policy_probabilities",
    "load_policy_value_artifact",
    "policy_value_loss",
    "save_policy_value_artifact",
)

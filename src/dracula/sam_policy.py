"""Standalone policy classifier for symmetry-aware Sam-32 imitation."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import torch
from torch import Tensor, nn

from dracula.cards import CARD_COUNT, CARD_IDS
from dracula.randomness import Sha256CounterStream, derive_seed, seed_hex

MODEL_SCHEMA_VERSION = "dracula-sam-policy-v1"
ARTIFACT_SCHEMA_VERSION = "dracula-sam-policy-artifact-v1"
OBSERVATION_SCHEMA_VERSION = "dracula-observation-v1"
ACTION_SCHEMA_VERSION = "dracula-action-map-v1"
REPRESENTATIVE_MASK_SCHEMA_VERSION = "dracula-sam-representative-mask-v1"
DESTINATION_SYMMETRY_SCHEMA_VERSION = (
    "dracula-early-destination-symmetry-v1"
)
INITIALIZATION_SCHEMA_VERSION = "dracula-sam-policy-initialization-v1"
TRAINING_SNAPSHOT_SCHEMA_VERSION = "dracula-sam-policy-snapshot-v1"
OPTIMIZER_COMPATIBILITY_VERSION = "dracula-sam-policy-optimizer-v1"
MODEL_INITIALIZATION_NAMESPACE = "dracula-sam-policy-initialization-v1"
INFERENCE_DESTINATION_SCOPE = "sam-policy-argmax-result-v1"

OBSERVATION_SIZE = 875
HAND_SLOT_COUNT = 4
COFFIN_POSITION_COUNT = 9
POLICY_POSITION_COUNT = 8
ACTION_COUNT = HAND_SLOT_COUNT * POLICY_POSITION_COUNT
POLICY_GRID_INDICES = (0, 1, 2, 3, 5, 6, 7, 8)
PARAMETER_COUNT = 738_569

HAND_FEATURES = HAND_SLOT_COUNT * CARD_COUNT
COFFIN_FEATURES = COFFIN_POSITION_COUNT * CARD_COUNT
STATUS_FEATURES = 3 * CARD_COUNT
CONTEXT_FEATURES = 11
HAND_START = 0
COFFIN_START = HAND_START + HAND_FEATURES
STATUS_START = COFFIN_START + COFFIN_FEATURES
CONTEXT_START = STATUS_START + STATUS_FEATURES

_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_SOURCE_REVISION = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_POLICY_POSITION_BY_GRID_INDEX = {
    grid_index: position
    for position, grid_index in enumerate(POLICY_GRID_INDICES)
}
_AUTHORIZED_PAIRED_DESTINATIONS = frozenset(
    {
        (0, (0, 2)),
        (0, (0, 6)),
        (1, (1, 7)),
        (2, (2, 8)),
        (3, (3, 5)),
        (6, (6, 8)),
        (8, (6, 8)),
    }
)


class SamPolicyContractError(ValueError):
    """A model input, mask projection, or artifact violates the contract."""


class StrategicGroupLike(Protocol):
    hand_slot: int
    representative_action_index: int
    representative_grid_index: int
    member_action_indices: tuple[int, ...]
    member_grid_indices: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class RepresentativeActionGroup:
    hand_slot: int
    representative_action_index: int
    member_action_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        if (
            type(self.hand_slot) is not int
            or not 0 <= self.hand_slot < HAND_SLOT_COUNT
            or type(self.representative_action_index) is not int
            or not 0 <= self.representative_action_index < ACTION_COUNT
            or self.representative_action_index // POLICY_POSITION_COUNT
            != self.hand_slot
        ):
            raise SamPolicyContractError(
                "representative action identity is invalid"
            )
        if (
            not isinstance(self.member_action_indices, tuple)
            or len(self.member_action_indices) not in (1, 2)
            or tuple(sorted(self.member_action_indices))
            != self.member_action_indices
            or len(set(self.member_action_indices))
            != len(self.member_action_indices)
            or self.representative_action_index
            not in self.member_action_indices
            or any(
                type(action_index) is not int
                or not 0 <= action_index < ACTION_COUNT
                or action_index // POLICY_POSITION_COUNT != self.hand_slot
                for action_index in self.member_action_indices
            )
        ):
            raise SamPolicyContractError(
                "representative group members are invalid"
            )


@dataclass(frozen=True, slots=True, eq=False)
class RepresentativeActionProjection:
    mask: Tensor
    groups: tuple[RepresentativeActionGroup, ...]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.mask, Tensor)
            or self.mask.dtype is not torch.bool
            or self.mask.shape
            != (HAND_SLOT_COUNT, POLICY_POSITION_COUNT)
            or int(self.mask.sum().item()) != len(self.groups)
        ):
            raise SamPolicyContractError(
                "representative projection mask is invalid"
            )
        representatives = tuple(
            group.representative_action_index for group in self.groups
        )
        if (
            tuple(sorted(representatives)) != representatives
            or len(set(representatives)) != len(representatives)
            or any(
                not bool(self.mask.flatten()[representative].item())
                for representative in representatives
            )
        ):
            raise SamPolicyContractError(
                "representative projection groups are invalid"
            )

    def group_for(
        self, representative_action_index: int
    ) -> RepresentativeActionGroup:
        if type(representative_action_index) is not int:
            raise SamPolicyContractError(
                "representative action index must be an integer"
            )
        try:
            return next(
                group
                for group in self.groups
                if group.representative_action_index
                == representative_action_index
            )
        except StopIteration as error:
            raise SamPolicyContractError(
                "action is not a legal representative"
            ) from error


@dataclass(frozen=True, slots=True)
class SamPolicyArtifactMetadata:
    model_schema_version: str
    artifact_schema_version: str
    observation_schema_version: str
    action_schema_version: str
    representative_mask_schema_version: str
    destination_symmetry_schema_version: str
    initialization_schema_version: str
    training_snapshot_schema_version: str
    optimizer_compatibility_version: str
    parameter_count: int
    card_ids: tuple[str, ...]
    policy_grid_indices: tuple[int, ...]
    run_root_seed: str
    model_id: str
    initialization_ordinal: int
    initialization_seed_digest: str
    source_revision: str
    source_tree_digest: str
    training_config_digest: str
    corpus_snapshot_digest: str
    state_dict_digest: str


@dataclass(frozen=True, slots=True)
class LoadedSamPolicyArtifact:
    model: SamPolicyModel
    metadata: SamPolicyArtifactMetadata
    training_configuration: dict[str, object]


@dataclass(frozen=True, slots=True)
class _StructuredObservation:
    hand: Tensor
    coffin: Tensor
    statuses: Tensor
    context: Tensor


def _validate_engine_legal_mask(legal_mask: Tensor) -> tuple[int, ...]:
    if (
        not isinstance(legal_mask, Tensor)
        or legal_mask.dtype is not torch.bool
        or legal_mask.shape != (HAND_SLOT_COUNT, POLICY_POSITION_COUNT)
    ):
        raise SamPolicyContractError(
            "engine legal mask must be bool[4,8]"
        )
    legal = tuple(
        index
        for index, allowed in enumerate(legal_mask.flatten().tolist())
        if allowed
    )
    if not legal:
        raise SamPolicyContractError(
            "engine legal mask must contain an action"
        )
    return legal


def _copy_and_validate_group(
    group: StrategicGroupLike,
) -> RepresentativeActionGroup:
    try:
        hand_slot = group.hand_slot
        representative = group.representative_action_index
        representative_grid = group.representative_grid_index
        members = group.member_action_indices
        member_grids = group.member_grid_indices
    except AttributeError as error:
        raise SamPolicyContractError(
            "strategic group fields are incomplete"
        ) from error
    copied = RepresentativeActionGroup(
        hand_slot,
        representative,
        members,
    )
    if (
        type(representative_grid) is not int
        or representative_grid
        != POLICY_GRID_INDICES[
            representative % POLICY_POSITION_COUNT
        ]
        or not isinstance(member_grids, tuple)
        or member_grids
        != tuple(
            POLICY_GRID_INDICES[
                member % POLICY_POSITION_COUNT
            ]
            for member in members
        )
    ):
        raise SamPolicyContractError(
            "strategic group grid and action indexes disagree"
        )
    if len(members) == 2 and (
        representative_grid,
        member_grids,
    ) not in _AUTHORIZED_PAIRED_DESTINATIONS:
        raise SamPolicyContractError(
            "paired destinations are outside the authoritative table"
        )
    return copied


def build_representative_action_projection(
    engine_legal_mask: Tensor,
    strategic_groups: Sequence[StrategicGroupLike],
) -> RepresentativeActionProjection:
    """Retain exactly one proxy action for every authoritative group."""

    legal = _validate_engine_legal_mask(engine_legal_mask)
    if (
        not isinstance(strategic_groups, Sequence)
        or not strategic_groups
    ):
        raise SamPolicyContractError(
            "strategic groups must be a nonempty sequence"
        )
    groups = tuple(
        sorted(
            (_copy_and_validate_group(group) for group in strategic_groups),
            key=lambda group: group.representative_action_index,
        )
    )
    members = tuple(
        member for group in groups for member in group.member_action_indices
    )
    if (
        len(set(members)) != len(members)
        or set(members) != set(legal)
    ):
        raise SamPolicyContractError(
            "strategic groups must partition engine-legal actions"
        )

    legal_slots = {
        action_index // POLICY_POSITION_COUNT for action_index in legal
    }
    grouped_slots = {group.hand_slot for group in groups}
    if grouped_slots != legal_slots:
        raise SamPolicyContractError(
            "strategic groups do not cover every legal hand card"
        )
    destination_shape: tuple[
        tuple[int, tuple[int, ...]], ...
    ] | None = None
    for hand_slot in sorted(legal_slots):
        shape = tuple(
            (
                group.representative_action_index
                % POLICY_POSITION_COUNT,
                tuple(
                    member % POLICY_POSITION_COUNT
                    for member in group.member_action_indices
                ),
            )
            for group in groups
            if group.hand_slot == hand_slot
        )
        if destination_shape is None:
            destination_shape = shape
        elif shape != destination_shape:
            raise SamPolicyContractError(
                "destination grouping must be identical for every legal card"
            )

    mask = torch.zeros_like(engine_legal_mask)
    flat_mask = mask.flatten()
    for group in groups:
        flat_mask[group.representative_action_index] = True
    return RepresentativeActionProjection(mask, groups)


def resolve_representative_action(
    projection: RepresentativeActionProjection,
    representative_action_index: int,
    choice_seed: bytes,
) -> int:
    """Resolve a paired proxy with the established SHA-256 fair coin."""

    if not isinstance(projection, RepresentativeActionProjection):
        raise SamPolicyContractError(
            "concrete resolution requires a representative projection"
        )
    seed_hex(choice_seed)
    group = projection.group_for(representative_action_index)
    if len(group.member_action_indices) == 1:
        return group.member_action_indices[0]
    return group.member_action_indices[
        Sha256CounterStream(choice_seed).randbelow(2)
    ]


def derive_policy_initialization(
    run_root_seed: str,
    model_id: str,
    initialization_ordinal: int,
) -> tuple[bytes, int]:
    if not isinstance(run_root_seed, str) or not run_root_seed:
        raise SamPolicyContractError(
            "run root seed must be a nonempty string"
        )
    if not isinstance(model_id, str) or not model_id:
        raise SamPolicyContractError(
            "model ID must be a nonempty string"
        )
    if (
        type(initialization_ordinal) is not int
        or initialization_ordinal < 0
    ):
        raise SamPolicyContractError(
            "initialization ordinal must be non-negative"
        )
    try:
        digest = derive_seed(
            MODEL_INITIALIZATION_NAMESPACE,
            run_root_seed,
            model_id,
            str(initialization_ordinal),
        )
    except (TypeError, ValueError) as error:
        raise SamPolicyContractError(
            "initialization identity is invalid"
        ) from error
    return digest, int.from_bytes(digest[:8], "big", signed=False)


def _validate_observation(observation: Tensor) -> bool:
    if (
        not isinstance(observation, Tensor)
        or observation.dtype is not torch.bool
    ):
        raise SamPolicyContractError(
            "observation must have Boolean dtype"
        )
    if observation.ndim == 1:
        if observation.shape != (OBSERVATION_SIZE,):
            raise SamPolicyContractError(
                "single observation must have shape [875]"
            )
        return True
    if observation.ndim == 2:
        if (
            observation.shape[0] < 1
            or observation.shape[1] != OBSERVATION_SIZE
        ):
            raise SamPolicyContractError(
                "batched observation must have shape [B,875]"
            )
        return False
    raise SamPolicyContractError(
        "observation must be one state or one batch"
    )


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
                nn.init.xavier_uniform_(
                    child.weight,
                    gain=1.0,
                    generator=generator,
                )
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


class _SamPolicyEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.card_embedding = nn.Embedding(CARD_COUNT, 32)
        self.hand_slot_embedding = nn.Embedding(HAND_SLOT_COUNT, 8)
        self.coffin_position_embedding = nn.Embedding(
            COFFIN_POSITION_COUNT, 8
        )
        self.hand_encoder = nn.Linear(41, 64)
        self.coffin_encoder = nn.Linear(41, 64)
        self.status_encoder = nn.Linear(STATUS_FEATURES, 96)
        self.context_encoder = nn.Linear(CONTEXT_FEATURES, 16)
        self.shared_input = nn.Linear(944, 512)
        self.shared_normalization = nn.LayerNorm(512, eps=1e-5)
        self.shared_output = nn.Linear(512, 256)
        self.activation = nn.GELU(approximate="none")

    def forward(
        self, observation: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        structured = _split_observation(observation)
        dtype = self.card_embedding.weight.dtype
        batch_size = observation.shape[0]

        hand_cards = (
            structured.hand.to(dtype) @ self.card_embedding.weight
        )
        hand_slots = self.hand_slot_embedding.weight.unsqueeze(0).expand(
            batch_size, -1, -1
        )
        hand_occupied = structured.hand.any(
            dim=-1, keepdim=True
        ).to(dtype)
        hand_features = self.activation(
            self.hand_encoder(
                torch.cat(
                    (hand_cards, hand_slots, hand_occupied),
                    dim=-1,
                )
            )
        )

        coffin_cards = (
            structured.coffin.to(dtype) @ self.card_embedding.weight
        )
        coffin_positions = (
            self.coffin_position_embedding.weight.unsqueeze(0).expand(
                batch_size, -1, -1
            )
        )
        coffin_occupied = structured.coffin.any(
            dim=-1, keepdim=True
        ).to(dtype)
        coffin_features = self.activation(
            self.coffin_encoder(
                torch.cat(
                    (
                        coffin_cards,
                        coffin_positions,
                        coffin_occupied,
                    ),
                    dim=-1,
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


class SamPolicyModel(nn.Module):
    """Structured player-visible classifier with 32 raw action logits."""

    parameter_count = PARAMETER_COUNT

    def __init__(
        self,
        *,
        run_root_seed: str,
        model_id: str,
        initialization_ordinal: int,
    ) -> None:
        super().__init__()
        digest, seed = derive_policy_initialization(
            run_root_seed,
            model_id,
            initialization_ordinal,
        )
        global_rng_state = torch.random.get_rng_state()
        try:
            self.encoder = _SamPolicyEncoder()
            self.pair_hidden = nn.Linear(384, 256)
            self.pair_normalization = nn.LayerNorm(256, eps=1e-5)
            self.pair_output = nn.Linear(256, 1)
            self.activation = nn.GELU(approximate="none")
            self.register_buffer(
                "policy_grid_indices",
                torch.tensor(POLICY_GRID_INDICES, dtype=torch.long),
                persistent=False,
            )
            self.float()
            _initialize(self, seed)
        finally:
            torch.random.set_rng_state(global_rng_state)
        self.run_root_seed = run_root_seed
        self.model_id = model_id
        self.initialization_ordinal = initialization_ordinal
        self.initialization_seed_digest = digest.hex()
        actual_count = sum(
            parameter.numel() for parameter in self.parameters()
        )
        if actual_count != PARAMETER_COUNT:
            raise RuntimeError(
                f"Sam policy parameter count is {actual_count}, "
                f"expected {PARAMETER_COUNT}"
            )

    def forward(self, observation: Tensor) -> Tensor:
        single = _validate_observation(observation)
        first_parameter = next(self.parameters())
        if first_parameter.dtype is not torch.float32:
            raise SamPolicyContractError(
                "model parameters must have float32 dtype"
            )
        if observation.device != first_parameter.device:
            raise SamPolicyContractError(
                "model and observation must share a device"
            )
        batch = observation.unsqueeze(0) if single else observation
        shared, hand_features, coffin_features = self.encoder(batch)
        destinations = coffin_features.index_select(
            1, self.policy_grid_indices
        )
        pair_features = torch.cat(
            (
                hand_features.unsqueeze(2).expand(
                    -1, -1, POLICY_POSITION_COUNT, -1
                ),
                destinations.unsqueeze(1).expand(
                    -1, HAND_SLOT_COUNT, -1, -1
                ),
                shared[:, None, None, :].expand(
                    -1,
                    HAND_SLOT_COUNT,
                    POLICY_POSITION_COUNT,
                    -1,
                ),
            ),
            dim=-1,
        )
        pair_features = self.activation(
            self.pair_hidden(pair_features)
        )
        pair_features = self.pair_normalization(pair_features)
        logits = self.pair_output(pair_features).squeeze(-1)
        return logits.squeeze(0) if single else logits


def _validate_logits_and_mask(
    logits: Tensor, representative_mask: Tensor
) -> None:
    if (
        not isinstance(logits, Tensor)
        or logits.dtype is not torch.float32
        or not torch.isfinite(logits).all()
    ):
        raise SamPolicyContractError(
            "policy logits must be finite float32"
        )
    if (
        not isinstance(representative_mask, Tensor)
        or representative_mask.dtype is not torch.bool
        or representative_mask.shape != logits.shape
    ):
        raise SamPolicyContractError(
            "representative mask must be Boolean with the logits shape"
        )
    if logits.device != representative_mask.device:
        raise SamPolicyContractError(
            "logits and representative mask must share a device"
        )
    if logits.ndim == 2:
        if logits.shape != (
            HAND_SLOT_COUNT,
            POLICY_POSITION_COUNT,
        ) or not representative_mask.any():
            raise SamPolicyContractError(
                "single logits and mask must have shape [4,8]"
            )
    elif logits.ndim == 3:
        if (
            logits.shape[1:]
            != (HAND_SLOT_COUNT, POLICY_POSITION_COUNT)
            or logits.shape[0] < 1
            or not representative_mask.flatten(start_dim=1).any(dim=1).all()
        ):
            raise SamPolicyContractError(
                "batched logits and mask must have shape [B,4,8]"
            )
    else:
        raise SamPolicyContractError(
            "logits must have shape [4,8] or [B,4,8]"
        )


def apply_representative_mask(
    logits: Tensor, representative_mask: Tensor
) -> Tensor:
    """Apply proxy legality outside the standalone model."""

    _validate_logits_and_mask(logits, representative_mask)
    return logits.masked_fill(~representative_mask, -torch.inf)


def representative_policy_probabilities(
    logits: Tensor, representative_mask: Tensor
) -> Tensor:
    masked = apply_representative_mask(logits, representative_mask)
    if logits.ndim == 2:
        return torch.softmax(
            masked.reshape(ACTION_COUNT), dim=0
        ).reshape(logits.shape)
    return torch.softmax(
        masked.flatten(start_dim=1), dim=1
    ).reshape(logits.shape)


def select_representative_action(
    logits: Tensor, representative_mask: Tensor
) -> int | Tensor:
    """Select masked argmax; flattening supplies the canonical tie-break."""

    masked = apply_representative_mask(logits, representative_mask)
    if logits.ndim == 2:
        return int(torch.argmax(masked.reshape(ACTION_COUNT)).item())
    return torch.argmax(masked.flatten(start_dim=1), dim=1)


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


def _canonical_configuration(
    configuration: Mapping[str, object],
) -> tuple[dict[str, object], str]:
    if not isinstance(configuration, Mapping) or any(
        not isinstance(key, str) for key in configuration
    ):
        raise SamPolicyContractError(
            "training configuration must be a string-keyed mapping"
        )
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
        raise SamPolicyContractError(
            "training configuration must be canonical JSON"
        ) from error
    if not isinstance(canonical, dict):
        raise SamPolicyContractError(
            "training configuration must be an object"
        )
    return canonical, hashlib.sha256(encoded).hexdigest()


def _require_digest(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or _DIGEST.fullmatch(value) is None
    ):
        raise SamPolicyContractError(
            f"{label} must be a lowercase SHA-256 digest"
        )
    return value


def build_sam_policy_artifact(
    model: SamPolicyModel,
    *,
    source_revision: str,
    source_tree_digest: str,
    training_configuration: Mapping[str, object],
    corpus_snapshot_digest: str,
) -> dict[str, object]:
    if not isinstance(model, SamPolicyModel):
        raise SamPolicyContractError(
            "artifact requires a SamPolicyModel"
        )
    if (
        not isinstance(source_revision, str)
        or _SOURCE_REVISION.fullmatch(source_revision) is None
    ):
        raise SamPolicyContractError(
            "source revision must be a lowercase Git object ID"
        )
    canonical_configuration, config_digest = (
        _canonical_configuration(training_configuration)
    )
    state_dict = {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }
    if any(
        tensor.dtype is not torch.float32
        or not torch.isfinite(tensor).all()
        for tensor in state_dict.values()
    ):
        raise SamPolicyContractError(
            "artifact model parameters must be finite float32"
        )
    metadata = SamPolicyArtifactMetadata(
        model_schema_version=MODEL_SCHEMA_VERSION,
        artifact_schema_version=ARTIFACT_SCHEMA_VERSION,
        observation_schema_version=OBSERVATION_SCHEMA_VERSION,
        action_schema_version=ACTION_SCHEMA_VERSION,
        representative_mask_schema_version=(
            REPRESENTATIVE_MASK_SCHEMA_VERSION
        ),
        destination_symmetry_schema_version=(
            DESTINATION_SYMMETRY_SCHEMA_VERSION
        ),
        initialization_schema_version=(
            INITIALIZATION_SCHEMA_VERSION
        ),
        training_snapshot_schema_version=(
            TRAINING_SNAPSHOT_SCHEMA_VERSION
        ),
        optimizer_compatibility_version=(
            OPTIMIZER_COMPATIBILITY_VERSION
        ),
        parameter_count=PARAMETER_COUNT,
        card_ids=CARD_IDS,
        policy_grid_indices=POLICY_GRID_INDICES,
        run_root_seed=model.run_root_seed,
        model_id=model.model_id,
        initialization_ordinal=model.initialization_ordinal,
        initialization_seed_digest=(
            model.initialization_seed_digest
        ),
        source_revision=source_revision,
        source_tree_digest=_require_digest(
            source_tree_digest, "source tree digest"
        ),
        training_config_digest=config_digest,
        corpus_snapshot_digest=_require_digest(
            corpus_snapshot_digest, "corpus snapshot digest"
        ),
        state_dict_digest=_state_dict_digest(state_dict),
    )
    return {
        "format_version": ARTIFACT_SCHEMA_VERSION,
        "metadata": asdict(metadata),
        "training_configuration": canonical_configuration,
        "state_dict": state_dict,
    }


def save_sam_policy_artifact(
    path: str | Path,
    model: SamPolicyModel,
    *,
    source_revision: str,
    source_tree_digest: str,
    training_configuration: Mapping[str, object],
    corpus_snapshot_digest: str,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = build_sam_policy_artifact(
        model,
        source_revision=source_revision,
        source_tree_digest=source_tree_digest,
        training_configuration=training_configuration,
        corpus_snapshot_digest=corpus_snapshot_digest,
    )
    temporary = destination.with_name(
        f".{destination.name}.tmp-{os.getpid()}"
    )
    try:
        torch.save(payload, temporary)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def load_sam_policy_artifact(
    path: str | Path,
) -> LoadedSamPolicyArtifact:
    try:
        payload = torch.load(
            Path(path), map_location="cpu", weights_only=True
        )
    except (OSError, RuntimeError, ValueError) as error:
        raise SamPolicyContractError(
            "Sam policy artifact could not be loaded"
        ) from error
    if not isinstance(payload, dict) or set(payload) != {
        "format_version",
        "metadata",
        "training_configuration",
        "state_dict",
    }:
        raise SamPolicyContractError(
            "Sam policy artifact payload is invalid"
        )
    if payload["format_version"] != ARTIFACT_SCHEMA_VERSION:
        raise SamPolicyContractError(
            "Sam policy artifact format is incompatible"
        )
    metadata_value = payload["metadata"]
    expected_fields = set(
        SamPolicyArtifactMetadata.__dataclass_fields__
    )
    if (
        not isinstance(metadata_value, dict)
        or set(metadata_value) != expected_fields
    ):
        raise SamPolicyContractError(
            "Sam policy artifact metadata is invalid"
        )
    try:
        metadata = SamPolicyArtifactMetadata(**metadata_value)
    except TypeError as error:
        raise SamPolicyContractError(
            "Sam policy artifact metadata is invalid"
        ) from error

    expected_contract = {
        "model_schema_version": MODEL_SCHEMA_VERSION,
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
        "action_schema_version": ACTION_SCHEMA_VERSION,
        "representative_mask_schema_version": (
            REPRESENTATIVE_MASK_SCHEMA_VERSION
        ),
        "destination_symmetry_schema_version": (
            DESTINATION_SYMMETRY_SCHEMA_VERSION
        ),
        "initialization_schema_version": (
            INITIALIZATION_SCHEMA_VERSION
        ),
        "training_snapshot_schema_version": (
            TRAINING_SNAPSHOT_SCHEMA_VERSION
        ),
        "optimizer_compatibility_version": (
            OPTIMIZER_COMPATIBILITY_VERSION
        ),
        "parameter_count": PARAMETER_COUNT,
        "card_ids": CARD_IDS,
        "policy_grid_indices": POLICY_GRID_INDICES,
    }
    for field, expected in expected_contract.items():
        if getattr(metadata, field) != expected:
            raise SamPolicyContractError(
                f"artifact {field} is incompatible"
            )
    for field in (
        "initialization_seed_digest",
        "source_tree_digest",
        "training_config_digest",
        "corpus_snapshot_digest",
        "state_dict_digest",
    ):
        _require_digest(
            getattr(metadata, field), field.replace("_", " ")
        )
    if (
        not isinstance(metadata.source_revision, str)
        or _SOURCE_REVISION.fullmatch(metadata.source_revision) is None
    ):
        raise SamPolicyContractError(
            "artifact source revision is invalid"
        )
    expected_seed, _ = derive_policy_initialization(
        metadata.run_root_seed,
        metadata.model_id,
        metadata.initialization_ordinal,
    )
    if metadata.initialization_seed_digest != expected_seed.hex():
        raise SamPolicyContractError(
            "artifact initialization identity is inconsistent"
        )
    configuration, config_digest = _canonical_configuration(
        payload["training_configuration"]  # type: ignore[arg-type]
    )
    if config_digest != metadata.training_config_digest:
        raise SamPolicyContractError(
            "artifact training configuration digest differs"
        )

    model = SamPolicyModel(
        run_root_seed=metadata.run_root_seed,
        model_id=metadata.model_id,
        initialization_ordinal=metadata.initialization_ordinal,
    )
    state_dict = payload["state_dict"]
    expected_state = model.state_dict()
    if (
        not isinstance(state_dict, dict)
        or set(state_dict) != set(expected_state)
    ):
        raise SamPolicyContractError(
            "artifact state-dict keys are incompatible"
        )
    for name, expected in expected_state.items():
        value = state_dict[name]
        if (
            not isinstance(value, Tensor)
            or value.device.type != "cpu"
            or value.dtype is not expected.dtype
            or value.shape != expected.shape
        ):
            raise SamPolicyContractError(
                f"artifact tensor is incompatible: {name}"
            )
        if not torch.isfinite(value).all():
            raise SamPolicyContractError(
                f"artifact tensor is non-finite: {name}"
            )
    if _state_dict_digest(state_dict) != metadata.state_dict_digest:
        raise SamPolicyContractError(
            "artifact state-dict digest differs"
        )
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as error:
        raise SamPolicyContractError(
            "artifact state dict could not be loaded"
        ) from error
    return LoadedSamPolicyArtifact(
        model, metadata, configuration
    )


__all__ = (
    "ACTION_SCHEMA_VERSION",
    "ACTION_COUNT",
    "ARTIFACT_SCHEMA_VERSION",
    "DESTINATION_SYMMETRY_SCHEMA_VERSION",
    "INFERENCE_DESTINATION_SCOPE",
    "INITIALIZATION_SCHEMA_VERSION",
    "MODEL_INITIALIZATION_NAMESPACE",
    "MODEL_SCHEMA_VERSION",
    "OBSERVATION_SCHEMA_VERSION",
    "OBSERVATION_SIZE",
    "OPTIMIZER_COMPATIBILITY_VERSION",
    "PARAMETER_COUNT",
    "POLICY_GRID_INDICES",
    "REPRESENTATIVE_MASK_SCHEMA_VERSION",
    "TRAINING_SNAPSHOT_SCHEMA_VERSION",
    "LoadedSamPolicyArtifact",
    "RepresentativeActionGroup",
    "RepresentativeActionProjection",
    "SamPolicyArtifactMetadata",
    "SamPolicyContractError",
    "SamPolicyModel",
    "apply_representative_mask",
    "build_representative_action_projection",
    "build_sam_policy_artifact",
    "derive_policy_initialization",
    "load_sam_policy_artifact",
    "representative_policy_probabilities",
    "resolve_representative_action",
    "save_sam_policy_artifact",
    "select_representative_action",
)

"""Permutation-invariant standalone policy for BGC visit distillation."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from dracula.cards import CARD_COUNT
from dracula.randomness import derive_seed

MODEL_SCHEMA_VERSION = "dracula-bgc-card-policy-v2"
OBSERVATION_SCHEMA_VERSION = "dracula-observation-card-set-v2"
ACTION_SCHEMA_VERSION = "dracula-card-candidate-action-map-v2"
INITIALIZATION_SCHEMA_VERSION = "dracula-bgc-card-policy-initialization-v2"
INITIALIZATION_NAMESPACE = "dracula-bgc-card-policy-initialization-v2"

LEGACY_OBSERVATION_SIZE = 875
LEGACY_HAND_BITS = 4 * CARD_COUNT
OBSERVATION_SIZE = LEGACY_OBSERVATION_SIZE - LEGACY_HAND_BITS
HAND_CANDIDATE_COUNT = 4
POLICY_POSITION_COUNT = 8
ACTION_COUNT = HAND_CANDIDATE_COUNT * POLICY_POSITION_COUNT
COFFIN_POSITION_COUNT = 9
POLICY_GRID_INDICES = (0, 1, 2, 3, 5, 6, 7, 8)

COFFIN_FEATURES = COFFIN_POSITION_COUNT * CARD_COUNT
STATUS_FEATURES = 3 * CARD_COUNT
CONTEXT_FEATURES = 11
COFFIN_START = 0
STATUS_START = COFFIN_FEATURES
CONTEXT_START = STATUS_START + STATUS_FEATURES
IN_HAND_START = STATUS_START + CARD_COUNT
PARAMETER_COUNT = 754_601


class BGCPolicyModelError(ValueError):
    """A card-set observation or model input violates the v2 contract."""


def derive_policy_initialization(
    run_root_seed: str, model_id: str, initialization_ordinal: int
) -> tuple[bytes, int]:
    if (
        not isinstance(run_root_seed, str)
        or not run_root_seed
        or not isinstance(model_id, str)
        or not model_id
        or type(initialization_ordinal) is not int
        or initialization_ordinal < 0
    ):
        raise BGCPolicyModelError("policy initialization identity is invalid")
    digest = derive_seed(
        INITIALIZATION_NAMESPACE,
        run_root_seed,
        model_id,
        str(initialization_ordinal),
    )
    return digest, int.from_bytes(digest[:8], "big", signed=False)


def compact_observation_from_legacy(observation: Tensor) -> Tensor:
    """Drop the redundant stable-slot field after proving exact membership."""

    if not isinstance(observation, Tensor) or observation.dtype is not torch.bool:
        raise BGCPolicyModelError("legacy observation must have Boolean dtype")
    single = observation.ndim == 1
    batch = observation.unsqueeze(0) if single else observation
    if batch.ndim != 2 or batch.shape[0] < 1 or batch.shape[1] != LEGACY_OBSERVATION_SIZE:
        raise BGCPolicyModelError("legacy observation must have shape [875] or [B,875]")
    hand = batch[:, :LEGACY_HAND_BITS].reshape(-1, HAND_CANDIDATE_COUNT, CARD_COUNT)
    if torch.any(hand.sum(dim=1) > 1) or torch.any(hand.sum(dim=2) > 1):
        raise BGCPolicyModelError("legacy hand slots are not unique one-hot cards")
    compact = batch[:, LEGACY_HAND_BITS:].clone()
    in_hand = compact[:, IN_HAND_START : IN_HAND_START + CARD_COUNT]
    if not torch.equal(hand.any(dim=1), in_hand):
        raise BGCPolicyModelError("legacy hand slots differ from in-hand membership")
    return compact.squeeze(0) if single else compact


def candidate_card_indices(observation: Tensor) -> tuple[Tensor, Tensor]:
    """Return fixed-card-order candidates and their padding mask."""

    if not isinstance(observation, Tensor) or observation.dtype is not torch.bool:
        raise BGCPolicyModelError("observation must have Boolean dtype")
    single = observation.ndim == 1
    batch = observation.unsqueeze(0) if single else observation
    if batch.ndim != 2 or batch.shape[0] < 1 or batch.shape[1] != OBSERVATION_SIZE:
        raise BGCPolicyModelError("observation must have shape [659] or [B,659]")
    in_hand = batch[:, IN_HAND_START : IN_HAND_START + CARD_COUNT]
    counts = in_hand.sum(dim=1)
    if torch.any(counts < 1) or torch.any(counts > HAND_CANDIDATE_COUNT):
        raise BGCPolicyModelError("active observations require one to four hand cards")
    card_ids = torch.arange(CARD_COUNT, device=batch.device).expand(batch.shape[0], -1)
    ordered = torch.where(in_hand, card_ids, torch.full_like(card_ids, CARD_COUNT))
    candidates = torch.sort(ordered, dim=1, stable=True).values[:, :HAND_CANDIDATE_COUNT]
    present = candidates < CARD_COUNT
    if single:
        return candidates.squeeze(0), present.squeeze(0)
    return candidates, present


def compact_action_tensor_from_legacy(observation: Tensor, values: Tensor) -> Tensor:
    """Relabel stable engine-slot rows as compact current-card rows."""

    if observation.dtype is not torch.bool:
        raise BGCPolicyModelError("legacy observation must have Boolean dtype")
    single = observation.ndim == 1
    observations = observation.unsqueeze(0) if single else observation
    value_batch = values.unsqueeze(0) if single else values
    if (
        observations.ndim != 2
        or observations.shape[1] != LEGACY_OBSERVATION_SIZE
        or value_batch.ndim != 3
        or value_batch.shape != (observations.shape[0], HAND_CANDIDATE_COUNT, POLICY_POSITION_COUNT)
    ):
        raise BGCPolicyModelError("legacy action tensor shape differs")
    compact = compact_observation_from_legacy(observations)
    candidates, candidate_present = candidate_card_indices(compact)
    hand = observations[:, :LEGACY_HAND_BITS].reshape(-1, HAND_CANDIDATE_COUNT, CARD_COUNT)
    slot_present = hand.any(dim=2)
    slot_cards = hand.to(torch.int64).argmax(dim=2)
    matches = (
        slot_cards[:, :, None] == candidates[:, None, :]
    ) & slot_present[:, :, None] & candidate_present[:, None, :]
    if not torch.equal(matches.sum(dim=1), candidate_present.to(torch.int64)):
        raise BGCPolicyModelError("legacy slots cannot map uniquely to card candidates")
    result = torch.zeros_like(value_batch)
    for candidate_row in range(HAND_CANDIDATE_COUNT):
        source_rows = matches[:, :, candidate_row].to(torch.int64).argmax(dim=1)
        gathered = value_batch.gather(
            1,
            source_rows[:, None, None].expand(-1, 1, POLICY_POSITION_COUNT),
        ).squeeze(1)
        result[:, candidate_row] = torch.where(
            candidate_present[:, candidate_row, None], gathered, torch.zeros_like(gathered)
        )
    return result.squeeze(0) if single else result


def legacy_action_index_from_compact(observation: Tensor, compact_action_index: int) -> int:
    """Map one compact card-row action back to the engine's stable slot row."""

    if observation.ndim != 1 or observation.shape != (LEGACY_OBSERVATION_SIZE,):
        raise BGCPolicyModelError("legacy action mapping requires one [875] observation")
    if type(compact_action_index) is not int or not 0 <= compact_action_index < ACTION_COUNT:
        raise BGCPolicyModelError("compact action index is invalid")
    candidate_row, destination = divmod(compact_action_index, POLICY_POSITION_COUNT)
    compact = compact_observation_from_legacy(observation)
    candidates, present = candidate_card_indices(compact)
    if not bool(present[candidate_row].item()):
        raise BGCPolicyModelError("compact action names a padded card row")
    card_id = int(candidates[candidate_row].item())
    hand = observation[:LEGACY_HAND_BITS].reshape(HAND_CANDIDATE_COUNT, CARD_COUNT)
    matches = torch.nonzero(hand[:, card_id], as_tuple=False).flatten()
    if len(matches) != 1:
        raise BGCPolicyModelError("compact card does not identify one engine slot")
    return int(matches[0].item()) * POLICY_POSITION_COUNT + destination


def compact_action_index_from_legacy(observation: Tensor, legacy_action_index: int) -> int:
    """Map one engine stable-slot action to its compact current-card row."""

    if observation.ndim != 1 or observation.shape != (LEGACY_OBSERVATION_SIZE,):
        raise BGCPolicyModelError("compact action mapping requires one [875] observation")
    if type(legacy_action_index) is not int or not 0 <= legacy_action_index < ACTION_COUNT:
        raise BGCPolicyModelError("legacy action index is invalid")
    slot, destination = divmod(legacy_action_index, POLICY_POSITION_COUNT)
    hand = observation[:LEGACY_HAND_BITS].reshape(HAND_CANDIDATE_COUNT, CARD_COUNT)
    card_matches = torch.nonzero(hand[slot], as_tuple=False).flatten()
    if len(card_matches) != 1:
        raise BGCPolicyModelError("legacy action names an empty hand slot")
    candidates, present = candidate_card_indices(compact_observation_from_legacy(observation))
    rows = torch.nonzero(
        present & candidates.eq(int(card_matches[0].item())), as_tuple=False
    ).flatten()
    if len(rows) != 1:
        raise BGCPolicyModelError("legacy card does not identify one compact row")
    return int(rows[0].item()) * POLICY_POSITION_COUNT + destination


def _initialize(module: nn.Module, seed: int) -> None:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    with torch.no_grad():
        for child in module.modules():
            if isinstance(child, nn.Linear):
                nn.init.xavier_uniform_(child.weight, generator=generator)
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


class BGCPolicyModel(nn.Module):
    """Score four current-card candidates against eight destinations."""

    parameter_count = PARAMETER_COUNT

    def __init__(self, *, run_root_seed: str, model_id: str, initialization_ordinal: int) -> None:
        super().__init__()
        digest, seed = derive_policy_initialization(
            run_root_seed, model_id, initialization_ordinal
        )
        global_state = torch.random.get_rng_state()
        try:
            self.card_embedding = nn.Embedding(CARD_COUNT, 32)
            self.coffin_position_embedding = nn.Embedding(COFFIN_POSITION_COUNT, 8)
            self.card_encoder = nn.Linear(32, 64)
            self.hand_set_encoder = nn.Linear(64, 256)
            self.coffin_encoder = nn.Linear(41, 64)
            self.status_encoder = nn.Linear(STATUS_FEATURES, 96)
            self.context_encoder = nn.Linear(CONTEXT_FEATURES, 16)
            self.shared_input = nn.Linear(944, 512)
            self.shared_normalization = nn.LayerNorm(512, eps=1e-5)
            self.shared_output = nn.Linear(512, 256)
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
            torch.random.set_rng_state(global_state)
        self.run_root_seed = run_root_seed
        self.model_id = model_id
        self.initialization_ordinal = initialization_ordinal
        self.initialization_seed_digest = digest.hex()
        if sum(parameter.numel() for parameter in self.parameters()) != PARAMETER_COUNT:
            raise RuntimeError("BGC card policy parameter count differs")

    def forward(self, observation: Tensor) -> Tensor:
        single = observation.ndim == 1
        batch = observation.unsqueeze(0) if single else observation
        candidates, present = candidate_card_indices(batch)
        safe_candidates = candidates.clamp_max(CARD_COUNT - 1)
        candidate_embeddings = self.card_embedding(safe_candidates)
        candidate_features = self.activation(self.card_encoder(candidate_embeddings))
        candidate_features = candidate_features * present.unsqueeze(-1).to(candidate_features.dtype)
        hand_summary = self.activation(self.hand_set_encoder(candidate_features.sum(dim=1)))

        coffin = batch[:, COFFIN_START:STATUS_START].reshape(-1, COFFIN_POSITION_COUNT, CARD_COUNT)
        coffin_cards = coffin.to(candidate_features.dtype) @ self.card_embedding.weight
        positions = self.coffin_position_embedding.weight.unsqueeze(0).expand(batch.shape[0], -1, -1)
        occupied = coffin.any(dim=2, keepdim=True).to(candidate_features.dtype)
        coffin_features = self.activation(
            self.coffin_encoder(torch.cat((coffin_cards, positions, occupied), dim=2))
        )
        statuses = batch[:, STATUS_START:CONTEXT_START].to(candidate_features.dtype)
        context = batch[:, CONTEXT_START:].to(candidate_features.dtype)
        status_features = self.activation(self.status_encoder(statuses))
        context_features = self.activation(self.context_encoder(context))
        fused = torch.cat(
            (hand_summary, coffin_features.flatten(start_dim=1), status_features, context_features),
            dim=1,
        )
        shared = self.activation(self.shared_input(fused))
        shared = self.shared_normalization(shared)
        shared = self.activation(self.shared_output(shared))
        destinations = coffin_features.index_select(1, self.policy_grid_indices)
        pairs = torch.cat(
            (
                candidate_features[:, :, None, :].expand(-1, -1, POLICY_POSITION_COUNT, -1),
                destinations[:, None, :, :].expand(-1, HAND_CANDIDATE_COUNT, -1, -1),
                shared[:, None, None, :].expand(-1, HAND_CANDIDATE_COUNT, POLICY_POSITION_COUNT, -1),
            ),
            dim=3,
        )
        logits = self.pair_output(
            self.pair_normalization(self.activation(self.pair_hidden(pairs)))
        ).squeeze(-1)
        return logits.squeeze(0) if single else logits


__all__ = (
    "ACTION_COUNT",
    "ACTION_SCHEMA_VERSION",
    "BGCPolicyModel",
    "BGCPolicyModelError",
    "HAND_CANDIDATE_COUNT",
    "INITIALIZATION_SCHEMA_VERSION",
    "MODEL_SCHEMA_VERSION",
    "OBSERVATION_SCHEMA_VERSION",
    "OBSERVATION_SIZE",
    "PARAMETER_COUNT",
    "POLICY_GRID_INDICES",
    "POLICY_POSITION_COUNT",
    "candidate_card_indices",
    "compact_action_tensor_from_legacy",
    "compact_action_index_from_legacy",
    "compact_observation_from_legacy",
    "derive_policy_initialization",
    "legacy_action_index_from_compact",
)

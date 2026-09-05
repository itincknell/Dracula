"""Define the permutation-invariant standalone BGC policy network.

The model reads a player-visible 659-bit observation, derives the current cards
without permanent hand-slot identities, and returns raw scores for 4×8 actions.
Legality and strategic-symmetry masking remain outside the network.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from dracula.bridge import (
    COFFIN_POSITION_COUNT,
    HAND_SLOT_COUNT,
    POLICY_GRID_INDICES,
    POLICY_POSITION_COUNT,
)
from dracula.cards import CARD_COUNT

HAND_CANDIDATE_COUNT = HAND_SLOT_COUNT
COFFIN_FEATURES = COFFIN_POSITION_COUNT * CARD_COUNT
STATUS_FEATURES = 3 * CARD_COUNT
CONTEXT_FEATURES = 11
OBSERVATION_SIZE = COFFIN_FEATURES + STATUS_FEATURES + CONTEXT_FEATURES
# The compact 659-bit input is 9×54 coffin occupancy, 3×54 card status, and
# eleven lifecycle bits. Current hand membership is one status row.
COFFIN_START = 0
STATUS_START = COFFIN_FEATURES
CONTEXT_START = STATUS_START + STATUS_FEATURES
IN_HAND_START = STATUS_START + CARD_COUNT

CARD_EMBEDDING_WIDTH = 32
POSITION_EMBEDDING_WIDTH = 8
CARD_FEATURE_WIDTH = 64
HAND_SUMMARY_WIDTH = 256
COFFIN_INPUT_WIDTH = CARD_EMBEDDING_WIDTH + POSITION_EMBEDDING_WIDTH + 1
COFFIN_FEATURE_WIDTH = 64
STATUS_FEATURE_WIDTH = 96
CONTEXT_FEATURE_WIDTH = 16
SHARED_INPUT_WIDTH = (
    HAND_SUMMARY_WIDTH
    + COFFIN_POSITION_COUNT * COFFIN_FEATURE_WIDTH
    + STATUS_FEATURE_WIDTH
    + CONTEXT_FEATURE_WIDTH
)
SHARED_HIDDEN_WIDTH = 512
SHARED_STATE_WIDTH = 256
PAIR_INPUT_WIDTH = CARD_FEATURE_WIDTH + COFFIN_FEATURE_WIDTH + SHARED_STATE_WIDTH
PAIR_HIDDEN_WIDTH = 256
PARAMETER_COUNT = 754_601


class BGCPolicyModelError(ValueError):
    """A card-set observation or model input violates the policy contract."""


def pack_hand_card_indices(observation: Tensor) -> tuple[Tensor, Tensor]:
    """Extract the current hand as up to four canonical model rows.

    The observation records hand membership as 54 Boolean card-ID positions.
    This returns those card indexes in ascending project order. Unused rows
    contain the sentinel ``CARD_COUNT`` and are false in the returned mask.
    """

    if not isinstance(observation, Tensor):
        raise BGCPolicyModelError("observation must be a tensor")
    if observation.dtype is not torch.bool:
        raise BGCPolicyModelError("observation must have Boolean dtype")
    single = observation.ndim == 1
    batch = observation.unsqueeze(0) if single else observation
    if batch.ndim != 2 or batch.shape[0] < 1 or batch.shape[1] != OBSERVATION_SIZE:
        raise BGCPolicyModelError("observation must have shape [659] or [B,659]")

    in_hand = batch[:, IN_HAND_START : IN_HAND_START + CARD_COUNT]
    cards_per_hand = in_hand.sum(dim=1)
    if torch.any(cards_per_hand < 1) or torch.any(
        cards_per_hand > HAND_CANDIDATE_COUNT
    ):
        raise BGCPolicyModelError("active observations require one to four hand cards")

    card_indexes = torch.arange(CARD_COUNT, device=batch.device)
    padding = torch.full_like(card_indexes, CARD_COUNT)
    # Replacing absent cards with 54 turns a hand such as cards 2 and 20 into
    # [2, 20, 54, 54] after sorting and taking the first four entries.
    padded_card_indexes = torch.where(in_hand, card_indexes, padding)
    candidates = torch.sort(padded_card_indexes, dim=1, stable=True).values[
        :, :HAND_CANDIDATE_COUNT
    ]
    candidate_is_present = candidates < CARD_COUNT
    if single:
        return candidates.squeeze(0), candidate_is_present.squeeze(0)
    return candidates, candidate_is_present


def _initialize(module: nn.Module, seed: int) -> None:
    """Initialize every learned layer from one model-local CPU generator."""

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

    def __init__(self, seed: int = 0) -> None:
        super().__init__()
        # Module constructors consume PyTorch's global RNG before explicit
        # initialization, so preserve it across the complete construction.
        global_state = torch.random.get_rng_state()
        try:
            self.card_embedding = nn.Embedding(CARD_COUNT, CARD_EMBEDDING_WIDTH)
            self.coffin_position_embedding = nn.Embedding(
                COFFIN_POSITION_COUNT,
                POSITION_EMBEDDING_WIDTH,
            )
            self.card_encoder = nn.Linear(
                CARD_EMBEDDING_WIDTH,
                CARD_FEATURE_WIDTH,
            )
            self.hand_set_encoder = nn.Linear(
                CARD_FEATURE_WIDTH,
                HAND_SUMMARY_WIDTH,
            )
            self.coffin_encoder = nn.Linear(
                COFFIN_INPUT_WIDTH,
                COFFIN_FEATURE_WIDTH,
            )
            self.status_encoder = nn.Linear(
                STATUS_FEATURES,
                STATUS_FEATURE_WIDTH,
            )
            self.context_encoder = nn.Linear(
                CONTEXT_FEATURES,
                CONTEXT_FEATURE_WIDTH,
            )
            self.shared_input = nn.Linear(
                SHARED_INPUT_WIDTH,
                SHARED_HIDDEN_WIDTH,
            )
            self.shared_normalization = nn.LayerNorm(
                SHARED_HIDDEN_WIDTH,
                eps=1e-5,
            )
            self.shared_output = nn.Linear(
                SHARED_HIDDEN_WIDTH,
                SHARED_STATE_WIDTH,
            )
            self.pair_hidden = nn.Linear(PAIR_INPUT_WIDTH, PAIR_HIDDEN_WIDTH)
            self.pair_normalization = nn.LayerNorm(PAIR_HIDDEN_WIDTH, eps=1e-5)
            self.pair_output = nn.Linear(PAIR_HIDDEN_WIDTH, 1)
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

    def _encode_hand(self, batch: Tensor) -> tuple[Tensor, Tensor]:
        """Encode each current card and an order-independent hand summary."""

        candidates, present = pack_hand_card_indices(batch)
        safe_candidates = candidates.clamp_max(CARD_COUNT - 1)
        candidate_embeddings = self.card_embedding(safe_candidates)
        candidate_features = self.activation(self.card_encoder(candidate_embeddings))
        candidate_features = candidate_features * present.unsqueeze(-1).to(
            candidate_features.dtype
        )
        # Sum pooling removes candidate-row order from the shared hand feature.
        hand_summary = self.activation(
            self.hand_set_encoder(candidate_features.sum(dim=1))
        )
        return candidate_features, hand_summary

    def _encode_coffin(self, batch: Tensor, dtype: torch.dtype) -> Tensor:
        """Encode the card, location, and occupancy of all nine positions."""

        coffin = batch[:, COFFIN_START:STATUS_START].reshape(
            -1, COFFIN_POSITION_COUNT, CARD_COUNT
        )
        # Multiplying the one-hot rows by the embedding table retrieves each
        # position's card embedding; empty positions produce an all-zero vector.
        coffin_cards = coffin.to(dtype) @ self.card_embedding.weight
        positions = self.coffin_position_embedding.weight.unsqueeze(0).expand(
            batch.shape[0], -1, -1
        )
        occupied = coffin.any(dim=2, keepdim=True).to(dtype)
        return self.activation(
            self.coffin_encoder(torch.cat((coffin_cards, positions, occupied), dim=2))
        )

    def _encode_shared_state(
        self,
        batch: Tensor,
        hand_summary: Tensor,
        coffin_features: Tensor,
    ) -> Tensor:
        """Combine the hand, coffin, card statuses, and lifecycle context."""

        dtype = hand_summary.dtype
        statuses = batch[:, STATUS_START:CONTEXT_START].to(dtype)
        context = batch[:, CONTEXT_START:].to(dtype)
        status_features = self.activation(self.status_encoder(statuses))
        context_features = self.activation(self.context_encoder(context))
        fused = torch.cat(
            (
                hand_summary,
                coffin_features.flatten(start_dim=1),
                status_features,
                context_features,
            ),
            dim=1,
        )
        shared = self.activation(self.shared_input(fused))
        shared = self.shared_normalization(shared)
        return self.activation(self.shared_output(shared))

    def _score_pairs(
        self,
        candidate_features: Tensor,
        coffin_features: Tensor,
        shared: Tensor,
    ) -> Tensor:
        """Score every current-card and non-center destination feature pair."""

        destinations = coffin_features.index_select(1, self.policy_grid_indices)
        pairs = torch.cat(
            (
                candidate_features[:, :, None, :].expand(
                    -1, -1, POLICY_POSITION_COUNT, -1
                ),
                destinations[:, None, :, :].expand(
                    -1, HAND_CANDIDATE_COUNT, -1, -1
                ),
                shared[:, None, None, :].expand(
                    -1, HAND_CANDIDATE_COUNT, POLICY_POSITION_COUNT, -1
                ),
            ),
            dim=3,
        )
        return self.pair_output(
            self.pair_normalization(self.activation(self.pair_hidden(pairs)))
        ).squeeze(-1)

    def forward(self, observation: Tensor) -> Tensor:
        """Return raw 4×8 candidate-card/destination logits."""

        single = observation.ndim == 1
        batch = observation.unsqueeze(0) if single else observation
        candidate_features, hand_summary = self._encode_hand(batch)
        coffin_features = self._encode_coffin(batch, candidate_features.dtype)
        shared = self._encode_shared_state(batch, hand_summary, coffin_features)
        logits = self._score_pairs(candidate_features, coffin_features, shared)
        return logits.squeeze(0) if single else logits


__all__ = (
    "BGCPolicyModel",
    "BGCPolicyModelError",
    "HAND_CANDIDATE_COUNT",
    "OBSERVATION_SIZE",
    "PARAMETER_COUNT",
    "POLICY_GRID_INDICES",
    "POLICY_POSITION_COUNT",
    "pack_hand_card_indices",
)

"""Define the permutation-invariant standalone pi1 policy network.

The model reads a player-visible 659-bit observation, derives the current cards
without permanent hand-slot identities, and returns raw scores for 4×8 actions.
Legality and strategic-symmetry masking remain outside the network.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from dracula.decision.bridge import (
    COFFIN_POSITION_COUNT,
    HAND_SLOT_COUNT,
    POLICY_GRID_INDICES,
    POLICY_POSITION_COUNT,
)
from dracula.game.cards import CARD_COUNT

# Observation layout. The coffin stores one 54-card occupancy row for each of
# its nine positions; the status block stores three 54-card sets; the final
# eleven bits describe the current round and player-relative context.
COFFIN_FEATURES = COFFIN_POSITION_COUNT * CARD_COUNT
STATUS_FEATURES = 3 * CARD_COUNT
CONTEXT_FEATURES = 11
OBSERVATION_SIZE = COFFIN_FEATURES + STATUS_FEATURES + CONTEXT_FEATURES
COFFIN_START = 0
STATUS_START = COFFIN_FEATURES
CONTEXT_START = STATUS_START + STATUS_FEATURES
IN_HAND_START = STATUS_START + CARD_COUNT

# At most four cards remain in hand. They are packed into temporary candidate
# rows for scoring; the rows themselves have no learned identity.
HAND_CANDIDATE_COUNT = HAND_SLOT_COUNT

# A shared embedding identifies each card wherever it appears. Coffin positions
# also need a small learned location embedding because their geometry matters.
CARD_EMBEDDING_WIDTH = 32
POSITION_EMBEDDING_WIDTH = 8

# Each current card is expanded into the feature vector used both by the
# card/destination scorer and by the pooled representation of the whole hand.
CARD_FEATURE_WIDTH = 64
HAND_SUMMARY_WIDTH = 256

# A coffin-position encoder receives its card embedding, position embedding,
# and one occupancy bit, then emits one feature vector per grid position.
COFFIN_INPUT_WIDTH = CARD_EMBEDDING_WIDTH + POSITION_EMBEDDING_WIDTH + 1
COFFIN_FEATURE_WIDTH = 64

# The remaining observation blocks are encoded separately before fusion.
STATUS_FEATURE_WIDTH = 96
CONTEXT_FEATURE_WIDTH = 16

# The shared body sees the pooled hand, all nine encoded coffin positions,
# global card status, and lifecycle context. It compresses that complete view
# into the state feature supplied to every candidate action.
SHARED_INPUT_WIDTH = (
    HAND_SUMMARY_WIDTH
    + COFFIN_POSITION_COUNT * COFFIN_FEATURE_WIDTH
    + STATUS_FEATURE_WIDTH
    + CONTEXT_FEATURE_WIDTH
)
SHARED_HIDDEN_WIDTH = 512
SHARED_STATE_WIDTH = 256

# The same head scores each card/destination pair from the candidate card, its
# destination position, and the shared state. This produces the raw 4×8 logits.
PAIR_INPUT_WIDTH = CARD_FEATURE_WIDTH + COFFIN_FEATURE_WIDTH + SHARED_STATE_WIDTH
PAIR_HIDDEN_WIDTH = 256

# Tests and artifact loading enforce this total so architecture drift cannot
# silently make the selected weights incompatible with the runtime model.
PARAMETER_COUNT = 754_601


class PolicyModelError(ValueError):
    """A card-set observation or model input violates the policy contract."""


def _observation_batch(observation: Tensor) -> tuple[Tensor, bool]:
    """Validate one model input and expose a uniform batched view."""

    if not isinstance(observation, Tensor):
        raise PolicyModelError("observation must be a tensor")
    if observation.dtype is not torch.bool:
        raise PolicyModelError("observation must have Boolean dtype")
    single = observation.ndim == 1
    batch = observation.unsqueeze(0) if single else observation
    if batch.ndim != 2 or batch.shape[0] < 1 or batch.shape[1] != OBSERVATION_SIZE:
        raise PolicyModelError("observation must have shape [659] or [B,659]")
    return batch, single


def pack_hand_card_indices(observation: Tensor) -> tuple[Tensor, Tensor]:
    """Map the cards still in hand into the model's four temporary card rows.

    One 54-bit observation block marks which card IDs are currently in hand.
    This function extracts those IDs in ascending project card order and puts
    them in candidate rows zero through three. For example, hand-card IDs 2
    and 20 become candidate rows ``[2, 20, empty, empty]``.

    The first returned tensor contains the card ID assigned to each candidate
    row; an empty row contains ``CARD_COUNT`` (54). The second tensor has the
    same shape and identifies which rows contain cards. A single observation
    returns two ``[4]`` tensors; a batch returns two ``[batch, 4]`` tensors.
    """

    batch, single = _observation_batch(observation)
    # This observation slice has one Boolean column per canonical card ID.
    hand_membership = batch[:, IN_HAND_START : IN_HAND_START + CARD_COUNT]
    cards_per_hand = hand_membership.sum(dim=1)
    if torch.any(cards_per_hand < 1) or torch.any(
        cards_per_hand > HAND_CANDIDATE_COUNT
    ):
        raise PolicyModelError("active observations require one to four hand cards")

    # These are the 54 valid card IDs: [0, 1, ..., 53].
    all_card_ids = torch.arange(CARD_COUNT, device=batch.device)

    # The out-of-range ID 54 marks an absent card. Matching the first tensor's
    # shape, dtype, and device lets torch.where replace absent entries directly.
    absent_card_markers = torch.full_like(all_card_ids, CARD_COUNT)

    # Keep each present card's ID and replace every absent card with 54. Sorting
    # moves the one-to-four real IDs to the front and all empty markers behind.
    marked_card_ids = torch.where(
        hand_membership,     # Boolean tensor
        all_card_ids,        # Assign card_id in [0:53] when true
        absent_card_markers, # Assign 54 when false
    )
    # Sort the 54 entries within each observation. Valid card IDs come first
    # because all of them are smaller than the absent-card marker 54.
    sorted_card_ids = torch.sort(marked_card_ids, dim=1, stable=True).values

    # A hand contains at most four cards, so the later columns can be discarded.
    candidate_card_ids = sorted_card_ids[:, :HAND_CANDIDATE_COUNT]
    candidate_is_present = candidate_card_ids < CARD_COUNT

    if single:
        return candidate_card_ids.squeeze(0), candidate_is_present.squeeze(0)
    return candidate_card_ids, candidate_is_present


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


class PolicyModel(nn.Module):
    """Score four current-card candidates against eight destinations."""

    parameter_count = PARAMETER_COUNT

    def __init__(self, seed: int = 0) -> None:
        super().__init__()
        self._build_input_encoders()
        self._build_shared_body()
        self._build_pair_scorer()
        self.activation = nn.GELU(approximate="none")
        self.register_buffer(
            "policy_grid_indices",
            torch.tensor(POLICY_GRID_INDICES, dtype=torch.long),
            persistent=False,
        )
        self.float()
        _initialize(self, seed)

    def _build_input_encoders(self) -> None:
        """Register card, hand-set, coffin, status, and context encoders."""

        self.card_embedding = nn.Embedding(CARD_COUNT, CARD_EMBEDDING_WIDTH)
        self.coffin_position_embedding = nn.Embedding(
            COFFIN_POSITION_COUNT,
            POSITION_EMBEDDING_WIDTH,
        )
        self.card_encoder = nn.Linear(CARD_EMBEDDING_WIDTH, CARD_FEATURE_WIDTH)
        self.hand_set_encoder = nn.Linear(CARD_FEATURE_WIDTH, HAND_SUMMARY_WIDTH)
        self.coffin_encoder = nn.Linear(COFFIN_INPUT_WIDTH, COFFIN_FEATURE_WIDTH)
        self.status_encoder = nn.Linear(STATUS_FEATURES, STATUS_FEATURE_WIDTH)
        self.context_encoder = nn.Linear(CONTEXT_FEATURES, CONTEXT_FEATURE_WIDTH)

    def _build_shared_body(self) -> None:
        """Register the layers that combine all encoded state features."""

        self.shared_input = nn.Linear(SHARED_INPUT_WIDTH, SHARED_HIDDEN_WIDTH)
        self.shared_normalization = nn.LayerNorm(SHARED_HIDDEN_WIDTH, eps=1e-5)
        self.shared_output = nn.Linear(SHARED_HIDDEN_WIDTH, SHARED_STATE_WIDTH)

    def _build_pair_scorer(self) -> None:
        """Register the head shared by every card-and-destination pair."""

        self.pair_hidden = nn.Linear(PAIR_INPUT_WIDTH, PAIR_HIDDEN_WIDTH)
        self.pair_normalization = nn.LayerNorm(PAIR_HIDDEN_WIDTH, eps=1e-5)
        self.pair_output = nn.Linear(PAIR_HIDDEN_WIDTH, 1)

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
        """Return one logit for every current-card and destination pairing.

        Candidate cards have shape ``[batch, 4, 64]``, coffin positions have
        shape ``[batch, 9, 64]``, and the shared visible-state feature has shape
        ``[batch, 256]``. Each output cell combines one exact card, one
        non-center destination, and that shared state. The result has shape
        ``[batch, 4, 8]``.
        """

        # The policy has eight destination columns
        destinations = coffin_features.index_select(1, self.policy_grid_indices)

        # Repeat each candidate card across all eight destinations and each
        # destination across all four card rows, producing the 4x8 action grid.
        card_for_each_destination = candidate_features[:, :, None, :].expand(
            -1,
            -1,
            POLICY_POSITION_COUNT,
            -1,
        )
        destination_for_each_card = destinations[:, None, :, :].expand(
            -1,
            HAND_CANDIDATE_COUNT,
            -1,
            -1,
        )

        # Every action candidate receives the same summary of the complete
        # player-visible state from which that action is being evaluated.
        state_for_each_pair = shared[:, None, None, :].expand(
            -1,
            HAND_CANDIDATE_COUNT,
            POLICY_POSITION_COUNT,
            -1,
        )

        # Each scorer input contains 64 card features, 64 destination features,
        # and 256 shared-state features: 384 values in total.
        pairs = torch.cat(
            (
                card_for_each_destination,
                destination_for_each_card,
                state_for_each_pair,
            ),
            dim=3,
        )

        pair_features = self.activation(self.pair_hidden(pairs))
        normalized_pairs = self.pair_normalization(pair_features)
        logits = self.pair_output(normalized_pairs)
        return logits.squeeze(-1)

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
    "PolicyModel",
    "PolicyModelError",
    "HAND_CANDIDATE_COUNT",
    "OBSERVATION_SIZE",
    "PARAMETER_COUNT",
    "POLICY_GRID_INDICES",
    "POLICY_POSITION_COUNT",
    "pack_hand_card_indices",
)

"""Version 1 PyTorch policy and critic defined by the model design."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from dracula.cards import CARD_COUNT

HAND_SLOT_COUNT = 4
COFFIN_POSITION_COUNT = 9
POLICY_POSITION_COUNT = 8
OBSERVATION_SIZE = 875
HIDDEN_SIZE = 128

HAND_FEATURES = HAND_SLOT_COUNT * CARD_COUNT
COFFIN_FEATURES = COFFIN_POSITION_COUNT * CARD_COUNT
STATUS_FEATURES = 3 * CARD_COUNT
CONTEXT_FEATURES = 11

HAND_START = 0
COFFIN_START = HAND_START + HAND_FEATURES
STATUS_START = COFFIN_START + COFFIN_FEATURES
CONTEXT_START = STATUS_START + STATUS_FEATURES

POLICY_GRID_INDICES = (0, 1, 2, 3, 5, 6, 7, 8)


@dataclass(frozen=True)
class _StructuredObservation:
    hand: Tensor
    coffin: Tensor
    statuses: Tensor
    context: Tensor


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


def _initialize_module(module: nn.Module, seed: int) -> None:
    """Apply the documented initialization without retaining global RNG changes."""

    rng_state = torch.random.get_rng_state()
    try:
        torch.manual_seed(seed)
        with torch.no_grad():
            for child in module.modules():
                if isinstance(child, nn.Linear):
                    nn.init.xavier_uniform_(child.weight)
                    nn.init.zeros_(child.bias)
                elif isinstance(child, nn.Embedding):
                    nn.init.normal_(
                        child.weight,
                        mean=0.0,
                        std=1.0 / math.sqrt(child.embedding_dim),
                    )
                elif isinstance(child, nn.LayerNorm):
                    nn.init.ones_(child.weight)
                    nn.init.zeros_(child.bias)

            for child in module.modules():
                if isinstance(child, nn.GRUCell):
                    nn.init.xavier_uniform_(child.weight_ih)
                    for gate in child.weight_hh.chunk(3, dim=0):
                        nn.init.orthogonal_(gate)
                    nn.init.zeros_(child.bias_ih)
                    nn.init.zeros_(child.bias_hh)
    finally:
        torch.random.set_rng_state(rng_state)


def _model_device(module: nn.Module) -> torch.device:
    return next(module.parameters()).device


def _validate_observation(observation: Tensor) -> bool:
    if observation.dtype is not torch.bool:
        raise TypeError("observation must have Boolean dtype")
    if observation.ndim == 1:
        if observation.shape != (OBSERVATION_SIZE,):
            raise ValueError(f"single observation must have shape ({OBSERVATION_SIZE},)")
        return True
    if observation.ndim == 2:
        if observation.shape[0] == 0 or observation.shape[1] != OBSERVATION_SIZE:
            raise ValueError(f"batched observation must have shape (batch, {OBSERVATION_SIZE})")
        return False
    raise ValueError("observation must be a single state or a batch of states")


class _PolicyStructuredEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.card_embedding = nn.Embedding(CARD_COUNT, 32)
        self.hand_slot_embedding = nn.Embedding(HAND_SLOT_COUNT, 8)
        self.grid_position_embedding = nn.Embedding(COFFIN_POSITION_COUNT, 8)

        self.hand_encoder = nn.Linear(41, 64)
        self.coffin_encoder = nn.Linear(41, 64)
        self.status_encoder = nn.Linear(STATUS_FEATURES, 96)
        self.context_encoder = nn.Linear(CONTEXT_FEATURES, 16)

        self.fusion_input = nn.Linear(944, 256)
        self.fusion_normalization = nn.LayerNorm(256)
        self.fusion_output = nn.Linear(256, 128)
        self.activation = nn.GELU()

    def forward(self, observation: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        structured = _split_observation(observation)
        dtype = self.card_embedding.weight.dtype

        hand_cards = structured.hand.to(dtype) @ self.card_embedding.weight
        hand_slots = self.hand_slot_embedding.weight.unsqueeze(0).expand(
            observation.shape[0], -1, -1
        )
        hand_occupied = structured.hand.any(dim=-1, keepdim=True).to(dtype)
        hand_features = self.activation(
            self.hand_encoder(torch.cat((hand_cards, hand_slots, hand_occupied), dim=-1))
        )

        coffin_cards = structured.coffin.to(dtype) @ self.card_embedding.weight
        grid_positions = self.grid_position_embedding.weight.unsqueeze(0).expand(
            observation.shape[0], -1, -1
        )
        coffin_occupied = structured.coffin.any(dim=-1, keepdim=True).to(dtype)
        coffin_features = self.activation(
            self.coffin_encoder(
                torch.cat((coffin_cards, grid_positions, coffin_occupied), dim=-1)
            )
        )

        status_features = self.activation(self.status_encoder(structured.statuses.to(dtype)))
        context_features = self.activation(self.context_encoder(structured.context.to(dtype)))
        fused = torch.cat(
            (
                hand_features.flatten(start_dim=1),
                coffin_features.flatten(start_dim=1),
                status_features,
                context_features,
            ),
            dim=-1,
        )
        encoded = self.activation(self.fusion_input(fused))
        encoded = self.fusion_normalization(encoded)
        encoded = self.activation(self.fusion_output(encoded))
        return encoded, hand_features, coffin_features


class Policy(nn.Module):
    """Structured recurrent policy returning unmasked 4-by-8 action logits."""

    parameter_count = 443_145

    def __init__(self, *, seed: int) -> None:
        super().__init__()
        rng_state = torch.random.get_rng_state()
        try:
            self.encoder = _PolicyStructuredEncoder()
            self.recurrent = nn.GRUCell(160, HIDDEN_SIZE)
            self.pair_input = nn.Linear(257, 128)
            self.pair_normalization = nn.LayerNorm(128)
            self.pair_output = nn.Linear(128, 1)
            self.activation = nn.GELU()
            self.register_buffer(
                "policy_grid_indices",
                torch.tensor(POLICY_GRID_INDICES, dtype=torch.long),
                persistent=False,
            )
            _initialize_module(self, seed)
        finally:
            torch.random.set_rng_state(rng_state)

    def initial_hidden(
        self,
        batch_size: int | None = None,
        *,
        device: torch.device | str | None = None,
    ) -> Tensor:
        """Create the all-zero hidden state required at a game boundary."""

        if batch_size is not None and batch_size < 1:
            raise ValueError("batch_size must be positive")
        shape = (HIDDEN_SIZE,) if batch_size is None else (batch_size, HIDDEN_SIZE)
        parameter = next(self.parameters())
        return torch.zeros(
            shape,
            dtype=parameter.dtype,
            device=parameter.device if device is None else device,
        )

    def forward(
        self,
        observation: Tensor,
        legal_mask: Tensor,
        hidden_state: Tensor,
    ) -> tuple[Tensor, Tensor]:
        single = _validate_observation(observation)
        expected_mask_shape = (
            (HAND_SLOT_COUNT, POLICY_POSITION_COUNT)
            if single
            else (observation.shape[0], HAND_SLOT_COUNT, POLICY_POSITION_COUNT)
        )
        expected_hidden_shape = (
            (HIDDEN_SIZE,) if single else (observation.shape[0], HIDDEN_SIZE)
        )
        if legal_mask.dtype is not torch.bool:
            raise TypeError("legal_mask must have Boolean dtype")
        if legal_mask.shape != expected_mask_shape:
            raise ValueError(f"legal_mask must have shape {expected_mask_shape}")
        if hidden_state.dtype is not torch.float32:
            raise TypeError("hidden_state must have float32 dtype")
        if hidden_state.shape != expected_hidden_shape:
            raise ValueError(f"hidden_state must have shape {expected_hidden_shape}")
        if observation.device != legal_mask.device or observation.device != hidden_state.device:
            raise ValueError("observation, legal_mask, and hidden_state must share a device")
        if observation.device != _model_device(self):
            raise ValueError("model and inputs must share a device")
        if not torch.isfinite(hidden_state).all():
            raise ValueError("hidden_state must contain only finite values")

        observation_batch = observation.unsqueeze(0) if single else observation
        mask_batch = legal_mask.unsqueeze(0) if single else legal_mask
        hidden_batch = hidden_state.unsqueeze(0) if single else hidden_state
        if not mask_batch.flatten(start_dim=1).any(dim=1).all():
            raise ValueError("every policy state must have at least one legal action")

        encoded, hand_features, coffin_features = self.encoder(observation_batch)
        mask_values = mask_batch.to(encoded.dtype)
        recurrent_input = torch.cat((encoded, mask_values.flatten(start_dim=1)), dim=-1)
        next_hidden = self.recurrent(recurrent_input, hidden_batch)

        destination_features = coffin_features.index_select(1, self.policy_grid_indices)
        pair_features = torch.cat(
            (
                hand_features.unsqueeze(2).expand(-1, -1, POLICY_POSITION_COUNT, -1),
                destination_features.unsqueeze(1).expand(-1, HAND_SLOT_COUNT, -1, -1),
                next_hidden[:, None, None, :].expand(
                    -1, HAND_SLOT_COUNT, POLICY_POSITION_COUNT, -1
                ),
                mask_values.unsqueeze(-1),
            ),
            dim=-1,
        )
        pair_values = self.activation(self.pair_input(pair_features))
        pair_values = self.pair_normalization(pair_values)
        raw_logits = self.pair_output(pair_values).squeeze(-1)

        if single:
            return raw_logits.squeeze(0), next_hidden.squeeze(0)
        return raw_logits, next_hidden


class _CriticStructuredEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.card_embedding = nn.Embedding(CARD_COUNT, 32)
        self.hand_slot_embedding = nn.Embedding(HAND_SLOT_COUNT, 8)
        self.grid_position_embedding = nn.Embedding(COFFIN_POSITION_COUNT, 8)

        self.hand_encoder = nn.Linear(41, 64)
        self.coffin_encoder = nn.Linear(41, 64)
        self.status_encoder = nn.Linear(STATUS_FEATURES, 64)
        self.context_encoder = nn.Linear(CONTEXT_FEATURES, 16)

        self.fusion_input = nn.Linear(912, 128)
        self.fusion_normalization = nn.LayerNorm(128)
        self.fusion_output = nn.Linear(128, 64)
        self.value_head = nn.Linear(64, 1)
        self.activation = nn.GELU()

    def forward(self, observation: Tensor) -> Tensor:
        structured = _split_observation(observation)
        dtype = self.card_embedding.weight.dtype

        hand_cards = structured.hand.to(dtype) @ self.card_embedding.weight
        hand_slots = self.hand_slot_embedding.weight.unsqueeze(0).expand(
            observation.shape[0], -1, -1
        )
        hand_occupied = structured.hand.any(dim=-1, keepdim=True).to(dtype)
        hand_features = self.activation(
            self.hand_encoder(torch.cat((hand_cards, hand_slots, hand_occupied), dim=-1))
        )

        coffin_cards = structured.coffin.to(dtype) @ self.card_embedding.weight
        grid_positions = self.grid_position_embedding.weight.unsqueeze(0).expand(
            observation.shape[0], -1, -1
        )
        coffin_occupied = structured.coffin.any(dim=-1, keepdim=True).to(dtype)
        coffin_features = self.activation(
            self.coffin_encoder(
                torch.cat((coffin_cards, grid_positions, coffin_occupied), dim=-1)
            )
        )

        status_features = self.activation(self.status_encoder(structured.statuses.to(dtype)))
        context_features = self.activation(self.context_encoder(structured.context.to(dtype)))
        fused = torch.cat(
            (
                hand_features.flatten(start_dim=1),
                coffin_features.flatten(start_dim=1),
                status_features,
                context_features,
            ),
            dim=-1,
        )
        value_features = self.activation(self.fusion_input(fused))
        value_features = self.fusion_normalization(value_features)
        value_features = self.activation(self.fusion_output(value_features))
        return self.value_head(value_features).squeeze(-1)


class Critic(nn.Module):
    """Independent feed-forward estimate of normalized round return."""

    parameter_count = 143_273

    def __init__(self, *, seed: int) -> None:
        super().__init__()
        rng_state = torch.random.get_rng_state()
        try:
            self.encoder = _CriticStructuredEncoder()
            _initialize_module(self, seed)
        finally:
            torch.random.set_rng_state(rng_state)

    def forward(self, observation: Tensor) -> Tensor:
        single = _validate_observation(observation)
        if observation.device != _model_device(self):
            raise ValueError("model and observation must share a device")
        observation_batch = observation.unsqueeze(0) if single else observation
        values = self.encoder(observation_batch)
        return values.squeeze(0) if single else values

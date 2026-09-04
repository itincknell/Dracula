"""Direct player-visible observation and compact action mappings for pi1."""

from __future__ import annotations

import torch
from torch import Tensor

from dracula.bgc_policy_model import (
    ACTION_COUNT,
    COFFIN_POSITION_COUNT,
    COFFIN_START,
    CONTEXT_FEATURES,
    CONTEXT_START,
    HAND_CANDIDATE_COUNT,
    IN_HAND_START,
    OBSERVATION_SIZE,
    POLICY_POSITION_COUNT,
    STATUS_START,
)
from dracula.cards import CARD_COUNT, CARD_INDEX_BY_ID
from dracula.search.information import SearchInformationState

ROUND_COUNT = 6
PROGRESS_COUNT = 4
PLAYED_START = STATUS_START
UNSEEN_START = IN_HAND_START + CARD_COUNT
ROUND_START = CONTEXT_START
PROGRESS_START = ROUND_START + ROUND_COUNT
DEALER_INDEX = PROGRESS_START + PROGRESS_COUNT


class PolicyObservationError(ValueError):
    """A typed information state cannot map to the pi1 tensor contract."""


def encode_policy_observation(information: SearchInformationState) -> Tensor:
    """Encode the actor-visible state directly as the selected bool[659] layout."""

    if not isinstance(information, SearchInformationState):
        raise PolicyObservationError("pi1 requires a SearchInformationState")
    observation = torch.zeros(OBSERVATION_SIZE, dtype=torch.bool)
    for position, card_id in enumerate(information.coffin):
        if card_id is not None:
            observation[
                COFFIN_START + position * CARD_COUNT + CARD_INDEX_BY_ID[card_id]
            ] = True
    for card_id in information.played_card_ids:
        observation[PLAYED_START + CARD_INDEX_BY_ID[card_id]] = True
    for card_id in information.own_hand:
        if card_id is not None:
            observation[IN_HAND_START + CARD_INDEX_BY_ID[card_id]] = True
    for card_id in information.unseen_card_ids:
        observation[UNSEEN_START + CARD_INDEX_BY_ID[card_id]] = True
    own_decision_index = sum(
        move.player is information.player
        for move in information.current_round_moves
    )
    observation[ROUND_START + information.round_number - 1] = True
    observation[PROGRESS_START + own_decision_index] = True
    observation[DEALER_INDEX] = information.player is information.dealer
    if observation.shape != (OBSERVATION_SIZE,) or CONTEXT_FEATURES != 11:
        raise PolicyObservationError("pi1 observation layout differs")
    return observation


def _candidate_slots(information: SearchInformationState) -> tuple[int, ...]:
    """Map canonical current-card rows to the state's stable engine slots."""

    slots = tuple(
        slot for slot, card_id in enumerate(information.own_hand) if card_id is not None
    )
    if not 1 <= len(slots) <= HAND_CANDIDATE_COUNT:
        raise PolicyObservationError("active information has no current card")
    # Engine hands retain fixed card-ID order, so occupied slots already have
    # the exact canonical order used by candidate_card_indices.
    return slots


def candidate_action_tensor(
    information: SearchInformationState, values: Tensor
) -> Tensor:
    """Relabel engine-slot action rows as canonical current-card rows."""

    if not isinstance(values, Tensor) or values.shape != (
        HAND_CANDIDATE_COUNT,
        POLICY_POSITION_COUNT,
    ):
        raise PolicyObservationError("action tensor must have shape [4,8]")
    result = torch.zeros_like(values)
    for candidate_row, slot in enumerate(_candidate_slots(information)):
        result[candidate_row] = values[slot]
    return result


def engine_action_index_from_candidate(
    information: SearchInformationState, candidate_action_index: int
) -> int:
    """Map a compact candidate action back to its legal engine-slot index."""

    if (
        type(candidate_action_index) is not int
        or not 0 <= candidate_action_index < ACTION_COUNT
    ):
        raise PolicyObservationError("candidate action index is invalid")
    candidate_row, destination = divmod(
        candidate_action_index, POLICY_POSITION_COUNT
    )
    slots = _candidate_slots(information)
    if candidate_row >= len(slots):
        raise PolicyObservationError("candidate action names a padding row")
    return slots[candidate_row] * POLICY_POSITION_COUNT + destination


def candidate_action_index_from_engine(
    information: SearchInformationState, engine_action_index: int
) -> int:
    """Map an engine-slot action to the corresponding current-card row."""

    if type(engine_action_index) is not int or not 0 <= engine_action_index < ACTION_COUNT:
        raise PolicyObservationError("engine action index is invalid")
    slot, destination = divmod(engine_action_index, POLICY_POSITION_COUNT)
    try:
        candidate_row = _candidate_slots(information).index(slot)
    except ValueError as error:
        raise PolicyObservationError("engine action names an empty hand slot") from error
    return candidate_row * POLICY_POSITION_COUNT + destination


__all__ = (
    "PolicyObservationError",
    "candidate_action_index_from_engine",
    "candidate_action_tensor",
    "encode_policy_observation",
    "engine_action_index_from_candidate",
)

"""Encode player-visible state and translate model actions into engine moves.

The engine stores the four dealt cards in slots 0–3. Playing a card empties its
original slot, so a remaining hand might be ``(None, card B, None, card D)``.
The model does not attach meaning to those permanent slot numbers. It packs the
remaining cards in canonical card-ID order instead:

``model row 0 -> card B -> engine slot 1``
``model row 1 -> card D -> engine slot 3``
``model rows 2 and 3 -> unused padding``

Each row contains scores for eight destinations. The engine and model therefore
both use a ``[4, 8]`` action array, but their rows can identify different card
locations. This module performs that conversion in both directions.
"""

from __future__ import annotations

import torch
from torch import Tensor

from dracula.action_contract import (
    build_representative_action_mask,
    select_representative_action,
)
from dracula.bgc_policy_model import (
    ACTION_COUNT,
    BGCPolicyModel,
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
from dracula.strategic_actions import (
    StrategicActionGroup,
    strategic_action_groups,
    strategic_group_for_representative,
)

ROUND_COUNT = 6
PROGRESS_COUNT = 4
PLAYED_START = STATUS_START
UNSEEN_START = IN_HAND_START + CARD_COUNT
ROUND_START = CONTEXT_START
PROGRESS_START = ROUND_START + ROUND_COUNT
DEALER_INDEX = PROGRESS_START + PROGRESS_COUNT

if DEALER_INDEX + 1 != OBSERVATION_SIZE or CONTEXT_FEATURES != 11:
    raise RuntimeError("policy observation constants do not match the model input")


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
    return observation


def _engine_slots_for_model_rows(
    information: SearchInformationState,
) -> tuple[int, ...]:
    """Return the stable engine slot represented by each packed model row."""

    engine_slots = tuple(
        engine_slot
        for engine_slot, card_id in enumerate(information.own_hand)
        if card_id is not None
    )
    # The original hand is sorted once when dealt. Removing played cards leaves
    # the occupied engine slots in the same card-ID order used by model rows.
    return engine_slots


def pack_action_rows_for_model(
    information: SearchInformationState, engine_slot_values: Tensor
) -> Tensor:
    """Pack action data from stable engine-slot rows into model candidate rows.

    For an engine hand ``(card, None, card, None)``, engine rows 0 and 2 become
    model rows 0 and 1; model rows 2 and 3 remain zero padding.
    """

    if not isinstance(engine_slot_values, Tensor) or engine_slot_values.shape != (
        HAND_CANDIDATE_COUNT,
        POLICY_POSITION_COUNT,
    ):
        raise PolicyObservationError("action tensor must have shape [4,8]")
    candidate_values = torch.zeros_like(engine_slot_values)
    for candidate_row, engine_slot in enumerate(
        _engine_slots_for_model_rows(information)
    ):
        candidate_values[candidate_row] = engine_slot_values[engine_slot]
    return candidate_values


def engine_action_index_from_model(
    information: SearchInformationState, model_action_index: int
) -> int:
    """Translate one flattened model action back to its stable engine slot."""

    if (
        type(model_action_index) is not int
        or not 0 <= model_action_index < ACTION_COUNT
    ):
        raise PolicyObservationError("model action index is invalid")
    model_row, destination = divmod(model_action_index, POLICY_POSITION_COUNT)
    engine_slots = _engine_slots_for_model_rows(information)
    if model_row >= len(engine_slots):
        raise PolicyObservationError("model action names a padding row")
    engine_slot = engine_slots[model_row]
    return engine_slot * POLICY_POSITION_COUNT + destination


def select_policy_group(
    model: BGCPolicyModel,
    information: SearchInformationState,
) -> StrategicActionGroup:
    """Select one legal strategic group from an actor-visible state.

    The representative mask is converted to model rows before masked argmax;
    the selected action is then converted back to its engine-row group.
    """

    # Groups and this first mask use permanent engine hand-slot indexes.
    groups = strategic_action_groups(information)
    engine_mask = torch.tensor(information.legal_mask, dtype=torch.bool)
    representative_mask = build_representative_action_mask(engine_mask, groups)

    # The network's rows contain only the remaining cards, packed without holes.
    model_mask = pack_action_rows_for_model(information, representative_mask)
    with torch.inference_mode():
        logits = model(encode_policy_observation(information))

    model_action = select_representative_action(logits, model_mask)

    # Convert the packed model row back to the engine slot used by the groups.
    representative = engine_action_index_from_model(information, model_action)
    return strategic_group_for_representative(groups, representative)


__all__ = (
    "PolicyObservationError",
    "encode_policy_observation",
    "engine_action_index_from_model",
    "pack_action_rows_for_model",
    "select_policy_group",
)

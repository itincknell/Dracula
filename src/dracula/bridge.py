"""Player-relative tensors and action mappings at the engine-model boundary."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import torch
from torch import Tensor

from dracula.cards import CARD_COUNT, CARD_INDEX_BY_ID, sort_card_ids
from dracula.engine import (
    EngineMove,
    EnginePlayer,
    EngineState,
    EngineStatus,
    EngineTransition,
    SimulationEngineState,
    apply_move,
    legal_moves,
    legal_simulation_moves,
    state_fingerprint,
    validate_state,
)
from dracula.models import (
    COFFIN_START,
    CONTEXT_START,
    HAND_START,
    HIDDEN_SIZE,
    OBSERVATION_SIZE,
    POLICY_GRID_INDICES,
    STATUS_START,
)

# These offsets and shapes are persisted model and dataset contracts.
ACTION_COUNT = 32
HAND_SLOT_COUNT = 4
COFFIN_POSITION_COUNT = 9
POLICY_POSITION_COUNT = 8
ROUND_COUNT = 6
HIDDEN_STATE_BYTES = HIDDEN_SIZE * 4
PLAYED_STATUS_START = STATUS_START
IN_HAND_STATUS_START = PLAYED_STATUS_START + CARD_COUNT
HIDDEN_STATUS_START = IN_HAND_STATUS_START + CARD_COUNT
ROUND_START = CONTEXT_START
PROGRESS_START = ROUND_START + ROUND_COUNT
DEALER_INDEX = PROGRESS_START + HAND_SLOT_COUNT

_POLICY_POSITION_BY_GRID_INDEX = {
    grid_index: position for position, grid_index in enumerate(POLICY_GRID_INDICES)
}
_FINGERPRINT = re.compile(r"^[0-9a-f]{64}$")


class BridgeContractViolation(ValueError):
    """The engine state and model-boundary value do not agree."""


# This enum literal remains serialized in historical recurrent records. Active
# standalone policies also bypass the forced move but carry no recurrent state.
class PolicyTurnKind(StrEnum):
    """Whether a turn is learned or the unique forced final placement."""

    LEARNED = "learned"
    FORCED_RECURRENT_TRANSITION = "forced_recurrent_transition"


def _coerce_player(player: EnginePlayer) -> EnginePlayer:
    try:
        return EnginePlayer(player)
    except (TypeError, ValueError) as error:
        raise BridgeContractViolation(f"unknown player: {player!r}") from error


def _validate_fingerprint(value: object) -> None:
    if not isinstance(value, str) or _FINGERPRINT.fullmatch(value) is None:
        raise BridgeContractViolation("state fingerprint must be a lowercase SHA-256 digest")


def _validate_policy_input_tensors(observation: Tensor, legal_mask: Tensor) -> None:
    """Verify tensor shapes and every redundant observation consistency check."""

    if not isinstance(observation, Tensor) or observation.dtype is not torch.bool:
        raise BridgeContractViolation("observation must be a Boolean tensor")
    if observation.shape != (OBSERVATION_SIZE,):
        raise BridgeContractViolation(f"observation must have shape ({OBSERVATION_SIZE},)")
    if not isinstance(legal_mask, Tensor) or legal_mask.dtype is not torch.bool:
        raise BridgeContractViolation("legal mask must be a Boolean tensor")
    if legal_mask.shape != (HAND_SLOT_COUNT, POLICY_POSITION_COUNT):
        raise BridgeContractViolation("legal mask must have shape (4, 8)")
    if observation.device != legal_mask.device:
        raise BridgeContractViolation("observation and legal mask must share a device")

    hand = observation[HAND_START:COFFIN_START].reshape(HAND_SLOT_COUNT, CARD_COUNT)
    coffin = observation[COFFIN_START:STATUS_START].reshape(
        COFFIN_POSITION_COUNT, CARD_COUNT
    )
    statuses = observation[STATUS_START:CONTEXT_START].reshape(3, CARD_COUNT)
    round_field = observation[ROUND_START:PROGRESS_START]
    progress = observation[PROGRESS_START:DEALER_INDEX]

    if torch.any(hand.sum(dim=1) > 1) or torch.any(hand.sum(dim=0) > 1):
        raise BridgeContractViolation("hand positions must contain unique one-hot cards")
    if torch.any(coffin.sum(dim=1) > 1) or torch.any(coffin.sum(dim=0) > 1):
        raise BridgeContractViolation("coffin positions must contain unique one-hot cards")
    if torch.any(hand.any(dim=0) & coffin.any(dim=0)):
        raise BridgeContractViolation("a card cannot occupy both hand and coffin")
    if not torch.all(statuses.sum(dim=0) == 1):
        raise BridgeContractViolation("card statuses must partition all 54 cards")
    if not torch.equal(statuses[1], hand.any(dim=0)):
        raise BridgeContractViolation("in-hand status must match occupied hand rows")
    if torch.any(coffin.any(dim=0) & ~statuses[0]):
        raise BridgeContractViolation("coffin cards must have played status")
    if int(round_field.sum().item()) != 1:
        raise BridgeContractViolation("round field must be one-hot")
    if int(progress.sum().item()) != 1:
        raise BridgeContractViolation("own-decision progress must be one-hot")
    if torch.any(legal_mask[~hand.any(dim=1)]):
        raise BridgeContractViolation("empty hand slots cannot contain legal actions")
    if int(legal_mask.sum().item()) < 1:
        raise BridgeContractViolation("an active policy input must contain a legal action")


@dataclass(frozen=True, slots=True, eq=False)
class PolicyInput:
    """Actor-visible Boolean observation and matching legal-action mask."""

    observation: Tensor
    legal_mask: Tensor

    def __post_init__(self) -> None:
        _validate_policy_input_tensors(self.observation, self.legal_mask)


@dataclass(frozen=True, slots=True, eq=False)
class PolicyTurnContext:
    """Validated mapping between one engine turn and its 32 policy actions."""

    state_fingerprint: str
    player: EnginePlayer
    round_number: int
    own_decision_index: int
    kind: PolicyTurnKind
    input: PolicyInput
    action_table: tuple[EngineMove | None, ...]
    forced_move: EngineMove | None

    def __post_init__(self) -> None:
        _validate_fingerprint(self.state_fingerprint)
        if not isinstance(self.player, EnginePlayer):
            raise BridgeContractViolation("context player must be an EnginePlayer")
        if type(self.round_number) is not int or not 1 <= self.round_number <= ROUND_COUNT:
            raise BridgeContractViolation("context round number must be between 1 and 6")
        if type(self.own_decision_index) is not int or not 0 <= self.own_decision_index < 4:
            raise BridgeContractViolation("own decision index must be between 0 and 3")
        if not isinstance(self.kind, PolicyTurnKind):
            raise BridgeContractViolation("context has an invalid transition kind")
        if not isinstance(self.input, PolicyInput):
            raise BridgeContractViolation("context input must be a PolicyInput")
        if not isinstance(self.action_table, tuple) or len(self.action_table) != ACTION_COUNT:
            raise BridgeContractViolation("action table must contain 32 entries")

        round_field = self.input.observation[ROUND_START:PROGRESS_START]
        progress = self.input.observation[PROGRESS_START:DEALER_INDEX]
        if not bool(round_field[self.round_number - 1].item()):
            raise BridgeContractViolation("context round disagrees with policy input")
        if not bool(progress[self.own_decision_index].item()):
            raise BridgeContractViolation("context progress disagrees with policy input")

        legal_count = 0
        flat_mask = self.input.legal_mask.flatten()
        for action_index, move in enumerate(self.action_table):
            is_legal = bool(flat_mask[action_index].item())
            if (move is not None) != is_legal:
                raise BridgeContractViolation("legal mask and action table disagree")
            if move is not None:
                legal_count += 1
                if move.player is not self.player:
                    raise BridgeContractViolation("action table contains another player's move")
                if action_index_for_move(move, self.player) != action_index:
                    raise BridgeContractViolation("action table contains a move at the wrong index")

        if self.kind is PolicyTurnKind.FORCED_RECURRENT_TRANSITION:
            if legal_count != 1 or self.forced_move is None:
                raise BridgeContractViolation("forced context must contain exactly one move")
            if self.own_decision_index != 3 or not bool(
                self.input.observation[DEALER_INDEX].item()
            ):
                raise BridgeContractViolation("forced context must be the dealer's fourth turn")
            forced_index = next(
                index for index, move in enumerate(self.action_table) if move is not None
            )
            if self.action_table[forced_index] != self.forced_move:
                raise BridgeContractViolation("forced move does not match the action table")
        elif legal_count < 2 or self.forced_move is not None:
            raise BridgeContractViolation("learned context must contain at least two moves")


def transpose_grid_index(grid_index: int) -> int:
    """Transpose a 3×3 position across its main diagonal."""

    if type(grid_index) is not int or not 0 <= grid_index < COFFIN_POSITION_COUNT:
        raise BridgeContractViolation("grid index must be between 0 and 8")
    return 3 * (grid_index % 3) + grid_index // 3


def player_relative_grid_index(player: EnginePlayer, global_grid_index: int) -> int:
    """Express Queen rows and King columns in one player-relative frame.

    Queen uses the authoritative layout. King transposes it so the columns King
    scores occupy the same tensor directions as Queen's rows.
    """

    player = _coerce_player(player)
    if player is EnginePlayer.QUEEN:
        if type(global_grid_index) is not int or not 0 <= global_grid_index < 9:
            raise BridgeContractViolation("grid index must be between 0 and 8")
        return global_grid_index
    return transpose_grid_index(global_grid_index)


def global_grid_index(player: EnginePlayer, policy_grid_index: int) -> int:
    """Map a relative position back; the same transpose is its own inverse."""

    return player_relative_grid_index(player, policy_grid_index)


def move_for_action_index(player: EnginePlayer, action_index: int) -> EngineMove:
    """Decode one flattened hand-slot/destination policy action."""

    player = _coerce_player(player)
    if type(action_index) is not int or not 0 <= action_index < ACTION_COUNT:
        raise BridgeContractViolation("action index must be between 0 and 31")
    hand_slot, policy_position = divmod(action_index, POLICY_POSITION_COUNT)
    policy_grid_index = POLICY_GRID_INDICES[policy_position]
    return EngineMove(player, hand_slot, global_grid_index(player, policy_grid_index))


def action_index_for_move(move: EngineMove, player: EnginePlayer | None = None) -> int:
    """Encode an authoritative move in the selected player's 32-action frame."""

    if not isinstance(move, EngineMove):
        raise BridgeContractViolation("move must be an EngineMove")
    perspective = move.player if player is None else _coerce_player(player)
    if not isinstance(move.player, EnginePlayer) or move.player is not perspective:
        raise BridgeContractViolation("move player and policy perspective disagree")
    if type(move.hand_slot) is not int or not 0 <= move.hand_slot < HAND_SLOT_COUNT:
        raise BridgeContractViolation("hand slot must be between 0 and 3")
    policy_grid_index = player_relative_grid_index(perspective, move.global_grid_index)
    try:
        policy_position = _POLICY_POSITION_BY_GRID_INDEX[policy_grid_index]
    except KeyError as error:
        raise BridgeContractViolation("the center is not a policy action position") from error
    return POLICY_POSITION_COUNT * move.hand_slot + policy_position


def encode_hand_positions(hand: tuple[str | None, ...]) -> Tensor:
    """Encode stable hand slots as a Boolean 4×54 tensor."""

    if not isinstance(hand, tuple) or len(hand) != HAND_SLOT_COUNT:
        raise BridgeContractViolation("hand must be a four-slot tuple")
    occupied = tuple(card_id for card_id in hand if card_id is not None)
    if len(set(occupied)) != len(occupied) or any(
        card_id not in CARD_INDEX_BY_ID for card_id in occupied
    ):
        raise BridgeContractViolation("hand contains duplicate or unknown cards")
    if occupied != sort_card_ids(occupied):
        raise BridgeContractViolation("occupied hand slots must retain canonical order")
    encoded = torch.zeros(HAND_SLOT_COUNT, CARD_COUNT, dtype=torch.bool)
    for hand_slot, card_id in enumerate(hand):
        if card_id is not None:
            encoded[hand_slot, CARD_INDEX_BY_ID[card_id]] = True
    return encoded


def encode_player_relative_coffin(
    coffin: tuple[str | None, ...], player: EnginePlayer
) -> Tensor:
    """Encode a coffin as a Boolean 9×54 tensor in the player's frame."""

    player = _coerce_player(player)
    if not isinstance(coffin, tuple) or len(coffin) != COFFIN_POSITION_COUNT:
        raise BridgeContractViolation("coffin must be a nine-slot tuple")
    occupied = tuple(card_id for card_id in coffin if card_id is not None)
    if len(set(occupied)) != len(occupied) or any(
        card_id not in CARD_INDEX_BY_ID for card_id in occupied
    ):
        raise BridgeContractViolation("coffin contains duplicate or unknown cards")
    encoded = torch.zeros(COFFIN_POSITION_COUNT, CARD_COUNT, dtype=torch.bool)
    for authoritative_index, card_id in enumerate(coffin):
        if card_id is not None:
            policy_index = player_relative_grid_index(player, authoritative_index)
            encoded[policy_index, CARD_INDEX_BY_ID[card_id]] = True
    return encoded


def _encode_observation_unchecked(state: EngineState, player: EnginePlayer) -> Tensor:
    """Encode only policy-visible fields after the public caller validates state."""

    hand = encode_hand_positions(state.hands[player])
    coffin = encode_player_relative_coffin(state.coffin, player)
    observation = torch.zeros(OBSERVATION_SIZE, dtype=torch.bool)
    observation[HAND_START:COFFIN_START] = hand.flatten()
    observation[COFFIN_START:STATUS_START] = coffin.flatten()

    # Status distinguishes public cards, the actor's hand, and one unseen pool;
    # it never encodes the opponent-hand and stock locations separately.
    played_card_ids = {
        card_id
        for result in state.completed_rounds
        for card_id in result.coffin
    }
    played_card_ids.update(card_id for card_id in state.coffin if card_id is not None)
    in_hand_card_ids = {card_id for card_id in state.hands[player] if card_id is not None}
    for card_id in played_card_ids:
        observation[PLAYED_STATUS_START + CARD_INDEX_BY_ID[card_id]] = True
    for card_id in in_hand_card_ids:
        observation[IN_HAND_STATUS_START + CARD_INDEX_BY_ID[card_id]] = True
    visible_card_ids = played_card_ids | in_hand_card_ids
    for card_id, card_index in CARD_INDEX_BY_ID.items():
        if card_id not in visible_card_ids:
            observation[HIDDEN_STATUS_START + card_index] = True

    own_decision_index = sum(
        move.player is player for move in state.current_round_moves
    )
    observation[ROUND_START + state.round_number - 1] = True
    observation[PROGRESS_START + own_decision_index] = True
    observation[DEALER_INDEX] = player is state.dealer
    return observation


def build_policy_turn_context(
    state: EngineState, player: EnginePlayer
) -> PolicyTurnContext:
    """Build the actor-visible input and exact lookup for one legal engine turn."""

    validate_state(state)
    player = _coerce_player(player)
    if state.status is not EngineStatus.PLAYING or state.active_player is not player:
        raise BridgeContractViolation(
            "policy context requires the active player in a playing round"
        )

    engine_legal_moves = legal_moves(state, player)
    # The fixed 32-entry table is the authoritative bridge from model indexes
    # back to engine moves; ``None`` and the Boolean mask must agree exactly.
    action_table: list[EngineMove | None] = [None] * ACTION_COUNT
    for move in engine_legal_moves:
        action_index = action_index_for_move(move, player)
        if action_table[action_index] is not None:
            raise BridgeContractViolation("engine legal moves collide in the action table")
        action_table[action_index] = move
    legal_mask = torch.tensor(
        [move is not None for move in action_table], dtype=torch.bool
    ).reshape(HAND_SLOT_COUNT, POLICY_POSITION_COUNT)
    observation = _encode_observation_unchecked(state, player)
    own_decision_index = sum(
        move.player is player for move in state.current_round_moves
    )
    if not 0 <= own_decision_index < 4:
        raise BridgeContractViolation("active player has invalid decision progress")
    kind = (
        PolicyTurnKind.FORCED_RECURRENT_TRANSITION
        if len(engine_legal_moves) == 1
        else PolicyTurnKind.LEARNED
    )
    forced_move = engine_legal_moves[0] if len(engine_legal_moves) == 1 else None
    return PolicyTurnContext(
        state_fingerprint=state_fingerprint(state),
        player=player,
        round_number=state.round_number,
        own_decision_index=own_decision_index,
        kind=kind,
        input=PolicyInput(observation, legal_mask),
        action_table=tuple(action_table),
        forced_move=forced_move,
    )


def build_simulation_policy_input(
    state: SimulationEngineState, player: EnginePlayer
) -> PolicyInput:
    """Project a trusted sampled state without whole-deck validation."""

    if not isinstance(state, SimulationEngineState):
        raise BridgeContractViolation(
            "simulation policy input requires a sampled engine state"
        )
    player = _coerce_player(player)
    if state.status is not EngineStatus.PLAYING or state.active_player is not player:
        raise BridgeContractViolation(
            "simulation policy input requires the active player"
        )
    action_table: list[EngineMove | None] = [None] * ACTION_COUNT
    for move in legal_simulation_moves(state, player):
        action_index = action_index_for_move(move, player)
        if action_table[action_index] is not None:
            raise BridgeContractViolation(
                "simulation legal moves collide in the action table"
            )
        action_table[action_index] = move
    return PolicyInput(
        _encode_observation_unchecked(state, player),
        torch.tensor(
            [move is not None for move in action_table], dtype=torch.bool
        ).reshape(HAND_SLOT_COUNT, POLICY_POSITION_COUNT),
    )


def _policy_inputs_equal(first: PolicyInput, second: PolicyInput) -> bool:
    return torch.equal(first.observation.cpu(), second.observation.cpu()) and torch.equal(
        first.legal_mask.cpu(), second.legal_mask.cpu()
    )


def validate_policy_turn_context(state: EngineState, context: PolicyTurnContext) -> None:
    """Reject a context not derived exactly from the supplied private state."""

    if not isinstance(context, PolicyTurnContext):
        raise BridgeContractViolation("context must be a PolicyTurnContext")
    expected = build_policy_turn_context(state, context.player)
    if (
        context.state_fingerprint != expected.state_fingerprint
        or context.round_number != expected.round_number
        or context.own_decision_index != expected.own_decision_index
        or context.kind is not expected.kind
        or context.action_table != expected.action_table
        or context.forced_move != expected.forced_move
        or not _policy_inputs_equal(context.input, expected.input)
    ):
        raise BridgeContractViolation("policy context does not exactly match engine state")


def resolve_policy_action(
    state: EngineState,
    context: PolicyTurnContext,
    action_index: int | None = None,
) -> tuple[int, EngineMove]:
    """Resolve a selected or forced policy index to one legal engine move."""

    validate_policy_turn_context(state, context)
    if context.kind is PolicyTurnKind.FORCED_RECURRENT_TRANSITION:
        forced_index = next(
            index for index, move in enumerate(context.action_table) if move is not None
        )
        if action_index is not None and action_index != forced_index:
            raise BridgeContractViolation("selected action does not match the forced move")
        action_index = forced_index
    elif action_index is None:
        raise BridgeContractViolation("a learned turn requires a selected action index")
    if type(action_index) is not int or not 0 <= action_index < ACTION_COUNT:
        raise BridgeContractViolation("action index must be between 0 and 31")
    move = context.action_table[action_index]
    if move is None:
        raise BridgeContractViolation("selected action index is masked")
    return action_index, move


def apply_policy_action(
    state: EngineState,
    context: PolicyTurnContext,
    action_index: int | None = None,
) -> EngineTransition:
    """Resolve and apply one policy action through the authoritative engine."""

    _, move = resolve_policy_action(state, context, action_index)
    return apply_move(state, move)


_LEGACY_EXPORTS = frozenset(
    {
        "PolicyTransition",
        "build_policy_transition",
        "hidden_state_from_bytes",
        "hidden_state_to_bytes",
        "validate_policy_transition",
    }
)


def __getattr__(name: str) -> Any:
    # Recurrent compatibility types stay importable without joining the active
    # bridge's dependency graph or import-time behavior.
    if name not in _LEGACY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from dracula import bridge_legacy

    value = getattr(bridge_legacy, name)
    globals()[name] = value
    return value

"""Translate between engine moves and the fixed policy action space.

The bridge builds actor-relative legal action tables and maps their indexes
back to concrete moves without exposing authoritative hidden engine state.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

from dracula.game.engine import (
    EngineMove,
    EnginePlayer,
    EngineState,
    EngineStatus,
    EngineTransition,
    apply_move,
    legal_moves,
    state_fingerprint,
    validate_state,
)

ACTION_COUNT = 32
HAND_SLOT_COUNT = 4
COFFIN_POSITION_COUNT = 9
POLICY_POSITION_COUNT = 8
ROUND_COUNT = 6
POLICY_GRID_INDICES = (0, 1, 2, 3, 5, 6, 7, 8)

_POLICY_POSITION_BY_GRID_INDEX = {
    grid_index: position for position, grid_index in enumerate(POLICY_GRID_INDICES)
}
_FINGERPRINT = re.compile(r"^[0-9a-f]{64}$")


class BridgeContractViolation(ValueError):
    """An engine state, action index, or turn context is inconsistent."""


class PolicyTurnKind(StrEnum):
    """Distinguish a model decision from the unique forced placement."""

    LEARNED = "learned"
    FORCED = "forced"


def _coerce_player(player: EnginePlayer) -> EnginePlayer:
    try:
        return EnginePlayer(player)
    except (TypeError, ValueError) as error:
        raise BridgeContractViolation(f"unknown player: {player!r}") from error


@dataclass(frozen=True, slots=True)
class PolicyTurnContext:
    """Immutable lookup from the 32 policy indexes to legal engine moves."""

    state_fingerprint: str
    player: EnginePlayer
    round_number: int
    own_decision_index: int
    kind: PolicyTurnKind
    action_table: tuple[EngineMove | None, ...]
    forced_move: EngineMove | None

    def __post_init__(self) -> None:
        if _FINGERPRINT.fullmatch(self.state_fingerprint) is None:
            raise BridgeContractViolation(
                "state fingerprint must be a lowercase SHA-256 digest"
            )
        if not isinstance(self.player, EnginePlayer):
            raise BridgeContractViolation("context player must be an EnginePlayer")
        if type(self.round_number) is not int or not 1 <= self.round_number <= ROUND_COUNT:
            raise BridgeContractViolation("context round number must be between 1 and 6")
        if (
            type(self.own_decision_index) is not int
            or not 0 <= self.own_decision_index < HAND_SLOT_COUNT
        ):
            raise BridgeContractViolation("own decision index must be between 0 and 3")
        if not isinstance(self.kind, PolicyTurnKind):
            raise BridgeContractViolation("context has an invalid turn kind")
        if not isinstance(self.action_table, tuple) or len(self.action_table) != ACTION_COUNT:
            raise BridgeContractViolation("action table must contain 32 entries")

        moves = tuple(move for move in self.action_table if move is not None)
        if not moves or any(move.player is not self.player for move in moves):
            raise BridgeContractViolation("action table contains invalid player moves")
        for action_index, move in enumerate(self.action_table):
            if move is not None and action_index_for_move(move, self.player) != action_index:
                raise BridgeContractViolation("action table contains a move at the wrong index")
        if self.kind is PolicyTurnKind.FORCED:
            if len(moves) != 1 or self.forced_move != moves[0]:
                raise BridgeContractViolation("forced context must contain exactly one move")
            if self.own_decision_index != 3:
                raise BridgeContractViolation("forced placement must be the dealer's fourth turn")
        elif len(moves) < 2 or self.forced_move is not None:
            raise BridgeContractViolation("learned context must contain at least two moves")


def transpose_grid_index(grid_index: int) -> int:
    """Transpose a 3×3 position across its main diagonal."""

    if type(grid_index) is not int or not 0 <= grid_index < COFFIN_POSITION_COUNT:
        raise BridgeContractViolation("grid index must be between 0 and 8")
    return 3 * (grid_index % 3) + grid_index // 3


def player_relative_grid_index(
    player: EnginePlayer, global_grid_index: int
) -> int:
    """Express Queen rows and King columns in one player-relative frame."""

    player = _coerce_player(player)
    if player is EnginePlayer.QUEEN:
        if (
            type(global_grid_index) is not int
            or not 0 <= global_grid_index < COFFIN_POSITION_COUNT
        ):
            raise BridgeContractViolation("grid index must be between 0 and 8")
        return global_grid_index
    return transpose_grid_index(global_grid_index)


def global_grid_index(player: EnginePlayer, policy_grid_index: int) -> int:
    """Map a player-relative position back to the authoritative grid."""

    return player_relative_grid_index(player, policy_grid_index)


def move_for_action_index(player: EnginePlayer, action_index: int) -> EngineMove:
    """Decode one flattened hand-slot/destination action."""

    player = _coerce_player(player)
    if type(action_index) is not int or not 0 <= action_index < ACTION_COUNT:
        raise BridgeContractViolation("action index must be between 0 and 31")
    hand_slot, policy_position = divmod(action_index, POLICY_POSITION_COUNT)
    relative_grid = POLICY_GRID_INDICES[policy_position]
    return EngineMove(player, hand_slot, global_grid_index(player, relative_grid))


def action_index_for_move(
    move: EngineMove, player: EnginePlayer | None = None
) -> int:
    """Encode an authoritative move in the selected player's action frame."""

    if not isinstance(move, EngineMove):
        raise BridgeContractViolation("move must be an EngineMove")
    perspective = move.player if player is None else _coerce_player(player)
    if not isinstance(move.player, EnginePlayer) or move.player is not perspective:
        raise BridgeContractViolation("move player and policy perspective disagree")
    if type(move.hand_slot) is not int or not 0 <= move.hand_slot < HAND_SLOT_COUNT:
        raise BridgeContractViolation("hand slot must be between 0 and 3")
    relative_grid = player_relative_grid_index(perspective, move.global_grid_index)
    try:
        position = _POLICY_POSITION_BY_GRID_INDEX[relative_grid]
    except KeyError as error:
        raise BridgeContractViolation("the center is not a policy destination") from error
    return POLICY_POSITION_COUNT * move.hand_slot + position


def build_policy_turn_context(
    state: EngineState, player: EnginePlayer
) -> PolicyTurnContext:
    """Build the exact legal-action lookup for the active player."""

    validate_state(state)
    player = _coerce_player(player)
    if state.status is not EngineStatus.PLAYING or state.active_player is not player:
        raise BridgeContractViolation(
            "policy context requires the active player in a playing round"
        )
    moves = legal_moves(state, player)
    action_table: list[EngineMove | None] = [None] * ACTION_COUNT
    for move in moves:
        index = action_index_for_move(move, player)
        if action_table[index] is not None:
            raise BridgeContractViolation("legal engine moves collide in the action table")
        action_table[index] = move
    own_decision_index = sum(
        move.player is player for move in state.current_round_moves
    )
    if not 0 <= own_decision_index < HAND_SLOT_COUNT:
        raise BridgeContractViolation("active player has invalid decision progress")
    forced_move = moves[0] if len(moves) == 1 else None
    return PolicyTurnContext(
        state_fingerprint=state_fingerprint(state),
        player=player,
        round_number=state.round_number,
        own_decision_index=own_decision_index,
        kind=PolicyTurnKind.FORCED if forced_move is not None else PolicyTurnKind.LEARNED,
        action_table=tuple(action_table),
        forced_move=forced_move,
    )


def validate_policy_turn_context(
    state: EngineState, context: PolicyTurnContext
) -> None:
    """Reject a context not derived exactly from the supplied state."""

    if not isinstance(context, PolicyTurnContext):
        raise BridgeContractViolation("context must be a PolicyTurnContext")
    if context != build_policy_turn_context(state, context.player):
        raise BridgeContractViolation("policy context does not exactly match engine state")


def resolve_policy_action(
    state: EngineState,
    context: PolicyTurnContext,
    action_index: int | None = None,
) -> tuple[int, EngineMove]:
    """Resolve a selected or forced index to one legal engine move."""

    validate_policy_turn_context(state, context)
    if context.kind is PolicyTurnKind.FORCED:
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


__all__ = (
    "ACTION_COUNT",
    "BridgeContractViolation",
    "HAND_SLOT_COUNT",
    "POLICY_GRID_INDICES",
    "POLICY_POSITION_COUNT",
    "PolicyTurnContext",
    "PolicyTurnKind",
    "action_index_for_move",
    "apply_policy_action",
    "build_policy_turn_context",
    "global_grid_index",
    "move_for_action_index",
    "player_relative_grid_index",
    "resolve_policy_action",
    "transpose_grid_index",
    "validate_policy_turn_context",
)

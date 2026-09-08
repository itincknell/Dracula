"""Provide the public gameplay operations for the deterministic engine.

This facade creates games, enumerates legal moves, applies immutable state
transitions, and advances rounds while delegating types, scoring, and checks.
"""

from __future__ import annotations

from dataclasses import replace

from dracula.game.dealing import (
    as_hand,
    create_initial_deal,
    deal_round,
    initial_dealer,
    shuffled_deck,
    starting_coffin,
)
from dracula.game.serialization import (
    canonical_state_data,
    canonical_state_json,
    state_fingerprint,
)
from dracula.game.types import (
    CENTER_GRID_INDEX,
    COFFIN_SIZE,
    HAND_SIZE,
    MOVES_PER_ROUND,
    ROUNDS_PER_GAME,
    Coffin,
    EngineMove,
    EnginePlayedMove,
    EnginePlayer,
    EngineRoundResult,
    EngineState,
    EngineStatus,
    EngineTransition,
    GameOutcome,
    GameOutcomeReason,
    Hand,
    InvalidGridIndex,
    InvalidLifecycleTransition,
    LineOrientation,
    LineScore,
    MalformedState,
    MultiplierReason,
    NonAdjacentDestination,
    OccupiedDestination,
    PlayerValues,
    RoundDeal,
    RuleViolation,
    SimulationEngineState,
    UnavailableHandSlot,
    WrongActivePlayer,
    empty_adjacent_grid_indices,
    other_player,
)
from dracula.game.validation import validate_state
from dracula.game.scoring import (
    make_round_result,
    resolve_game_outcome,
    resolve_round_scores,
    score_coffin,
    score_line,
)


def create_game(seed: str) -> EngineState:
    """Create and validate the first active round for a deterministic game seed."""

    if not isinstance(seed, str):
        raise TypeError("seed must be a string")
    deal = create_initial_deal(seed)
    state = EngineState(
        seed=seed,
        status=EngineStatus.PLAYING,
        round_number=1,
        dealer=deal.dealer,
        active_player=deal.non_dealer,
        stock=deal.remaining_stock,
        hands=PlayerValues(
            queen=as_hand(deal.queen_hand),
            king=as_hand(deal.king_hand),
        ),
        coffin=starting_coffin(deal.center_card),
        current_round_moves=(),
        pending_round_result=None,
        completed_rounds=(),
        total_scores=PlayerValues(queen=0, king=0),
    )
    validate_state(state)
    return state


def _legal_moves_for_active_player(
    state: EngineState, player: EnginePlayer
) -> tuple[EngineMove, ...]:
    """Enumerate legal placements assuming the enclosing state is trusted."""

    if state.status is not EngineStatus.PLAYING:
        raise InvalidLifecycleTransition("moves are available only while a round is playing")
    try:
        player = EnginePlayer(player)
    except (TypeError, ValueError) as error:
        raise WrongActivePlayer(f"unknown player: {player!r}") from error
    if player is not state.active_player:
        raise WrongActivePlayer(f"{player.value} is not the active player")

    destinations = empty_adjacent_grid_indices(state.coffin)
    # This nesting defines the stable legal-move order: hand slot first, then
    # global destination index.
    return tuple(
        EngineMove(player, hand_slot, grid_index)
        for hand_slot, card_id in enumerate(state.hands[player])
        if card_id is not None
        for grid_index in destinations
    )


def legal_moves(state: EngineState, player: EnginePlayer) -> tuple[EngineMove, ...]:
    """Return canonical legal moves after validating the complete private state."""

    validate_state(state)
    return _legal_moves_for_active_player(state, player)


def legal_simulation_moves(
    state: SimulationEngineState, player: EnginePlayer
) -> tuple[EngineMove, ...]:
    """Return legal moves without repeating whole-deck validation."""

    if not isinstance(state, SimulationEngineState):
        raise TypeError("fast simulation legality requires SimulationEngineState")
    return _legal_moves_for_active_player(state, player)


def _apply_move_state(
    state: EngineState, move: EngineMove
) -> tuple[EngineState, EnginePlayedMove, EngineRoundResult | None]:
    """Apply rule checks and construct new state without whole-state validation."""

    if state.status is not EngineStatus.PLAYING:
        raise InvalidLifecycleTransition("a move cannot be applied outside active play")
    if not isinstance(move, EngineMove):
        raise RuleViolation("move must be an EngineMove")
    try:
        player = EnginePlayer(move.player)
    except (TypeError, ValueError) as error:
        raise WrongActivePlayer(f"unknown player: {move.player!r}") from error
    if player is not state.active_player:
        raise WrongActivePlayer(f"{player.value} is not the active player")
    if type(move.hand_slot) is not int or not 0 <= move.hand_slot < HAND_SIZE:
        raise UnavailableHandSlot("hand slot must be between 0 and 3")
    card_id = state.hands[player][move.hand_slot]
    if card_id is None:
        raise UnavailableHandSlot(f"hand slot {move.hand_slot} is empty")
    if type(move.global_grid_index) is not int or not 0 <= move.global_grid_index < COFFIN_SIZE:
        raise InvalidGridIndex("grid index must be between 0 and 8")
    if state.coffin[move.global_grid_index] is not None:
        raise OccupiedDestination(f"coffin position {move.global_grid_index} is occupied")
    if move.global_grid_index not in empty_adjacent_grid_indices(state.coffin):
        raise NonAdjacentDestination(
            f"coffin position {move.global_grid_index} has no orthogonally adjacent card"
        )

    hand = list(state.hands[player])
    hand[move.hand_slot] = None
    hands = state.hands.updated(player, tuple(hand))  # type: ignore[arg-type]
    coffin = list(state.coffin)
    coffin[move.global_grid_index] = card_id
    played_move = EnginePlayedMove(
        player=player,
        card_id=card_id,
        hand_slot=move.hand_slot,
        global_grid_index=move.global_grid_index,
        turn_number=len(state.current_round_moves) + 1,
    )
    next_state = replace(
        state,
        hands=hands,
        coffin=tuple(coffin),  # type: ignore[arg-type]
        current_round_moves=state.current_round_moves + (played_move,),
        active_player=other_player(player),
    )
    round_result = None
    if played_move.turn_number == MOVES_PER_ROUND:
        # The eighth placement immediately computes and exposes the round score;
        # archiving and dealing remain a separate lifecycle transition.
        round_result = make_round_result(next_state)
        next_state = replace(
            next_state,
            status=EngineStatus.ROUND_COMPLETE,
            active_player=None,
            pending_round_result=round_result,
            total_scores=PlayerValues(
                queen=state.total_scores.queen + round_result.round_scores.queen,
                king=state.total_scores.king + round_result.round_scores.king,
            ),
        )
    return next_state, played_move, round_result


def apply_move(state: EngineState, move: EngineMove) -> EngineTransition:
    """Validate and apply one authoritative move as an immutable transition."""

    validate_state(state)
    next_state, played_move, round_result = _apply_move_state(state, move)
    validate_state(next_state)
    return EngineTransition(
        previous_state=state,
        state=next_state,
        move=move,
        played_move=played_move,
        round_result=round_result,
        state_fingerprint=state_fingerprint(next_state),
    )


def apply_simulation_move(
    state: SimulationEngineState, move: EngineMove
) -> SimulationEngineState:
    """Apply one search transition without repeating whole-deck validation."""

    if not isinstance(state, SimulationEngineState):
        raise TypeError("fast simulation transitions require SimulationEngineState")
    next_state, _, _ = _apply_move_state(state, move)
    if not isinstance(next_state, SimulationEngineState):
        raise MalformedState("simulation transition lost its sampled-deck provenance")
    return next_state


def advance_after_round(state: EngineState) -> EngineState:
    """Archive a pending round and either deal the next one or finish the game."""

    validate_state(state)
    if isinstance(state, SimulationEngineState):
        raise InvalidLifecycleTransition("round-local simulation cannot advance rounds")
    if state.status is not EngineStatus.ROUND_COMPLETE or state.pending_round_result is None:
        raise InvalidLifecycleTransition("only a completed round can be advanced")

    completed_rounds = state.completed_rounds + (state.pending_round_result,)
    if state.round_number == ROUNDS_PER_GAME:
        # A completed game retains archived results and totals but clears every
        # round-local card location.
        terminal = replace(
            state,
            status=EngineStatus.GAME_COMPLETE,
            active_player=None,
            hands=PlayerValues(
                queen=(None, None, None, None), king=(None, None, None, None)
            ),
            coffin=(None, None, None, None, None, None, None, None, None),
            current_round_moves=(),
            pending_round_result=None,
            completed_rounds=completed_rounds,
        )
        validate_state(terminal)
        return terminal

    # Only the canonical stock tail advances between rounds; cards from prior
    # coffins remain represented in completed results.
    dealer = other_player(state.dealer)
    deal = deal_round(state.stock, dealer)
    next_state = EngineState(
        seed=state.seed,
        status=EngineStatus.PLAYING,
        round_number=state.round_number + 1,
        dealer=dealer,
        active_player=deal.non_dealer,
        stock=deal.remaining_stock,
        hands=PlayerValues(
            queen=as_hand(deal.queen_hand), king=as_hand(deal.king_hand)
        ),
        coffin=starting_coffin(deal.center_card),
        current_round_moves=(),
        pending_round_result=None,
        completed_rounds=completed_rounds,
        total_scores=state.total_scores,
    )
    validate_state(next_state)
    return next_state


def derive_game_outcome(state: EngineState) -> GameOutcome:
    """Return the rules-defined outcome of a validated completed game."""

    validate_state(state)
    if state.status is not EngineStatus.GAME_COMPLETE:
        raise InvalidLifecycleTransition("a final outcome requires a completed game")
    return resolve_game_outcome(state.total_scores, state.completed_rounds[-1].round_scores)


__all__ = [
    "CENTER_GRID_INDEX", "COFFIN_SIZE", "HAND_SIZE", "MOVES_PER_ROUND",
    "ROUNDS_PER_GAME", "Coffin", "EngineMove",
    "EnginePlayedMove", "EnginePlayer", "EngineRoundResult", "EngineState", "EngineStatus",
    "EngineTransition", "GameOutcome", "GameOutcomeReason", "Hand", "InvalidGridIndex",
    "InvalidLifecycleTransition", "LineOrientation", "LineScore", "MalformedState",
    "MultiplierReason", "NonAdjacentDestination", "OccupiedDestination", "PlayerValues",
    "RoundDeal", "RuleViolation", "SimulationEngineState", "UnavailableHandSlot",
    "WrongActivePlayer", "advance_after_round", "apply_move", "apply_simulation_move",
    "canonical_state_data", "canonical_state_json", "create_game", "create_initial_deal",
    "deal_round", "derive_game_outcome", "initial_dealer", "legal_moves",
    "legal_simulation_moves", "other_player", "resolve_game_outcome",
    "resolve_round_scores", "score_coffin", "score_line", "shuffled_deck",
    "state_fingerprint", "validate_state",
]

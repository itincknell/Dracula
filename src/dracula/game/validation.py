"""Validate complete private engine states at their ownership boundary.

Checks cover lifecycle consistency, deal provenance, legal history, scoring,
and conservation of all cards. Transition code relies on these guarantees.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from dracula.game.cards import CARD_IDS, CARD_INDEX_BY_ID, sort_card_ids
from dracula.game.dealing import deal_round, initial_dealer, shuffled_deck
from dracula.game.types import (
    CENTER_GRID_INDEX,
    COFFIN_SIZE,
    HAND_SIZE,
    MOVES_PER_ROUND,
    ROUNDS_PER_GAME,
    EnginePlayer,
    EnginePlayedMove,
    EngineRoundResult,
    EngineState,
    EngineStatus,
    LineScore,
    MalformedState,
    PlayerValues,
    SimulationEngineState,
    orthogonally_adjacent,
    other_player,
)
from dracula.game.scoring import round_scores_from_lines, score_coffin


@dataclass(frozen=True, slots=True)
class _ValidatedStateParts:
    """Validated views reused by the cross-field consistency checks."""

    hands: PlayerValues[object]
    current_cards: tuple[str, ...]
    moves: tuple[EnginePlayedMove, ...]
    completed: tuple[EngineRoundResult, ...]
    pending: EngineRoundResult | None
    totals: PlayerValues[object]


def _validate_player_values(value: object, label: str) -> PlayerValues[object]:
    if not isinstance(value, PlayerValues):
        raise MalformedState(f"{label} must be PlayerValues")
    return value


def _validate_card_id(card_id: object, label: str) -> str:
    if not isinstance(card_id, str) or card_id not in CARD_INDEX_BY_ID:
        raise MalformedState(f"{label} contains an unknown card ID: {card_id!r}")
    return card_id


def _validate_played_moves(
    moves: object,
    dealer: EnginePlayer,
    coffin: Sequence[str | None],
    label: str,
) -> tuple[EnginePlayedMove, ...]:
    """Validate alternating actors, slot use, adjacency, and coffin agreement."""

    if not isinstance(moves, tuple) or len(moves) > MOVES_PER_ROUND:
        raise MalformedState(f"{label} must be a tuple of at most eight moves")
    occupied = {CENTER_GRID_INDEX}
    non_dealer = other_player(dealer)
    seen_slots: set[tuple[EnginePlayer, int]] = set()
    for index, move in enumerate(moves):
        if not isinstance(move, EnginePlayedMove):
            raise MalformedState(f"{label} contains a non-played-move value")
        expected_player = non_dealer if index % 2 == 0 else dealer
        if move.player is not expected_player:
            raise MalformedState(f"{label} has an invalid player at turn {index + 1}")
        if move.turn_number != index + 1:
            raise MalformedState(f"{label} has a non-sequential turn number")
        if type(move.hand_slot) is not int or not 0 <= move.hand_slot < HAND_SIZE:
            raise MalformedState(f"{label} contains an invalid hand slot")
        slot_key = (move.player, move.hand_slot)
        if slot_key in seen_slots:
            raise MalformedState(f"{label} reuses a hand slot")
        seen_slots.add(slot_key)
        _validate_card_id(move.card_id, f"{label} move")
        grid_index = move.global_grid_index
        if type(grid_index) is not int or not 0 <= grid_index < COFFIN_SIZE:
            raise MalformedState(f"{label} contains an invalid grid index")
        if grid_index in occupied:
            raise MalformedState(f"{label} reuses an occupied grid index")
        if not any(orthogonally_adjacent(grid_index, prior) for prior in occupied):
            raise MalformedState(f"{label} contains a non-adjacent placement")
        if coffin[grid_index] != move.card_id:
            raise MalformedState(f"{label} move does not match the coffin")
        occupied.add(grid_index)
    return moves


def _validate_line_score(line: object, expected: LineScore, label: str) -> None:
    if line != expected:
        raise MalformedState(f"{label} does not match rules-defined scoring")


def _validate_round_result(result: object, label: str) -> EngineRoundResult:
    """Recompute and verify every rule-derived field of one round result."""

    if not isinstance(result, EngineRoundResult):
        raise MalformedState(f"{label} must be an EngineRoundResult")
    if type(result.round_number) is not int or not 1 <= result.round_number <= ROUNDS_PER_GAME:
        raise MalformedState(f"{label} has an invalid round number")
    if not isinstance(result.dealer, EnginePlayer):
        raise MalformedState(f"{label} has an invalid dealer")
    if not isinstance(result.coffin, tuple) or len(result.coffin) != COFFIN_SIZE:
        raise MalformedState(f"{label} must contain a nine-card coffin")
    for card_id in result.coffin:
        _validate_card_id(card_id, f"{label} coffin")
    if len(set(result.coffin)) != COFFIN_SIZE:
        raise MalformedState(f"{label} coffin contains duplicate cards")
    moves = _validate_played_moves(result.moves, result.dealer, result.coffin, f"{label} moves")
    if len(moves) != MOVES_PER_ROUND or result.coffin[CENTER_GRID_INDEX] in {
        move.card_id for move in moves
    }:
        raise MalformedState(f"{label} does not contain eight placements and one center card")
    expected_lines = score_coffin(result.coffin)
    actual_lines = _validate_player_values(result.line_scores, f"{label} line scores")
    for player in EnginePlayer:
        lines = actual_lines[player]
        if not isinstance(lines, tuple) or len(lines) != 3:
            raise MalformedState(f"{label} must contain three line scores per player")
        for index, line in enumerate(lines):
            _validate_line_score(line, expected_lines[player][index], f"{label} line score")
    expected_round_scores = round_scores_from_lines(expected_lines)
    if result.round_scores != expected_round_scores:
        raise MalformedState(f"{label} round scores do not match its line scores")
    return result


def _result_hand(result: EngineRoundResult, player: EnginePlayer) -> tuple[str, ...]:
    """Recover a player's original canonical hand from completed move records."""

    cards_by_slot = {
        move.hand_slot: move.card_id for move in result.moves if move.player is player
    }
    if set(cards_by_slot) != set(range(HAND_SIZE)):
        raise MalformedState("a completed round does not consume every hand slot")
    return tuple(cards_by_slot[slot] for slot in range(HAND_SIZE))


def _validate_state_header(state: EngineState) -> None:
    """Validate scalar identity, lifecycle, stock type, and stock uniqueness."""

    if not isinstance(state, EngineState):
        raise MalformedState("state must be an EngineState")
    if not isinstance(state.seed, str):
        raise MalformedState("seed must be a string")
    if not isinstance(state.status, EngineStatus):
        raise MalformedState("state has an invalid status")
    if type(state.round_number) is not int or not 1 <= state.round_number <= ROUNDS_PER_GAME:
        raise MalformedState("round number must be between 1 and 6")
    if not isinstance(state.dealer, EnginePlayer):
        raise MalformedState("state has an invalid dealer")
    if state.active_player is not None and not isinstance(state.active_player, EnginePlayer):
        raise MalformedState("state has an invalid active player")
    if not isinstance(state.stock, tuple):
        raise MalformedState("stock must be a tuple")
    for card_id in state.stock:
        _validate_card_id(card_id, "stock")
    if len(set(state.stock)) != len(state.stock):
        raise MalformedState("stock contains duplicate cards")


def _validate_state_parts(state: EngineState) -> _ValidatedStateParts:
    """Validate nested state values and return normalized views for later checks."""

    hands = _validate_player_values(state.hands, "hands")
    current_cards: list[str] = []
    for player in EnginePlayer:
        hand = hands[player]
        if not isinstance(hand, tuple) or len(hand) != HAND_SIZE:
            raise MalformedState("each hand must be a four-slot tuple")
        for card_id in hand:
            if card_id is not None:
                current_cards.append(_validate_card_id(card_id, f"{player.value} hand"))

    if not isinstance(state.coffin, tuple) or len(state.coffin) != COFFIN_SIZE:
        raise MalformedState("coffin must be a nine-slot tuple")
    for card_id in state.coffin:
        if card_id is not None:
            current_cards.append(_validate_card_id(card_id, "coffin"))
    moves = _validate_played_moves(
        state.current_round_moves, state.dealer, state.coffin, "current-round moves"
    )
    completed = state.completed_rounds
    if not isinstance(completed, tuple):
        raise MalformedState("completed rounds must be a tuple")
    completed_results = tuple(
        _validate_round_result(result, f"completed round {index + 1}")
        for index, result in enumerate(completed)
    )
    pending = (
        None
        if state.pending_round_result is None
        else _validate_round_result(state.pending_round_result, "pending round")
    )
    totals = _validate_player_values(state.total_scores, "total scores")
    if any(type(value) is not int or value < 0 for value in (totals.queen, totals.king)):
        raise MalformedState("total scores must be non-negative integers")
    return _ValidatedStateParts(
        hands,
        tuple(current_cards),
        moves,
        completed_results,
        pending,
        totals,
    )


def _expected_deck(state: EngineState) -> tuple[EnginePlayer, tuple[str, ...]]:
    """Return the dealer/deck provenance authoritative for this state type."""

    if isinstance(state, SimulationEngineState):
        expected_deck = state.simulation_deck
        if not isinstance(expected_deck, tuple) or len(expected_deck) != len(CARD_IDS):
            raise MalformedState("a simulation deck must contain exactly 54 cards")
        for card_id in expected_deck:
            _validate_card_id(card_id, "simulation deck")
        if len(set(expected_deck)) != len(CARD_IDS):
            raise MalformedState("a simulation deck must contain every card exactly once")
        # Simulations retain the current dealer, so recover the round-one dealer
        # before validating the sampled deck's complete deal history.
        expected_dealer = (
            state.dealer if state.round_number % 2 == 1 else other_player(state.dealer)
        )
        return expected_dealer, expected_deck
    return initial_dealer(state.seed), shuffled_deck(state.seed)


def _validate_completed_deals(
    completed: tuple[EngineRoundResult, ...],
    expected_dealer: EnginePlayer,
    expected_deck: tuple[str, ...],
) -> None:
    """Verify archived rounds against their exact positions in the source deck."""

    # Re-derive each deal to reject histories inconsistent with the sealed deck.
    for round_index, result in enumerate(completed, start=1):
        dealer = expected_dealer if round_index % 2 == 1 else other_player(expected_dealer)
        if result.round_number != round_index or result.dealer is not dealer:
            raise MalformedState("completed-round order or dealer alternation is invalid")
        expected_deal = deal_round(expected_deck[(round_index - 1) * 9 :], dealer)
        if result.coffin[CENTER_GRID_INDEX] != expected_deal.center_card:
            raise MalformedState("a completed round has the wrong dealt center card")
        for player in EnginePlayer:
            if _result_hand(result, player) != expected_deal.hand_for(player):
                raise MalformedState("a completed round does not match its dealt hands")


def _validate_current_lifecycle(
    state: EngineState,
    parts: _ValidatedStateParts,
    expected_deck: tuple[str, ...],
) -> None:
    """Verify the active round's status, turn, deal, hands, stock, and pending result."""

    hands = parts.hands
    moves = parts.moves
    pending = parts.pending
    if state.status is EngineStatus.GAME_COMPLETE:
        if state.round_number != ROUNDS_PER_GAME or len(parts.completed) != ROUNDS_PER_GAME:
            raise MalformedState("a complete game must contain six completed rounds")
        if state.active_player is not None or state.stock or moves or pending is not None:
            raise MalformedState("a complete game contains active-round data")
        if any(card_id is not None for hand in (hands.queen, hands.king) for card_id in hand):
            raise MalformedState("a complete game contains hand cards")
        if any(card_id is not None for card_id in state.coffin):
            raise MalformedState("a complete game contains a current coffin")
        return

    expected_completed_count = state.round_number - 1
    if len(parts.completed) != expected_completed_count:
        raise MalformedState("completed-round count does not match the current round")
    if len(state.stock) != len(CARD_IDS) - 9 * state.round_number:
        raise MalformedState("stock length does not match the current round")
    if state.coffin[CENTER_GRID_INDEX] is None:
        raise MalformedState("the current coffin is missing its center card")
    if state.status is EngineStatus.PLAYING:
        if pending is not None or len(moves) >= MOVES_PER_ROUND:
            raise MalformedState("a playing round has completed-round data")
        expected_active = other_player(state.dealer) if len(moves) % 2 == 0 else state.dealer
        if state.active_player is not expected_active:
            raise MalformedState("active player does not match turn order")
    else:
        if state.active_player is not None or len(moves) != MOVES_PER_ROUND or pending is None:
            raise MalformedState("a completed round has an invalid lifecycle shape")
        if pending.round_number != state.round_number or pending.dealer is not state.dealer:
            raise MalformedState("pending result does not identify the current round")
        if pending.coffin != state.coffin or pending.moves != moves:
            raise MalformedState("pending result does not match the current round")

    used_by_player: dict[EnginePlayer, dict[int, str]] = {
        EnginePlayer.QUEEN: {},
        EnginePlayer.KING: {},
    }
    for move in moves:
        used_by_player[move.player][move.hand_slot] = move.card_id
    for player in EnginePlayer:
        hand = hands[player]
        reconstructed: list[str] = []
        for slot, card_id in enumerate(hand):
            moved_card = used_by_player[player].get(slot)
            # Each original slot is represented in exactly one place: the live
            # hand while held, or move history after it has been played.
            if (card_id is None) == (moved_card is None):
                raise MalformedState("each hand slot must be either held or played")
            reconstructed.append(card_id if card_id is not None else moved_card)  # type: ignore[arg-type]
        if tuple(reconstructed) != sort_card_ids(reconstructed):
            raise MalformedState("hand slots do not retain canonical card order")

    expected_deal = deal_round(expected_deck[(state.round_number - 1) * 9 :], state.dealer)
    if state.stock != expected_deal.remaining_stock:
        raise MalformedState("stock order does not match the deterministic deal")
    if state.coffin[CENTER_GRID_INDEX] != expected_deal.center_card:
        raise MalformedState("current center card does not match the deterministic deal")
    for player in EnginePlayer:
        reconstructed = tuple(
            card_id if card_id is not None else used_by_player[player][slot]
            for slot, card_id in enumerate(hands[player])
        )
        if reconstructed != expected_deal.hand_for(player):
            raise MalformedState("current hand slots do not match the deterministic deal")


def _validate_conservation_and_scores(
    state: EngineState, parts: _ValidatedStateParts
) -> None:
    """Verify unique ownership of all cards and recomputed cumulative scores."""

    # Every card must occupy exactly one live or archived location.
    owned_cards = list(state.stock)
    owned_cards.extend(parts.current_cards)
    for result in parts.completed:
        owned_cards.extend(result.coffin)
    if len(owned_cards) != len(CARD_IDS) or set(owned_cards) != set(CARD_IDS):
        raise MalformedState("the state does not conserve all 54 unique cards")

    # Totals include a pending result immediately after the round's final move,
    # before that result is archived by ``advance_after_round``.
    scored_results = parts.completed + ((parts.pending,) if parts.pending is not None else ())
    expected_totals = PlayerValues(
        queen=sum(result.round_scores.queen for result in scored_results),
        king=sum(result.round_scores.king for result in scored_results),
    )
    if parts.totals != expected_totals:
        raise MalformedState("total scores do not match recorded round scores")


def validate_state(state: EngineState) -> None:
    """Reject structural, deterministic-deal, lifecycle, and conservation errors."""

    _validate_state_header(state)
    parts = _validate_state_parts(state)
    expected_dealer, expected_deck = _expected_deck(state)
    _validate_completed_deals(parts.completed, expected_dealer, expected_deck)
    current_expected_dealer = (
        expected_dealer if state.round_number % 2 == 1 else other_player(expected_dealer)
    )
    if state.dealer is not current_expected_dealer:
        raise MalformedState("current dealer does not match round alternation")
    _validate_current_lifecycle(state, parts, expected_deck)
    _validate_conservation_and_scores(state, parts)

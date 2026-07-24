"""Pure, deterministic Dracula game rules and immutable engine state."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Generic, TypeVar

from dracula.cards import (
    CARD_IDS,
    CARD_INDEX_BY_ID,
    CARD_SCHEMA_VERSION,
    Card,
    card_by_id,
    sort_card_ids,
)
from dracula.randomness import (
    RANDOMNESS_SCHEMA_VERSION,
    derive_seed,
    seed_integer,
    shuffled,
)

ENGINE_VERSION = "dracula-engine-v1"
RULES_VERSION = "dracula-rules-v1"
SHUFFLE_NAMESPACE = "dracula-engine-shuffle-v1"
INITIAL_DEALER_NAMESPACE = "dracula-engine-initial-dealer-v1"

HAND_SIZE = 4
COFFIN_SIZE = 9
CENTER_GRID_INDEX = 4
MOVES_PER_ROUND = 8
ROUNDS_PER_GAME = 6


class EnginePlayer(StrEnum):
    QUEEN = "queen"
    KING = "king"


class EngineStatus(StrEnum):
    PLAYING = "playing"
    ROUND_COMPLETE = "round_complete"
    GAME_COMPLETE = "game_complete"


class LineOrientation(StrEnum):
    ROW = "row"
    COLUMN = "column"


class MultiplierReason(StrEnum):
    VAMPIRE = "vampire"
    NONE = "none"
    SUIT_PAIR = "suit_pair"
    SAME_COLOR = "same_color"
    SAME_SUIT = "same_suit"


class GameOutcomeReason(StrEnum):
    TOTAL_SCORE = "total_score"
    SIXTH_ROUND_SCORE = "sixth_round_score"
    TIE = "tie"


class RuleViolation(Exception):
    """Base class for rejected state transitions and rule inputs."""


class MalformedState(RuleViolation):
    pass


class WrongActivePlayer(RuleViolation):
    pass


class UnavailableHandSlot(RuleViolation):
    pass


class InvalidGridIndex(RuleViolation):
    pass


class OccupiedDestination(RuleViolation):
    pass


class NonAdjacentDestination(RuleViolation):
    pass


class InvalidLifecycleTransition(RuleViolation):
    pass


T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class PlayerValues(Generic[T]):
    queen: T
    king: T

    def __getitem__(self, player: EnginePlayer) -> T:
        player = EnginePlayer(player)
        return self.queen if player is EnginePlayer.QUEEN else self.king

    def updated(self, player: EnginePlayer, value: T) -> PlayerValues[T]:
        player = EnginePlayer(player)
        if player is EnginePlayer.QUEEN:
            return replace(self, queen=value)
        return replace(self, king=value)


Hand = tuple[str | None, str | None, str | None, str | None]
Coffin = tuple[
    str | None,
    str | None,
    str | None,
    str | None,
    str | None,
    str | None,
    str | None,
    str | None,
    str | None,
]


def other_player(player: EnginePlayer) -> EnginePlayer:
    player = EnginePlayer(player)
    return EnginePlayer.KING if player is EnginePlayer.QUEEN else EnginePlayer.QUEEN


@dataclass(frozen=True, slots=True)
class RoundDeal:
    dealer: EnginePlayer
    queen_hand: tuple[str, ...]
    king_hand: tuple[str, ...]
    center_card: str
    remaining_stock: tuple[str, ...]

    @property
    def non_dealer(self) -> EnginePlayer:
        return other_player(self.dealer)

    def hand_for(self, player: EnginePlayer) -> tuple[str, ...]:
        player = EnginePlayer(player)
        return self.queen_hand if player is EnginePlayer.QUEEN else self.king_hand


@dataclass(frozen=True, slots=True)
class EngineMove:
    player: EnginePlayer
    hand_slot: int
    global_grid_index: int


@dataclass(frozen=True, slots=True)
class EnginePlayedMove:
    player: EnginePlayer
    card_id: str
    hand_slot: int
    global_grid_index: int
    turn_number: int


@dataclass(frozen=True, slots=True)
class LineScore:
    orientation: LineOrientation
    line_index: int
    cards: tuple[str, str, str]
    values: tuple[int, int, int]
    base_value: int
    multiplier: int
    multiplier_reason: MultiplierReason
    total: int


@dataclass(frozen=True, slots=True)
class EngineRoundResult:
    round_number: int
    dealer: EnginePlayer
    coffin: tuple[str, str, str, str, str, str, str, str, str]
    moves: tuple[EnginePlayedMove, ...]
    line_scores: PlayerValues[tuple[LineScore, LineScore, LineScore]]
    round_scores: PlayerValues[int]


@dataclass(frozen=True, slots=True)
class EngineState:
    seed: str
    status: EngineStatus
    round_number: int
    dealer: EnginePlayer
    active_player: EnginePlayer | None
    stock: tuple[str, ...]
    hands: PlayerValues[Hand]
    coffin: Coffin
    current_round_moves: tuple[EnginePlayedMove, ...]
    pending_round_result: EngineRoundResult | None
    completed_rounds: tuple[EngineRoundResult, ...]
    total_scores: PlayerValues[int]


@dataclass(frozen=True, slots=True)
class SimulationEngineState(EngineState):
    """Engine state whose private deck was sampled from a player's belief."""

    simulation_deck: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class EngineTransition:
    previous_state: EngineState
    state: EngineState
    move: EngineMove
    played_move: EnginePlayedMove
    round_result: EngineRoundResult | None
    state_fingerprint: str


@dataclass(frozen=True, slots=True)
class GameOutcome:
    winner: EnginePlayer | None
    reason: GameOutcomeReason
    total_scores: PlayerValues[int]
    sixth_round_scores: PlayerValues[int]


def shuffled_deck(game_seed: str) -> tuple[str, ...]:
    shuffle_seed = derive_seed(SHUFFLE_NAMESPACE, game_seed)
    return shuffled(CARD_IDS, shuffle_seed)


def initial_dealer(game_seed: str) -> EnginePlayer:
    dealer_seed = derive_seed(INITIAL_DEALER_NAMESPACE, game_seed)
    return EnginePlayer.QUEEN if seed_integer(dealer_seed) % 2 == 0 else EnginePlayer.KING


def deal_round(stock: Sequence[str], dealer: EnginePlayer) -> RoundDeal:
    dealer = EnginePlayer(dealer)
    stock_cards = tuple(stock)
    if len(stock_cards) < 9:
        raise ValueError("a round requires at least nine stock cards")
    for card_id in stock_cards:
        card_by_id(card_id)
    if len(set(stock_cards)) != len(stock_cards):
        raise ValueError("stock cannot contain duplicate cards")

    non_dealer = other_player(dealer)
    non_dealer_hand = sort_card_ids(stock_cards[0:2] + stock_cards[4:6])
    dealer_hand = sort_card_ids(stock_cards[2:4] + stock_cards[6:8])
    hands = PlayerValues(
        queen=non_dealer_hand if non_dealer is EnginePlayer.QUEEN else dealer_hand,
        king=non_dealer_hand if non_dealer is EnginePlayer.KING else dealer_hand,
    )
    return RoundDeal(
        dealer=dealer,
        queen_hand=hands.queen,
        king_hand=hands.king,
        center_card=stock_cards[8],
        remaining_stock=stock_cards[9:],
    )


def create_initial_deal(game_seed: str) -> RoundDeal:
    return deal_round(shuffled_deck(game_seed), initial_dealer(game_seed))


def _as_hand(cards: tuple[str, ...]) -> Hand:
    if len(cards) != HAND_SIZE:
        raise ValueError("a hand must contain four cards")
    return cards  # type: ignore[return-value]


def _starting_coffin(center_card: str) -> Coffin:
    return (None, None, None, None, center_card, None, None, None, None)


def create_game(seed: str) -> EngineState:
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
            queen=_as_hand(deal.queen_hand),
            king=_as_hand(deal.king_hand),
        ),
        coffin=_starting_coffin(deal.center_card),
        current_round_moves=(),
        pending_round_result=None,
        completed_rounds=(),
        total_scores=PlayerValues(queen=0, king=0),
    )
    validate_state(state)
    return state


def _orthogonally_adjacent(first: int, second: int) -> bool:
    first_row, first_column = divmod(first, 3)
    second_row, second_column = divmod(second, 3)
    return abs(first_row - second_row) + abs(first_column - second_column) == 1


def _has_occupied_neighbor(coffin: Sequence[str | None], grid_index: int) -> bool:
    return any(
        card_id is not None and _orthogonally_adjacent(grid_index, occupied_index)
        for occupied_index, card_id in enumerate(coffin)
    )


def _legal_moves_for_active_player(
    state: EngineState, player: EnginePlayer
) -> tuple[EngineMove, ...]:
    if state.status is not EngineStatus.PLAYING:
        raise InvalidLifecycleTransition("moves are available only while a round is playing")
    try:
        player = EnginePlayer(player)
    except (TypeError, ValueError) as error:
        raise WrongActivePlayer(f"unknown player: {player!r}") from error
    if player is not state.active_player:
        raise WrongActivePlayer(f"{player.value} is not the active player")

    destinations = tuple(
        grid_index
        for grid_index, card_id in enumerate(state.coffin)
        if card_id is None and _has_occupied_neighbor(state.coffin, grid_index)
    )
    return tuple(
        EngineMove(player, hand_slot, grid_index)
        for hand_slot, card_id in enumerate(state.hands[player])
        if card_id is not None
        for grid_index in destinations
    )


def legal_moves(state: EngineState, player: EnginePlayer) -> tuple[EngineMove, ...]:
    validate_state(state)
    return _legal_moves_for_active_player(state, player)


def legal_simulation_moves(
    state: SimulationEngineState, player: EnginePlayer
) -> tuple[EngineMove, ...]:
    """Return legal moves without repeating whole-deck validation."""

    if not isinstance(state, SimulationEngineState):
        raise TypeError("fast simulation legality requires SimulationEngineState")
    return _legal_moves_for_active_player(state, player)


def _line_values(cards: Sequence[Card], orientation: LineOrientation) -> tuple[int, int, int]:
    if orientation is LineOrientation.ROW:
        return tuple(card.horizontal_value for card in cards)  # type: ignore[return-value]
    return tuple(card.vertical_value for card in cards)  # type: ignore[return-value]


def score_line(
    card_ids: Sequence[str],
    orientation: LineOrientation,
    line_index: int = 0,
) -> LineScore:
    try:
        orientation = LineOrientation(orientation)
    except (TypeError, ValueError) as error:
        raise ValueError(f"unknown line orientation: {orientation!r}") from error
    if type(line_index) is not int or not 0 <= line_index < 3:
        raise ValueError("line index must be between 0 and 2")
    if len(card_ids) != 3:
        raise ValueError("a line must contain exactly three cards")

    cards = tuple(card_by_id(card_id) for card_id in card_ids)
    ids = tuple(card.card_id for card in cards)
    values = _line_values(cards, orientation)
    base_value = sum(values)
    if any(card.is_vampire for card in cards):
        multiplier = 0
        reason = MultiplierReason.VAMPIRE
    else:
        suit_counts = Counter(card.suit for card in cards)
        colors = {card.color for card in cards}
        if len(suit_counts) == 1:
            multiplier = 5
            reason = MultiplierReason.SAME_SUIT
        elif len(colors) == 1:
            multiplier = 3
            reason = MultiplierReason.SAME_COLOR
        elif max(suit_counts.values()) >= 2:
            multiplier = 2
            reason = MultiplierReason.SUIT_PAIR
        else:
            multiplier = 1
            reason = MultiplierReason.NONE
    return LineScore(
        orientation=orientation,
        line_index=line_index,
        cards=ids,  # type: ignore[arg-type]
        values=values,
        base_value=base_value,
        multiplier=multiplier,
        multiplier_reason=reason,
        total=base_value * multiplier,
    )


def _line_cards(
    coffin: Sequence[str], indices: tuple[int, int, int]
) -> tuple[str, str, str]:
    return coffin[indices[0]], coffin[indices[1]], coffin[indices[2]]


def score_coffin(
    coffin: Sequence[str],
) -> PlayerValues[tuple[LineScore, LineScore, LineScore]]:
    if len(coffin) != COFFIN_SIZE:
        raise ValueError("a completed coffin must contain nine cards")
    for card_id in coffin:
        card_by_id(card_id)
    rows = ((0, 1, 2), (3, 4, 5), (6, 7, 8))
    columns = ((0, 3, 6), (1, 4, 7), (2, 5, 8))
    queen_lines = tuple(
        score_line(_line_cards(coffin, indices), LineOrientation.ROW, line_index)
        for line_index, indices in enumerate(rows)
    )
    king_lines = tuple(
        score_line(_line_cards(coffin, indices), LineOrientation.COLUMN, line_index)
        for line_index, indices in enumerate(columns)
    )
    return PlayerValues(queen=queen_lines, king=king_lines)  # type: ignore[arg-type]


def resolve_round_scores(
    queen_line_scores: Sequence[int], king_line_scores: Sequence[int]
) -> PlayerValues[int]:
    if len(queen_line_scores) != 3 or len(king_line_scores) != 3:
        raise ValueError("each player must have exactly three line scores")
    if any(type(value) is not int or value < 0 for value in (*queen_line_scores, *king_line_scores)):
        raise ValueError("line scores must be non-negative integers")
    queen_sorted = sorted(queen_line_scores, reverse=True)
    king_sorted = sorted(king_line_scores, reverse=True)
    selected_index = 2
    for index in range(2):
        if queen_sorted[index] != king_sorted[index]:
            selected_index = index
            break
    return PlayerValues(queen=queen_sorted[selected_index], king=king_sorted[selected_index])


def _make_round_result(state: EngineState) -> EngineRoundResult:
    if any(card_id is None for card_id in state.coffin):
        raise MalformedState("a round result requires a completed coffin")
    coffin = state.coffin  # type: ignore[assignment]
    line_scores = score_coffin(coffin)
    round_scores = resolve_round_scores(
        tuple(line.total for line in line_scores.queen),
        tuple(line.total for line in line_scores.king),
    )
    return EngineRoundResult(
        round_number=state.round_number,
        dealer=state.dealer,
        coffin=coffin,
        moves=state.current_round_moves,
        line_scores=line_scores,
        round_scores=round_scores,
    )


def _apply_move_state(
    state: EngineState, move: EngineMove
) -> tuple[EngineState, EnginePlayedMove, EngineRoundResult | None]:
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
    if not _has_occupied_neighbor(state.coffin, move.global_grid_index):
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
        round_result = _make_round_result(next_state)
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
    validate_state(state)
    if isinstance(state, SimulationEngineState):
        raise InvalidLifecycleTransition("round-local simulation cannot advance rounds")
    if state.status is not EngineStatus.ROUND_COMPLETE or state.pending_round_result is None:
        raise InvalidLifecycleTransition("only a completed round can be advanced")

    completed_rounds = state.completed_rounds + (state.pending_round_result,)
    if state.round_number == ROUNDS_PER_GAME:
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
            queen=_as_hand(deal.queen_hand), king=_as_hand(deal.king_hand)
        ),
        coffin=_starting_coffin(deal.center_card),
        current_round_moves=(),
        pending_round_result=None,
        completed_rounds=completed_rounds,
        total_scores=state.total_scores,
    )
    validate_state(next_state)
    return next_state


def resolve_game_outcome(
    total_scores: PlayerValues[int], sixth_round_scores: PlayerValues[int]
) -> GameOutcome:
    for label, scores in (("total", total_scores), ("sixth-round", sixth_round_scores)):
        if not isinstance(scores, PlayerValues) or any(
            type(value) is not int or value < 0 for value in (scores.queen, scores.king)
        ):
            raise ValueError(f"{label} scores must be non-negative player values")
    if total_scores.queen != total_scores.king:
        winner = (
            EnginePlayer.QUEEN
            if total_scores.queen > total_scores.king
            else EnginePlayer.KING
        )
        reason = GameOutcomeReason.TOTAL_SCORE
    elif sixth_round_scores.queen != sixth_round_scores.king:
        winner = (
            EnginePlayer.QUEEN
            if sixth_round_scores.queen > sixth_round_scores.king
            else EnginePlayer.KING
        )
        reason = GameOutcomeReason.SIXTH_ROUND_SCORE
    else:
        winner = None
        reason = GameOutcomeReason.TIE
    return GameOutcome(winner, reason, total_scores, sixth_round_scores)


def derive_game_outcome(state: EngineState) -> GameOutcome:
    validate_state(state)
    if state.status is not EngineStatus.GAME_COMPLETE:
        raise InvalidLifecycleTransition("a final outcome requires a completed game")
    return resolve_game_outcome(state.total_scores, state.completed_rounds[-1].round_scores)


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
        if not any(_orthogonally_adjacent(grid_index, prior) for prior in occupied):
            raise MalformedState(f"{label} contains a non-adjacent placement")
        if coffin[grid_index] != move.card_id:
            raise MalformedState(f"{label} move does not match the coffin")
        occupied.add(grid_index)
    return moves


def _validate_line_score(line: object, expected: LineScore, label: str) -> None:
    if line != expected:
        raise MalformedState(f"{label} does not match rules-defined scoring")


def _validate_round_result(result: object, label: str) -> EngineRoundResult:
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
    expected_round_scores = resolve_round_scores(
        tuple(line.total for line in expected_lines.queen),
        tuple(line.total for line in expected_lines.king),
    )
    if result.round_scores != expected_round_scores:
        raise MalformedState(f"{label} round scores do not match its line scores")
    return result


def _result_hand(result: EngineRoundResult, player: EnginePlayer) -> tuple[str, ...]:
    cards_by_slot = {
        move.hand_slot: move.card_id for move in result.moves if move.player is player
    }
    if set(cards_by_slot) != set(range(HAND_SIZE)):
        raise MalformedState("a completed round does not consume every hand slot")
    return tuple(cards_by_slot[slot] for slot in range(HAND_SIZE))


def validate_state(state: EngineState) -> None:
    """Reject structurally invalid states and impossible lifecycle combinations."""

    if not isinstance(state, EngineState):
        raise MalformedState("state must be an EngineState")
    if not isinstance(state.seed, str):
        raise MalformedState("seed must be a string")
    try:
        derive_seed(SHUFFLE_NAMESPACE, state.seed)
    except ValueError as error:
        raise MalformedState("seed contains an invalid separator") from error
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

    if isinstance(state, SimulationEngineState):
        expected_deck = state.simulation_deck
        if not isinstance(expected_deck, tuple) or len(expected_deck) != len(CARD_IDS):
            raise MalformedState("a simulation deck must contain exactly 54 cards")
        for card_id in expected_deck:
            _validate_card_id(card_id, "simulation deck")
        if len(set(expected_deck)) != len(CARD_IDS):
            raise MalformedState("a simulation deck must contain every card exactly once")
        expected_dealer = (
            state.dealer
            if state.round_number % 2 == 1
            else other_player(state.dealer)
        )
    else:
        expected_dealer = initial_dealer(state.seed)
        expected_deck = shuffled_deck(state.seed)
    for round_index, result in enumerate(completed_results, start=1):
        dealer = expected_dealer if round_index % 2 == 1 else other_player(expected_dealer)
        if result.round_number != round_index or result.dealer is not dealer:
            raise MalformedState("completed-round order or dealer alternation is invalid")
        expected_deal = deal_round(expected_deck[(round_index - 1) * 9 :], dealer)
        if result.coffin[CENTER_GRID_INDEX] != expected_deal.center_card:
            raise MalformedState("a completed round has the wrong dealt center card")
        for player in EnginePlayer:
            if _result_hand(result, player) != expected_deal.hand_for(player):
                raise MalformedState("a completed round does not match its dealt hands")
    current_expected_dealer = (
        expected_dealer if state.round_number % 2 == 1 else other_player(expected_dealer)
    )
    if state.dealer is not current_expected_dealer:
        raise MalformedState("current dealer does not match round alternation")

    if state.status is EngineStatus.GAME_COMPLETE:
        if state.round_number != ROUNDS_PER_GAME or len(completed_results) != ROUNDS_PER_GAME:
            raise MalformedState("a complete game must contain six completed rounds")
        if state.active_player is not None or state.stock or moves or pending is not None:
            raise MalformedState("a complete game contains active-round data")
        if any(card_id is not None for hand in (hands.queen, hands.king) for card_id in hand):
            raise MalformedState("a complete game contains hand cards")
        if any(card_id is not None for card_id in state.coffin):
            raise MalformedState("a complete game contains a current coffin")
    else:
        expected_completed_count = state.round_number - 1
        if len(completed_results) != expected_completed_count:
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
                if (card_id is None) == (moved_card is None):
                    raise MalformedState("each hand slot must be either held or played")
                reconstructed.append(card_id if card_id is not None else moved_card)  # type: ignore[arg-type]
            if tuple(reconstructed) != sort_card_ids(reconstructed):
                raise MalformedState("hand slots do not retain canonical card order")

        expected_deal = deal_round(
            expected_deck[(state.round_number - 1) * 9 :], state.dealer
        )
        if state.stock != expected_deal.remaining_stock:
            raise MalformedState("stock order does not match the deterministic deal")
        if state.coffin[CENTER_GRID_INDEX] != expected_deal.center_card:
            raise MalformedState("current center card does not match the deterministic deal")
        for player in EnginePlayer:
            reconstructed = tuple(
                card_id
                if card_id is not None
                else used_by_player[player][slot]
                for slot, card_id in enumerate(hands[player])
            )
            if reconstructed != expected_deal.hand_for(player):
                raise MalformedState("current hand slots do not match the deterministic deal")

    owned_cards = list(state.stock)
    owned_cards.extend(current_cards)
    for result in completed_results:
        owned_cards.extend(result.coffin)
    if len(owned_cards) != len(CARD_IDS) or set(owned_cards) != set(CARD_IDS):
        raise MalformedState("the state does not conserve all 54 unique cards")

    scored_results = completed_results + ((pending,) if pending is not None else ())
    expected_totals = PlayerValues(
        queen=sum(result.round_scores.queen for result in scored_results),
        king=sum(result.round_scores.king for result in scored_results),
    )
    if totals != expected_totals:
        raise MalformedState("total scores do not match recorded round scores")


def _player_values_json(values: PlayerValues[object], encode: object) -> dict[str, object]:
    encoder = encode if callable(encode) else lambda value: value
    return {
        EnginePlayer.QUEEN.value: encoder(values.queen),
        EnginePlayer.KING.value: encoder(values.king),
    }


def _played_move_json(move: EnginePlayedMove) -> dict[str, object]:
    return {
        "player": move.player.value,
        "card_id": move.card_id,
        "hand_slot": move.hand_slot,
        "global_grid_index": move.global_grid_index,
        "turn_number": move.turn_number,
    }


def _line_score_json(line: LineScore) -> dict[str, object]:
    return {
        "orientation": line.orientation.value,
        "line_index": line.line_index,
        "cards": list(line.cards),
        "values": list(line.values),
        "base_value": line.base_value,
        "multiplier": line.multiplier,
        "multiplier_reason": line.multiplier_reason.value,
        "total": line.total,
    }


def _round_result_json(result: EngineRoundResult) -> dict[str, object]:
    return {
        "round_number": result.round_number,
        "dealer": result.dealer.value,
        "coffin": list(result.coffin),
        "moves": [_played_move_json(move) for move in result.moves],
        "line_scores": _player_values_json(
            result.line_scores,
            lambda lines: [_line_score_json(line) for line in lines],
        ),
        "round_scores": _player_values_json(result.round_scores, lambda value: value),
    }


def canonical_state_data(state: EngineState) -> dict[str, object]:
    validate_state(state)
    return {
        "rules_version": RULES_VERSION,
        "card_schema_version": CARD_SCHEMA_VERSION,
        "engine_version": ENGINE_VERSION,
        "randomness_schema_version": RANDOMNESS_SCHEMA_VERSION,
        "state": {
            "seed": state.seed,
            "status": state.status.value,
            "round_number": state.round_number,
            "dealer": state.dealer.value,
            "active_player": None if state.active_player is None else state.active_player.value,
            "stock": list(state.stock),
            "hands": _player_values_json(state.hands, lambda hand: list(hand)),
            "coffin": list(state.coffin),
            "current_round_moves": [
                _played_move_json(move) for move in state.current_round_moves
            ],
            "pending_round_result": (
                None
                if state.pending_round_result is None
                else _round_result_json(state.pending_round_result)
            ),
            "completed_rounds": [
                _round_result_json(result) for result in state.completed_rounds
            ],
            "total_scores": _player_values_json(state.total_scores, lambda value: value),
        },
    }


def canonical_state_json(state: EngineState) -> str:
    return json.dumps(
        canonical_state_data(state),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=False,
    )


def state_fingerprint(state: EngineState) -> str:
    return hashlib.sha256(canonical_state_json(state).encode("utf-8")).hexdigest()

"""Project private engine state into an immutable actor-visible state.

The projection retains public history and the acting player's own cards while
erasing the opponent hand, stock order, engine seed, and sampled-world details.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import cast

from dracula.bridge import (
    COFFIN_POSITION_COUNT,
    HAND_SLOT_COUNT,
    POLICY_GRID_INDICES,
    player_relative_grid_index,
)
from dracula.cards import CARD_COUNT, CARD_IDS, CARD_INDEX_BY_ID, sort_card_ids
from dracula.engine import (
    CENTER_GRID_INDEX,
    MOVES_PER_ROUND,
    ROUNDS_PER_GAME,
    EnginePlayedMove,
    EnginePlayer,
    EngineRoundResult,
    EngineState,
    EngineStatus,
    PlayerValues,
    SimulationEngineState,
    other_player,
    score_coffin,
    validate_state,
)
from dracula.engine_types import empty_adjacent_grid_indices, orthogonally_adjacent
from dracula.scoring import round_scores_from_lines

# Rows are stable hand slots; columns are the eight non-center policy positions.
LegalMask = tuple[
    tuple[bool, bool, bool, bool, bool, bool, bool, bool],
    tuple[bool, bool, bool, bool, bool, bool, bool, bool],
    tuple[bool, bool, bool, bool, bool, bool, bool, bool],
    tuple[bool, bool, bool, bool, bool, bool, bool, bool],
]


class InformationContractViolation(ValueError):
    """A player-visible state or public history is inconsistent."""


@dataclass(frozen=True, slots=True)
class PublicPlayedMove:
    """Played-card fact with the private original hand slot removed."""

    player: EnginePlayer
    card_id: str
    global_grid_index: int
    turn_number: int


@dataclass(frozen=True, slots=True)
class PublicRoundRecord:
    """Completed round facts visible to both players."""

    round_number: int
    dealer: EnginePlayer
    coffin: tuple[str, str, str, str, str, str, str, str, str]
    moves: tuple[PublicPlayedMove, ...]
    round_scores: PlayerValues[int]


@dataclass(frozen=True, slots=True)
class PublicGameHistory:
    """Validated public history for an active round, with no hidden locations."""

    status: EngineStatus
    round_number: int
    dealer: EnginePlayer
    active_player: EnginePlayer
    total_scores: PlayerValues[int]
    completed_rounds: tuple[PublicRoundRecord, ...]
    current_coffin: tuple[str | None, ...]
    current_round_moves: tuple[PublicPlayedMove, ...]

    def __post_init__(self) -> None:
        _validate_public_history(self)


@dataclass(frozen=True, slots=True)
class SearchInformationState:
    """Complete decision information available to exactly one active player."""

    player: EnginePlayer
    round_number: int
    dealer: EnginePlayer
    active_player: EnginePlayer
    turn_number: int
    total_scores: PlayerValues[int]
    completed_rounds: tuple[PublicRoundRecord, ...]
    own_hand: tuple[str | None, str | None, str | None, str | None]
    coffin: tuple[str | None, ...]
    current_round_moves: tuple[PublicPlayedMove, ...]
    played_card_ids: tuple[str, ...]
    unseen_card_ids: tuple[str, ...]
    opponent_remaining_count: int
    stock_count: int
    legal_mask: LegalMask

    def __post_init__(self) -> None:
        _validate_information_state(self)


def _card_ids(values: tuple[str | None, ...]) -> tuple[str, ...]:
    return tuple(value for value in values if value is not None)


def _validate_card_ids(values: tuple[str, ...], label: str) -> None:
    if any(value not in CARD_INDEX_BY_ID for value in values):
        raise InformationContractViolation(f"{label} contains an unknown card")
    if len(values) != len(set(values)):
        raise InformationContractViolation(f"{label} contains duplicate cards")


def _validate_public_moves(
    moves: tuple[PublicPlayedMove, ...],
    dealer: EnginePlayer,
    coffin: tuple[str | None, ...],
    *,
    complete: bool,
) -> None:
    """Validate public turn order and board growth without private hand slots."""

    expected_count = MOVES_PER_ROUND if complete else range(MOVES_PER_ROUND)
    if complete:
        valid_count = len(moves) == expected_count
    else:
        valid_count = len(moves) in expected_count
    if not isinstance(moves, tuple) or not valid_count:
        qualifier = "eight" if complete else "fewer than eight"
        raise InformationContractViolation(f"public history must contain {qualifier} moves")

    occupied = {CENTER_GRID_INDEX}
    non_dealer = other_player(dealer)
    seen_cards: set[str] = set()
    for index, move in enumerate(moves):
        if not isinstance(move, PublicPlayedMove):
            raise InformationContractViolation("public history contains an invalid move")
        expected_player = non_dealer if index % 2 == 0 else dealer
        if move.player is not expected_player or move.turn_number != index + 1:
            raise InformationContractViolation("public move order or actor is invalid")
        if move.card_id not in CARD_INDEX_BY_ID or move.card_id in seen_cards:
            raise InformationContractViolation("public moves contain invalid cards")
        grid_index = move.global_grid_index
        if type(grid_index) is not int or not 0 <= grid_index < COFFIN_POSITION_COUNT:
            raise InformationContractViolation("public move has an invalid position")
        if grid_index in occupied or not any(
            orthogonally_adjacent(grid_index, prior) for prior in occupied
        ):
            raise InformationContractViolation("public move history is not a legal coffin growth")
        if coffin[grid_index] != move.card_id:
            raise InformationContractViolation("public move does not match the coffin")
        occupied.add(grid_index)
        seen_cards.add(move.card_id)


def _validate_public_round(record: PublicRoundRecord) -> None:
    """Verify one public round's cards, moves, dealer, and recomputed scores."""

    if not isinstance(record, PublicRoundRecord):
        raise InformationContractViolation("completed history contains an invalid round")
    if type(record.round_number) is not int or not 1 <= record.round_number <= ROUNDS_PER_GAME:
        raise InformationContractViolation("public round number is invalid")
    if not isinstance(record.dealer, EnginePlayer):
        raise InformationContractViolation("public round dealer is invalid")
    if not isinstance(record.coffin, tuple) or len(record.coffin) != COFFIN_POSITION_COUNT:
        raise InformationContractViolation("public round coffin must contain nine cards")
    _validate_card_ids(record.coffin, "public round coffin")
    _validate_public_moves(record.moves, record.dealer, record.coffin, complete=True)
    if record.coffin[CENTER_GRID_INDEX] in {move.card_id for move in record.moves}:
        raise InformationContractViolation("public round center cannot also be a move")
    if record.round_scores != round_scores_from_lines(
        score_coffin(record.coffin)
    ):
        raise InformationContractViolation("public round scores do not match its coffin")


def _validate_public_history(history: PublicGameHistory) -> None:
    """Verify active lifecycle, archived rounds, totals, and public card uniqueness."""

    _validate_public_history_fields(
        status=history.status,
        round_number=history.round_number,
        dealer=history.dealer,
        active_player=history.active_player,
        total_scores=history.total_scores,
        completed_rounds=history.completed_rounds,
        current_coffin=history.current_coffin,
        current_round_moves=history.current_round_moves,
    )


def _validate_public_history_fields(
    *,
    status: EngineStatus,
    round_number: int,
    dealer: EnginePlayer,
    active_player: EnginePlayer,
    total_scores: PlayerValues[int],
    completed_rounds: tuple[PublicRoundRecord, ...],
    current_coffin: tuple[str | None, ...],
    current_round_moves: tuple[PublicPlayedMove, ...],
) -> None:
    """Validate public-history fields shared by both visible-state records."""

    if status is not EngineStatus.PLAYING:
        raise InformationContractViolation("search information requires an active round")
    if type(round_number) is not int or not 1 <= round_number <= ROUNDS_PER_GAME:
        raise InformationContractViolation("current public round number is invalid")
    if not isinstance(dealer, EnginePlayer) or not isinstance(active_player, EnginePlayer):
        raise InformationContractViolation("public roles are invalid")
    if not isinstance(completed_rounds, tuple) or len(completed_rounds) != round_number - 1:
        raise InformationContractViolation("completed public rounds do not match current round")
    for index, record in enumerate(completed_rounds, start=1):
        _validate_public_round(record)
        if record.round_number != index:
            raise InformationContractViolation("completed public rounds are out of order")
        expected_dealer = (
            dealer if (round_number - index) % 2 == 0 else other_player(dealer)
        )
        if record.dealer is not expected_dealer:
            raise InformationContractViolation("public dealer alternation is invalid")
    expected_totals = PlayerValues(
        queen=sum(record.round_scores.queen for record in completed_rounds),
        king=sum(record.round_scores.king for record in completed_rounds),
    )
    if total_scores != expected_totals:
        raise InformationContractViolation("public totals do not match completed rounds")
    if not isinstance(current_coffin, tuple) or len(current_coffin) != COFFIN_POSITION_COUNT:
        raise InformationContractViolation("current public coffin must contain nine positions")
    current_cards = _card_ids(current_coffin)
    _validate_card_ids(current_cards, "current public coffin")
    if current_coffin[CENTER_GRID_INDEX] is None:
        raise InformationContractViolation("current public coffin is missing its center")
    _validate_public_moves(
        current_round_moves,
        dealer,
        current_coffin,
        complete=False,
    )
    expected_active = (
        other_player(dealer) if len(current_round_moves) % 2 == 0 else dealer
    )
    if active_player is not expected_active:
        raise InformationContractViolation("active player does not match public turn order")

    public_cards = [card for record in completed_rounds for card in record.coffin]
    public_cards.extend(current_cards)
    if len(public_cards) != len(set(public_cards)):
        raise InformationContractViolation("public history reuses a physical card")


def _public_move(move: EnginePlayedMove) -> PublicPlayedMove:
    return PublicPlayedMove(
        player=move.player,
        card_id=move.card_id,
        global_grid_index=move.global_grid_index,
        turn_number=move.turn_number,
    )


def _public_round(result: EngineRoundResult) -> PublicRoundRecord:
    return PublicRoundRecord(
        round_number=result.round_number,
        dealer=result.dealer,
        coffin=result.coffin,
        moves=tuple(_public_move(move) for move in result.moves),
        round_scores=result.round_scores,
    )


def _public_history_from_validated_engine(state: EngineState) -> PublicGameHistory:
    if state.status is not EngineStatus.PLAYING or state.active_player is None:
        raise InformationContractViolation("search information requires the active player")
    return PublicGameHistory(
        status=state.status,
        round_number=state.round_number,
        dealer=state.dealer,
        active_player=state.active_player,
        total_scores=state.total_scores,
        completed_rounds=tuple(_public_round(result) for result in state.completed_rounds),
        current_coffin=state.coffin,
        current_round_moves=tuple(_public_move(move) for move in state.current_round_moves),
    )


def public_history_from_engine(state: EngineState) -> PublicGameHistory:
    """Project public history without returning any private engine location."""

    validate_state(state)
    return _public_history_from_validated_engine(state)


def _transpose_coffin_for_player(
    coffin: tuple[str | None, ...], player: EnginePlayer
) -> tuple[str | None, ...]:
    """Transpose a coffin between global and player-relative scoring frames."""

    result: list[str | None] = [None] * COFFIN_POSITION_COUNT
    for global_index, card_id in enumerate(coffin):
        result[player_relative_grid_index(player, global_index)] = card_id
    return tuple(result)


def information_state_from_engine(
    state: EngineState, player: EnginePlayer | None = None
) -> SearchInformationState:
    """Erase private engine locations into the active actor's typed view."""

    validate_state(state)
    selected = state.active_player if player is None else EnginePlayer(player)
    if selected is None or state.active_player is not selected:
        raise InformationContractViolation(
            "search information requires the active player"
        )
    return _information_state_from_validated_engine(state, selected)


def information_state_from_simulation(
    state: SimulationEngineState,
) -> SearchInformationState:
    """Give a simulated actor the same private-data erasure used at the root."""

    if not isinstance(state, SimulationEngineState) or state.active_player is None:
        raise InformationContractViolation(
            "simulation information requires an active sampled state"
        )
    # Search transitions preserve the sampled state's engine invariants. Avoid
    # repeating whole-deck validation at every simulated decision.
    return _information_state_from_validated_engine(state, state.active_player)


def _information_state_from_validated_engine(
    state: EngineState, player: EnginePlayer
) -> SearchInformationState:
    """Construct the complete decision state visible to the acting player."""

    own_hand = state.hands[player]
    completed_rounds = tuple(_public_round(result) for result in state.completed_rounds)
    current_round_moves = tuple(_public_move(move) for move in state.current_round_moves)
    relative_coffin = _transpose_coffin_for_player(state.coffin, player)
    completed_cards = tuple(
        card_id for result in state.completed_rounds for card_id in result.coffin
    )
    # One global order keeps information fingerprints and paired-destination
    # seed choices stable across completed and current rounds.
    played = sort_card_ids((*completed_cards, *_card_ids(state.coffin)))
    own_cards = set(_card_ids(own_hand))
    visible = set(played) | own_cards
    unseen = tuple(card_id for card_id in CARD_IDS if card_id not in visible)
    opponent = other_player(player)
    opponent_remaining = HAND_SLOT_COUNT - sum(
        move.player is opponent for move in state.current_round_moves
    )
    return SearchInformationState(
        player=player,
        round_number=state.round_number,
        dealer=state.dealer,
        active_player=player,
        turn_number=len(state.current_round_moves) + 1,
        total_scores=state.total_scores,
        completed_rounds=completed_rounds,
        own_hand=own_hand,
        coffin=relative_coffin,
        current_round_moves=current_round_moves,
        played_card_ids=played,
        unseen_card_ids=unseen,
        opponent_remaining_count=opponent_remaining,
        stock_count=CARD_COUNT - COFFIN_POSITION_COUNT * state.round_number,
        legal_mask=_legal_mask_from_visible_state(own_hand, relative_coffin),
    )


def _legal_mask_from_visible_state(
    own_hand: tuple[str | None, str | None, str | None, str | None],
    coffin: tuple[str | None, ...],
) -> LegalMask:
    """Derive action legality from only the actor's visible hand and coffin."""

    destinations = set(empty_adjacent_grid_indices(coffin))
    return cast(
        LegalMask,
        tuple(
            tuple(
                own_hand[hand_slot] is not None and grid_index in destinations
                for grid_index in POLICY_GRID_INDICES
            )
            for hand_slot in range(HAND_SLOT_COUNT)
        ),
    )


def _validate_information_state(state: SearchInformationState) -> None:
    """Verify actor ownership, public history, card partition, counts, and legality."""

    if not isinstance(state.player, EnginePlayer) or state.active_player is not state.player:
        raise InformationContractViolation("information state must belong to its active player")
    if not isinstance(state.dealer, EnginePlayer):
        raise InformationContractViolation("information dealer is invalid")
    if type(state.turn_number) is not int or not 1 <= state.turn_number <= MOVES_PER_ROUND:
        raise InformationContractViolation("information turn number is invalid")
    # Queen's mapping is the identity and King's transpose is self-inverse.
    global_coffin = _transpose_coffin_for_player(state.coffin, state.player)
    _validate_public_history_fields(
        status=EngineStatus.PLAYING,
        round_number=state.round_number,
        dealer=state.dealer,
        active_player=state.active_player,
        total_scores=state.total_scores,
        completed_rounds=state.completed_rounds,
        current_coffin=global_coffin,
        current_round_moves=state.current_round_moves,
    )
    if state.turn_number != len(state.current_round_moves) + 1:
        raise InformationContractViolation("information turn and public history disagree")
    if not isinstance(state.own_hand, tuple) or len(state.own_hand) != HAND_SLOT_COUNT:
        raise InformationContractViolation("own hand must retain four stable slots")
    own_cards = _card_ids(state.own_hand)
    _validate_card_ids(own_cards, "own hand")
    if own_cards != sort_card_ids(own_cards):
        raise InformationContractViolation("own hand cards are not in canonical slot order")
    if not isinstance(state.coffin, tuple) or len(state.coffin) != COFFIN_POSITION_COUNT:
        raise InformationContractViolation("information coffin must contain nine positions")
    current_cards = _card_ids(state.coffin)
    _validate_card_ids(current_cards, "information coffin")
    completed_cards = tuple(
        card for record in state.completed_rounds for card in record.coffin
    )
    expected_played = sort_card_ids((*completed_cards, *current_cards))
    if state.played_card_ids != expected_played:
        raise InformationContractViolation(
            "played cards must match public cards in canonical order"
        )
    if state.unseen_card_ids != sort_card_ids(state.unseen_card_ids):
        raise InformationContractViolation("unseen cards must use canonical order")
    _validate_card_ids(state.unseen_card_ids, "unseen pool")
    # These three actor-visible sets must partition the physical deck without
    # revealing how unseen cards divide between opponent hand and stock.
    physical = (*state.played_card_ids, *own_cards, *state.unseen_card_ids)
    if len(physical) != CARD_COUNT or set(physical) != set(CARD_IDS):
        raise InformationContractViolation("information state must partition all 54 cards")
    opponent_moves = sum(
        move.player is other_player(state.player) for move in state.current_round_moves
    )
    if state.opponent_remaining_count != HAND_SLOT_COUNT - opponent_moves:
        raise InformationContractViolation("opponent remaining count is inconsistent")
    if len(own_cards) != HAND_SLOT_COUNT - sum(
        move.player is state.player for move in state.current_round_moves
    ):
        raise InformationContractViolation("own remaining hand count is inconsistent")
    if state.stock_count != CARD_COUNT - COFFIN_POSITION_COUNT * state.round_number:
        raise InformationContractViolation("stock count is inconsistent with round")
    if len(state.unseen_card_ids) != state.opponent_remaining_count + state.stock_count:
        raise InformationContractViolation("unseen pool does not match hidden location counts")
    if state.legal_mask != _legal_mask_from_visible_state(state.own_hand, state.coffin):
        raise InformationContractViolation("legal mask does not match visible hand and coffin")


def _relative_player(root: EnginePlayer, actor: EnginePlayer) -> str:
    return "self" if actor is root else "opponent"


def _relative_public_move(
    move: PublicPlayedMove, root: EnginePlayer
) -> dict[str, object]:
    return {
        "actor": _relative_player(root, move.player),
        "card_id": move.card_id,
        "grid_index": player_relative_grid_index(root, move.global_grid_index),
        "turn_number": move.turn_number,
    }


def _relative_round(record: PublicRoundRecord, root: EnginePlayer) -> dict[str, object]:
    """Encode one public round without preserving absolute actor orientation."""

    coffin = _transpose_coffin_for_player(record.coffin, root)
    opponent = other_player(root)
    return {
        "round_number": record.round_number,
        "dealer": _relative_player(root, record.dealer),
        "coffin": coffin,
        "moves": [_relative_public_move(move, root) for move in record.moves],
        "round_scores": {
            "self": record.round_scores[root],
            "opponent": record.round_scores[opponent],
        },
    }


def canonical_information_data(state: SearchInformationState) -> dict[str, object]:
    """Return actor-relative information in fingerprint-stable field order."""

    opponent = other_player(state.player)
    return {
        "round_number": state.round_number,
        "dealer": _relative_player(state.player, state.dealer),
        "active_player": "self",
        "turn_number": state.turn_number,
        "total_scores": {
            "self": state.total_scores[state.player],
            "opponent": state.total_scores[opponent],
        },
        "completed_rounds": [
            _relative_round(record, state.player) for record in state.completed_rounds
        ],
        "own_hand": list(state.own_hand),
        "coffin": list(state.coffin),
        "current_round_moves": [
            _relative_public_move(move, state.player)
            for move in state.current_round_moves
        ],
        "played_card_ids": list(state.played_card_ids),
        "unseen_card_ids": list(state.unseen_card_ids),
        "opponent_remaining_count": state.opponent_remaining_count,
        "stock_count": state.stock_count,
        "legal_mask": [list(row) for row in state.legal_mask],
    }


def canonical_information_json(state: SearchInformationState) -> str:
    """Serialize canonical actor information without hidden-location data."""

    # Dictionary insertion order is part of the information fingerprint.
    return json.dumps(
        canonical_information_data(state),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=False,
    )


def information_state_fingerprint(state: SearchInformationState) -> str:
    """Identify states indistinguishable to the acting player."""

    return hashlib.sha256(canonical_information_json(state).encode("utf-8")).hexdigest()

"""Immutable player views and deterministic hidden-card sampling."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import torch

from dracula.bridge import (
    COFFIN_START,
    COFFIN_POSITION_COUNT,
    DEALER_INDEX,
    HAND_SLOT_COUNT,
    HIDDEN_STATUS_START,
    IN_HAND_STATUS_START,
    OBSERVATION_SIZE,
    PLAYED_STATUS_START,
    PROGRESS_START,
    ROUND_START,
    PolicyInput,
    build_policy_turn_context,
    build_simulation_policy_input,
    global_grid_index,
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
    resolve_round_scores,
    score_coffin,
    validate_state,
)
from dracula.randomness import Sha256CounterStream, derive_seed, seed_hex, shuffled

INFORMATION_STATE_SCHEMA_VERSION = "dracula-search-information-v1"
SEARCH_REQUEST_NAMESPACE = "dracula-search-request-v1"
BELIEF_SAMPLE_NAMESPACE = "dracula-search-determinization-v1"
TREE_SELECTION_NAMESPACE = "dracula-search-expansion-v1"
ROLLOUT_CHOICE_NAMESPACE = "dracula-search-opponent-v1"

LegalMask = tuple[
    tuple[bool, bool, bool, bool, bool, bool, bool, bool],
    tuple[bool, bool, bool, bool, bool, bool, bool, bool],
    tuple[bool, bool, bool, bool, bool, bool, bool, bool],
    tuple[bool, bool, bool, bool, bool, bool, bool, bool],
]


class InformationContractViolation(ValueError):
    """A public view, information state, or determinization is inconsistent."""


@dataclass(frozen=True, slots=True)
class PublicPlayedMove:
    player: EnginePlayer
    card_id: str
    global_grid_index: int
    turn_number: int


@dataclass(frozen=True, slots=True)
class PublicRoundRecord:
    round_number: int
    dealer: EnginePlayer
    coffin: tuple[str, str, str, str, str, str, str, str, str]
    moves: tuple[PublicPlayedMove, ...]
    round_scores: PlayerValues[int]


@dataclass(frozen=True, slots=True)
class PublicGameHistory:
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
    schema_version: str
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


@dataclass(frozen=True, slots=True)
class SampledDeterminization:
    root_player: EnginePlayer
    sample_seed_digest: str
    opponent_remaining_hand: tuple[str, ...]
    sampled_stock: tuple[str, ...]
    state: SimulationEngineState


def _card_ids(values: tuple[str | None, ...]) -> tuple[str, ...]:
    return tuple(value for value in values if value is not None)


def _validate_card_ids(values: tuple[str, ...], label: str) -> None:
    if any(value not in CARD_INDEX_BY_ID for value in values):
        raise InformationContractViolation(f"{label} contains an unknown card")
    if len(values) != len(set(values)):
        raise InformationContractViolation(f"{label} contains duplicate cards")


def _adjacent(first: int, second: int) -> bool:
    first_row, first_column = divmod(first, 3)
    second_row, second_column = divmod(second, 3)
    return abs(first_row - second_row) + abs(first_column - second_column) == 1


def _validate_public_moves(
    moves: tuple[PublicPlayedMove, ...],
    dealer: EnginePlayer,
    coffin: tuple[str | None, ...],
    *,
    complete: bool,
) -> None:
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
        if grid_index in occupied or not any(_adjacent(grid_index, prior) for prior in occupied):
            raise InformationContractViolation("public move history is not a legal coffin growth")
        if coffin[grid_index] != move.card_id:
            raise InformationContractViolation("public move does not match the coffin")
        occupied.add(grid_index)
        seen_cards.add(move.card_id)


def _round_scores(coffin: tuple[str, ...]) -> PlayerValues[int]:
    lines = score_coffin(coffin)
    return resolve_round_scores(
        tuple(line.total for line in lines.queen),
        tuple(line.total for line in lines.king),
    )


def _validate_public_round(record: PublicRoundRecord) -> None:
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
    if record.round_scores != _round_scores(record.coffin):
        raise InformationContractViolation("public round scores do not match its coffin")


def _validate_public_history(history: PublicGameHistory) -> None:
    if history.status is not EngineStatus.PLAYING:
        raise InformationContractViolation("search information requires an active round")
    if type(history.round_number) is not int or not 1 <= history.round_number <= ROUNDS_PER_GAME:
        raise InformationContractViolation("current public round number is invalid")
    if not isinstance(history.dealer, EnginePlayer) or not isinstance(
        history.active_player, EnginePlayer
    ):
        raise InformationContractViolation("public roles are invalid")
    if not isinstance(history.completed_rounds, tuple) or len(history.completed_rounds) != (
        history.round_number - 1
    ):
        raise InformationContractViolation("completed public rounds do not match current round")
    for index, record in enumerate(history.completed_rounds, start=1):
        _validate_public_round(record)
        if record.round_number != index:
            raise InformationContractViolation("completed public rounds are out of order")
        expected_dealer = (
            history.dealer
            if (history.round_number - index) % 2 == 0
            else other_player(history.dealer)
        )
        if record.dealer is not expected_dealer:
            raise InformationContractViolation("public dealer alternation is invalid")
    expected_totals = PlayerValues(
        queen=sum(record.round_scores.queen for record in history.completed_rounds),
        king=sum(record.round_scores.king for record in history.completed_rounds),
    )
    if history.total_scores != expected_totals:
        raise InformationContractViolation("public totals do not match completed rounds")
    if not isinstance(history.current_coffin, tuple) or len(history.current_coffin) != (
        COFFIN_POSITION_COUNT
    ):
        raise InformationContractViolation("current public coffin must contain nine positions")
    current_cards = _card_ids(history.current_coffin)
    _validate_card_ids(current_cards, "current public coffin")
    if history.current_coffin[CENTER_GRID_INDEX] is None:
        raise InformationContractViolation("current public coffin is missing its center")
    _validate_public_moves(
        history.current_round_moves,
        history.dealer,
        history.current_coffin,
        complete=False,
    )
    expected_active = (
        other_player(history.dealer)
        if len(history.current_round_moves) % 2 == 0
        else history.dealer
    )
    if history.active_player is not expected_active:
        raise InformationContractViolation("active player does not match public turn order")

    public_cards = [card for record in history.completed_rounds for card in record.coffin]
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


def _decode_slots(bits: list[list[bool]], label: str) -> tuple[str | None, ...]:
    decoded: list[str | None] = []
    for row in bits:
        indexes = [index for index, present in enumerate(row) if present]
        if len(indexes) > 1:
            raise InformationContractViolation(f"{label} contains a non-one-hot position")
        decoded.append(None if not indexes else CARD_IDS[indexes[0]])
    return tuple(decoded)


def _relative_coffin(
    coffin: tuple[str | None, ...], player: EnginePlayer
) -> tuple[str | None, ...]:
    result: list[str | None] = [None] * COFFIN_POSITION_COUNT
    for global_index, card_id in enumerate(coffin):
        result[player_relative_grid_index(player, global_index)] = card_id
    return tuple(result)


def _immutable_mask(policy_input: PolicyInput) -> LegalMask:
    rows = policy_input.legal_mask.detach().cpu().tolist()
    return tuple(tuple(bool(value) for value in row) for row in rows)  # type: ignore[return-value]


def build_information_state(
    policy_input: PolicyInput,
    public_history: PublicGameHistory,
    player: EnginePlayer,
) -> SearchInformationState:
    """Combine the established visible tensor with slot-free public history."""

    if not isinstance(policy_input, PolicyInput):
        raise InformationContractViolation("policy input must satisfy the bridge contract")
    if not isinstance(public_history, PublicGameHistory):
        raise InformationContractViolation("public history must satisfy its contract")
    try:
        player = EnginePlayer(player)
    except (TypeError, ValueError) as error:
        raise InformationContractViolation("information player is invalid") from error
    if public_history.active_player is not player:
        raise InformationContractViolation("information can be built only for the active player")

    observation = policy_input.observation.detach().cpu()
    hand_bits = observation[: HAND_SLOT_COUNT * CARD_COUNT].reshape(
        HAND_SLOT_COUNT, CARD_COUNT
    ).tolist()
    coffin_start = HAND_SLOT_COUNT * CARD_COUNT
    coffin_bits = observation[
        coffin_start : coffin_start + COFFIN_POSITION_COUNT * CARD_COUNT
    ].reshape(COFFIN_POSITION_COUNT, CARD_COUNT).tolist()
    own_hand = _decode_slots(hand_bits, "own hand")
    coffin = _decode_slots(coffin_bits, "coffin")
    played_card_ids = tuple(
        CARD_IDS[index]
        for index, present in enumerate(
            observation[PLAYED_STATUS_START:IN_HAND_STATUS_START].tolist()
        )
        if present
    )
    unseen_card_ids = tuple(
        CARD_IDS[index]
        for index, present in enumerate(
            observation[HIDDEN_STATUS_START:ROUND_START].tolist()
        )
        if present
    )
    round_bits = observation[ROUND_START:PROGRESS_START].tolist()
    progress_bits = observation[PROGRESS_START:DEALER_INDEX].tolist()
    round_number = round_bits.index(True) + 1
    own_decision_index = progress_bits.index(True)
    dealer = player if bool(observation[DEALER_INDEX].item()) else other_player(player)

    if round_number != public_history.round_number or dealer is not public_history.dealer:
        raise InformationContractViolation("visible tensor and public lifecycle disagree")
    if coffin != _relative_coffin(public_history.current_coffin, player):
        raise InformationContractViolation("visible tensor and public coffin disagree")
    public_played = tuple(
        sort_card_ids(
            card
            for record in public_history.completed_rounds
            for card in record.coffin
        )
    ) + tuple(sort_card_ids(_card_ids(public_history.current_coffin)))
    if tuple(sort_card_ids(played_card_ids)) != tuple(sort_card_ids(public_played)):
        raise InformationContractViolation("visible played cards and public history disagree")
    if own_decision_index != sum(
        move.player is player for move in public_history.current_round_moves
    ):
        raise InformationContractViolation("visible progress and public move history disagree")

    opponent = other_player(player)
    opponent_remaining = HAND_SLOT_COUNT - sum(
        move.player is opponent for move in public_history.current_round_moves
    )
    stock_count = CARD_COUNT - COFFIN_POSITION_COUNT * round_number
    return SearchInformationState(
        schema_version=INFORMATION_STATE_SCHEMA_VERSION,
        player=player,
        round_number=round_number,
        dealer=dealer,
        active_player=player,
        turn_number=len(public_history.current_round_moves) + 1,
        total_scores=public_history.total_scores,
        completed_rounds=public_history.completed_rounds,
        own_hand=own_hand,  # type: ignore[arg-type]
        coffin=coffin,
        current_round_moves=public_history.current_round_moves,
        played_card_ids=played_card_ids,
        unseen_card_ids=unseen_card_ids,
        opponent_remaining_count=opponent_remaining,
        stock_count=stock_count,
        legal_mask=_immutable_mask(policy_input),
    )


def information_state_from_engine(
    state: EngineState, player: EnginePlayer | None = None
) -> SearchInformationState:
    """Application boundary; private fields are erased by existing projections."""

    validate_state(state)
    selected = state.active_player if player is None else EnginePlayer(player)
    if selected is None:
        raise InformationContractViolation("search information requires the active player")
    context = build_policy_turn_context(state, selected)
    return build_information_state(
        context.input, _public_history_from_validated_engine(state), selected
    )


def project_simulation_information_state(
    state: SimulationEngineState,
) -> SearchInformationState:
    """Project an engine-created simulation state without whole-deck revalidation."""

    if not isinstance(state, SimulationEngineState):
        raise InformationContractViolation("simulation projection requires sampled state")
    if state.active_player is None:
        raise InformationContractViolation("simulation projection requires an active player")
    policy_input = build_simulation_policy_input(state, state.active_player)
    return build_information_state(
        policy_input,
        _public_history_from_validated_engine(state),
        state.active_player,
    )


def information_state_from_simulation(
    state: SimulationEngineState,
) -> SearchInformationState:
    """Give the simulated active actor the same projection used at the root."""

    if not isinstance(state, SimulationEngineState):
        raise InformationContractViolation("simulation projection requires sampled state")
    validate_state(state)
    return project_simulation_information_state(state)


def policy_input_from_information_state(
    state: SearchInformationState,
) -> PolicyInput:
    """Encode the typed player view without consulting a sampled or private world."""

    _validate_information_state(state)
    observation = torch.zeros(OBSERVATION_SIZE, dtype=torch.bool)
    for hand_slot, card_id in enumerate(state.own_hand):
        if card_id is not None:
            observation[hand_slot * CARD_COUNT + CARD_INDEX_BY_ID[card_id]] = True
    for position, card_id in enumerate(state.coffin):
        if card_id is not None:
            observation[
                COFFIN_START + position * CARD_COUNT + CARD_INDEX_BY_ID[card_id]
            ] = True
    for card_id in state.played_card_ids:
        observation[PLAYED_STATUS_START + CARD_INDEX_BY_ID[card_id]] = True
    for card_id in _card_ids(state.own_hand):
        observation[IN_HAND_STATUS_START + CARD_INDEX_BY_ID[card_id]] = True
    for card_id in state.unseen_card_ids:
        observation[HIDDEN_STATUS_START + CARD_INDEX_BY_ID[card_id]] = True
    own_decision_index = sum(
        move.player is state.player for move in state.current_round_moves
    )
    observation[ROUND_START + state.round_number - 1] = True
    observation[PROGRESS_START + own_decision_index] = True
    observation[DEALER_INDEX] = state.player is state.dealer
    legal_mask = torch.tensor(state.legal_mask, dtype=torch.bool)
    return PolicyInput(observation, legal_mask)


def _expected_legal_mask(state: SearchInformationState) -> LegalMask:
    occupied = {index for index, card in enumerate(state.coffin) if card is not None}
    destinations = {
        index
        for index, card in enumerate(state.coffin)
        if card is None and any(_adjacent(index, prior) for prior in occupied)
    }
    policy_positions = (0, 1, 2, 3, 5, 6, 7, 8)
    return tuple(
        tuple(
            state.own_hand[hand_slot] is not None and grid_index in destinations
            for grid_index in policy_positions
        )
        for hand_slot in range(HAND_SLOT_COUNT)
    )  # type: ignore[return-value]


def _validate_information_state(state: SearchInformationState) -> None:
    if state.schema_version != INFORMATION_STATE_SCHEMA_VERSION:
        raise InformationContractViolation("information schema version is unsupported")
    if not isinstance(state.player, EnginePlayer) or state.active_player is not state.player:
        raise InformationContractViolation("information state must belong to its active player")
    if not isinstance(state.dealer, EnginePlayer):
        raise InformationContractViolation("information dealer is invalid")
    if type(state.turn_number) is not int or not 1 <= state.turn_number <= MOVES_PER_ROUND:
        raise InformationContractViolation("information turn number is invalid")
    history = PublicGameHistory(
        status=EngineStatus.PLAYING,
        round_number=state.round_number,
        dealer=state.dealer,
        active_player=state.active_player,
        total_scores=state.total_scores,
        completed_rounds=state.completed_rounds,
        current_coffin=tuple(
            state.coffin[player_relative_grid_index(state.player, index)]
            for index in range(COFFIN_POSITION_COUNT)
        ),
        current_round_moves=state.current_round_moves,
    )
    if state.turn_number != len(history.current_round_moves) + 1:
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
    expected_played = tuple(
        sort_card_ids(
            card for record in state.completed_rounds for card in record.coffin
        )
    ) + tuple(sort_card_ids(current_cards))
    if tuple(sort_card_ids(state.played_card_ids)) != tuple(sort_card_ids(expected_played)):
        raise InformationContractViolation("played-card set does not match public cards")
    if state.unseen_card_ids != sort_card_ids(state.unseen_card_ids):
        raise InformationContractViolation("unseen cards must use canonical order")
    _validate_card_ids(state.unseen_card_ids, "unseen pool")
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
    if state.legal_mask != _expected_legal_mask(state):
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
    coffin = [None] * COFFIN_POSITION_COUNT
    for global_index, card_id in enumerate(record.coffin):
        coffin[player_relative_grid_index(root, global_index)] = card_id
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
    _validate_information_state(state)
    opponent = other_player(state.player)
    return {
        "information_state_schema_version": state.schema_version,
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
    return json.dumps(
        canonical_information_data(state),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=False,
    )


def information_state_fingerprint(state: SearchInformationState) -> str:
    return hashlib.sha256(canonical_information_json(state).encode("utf-8")).hexdigest()


def _validate_counter(value: int, label: str) -> str:
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return str(value)


def derive_search_request_seed(
    fixture_id: str,
    state: SearchInformationState,
    search_config_digest: str,
) -> bytes:
    return derive_seed(
        SEARCH_REQUEST_NAMESPACE,
        fixture_id,
        information_state_fingerprint(state),
        state.player.value,
        str(state.round_number),
        str(state.turn_number),
        search_config_digest,
    )


def derive_belief_sample_seed(request_seed: bytes, simulation_index: int) -> bytes:
    return derive_seed(
        BELIEF_SAMPLE_NAMESPACE,
        seed_hex(request_seed),
        _validate_counter(simulation_index, "simulation index"),
    )


def derive_tree_selection_seed(
    request_seed: bytes, simulation_index: int, node_digest: str
) -> bytes:
    return derive_seed(
        TREE_SELECTION_NAMESPACE,
        seed_hex(request_seed),
        _validate_counter(simulation_index, "simulation index"),
        node_digest,
    )


def derive_rollout_choice_seed(
    request_seed: bytes, simulation_index: int, rollout_ply: int
) -> bytes:
    return derive_seed(
        ROLLOUT_CHOICE_NAMESPACE,
        seed_hex(request_seed),
        _validate_counter(simulation_index, "simulation index"),
        _validate_counter(rollout_ply, "rollout ply"),
    )


def sample_uniform_action_index(
    state: SearchInformationState, choice_seed: bytes
) -> int:
    """Sample only from actions present in the actor's immutable legal mask."""

    seed_hex(choice_seed)
    legal = tuple(
        index
        for index, allowed in enumerate(value for row in state.legal_mask for value in row)
        if allowed
    )
    if not legal:
        raise InformationContractViolation("an active information state has no legal action")
    return legal[Sha256CounterStream(choice_seed).randbelow(len(legal))]


def _private_moves(
    moves: tuple[PublicPlayedMove, ...],
    original_hands: PlayerValues[tuple[str, ...]],
) -> tuple[EnginePlayedMove, ...]:
    slots = {
        player: {card_id: index for index, card_id in enumerate(original_hands[player])}
        for player in EnginePlayer
    }
    return tuple(
        EnginePlayedMove(
            player=move.player,
            card_id=move.card_id,
            hand_slot=slots[move.player][move.card_id],
            global_grid_index=move.global_grid_index,
            turn_number=move.turn_number,
        )
        for move in moves
    )


def _completed_round_result(record: PublicRoundRecord) -> EngineRoundResult:
    hands = PlayerValues(
        queen=sort_card_ids(
            move.card_id for move in record.moves if move.player is EnginePlayer.QUEEN
        ),
        king=sort_card_ids(
            move.card_id for move in record.moves if move.player is EnginePlayer.KING
        ),
    )
    lines = score_coffin(record.coffin)
    return EngineRoundResult(
        round_number=record.round_number,
        dealer=record.dealer,
        coffin=record.coffin,
        moves=_private_moves(record.moves, hands),
        line_scores=lines,
        round_scores=record.round_scores,
    )


def _original_hand(
    remaining: tuple[str, ...],
    moves: tuple[PublicPlayedMove, ...],
    player: EnginePlayer,
) -> tuple[str, ...]:
    original = sort_card_ids(
        (*remaining, *(move.card_id for move in moves if move.player is player))
    )
    if len(original) != HAND_SLOT_COUNT or len(set(original)) != HAND_SLOT_COUNT:
        raise InformationContractViolation("sampled original hand must contain four cards")
    return original


def _remaining_slots(
    original: tuple[str, ...], moves: tuple[PublicPlayedMove, ...], player: EnginePlayer
) -> tuple[str | None, str | None, str | None, str | None]:
    played = {move.card_id for move in moves if move.player is player}
    return tuple(  # type: ignore[return-value]
        None if card_id in played else card_id for card_id in original
    )


def _deal_segment(
    hands: PlayerValues[tuple[str, ...]], dealer: EnginePlayer, center_card: str
) -> tuple[str, ...]:
    non_dealer_hand = hands[other_player(dealer)]
    dealer_hand = hands[dealer]
    return (
        non_dealer_hand[0],
        non_dealer_hand[1],
        dealer_hand[0],
        dealer_hand[1],
        non_dealer_hand[2],
        non_dealer_hand[3],
        dealer_hand[2],
        dealer_hand[3],
        center_card,
    )


def _global_coffin(state: SearchInformationState) -> tuple[str | None, ...]:
    result: list[str | None] = [None] * COFFIN_POSITION_COUNT
    for policy_index, card_id in enumerate(state.coffin):
        result[global_grid_index(state.player, policy_index)] = card_id
    return tuple(result)


def sample_determinization(
    information: SearchInformationState, sample_seed: bytes
) -> SampledDeterminization:
    """Sample a complete private world using only a player information state."""

    _validate_information_state(information)
    sample_seed_digest = seed_hex(sample_seed)
    hidden_order = shuffled(information.unseen_card_ids, sample_seed)
    opponent_remaining = sort_card_ids(
        hidden_order[: information.opponent_remaining_count]
    )
    sampled_stock = hidden_order[information.opponent_remaining_count :]
    if len(sampled_stock) != information.stock_count:
        raise InformationContractViolation("sampled stock count is inconsistent")

    root = information.player
    opponent = other_player(root)
    root_original = _original_hand(
        _card_ids(information.own_hand), information.current_round_moves, root
    )
    opponent_original = _original_hand(
        opponent_remaining, information.current_round_moves, opponent
    )
    current_hands = PlayerValues(
        queen=root_original if root is EnginePlayer.QUEEN else opponent_original,
        king=root_original if root is EnginePlayer.KING else opponent_original,
    )

    completed_results = tuple(
        _completed_round_result(record) for record in information.completed_rounds
    )
    deck_parts: list[str] = []
    for result in completed_results:
        result_hands = PlayerValues(
            queen=tuple(
                move.card_id
                for move in sorted(
                    (move for move in result.moves if move.player is EnginePlayer.QUEEN),
                    key=lambda move: move.hand_slot,
                )
            ),
            king=tuple(
                move.card_id
                for move in sorted(
                    (move for move in result.moves if move.player is EnginePlayer.KING),
                    key=lambda move: move.hand_slot,
                )
            ),
        )
        deck_parts.extend(
            _deal_segment(
                result_hands,
                result.dealer,
                result.coffin[CENTER_GRID_INDEX],
            )
        )
    global_coffin = _global_coffin(information)
    center_card = global_coffin[CENTER_GRID_INDEX]
    if center_card is None:
        raise InformationContractViolation("sampled round requires its public center")
    deck_parts.extend(_deal_segment(current_hands, information.dealer, center_card))
    deck_parts.extend(sampled_stock)
    simulation_deck = tuple(deck_parts)

    private_moves = _private_moves(information.current_round_moves, current_hands)
    simulated_hands = PlayerValues(
        queen=_remaining_slots(
            current_hands.queen, information.current_round_moves, EnginePlayer.QUEEN
        ),
        king=_remaining_slots(
            current_hands.king, information.current_round_moves, EnginePlayer.KING
        ),
    )
    if simulated_hands[root] != information.own_hand:
        raise InformationContractViolation("sample changed the root player's stable slots")
    state = SimulationEngineState(
        seed=f"search-simulation-v1:{sample_seed_digest}",
        status=EngineStatus.PLAYING,
        round_number=information.round_number,
        dealer=information.dealer,
        active_player=information.active_player,
        stock=sampled_stock,
        hands=simulated_hands,
        coffin=global_coffin,  # type: ignore[arg-type]
        current_round_moves=private_moves,
        pending_round_result=None,
        completed_rounds=completed_results,
        total_scores=information.total_scores,
        simulation_deck=simulation_deck,
    )
    if project_simulation_information_state(state) != information:
        raise InformationContractViolation("sample does not reproduce its root information state")
    return SampledDeterminization(
        root_player=root,
        sample_seed_digest=sample_seed_digest,
        opponent_remaining_hand=opponent_remaining,
        sampled_stock=sampled_stock,
        state=state,
    )

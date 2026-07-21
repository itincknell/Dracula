"""Rule, lifecycle, immutability, and replay tests for the game engine."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError, replace

import pytest

from dracula.cards import CARD_IDS
from dracula.engine import (
    CENTER_GRID_INDEX,
    MOVES_PER_ROUND,
    ROUNDS_PER_GAME,
    EngineMove,
    EnginePlayer,
    EngineStatus,
    GameOutcomeReason,
    InvalidGridIndex,
    InvalidLifecycleTransition,
    LineOrientation,
    MalformedState,
    MultiplierReason,
    NonAdjacentDestination,
    OccupiedDestination,
    PlayerValues,
    RuleViolation,
    UnavailableHandSlot,
    WrongActivePlayer,
    advance_after_round,
    apply_move,
    canonical_state_json,
    create_game,
    derive_game_outcome,
    legal_moves,
    other_player,
    resolve_game_outcome,
    resolve_round_scores,
    score_coffin,
    score_line,
    state_fingerprint,
    validate_state,
)

GOLDEN_GAME_SEED = "engine-contract-fixture-1"
GOLDEN_INITIAL_FINGERPRINT = (
    "0775ee468afff2b980efe2b4283dfa7b4c3ad8ee3938f3ac3ae226c44b509fc5"
)
GOLDEN_FIRST_MOVE_FINGERPRINT = (
    "a2f776952d2b57303361df814e81a0637cb6c024d8f940b17f99cd53f6385d71"
)
GOLDEN_FINAL_FINGERPRINT = (
    "a425fd919af1ef6e50879562445881f801475cd00712a4fa5bb4a6de23fb60ed"
)


def _play_round(state):
    accepted = []
    while state.status is EngineStatus.PLAYING:
        move = legal_moves(state, state.active_player)[0]
        transition = apply_move(state, move)
        accepted.append(transition)
        state = transition.state
    return state, tuple(accepted)


def _play_game(seed: str = GOLDEN_GAME_SEED):
    state = create_game(seed)
    while state.status is not EngineStatus.GAME_COMPLETE:
        state, _ = _play_round(state)
        state = advance_after_round(state)
    return state


def _owned_card_ids(state) -> tuple[str, ...]:
    cards = list(state.stock)
    cards.extend(
        card_id
        for hand in (state.hands.queen, state.hands.king)
        for card_id in hand
        if card_id is not None
    )
    cards.extend(card_id for card_id in state.coffin if card_id is not None)
    for result in state.completed_rounds:
        cards.extend(result.coffin)
    return tuple(cards)


# Conservation makes card identity trustworthy across deals, moves, and archives.
@pytest.mark.parametrize("seed", ("conservation-0", "conservation-1", "conservation-2"))
def test_all_54_cards_have_one_ownership_location_throughout_a_game(seed: str) -> None:
    state = create_game(seed)
    while True:
        validate_state(state)
        owned = _owned_card_ids(state)
        assert len(owned) == 54
        assert len(set(owned)) == 54
        assert set(owned) == set(CARD_IDS)
        if state.status is EngineStatus.GAME_COMPLETE:
            break
        if state.status is EngineStatus.ROUND_COMPLETE:
            state = advance_after_round(state)
        else:
            moves = legal_moves(state, state.active_player)
            choice = (len(seed) + state.round_number + 3 * len(state.current_round_moves)) % len(
                moves
            )
            state = apply_move(state, moves[choice]).state


# Golden fingerprints detect any accidental change to replay-visible state encoding.
def test_seeded_creation_and_fingerprints_are_repeatable_and_golden() -> None:
    first = create_game(GOLDEN_GAME_SEED)
    repeated = create_game(GOLDEN_GAME_SEED)
    assert first == repeated
    assert canonical_state_json(first) == canonical_state_json(repeated)
    assert state_fingerprint(first) == GOLDEN_INITIAL_FINGERPRINT
    assert json.loads(canonical_state_json(first))["state"]["stock"] == list(first.stock)

    first_transition = apply_move(first, legal_moves(first, first.active_player)[0])
    assert first_transition.state_fingerprint == GOLDEN_FIRST_MOVE_FINGERPRINT
    assert _play_game().status is EngineStatus.GAME_COMPLETE
    assert state_fingerprint(_play_game()) == GOLDEN_FINAL_FINGERPRINT


# Pure transitions let callers retry and compare states without defensive copying.
def test_apply_and_advance_do_not_mutate_their_input_states() -> None:
    initial = create_game(GOLDEN_GAME_SEED)
    before = canonical_state_json(initial)
    transition = apply_move(initial, legal_moves(initial, initial.active_player)[0])
    assert canonical_state_json(initial) == before
    assert transition.previous_state is initial
    assert transition.state is not initial
    with pytest.raises(FrozenInstanceError):
        initial.round_number = 2  # type: ignore[misc]

    completed, _ = _play_round(initial)
    completed_before = canonical_state_json(completed)
    next_round = advance_after_round(completed)
    assert canonical_state_json(completed) == completed_before
    assert next_round is not completed


# Legal ordering is part of the action-table and deterministic replay contract.
def test_legal_moves_use_stable_slot_then_grid_ordering() -> None:
    state = create_game(GOLDEN_GAME_SEED)
    moves = legal_moves(state, state.active_player)
    assert moves == tuple(
        EngineMove(state.active_player, hand_slot, grid_index)
        for hand_slot in range(4)
        for grid_index in (1, 3, 5, 7)
    )


# Orthogonal growth prevents diagonal gaps from becoming playable coffin cells.
def test_orthogonal_adjacency_accepts_edges_and_rejects_diagonal_only_cells() -> None:
    state = create_game(GOLDEN_GAME_SEED)
    for grid_index in (1, 3, 5, 7):
        apply_move(state, EngineMove(state.active_player, 0, grid_index))
    for grid_index in (0, 2, 6, 8):
        with pytest.raises(NonAdjacentDestination):
            apply_move(state, EngineMove(state.active_player, 0, grid_index))


# Directional values encode the Queen-row and King-column rules without role inference.
def test_jack_queen_and_king_values_are_directional() -> None:
    queen_row = score_line(("QH", "5C", "9D"), LineOrientation.ROW)
    queen_column = score_line(("QH", "5C", "9D"), LineOrientation.COLUMN)
    king_row = score_line(("KH", "5C", "9D"), LineOrientation.ROW)
    king_column = score_line(("KH", "5C", "9D"), LineOrientation.COLUMN)
    jack_row = score_line(("JH", "5C", "9D"), LineOrientation.ROW)
    jack_column = score_line(("JH", "5C", "9D"), LineOrientation.COLUMN)
    assert queen_row.values == (10, 5, 9)
    assert queen_column.values == (0, 5, 9)
    assert king_row.values == (0, 5, 9)
    assert king_column.values == (10, 5, 9)
    assert jack_row.values == jack_column.values == (0, 5, 9)


# Only the strongest matching pattern applies; suit and color multipliers never compound.
@pytest.mark.parametrize(
    ("cards", "multiplier", "reason", "total"),
    (
        (("8H", "9H", "10H"), 5, MultiplierReason.SAME_SUIT, 135),
        (("8H", "9H", "10D"), 3, MultiplierReason.SAME_COLOR, 81),
        (("8H", "3H", "8C"), 2, MultiplierReason.SUIT_PAIR, 38),
        (("8H", "9C", "10D"), 1, MultiplierReason.NONE, 27),
    ),
)
def test_highest_multiplier_applies_without_compounding(
    cards: tuple[str, str, str],
    multiplier: int,
    reason: MultiplierReason,
    total: int,
) -> None:
    line = score_line(cards, LineOrientation.ROW)
    assert line.multiplier == multiplier
    assert line.multiplier_reason is reason
    assert line.total == total


# Either physical Vampire nullifies the complete line in both orientations.
@pytest.mark.parametrize("vampire", ("V1", "V2"))
@pytest.mark.parametrize("orientation", tuple(LineOrientation))
def test_vampire_lines_score_zero(vampire: str, orientation: LineOrientation) -> None:
    line = score_line(("10H", vampire, "9H"), orientation)
    assert line.base_value == 19
    assert line.multiplier == 0
    assert line.multiplier_reason is MultiplierReason.VAMPIRE
    assert line.total == 0


# Tie resolution must descend in lockstep and accept the third level even when tied.
@pytest.mark.parametrize(
    ("queen", "king", "expected"),
    (
        ((40, 20, 10), (30, 25, 15), PlayerValues(40, 30)),
        ((30, 24, 8), (30, 18, 11), PlayerValues(24, 18)),
        ((30, 20, 9), (30, 20, 7), PlayerValues(9, 7)),
        ((30, 20, 7), (30, 20, 7), PlayerValues(7, 7)),
    ),
)
def test_round_score_resolves_at_each_tie_level(
    queen: tuple[int, int, int],
    king: tuple[int, int, int],
    expected: PlayerValues[int],
) -> None:
    assert resolve_round_scores(queen, king) == expected


# Turn ownership and dealer alternation determine which private hand may change.
def test_non_dealer_starts_and_dealer_alternates_across_rounds() -> None:
    state = create_game(GOLDEN_GAME_SEED)
    first_dealer = state.dealer
    for round_index in range(ROUNDS_PER_GAME):
        expected_dealer = first_dealer if round_index % 2 == 0 else other_player(first_dealer)
        assert state.dealer is expected_dealer
        assert state.active_player is other_player(state.dealer)
        completed, transitions = _play_round(state)
        assert tuple(item.played_move.player for item in transitions) == (
            other_player(expected_dealer),
            expected_dealer,
        ) * 4
        state = advance_after_round(completed)


# Every round consumes all four stable slots from both hands before scoring.
def test_exactly_eight_moves_complete_each_round() -> None:
    state = create_game(GOLDEN_GAME_SEED)
    completed, transitions = _play_round(state)
    assert len(transitions) == MOVES_PER_ROUND
    assert len(completed.current_round_moves) == MOVES_PER_ROUND
    assert completed.status is EngineStatus.ROUND_COMPLETE
    assert completed.active_player is None
    assert completed.hands == PlayerValues((None,) * 4, (None,) * 4)
    assert transitions[-1].round_result is completed.pending_round_result

    with pytest.raises(InvalidLifecycleTransition):
        apply_move(completed, transitions[-1].move)


# Six nine-card rounds exhaust the deck exactly and retain six scored archives.
def test_six_rounds_consume_the_deck_and_produce_a_terminal_outcome() -> None:
    state = create_game(GOLDEN_GAME_SEED)
    for expected_round in range(1, ROUNDS_PER_GAME + 1):
        assert state.round_number == expected_round
        assert len(state.stock) == 54 - 9 * expected_round
        state, _ = _play_round(state)
        state = advance_after_round(state)
    assert state.status is EngineStatus.GAME_COMPLETE
    assert state.round_number == 6
    assert state.stock == ()
    assert len(state.completed_rounds) == 6
    assert sum(result.round_scores.queen for result in state.completed_rounds) == state.total_scores.queen
    assert sum(result.round_scores.king for result in state.completed_rounds) == state.total_scores.king
    assert derive_game_outcome(state).winner is EnginePlayer.QUEEN


# The sixth round breaks only a tied cumulative total; an equal sixth round remains a tie.
@pytest.mark.parametrize(
    ("totals", "sixth", "winner", "reason"),
    (
        (PlayerValues(101, 100), PlayerValues(1, 99), EnginePlayer.QUEEN, GameOutcomeReason.TOTAL_SCORE),
        (
            PlayerValues(100, 100),
            PlayerValues(20, 21),
            EnginePlayer.KING,
            GameOutcomeReason.SIXTH_ROUND_SCORE,
        ),
        (PlayerValues(100, 100), PlayerValues(20, 20), None, GameOutcomeReason.TIE),
    ),
)
def test_final_outcome_uses_total_then_sixth_round_tie_break(
    totals: PlayerValues[int],
    sixth: PlayerValues[int],
    winner: EnginePlayer | None,
    reason: GameOutcomeReason,
) -> None:
    outcome = resolve_game_outcome(totals, sixth)
    assert outcome.winner is winner
    assert outcome.reason is reason


@pytest.mark.parametrize(
    ("move_factory", "violation"),
    (
        (
            lambda state: EngineMove(other_player(state.active_player), 0, 1),
            WrongActivePlayer,
        ),
        (lambda state: EngineMove(state.active_player, -1, 1), UnavailableHandSlot),
        (lambda state: EngineMove(state.active_player, 4, 1), UnavailableHandSlot),
        (lambda state: EngineMove(state.active_player, 0, -1), InvalidGridIndex),
        (lambda state: EngineMove(state.active_player, 0, 9), InvalidGridIndex),
        (lambda state: EngineMove(state.active_player, 0, CENTER_GRID_INDEX), OccupiedDestination),
        (lambda state: EngineMove(state.active_player, 0, 0), NonAdjacentDestination),
    ),
)
def test_each_invalid_move_category_has_a_typed_violation(move_factory, violation) -> None:
    state = create_game(GOLDEN_GAME_SEED)
    with pytest.raises(violation):
        apply_move(state, move_factory(state))


def test_empty_hand_slot_and_malformed_move_are_rejected() -> None:
    state = create_game(GOLDEN_GAME_SEED)
    first = legal_moves(state, state.active_player)[0]
    state = apply_move(state, first).state
    state = apply_move(state, legal_moves(state, state.active_player)[0]).state
    with pytest.raises(UnavailableHandSlot):
        apply_move(state, EngineMove(state.active_player, first.hand_slot, 3))
    with pytest.raises(RuleViolation):
        apply_move(state, object())  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "mutate",
    (
        lambda state: replace(state, status="playing"),
        lambda state: replace(state, round_number=0),
        lambda state: replace(state, dealer=other_player(state.dealer)),
        lambda state: replace(state, active_player=state.dealer),
        lambda state: replace(state, stock=(state.stock[0],) + state.stock[:-1]),
        lambda state: replace(
            state, stock=(state.stock[1], state.stock[0], *state.stock[2:])
        ),
        lambda state: replace(state, hands=PlayerValues(state.hands.queen[:3], state.hands.king)),
        lambda state: replace(
            state,
            hands=PlayerValues(
                (state.hands.queen[1], state.hands.queen[0], *state.hands.queen[2:]),
                state.hands.king,
            ),
        ),
        lambda state: replace(state, coffin=state.coffin[:8]),
        lambda state: replace(
            state,
            coffin=state.coffin[:CENTER_GRID_INDEX]
            + (None,)
            + state.coffin[CENTER_GRID_INDEX + 1 :],
        ),
        lambda state: replace(state, current_round_moves=[]),
        lambda state: replace(state, pending_round_result="result"),
        lambda state: replace(state, total_scores=PlayerValues(-1, 0)),
        lambda state: replace(state, total_scores=PlayerValues(1, 0)),
    ),
)
def test_malformed_created_state_variants_are_rejected(mutate) -> None:
    with pytest.raises(MalformedState):
        validate_state(mutate(create_game(GOLDEN_GAME_SEED)))


def test_malformed_move_history_and_round_result_are_rejected() -> None:
    state = create_game(GOLDEN_GAME_SEED)
    state = apply_move(state, legal_moves(state, state.active_player)[0]).state
    bad_turn = replace(state.current_round_moves[0], turn_number=2)
    with pytest.raises(MalformedState):
        validate_state(replace(state, current_round_moves=(bad_turn,)))

    completed, _ = _play_round(create_game(GOLDEN_GAME_SEED))
    result = completed.pending_round_result
    assert result is not None
    bad_result = replace(result, round_scores=PlayerValues(999, result.round_scores.king))
    with pytest.raises(MalformedState):
        validate_state(replace(completed, pending_round_result=bad_result))


def test_invalid_lifecycle_operations_are_rejected() -> None:
    playing = create_game(GOLDEN_GAME_SEED)
    with pytest.raises(InvalidLifecycleTransition):
        advance_after_round(playing)
    with pytest.raises(InvalidLifecycleTransition):
        derive_game_outcome(playing)

    completed, _ = _play_round(playing)
    with pytest.raises(InvalidLifecycleTransition):
        legal_moves(completed, EnginePlayer.QUEEN)

    terminal = _play_game()
    with pytest.raises(InvalidLifecycleTransition):
        advance_after_round(terminal)
    with pytest.raises(InvalidLifecycleTransition):
        legal_moves(terminal, EnginePlayer.QUEEN)


def test_scoring_rejects_malformed_inputs() -> None:
    with pytest.raises(ValueError):
        score_line(("AC", "2C"), LineOrientation.ROW)
    with pytest.raises(ValueError):
        score_line(("AC", "2C", "not-a-card"), LineOrientation.ROW)
    with pytest.raises(ValueError):
        score_line(("AC", "2C", "3C"), "diagonal")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        score_coffin(CARD_IDS[:8])
    with pytest.raises(ValueError):
        resolve_round_scores((1, 2), (1, 2, 3))
    with pytest.raises(ValueError):
        resolve_round_scores((1, -2, 3), (1, 2, 3))

"""Contract fixtures for the player-relative engine-model bridge."""

from __future__ import annotations

import hashlib
import math
from dataclasses import replace

import pytest
import torch

from dracula.bridge import (
    ACTION_COUNT,
    DEALER_INDEX,
    HIDDEN_STATE_BYTES,
    HIDDEN_STATUS_START,
    IN_HAND_STATUS_START,
    PLAYED_STATUS_START,
    PROGRESS_START,
    ROUND_START,
    BridgeContractViolation,
    PolicyInput,
    PolicyTransition,
    PolicyTurnContext,
    PolicyTurnKind,
    _encode_observation_unchecked,
    action_index_for_move,
    apply_policy_action,
    build_policy_transition,
    build_policy_turn_context,
    encode_hand_positions,
    encode_player_relative_coffin,
    global_grid_index,
    hidden_state_from_bytes,
    hidden_state_to_bytes,
    move_for_action_index,
    player_relative_grid_index,
    resolve_policy_action,
    transpose_grid_index,
    validate_policy_transition,
    validate_policy_turn_context,
)
from dracula.cards import CARD_COUNT
from dracula.engine import (
    EngineMove,
    EnginePlayer,
    EngineState,
    EngineStatus,
    MalformedState,
    advance_after_round,
    create_game,
    deal_round,
    legal_moves,
    other_player,
    state_fingerprint,
)
from dracula.models import COFFIN_START, CONTEXT_START, OBSERVATION_SIZE, STATUS_START

GOLDEN_GAME_SEED = "engine-contract-fixture-1"
GOLDEN_STATE_FINGERPRINT = (
    "0775ee468afff2b980efe2b4283dfa7b4c3ad8ee3938f3ac3ae226c44b509fc5"
)


def _first_action_index(context: PolicyTurnContext) -> int:
    return next(index for index, move in enumerate(context.action_table) if move is not None)


def _advance_with_first_action(state: EngineState) -> EngineState:
    context = build_policy_turn_context(state, state.active_player)
    return apply_policy_action(state, context, _first_action_index(context)).state


def _transpose_coffin(coffin: tuple[str | None, ...]) -> tuple[str | None, ...]:
    result: list[str | None] = [None] * 9
    for global_index, card_id in enumerate(coffin):
        result[transpose_grid_index(global_index)] = card_id
    return tuple(result)


def _tensor_digest(tensor: torch.Tensor) -> str:
    values = bytes(tensor.to(dtype=torch.uint8).flatten().tolist())
    return hashlib.sha256(values).hexdigest()


# Transposing an authoritative King board gives the same normalized view as Queen's board.
def test_queen_and_king_coffin_views_are_transpose_equivalent() -> None:
    queen_coffin = (
        "AC",
        None,
        "2D",
        "3H",
        "4S",
        None,
        "5C",
        "6D",
        "V1",
    )
    king_authoritative = _transpose_coffin(queen_coffin)
    queen_view = encode_player_relative_coffin(queen_coffin, EnginePlayer.QUEEN)
    king_view = encode_player_relative_coffin(king_authoritative, EnginePlayer.KING)
    assert torch.equal(queen_view, king_view)

    for hand_slot in range(4):
        for queen_global_index in (0, 1, 2, 3, 5, 6, 7, 8):
            king_global_index = transpose_grid_index(queen_global_index)
            queen_move = EngineMove(EnginePlayer.QUEEN, hand_slot, queen_global_index)
            king_move = EngineMove(EnginePlayer.KING, hand_slot, king_global_index)
            assert action_index_for_move(queen_move) == action_index_for_move(king_move)


# A self-inverse transform prevents coordinate drift on policy-to-engine round trips.
def test_grid_transpose_is_self_inverse() -> None:
    for grid_index in range(9):
        assert transpose_grid_index(transpose_grid_index(grid_index)) == grid_index
        assert player_relative_grid_index(EnginePlayer.QUEEN, grid_index) == grid_index
        assert global_grid_index(EnginePlayer.KING, transpose_grid_index(grid_index)) == grid_index


# Canonical sorting makes hand tensors independent of within-pair deal order.
def test_canonical_hand_encoding_is_permutation_invariant_after_dealing() -> None:
    first_stock = ("AC", "2C", "3C", "4C", "5C", "6C", "7C", "8C", "9C")
    permuted_stock = ("6C", "2C", "8C", "4C", "5C", "AC", "7C", "3C", "9C")
    first = deal_round(first_stock, EnginePlayer.KING)
    permuted = deal_round(permuted_stock, EnginePlayer.KING)
    assert first.queen_hand == permuted.queen_hand
    assert first.king_hand == permuted.king_hand
    assert torch.equal(
        encode_hand_positions(first.queen_hand), encode_hand_positions(permuted.queen_hand)
    )
    assert torch.equal(
        encode_hand_positions(first.king_hand), encode_hand_positions(permuted.king_hand)
    )


# Played, own-hand, and hidden classes must assign every physical card exactly once.
def test_card_status_vectors_xor_to_all_54_true_values() -> None:
    state = create_game(GOLDEN_GAME_SEED)
    state = _advance_with_first_action(state)
    context = build_policy_turn_context(state, state.active_player)
    observation = context.input.observation
    played = observation[PLAYED_STATUS_START:IN_HAND_STATUS_START]
    in_hand = observation[IN_HAND_STATUS_START:HIDDEN_STATUS_START]
    hidden = observation[HIDDEN_STATUS_START:CONTEXT_START]
    assert played.shape == in_hand.shape == hidden.shape == (CARD_COUNT,)
    assert torch.all(torch.logical_xor(torch.logical_xor(played, in_hand), hidden))
    assert torch.all(played.to(torch.int8) + in_hand.to(torch.int8) + hidden.to(torch.int8) == 1)


# Fixed field boundaries let policy artifacts interpret every one of the 875 bits identically.
def test_hand_coffin_status_round_progress_and_dealer_fields_match_engine_state() -> None:
    state = create_game(GOLDEN_GAME_SEED)
    player = state.active_player
    assert player is EnginePlayer.QUEEN
    context = build_policy_turn_context(state, player)
    observation = context.input.observation

    hand = observation[:COFFIN_START].reshape(4, CARD_COUNT)
    coffin = observation[COFFIN_START:STATUS_START].reshape(9, CARD_COUNT)
    assert torch.equal(hand, encode_hand_positions(state.hands[player]))
    assert torch.equal(coffin, encode_player_relative_coffin(state.coffin, player))
    assert hand.sum().item() == 4
    assert coffin.sum().item() == 1
    assert observation[PLAYED_STATUS_START:IN_HAND_STATUS_START].sum().item() == 1
    assert observation[IN_HAND_STATUS_START:HIDDEN_STATUS_START].sum().item() == 4
    assert observation[HIDDEN_STATUS_START:CONTEXT_START].sum().item() == 49
    assert observation[ROUND_START:PROGRESS_START].tolist() == [True] + [False] * 5
    assert observation[PROGRESS_START:DEALER_INDEX].tolist() == [True, False, False, False]
    assert not bool(observation[DEALER_INDEX].item())

    state = _advance_with_first_action(state)
    dealer_context = build_policy_turn_context(state, state.active_player)
    assert dealer_context.player is state.dealer
    assert bool(dealer_context.input.observation[DEALER_INDEX].item())


# Hidden-card location and ordering are deliberately erased before model encoding.
def test_hidden_opponent_hand_and_stock_rearrangements_do_not_change_observation() -> None:
    state = create_game(GOLDEN_GAME_SEED)
    player = state.active_player
    assert player is not None
    baseline = _encode_observation_unchecked(state, player)

    opponent = other_player(player)
    opponent_hand = list(state.hands[opponent])
    stock = list(reversed(state.stock))
    opponent_hand[0], stock[0] = stock[0], opponent_hand[0]
    rearranged = replace(
        state,
        seed="a-seed-that-must-not-enter-policy-input",
        stock=tuple(stock),
        hands=state.hands.updated(opponent, tuple(opponent_hand)),
    )
    assert torch.equal(_encode_observation_unchecked(rearranged, player), baseline)

    # The public bridge still rejects the rearranged authoritative state.
    with pytest.raises(MalformedState):
        build_policy_turn_context(rearranged, player)


# One action index must identify each legal engine move, with no aliases or omissions.
@pytest.mark.parametrize("accepted_moves", (0, 3, 7))
def test_legal_mask_action_table_and_engine_moves_are_an_exact_bijection(
    accepted_moves: int,
) -> None:
    state = create_game(GOLDEN_GAME_SEED)
    for _ in range(accepted_moves):
        state = _advance_with_first_action(state)
    player = state.active_player
    assert player is not None
    context = build_policy_turn_context(state, player)
    engine_moves = legal_moves(state, player)

    table_moves = tuple(move for move in context.action_table if move is not None)
    assert len(table_moves) == len(set(table_moves)) == len(engine_moves)
    assert set(table_moves) == set(engine_moves)
    assert int(context.input.legal_mask.sum().item()) == len(engine_moves)
    for move in engine_moves:
        action_index = action_index_for_move(move, player)
        assert context.action_table[action_index] == move
        assert bool(context.input.legal_mask.flatten()[action_index].item())


# A false mask bit must never retain a move that can bypass action masking.
def test_every_masked_action_maps_to_none() -> None:
    state = create_game(GOLDEN_GAME_SEED)
    context = build_policy_turn_context(state, state.active_player)
    for action_index, is_legal in enumerate(context.input.legal_mask.flatten().tolist()):
        assert (context.action_table[action_index] is None) is (not is_legal)


# Player-relative action coordinates must recover the identical authoritative move.
@pytest.mark.parametrize("player", tuple(EnginePlayer))
def test_action_mapping_round_trips_between_policy_and_global_coordinates(
    player: EnginePlayer,
) -> None:
    for action_index in range(ACTION_COUNT):
        move = move_for_action_index(player, action_index)
        assert action_index_for_move(move, player) == action_index
        policy_index = player_relative_grid_index(player, move.global_grid_index)
        assert global_grid_index(player, policy_index) == move.global_grid_index


# Progress is local to each player and advances through the same four one-hot positions.
def test_dealer_and_non_dealer_each_receive_four_consistent_progress_positions() -> None:
    state = create_game(GOLDEN_GAME_SEED)
    dealer = state.dealer
    contexts = {EnginePlayer.QUEEN: [], EnginePlayer.KING: []}
    while state.status is EngineStatus.PLAYING:
        player = state.active_player
        assert player is not None
        context = build_policy_turn_context(state, player)
        contexts[player].append(context)
        progress = context.input.observation[PROGRESS_START:DEALER_INDEX]
        assert int(progress.sum().item()) == 1
        assert bool(progress[context.own_decision_index].item())
        assert bool(context.input.observation[DEALER_INDEX].item()) is (player is dealer)
        state = apply_policy_action(state, context, _first_action_index(context)).state

    for player in EnginePlayer:
        assert [context.own_decision_index for context in contexts[player]] == [0, 1, 2, 3]


# The eighth placement advances recurrence but cannot contribute an actor decision.
def test_eighth_placement_is_the_unique_forced_recurrent_transition() -> None:
    state = create_game(GOLDEN_GAME_SEED)
    for _ in range(7):
        state = _advance_with_first_action(state)
    context = build_policy_turn_context(state, state.active_player)
    assert context.player is state.dealer
    assert context.own_decision_index == 3
    assert context.kind is PolicyTurnKind.FORCED_RECURRENT_TRANSITION
    assert int(context.input.legal_mask.sum().item()) == 1
    assert sum(move is not None for move in context.action_table) == 1
    assert context.forced_move is not None

    action_index, move = resolve_policy_action(state, context)
    assert move == context.forced_move == context.action_table[action_index]
    transition = apply_policy_action(state, context)
    assert transition.played_move.turn_number == 8
    assert transition.state.status is EngineStatus.ROUND_COMPLETE


# Six rounds must preserve one uninterrupted 24-step recurrent sequence per player.
def test_full_game_transition_counts_hidden_continuity_and_loss_masks() -> None:
    state = create_game(GOLDEN_GAME_SEED)
    hidden = {
        EnginePlayer.QUEEN: bytes(HIDDEN_STATE_BYTES),
        EnginePlayer.KING: bytes(HIDDEN_STATE_BYTES),
    }
    transitions: dict[EnginePlayer, list[PolicyTransition]] = {
        EnginePlayer.QUEEN: [],
        EnginePlayer.KING: [],
    }
    while state.status is not EngineStatus.GAME_COMPLETE:
        while state.status is EngineStatus.PLAYING:
            player = state.active_player
            assert player is not None
            context = build_policy_turn_context(state, player)
            action_index = _first_action_index(context)
            engine_transition = apply_policy_action(state, context, action_index)
            hidden_out = hidden_state_to_bytes(
                torch.full((128,), len(transitions[player]) + 1, dtype=torch.float32)
            )
            learned = context.kind is PolicyTurnKind.LEARNED
            policy_transition = build_policy_transition(
                context=context,
                engine_transition=engine_transition,
                fixture_id="fixture-0",
                learner_id=f"policy-{player.value}",
                learner_policy_version="version-0",
                opponent_id=f"policy-{other_player(player).value}",
                opponent_policy_version="version-0",
                recurrent_step_index=len(transitions[player]),
                policy_hidden_in=hidden[player],
                policy_hidden_out=hidden_out,
                action_log_probability=-math.log(
                    int(context.input.legal_mask.sum().item())
                )
                if learned
                else None,
                critic_value=0.0 if learned else None,
            )
            assert policy_transition.policy_hidden_in == hidden[player]
            hidden[player] = policy_transition.policy_hidden_out
            transitions[player].append(policy_transition)
            state = engine_transition.state
        state = advance_after_round(state)

    for player in EnginePlayer:
        player_transitions = transitions[player]
        assert len(player_transitions) == 24
        assert [item.recurrent_step_index for item in player_transitions] == list(range(24))
        assert sum(item.kind is PolicyTurnKind.LEARNED for item in player_transitions) == 21
        assert sum(
            item.kind is PolicyTurnKind.FORCED_RECURRENT_TRANSITION
            for item in player_transitions
        ) == 3
        assert sum(item.actor_loss_mask for item in player_transitions) == 21
        assert all(
            current.policy_hidden_out == following.policy_hidden_in
            for current, following in zip(
                player_transitions[:-1], player_transitions[1:], strict=True
            )
        )


# The fixture hash pins field order while excluding seed and hidden-location details.
def test_golden_policy_input_and_action_contract_fixture() -> None:
    state = create_game(GOLDEN_GAME_SEED)
    context = build_policy_turn_context(state, EnginePlayer.QUEEN)
    assert context.state_fingerprint == GOLDEN_STATE_FINGERPRINT == state_fingerprint(state)
    assert context.input.observation.shape == (OBSERVATION_SIZE,)
    assert context.input.legal_mask.shape == (4, 8)
    assert context.input.observation.dtype is torch.bool
    assert context.input.legal_mask.dtype is torch.bool
    assert [index for index, move in enumerate(context.action_table) if move is not None] == [
        1,
        3,
        4,
        6,
        9,
        11,
        12,
        14,
        17,
        19,
        20,
        22,
        25,
        27,
        28,
        30,
    ]
    assert _tensor_digest(context.input.observation) == (
        "c27790f1548eeae45615f366a75de492f6fd0d0861d0197159ffff64fa057f62"
    )
    assert _tensor_digest(context.input.legal_mask) == (
        "08cc9332a54de8d03e7bee2eb7f792f814cc500a46d5e59594ff270d7bbf8cd0"
    )


# A context is valid only for the exact pre-action engine fingerprint and projection.
def test_context_and_transition_validation_reject_stale_or_forged_contract_values() -> None:
    state = create_game(GOLDEN_GAME_SEED)
    context = build_policy_turn_context(state, state.active_player)
    validate_policy_turn_context(state, context)

    forged_fingerprint = replace(context, state_fingerprint="0" * 64)
    with pytest.raises(BridgeContractViolation):
        validate_policy_turn_context(state, forged_fingerprint)

    observation = context.input.observation.clone()
    observation[DEALER_INDEX] = ~observation[DEALER_INDEX]
    forged_input = replace(context, input=PolicyInput(observation, context.input.legal_mask))
    with pytest.raises(BridgeContractViolation):
        validate_policy_turn_context(state, forged_input)

    engine_transition = apply_policy_action(state, context, _first_action_index(context))
    with pytest.raises(BridgeContractViolation):
        validate_policy_turn_context(engine_transition.state, context)

    learned_transition = build_policy_transition(
        context=context,
        engine_transition=engine_transition,
        fixture_id="fixture-0",
        learner_id="learner-0",
        learner_policy_version="version-0",
        opponent_id="opponent-0",
        opponent_policy_version="version-0",
        recurrent_step_index=0,
        policy_hidden_in=bytes(HIDDEN_STATE_BYTES),
        policy_hidden_out=bytes(HIDDEN_STATE_BYTES),
        action_log_probability=-1.0,
        critic_value=0.0,
    )
    validate_policy_transition(learned_transition, state)
    with pytest.raises(BridgeContractViolation):
        validate_policy_transition(learned_transition, engine_transition.state)


# Hidden artifacts use one portable little-endian float32[128] representation.
def test_hidden_state_bytes_round_trip_exactly_and_reject_malformed_values() -> None:
    hidden = torch.linspace(-1.0, 1.0, 128, dtype=torch.float32)
    encoded = hidden_state_to_bytes(hidden)
    assert len(encoded) == HIDDEN_STATE_BYTES == 512
    assert torch.equal(hidden_state_from_bytes(encoded), hidden)
    with pytest.raises(BridgeContractViolation):
        hidden_state_from_bytes(encoded[:-1])
    with pytest.raises(BridgeContractViolation):
        hidden_state_to_bytes(torch.full((128,), float("nan")))


def test_policy_input_rejects_wrong_shapes_dtypes_and_status_partitions() -> None:
    state = create_game(GOLDEN_GAME_SEED)
    context = build_policy_turn_context(state, state.active_player)
    with pytest.raises(BridgeContractViolation):
        PolicyInput(context.input.observation.float(), context.input.legal_mask)
    with pytest.raises(BridgeContractViolation):
        PolicyInput(context.input.observation[:-1], context.input.legal_mask)
    malformed = context.input.observation.clone()
    card_index = 0
    malformed[PLAYED_STATUS_START + card_index] = True
    malformed[HIDDEN_STATUS_START + card_index] = True
    with pytest.raises(BridgeContractViolation):
        PolicyInput(malformed, context.input.legal_mask)


def test_masked_and_missing_learned_actions_are_rejected() -> None:
    state = create_game(GOLDEN_GAME_SEED)
    context = build_policy_turn_context(state, state.active_player)
    with pytest.raises(BridgeContractViolation):
        resolve_policy_action(state, context)
    masked_index = next(
        index for index, move in enumerate(context.action_table) if move is None
    )
    with pytest.raises(BridgeContractViolation):
        resolve_policy_action(state, context, masked_index)


def test_policy_context_rejects_inactive_players_and_completed_rounds() -> None:
    state = create_game(GOLDEN_GAME_SEED)
    with pytest.raises(BridgeContractViolation):
        build_policy_turn_context(state, other_player(state.active_player))
    while state.status is EngineStatus.PLAYING:
        state = _advance_with_first_action(state)
    with pytest.raises(BridgeContractViolation):
        build_policy_turn_context(state, EnginePlayer.QUEEN)

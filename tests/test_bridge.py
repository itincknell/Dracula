"""Verify the boundary between engine moves and policy action indexes.

Tests cover role-relative grids, legal action tables, forced placements, and
round trips from a selected policy index back to one concrete engine move.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from dracula.decision.bridge import (
    ACTION_COUNT,
    BridgeContractViolation,
    PolicyTurnKind,
    action_index_for_move,
    apply_policy_action,
    build_policy_turn_context,
    global_grid_index,
    move_for_action_index,
    player_relative_grid_index,
    resolve_policy_action,
    transpose_grid_index,
)
from dracula.game.engine import EnginePlayer, apply_move, create_game, legal_moves, other_player


def test_grid_transpose_is_its_own_inverse() -> None:
    for index in range(9):
        assert transpose_grid_index(transpose_grid_index(index)) == index
        assert player_relative_grid_index(EnginePlayer.QUEEN, index) == index
        assert global_grid_index(
            EnginePlayer.KING,
            player_relative_grid_index(EnginePlayer.KING, index),
        ) == index


def test_action_indexes_round_trip_for_both_players() -> None:
    for player in EnginePlayer:
        for action_index in range(ACTION_COUNT):
            move = move_for_action_index(player, action_index)
            assert action_index_for_move(move, player) == action_index


def test_context_matches_engine_legality_in_canonical_order() -> None:
    state = create_game("bridge-context")
    assert state.active_player is not None
    context = build_policy_turn_context(state, state.active_player)
    legal = legal_moves(state, state.active_player)
    assert context.kind is PolicyTurnKind.LEARNED
    assert tuple(move for move in context.action_table if move is not None) == tuple(
        sorted(legal, key=action_index_for_move)
    )
    for index, move in enumerate(context.action_table):
        assert (move is not None) == (index in map(action_index_for_move, legal))


def test_policy_action_applies_the_exact_selected_move() -> None:
    state = create_game("bridge-apply")
    assert state.active_player is not None
    context = build_policy_turn_context(state, state.active_player)
    selected = next(index for index, move in enumerate(context.action_table) if move)
    _, move = resolve_policy_action(state, context, selected)
    assert apply_policy_action(state, context, selected) == apply_move(state, move)


def test_stale_or_other_player_context_is_rejected() -> None:
    state = create_game("bridge-stale")
    assert state.active_player is not None
    context = build_policy_turn_context(state, state.active_player)
    selected = next(index for index, move in enumerate(context.action_table) if move)
    advanced = apply_policy_action(state, context, selected).state
    with pytest.raises(BridgeContractViolation, match="active player"):
        resolve_policy_action(advanced, context, selected)
    with pytest.raises(BridgeContractViolation, match="active player"):
        build_policy_turn_context(state, other_player(state.active_player))
    with pytest.raises(BridgeContractViolation):
        resolve_policy_action(state, replace(context, state_fingerprint="0" * 64), selected)


def test_forced_eighth_placement_bypasses_selection() -> None:
    state = create_game("bridge-forced")
    for _ in range(7):
        assert state.active_player is not None
        state = apply_move(state, legal_moves(state, state.active_player)[0]).state
    assert state.active_player is not None
    context = build_policy_turn_context(state, state.active_player)
    assert context.kind is PolicyTurnKind.FORCED
    assert context.forced_move is not None
    selected, move = resolve_policy_action(state, context)
    assert context.action_table[selected] == move == context.forced_move

"""Teacher v2 action grouping and concrete-destination invariants."""

from __future__ import annotations

import inspect

import pytest

import dracula.search.strategic as strategic_module
from dracula.bridge import global_grid_index
from dracula.engine import (
    EngineMove,
    EngineStatus,
    apply_move,
    apply_simulation_move,
    create_game,
    legal_moves,
    legal_simulation_moves,
)
from dracula.randomness import derive_seed
from dracula.search import (
    ShallowGreedyResponseEvaluator,
    ShallowResponseConfig,
    StrategicInformationSetSearch,
    StrategicSearchConfig,
    derive_strategic_destination_choice_seed,
    derive_strategic_search_request_seed,
    information_state_from_engine,
    information_state_from_simulation,
    sample_determinization,
    select_concrete_action_index,
    strategic_action_groups,
)


AUTHORIZED_CASES = (
    ({5}, ((2, (2, 8)), (4, (4, 6))), 8),
    ({2, 5}, ((1, (1, 3)), (4, (4, 6)), (8, (8,))), 12),
    ({5, 8}, ((2, (2,)), (4, (4, 6)), (9, (7, 9))), 12),
    ({4, 5}, ((1, (1, 7)), (2, (2, 8)), (6, (6,))), 12),
    ({5, 6}, ((2, (2, 8)), (3, (3, 9)), (4, (4,))), 12),
    ({4, 5, 6}, ((1, (1, 7)), (2, (2, 8)), (3, (3, 9))), 9),
    ({2, 5, 8}, ((1, (1, 3)), (4, (4, 6)), (7, (7, 9))), 9),
)


def _apply_relative_destination(state, perspective, position: int):
    actor = state.active_player
    hand_slot = next(
        index for index, card_id in enumerate(state.hands[actor])
        if card_id is not None
    )
    move = EngineMove(
        actor,
        hand_slot,
        global_grid_index(perspective, position - 1),
    )
    assert move in legal_moves(state, actor)
    return apply_move(state, move).state


def _state_for_pattern(occupied: set[int], game_seed: str):
    state = create_game(game_seed)
    non_center = sorted(occupied - {5})
    if len(non_center) == 0:
        return state
    if len(non_center) == 1:
        return _apply_relative_destination(
            state,
            state.dealer,
            non_center[0],
        )
    perspective = state.active_player
    for position in non_center:
        state = _apply_relative_destination(state, perspective, position)
    return state


def _group_shape(information):
    return tuple(
        (
            group.hand_slot,
            group.representative_grid_index + 1,
            tuple(index + 1 for index in group.member_grid_indices),
        )
        for group in strategic_action_groups(information)
    )


# Every available card receives exactly the definitive destination groups.
@pytest.mark.parametrize("game_seed", ("engine-contract-fixture-1", "search-role-0"))
@pytest.mark.parametrize(
    ("occupied", "destination_groups", "expected_action_count"),
    AUTHORIZED_CASES,
)
def test_every_authorized_pattern_has_exact_strategic_actions(
    game_seed: str,
    occupied: set[int],
    destination_groups,
    expected_action_count: int,
) -> None:
    information = information_state_from_engine(
        _state_for_pattern(occupied, game_seed)
    )
    available_slots = tuple(
        index
        for index, card_id in enumerate(information.own_hand)
        if card_id is not None
    )
    expected = tuple(
        (hand_slot, representative, members)
        for hand_slot in available_slots
        for representative, members in destination_groups
    )

    assert _group_shape(information) == expected
    assert len(strategic_action_groups(information)) == expected_action_count


# The fair coin is derived and reproducible, while both mirror members remain reachable.
def test_paired_group_coin_is_reproducible_and_selects_both_members() -> None:
    information = information_state_from_engine(
        create_game("engine-contract-fixture-1")
    )
    group = strategic_action_groups(information)[0]
    request_seed = derive_strategic_search_request_seed(
        "coin-coverage",
        information,
        StrategicSearchConfig(8, 1).digest,
    )
    choices = []
    for index in range(64):
        choice_seed = derive_strategic_destination_choice_seed(
            request_seed,
            "coin-coverage",
            information,
            group.representative_action_index,
            index,
        )
        first = select_concrete_action_index(
            information,
            group.representative_action_index,
            choice_seed,
        )
        second = select_concrete_action_index(
            information,
            group.representative_action_index,
            choice_seed,
        )
        assert first == second
        choices.append(first)

    assert set(choices) == set(group.member_action_indices)


# Concrete resolution happens after search and cannot alter pooled group evidence.
def test_root_coin_cannot_change_group_visits_or_values(monkeypatch) -> None:
    information = information_state_from_engine(
        create_game("engine-contract-fixture-1")
    )
    config = StrategicSearchConfig(8, 1)
    request_seed = derive_strategic_search_request_seed(
        "root-coin-independence",
        information,
        config.digest,
    )
    original = strategic_module.derive_strategic_destination_choice_seed
    alternate_seeds = []
    for index in range(64):
        candidate = derive_seed(
            "strategic-root-coin-test-v1",
            str(index),
        )
        if not alternate_seeds:
            alternate_seeds.append(candidate)
        elif (
            strategic_module.Sha256CounterStream(candidate).randbelow(2)
            != strategic_module.Sha256CounterStream(
                alternate_seeds[0]
            ).randbelow(2)
        ):
            alternate_seeds.append(candidate)
            break
    assert len(alternate_seeds) == 2

    mode = 0

    def root_choice_seed(
        seed,
        scope,
        actor_information,
        representative_action_index,
        choice_index,
    ):
        if scope == "root-result-group":
            return alternate_seeds[mode]
        return original(
            seed,
            scope,
            actor_information,
            representative_action_index,
            choice_index,
        )

    monkeypatch.setattr(
        strategic_module,
        "derive_strategic_destination_choice_seed",
        root_choice_seed,
    )
    first = StrategicInformationSetSearch(config).search(
        information,
        request_seed,
    )
    mode = 1
    second = StrategicInformationSetSearch(config).search(
        information,
        request_seed,
    )

    assert first.selected_representative_action_index == (
        second.selected_representative_action_index
    )
    assert first.selected_action_index != second.selected_action_index
    assert first.selected_representative_grid_index == (
        second.selected_representative_grid_index
    )
    assert {
        first.selected_concrete_grid_index,
        second.selected_concrete_grid_index,
    } == set(
        next(
            diagnostic.group.member_grid_indices
            for diagnostic in first.group_diagnostics
            if diagnostic.group.representative_action_index
            == first.selected_representative_action_index
        )
    )
    assert first.mean_action_values == second.mean_action_values
    assert first.group_diagnostics == second.group_diagnostics
    assert sorted(first.action_visits) == sorted(second.action_visits)


@pytest.mark.parametrize("relative_pair", ((1, 7), (3, 5)))
def test_reflected_continuations_have_identical_engine_scores(
    relative_pair: tuple[int, int],
) -> None:
    information = information_state_from_engine(
        create_game("engine-contract-fixture-1")
    )
    sampled = sample_determinization(
        information,
        derive_seed("strategic-reflection-test-v1", str(relative_pair)),
    ).state
    root = information.player
    first_global = global_grid_index(root, relative_pair[0])
    second_global = global_grid_index(root, relative_pair[1])
    if first_global // 3 == second_global // 3:
        reflect = lambda index: 3 * (index // 3) + (2 - index % 3)
    else:
        assert first_global % 3 == second_global % 3
        reflect = lambda index: 3 * (2 - index // 3) + index % 3

    first = apply_simulation_move(
        sampled,
        EngineMove(root, 0, first_global),
    )
    second = apply_simulation_move(
        sampled,
        EngineMove(root, 0, second_global),
    )
    while first.status is EngineStatus.PLAYING:
        first_move = legal_simulation_moves(
            first,
            first.active_player,
        )[0]
        second_move = EngineMove(
            first_move.player,
            first_move.hand_slot,
            reflect(first_move.global_grid_index),
        )
        assert second_move in legal_simulation_moves(
            second,
            second.active_player,
        )
        first = apply_simulation_move(first, first_move)
        second = apply_simulation_move(second, second_move)

    assert first.pending_round_result is not None
    assert second.pending_round_result is not None
    assert (
        first.pending_round_result.round_scores
        == second.pending_round_result.round_scores
    )


# An unlisted board retains the engine's full action set without grouping.
def test_unlisted_pattern_keeps_every_legal_action_independent() -> None:
    state = create_game("engine-contract-fixture-1")
    for _ in range(3):
        state = apply_move(
            state,
            legal_moves(state, state.active_player)[0],
        ).state
    information = information_state_from_engine(state)
    legal = {
        index
        for index, allowed in enumerate(
            value for row in information.legal_mask for value in row
        )
        if allowed
    }
    groups = strategic_action_groups(information)

    assert {
        group.representative_action_index for group in groups
    } == legal
    assert all(
        group.member_action_indices
        == (group.representative_action_index,)
        for group in groups
    )


# Spatial groups ignore card identity, but each hand slot remains a distinct action.
def test_card_substitution_does_not_merge_distinct_hand_choices() -> None:
    first = information_state_from_engine(
        create_game("engine-contract-fixture-1")
    )
    second = information_state_from_engine(create_game("different-cards"))

    assert first.own_hand != second.own_hand
    assert _group_shape(first) == _group_shape(second)
    groups = strategic_action_groups(first)
    assert len(groups) == 8
    assert len(
        {group.representative_action_index for group in groups}
    ) == len(groups)
    assert {group.hand_slot for group in groups} == {0, 1, 2, 3}


# The response evaluator enumerates strategic groups, not mirrored concrete actions.
def test_shallow_response_uses_the_same_group_reduction() -> None:
    information = information_state_from_engine(
        create_game("engine-contract-fixture-1")
    )
    response = ShallowGreedyResponseEvaluator(
        ShallowResponseConfig(1)
    ).evaluate(information)

    assert len(response.legal_action_indices) == 16
    assert response.candidate_action_count == 8
    assert response.terminal_evaluation_count == 8
    assert len(response.group_diagnostics) == 8
    for diagnostic in response.group_diagnostics:
        assert diagnostic.visits == 1
        assert len(diagnostic.group.member_action_indices) == 2
        member_values = {
            response.mean_action_values[index]
            for index in diagnostic.group.member_action_indices
        }
        assert member_values == {diagnostic.mean_value}


# Grouping and coin selection depend only on the immutable actor information state.
def test_hidden_world_substitution_cannot_enter_grouping_or_coin_selection() -> None:
    information = information_state_from_engine(
        create_game("engine-contract-fixture-1")
    )
    world_a = sample_determinization(
        information,
        derive_seed("strategic-symmetry-hidden-test-v1", "a"),
    ).state
    world_b = sample_determinization(
        information,
        derive_seed("strategic-symmetry-hidden-test-v1", "b"),
    ).state
    view_a = information_state_from_simulation(world_a)
    view_b = information_state_from_simulation(world_b)
    group = strategic_action_groups(view_a)[0]
    choice_seed = derive_seed("strategic-symmetry-coin-test-v1", "same")

    assert view_a == view_b == information
    assert strategic_action_groups(view_a) == strategic_action_groups(view_b)
    assert select_concrete_action_index(
        view_a,
        group.representative_action_index,
        choice_seed,
    ) == select_concrete_action_index(
        view_b,
        group.representative_action_index,
        choice_seed,
    )
    source = inspect.getsource(strategic_module.strategic_action_groups)
    source += inspect.getsource(
        strategic_module.derive_strategic_destination_choice_seed
    )
    for forbidden in (
        "opponent_hand",
        "stock_order",
        "engine_seed",
        "model_path",
        "hidden_state",
    ):
        assert forbidden not in source


# A forced final placement bypasses strategic grouping exactly as before.
def test_forced_root_behavior_is_unchanged() -> None:
    state = create_game("engine-contract-fixture-1")
    for _ in range(7):
        state = apply_move(
            state,
            legal_moves(state, state.active_player)[0],
        ).state
    information = information_state_from_engine(state)
    config = StrategicSearchConfig(8, 4)
    result = StrategicInformationSetSearch(config).search(
        information,
        derive_strategic_search_request_seed(
            "forced-symmetry",
            information,
            config.digest,
        ),
    )

    assert result.simulation_count == 0
    assert result.group_diagnostics == ()
    assert (
        result.selected_representative_action_index
        == result.selected_action_index
    )

"""Belief-greedy search invariants and information-boundary tests."""

from __future__ import annotations

from dracula.bridge import move_for_action_index
from dracula.engine import (
    apply_move,
    create_game,
    legal_moves,
    other_player,
    state_fingerprint,
)
from dracula.randomness import derive_seed
from dracula.search import (
    BeliefGreedyInformationSetSearch,
    BeliefGreedyResponseEvaluator,
    BeliefGreedySearchConfig,
    SearchInterrupted,
    derive_belief_greedy_request_seed,
    information_state_fingerprint,
    information_state_from_engine,
    sample_determinization,
    strategic_action_groups,
)


def _advance(state, count: int):
    for _ in range(count):
        state = apply_move(
            state,
            legal_moves(state, state.active_player)[0],
        ).state
    return state


def _request(information, config, fixture: str = "belief-greedy-test"):
    return derive_belief_greedy_request_seed(
        fixture,
        information,
        config.digest,
    )


def test_response_evaluates_every_group_under_shared_belief_samples() -> None:
    information = information_state_from_engine(
        _advance(create_game("belief-response-counts"), 4)
    )
    config = BeliefGreedySearchConfig(belief_completion_count=3)
    result = BeliefGreedyResponseEvaluator(config).evaluate(information)
    groups = strategic_action_groups(information, True)

    assert result.belief_completion_count == 3
    assert result.potential_evaluation_count == len(groups) * 3
    assert tuple(item.group for item in result.group_diagnostics) == groups
    assert all(item.visits == 3 for item in result.group_diagnostics)
    assert result.selected_representative_action_index == min(
        (item.group.representative_action_index for item in result.group_diagnostics),
        key=lambda action: (
            -next(
                item.mean_value
                for item in result.group_diagnostics
                if item.group.representative_action_index == action
            ),
            action,
        ),
    )


def test_response_is_reproducible_and_independent_of_hidden_world() -> None:
    root = information_state_from_engine(create_game("belief-hidden-boundary"))
    hidden_a = sample_determinization(
        root,
        derive_seed("belief-hidden-test-v1", "a"),
    ).state
    hidden_b = sample_determinization(
        root,
        derive_seed("belief-hidden-test-v1", "b"),
    ).state
    assert hidden_a.hands[other_player(root.player)] != hidden_b.hands[
        other_player(root.player)
    ]
    view_a = information_state_from_engine(hidden_a)
    view_b = information_state_from_engine(hidden_b)
    assert view_a == view_b == root

    evaluator = BeliefGreedyResponseEvaluator(
        BeliefGreedySearchConfig(belief_completion_count=4)
    )
    assert evaluator.evaluate(view_a) == evaluator.evaluate(view_b)


def test_outer_search_visits_all_root_groups_and_returns_a_legal_move() -> None:
    state = create_game("belief-root-coverage")
    information = information_state_from_engine(state)
    config = BeliefGreedySearchConfig(
        outer_simulation_budget=32,
        belief_completion_count=2,
    )
    result = BeliefGreedyInformationSetSearch(config).search(
        information,
        _request(information, config),
    )
    representatives = {
        item.group.representative_action_index
        for item in result.group_diagnostics
    }

    assert result.simulation_count == 32
    assert sum(item.visits for item in result.group_diagnostics) == 32
    assert all(item.visits > 0 for item in result.group_diagnostics)
    assert result.selected_representative_action_index in representatives
    move = move_for_action_index(
        information.player,
        result.selected_action_index,
    )
    assert move in legal_moves(state, information.player)


def test_search_replay_is_exact_and_authoritative_state_is_immutable() -> None:
    state = _advance(create_game("belief-replay"), 3)
    before = state_fingerprint(state)
    information = information_state_from_engine(state)
    config = BeliefGreedySearchConfig(
        outer_simulation_budget=32,
        belief_completion_count=2,
    )
    request = _request(information, config)
    planner = BeliefGreedyInformationSetSearch(config)

    first = planner.search(information, request)
    repeated = planner.search(information, request)
    assert first.selected_action_index == repeated.selected_action_index
    assert first.action_visits == repeated.action_visits
    assert first.mean_action_values == repeated.mean_action_values
    assert first.group_diagnostics == repeated.group_diagnostics
    assert first.principal_continuation == repeated.principal_continuation
    assert state_fingerprint(state) == before


def test_forced_placement_bypasses_search_and_response_work() -> None:
    state = _advance(create_game("belief-forced"), 7)
    information = information_state_from_engine(state)
    config = BeliefGreedySearchConfig()
    result = BeliefGreedyInformationSetSearch(config).search(
        information,
        _request(information, config),
    )

    assert result.simulation_count == 0
    assert result.response_request_count == 0
    assert result.response_potential_evaluation_count == 0
    assert len(legal_moves(state, information.player)) == 1
    assert move_for_action_index(
        information.player,
        result.selected_action_index,
    ) in legal_moves(state, information.player)


def test_interruption_returns_no_partial_result() -> None:
    information = information_state_from_engine(create_game("belief-interrupt"))
    before = information_state_fingerprint(information)
    config = BeliefGreedySearchConfig(
        outer_simulation_budget=32,
        belief_completion_count=2,
    )
    calls = 0

    def stop() -> bool:
        nonlocal calls
        calls += 1
        return calls > 5

    planner = BeliefGreedyInformationSetSearch(config)
    try:
        planner.search(information, _request(information, config), stop)
    except SearchInterrupted:
        pass
    else:
        raise AssertionError("the configured interruption was not observed")
    assert information_state_fingerprint(information) == before

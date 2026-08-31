"""Sam-128 symmetry, information-boundary, and regression invariants."""

from __future__ import annotations

import hashlib
import inspect
import json
from dataclasses import asdict

import pytest

from dracula.bridge import global_grid_index, move_for_action_index
from dracula.engine import (
    EngineMove,
    EnginePlayer,
    apply_move,
    apply_simulation_move,
    create_game,
    legal_moves,
    legal_simulation_moves,
)
from dracula.randomness import derive_seed
from dracula.search import (
    SAM_TEACHER_DETERMINIZATION_NAMESPACE,
    SAM_TEACHER_REQUEST_NAMESPACE,
    SAM_TEACHER_RESPONSE_DETERMINIZATION_NAMESPACE,
    SAM_TEACHER_RESPONSE_REQUEST_NAMESPACE,
    SAM_TEACHER_RESPONSE_ROLLOUT_NAMESPACE,
    SAM_TEACHER_RESPONSE_SELECTION_NAMESPACE,
    SAM_TEACHER_SEARCH_SCHEMA_VERSION,
    SAM_TEACHER_SELECTION_NAMESPACE,
    SamTeacherInformationSetSearch,
    SamTeacherNodeKey,
    SamTeacherSearchConfig,
    derive_sam_teacher_destination_seed,
    derive_sam_teacher_determinization_seed,
    derive_sam_teacher_request_seed,
    derive_sam_teacher_response_determinization_seed,
    derive_sam_teacher_response_request_seed,
    derive_sam_teacher_response_rollout_seed,
    derive_sam_teacher_response_selection_seed,
    derive_sam_teacher_selection_seed,
    information_state_from_engine,
    information_state_from_simulation,
    normalized_round_return,
    sample_determinization,
    sam_teacher_action_groups,
    select_concrete_action_index,
)
from dracula.search.nested_strategic import (
    NestedStrategicInformationSetSearch,
    NestedStrategicSearchConfig,
    derive_nested_strategic_search_request_seed,
)
from dracula.search.strategic import StrategicSearchConfig


AUTHORIZED_CASES = (
    ({5}, ((2, (2, 8)), (4, (4, 6))), 8),
    ({2, 5}, ((1, (1, 3)), (4, (4, 6)), (8, (8,))), 12),
    ({5, 8}, ((2, (2,)), (4, (4, 6)), (9, (7, 9))), 12),
    ({4, 5}, ((1, (1, 7)), (2, (2, 8)), (6, (6,))), 12),
    ({5, 6}, ((2, (2, 8)), (3, (3, 9)), (4, (4,))), 12),
    ({4, 5, 6}, ((1, (1, 7)), (2, (2, 8)), (3, (3, 9))), 9),
    ({2, 5, 8}, ((1, (1, 3)), (4, (4, 6)), (7, (7, 9))), 9),
)


def _advance(state, count: int):
    for _ in range(count):
        state = apply_move(
            state,
            legal_moves(state, state.active_player)[0],
        ).state
    return state


def _apply_relative_destination(state, perspective, position: int):
    actor = state.active_player
    hand_slot = next(
        index
        for index, card_id in enumerate(state.hands[actor])
        if card_id is not None
    )
    move = EngineMove(
        actor,
        hand_slot,
        global_grid_index(perspective, position - 1),
    )
    assert move in legal_moves(state, actor)
    return apply_move(state, move).state


def _state_for_pattern(occupied: set[int], seed: str):
    state = create_game(seed)
    non_center = sorted(occupied - {5})
    if not non_center:
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
        for group in sam_teacher_action_groups(information)
    )


# Sam-128 uses every row of the exhaustive table and repeats it independently
# for each remaining card; only the two line states have three cards remaining.
@pytest.mark.parametrize(
    ("occupied", "destination_groups", "expected_count"),
    AUTHORIZED_CASES,
)
@pytest.mark.parametrize(
    "game_seed",
    ("engine-contract-fixture-1", "search-role-0"),
)
def test_sam_teacher_uses_exact_authoritative_groups(
    occupied,
    destination_groups,
    expected_count: int,
    game_seed: str,
) -> None:
    information = information_state_from_engine(
        _state_for_pattern(occupied, game_seed)
    )
    slots = tuple(
        index
        for index, card_id in enumerate(information.own_hand)
        if card_id is not None
    )
    assert _group_shape(information) == tuple(
        (slot, representative, members)
        for slot in slots
        for representative, members in destination_groups
    )
    assert len(sam_teacher_action_groups(information)) == expected_count


# The conditional dealer response has three destination groups for each of
# three remaining cards, rather than merging the card-choice dimension.
@pytest.mark.parametrize("occupied", ({4, 5, 6}, {2, 5, 8}))
def test_conditional_line_response_is_three_groups_by_three_cards(
    occupied,
) -> None:
    information = information_state_from_engine(
        _state_for_pattern(occupied, "engine-contract-fixture-1")
    )
    groups = sam_teacher_action_groups(information)
    assert len(groups) == 9
    assert {group.hand_slot for group in groups} == {
        index
        for index, card_id in enumerate(information.own_hand)
        if card_id is not None
    }
    assert all(
        sum(group.hand_slot == slot for group in groups) == 3
        for slot in {group.hand_slot for group in groups}
    )


# Card substitutions cannot change spatial groups, while all four opening hand
# slots remain distinct strategic choices.
def test_different_cards_remain_distinct_strategic_actions() -> None:
    first = information_state_from_engine(
        create_game("engine-contract-fixture-1")
    )
    second = information_state_from_engine(create_game("different-cards"))
    assert first.own_hand != second.own_hand
    assert _group_shape(first) == _group_shape(second)
    assert len(sam_teacher_action_groups(first)) == 8
    assert {group.hand_slot for group in sam_teacher_action_groups(first)} == {
        0,
        1,
        2,
        3,
    }


def test_sam_teacher_configuration_and_seed_domains_are_distinct() -> None:
    information = information_state_from_engine(
        create_game("engine-contract-fixture-1")
    )
    config = SamTeacherSearchConfig()
    request = derive_sam_teacher_request_seed(information, config.digest)
    node = SamTeacherNodeKey(
        information.player,
        hashlib.sha256(b"node").hexdigest(),
    )
    response = derive_sam_teacher_response_request_seed(
        information,
        config.response_digest,
    )
    seeds = {
        request,
        derive_sam_teacher_determinization_seed(request, 0),
        derive_sam_teacher_selection_seed(request, 0, node),
        response,
        derive_sam_teacher_response_determinization_seed(response, 0),
        derive_sam_teacher_response_selection_seed(response, 0, node),
        derive_sam_teacher_response_rollout_seed(
            response,
            0,
            0,
            information,
        ),
    }
    assert len(seeds) == 7
    assert request == derive_sam_teacher_request_seed(
        information,
        config.digest,
    )
    assert config.digest not in {
        NestedStrategicSearchConfig(128, 128).digest,
        StrategicSearchConfig(128, 4).digest,
    }
    assert SAM_TEACHER_SEARCH_SCHEMA_VERSION.endswith("-v1")
    for namespace in (
        SAM_TEACHER_REQUEST_NAMESPACE,
        SAM_TEACHER_DETERMINIZATION_NAMESPACE,
        SAM_TEACHER_SELECTION_NAMESPACE,
        SAM_TEACHER_RESPONSE_REQUEST_NAMESPACE,
        SAM_TEACHER_RESPONSE_DETERMINIZATION_NAMESPACE,
        SAM_TEACHER_RESPONSE_SELECTION_NAMESPACE,
        SAM_TEACHER_RESPONSE_ROLLOUT_NAMESPACE,
    ):
        assert namespace.endswith("-v1")


# The established derived fair coin remains reproducible and reaches both
# concrete members of every authorized pair without changing its group.
@pytest.mark.parametrize(
    "occupied",
    tuple(case[0] for case in AUTHORIZED_CASES),
)
def test_sam_teacher_fair_coin_reaches_both_mirror_members(
    occupied,
) -> None:
    information = information_state_from_engine(
        _state_for_pattern(occupied, "engine-contract-fixture-1")
    )
    config = SamTeacherSearchConfig()
    request = derive_sam_teacher_request_seed(information, config.digest)
    paired_groups = tuple(
        group
        for group in sam_teacher_action_groups(information)
        if len(group.member_action_indices) == 2
    )
    assert paired_groups
    for group in paired_groups:
        choices = []
        for index in range(64):
            choice_seed = derive_sam_teacher_destination_seed(
                request,
                "sam-128-coin-test",
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


# Root statistics live on strategic groups. Only one fair-coin member carries
# the pooled visit count, and both members expose the same pooled mean.
def test_mirrored_destinations_pool_outer_statistics() -> None:
    information = information_state_from_engine(
        create_game("engine-contract-fixture-1")
    )
    config = SamTeacherSearchConfig(8, 32)
    result = SamTeacherInformationSetSearch(config).search(
        information,
        derive_sam_teacher_request_seed(information, config.digest),
    )
    assert len(result.group_diagnostics) == 8
    assert sum(
        diagnostic.visits for diagnostic in result.group_diagnostics
    ) == 8
    for diagnostic in result.group_diagnostics:
        assert diagnostic.visits == 1
        assert len(diagnostic.group.member_action_indices) == 2
        member_visits = tuple(
            result.action_visits[index]
            for index in diagnostic.group.member_action_indices
        )
        assert sorted(member_visits) == [0, 1]
        assert {
            result.mean_action_values[index]
            for index in diagnostic.group.member_action_indices
        } == {diagnostic.mean_value}


# Actor-local response UCT uses the identical card-by-destination groups rather
# than reverting to the historical concrete 32-action response search.
def test_actor_response_search_uses_the_same_grouped_actions() -> None:
    information = information_state_from_engine(
        create_game("engine-contract-fixture-1")
    )
    planner = SamTeacherInformationSetSearch(
        SamTeacherSearchConfig(8, 32)
    )
    response = planner._actor_response(information, None)
    assert tuple(
        diagnostic.group for diagnostic in response.group_diagnostics
    ) == sam_teacher_action_groups(information)
    assert sum(
        diagnostic.visits for diagnostic in response.group_diagnostics
    ) == response.simulation_count == 32
    assert all(diagnostic.visits > 0 for diagnostic in response.group_diagnostics)


# The same information, configuration, and request reproduce the complete
# strategic result; elapsed wall time is intentionally excluded from equality.
def test_sam_teacher_is_deterministic_and_returns_a_legal_action() -> None:
    state = _advance(create_game("engine-contract-fixture-1"), 5)
    information = information_state_from_engine(state)
    config = SamTeacherSearchConfig(8, 8)
    request = derive_sam_teacher_request_seed(information, config.digest)
    first = SamTeacherInformationSetSearch(config).search(
        information,
        request,
    )
    repeated = SamTeacherInformationSetSearch(config).search(
        information,
        request,
    )
    assert first == repeated
    assert first.selected_group is not None
    assert first.selected_action_index in first.selected_group.member_action_indices
    assert first.selected_action_index in {
        index
        for index, allowed in enumerate(
            value for row in information.legal_mask for value in row
        )
        if allowed
    }


# Actor-local response calls receive only the currently acting player's
# immutable view; no enclosing determinization or request seed crosses it.
def test_sam_teacher_opponent_searches_receive_only_actor_views(
    monkeypatch,
) -> None:
    state = _advance(create_game("engine-contract-fixture-1"), 5)
    information = information_state_from_engine(state)
    config = SamTeacherSearchConfig(8, 8)
    planner = SamTeacherInformationSetSearch(config)
    observed = []
    original = SamTeacherInformationSetSearch._actor_response

    def capture(self, actor_information, should_stop):
        observed.append(actor_information)
        return original(self, actor_information, should_stop)

    monkeypatch.setattr(
        SamTeacherInformationSetSearch,
        "_actor_response",
        capture,
    )
    planner.search(
        information,
        derive_sam_teacher_request_seed(information, config.digest),
    )
    assert observed
    assert all(item.active_player is item.player for item in observed)
    assert all(not hasattr(item, "opponent_hand") for item in observed)
    assert all(not hasattr(item, "stock") for item in observed)
    assert tuple(
        inspect.signature(
            SamTeacherInformationSetSearch._actor_response
        ).parameters
    ) == ("self", "actor_information", "should_stop")


# Hidden allocations that project to one actor view produce the same response
# identity and response search result.
def test_sam_teacher_response_is_invariant_to_hidden_world_substitution() -> None:
    information = information_state_from_engine(
        _advance(create_game("engine-contract-fixture-1"), 5)
    )
    first_world = sample_determinization(
        information,
        derive_seed("sam-128-hidden-world-test-v1", "a"),
    ).state
    second_world = sample_determinization(
        information,
        derive_seed("sam-128-hidden-world-test-v1", "b"),
    ).state
    first_view = information_state_from_simulation(first_world)
    second_view = information_state_from_simulation(second_world)
    planner = SamTeacherInformationSetSearch(
        SamTeacherSearchConfig(8, 8)
    )
    assert first_view == second_view == information
    assert planner._actor_response(first_view, None) == planner._actor_response(
        second_view,
        None,
    )


# On the seventh learned placement, every outer sample reaches the engine's
# forced eighth placement, so its backed-up value can be reconstructed exactly.
def test_sam_teacher_terminal_values_equal_exact_engine_scoring() -> None:
    state = _advance(create_game("engine-contract-fixture-1"), 6)
    information = information_state_from_engine(state)
    groups = sam_teacher_action_groups(information)
    config = SamTeacherSearchConfig(2, 2)
    request = derive_sam_teacher_request_seed(information, config.digest)
    result = SamTeacherInformationSetSearch(config).search(
        information,
        request,
    )
    expected = {}
    for simulation_index, group in enumerate(groups):
        sampled = sample_determinization(
            information,
            derive_sam_teacher_determinization_seed(
                request,
                simulation_index,
            ),
        ).state
        concrete = select_concrete_action_index(
            information,
            group.representative_action_index,
            derive_sam_teacher_destination_seed(
                request,
                "sam-128-outer-tree",
                information,
                group.representative_action_index,
                simulation_index,
            ),
        )
        after_root = apply_simulation_move(
            sampled,
            move_for_action_index(sampled.active_player, concrete),
        )
        forced = legal_simulation_moves(
            after_root,
            after_root.active_player,
        )
        assert len(forced) == 1
        terminal = apply_simulation_move(after_root, forced[0])
        assert terminal.pending_round_result is not None
        expected[group.representative_action_index] = (
            normalized_round_return(
                terminal.pending_round_result,
                information.player,
            )
        )
    assert {
        diagnostic.group.representative_action_index: diagnostic.mean_value
        for diagnostic in result.group_diagnostics
    } == expected


# A forced eighth placement is an engine transition, not an outer or response
# search decision, regardless of the configured 128x128 budgets.
def test_sam_teacher_forced_placement_bypasses_both_budgets() -> None:
    information = information_state_from_engine(
        _advance(create_game("engine-contract-fixture-1"), 7)
    )
    result = SamTeacherInformationSetSearch().search(information)
    assert result.simulation_count == 0
    assert result.information_set_count == 0
    assert result.response_request_count == 0
    assert result.response_simulation_count == 0
    assert result.total_terminal_evaluation_count == 0
    assert result.group_diagnostics == ()


# The live 32x32 Sam implementation remains a frozen comparison control.
def test_live_sam_32x32_golden_result_is_unchanged() -> None:
    information = information_state_from_engine(
        _advance(create_game("engine-contract-fixture-1"), 5)
    )
    config = NestedStrategicSearchConfig(32, 32)
    result = NestedStrategicInformationSetSearch(config).search(
        information,
        derive_nested_strategic_search_request_seed(
            "sam-32-golden",
            information,
            config.digest,
        ),
    )
    payload = asdict(result)
    payload.pop("elapsed_seconds")
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    )
    assert config.digest == (
        "15dfad00db2963b81d7d94745b431cc2d649a51b0847c097a80be2da02e45385"
    )
    assert hashlib.sha256(encoded.encode("utf-8")).hexdigest() == (
        "ddf703958adc9f210a6ca070aef8e595f3f78425a68f2715e460d633fd24c242"
    )

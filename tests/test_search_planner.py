"""Information-set search invariants and privileged-diagnostic isolation."""

from __future__ import annotations

import inspect

import pytest

import dracula.search as public_search
from dracula.bridge import build_policy_turn_context
from dracula.engine import (
    EnginePlayer,
    EngineRoundResult,
    EngineStatus,
    MultiplierReason,
    PlayerValues,
    apply_move,
    create_game,
    legal_moves,
    other_player,
    resolve_round_scores,
    score_coffin,
    state_fingerprint,
)
from dracula.search import (
    InformationSetSearch,
    SearchConfig,
    SearchContractViolation,
    SearchInterrupted,
    derive_belief_sample_seed,
    derive_rollout_choice_seed,
    derive_search_request_seed,
    information_state_from_engine,
    normalized_round_return,
    sample_determinization,
    sample_uniform_action_index,
)
from dracula.search.diagnostic import solve_perfect_information_round


def _advance(state, count: int):
    for _ in range(count):
        state = apply_move(state, legal_moves(state, state.active_player)[0]).state
    return state


def _request(information, config: SearchConfig, fixture: str = "planner-test") -> bytes:
    return derive_search_request_seed(fixture, information, config.digest)


def _search_at_turn(move_count: int, budget: int = 32):
    state = _advance(create_game("engine-contract-fixture-1"), move_count)
    information = information_state_from_engine(state)
    config = SearchConfig(simulation_budget=budget)
    return state, information, InformationSetSearch(config).search(
        information, _request(information, config)
    )


# Root statistics are indexed by the bridge action table and cannot invent actions.
def test_root_visits_exactly_cover_engine_legal_actions_and_backups() -> None:
    state, information, result = _search_at_turn(0, 32)
    context = build_policy_turn_context(state, state.active_player)
    expected = {
        index for index, move in enumerate(context.action_table) if move is not None
    }
    visited = {index for index, count in enumerate(result.action_visits) if count}

    assert visited == expected
    assert sum(result.action_visits) == result.simulation_count == 32
    assert all(result.action_visits[index] >= 1 for index in expected)
    assert {
        index for index, value in enumerate(result.mean_action_values) if value is not None
    } == expected
    assert result.selected_action_index in expected


# A two-action late state makes each initial backup independently reproducible.
def test_late_round_action_means_equal_the_exact_sampled_terminal_returns() -> None:
    state = _advance(create_game("engine-contract-fixture-1"), 6)
    information = information_state_from_engine(state)
    legal = tuple(
        index
        for index, allowed in enumerate(
            value for row in information.legal_mask for value in row
        )
        if allowed
    )
    assert len(legal) == 2
    config = SearchConfig(simulation_budget=2)
    request = _request(information, config, "exact-backup")
    result = InformationSetSearch(config).search(information, request)

    expected: dict[int, float] = {}
    for simulation_index, action_index in enumerate(legal):
        sampled = sample_determinization(
            information, derive_belief_sample_seed(request, simulation_index)
        )
        context = build_policy_turn_context(sampled.state, sampled.state.active_player)
        move = context.action_table[action_index]
        assert move is not None
        after_root = apply_move(sampled.state, move).state
        forced = legal_moves(after_root, after_root.active_player)
        assert len(forced) == 1
        terminal = apply_move(after_root, forced[0]).state
        assert terminal.pending_round_result is not None
        expected[action_index] = normalized_round_return(
            terminal.pending_round_result, information.player
        )

    assert result.action_visits[legal[0]] == result.action_visits[legal[1]] == 1
    assert result.mean_action_values[legal[0]] == expected[legal[0]]
    assert result.mean_action_values[legal[1]] == expected[legal[1]]


# Zero-sum backup is one exact engine score difference viewed from opposite seats.
def test_round_return_reverses_sign_between_player_perspectives() -> None:
    terminal = _advance(create_game("engine-contract-fixture-1"), 8)
    assert terminal.pending_round_result is not None
    queen = normalized_round_return(terminal.pending_round_result, EnginePlayer.QUEEN)
    king = normalized_round_return(terminal.pending_round_result, EnginePlayer.KING)
    assert queen == -king


# Search reads the engine result after every multiplier and ranked-line rule is resolved.
@pytest.mark.parametrize(
    ("first_row", "multiplier", "reason"),
    (
        (("8H", "9H", "10H"), 5, MultiplierReason.SAME_SUIT),
        (("8H", "9H", "10D"), 3, MultiplierReason.SAME_COLOR),
        (("8H", "3H", "8C"), 2, MultiplierReason.SUIT_PAIR),
        (("10H", "V1", "9H"), 0, MultiplierReason.VAMPIRE),
    ),
)
def test_terminal_return_uses_exact_multiplier_vampire_and_ranked_line_score(
    first_row: tuple[str, str, str],
    multiplier: int,
    reason: MultiplierReason,
) -> None:
    tail = tuple(
        card
        for card in ("AC", "2C", "3D", "4S", "5C", "V2", "6D", "7S")
        if card not in first_row
    )[:6]
    coffin = (*first_row, *tail)
    lines = score_coffin(coffin)
    assert lines.queen[0].multiplier == multiplier
    assert lines.queen[0].multiplier_reason is reason
    scores = resolve_round_scores(
        tuple(line.total for line in lines.queen),
        tuple(line.total for line in lines.king),
    )
    result = EngineRoundResult(
        round_number=1,
        dealer=EnginePlayer.QUEEN,
        coffin=coffin,
        moves=(),
        line_scores=lines,
        round_scores=scores,
    )
    assert normalized_round_return(result, EnginePlayer.QUEEN) == (
        scores.queen - scores.king
    ) / 150


# Forced placements traverse the engine but do not allocate or consume search work.
def test_forced_root_bypasses_search_and_forced_rollout_has_no_decision_budget() -> None:
    _, information, forced = _search_at_turn(7, 100)
    assert forced.simulation_count == 0
    assert forced.information_set_count == 0
    assert sum(forced.action_visits) == 0
    assert information.legal_mask[forced.selected_action_index // 8][
        forced.selected_action_index % 8
    ]

    _, _, learned = _search_at_turn(6, 12)
    assert learned.simulation_count == sum(learned.action_visits) == 12
    assert learned.principal_continuation is not None
    assert len(learned.principal_continuation.steps) == 2
    assert learned.principal_continuation.steps[-1].forced is True


# Configuration plus information plus seed is the complete reproducibility boundary.
def test_search_is_deterministic_and_budget_changes_preserve_invariants() -> None:
    state = create_game("engine-contract-fixture-1")
    information = information_state_from_engine(state)
    small_config = SearchConfig(simulation_budget=24, exploration_constant=1.1)
    small_seed = _request(information, small_config, "repeat")
    first = InformationSetSearch(small_config).search(information, small_seed)
    repeated = InformationSetSearch(small_config).search(information, small_seed)
    assert first == repeated

    large_config = SearchConfig(simulation_budget=48, exploration_constant=1.1)
    larger = InformationSetSearch(large_config).search(
        information, _request(information, large_config, "repeat")
    )
    legal_count = len(legal_moves(state, state.active_player))
    assert sum(larger.action_visits) == larger.simulation_count == 48
    assert sum(count > 0 for count in larger.action_visits) == legal_count
    assert larger.selected_action_index in {
        index for index, count in enumerate(larger.action_visits) if count
    }


# Indistinguishable private worlds collapse to one root and one search result.
def test_hidden_world_permutations_cannot_change_root_identity_or_search() -> None:
    root = information_state_from_engine(create_game("engine-contract-fixture-1"))
    request = derive_belief_sample_seed(_request(root, SearchConfig(24)), 99)
    first_world = sample_determinization(root, request).state
    second_world = sample_determinization(
        root, derive_belief_sample_seed(request, 1)
    ).state
    first_view = information_state_from_engine(first_world)
    second_view = information_state_from_engine(second_world)
    config = SearchConfig(24)
    search_seed = _request(first_view, config, "same-information")

    assert first_world.hands[other_player(root.player)] != second_world.hands[
        other_player(root.player)
    ]
    assert InformationSetSearch(config).search(first_view, search_seed) == (
        InformationSetSearch(config).search(second_view, search_seed)
    )


# Cancellation discards local statistics; immutable authoritative state stays byte-identical.
def test_interrupted_search_cannot_mutate_authoritative_engine_state() -> None:
    state = create_game("engine-contract-fixture-1")
    before = state_fingerprint(state)
    information = information_state_from_engine(state)
    config = SearchConfig(32)
    checks = 0

    def stop() -> bool:
        nonlocal checks
        checks += 1
        return checks == 6

    with pytest.raises(SearchInterrupted):
        InformationSetSearch(config).search(
            information, _request(information, config, "cancel"), stop
        )
    assert state_fingerprint(state) == before
    assert state.status is EngineStatus.PLAYING


# The deployable API cannot receive private state; privileged analysis requires an explicit guard.
def test_perfect_information_solver_is_separate_and_cannot_enter_live_path() -> None:
    state = _advance(create_game("engine-contract-fixture-1"), 6)
    planner = InformationSetSearch(SearchConfig(8))
    assert "EngineState" not in str(inspect.signature(planner.search))
    assert not hasattr(public_search, "solve_perfect_information_round")
    with pytest.raises(SearchContractViolation):
        planner.search(state, b"x" * 32)  # type: ignore[arg-type]
    with pytest.raises(PermissionError):
        solve_perfect_information_round(state)

    before = state_fingerprint(state)
    diagnostic = solve_perfect_information_round(
        state, diagnostic_only=True, maximum_states=10_000
    )
    assert diagnostic.selected_action_index in {
        index
        for index, move in enumerate(
            build_policy_turn_context(state, state.active_player).action_table
        )
        if move is not None
    }
    assert state_fingerprint(state) == before


# Rollout sampling is driven by the acting player's view and an explicit seed only.
def test_opponent_rollout_choice_uses_its_information_and_exact_engine_legality() -> None:
    state = create_game("engine-contract-fixture-1")
    root = information_state_from_engine(state)
    sample = sample_determinization(
        root, derive_belief_sample_seed(_request(root, SearchConfig(24)), 0)
    ).state
    root_move = legal_moves(sample, sample.active_player)[0]
    opponent_state = apply_move(sample, root_move).state
    opponent_view = information_state_from_engine(opponent_state)
    choice_seed = derive_rollout_choice_seed(b"r" * 32, 0, 1)
    action_index = sample_uniform_action_index(opponent_view, choice_seed)
    context = build_policy_turn_context(opponent_state, opponent_state.active_player)
    assert context.action_table[action_index] in legal_moves(
        opponent_state, opponent_state.active_player
    )

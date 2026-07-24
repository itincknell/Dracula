"""Teacher v2 shallow-response information, value, and replay invariants."""

from __future__ import annotations

import hashlib
import inspect
import json
from dataclasses import asdict

import pytest

import dracula.search as public_search
import dracula.search.strategic as strategic_module
from dracula.bridge import action_index_for_move, build_policy_turn_context
from dracula.engine import (
    EnginePlayer,
    EngineStatus,
    apply_move,
    apply_simulation_move,
    create_game,
    legal_moves,
    other_player,
    state_fingerprint,
)
from dracula.randomness import derive_seed
from dracula.search import (
    GREEDY_RESPONSE_DETERMINIZATION_NAMESPACE,
    GREEDY_RESPONSE_REQUEST_NAMESPACE,
    GREEDY_RESPONSE_ROLLOUT_NAMESPACE,
    GREEDY_RESPONSE_SCHEMA_VERSION,
    STRATEGIC_DETERMINIZATION_NAMESPACE,
    STRATEGIC_SEARCH_REQUEST_NAMESPACE,
    STRATEGIC_SEARCH_SCHEMA_VERSION,
    STRATEGIC_SELECTION_NAMESPACE,
    InformationSetSearch,
    SearchConfig,
    SearchContractViolation,
    SearchInterrupted,
    SearchResult,
    ShallowGreedyResponseEvaluator,
    ShallowResponseConfig,
    StrategicInformationSetSearch,
    StrategicNodeKey,
    StrategicSearchConfig,
    actor_relative_value,
    derive_greedy_response_determinization_seed,
    derive_greedy_response_request_seed,
    derive_greedy_response_rollout_seed,
    derive_search_request_seed,
    derive_strategic_determinization_seed,
    derive_strategic_search_request_seed,
    derive_strategic_selection_seed,
    information_state_fingerprint,
    information_state_from_engine,
    information_state_from_simulation,
    normalized_round_return,
    sample_determinization,
)
from dracula.search.diagnostic import (
    solve_adversarial_perfect_information_round,
)


def _advance(state, count: int):
    for _ in range(count):
        state = apply_move(state, legal_moves(state, state.active_player)[0]).state
    return state


def _strategic_request(information, config, label: str = "strategic-test"):
    return derive_strategic_search_request_seed(
        label, information, config.digest
    )


def _legal_action_indexes(information) -> tuple[int, ...]:
    return tuple(
        index
        for index, allowed in enumerate(
            value for row in information.legal_mask for value in row
        )
        if allowed
    )


@pytest.mark.parametrize("completions", (1, 2, 4))
def test_response_enumerates_every_candidate_for_every_completion(
    completions: int,
) -> None:
    information = information_state_from_engine(
        _advance(create_game("engine-contract-fixture-1"), 5)
    )
    legal = _legal_action_indexes(information)
    response = ShallowGreedyResponseEvaluator(
        ShallowResponseConfig(completions)
    ).evaluate(information)

    # Every legal action receives K exact terminal observations; masked actions
    # receive no value that could influence the greedy comparison.
    assert response.legal_action_indices == legal
    assert response.candidate_action_count == len(legal)
    assert response.terminal_evaluation_count == len(legal) * completions
    assert all(response.mean_action_values[index] is not None for index in legal)
    assert all(
        response.mean_action_values[index] is None
        for index in set(range(32)) - set(legal)
    )


def test_candidate_actions_share_one_hidden_world_per_completion(
    monkeypatch,
) -> None:
    information = information_state_from_engine(
        _advance(create_game("engine-contract-fixture-1"), 5)
    )
    legal = _legal_action_indexes(information)
    sampled_states = []
    candidate_starts: dict[int, int] = {}
    original_sample = strategic_module.sample_determinization
    original_apply = strategic_module.apply_simulation_move

    def capture_sample(actor_information, seed):
        sampled = original_sample(actor_information, seed)
        sampled_states.append(sampled.state)
        return sampled

    def capture_apply(state, move):
        for completion_index, shared in enumerate(sampled_states):
            if state is shared:
                candidate_starts[completion_index] = (
                    candidate_starts.get(completion_index, 0) + 1
                )
                break
        return original_apply(state, move)

    monkeypatch.setattr(
        strategic_module, "sample_determinization", capture_sample
    )
    monkeypatch.setattr(strategic_module, "apply_simulation_move", capture_apply)
    ShallowGreedyResponseEvaluator(ShallowResponseConfig(2)).evaluate(
        information
    )

    # Sampling once per completion, rather than once per candidate, is the
    # common-random-world invariant used to compare actions fairly.
    assert len(sampled_states) == 2
    assert candidate_starts == {0: len(legal), 1: len(legal)}


def test_outer_continuation_uses_actor_local_views_for_both_players(
    monkeypatch,
) -> None:
    state = _advance(create_game("engine-contract-fixture-1"), 4)
    information = information_state_from_engine(state)
    config = StrategicSearchConfig(6, 1)
    planner = StrategicInformationSetSearch(config)
    observed = []
    original = StrategicInformationSetSearch._actor_response

    def capture(self, actor_information, should_stop):
        observed.append(actor_information)
        return original(self, actor_information, should_stop)

    monkeypatch.setattr(StrategicInformationSetSearch, "_actor_response", capture)
    result = planner.search(information, _strategic_request(information, config))

    assert {item.player for item in observed} == set(EnginePlayer)
    assert all(item.active_player is item.player for item in observed)
    assert all(not hasattr(item, "opponent_hand") for item in observed)
    assert all(not hasattr(item, "stock") for item in observed)
    assert result.response_request_count >= len(observed) > 0


def test_response_boundary_has_no_private_or_model_channel() -> None:
    parameters = tuple(
        inspect.signature(StrategicInformationSetSearch._actor_response).parameters
    )
    assert parameters == ("self", "information", "should_stop")
    source = inspect.getsource(StrategicInformationSetSearch._actor_response)
    for forbidden in (
        "request_seed",
        "opponent_hand",
        "stock_order",
        "engine_seed",
        "model_path",
        "hidden_state",
    ):
        assert forbidden not in source
    module_source = inspect.getsource(
        inspect.getmodule(StrategicInformationSetSearch)
    )
    for forbidden in ("policy_adapter", "model_path", "raw_logits", "hidden_state"):
        assert forbidden not in module_source


def test_seed_namespaces_are_reproducible_and_separated() -> None:
    information = information_state_from_engine(
        create_game("engine-contract-fixture-1")
    )
    config = StrategicSearchConfig(16, 2)
    response_seed = derive_greedy_response_request_seed(
        information, config.response_config.digest
    )
    request = _strategic_request(information, config, "namespaces")
    node = StrategicNodeKey(
        information.player, information_state_fingerprint(information)
    )
    values = {
        request,
        derive_strategic_determinization_seed(request, 0),
        derive_strategic_selection_seed(request, 0, node),
        response_seed,
        derive_greedy_response_determinization_seed(response_seed, 0),
        derive_greedy_response_rollout_seed(
            response_seed, 0, 0, 0, information
        ),
    }

    assert len(values) == 6
    assert request == _strategic_request(information, config, "namespaces")
    assert STRATEGIC_SEARCH_SCHEMA_VERSION.endswith("-v2")
    assert STRATEGIC_SEARCH_REQUEST_NAMESPACE.endswith("-v2")
    assert STRATEGIC_DETERMINIZATION_NAMESPACE.endswith("-v2")
    assert STRATEGIC_SELECTION_NAMESPACE.endswith("-v2")
    assert GREEDY_RESPONSE_SCHEMA_VERSION.endswith("-v1")
    assert GREEDY_RESPONSE_REQUEST_NAMESPACE.endswith("-v1")
    assert GREEDY_RESPONSE_DETERMINIZATION_NAMESPACE.endswith("-v1")
    assert GREEDY_RESPONSE_ROLLOUT_NAMESPACE.endswith("-v1")


def test_hidden_world_substitutions_cannot_change_response_policy() -> None:
    state = _advance(create_game("engine-contract-fixture-1"), 5)
    information = information_state_from_engine(state)
    world_a = sample_determinization(
        information, derive_seed("strategic-hidden-test-v2", "a")
    ).state
    world_b = sample_determinization(
        information, derive_seed("strategic-hidden-test-v2", "b")
    ).state
    view_a = information_state_from_simulation(world_a)
    view_b = information_state_from_simulation(world_b)
    evaluator = ShallowGreedyResponseEvaluator(ShallowResponseConfig(2))

    assert (
        world_a.hands[other_player(information.player)]
        != world_b.hands[other_player(information.player)]
        or world_a.stock != world_b.stock
    )
    assert view_a == view_b == information
    assert derive_greedy_response_request_seed(
        view_a, evaluator.config.digest
    ) == derive_greedy_response_request_seed(view_b, evaluator.config.digest)
    assert evaluator.evaluate(view_a) == evaluator.evaluate(view_b)


def test_node_identity_includes_actor_and_information_fingerprint() -> None:
    digest = "a" * 64
    assert StrategicNodeKey(EnginePlayer.QUEEN, digest) != StrategicNodeKey(
        EnginePlayer.KING, digest
    )


@pytest.mark.parametrize(
    "game_seed", ("engine-contract-fixture-1", "search-role-0")
)
def test_response_uses_exact_actor_relative_terminal_values(game_seed: str) -> None:
    information = information_state_from_engine(
        _advance(create_game(game_seed), 6)
    )
    config = ShallowResponseConfig(1)
    evaluator = ShallowGreedyResponseEvaluator(config)
    response = evaluator.evaluate(information)
    response_seed = derive_greedy_response_request_seed(
        information, config.digest
    )
    shared = sample_determinization(
        information,
        derive_greedy_response_determinization_seed(response_seed, 0),
    ).state
    legal = _legal_action_indexes(information)
    expected = {}
    terminal_results = {}

    for action_index in legal:
        move = build_policy_turn_context(
            shared, shared.active_player
        ).action_table[action_index]
        assert move is not None
        after_actor = apply_simulation_move(shared, move)
        forced = legal_moves(after_actor, after_actor.active_player)
        assert len(forced) == 1
        terminal = apply_simulation_move(after_actor, forced[0])
        assert terminal.pending_round_result is not None
        terminal_results[action_index] = terminal.pending_round_result
        expected[action_index] = normalized_round_return(
            terminal.pending_round_result, information.player
        )

    assert all(
        response.mean_action_values[index] == expected[index] for index in legal
    )
    assert response.selected_action_index == min(
        legal, key=lambda index: (-expected[index], index)
    )
    assert all(
        expected[index]
        == -normalized_round_return(
            terminal_results[index],
            other_player(information.player),
        )
        for index in legal
    )


def test_equal_response_values_use_canonical_action_tie_break(
    monkeypatch,
) -> None:
    information = information_state_from_engine(
        _advance(create_game("engine-contract-fixture-1"), 6)
    )
    legal = _legal_action_indexes(information)
    monkeypatch.setattr(
        strategic_module, "normalized_round_return", lambda _result, _player: 0.0
    )

    response = ShallowGreedyResponseEvaluator().evaluate(information)

    assert response.selected_action_index == min(legal)
    assert all(response.mean_action_values[index] == 0.0 for index in legal)


def test_actor_relative_value_conversion_is_exact() -> None:
    assert actor_relative_value(0.375, EnginePlayer.QUEEN, EnginePlayer.QUEEN) == 0.375
    assert actor_relative_value(0.375, EnginePlayer.QUEEN, EnginePlayer.KING) == -0.375
    assert actor_relative_value(-0.2, EnginePlayer.KING, EnginePlayer.QUEEN) == 0.2
    with pytest.raises(SearchContractViolation):
        actor_relative_value(float("nan"), EnginePlayer.QUEEN, EnginePlayer.KING)


@pytest.mark.parametrize(
    "game_seed", ("engine-contract-fixture-1", "search-role-0")
)
def test_outer_terminal_values_and_backup_signs_are_exact(game_seed: str) -> None:
    information = information_state_from_engine(
        _advance(create_game(game_seed), 6)
    )
    config = StrategicSearchConfig(2, 1)
    request = _strategic_request(information, config, f"exact-{game_seed}")
    result = StrategicInformationSetSearch(config).search(information, request)
    legal = _legal_action_indexes(information)

    for simulation_index, action_index in enumerate(legal):
        sampled = sample_determinization(
            information,
            derive_strategic_determinization_seed(request, simulation_index),
        )
        move = build_policy_turn_context(
            sampled.state, sampled.state.active_player
        ).action_table[action_index]
        assert move is not None
        after_root = apply_simulation_move(sampled.state, move)
        forced = legal_moves(after_root, after_root.active_player)
        terminal = apply_simulation_move(after_root, forced[0])
        assert terminal.pending_round_result is not None
        value = normalized_round_return(
            terminal.pending_round_result, information.player
        )
        assert result.mean_action_values[action_index] == value
        assert value == -normalized_round_return(
            terminal.pending_round_result, other_player(information.player)
        )


def test_legal_actions_are_stable_across_determinizations_and_responses() -> None:
    information = information_state_from_engine(
        _advance(create_game("engine-contract-fixture-1"), 5)
    )
    legal = set(_legal_action_indexes(information))
    for index in range(8):
        sampled = sample_determinization(
            information, derive_seed("strategic-legality-test-v2", str(index))
        )
        sampled_view = information_state_from_simulation(sampled.state)
        assert set(_legal_action_indexes(sampled_view)) == legal

    config = StrategicSearchConfig(6, 1)
    result = StrategicInformationSetSearch(config).search(
        information, _strategic_request(information, config, "legality")
    )
    assert {index for index, count in enumerate(result.action_visits) if count} == legal
    assert result.selected_action_index in legal


def test_forced_placements_consume_no_response_or_decision_budget() -> None:
    forced_information = information_state_from_engine(
        _advance(create_game("engine-contract-fixture-1"), 7)
    )
    config = StrategicSearchConfig(20, 4)
    forced = StrategicInformationSetSearch(config).search(
        forced_information,
        _strategic_request(forced_information, config, "forced-root"),
    )
    assert forced.simulation_count == 0
    assert forced.response_request_count == 0
    assert forced.response_terminal_evaluation_count == 0
    assert forced.total_terminal_evaluation_count == 0
    assert forced.information_set_count == 0

    learned_information = information_state_from_engine(
        _advance(create_game("engine-contract-fixture-1"), 6)
    )
    learned_config = StrategicSearchConfig(2, 4)
    learned = StrategicInformationSetSearch(learned_config).search(
        learned_information,
        _strategic_request(learned_information, learned_config, "forced-tail"),
    )
    assert learned.response_request_count == 0
    assert learned.principal_continuation is not None
    assert learned.principal_continuation.steps[-1].forced is True


def test_search_is_deterministic_and_accounts_for_all_work() -> None:
    information = information_state_from_engine(
        _advance(create_game("engine-contract-fixture-1"), 5)
    )
    config = StrategicSearchConfig(6, 2)
    request = _strategic_request(information, config, "repeat")
    first = StrategicInformationSetSearch(config).search(information, request)
    repeated = StrategicInformationSetSearch(config).search(information, request)

    assert first == repeated
    assert sum(first.action_visits) == first.simulation_count == 6
    assert first.response_request_count == (
        first.unique_response_evaluation_count + first.response_cache_hit_count
    )
    assert first.response_terminal_evaluation_count == (
        config.response_completions_per_action
        * first.response_candidate_action_count
    )
    assert first.total_terminal_evaluation_count == (
        first.simulation_count + first.response_terminal_evaluation_count
    )
    assert first.peak_resident_memory_bytes > 0


def test_interruption_cannot_mutate_authoritative_state_or_return_result() -> None:
    state = _advance(create_game("engine-contract-fixture-1"), 5)
    before = state_fingerprint(state)
    information = information_state_from_engine(state)
    config = StrategicSearchConfig(6, 2)
    checks = 0

    def stop() -> bool:
        nonlocal checks
        checks += 1
        return checks == 5

    with pytest.raises(SearchInterrupted):
        StrategicInformationSetSearch(config).search(
            information,
            _strategic_request(information, config, "interrupt"),
            stop,
        )
    assert state_fingerprint(state) == before
    assert state.status is EngineStatus.PLAYING


@pytest.mark.parametrize("completions", (0, 3, 5))
def test_response_configuration_rejects_unsupported_counts(
    completions: int,
) -> None:
    with pytest.raises(SearchContractViolation):
        ShallowResponseConfig(completions)
    with pytest.raises(SearchContractViolation):
        StrategicSearchConfig(10, completions)


@pytest.mark.parametrize(
    ("move_count", "budget", "label", "expected_digest"),
    (
        (0, 16, "early", "1022e050ac3001580f143b7804bdc432e6601fb3078690a547567a6a602fd316"),
        (3, 24, "middle", "f57ec6655efc8f60c6f70bfbc134b362ddb13b5293f36fbbbf1ade8878b7e86a"),
        (6, 8, "late", "a9858248b3d19e30e6155ae0c5979653288a5c4ac3bb4038ce50bda8317d382d"),
    ),
)
def test_v1_behavior_remains_frozen_on_golden_fixtures(
    move_count: int, budget: int, label: str, expected_digest: str
) -> None:
    information = information_state_from_engine(
        _advance(create_game("engine-contract-fixture-1"), move_count)
    )
    config = SearchConfig(budget)
    request = derive_search_request_seed(
        f"v1-golden-{label}", information, config.digest
    )
    result = InformationSetSearch(config).search(information, request)
    payload = json.dumps(asdict(result), sort_keys=True, separators=(",", ":"))
    assert hashlib.sha256(payload.encode("utf-8")).hexdigest() == expected_digest


def test_result_is_teacher_and_gameplay_boundary_compatible() -> None:
    state = _advance(create_game("engine-contract-fixture-1"), 5)
    information = information_state_from_engine(state)
    config = StrategicSearchConfig(6, 2)
    result = StrategicInformationSetSearch(config).search(
        information, _strategic_request(information, config, "collector")
    )

    assert isinstance(result, SearchResult)
    assert result.information_state_fingerprint == information_state_fingerprint(
        information
    )
    assert result.config_digest == config.digest
    assert sum(result.action_visits) == result.simulation_count
    policy = tuple(count / result.simulation_count for count in result.action_visits)
    assert sum(policy) == pytest.approx(1.0)
    assert all(
        probability == 0.0
        for probability, legal in zip(
            policy,
            (value for row in information.legal_mask for value in row),
            strict=True,
        )
        if not legal
    )
    assert build_policy_turn_context(
        state, state.active_player
    ).action_table[result.selected_action_index] in legal_moves(
        state, state.active_player
    )


def test_adversarial_perfect_information_control_is_diagnostic_only() -> None:
    state = _advance(create_game("engine-contract-fixture-1"), 6)
    before = state_fingerprint(state)
    assert not hasattr(public_search, "solve_adversarial_perfect_information_round")
    with pytest.raises(PermissionError):
        solve_adversarial_perfect_information_round(state)
    result = solve_adversarial_perfect_information_round(
        state, diagnostic_only=True, maximum_states=10_000
    )
    assert result.selected_action_index in {
        action_index_for_move(move, state.active_player)
        for move in legal_moves(state, state.active_player)
    }
    assert state_fingerprint(state) == before

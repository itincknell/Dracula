"""Neural-guided information-set search invariants."""

from __future__ import annotations

import inspect
from dataclasses import fields

import pytest
import torch

import dracula.search.guided as guided_module
from dracula.engine import (
    EnginePlayer,
    EngineStatus,
    apply_move,
    create_game,
    legal_moves,
    state_fingerprint,
)
from dracula.policy_value import PolicyValueModel
from dracula.search import (
    GuidedInformationSetSearch,
    GuidedLeafValueMode,
    GuidedSearchConfig,
    InformationSetSearch,
    PolicyValueEvaluation,
    PolicyValueModelEvaluator,
    SearchConfig,
    SearchInformationState,
    SearchInterrupted,
    UniformPolicyValueEvaluator,
    derive_search_request_seed,
    information_state_fingerprint,
    information_state_from_engine,
    player_relative_value_for_root,
)


def _advance(state, count: int):
    for _ in range(count):
        state = apply_move(state, legal_moves(state, state.active_player)[0]).state
    return state


def _deterministic_result(result):
    return (
        result.information_state_fingerprint,
        result.config_digest,
        result.evaluator_digest,
        result.selected_action_index,
        result.action_visits,
        result.mean_action_values,
        result.simulation_count,
        result.information_set_count,
        result.model_evaluation_count,
        result.terminal_evaluation_count,
        result.cutoff_evaluation_count,
        result.principal_continuation,
    )


class _RecordingEvaluator:
    def __init__(self, *, value: float = 0.0, favored_action: int | None = None) -> None:
        self.states: list[SearchInformationState] = []
        self.value = value
        self.favored_action = favored_action

    @property
    def digest(self) -> str:
        return "7" * 64

    def evaluate_batch(self, states):
        results = []
        for state in states:
            assert isinstance(state, SearchInformationState)
            self.states.append(state)
            legal = tuple(
                index
                for index, allowed in enumerate(
                    value for row in state.legal_mask for value in row
                )
                if allowed
            )
            if self.favored_action in legal:
                priors = tuple(
                    1.0 if index == self.favored_action else 0.0
                    for index in range(32)
                )
            else:
                priors = tuple(
                    1.0 / len(legal) if index in legal else 0.0
                    for index in range(32)
                )
            results.append(
                PolicyValueEvaluation(
                    information_state_fingerprint(state), priors, self.value
                )
            )
        return tuple(results)


# Uniform policy guidance must preserve the fixed control during canonical coverage.
def test_uniform_priors_reproduce_search_only_selection_contract() -> None:
    state = create_game("guided-uniform-control")
    information = information_state_from_engine(state)
    legal_count = len(legal_moves(state, state.active_player))
    request_seed = b"u" * 32
    control = InformationSetSearch(SearchConfig(legal_count)).search(
        information, request_seed
    )
    guided = GuidedInformationSetSearch(
        UniformPolicyValueEvaluator(), GuidedSearchConfig(legal_count)
    ).search(information, request_seed)
    assert guided.action_visits == control.action_visits
    assert guided.mean_action_values == control.mean_action_values
    assert guided.selected_action_index == control.selected_action_index
    assert guided.terminal_evaluation_count == legal_count
    assert guided.cutoff_evaluation_count == 0


# PUCT may bias exploration only among legal actions after every action has evidence.
def test_policy_priors_influence_puct_without_bypassing_legality() -> None:
    legal = (2, 5, 11)
    priors = tuple(0.9 if index == 5 else 0.05 if index in legal else 0.0 for index in range(32))
    node = guided_module._GuidedNode.create(legal, priors)
    node.visits = 3
    for index in legal:
        node.action_visits[index] = 1
    selected = guided_module._select_puct_action(
        node, GuidedSearchConfig(3), b"p" * 32
    )
    assert selected == 5
    assert selected in legal


# Every simulated actor is evaluated from that actor's typed sampled view.
def test_opponent_inference_receives_only_its_sampled_information_state() -> None:
    state = create_game("guided-opponent-view")
    root = information_state_from_engine(state)
    evaluator = _RecordingEvaluator()
    planner = GuidedInformationSetSearch(evaluator, GuidedSearchConfig(16))
    planner.search(root, b"o" * 32)
    assert {item.player for item in evaluator.states} == set(EnginePlayer)
    opponent_states = [item for item in evaluator.states if item.player is not root.player]
    assert opponent_states
    assert all(item.active_player is item.player for item in opponent_states)
    assert {field.name for field in fields(SearchInformationState)}.isdisjoint(
        {"hands", "opponent_hand", "stock", "engine_seed", "model_path"}
    )
    assert "EngineState" not in str(inspect.signature(evaluator.evaluate_batch))


# A value is actor-relative at inference and root-relative at backup.
def test_value_conversion_reverses_for_the_other_actor_and_cutoff_backup() -> None:
    assert player_relative_value_for_root(0.4, EnginePlayer.QUEEN, EnginePlayer.QUEEN) == 0.4
    assert player_relative_value_for_root(0.4, EnginePlayer.KING, EnginePlayer.QUEEN) == -0.4

    state = create_game("guided-value-sign")
    information = information_state_from_engine(state)
    evaluator = _RecordingEvaluator(value=0.4)
    config = GuidedSearchConfig(
        16, leaf_value_mode=GuidedLeafValueMode.EXPERIMENTAL_MODEL
    )
    result = GuidedInformationSetSearch(evaluator, config).search(
        information, b"v" * 32
    )
    values = [value for value in result.mean_action_values if value is not None]
    assert values == pytest.approx([-0.4] * len(values))
    assert result.cutoff_evaluation_count == 16
    assert result.terminal_evaluation_count == 0


# The active terminal profile must return the engine's exact completed-round value.
def test_exact_terminal_profile_matches_search_only_engine_results() -> None:
    state = _advance(create_game("guided-exact-terminal"), 6)
    information = information_state_from_engine(state)
    request_seed = b"e" * 32
    control = InformationSetSearch(SearchConfig(2)).search(information, request_seed)
    result = GuidedInformationSetSearch(
        UniformPolicyValueEvaluator(), GuidedSearchConfig(2)
    ).search(information, request_seed)
    assert result.action_visits == control.action_visits
    assert result.mean_action_values == control.mean_action_values
    assert result.terminal_evaluation_count == 2
    assert result.cutoff_evaluation_count == 0
    assert result.principal_continuation is not None
    assert result.principal_continuation.value_source == "exact-terminal"


# Forced placements are engine work, not neural decisions or search simulations.
def test_forced_root_bypasses_model_inference_and_decision_budget() -> None:
    state = _advance(create_game("guided-forced"), 7)
    information = information_state_from_engine(state)
    evaluator = _RecordingEvaluator()
    result = GuidedInformationSetSearch(evaluator, GuidedSearchConfig(100)).search(
        information, b"f" * 32
    )
    assert result.simulation_count == 0
    assert result.model_evaluation_count == 0
    assert evaluator.states == []
    assert sum(result.action_visits) == 0


# A fixed model, configuration, information state, and seed fix all decision evidence.
def test_fixed_seed_and_weights_reproduce_visits_values_and_action() -> None:
    state = _advance(create_game("guided-repeat"), 4)
    information = information_state_from_engine(state)
    model = PolicyValueModel(
        run_root_seed="guided-repeat-root",
        model_id="guided-repeat-model",
        initialization_ordinal=0,
    )
    evaluator = PolicyValueModelEvaluator(model)
    config = GuidedSearchConfig(24)
    planner = GuidedInformationSetSearch(evaluator, config)
    request_seed = derive_search_request_seed("guided-repeat", information, planner.digest)
    first = planner.search(information, request_seed)
    repeated = planner.search(information, request_seed)
    assert _deterministic_result(first) == _deterministic_result(repeated)
    assert first.elapsed_seconds >= 0
    assert repeated.elapsed_seconds >= 0


# Batched model evaluation is the same pure mapping as individual evaluation.
def test_batched_leaf_evaluation_matches_single_evaluation() -> None:
    state = create_game("guided-batch")
    states = []
    for _ in range(4):
        states.append(information_state_from_engine(state))
        state = apply_move(state, legal_moves(state, state.active_player)[0]).state
    evaluator = PolicyValueModelEvaluator(
        PolicyValueModel(
            run_root_seed="guided-batch-root",
            model_id="guided-batch-model",
            initialization_ordinal=0,
        )
    )
    batched = evaluator.evaluate_batch(states)
    individual = tuple(evaluator.evaluate_batch((state,))[0] for state in states)
    for left, right in zip(batched, individual, strict=True):
        assert left.information_state_fingerprint == right.information_state_fingerprint
        assert left.legal_priors == pytest.approx(right.legal_priors, abs=1e-6, rel=1e-5)
        assert left.player_relative_value == pytest.approx(
            right.player_relative_value, abs=1e-6, rel=1e-5
        )


# Equivalent private worlds cannot affect the evaluator boundary or guided result.
def test_authoritative_hidden_data_cannot_reach_the_evaluator() -> None:
    root = information_state_from_engine(create_game("guided-hidden-boundary"))
    evaluator = _RecordingEvaluator()
    planner = GuidedInformationSetSearch(evaluator, GuidedSearchConfig(16))
    before = information_state_fingerprint(root)
    result = planner.search(root, b"h" * 32)
    assert result.information_state_fingerprint == before
    assert all(isinstance(item, SearchInformationState) for item in evaluator.states)
    assert all(not hasattr(item, "stock_order") for item in evaluator.states)
    assert all(not hasattr(item, "opponent_hand") for item in evaluator.states)


# Cancellation discards local tree diagnostics and cannot mutate the authoritative game.
def test_interruption_cannot_mutate_state_or_return_partial_diagnostics() -> None:
    state = create_game("guided-interruption")
    before = state_fingerprint(state)
    information = information_state_from_engine(state)
    checks = 0

    def stop() -> bool:
        nonlocal checks
        checks += 1
        return checks == 8

    with pytest.raises(SearchInterrupted):
        GuidedInformationSetSearch(
            UniformPolicyValueEvaluator(), GuidedSearchConfig(32)
        ).search(information, b"i" * 32, stop)
    assert state_fingerprint(state) == before
    assert state.status is EngineStatus.PLAYING

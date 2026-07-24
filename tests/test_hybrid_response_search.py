"""Experimental response-ranker hybrid information and search invariants."""

from __future__ import annotations

import inspect

import pytest
import torch

import dracula.search.strategic as strategic_module
from dracula.api.app import create_app
from dracula.api.repository import InMemoryGameRepository
from dracula.engine import (
    EngineStatus,
    apply_move,
    apply_simulation_move,
    create_game,
    legal_moves,
    other_player,
    state_fingerprint,
)
from dracula.policy_value import PolicyValueModel
from dracula.search import (
    ResponseRankerGroupEvaluator,
    SearchInformationState,
    StrategicInformationSetSearch,
    StrategicResponseMode,
    StrategicSearchConfig,
    derive_strategic_determinization_seed,
    derive_strategic_search_request_seed,
    information_state_from_engine,
    normalized_round_return,
    sample_determinization,
    strategic_action_groups,
)
from dracula.search_policy import InlineStrategicSearchExecutor


def _advance(state, count: int):
    for _ in range(count):
        state = apply_move(state, legal_moves(state, state.active_player)[0]).state
    return state


class _FixedRanker:
    def __init__(self, digest: str = "a" * 64, *, reverse: bool = False) -> None:
        self.artifact_digest = digest
        self.reverse = reverse
        self.calls: list[SearchInformationState] = []

    def rank(self, information, groups):
        self.calls.append(information)
        representatives = tuple(
            group.representative_action_index for group in groups
        )
        return tuple(reversed(representatives)) if self.reverse else representatives


def _config(
    mode: StrategicResponseMode,
    budget: int,
    ranker: _FixedRanker,
) -> StrategicSearchConfig:
    return StrategicSearchConfig(
        outer_simulation_budget=budget,
        response_completions_per_action=4,
        response_mode=mode,
        response_ranker_artifact_digest=ranker.artifact_digest,
    )


def _request(information, config, label: str):
    return derive_strategic_search_request_seed(
        label, information, config.digest
    )


# Adding hybrid configuration must not alter the default Teacher v2 digest,
# root visits, values, concrete action, or shallow-response accounting.
def test_pure_teacher_v2_remains_behaviorally_frozen() -> None:
    information = information_state_from_engine(
        _advance(create_game("hybrid-pure-control"), 5)
    )
    original = StrategicSearchConfig(6, 4)
    explicit = StrategicSearchConfig(
        6,
        4,
        response_mode=StrategicResponseMode.PURE,
    )
    assert original.digest == explicit.digest
    request = _request(information, original, "hybrid-pure-control")
    first = StrategicInformationSetSearch(original).search(information, request)
    second = StrategicInformationSetSearch(explicit).search(information, request)
    assert first == second
    assert first.response_model_call_count == 0


# The concrete student implementation can receive only the typed actor view and
# must return an exact permutation of the legal strategic representatives.
def test_student_inference_is_information_only_and_groups_legality_exactly() -> None:
    information = information_state_from_engine(
        _advance(create_game("hybrid-student-information"), 4)
    )
    groups = strategic_action_groups(information)
    model = PolicyValueModel(
        run_root_seed="hybrid-student",
        model_id="hybrid-student",
        initialization_ordinal=0,
    )
    evaluator = ResponseRankerGroupEvaluator(model, "b" * 64)
    ranking = evaluator.rank(information, groups)

    assert set(ranking) == {
        group.representative_action_index for group in groups
    }
    assert not hasattr(information, "opponent_hand")
    assert not hasattr(information, "stock")
    parameters = tuple(
        inspect.signature(ResponseRankerGroupEvaluator.rank).parameters
    )
    assert parameters == ("self", "information", "groups")


# Each completion samples one common hidden world, then evaluates exactly the
# two neural-shortlisted candidates from immutable copies of that world.
def test_student_top_2_uses_shared_determinizations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    information = information_state_from_engine(
        _advance(create_game("hybrid-top-two-worlds"), 5)
    )
    ranker = _FixedRanker(reverse=True)
    config = _config(StrategicResponseMode.STUDENT_TOP_2, 6, ranker)
    planner = StrategicInformationSetSearch(config, response_ranker=ranker)
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

    monkeypatch.setattr(strategic_module, "sample_determinization", capture_sample)
    monkeypatch.setattr(strategic_module, "apply_simulation_move", capture_apply)
    response = planner._actor_response(information, None)

    assert len(sampled_states) == 4
    assert candidate_starts == {index: 2 for index in range(4)}
    assert response.candidate_action_count == 2
    assert response.terminal_evaluation_count == 8
    assert response.model_call_count == 1
    assert response.selected_representative_action_index in (
        response.shortlisted_representative_action_indices
    )


# Top-3 is the only additional experimental shortlist. It shares each sampled
# world across exactly three candidates and retains the terminal evaluator.
def test_student_top_3_uses_shared_determinizations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    information = information_state_from_engine(
        _advance(create_game("hybrid-top-three-worlds"), 5)
    )
    ranker = _FixedRanker(reverse=True)
    config = _config(StrategicResponseMode.STUDENT_TOP_3, 6, ranker)
    planner = StrategicInformationSetSearch(config, response_ranker=ranker)
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

    monkeypatch.setattr(strategic_module, "sample_determinization", capture_sample)
    monkeypatch.setattr(strategic_module, "apply_simulation_move", capture_apply)
    response = planner._actor_response(information, None)

    assert len(sampled_states) == 4
    assert candidate_starts == {index: 3 for index in range(4)}
    assert response.candidate_action_count == 3
    assert response.terminal_evaluation_count == 12
    assert response.model_call_count == 1
    assert response.selected_representative_action_index in (
        response.shortlisted_representative_action_indices
    )


# Direct inference selects one legal group without terminal response work; its
# paired concrete destination is still resolved by the existing derived coin.
def test_student_direct_is_legal_and_runs_no_inner_terminal_evaluation() -> None:
    information = information_state_from_engine(
        _advance(create_game("hybrid-direct"), 5)
    )
    ranker = _FixedRanker(reverse=True)
    config = _config(StrategicResponseMode.STUDENT_DIRECT, 6, ranker)
    response = StrategicInformationSetSearch(
        config, response_ranker=ranker
    )._actor_response(information, None)
    selected_group = next(
        group
        for group in strategic_action_groups(information)
        if group.representative_action_index
        == response.selected_representative_action_index
    )

    assert response.selected_action_index in selected_group.member_action_indices
    assert response.candidate_action_count == 0
    assert response.terminal_evaluation_count == 0
    assert response.model_call_count == 1


# A forced placement remains an engine operation. It cannot call the student or
# spend either the outer or shallow-response decision budget.
@pytest.mark.parametrize(
    "mode",
    (
        StrategicResponseMode.STUDENT_DIRECT,
        StrategicResponseMode.STUDENT_TOP_2,
        StrategicResponseMode.STUDENT_TOP_3,
    ),
)
def test_forced_placement_bypasses_student_and_response_evaluation(mode) -> None:
    state = _advance(create_game("hybrid-forced"), 7)
    information = information_state_from_engine(state)
    ranker = _FixedRanker()
    config = _config(mode, 32, ranker)
    result = StrategicInformationSetSearch(
        config, response_ranker=ranker
    ).search(information, _request(information, config, "hybrid-forced"))

    assert result.simulation_count == 0
    assert result.response_request_count == 0
    assert result.response_terminal_evaluation_count == 0
    assert result.response_model_call_count == 0
    assert ranker.calls == []


# Retry identity includes the artifact and shortlist configuration, while a
# repeated request reproduces the complete outer visit and action result.
@pytest.mark.parametrize(
    "mode",
    (
        StrategicResponseMode.STUDENT_DIRECT,
        StrategicResponseMode.STUDENT_TOP_2,
        StrategicResponseMode.STUDENT_TOP_3,
    ),
)
def test_hybrid_retries_are_exact_and_artifact_bound(mode) -> None:
    state = _advance(create_game("hybrid-retry"), 5)
    before = state_fingerprint(state)
    information = information_state_from_engine(state)
    group_count = len(strategic_action_groups(information))
    ranker = _FixedRanker()
    config = _config(mode, group_count, ranker)
    request = _request(information, config, f"hybrid-retry-{mode.value}")
    first = StrategicInformationSetSearch(
        config, response_ranker=ranker
    ).search(information, request)
    second = StrategicInformationSetSearch(
        config, response_ranker=ranker
    ).search(information, request)
    other_ranker = _FixedRanker("c" * 64)
    other_config = _config(mode, group_count, other_ranker)

    assert first == second
    assert first.selected_action_index in {
        index
        for index, allowed in enumerate(
            value for row in information.legal_mask for value in row
        )
        if allowed
    }
    assert first.response_model_call_count == first.unique_response_evaluation_count
    assert config.digest != other_config.digest
    assert state_fingerprint(state) == before


# The outer loop still terminates in the engine and backs up its exact
# player-relative round score; the response model is not a value cutoff.
def test_hybrid_outer_terminal_values_are_exact_engine_results() -> None:
    information = information_state_from_engine(
        _advance(create_game("engine-contract-fixture-1"), 6)
    )
    ranker = _FixedRanker()
    config = _config(StrategicResponseMode.STUDENT_TOP_2, 2, ranker)
    request = _request(information, config, "hybrid-terminal")
    result = StrategicInformationSetSearch(
        config, response_ranker=ranker
    ).search(information, request)
    legal = tuple(
        index
        for index, allowed in enumerate(
            value for row in information.legal_mask for value in row
        )
        if allowed
    )
    for simulation_index, action_index in enumerate(legal):
        sampled = sample_determinization(
            information,
            derive_strategic_determinization_seed(request, simulation_index),
        )
        move = strategic_module._move_for_action(sampled.state, action_index)
        state = apply_simulation_move(sampled.state, move)
        forced = legal_moves(state, state.active_player)[0]
        terminal = apply_simulation_move(state, forced)
        assert terminal.status is EngineStatus.ROUND_COMPLETE
        value = normalized_round_return(
            terminal.pending_round_result, information.player
        )
        assert result.mean_action_values[action_index] == value
        assert value == -normalized_round_return(
            terminal.pending_round_result,
            other_player(information.player),
        )


# Browser selection is explicit; neither experimental mode can silently reuse
# pure Teacher v2 or a missing ranker artifact.
def test_app_selects_explicit_hybrid_modes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FixedRanker()
    monkeypatch.setattr(
        ResponseRankerGroupEvaluator,
        "from_artifact",
        classmethod(lambda cls, _path: fake),
    )
    monkeypatch.setenv("DRACULA_SEARCH_SIMULATIONS", "32")
    monkeypatch.setenv("DRACULA_SEARCH_RESPONSE_COMPLETIONS", "4")
    monkeypatch.setenv("DRACULA_RESPONSE_RANKER_ARTIFACT", "ranker.pt")

    for mode in ("student-direct", "student-top-2", "student-top-3"):
        monkeypatch.setenv("DRACULA_OPPONENT_MODE", f"search-v2-{mode}")
        app = create_app(
            repository=InMemoryGameRepository(), narration_enabled=False
        )
        executor = app.state.gameplay_service.policy_executor
        assert isinstance(executor, InlineStrategicSearchExecutor)
        assert executor.planner.config.response_mode.value == mode
        assert mode in executor.descriptor.policy_id

    monkeypatch.delenv("DRACULA_RESPONSE_RANKER_ARTIFACT")
    with pytest.raises(ValueError, match="must be a nonempty path"):
        create_app(
            repository=InMemoryGameRepository(), narration_enabled=False
        )

"""Exercise the retained BGC search and both continuation implementations.

The suite protects deterministic UCT statistics, sampled-world privacy, exact
terminal scoring, belief-greedy responses, and policy-based responses.
"""

from __future__ import annotations

import math
from pathlib import Path

from dracula.bgc_policy import save_bgc_policy_artifact
from dracula.bgc_policy_model import BGCPolicyModel, OBSERVATION_SIZE
from dracula.bridge import build_policy_turn_context, move_for_action_index
from dracula.engine import (
    EngineStatus,
    apply_move,
    apply_simulation_move,
    create_game,
    legal_moves,
    other_player,
)
from dracula.policy_observation import encode_policy_observation
from dracula.randomness import derive_seed, seed_hex
from dracula.search.belief_greedy import (
    BeliefGreedyContinuation,
    BeliefGreedyContinuationConfig,
    BeliefGreedyInformationSetSearch,
)
from dracula.search.bgc import (
    BGC_DETERMINIZATION_NAMESPACE,
    BGCInformationSetSearch,
    BGCSearchConfig,
    derive_bgc_request_seed,
)
from dracula.search.contracts import SearchInterrupted, normalized_round_return
from dracula.search.determinization import sample_determinization
from dracula.search.information import (
    SearchInformationState,
    canonical_information_data,
    information_state_fingerprint,
    information_state_from_engine,
    information_state_from_simulation,
)
from dracula.search.policy_continuation import (
    PolicyContinuation,
    PolicyContinuationInformationSetSearch,
)
from dracula.strategic_actions import strategic_action_groups


def _advance(state, count: int):
    for _ in range(count):
        assert state.active_player is not None
        state = apply_move(
            state,
            legal_moves(state, state.active_player)[0],
        ).state
    return state


def _policy_artifact(tmp_path: Path) -> Path:
    model = BGCPolicyModel(
        run_root_seed="bgc-continuation-test",
        model_id="policy-continuation",
        initialization_ordinal=0,
    )
    path = tmp_path / "policy.pt"
    save_bgc_policy_artifact(
        path,
        model,
        source_revision="a" * 40,
        source_tree_digest="b" * 64,
        training_configuration={"run": {"id": "bgc-continuation-test"}},
        corpus_snapshot_digest="c" * 64,
        dataset_digest="d" * 64,
    )
    return path


def test_retained_bgc_configuration_is_128_outer_by_8_belief_completions() -> None:
    search = BGCSearchConfig()
    continuation = BeliefGreedyContinuationConfig()

    assert search.outer_simulation_budget == 128
    assert search.outer_exploration_constant == math.sqrt(2.0)
    assert continuation.belief_completion_count == 8


def test_determinizations_change_hidden_locations_but_not_actor_information() -> None:
    information = information_state_from_engine(create_game("bgc-hidden-boundary"))
    first = sample_determinization(
        information,
        derive_seed("bgc-hidden-test-v1", "first"),
    )
    second = sample_determinization(
        information,
        derive_seed("bgc-hidden-test-v1", "second"),
    )

    opponent = other_player(information.player)
    assert first.state.hands[opponent] != second.state.hands[opponent]
    assert information_state_from_simulation(first.state) == information
    assert information_state_from_simulation(second.state) == information


def test_belief_greedy_continuation_scores_every_group_on_shared_samples() -> None:
    information = information_state_from_engine(
        _advance(create_game("bgc-greedy-response"), 4)
    )
    continuation = BeliefGreedyContinuation(
        BeliefGreedyContinuationConfig(belief_completion_count=3)
    )
    decision = continuation.select(information)
    groups = strategic_action_groups(information)

    assert tuple(item.group for item in decision.group_statistics) == groups
    assert all(item.visits == 3 for item in decision.group_statistics)
    assert decision.terminal_evaluation_count == len(groups) * 3
    assert decision.model_inference_count == 0
    best = min(
        decision.group_statistics,
        key=lambda item: (-float(item.mean_value), item.group.representative_action_index),
    )
    assert decision.selected_group == best.group
    assert decision.selected_action_index in best.group.member_action_indices


def test_outer_uct_visits_only_strategic_representatives() -> None:
    state = create_game("bgc-root-groups")
    information = information_state_from_engine(state)
    search = BeliefGreedyInformationSetSearch(
        BGCSearchConfig(outer_simulation_budget=16),
        BeliefGreedyContinuationConfig(belief_completion_count=2),
    )
    request_seed = derive_bgc_request_seed(
        "bgc-root-groups",
        information,
        search.digest,
    )
    result = search.search(information, request_seed)
    representatives = {
        group.representative_action_index
        for group in strategic_action_groups(information)
    }

    assert result.simulation_count == 16
    assert sum(result.representative_action_visits) == 16
    assert all(
        visits == 0
        for index, visits in enumerate(result.representative_action_visits)
        if index not in representatives
    )
    assert all(item.visits > 0 for item in result.group_statistics)
    assert result.selected_group in strategic_action_groups(information)
    context = build_policy_turn_context(state, information.player)
    assert context.action_table[result.selected_action_index] is not None


def test_bgc_replay_is_exact_and_interruption_returns_no_result() -> None:
    information = information_state_from_engine(
        _advance(create_game("bgc-replay"), 3)
    )
    search = BeliefGreedyInformationSetSearch(
        BGCSearchConfig(outer_simulation_budget=16),
        BeliefGreedyContinuationConfig(belief_completion_count=2),
    )
    request_seed = derive_bgc_request_seed("bgc-replay", information, search.digest)
    first = search.search(information, request_seed)
    second = search.search(information, request_seed)

    assert first.selected_group == second.selected_group
    assert first.selected_action_index == second.selected_action_index
    assert first.representative_action_visits == second.representative_action_visits
    assert first.representative_mean_values == second.representative_mean_values
    assert first.principal_continuation == second.principal_continuation

    calls = 0

    def stop() -> bool:
        nonlocal calls
        calls += 1
        return calls > 5

    try:
        search.search(information, request_seed, stop)
    except SearchInterrupted:
        pass
    else:
        raise AssertionError("search ignored its interruption callback")
    assert information_state_fingerprint(information) == (
        first.information_state_fingerprint
    )


def test_principal_continuation_uses_exact_engine_terminal_scoring() -> None:
    information = information_state_from_engine(
        _advance(create_game("bgc-terminal-score"), 3)
    )
    search = BeliefGreedyInformationSetSearch(
        BGCSearchConfig(outer_simulation_budget=16),
        BeliefGreedyContinuationConfig(belief_completion_count=2),
    )
    request_seed = derive_bgc_request_seed(
        "bgc-terminal-score",
        information,
        search.digest,
    )
    result = search.search(information, request_seed)
    principal = result.principal_continuation
    assert principal is not None
    state = sample_determinization(
        information,
        derive_seed(
            BGC_DETERMINIZATION_NAMESPACE,
            seed_hex(request_seed),
            str(principal.simulation_index),
        ),
    ).state
    for step in principal.steps:
        assert state.active_player is not None
        move = move_for_action_index(state.active_player, step.action_index)
        assert state.hands[move.player][move.hand_slot] == step.card_id
        state = apply_simulation_move(state, move)

    assert state.status is EngineStatus.ROUND_COMPLETE
    assert state.pending_round_result is not None
    assert principal.terminal_value == normalized_round_return(
        state.pending_round_result,
        information.player,
    )


def test_forced_placement_bypasses_uct_and_continuation() -> None:
    state = _advance(create_game("bgc-forced"), 7)
    information = information_state_from_engine(state)
    search = BeliefGreedyInformationSetSearch()
    result = search.search(
        information,
        derive_bgc_request_seed("bgc-forced", information, search.digest),
    )

    assert result.simulation_count == 0
    assert result.continuation_request_count == 0
    assert result.continuation_terminal_evaluation_count == 0
    context = build_policy_turn_context(state, information.player)
    assert context.action_table[result.selected_action_index] is not None


def test_policy_continuation_uses_final_model_shape_and_actor_view(
    tmp_path: Path,
) -> None:
    information = information_state_from_engine(
        _advance(create_game("bgc-policy-response"), 2)
    )
    continuation = PolicyContinuation.from_artifact(_policy_artifact(tmp_path))
    decision = continuation.select(information)

    assert encode_policy_observation(information).shape == (OBSERVATION_SIZE,)
    assert OBSERVATION_SIZE == 659
    assert decision.information_state_fingerprint == information_state_fingerprint(
        information
    )
    assert decision.selected_group in strategic_action_groups(information)
    assert decision.selected_action_index in decision.selected_group.member_action_indices
    assert decision.model_inference_count == 1
    assert decision.terminal_evaluation_count == 0


def test_phase_two_policy_continuation_changes_only_the_response_policy(
    tmp_path: Path,
) -> None:
    state = create_game("bgc-policy-search")
    information = information_state_from_engine(state)
    config = BGCSearchConfig(outer_simulation_budget=16)
    continuation = PolicyContinuation.from_artifact(_policy_artifact(tmp_path))
    search = PolicyContinuationInformationSetSearch(continuation, config)
    result = search.search(
        information,
        derive_bgc_request_seed("bgc-policy-search", information, search.digest),
    )
    replay = search.search(
        information,
        derive_bgc_request_seed("bgc-policy-search", information, search.digest),
    )

    assert search.config == config
    assert result.simulation_count == config.outer_simulation_budget
    assert sum(result.representative_action_visits) == config.outer_simulation_budget
    assert result.continuation_model_inference_count > 0
    assert result.continuation_terminal_evaluation_count == 0
    assert replay.selected_group == result.selected_group
    assert replay.selected_action_index == result.selected_action_index
    assert replay.representative_action_visits == result.representative_action_visits
    assert replay.representative_mean_values == result.representative_mean_values
    context = build_policy_turn_context(state, information.player)
    assert context.action_table[result.selected_action_index] is not None


def test_every_simulated_continuation_receives_only_actor_information() -> None:
    class RecordingContinuation:
        def __init__(self) -> None:
            self.delegate = BeliefGreedyContinuation(
                BeliefGreedyContinuationConfig(belief_completion_count=1)
            )
            self.calls = 0

        @property
        def digest(self) -> str:
            return self.delegate.digest

        def select(self, information, should_stop=None):
            assert isinstance(information, SearchInformationState)
            assert set(canonical_information_data(information)).isdisjoint(
                {"seed", "stock", "opponent_hand", "simulation_deck", "search_tree"}
            )
            self.calls += 1
            return self.delegate.select(information, should_stop)

    information = information_state_from_engine(create_game("bgc-actor-privacy"))
    continuation = RecordingContinuation()
    search = BGCInformationSetSearch(
        continuation,
        BGCSearchConfig(outer_simulation_budget=8),
    )
    search.search(
        information,
        derive_bgc_request_seed("bgc-actor-privacy", information, search.digest),
    )
    assert continuation.calls > 0

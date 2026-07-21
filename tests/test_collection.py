"""Deterministic schedule, collection, replay, and artifact contract tests."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from dracula.bridge import (
    HIDDEN_STATE_BYTES,
    PolicyTurnKind,
    apply_policy_action,
    build_policy_turn_context,
    hidden_state_from_bytes,
)
from dracula.collection import (
    ActorBucket,
    CollectionContractViolation,
    PolicyVersion,
    action_sampling_seed,
    artifact_bytes,
    build_collection_schedule,
    build_smoke_schedule,
    collect_schedule,
    load_collection_artifact,
    replay_behavior_hidden_states,
    sample_masked_action,
    save_collection_artifact,
    seal_collection,
)
from dracula.engine import (
    EnginePlayer,
    EngineStatus,
    advance_after_round,
    create_game,
    other_player,
)
from dracula.models import Critic, Policy
from dracula.randomness import derive_pytorch_seed

MODEL_INITIALIZATION_NAMESPACE = "dracula-model-initialization-v1"
RUN_ROOT_SEED = "collection-test-run-root"
CRITIC_VERSION = "critic-v0"
POLICY_A = PolicyVersion("policy-0", "policy-0-v0")
POLICY_B = PolicyVersion("policy-1", "policy-1-v0")


def _model_seed(model_kind: str, stable_id: str) -> int:
    return derive_pytorch_seed(
        MODEL_INITIALIZATION_NAMESPACE,
        "collection-model-root",
        model_kind,
        stable_id,
        "0",
    )


@pytest.fixture(scope="module")
def smoke_models() -> tuple[dict[PolicyVersion, Policy], Critic]:
    return (
        {
            POLICY_A: Policy(seed=_model_seed("policy", POLICY_A.policy_id)),
            POLICY_B: Policy(seed=_model_seed("policy", POLICY_B.policy_id)),
        },
        Critic(seed=_model_seed("critic", "critic-0")),
    )


@pytest.fixture(scope="module")
def smoke_collection(smoke_models):
    policies, critic = smoke_models
    schedule = build_smoke_schedule(POLICY_A, POLICY_B, "collection-lane-0")
    return collect_schedule(
        run_root_seed=RUN_ROOT_SEED,
        schedule=schedule,
        policies=policies,
        critic=critic,
        critic_version=CRITIC_VERSION,
    )


# Schedule cardinalities define the exact amount of evidence in one frozen update window.
def test_full_five_policy_twelve_lane_schedule_has_exact_counts() -> None:
    policies = tuple(
        PolicyVersion(f"policy-{index}", f"policy-{index}-v0") for index in range(5)
    )
    lanes = tuple(f"collection-lane-{index:02d}" for index in range(12))
    schedule = build_collection_schedule(policies, lanes)
    assert schedule.generation_count == 4
    assert schedule.counts.engine_games == 480
    assert schedule.counts.trajectories == 960
    assert schedule.counts.recurrent_steps == 23_040
    assert schedule.counts.learned_rows == 20_160
    for policy in policies:
        policy_games = sum(
            policy in (fixture.policy_a, fixture.policy_b)
            for fixture in schedule.fixtures
        )
        assert policy_games == 192
        assert policy_games * 24 == 4_608
        assert policy_games * 21 == 4_032


# Pair generation must never spend compute teaching a policy against itself.
def test_full_schedule_contains_no_self_matches() -> None:
    policies = tuple(
        PolicyVersion(f"policy-{index}", f"version-{index}") for index in range(5)
    )
    schedule = build_collection_schedule(
        policies,
        tuple(f"lane-{index}" for index in range(12)),
        generation_count=1,
    )
    assert len(schedule.fixtures) == 120
    assert all(fixture.policy_a != fixture.policy_b for fixture in schedule.fixtures)
    assert all(
        fixture.policy_a.policy_id != fixture.policy_b.policy_id
        for fixture in schedule.fixtures
    )


# Twelve consecutive lane indexes balance each ordered pair across both rules roles.
def test_full_generation_balances_queen_and_king_for_every_pair() -> None:
    policies = tuple(
        PolicyVersion(f"policy-{index}", f"version-{index}") for index in range(5)
    )
    schedule = build_collection_schedule(
        policies,
        tuple(f"lane-{index}" for index in range(12)),
        generation_count=1,
        start_game_counter=7,
    )
    for policy_a, policy_b in (
        (policies[first], policies[second])
        for first in range(5)
        for second in range(first + 1, 5)
    ):
        pair_fixtures = tuple(
            fixture
            for fixture in schedule.fixtures
            if (fixture.policy_a, fixture.policy_b) == (policy_a, policy_b)
        )
        assert len(pair_fixtures) == 12
        assert sum(fixture.queen == policy_a for fixture in pair_fixtures) == 6
        assert sum(fixture.queen == policy_b for fixture in pair_fixtures) == 6
        for fixture in pair_fixtures:
            expected_queen = (
                policy_a
                if (fixture.lane_index + fixture.game_counter) % 2 == 0
                else policy_b
            )
            assert fixture.queen == expected_queen


# The smoke profile is the smallest real collection and retains full-game trajectories.
def test_smoke_collector_has_exact_game_trajectory_and_row_counts(
    smoke_collection,
) -> None:
    assert smoke_collection.schedule.lane_count == 1
    assert smoke_collection.schedule.generation_count == 1
    assert smoke_collection.schedule.counts.engine_games == 1
    assert len(smoke_collection.fixture_results) == 1
    assert len(smoke_collection.actor_buckets) == 2
    assert len(smoke_collection.critic_rows) == 42
    for bucket in smoke_collection.actor_buckets:
        assert len(bucket.trajectories) == 1
        assert len(bucket.trajectories[0].transitions) == 24
        assert bucket.learned_row_count == 21


# Named fixture and sampling seeds make actions and hidden states repeat exactly.
def test_identical_fixture_collection_reproduces_actions_hidden_states_and_hash(
    smoke_models, smoke_collection
) -> None:
    policies, critic = smoke_models
    repeated = collect_schedule(
        run_root_seed=RUN_ROOT_SEED,
        schedule=smoke_collection.schedule,
        policies=policies,
        critic=critic,
        critic_version=CRITIC_VERSION,
    )
    first_artifact = seal_collection(smoke_collection)
    repeated_artifact = seal_collection(repeated)
    assert repeated_artifact.content_hash == first_artifact.content_hash
    assert artifact_bytes(repeated_artifact) == artifact_bytes(first_artifact)

    for first_bucket, repeated_bucket in zip(
        smoke_collection.actor_buckets, repeated.actor_buckets, strict=True
    ):
        first = first_bucket.trajectories[0].transitions
        second = repeated_bucket.trajectories[0].transitions
        assert [transition.action_index for transition in first] == [
            transition.action_index for transition in second
        ]
        assert [transition.policy_hidden_in for transition in first] == [
            transition.policy_hidden_in for transition in second
        ]
        assert [transition.policy_hidden_out for transition in first] == [
            transition.policy_hidden_out for transition in second
        ]


# Every actor bucket is owned by one exact stable ID and behavior version.
def test_actor_buckets_cannot_mix_policy_versions(smoke_collection) -> None:
    assert tuple(bucket.learner for bucket in smoke_collection.actor_buckets) == (
        POLICY_A,
        POLICY_B,
    )
    first_trajectory = smoke_collection.actor_buckets[0].trajectories[0]
    second_trajectory = smoke_collection.actor_buckets[1].trajectories[0]
    with pytest.raises(CollectionContractViolation):
        ActorBucket(POLICY_A, (first_trajectory, second_trajectory))
    with pytest.raises(CollectionContractViolation):
        replace(smoke_collection.actor_buckets[0], learner=POLICY_B)


# Stored behavior hidden evidence must replay bit-exactly under unchanged CPU weights.
def test_stored_hidden_states_match_behavior_policy_replay(
    smoke_models, smoke_collection
) -> None:
    policies, _ = smoke_models
    for bucket in smoke_collection.actor_buckets:
        replay_behavior_hidden_states(bucket.trajectories[0], policies[bucket.learner])


# A forced update keeps action and recurrent evidence while remaining absent from losses.
def test_forced_transitions_retain_evidence_without_actor_or_critic_targets(
    smoke_collection,
) -> None:
    for bucket in smoke_collection.actor_buckets:
        forced = tuple(
            transition
            for transition in bucket.trajectories[0].transitions
            if transition.kind is PolicyTurnKind.FORCED_RECURRENT_TRANSITION
        )
        assert len(forced) == 3
        for transition in forced:
            assert 0 <= transition.action_index < 32
            assert len(transition.policy_hidden_in) == HIDDEN_STATE_BYTES
            assert len(transition.policy_hidden_out) == HIDDEN_STATE_BYTES
            assert transition.action_log_probability is None
            assert transition.critic_value is None
            assert transition.round_return is None
            assert not transition.actor_loss_mask


# Round targets attach only to learned rows owned by that player in that round.
def test_round_returns_attach_to_the_correct_player_and_round(smoke_collection) -> None:
    fixture_result = smoke_collection.fixture_results[0]
    trajectories = {
        bucket.trajectories[0].player: bucket.trajectories[0]
        for bucket in smoke_collection.actor_buckets
    }
    state = create_game(fixture_result.fixture.game_seed)
    next_step = {EnginePlayer.QUEEN: 0, EnginePlayer.KING: 0}
    while state.status is not EngineStatus.GAME_COMPLETE:
        while state.status is EngineStatus.PLAYING:
            player = state.active_player
            assert player is not None
            transition = trajectories[player].transitions[next_step[player]]
            context = build_policy_turn_context(state, player)
            assert transition.state_fingerprint == context.state_fingerprint
            state = apply_policy_action(state, context, transition.action_index).state
            next_step[player] += 1
        result = state.pending_round_result
        assert result is not None
        expected_queen = (
            result.round_scores.queen - result.round_scores.king
        ) / 150.0
        assert fixture_result.round_returns.queen[state.round_number - 1] == expected_queen
        assert fixture_result.round_returns.king[state.round_number - 1] == -expected_queen
        state = advance_after_round(state)

    for player in EnginePlayer:
        expected_returns = fixture_result.round_returns[player]
        trajectory = trajectories[player]
        for round_number, expected_return in enumerate(expected_returns, start=1):
            round_steps = tuple(
                transition
                for transition in trajectory.transitions
                if transition.round_number == round_number
            )
            assert len(round_steps) == 4
            for transition in round_steps:
                if transition.actor_loss_mask:
                    assert transition.round_return == expected_return
                else:
                    assert transition.round_return is None
        opposite = fixture_result.round_returns[other_player(player)]
        assert all(
            own == -other for own, other in zip(expected_returns, opposite, strict=True)
        )


# Match boundaries reset hidden state and make a fixture independent of its predecessor.
def test_fixture_local_state_does_not_leak_between_matches(smoke_models) -> None:
    policies, critic = smoke_models
    two_game_schedule = build_collection_schedule(
        (POLICY_A, POLICY_B),
        ("collection-lane-0",),
        generation_count=2,
    )
    combined = collect_schedule(
        run_root_seed=RUN_ROOT_SEED,
        schedule=two_game_schedule,
        policies=policies,
        critic=critic,
        critic_version=CRITIC_VERSION,
    )
    standalone_schedule = build_smoke_schedule(
        POLICY_A, POLICY_B, "collection-lane-0", game_counter=1
    )
    standalone = collect_schedule(
        run_root_seed=RUN_ROOT_SEED,
        schedule=standalone_schedule,
        policies=policies,
        critic=critic,
        critic_version=CRITIC_VERSION,
    )

    for combined_bucket, standalone_bucket in zip(
        combined.actor_buckets, standalone.actor_buckets, strict=True
    ):
        first, second = combined_bucket.trajectories
        standalone_trajectory = standalone_bucket.trajectories[0]
        assert first.transitions[0].policy_hidden_in == bytes(HIDDEN_STATE_BYTES)
        assert second.transitions[0].policy_hidden_in == bytes(HIDDEN_STATE_BYTES)
        assert second.fixture_id == standalone_trajectory.fixture_id
        assert [transition.action_index for transition in second.transitions] == [
            transition.action_index for transition in standalone_trajectory.transitions
        ]
        assert [transition.policy_hidden_out for transition in second.transitions] == [
            transition.policy_hidden_out
            for transition in standalone_trajectory.transitions
        ]


# Sealing must preserve tensor bits, dtypes, ordering, versions, and its canonical hash.
def test_sealed_artifact_round_trip_is_lossless_and_deterministic(
    smoke_collection, tmp_path
) -> None:
    artifact = seal_collection(smoke_collection)
    path = tmp_path / "collection.json"
    save_collection_artifact(artifact, path)
    loaded = load_collection_artifact(path)
    assert loaded.content_hash == artifact.content_hash
    assert artifact_bytes(loaded) == artifact_bytes(artifact) == path.read_bytes()
    assert loaded.collection.schedule.policies == smoke_collection.schedule.policies
    assert loaded.collection.critic_version == smoke_collection.critic_version

    for original_bucket, loaded_bucket in zip(
        smoke_collection.actor_buckets, loaded.collection.actor_buckets, strict=True
    ):
        assert loaded_bucket.learner == original_bucket.learner
        for original_trajectory, loaded_trajectory in zip(
            original_bucket.trajectories,
            loaded_bucket.trajectories,
            strict=True,
        ):
            for original, restored in zip(
                original_trajectory.transitions,
                loaded_trajectory.transitions,
                strict=True,
            ):
                assert restored.action_index == original.action_index
                assert restored.policy_hidden_in == original.policy_hidden_in
                assert restored.policy_hidden_out == original.policy_hidden_out
                assert restored.policy_input.observation.device.type == "cpu"
                assert restored.policy_input.observation.dtype is torch.bool
                assert restored.policy_input.legal_mask.device.type == "cpu"
                assert restored.policy_input.legal_mask.dtype is torch.bool
                assert torch.equal(
                    restored.policy_input.observation,
                    original.policy_input.observation,
                )
                assert torch.equal(
                    restored.policy_input.legal_mask,
                    original.policy_input.legal_mask,
                )


# Temperature-one sampling must use only the named local generator and hard legal mask.
def test_recorded_learned_action_matches_derived_temperature_one_sample(
    smoke_models, smoke_collection
) -> None:
    policies, _ = smoke_models
    trajectory = smoke_collection.actor_buckets[0].trajectories[0]
    transition = next(item for item in trajectory.transitions if item.actor_loss_mask)
    policy = policies[trajectory.learner].cpu().eval()
    hidden = hidden_state_from_bytes(transition.policy_hidden_in)
    with torch.inference_mode():
        raw_logits, _ = policy(
            transition.policy_input.observation,
            transition.policy_input.legal_mask,
            hidden,
        )
    seed = action_sampling_seed(
        RUN_ROOT_SEED,
        transition.fixture_id,
        transition.learner_policy_version,
        transition.player,
        transition.round_number,
        transition.recurrent_step_index,
    )
    action_index, log_probability = sample_masked_action(
        raw_logits, transition.policy_input.legal_mask, seed=seed
    )
    assert action_index == transition.action_index
    assert log_probability == transition.action_log_probability


# Collection snapshots models instead of mutating caller weights, gradients, or RNG state.
def test_collection_uses_frozen_cpu_snapshots_without_mutating_callers(smoke_models) -> None:
    policies, critic = smoke_models
    policy_states = {
        identity: {name: tensor.clone() for name, tensor in model.state_dict().items()}
        for identity, model in policies.items()
    }
    critic_state = {name: tensor.clone() for name, tensor in critic.state_dict().items()}
    torch.manual_seed(811)
    expected_next_random = torch.rand(1)
    torch.manual_seed(811)
    collect_schedule(
        run_root_seed="frozen-snapshot-run",
        schedule=build_smoke_schedule(POLICY_A, POLICY_B, "frozen-lane"),
        policies=policies,
        critic=critic,
        critic_version=CRITIC_VERSION,
    )
    assert torch.equal(torch.rand(1), expected_next_random)
    for identity, model in policies.items():
        assert all(parameter.grad is None for parameter in model.parameters())
        for name, tensor in model.state_dict().items():
            assert torch.equal(tensor, policy_states[identity][name])
    assert all(parameter.grad is None for parameter in critic.parameters())
    for name, tensor in critic.state_dict().items():
        assert torch.equal(tensor, critic_state[name])

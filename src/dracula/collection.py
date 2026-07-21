"""Deterministic CPU self-play collection and sealed trajectory artifacts."""

from __future__ import annotations

import base64
import copy
import hashlib
import json
import math
import os
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from itertools import combinations
from pathlib import Path

import torch
from torch import Tensor

from dracula.bridge import (
    HIDDEN_STATE_BYTES,
    PolicyInput,
    PolicyTransition,
    PolicyTurnKind,
    apply_policy_action,
    build_policy_transition,
    build_policy_turn_context,
    hidden_state_from_bytes,
    hidden_state_to_bytes,
)
from dracula.cards import CARD_SCHEMA_VERSION
from dracula.engine import (
    ENGINE_VERSION,
    RULES_VERSION,
    EnginePlayer,
    EngineStatus,
    PlayerValues,
    advance_after_round,
    create_game,
    other_player,
    state_fingerprint,
)
from dracula.models import Critic, Policy
from dracula.randomness import (
    RANDOMNESS_SCHEMA_VERSION,
    derive_pytorch_seed,
    derive_seed,
)

COLLECTION_FORMAT_VERSION = "dracula-collection-v1"
FIXTURE_GAME_NAMESPACE = "dracula-fixture-game-v1"
MATCH_FIXTURE_NAMESPACE = "dracula-match-fixture-v1"
ACTION_SAMPLING_NAMESPACE = "dracula-action-sampling-v1"
COLLECTION_PHASE = "collection"
ROUND_RETURN_SCALE = 150.0


class CollectionContractViolation(ValueError):
    """A schedule, trajectory, or artifact violates the collection contract."""


def _validate_identifier(value: object, label: str) -> None:
    if not isinstance(value, str) or not value or "\0" in value:
        raise CollectionContractViolation(f"{label} must be a nonempty string without NUL")


def _validate_digest(value: object, label: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise CollectionContractViolation(f"{label} must be a lowercase SHA-256 digest")


@dataclass(frozen=True, slots=True, order=True)
class PolicyVersion:
    policy_id: str
    version: str

    def __post_init__(self) -> None:
        _validate_identifier(self.policy_id, "policy ID")
        _validate_identifier(self.version, "policy version")


@dataclass(frozen=True, slots=True)
class BehaviorPolicyRecord:
    identity: PolicyVersion
    parameter_fingerprint: str

    def __post_init__(self) -> None:
        if not isinstance(self.identity, PolicyVersion):
            raise CollectionContractViolation("behavior identity must be a PolicyVersion")
        _validate_digest(self.parameter_fingerprint, "policy parameter fingerprint")


@dataclass(frozen=True, slots=True)
class MatchFixture:
    fixture_id: str
    generation_index: int
    lane_index: int
    game_counter: int
    lane_root_seed_digest: str
    game_seed: str
    policy_a: PolicyVersion
    policy_b: PolicyVersion
    queen: PolicyVersion
    king: PolicyVersion
    randomness_schema_version: str = RANDOMNESS_SCHEMA_VERSION
    phase: str = COLLECTION_PHASE

    def __post_init__(self) -> None:
        _validate_digest(self.fixture_id, "fixture ID")
        _validate_digest(self.lane_root_seed_digest, "lane-root seed digest")
        _validate_digest(self.game_seed, "game seed")
        if type(self.generation_index) is not int or self.generation_index < 0:
            raise CollectionContractViolation("generation index must be non-negative")
        if type(self.lane_index) is not int or self.lane_index < 0:
            raise CollectionContractViolation("lane index must be non-negative")
        if type(self.game_counter) is not int or self.game_counter < 0:
            raise CollectionContractViolation("game counter must be non-negative")
        if self.policy_a >= self.policy_b:
            raise CollectionContractViolation("fixture policy pair must use stable ID order")
        if {self.queen, self.king} != {self.policy_a, self.policy_b}:
            raise CollectionContractViolation("fixture roles must contain the policy pair")
        if self.queen == self.king:
            raise CollectionContractViolation("self matches are not valid fixtures")
        expected_queen = (
            self.policy_a
            if (self.lane_index + self.game_counter) % 2 == 0
            else self.policy_b
        )
        if self.queen != expected_queen:
            raise CollectionContractViolation("fixture role assignment is not deterministic")
        if self.randomness_schema_version != RANDOMNESS_SCHEMA_VERSION:
            raise CollectionContractViolation("fixture randomness version is unsupported")
        if self.phase != COLLECTION_PHASE:
            raise CollectionContractViolation("only collection fixtures are supported")


@dataclass(frozen=True, slots=True)
class CollectionCounts:
    engine_games: int
    trajectories: int
    recurrent_steps: int
    learned_rows: int


@dataclass(frozen=True, slots=True)
class CollectionSchedule:
    policies: tuple[PolicyVersion, ...]
    lane_count: int
    generation_count: int
    start_game_counter: int
    fixtures: tuple[MatchFixture, ...]

    def __post_init__(self) -> None:
        if len(self.policies) < 2 or tuple(sorted(self.policies)) != self.policies:
            raise CollectionContractViolation("schedule policies must be sorted and distinct")
        if len(set(policy.policy_id for policy in self.policies)) != len(self.policies):
            raise CollectionContractViolation("active policy IDs must be unique")
        if type(self.lane_count) is not int or self.lane_count < 1:
            raise CollectionContractViolation("lane count must be positive")
        if type(self.generation_count) is not int or self.generation_count < 1:
            raise CollectionContractViolation("generation count must be positive")
        if type(self.start_game_counter) is not int or self.start_game_counter < 0:
            raise CollectionContractViolation("starting game counter must be non-negative")
        expected_games = (
            len(self.policies)
            * (len(self.policies) - 1)
            // 2
            * self.lane_count
            * self.generation_count
        )
        if len(self.fixtures) != expected_games:
            raise CollectionContractViolation("fixture count does not match schedule dimensions")
        fixture_ids = {fixture.fixture_id for fixture in self.fixtures}
        if len(fixture_ids) != len(self.fixtures):
            raise CollectionContractViolation("fixture IDs must be unique")

    @property
    def counts(self) -> CollectionCounts:
        games = len(self.fixtures)
        return CollectionCounts(
            engine_games=games,
            trajectories=2 * games,
            recurrent_steps=48 * games,
            learned_rows=42 * games,
        )


def build_collection_schedule(
    policies: Sequence[PolicyVersion],
    lane_root_seeds: Sequence[str],
    *,
    generation_count: int = 4,
    start_game_counter: int = 0,
) -> CollectionSchedule:
    ordered_policies = tuple(sorted(policies))
    if len(set(ordered_policies)) != len(ordered_policies):
        raise CollectionContractViolation("policy identities must be unique")
    if len(set(policy.policy_id for policy in ordered_policies)) != len(ordered_policies):
        raise CollectionContractViolation("active policy IDs must be unique")
    lane_roots = tuple(lane_root_seeds)
    if not lane_roots or len(set(lane_roots)) != len(lane_roots):
        raise CollectionContractViolation("lane root seeds must be nonempty and distinct")
    for lane_root in lane_roots:
        _validate_identifier(lane_root, "lane root seed")
    if type(generation_count) is not int or generation_count < 1:
        raise CollectionContractViolation("generation count must be positive")
    if type(start_game_counter) is not int or start_game_counter < 0:
        raise CollectionContractViolation("starting game counter must be non-negative")

    fixtures: list[MatchFixture] = []
    for generation_index in range(generation_count):
        game_counter = start_game_counter + generation_index
        for lane_index, lane_root in enumerate(lane_roots):
            game_seed = derive_seed(
                FIXTURE_GAME_NAMESPACE, lane_root, str(game_counter)
            ).hex()
            lane_digest = hashlib.sha256(lane_root.encode("utf-8")).hexdigest()
            for policy_a, policy_b in combinations(ordered_policies, 2):
                fixture_id = derive_seed(
                    MATCH_FIXTURE_NAMESPACE,
                    COLLECTION_PHASE,
                    game_seed,
                    policy_a.policy_id,
                    policy_a.version,
                    policy_b.policy_id,
                    policy_b.version,
                ).hex()
                queen = (
                    policy_a
                    if (lane_index + game_counter) % 2 == 0
                    else policy_b
                )
                king = policy_b if queen == policy_a else policy_a
                fixtures.append(
                    MatchFixture(
                        fixture_id=fixture_id,
                        generation_index=generation_index,
                        lane_index=lane_index,
                        game_counter=game_counter,
                        lane_root_seed_digest=lane_digest,
                        game_seed=game_seed,
                        policy_a=policy_a,
                        policy_b=policy_b,
                        queen=queen,
                        king=king,
                    )
                )
    return CollectionSchedule(
        policies=ordered_policies,
        lane_count=len(lane_roots),
        generation_count=generation_count,
        start_game_counter=start_game_counter,
        fixtures=tuple(fixtures),
    )


def build_smoke_schedule(
    policy_a: PolicyVersion,
    policy_b: PolicyVersion,
    lane_root_seed: str,
    *,
    game_counter: int = 0,
) -> CollectionSchedule:
    return build_collection_schedule(
        (policy_a, policy_b),
        (lane_root_seed,),
        generation_count=1,
        start_game_counter=game_counter,
    )


@dataclass(frozen=True, slots=True, eq=False)
class PlayerTrajectory:
    fixture_id: str
    learner: PolicyVersion
    opponent: PolicyVersion
    player: EnginePlayer
    transitions: tuple[PolicyTransition, ...]

    def __post_init__(self) -> None:
        _validate_digest(self.fixture_id, "trajectory fixture ID")
        if not isinstance(self.learner, PolicyVersion) or not isinstance(
            self.opponent, PolicyVersion
        ):
            raise CollectionContractViolation("trajectory policies must be versioned")
        if self.learner == self.opponent:
            raise CollectionContractViolation("trajectory cannot be a self match")
        if not isinstance(self.player, EnginePlayer):
            raise CollectionContractViolation("trajectory player must be an EnginePlayer")
        if not isinstance(self.transitions, tuple) or len(self.transitions) != 24:
            raise CollectionContractViolation("a trajectory must contain 24 transitions")
        learned_count = 0
        forced_count = 0
        for step, transition in enumerate(self.transitions):
            if not isinstance(transition, PolicyTransition):
                raise CollectionContractViolation("trajectory contains an invalid transition")
            if (
                transition.fixture_id != self.fixture_id
                or transition.learner_id != self.learner.policy_id
                or transition.learner_policy_version != self.learner.version
                or transition.opponent_id != self.opponent.policy_id
                or transition.opponent_policy_version != self.opponent.version
                or transition.player is not self.player
                or transition.recurrent_step_index != step
            ):
                raise CollectionContractViolation("trajectory transition identity is inconsistent")
            if (
                transition.policy_input.observation.device.type != "cpu"
                or transition.policy_input.legal_mask.device.type != "cpu"
            ):
                raise CollectionContractViolation("trajectory tensors must remain on CPU")
            if step and (
                self.transitions[step - 1].policy_hidden_out
                != transition.policy_hidden_in
            ):
                raise CollectionContractViolation("trajectory hidden-state chain is discontinuous")
            if transition.kind is PolicyTurnKind.LEARNED:
                learned_count += 1
                if transition.round_return is None:
                    raise CollectionContractViolation("learned transition is missing its return")
            else:
                forced_count += 1
                if transition.round_return is not None:
                    raise CollectionContractViolation("forced transition cannot carry a return")
        if learned_count != 21 or forced_count != 3:
            raise CollectionContractViolation(
                "trajectory must contain 21 learned and 3 forced steps"
            )


@dataclass(frozen=True, slots=True)
class FixtureResult:
    fixture: MatchFixture
    terminal_state_fingerprint: str
    round_returns: PlayerValues[tuple[float, float, float, float, float, float]]

    def __post_init__(self) -> None:
        if not isinstance(self.fixture, MatchFixture):
            raise CollectionContractViolation("fixture result requires fixture metadata")
        _validate_digest(self.terminal_state_fingerprint, "terminal state fingerprint")
        if not isinstance(self.round_returns, PlayerValues):
            raise CollectionContractViolation("round returns must be player-keyed")
        if len(self.round_returns.queen) != 6 or len(self.round_returns.king) != 6:
            raise CollectionContractViolation("fixture result must contain six round returns")
        for queen_return, king_return in zip(
            self.round_returns.queen, self.round_returns.king, strict=True
        ):
            if not math.isfinite(queen_return) or not -1.0 <= queen_return <= 1.0:
                raise CollectionContractViolation("round return is invalid")
            if queen_return != -king_return:
                raise CollectionContractViolation("player round returns must be opposites")


@dataclass(frozen=True, slots=True, eq=False)
class CriticRow:
    fixture_id: str
    learner: PolicyVersion
    player: EnginePlayer
    round_number: int
    recurrent_step_index: int
    observation: Tensor
    round_return: float

    def __post_init__(self) -> None:
        _validate_digest(self.fixture_id, "critic-row fixture ID")
        if not isinstance(self.learner, PolicyVersion):
            raise CollectionContractViolation("critic row requires a policy version")
        if not isinstance(self.player, EnginePlayer):
            raise CollectionContractViolation("critic row requires an engine player")
        if type(self.round_number) is not int or not 1 <= self.round_number <= 6:
            raise CollectionContractViolation("critic-row round must be between 1 and 6")
        if type(self.recurrent_step_index) is not int or not 0 <= self.recurrent_step_index < 24:
            raise CollectionContractViolation("critic-row step must be between 0 and 23")
        if (
            not isinstance(self.observation, Tensor)
            or self.observation.device.type != "cpu"
            or self.observation.dtype is not torch.bool
            or self.observation.shape != (875,)
        ):
            raise CollectionContractViolation("critic observation must be CPU bool[875]")
        if not math.isfinite(self.round_return) or not -1.0 <= self.round_return <= 1.0:
            raise CollectionContractViolation("critic-row return must be between -1 and 1")


@dataclass(frozen=True, slots=True, eq=False)
class ActorBucket:
    learner: PolicyVersion
    trajectories: tuple[PlayerTrajectory, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.learner, PolicyVersion):
            raise CollectionContractViolation("actor bucket requires a policy version")
        if any(trajectory.learner != self.learner for trajectory in self.trajectories):
            raise CollectionContractViolation("actor bucket mixes behavior policy versions")

    @property
    def learned_row_count(self) -> int:
        return sum(
            transition.actor_loss_mask
            for trajectory in self.trajectories
            for transition in trajectory.transitions
        )


@dataclass(frozen=True, slots=True, eq=False)
class CollectionData:
    run_root_seed: str
    schedule: CollectionSchedule
    behavior_policies: tuple[BehaviorPolicyRecord, ...]
    critic_version: str
    critic_parameter_fingerprint: str
    fixture_results: tuple[FixtureResult, ...]
    actor_buckets: tuple[ActorBucket, ...]
    critic_rows: tuple[CriticRow, ...]
    format_version: str = COLLECTION_FORMAT_VERSION
    rules_version: str = RULES_VERSION
    card_schema_version: str = CARD_SCHEMA_VERSION
    engine_version: str = ENGINE_VERSION
    randomness_schema_version: str = RANDOMNESS_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_identifier(self.run_root_seed, "run root seed")
        _validate_identifier(self.critic_version, "critic version")
        _validate_digest(self.critic_parameter_fingerprint, "critic parameter fingerprint")
        if self.format_version != COLLECTION_FORMAT_VERSION:
            raise CollectionContractViolation("collection format version is unsupported")
        if (
            self.rules_version != RULES_VERSION
            or self.card_schema_version != CARD_SCHEMA_VERSION
            or self.engine_version != ENGINE_VERSION
            or self.randomness_schema_version != RANDOMNESS_SCHEMA_VERSION
        ):
            raise CollectionContractViolation("collection contract versions do not match runtime")
        identities = tuple(record.identity for record in self.behavior_policies)
        if identities != self.schedule.policies:
            raise CollectionContractViolation("behavior policy records do not match schedule")
        bucket_identities = tuple(bucket.learner for bucket in self.actor_buckets)
        if bucket_identities != self.schedule.policies:
            raise CollectionContractViolation("actor buckets do not preserve policy-version order")
        if len(self.fixture_results) != self.schedule.counts.engine_games:
            raise CollectionContractViolation("fixture result count does not match schedule")
        if tuple(result.fixture for result in self.fixture_results) != self.schedule.fixtures:
            raise CollectionContractViolation("fixture result order does not match schedule")
        if sum(len(bucket.trajectories) for bucket in self.actor_buckets) != (
            self.schedule.counts.trajectories
        ):
            raise CollectionContractViolation("trajectory count does not match schedule")
        if len(self.critic_rows) != self.schedule.counts.learned_rows:
            raise CollectionContractViolation("critic-row count does not match schedule")
        fixture_trajectory_counts: dict[str, int] = {}
        trajectories_by_fixture: dict[str, list[PlayerTrajectory]] = {}
        for bucket in self.actor_buckets:
            for trajectory in bucket.trajectories:
                fixture_trajectory_counts[trajectory.fixture_id] = (
                    fixture_trajectory_counts.get(trajectory.fixture_id, 0) + 1
                )
                trajectories_by_fixture.setdefault(trajectory.fixture_id, []).append(
                    trajectory
                )
        if any(
            fixture_trajectory_counts.get(fixture.fixture_id) != 2
            for fixture in self.schedule.fixtures
        ):
            raise CollectionContractViolation("each fixture must produce two trajectories")
        expected_critic_rows: list[tuple[PlayerTrajectory, PolicyTransition]] = []
        for fixture in self.schedule.fixtures:
            trajectories = trajectories_by_fixture.get(fixture.fixture_id, [])
            by_learner = {trajectory.learner: trajectory for trajectory in trajectories}
            if set(by_learner) != {fixture.policy_a, fixture.policy_b}:
                raise CollectionContractViolation(
                    "fixture trajectories do not contain both scheduled policies"
                )
            for identity in (fixture.policy_a, fixture.policy_b):
                trajectory = by_learner[identity]
                expected_player = (
                    EnginePlayer.QUEEN
                    if identity == fixture.queen
                    else EnginePlayer.KING
                )
                if trajectory.player is not expected_player:
                    raise CollectionContractViolation(
                        "trajectory player does not match fixture role"
                    )
                expected_critic_rows.extend(
                    (trajectory, transition)
                    for transition in trajectory.transitions
                    if transition.actor_loss_mask
                )
        for row, (trajectory, transition) in zip(
            self.critic_rows, expected_critic_rows, strict=True
        ):
            if (
                row.fixture_id != trajectory.fixture_id
                or row.learner != trajectory.learner
                or row.player is not trajectory.player
                or row.round_number != transition.round_number
                or row.recurrent_step_index != transition.recurrent_step_index
                or row.round_return != transition.round_return
                or not torch.equal(row.observation, transition.policy_input.observation)
            ):
                raise CollectionContractViolation(
                    "critic row does not match its learned transition"
                )


@dataclass(frozen=True, slots=True, eq=False)
class SealedCollectionArtifact:
    collection: CollectionData
    content_hash: str

    def __post_init__(self) -> None:
        if not isinstance(self.collection, CollectionData):
            raise CollectionContractViolation("sealed artifact requires collection data")
        _validate_digest(self.content_hash, "collection content hash")
        if self.content_hash != _collection_content_hash(self.collection):
            raise CollectionContractViolation("collection content hash does not match payload")


@dataclass(frozen=True, slots=True)
class _FrozenModels:
    policies: Mapping[PolicyVersion, Policy]
    policy_records: tuple[BehaviorPolicyRecord, ...]
    critic: Critic
    critic_fingerprint: str


def module_fingerprint(module: torch.nn.Module) -> str:
    """Return a stable digest of a module's named tensor state."""

    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        cpu_tensor = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(cpu_tensor.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(",".join(str(size) for size in cpu_tensor.shape).encode("ascii"))
        digest.update(b"\0")
        digest.update(cpu_tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _freeze_models(
    schedule: CollectionSchedule,
    policies: Mapping[PolicyVersion, Policy],
    critic: Critic,
) -> _FrozenModels:
    if set(policies) != set(schedule.policies):
        raise CollectionContractViolation("policy models do not match scheduled versions")
    frozen_policies: dict[PolicyVersion, Policy] = {}
    records: list[BehaviorPolicyRecord] = []
    for identity in schedule.policies:
        model = policies[identity]
        if not isinstance(model, Policy):
            raise CollectionContractViolation("behavior model must be a Policy")
        frozen = copy.deepcopy(model).cpu().eval()
        frozen.requires_grad_(False)
        frozen_policies[identity] = frozen
        records.append(BehaviorPolicyRecord(identity, module_fingerprint(frozen)))
    if not isinstance(critic, Critic):
        raise CollectionContractViolation("V_old model must be a Critic")
    frozen_critic = copy.deepcopy(critic).cpu().eval()
    frozen_critic.requires_grad_(False)
    return _FrozenModels(
        policies=frozen_policies,
        policy_records=tuple(records),
        critic=frozen_critic,
        critic_fingerprint=module_fingerprint(frozen_critic),
    )


def normalized_round_return(score: int, other_score: int) -> float:
    """Scale the rules-defined round-score difference to the fixed [-1, 1] target."""

    if type(score) is not int or type(other_score) is not int:
        raise CollectionContractViolation("round scores must be integers")
    if not 0 <= score <= 150 or not 0 <= other_score <= 150:
        raise CollectionContractViolation("round scores must be between 0 and 150")
    return (score - other_score) / ROUND_RETURN_SCALE


def action_sampling_seed(
    run_root_seed: str,
    fixture_id: str,
    behavior_policy_version: str,
    player: EnginePlayer,
    round_number: int,
    recurrent_step_index: int,
) -> int:
    return derive_pytorch_seed(
        ACTION_SAMPLING_NAMESPACE,
        run_root_seed,
        fixture_id,
        behavior_policy_version,
        EnginePlayer(player).value,
        str(round_number),
        str(recurrent_step_index),
    )


def sample_masked_action(
    raw_logits: Tensor, legal_mask: Tensor, *, seed: int
) -> tuple[int, float]:
    if type(seed) is not int or not 0 <= seed < 1 << 64:
        raise CollectionContractViolation(
            "sampling seed must be an unsigned 64-bit integer"
        )
    if (
        raw_logits.device.type != "cpu"
        or raw_logits.dtype is not torch.float32
        or raw_logits.shape != (4, 8)
    ):
        raise CollectionContractViolation("raw logits must be CPU float32[4,8]")
    if (
        legal_mask.device.type != "cpu"
        or legal_mask.dtype is not torch.bool
        or legal_mask.shape != (4, 8)
        or not legal_mask.any()
    ):
        raise CollectionContractViolation("legal mask must be nonempty CPU bool[4,8]")
    if not torch.isfinite(raw_logits).all():
        raise CollectionContractViolation("raw logits must be finite")
    flattened_logits = raw_logits.flatten()
    flattened_mask = legal_mask.flatten()
    masked_logits = flattened_logits.masked_fill(~flattened_mask, float("-inf"))
    probabilities = torch.softmax(masked_logits, dim=0)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    action_index = int(torch.multinomial(probabilities, 1, generator=generator).item())
    # Store the same stable log-softmax quantity recomputed by PPO replay.
    log_probability = float(torch.log_softmax(masked_logits, dim=0)[action_index].item())
    return action_index, log_probability


def _attach_round_return(
    transitions: list[PolicyTransition], round_number: int, round_return: float
) -> None:
    learned_in_round = 0
    for index, transition in enumerate(transitions):
        if transition.round_number != round_number:
            continue
        if transition.kind is PolicyTurnKind.LEARNED:
            transitions[index] = replace(transition, round_return=round_return)
            learned_in_round += 1
    if learned_in_round not in (3, 4):
        raise CollectionContractViolation("round did not produce the expected learned rows")


def _collect_fixture(
    fixture: MatchFixture,
    run_root_seed: str,
    frozen: _FrozenModels,
) -> tuple[FixtureResult, tuple[PlayerTrajectory, PlayerTrajectory]]:
    state = create_game(fixture.game_seed)
    identities = PlayerValues(queen=fixture.queen, king=fixture.king)
    hidden = PlayerValues(
        queen=frozen.policies[fixture.queen].initial_hidden(),
        king=frozen.policies[fixture.king].initial_hidden(),
    )
    transition_lists: PlayerValues[list[PolicyTransition]] = PlayerValues(
        queen=[], king=[]
    )
    round_returns: PlayerValues[list[float]] = PlayerValues(queen=[], king=[])

    with torch.inference_mode():
        while state.status is not EngineStatus.GAME_COMPLETE:
            while state.status is EngineStatus.PLAYING:
                player = state.active_player
                if player is None:
                    raise CollectionContractViolation("playing state has no active player")
                identity = identities[player]
                opponent_identity = identities[other_player(player)]
                policy = frozen.policies[identity]
                context = build_policy_turn_context(state, player)
                hidden_in = hidden[player]
                raw_logits, hidden_out = policy(
                    context.input.observation,
                    context.input.legal_mask,
                    hidden_in,
                )
                recurrent_step_index = len(transition_lists[player])
                if context.kind is PolicyTurnKind.LEARNED:
                    sampling_seed = action_sampling_seed(
                        run_root_seed,
                        fixture.fixture_id,
                        identity.version,
                        player,
                        context.round_number,
                        recurrent_step_index,
                    )
                    action_index, log_probability = sample_masked_action(
                        raw_logits, context.input.legal_mask, seed=sampling_seed
                    )
                    critic_value = float(
                        frozen.critic(context.input.observation).item()
                    )
                else:
                    action_index = None
                    log_probability = None
                    critic_value = None
                engine_transition = apply_policy_action(state, context, action_index)
                policy_transition = build_policy_transition(
                    context=context,
                    engine_transition=engine_transition,
                    fixture_id=fixture.fixture_id,
                    learner_id=identity.policy_id,
                    learner_policy_version=identity.version,
                    opponent_id=opponent_identity.policy_id,
                    opponent_policy_version=opponent_identity.version,
                    recurrent_step_index=recurrent_step_index,
                    policy_hidden_in=hidden_state_to_bytes(hidden_in),
                    policy_hidden_out=hidden_state_to_bytes(hidden_out),
                    action_log_probability=log_probability,
                    critic_value=critic_value,
                )
                transition_lists[player].append(policy_transition)
                hidden = hidden.updated(player, hidden_out)
                state = engine_transition.state

            result = state.pending_round_result
            if result is None:
                raise CollectionContractViolation("completed round has no result")
            queen_return = normalized_round_return(
                result.round_scores.queen, result.round_scores.king
            )
            king_return = -queen_return
            round_returns.queen.append(queen_return)
            round_returns.king.append(king_return)
            _attach_round_return(
                transition_lists.queen, state.round_number, queen_return
            )
            _attach_round_return(transition_lists.king, state.round_number, king_return)
            state = advance_after_round(state)

    if len(transition_lists.queen) != 24 or len(transition_lists.king) != 24:
        raise CollectionContractViolation("fixture did not produce two 24-step trajectories")
    fixture_result = FixtureResult(
        fixture=fixture,
        terminal_state_fingerprint=state_fingerprint(state),
        round_returns=PlayerValues(
            queen=tuple(round_returns.queen),  # type: ignore[arg-type]
            king=tuple(round_returns.king),  # type: ignore[arg-type]
        ),
    )
    trajectories_by_identity = {
        fixture.queen: PlayerTrajectory(
            fixture.fixture_id,
            fixture.queen,
            fixture.king,
            EnginePlayer.QUEEN,
            tuple(transition_lists.queen),
        ),
        fixture.king: PlayerTrajectory(
            fixture.fixture_id,
            fixture.king,
            fixture.queen,
            EnginePlayer.KING,
            tuple(transition_lists.king),
        ),
    }
    return fixture_result, (
        trajectories_by_identity[fixture.policy_a],
        trajectories_by_identity[fixture.policy_b],
    )


def collect_schedule(
    *,
    run_root_seed: str,
    schedule: CollectionSchedule,
    policies: Mapping[PolicyVersion, Policy],
    critic: Critic,
    critic_version: str,
    progress_callback: Callable[[int, int], None] | None = None,
) -> CollectionData:
    _validate_identifier(run_root_seed, "run root seed")
    _validate_identifier(critic_version, "critic version")
    frozen = _freeze_models(schedule, policies, critic)
    trajectories_by_policy: dict[PolicyVersion, list[PlayerTrajectory]] = {
        policy: [] for policy in schedule.policies
    }
    fixture_results: list[FixtureResult] = []
    critic_rows: list[CriticRow] = []

    fixture_count = len(schedule.fixtures)
    for completed, fixture in enumerate(schedule.fixtures, start=1):
        fixture_result, trajectories = _collect_fixture(
            fixture, run_root_seed, frozen
        )
        fixture_results.append(fixture_result)
        for trajectory in trajectories:
            trajectories_by_policy[trajectory.learner].append(trajectory)
            for transition in trajectory.transitions:
                if not transition.actor_loss_mask:
                    continue
                if transition.round_return is None:
                    raise CollectionContractViolation("learned row is missing round return")
                critic_rows.append(
                    CriticRow(
                        fixture_id=fixture.fixture_id,
                        learner=trajectory.learner,
                        player=trajectory.player,
                        round_number=transition.round_number,
                        recurrent_step_index=transition.recurrent_step_index,
                        observation=transition.policy_input.observation.clone(),
                        round_return=transition.round_return,
                    )
                )
        if progress_callback is not None:
            progress_callback(completed, fixture_count)

    actor_buckets = tuple(
        ActorBucket(policy, tuple(trajectories_by_policy[policy]))
        for policy in schedule.policies
    )
    return CollectionData(
        run_root_seed=run_root_seed,
        schedule=schedule,
        behavior_policies=frozen.policy_records,
        critic_version=critic_version,
        critic_parameter_fingerprint=frozen.critic_fingerprint,
        fixture_results=tuple(fixture_results),
        actor_buckets=actor_buckets,
        critic_rows=tuple(critic_rows),
    )


def replay_behavior_hidden_states(
    trajectory: PlayerTrajectory, policy: Policy
) -> None:
    if not isinstance(policy, Policy):
        raise CollectionContractViolation("replay requires a Policy")
    replay_policy = copy.deepcopy(policy).cpu().eval()
    replay_policy.requires_grad_(False)
    hidden = replay_policy.initial_hidden()
    with torch.inference_mode():
        for transition in trajectory.transitions:
            if hidden_state_to_bytes(hidden) != transition.policy_hidden_in:
                raise CollectionContractViolation("stored hidden input does not match replay")
            _, hidden = replay_policy(
                transition.policy_input.observation,
                transition.policy_input.legal_mask,
                hidden,
            )
            if hidden_state_to_bytes(hidden) != transition.policy_hidden_out:
                raise CollectionContractViolation("stored hidden output does not match replay")


def _pack_bool_tensor(tensor: Tensor) -> dict[str, object]:
    if tensor.device.type != "cpu" or tensor.dtype is not torch.bool:
        raise CollectionContractViolation("artifact tensors must be Boolean and on CPU")
    flat = tensor.flatten().tolist()
    packed = bytearray((len(flat) + 7) // 8)
    for index, value in enumerate(flat):
        if value:
            packed[index // 8] |= 1 << (7 - index % 8)
    return {
        "dtype": "bool",
        "shape": list(tensor.shape),
        "bit_order": "msb0",
        "data": base64.b64encode(bytes(packed)).decode("ascii"),
    }


def _unpack_bool_tensor(data: object, expected_shape: tuple[int, ...]) -> Tensor:
    if not isinstance(data, dict):
        raise CollectionContractViolation("packed tensor must be an object")
    if (
        data.get("dtype") != "bool"
        or data.get("shape") != list(expected_shape)
        or data.get("bit_order") != "msb0"
        or not isinstance(data.get("data"), str)
    ):
        raise CollectionContractViolation("packed tensor contract is invalid")
    try:
        packed = base64.b64decode(data["data"], validate=True)
    except (ValueError, TypeError) as error:
        raise CollectionContractViolation("packed tensor data is invalid base64") from error
    value_count = math.prod(expected_shape)
    if len(packed) != (value_count + 7) // 8:
        raise CollectionContractViolation("packed tensor byte length is invalid")
    values = [
        bool(packed[index // 8] & (1 << (7 - index % 8)))
        for index in range(value_count)
    ]
    unused_bits = len(packed) * 8 - value_count
    if unused_bits and packed[-1] & ((1 << unused_bits) - 1):
        raise CollectionContractViolation("packed tensor has nonzero padding bits")
    return torch.tensor(values, dtype=torch.bool).reshape(expected_shape)


def _float_data(value: float | None) -> str | None:
    return None if value is None else value.hex()


def _float_from_data(value: object) -> float | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise CollectionContractViolation("serialized float must be hexadecimal text")
    try:
        number = float.fromhex(value)
    except ValueError as error:
        raise CollectionContractViolation("serialized float is invalid") from error
    if not math.isfinite(number):
        raise CollectionContractViolation("serialized float must be finite")
    return number


def _identity_data(identity: PolicyVersion) -> dict[str, str]:
    return {"policy_id": identity.policy_id, "version": identity.version}


def _identity_from_data(data: object) -> PolicyVersion:
    if not isinstance(data, dict):
        raise CollectionContractViolation("policy identity must be an object")
    return PolicyVersion(data.get("policy_id"), data.get("version"))  # type: ignore[arg-type]


def _fixture_data(fixture: MatchFixture) -> dict[str, object]:
    return {
        "fixture_id": fixture.fixture_id,
        "generation_index": fixture.generation_index,
        "lane_index": fixture.lane_index,
        "game_counter": fixture.game_counter,
        "lane_root_seed_digest": fixture.lane_root_seed_digest,
        "game_seed": fixture.game_seed,
        "policy_a": _identity_data(fixture.policy_a),
        "policy_b": _identity_data(fixture.policy_b),
        "queen": _identity_data(fixture.queen),
        "king": _identity_data(fixture.king),
        "randomness_schema_version": fixture.randomness_schema_version,
        "phase": fixture.phase,
    }


def _fixture_from_data(data: object) -> MatchFixture:
    if not isinstance(data, dict):
        raise CollectionContractViolation("fixture must be an object")
    return MatchFixture(
        fixture_id=data.get("fixture_id"),  # type: ignore[arg-type]
        generation_index=data.get("generation_index"),  # type: ignore[arg-type]
        lane_index=data.get("lane_index"),  # type: ignore[arg-type]
        game_counter=data.get("game_counter"),  # type: ignore[arg-type]
        lane_root_seed_digest=data.get("lane_root_seed_digest"),  # type: ignore[arg-type]
        game_seed=data.get("game_seed"),  # type: ignore[arg-type]
        policy_a=_identity_from_data(data.get("policy_a")),
        policy_b=_identity_from_data(data.get("policy_b")),
        queen=_identity_from_data(data.get("queen")),
        king=_identity_from_data(data.get("king")),
        randomness_schema_version=data.get("randomness_schema_version"),  # type: ignore[arg-type]
        phase=data.get("phase"),  # type: ignore[arg-type]
    )


def _transition_data(transition: PolicyTransition) -> dict[str, object]:
    return {
        "fixture_id": transition.fixture_id,
        "learner_id": transition.learner_id,
        "learner_policy_version": transition.learner_policy_version,
        "opponent_id": transition.opponent_id,
        "opponent_policy_version": transition.opponent_policy_version,
        "player": transition.player.value,
        "state_fingerprint": transition.state_fingerprint,
        "round_number": transition.round_number,
        "own_decision_index": transition.own_decision_index,
        "recurrent_step_index": transition.recurrent_step_index,
        "kind": transition.kind.value,
        "policy_input": {
            "observation": _pack_bool_tensor(transition.policy_input.observation),
            "legal_mask": _pack_bool_tensor(transition.policy_input.legal_mask),
        },
        "policy_hidden_in": base64.b64encode(transition.policy_hidden_in).decode("ascii"),
        "action_index": transition.action_index,
        "action_log_probability": _float_data(transition.action_log_probability),
        "policy_hidden_out": base64.b64encode(transition.policy_hidden_out).decode("ascii"),
        "critic_value": _float_data(transition.critic_value),
        "round_return": _float_data(transition.round_return),
        "actor_loss_mask": transition.actor_loss_mask,
    }


def _hidden_from_data(value: object) -> bytes:
    if not isinstance(value, str):
        raise CollectionContractViolation("serialized hidden state must be base64 text")
    try:
        hidden = base64.b64decode(value, validate=True)
    except (ValueError, TypeError) as error:
        raise CollectionContractViolation("serialized hidden state is invalid") from error
    if len(hidden) != HIDDEN_STATE_BYTES:
        raise CollectionContractViolation("serialized hidden-state length is invalid")
    hidden_state_from_bytes(hidden)
    return hidden


def _transition_from_data(data: object) -> PolicyTransition:
    if not isinstance(data, dict) or not isinstance(data.get("policy_input"), dict):
        raise CollectionContractViolation("transition must be an object")
    policy_input_data = data["policy_input"]
    return PolicyTransition(
        fixture_id=data.get("fixture_id"),  # type: ignore[arg-type]
        learner_id=data.get("learner_id"),  # type: ignore[arg-type]
        learner_policy_version=data.get("learner_policy_version"),  # type: ignore[arg-type]
        opponent_id=data.get("opponent_id"),  # type: ignore[arg-type]
        opponent_policy_version=data.get("opponent_policy_version"),  # type: ignore[arg-type]
        player=EnginePlayer(data.get("player")),
        state_fingerprint=data.get("state_fingerprint"),  # type: ignore[arg-type]
        round_number=data.get("round_number"),  # type: ignore[arg-type]
        own_decision_index=data.get("own_decision_index"),  # type: ignore[arg-type]
        recurrent_step_index=data.get("recurrent_step_index"),  # type: ignore[arg-type]
        kind=PolicyTurnKind(data.get("kind")),
        policy_input=PolicyInput(
            _unpack_bool_tensor(policy_input_data.get("observation"), (875,)),
            _unpack_bool_tensor(policy_input_data.get("legal_mask"), (4, 8)),
        ),
        policy_hidden_in=_hidden_from_data(data.get("policy_hidden_in")),
        action_index=data.get("action_index"),  # type: ignore[arg-type]
        action_log_probability=_float_from_data(data.get("action_log_probability")),
        policy_hidden_out=_hidden_from_data(data.get("policy_hidden_out")),
        critic_value=_float_from_data(data.get("critic_value")),
        round_return=_float_from_data(data.get("round_return")),
        actor_loss_mask=data.get("actor_loss_mask"),  # type: ignore[arg-type]
    )


def _trajectory_data(trajectory: PlayerTrajectory) -> dict[str, object]:
    return {
        "fixture_id": trajectory.fixture_id,
        "learner": _identity_data(trajectory.learner),
        "opponent": _identity_data(trajectory.opponent),
        "player": trajectory.player.value,
        "transitions": [
            _transition_data(transition) for transition in trajectory.transitions
        ],
    }


def _trajectory_from_data(data: object) -> PlayerTrajectory:
    if not isinstance(data, dict) or not isinstance(data.get("transitions"), list):
        raise CollectionContractViolation("trajectory must be an object")
    return PlayerTrajectory(
        fixture_id=data.get("fixture_id"),  # type: ignore[arg-type]
        learner=_identity_from_data(data.get("learner")),
        opponent=_identity_from_data(data.get("opponent")),
        player=EnginePlayer(data.get("player")),
        transitions=tuple(
            _transition_from_data(transition) for transition in data["transitions"]
        ),
    )


def _collection_payload(collection: CollectionData) -> dict[str, object]:
    return {
        "format_version": collection.format_version,
        "rules_version": collection.rules_version,
        "card_schema_version": collection.card_schema_version,
        "engine_version": collection.engine_version,
        "randomness_schema_version": collection.randomness_schema_version,
        "run_root_seed": collection.run_root_seed,
        "critic_version": collection.critic_version,
        "critic_parameter_fingerprint": collection.critic_parameter_fingerprint,
        "schedule": {
            "policies": [_identity_data(policy) for policy in collection.schedule.policies],
            "lane_count": collection.schedule.lane_count,
            "generation_count": collection.schedule.generation_count,
            "start_game_counter": collection.schedule.start_game_counter,
            "fixtures": [_fixture_data(fixture) for fixture in collection.schedule.fixtures],
        },
        "behavior_policies": [
            {
                "identity": _identity_data(record.identity),
                "parameter_fingerprint": record.parameter_fingerprint,
            }
            for record in collection.behavior_policies
        ],
        "fixture_results": [
            {
                "fixture": _fixture_data(result.fixture),
                "terminal_state_fingerprint": result.terminal_state_fingerprint,
                "round_returns": {
                    "queen": [_float_data(value) for value in result.round_returns.queen],
                    "king": [_float_data(value) for value in result.round_returns.king],
                },
            }
            for result in collection.fixture_results
        ],
        "actor_buckets": [
            {
                "learner": _identity_data(bucket.learner),
                "trajectories": [
                    _trajectory_data(trajectory) for trajectory in bucket.trajectories
                ],
            }
            for bucket in collection.actor_buckets
        ],
        "critic_rows": [
            {
                "fixture_id": row.fixture_id,
                "learner": _identity_data(row.learner),
                "player": row.player.value,
                "round_number": row.round_number,
                "recurrent_step_index": row.recurrent_step_index,
                "observation": _pack_bool_tensor(row.observation),
                "round_return": _float_data(row.round_return),
            }
            for row in collection.critic_rows
        ],
    }


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def _collection_content_hash(collection: CollectionData) -> str:
    return hashlib.sha256(_canonical_json_bytes(_collection_payload(collection))).hexdigest()


def seal_collection(collection: CollectionData) -> SealedCollectionArtifact:
    return SealedCollectionArtifact(collection, _collection_content_hash(collection))


def artifact_bytes(artifact: SealedCollectionArtifact) -> bytes:
    if artifact.content_hash != _collection_content_hash(artifact.collection):
        raise CollectionContractViolation("collection changed after sealing")
    return _canonical_json_bytes(
        {
            "content_hash": artifact.content_hash,
            "payload": _collection_payload(artifact.collection),
        }
    )


def save_collection_artifact(
    artifact: SealedCollectionArtifact, path: str | os.PathLike[str]
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = artifact_bytes(artifact)
    temporary_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=destination.parent, prefix=f".{destination.name}.", delete=False
        ) as temporary:
            temporary_path = temporary.name
            temporary.write(payload)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, destination)
    finally:
        if temporary_path is not None and os.path.exists(temporary_path):
            os.unlink(temporary_path)


def _collection_from_payload(payload: object) -> CollectionData:
    if not isinstance(payload, dict) or not isinstance(payload.get("schedule"), dict):
        raise CollectionContractViolation("collection payload must be an object")
    schedule_data = payload["schedule"]
    if not isinstance(schedule_data.get("policies"), list) or not isinstance(
        schedule_data.get("fixtures"), list
    ):
        raise CollectionContractViolation("serialized schedule is invalid")
    schedule = CollectionSchedule(
        policies=tuple(
            _identity_from_data(identity) for identity in schedule_data["policies"]
        ),
        lane_count=schedule_data.get("lane_count"),  # type: ignore[arg-type]
        generation_count=schedule_data.get("generation_count"),  # type: ignore[arg-type]
        start_game_counter=schedule_data.get("start_game_counter"),  # type: ignore[arg-type]
        fixtures=tuple(
            _fixture_from_data(fixture) for fixture in schedule_data["fixtures"]
        ),
    )
    behavior_data = payload.get("behavior_policies")
    fixture_result_data = payload.get("fixture_results")
    bucket_data = payload.get("actor_buckets")
    critic_data = payload.get("critic_rows")
    if not all(
        isinstance(value, list)
        for value in (behavior_data, fixture_result_data, bucket_data, critic_data)
    ):
        raise CollectionContractViolation("serialized collection arrays are invalid")
    behavior_policies = tuple(
        BehaviorPolicyRecord(
            _identity_from_data(record.get("identity")),
            record.get("parameter_fingerprint"),
        )
        for record in behavior_data
        if isinstance(record, dict)
    )
    if len(behavior_policies) != len(behavior_data):
        raise CollectionContractViolation("serialized behavior policy is invalid")
    fixture_results: list[FixtureResult] = []
    for result in fixture_result_data:
        if not isinstance(result, dict) or not isinstance(result.get("round_returns"), dict):
            raise CollectionContractViolation("serialized fixture result is invalid")
        returns = result["round_returns"]
        queen_values = returns.get("queen")
        king_values = returns.get("king")
        if not isinstance(queen_values, list) or not isinstance(king_values, list):
            raise CollectionContractViolation("serialized round returns are invalid")
        fixture_results.append(
            FixtureResult(
                fixture=_fixture_from_data(result.get("fixture")),
                terminal_state_fingerprint=result.get(  # type: ignore[arg-type]
                    "terminal_state_fingerprint"
                ),
                round_returns=PlayerValues(
                    queen=tuple(  # type: ignore[arg-type]
                        _float_from_data(value) for value in queen_values
                    ),
                    king=tuple(  # type: ignore[arg-type]
                        _float_from_data(value) for value in king_values
                    ),
                ),
            )
        )
    actor_buckets = tuple(
        ActorBucket(
            learner=_identity_from_data(bucket.get("learner")),
            trajectories=tuple(
                _trajectory_from_data(trajectory)
                for trajectory in bucket.get("trajectories", [])
            ),
        )
        for bucket in bucket_data
        if isinstance(bucket, dict) and isinstance(bucket.get("trajectories"), list)
    )
    if len(actor_buckets) != len(bucket_data):
        raise CollectionContractViolation("serialized actor bucket is invalid")
    critic_rows: list[CriticRow] = []
    for row in critic_data:
        if not isinstance(row, dict):
            raise CollectionContractViolation("serialized critic row is invalid")
        round_return = _float_from_data(row.get("round_return"))
        if round_return is None:
            raise CollectionContractViolation("critic row is missing a return")
        critic_rows.append(
            CriticRow(
                fixture_id=row.get("fixture_id"),  # type: ignore[arg-type]
                learner=_identity_from_data(row.get("learner")),
                player=EnginePlayer(row.get("player")),
                round_number=row.get("round_number"),  # type: ignore[arg-type]
                recurrent_step_index=row.get("recurrent_step_index"),  # type: ignore[arg-type]
                observation=_unpack_bool_tensor(row.get("observation"), (875,)),
                round_return=round_return,
            )
        )
    return CollectionData(
        run_root_seed=payload.get("run_root_seed"),  # type: ignore[arg-type]
        schedule=schedule,
        behavior_policies=behavior_policies,
        critic_version=payload.get("critic_version"),  # type: ignore[arg-type]
        critic_parameter_fingerprint=payload.get(  # type: ignore[arg-type]
            "critic_parameter_fingerprint"
        ),
        fixture_results=tuple(fixture_results),
        actor_buckets=actor_buckets,
        critic_rows=tuple(critic_rows),
        format_version=payload.get("format_version"),  # type: ignore[arg-type]
        rules_version=payload.get("rules_version"),  # type: ignore[arg-type]
        card_schema_version=payload.get("card_schema_version"),  # type: ignore[arg-type]
        engine_version=payload.get("engine_version"),  # type: ignore[arg-type]
        randomness_schema_version=payload.get(  # type: ignore[arg-type]
            "randomness_schema_version"
        ),
    )


def load_collection_artifact(
    path: str | os.PathLike[str],
) -> SealedCollectionArtifact:
    try:
        envelope = json.loads(Path(path).read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        raise CollectionContractViolation("collection artifact is not valid JSON") from error
    if not isinstance(envelope, dict):
        raise CollectionContractViolation("collection artifact envelope is invalid")
    content_hash = envelope.get("content_hash")
    payload = envelope.get("payload")
    _validate_digest(content_hash, "collection content hash")
    if hashlib.sha256(_canonical_json_bytes(payload)).hexdigest() != content_hash:
        raise CollectionContractViolation("collection artifact content hash is invalid")
    collection = _collection_from_payload(payload)
    return SealedCollectionArtifact(collection, content_hash)

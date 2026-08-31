"""Held-out self-play evaluation and critic validation."""

from __future__ import annotations

import copy
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations

import torch
from torch import Tensor, nn

from dracula.bridge import PolicyTurnKind, apply_policy_action, build_policy_turn_context
from dracula.collection import (
    FIXTURE_GAME_NAMESPACE,
    MATCH_FIXTURE_NAMESPACE,
    PolicyVersion,
    normalized_round_return,
    sample_masked_action,
)
from dracula.engine import (
    EnginePlayer,
    EngineStatus,
    PlayerValues,
    advance_after_round,
    create_game,
    derive_game_outcome,
)
from dracula.models import Critic, Policy
from dracula.randomness import derive_pytorch_seed, derive_seed

EVALUATION_SAMPLING_NAMESPACE = "dracula-evaluation-sampling-v1"
EVALUATION_PHASE = "evaluation"


class EvaluationContractViolation(ValueError):
    """Held-out fixtures or model inputs violate the evaluation contract."""


@dataclass(frozen=True, slots=True)
class EvaluationFixture:
    fixture_id: str
    lane_index: int
    generation_index: int
    game_counter: int
    game_seed: str
    policy_a: PolicyVersion
    policy_b: PolicyVersion
    queen: PolicyVersion
    king: PolicyVersion


@dataclass(frozen=True, slots=True)
class EvaluationSchedule:
    policies: tuple[PolicyVersion, ...]
    lane_count: int
    generation_count: int
    start_game_counter: int
    fixtures: tuple[EvaluationFixture, ...]


@dataclass(frozen=True, slots=True)
class EvaluationFixtureResult:
    fixture_id: str
    queen: PolicyVersion
    king: PolicyVersion
    winner: PolicyVersion | None
    queen_round_returns: tuple[float, ...]
    king_round_returns: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class SplitMetrics:
    games: int
    wins: int
    ties: int
    victory_percentage: float | None
    mean_round_return: float | None


@dataclass(frozen=True, slots=True)
class PolicyEvaluationMetrics:
    policy: PolicyVersion
    overall: SplitMetrics
    queen: SplitMetrics
    king: SplitMetrics


@dataclass(frozen=True, slots=True)
class CriticValidationMetrics:
    row_count: int
    critic_mse: float
    zero_predictor_mse: float
    required_improvement: float
    passed: bool


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    schedule: EvaluationSchedule
    fixture_results: tuple[EvaluationFixtureResult, ...]
    policy_metrics: tuple[PolicyEvaluationMetrics, ...]
    critic_validation: CriticValidationMetrics


@dataclass(frozen=True, slots=True)
class _ValidationRow:
    observation: Tensor
    round_return: float


def build_evaluation_schedule(
    policies: Sequence[PolicyVersion],
    lane_root_seeds: Sequence[str],
    *,
    generation_count: int = 1,
    start_game_counter: int = 0,
) -> EvaluationSchedule:
    ordered = tuple(sorted(policies))
    lanes = tuple(lane_root_seeds)
    if len(ordered) < 2 or len(set(ordered)) != len(ordered):
        raise EvaluationContractViolation("evaluation policies must be distinct and sorted")
    if len({identity.policy_id for identity in ordered}) != len(ordered):
        raise EvaluationContractViolation("evaluation policy IDs must be unique")
    if not lanes or len(set(lanes)) != len(lanes):
        raise EvaluationContractViolation("held-out lane roots must be nonempty and distinct")
    if type(generation_count) is not int or generation_count < 1:
        raise EvaluationContractViolation("evaluation generations must be positive")
    if type(start_game_counter) is not int or start_game_counter < 0:
        raise EvaluationContractViolation("evaluation counter must be non-negative")

    fixtures: list[EvaluationFixture] = []
    for generation_index in range(generation_count):
        game_counter = start_game_counter + generation_index
        for lane_index, lane_root in enumerate(lanes):
            game_seed = derive_seed(
                FIXTURE_GAME_NAMESPACE, lane_root, str(game_counter)
            ).hex()
            for policy_a, policy_b in combinations(ordered, 2):
                fixture_id = derive_seed(
                    MATCH_FIXTURE_NAMESPACE,
                    EVALUATION_PHASE,
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
                    EvaluationFixture(
                        fixture_id=fixture_id,
                        lane_index=lane_index,
                        generation_index=generation_index,
                        game_counter=game_counter,
                        game_seed=game_seed,
                        policy_a=policy_a,
                        policy_b=policy_b,
                        queen=queen,
                        king=king,
                    )
                )
    return EvaluationSchedule(
        policies=ordered,
        lane_count=len(lanes),
        generation_count=generation_count,
        start_game_counter=start_game_counter,
        fixtures=tuple(fixtures),
    )


def evaluate_population(
    *,
    run_root_seed: str,
    schedule: EvaluationSchedule,
    policies: Mapping[PolicyVersion, Policy],
    critic: Critic,
    required_improvement: float = 0.05,
    progress_callback: Callable[[int, int], None] | None = None,
) -> EvaluationResult:
    if set(policies) != set(schedule.policies):
        raise EvaluationContractViolation("evaluation models do not match the schedule")
    if not math.isfinite(required_improvement) or not 0 <= required_improvement <= 1:
        raise EvaluationContractViolation("critic improvement must be in [0, 1]")
    frozen_policies = _freeze_policies(policies, schedule)
    frozen_critic = copy.deepcopy(critic).cpu().eval().requires_grad_(False)
    fixture_results: list[EvaluationFixtureResult] = []
    validation_rows: list[_ValidationRow] = []
    fixture_count = len(schedule.fixtures)
    for completed, fixture in enumerate(schedule.fixtures, start=1):
        result, rows = _evaluate_fixture(fixture, run_root_seed, frozen_policies)
        fixture_results.append(result)
        validation_rows.extend(rows)
        if progress_callback is not None:
            progress_callback(completed, fixture_count)
    policy_metrics = tuple(
        _policy_metrics(identity, fixture_results) for identity in schedule.policies
    )
    critic_validation = _critic_validation(
        frozen_critic, validation_rows, required_improvement
    )
    return EvaluationResult(
        schedule=schedule,
        fixture_results=tuple(fixture_results),
        policy_metrics=policy_metrics,
        critic_validation=critic_validation,
    )


def evaluate_matchups(
    *,
    run_root_seed: str,
    schedule: EvaluationSchedule,
    policies: Mapping[PolicyVersion, nn.Module],
    progress_callback: Callable[[int, int], None] | None = None,
) -> tuple[EvaluationFixtureResult, ...]:
    """Play scheduled matchups without constructing critic-validation rows."""

    frozen_policies = _freeze_policies(policies, schedule)
    fixture_results: list[EvaluationFixtureResult] = []
    fixture_count = len(schedule.fixtures)
    for completed, fixture in enumerate(schedule.fixtures, start=1):
        result, rows = _evaluate_fixture(
            fixture,
            run_root_seed,
            frozen_policies,
            collect_validation_rows=False,
        )
        if rows:
            raise EvaluationContractViolation("matchup evaluation retained critic rows")
        fixture_results.append(result)
        if progress_callback is not None:
            progress_callback(completed, fixture_count)
    return tuple(fixture_results)


def _freeze_policies(
    policies: Mapping[PolicyVersion, nn.Module], schedule: EvaluationSchedule
) -> dict[PolicyVersion, nn.Module]:
    if set(policies) != set(schedule.policies):
        raise EvaluationContractViolation("evaluation models do not match the schedule")
    return {
        identity: copy.deepcopy(policies[identity]).cpu().eval().requires_grad_(False)
        for identity in schedule.policies
    }


def _evaluate_fixture(
    fixture: EvaluationFixture,
    run_root_seed: str,
    policies: Mapping[PolicyVersion, nn.Module],
    *,
    collect_validation_rows: bool = True,
) -> tuple[EvaluationFixtureResult, tuple[_ValidationRow, ...]]:
    state = create_game(fixture.game_seed)
    identities = PlayerValues(queen=fixture.queen, king=fixture.king)
    hidden = PlayerValues(
        queen=policies[fixture.queen].initial_hidden(),
        king=policies[fixture.king].initial_hidden(),
    )
    learned_by_round: PlayerValues[list[list[Tensor]]] = PlayerValues(
        queen=[[] for _ in range(6)], king=[[] for _ in range(6)]
    )
    round_returns: PlayerValues[list[float]] = PlayerValues(queen=[], king=[])

    with torch.inference_mode():
        while state.status is not EngineStatus.GAME_COMPLETE:
            while state.status is EngineStatus.PLAYING:
                player = state.active_player
                if player is None:
                    raise EvaluationContractViolation("playing evaluation has no active player")
                identity = identities[player]
                context = build_policy_turn_context(state, player)
                logits, hidden_out = policies[identity](
                    context.input.observation,
                    context.input.legal_mask,
                    hidden[player],
                )
                if context.kind is PolicyTurnKind.LEARNED:
                    recurrent_step = (context.round_number - 1) * 4 + context.own_decision_index
                    seed = derive_pytorch_seed(
                        EVALUATION_SAMPLING_NAMESPACE,
                        run_root_seed,
                        fixture.fixture_id,
                        identity.version,
                        player.value,
                        str(context.round_number),
                        str(recurrent_step),
                    )
                    action_index, _ = sample_masked_action(
                        logits, context.input.legal_mask, seed=seed
                    )
                    if collect_validation_rows:
                        learned_by_round[player][context.round_number - 1].append(
                            context.input.observation.clone()
                        )
                else:
                    action_index = None
                transition = apply_policy_action(state, context, action_index)
                hidden = hidden.updated(player, hidden_out)
                state = transition.state

            result = state.pending_round_result
            if result is None:
                raise EvaluationContractViolation("evaluation round has no result")
            queen_return = normalized_round_return(
                result.round_scores.queen, result.round_scores.king
            )
            round_returns.queen.append(queen_return)
            round_returns.king.append(-queen_return)
            state = advance_after_round(state)

    outcome = derive_game_outcome(state)
    winner = None if outcome.winner is None else identities[outcome.winner]
    rows: list[_ValidationRow] = []
    for player in EnginePlayer:
        for round_index, observations in enumerate(learned_by_round[player]):
            rows.extend(
                _ValidationRow(observation, round_returns[player][round_index])
                for observation in observations
            )
    if collect_validation_rows and len(rows) != 42:
        raise EvaluationContractViolation("evaluation fixture must produce 42 learned rows")
    return (
        EvaluationFixtureResult(
            fixture_id=fixture.fixture_id,
            queen=fixture.queen,
            king=fixture.king,
            winner=winner,
            queen_round_returns=tuple(round_returns.queen),
            king_round_returns=tuple(round_returns.king),
        ),
        tuple(rows),
    )


def _policy_metrics(
    identity: PolicyVersion, results: Sequence[EvaluationFixtureResult]
) -> PolicyEvaluationMetrics:
    relevant = [result for result in results if identity in (result.queen, result.king)]
    queen_results = [result for result in relevant if result.queen == identity]
    king_results = [result for result in relevant if result.king == identity]
    return PolicyEvaluationMetrics(
        policy=identity,
        overall=_split_metrics(identity, relevant),
        queen=_split_metrics(identity, queen_results),
        king=_split_metrics(identity, king_results),
    )


def _split_metrics(
    identity: PolicyVersion, results: Sequence[EvaluationFixtureResult]
) -> SplitMetrics:
    games = len(results)
    wins = sum(result.winner == identity for result in results)
    ties = sum(result.winner is None for result in results)
    returns: list[float] = []
    for result in results:
        returns.extend(
            result.queen_round_returns
            if result.queen == identity
            else result.king_round_returns
        )
    return SplitMetrics(
        games=games,
        wins=wins,
        ties=ties,
        victory_percentage=(wins + 0.5 * ties) / games if games else None,
        mean_round_return=sum(returns) / len(returns) if returns else None,
    )


def _critic_validation(
    critic: Critic,
    rows: Sequence[_ValidationRow],
    required_improvement: float,
) -> CriticValidationMetrics:
    if not rows:
        raise EvaluationContractViolation("critic validation requires held-out rows")
    squared_error = 0.0
    zero_squared_error = 0.0
    with torch.inference_mode():
        for start in range(0, len(rows), 512):
            batch = rows[start : start + 512]
            observations = torch.stack([row.observation for row in batch]).clone()
            targets = torch.tensor([row.round_return for row in batch], dtype=torch.float32)
            predictions = critic(observations)
            if not torch.isfinite(predictions).all():
                raise EvaluationContractViolation("critic validation produced non-finite values")
            squared_error += float(torch.square(predictions - targets).sum().item())
            zero_squared_error += float(torch.square(targets).sum().item())
    critic_mse = squared_error / len(rows)
    zero_mse = zero_squared_error / len(rows)
    return CriticValidationMetrics(
        row_count=len(rows),
        critic_mse=critic_mse,
        zero_predictor_mse=zero_mse,
        required_improvement=required_improvement,
        passed=critic_mse <= (1.0 - required_improvement) * zero_mse,
    )

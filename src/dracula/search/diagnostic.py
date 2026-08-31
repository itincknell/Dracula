"""Privileged search diagnostics excluded from the deployable search API."""

from __future__ import annotations

import math
from dataclasses import dataclass

from dracula.bridge import ACTION_COUNT, action_index_for_move
from dracula.engine import (
    EngineState,
    EngineStatus,
    SimulationEngineState,
    apply_simulation_move,
    legal_simulation_moves,
    shuffled_deck,
    validate_state,
)
from dracula.search.planner import normalized_round_return


class DiagnosticSearchLimit(RuntimeError):
    """The privileged exact solver exceeded its explicit state limit."""


@dataclass(frozen=True, slots=True)
class PerfectInformationResult:
    selected_action_index: int
    action_values: tuple[float | None, ...]
    evaluated_states: int


def _simulation_state(state: EngineState) -> SimulationEngineState:
    if isinstance(state, SimulationEngineState):
        return state
    return SimulationEngineState(
        seed=state.seed,
        status=state.status,
        round_number=state.round_number,
        dealer=state.dealer,
        active_player=state.active_player,
        stock=state.stock,
        hands=state.hands,
        coffin=state.coffin,
        current_round_moves=state.current_round_moves,
        pending_round_result=state.pending_round_result,
        completed_rounds=state.completed_rounds,
        total_scores=state.total_scores,
        simulation_deck=shuffled_deck(state.seed),
    )


def _round_state_key(state: SimulationEngineState) -> tuple[object, ...]:
    return (
        state.status,
        state.active_player,
        state.hands,
        state.coffin,
        state.current_round_moves,
    )


def solve_perfect_information_round(
    state: EngineState,
    *,
    diagnostic_only: bool = False,
    maximum_states: int = 100_000,
) -> PerfectInformationResult:
    """Solve a known world against a uniform opponent for upper-bound analysis."""

    if diagnostic_only is not True:
        raise PermissionError("perfect-information search requires diagnostic_only=True")
    validate_state(state)
    if state.status is not EngineStatus.PLAYING or state.active_player is None:
        raise ValueError("perfect-information search requires an active round")
    if type(maximum_states) is not int or maximum_states < 1:
        raise ValueError("maximum states must be a positive integer")

    simulation = _simulation_state(state)
    root_player = simulation.active_player
    evaluated_states = 0
    memo: dict[tuple[object, ...], float] = {}

    def value(current: SimulationEngineState) -> float:
        nonlocal evaluated_states
        if current.status is EngineStatus.ROUND_COMPLETE:
            if current.pending_round_result is None:
                raise ValueError("completed engine round is missing its result")
            return normalized_round_return(current.pending_round_result, root_player)

        key = _round_state_key(current)
        cached = memo.get(key)
        if cached is not None:
            return cached
        evaluated_states += 1
        if evaluated_states > maximum_states:
            raise DiagnosticSearchLimit(
                f"perfect-information search exceeded {maximum_states} states"
            )
        actor = current.active_player
        if actor is None:
            raise ValueError("playing engine state is missing its active player")
        moves = legal_simulation_moves(current, actor)
        child_values = tuple(
            value(apply_simulation_move(current, move)) for move in moves
        )
        if len(child_values) == 1:
            result = child_values[0]
        elif actor is root_player:
            result = max(child_values)
        else:
            result = sum(child_values) / len(child_values)
        memo[key] = result
        return result

    action_values: list[float | None] = [None] * ACTION_COUNT
    for move in legal_simulation_moves(simulation, root_player):
        action_values[action_index_for_move(move, root_player)] = value(
            apply_simulation_move(simulation, move)
        )
    selected = min(
        (index for index, action_value in enumerate(action_values) if action_value is not None),
        key=lambda index: (-float(action_values[index]), index),
    )
    return PerfectInformationResult(selected, tuple(action_values), evaluated_states)


def solve_adversarial_perfect_information_round(
    state: EngineState,
    *,
    diagnostic_only: bool = False,
    maximum_states: int = 100_000,
) -> PerfectInformationResult:
    """Solve a known round with maximizing root and minimizing opponent."""

    if diagnostic_only is not True:
        raise PermissionError("perfect-information search requires diagnostic_only=True")
    validate_state(state)
    if state.status is not EngineStatus.PLAYING or state.active_player is None:
        raise ValueError("perfect-information search requires an active round")
    if type(maximum_states) is not int or maximum_states < 1:
        raise ValueError("maximum states must be a positive integer")

    simulation = _simulation_state(state)
    root_player = simulation.active_player
    evaluated_states = 0
    def value(
        current: SimulationEngineState, alpha: float, beta: float
    ) -> float:
        nonlocal evaluated_states
        if current.status is EngineStatus.ROUND_COMPLETE:
            if current.pending_round_result is None:
                raise ValueError("completed engine round is missing its result")
            return normalized_round_return(current.pending_round_result, root_player)

        evaluated_states += 1
        if evaluated_states > maximum_states:
            raise DiagnosticSearchLimit(
                f"perfect-information search exceeded {maximum_states} states"
            )
        actor = current.active_player
        if actor is None:
            raise ValueError("playing engine state is missing its active player")
        if actor is root_player:
            result = -math.inf
            for move in legal_simulation_moves(current, actor):
                result = max(
                    result,
                    value(apply_simulation_move(current, move), alpha, beta),
                )
                alpha = max(alpha, result)
                if alpha >= beta:
                    break
            return result

        result = math.inf
        for move in legal_simulation_moves(current, actor):
            result = min(
                result,
                value(apply_simulation_move(current, move), alpha, beta),
            )
            beta = min(beta, result)
            if alpha >= beta:
                break
        return result

    action_values: list[float | None] = [None] * ACTION_COUNT
    for move in legal_simulation_moves(simulation, root_player):
        action_values[action_index_for_move(move, root_player)] = value(
            apply_simulation_move(simulation, move),
            -math.inf,
            math.inf,
        )
    selected = min(
        (
            index
            for index, action_value in enumerate(action_values)
            if action_value is not None
        ),
        key=lambda index: (-float(action_values[index]), index),
    )
    return PerfectInformationResult(selected, tuple(action_values), evaluated_states)

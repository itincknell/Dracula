"""Privileged search diagnostics excluded from the deployable search API."""

from __future__ import annotations

from dataclasses import dataclass

from dracula.bridge import ACTION_COUNT, action_index_for_move
from dracula.engine import (
    EnginePlayer,
    EngineState,
    EngineStatus,
    apply_move,
    legal_moves,
    state_fingerprint,
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

    root_player = state.active_player
    evaluated_states = 0
    memo: dict[str, float] = {}

    def value(current: EngineState) -> float:
        nonlocal evaluated_states
        if current.status is EngineStatus.ROUND_COMPLETE:
            if current.pending_round_result is None:
                raise ValueError("completed engine round is missing its result")
            return normalized_round_return(current.pending_round_result, root_player)

        fingerprint = state_fingerprint(current)
        cached = memo.get(fingerprint)
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
        moves = legal_moves(current, actor)
        child_values = tuple(value(apply_move(current, move).state) for move in moves)
        if len(child_values) == 1:
            result = child_values[0]
        elif actor is root_player:
            result = max(child_values)
        else:
            result = sum(child_values) / len(child_values)
        memo[fingerprint] = result
        return result

    action_values: list[float | None] = [None] * ACTION_COUNT
    for move in legal_moves(state, root_player):
        action_values[action_index_for_move(move, root_player)] = value(
            apply_move(state, move).state
        )
    selected = min(
        (index for index, action_value in enumerate(action_values) if action_value is not None),
        key=lambda index: (-float(action_values[index]), index),
    )
    return PerfectInformationResult(selected, tuple(action_values), evaluated_states)


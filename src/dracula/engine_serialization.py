"""Serialize and restore complete private engine states canonically.

This format supports trusted local persistence and deterministic fingerprints.
Its output contains hidden game data and must never become a public response.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

from dracula.engine_types import (
    EnginePlayedMove,
    EnginePlayer,
    EngineRoundResult,
    EngineState,
    EngineStatus,
    LineOrientation,
    LineScore,
    MultiplierReason,
    PlayerValues,
)
from dracula.engine_validation import validate_state


def _player_values_json(values: PlayerValues[object], encode: object) -> dict[str, object]:
    """Encode a role pair with stable Queen-then-King field order."""

    encoder = encode if callable(encode) else lambda value: value
    return {
        EnginePlayer.QUEEN.value: encoder(values.queen),
        EnginePlayer.KING.value: encoder(values.king),
    }


def _played_move_json(move: EnginePlayedMove) -> dict[str, object]:
    """Encode every accepted-move field used by private fingerprints."""

    return {
        "player": move.player.value,
        "card_id": move.card_id,
        "hand_slot": move.hand_slot,
        "global_grid_index": move.global_grid_index,
        "turn_number": move.turn_number,
    }


def _line_score_json(line: LineScore) -> dict[str, object]:
    """Encode the complete auditable calculation for one line."""

    return {
        "orientation": line.orientation.value,
        "line_index": line.line_index,
        "cards": list(line.cards),
        "values": list(line.values),
        "base_value": line.base_value,
        "multiplier": line.multiplier,
        "multiplier_reason": line.multiplier_reason.value,
        "total": line.total,
    }


def _round_result_json(result: EngineRoundResult) -> dict[str, object]:
    """Encode one completed result, including its verified scoring evidence."""

    return {
        "round_number": result.round_number,
        "dealer": result.dealer.value,
        "coffin": list(result.coffin),
        "moves": [_played_move_json(move) for move in result.moves],
        "line_scores": _player_values_json(
            result.line_scores,
            lambda lines: [_line_score_json(line) for line in lines],
        ),
        "round_scores": _player_values_json(result.round_scores, lambda value: value),
    }


def canonical_state_data(state: EngineState) -> dict[str, object]:
    """Return the complete private state in canonical fingerprint order."""

    validate_state(state)
    return {
        "state": {
            "seed": state.seed,
            "status": state.status.value,
            "round_number": state.round_number,
            "dealer": state.dealer.value,
            "active_player": None if state.active_player is None else state.active_player.value,
            "stock": list(state.stock),
            "hands": _player_values_json(state.hands, lambda hand: list(hand)),
            "coffin": list(state.coffin),
            "current_round_moves": [
                _played_move_json(move) for move in state.current_round_moves
            ],
            "pending_round_result": (
                None
                if state.pending_round_result is None
                else _round_result_json(state.pending_round_result)
            ),
            "completed_rounds": [
                _round_result_json(result) for result in state.completed_rounds
            ],
            "total_scores": _player_values_json(state.total_scores, lambda value: value),
        },
    }


def _played_move_from_data(value: Mapping[str, Any]) -> EnginePlayedMove:
    return EnginePlayedMove(
        player=EnginePlayer(value["player"]),
        card_id=str(value["card_id"]),
        hand_slot=int(value["hand_slot"]),
        global_grid_index=int(value["global_grid_index"]),
        turn_number=int(value["turn_number"]),
    )


def _line_score_from_data(value: Mapping[str, Any]) -> LineScore:
    return LineScore(
        orientation=LineOrientation(value["orientation"]),
        line_index=int(value["line_index"]),
        cards=tuple(value["cards"]),  # type: ignore[arg-type]
        values=tuple(int(item) for item in value["values"]),  # type: ignore[arg-type]
        base_value=int(value["base_value"]),
        multiplier=int(value["multiplier"]),
        multiplier_reason=MultiplierReason(value["multiplier_reason"]),
        total=int(value["total"]),
    )


def _player_values_from_data(
    value: Mapping[str, Any], decode: Any
) -> PlayerValues[Any]:
    return PlayerValues(queen=decode(value["queen"]), king=decode(value["king"]))


def _round_result_from_data(value: Mapping[str, Any]) -> EngineRoundResult:
    return EngineRoundResult(
        round_number=int(value["round_number"]),
        dealer=EnginePlayer(value["dealer"]),
        coffin=tuple(value["coffin"]),  # type: ignore[arg-type]
        moves=tuple(_played_move_from_data(item) for item in value["moves"]),
        line_scores=_player_values_from_data(
            value["line_scores"],
            lambda lines: tuple(_line_score_from_data(line) for line in lines),
        ),
        round_scores=_player_values_from_data(value["round_scores"], int),
    )


def engine_state_from_data(value: Mapping[str, Any]) -> EngineState:
    """Decode and fully validate a canonical private engine-state mapping."""

    state = EngineState(
        seed=str(value["seed"]),
        status=EngineStatus(value["status"]),
        round_number=int(value["round_number"]),
        dealer=EnginePlayer(value["dealer"]),
        active_player=(
            None
            if value["active_player"] is None
            else EnginePlayer(value["active_player"])
        ),
        stock=tuple(value["stock"]),
        hands=_player_values_from_data(value["hands"], lambda hand: tuple(hand)),
        coffin=tuple(value["coffin"]),  # type: ignore[arg-type]
        current_round_moves=tuple(
            _played_move_from_data(item) for item in value["current_round_moves"]
        ),
        pending_round_result=(
            None
            if value["pending_round_result"] is None
            else _round_result_from_data(value["pending_round_result"])
        ),
        completed_rounds=tuple(
            _round_result_from_data(item) for item in value["completed_rounds"]
        ),
        total_scores=_player_values_from_data(value["total_scores"], int),
    )
    validate_state(state)
    return state


def canonical_state_json(state: EngineState) -> str:
    """Serialize the complete private state with fingerprint-stable field order."""

    # Dictionary insertion order is part of the fingerprint contract.
    return json.dumps(
        canonical_state_data(state),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=False,
    )


def state_fingerprint(state: EngineState) -> str:
    """Return the SHA-256 fingerprint of canonical private state data."""

    return hashlib.sha256(canonical_state_json(state).encode("utf-8")).hexdigest()

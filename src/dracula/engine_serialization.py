"""Canonical private engine serialization and state fingerprints."""

from __future__ import annotations

import hashlib
import json

from dracula.cards import CARD_SCHEMA_VERSION
from dracula.engine_types import (
    ENGINE_VERSION,
    RULES_VERSION,
    EnginePlayedMove,
    EnginePlayer,
    EngineRoundResult,
    EngineState,
    LineScore,
    PlayerValues,
)
from dracula.engine_validation import validate_state
from dracula.randomness import RANDOMNESS_SCHEMA_VERSION


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
        "rules_version": RULES_VERSION,
        "card_schema_version": CARD_SCHEMA_VERSION,
        "engine_version": ENGINE_VERSION,
        "randomness_schema_version": RANDOMNESS_SCHEMA_VERSION,
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


def canonical_state_json(state: EngineState) -> str:
    """Serialize the complete private state with compatibility-stable field order."""

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

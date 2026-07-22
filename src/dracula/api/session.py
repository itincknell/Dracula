"""Persisted application state and public projections over the pure engine."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import math
import re
import struct
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Mapping
from uuid import UUID, uuid5, NAMESPACE_URL

from pydantic import TypeAdapter

from dracula.api.contracts import (
    GameCompletePhase,
    HumanGameView,
    HumanTurnPhase,
    LegalMove,
    LineScore as ApiLineScore,
    OpponentTurnPhase,
    PlayedMove,
    PlayerScore,
    PublicEvent,
    ResumablePhase,
    RoundRecord,
    ScoringPhase,
    ScoringStep,
)
from dracula.cards import Suit, card_by_id
from dracula.engine import (
    EnginePlayedMove,
    EnginePlayer,
    EngineRoundResult,
    EngineState,
    EngineStatus,
    LineOrientation,
    LineScore,
    MultiplierReason,
    PlayerValues,
    canonical_state_data,
    derive_game_outcome,
    legal_moves,
    other_player,
    validate_state,
)

HIDDEN_FLOAT_COUNT = 128
HIDDEN_STATE_BYTES = HIDDEN_FLOAT_COUNT * 4
SESSION_SCHEMA_VERSION = "dracula-game-session-v1"
_PHASE_ADAPTER = TypeAdapter(ResumablePhase)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class PolicySession:
    policy_id: str
    policy_version: str
    artifact_id: str
    artifact_sha256: str
    observation_schema_version: str
    action_schema_version: str
    hidden_state_schema_version: str
    inference_profile: str
    hidden_state: bytes


@dataclass(frozen=True, slots=True)
class PolicyTurnClaim:
    job_id: str
    request_id: str
    request_hash: str
    expected_version: int


@dataclass(frozen=True, slots=True)
class IdempotencyRecord:
    request_id: str
    request_hash: str
    status_code: int
    response_json: str


@dataclass(frozen=True, slots=True)
class GameSession:
    game_id: UUID
    version: int
    revision: int
    human_role: EnginePlayer
    engine_state: EngineState
    policy_session: PolicySession
    move_secret: bytes
    phase: ResumablePhase
    events: tuple[PublicEvent, ...]
    idempotency_records: tuple[IdempotencyRecord, ...]
    policy_turn_claim: PolicyTurnClaim | None = None


def zero_hidden_state() -> bytes:
    return bytes(HIDDEN_STATE_BYTES)


def validate_hidden_state(value: bytes) -> None:
    if not isinstance(value, bytes) or len(value) != HIDDEN_STATE_BYTES:
        raise ValueError(f"policy hidden state must contain {HIDDEN_STATE_BYTES} bytes")
    if not all(math.isfinite(item) for item in struct.unpack("<128f", value)):
        raise ValueError("policy hidden state contains a non-finite value")


def validate_session(session: GameSession) -> None:
    if not isinstance(session, GameSession):
        raise TypeError("session must be a GameSession")
    validate_state(session.engine_state)
    if session.version < 0 or session.revision < 0:
        raise ValueError("session versions must be non-negative")
    if not isinstance(session.human_role, EnginePlayer):
        raise ValueError("session human role is invalid")
    if len(session.move_secret) < 32:
        raise ValueError("move token secret must contain at least 32 bytes")
    validate_hidden_state(session.policy_session.hidden_state)
    policy = session.policy_session
    if any(
        not isinstance(value, str) or not value
        for value in (
            policy.policy_id,
            policy.policy_version,
            policy.artifact_id,
            policy.artifact_sha256,
            policy.observation_schema_version,
            policy.action_schema_version,
            policy.hidden_state_schema_version,
            policy.inference_profile,
        )
    ):
        raise ValueError("policy session metadata must be nonempty")
    if policy.artifact_sha256 != "none" and (
        _SHA256.fullmatch(policy.artifact_sha256) is None
        or policy.artifact_id != f"sha256:{policy.artifact_sha256}"
    ):
        raise ValueError("policy artifact digest and ID do not match")
    expected_sequences = tuple(range(1, len(session.events) + 1))
    if tuple(event.sequence for event in session.events) != expected_sequences:
        raise ValueError("public event sequence is not contiguous")
    if any(event.game_id != session.game_id for event in session.events):
        raise ValueError("public event belongs to another game")
    request_ids = tuple(record.request_id for record in session.idempotency_records)
    if len(request_ids) != len(set(request_ids)):
        raise ValueError("session contains duplicate idempotency records")


def _played_move_from_data(value: Mapping[str, Any]) -> EnginePlayedMove:
    return EnginePlayedMove(
        player=EnginePlayer(value["player"]),
        card_id=str(value["card_id"]),
        hand_slot=int(value["hand_slot"]),
        global_grid_index=int(value["global_grid_index"]),
        turn_number=int(value["turn_number"]),
    )


def _line_from_data(value: Mapping[str, Any]) -> LineScore:
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


def _player_values_from_data(value: Mapping[str, Any], decode: Any) -> PlayerValues[Any]:
    return PlayerValues(queen=decode(value["queen"]), king=decode(value["king"]))


def _round_from_data(value: Mapping[str, Any]) -> EngineRoundResult:
    return EngineRoundResult(
        round_number=int(value["round_number"]),
        dealer=EnginePlayer(value["dealer"]),
        coffin=tuple(value["coffin"]),  # type: ignore[arg-type]
        moves=tuple(_played_move_from_data(item) for item in value["moves"]),
        line_scores=_player_values_from_data(
            value["line_scores"], lambda lines: tuple(_line_from_data(line) for line in lines)
        ),
        round_scores=_player_values_from_data(value["round_scores"], int),
    )


def engine_state_from_data(value: Mapping[str, Any]) -> EngineState:
    state = EngineState(
        seed=str(value["seed"]),
        status=EngineStatus(value["status"]),
        round_number=int(value["round_number"]),
        dealer=EnginePlayer(value["dealer"]),
        active_player=(
            None if value["active_player"] is None else EnginePlayer(value["active_player"])
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
            else _round_from_data(value["pending_round_result"])
        ),
        completed_rounds=tuple(_round_from_data(item) for item in value["completed_rounds"]),
        total_scores=_player_values_from_data(value["total_scores"], int),
    )
    validate_state(state)
    return state


def session_core_data(session: GameSession) -> dict[str, Any]:
    """Encode private state explicitly; repositories store events and requests separately."""

    validate_session(session)
    engine_data = canonical_state_data(session.engine_state)["state"]
    policy = session.policy_session
    return {
        "schema_version": SESSION_SCHEMA_VERSION,
        "game_id": str(session.game_id),
        "version": session.version,
        "revision": session.revision,
        "human_role": session.human_role.value,
        "engine_state": engine_data,
        "policy_session": {
            "policy_id": policy.policy_id,
            "policy_version": policy.policy_version,
            "artifact_id": policy.artifact_id,
            "artifact_sha256": policy.artifact_sha256,
            "observation_schema_version": policy.observation_schema_version,
            "action_schema_version": policy.action_schema_version,
            "hidden_state_schema_version": policy.hidden_state_schema_version,
            "inference_profile": policy.inference_profile,
            "hidden_state": base64.b64encode(policy.hidden_state).decode("ascii"),
        },
        "move_secret": base64.b64encode(session.move_secret).decode("ascii"),
        "phase": _PHASE_ADAPTER.dump_python(session.phase, mode="json"),
        "policy_turn_claim": (
            None
            if session.policy_turn_claim is None
            else {
                "job_id": session.policy_turn_claim.job_id,
                "request_id": session.policy_turn_claim.request_id,
                "request_hash": session.policy_turn_claim.request_hash,
                "expected_version": session.policy_turn_claim.expected_version,
            }
        ),
    }


def session_from_core_data(
    value: Mapping[str, Any],
    events: tuple[PublicEvent, ...],
    idempotency_records: tuple[IdempotencyRecord, ...],
) -> GameSession:
    if value.get("schema_version") != SESSION_SCHEMA_VERSION:
        raise ValueError("unsupported game session schema")
    policy_value = value["policy_session"]
    claim_value = value["policy_turn_claim"]
    session = GameSession(
        game_id=UUID(value["game_id"]),
        version=int(value["version"]),
        revision=int(value["revision"]),
        human_role=EnginePlayer(value["human_role"]),
        engine_state=engine_state_from_data(value["engine_state"]),
        policy_session=PolicySession(
            policy_id=str(policy_value["policy_id"]),
            policy_version=str(policy_value["policy_version"]),
            artifact_id=str(policy_value["artifact_id"]),
            artifact_sha256=str(policy_value.get("artifact_sha256", "none")),
            observation_schema_version=str(policy_value["observation_schema_version"]),
            action_schema_version=str(policy_value["action_schema_version"]),
            hidden_state_schema_version=str(policy_value["hidden_state_schema_version"]),
            inference_profile=str(policy_value["inference_profile"]),
            hidden_state=base64.b64decode(policy_value["hidden_state"], validate=True),
        ),
        move_secret=base64.b64decode(value["move_secret"], validate=True),
        phase=_PHASE_ADAPTER.validate_python(value["phase"]),
        events=events,
        idempotency_records=idempotency_records,
        policy_turn_claim=(
            None
            if claim_value is None
            else PolicyTurnClaim(
                job_id=str(claim_value["job_id"]),
                request_id=str(claim_value["request_id"]),
                request_hash=str(claim_value["request_hash"]),
                expected_version=int(claim_value["expected_version"]),
            )
        ),
    )
    validate_session(session)
    return session


def request_fingerprint(body: Mapping[str, Any]) -> str:
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def response_record(
    request_id: str, request_hash: str, status_code: int, body: Mapping[str, Any]
) -> IdempotencyRecord:
    return IdempotencyRecord(
        request_id=request_id,
        request_hash=request_hash,
        status_code=status_code,
        response_json=json.dumps(body, sort_keys=True, separators=(",", ":")),
    )


def response_body(record: IdempotencyRecord) -> dict[str, Any]:
    value = json.loads(record.response_json)
    if not isinstance(value, dict):
        raise ValueError("persisted idempotency response is not an object")
    return value


def make_event(
    game_id: UUID,
    sequence: int,
    event_type: str,
    occurred_at: datetime,
    prior_version: int,
    resulting_version: int,
    request_id: str | None,
    payload: Mapping[str, int | str | bool | None],
) -> PublicEvent:
    return PublicEvent(
        game_id=game_id,
        event_id=str(uuid5(NAMESPACE_URL, f"dracula:{game_id}:{sequence}")),
        sequence=sequence,
        event_type=event_type,  # type: ignore[arg-type]
        occurred_at=occurred_at,
        prior_version=prior_version,
        resulting_version=resulting_version,
        request_id=request_id,
        payload=dict(payload),
    )


def move_id_for(session: GameSession, hand_slot: int, position: int, card_id: str) -> str:
    message = (
        f"move-v1\0{session.game_id}\0{session.version}\0{hand_slot}\0{position}\0{card_id}"
    ).encode("utf-8")
    digest = hmac.new(session.move_secret, message, hashlib.sha256).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


def resolve_human_move_id(session: GameSession, move_id: str) -> tuple[int, int] | None:
    state = session.engine_state
    if state.status is not EngineStatus.PLAYING or state.active_player is not session.human_role:
        return None
    for move in legal_moves(state, session.human_role):
        card_id = state.hands[session.human_role][move.hand_slot]
        assert card_id is not None
        candidate = move_id_for(session, move.hand_slot, move.global_grid_index, card_id)
        if hmac.compare_digest(candidate, move_id):
            return move.hand_slot, move.global_grid_index
    return None


def phase_for_state(state: EngineState, human_role: EnginePlayer) -> ResumablePhase:
    if state.status is EngineStatus.GAME_COMPLETE:
        outcome = derive_game_outcome(state)
        winner = (
            "tie"
            if outcome.winner is None
            else "human"
            if outcome.winner is human_role
            else "opponent"
        )
        return GameCompletePhase(outcome=winner)
    if state.status is EngineStatus.ROUND_COMPLETE:
        return ScoringPhase(round_number=state.round_number, next_step_index=0)
    if state.active_player is human_role:
        return HumanTurnPhase()
    return OpponentTurnPhase(status="ready")


def _player_score(values: PlayerValues[int], human_role: EnginePlayer) -> PlayerScore:
    return PlayerScore(human=values[human_role], opponent=values[other_player(human_role)])


def _played_move(move: EnginePlayedMove) -> PlayedMove:
    return PlayedMove(
        player=move.player.value,
        card_id=move.card_id,
        hand_slot=move.hand_slot,
        position=move.global_grid_index,
        turn_number=move.turn_number,
    )


def _multiplier_label(line: LineScore) -> str:
    cards = tuple(card_by_id(card_id) for card_id in line.cards)
    if line.multiplier_reason is MultiplierReason.VAMPIRE:
        return "Vampire"
    if line.multiplier_reason is MultiplierReason.NONE:
        return "No Multiplier"
    if line.multiplier_reason is MultiplierReason.SAME_COLOR:
        return f"3× {cards[0].color.value.title()}"
    suit_counts = {suit: sum(card.suit is suit for card in cards) for suit in Suit}
    suit = max(suit_counts, key=suit_counts.get)  # type: ignore[arg-type]
    prefix = "2×" if line.multiplier_reason is MultiplierReason.SUIT_PAIR else "3×"
    names = {Suit.CLUBS: "Clubs", Suit.DIAMONDS: "Diamonds", Suit.HEARTS: "Hearts", Suit.SPADES: "Spades"}
    return f"{prefix} {names[suit]}"


def _highlighted_cards(line: LineScore) -> tuple[str, ...]:
    cards = tuple(card_by_id(card_id) for card_id in line.cards)
    if line.multiplier_reason is MultiplierReason.NONE:
        return ()
    if line.multiplier_reason is MultiplierReason.VAMPIRE:
        return tuple(card.card_id for card in cards if card.is_vampire)
    if line.multiplier_reason is MultiplierReason.SUIT_PAIR:
        counts = {suit: sum(card.suit is suit for card in cards) for suit in Suit}
        repeated = max(counts, key=counts.get)  # type: ignore[arg-type]
        return tuple(card.card_id for card in cards if card.suit is repeated)
    return line.cards


def _api_line(line: LineScore) -> ApiLineScore:
    return ApiLineScore(
        direction=line.orientation.value,
        index=line.line_index,
        card_ids=line.cards,
        base_value=line.base_value,
        multiplier=line.multiplier,  # type: ignore[arg-type]
        multiplier_reason=line.multiplier_reason.value,
        multiplier_label=_multiplier_label(line),
        highlighted_card_ids=_highlighted_cards(line),
        total=line.total,
    )


def _scoring_sequence(
    result: EngineRoundResult,
    human_role: EnginePlayer,
    previous_totals: PlayerValues[int],
) -> tuple[ScoringStep, ...]:
    steps: list[ScoringStep] = []
    for player in (result.dealer, other_player(result.dealer)):
        lines = result.line_scores[player]
        ranked_indices = sorted(range(3), key=lambda index: (-lines[index].total, index))
        rank_by_index = {line_index: rank + 1 for rank, line_index in enumerate(ranked_indices)}
        for line_index, line in enumerate(lines):
            steps.append(
                ScoringStep(
                    kind="score_line",
                    player=player.value,
                    line=_api_line(line),
                    details={
                        "value_1": line.values[0],
                        "value_2": line.values[1],
                        "value_3": line.values[2],
                        "base_value": line.base_value,
                        "multiplier": line.multiplier,
                        "total": line.total,
                        "rank": rank_by_index[line_index],
                    },
                )
            )
    human_totals = sorted(
        (line.total for line in result.line_scores[human_role]), reverse=True
    )
    opponent = other_player(human_role)
    opponent_totals = sorted(
        (line.total for line in result.line_scores[opponent]), reverse=True
    )
    selected_rank = 2
    for rank in range(3):
        tied = human_totals[rank] == opponent_totals[rank]
        steps.append(
            ScoringStep(
                kind="compare_candidates",
                player=None,
                line=None,
                details={
                    "rank": rank + 1,
                    "human_score": human_totals[rank],
                    "opponent_score": opponent_totals[rank],
                    "tied": tied,
                },
            )
        )
        if rank < 2 and not tied:
            selected_rank = rank
            break
    steps.append(
        ScoringStep(
            kind="select_round_score",
            player=None,
            line=None,
            details={
                "rank": selected_rank + 1,
                "human_score": result.round_scores[human_role],
                "opponent_score": result.round_scores[opponent],
            },
        )
    )
    steps.append(
        ScoringStep(
            kind="update_total",
            player=None,
            line=None,
            details={
                "human_previous": previous_totals[human_role],
                "human_round": result.round_scores[human_role],
                "human_total": previous_totals[human_role] + result.round_scores[human_role],
                "opponent_previous": previous_totals[opponent],
                "opponent_round": result.round_scores[opponent],
                "opponent_total": previous_totals[opponent] + result.round_scores[opponent],
            },
        )
    )
    return tuple(steps)


def round_record(
    result: EngineRoundResult,
    human_role: EnginePlayer,
    previous_totals: PlayerValues[int],
) -> RoundRecord:
    line_order = (result.dealer, other_player(result.dealer))
    return RoundRecord(
        round_number=result.round_number,
        dealer=result.dealer.value,
        coffin=result.coffin,
        moves=tuple(_played_move(move) for move in result.moves),
        line_scores=tuple(
            _api_line(line) for player in line_order for line in result.line_scores[player]
        ),
        scoring_sequence=_scoring_sequence(result, human_role, previous_totals),
        round_scores=_player_score(result.round_scores, human_role),
    )


def project_human_game(session: GameSession, *, narration_enabled: bool = False) -> HumanGameView:
    validate_session(session)
    state = session.engine_state
    running = PlayerValues(queen=0, king=0)
    completed: list[RoundRecord] = []
    for result in state.completed_rounds:
        completed.append(round_record(result, session.human_role, running))
        running = PlayerValues(
            queen=running.queen + result.round_scores.queen,
            king=running.king + result.round_scores.king,
        )
    pending = (
        None
        if state.pending_round_result is None
        else round_record(state.pending_round_result, session.human_role, running)
    )
    legal: list[LegalMove] = []
    if (
        state.status is EngineStatus.PLAYING
        and state.active_player is session.human_role
        and isinstance(session.phase, HumanTurnPhase)
    ):
        for move in legal_moves(state, session.human_role):
            card_id = state.hands[session.human_role][move.hand_slot]
            assert card_id is not None
            legal.append(
                LegalMove(
                    move_id=move_id_for(
                        session, move.hand_slot, move.global_grid_index, card_id
                    ),
                    card_id=card_id,
                    hand_slot=move.hand_slot,
                    position=move.global_grid_index,
                )
            )
    return HumanGameView(
        game_id=session.game_id,
        version=session.version,
        status=state.status.value,
        round_number=state.round_number,
        turn_number=len(state.current_round_moves),
        dealer=state.dealer.value,
        active_player=None if state.active_player is None else state.active_player.value,
        human_role=session.human_role.value,
        opponent_role=other_player(session.human_role).value,
        coffin=state.coffin,
        current_round_moves=tuple(_played_move(move) for move in state.current_round_moves),
        pending_round_result=pending,
        completed_rounds=tuple(completed),
        total_scores=_player_score(state.total_scores, session.human_role),
        phase=session.phase,
        latest_event_sequence=len(session.events),
        narration_enabled=narration_enabled,
        human_hand=state.hands[session.human_role],
        legal_moves=tuple(legal),
        events=session.events,
    )

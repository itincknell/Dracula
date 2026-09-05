"""Represent trusted local gameplay sessions around private engine state.

This local-only compatibility layer defines and serializes private session,
policy, idempotency, and turn-claim records for SQLite-backed play. Public view
projection is owned separately by :mod:`dracula.api.local_projection`.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import re
import struct
from dataclasses import dataclass
from typing import Any, Mapping
from uuid import UUID

from pydantic import TypeAdapter

from dracula.api.contracts import (
    PublicEvent,
    ResumablePhase,
)
from dracula.api.policy import HIDDEN_STATE_BYTES
from dracula.engine import (
    EnginePlayer,
    EngineState,
    canonical_state_data,
    validate_state,
)
from dracula.engine_serialization import engine_state_from_data

SESSION_SCHEMA_VERSION = "dracula-game-session-v1"
_PHASE_ADAPTER = TypeAdapter(ResumablePhase)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class PolicySession:
    """Controller identity and opaque policy state pinned to a local game."""

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
    """Persisted ownership of one local opponent inference transaction."""

    job_id: str
    request_id: str
    request_hash: str
    expected_version: int


@dataclass(frozen=True, slots=True)
class IdempotencyRecord:
    """Exact response sealed for one local request ID and body digest."""

    request_id: str
    request_hash: str
    status_code: int
    response_json: str


@dataclass(frozen=True, slots=True)
class GameSession:
    """Complete private state and transaction metadata for local persistence."""

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


def validate_hidden_state(value: bytes) -> None:
    """Validate the retained fixed-width recurrent policy state."""

    if not isinstance(value, bytes) or len(value) != HIDDEN_STATE_BYTES:
        raise ValueError(f"policy hidden state must contain {HIDDEN_STATE_BYTES} bytes")
    if not all(math.isfinite(item) for item in struct.unpack("<128f", value)):
        raise ValueError("policy hidden state contains a non-finite value")


def validate_session(session: GameSession) -> None:
    """Verify private engine, policy, event, and idempotency consistency."""

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


def session_core_data(session: GameSession) -> dict[str, Any]:
    """Encode private state explicitly; repositories store events and requests separately."""

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
    """Reconstruct and validate a local session from normalized storage records."""

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
    """Hash canonical request content excluding its idempotency identifier."""

    encoded = json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def response_record(
    request_id: str, request_hash: str, status_code: int, body: Mapping[str, Any]
) -> IdempotencyRecord:
    """Seal one canonical JSON response for exact local request replay."""

    return IdempotencyRecord(
        request_id=request_id,
        request_hash=request_hash,
        status_code=status_code,
        response_json=json.dumps(body, sort_keys=True, separators=(",", ":")),
    )


def response_body(record: IdempotencyRecord) -> dict[str, Any]:
    """Decode a previously sealed local idempotency response."""

    value = json.loads(record.response_json)
    if not isinstance(value, dict):
        raise ValueError("persisted idempotency response is not an object")
    return value

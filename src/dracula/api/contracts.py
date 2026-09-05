"""Define browser-facing values shared by Dracula API projections.

These Pydantic models describe public cards, scores, phases, and local gameplay
messages. They deliberately exclude authoritative engine and model internals.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

Player = Literal["queen", "king"]
GameStatus = Literal["playing", "round_complete", "game_complete"]
CardSlot = tuple[str | None, str | None, str | None, str | None]
CoffinSlots = tuple[
    str | None,
    str | None,
    str | None,
    str | None,
    str | None,
    str | None,
    str | None,
    str | None,
    str | None,
]


class ContractModel(BaseModel):
    """Reject fields that are not part of the public wire contract."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class HealthResponse(ContractModel):
    """Local API health and configured dependency status."""

    status: Literal["ok"] = "ok"
    narration_enabled: bool


class PlayerScore(ContractModel):
    """One score pair expressed as human and opponent values."""

    human: int = Field(ge=0)
    opponent: int = Field(ge=0)


class PlayedMove(ContractModel):
    """Public accepted placement including its original hand slot."""

    player: Player
    card_id: str
    hand_slot: int = Field(ge=0, le=3)
    position: int = Field(ge=0, le=8)
    turn_number: int = Field(ge=1, le=8)


class LineScore(ContractModel):
    """Browser-safe arithmetic and highlighting facts for one scored line."""

    direction: Literal["row", "column"]
    index: int = Field(ge=0, le=2)
    card_ids: tuple[str, str, str]
    base_value: int = Field(ge=0)
    multiplier: Literal[0, 1, 2, 3, 5]
    multiplier_reason: Literal[
        "none",
        "suit_pair",
        "same_color",
        "same_suit",
        "vampire",
    ]
    multiplier_label: str
    highlighted_card_ids: tuple[str, ...]
    total: int = Field(ge=0)


class ScoringStep(ContractModel):
    """One deterministic instruction in the round-scoring animation."""

    kind: Literal[
        "score_line",
        "compare_candidates",
        "select_round_score",
        "update_total",
    ]
    player: Player | None
    line: LineScore | None
    details: dict[str, int | str | bool]


class RoundRecord(ContractModel):
    """Public cards, moves, calculations, and result for one finished round."""

    round_number: int = Field(ge=1, le=6)
    dealer: Player
    coffin: tuple[str, str, str, str, str, str, str, str, str]
    moves: tuple[PlayedMove, ...]
    line_scores: tuple[LineScore, ...]
    scoring_sequence: tuple[ScoringStep, ...]
    round_scores: PlayerScore


class LegalMove(ContractModel):
    """Opaque local-session move token and its visible placement details."""

    move_id: str
    card_id: str
    hand_slot: int = Field(ge=0, le=3)
    position: int = Field(ge=0, le=8)


class PublicEvent(ContractModel):
    """Append-only local gameplay event with contiguous sequence identity."""

    game_id: UUID
    event_id: str
    sequence: int = Field(ge=1)
    event_type: Literal[
        "game_created",
        "round_started",
        "move_accepted",
        "row_score_ready",
        "column_score_ready",
        "round_completed",
        "game_completed",
    ]
    occurred_at: datetime
    prior_version: int = Field(ge=0)
    resulting_version: int = Field(ge=0)
    request_id: str | None
    payload: dict[str, int | str | bool | None]


class HumanTurnPhase(ContractModel):
    """Phase in which the browser may submit one human placement."""

    kind: Literal["human_turn"] = "human_turn"


class OpponentTurnPhase(ContractModel):
    """Ready, pending, or failed local opponent-turn phase."""

    kind: Literal["opponent_turn"] = "opponent_turn"
    status: Literal["ready", "pending", "failed"]
    job_id: str | None = None
    retryable: bool = True


class NarrationPhase(ContractModel):
    """Local phase exposing narration generated for the current game state."""

    kind: Literal["narration"] = "narration"
    status: Literal["ready", "pending", "failed"]
    event_id: str
    narration_id: str | None = None
    required: bool


class ScoringPhase(ContractModel):
    """Phase identifying the completed round awaiting score presentation."""

    kind: Literal["scoring"] = "scoring"
    round_number: int = Field(ge=1, le=6)
    next_step_index: int = Field(ge=0)
    pending_narration_id: str | None = None


class RoundAdvancePhase(ContractModel):
    """Local phase awaiting acknowledgement of a scored round."""

    kind: Literal["round_advance"] = "round_advance"
    round_number: int = Field(ge=1, le=6)


class GameCompletePhase(ContractModel):
    """Terminal phase carrying only the public game outcome."""

    kind: Literal["game_complete"] = "game_complete"
    outcome: Literal["human", "opponent", "tie"]


ResumablePhase = Annotated[
    HumanTurnPhase
    | OpponentTurnPhase
    | NarrationPhase
    | ScoringPhase
    | RoundAdvancePhase
    | GameCompletePhase,
    Field(discriminator="kind"),
]


class PublicGameView(ContractModel):
    """Shared public lifecycle, board, score, and phase fields."""

    game_id: UUID
    version: int = Field(ge=0)
    status: GameStatus
    round_number: int = Field(ge=1, le=6)
    turn_number: int = Field(ge=0, le=8)
    dealer: Player
    active_player: Player | None
    human_role: Player
    opponent_role: Player
    coffin: CoffinSlots
    current_round_moves: tuple[PlayedMove, ...]
    pending_round_result: RoundRecord | None
    completed_rounds: tuple[RoundRecord, ...]
    total_scores: PlayerScore
    phase: ResumablePhase
    latest_event_sequence: int = Field(ge=0)
    narration_enabled: bool


class HumanGameView(PublicGameView):
    """Local-session game view extended with human hand and legal moves."""

    human_hand: CardSlot
    legal_moves: tuple[LegalMove, ...]
    events: tuple[PublicEvent, ...]


class EventsResponse(ContractModel):
    """Local append-only events after a requested sequence number."""

    events: tuple[PublicEvent, ...]
    latest_sequence: int = Field(ge=0)


class CreateGameRequest(ContractModel):
    """Idempotent local request selecting role and optional game seed."""

    human_role: Player
    request_id: UUID
    seed: str | None = None


class MoveRequest(ContractModel):
    """Idempotent local placement request using an opaque legal-move token."""

    move_id: str
    expected_version: int = Field(ge=0)
    request_id: UUID


class VersionedMutationRequest(ContractModel):
    """Idempotent local mutation bound to the caller's current game version."""

    expected_version: int = Field(ge=0)
    request_id: UUID


class ApiErrorResponse(ContractModel):
    """Local API error with optional authoritative recovery view."""

    code: Literal[
        "not_found",
        "stale_version",
        "wrong_turn",
        "wrong_phase",
        "already_advanced",
        "invalid_move",
        "request_id_conflict",
        "rate_limited",
        "dependency_unavailable",
        "validation_error",
    ]
    message: str
    retryable: bool
    current_version: int | None = Field(default=None, ge=0)
    current_game: HumanGameView | None = None

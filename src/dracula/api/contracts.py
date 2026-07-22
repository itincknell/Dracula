"""Versioned public API contracts for browser gameplay."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

API_VERSION = "dracula-api-v1"
HEALTH_SCHEMA_VERSION = "dracula-health-v1"
GAME_VIEW_SCHEMA_VERSION = "dracula-human-game-view-v1"
EVENT_SCHEMA_VERSION = "dracula-public-event-v1"
ERROR_SCHEMA_VERSION = "dracula-error-v1"

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
    schema_version: Literal["dracula-health-v1"] = HEALTH_SCHEMA_VERSION
    api_version: Literal["dracula-api-v1"] = API_VERSION
    status: Literal["ok"] = "ok"
    narration_enabled: bool


class PlayerScore(ContractModel):
    human: int = Field(ge=0)
    opponent: int = Field(ge=0)


class PlayedMove(ContractModel):
    player: Player
    card_id: str
    hand_slot: int = Field(ge=0, le=3)
    position: int = Field(ge=0, le=8)
    turn_number: int = Field(ge=1, le=8)


class LineScore(ContractModel):
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
    round_number: int = Field(ge=1, le=6)
    dealer: Player
    coffin: tuple[str, str, str, str, str, str, str, str, str]
    moves: tuple[PlayedMove, ...]
    line_scores: tuple[LineScore, ...]
    scoring_sequence: tuple[ScoringStep, ...]
    round_scores: PlayerScore


class LegalMove(ContractModel):
    move_id: str
    card_id: str
    hand_slot: int = Field(ge=0, le=3)
    position: int = Field(ge=0, le=8)


class PublicEvent(ContractModel):
    schema_version: Literal["dracula-public-event-v1"] = EVENT_SCHEMA_VERSION
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
    kind: Literal["human_turn"] = "human_turn"


class OpponentTurnPhase(ContractModel):
    kind: Literal["opponent_turn"] = "opponent_turn"
    status: Literal["ready", "pending", "failed"]
    job_id: str | None = None
    retryable: bool = True


class NarrationPhase(ContractModel):
    kind: Literal["narration"] = "narration"
    status: Literal["ready", "pending", "failed"]
    event_id: str
    narration_id: str | None = None
    required: bool


class ScoringPhase(ContractModel):
    kind: Literal["scoring"] = "scoring"
    round_number: int = Field(ge=1, le=6)
    next_step_index: int = Field(ge=0)
    pending_narration_id: str | None = None


class RoundAdvancePhase(ContractModel):
    kind: Literal["round_advance"] = "round_advance"
    round_number: int = Field(ge=1, le=6)


class GameCompletePhase(ContractModel):
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
    schema_version: Literal["dracula-human-game-view-v1"] = GAME_VIEW_SCHEMA_VERSION
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
    human_hand: CardSlot
    legal_moves: tuple[LegalMove, ...]
    events: tuple[PublicEvent, ...]


class EventsResponse(ContractModel):
    events: tuple[PublicEvent, ...]
    latest_sequence: int = Field(ge=0)


class CreateGameRequest(ContractModel):
    human_role: Player
    request_id: UUID
    seed: str | None = None

    @field_validator("seed")
    @classmethod
    def reject_seed_separator(cls, value: str | None) -> str | None:
        if value is not None and "\0" in value:
            raise ValueError("seed cannot contain NUL")
        return value


class MoveRequest(ContractModel):
    move_id: str
    expected_version: int = Field(ge=0)
    request_id: UUID


class VersionedMutationRequest(ContractModel):
    expected_version: int = Field(ge=0)
    request_id: UUID


class ApiErrorResponse(ContractModel):
    schema_version: Literal["dracula-error-v1"] = ERROR_SCHEMA_VERSION
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

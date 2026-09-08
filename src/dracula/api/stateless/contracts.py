"""Define the JSON exchanged by the stateless production gameplay API.

The browser carries a visible initial seed and ordered accepted-command history.
Requests describe commands rather than computed state, and responses expose only
the recovery envelope plus the public game projection. Each class is a Pydantic
model: FastAPI uses its type annotations and ``Field`` constraints to validate
incoming JSON before calling a route handler.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, model_validator

from dracula.api.contracts import (
    CardSlot,
    CoffinSlots,
    ContractModel,
    GameCompletePhase,
    GameStatus,
    HumanTurnPhase,
    LineScore,
    Player,
    PlayerScore,
    ScoringPhase,
    ScoringStep,
)

MAX_GAME_SEED_LENGTH = 256
# A complete game contains one role choice, 24 human placements, and six round
# advances; advancing round six changes the engine to its final game state.
MAX_ACCEPTED_COMMANDS = 31


class SelectRoleCommand(ContractModel):
    """First history command fixing the human scoring role for all rounds."""

    type: Literal["select_role"] = "select_role"
    human_role: Player


class PlaceCardCommand(ContractModel):
    """Human request to place one current hand slot at one coffin position."""

    type: Literal["place"] = "place"
    hand_slot: int = Field(ge=0, le=3)
    position: int = Field(ge=0, le=8)


class AdvanceRoundCommand(ContractModel):
    """Human acknowledgement that advances a scored round."""

    type: Literal["advance_round"] = "advance_round"


# The JSON ``type`` key is the discriminator: it tells Pydantic which command
# class must validate the remaining fields.
AcceptedGameCommand = Annotated[
    SelectRoleCommand | PlaceCardCommand | AdvanceRoundCommand,
    Field(discriminator="type"),
]
RequestedGameCommand = Annotated[
    PlaceCardCommand | AdvanceRoundCommand,
    Field(discriminator="type"),
]


class RecoveryEnvelope(ContractModel):
    """Authoritative browser-held seed and ordered accepted-command history."""

    seed: str = Field(min_length=1, max_length=MAX_GAME_SEED_LENGTH)
    history: tuple[AcceptedGameCommand, ...] = Field(
        min_length=1,
        max_length=MAX_ACCEPTED_COMMANDS,
    )

    @model_validator(mode="after")
    def require_one_leading_role(self) -> RecoveryEnvelope:
        """Check command ordering after Pydantic has built every command model."""

        # ``mode="after"`` means ``self.history`` already contains typed
        # SelectRoleCommand, PlaceCardCommand, and AdvanceRoundCommand objects.
        if not isinstance(self.history[0], SelectRoleCommand):
            raise ValueError("history must begin with role selection")
        if any(isinstance(command, SelectRoleCommand) for command in self.history[1:]):
            raise ValueError("role selection may appear only once")
        return self


class StartStatelessGameRequest(ContractModel):
    """New-game request with an optional caller-visible deterministic seed."""

    human_role: Player
    seed: str | None = Field(
        default=None,
        min_length=1,
        max_length=MAX_GAME_SEED_LENGTH,
    )


class ApplyStatelessCommandRequest(ContractModel):
    """Current recovery envelope plus exactly one proposed next command."""

    envelope: RecoveryEnvelope
    command: RequestedGameCommand


class ResumeStatelessGameRequest(ContractModel):
    """Request to reconstruct a game without adding a command."""

    envelope: RecoveryEnvelope


NarrationCueType = Literal["opening", "round_transition", "final_result"]


class NarrationRequest(ContractModel):
    """Separate narration cue request grounded by replaying the envelope."""

    envelope: RecoveryEnvelope
    cue_type: NarrationCueType


class NarrationResponse(ContractModel):
    """Ready plain text or an explicit empty unavailable state."""

    cue_type: NarrationCueType
    status: Literal["ready", "unavailable"]
    text: str | None = Field(default=None, max_length=400)

    @model_validator(mode="after")
    def require_text_only_when_ready(self) -> NarrationResponse:
        """Keep the status and optional text fields consistent."""

        if self.status == "ready" and not self.text:
            raise ValueError("ready narration requires text")
        if self.status == "unavailable" and self.text is not None:
            raise ValueError("unavailable narration cannot contain fallback text")
        return self


class StatelessLegalMove(ContractModel):
    """Browser-safe human placement including the visible card identity."""

    card_id: str
    hand_slot: int = Field(ge=0, le=3)
    position: int = Field(ge=0, le=8)


class VisiblePlayedMove(ContractModel):
    """Accepted public placement with no private original hand slot."""

    player: Player
    card_id: str
    position: int = Field(ge=0, le=8)
    turn_number: int = Field(ge=1, le=8)


class StatelessRoundRecord(ContractModel):
    """Public scoring and animation facts for one completed round."""

    round_number: int = Field(ge=1, le=6)
    dealer: Player
    coffin: tuple[str, str, str, str, str, str, str, str, str]
    moves: tuple[VisiblePlayedMove, ...]
    line_scores: tuple[LineScore, ...]
    scoring_sequence: tuple[ScoringStep, ...]
    round_scores: PlayerScore


# Stateless responses describe only phases derived from the engine. Narration
# and the pause before advancing a round are frontend presentation states.
StatelessPhase = Annotated[
    HumanTurnPhase | ScoringPhase | GameCompletePhase,
    Field(discriminator="kind"),
]


class StatelessHumanGameView(ContractModel):
    """Complete browser-safe projection of the reconstructed game.

    This is a view, not the authoritative engine state. It intentionally omits
    the opponent hand, stock, seed, policy input, and policy output.
    """

    status: GameStatus
    round_number: int = Field(ge=1, le=6)
    turn_number: int = Field(ge=0, le=8)
    dealer: Player
    active_player: Player | None
    human_role: Player
    opponent_role: Player
    coffin: CoffinSlots
    current_round_moves: tuple[VisiblePlayedMove, ...]
    pending_round_result: StatelessRoundRecord | None
    completed_rounds: tuple[StatelessRoundRecord, ...]
    total_scores: PlayerScore
    phase: StatelessPhase
    human_hand: CardSlot
    legal_moves: tuple[StatelessLegalMove, ...]


class StatelessGameResponse(ContractModel):
    """Updated recovery envelope paired with its authoritative public view."""

    envelope: RecoveryEnvelope
    game: StatelessHumanGameView


class StatelessHealthResponse(ContractModel):
    """Deployment health without process, model, or private-game details."""

    status: Literal["ok"] = "ok"
    opponent_configured: bool
    narration_enabled: bool


StatelessErrorCode = Literal[
    "validation_error",
    "invalid_history",
    "invalid_command",
    "wrong_turn",
    "wrong_phase",
    "ineligible_cue",
    "dependency_unavailable",
]


class StatelessApiErrorResponse(ContractModel):
    """Closed public error categories safe for browser recovery handling."""

    code: StatelessErrorCode
    message: str
    retryable: bool = False

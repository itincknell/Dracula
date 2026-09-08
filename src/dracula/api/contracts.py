"""Define browser-facing score and lifecycle values for stateless gameplay.

These models are shared by projection and transport code. They contain only
public game facts; request envelopes and complete game views live in the
stateless API package.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

# ``Literal`` restricts a string field to the listed wire values. These aliases
# describe JSON shapes; they do not create new runtime classes.
Player = Literal["queen", "king"]
GameStatus = Literal["playing", "round_complete", "game_complete"]

# Fixed-length tuples let Pydantic reject a board or hand with the wrong number
# of positions while preserving ``null`` for each currently empty position.
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


@dataclass(frozen=True, slots=True)
class ServiceResponse:
    """Carry a service result without importing FastAPI into service code.

    Route modules convert this small value into a framework ``JSONResponse``.
    Keeping that conversion at the edge lets gameplay services be tested
    without constructing an HTTP request.
    """

    status_code: int
    body: dict[str, Any]


class ContractModel(BaseModel):
    """Common Pydantic behavior for JSON accepted from or sent to a browser.

    ``extra="forbid"`` rejects unexpected JSON keys instead of ignoring them.
    ``frozen=True`` prevents code from changing a validated model afterward.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)


class PlayerScore(ContractModel):
    """One score pair expressed as human and opponent values."""

    human: int = Field(ge=0)
    opponent: int = Field(ge=0)


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
    """One deterministic instruction in the round-scoring animation.

    ``kind`` tells the frontend how to interpret ``line`` and ``details``.
    The details mapping contains only JSON primitive values, never engine data.
    """

    kind: Literal[
        "score_line",
        "compare_candidates",
        "select_round_score",
        "update_total",
    ]
    player: Player | None
    line: LineScore | None
    details: dict[str, int | str | bool]


class HumanTurnPhase(ContractModel):
    """Phase in which the browser may submit one human placement."""

    kind: Literal["human_turn"] = "human_turn"


class ScoringPhase(ContractModel):
    """Phase identifying the completed round awaiting score presentation."""

    kind: Literal["scoring"] = "scoring"
    round_number: int = Field(ge=1, le=6)


class GameCompletePhase(ContractModel):
    """Terminal phase carrying only the public game outcome."""

    kind: Literal["game_complete"] = "game_complete"
    outcome: Literal["human", "opponent", "tie"]

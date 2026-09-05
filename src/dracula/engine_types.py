"""Define the engine's immutable domain values and lifecycle vocabulary.

These types describe players, moves, scores, complete private game states, and
typed rule failures. They contain structure rather than transition logic.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Generic, Sequence, TypeVar

# These identifiers participate in persisted artifacts and deterministic seed
# derivation. Changing them is a compatibility change, not a naming cleanup.
ENGINE_VERSION = "dracula-engine-v1"
RULES_VERSION = "dracula-rules-v1"
SHUFFLE_NAMESPACE = "dracula-engine-shuffle-v1"
INITIAL_DEALER_NAMESPACE = "dracula-engine-initial-dealer-v1"

# Coffin indexes are zero-based and row-major; index 4 is the center.
HAND_SIZE = 4
COFFIN_SIZE = 9
CENTER_GRID_INDEX = 4
MOVES_PER_ROUND = 8
ROUNDS_PER_GAME = 6


class EnginePlayer(StrEnum):
    """Fixed scoring role: Queen owns rows and King owns columns."""

    QUEEN = "queen"
    KING = "king"


class EngineStatus(StrEnum):
    """Lifecycle stage controlling which engine transitions are permitted."""

    PLAYING = "playing"
    ROUND_COMPLETE = "round_complete"
    GAME_COMPLETE = "game_complete"


class LineOrientation(StrEnum):
    """Direction used to select a card's Queen or King value."""

    ROW = "row"
    COLUMN = "column"


class MultiplierReason(StrEnum):
    """Rules condition responsible for a line's single applied multiplier."""

    VAMPIRE = "vampire"
    NONE = "none"
    SUIT_PAIR = "suit_pair"
    SAME_COLOR = "same_color"
    SAME_SUIT = "same_suit"


class GameOutcomeReason(StrEnum):
    """Score tier that determined the final winner or tie."""

    TOTAL_SCORE = "total_score"
    SIXTH_ROUND_SCORE = "sixth_round_score"
    TIE = "tie"


class RuleViolation(Exception):
    """Base class for rejected state transitions and rule inputs."""


class MalformedState(RuleViolation):
    """The supplied state violates a structural or cross-field invariant."""


class WrongActivePlayer(RuleViolation):
    """A move or query names someone other than the current actor."""


class UnavailableHandSlot(RuleViolation):
    """A move names an out-of-range slot or a card already played."""


class InvalidGridIndex(RuleViolation):
    """A move names a position outside the nine-cell coffin."""


class OccupiedDestination(RuleViolation):
    """A move targets a coffin position that already contains a card."""


class NonAdjacentDestination(RuleViolation):
    """A move targets an empty position without an edge-sharing neighbor."""


class InvalidLifecycleTransition(RuleViolation):
    """An operation is unavailable in the state's current lifecycle stage."""


T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class PlayerValues(Generic[T]):
    """Immutable Queen/King pair whose two values share one declared type."""

    queen: T
    king: T

    def __getitem__(self, player: EnginePlayer) -> T:
        """Return the value belonging to the requested scoring role."""

        player = EnginePlayer(player)
        return self.queen if player is EnginePlayer.QUEEN else self.king

    def updated(self, player: EnginePlayer, value: T) -> PlayerValues[T]:
        """Return a new pair with only the requested player's value replaced."""

        player = EnginePlayer(player)
        if player is EnginePlayer.QUEEN:
            return replace(self, queen=value)
        return replace(self, king=value)


# A hand keeps its four original slots; ``None`` marks a card already played.
Hand = tuple[str | None, str | None, str | None, str | None]
# A coffin keeps all nine positions; ``None`` marks a currently empty position.
Coffin = tuple[
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


def other_player(player: EnginePlayer) -> EnginePlayer:
    """Return the scoring role opposite ``player``."""

    player = EnginePlayer(player)
    return EnginePlayer.KING if player is EnginePlayer.QUEEN else EnginePlayer.QUEEN


def orthogonally_adjacent(first: int, second: int) -> bool:
    """Return whether two zero-based row-major coffin positions share an edge."""

    # For a three-column grid, divmod yields the row as quotient and column as
    # remainder. Edge-sharing cells are one total step apart across both axes.
    first_row, first_column = divmod(first, 3)
    second_row, second_column = divmod(second, 3)
    row_distance = abs(first_row - second_row)
    column_distance = abs(first_column - second_column)
    return row_distance + column_distance == 1


def empty_adjacent_grid_indices(
    coffin: Sequence[object | None],
) -> tuple[int, ...]:
    """Return empty coffin positions sharing an edge with an occupied position."""

    occupied = tuple(index for index, value in enumerate(coffin) if value is not None)
    return tuple(
        destination
        for destination, value in enumerate(coffin)
        if value is None
        and any(
            orthogonally_adjacent(destination, occupied_index)
            for occupied_index in occupied
        )
    )


@dataclass(frozen=True, slots=True)
class RoundDeal:
    """Private cards and remaining stock produced by one deterministic deal."""

    dealer: EnginePlayer
    queen_hand: tuple[str, ...]
    king_hand: tuple[str, ...]
    center_card: str
    remaining_stock: tuple[str, ...]

    @property
    def non_dealer(self) -> EnginePlayer:
        """Return the player who acts first in this round."""

        return other_player(self.dealer)

    def hand_for(self, player: EnginePlayer) -> tuple[str, ...]:
        """Return one player's four canonically ordered dealt cards."""

        player = EnginePlayer(player)
        return self.queen_hand if player is EnginePlayer.QUEEN else self.king_hand


@dataclass(frozen=True, slots=True)
class EngineMove:
    """Requested placement expressed against the player's current hand slots."""

    player: EnginePlayer
    hand_slot: int
    global_grid_index: int


@dataclass(frozen=True, slots=True)
class EnginePlayedMove:
    """Accepted placement recorded with its resolved card and turn number."""

    player: EnginePlayer
    card_id: str
    hand_slot: int
    global_grid_index: int
    turn_number: int


@dataclass(frozen=True, slots=True)
class LineScore:
    """Auditable components of one completed row or column score."""

    orientation: LineOrientation
    line_index: int
    cards: tuple[str, str, str]
    values: tuple[int, int, int]
    base_value: int
    multiplier: int
    multiplier_reason: MultiplierReason
    total: int


@dataclass(frozen=True, slots=True)
class EngineRoundResult:
    """Public result facts derived from one completed coffin."""

    round_number: int
    dealer: EnginePlayer
    coffin: tuple[str, str, str, str, str, str, str, str, str]
    moves: tuple[EnginePlayedMove, ...]
    line_scores: PlayerValues[tuple[LineScore, LineScore, LineScore]]
    round_scores: PlayerValues[int]


@dataclass(frozen=True, slots=True)
class EngineState:
    """Complete private game state; never a public gameplay projection.

    Played hand slots remain present as ``None``. A pending round is already
    included in ``total_scores`` but is not archived in ``completed_rounds``.
    """

    seed: str
    status: EngineStatus
    round_number: int
    dealer: EnginePlayer
    active_player: EnginePlayer | None
    stock: tuple[str, ...]
    hands: PlayerValues[Hand]
    coffin: Coffin
    current_round_moves: tuple[EnginePlayedMove, ...]
    pending_round_result: EngineRoundResult | None
    completed_rounds: tuple[EngineRoundResult, ...]
    total_scores: PlayerValues[int]


@dataclass(frozen=True, slots=True)
class SimulationEngineState(EngineState):
    """Engine state whose complete private deck was sampled from actor knowledge."""

    simulation_deck: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class EngineTransition:
    """Private before/after states and the accepted move joining them.

    ``state_fingerprint`` identifies the resulting complete private state.
    """

    previous_state: EngineState
    state: EngineState
    move: EngineMove
    played_move: EnginePlayedMove
    round_result: EngineRoundResult | None
    state_fingerprint: str


@dataclass(frozen=True, slots=True)
class GameOutcome:
    """Final winner and the score tier that decided the game."""

    winner: EnginePlayer | None
    reason: GameOutcomeReason
    total_scores: PlayerValues[int]
    sixth_round_scores: PlayerValues[int]

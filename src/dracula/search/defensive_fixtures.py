"""Versioned defensive acceptance positions with exact continuation evidence."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from dracula.engine import (
    EngineMove,
    EnginePlayer,
    EngineState,
    EngineStatus,
    advance_after_round,
    apply_move,
    create_game,
    legal_moves,
)
from dracula.randomness import Sha256CounterStream, derive_seed
from dracula.search.strategic_fixtures import RoundStage, round_stage

DEFENSIVE_FIXTURE_SCHEMA_VERSION = "dracula-defensive-fixtures-v1"
# The catalog was discovered from the already frozen replay stream so its public
# engine histories remain comparable with the version 1 strategic fixtures.
DEFENSIVE_FIXTURE_SETUP_NAMESPACE = "strategic-fixture-setup-v1"
DEFENSIVE_FIXTURE_TARGET_ROUND = 6


class DefensiveBehavior(StrEnum):
    AVOID_SUIT_ENABLEMENT = "avoid_suit_enablement"
    AVOID_COLOR_ENABLEMENT = "avoid_color_enablement"
    BLOCK_VISIBLE_MULTIPLIER = "block_visible_multiplier"
    AVOID_DANGEROUS_INTERSECTION = "avoid_dangerous_intersection"
    DEFENSIVE_VAMPIRE = "defensive_vampire"
    AVOID_VAMPIRE_DESTRUCTION = "avoid_vampire_destruction"
    DEFENSIVE_DENIAL_OVER_OFFENSE = "defensive_denial_over_offense"
    CONSTRUCTIVE_WHEN_BLOCK_VALUELESS = "constructive_when_block_valueless"
    OFFENSE_DEFENSE_CONFLICT = "offense_defense_conflict"
    FORCED_TRANSITION = "forced_transition"


@dataclass(frozen=True, slots=True)
class DefensiveFixture:
    fixture_id: str
    title: str
    behaviors: tuple[DefensiveBehavior, ...]
    game_seed: str
    accepted_moves: int
    player: EnginePlayer
    dealer: bool
    stage: RoundStage
    expected_action_indices: tuple[int, ...]
    exact_value: float | None
    exact_margin: float | None
    rationale: str


def _fixture_move(state: EngineState, game_seed: str, step: int) -> EngineMove:
    moves = legal_moves(state, state.active_player)
    stream = Sha256CounterStream(
        derive_seed(DEFENSIVE_FIXTURE_SETUP_NAMESPACE, game_seed, str(step))
    )
    return moves[stream.randbelow(len(moves))]


def replay_defensive_fixture(fixture: DefensiveFixture) -> EngineState:
    """Rebuild a position through the public engine using its fixture seed."""

    state = create_game(fixture.game_seed)
    step = 0
    while state.round_number < DEFENSIVE_FIXTURE_TARGET_ROUND:
        while state.status is EngineStatus.PLAYING:
            state = apply_move(
                state, _fixture_move(state, fixture.game_seed, step)
            ).state
            step += 1
        state = advance_after_round(state)
    for _ in range(fixture.accepted_moves):
        state = apply_move(
            state, _fixture_move(state, fixture.game_seed, step)
        ).state
        step += 1
    if state.status is not EngineStatus.PLAYING or state.active_player is None:
        raise ValueError(f"fixture {fixture.fixture_id} is not an active decision")
    if (
        state.active_player is not fixture.player
        or (state.active_player is state.dealer) is not fixture.dealer
        or round_stage(fixture.accepted_moves) is not fixture.stage
    ):
        raise ValueError(f"fixture {fixture.fixture_id} lifecycle metadata drifted")
    return state


DEFENSIVE_FIXTURES = (
    DefensiveFixture(
        "king-early-dealer-constructive",
        "King completes a strong color column when blocking has no value",
        (DefensiveBehavior.CONSTRUCTIVE_WHEN_BLOCK_VALUELESS,),
        "defense-early-00003",
        1,
        EnginePlayer.KING,
        True,
        RoundStage.EARLY,
        (19,),
        0.26666666666666666,
        0.22,
        "KD at position 1 completes the red KD-10D-QH column for 60 points; "
        "exact minimax rejects speculative denial elsewhere.",
    ),
    DefensiveFixture(
        "queen-early-dealer-safe-intersection",
        "Queen avoids the dangerous early shared intersections",
        (DefensiveBehavior.AVOID_DANGEROUS_INTERSECTION,),
        "defense-early-00008",
        1,
        EnginePlayer.QUEEN,
        True,
        RoundStage.EARLY,
        (28,),
        -0.04,
        0.013333333333333336,
        "JS at position 5 is the unique exact minimax action after the visible "
        "7C placement; every alternate intersection has a lower terminal floor.",
    ),
    DefensiveFixture(
        "queen-early-nondealer-avoid-suit",
        "Queen avoids giving the opponent an early suit pair",
        (DefensiveBehavior.AVOID_SUIT_ENABLEMENT,),
        "defense-early-00000",
        2,
        EnginePlayer.QUEEN,
        False,
        RoundStage.EARLY,
        (1, 9),
        -0.05333333333333334,
        0.04,
        "6C or QD at position 1 preserves the exact minimax value without "
        "placing 6H into the opponent's visible Heart construction.",
    ),
    DefensiveFixture(
        "king-early-nondealer-safe-intersection",
        "King avoids feeding either exposed scoring line",
        (DefensiveBehavior.AVOID_DANGEROUS_INTERSECTION,),
        "defense-early-00010",
        2,
        EnginePlayer.KING,
        False,
        RoundStage.EARLY,
        (4, 23),
        0.16,
        0.08666666666666667,
        "8C at position 7 or JD at position 8 retains the exact floor while "
        "the alternatives enter more valuable shared lines.",
    ),
    DefensiveFixture(
        "king-middle-avoid-suit",
        "King refuses to extend the visible Heart construction",
        (DefensiveBehavior.AVOID_SUIT_ENABLEMENT,),
        "defense-discovery-00000",
        4,
        EnginePlayer.KING,
        False,
        RoundStage.MIDDLE,
        (25,),
        -0.04,
        0.06,
        "QH at position 3 is the unique exact action; placements beside 8H "
        "create the suit pair that the opponent can exploit.",
    ),
    DefensiveFixture(
        "queen-middle-avoid-color",
        "Queen avoids extending a visible red construction",
        (DefensiveBehavior.AVOID_COLOR_ENABLEMENT,),
        "defense-discovery-00014",
        4,
        EnginePlayer.QUEEN,
        False,
        RoundStage.MIDDLE,
        (18,),
        -0.02666666666666667,
        0.08,
        "2S at position 2 avoids the red pair created by KD and preserves the "
        "best exact terminal floor.",
    ),
    DefensiveFixture(
        "queen-middle-defensive-vampire",
        "Queen zeros the opponent's live scoring column with a Vampire",
        (DefensiveBehavior.DEFENSIVE_VAMPIRE,),
        "defense-discovery-00006",
        5,
        EnginePlayer.QUEEN,
        True,
        RoundStage.MIDDLE,
        (31,),
        0.013333333333333334,
        0.13333333333333333,
        "V2 at position 8 zeros the live 8D-KD column and dominates the "
        "ordinary blocking card under exact continuation analysis.",
    ),
    DefensiveFixture(
        "queen-middle-avoid-vampire-destruction",
        "Queen invests only in a line the opponent cannot erase",
        (DefensiveBehavior.AVOID_VAMPIRE_DESTRUCTION,),
        "vamp-risk-00061",
        4,
        EnginePlayer.QUEEN,
        False,
        RoundStage.MIDDLE,
        (23,),
        0.14,
        0.10666666666666669,
        "10H at position 8 completes the safe 36-point row; the opponent's "
        "known remaining Vampire can erase each tempting open-line investment.",
    ),
    DefensiveFixture(
        "king-middle-denial-over-offense",
        "King chooses denial over a modest isolated improvement",
        (
            DefensiveBehavior.BLOCK_VISIBLE_MULTIPLIER,
            DefensiveBehavior.DEFENSIVE_DENIAL_OVER_OFFENSE,
            DefensiveBehavior.OFFENSE_DEFENSE_CONFLICT,
        ),
        "defensive-candidate-00005",
        4,
        EnginePlayer.KING,
        False,
        RoundStage.MIDDLE,
        (5,),
        0.12,
        0.09999999999999999,
        "6D at position 2 breaks the visible 4H-AH row while completing the "
        "54-point King column; lower-value construction elsewhere loses the floor.",
    ),
    DefensiveFixture(
        "queen-middle-constructive-over-empty-block",
        "Queen keeps constructive play when a nominal block has no value",
        (DefensiveBehavior.CONSTRUCTIVE_WHEN_BLOCK_VALUELESS,),
        "defensive-candidate-00017",
        5,
        EnginePlayer.QUEEN,
        True,
        RoundStage.MIDDLE,
        (14,),
        0.26666666666666666,
        0.44666666666666666,
        "7C at position 7 completes the dominant Club row; the apparent block "
        "with 3H gives up substantially more than it denies.",
    ),
    DefensiveFixture(
        "queen-late-block",
        "Queen uses the final choice to deny the stronger column",
        (DefensiveBehavior.BLOCK_VISIBLE_MULTIPLIER,),
        "defensive-candidate-00000",
        6,
        EnginePlayer.QUEEN,
        False,
        RoundStage.LATE,
        (7,),
        0.21333333333333335,
        0.013333333333333336,
        "3H at position 8 both completes the live Heart row and breaks the "
        "opponent's Spade column; exact terminal scores resolve the close conflict.",
    ),
    DefensiveFixture(
        "king-late-block",
        "King blocks the opponent's final Heart row",
        (DefensiveBehavior.BLOCK_VISIBLE_MULTIPLIER,),
        "defensive-candidate-00005",
        6,
        EnginePlayer.KING,
        False,
        RoundStage.LATE,
        (2,),
        0.06,
        0.17333333333333334,
        "6D at position 6 breaks the visible 5H-2H construction and dominates "
        "the only alternate placement under exact terminal scoring.",
    ),
    DefensiveFixture(
        "queen-late-dealer-forced",
        "Queen dealer's final placement is forced",
        (DefensiveBehavior.FORCED_TRANSITION,),
        "defense-discovery-00006",
        7,
        EnginePlayer.QUEEN,
        True,
        RoundStage.LATE,
        (29,),
        None,
        None,
        "The sole action is recorded for lifecycle coverage and is excluded "
        "from strategic pass rates.",
    ),
    DefensiveFixture(
        "king-late-dealer-forced",
        "King dealer's final placement is forced",
        (DefensiveBehavior.FORCED_TRANSITION,),
        "defense-discovery-00017",
        7,
        EnginePlayer.KING,
        True,
        RoundStage.LATE,
        (26,),
        None,
        None,
        "The sole action is recorded for lifecycle coverage and is excluded "
        "from strategic pass rates.",
    ),
)


__all__ = (
    "DEFENSIVE_FIXTURE_SCHEMA_VERSION",
    "DEFENSIVE_FIXTURES",
    "DefensiveBehavior",
    "DefensiveFixture",
    "replay_defensive_fixture",
)

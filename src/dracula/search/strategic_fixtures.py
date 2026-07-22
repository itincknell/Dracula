"""Versioned strategic positions with reproducible engine histories."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from dracula.bridge import action_index_for_move, build_policy_turn_context
from dracula.cards import Card, card_by_id
from dracula.engine import (
    EngineMove,
    EnginePlayer,
    EngineState,
    EngineStatus,
    LineOrientation,
    LineScore,
    MultiplierReason,
    advance_after_round,
    apply_move,
    create_game,
    legal_moves,
    other_player,
    score_line,
)
from dracula.randomness import Sha256CounterStream, derive_seed

STRATEGIC_FIXTURE_SCHEMA_VERSION = "dracula-strategic-fixtures-v1"
FIXTURE_SETUP_NAMESPACE = "strategic-fixture-setup-v1"
FIXTURE_TARGET_ROUND = 6


class StrategicBehavior(StrEnum):
    SAME_SUIT_COMPLETION = "same_suit_completion"
    SAME_COLOR_COMPLETION = "same_color_completion"
    HIGH_FACE_VALUE = "high_face_value"
    BLOCK_VISIBLE_MULTIPLIER = "block_visible_multiplier"
    AVOID_VAMPIRE_LINE = "avoid_vampire_line"
    DEFENSIVE_VAMPIRE = "defensive_vampire"
    PRESERVE_VALUABLE_CARD = "preserve_valuable_card"
    EXPLOIT_OPPONENT_PLACEMENT = "exploit_opponent_placement"
    OFFENSE_DEFENSE_CONFLICT = "offense_defense_conflict"


class RoundStage(StrEnum):
    EARLY = "early"
    MIDDLE = "middle"
    LATE = "late"


class FixtureEvidence(StrEnum):
    EXHAUSTIVE = "exhaustive_continuations"
    CONVERGED_SEARCH = "converged_information_set_search"
    FORCED = "forced_engine_transition"


@dataclass(frozen=True, slots=True)
class StrategicFixture:
    fixture_id: str
    title: str
    behaviors: tuple[StrategicBehavior, ...]
    game_seed: str
    accepted_moves: int
    player: EnginePlayer
    dealer: bool
    stage: RoundStage
    expected_action_indices: tuple[int, ...]
    evidence: FixtureEvidence
    evidence_value: float | None
    evidence_margin: float | None
    rationale: str
    comparison_action_index: int | None = None


@dataclass(frozen=True, slots=True)
class ActionStrategicFeatures:
    action_index: int
    move: EngineMove
    card_id: str
    card: Card
    directional_value: int
    own_line: tuple[int, int, int]
    opponent_line: tuple[int, int, int]
    own_line_score: LineScore | None
    opponent_line_score: LineScore | None
    completes_same_suit: bool
    completes_same_color: bool
    blocks_visible_multiplier: bool
    uses_vampire_defensively: bool
    invests_in_vampire_line: bool
    exploits_opponent_placement: bool


def round_stage(accepted_moves: int) -> RoundStage:
    if type(accepted_moves) is not int or not 0 <= accepted_moves <= 7:
        raise ValueError("accepted moves must be between zero and seven")
    if accepted_moves <= 2:
        return RoundStage.EARLY
    if accepted_moves <= 5:
        return RoundStage.MIDDLE
    return RoundStage.LATE


def _fixture_move(state: EngineState, game_seed: str, step: int) -> EngineMove:
    moves = legal_moves(state, state.active_player)
    stream = Sha256CounterStream(
        derive_seed(FIXTURE_SETUP_NAMESPACE, game_seed, str(step))
    )
    return moves[stream.randbelow(len(moves))]


def replay_fixture(fixture: StrategicFixture) -> EngineState:
    """Rebuild one fixture entirely through public engine operations."""

    state = create_game(fixture.game_seed)
    step = 0
    while state.round_number < FIXTURE_TARGET_ROUND:
        while state.status is EngineStatus.PLAYING:
            state = apply_move(
                state, _fixture_move(state, fixture.game_seed, step)
            ).state
            step += 1
        state = advance_after_round(state)
    for _ in range(fixture.accepted_moves):
        state = apply_move(state, _fixture_move(state, fixture.game_seed, step)).state
        step += 1
    if state.status is not EngineStatus.PLAYING or state.active_player is None:
        raise ValueError(f"fixture {fixture.fixture_id} did not produce an active decision")
    if (
        state.active_player is not fixture.player
        or (state.active_player is state.dealer) is not fixture.dealer
        or round_stage(fixture.accepted_moves) is not fixture.stage
    ):
        raise ValueError(f"fixture {fixture.fixture_id} lifecycle metadata drifted")
    return state


def _lines(player: EnginePlayer) -> tuple[tuple[int, int, int], ...]:
    if player is EnginePlayer.QUEEN:
        return ((0, 1, 2), (3, 4, 5), (6, 7, 8))
    return ((0, 3, 6), (1, 4, 7), (2, 5, 8))


def _orientation(player: EnginePlayer) -> LineOrientation:
    return (
        LineOrientation.ROW
        if player is EnginePlayer.QUEEN
        else LineOrientation.COLUMN
    )


def _line_for_position(
    player: EnginePlayer, grid_index: int
) -> tuple[int, int, int]:
    return next(line for line in _lines(player) if grid_index in line)


def _completed_line_score(
    coffin: list[str | None], line: tuple[int, int, int], player: EnginePlayer
) -> LineScore | None:
    cards = tuple(coffin[index] for index in line)
    if any(card_id is None for card_id in cards):
        return None
    return score_line(cards, _orientation(player))  # type: ignore[arg-type]


def _directional_value(card: Card, player: EnginePlayer) -> int:
    return (
        card.horizontal_value
        if player is EnginePlayer.QUEEN
        else card.vertical_value
    )


def action_strategic_features(
    state: EngineState, move: EngineMove
) -> ActionStrategicFeatures:
    if move not in legal_moves(state, state.active_player):
        raise ValueError("strategic features require an exact engine-legal action")
    player = move.player
    opponent = other_player(player)
    card_id = state.hands[player][move.hand_slot]
    if card_id is None:
        raise ValueError("strategic action names an empty hand slot")
    card = card_by_id(card_id)
    own_line = _line_for_position(player, move.global_grid_index)
    opponent_line = _line_for_position(opponent, move.global_grid_index)
    coffin = list(state.coffin)
    coffin[move.global_grid_index] = card_id
    own_score = _completed_line_score(coffin, own_line, player)
    opponent_score = _completed_line_score(coffin, opponent_line, opponent)

    visible_pair = tuple(
        card_by_id(state.coffin[index])
        for index in opponent_line
        if state.coffin[index] is not None
    )
    opportunity_multiplier = 1
    if len(visible_pair) == 2:
        if visible_pair[0].suit == visible_pair[1].suit:
            opportunity_multiplier = 5
        elif visible_pair[0].color == visible_pair[1].color:
            opportunity_multiplier = 3
    blocks = (
        opportunity_multiplier > 1
        and opponent_score is not None
        and opponent_score.multiplier < opportunity_multiplier
    )
    opponent_positions = {
        played.global_grid_index
        for played in state.current_round_moves
        if played.player is opponent
    }
    return ActionStrategicFeatures(
        action_index=action_index_for_move(move, player),
        move=move,
        card_id=card_id,
        card=card,
        directional_value=_directional_value(card, player),
        own_line=own_line,
        opponent_line=opponent_line,
        own_line_score=own_score,
        opponent_line_score=opponent_score,
        completes_same_suit=(
            own_score is not None
            and own_score.multiplier_reason is MultiplierReason.SAME_SUIT
        ),
        completes_same_color=(
            own_score is not None
            and own_score.multiplier_reason is MultiplierReason.SAME_COLOR
        ),
        blocks_visible_multiplier=blocks and not card.is_vampire,
        uses_vampire_defensively=blocks and card.is_vampire,
        invests_in_vampire_line=any(
            card_id in {"V1", "V2"}
            for card_id in (state.coffin[index] for index in own_line)
        ),
        exploits_opponent_placement=(
            own_score is not None
            and own_score.multiplier > 1
            and any(index in opponent_positions for index in own_line)
        ),
    )


def fixture_action_features(
    fixture: StrategicFixture,
) -> dict[int, ActionStrategicFeatures]:
    state = replay_fixture(fixture)
    return {
        feature.action_index: feature
        for feature in (
            action_strategic_features(state, move)
            for move in legal_moves(state, state.active_player)
        )
    }


def validate_fixture_semantics(fixture: StrategicFixture) -> None:
    state = replay_fixture(fixture)
    context = build_policy_turn_context(state, state.active_player)
    legal = {
        index for index, move in enumerate(context.action_table) if move is not None
    }
    expected = set(fixture.expected_action_indices)
    if not expected or not expected <= legal:
        raise ValueError(f"fixture {fixture.fixture_id} expected actions are not legal")
    features = fixture_action_features(fixture)
    selected = [features[index] for index in fixture.expected_action_indices]

    for behavior in fixture.behaviors:
        if behavior is StrategicBehavior.SAME_SUIT_COMPLETION:
            valid = all(feature.completes_same_suit for feature in selected)
        elif behavior is StrategicBehavior.SAME_COLOR_COMPLETION:
            valid = all(feature.completes_same_color for feature in selected) and not any(
                feature.completes_same_suit for feature in features.values()
            )
        elif behavior is StrategicBehavior.HIGH_FACE_VALUE:
            valid = all(
                feature.own_line_score is not None
                and any(
                    other.move.global_grid_index == feature.move.global_grid_index
                    and other.own_line_score is not None
                    and other.own_line_score.multiplier_reason
                    is feature.own_line_score.multiplier_reason
                    and other.own_line_score.total < feature.own_line_score.total
                    for other in features.values()
                )
                for feature in selected
            )
        elif behavior is StrategicBehavior.BLOCK_VISIBLE_MULTIPLIER:
            valid = all(feature.blocks_visible_multiplier for feature in selected)
        elif behavior is StrategicBehavior.AVOID_VAMPIRE_LINE:
            valid = all(not feature.invests_in_vampire_line for feature in selected) and any(
                feature.invests_in_vampire_line for feature in features.values()
            )
        elif behavior is StrategicBehavior.DEFENSIVE_VAMPIRE:
            valid = all(feature.uses_vampire_defensively for feature in selected)
        elif behavior is StrategicBehavior.PRESERVE_VALUABLE_CARD:
            valid = all(
                any(
                    other.move.global_grid_index == feature.move.global_grid_index
                    and other.directional_value > feature.directional_value
                    for other in features.values()
                )
                for feature in selected
            )
        elif behavior is StrategicBehavior.EXPLOIT_OPPONENT_PLACEMENT:
            valid = all(feature.exploits_opponent_placement for feature in selected)
        else:
            offensive = any(
                feature.completes_same_suit or feature.completes_same_color
                for feature in features.values()
            )
            defensive = any(
                feature.blocks_visible_multiplier
                or feature.uses_vampire_defensively
                for feature in features.values()
            )
            valid = offensive and defensive
        if not valid:
            raise ValueError(
                f"fixture {fixture.fixture_id} does not exhibit {behavior.value}"
            )


STRATEGIC_FIXTURES = (
    StrategicFixture(
        "queen-early-nondealer-suit",
        "Queen completes a 135-point Club row",
        (StrategicBehavior.SAME_SUIT_COMPLETION,),
        "strategy-candidate-00006",
        2,
        EnginePlayer.QUEEN,
        False,
        RoundStage.EARLY,
        (3,),
        FixtureEvidence.CONVERGED_SEARCH,
        0.61747,
        None,
        "7C at position 3 completes 7C-10C-QC with the Queen value of 10.",
    ),
    StrategicFixture(
        "queen-early-dealer-suit",
        "Queen completes the available Club row while retaining flexibility",
        (StrategicBehavior.SAME_SUIT_COMPLETION,),
        "strategy-candidate-00119",
        1,
        EnginePlayer.QUEEN,
        True,
        RoundStage.EARLY,
        (3, 11),
        FixtureEvidence.CONVERGED_SEARCH,
        0.3658,
        None,
        "Either 9C or QC at position 3 completes a five-times Club row.",
    ),
    StrategicFixture(
        "king-early-nondealer-suit",
        "King completes an 80-point Club column",
        (StrategicBehavior.SAME_SUIT_COMPLETION,),
        "strategy-candidate-00009",
        2,
        EnginePlayer.KING,
        False,
        RoundStage.EARLY,
        (11,),
        FixtureEvidence.CONVERGED_SEARCH,
        0.1911,
        None,
        "KC at position 1 completes KC-6C-QC with the King value of 10.",
    ),
    StrategicFixture(
        "king-early-dealer-vampire",
        "King uses a Vampire against a visible Club row",
        (StrategicBehavior.DEFENSIVE_VAMPIRE,),
        "strategy-candidate-00006",
        1,
        EnginePlayer.KING,
        True,
        RoundStage.EARLY,
        (25,),
        FixtureEvidence.CONVERGED_SEARCH,
        0.17059,
        None,
        "V1 at position 3 zeros the Queen's visible 10C-QC construction.",
    ),
    StrategicFixture(
        "queen-middle-offense-defense",
        "Queen chooses construction where offense and denial conflict",
        (StrategicBehavior.OFFENSE_DEFENSE_CONFLICT,),
        "strategy-candidate-00006",
        4,
        EnginePlayer.QUEEN,
        False,
        RoundStage.MIDDLE,
        (27,),
        FixtureEvidence.EXHAUSTIVE,
        0.18666666666666668,
        0.026666666666666672,
        "Exhaustive continuations favor completing the 60-point black row.",
    ),
    StrategicFixture(
        "queen-middle-color-exploit",
        "Queen uses an opponent card to complete a same-color row",
        (
            StrategicBehavior.SAME_COLOR_COMPLETION,
            StrategicBehavior.EXPLOIT_OPPONENT_PLACEMENT,
        ),
        "strategy-candidate-00000",
        5,
        EnginePlayer.QUEEN,
        True,
        RoundStage.MIDDLE,
        (29,),
        FixtureEvidence.EXHAUSTIVE,
        0.22666666666666668,
        0.05,
        "JS completes a black row using the opponent's 10S when no suit line is available.",
    ),
    StrategicFixture(
        "king-middle-suit",
        "King completes the dominant Spade column",
        (StrategicBehavior.SAME_SUIT_COMPLETION,),
        "strategy-candidate-00004",
        5,
        EnginePlayer.KING,
        True,
        RoundStage.MIDDLE,
        (24,),
        FixtureEvidence.EXHAUSTIVE,
        0.4066666666666667,
        0.26666666666666666,
        "9S completes a 130-point Spade column.",
    ),
    StrategicFixture(
        "queen-middle-avoid-vampire",
        "Queen selects the higher live combination and avoids a Vampire",
        (
            StrategicBehavior.HIGH_FACE_VALUE,
            StrategicBehavior.AVOID_VAMPIRE_LINE,
        ),
        "strategy-candidate-00013",
        5,
        EnginePlayer.QUEEN,
        True,
        RoundStage.MIDDLE,
        (4,),
        FixtureEvidence.EXHAUSTIVE,
        -0.2,
        0.09,
        "9C makes 57 instead of 30 with KS and avoids investing in the V2 row.",
    ),
    StrategicFixture(
        "king-middle-preserve-card",
        "King retains the stronger Spade for the final construction",
        (StrategicBehavior.PRESERVE_VALUABLE_CARD,),
        "strategy-candidate-00000",
        4,
        EnginePlayer.KING,
        False,
        RoundStage.MIDDLE,
        (10,),
        FixtureEvidence.EXHAUSTIVE,
        -0.09555555555555556,
        0.012222222222222218,
        "6S is preferred to 7S at the same position, preserving 7S for the last turn.",
        comparison_action_index=18,
    ),
    StrategicFixture(
        "king-late-block",
        "King breaks a visible Heart row",
        (StrategicBehavior.BLOCK_VISIBLE_MULTIPLIER,),
        "strategy-candidate-00001",
        6,
        EnginePlayer.KING,
        False,
        RoundStage.LATE,
        (10,),
        FixtureEvidence.EXHAUSTIVE,
        0.05333333333333334,
        0.23333333333333334,
        "3C at position 6 prevents the Queen's 9H-3H line from reaching five-times Hearts.",
    ),
    StrategicFixture(
        "queen-late-offense-defense",
        "Queen resolves a final offense-versus-denial choice",
        (StrategicBehavior.OFFENSE_DEFENSE_CONFLICT,),
        "strategy-candidate-00019",
        6,
        EnginePlayer.QUEEN,
        False,
        RoundStage.LATE,
        (20,),
        FixtureEvidence.EXHAUSTIVE,
        0.02666666666666667,
        0.04,
        "The live same-color completion exceeds the alternative defensive placement.",
    ),
    StrategicFixture(
        "dealer-late-forced",
        "Dealer's final placement is forced",
        (),
        "strategy-candidate-00000",
        7,
        EnginePlayer.QUEEN,
        True,
        RoundStage.LATE,
        (25,),
        FixtureEvidence.FORCED,
        None,
        None,
        "The sole engine-legal action is a lifecycle control, not a strategic pass case.",
    ),
)

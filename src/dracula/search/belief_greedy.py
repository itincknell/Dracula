"""Choose continuation moves by scoring plausible completed coffins.

Information-set UCT calls this policy when it needs a move to continue one of
its trial rounds. The policy receives only the information visible to the
player whose simulated turn it is.

To choose one move, the "belief greedy" policy follows these steps:

1. List every legal choice of card and strategically distinct destination.
2. Draw eight possible hands that the opponent could hold from the cards this
   player has not seen.
3. For each possible opponent hand, evaluate every legal choice from step 1.
   Put the chosen card at that choice's representative destination. Then make a
   complete hypothetical coffin by placing this player's remaining cards and
   the sampled opponent cards into every empty position. Each remaining card
   receives a stable pseudo-random priority derived from that sample's seed and
   the card's identity. Cards sorted by those priorities fill the empty coffin
   positions in grid order. This is not intended to imitate tactical play: it
   supplies a reproducible, varied completion in which a given card keeps the
   same priority while different candidate moves are compared. No further move
   selection occurs inside the completion.
4. Score each completed coffin. If Queen is choosing, the result is Queen's
   score minus King's. If King is choosing, it is King's score minus Queen's.
5. Average each candidate's eight results and choose the highest average.

All candidates use the same possible opponent hand and completion order within
each of the eight samples. This makes their scores directly comparable instead
of giving each candidate different sampling luck. Canonical action order breaks
an exact tie. If the winning destination group is a mirrored pair, the
deterministic fair coin chooses its concrete destination only after this
comparison is complete.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

from dracula.decision.bridge import ACTION_COUNT
from dracula.game.engine import EnginePlayer, score_coffin
from dracula.game.scoring import round_scores_from_lines
from dracula.randomness import shuffled, stable_seed
from dracula.search.contracts import (
    ContinuationDecision,
    ROUND_SCORE_NORMALIZER,
    SearchInterrupted,
    StrategicGroupStatistics,
)
from dracula.search.information_set_uct import (
    InformationSetUCTConfig,
    InformationSetUCTSearch,
)
from dracula.decision.information import (
    SearchInformationState,
    information_state_fingerprint,
)
from dracula.decision.strategic_actions import (
    StrategicActionGroup,
    select_concrete_action_index,
    strategic_action_groups,
)


@dataclass(frozen=True, slots=True)
class BeliefGreedyContinuationConfig:
    """Set how many hypothetical opponent hands evaluate each candidate move."""

    belief_completion_count: int = 8


def _response_seed(
    information: SearchInformationState,
    completion_count: int,
) -> int:
    """Return the local random seed for one visible response evaluation."""

    return stable_seed(
        information_state_fingerprint(information),
        completion_count,
    )


def _completion_order(cards: tuple[str, ...], seed: int) -> tuple[str, ...]:
    """Order sampled remaining cards reproducibly, independent of tuple order.

    Each card receives its own derived key, so the same card receives the same
    priority when different candidate groups are compared against one sample.
    """

    return tuple(
        sorted(
            cards,
            key=lambda card_id: stable_seed(seed, card_id),
        )
    )


def _sample_opponent_hand(
    information: SearchInformationState,
    completion_seed: int,
) -> tuple[str, ...]:
    """Draw the unseen cards used as one possible opponent hand."""

    sampled = shuffled(information.unseen_card_ids, completion_seed)
    opponent_hand = sampled[: information.opponent_remaining_count]
    return tuple(sorted(opponent_hand))


def _fill_remaining_positions(
    coffin: list[str | None],
    remaining_cards: tuple[str, ...],
    completion_seed: int,
) -> tuple[str, ...]:
    """Fill open grid positions using one sample's stable card priorities."""

    empty_positions = tuple(
        index for index, card_id in enumerate(coffin) if card_id is None
    )
    ordered_cards = _completion_order(remaining_cards, completion_seed)

    # Strict pairing enforces the central completion invariant: after the
    # candidate move, the two sampled hands exactly fill the current round.
    for position, card_id in zip(empty_positions, ordered_cards, strict=True):
        coffin[position] = card_id
    return cast(tuple[str, ...], tuple(coffin))


def _completed_coffin(
    information: SearchInformationState,
    group: StrategicActionGroup,
    opponent_hand: tuple[str, ...],
    completion_seed: int,
) -> tuple[str, ...]:
    """Place one candidate and complete the board under one sampled belief."""

    coffin = list(information.coffin)
    # Strategic groups come from the information state's legal actions, which
    # guarantees that the named hand slot contains a card.
    coffin[group.representative_grid_index] = information.own_hand[group.hand_slot]
    own_remaining = tuple(
        card_id
        for hand_slot, card_id in enumerate(information.own_hand)
        if hand_slot != group.hand_slot and card_id is not None
    )
    return _fill_remaining_positions(
        coffin,
        own_remaining + opponent_hand,
        completion_seed,
    )


def _actor_relative_value(
    coffin: tuple[str, ...],
    actor: EnginePlayer,
) -> float:
    """Score one full coffin as the choosing player's normalized advantage."""

    lines = score_coffin(coffin)
    scores = round_scores_from_lines(lines)
    differential = (
        scores.queen - scores.king
        if actor is EnginePlayer.QUEEN
        else scores.king - scores.queen
    )
    return differential / ROUND_SCORE_NORMALIZER


def _group_value(
    information: SearchInformationState,
    group: StrategicActionGroup,
    opponent_hand: tuple[str, ...],
    completion_seed: int,
) -> float:
    """Evaluate one candidate against one possible opponent hand."""

    coffin = _completed_coffin(
        information,
        group,
        opponent_hand,
        completion_seed,
    )
    return _actor_relative_value(coffin, information.player)


def _evaluate_group_values(
    information: SearchInformationState,
    groups: tuple[StrategicActionGroup, ...],
    request_seed: int,
    completion_count: int,
    should_stop: Callable[[], bool] | None,
) -> list[float]:
    """Accumulate each group's value over identical hidden-card samples."""

    value_sums = [0.0] * ACTION_COUNT
    for completion_index in range(completion_count):
        if should_stop is not None and should_stop():
            raise SearchInterrupted("belief-greedy continuation interrupted")
        completion_seed = stable_seed(request_seed, completion_index)
        opponent_hand = _sample_opponent_hand(information, completion_seed)

        # Reusing the sample isolates candidate choice from sampling luck.
        for group in groups:
            value_sums[group.representative_action_index] += _group_value(
                information,
                group,
                opponent_hand,
                completion_seed,
            )
    return value_sums


def _mean_group_value(
    value_sums: list[float],
    group: StrategicActionGroup,
    completion_count: int,
) -> float:
    """Return one candidate's mean over all sampled completions."""

    return value_sums[group.representative_action_index] / completion_count


def _group_statistics(
    groups: tuple[StrategicActionGroup, ...],
    value_sums: list[float],
    completion_count: int,
) -> tuple[StrategicGroupStatistics, ...]:
    """Convert accumulated sample values into continuation diagnostics."""

    return tuple(
        StrategicGroupStatistics(
            group,
            completion_count,
            _mean_group_value(value_sums, group, completion_count),
        )
        for group in groups
    )


def _best_group(
    groups: tuple[StrategicActionGroup, ...],
    value_sums: list[float],
    completion_count: int,
) -> StrategicActionGroup:
    """Choose the highest mean value, resolving exact ties canonically."""

    return min(
        groups,
        key=lambda group: (
            -_mean_group_value(value_sums, group, completion_count),
            group.representative_action_index,
        ),
    )


class BeliefGreedyContinuation:
    """Rank every strategic group over shared samples of the hidden cards."""

    def __init__(
        self,
        config: BeliefGreedyContinuationConfig = BeliefGreedyContinuationConfig(),
    ) -> None:
        self.config = config

    def select(
        self,
        information: SearchInformationState,
        should_stop: Callable[[], bool] | None = None,
    ) -> ContinuationDecision:
        """Evaluate all legal groups over shared beliefs and choose the best mean."""

        groups = strategic_action_groups(information)
        completion_count = self.config.belief_completion_count
        request_seed = _response_seed(information, completion_count)
        # Every group sees the same hidden-card samples, so differences reflect
        # the candidate action rather than independent sampling luck.
        value_sums = _evaluate_group_values(
            information,
            groups,
            request_seed,
            completion_count,
            should_stop,
        )
        statistics = _group_statistics(groups, value_sums, completion_count)
        selected_group = _best_group(groups, value_sums, completion_count)

        # The fair coin affects only the concrete member returned to the engine.
        return ContinuationDecision(
            selected_group=selected_group,
            selected_action_index=select_concrete_action_index(
                selected_group,
                request_seed,
                selected_group.representative_action_index,
            ),
            group_statistics=statistics,
            terminal_evaluation_count=len(groups) * completion_count,
            model_inference_count=0,
        )


class BeliefGreedyInformationSetSearch(InformationSetUCTSearch):
    """Run information-set UCT with belief-greedy continuation moves."""

    def __init__(
        self,
        search_config: InformationSetUCTConfig = InformationSetUCTConfig(),
        continuation_config: BeliefGreedyContinuationConfig = (
            BeliefGreedyContinuationConfig()
        ),
    ) -> None:
        super().__init__(
            BeliefGreedyContinuation(continuation_config),
            search_config,
        )


__all__ = (
    "BeliefGreedyContinuation",
    "BeliefGreedyContinuationConfig",
    "BeliefGreedyInformationSetSearch",
)

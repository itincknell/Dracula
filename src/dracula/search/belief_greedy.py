"""Choose BGC continuation moves with one-ply belief-greedy evaluation.

This policy replaces a nested response search. It samples eight plausible
opponent hands from the acting player's unseen cards. For each sample, every
legal strategic group is placed first and the remaining round is completed in
one reproducible order. Groups are ranked by their mean exact round-score
difference across those shared samples.

The method reasons about hidden cards probabilistically but never receives the
authoritative opponent hand or stock order.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from dracula.bridge import ACTION_COUNT
from dracula.engine import EnginePlayer, resolve_round_scores, score_coffin
from dracula.randomness import shuffled, stable_seed
from dracula.search.contracts import (
    ContinuationDecision,
    ROUND_SCORE_NORMALIZER,
    SearchInterrupted,
    StrategicGroupStatistics,
)
from dracula.search.bgc import BGCInformationSetSearch, BGCSearchConfig
from dracula.search.information import (
    SearchInformationState,
    information_state_fingerprint,
)
from dracula.strategic_actions import (
    StrategicActionGroup,
    select_concrete_action_index,
    strategic_action_groups,
)


@dataclass(frozen=True, slots=True)
class BeliefGreedyContinuationConfig:
    """Sampling budget for the baseline BGC one-ply response policy."""

    belief_completion_count: int = 8

    def __post_init__(self) -> None:
        if self.belief_completion_count < 1:
            raise ValueError("belief completion count must be positive")


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


def _group_value(
    information: SearchInformationState,
    group: StrategicActionGroup,
    opponent_hand: tuple[str, ...],
    completion_seed: int,
) -> float:
    """Return one group's exact score after one sampled round completion.

    Evaluation uses the group's designated representative destination. A paired
    concrete destination is chosen only after the strategic group wins the mean
    comparison, so it cannot influence the group's estimated value.
    """

    card_id = information.own_hand[group.hand_slot]
    if card_id is None:
        raise ValueError("strategic group refers to an unavailable card")
    coffin = list(information.coffin)
    coffin[group.representative_grid_index] = card_id
    own_remaining = tuple(
        candidate
        for hand_slot, candidate in enumerate(information.own_hand)
        if hand_slot != group.hand_slot and candidate is not None
    )
    empty_positions = tuple(
        index for index, candidate in enumerate(coffin) if candidate is None
    )
    future_cards = own_remaining + opponent_hand
    if len(future_cards) != len(empty_positions):
        raise ValueError("belief completion does not fill the current round")
    # This continuation is deliberately one-ply: after the candidate action,
    # a deterministic completion assigns the remaining sampled cards to spaces.
    for position, future_card in zip(
        empty_positions,
        _completion_order(future_cards, completion_seed),
        strict=True,
    ):
        coffin[position] = future_card

    complete_coffin = tuple(coffin)
    if any(card is None for card in complete_coffin):
        raise ValueError("belief completion left an empty coffin position")
    lines = score_coffin(complete_coffin)  # type: ignore[arg-type]
    scores = resolve_round_scores(
        tuple(line.total for line in lines.queen),
        tuple(line.total for line in lines.king),
    )
    actor = information.player
    differential = (
        scores.queen - scores.king
        if actor is EnginePlayer.QUEEN
        else scores.king - scores.queen
    )
    return differential / ROUND_SCORE_NORMALIZER


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
        # Only the opponent cards needed to finish this round are sampled. The
        # same hand and completion order are then reused for every candidate.
        opponent_hand = tuple(
            sorted(
                shuffled(information.unseen_card_ids, completion_seed)[
                    : information.opponent_remaining_count
                ]
            )
        )
        for group in groups:
            value_sums[group.representative_action_index] += _group_value(
                information,
                group,
                opponent_hand,
                completion_seed,
            )
    return value_sums


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
        statistics = tuple(
            StrategicGroupStatistics(
                group,
                completion_count,
                value_sums[group.representative_action_index] / completion_count,
            )
            for group in groups
        )
        selected_group = min(
            groups,
            key=lambda group: (
                -value_sums[group.representative_action_index] / completion_count,
                group.representative_action_index,
            ),
        )
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


class BeliefGreedyInformationSetSearch(BGCInformationSetSearch):
    """BGC-128 with the baseline eight-completion greedy continuation."""

    def __init__(
        self,
        search_config: BGCSearchConfig = BGCSearchConfig(),
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

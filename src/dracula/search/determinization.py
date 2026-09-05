"""Build complete private engine worlds for information-set search.

``SearchInformationState`` deliberately omits the opponent's remaining cards
and future stock order. The simulation engine nevertheless needs concrete cards
in those locations. Each determinization assigns the visible state's unseen
cards to one plausible opponent hand and stock, reconstructs a runnable private
state, and verifies that hiding those sampled details yields the original view.

The sampled world exists only inside search. It is neither an authoritative
game state nor input to a continuation policy.
"""

from __future__ import annotations

from dataclasses import dataclass

from dracula.bridge import COFFIN_POSITION_COUNT, HAND_SLOT_COUNT, global_grid_index
from dracula.cards import sort_card_ids
from dracula.engine import (
    CENTER_GRID_INDEX,
    EnginePlayedMove,
    EnginePlayer,
    EngineRoundResult,
    EngineStatus,
    PlayerValues,
    SimulationEngineState,
    other_player,
    score_coffin,
)
from dracula.randomness import seed_hex, shuffled
from dracula.search.information import (
    InformationContractViolation,
    PublicPlayedMove,
    PublicRoundRecord,
    SearchInformationState,
    information_state_from_simulation,
)


@dataclass(frozen=True, slots=True)
class SampledDeterminization:
    """One reproducible assignment of hidden cards used by a simulation.

    ``state`` is the complete runnable engine state. The separately retained
    opponent hand, stock, and seed digest support private diagnostics and tests;
    none may cross the search boundary.
    """

    root_player: EnginePlayer
    sample_seed_digest: str
    opponent_remaining_hand: tuple[str, ...]
    sampled_stock: tuple[str, ...]
    state: SimulationEngineState


def _present_cards(values: tuple[str | None, ...]) -> tuple[str, ...]:
    """Return cards still present in a stable-slot hand or coffin."""

    return tuple(value for value in values if value is not None)


def _private_moves(
    moves: tuple[PublicPlayedMove, ...],
    original_hands: PlayerValues[tuple[str, ...]],
) -> tuple[EnginePlayedMove, ...]:
    """Restore the private hand-slot index omitted from public move history.

    Engine hands are canonically sorted when dealt and keep four stable slots as
    cards are played. Reconstructing the original hand therefore determines the
    unique slot occupied by every publicly recorded card.
    """

    slots = {
        player: {card_id: index for index, card_id in enumerate(original_hands[player])}
        for player in EnginePlayer
    }
    return tuple(
        EnginePlayedMove(
            player=move.player,
            card_id=move.card_id,
            hand_slot=slots[move.player][move.card_id],
            global_grid_index=move.global_grid_index,
            turn_number=move.turn_number,
        )
        for move in moves
    )


def _completed_round_result(record: PublicRoundRecord) -> EngineRoundResult:
    """Rebuild an engine round result from cards that are already public.

    Completed-round history omits stable hand slots and derived line scores.
    Both can be recovered exactly because all nine cards and all eight moves are
    public after the round.
    """

    hands = PlayerValues(
        queen=sort_card_ids(
            move.card_id
            for move in record.moves
            if move.player is EnginePlayer.QUEEN
        ),
        king=sort_card_ids(
            move.card_id
            for move in record.moves
            if move.player is EnginePlayer.KING
        ),
    )
    return EngineRoundResult(
        round_number=record.round_number,
        dealer=record.dealer,
        coffin=record.coffin,
        moves=_private_moves(record.moves, hands),
        line_scores=score_coffin(record.coffin),
        round_scores=record.round_scores,
    )


def _original_hand(
    remaining: tuple[str, ...],
    moves: tuple[PublicPlayedMove, ...],
    player: EnginePlayer,
) -> tuple[str, ...]:
    """Recover the four cards originally dealt to one player this round.

    For the root player, remaining cards are visible. For the opponent, they
    come from the current hidden-card sample. Cards already played are public in
    either case.
    """

    original = sort_card_ids(
        (*remaining, *(move.card_id for move in moves if move.player is player))
    )
    if len(original) != HAND_SLOT_COUNT or len(set(original)) != HAND_SLOT_COUNT:
        raise InformationContractViolation(
            "sampled original hand must contain four distinct cards"
        )
    return original


def _remaining_slots(
    original: tuple[str, ...],
    moves: tuple[PublicPlayedMove, ...],
    player: EnginePlayer,
) -> tuple[str | None, str | None, str | None, str | None]:
    """Recreate a four-slot hand with holes for cards already played."""

    played = {move.card_id for move in moves if move.player is player}
    return tuple(  # type: ignore[return-value]
        None if card_id in played else card_id for card_id in original
    )


def _deal_segment(
    hands: PlayerValues[tuple[str, ...]],
    dealer: EnginePlayer,
    center_card: str,
) -> tuple[str, ...]:
    """Invert the fixed packet deal into its nine-card deck segment."""

    non_dealer_hand = hands[other_player(dealer)]
    dealer_hand = hands[dealer]
    return (
        non_dealer_hand[0],
        non_dealer_hand[1],
        dealer_hand[0],
        dealer_hand[1],
        non_dealer_hand[2],
        non_dealer_hand[3],
        dealer_hand[2],
        dealer_hand[3],
        center_card,
    )


def _global_coffin(
    information: SearchInformationState,
) -> tuple[str | None, ...]:
    """Undo the actor-relative board orientation required by engine state."""

    result: list[str | None] = [None] * COFFIN_POSITION_COUNT
    for relative_index, card_id in enumerate(information.coffin):
        result[global_grid_index(information.player, relative_index)] = card_id
    return tuple(result)


def _sample_hidden_cards(
    information: SearchInformationState,
    sample_seed: bytes,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Partition unseen cards into one plausible opponent hand and stock."""

    # The visible opponent-card count fixes the partition boundary. Shuffling
    # changes which unseen cards occupy those hidden locations, not their counts.
    hidden_order = shuffled(information.unseen_card_ids, sample_seed)
    opponent_remaining = sort_card_ids(
        hidden_order[: information.opponent_remaining_count]
    )
    sampled_stock = hidden_order[information.opponent_remaining_count :]
    if len(sampled_stock) != information.stock_count:
        raise InformationContractViolation("sampled stock count is inconsistent")
    return opponent_remaining, sampled_stock


def _reconstruct_simulation_deck(
    information: SearchInformationState,
    completed_results: tuple[EngineRoundResult, ...],
    original_hands: PlayerValues[tuple[str, ...]],
    sampled_stock: tuple[str, ...],
    current_center: str,
) -> tuple[str, ...]:
    """Recreate the full dealt-deck order required by simulation validation.

    Completed rounds are entirely public. The current round combines visible
    cards with the sampled opponent hand, and sampled stock supplies all future
    rounds.
    """

    deck_parts: list[str] = []
    for result in completed_results:
        # Public move records retain original hand slots. Sorting by those slots
        # restores the two dealt hands before the packet deal is inverted.
        completed_hands = PlayerValues(
            queen=tuple(
                move.card_id
                for move in sorted(
                    (
                        move
                        for move in result.moves
                        if move.player is EnginePlayer.QUEEN
                    ),
                    key=lambda move: move.hand_slot,
                )
            ),
            king=tuple(
                move.card_id
                for move in sorted(
                    (
                        move
                        for move in result.moves
                        if move.player is EnginePlayer.KING
                    ),
                    key=lambda move: move.hand_slot,
                )
            ),
        )
        # Every completed round accounts for one contiguous nine-card deal.
        deck_parts.extend(
            _deal_segment(
                completed_hands,
                result.dealer,
                result.coffin[CENTER_GRID_INDEX],
            )
        )
    # The current nine-card segment includes the sampled opponent hand. All
    # remaining sampled cards retain their order as the undealt stock.
    deck_parts.extend(
        _deal_segment(original_hands, information.dealer, current_center)
    )
    deck_parts.extend(sampled_stock)
    return tuple(deck_parts)


def _build_simulation_state(
    information: SearchInformationState,
    opponent_remaining: tuple[str, ...],
    sampled_stock: tuple[str, ...],
    sample_seed_digest: str,
) -> SimulationEngineState:
    """Restore private engine fields around one sampled hidden-card partition."""

    # Public moves reveal cards already played. Combining them with each
    # player's remaining cards recovers the stable four-slot hands originally dealt.
    root = information.player
    opponent = other_player(root)
    root_original = _original_hand(
        _present_cards(information.own_hand),
        information.current_round_moves,
        root,
    )
    opponent_original = _original_hand(
        opponent_remaining,
        information.current_round_moves,
        opponent,
    )
    original_hands = PlayerValues(
        queen=root_original if root is EnginePlayer.QUEEN else opponent_original,
        king=root_original if root is EnginePlayer.KING else opponent_original,
    )

    # Completed rounds need their derived engine records, while the current
    # coffin must be rotated out of the root player's relative orientation.
    completed_results = tuple(
        _completed_round_result(record) for record in information.completed_rounds
    )
    global_coffin = _global_coffin(information)
    center_card = global_coffin[CENTER_GRID_INDEX]
    if center_card is None:
        raise InformationContractViolation("sampled round requires its public center")

    # Engine hand slots never collapse after a play; consumed cards become holes
    # so public action indexes still map to their original dealt positions.
    sampled_hands = PlayerValues(
        queen=_remaining_slots(
            original_hands.queen,
            information.current_round_moves,
            EnginePlayer.QUEEN,
        ),
        king=_remaining_slots(
            original_hands.king,
            information.current_round_moves,
            EnginePlayer.KING,
        ),
    )
    # The reconstructed deck is a conservation witness used by simulation-state
    # validation. Search decisions still receive only a projected information state.
    return SimulationEngineState(
        seed=f"search-simulation-v1:{sample_seed_digest}",
        status=EngineStatus.PLAYING,
        round_number=information.round_number,
        dealer=information.dealer,
        active_player=information.active_player,
        stock=sampled_stock,
        hands=sampled_hands,
        coffin=global_coffin,  # type: ignore[arg-type]
        current_round_moves=_private_moves(
            information.current_round_moves,
            original_hands,
        ),
        pending_round_result=None,
        completed_rounds=completed_results,
        total_scores=information.total_scores,
        simulation_deck=_reconstruct_simulation_deck(
            information,
            completed_results,
            original_hands,
            sampled_stock,
            center_card,
        ),
    )


def sample_determinization(
    information: SearchInformationState,
    sample_seed: bytes,
) -> SampledDeterminization:
    """Create one runnable world compatible with an actor-visible position.

    The seed deterministically shuffles only ``unseen_card_ids``. The number of
    cards known to remain in the opponent hand determines the first partition;
    the rest become future stock. Public history and the sampled current hands
    are then assembled into the exact private engine representation.
    """

    sample_seed_digest = seed_hex(sample_seed)
    opponent_remaining, sampled_stock = _sample_hidden_cards(
        information,
        sample_seed,
    )
    state = _build_simulation_state(
        information,
        opponent_remaining,
        sampled_stock,
        sample_seed_digest,
    )
    # Projection must erase the sampled opponent hand and stock partition.
    if information_state_from_simulation(state) != information:
        raise InformationContractViolation(
            "sampled world does not reproduce its source information"
        )
    # Hidden assignments remain available only to the enclosing search and its
    # diagnostics; continuation policies receive projections derived from state.
    return SampledDeterminization(
        root_player=information.player,
        sample_seed_digest=sample_seed_digest,
        opponent_remaining_hand=opponent_remaining,
        sampled_stock=sampled_stock,
        state=state,
    )


__all__ = ("SampledDeterminization", "sample_determinization")

"""Information-safe reconstruction of sampled private search worlds."""

from __future__ import annotations

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
    SampledDeterminization,
    SearchInformationState,
    _card_ids,
    _validate_information_state,
    project_simulation_information_state,
)


def _private_moves(
    moves: tuple[PublicPlayedMove, ...],
    original_hands: PlayerValues[tuple[str, ...]],
) -> tuple[EnginePlayedMove, ...]:
    """Restore stable hand slots after sampling each original four-card hand."""

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
    """Rebuild validated private move slots for an already public round."""

    hands = PlayerValues(
        queen=sort_card_ids(
            move.card_id for move in record.moves if move.player is EnginePlayer.QUEEN
        ),
        king=sort_card_ids(
            move.card_id for move in record.moves if move.player is EnginePlayer.KING
        ),
    )
    lines = score_coffin(record.coffin)
    return EngineRoundResult(
        round_number=record.round_number,
        dealer=record.dealer,
        coffin=record.coffin,
        moves=_private_moves(record.moves, hands),
        line_scores=lines,
        round_scores=record.round_scores,
    )


def _original_hand(
    remaining: tuple[str, ...],
    moves: tuple[PublicPlayedMove, ...],
    player: EnginePlayer,
) -> tuple[str, ...]:
    """Recombine held and played cards into one canonical four-card hand."""

    original = sort_card_ids(
        (*remaining, *(move.card_id for move in moves if move.player is player))
    )
    if len(original) != HAND_SLOT_COUNT or len(set(original)) != HAND_SLOT_COUNT:
        raise InformationContractViolation("sampled original hand must contain four cards")
    return original


def _remaining_slots(
    original: tuple[str, ...],
    moves: tuple[PublicPlayedMove, ...],
    player: EnginePlayer,
) -> tuple[str | None, str | None, str | None, str | None]:
    """Mark played cards absent while retaining their original canonical slots."""

    played = {move.card_id for move in moves if move.player is player}
    return tuple(  # type: ignore[return-value]
        None if card_id in played else card_id for card_id in original
    )


def _deal_segment(
    hands: PlayerValues[tuple[str, ...]], dealer: EnginePlayer, center_card: str
) -> tuple[str, ...]:
    """Invert packet dealing into the nine-card source-deck segment."""

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


def _global_coffin(state: SearchInformationState) -> tuple[str | None, ...]:
    """Invert the root player's relative coffin coordinates."""

    result: list[str | None] = [None] * COFFIN_POSITION_COUNT
    for policy_index, card_id in enumerate(state.coffin):
        result[global_grid_index(state.player, policy_index)] = card_id
    return tuple(result)


def sample_determinization(
    information: SearchInformationState, sample_seed: bytes
) -> SampledDeterminization:
    """Sample a complete private world using only a player information state."""

    _validate_information_state(information)
    sample_seed_digest = seed_hex(sample_seed)
    # Only the actor's unseen pool and public location counts determine this
    # split; the authoritative opponent hand and stock order are unavailable.
    hidden_order = shuffled(information.unseen_card_ids, sample_seed)
    opponent_remaining = sort_card_ids(
        hidden_order[: information.opponent_remaining_count]
    )
    sampled_stock = hidden_order[information.opponent_remaining_count :]
    if len(sampled_stock) != information.stock_count:
        raise InformationContractViolation("sampled stock count is inconsistent")

    root = information.player
    opponent = other_player(root)
    root_original = _original_hand(
        _card_ids(information.own_hand), information.current_round_moves, root
    )
    opponent_original = _original_hand(
        opponent_remaining, information.current_round_moves, opponent
    )
    current_hands = PlayerValues(
        queen=root_original if root is EnginePlayer.QUEEN else opponent_original,
        king=root_original if root is EnginePlayer.KING else opponent_original,
    )

    # Reconstruct a complete sampled deck so ordinary engine validation and
    # transitions can operate without a separate simulation ruleset.
    completed_results = tuple(
        _completed_round_result(record) for record in information.completed_rounds
    )
    deck_parts: list[str] = []
    for result in completed_results:
        result_hands = PlayerValues(
            queen=tuple(
                move.card_id
                for move in sorted(
                    (move for move in result.moves if move.player is EnginePlayer.QUEEN),
                    key=lambda move: move.hand_slot,
                )
            ),
            king=tuple(
                move.card_id
                for move in sorted(
                    (move for move in result.moves if move.player is EnginePlayer.KING),
                    key=lambda move: move.hand_slot,
                )
            ),
        )
        deck_parts.extend(
            _deal_segment(
                result_hands,
                result.dealer,
                result.coffin[CENTER_GRID_INDEX],
            )
        )
    global_coffin = _global_coffin(information)
    center_card = global_coffin[CENTER_GRID_INDEX]
    if center_card is None:
        raise InformationContractViolation("sampled round requires its public center")
    deck_parts.extend(_deal_segment(current_hands, information.dealer, center_card))
    deck_parts.extend(sampled_stock)

    private_moves = _private_moves(information.current_round_moves, current_hands)
    simulated_hands = PlayerValues(
        queen=_remaining_slots(
            current_hands.queen, information.current_round_moves, EnginePlayer.QUEEN
        ),
        king=_remaining_slots(
            current_hands.king, information.current_round_moves, EnginePlayer.KING
        ),
    )
    if simulated_hands[root] != information.own_hand:
        raise InformationContractViolation("sample changed the root player's stable slots")
    state = SimulationEngineState(
        seed=f"search-simulation-v1:{sample_seed_digest}",
        status=EngineStatus.PLAYING,
        round_number=information.round_number,
        dealer=information.dealer,
        active_player=information.active_player,
        stock=sampled_stock,
        hands=simulated_hands,
        coffin=global_coffin,  # type: ignore[arg-type]
        current_round_moves=private_moves,
        pending_round_result=None,
        completed_rounds=completed_results,
        total_scores=information.total_scores,
        simulation_deck=tuple(deck_parts),
    )
    # Round-tripping must erase the sample and recover the exact source view.
    if project_simulation_information_state(state) != information:
        raise InformationContractViolation(
            "sample does not reproduce its root information state"
        )
    return SampledDeterminization(
        root_player=root,
        sample_seed_digest=sample_seed_digest,
        opponent_remaining_hand=opponent_remaining,
        sampled_stock=sampled_stock,
        state=state,
    )

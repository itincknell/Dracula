"""Verify actor-visible information construction and privacy guarantees.

The suite checks public history, hand and unseen-card partitions, legal masks,
canonical fingerprints, hidden-state equivalence, and malformed inputs.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError, fields, replace

import pytest

from dracula.decision.bridge import action_index_for_move
from dracula.game.cards import CARD_IDS, sort_card_ids
from dracula.game.engine import (
    EnginePlayer,
    EngineStatus,
    PlayerValues,
    SimulationEngineState,
    advance_after_round,
    apply_move,
    create_game,
    legal_moves,
    other_player,
)
from dracula.game.dealing import as_hand, deal_round, shuffled_deck, starting_coffin
from dracula.decision.information import (
    PublicGameHistory,
    PublicPlayedMove,
    canonical_information_data,
    information_state_fingerprint,
    information_state_from_engine,
    public_history_from_engine,
)


def _advance(state, count: int):
    for _ in range(count):
        assert state.active_player is not None
        state = apply_move(state, legal_moves(state, state.active_player)[0]).state
    return state


def _hidden_location_variant(seed: str) -> SimulationEngineState:
    """Move one dealer card into stock without changing the actor-visible state."""

    original = create_game(seed)
    sampled_deck = list(shuffled_deck(seed))
    # Index 2 belongs to the dealer's first packet; index 9 begins hidden stock.
    sampled_deck[2], sampled_deck[9] = sampled_deck[9], sampled_deck[2]
    simulation_deck = tuple(sampled_deck)
    deal = deal_round(simulation_deck, original.dealer)
    return SimulationEngineState(
        seed=seed,
        status=EngineStatus.PLAYING,
        round_number=1,
        dealer=deal.dealer,
        active_player=deal.non_dealer,
        stock=deal.remaining_stock,
        hands=PlayerValues(
            queen=as_hand(deal.queen_hand),
            king=as_hand(deal.king_hand),
        ),
        coffin=starting_coffin(deal.center_card),
        current_round_moves=(),
        pending_round_result=None,
        completed_rounds=(),
        total_scores=PlayerValues(queen=0, king=0),
        simulation_deck=simulation_deck,
    )


def test_information_state_is_immutable_and_contains_only_actor_visible_cards() -> None:
    state = _advance(create_game("information-visible"), 3)
    information = information_state_from_engine(state)
    player = state.active_player
    assert player is not None
    assert information.player is player
    assert information.own_hand == state.hands[player]
    assert set(information.unseen_card_ids) == {
        card
        for card in (*state.hands[other_player(player)], *state.stock)
        if card is not None
    }
    assert set(information.played_card_ids) == {
        card for card in state.coffin if card is not None
    }
    assert not hasattr(information, "opponent_hand")
    assert not hasattr(information, "stock")
    assert not hasattr(information, "seed")
    assert {field.name for field in fields(PublicGameHistory)}.isdisjoint(
        {"seed", "stock", "hands", "opponent_hand"}
    )
    assert all("slot" not in field.name for field in fields(PublicPlayedMove))
    with pytest.raises(FrozenInstanceError):
        information.round_number = 2  # type: ignore[misc]


def test_information_sets_partition_all_cards_and_recompute_legality() -> None:
    state = create_game("information-partition")
    for placement in range(8):
        information = information_state_from_engine(state)
        own = {card for card in information.own_hand if card is not None}
        assert (
            set(information.played_card_ids)
            | own
            | set(information.unseen_card_ids)
        ) == set(CARD_IDS)
        assert information.played_card_ids == sort_card_ids(
            information.played_card_ids
        )
        assert sum(map(len, (information.played_card_ids, own, information.unseen_card_ids))) == len(CARD_IDS)
        legal_indexes = {
            index
            for index, allowed in enumerate(
                value for row in information.legal_mask for value in row
            )
            if allowed
        }
        assert legal_indexes
        assert legal_indexes == {
            action_index_for_move(move, information.player)
            for move in legal_moves(state, information.player)
        }
        assert all(information.own_hand[index // 8] is not None for index in legal_indexes)
        if placement < 7:
            assert state.active_player is not None
            state = apply_move(state, legal_moves(state, state.active_player)[0]).state


def test_hidden_location_variants_have_identical_information_and_fingerprints() -> None:
    seed = "information-hidden-equivalence"
    original = create_game(seed)
    variant = _hidden_location_variant(seed)
    assert original.active_player is not None
    opponent = other_player(original.active_player)
    assert original.hands[opponent] != variant.hands[opponent]
    assert original.stock != variant.stock

    original_information = information_state_from_engine(original)
    variant_information = information_state_from_engine(variant)
    assert original_information == variant_information
    assert information_state_fingerprint(original_information) == (
        information_state_fingerprint(variant_information)
    )


def test_visible_move_changes_canonical_information_and_fingerprint() -> None:
    state = create_game("information-visible-fingerprint")
    assert state.active_player is not None
    options = legal_moves(state, state.active_player)
    first = options[0]
    second = next(
        move
        for move in options
        if move.hand_slot == first.hand_slot
        and move.global_grid_index != first.global_grid_index
    )
    first_information = information_state_from_engine(apply_move(state, first).state)
    second_information = information_state_from_engine(apply_move(state, second).state)

    assert canonical_information_data(first_information) != canonical_information_data(
        second_information
    )
    assert information_state_fingerprint(first_information) != (
        information_state_fingerprint(second_information)
    )


def test_information_fingerprint_has_a_stable_golden_fixture() -> None:
    information = information_state_from_engine(
        _advance(create_game("information-fingerprint-golden"), 4)
    )
    assert information_state_fingerprint(information) == (
        "11f16230d24a1707966567d7b63016ff65389471f72979179fb8efa4d40f79b4"
    )


def test_complete_game_information_tracks_public_history_for_both_roles() -> None:
    state = create_game("information-complete-history")
    observed_players: set[EnginePlayer] = set()
    observed_dealer_statuses: set[bool] = set()

    while state.status is not EngineStatus.GAME_COMPLETE:
        if state.status is EngineStatus.ROUND_COMPLETE:
            state = advance_after_round(state)
            continue

        information = information_state_from_engine(state)
        public_history = public_history_from_engine(state)
        observed_players.add(information.player)
        observed_dealer_statuses.add(information.player is information.dealer)

        assert information.completed_rounds == public_history.completed_rounds
        assert len(information.completed_rounds) == information.round_number - 1
        assert information.total_scores == PlayerValues(
            queen=sum(
                record.round_scores.queen for record in information.completed_rounds
            ),
            king=sum(
                record.round_scores.king for record in information.completed_rounds
            ),
        )
        completed_cards = tuple(
            card
            for record in information.completed_rounds
            for card in record.coffin
        )
        current_cards = tuple(card for card in information.coffin if card is not None)
        assert information.played_card_ids == sort_card_ids(
            (*completed_cards, *current_cards)
        )

        assert state.active_player is not None
        state = apply_move(state, legal_moves(state, state.active_player)[0]).state

    assert observed_players == set(EnginePlayer)
    assert observed_dealer_statuses == {False, True}


def test_information_state_does_not_encode_unseen_card_locations() -> None:
    state = create_game("information-hidden-locations")
    information = information_state_from_engine(state)
    serialized = canonical_information_data(information)
    assert set(serialized).isdisjoint(
        {"opponent_hand", "stock", "stock_order", "engine_seed", "hands"}
    )
    assert "unseen_card_ids" in serialized
    assert information.opponent_remaining_count == 4
    assert information.stock_count == 45


def test_information_state_validation_rejects_partition_mask_and_played_order() -> None:
    information = information_state_from_engine(
        _advance(create_game("information-invalid"), 2)
    )
    with pytest.raises(ValueError, match="partition"):
        replace(
            information,
            unseen_card_ids=information.unseen_card_ids[:-1],
        )
    with pytest.raises(ValueError, match="legal mask"):
        replace(
            information,
            legal_mask=tuple(
                tuple(False for _ in row) for row in information.legal_mask
            ),
        )
    with pytest.raises(ValueError, match="played cards"):
        replace(
            information,
            played_card_ids=tuple(reversed(information.played_card_ids)),
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (
            lambda information: replace(
                information,
                active_player=other_player(information.player),
            ),
            "active player",
        ),
        (
            lambda information: replace(
                information,
                opponent_remaining_count=information.opponent_remaining_count + 1,
            ),
            "opponent remaining count",
        ),
        (
            lambda information: replace(
                information,
                current_round_moves=(
                    replace(information.current_round_moves[0], turn_number=2),
                    *information.current_round_moves[1:],
                ),
            ),
            "public move order",
        ),
    ),
    ids=("actor-mismatch", "count-mismatch", "public-history-corruption"),
)
def test_information_state_validation_rejects_distinct_invariant_failures(
    mutation,
    message: str,
) -> None:
    information = information_state_from_engine(
        _advance(create_game("information-invalid-families"), 2)
    )
    with pytest.raises(ValueError, match=message):
        mutation(information)


def test_player_view_retains_canonical_hand_slots_for_engine_resolution() -> None:
    state = _advance(create_game("information-slots"), 5)
    information = information_state_from_engine(state)
    occupied = tuple(card for card in information.own_hand if card is not None)
    assert occupied == sort_card_ids(occupied)
    assert len(information.own_hand) == 4

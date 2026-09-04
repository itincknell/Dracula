"""Actor-visible information-state and privacy invariants."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, fields, replace

import pytest

from dracula.cards import CARD_IDS, sort_card_ids
from dracula.engine import (
    EnginePlayer,
    apply_move,
    create_game,
    legal_moves,
    other_player,
)
from dracula.search.information import (
    PublicGameHistory,
    PublicPlayedMove,
    SearchInformationState,
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
        assert all(information.own_hand[index // 8] is not None for index in legal_indexes)
        if placement < 7:
            assert state.active_player is not None
            state = apply_move(state, legal_moves(state, state.active_player)[0]).state


def test_public_history_and_information_fingerprint_are_deterministic() -> None:
    state = _advance(create_game("information-repeat"), 4)
    first = information_state_from_engine(state)
    second = information_state_from_engine(state)
    assert first == second
    assert public_history_from_engine(state) == public_history_from_engine(state)
    assert information_state_fingerprint(first) == information_state_fingerprint(second)
    assert set(canonical_information_data(first)).isdisjoint(
        {"seed", "stock", "opponent_hand", "engine_state", "search_tree"}
    )


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


def test_information_state_validation_rejects_private_or_inconsistent_substitutions() -> None:
    information = information_state_from_engine(create_game("information-invalid"))
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


def test_player_view_retains_canonical_hand_slots_for_engine_resolution() -> None:
    state = _advance(create_game("information-slots"), 5)
    information = information_state_from_engine(state)
    occupied = tuple(card for card in information.own_hand if card is not None)
    assert occupied == sort_card_ids(occupied)
    assert len(information.own_hand) == 4
    assert isinstance(information, SearchInformationState)

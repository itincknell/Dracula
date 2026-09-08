"""Protect card identity, local randomness, and deterministic dealing."""

from __future__ import annotations

import hashlib

import pytest

from dracula.game.cards import (
    CARD_BY_ID,
    CARD_IDS,
    CARD_INDEX_BY_ID,
    CARDS,
    Color,
    Suit,
    card_by_index,
)
from dracula.game.engine import (
    EnginePlayer,
    create_initial_deal,
    deal_round,
    initial_dealer,
    shuffled_deck,
)
from dracula.randomness import deterministic_random, stable_seed

EXPECTED_CARD_IDS = (
    "AC", "2C", "3C", "4C", "5C", "6C", "7C", "8C", "9C", "10C", "JC", "QC", "KC",
    "AD", "2D", "3D", "4D", "5D", "6D", "7D", "8D", "9D", "10D", "JD", "QD", "KD",
    "AH", "2H", "3H", "4H", "5H", "6H", "7H", "8H", "9H", "10H", "JH", "QH", "KH",
    "AS", "2S", "3S", "4S", "5S", "6S", "7S", "8S", "9S", "10S", "JS", "QS", "KS",
    "V1", "V2",
)


def test_card_ids_and_indexes_match_the_contract_exactly() -> None:
    assert CARD_IDS == EXPECTED_CARD_IDS
    assert len(CARDS) == len(CARD_BY_ID) == len(CARD_INDEX_BY_ID) == 54
    for expected_index, expected_id in enumerate(EXPECTED_CARD_IDS):
        assert CARDS[expected_index].card_id == expected_id
        assert CARD_BY_ID[expected_id].index == expected_index
        assert CARD_INDEX_BY_ID[expected_id] == expected_index
        assert card_by_index(expected_index) is CARDS[expected_index]


def test_card_metadata_preserves_rules_relevant_identity() -> None:
    assert CARD_BY_ID["QC"].suit is Suit.CLUBS
    assert CARD_BY_ID["QC"].color is Color.BLACK
    assert CARD_BY_ID["QC"].horizontal_value == 10
    assert CARD_BY_ID["QC"].vertical_value == 0
    assert CARD_BY_ID["KS"].horizontal_value == 0
    assert CARD_BY_ID["KS"].vertical_value == 10
    assert CARD_BY_ID["V1"].is_vampire
    assert CARD_BY_ID["V2"].is_vampire


def test_local_seed_and_random_generator_are_repeatable() -> None:
    assert stable_seed("deal", 12) == 2589705311707625369
    assert stable_seed("deal", 12) != stable_seed("deal", 13)
    first = deterministic_random("fixture", 9)
    second = deterministic_random("fixture", 9)
    assert [first.randrange(1000) for _ in range(20)] == [
        second.randrange(1000) for _ in range(20)
    ]


def test_golden_shuffle_and_initial_dealer_are_stable() -> None:
    seed = "engine-contract-fixture-1"
    deck = shuffled_deck(seed)
    assert hashlib.sha256(",".join(deck).encode()).hexdigest() == (
        "ce2dce5957b4f855994007f264fa41864bd4e0bee6b41dbedebfe04d3074260d"
    )
    assert set(deck) == set(CARD_IDS)
    assert initial_dealer(seed) is EnginePlayer.QUEEN


@pytest.mark.parametrize("dealer", [EnginePlayer.QUEEN, EnginePlayer.KING])
def test_pair_deal_always_gives_the_first_pair_to_the_non_dealer(
    dealer: EnginePlayer,
) -> None:
    stock = ("AC", "2C", "3C", "4C", "5C", "6C", "7C", "8C", "9C")
    deal = deal_round(stock, dealer)
    assert deal.hand_for(deal.non_dealer) == ("AC", "2C", "5C", "6C")
    assert deal.hand_for(dealer) == ("3C", "4C", "7C", "8C")
    assert deal.center_card == "9C"
    assert deal.remaining_stock == ()


def test_golden_initial_pair_deal_and_canonical_hand_sort() -> None:
    deal = create_initial_deal("engine-contract-fixture-1")
    assert deal.dealer is EnginePlayer.QUEEN
    assert deal.non_dealer is EnginePlayer.KING
    assert deal.queen_hand == ("6C", "6D", "JD", "KS")
    assert deal.king_hand == ("2D", "5D", "4S", "8S")
    assert deal.center_card == "5C"
    assert len(deal.remaining_stock) == 45
    assert deal.remaining_stock[0] == "AS"
    assert deal.remaining_stock[-1] == "7C"

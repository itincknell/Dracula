"""Golden tests for card identity, deterministic randomness, and dealing."""

from __future__ import annotations

import hashlib

import pytest

from dracula.cards import (
    CARD_BY_ID,
    CARD_IDS,
    CARD_INDEX_BY_ID,
    CARDS,
    Color,
    Suit,
    card_by_index,
)
from dracula.engine import (
    INITIAL_DEALER_NAMESPACE,
    SHUFFLE_NAMESPACE,
    EnginePlayer,
    create_initial_deal,
    deal_round,
    initial_dealer,
    shuffled_deck,
)
from dracula.randomness import Sha256CounterStream, derive_seed

EXPECTED_CARD_IDS = (
    "AC",
    "2C",
    "3C",
    "4C",
    "5C",
    "6C",
    "7C",
    "8C",
    "9C",
    "10C",
    "JC",
    "QC",
    "KC",
    "AD",
    "2D",
    "3D",
    "4D",
    "5D",
    "6D",
    "7D",
    "8D",
    "9D",
    "10D",
    "JD",
    "QD",
    "KD",
    "AH",
    "2H",
    "3H",
    "4H",
    "5H",
    "6H",
    "7H",
    "8H",
    "9H",
    "10H",
    "JH",
    "QH",
    "KH",
    "AS",
    "2S",
    "3S",
    "4S",
    "5S",
    "6S",
    "7S",
    "8S",
    "9S",
    "10S",
    "JS",
    "QS",
    "KS",
    "V1",
    "V2",
)

GOLDEN_GAME_SEED = "engine-contract-fixture-1"
GOLDEN_DECK = (
    "4D",
    "10S",
    "10H",
    "9C",
    "AS",
    "6H",
    "3H",
    "8H",
    "7H",
    "2S",
    "V2",
    "7C",
    "2D",
    "AH",
    "9S",
    "6S",
    "4C",
    "JS",
    "5C",
    "3C",
    "6C",
    "JC",
    "JH",
    "V1",
    "7D",
    "AD",
    "QH",
    "6D",
    "QD",
    "7S",
    "KH",
    "QC",
    "8S",
    "JD",
    "KC",
    "AC",
    "4S",
    "5D",
    "3D",
    "10D",
    "3S",
    "8D",
    "2C",
    "2H",
    "4H",
    "KS",
    "9H",
    "KD",
    "QS",
    "5H",
    "9D",
    "8C",
    "5S",
    "10C",
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


def test_seed_derivation_is_repeatable_and_namespace_separated() -> None:
    first = derive_seed("dracula-test-first-v1", "root", "0")
    repeated = derive_seed("dracula-test-first-v1", "root", "0")
    second_namespace = derive_seed("dracula-test-second-v1", "root", "0")

    assert first == repeated
    assert first != second_namespace
    assert len(first) == 32

    first_stream = Sha256CounterStream(first)
    repeated_stream = Sha256CounterStream(repeated)
    second_stream = Sha256CounterStream(second_namespace)
    first_blocks = tuple(first_stream.next_block() for _ in range(4))
    assert first_blocks == tuple(repeated_stream.next_block() for _ in range(4))
    assert first_blocks != tuple(second_stream.next_block() for _ in range(4))


@pytest.mark.parametrize(
    ("namespace", "components"),
    (
        ("dracula-invalid", ("root",)),
        ("dracula-invalid-v1\0suffix", ("root",)),
        ("dracula-valid-v1", ("root\0suffix",)),
    ),
)
def test_seed_fields_reject_unversioned_namespaces_and_nul(
    namespace: str, components: tuple[str, ...]
) -> None:
    with pytest.raises(ValueError):
        derive_seed(namespace, *components)


class _ControlledStream(Sha256CounterStream):
    __slots__ = ("_blocks",)

    def __init__(self, blocks: tuple[bytes, ...]) -> None:
        super().__init__(bytes(32))
        self._blocks = iter(blocks)

    def next_block(self) -> bytes:
        return next(self._blocks)


def test_randbelow_stays_in_bounds_and_rejects_the_biased_tail() -> None:
    stream = Sha256CounterStream(derive_seed("dracula-randbelow-test-v1", "root"))
    for upper_bound in (1, 2, 3, 7, 32, 54):
        assert all(0 <= stream.randbelow(upper_bound) < upper_bound for _ in range(100))

    # For n=3, the maximum 256-bit integer is the one-element biased tail.
    # Consuming the following block proves rejection occurs before modulo.
    maximum = ((1 << 256) - 1).to_bytes(32, "big")
    accepted = (1).to_bytes(32, "big")
    controlled = _ControlledStream((maximum, accepted))
    assert controlled.randbelow(3) == 1

    with pytest.raises(ValueError):
        stream.randbelow(0)
    with pytest.raises(ValueError):
        stream.randbelow((1 << 256) + 1)


def test_golden_shuffle_and_initial_dealer_match_the_contract() -> None:
    assert derive_seed(SHUFFLE_NAMESPACE, GOLDEN_GAME_SEED).hex() == (
        "47f9be37aef2409c1b4bf610dc40012aa03b58ceeb775314fec9312e43b22aef"
    )
    assert derive_seed(INITIAL_DEALER_NAMESPACE, GOLDEN_GAME_SEED).hex() == (
        "3736def5b5df2e7f99d7e6b6a60dc83303463feae387d4cdb8342f61124a87db"
    )
    deck = shuffled_deck(GOLDEN_GAME_SEED)
    assert deck == GOLDEN_DECK
    assert hashlib.sha256(",".join(deck).encode("utf-8")).hexdigest() == (
        "f4d82c3e12ab9d6cf37d777400e69a702e751ced8d5215d1716248a2aa66583f"
    )
    assert initial_dealer(GOLDEN_GAME_SEED) is EnginePlayer.KING


@pytest.mark.parametrize("dealer", [EnginePlayer.QUEEN, EnginePlayer.KING])
def test_pair_deal_always_gives_the_first_pair_to_the_non_dealer(
    dealer: EnginePlayer,
) -> None:
    stock = ("AC", "2C", "3C", "4C", "5C", "6C", "7C", "8C", "9C")
    deal = deal_round(stock, dealer)

    # Hand slots reflect canonical card order, while ownership remains tied to
    # stock positions 0-1/4-5 and 2-3/6-7 rather than player identity.
    assert deal.hand_for(deal.non_dealer) == ("AC", "2C", "5C", "6C")
    assert deal.hand_for(dealer) == ("3C", "4C", "7C", "8C")
    assert deal.center_card == "9C"
    assert deal.remaining_stock == ()


def test_golden_initial_pair_deal_and_canonical_hand_sort() -> None:
    deal = create_initial_deal(GOLDEN_GAME_SEED)
    assert deal.dealer is EnginePlayer.KING
    assert deal.non_dealer is EnginePlayer.QUEEN
    assert deal.queen_hand == ("4D", "6H", "AS", "10S")
    assert deal.king_hand == ("9C", "3H", "8H", "10H")
    assert deal.center_card == "7H"
    assert len(deal.remaining_stock) == 45
    assert deal.remaining_stock[0] == "2S"
    assert deal.remaining_stock[-1] == "10C"

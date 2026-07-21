"""Canonical Dracula card definitions and indexes."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType

CARD_SCHEMA_VERSION = "dracula-cards-v1"

RANKS = ("A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K")


class Suit(StrEnum):
    CLUBS = "C"
    DIAMONDS = "D"
    HEARTS = "H"
    SPADES = "S"


class Color(StrEnum):
    BLACK = "black"
    RED = "red"


@dataclass(frozen=True, slots=True)
class Card:
    card_id: str
    index: int
    rank: str | None
    suit: Suit | None

    @property
    def is_vampire(self) -> bool:
        return self.suit is None

    @property
    def color(self) -> Color | None:
        if self.suit in (Suit.CLUBS, Suit.SPADES):
            return Color.BLACK
        if self.suit in (Suit.DIAMONDS, Suit.HEARTS):
            return Color.RED
        return None

    @property
    def horizontal_value(self) -> int:
        if self.rank == "A":
            return 1
        if self.rank is not None and self.rank.isdecimal():
            return int(self.rank)
        if self.rank == "Q":
            return 10
        return 0

    @property
    def vertical_value(self) -> int:
        if self.rank == "A":
            return 1
        if self.rank is not None and self.rank.isdecimal():
            return int(self.rank)
        if self.rank == "K":
            return 10
        return 0


def _build_cards() -> tuple[Card, ...]:
    cards: list[Card] = []
    for suit in Suit:
        for rank in RANKS:
            cards.append(Card(f"{rank}{suit.value}", len(cards), rank, suit))
    cards.extend(
        (
            Card("V1", len(cards), None, None),
            Card("V2", len(cards) + 1, None, None),
        )
    )
    return tuple(cards)


CARDS = _build_cards()
CARD_IDS = tuple(card.card_id for card in CARDS)
CARD_COUNT = len(CARDS)
CARD_BY_ID: Mapping[str, Card] = MappingProxyType({card.card_id: card for card in CARDS})
CARD_INDEX_BY_ID: Mapping[str, int] = MappingProxyType(
    {card.card_id: card.index for card in CARDS}
)


def card_by_id(card_id: str) -> Card:
    try:
        return CARD_BY_ID[card_id]
    except KeyError as error:
        raise ValueError(f"unknown card ID: {card_id!r}") from error


def card_by_index(index: int) -> Card:
    if type(index) is not int or not 0 <= index < CARD_COUNT:
        raise ValueError(f"card index must be between 0 and {CARD_COUNT - 1}")
    return CARDS[index]


def sort_card_ids(card_ids: Iterable[str]) -> tuple[str, ...]:
    cards = tuple(card_by_id(card_id) for card_id in card_ids)
    return tuple(card.card_id for card in sorted(cards, key=lambda card: card.index))

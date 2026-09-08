"""Create deterministic decks and deal one Dracula round.

The functions here preserve the game's packet order and canonical hand order.
They return deal data but do not construct or mutate complete engine states.
"""

from __future__ import annotations

from collections.abc import Sequence

from dracula.game.cards import CARD_IDS, card_by_id, sort_card_ids
from dracula.game.types import (
    CENTER_GRID_INDEX,
    COFFIN_SIZE,
    HAND_SIZE,
    Coffin,
    EnginePlayer,
    Hand,
    PlayerValues,
    RoundDeal,
    other_player,
)
from dracula.randomness import deterministic_random


def _game_randomization(game_seed: str) -> tuple[tuple[str, ...], EnginePlayer]:
    """Produce the complete deck and dealer from one game-local generator."""

    generator = deterministic_random(game_seed)
    deck = list(CARD_IDS)
    generator.shuffle(deck)
    dealer = (EnginePlayer.QUEEN, EnginePlayer.KING)[generator.randrange(2)]
    return tuple(deck), dealer


def shuffled_deck(game_seed: str) -> tuple[str, ...]:
    """Return the reproducible deck for ``game_seed``."""

    return _game_randomization(game_seed)[0]


def initial_dealer(game_seed: str) -> EnginePlayer:
    """Return the reproducible initial dealer for ``game_seed``."""

    return _game_randomization(game_seed)[1]


def deal_round(stock: Sequence[str], dealer: EnginePlayer) -> RoundDeal:
    """Deal one round from the stock front while preserving the remaining order."""

    dealer = EnginePlayer(dealer)
    stock_cards = tuple(stock)
    if len(stock_cards) < 9:
        raise ValueError("a round requires at least nine stock cards")
    for card_id in stock_cards:
        card_by_id(card_id)
    if len(set(stock_cards)) != len(stock_cards):
        raise ValueError("stock cannot contain duplicate cards")

    # Two-card packets alternate, beginning with the non-dealer. Naming the
    # slices keeps this rules-defined order visible instead of hiding it in
    # concatenation arithmetic.
    non_dealer = other_player(dealer)
    first_non_dealer_packet = stock_cards[0:2]
    first_dealer_packet = stock_cards[2:4]
    second_non_dealer_packet = stock_cards[4:6]
    second_dealer_packet = stock_cards[6:8]
    non_dealer_hand = sort_card_ids(
        first_non_dealer_packet + second_non_dealer_packet
    )
    dealer_hand = sort_card_ids(first_dealer_packet + second_dealer_packet)
    hands = PlayerValues(
        queen=non_dealer_hand if non_dealer is EnginePlayer.QUEEN else dealer_hand,
        king=non_dealer_hand if non_dealer is EnginePlayer.KING else dealer_hand,
    )
    return RoundDeal(
        dealer=dealer,
        queen_hand=hands.queen,
        king_hand=hands.king,
        center_card=stock_cards[8],
        remaining_stock=stock_cards[9:],
    )


def create_initial_deal(game_seed: str) -> RoundDeal:
    """Create round one's deal from one deterministic game randomization."""

    deck, dealer = _game_randomization(game_seed)
    return deal_round(deck, dealer)


def as_hand(cards: tuple[str, ...]) -> Hand:
    """Convert four ordered card IDs to the engine's fixed hand shape."""

    if len(cards) != HAND_SIZE:
        raise ValueError("a hand must contain four cards")
    return cards  # type: ignore[return-value]


def starting_coffin(center_card: str) -> Coffin:
    """Create an empty coffin whose center contains the dealt first nail."""

    values: list[str | None] = [None] * COFFIN_SIZE
    values[CENTER_GRID_INDEX] = center_card
    return tuple(values)  # type: ignore[return-value]

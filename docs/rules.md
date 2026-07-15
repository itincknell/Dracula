# Dracula rules

## Ruleset

This project is based on the standard two-player game created by David Parlett.
The application uses **round** for what the published rules call a **deal**. A
game consists of six rounds. It uses the project variant in which the two
Vampires are shuffled into the deck and dealt as ordinary cards.

## Components

- One standard 52-card deck plus two Jokers, for 54 cards total.
- The Jokers are called **Vampires**.
- One shared 3×3 layout, called the **coffin**.
- A running score for each player.

## Setup

1. Shuffle all 54 cards, including both Vampires, to form the stock.
2. Select the first dealer using the game seed.
3. Before the first round, the human player chooses a scoring direction:
   - **Queen:** horizontal rows.
   - **King:** vertical columns.
4. The LLM opponent receives the opposite direction. These directions remain
   fixed for all six rounds and do not depend on the dealer.

The deal alternates each round, so each player deals three times.

## Dealing a round

1. Deal four private cards to each player.
2. Place the next stock card face up in the center of the coffin. This is the
   first nail.
3. Leave the remaining stock face down without changing its order.

There is no draw during a round. A player's hand is private. The coffin, played
cards, scores, dealer, and scoring directions are public. An unplayed Vampire is
hidden when it is in a hand or the stock.

## Playing a round

The non-dealer plays first. Players alternate until all eight empty coffin
positions are filled, giving each player four turns.

On a turn, the player places one card from their hand into a vacant coffin
position. A placement is legal when:

- The position is one of the nine cells in the 3×3 coffin.
- The position is empty.
- It shares an edge with a card already in the coffin. Diagonal contact alone
  is not sufficient.

A Vampire follows the same placement rules as every other card. It may be dealt
to either player or appear face up as the center card.

## Scoring a line

Each completed coffin has three horizontal rows and three vertical columns.
Only rows are relevant to the Queen player; only columns are relevant to the
King player.

### Card values

| Card | Horizontal row | Vertical column |
| --- | ---: | ---: |
| Ace | 1 | 1 |
| 2 through 10 | Face value | Face value |
| Jack | 0 | 0 |
| Queen | 10 | 0 |
| King | 0 | 10 |

The suit and color of a zero-valued face card still count when determining a
line multiplier.

### Multiplier

Add the three card values, then apply the highest applicable multiplier:

| Cards in the line | Multiplier |
| --- | ---: |
| Three cards of the same suit | ×5 |
| Otherwise, three cards of the same color | ×3 |
| Otherwise, at least two cards of the same suit | ×2 |
| None of the above | ×1 |

The standard rules apply one multiplier; they are not compounded.

### Vampire

A line containing a Vampire scores zero. The Vampire therefore sets both its
horizontal row and vertical column to zero.

## Selecting each player's round score

1. Calculate the Queen player's three row scores and sort them from highest to
   lowest.
2. Calculate the King player's three column scores and sort them from highest to
   lowest.
3. If the two highest scores differ, each player records their highest score.
4. If the highest scores tie, each player instead records their second-highest
   score.
5. If the second-highest scores also tie, each player records their third-highest
   score, even if those scores also tie.

## Ending a round

Record both round scores and remove the nine coffin cards from play. Do not
return them to the stock. Both players have played all four cards. The other
player becomes dealer and deals the next round from the existing stock.

## Ending the game

After six rounds, add each player's six round scores. The higher total wins. If
the totals tie, the player with the higher sixth-round score wins. If both the
totals and sixth-round scores tie, the game is a tie. The stock is empty because
each of the six rounds uses nine cards.

## Scoring examples

### Basic suit multiplier

A horizontal line of 8♥, 3♥, and 8♣ has a base value of 19. Two Hearts produce a
×2 multiplier, so the row scores **38**.

### Color takes precedence over a suit pair

A horizontal line of 8♥, 9♥, and 10♦ has a base value of 27. All three cards are
red, so the highest applicable multiplier is ×3 and the row scores **81**.

### Three of one suit

A horizontal line of 8♥, 9♥, and 10♥ has a base value of 27 and a ×5 multiplier,
so the row scores **135**.

### Directional face-card value

Q♥, 5♥, and 9♣ score 48 as a row: `(10 + 5 + 9) × 2`. The same cards score 28 as
a column: `(0 + 5 + 9) × 2`. The Queen's Heart still supplies the matching suit
when its column value is zero.

### Tied best lines

If the Queen player's sorted row scores are `[30, 24, 8]` and the King player's
sorted column scores are `[30, 18, 11]`, the tied scores of 30 are not recorded.
The round scores are **24** for Queen and **18** for King.

## Sources

- [David Parlett's current rules](https://www.parlettgames.uk/oricards/dracula.html)
- [Freeze-Dried Games Pack summary](https://kevan.org/fdgp/index.php?view=2#Dracula)

The project rules for shuffling and dealing the Vampires and for human role
selection supersede the corresponding setup rules in Parlett's current version.

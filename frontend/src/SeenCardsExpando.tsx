/**
 * Presents the player's optional ledger of every publicly seen card.
 * The ledger derives its contents from the current public game view and keeps
 * its own expanded/collapsed presentation state. It has no gameplay effects.
 */
import { useState } from "react";

import { cardName } from "./cardAssets";
import type { HumanGameView } from "./gameView";

const CARD_RANKS = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"] as const;
const CARD_SUITS = [
  { id: "C", name: "Clubs" },
  { id: "D", name: "Diamonds" },
  { id: "H", name: "Hearts" },
  { id: "S", name: "Spades" },
] as const;

export function seenCardIds(view: HumanGameView): ReadonlySet<string> {
  const cards = new Set<string>();
  for (const round of view.completed_rounds) {
    for (const cardId of round.coffin) cards.add(cardId);
  }
  if (view.pending_round_result !== null) {
    for (const cardId of view.pending_round_result.coffin) cards.add(cardId);
  }
  for (const cardId of view.coffin) {
    if (cardId !== null) cards.add(cardId);
  }
  for (const cardId of view.human_hand) {
    if (cardId !== null) cards.add(cardId);
  }
  return cards;
}

export function SeenCardsExpando({ view }: { view: HumanGameView }) {
  const [expanded, setExpanded] = useState(false);
  const seen = seenCardIds(view);
  const panelId = "seen-card-ledger";

  return (
    <section className="cheat-sheet-window" aria-label="Cheat sheet">
      <button
        className="cheat-sheet-expando"
        type="button"
        aria-controls={panelId}
        aria-expanded={expanded}
        onClick={() => setExpanded((current) => !current)}
      >
        <span>Cheat Sheet</span>
        <span aria-hidden="true">{expanded ? "−" : "+"}</span>
      </button>
      {expanded ? (
        <div className="seen-card-ledger" id={panelId}>
          <div className="seen-card-grid-scroll">
            <table className="seen-card-grid">
              <caption>{seen.size} of 54 cards seen</caption>
              <thead>
                <tr>
                  <th scope="col" aria-label="Suit" />
                  {CARD_RANKS.map((rank) => <th scope="col" key={rank}>{rank}</th>)}
                </tr>
              </thead>
              <tbody>
                {CARD_SUITS.map((suit) => (
                  <tr key={suit.id}>
                    <th scope="row">{suit.name}</th>
                    {CARD_RANKS.map((rank) => {
                      const cardId = `${rank}${suit.id}`;
                      const cardSeen = seen.has(cardId);
                      return (
                        <td key={cardId} data-seen={cardSeen} aria-label={`${cardName(cardId)}: ${cardSeen ? "seen" : "not seen"}`}>
                          <span aria-hidden="true">{cardSeen ? "X" : ""}</span>
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="vampire-ledger" aria-label="Vampires">
            {["V1", "V2"].map((cardId) => {
              const cardSeen = seen.has(cardId);
              return (
                <div className="vampire-ledger-cell" key={cardId} data-seen={cardSeen} aria-label={`${cardName(cardId)}: ${cardSeen ? "seen" : "not seen"}`}>
                  <strong>{cardId}</strong><span aria-hidden="true">{cardSeen ? "X" : ""}</span>
                </div>
              );
            })}
          </div>
        </div>
      ) : null}
    </section>
  );
}

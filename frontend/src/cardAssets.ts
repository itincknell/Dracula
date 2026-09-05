/**
 * Maps canonical card identifiers to bundled image assets and readable names.
 * The generated asset manifest is the single source for card URLs used by the
 * board, hand, scoring animation, and accessibility labels.
 */
import sourceAssets from "../card-assets.json";

export const CANONICAL_CARD_IDS = Object.freeze(Object.keys(sourceAssets));

const CARD_ASSET_URLS = Object.freeze(
  Object.fromEntries(
    CANONICAL_CARD_IDS.map((cardId) => [
      cardId,
      `${import.meta.env.BASE_URL}cards/${cardId}.${cardId === "V1" || cardId === "V2" ? "jpg" : "png"}`,
    ]),
  ),
);

const rankNames: Readonly<Record<string, string>> = {
  A: "Ace",
  J: "Jack",
  Q: "Queen",
  K: "King",
};

const suitNames: Readonly<Record<string, string>> = {
  C: "Clubs",
  D: "Diamonds",
  H: "Hearts",
  S: "Spades",
};

export function cardAsset(cardId: string): string {
  const asset = CARD_ASSET_URLS[cardId];
  if (asset === undefined) throw new TypeError(`unknown canonical card ID: ${cardId}`);
  return asset;
}

export function cardName(cardId: string): string {
  if (cardId === "V1") return "Vampire 1";
  if (cardId === "V2") return "Vampire 2";
  const suit = suitNames[cardId.at(-1) ?? ""];
  const rankId = cardId.slice(0, -1);
  const rank = rankNames[rankId] ?? rankId;
  if (suit === undefined || rank.length === 0 || !CANONICAL_CARD_IDS.includes(cardId)) {
    throw new TypeError(`unknown canonical card ID: ${cardId}`);
  }
  return `${rank} of ${suit}`;
}

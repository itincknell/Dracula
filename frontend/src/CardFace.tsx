/**
 * Renders one canonical card image with its accessible card name.
 * This intentionally small component centralizes asset lookup and decorative
 * image handling for every gameplay and scoring surface.
 */
import { cardAsset, cardName } from "./cardAssets";

export function CardFace({ cardId, decorative = false }: { cardId: string; decorative?: boolean }) {
  return <img className="card-face" src={cardAsset(cardId)} alt={decorative ? "" : cardName(cardId)} draggable={false} />;
}

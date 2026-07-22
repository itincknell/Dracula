import { cardAsset, cardName } from "./cardAssets";

export function CardFace({ cardId, decorative = false }: { cardId: string; decorative?: boolean }) {
  return <img className="card-face" src={cardAsset(cardId)} alt={decorative ? "" : cardName(cardId)} draggable={false} />;
}

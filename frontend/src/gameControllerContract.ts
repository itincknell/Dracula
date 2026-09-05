/**
 * Defines the observable controller interface consumed by the React view.
 * It separates gameplay commands and subscriptions from the stateless store's
 * transport, replay persistence, and narration implementation.
 */
import { useSyncExternalStore } from "react";

import type { LegalMove, Player } from "./contractPrimitives";
import type { HumanGameView, PresentationError } from "./gameView";

export type PendingAction =
  | "create_game"
  | "load_game"
  | "human_move"
  | "opponent_turn"
  | "round_advance";

export interface PresentationState {
  pending: PendingAction | null;
  selectedHandSlot: number | null;
  error: PresentationError | null;
}

export interface NarrationState {
  enabled: boolean;
  pending: boolean;
  messages: readonly string[];
}

export interface GameStoreState {
  view: HumanGameView | null;
  presentation: PresentationState;
  narration: NarrationState;
}

export interface GameControllerContract {
  getSnapshot(): GameStoreState;
  subscribe(listener: () => void): () => void;
  createGame(humanRole: Player): Promise<HumanGameView | null>;
  loadGame(): Promise<HumanGameView | null>;
  clearError(): void;
  clearGame(): void;
  selectCard(handSlot: number): boolean;
  legalMoveAt(position: number): LegalMove | null;
  playPosition(position: number): Promise<boolean>;
  advanceRound(): Promise<boolean>;
  revealNarration(): void;
}

export function useGameStore(controller: GameControllerContract): GameStoreState {
  return useSyncExternalStore(controller.subscribe, controller.getSnapshot, controller.getSnapshot);
}

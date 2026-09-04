import { useSyncExternalStore } from "react";

import type { LegalMove, Player } from "./contractPrimitives";
import type { ApiErrorResponse, HumanGameView } from "./statefulContracts";

export type PendingAction =
  | "create_game"
  | "load_game"
  | "human_move"
  | "opponent_turn"
  | "round_advance";

export interface PresentationState {
  pending: PendingAction | null;
  selectedHandSlot: number | null;
  error: ApiErrorResponse | null;
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
  loadGame(gameId: string): Promise<HumanGameView | null>;
  clearError(): void;
  clearGame(): void;
  selectCard(handSlot: number): boolean;
  legalMoveAt(position: number): LegalMove | null;
  playPosition(position: number): Promise<boolean>;
  playMove(moveId: string): Promise<boolean>;
  progressOpponent(options?: { retryFailed?: boolean }): Promise<void>;
  advanceRound(): Promise<boolean>;
  revealNarration(): void;
}

export function useGameStore(controller: GameControllerContract): GameStoreState {
  return useSyncExternalStore(controller.subscribe, controller.getSnapshot, controller.getSnapshot);
}

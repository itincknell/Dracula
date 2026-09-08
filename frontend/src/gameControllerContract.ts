/**
 * Defines the observable controller interface consumed by the React view.
 * It separates gameplay commands and subscriptions from the stateless store's
 * transport, replay persistence, and narration implementation.
 */
import { useSyncExternalStore } from "react";

import type { LegalMove, Player } from "./contractPrimitives";
import type { HumanGameView, PresentationError } from "./gameView";

/**
 * The one frontend operation currently in progress.
 *
 * Creation and loading contact the stateless API; `opponent_turn` is the
 * deliberate visible delay before publishing Dracula's already-authoritative
 * response; and `round_advance`
 * acknowledges the completed scoring presentation.
 */
export type PendingAction =
  | "create_game"
  | "load_game"
  | "opponent_turn"
  | "round_advance";

/** UI state for one in-flight action, selected card, and displayable failure. */
export interface PresentationState {
  /** Names the operation currently disabling conflicting user input. */
  pending: PendingAction | null;
  /** Stable engine hand slot selected by the human, or null if none is selected. */
  selectedHandSlot: number | null;
  error: PresentationError | null;
}

/** Dialogue already accepted for display plus the state of the current request. */
export interface NarrationState {
  /** True only while the separate narration request is outstanding. */
  pending: boolean;
  /** Ordered successful cues for this game; unavailable cues add no entry. */
  messages: readonly string[];
}

/** The immutable snapshot observed by React on each controller notification. */
export interface GameStoreState {
  /** Null before creation and after local recovery data is cleared. */
  view: HumanGameView | null;
  presentation: PresentationState;
  narration: NarrationState;
}

/**
 * Operations available to the React application.
 *
 * The concrete stateless controller owns HTTP calls, replay-envelope storage,
 * opponent timing, and narration. Components receive this smaller boundary so
 * they can render snapshots and submit user intent without manipulating those
 * mechanisms directly. The test controller implements the same contract.
 */
export interface GameControllerContract {
  /** Return the same snapshot object until the controller publishes a change. */
  getSnapshot(): GameStoreState;
  /** Register a change listener and return its unsubscribe function. */
  subscribe(listener: () => void): () => void;
  /** Start a new game and return its first settled public view. */
  createGame(humanRole: Player): Promise<HumanGameView | null>;
  /** Recover the browser's saved game, if any. */
  loadGame(): Promise<HumanGameView | null>;
  clearError(): void;
  /** Forget local recovery and presentation state; it does not call a server session. */
  clearGame(): void;
  /** Select an occupied human hand slot when the current phase permits it. */
  selectCard(handSlot: number): boolean;
  /** Find the legal move pairing the selected card with a coffin position. */
  legalMoveAt(position: number): LegalMove | null;
  /** Preview and submit the selected human card placement. */
  playPosition(position: number): Promise<boolean>;
  /** Acknowledge completed scoring and deal the next round. */
  advanceRound(): Promise<boolean>;
  /** Make a fetched narration cue visible after its coordinated animation point. */
  revealNarration(): void;
}

/**
 * Subscribe React to the controller's external immutable store.
 *
 * `useSyncExternalStore` coordinates snapshot reads with subscription updates,
 * including React's concurrent rendering. The third argument supplies the
 * same deterministic snapshot path if this component is rendered on a server.
 */
export function useGameStore(controller: GameControllerContract): GameStoreState {
  return useSyncExternalStore(controller.subscribe, controller.getSnapshot, controller.getSnapshot);
}

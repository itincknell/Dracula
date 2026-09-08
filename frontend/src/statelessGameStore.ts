/**
 * Owns frontend gameplay state for the stateless production protocol.
 * The store sequences commands, replay persistence, intermediate move display,
 * minimum opponent delay, narration requests, and observable UI state.
 */
import type {
  LegalMove,
  Player,
} from "./contractPrimitives";
import type {
  HumanGameView,
  PresentationError,
} from "./gameView";
import type {
  RecoveryEnvelope,
  StatelessGameResponse,
} from "./statelessContracts";
import type {
  GameControllerContract,
  GameStoreState,
  NarrationState,
  PresentationState,
} from "./gameControllerContract";
import { StatelessApiError, type StatelessApiClient } from "./statelessApi";
import { StatelessNarrationCoordinator } from "./statelessNarration";
import {
  defaultBrowserStorage,
  RecoveryEnvelopeStore,
  type BrowserStorage,
} from "./statelessRecovery";
import {
  humanPlacementPreview,
  opponentOpeningPreview,
} from "./statelessProjection";

export const MINIMUM_DRACULA_TURN_MS = 600;

/** Real browser delay; tests inject an immediate or manually released Promise. */
const defaultDraculaTurnDelay = (): Promise<void> =>
  new Promise((resolve) => window.setTimeout(resolve, MINIMUM_DRACULA_TURN_MS));

/** Construct interaction state with no selection, request, or visible error. */
const emptyPresentation = (): PresentationState => ({
  pending: null,
  selectedHandSlot: null,
  error: null,
});

/** Construct narration state for a new or deliberately cleared game. */
const emptyNarration = (): NarrationState => ({
  pending: false,
  messages: [],
});

/** Convert transport and validated API failures into the small UI error shape. */
function statelessError(error: unknown): PresentationError {
  if (!(error instanceof StatelessApiError)) {
    return {
      message: error instanceof Error ? error.message : "The game service is unavailable.",
      retryable: true,
    };
  }
  return {
    message: error.response.message,
    retryable: error.response.retryable,
  };
}

/** Identify recovery failures that cannot succeed by repeating the same history. */
function invalidSavedHistory(error: unknown): boolean {
  return error instanceof StatelessApiError &&
    !error.response.retryable &&
    ["invalid_history", "validation_error"].includes(
      error.response.code,
    );
}

export class StatelessGameController implements GameControllerContract {
  // React reads this immutable-by-replacement object through getSnapshot().
  // Nested objects are also replaced whenever their values change.
  private state: GameStoreState = {
    view: null,
    presentation: emptyPresentation(),
    narration: emptyNarration(),
  };
  private readonly listeners = new Set<() => void>();
  // React StrictMode can ask for recovery twice; both callers share one replay.
  private loadPromise: Promise<HumanGameView | null> | null = null;
  private envelope: RecoveryEnvelope | null = null;
  private readonly recovery: RecoveryEnvelopeStore;
  private readonly narrationCoordinator: StatelessNarrationCoordinator;

  /** Compose transport, persistent recovery, visible delay, and narration. */
  constructor(
    private readonly api: StatelessApiClient,
    storage: BrowserStorage = defaultBrowserStorage(),
    private readonly draculaTurnDelay: () => Promise<void> = defaultDraculaTurnDelay,
  ) {
    this.recovery = new RecoveryEnvelopeStore(storage);
    this.narrationCoordinator = new StatelessNarrationCoordinator(
      api,
      () => this.state.narration,
      (changes) => this.updateNarration(changes),
    );
  }

  readonly getSnapshot = (): GameStoreState => this.state;

  /** Subscribe one React external-store listener. */
  readonly subscribe = (listener: () => void): (() => void) => {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  };

  /** Replace the snapshot and notify every mounted React subscriber. */
  private publish(state: GameStoreState): void {
    this.state = state;
    this.listeners.forEach((listener) => listener());
  }

  private updatePresentation(changes: Partial<PresentationState>): void {
    this.publish({
      ...this.state,
      presentation: { ...this.state.presentation, ...changes },
    });
  }

  private updateNarration(changes: Partial<NarrationState>): void {
    this.publish({
      ...this.state,
      narration: { ...this.state.narration, ...changes },
    });
  }

  /** Keep the latest accepted envelope in memory and browser storage together. */
  private saveEnvelope(envelope: RecoveryEnvelope): void {
    this.envelope = this.recovery.save(envelope);
  }

  /** Publish a server-computed Dracula opener only after the visible delay. */
  private async holdOpponentOpening(
    response: StatelessGameResponse,
  ): Promise<void> {
    const preview = opponentOpeningPreview(response.game);
    if (preview === null) return;
    // The server has already accepted both opening actions. Publishing a view
    // with only Dracula's move withheld gives the opening the same visible pace
    // as later turns without changing the recovery history.
    this.saveEnvelope(response.envelope);
    this.publish({
      view: preview,
      presentation: {
        pending: "opponent_turn",
        selectedHandSlot: null,
        error: null,
      },
      narration: this.state.narration,
    });
    await this.draculaTurnDelay();
  }

  /** Make one validated server response the new authoritative browser view. */
  private acceptResponse(response: StatelessGameResponse): HumanGameView {
    this.saveEnvelope(response.envelope);
    const view = response.game;
    this.publish({
      view,
      presentation: {
        ...this.state.presentation,
        pending: null,
        selectedHandSlot: null,
        error: null,
      },
      narration: this.state.narration,
    });
    return view;
  }

  /** End the active interaction and expose a safe transport or API error. */
  private reject(error: unknown): void {
    this.publish({
      ...this.state,
      presentation: {
        pending: null,
        selectedHandSlot: null,
        error: statelessError(error),
      },
    });
  }

  /** Return to the start screen with a non-retryable recovery explanation. */
  private rejectRecovery(message: string): void {
    this.publish({
      view: null,
      presentation: {
        ...emptyPresentation(),
        error: { message, retryable: false },
      },
      narration: emptyNarration(),
    });
  }

  /** Remove a replay history the server has proved impossible. */
  private discardInvalidRecovery(): void {
    this.recovery.clear();
    this.envelope = null;
    this.rejectRecovery("The saved game could not be replayed and has been cleared.");
  }

  /** Create a game, pace a possible Dracula opener, then request opening text. */
  async createGame(humanRole: Player): Promise<HumanGameView | null> {
    // A second click while creation or another transition is active must not
    // start a competing game or overwrite its recovery envelope.
    if (this.state.presentation.pending !== null) return null;
    this.narrationCoordinator.cancel();
    this.publish({
      view: null,
      presentation: { ...emptyPresentation(), pending: "create_game" },
      narration: emptyNarration(),
    });
    try {
      const response = await this.api.startGame({ human_role: humanRole });
      await this.holdOpponentOpening(response);
      const view = this.acceptResponse(response);
      this.narrationCoordinator.requestEligible(view, response.envelope, { opening: true });
      return view;
    } catch (error) {
      this.reject(error);
      return null;
    }
  }

  /** Load, replay, and publish the envelope currently stored by this browser. */
  private async restoreGame(): Promise<HumanGameView | null> {
    let envelope: RecoveryEnvelope | null;
    try {
      envelope = this.recovery.load();
    } catch (error) {
      const message = error instanceof Error ? error.message : "The saved game was malformed.";
      this.rejectRecovery(message);
      return null;
    }
    if (envelope === null) {
      this.rejectRecovery("No saved game is available. Start a new game.");
      return null;
    }
    try {
      const response = await this.api.resumeGame({ envelope });
      const view = this.acceptResponse(response);
      this.narrationCoordinator.requestEligible(view, response.envelope, {
        // A role-only history is the initial settled state, so reloading that
        // screen may re-request its optional opening narration.
        opening: response.envelope.history.length === 1,
      });
      return view;
    } catch (error) {
      if (invalidSavedHistory(error)) this.discardInvalidRecovery();
      else this.reject(error);
      return null;
    }
  }

  /**
   * Recover at most once while React StrictMode or multiple components wait.
   * Every concurrent caller receives the same Promise and reconstructed view.
   */
  loadGame(): Promise<HumanGameView | null> {
    if (this.loadPromise !== null) return this.loadPromise;
    this.updatePresentation({ pending: "load_game", error: null });
    this.loadPromise = this.restoreGame().finally(() => {
      this.loadPromise = null;
    });
    return this.loadPromise;
  }

  /** Dismiss the current presentation error without changing gameplay state. */
  clearError(): void {
    this.updatePresentation({ error: null });
  }

  /** Forget this browser's game; the stateless server has no session to delete. */
  clearGame(): void {
    this.recovery.clear();
    this.envelope = null;
    this.narrationCoordinator.cancel();
    this.publish({
      view: null,
      presentation: emptyPresentation(),
      narration: emptyNarration(),
    });
  }

  /** Select a card only when at least one current legal move uses its hand slot. */
  selectCard(handSlot: number): boolean {
    const { view, presentation } = this.state;
    if (
      view === null || presentation.pending !== null || view.phase.kind !== "human_turn" ||
      view.human_hand[handSlot] === null ||
      !view.legal_moves.some((move) => move.hand_slot === handSlot)
    ) return false;
    this.updatePresentation({ selectedHandSlot: handSlot, error: null });
    return true;
  }

  /** Resolve the selected hand slot and clicked coffin cell to a server legal move. */
  legalMoveAt(position: number): LegalMove | null {
    const { view, presentation } = this.state;
    if (view === null || presentation.selectedHandSlot === null) return null;
    return view.legal_moves.find((move) =>
      move.hand_slot === presentation.selectedHandSlot && move.position === position
    ) ?? null;
  }

  /** Submit the legal move at one clicked coffin position, if one exists. */
  async playPosition(position: number): Promise<boolean> {
    const move = this.legalMoveAt(position);
    return move === null ? false : this.playMove(move);
  }

  /**
   * Recheck a move immediately before submission.
   *
   * The view may have changed between a click selecting a card and a later
   * click selecting a destination. Matching every public move field prevents a
   * stale selection from being submitted against the newer envelope.
   */
  private playableMove(move: LegalMove): {
    view: HumanGameView;
    envelope: RecoveryEnvelope;
  } | null {
    const { view, presentation } = this.state;
    const remainsLegal = view?.legal_moves.some((candidate) =>
      candidate.card_id === move.card_id &&
      candidate.hand_slot === move.hand_slot &&
      candidate.position === move.position
    );
    if (
      view === null || !remainsLegal || this.envelope === null ||
      presentation.pending !== null || view.phase.kind !== "human_turn"
    ) return null;
    return { view, envelope: this.envelope };
  }

  /** Publish a reversible local preview of the human placement. */
  private showHumanPlacement(view: HumanGameView, move: LegalMove): void {
    // Show the proposed human choice immediately while the authoritative
    // request and Dracula's minimum delay run. A failure restores `view`.
    this.publish({
      ...this.state,
      view: humanPlacementPreview(view, move),
      presentation: {
        ...this.state.presentation,
        pending: "opponent_turn",
        selectedHandSlot: null,
        error: null,
      },
    });
  }

  /** Coordinate optimistic display, authoritative submission, and Dracula pacing. */
  private async playMove(move: LegalMove): Promise<boolean> {
    const playable = this.playableMove(move);
    if (playable === null) return false;
    const { envelope, view } = playable;
    this.showHumanPlacement(view, move);
    try {
      // Network work and the minimum Dracula delay run together. Fast inference
      // remains legible while slow inference incurs no extra artificial wait.
      const [result] = await Promise.all([
        this.api.applyCommand({
          envelope,
          command: { type: "place", hand_slot: move.hand_slot, position: move.position },
        }),
        this.draculaTurnDelay(),
      ]);
      const accepted = this.acceptResponse(result);
      this.narrationCoordinator.requestEligible(accepted, result.envelope);
      return true;
    } catch (error) {
      // The preview never entered recovery history. Restore the exact prior
      // authoritative view and leave the accepted envelope untouched.
      this.publish({
        ...this.state,
        view,
        presentation: {
          pending: null,
          selectedHandSlot: null,
          error: statelessError(error),
        },
      });
      return false;
    }
  }

  /** Acknowledge scoring, clear its dialogue, and reveal any next-round opener. */
  async advanceRound(): Promise<boolean> {
    const { view, presentation } = this.state;
    if (
      view === null || this.envelope === null || presentation.pending !== null ||
      view.phase.kind !== "scoring"
    ) return false;
    this.updatePresentation({ pending: "round_advance", error: null });
    // Round-transition dialogue belongs to the completed round and disappears
    // only when the user explicitly deals the next one.
    if (view.status === "round_complete") this.narrationCoordinator.clear();
    try {
      const response = await this.api.applyCommand({
        envelope: this.envelope,
        command: { type: "advance_round" },
      });
      await this.holdOpponentOpening(response);
      const accepted = this.acceptResponse(response);
      if (accepted.status === "game_complete") {
        this.narrationCoordinator.requestEligible(accepted, response.envelope);
      }
      return true;
    } catch (error) {
      this.reject(error);
      return false;
    }
  }

  /** Called by the scoring timeline when a prefetched round cue may appear. */
  revealNarration(): void {
    this.narrationCoordinator.reveal();
  }
}

import type {
  LegalMove,
  Player,
} from "./contractPrimitives";
import type {
  ApiErrorResponse,
  HumanGameView,
} from "./statefulContracts";
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
  projectStatelessResponse,
} from "./statelessProjection";

export { RECOVERY_STORAGE_KEY, type BrowserStorage } from "./statelessRecovery";
export { projectStatelessResponse } from "./statelessProjection";

export const MINIMUM_DRACULA_TURN_MS = 600;

const defaultDraculaTurnDelay = (): Promise<void> =>
  new Promise((resolve) => window.setTimeout(resolve, MINIMUM_DRACULA_TURN_MS));

const emptyPresentation = (): PresentationState => ({
  pending: null,
  selectedHandSlot: null,
  error: null,
});

const narrationState = (
  changes: Partial<NarrationState> = {},
): NarrationState => ({
  enabled: true,
  pending: false,
  messages: [],
  ...changes,
});

const networkError = (error: unknown): ApiErrorResponse => ({
  schema_version: "dracula-error-v1",
  code: "dependency_unavailable",
  message: error instanceof Error ? error.message : "The game service is unavailable.",
  retryable: true,
  current_version: null,
  current_game: null,
});

const recoveryError = (message: string): ApiErrorResponse => ({
  schema_version: "dracula-error-v1",
  code: "validation_error",
  message,
  retryable: false,
  current_version: null,
  current_game: null,
});

function statelessError(error: unknown): ApiErrorResponse {
  if (!(error instanceof StatelessApiError)) return networkError(error);
  return {
    schema_version: "dracula-error-v1",
    code: error.response.code === "dependency_unavailable"
      ? "dependency_unavailable"
      : error.response.code === "wrong_turn"
        ? "wrong_turn"
        : error.response.code === "wrong_phase"
          ? "wrong_phase"
          : "validation_error",
    message: error.response.message,
    retryable: error.response.retryable,
    current_version: null,
    current_game: null,
  };
}

export class StatelessGameController implements GameControllerContract {
  private state: GameStoreState = {
    view: null,
    presentation: emptyPresentation(),
    narration: narrationState(),
  };
  private readonly listeners = new Set<() => void>();
  private loadPromise: Promise<HumanGameView | null> | null = null;
  private envelope: RecoveryEnvelope | null = null;
  private readonly recovery: RecoveryEnvelopeStore;
  private readonly narrationCoordinator: StatelessNarrationCoordinator;

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

  readonly subscribe = (listener: () => void): (() => void) => {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  };

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

  private saveEnvelope(envelope: RecoveryEnvelope): void {
    this.envelope = this.recovery.save(envelope);
  }

  private async holdOpponentOpening(response: StatelessGameResponse): Promise<void> {
    const preview = opponentOpeningPreview(projectStatelessResponse(response));
    if (preview === null) return;
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

  private acceptResponse(response: StatelessGameResponse): HumanGameView {
    this.saveEnvelope(response.envelope);
    const view = projectStatelessResponse(response);
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

  async createGame(humanRole: Player): Promise<HumanGameView | null> {
    if (this.state.presentation.pending !== null) return null;
    this.narrationCoordinator.cancel();
    this.publish({
      view: null,
      presentation: { ...emptyPresentation(), pending: "create_game" },
      narration: narrationState(),
    });
    try {
      const result = await this.api.startGame({ human_role: humanRole });
      await this.holdOpponentOpening(result.data);
      const view = this.acceptResponse(result.data);
      this.narrationCoordinator.requestEligible(view, result.data.envelope, { opening: true });
      return view;
    } catch (error) {
      this.reject(error);
      return null;
    }
  }

  loadGame(_gameId: string): Promise<HumanGameView | null> {
    void _gameId;
    if (this.loadPromise !== null) return this.loadPromise;
    this.updatePresentation({ pending: "load_game", error: null });
    this.loadPromise = (async () => {
      try {
        const envelope = this.recovery.load();
        if (envelope === null) {
          throw new TypeError("No saved game is available. Start a new game.");
        }
        const result = await this.api.resumeGame({ envelope });
        const view = this.acceptResponse(result.data);
        this.narrationCoordinator.requestEligible(view, result.data.envelope, {
          opening: result.data.envelope.history.length === 1,
        });
        return view;
      } catch (error) {
        if (error instanceof TypeError && error.message.includes("saved game")) {
          this.publish({
            view: null,
            presentation: { ...emptyPresentation(), error: recoveryError(error.message) },
            narration: narrationState(),
          });
        } else if (
          error instanceof StatelessApiError &&
          !error.response.retryable &&
          ["invalid_history", "validation_error", "request_too_large"].includes(error.response.code)
        ) {
          this.recovery.clear();
          this.envelope = null;
          this.publish({
            view: null,
            presentation: {
              ...emptyPresentation(),
              error: recoveryError("The saved game could not be replayed and has been cleared."),
            },
            narration: narrationState(),
          });
        } else {
          this.reject(error);
        }
        return null;
      } finally {
        this.loadPromise = null;
      }
    })();
    return this.loadPromise;
  }

  clearError(): void {
    this.updatePresentation({ error: null });
  }

  clearGame(): void {
    this.recovery.clear();
    this.envelope = null;
    this.narrationCoordinator.cancel();
    this.publish({
      view: null,
      presentation: emptyPresentation(),
      narration: narrationState(),
    });
  }

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

  legalMoveAt(position: number): LegalMove | null {
    const { view, presentation } = this.state;
    if (view === null || presentation.selectedHandSlot === null) return null;
    return view.legal_moves.find((move) =>
      move.hand_slot === presentation.selectedHandSlot && move.position === position
    ) ?? null;
  }

  async playPosition(position: number): Promise<boolean> {
    const move = this.legalMoveAt(position);
    return move === null ? false : this.playMove(move.move_id);
  }

  async playMove(moveId: string): Promise<boolean> {
    const { view, presentation } = this.state;
    const move = view?.legal_moves.find((candidate) => candidate.move_id === moveId);
    if (
      view === null || move === undefined || this.envelope === null ||
      presentation.pending !== null || view.phase.kind !== "human_turn"
    ) return false;
    this.updatePresentation({ pending: "human_move", error: null });
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
    try {
      const [result] = await Promise.all([
        this.api.applyCommand({
          envelope: this.envelope,
          command: { type: "place", hand_slot: move.hand_slot, position: move.position },
        }),
        this.draculaTurnDelay(),
      ]);
      const accepted = this.acceptResponse(result.data);
      this.narrationCoordinator.requestEligible(accepted, result.data.envelope);
      return true;
    } catch (error) {
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

  progressOpponent(): Promise<void> {
    // Stateless commands settle every non-forced opponent response synchronously.
    return Promise.resolve();
  }

  async advanceRound(): Promise<boolean> {
    const { view, presentation } = this.state;
    if (
      view === null || this.envelope === null || presentation.pending !== null ||
      (view.phase.kind !== "scoring" && view.phase.kind !== "round_advance")
    ) return false;
    this.updatePresentation({ pending: "round_advance", error: null });
    const priorStatus = view.status;
    if (priorStatus === "round_complete") this.narrationCoordinator.clear();
    try {
      const result = await this.api.applyCommand({
        envelope: this.envelope,
        command: { type: "advance_round" },
      });
      await this.holdOpponentOpening(result.data);
      const accepted = this.acceptResponse(result.data);
      if (accepted.status === "game_complete") {
        this.narrationCoordinator.requestEligible(accepted, result.data.envelope);
      }
      return true;
    } catch (error) {
      this.reject(error);
      return false;
    }
  }

  revealNarration(): void {
    this.narrationCoordinator.reveal();
  }
}

import { ApiError, type ApiClient } from "./api";
import type { LegalMove, Player } from "./contractPrimitives";
import type { ApiErrorResponse, HumanGameView } from "./statefulContracts";
import type { GameStoreState, NarrationState, PresentationState } from "./gameControllerContract";

export interface GameControllerOptions {
  requestId?: () => string;
  opponentPollDelay?: () => Promise<void>;
}

const emptyPresentation = (): PresentationState => ({
  pending: null,
  selectedHandSlot: null,
  error: null,
});

const disabledNarration = (): NarrationState => ({
  enabled: false,
  pending: false,
  messages: [],
});

const networkError = (error: unknown): ApiErrorResponse => ({
  schema_version: "dracula-error-v1",
  code: "dependency_unavailable",
  message: error instanceof Error ? error.message : "The game service is unavailable.",
  retryable: true,
  current_version: null,
  current_game: null,
});

export class GameController {
  private state: GameStoreState = {
    view: null,
    presentation: emptyPresentation(),
    narration: disabledNarration(),
  };
  private readonly listeners = new Set<() => void>();
  private readonly requestId: () => string;
  private readonly opponentPollDelay: () => Promise<void>;
  private loadPromise: Promise<HumanGameView | null> | null = null;
  private opponentPromise: Promise<void> | null = null;

  constructor(
    private readonly api: ApiClient,
    options: GameControllerOptions = {},
  ) {
    this.requestId = options.requestId ?? (() => crypto.randomUUID());
    this.opponentPollDelay =
      options.opponentPollDelay ?? (() => new Promise((resolve) => setTimeout(resolve, 250)));
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

  private acceptView(view: HumanGameView, clearSelection = true): void {
    const selectedHandSlot = clearSelection ? null : this.state.presentation.selectedHandSlot;
    this.publish({
      view,
      narration: this.state.narration,
      presentation: {
        ...this.state.presentation,
        selectedHandSlot,
        error: null,
      },
    });
  }

  private reject(error: unknown): void {
    const response = error instanceof ApiError ? error.response : networkError(error);
    const view = response.current_game ?? this.state.view;
    this.publish({
      view,
      narration: this.state.narration,
      presentation: {
        pending: null,
        selectedHandSlot: null,
        error: response,
      },
    });
  }

  async createGame(humanRole: Player): Promise<HumanGameView | null> {
    if (this.state.presentation.pending !== null) return null;
    this.publish({
      view: null,
      presentation: { ...emptyPresentation(), pending: "create_game" },
      narration: disabledNarration(),
    });
    try {
      const result = await this.api.createGame({
        human_role: humanRole,
        request_id: this.requestId(),
      });
      this.acceptView(result.data);
      this.updatePresentation({ pending: null });
      return result.data;
    } catch (error) {
      this.reject(error);
      return null;
    }
  }

  loadGame(gameId: string): Promise<HumanGameView | null> {
    if (this.loadPromise !== null) return this.loadPromise;
    this.updatePresentation({ pending: "load_game", error: null });
    this.loadPromise = (async () => {
      try {
        const result = await this.api.getGame(gameId);
        this.acceptView(result.data);
        this.updatePresentation({ pending: null });
        return result.data;
      } catch (error) {
        this.reject(error);
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
    this.publish({
      view: null,
      presentation: emptyPresentation(),
      narration: disabledNarration(),
    });
  }

  revealNarration(): void {
    // Stateful local gameplay retains its existing narrator/event lifecycle.
  }

  selectCard(handSlot: number): boolean {
    const { view, presentation } = this.state;
    if (
      view === null ||
      presentation.pending !== null ||
      view.phase.kind !== "human_turn" ||
      view.human_hand[handSlot] === null ||
      !view.legal_moves.some((move) => move.hand_slot === handSlot)
    ) {
      return false;
    }
    this.updatePresentation({ selectedHandSlot: handSlot, error: null });
    return true;
  }

  legalMoveAt(position: number): LegalMove | null {
    const { view, presentation } = this.state;
    if (view === null || presentation.selectedHandSlot === null) return null;
    return (
      view.legal_moves.find(
        (move) =>
          move.hand_slot === presentation.selectedHandSlot && move.position === position,
      ) ?? null
    );
  }

  async playPosition(position: number): Promise<boolean> {
    const move = this.legalMoveAt(position);
    if (move === null) return false;
    return this.playMove(move.move_id);
  }

  async playMove(moveId: string): Promise<boolean> {
    const { view, presentation } = this.state;
    const move = view?.legal_moves.find((candidate) => candidate.move_id === moveId);
    if (
      view === null ||
      move === undefined ||
      presentation.pending !== null ||
      view.phase.kind !== "human_turn"
    ) {
      return false;
    }
    this.updatePresentation({ pending: "human_move", error: null });
    try {
      const result = await this.api.submitMove(view.game_id, {
        move_id: move.move_id,
        expected_version: view.version,
        request_id: this.requestId(),
      });
      this.acceptView(result.data);
      this.updatePresentation({ pending: null });
      return true;
    } catch (error) {
      this.reject(error);
      return false;
    }
  }

  progressOpponent(options: { retryFailed?: boolean } = {}): Promise<void> {
    if (this.opponentPromise !== null) return this.opponentPromise;
    const initial = this.state.view;
    if (initial?.phase.kind !== "opponent_turn") return Promise.resolve();
    if (initial.phase.status === "failed" && !options.retryFailed) return Promise.resolve();
    const opponentPhase = initial.phase;

    this.opponentPromise = (async () => {
      const pendingJob = opponentPhase.status === "pending" ? opponentPhase.job_id : null;
      const requestId = pendingJob ?? this.requestId();
      const expectedVersion = initial.version;
      this.updatePresentation({ pending: "opponent_turn", error: null });
      try {
        while (true) {
          const result = await this.api.opponentTurn(initial.game_id, {
            expected_version: expectedVersion,
            request_id: requestId,
          });
          this.acceptView(result.data);
          if (result.status === 200) break;
          await this.opponentPollDelay();
        }
        this.updatePresentation({ pending: null });
      } catch (error) {
        this.reject(error);
      } finally {
        this.opponentPromise = null;
      }
    })();
    return this.opponentPromise;
  }

  async advanceRound(): Promise<boolean> {
    const { view, presentation } = this.state;
    if (
      view === null ||
      presentation.pending !== null ||
      (view.phase.kind !== "scoring" && view.phase.kind !== "round_advance")
    ) {
      return false;
    }
    this.updatePresentation({ pending: "round_advance", error: null });
    try {
      const result = await this.api.advanceRound(view.game_id, view.round_number, {
        expected_version: view.version,
        request_id: this.requestId(),
      });
      this.acceptView(result.data);
      this.updatePresentation({ pending: null });
      return true;
    } catch (error) {
      this.reject(error);
      return false;
    }
  }
}

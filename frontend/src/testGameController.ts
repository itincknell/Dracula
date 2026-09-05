/**
 * Provides a controllable in-memory GameController for component tests.
 * Test cases can publish explicit game and presentation states without invoking
 * HTTP, browser persistence, or the production stateless store.
 */
import type { LegalMove, Player, ResumablePhase } from "./contractPrimitives";
import type {
  GameControllerContract,
  GameStoreState,
  PresentationState,
} from "./gameControllerContract";
import type { HumanGameView } from "./gameView";

export function testGameView(
  phase: ResumablePhase = {
    kind: "opponent_turn",
    status: "ready",
    job_id: null,
    retryable: true,
  },
  changes: Partial<HumanGameView> = {},
): HumanGameView {
  return {
    status: "playing",
    round_number: 1,
    turn_number: 0,
    dealer: "queen",
    active_player: "king",
    human_role: "queen",
    opponent_role: "king",
    coffin: [null, null, null, null, "4C", null, null, null, null],
    current_round_moves: [],
    pending_round_result: null,
    completed_rounds: [],
    total_scores: { human: 0, opponent: 0 },
    phase,
    human_hand: ["2C", "10D", "QH", "KS"],
    legal_moves: [],
    ...changes,
  };
}

export interface TestControllerActions {
  createGame?: (role: Player) => Promise<HumanGameView>;
  playMove?: (move: LegalMove) => Promise<HumanGameView>;
  advanceRound?: () => Promise<HumanGameView>;
  revealNarration?: () => void;
}

const presentation = (): PresentationState => ({
  pending: null,
  selectedHandSlot: null,
  error: null,
});

export class TestGameController implements GameControllerContract {
  private state: GameStoreState;
  private readonly listeners = new Set<() => void>();

  constructor(
    view: HumanGameView | null = null,
    private readonly actions: TestControllerActions = {},
  ) {
    this.state = {
      view,
      presentation: presentation(),
      narration: { enabled: false, pending: false, messages: [] },
    };
  }

  readonly getSnapshot = (): GameStoreState => this.state;

  readonly subscribe = (listener: () => void): (() => void) => {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  };

  private publish(changes: Partial<GameStoreState>): void {
    this.state = { ...this.state, ...changes };
    this.listeners.forEach((listener) => listener());
  }

  private updatePresentation(changes: Partial<PresentationState>): void {
    this.publish({
      presentation: { ...this.state.presentation, ...changes },
    });
  }

  async createGame(role: Player): Promise<HumanGameView | null> {
    if (this.state.presentation.pending !== null || this.actions.createGame === undefined) return null;
    this.updatePresentation({ pending: "create_game" });
    const view = await this.actions.createGame(role);
    this.publish({ view, presentation: presentation() });
    return view;
  }

  loadGame(): Promise<HumanGameView | null> {
    return Promise.resolve(this.state.view);
  }

  clearError(): void {
    this.updatePresentation({ error: null });
  }

  clearGame(): void {
    this.publish({ view: null, presentation: presentation() });
  }

  selectCard(handSlot: number): boolean {
    const view = this.state.view;
    if (
      view === null || this.state.presentation.pending !== null ||
      view.phase.kind !== "human_turn" || view.human_hand[handSlot] === null ||
      !view.legal_moves.some((move) => move.hand_slot === handSlot)
    ) return false;
    this.updatePresentation({ selectedHandSlot: handSlot });
    return true;
  }

  legalMoveAt(position: number): LegalMove | null {
    const view = this.state.view;
    const handSlot = this.state.presentation.selectedHandSlot;
    if (view === null || handSlot === null) return null;
    return view.legal_moves.find(
      (move) => move.hand_slot === handSlot && move.position === position,
    ) ?? null;
  }

  async playPosition(position: number): Promise<boolean> {
    const move = this.legalMoveAt(position);
    if (move === null || this.actions.playMove === undefined) return false;
    this.updatePresentation({ pending: "human_move", selectedHandSlot: null });
    const view = await this.actions.playMove(move);
    this.publish({ view, presentation: presentation() });
    return true;
  }

  async advanceRound(): Promise<boolean> {
    if (this.state.presentation.pending !== null || this.actions.advanceRound === undefined) return false;
    this.updatePresentation({ pending: "round_advance" });
    const view = await this.actions.advanceRound();
    this.publish({ view, presentation: presentation() });
    return true;
  }

  revealNarration(): void {
    this.actions.revealNarration?.();
  }
}

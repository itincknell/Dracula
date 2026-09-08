/**
 * Defines the presentation-facing game view used by React components.
 * The view is transport-neutral and contains only trusted public gameplay data,
 * pending presentation state, and user-displayable failures.
 */
import type {
  CoffinSlots,
  GameStatus,
  HandSlots,
  LegalMove,
  PlayedMove,
  Player,
  PlayerScore,
  PresentationPhase,
  RoundRecord,
} from "./contractPrimitives";

/**
 * Trusted public game data consumed by the presentation components.
 *
 * The fields mirror the validated server view except for `phase`, which may
 * temporarily become `opponent_turn` while the browser displays Dracula's
 * minimum move delay. Keeping that display-only phase here prevents it from
 * leaking into the HTTP contract or the replay history.
 */
export interface HumanGameView {
  /** Overall engine lifecycle after automatic opponent play has settled. */
  status: GameStatus;
  /** One-based current round. */
  round_number: number;
  /** Placements already made in the current round. */
  turn_number: number;
  dealer: Player;
  /** The engine's next actor, or null while scoring or after the game. */
  active_player: Player | null;
  human_role: Player;
  opponent_role: Player;
  /** Nine row-major public coffin positions. */
  coffin: CoffinSlots;
  current_round_moves: PlayedMove[];
  /** The round currently being animated, before explicit advancement. */
  pending_round_result: RoundRecord | null;
  /** Rounds already acknowledged and incorporated into the next deal. */
  completed_rounds: RoundRecord[];
  total_scores: PlayerScore;
  phase: PresentationPhase;
  /** The human's stable four-slot hand; played cards leave null entries. */
  human_hand: HandSlots;
  /** Server-approved human moves for the current settled state. */
  legal_moves: LegalMove[];
}

/** A safe message for the UI plus whether repeating the operation is sensible. */
export interface PresentationError {
  message: string;
  retryable: boolean;
}

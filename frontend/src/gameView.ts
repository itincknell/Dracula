/**
 * Defines the presentation-facing game view used by React components.
 * The view is transport-neutral and contains only trusted public gameplay data,
 * pending presentation state, and user-displayable failures.
 */
import type {
  GameStatus,
  LegalMove,
  PlayedMove,
  Player,
  PlayerScore,
  ResumablePhase,
  RoundRecord,
} from "./contractPrimitives";

export interface HumanGameView {
  status: GameStatus;
  round_number: number;
  turn_number: number;
  dealer: Player;
  active_player: Player | null;
  human_role: Player;
  opponent_role: Player;
  coffin: [
    string | null,
    string | null,
    string | null,
    string | null,
    string | null,
    string | null,
    string | null,
    string | null,
    string | null,
  ];
  current_round_moves: PlayedMove[];
  pending_round_result: RoundRecord | null;
  completed_rounds: RoundRecord[];
  total_scores: PlayerScore;
  phase: ResumablePhase;
  human_hand: [string | null, string | null, string | null, string | null];
  legal_moves: LegalMove[];
}

export interface PresentationError {
  message: string;
  retryable: boolean;
}

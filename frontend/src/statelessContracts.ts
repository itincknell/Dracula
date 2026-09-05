/**
 * Defines and validates the stateless production API wire format.
 * Raw server data becomes trusted frontend data only after these checks verify
 * the recovery envelope, public game projection, errors, and narration results.
 */
import {
  assertArray,
  assertBoolean,
  assertLineScore,
  assertNullableString,
  assertNumber,
  assertOneOf,
  assertPhase,
  assertPlayerScore,
  assertScoringStep,
  assertString,
  exactKeys,
  record,
  type GameStatus,
  type Player,
  type PlayedMove,
  type PlayerScore,
  type ResumablePhase,
  type RoundRecord,
} from "./contractPrimitives";

export type AcceptedGameCommand =
  | { type: "select_role"; human_role: Player }
  | { type: "place"; hand_slot: number; position: number }
  | { type: "advance_round" };

export type RequestedGameCommand = Exclude<AcceptedGameCommand, { type: "select_role" }>;

export interface RecoveryEnvelope {
  seed: string;
  history: AcceptedGameCommand[];
}

export interface StatelessLegalMove {
  card_id: string;
  hand_slot: number;
  position: number;
}

export type VisiblePlayedMove = PlayedMove;
export type StatelessRoundRecord = RoundRecord;

export interface StatelessHumanGameView {
  status: GameStatus;
  round_number: number;
  turn_number: number;
  dealer: Player;
  active_player: Player | null;
  human_role: Player;
  opponent_role: Player;
  coffin: [
    string | null, string | null, string | null,
    string | null, string | null, string | null,
    string | null, string | null, string | null,
  ];
  current_round_moves: VisiblePlayedMove[];
  pending_round_result: StatelessRoundRecord | null;
  completed_rounds: StatelessRoundRecord[];
  total_scores: PlayerScore;
  phase: ResumablePhase;
  human_hand: [string | null, string | null, string | null, string | null];
  legal_moves: StatelessLegalMove[];
}

export interface StatelessGameResponse {
  envelope: RecoveryEnvelope;
  game: StatelessHumanGameView;
}

export interface StatelessHealthResponse {
  status: "ok";
  gameplay_mode: "stateless";
  opponent_configured: boolean;
  narration_enabled: boolean;
  narration_configured: boolean;
}

export type NarrationCueType = "opening" | "round_transition" | "final_result";

export interface NarrationResponse {
  cue_type: NarrationCueType;
  status: "ready" | "unavailable";
  text: string | null;
}

export interface StatelessApiErrorResponse {
  code:
    | "validation_error"
    | "request_too_large"
    | "invalid_history"
    | "invalid_command"
    | "wrong_turn"
    | "wrong_phase"
    | "ineligible_cue"
    | "dependency_unavailable";
  message: string;
  retryable: boolean;
}

function assertAcceptedGameCommand(value: unknown, label: string): void {
  const item = record(value, label);
  assertString(item.type, `${label} type`);
  if (item.type === "select_role") {
    exactKeys(item, ["type", "human_role"], label);
    assertOneOf(item.human_role, ["queen", "king"], `${label} role`);
    return;
  }
  if (item.type === "place") {
    exactKeys(item, ["type", "hand_slot", "position"], label);
    assertNumber(item.hand_slot, `${label} hand slot`);
    assertNumber(item.position, `${label} position`);
    if (!Number.isInteger(item.hand_slot) || item.hand_slot < 0 || item.hand_slot > 3) {
      throw new TypeError(`${label} hand slot is out of range`);
    }
    if (!Number.isInteger(item.position) || item.position < 0 || item.position > 8) {
      throw new TypeError(`${label} position is out of range`);
    }
    return;
  }
  if (item.type === "advance_round") {
    exactKeys(item, ["type"], label);
    return;
  }
  throw new TypeError(`${label} type is unsupported`);
}

export function assertRecoveryEnvelope(value: unknown): asserts value is RecoveryEnvelope {
  const item = record(value, "recovery envelope");
  exactKeys(item, ["seed", "history"], "recovery envelope");
  assertString(item.seed, "recovery seed");
  if (item.seed.length < 1 || item.seed.length > 256 || item.seed.includes("\0")) {
    throw new TypeError("recovery seed is invalid");
  }
  assertArray(item.history, "recovery history");
  if (item.history.length < 1 || item.history.length > 31) {
    throw new TypeError("recovery history length is invalid");
  }
  item.history.forEach((command, index) => assertAcceptedGameCommand(command, `history command ${index}`));
  const first = item.history[0] as Record<string, unknown>;
  if (first.type !== "select_role") throw new TypeError("history must begin with role selection");
  if (item.history.slice(1).some((command) => (command as Record<string, unknown>).type === "select_role")) {
    throw new TypeError("role selection may appear only once");
  }
}

function assertVisiblePlayedMove(value: unknown): void {
  const item = record(value, "visible move");
  exactKeys(item, ["player", "card_id", "position", "turn_number"], "visible move");
  assertOneOf(item.player, ["queen", "king"], "visible move player");
  assertString(item.card_id, "visible move card");
  assertNumber(item.position, "visible move position");
  assertNumber(item.turn_number, "visible move turn");
}

function assertStatelessRoundRecord(value: unknown): void {
  const item = record(value, "stateless round record");
  exactKeys(
    item,
    ["round_number", "dealer", "coffin", "moves", "line_scores", "scoring_sequence", "round_scores"],
    "stateless round record",
  );
  assertNumber(item.round_number, "round number");
  assertOneOf(item.dealer, ["queen", "king"], "round dealer");
  assertArray(item.coffin, "round coffin");
  if (item.coffin.length !== 9) throw new TypeError("round coffin requires nine cards");
  item.coffin.forEach((card) => assertString(card, "round coffin card"));
  assertArray(item.moves, "round moves");
  item.moves.forEach(assertVisiblePlayedMove);
  assertArray(item.line_scores, "round line scores");
  item.line_scores.forEach(assertLineScore);
  assertArray(item.scoring_sequence, "scoring sequence");
  item.scoring_sequence.forEach(assertScoringStep);
  assertPlayerScore(item.round_scores);
}

function assertStatelessLegalMove(value: unknown): void {
  const item = record(value, "stateless legal move");
  exactKeys(item, ["card_id", "hand_slot", "position"], "stateless legal move");
  assertString(item.card_id, "legal move card");
  assertNumber(item.hand_slot, "legal move hand slot");
  assertNumber(item.position, "legal move position");
}

function assertStatelessHumanGameView(value: unknown): void {
  const item = record(value, "stateless game view");
  exactKeys(
    item,
    [
      "status", "round_number", "turn_number", "dealer", "active_player", "human_role",
      "opponent_role", "coffin", "current_round_moves", "pending_round_result",
      "completed_rounds", "total_scores", "phase", "human_hand", "legal_moves",
    ],
    "stateless game view",
  );
  assertOneOf(item.status, ["playing", "round_complete", "game_complete"], "game status");
  assertNumber(item.round_number, "game round");
  assertNumber(item.turn_number, "game turn");
  assertOneOf(item.dealer, ["queen", "king"], "dealer");
  if (item.active_player !== null) assertOneOf(item.active_player, ["queen", "king"], "active player");
  assertOneOf(item.human_role, ["queen", "king"], "human role");
  assertOneOf(item.opponent_role, ["queen", "king"], "opponent role");
  assertArray(item.coffin, "coffin");
  if (item.coffin.length !== 9) throw new TypeError("coffin requires nine slots");
  item.coffin.forEach((card) => assertNullableString(card, "coffin card"));
  assertArray(item.current_round_moves, "current round moves");
  item.current_round_moves.forEach(assertVisiblePlayedMove);
  if (item.pending_round_result !== null) assertStatelessRoundRecord(item.pending_round_result);
  assertArray(item.completed_rounds, "completed rounds");
  item.completed_rounds.forEach(assertStatelessRoundRecord);
  assertPlayerScore(item.total_scores);
  assertPhase(item.phase);
  assertArray(item.human_hand, "human hand");
  if (item.human_hand.length !== 4) throw new TypeError("human hand requires four slots");
  item.human_hand.forEach((card) => assertNullableString(card, "human hand card"));
  assertArray(item.legal_moves, "legal moves");
  item.legal_moves.forEach(assertStatelessLegalMove);
}

export function assertStatelessGameResponse(value: unknown): asserts value is StatelessGameResponse {
  const item = record(value, "stateless game response");
  exactKeys(item, ["envelope", "game"], "stateless game response");
  assertRecoveryEnvelope(item.envelope);
  assertStatelessHumanGameView(item.game);
}

export function assertStatelessHealthResponse(value: unknown): asserts value is StatelessHealthResponse {
  const item = record(value, "stateless health response");
  exactKeys(
    item,
    ["status", "gameplay_mode", "opponent_configured", "narration_enabled", "narration_configured"],
    "stateless health response",
  );
  assertOneOf(item.status, ["ok"], "health status");
  assertOneOf(item.gameplay_mode, ["stateless"], "gameplay mode");
  assertBoolean(item.opponent_configured, "opponent configured");
  assertBoolean(item.narration_enabled, "narration enabled");
  assertBoolean(item.narration_configured, "narration configured");
}

export function assertNarrationResponse(value: unknown): asserts value is NarrationResponse {
  const item = record(value, "narration response");
  exactKeys(item, ["cue_type", "status", "text"], "narration response");
  assertOneOf(item.cue_type, ["opening", "round_transition", "final_result"], "narration cue");
  assertOneOf(item.status, ["ready", "unavailable"], "narration status");
  assertNullableString(item.text, "narration text");
  if ((item.status === "ready") !== (typeof item.text === "string" && item.text.length > 0)) {
    throw new TypeError("narration status and text disagree");
  }
}

export function assertStatelessApiErrorResponse(
  value: unknown,
): asserts value is StatelessApiErrorResponse {
  const item = record(value, "stateless API error");
  exactKeys(item, ["code", "message", "retryable"], "stateless API error");
  assertOneOf(
    item.code,
    [
      "validation_error", "request_too_large", "invalid_history", "invalid_command",
      "wrong_turn", "wrong_phase", "ineligible_cue", "dependency_unavailable",
    ],
    "stateless API error code",
  );
  assertString(item.message, "stateless API error message");
  assertBoolean(item.retryable, "stateless API error retryable");
}

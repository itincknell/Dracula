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
  assertServerGamePhase,
  assertPlayerScore,
  assertScoringStep,
  assertString,
  exactKeys,
  record,
  type GameStatus,
  type CoffinSlots,
  type HandSlots,
  type LegalMove,
  type Player,
  type PlayedMove,
  type PlayerScore,
  type ServerGamePhase,
  type RoundRecord,
} from "./contractPrimitives";

/**
 * Commands accepted after a game has already been created.
 * A placement names one stable hand slot and one global coffin position;
 * advancement acknowledges the score presentation for a completed round.
 */
export type RequestedGameCommand =
  | { type: "place"; hand_slot: number; position: number }
  | { type: "advance_round" };

/**
 * Every command needed to replay a game from its seed.
 *
 * Role selection is stored as the first accepted command even though it is
 * supplied to the create-game endpoint rather than the command endpoint.
 */
export type AcceptedGameCommand =
  | { type: "select_role"; human_role: Player }
  | RequestedGameCommand;

/** The complete client-owned recipe for rebuilding one authoritative game. */
export interface RecoveryEnvelope {
  seed: string;
  /** Accepted commands in execution order, beginning with role selection. */
  history: AcceptedGameCommand[];
}

/**
 * Public engine state returned after the server has settled automatic play.
 *
 * This is the raw wire shape. `gameView.ts` defines the trusted presentation
 * copy, whose phase union additionally permits the browser-only opponent delay.
 */
export interface StatelessHumanGameView {
  status: GameStatus;
  /** One-based round number. */
  round_number: number;
  /** Number of placements already made in the current round. */
  turn_number: number;
  dealer: Player;
  /** Null once no further placement is possible in the current game state. */
  active_player: Player | null;
  human_role: Player;
  opponent_role: Player;
  coffin: CoffinSlots;
  current_round_moves: PlayedMove[];
  /**
   * The just-finished round while its score animation awaits acknowledgement.
   * It is not duplicated in `completed_rounds` until the next round advances.
   */
  pending_round_result: RoundRecord | null;
  completed_rounds: RoundRecord[];
  total_scores: PlayerScore;
  phase: ServerGamePhase;
  human_hand: HandSlots;
  /** Empty unless the stable phase is `human_turn`. */
  legal_moves: LegalMove[];
}

/** A successful gameplay response pairs rebuilt public state with its replay recipe. */
export interface StatelessGameResponse {
  envelope: RecoveryEnvelope;
  game: StatelessHumanGameView;
}

/** Minimal operational status used by the frontend before gameplay begins. */
/** The only three moments at which the frontend may request dialogue. */
export type NarrationCueType = "opening" | "round_transition" | "final_result";

/** A narration request either returns usable text or an explicit empty result. */
export interface NarrationResponse {
  cue_type: NarrationCueType;
  status: "ready" | "unavailable";
  text: string | null;
}

/**
 * Structured public errors returned by the stateless API.
 *
 * `validation_error` means the HTTP body did not match its declared shape.
 * `invalid_history` means the recovery commands could not be replayed.
 * `invalid_command`, `wrong_turn`, and `wrong_phase` reject a proposed move or
 * lifecycle action. `ineligible_cue` rejects narration at the wrong moment,
 * while `dependency_unavailable` reports an optional or required service that
 * could not answer.
 */
export interface StatelessApiErrorResponse {
  code:
    | "validation_error"
    | "invalid_history"
    | "invalid_command"
    | "wrong_turn"
    | "wrong_phase"
    | "ineligible_cue"
    | "dependency_unavailable";
  message: string;
  /** Tells the controller whether repeating the same request may be useful. */
  retryable: boolean;
}

/** Validate one command stored in an untrusted browser recovery history. */
function assertAcceptedGameCommand(value: unknown, label: string): void {
  const item = record(value, label);
  assertString(item.type, `${label} type`);

  // The command type is a discriminator: each case has a different exact
  // field set, and no command may smuggle unrelated state into replay.
  if (item.type === "select_role") {
    exactKeys(item, ["type", "human_role"], label);
    assertOneOf(item.human_role, ["queen", "king"], `${label} role`);
    return;
  }
  if (item.type === "place") {
    exactKeys(item, ["type", "hand_slot", "position"], label);
    assertNumber(item.hand_slot, `${label} hand slot`);
    assertNumber(item.position, `${label} position`);

    // Hand slots are the engine's four stable card positions. Coffin positions
    // use row-major indexes over all nine cells, including the occupied center.
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

/**
 * Validate the seed and ordered commands loaded from browser storage or HTTP.
 *
 * The limit of 31 commands covers one role selection, up to 24 human card
 * placements, and six round-advance commands. Gameplay replay later verifies
 * that the individually well-formed commands are possible in that exact order.
 */
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
  item.history.forEach((command, index) => {
    assertAcceptedGameCommand(command, `history command ${index}`);
  });

  // The seed does not imply which side the human chose. Requiring role
  // selection exactly once makes replay identity unambiguous.
  const first = item.history[0] as Record<string, unknown>;
  if (first.type !== "select_role") throw new TypeError("history must begin with role selection");
  const roleWasRepeated = item.history
    .slice(1)
    .some((command) => (command as Record<string, unknown>).type === "select_role");
  if (roleWasRepeated) {
    throw new TypeError("role selection may appear only once");
  }
}

/** Validate a public move; private hand-slot data is intentionally absent. */
function assertVisiblePlayedMove(value: unknown): void {
  const item = record(value, "visible move");
  exactKeys(item, ["player", "card_id", "position", "turn_number"], "visible move");
  assertOneOf(item.player, ["queen", "king"], "visible move player");
  assertString(item.card_id, "visible move card");
  assertNumber(item.position, "visible move position");
  assertNumber(item.turn_number, "visible move turn");
}

/** Validate one immutable completed-round record and all nested animation data. */
function assertStatelessRoundRecord(value: unknown): void {
  const item = record(value, "stateless round record");
  exactKeys(
    item,
    ["round_number", "dealer", "coffin", "moves", "line_scores", "scoring_sequence", "round_scores"],
    "stateless round record",
  );
  assertNumber(item.round_number, "round number");
  assertOneOf(item.dealer, ["queen", "king"], "round dealer");

  // A completed round always exposes the entire nine-card coffin.
  assertArray(item.coffin, "round coffin");
  if (item.coffin.length !== 9) throw new TypeError("round coffin requires nine cards");
  item.coffin.forEach((card) => assertString(card, "round coffin card"));
  assertArray(item.moves, "round moves");
  item.moves.forEach(assertVisiblePlayedMove);

  // Scores and the presentation sequence are server-authored. The frontend
  // renders them but never recalculates authoritative round results.
  assertArray(item.line_scores, "round line scores");
  item.line_scores.forEach(assertLineScore);
  assertArray(item.scoring_sequence, "scoring sequence");
  item.scoring_sequence.forEach(assertScoringStep);
  assertPlayerScore(item.round_scores);
}

/** Validate one move the server says the human may currently make. */
function assertStatelessLegalMove(value: unknown): void {
  const item = record(value, "stateless legal move");
  exactKeys(item, ["card_id", "hand_slot", "position"], "stateless legal move");
  assertString(item.card_id, "legal move card");
  assertNumber(item.hand_slot, "legal move hand slot");
  assertNumber(item.position, "legal move position");
}

/** Validate the complete public game projection before React can consume it. */
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

  // First validate the scalar lifecycle and role fields that explain the rest
  // of the response.
  assertOneOf(item.status, ["playing", "round_complete", "game_complete"], "game status");
  assertNumber(item.round_number, "game round");
  assertNumber(item.turn_number, "game turn");
  assertOneOf(item.dealer, ["queen", "king"], "dealer");
  if (item.active_player !== null) assertOneOf(item.active_player, ["queen", "king"], "active player");
  assertOneOf(item.human_role, ["queen", "king"], "human role");
  assertOneOf(item.opponent_role, ["queen", "king"], "opponent role");

  // The live coffin and current moves are public even before a round ends.
  assertArray(item.coffin, "coffin");
  if (item.coffin.length !== 9) throw new TypeError("coffin requires nine slots");
  item.coffin.forEach((card) => assertNullableString(card, "coffin card"));
  assertArray(item.current_round_moves, "current round moves");
  item.current_round_moves.forEach(assertVisiblePlayedMove);

  // A finished round occupies exactly one of these locations: pending while
  // scoring is shown, then completed after the browser advances the round.
  if (item.pending_round_result !== null) assertStatelessRoundRecord(item.pending_round_result);
  assertArray(item.completed_rounds, "completed rounds");
  item.completed_rounds.forEach(assertStatelessRoundRecord);
  assertPlayerScore(item.total_scores);
  assertServerGamePhase(item.phase);

  // Only the human hand is public. Stable slots retain null holes after cards
  // are played so submitted hand indexes continue to name the same cards.
  assertArray(item.human_hand, "human hand");
  if (item.human_hand.length !== 4) throw new TypeError("human hand requires four slots");
  item.human_hand.forEach((card) => assertNullableString(card, "human hand card"));
  assertArray(item.legal_moves, "legal moves");
  item.legal_moves.forEach(assertStatelessLegalMove);
}

/** Validate a successful response from any gameplay endpoint. */
export function assertStatelessGameResponse(value: unknown): asserts value is StatelessGameResponse {
  const item = record(value, "stateless game response");
  exactKeys(item, ["envelope", "game"], "stateless game response");
  assertRecoveryEnvelope(item.envelope);
  assertStatelessHumanGameView(item.game);
}

/**
 * Validate optional narration without making it part of gameplay correctness.
 * Ready responses require nonempty text; unavailable responses require null.
 */
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

/** Validate the API's public error shape before displaying or acting on it. */
export function assertStatelessApiErrorResponse(
  value: unknown,
): asserts value is StatelessApiErrorResponse {
  const item = record(value, "stateless API error");
  exactKeys(item, ["code", "message", "retryable"], "stateless API error");
  assertOneOf(
    item.code,
    [
      "validation_error", "invalid_history", "invalid_command",
      "wrong_turn", "wrong_phase", "ineligible_cue", "dependency_unavailable",
    ],
    "stateless API error code",
  );
  assertString(item.message, "stateless API error message");
  assertBoolean(item.retryable, "stateless API error retryable");
}

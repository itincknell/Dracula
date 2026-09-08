/**
 * Public gameplay values shared by the API client, controller, and React view.
 *
 * TypeScript interfaces disappear when the application is compiled, so they
 * cannot by themselves make JSON received from the server safe to use. The
 * assertion functions in the second half of this file perform the matching
 * runtime checks. Once a response passes those checks, the rest of the
 * frontend can work with these types without repeatedly validating them.
 *
 * This module contains only values that may be shown to the human player. It
 * has no representation of the opponent's hand, the undealt deck, policy
 * inputs, or other private engine state.
 */

/** The two roles used by the game rules and public API. */
export type Player = "queen" | "king";

/**
 * The engine lifecycle state after any automatic opponent move has settled.
 * `playing` permits more placements, `round_complete` awaits the scoring UI,
 * and `game_complete` follows acknowledgement of the sixth round.
 */
export type GameStatus = "playing" | "round_complete" | "game_complete";

/** Scores relabeled around the browser user rather than Queen and King. */
export interface PlayerScore {
  /** Score belonging to the role selected by the browser user. */
  human: number;
  /** Score belonging to Dracula, regardless of whether he is Queen or King. */
  opponent: number;
}

/** One publicly visible card placement in the current or a completed round. */
export interface PlayedMove {
  player: Player;
  card_id: string;
  /** Row-major position in the complete 3×3 coffin, including the center. */
  position: number;
  /** One-based placement number within the round. */
  turn_number: number;
}

/** The complete score calculation for one row or column of the coffin. */
export interface LineScore {
  direction: "row" | "column";
  /** Zero-based row or column number. */
  index: number;
  /** Cards in their visible order along the scored line. */
  card_ids: [string, string, string];
  /** Sum of the three card values before applying a multiplier. */
  base_value: number;
  /** Rules-defined factor; zero represents a line containing a Vampire. */
  multiplier: 0 | 1 | 2 | 3 | 5;
  /** Machine-readable rule that produced `multiplier`. */
  multiplier_reason: "none" | "suit_pair" | "same_color" | "same_suit" | "vampire";
  /** Human-readable multiplier name rendered during the scoring animation. */
  multiplier_label: string;
  /** Cards whose suit, color, or Vampire status created the multiplier. */
  highlighted_card_ids: string[];
  /** Final `base_value × multiplier` score. */
  total: number;
}

/**
 * One instruction in the server-authored round-scoring animation.
 *
 * `details` contains the small primitive values needed by the named step. For
 * example, `score_line` supplies its arithmetic while `compare_candidates`
 * supplies the two ranked totals being compared. Card and line structure stay
 * in `line` rather than being copied into this flexible field.
 */
export interface ScoringStep {
  /** Selects the calculation or comparison performed by this animation step. */
  kind: "score_line" | "compare_candidates" | "select_round_score" | "update_total";
  /** Role whose line is being scored; null for two-player comparison steps. */
  player: Player | null;
  /** Full line data for `score_line`; null for comparison and total steps. */
  line: LineScore | null;
  details: Record<string, number | string | boolean>;
}

/** A completed round and everything required to replay its score animation. */
export interface RoundRecord {
  /** One-based round number. */
  round_number: number;
  dealer: Player;
  coffin: CompletedCoffin;
  /** All eight player placements; the dealt center card is not a move. */
  moves: PlayedMove[];
  /** The dealer's three lines followed by the other player's three lines. */
  line_scores: LineScore[];
  scoring_sequence: ScoringStep[];
  round_scores: PlayerScore;
}

/** One engine-approved move the human can currently submit. */
export interface LegalMove {
  card_id: string;
  /** Stable position of the card in the four-slot engine hand. */
  hand_slot: number;
  /** Row-major position in the complete 3×3 coffin. */
  position: number;
}

/** Four stable hand positions; played cards leave `null` holes. */
export type HandSlots = [
  string | null,
  string | null,
  string | null,
  string | null,
];

/** Nine row-major coffin positions while a round is being played. */
export type CoffinSlots = [
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

/** A finished coffin contains a card in all nine positions. */
export type CompletedCoffin = [
  string,
  string,
  string,
  string,
  string,
  string,
  string,
  string,
  string,
];

/**
 * Stable phases returned by the stateless server.
 *
 * The server completes Dracula's automatic move before responding, so it
 * never returns an `opponent_turn` phase. A completed round remains in
 * `scoring` until the browser submits the explicit advance command.
 */
export type ServerGamePhase =
  | { kind: "human_turn" }
  | { kind: "scoring"; round_number: number }
  | { kind: "game_complete"; outcome: "human" | "opponent" | "tie" };

/** The browser adds this temporary phase while displaying Dracula's move delay. */
export type PresentationPhase = ServerGamePhase | { kind: "opponent_turn" };

/** An object whose fields remain unknown until a validator checks them. */
type JsonRecord = Record<string, unknown>;

/**
 * Require an ordinary JSON object and expose its fields as unknown values.
 *
 * Arrays and `null` also have JavaScript's `object` type, so they must be
 * rejected explicitly. The cast does not claim anything about the object's
 * fields; each caller still validates those fields before using them.
 */
export function record(value: unknown, label: string): JsonRecord {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new TypeError(`${label} must be an object`);
  }
  return value as JsonRecord;
}

/**
 * Require exactly the documented fields, independent of their JSON order.
 *
 * Rejecting extra fields catches server/frontend contract drift instead of
 * silently carrying an obsolete or accidentally private field into browser
 * state. Sorting makes the comparison insensitive to serialization order.
 */
export function exactKeys(value: JsonRecord, keys: readonly string[], label: string): void {
  const actual = Object.keys(value).sort();
  const expected = [...keys].sort();
  if (actual.length !== expected.length || actual.some((key, index) => key !== expected[index])) {
    throw new TypeError(`${label} has an unexpected field set`);
  }
}

/** Narrow an unknown JSON value to a string after checking it at runtime. */
export function assertString(value: unknown, label: string): asserts value is string {
  if (typeof value !== "string") throw new TypeError(`${label} must be a string`);
}

/** Require a real finite number; JSON-facing calculations cannot use NaN or infinity. */
export function assertNumber(value: unknown, label: string): asserts value is number {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new TypeError(`${label} must be a finite number`);
  }
}

/** Narrow an unknown JSON value to a Boolean. */
export function assertBoolean(value: unknown, label: string): asserts value is boolean {
  if (typeof value !== "boolean") throw new TypeError(`${label} must be a boolean`);
}

/**
 * Require one member of a string-literal set and preserve that narrow type.
 * The generic return assertion lets TypeScript recognize the checked value as
 * the same union represented by `choices`.
 */
export function assertOneOf<T extends string>(
  value: unknown,
  choices: readonly T[],
  label: string,
): asserts value is T {
  if (typeof value !== "string" || !choices.includes(value as T)) {
    throw new TypeError(`${label} has an unsupported value`);
  }
}

/** Require either `null` or a string without changing the caller's field type. */
export function assertNullableString(value: unknown, label: string): void {
  if (value !== null) assertString(value, label);
}

/** Narrow an unknown JSON value to an array whose entries still require checks. */
export function assertArray(value: unknown, label: string): asserts value is unknown[] {
  if (!Array.isArray(value)) throw new TypeError(`${label} must be an array`);
}

/** Validate the human-relative score pair used throughout the public API. */
export function assertPlayerScore(value: unknown): void {
  const item = record(value, "player score");
  exactKeys(item, ["human", "opponent"], "player score");
  assertNumber(item.human, "human score");
  assertNumber(item.opponent, "opponent score");
}

/** Validate one complete line-score object received from the server. */
export function assertLineScore(value: unknown): void {
  const item = record(value, "line score");
  // Checking the full field set first makes every later property access safe
  // and rejects stale score formats before the animation consumes them.
  exactKeys(
    item,
    [
      "direction",
      "index",
      "card_ids",
      "base_value",
      "multiplier",
      "multiplier_reason",
      "multiplier_label",
      "highlighted_card_ids",
      "total",
    ],
    "line score",
  );
  assertOneOf(item.direction, ["row", "column"], "line direction");
  assertNumber(item.index, "line index");

  // A scored row or column always names exactly three visible cards.
  assertArray(item.card_ids, "line cards");
  if (item.card_ids.length !== 3) throw new TypeError("line score requires three cards");
  item.card_ids.forEach((card) => assertString(card, "line card"));

  assertNumber(item.base_value, "line base value");
  // TypeScript's multiplier union does not exist at runtime, so the JSON value
  // must be checked against the five multipliers defined by the game rules.
  if (![0, 1, 2, 3, 5].includes(item.multiplier as number)) {
    throw new TypeError("line multiplier is unsupported");
  }
  assertOneOf(
    item.multiplier_reason,
    ["none", "suit_pair", "same_color", "same_suit", "vampire"],
    "multiplier reason",
  );
  assertString(item.multiplier_label, "multiplier label");

  // Highlighted cards drive the visual emphasis during the score animation.
  // The server may legitimately return zero, one, two, or three of them.
  assertArray(item.highlighted_card_ids, "highlighted cards");
  item.highlighted_card_ids.forEach((card) => assertString(card, "highlighted card"));
  assertNumber(item.total, "line total");
}

/** Validate one server-authored instruction in the scoring animation. */
export function assertScoringStep(value: unknown): void {
  const item = record(value, "scoring step");
  exactKeys(item, ["kind", "player", "line", "details"], "scoring step");
  assertOneOf(
    item.kind,
    ["score_line", "compare_candidates", "select_round_score", "update_total"],
    "scoring step kind",
  );

  // Comparison and total-update steps concern both players, so `player` and
  // `line` may be null. Line-scoring steps supply both values.
  if (item.player !== null) assertOneOf(item.player, ["queen", "king"], "scoring player");
  if (item.line !== null) assertLineScore(item.line);

  // Each step kind uses a different small set of named values. Restricting
  // them to JSON primitives prevents nested data from bypassing its own
  // explicit public contract.
  const details = record(item.details, "scoring details");
  Object.values(details).forEach((detail) => {
    if (!["string", "number", "boolean"].includes(typeof detail)) {
      throw new TypeError("scoring detail must be a primitive value");
    }
  });
}

/** Validate and narrow the server's discriminated gameplay-phase union. */
export function assertServerGamePhase(value: unknown): asserts value is ServerGamePhase {
  const item = record(value, "game phase");
  assertString(item.kind, "phase kind");

  // `kind` selects both the permitted fields and their validation. This keeps
  // impossible combinations, such as an outcome on a human turn, out of the
  // trusted presentation state.
  switch (item.kind) {
    case "human_turn":
      exactKeys(item, ["kind"], "human-turn phase");
      return;
    case "scoring":
      exactKeys(item, ["kind", "round_number"], "scoring phase");
      assertNumber(item.round_number, "scoring round");
      return;
    case "game_complete":
      exactKeys(item, ["kind", "outcome"], "game-complete phase");
      assertOneOf(item.outcome, ["human", "opponent", "tie"], "game outcome");
      return;
    default:
      throw new TypeError("game phase kind is unsupported");
  }
}

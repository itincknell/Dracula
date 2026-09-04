export type Player = "queen" | "king";
export type GameStatus = "playing" | "round_complete" | "game_complete";

export interface PlayerScore {
  human: number;
  opponent: number;
}

export interface PlayedMove {
  player: Player;
  card_id: string;
  hand_slot: number;
  position: number;
  turn_number: number;
}

export interface LineScore {
  direction: "row" | "column";
  index: number;
  card_ids: [string, string, string];
  base_value: number;
  multiplier: 0 | 1 | 2 | 3 | 5;
  multiplier_reason: "none" | "suit_pair" | "same_color" | "same_suit" | "vampire";
  multiplier_label: string;
  highlighted_card_ids: string[];
  total: number;
}

export interface ScoringStep {
  kind: "score_line" | "compare_candidates" | "select_round_score" | "update_total";
  player: Player | null;
  line: LineScore | null;
  details: Record<string, number | string | boolean>;
}

export interface RoundRecord {
  round_number: number;
  dealer: Player;
  coffin: [string, string, string, string, string, string, string, string, string];
  moves: PlayedMove[];
  line_scores: LineScore[];
  scoring_sequence: ScoringStep[];
  round_scores: PlayerScore;
}

export interface LegalMove {
  move_id: string;
  card_id: string;
  hand_slot: number;
  position: number;
}

export type ResumablePhase =
  | { kind: "human_turn" }
  | {
      kind: "opponent_turn";
      status: "ready" | "pending" | "failed";
      job_id: string | null;
      retryable: boolean;
    }
  | {
      kind: "narration";
      status: "ready" | "pending" | "failed";
      event_id: string;
      narration_id: string | null;
      required: boolean;
    }
  | {
      kind: "scoring";
      round_number: number;
      next_step_index: number;
      pending_narration_id: string | null;
    }
  | { kind: "round_advance"; round_number: number }
  | { kind: "game_complete"; outcome: "human" | "opponent" | "tie" };

export type JsonRecord = Record<string, unknown>;

export function record(value: unknown, label: string): JsonRecord {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new TypeError(`${label} must be an object`);
  }
  return value as JsonRecord;
}

export function exactKeys(value: JsonRecord, keys: readonly string[], label: string): void {
  const actual = Object.keys(value).sort();
  const expected = [...keys].sort();
  if (actual.length !== expected.length || actual.some((key, index) => key !== expected[index])) {
    throw new TypeError(`${label} has an unexpected field set`);
  }
}

export function assertString(value: unknown, label: string): asserts value is string {
  if (typeof value !== "string") throw new TypeError(`${label} must be a string`);
}

export function assertNumber(value: unknown, label: string): asserts value is number {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new TypeError(`${label} must be a finite number`);
  }
}

export function assertBoolean(value: unknown, label: string): asserts value is boolean {
  if (typeof value !== "boolean") throw new TypeError(`${label} must be a boolean`);
}

export function assertOneOf<T extends string>(
  value: unknown,
  choices: readonly T[],
  label: string,
): asserts value is T {
  if (typeof value !== "string" || !choices.includes(value as T)) {
    throw new TypeError(`${label} has an unsupported value`);
  }
}

export function assertNullableString(value: unknown, label: string): void {
  if (value !== null) assertString(value, label);
}

export function assertArray(value: unknown, label: string): asserts value is unknown[] {
  if (!Array.isArray(value)) throw new TypeError(`${label} must be an array`);
}

export function assertPlayerScore(value: unknown): void {
  const item = record(value, "player score");
  exactKeys(item, ["human", "opponent"], "player score");
  assertNumber(item.human, "human score");
  assertNumber(item.opponent, "opponent score");
}

export function assertPlayedMove(value: unknown): void {
  const item = record(value, "played move");
  exactKeys(item, ["player", "card_id", "hand_slot", "position", "turn_number"], "played move");
  assertOneOf(item.player, ["queen", "king"], "move player");
  assertString(item.card_id, "move card");
  assertNumber(item.hand_slot, "move hand slot");
  assertNumber(item.position, "move position");
  assertNumber(item.turn_number, "move turn");
}

export function assertLineScore(value: unknown): void {
  const item = record(value, "line score");
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
  assertArray(item.card_ids, "line cards");
  if (item.card_ids.length !== 3) throw new TypeError("line score requires three cards");
  item.card_ids.forEach((card) => assertString(card, "line card"));
  assertNumber(item.base_value, "line base value");
  if (![0, 1, 2, 3, 5].includes(item.multiplier as number)) {
    throw new TypeError("line multiplier is unsupported");
  }
  assertOneOf(
    item.multiplier_reason,
    ["none", "suit_pair", "same_color", "same_suit", "vampire"],
    "multiplier reason",
  );
  assertString(item.multiplier_label, "multiplier label");
  assertArray(item.highlighted_card_ids, "highlighted cards");
  item.highlighted_card_ids.forEach((card) => assertString(card, "highlighted card"));
  assertNumber(item.total, "line total");
}

export function assertScoringStep(value: unknown): void {
  const item = record(value, "scoring step");
  exactKeys(item, ["kind", "player", "line", "details"], "scoring step");
  assertOneOf(
    item.kind,
    ["score_line", "compare_candidates", "select_round_score", "update_total"],
    "scoring step kind",
  );
  if (item.player !== null) assertOneOf(item.player, ["queen", "king"], "scoring player");
  if (item.line !== null) assertLineScore(item.line);
  const details = record(item.details, "scoring details");
  Object.values(details).forEach((detail) => {
    if (!["string", "number", "boolean"].includes(typeof detail)) {
      throw new TypeError("scoring detail must be a primitive value");
    }
  });
}

export function assertRoundRecord(value: unknown): void {
  const item = record(value, "round record");
  exactKeys(
    item,
    [
      "round_number",
      "dealer",
      "coffin",
      "moves",
      "line_scores",
      "scoring_sequence",
      "round_scores",
    ],
    "round record",
  );
  assertNumber(item.round_number, "round number");
  assertOneOf(item.dealer, ["queen", "king"], "round dealer");
  assertArray(item.coffin, "round coffin");
  if (item.coffin.length !== 9) throw new TypeError("round coffin requires nine cards");
  item.coffin.forEach((card) => assertString(card, "round coffin card"));
  assertArray(item.moves, "round moves");
  item.moves.forEach(assertPlayedMove);
  assertArray(item.line_scores, "round line scores");
  item.line_scores.forEach(assertLineScore);
  assertArray(item.scoring_sequence, "scoring sequence");
  item.scoring_sequence.forEach(assertScoringStep);
  assertPlayerScore(item.round_scores);
}

export function assertPhase(value: unknown): void {
  const item = record(value, "resumable phase");
  assertString(item.kind, "phase kind");
  switch (item.kind) {
    case "human_turn":
      exactKeys(item, ["kind"], "human-turn phase");
      return;
    case "opponent_turn":
      exactKeys(item, ["kind", "status", "job_id", "retryable"], "opponent-turn phase");
      assertOneOf(item.status, ["ready", "pending", "failed"], "opponent status");
      assertNullableString(item.job_id, "opponent job ID");
      assertBoolean(item.retryable, "opponent retryable");
      return;
    case "narration":
      exactKeys(
        item,
        ["kind", "status", "event_id", "narration_id", "required"],
        "narration phase",
      );
      assertOneOf(item.status, ["ready", "pending", "failed"], "narration status");
      assertString(item.event_id, "narration event ID");
      assertNullableString(item.narration_id, "narration ID");
      assertBoolean(item.required, "narration required");
      return;
    case "scoring":
      exactKeys(
        item,
        ["kind", "round_number", "next_step_index", "pending_narration_id"],
        "scoring phase",
      );
      assertNumber(item.round_number, "scoring round");
      assertNumber(item.next_step_index, "scoring step index");
      assertNullableString(item.pending_narration_id, "pending narration ID");
      return;
    case "round_advance":
      exactKeys(item, ["kind", "round_number"], "round-advance phase");
      assertNumber(item.round_number, "advance round");
      return;
    case "game_complete":
      exactKeys(item, ["kind", "outcome"], "game-complete phase");
      assertOneOf(item.outcome, ["human", "opponent", "tie"], "game outcome");
      return;
    default:
      throw new TypeError("resumable phase kind is unsupported");
  }
}

export function assertLegalMove(value: unknown): void {
  const item = record(value, "legal move");
  exactKeys(item, ["move_id", "card_id", "hand_slot", "position"], "legal move");
  assertString(item.move_id, "move ID");
  assertString(item.card_id, "legal move card");
  assertNumber(item.hand_slot, "legal move hand slot");
  assertNumber(item.position, "legal move position");
}

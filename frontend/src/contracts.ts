export type Player = "queen" | "king";
export type GameStatus = "playing" | "round_complete" | "game_complete";

export interface HealthResponse {
  schema_version: "dracula-health-v1";
  api_version: "dracula-api-v1";
  status: "ok";
  narration_enabled: boolean;
}

export interface CreateGameRequest {
  human_role: Player;
  request_id: string;
  seed?: string | null;
}

export interface MoveRequest {
  move_id: string;
  expected_version: number;
  request_id: string;
}

export interface VersionedMutationRequest {
  expected_version: number;
  request_id: string;
}

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

export interface PublicGameView {
  schema_version: "dracula-human-game-view-v1";
  game_id: string;
  version: number;
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
  latest_event_sequence: number;
  narration_enabled: boolean;
}

export interface HumanGameView extends PublicGameView {
  human_hand: [string | null, string | null, string | null, string | null];
  legal_moves: LegalMove[];
  events: PublicEvent[];
}

export interface PublicEvent {
  schema_version: "dracula-public-event-v1";
  game_id: string;
  event_id: string;
  sequence: number;
  event_type:
    | "game_created"
    | "round_started"
    | "move_accepted"
    | "row_score_ready"
    | "column_score_ready"
    | "round_completed"
    | "game_completed";
  occurred_at: string;
  prior_version: number;
  resulting_version: number;
  request_id: string | null;
  payload: Record<string, number | string | boolean | null>;
}

export interface EventsResponse {
  events: PublicEvent[];
  latest_sequence: number;
}

export interface ApiErrorResponse {
  schema_version: "dracula-error-v1";
  code:
    | "not_found"
    | "stale_version"
    | "wrong_turn"
    | "wrong_phase"
    | "already_advanced"
    | "invalid_move"
    | "request_id_conflict"
    | "rate_limited"
    | "dependency_unavailable"
    | "validation_error";
  message: string;
  retryable: boolean;
  current_version: number | null;
  current_game: HumanGameView | null;
}

type JsonRecord = Record<string, unknown>;

function record(value: unknown, label: string): JsonRecord {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new TypeError(`${label} must be an object`);
  }
  return value as JsonRecord;
}

function exactKeys(value: JsonRecord, keys: readonly string[], label: string): void {
  const actual = Object.keys(value).sort();
  const expected = [...keys].sort();
  if (actual.length !== expected.length || actual.some((key, index) => key !== expected[index])) {
    throw new TypeError(`${label} has an unexpected field set`);
  }
}

function string(value: unknown, label: string): asserts value is string {
  if (typeof value !== "string") throw new TypeError(`${label} must be a string`);
}

function number(value: unknown, label: string): asserts value is number {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new TypeError(`${label} must be a finite number`);
  }
}

function boolean(value: unknown, label: string): asserts value is boolean {
  if (typeof value !== "boolean") throw new TypeError(`${label} must be a boolean`);
}

function oneOf<T extends string>(
  value: unknown,
  choices: readonly T[],
  label: string,
): asserts value is T {
  if (typeof value !== "string" || !choices.includes(value as T)) {
    throw new TypeError(`${label} has an unsupported value`);
  }
}

function nullableString(value: unknown, label: string): void {
  if (value !== null) string(value, label);
}

function array(value: unknown, label: string): asserts value is unknown[] {
  if (!Array.isArray(value)) throw new TypeError(`${label} must be an array`);
}

function assertPlayerScore(value: unknown): void {
  const item = record(value, "player score");
  exactKeys(item, ["human", "opponent"], "player score");
  number(item.human, "human score");
  number(item.opponent, "opponent score");
}

function assertPlayedMove(value: unknown): void {
  const item = record(value, "played move");
  exactKeys(item, ["player", "card_id", "hand_slot", "position", "turn_number"], "played move");
  oneOf(item.player, ["queen", "king"], "move player");
  string(item.card_id, "move card");
  number(item.hand_slot, "move hand slot");
  number(item.position, "move position");
  number(item.turn_number, "move turn");
}

function assertLineScore(value: unknown): void {
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
  oneOf(item.direction, ["row", "column"], "line direction");
  number(item.index, "line index");
  array(item.card_ids, "line cards");
  if (item.card_ids.length !== 3) throw new TypeError("line score requires three cards");
  item.card_ids.forEach((card) => string(card, "line card"));
  number(item.base_value, "line base value");
  if (![0, 1, 2, 3, 5].includes(item.multiplier as number)) {
    throw new TypeError("line multiplier is unsupported");
  }
  oneOf(
    item.multiplier_reason,
    ["none", "suit_pair", "same_color", "same_suit", "vampire"],
    "multiplier reason",
  );
  string(item.multiplier_label, "multiplier label");
  array(item.highlighted_card_ids, "highlighted cards");
  item.highlighted_card_ids.forEach((card) => string(card, "highlighted card"));
  number(item.total, "line total");
}

function assertScoringStep(value: unknown): void {
  const item = record(value, "scoring step");
  exactKeys(item, ["kind", "player", "line", "details"], "scoring step");
  oneOf(
    item.kind,
    ["score_line", "compare_candidates", "select_round_score", "update_total"],
    "scoring step kind",
  );
  if (item.player !== null) oneOf(item.player, ["queen", "king"], "scoring player");
  if (item.line !== null) assertLineScore(item.line);
  const details = record(item.details, "scoring details");
  Object.values(details).forEach((detail) => {
    if (!["string", "number", "boolean"].includes(typeof detail)) {
      throw new TypeError("scoring detail must be a primitive value");
    }
  });
}

function assertRoundRecord(value: unknown): void {
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
  number(item.round_number, "round number");
  oneOf(item.dealer, ["queen", "king"], "round dealer");
  array(item.coffin, "round coffin");
  if (item.coffin.length !== 9) throw new TypeError("round coffin requires nine cards");
  item.coffin.forEach((card) => string(card, "round coffin card"));
  array(item.moves, "round moves");
  item.moves.forEach(assertPlayedMove);
  array(item.line_scores, "round line scores");
  item.line_scores.forEach(assertLineScore);
  array(item.scoring_sequence, "scoring sequence");
  item.scoring_sequence.forEach(assertScoringStep);
  assertPlayerScore(item.round_scores);
}

function assertPhase(value: unknown): void {
  const item = record(value, "resumable phase");
  string(item.kind, "phase kind");
  switch (item.kind) {
    case "human_turn":
      exactKeys(item, ["kind"], "human-turn phase");
      return;
    case "opponent_turn":
      exactKeys(item, ["kind", "status", "job_id", "retryable"], "opponent-turn phase");
      oneOf(item.status, ["ready", "pending", "failed"], "opponent status");
      nullableString(item.job_id, "opponent job ID");
      boolean(item.retryable, "opponent retryable");
      return;
    case "narration":
      exactKeys(
        item,
        ["kind", "status", "event_id", "narration_id", "required"],
        "narration phase",
      );
      oneOf(item.status, ["ready", "pending", "failed"], "narration status");
      string(item.event_id, "narration event ID");
      nullableString(item.narration_id, "narration ID");
      boolean(item.required, "narration required");
      return;
    case "scoring":
      exactKeys(
        item,
        ["kind", "round_number", "next_step_index", "pending_narration_id"],
        "scoring phase",
      );
      number(item.round_number, "scoring round");
      number(item.next_step_index, "scoring step index");
      nullableString(item.pending_narration_id, "pending narration ID");
      return;
    case "round_advance":
      exactKeys(item, ["kind", "round_number"], "round-advance phase");
      number(item.round_number, "advance round");
      return;
    case "game_complete":
      exactKeys(item, ["kind", "outcome"], "game-complete phase");
      oneOf(item.outcome, ["human", "opponent", "tie"], "game outcome");
      return;
    default:
      throw new TypeError("resumable phase kind is unsupported");
  }
}

function assertLegalMove(value: unknown): void {
  const item = record(value, "legal move");
  exactKeys(item, ["move_id", "card_id", "hand_slot", "position"], "legal move");
  string(item.move_id, "move ID");
  string(item.card_id, "legal move card");
  number(item.hand_slot, "legal move hand slot");
  number(item.position, "legal move position");
}

export function assertHealthResponse(value: unknown): asserts value is HealthResponse {
  const item = record(value, "health response");
  exactKeys(item, ["schema_version", "api_version", "status", "narration_enabled"], "health response");
  oneOf(item.schema_version, ["dracula-health-v1"], "health schema");
  oneOf(item.api_version, ["dracula-api-v1"], "API version");
  oneOf(item.status, ["ok"], "health status");
  boolean(item.narration_enabled, "narration enabled");
}

export function assertHumanGameView(value: unknown): asserts value is HumanGameView {
  const item = record(value, "human game view");
  exactKeys(
    item,
    [
      "schema_version",
      "game_id",
      "version",
      "status",
      "round_number",
      "turn_number",
      "dealer",
      "active_player",
      "human_role",
      "opponent_role",
      "coffin",
      "current_round_moves",
      "pending_round_result",
      "completed_rounds",
      "total_scores",
      "phase",
      "latest_event_sequence",
      "narration_enabled",
      "human_hand",
      "legal_moves",
      "events",
    ],
    "human game view",
  );
  oneOf(item.schema_version, ["dracula-human-game-view-v1"], "game-view schema");
  string(item.game_id, "game ID");
  number(item.version, "game version");
  oneOf(item.status, ["playing", "round_complete", "game_complete"], "game status");
  number(item.round_number, "game round");
  number(item.turn_number, "game turn");
  oneOf(item.dealer, ["queen", "king"], "dealer");
  if (item.active_player !== null) oneOf(item.active_player, ["queen", "king"], "active player");
  oneOf(item.human_role, ["queen", "king"], "human role");
  oneOf(item.opponent_role, ["queen", "king"], "opponent role");
  array(item.coffin, "coffin");
  if (item.coffin.length !== 9) throw new TypeError("coffin requires nine slots");
  item.coffin.forEach((card) => nullableString(card, "coffin card"));
  array(item.current_round_moves, "current-round moves");
  item.current_round_moves.forEach(assertPlayedMove);
  if (item.pending_round_result !== null) assertRoundRecord(item.pending_round_result);
  array(item.completed_rounds, "completed rounds");
  item.completed_rounds.forEach(assertRoundRecord);
  assertPlayerScore(item.total_scores);
  assertPhase(item.phase);
  number(item.latest_event_sequence, "latest event sequence");
  boolean(item.narration_enabled, "narration enabled");
  array(item.human_hand, "human hand");
  if (item.human_hand.length !== 4) throw new TypeError("human hand requires four slots");
  item.human_hand.forEach((card) => nullableString(card, "human hand card"));
  array(item.legal_moves, "legal moves");
  item.legal_moves.forEach(assertLegalMove);
  array(item.events, "public events");
  item.events.forEach(assertPublicEvent);
}

export function assertPublicEvent(value: unknown): asserts value is PublicEvent {
  const item = record(value, "public event");
  exactKeys(
    item,
    [
      "schema_version",
      "game_id",
      "event_id",
      "sequence",
      "event_type",
      "occurred_at",
      "prior_version",
      "resulting_version",
      "request_id",
      "payload",
    ],
    "public event",
  );
  oneOf(item.schema_version, ["dracula-public-event-v1"], "event schema");
  string(item.game_id, "event game ID");
  string(item.event_id, "event ID");
  number(item.sequence, "event sequence");
  oneOf(
    item.event_type,
    [
      "game_created",
      "round_started",
      "move_accepted",
      "row_score_ready",
      "column_score_ready",
      "round_completed",
      "game_completed",
    ],
    "event type",
  );
  string(item.occurred_at, "event timestamp");
  number(item.prior_version, "prior version");
  number(item.resulting_version, "resulting version");
  nullableString(item.request_id, "request ID");
  const payload = record(item.payload, "event payload");
  Object.values(payload).forEach((entry) => {
    if (entry !== null && !["string", "number", "boolean"].includes(typeof entry)) {
      throw new TypeError("event payload values must be public primitives");
    }
  });
}

export function assertEventsResponse(value: unknown): asserts value is EventsResponse {
  const item = record(value, "events response");
  exactKeys(item, ["events", "latest_sequence"], "events response");
  array(item.events, "events");
  item.events.forEach(assertPublicEvent);
  number(item.latest_sequence, "latest event sequence");
}

export function assertApiErrorResponse(value: unknown): asserts value is ApiErrorResponse {
  const item = record(value, "API error");
  exactKeys(
    item,
    ["schema_version", "code", "message", "retryable", "current_version", "current_game"],
    "API error",
  );
  oneOf(item.schema_version, ["dracula-error-v1"], "error schema");
  oneOf(
    item.code,
    [
      "not_found",
      "stale_version",
      "wrong_turn",
      "wrong_phase",
      "already_advanced",
      "invalid_move",
      "request_id_conflict",
      "rate_limited",
      "dependency_unavailable",
      "validation_error",
    ],
    "error code",
  );
  string(item.message, "error message");
  boolean(item.retryable, "error retryable");
  if (item.current_version !== null) number(item.current_version, "current version");
  if (item.current_game !== null) assertHumanGameView(item.current_game);
}

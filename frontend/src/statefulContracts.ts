import {
  assertArray,
  assertBoolean,
  assertLegalMove,
  assertNullableString,
  assertNumber,
  assertOneOf,
  assertPhase,
  assertPlayedMove,
  assertPlayerScore,
  assertRoundRecord,
  assertString,
  exactKeys,
  record,
  type GameStatus,
  type LegalMove,
  type PlayedMove,
  type Player,
  type PlayerScore,
  type ResumablePhase,
  type RoundRecord,
} from "./contractPrimitives";

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

export function assertHealthResponse(value: unknown): asserts value is HealthResponse {
  const item = record(value, "health response");
  exactKeys(item, ["schema_version", "api_version", "status", "narration_enabled"], "health response");
  assertOneOf(item.schema_version, ["dracula-health-v1"], "health schema");
  assertOneOf(item.api_version, ["dracula-api-v1"], "API version");
  assertOneOf(item.status, ["ok"], "health status");
  assertBoolean(item.narration_enabled, "narration enabled");
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
  assertOneOf(item.schema_version, ["dracula-human-game-view-v1"], "game-view schema");
  assertString(item.game_id, "game ID");
  assertNumber(item.version, "game version");
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
  assertArray(item.current_round_moves, "current-round moves");
  item.current_round_moves.forEach(assertPlayedMove);
  if (item.pending_round_result !== null) assertRoundRecord(item.pending_round_result);
  assertArray(item.completed_rounds, "completed rounds");
  item.completed_rounds.forEach(assertRoundRecord);
  assertPlayerScore(item.total_scores);
  assertPhase(item.phase);
  assertNumber(item.latest_event_sequence, "latest event sequence");
  assertBoolean(item.narration_enabled, "narration enabled");
  assertArray(item.human_hand, "human hand");
  if (item.human_hand.length !== 4) throw new TypeError("human hand requires four slots");
  item.human_hand.forEach((card) => assertNullableString(card, "human hand card"));
  assertArray(item.legal_moves, "legal moves");
  item.legal_moves.forEach(assertLegalMove);
  assertArray(item.events, "public events");
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
  assertOneOf(item.schema_version, ["dracula-public-event-v1"], "event schema");
  assertString(item.game_id, "event game ID");
  assertString(item.event_id, "event ID");
  assertNumber(item.sequence, "event sequence");
  assertOneOf(
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
  assertString(item.occurred_at, "event timestamp");
  assertNumber(item.prior_version, "prior version");
  assertNumber(item.resulting_version, "resulting version");
  assertNullableString(item.request_id, "request ID");
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
  assertArray(item.events, "events");
  item.events.forEach(assertPublicEvent);
  assertNumber(item.latest_sequence, "latest event sequence");
}

export function assertApiErrorResponse(value: unknown): asserts value is ApiErrorResponse {
  const item = record(value, "API error");
  exactKeys(
    item,
    ["schema_version", "code", "message", "retryable", "current_version", "current_game"],
    "API error",
  );
  assertOneOf(
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
  assertOneOf(item.schema_version, ["dracula-error-v1"], "error schema");
  assertString(item.message, "error message");
  assertBoolean(item.retryable, "error retryable");
  if (item.current_version !== null) assertNumber(item.current_version, "current version");
  if (item.current_game !== null) assertHumanGameView(item.current_game);
}

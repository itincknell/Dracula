import {
  assertApiErrorResponse,
  assertEventsResponse,
  assertHealthResponse,
  assertHumanGameView,
  type ApiErrorResponse,
  type CreateGameRequest,
  type EventsResponse,
  type HealthResponse,
  type HumanGameView,
  type MoveRequest,
  type VersionedMutationRequest,
} from "./contracts";

export interface ApiResult<T, Status extends number> {
  status: Status;
  data: T;
}

export type GameMutationResult = ApiResult<HumanGameView, 200 | 202>;

export class ApiError extends Error {
  constructor(
    readonly status: number,
    readonly response: ApiErrorResponse,
  ) {
    super(response.message);
    this.name = "ApiError";
  }
}

export interface ApiClient {
  health(): Promise<HealthResponse>;
  createGame(request: CreateGameRequest): Promise<ApiResult<HumanGameView, 201>>;
  getGame(gameId: string): Promise<ApiResult<HumanGameView, 200>>;
  submitMove(gameId: string, request: MoveRequest): Promise<ApiResult<HumanGameView, 200>>;
  opponentTurn(gameId: string, request: VersionedMutationRequest): Promise<GameMutationResult>;
  advanceRound(
    gameId: string,
    roundNumber: number,
    request: VersionedMutationRequest,
  ): Promise<ApiResult<HumanGameView, 200>>;
  getEvents(gameId: string, afterSequence?: number): Promise<ApiResult<EventsResponse, 200>>;
}

type FetchImplementation = (input: RequestInfo | URL, init?: RequestInit) => Promise<Response>;

export function resolveApiBaseUrl(configured: string | undefined): string {
  const value = configured?.trim() || "/api";
  return value === "/" ? "" : value.replace(/\/+$/, "");
}

export function createApiClient(
  baseUrl: string,
  fetchImplementation: FetchImplementation = fetch,
): ApiClient {
  const resolvedBaseUrl = resolveApiBaseUrl(baseUrl);

  async function requestJson<T, Status extends number>(
    path: string,
    expectedStatuses: readonly Status[],
    validate: (value: unknown) => asserts value is T,
    init?: RequestInit,
  ): Promise<ApiResult<T, Status>> {
    const response = await fetchImplementation(`${resolvedBaseUrl}${path}`, {
      ...init,
      headers: {
        Accept: "application/json",
        ...(init?.body === undefined ? {} : { "Content-Type": "application/json" }),
        ...init?.headers,
      },
    });
    const payload: unknown = await response.json();
    if (!response.ok) {
      assertApiErrorResponse(payload);
      throw new ApiError(response.status, payload);
    }
    if (!expectedStatuses.includes(response.status as Status)) {
      throw new TypeError(`Unexpected successful API status ${response.status}`);
    }
    validate(payload);
    return { status: response.status as Status, data: payload };
  }

  const body = (value: unknown): string => JSON.stringify(value);

  return {
    async health(): Promise<HealthResponse> {
      return (await requestJson("/health", [200], assertHealthResponse)).data;
    },
    createGame(request) {
      return requestJson("/games", [201], assertHumanGameView, {
        method: "POST",
        body: body(request),
      });
    },
    getGame(gameId) {
      return requestJson(`/games/${encodeURIComponent(gameId)}`, [200], assertHumanGameView);
    },
    submitMove(gameId, request) {
      return requestJson(
        `/games/${encodeURIComponent(gameId)}/moves`,
        [200],
        assertHumanGameView,
        { method: "POST", body: body(request) },
      );
    },
    opponentTurn(gameId, request) {
      return requestJson(
        `/games/${encodeURIComponent(gameId)}/opponent-turn`,
        [200, 202],
        assertHumanGameView,
        { method: "POST", body: body(request) },
      );
    },
    advanceRound(gameId, roundNumber, request) {
      return requestJson(
        `/games/${encodeURIComponent(gameId)}/rounds/${roundNumber}/advance`,
        [200],
        assertHumanGameView,
        { method: "POST", body: body(request) },
      );
    },
    getEvents(gameId, afterSequence = 0) {
      const query = new URLSearchParams({ after_sequence: String(afterSequence) });
      return requestJson(
        `/games/${encodeURIComponent(gameId)}/events?${query}`,
        [200],
        assertEventsResponse,
      );
    },
  };
}

export const apiClient = createApiClient(resolveApiBaseUrl(import.meta.env.VITE_API_BASE_URL));

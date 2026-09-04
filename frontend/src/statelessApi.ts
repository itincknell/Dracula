import {
  assertNarrationResponse,
  assertStatelessApiErrorResponse,
  assertStatelessGameResponse,
  assertStatelessHealthResponse,
  type NarrationCueType,
  type NarrationResponse,
  type RecoveryEnvelope,
  type RequestedGameCommand,
  type StatelessApiErrorResponse,
  type StatelessGameResponse,
  type StatelessHealthResponse,
} from "./statelessContracts";
import type { Player } from "./contractPrimitives";

export interface ApiResult<T, Status extends number> {
  status: Status;
  data: T;
}

export class StatelessApiError extends Error {
  constructor(
    readonly status: number,
    readonly response: StatelessApiErrorResponse,
  ) {
    super(response.message);
    this.name = "StatelessApiError";
  }
}

export interface StatelessApiClient {
  health(): Promise<StatelessHealthResponse>;
  startGame(request: { human_role: Player; seed?: string }): Promise<ApiResult<StatelessGameResponse, 201>>;
  applyCommand(request: {
    envelope: RecoveryEnvelope;
    command: RequestedGameCommand;
  }): Promise<ApiResult<StatelessGameResponse, 200>>;
  resumeGame(request: {
    envelope: RecoveryEnvelope;
  }): Promise<ApiResult<StatelessGameResponse, 200>>;
  narrate(request: {
    envelope: RecoveryEnvelope;
    cue_type: NarrationCueType;
  }): Promise<ApiResult<NarrationResponse, 200>>;
}

type FetchImplementation = (input: RequestInfo | URL, init?: RequestInit) => Promise<Response>;

export function resolveStatelessApiOrigin(configured: string | undefined): string {
  const value = configured?.trim() || "/api";
  return value === "/" ? "" : value.replace(/\/+$/, "");
}

export function createStatelessApiClient(
  origin: string,
  fetchImplementation: FetchImplementation = fetch,
): StatelessApiClient {
  const base = resolveStatelessApiOrigin(origin);

  async function requestJson<T, Status extends number>(
    path: string,
    status: Status,
    validate: (value: unknown) => asserts value is T,
    init?: RequestInit,
  ): Promise<ApiResult<T, Status>> {
    const response = await fetchImplementation(`${base}${path}`, {
      ...init,
      cache: "no-store",
      headers: {
        Accept: "application/json",
        ...(init?.body === undefined ? {} : { "Content-Type": "application/json" }),
        ...init?.headers,
      },
    });
    const responseText = await response.text();
    if (responseText.trim().length === 0) {
      throw new TypeError("The game server returned no response. Please retry.");
    }
    let payload: unknown;
    try {
      payload = JSON.parse(responseText);
    } catch {
      throw new TypeError("The game server returned an invalid response. Please retry.");
    }
    if (!response.ok) {
      assertStatelessApiErrorResponse(payload);
      throw new StatelessApiError(response.status, payload);
    }
    if (response.status !== status) {
      throw new TypeError(`Unexpected successful API status ${response.status}`);
    }
    validate(payload);
    return { status, data: payload };
  }

  const post = <T, Status extends number>(
    path: string,
    status: Status,
    validate: (value: unknown) => asserts value is T,
    body: unknown,
  ) => requestJson(path, status, validate, {
    method: "POST",
    body: JSON.stringify(body),
  });

  return {
    async health() {
      return (await requestJson("/health", 200, assertStatelessHealthResponse)).data;
    },
    startGame(request) {
      return post("/games", 201, assertStatelessGameResponse, request);
    },
    applyCommand(request) {
      return post("/games/command", 200, assertStatelessGameResponse, request);
    },
    resumeGame(request) {
      return post("/games/resume", 200, assertStatelessGameResponse, request);
    },
    narrate(request) {
      return post("/narration", 200, assertNarrationResponse, request);
    },
  };
}

export const statelessApiClient = createStatelessApiClient(
  resolveStatelessApiOrigin(import.meta.env.VITE_API_ORIGIN),
);

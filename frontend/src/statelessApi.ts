/**
 * Implements the browser's HTTP client for the stateless gameplay API.
 * It resolves local or production origins, sends recovery envelopes and
 * commands, and validates every untrusted JSON response before returning it.
 */
import {
  assertNarrationResponse,
  assertStatelessApiErrorResponse,
  assertStatelessGameResponse,
  type NarrationCueType,
  type NarrationResponse,
  type RecoveryEnvelope,
  type RequestedGameCommand,
  type StatelessApiErrorResponse,
  type StatelessGameResponse,
} from "./statelessContracts";
import type { Player } from "./contractPrimitives";

/** An HTTP failure whose already-validated public details may be shown by the UI. */
export class StatelessApiError extends Error {
  constructor(readonly response: StatelessApiErrorResponse) {
    super(response.message);
    this.name = "StatelessApiError";
  }
}

/**
 * The requests used by the browser gameplay controller.
 *
 * Each method returns validated response data directly. HTTP status codes are
 * checked inside the client because callers cannot usefully act on a different
 * successful status for these fixed endpoints.
 */
export interface StatelessApiClient {
  startGame(request: { human_role: Player; seed?: string }): Promise<StatelessGameResponse>;
  applyCommand(request: {
    envelope: RecoveryEnvelope;
    command: RequestedGameCommand;
  }): Promise<StatelessGameResponse>;
  resumeGame(request: {
    envelope: RecoveryEnvelope;
  }): Promise<StatelessGameResponse>;
  narrate(request: {
    envelope: RecoveryEnvelope;
    cue_type: NarrationCueType;
  }): Promise<NarrationResponse>;
}

/** The subset of `fetch` injected by tests; production uses the browser function. */
type FetchImplementation = (input: RequestInfo | URL, init?: RequestInit) => Promise<Response>;

/**
 * Normalize the build-time API origin before endpoint paths are appended.
 * `/api` uses Vite's local proxy. A configured absolute origin has trailing
 * slashes removed, while `/` becomes the empty same-origin prefix.
 */
export function resolveStatelessApiOrigin(configured: string | undefined): string {
  const value = configured?.trim() || "/api";
  return value === "/" ? "" : value.replace(/\/+$/, "");
}

export function createStatelessApiClient(
  origin: string | undefined,
  fetchImplementation: FetchImplementation = fetch,
): StatelessApiClient {
  const base = resolveStatelessApiOrigin(origin);

  /** Send one request, decode its body once, and establish the trusted boundary. */
  async function requestJson<T, Status extends number>(
    path: string,
    status: Status,
    validate: (value: unknown) => asserts value is T,
    init?: RequestInit,
  ): Promise<T> {
    // `no-store` prevents an intermediary browser cache from replaying a stale
    // game response. Idempotence comes from the recovery envelope, not HTTP
    // response caching.
    const response = await fetchImplementation(`${base}${path}`, {
      ...init,
      cache: "no-store",
      headers: {
        Accept: "application/json",
        ...(init?.body === undefined ? {} : { "Content-Type": "application/json" }),
        ...init?.headers,
      },
    });

    // Read text before parsing so an interrupted response produces a useful
    // message instead of the browser's opaque "Unexpected end of JSON" error.
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

    // Error bodies have their own public contract. Successfully shaped errors
    // retain their retry guidance without exposing an arbitrary server body.
    if (!response.ok) {
      assertStatelessApiErrorResponse(payload);
      throw new StatelessApiError(payload);
    }

    // Each endpoint has one documented success code. Accepting another code
    // could conceal a changed server contract even if its body looked familiar.
    if (response.status !== status) {
      throw new TypeError(`Unexpected successful API status ${response.status}`);
    }
    validate(payload);
    return payload;
  }

  /** Serialize the JSON body and apply the common POST request contract. */
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
  import.meta.env.VITE_API_ORIGIN,
);

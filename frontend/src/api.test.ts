import { describe, expect, it, vi } from "vitest";

import errorFixture from "../../contracts/v1/api-error.json";
import healthFixture from "../../contracts/v1/health.json";
import gameViewFixture from "../../contracts/v1/human-game-view.json";
import publicEventFixture from "../../contracts/v1/public-event.json";
import { createApiClient, resolveApiBaseUrl } from "./api";

describe("API client configuration", () => {
  it("uses one client implementation for local and deployed base URLs", async () => {
    const request = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      void input;
      void init;
      return new Response(JSON.stringify(healthFixture), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      });
    });

    await createApiClient("http://127.0.0.1:8000/", request).health();
    await createApiClient("https://api.example.test/v1", request).health();

    expect(request.mock.calls.map((call) => call[0])).toEqual([
      "http://127.0.0.1:8000/health",
      "https://api.example.test/v1/health",
    ]);
  });

  it("uses the same-origin API path when no environment override is supplied", () => {
    expect(resolveApiBaseUrl(undefined)).toBe("/api");
  });

  it("supports every gameplay endpoint and preserves 200/202 opponent statuses", async () => {
    const statuses = [201, 200, 200, 202, 200, 200, 200];
    const payloads = [
      gameViewFixture,
      gameViewFixture,
      gameViewFixture,
      gameViewFixture,
      gameViewFixture,
      gameViewFixture,
      { events: [publicEventFixture], latest_sequence: 1 },
    ];
    const request = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      void input;
      void init;
      const status = statuses.shift() ?? 500;
      return new Response(JSON.stringify(payloads.shift()), {
        status,
        headers: { "Content-Type": "application/json" },
      });
    });
    const client = createApiClient("/api", request);
    await client.createGame({ human_role: "queen", request_id: "request-1" });
    await client.getGame("game/one");
    await client.submitMove("game-1", {
      move_id: "move-1",
      expected_version: 0,
      request_id: "request-2",
    });
    expect((await client.opponentTurn("game-1", { expected_version: 1, request_id: "job" })).status).toBe(202);
    expect((await client.opponentTurn("game-1", { expected_version: 1, request_id: "job" })).status).toBe(200);
    await client.advanceRound("game-1", 1, { expected_version: 8, request_id: "advance" });
    await client.getEvents("game-1", 4);

    expect(request.mock.calls.map(([url]) => url)).toEqual([
      "/api/games",
      "/api/games/game%2Fone",
      "/api/games/game-1/moves",
      "/api/games/game-1/opponent-turn",
      "/api/games/game-1/opponent-turn",
      "/api/games/game-1/rounds/1/advance",
      "/api/games/game-1/events?after_sequence=4",
    ]);
  });

  it("surfaces the stable error body to reconciliation code", async () => {
    const request = vi.fn(async () => new Response(JSON.stringify(errorFixture), {
      status: 409,
      headers: { "Content-Type": "application/json" },
    }));
    const client = createApiClient("/api", request);
    await expect(client.getGame("stale")).rejects.toMatchObject({
      status: 409,
      response: { code: "stale_version" },
    });
  });
});

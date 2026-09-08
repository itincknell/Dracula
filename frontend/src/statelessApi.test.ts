/**
 * Exercises the stateless HTTP client's origin selection and response handling.
 * Mocked fetch calls verify request shapes, trusted parsing, and public errors
 * without duplicating server-side gameplay tests.
 */
import { describe, expect, it, vi } from "vitest";

import {
  assertStatelessGameResponse,
  type StatelessGameResponse,
} from "./statelessContracts";
import { createStatelessApiClient, resolveStatelessApiOrigin } from "./statelessApi";

const response = (): StatelessGameResponse => ({
  envelope: {
    seed: "browser-seed",
    history: [{ type: "select_role", human_role: "queen" }],
  },
  game: {
    status: "playing",
    round_number: 1,
    turn_number: 0,
    dealer: "queen",
    active_player: "queen",
    human_role: "queen",
    opponent_role: "king",
    coffin: [null, null, null, null, "4C", null, null, null, null],
    current_round_moves: [],
    pending_round_result: null,
    completed_rounds: [],
    total_scores: { human: 0, opponent: 0 },
    phase: { kind: "human_turn" },
    human_hand: ["2C", "10D", "QH", "KS"],
    legal_moves: [{ card_id: "2C", hand_slot: 0, position: 1 }],
  },
});

describe("stateless production API client", () => {
  it("uses the build-time origin while retaining the local /api default", () => {
    expect(resolveStatelessApiOrigin(undefined)).toBe("/api");
    expect(resolveStatelessApiOrigin("https://api.example.test/")).toBe("https://api.example.test");
  });

  it("uses only the stateless gameplay and narration endpoints", async () => {
    const game = response();
    const replies: Array<[number, unknown]> = [
      [201, game],
      [200, game],
      [200, game],
      [200, { cue_type: "opening", status: "ready", text: "Enter." }],
    ];
    const request = vi.fn(async (_input: RequestInfo | URL, _init?: RequestInit) => {
      void _input;
      void _init;
      const next = replies.shift();
      if (next === undefined) throw new Error("unexpected request");
      return new Response(JSON.stringify(next[1]), {
        status: next[0],
        headers: { "Content-Type": "application/json" },
      });
    });
    const client = createStatelessApiClient("https://api.example.test", request);
    await client.startGame({ human_role: "queen" });
    await client.applyCommand({
      envelope: game.envelope,
      command: { type: "place", hand_slot: 0, position: 1 },
    });
    await client.resumeGame({ envelope: game.envelope });
    await client.narrate({ envelope: game.envelope, cue_type: "opening" });

    expect(request.mock.calls.map(([url]) => url)).toEqual([
      "https://api.example.test/games",
      "https://api.example.test/games/command",
      "https://api.example.test/games/resume",
      "https://api.example.test/narration",
    ]);
    expect(request.mock.calls.every(([, init]) => init?.cache === "no-store")).toBe(true);
  });

  it("rejects malformed success payloads", async () => {
    const request = vi.fn(async () => new Response(JSON.stringify({ unexpected: true }), {
      status: 201,
      headers: { "Content-Type": "application/json" },
    }));
    await expect(createStatelessApiClient("/api", request).startGame({ human_role: "queen" }))
      .rejects.toBeInstanceOf(TypeError);
  });

  it("rejects an unsupported phase in a server response", () => {
    const payload = response();
    expect(() => assertStatelessGameResponse({
      ...payload,
      game: { ...payload.game, phase: { kind: "waiting" } },
    })).toThrow("game phase kind is unsupported");
  });

  it("reports an empty interrupted response without exposing a JSON parser error", async () => {
    const request = vi.fn(async () => new Response(null, { status: 502 }));
    await expect(createStatelessApiClient("/api", request).startGame({ human_role: "queen" }))
      .rejects.toThrow("The game server returned no response. Please retry.");
  });
});

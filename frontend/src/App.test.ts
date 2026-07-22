import { describe, expect, it } from "vitest";

import { resolveRoute } from "./App";

describe("application routes", () => {
  it("resolves start, game, and rules routes without a routing dependency", () => {
    expect(resolveRoute("/")).toEqual({ kind: "start" });
    expect(resolveRoute("/games/game-123")).toEqual({ kind: "game", gameId: "game-123" });
    expect(resolveRoute("/rules")).toEqual({ kind: "rules" });
  });
});

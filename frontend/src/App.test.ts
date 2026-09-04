import { describe, expect, it } from "vitest";

import { resolveRoute } from "./App";

describe("application routes", () => {
  it("resolves start, game, and rules routes without a routing dependency", () => {
    expect(resolveRoute("#/"))
      .toEqual({ kind: "start" });
    expect(resolveRoute("#/game")).toEqual({ kind: "game" });
    expect(resolveRoute("#/rules")).toEqual({ kind: "rules" });
  });
});

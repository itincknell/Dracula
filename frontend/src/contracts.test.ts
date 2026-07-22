import { describe, expect, it } from "vitest";

import apiErrorFixture from "../../contracts/v1/api-error.json";
import healthFixture from "../../contracts/v1/health.json";
import gameViewFixture from "../../contracts/v1/human-game-view.json";
import publicEventFixture from "../../contracts/v1/public-event.json";
import {
  assertApiErrorResponse,
  assertHealthResponse,
  assertHumanGameView,
  assertPublicEvent,
} from "./contracts";

describe("versioned API fixtures", () => {
  it("validates the health contract", () => {
    expect(() => assertHealthResponse(healthFixture)).not.toThrow();
  });

  it("validates the human game-view contract", () => {
    expect(() => assertHumanGameView(gameViewFixture)).not.toThrow();
  });

  it("validates the public-event contract", () => {
    expect(() => assertPublicEvent(publicEventFixture)).not.toThrow();
  });

  it("validates the API-error contract", () => {
    expect(() => assertApiErrorResponse(apiErrorFixture)).not.toThrow();
  });

  // Exact field checks prevent an accidental private engine field from becoming public.
  it("rejects private fields in a human game view", () => {
    expect(() =>
      assertHumanGameView({ ...gameViewFixture, stock: ["V1"], seed: "private" }),
    ).toThrow(/unexpected field set/);
  });
});

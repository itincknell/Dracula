/**
 * Verifies that every canonical card resolves to a real bundled image asset.
 * The suite protects manifest completeness, readable naming, and file presence
 * rather than testing browser image rendering.
 */
import { readFile, stat } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { describe, expect, it } from "vitest";

import sourceAssets from "../card-assets.json";
import { CANONICAL_CARD_IDS, cardAsset, cardName } from "./cardAssets";

const frontendRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const projectRoot = path.resolve(frontendRoot, "..");
const sourceRoot = path.join(projectRoot, "Cards (large)");
const customRoot = path.join(frontendRoot, "assets", "cards");
const copiedRoot = path.join(frontendRoot, "public", "cards");

const ranks = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"];
const expectedIds = [
  ...["C", "D", "H", "S"].flatMap((suit) => ranks.map((rank) => `${rank}${suit}`)),
  "V1",
  "V2",
];

describe("card assets", () => {
  it("keeps the public mapping in canonical engine order", () => {
    expect(CANONICAL_CARD_IDS).toEqual(expectedIds);
    expect(Object.keys(sourceAssets)).toEqual(expectedIds);
    expect(cardAsset("AC")).toBe("/cards/AC.png");
    expect(cardAsset("V2")).toBe("/cards/V2.jpg");
    expect(() => cardAsset("not-a-card")).toThrow("unknown canonical card ID");
  });

  it("provides readable labels for suited cards and Vampires", () => {
    expect(cardName("AC")).toBe("Ace of Clubs");
    expect(cardName("10H")).toBe("10 of Hearts");
    expect(cardName("V1")).toBe("Vampire 1");
  });

  it("copies every selected large-pack file byte-for-byte", async () => {
    await stat(sourceRoot);
    for (const [cardId, sourceName] of Object.entries(sourceAssets)) {
      const [source, copied] = await Promise.all([
        readFile(path.join(sourceRoot, sourceName)),
        readFile(path.join(copiedRoot, `${cardId}.png`)),
      ]);
      expect(copied.equals(source), `${cardId} must be copied without transformation`).toBe(true);
    }
  });

  it("copies the project-owned Vampire faces without transforming them", async () => {
    for (const cardId of ["V1", "V2"]) {
      const [source, copied] = await Promise.all([
        readFile(path.join(customRoot, `${cardId}.jpg`)),
        readFile(path.join(copiedRoot, `${cardId}.jpg`)),
      ]);
      expect(copied.equals(source), `${cardId} must preserve its custom raster source`).toBe(true);
    }
  });

  it("does not select the retained medium pack", async () => {
    const script = await readFile(path.join(frontendRoot, "scripts", "prepare-card-assets.mjs"), "utf8");
    expect(script).toContain('"Cards (large)"');
    expect(script).not.toContain("Cards (medium)");
  });
});

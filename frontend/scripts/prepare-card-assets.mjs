/**
 * Copies canonical playing-card assets into the generated public directory.
 * The script validates the manifest and custom Vampire images so development
 * and production builds expose one complete, deterministic card set.
 */
import { copyFile, mkdir, readFile, stat } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const frontendRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const sourceRoot = path.resolve(frontendRoot, "..", "Cards (large)");
const customRoot = path.join(frontendRoot, "assets", "cards");
const destinationRoot = path.join(frontendRoot, "public", "cards");
const manifestPath = path.join(frontendRoot, "card-assets.json");
const manifest = JSON.parse(await readFile(manifestPath, "utf8"));

// The engine's card IDs are suit-major. Building that order here lets the
// script reject missing, extra, reordered, or duplicate manifest entries
// before Vite copies any card images.
const canonicalRanks = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"];
const canonicalIds = ["C", "D", "H", "S"].flatMap((suit) =>
  canonicalRanks.map((rank) => `${rank}${suit}`),
);
const assets = Object.entries(manifest).map(([cardId, source]) => ({
  destination: `${cardId}.png`,
  source,
}));

if (
  Object.keys(manifest).some((cardId, index) => cardId !== canonicalIds[index]) ||
  assets.length !== canonicalIds.length ||
  new Set(assets.map(({ source }) => source)).size !== canonicalIds.length
) {
  throw new Error("card-assets.json must map the 52 suited card IDs in canonical order to unique source files");
}

async function exists(file) {
  try {
    await stat(file);
    return true;
  } catch {
    return false;
  }
}

await mkdir(destinationRoot, { recursive: true });
if (!(await exists(sourceRoot))) {
  // CI does not contain the licensed source pack. It may use the already
  // selected public copies, but only if the entire canonical set is present.
  const missing = [];
  for (const asset of assets) {
    if (!(await exists(path.join(destinationRoot, asset.destination)))) missing.push(asset.destination);
  }
  if (missing.length > 0) {
    throw new Error(`Kenney source pack is unavailable and copied assets are missing: ${missing.join(", ")}`);
  }
} else {
  // A development checkout with the source pack refreshes all public copies
  // from the manifest instead of trusting possibly stale generated assets.
  for (const asset of assets) {
    await copyFile(
      path.join(sourceRoot, asset.source),
      path.join(destinationRoot, asset.destination),
    );
  }
}

for (const vampireId of ["V1", "V2"]) {
  // Vampire cards are custom artwork and do not belong to the 52-card source
  // manifest, but the browser expects them in the same public directory.
  await copyFile(
    path.join(customRoot, `${vampireId}.jpg`),
    path.join(destinationRoot, `${vampireId}.jpg`),
  );
}

const portraits = [
  "dracula-angry-frown.jpg",
  "dracula-angrier-frown.jpg",
  "dracula-angriest-grimace.jpg",
  "dracula-winning-grin.jpg",
];
// Portraits are authored directly in public/ because their paths are stable UI
// assets rather than generated card-pack selections.
for (const portrait of portraits) {
  if (!(await exists(path.join(frontendRoot, "public", "portraits", portrait)))) {
    throw new Error(`required Dracula portrait is missing: ${portrait}`);
  }
}

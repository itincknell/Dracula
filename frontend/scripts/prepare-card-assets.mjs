import { copyFile, mkdir, readFile, stat } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const frontendRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const projectRoot = path.resolve(frontendRoot, "..");
const sourceRoot = path.resolve(frontendRoot, "..", "Cards (large)");
const customRoot = path.join(frontendRoot, "assets", "cards");
const destinationRoot = path.join(frontendRoot, "public", "cards");
const manifestPath = path.join(frontendRoot, "card-assets.json");
const manifest = JSON.parse(await readFile(manifestPath, "utf8"));
const canonicalRanks = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"];
const canonicalIds = [
  ...["C", "D", "H", "S"].flatMap((suit) => canonicalRanks.map((rank) => `${rank}${suit}`)),
  "V1",
  "V2",
];
const assets = Object.entries(manifest).map(([cardId, source]) => ({
  destination: `${cardId}.png`,
  source,
}));

if (
  Object.keys(manifest).some((cardId, index) => cardId !== canonicalIds[index]) ||
  assets.length !== canonicalIds.length ||
  new Set(assets.map(({ source }) => source)).size !== canonicalIds.length
) {
  throw new Error("card-assets.json must map the 54 canonical card IDs in canonical order to unique source files");
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
  const missing = [];
  for (const asset of assets) {
    if (!(await exists(path.join(destinationRoot, asset.destination)))) missing.push(asset.destination);
  }
  if (missing.length > 0) {
    throw new Error(`Kenney source pack is unavailable and copied assets are missing: ${missing.join(", ")}`);
  }
} else {
  for (const asset of assets) {
    await copyFile(
      path.join(sourceRoot, asset.source),
      path.join(destinationRoot, asset.destination),
    );
  }
}

for (const vampireId of ["V1", "V2"]) {
  await copyFile(
    path.join(customRoot, `${vampireId}.jpg`),
    path.join(destinationRoot, `${vampireId}.jpg`),
  );
}

await copyFile(
  path.join(projectRoot, "Dracula.png"),
  path.join(frontendRoot, "public", "dracula.png"),
);

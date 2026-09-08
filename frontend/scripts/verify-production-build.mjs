/**
 * Inspects the generated frontend distribution before publication.
 * It verifies required entry files, the application base and API prefix, and
 * the absence of localhost references from production JavaScript and HTML.
 */
import { access, readdir, readFile } from "node:fs/promises";
import { extname, resolve } from "node:path";

const dist = resolve("dist");
const forbidden = ["localhost", "127.0.0.1", "[::1]"];
const expectedBase = process.env.VITE_BASE_PATH || "/Dracula/";
const expectedApi = process.env.VITE_API_ORIGIN || "/Dracula/api";

async function files(directory) {
  // Vite may create nested asset directories, so production inspection walks
  // the complete distribution rather than assuming one bundle filename.
  const entries = await readdir(directory, { withFileTypes: true });
  const nested = await Promise.all(
    entries.map((entry) => {
      const path = resolve(directory, entry.name);
      return entry.isDirectory() ? files(path) : [path];
    }),
  );
  return nested.flat();
}

const paths = await files(dist);
const text = [];
for (const path of paths) {
  // Inspect source text, not binary images or fonts decoded as fake UTF-8.
  if (![".html", ".js", ".css", ".json", ".svg"].includes(extname(path))) continue;
  const contents = await readFile(path, "utf8");
  text.push(contents);
  const match = forbidden.find((value) => contents.includes(value));
  if (match !== undefined) {
    throw new Error(`Production output ${path} contains forbidden local URL text: ${match}`);
  }
}

const index = await readFile(resolve(dist, "index.html"), "utf8");
// Checking the emitted index catches an incorrectly configured Vite base even
// when the source configuration itself appears correct.
if (!index.includes(`${expectedBase}assets/`)) {
  throw new Error(`production index does not use the ${expectedBase} asset base`);
}
const bundle = text.join("\n");
// The API origin is compiled into the browser bundle. Its presence verifies
// that the release build did not silently inherit the local /api proxy.
if (!bundle.includes(expectedApi)) {
  throw new Error(`production output does not contain configured API origin ${expectedApi}`);
}
for (const asset of [
  "portraits/dracula-angry-frown.jpg",
  "portraits/dracula-angrier-frown.jpg",
  "portraits/dracula-angriest-grimace.jpg",
  "portraits/dracula-winning-grin.jpg",
]) {
  if (!bundle.includes(asset)) throw new Error(`production output does not reference asset ${asset}`);
  await access(resolve(dist, asset));
}
await access(resolve(dist, "cards"));

console.log(`production build verified base=${expectedBase} api=${expectedApi}`);

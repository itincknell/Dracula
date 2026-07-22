import { readdir, readFile } from "node:fs/promises";
import { resolve } from "node:path";

const dist = resolve("dist");
const forbidden = ["localhost", "127.0.0.1", "[::1]"];

async function files(directory) {
  const entries = await readdir(directory, { withFileTypes: true });
  const nested = await Promise.all(
    entries.map((entry) => {
      const path = resolve(directory, entry.name);
      return entry.isDirectory() ? files(path) : [path];
    }),
  );
  return nested.flat();
}

for (const path of await files(dist)) {
  const contents = await readFile(path, "utf8");
  const match = forbidden.find((value) => contents.includes(value));
  if (match !== undefined) {
    throw new Error(`Production output ${path} contains forbidden local URL text: ${match}`);
  }
}

console.log("production build contains no hard-coded local API URL");

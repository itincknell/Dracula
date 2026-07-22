import { defineConfig } from "@playwright/test";
import path from "node:path";
import { fileURLToPath } from "node:url";

const frontendRoot = path.dirname(fileURLToPath(import.meta.url));
const projectRoot = path.resolve(frontendRoot, "..");
const candidate = path.resolve(
  process.env.DRACULA_POLICY_ARCHIVE ??
    path.join(projectRoot, "runs/training-004/archives/policy-2-policy-2-v20.pt"),
);
const database = path.join(projectRoot, ".local", "dracula-e2e.sqlite3");

function shellValue(value: string): string {
  return `'${value.replaceAll("'", `'\\''`)}'`;
}

process.env.DRACULA_E2E_DATABASE = database;

export default defineConfig({
  testDir: "./e2e",
  outputDir: "./test-results",
  timeout: 180_000,
  expect: { timeout: 20_000 },
  fullyParallel: false,
  workers: 1,
  reporter: [["list"], ["html", { open: "never", outputFolder: "playwright-report" }]],
  use: {
    baseURL: "http://127.0.0.1:4174",
    channel: "chrome",
    headless: true,
    reducedMotion: "reduce",
    screenshot: "only-on-failure",
    trace: "retain-on-failure",
    viewport: { width: 1440, height: 900 },
  },
  webServer: [
    {
      command: [
        `rm -f ${shellValue(database)} ${shellValue(`${database}-shm`)} ${shellValue(`${database}-wal`)}`,
        "&&",
        `DRACULA_DATABASE_PATH=${shellValue(database)}`,
        `DRACULA_LOCAL_GAME_SEED=${shellValue("dracula-browser-e2e-v1")}`,
        "DRACULA_NARRATION_ENABLED=false",
        `DRACULA_POLICY_ARCHIVE=${shellValue(candidate)}`,
        "DRACULA_POLICY_INFERENCE_PROFILE=argmax-v1",
        ".venv/bin/uvicorn dracula.api.app:app --host 127.0.0.1 --port 8011 --no-access-log",
      ].join(" "),
      cwd: projectRoot,
      port: 8011,
      reuseExistingServer: false,
      timeout: 60_000,
    },
    {
      command: [
        `VITE_SCORING_REDUCED_STEP_MS=${shellValue("40")}`,
        "npm run build",
        "&&",
        `DRACULA_API_PROXY_TARGET=${shellValue("http://127.0.0.1:8011")}`,
        "npm run preview -- --host 127.0.0.1 --port 4174",
      ].join(" "),
      cwd: frontendRoot,
      port: 4174,
      reuseExistingServer: false,
      timeout: 60_000,
    },
  ],
});

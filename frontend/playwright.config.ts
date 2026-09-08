/**
 * Configures production-shaped browser tests for the built frontend.
 * Playwright builds the frontend and starts FastAPI serving it with
 * the selected local π1 artifact, then runs one serial desktop/mobile project.
 */
import { defineConfig } from "@playwright/test";
import path from "node:path";
import { fileURLToPath } from "node:url";

const frontendRoot = path.dirname(fileURLToPath(import.meta.url));
const projectRoot = path.resolve(frontendRoot, "..");
const policy = path.resolve(
  process.env.DRACULA_POLICY_ARTIFACT ??
    path.join(projectRoot, "runs/bgc-policy-pi1-001/artifacts/pi1-policy.pt"),
);

function shellValue(value: string): string {
  // The policy path becomes part of a shell command assembled for Playwright.
  // Quote embedded apostrophes so paths remain one environment value.
  return `'${value.replaceAll("'", `'\\''`)}'`;
}

export default defineConfig({
  testDir: "./e2e",
  testMatch: "production-stateless.spec.ts",
  outputDir: "./test-results/production",
  timeout: 180_000,
  expect: { timeout: 20_000 },
  fullyParallel: false,
  workers: 1,
  reporter: [["list"]],
  use: {
    baseURL: "http://127.0.0.1:8012/Dracula/",
    channel: "chrome",
    headless: true,
    reducedMotion: "reduce",
    screenshot: "only-on-failure",
    trace: "retain-on-failure",
    viewport: { width: 1440, height: 900 },
  },
  webServer: [
    {
      // The Python fixture serves the real stateless app and built frontend with a fixed game seed and
      // selected local policy. It replaces only Bedrock with deterministic text.
      command: [
        "VITE_API_ORIGIN=/Dracula/api VITE_BASE_PATH=/Dracula/ VITE_SCORING_REDUCED_STEP_MS=20 npm --prefix frontend run build &&",
        "DRACULA_LOCAL_GAME_SEED=dracula-stateless-frontend-v1",
        `DRACULA_POLICY_ARTIFACT=${shellValue(policy)}`,
        ".venv/bin/uvicorn frontend_production_app:app",
        "--app-dir tests --host 127.0.0.1 --port 8012 --no-access-log",
      ].join(" "),
      cwd: projectRoot,
      port: 8012,
      reuseExistingServer: false,
      timeout: 60_000,
    },
  ],
});

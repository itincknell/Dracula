/**
 * Exercises the built frontend against a production-shaped stateless server.
 * Browser tests cover complete play, reload recovery, responsive layout, and
 * public asset/API configuration across desktop and mobile viewports.
 */
import { expect, test, type Page } from "@playwright/test";

interface Envelope {
  seed: string;
  history: Array<Record<string, unknown>>;
}

interface GameResponse {
  envelope: Envelope;
  game: {
    status: "playing" | "round_complete" | "game_complete";
    round_number: number;
    phase: { kind: string };
    legal_moves: Array<{ hand_slot: number; position: number }>;
    completed_rounds: unknown[];
  };
}

const storageKey = "dracula.recovery-envelope";

test("starting overlay covers the role card without moving its buttons", async ({ page }) => {
  // Hold the initial response to inspect exactly what a cold start displays.
  let release!: () => void;
  const pending = new Promise<void>((resolve) => { release = resolve; });
  await page.route("**/api/games", async (route) => {
    await pending;
    await route.abort();
  });
  try {
    for (const viewport of [{ width: 390, height: 844 }, { width: 1440, height: 900 }]) {
      await page.setViewportSize(viewport);
      await page.goto("./");
      await page.evaluate(() => document.fonts.ready);
      const card = page.locator(".start-card");
      const queen = page.getByRole("button", { name: "Start as Queen" });
      const beforeCard = await card.boundingBox();
      const beforeButton = await queen.boundingBox();
      await queen.click();
      await expect(page.getByRole("status")).toHaveText("Starting game");
      await expect(queen).toBeDisabled();
      await expect(page.getByRole("button", { name: "Start as King" })).toBeDisabled();
      expect(await card.boundingBox()).toEqual(beforeCard);
      expect(await queen.boundingBox()).toEqual(beforeButton);
      const overlay = await page.locator(".start-overlay").boundingBox();
      expect(overlay!.width).toBeGreaterThan(beforeCard!.width - 10);
      expect(overlay!.height).toBeGreaterThan(beforeCard!.height - 10);
    }
  } finally {
    release();
    await page.unrouteAll({ behavior: "wait" });
  }
});

async function waitForCommittedResponse(page: Page, game: GameResponse): Promise<void> {
  await page.waitForFunction(
    ({ key, envelope }) => localStorage.getItem(key) === JSON.stringify(envelope),
    { key: storageKey, envelope: game.envelope },
  );
  const firstMove = game.game.legal_moves[0];
  if (firstMove !== undefined) {
    await expect(page.locator(`button.hand-card[data-hand-slot="${firstMove.hand_slot}"]`)).toBeEnabled();
  }
}

async function start(page: Page, role: "Queen" | "King" = "Queen"): Promise<GameResponse> {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.goto("./");
  expect(await page.evaluate(() => matchMedia("(prefers-reduced-motion: reduce)").matches)).toBe(true);
  const response = page.waitForResponse((candidate) =>
    candidate.request().method() === "POST" &&
    new URL(candidate.url()).pathname === "/Dracula/api/games",
  );
  await page.getByRole("button", { name: `Start as ${role}` }).click();
  const started = await response;
  expect(started.status()).toBe(201);
  const game = await started.json() as GameResponse;
  await expect(page).toHaveURL(/\/Dracula\/#\/game$/);
  await waitForCommittedResponse(page, game);
  return game;
}

async function expectDialogue(page: Page, text: string): Promise<void> {
  const dialogue = page.locator(".commentary-dialogue");
  await expect(dialogue).toBeVisible();
  await expect(dialogue).toContainText(text);
}

async function playFirstLegal(page: Page, game: GameResponse): Promise<GameResponse> {
  const move = game.game.legal_moves[0];
  if (move === undefined) throw new Error("human turn has no legal move");
  await page.locator(`[data-hand-slot="${move.hand_slot}"]`).click();
  const response = page.waitForResponse((candidate) =>
    candidate.request().method() === "POST" &&
    new URL(candidate.url()).pathname === "/Dracula/api/games/command",
  );
  await page.locator(`button.legal-target[data-position="${move.position}"]`).click();
  const accepted = await response;
  expect(accepted.status()).toBe(200);
  const acceptedGame = await accepted.json() as GameResponse;
  await waitForCommittedResponse(page, acceptedGame);
  return acceptedGame;
}

async function advance(page: Page): Promise<GameResponse> {
  const response = page.waitForResponse((candidate) =>
    candidate.request().method() === "POST" &&
    new URL(candidate.url()).pathname === "/Dracula/api/games/command",
  );
  await page.getByRole("button", { name: "Deal Next Round" }).click();
  const game = await (await response).json() as GameResponse;
  await waitForCommittedResponse(page, game);
  return game;
}

test("completes six stateless rounds, reloads from the envelope, and times narration", async ({ page, context }) => {
  // A full-length closing line must retain room above the score card when an
  // already-completed desktop presentation is resized to a narrow phone.
  const closingText = "Round transition cue. Your pitiful little triumph means nothing, human! Those three black cards have merely delayed the inevitable: I shall reclaim my magnificent lead, crush your miserable ambitions, and hang this fleeting victory in my castle as a monument to your foolishness!";
  await page.route("**/api/narration", async (route) => {
    const request = route.request().postDataJSON() as { cue_type: string };
    if (request.cue_type !== "round_transition") return route.continue();
    await route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({
      cue_type: request.cue_type, status: "ready", text: closingText,
    }) });
  });
  let current = await start(page);
  await expectDialogue(page, "Opening cue");
  const opening = current;
  const openingResume = page.waitForResponse((candidate) =>
    new URL(candidate.url()).pathname === "/Dracula/api/games/resume",
  );
  await page.reload();
  current = await (await openingResume).json() as GameResponse;
  expect(current).toEqual(opening);
  await expectDialogue(page, "Opening cue");

  const rules = context.waitForEvent("page");
  await page.getByRole("link", { name: "Rules" }).click();
  const rulesPage = await rules;
  await expect(rulesPage).toHaveURL(/\/Dracula\/#\/rules$/);
  await expect(rulesPage.getByRole("heading", { name: "How to play" })).toBeVisible();
  await rulesPage.close();

  const stored = await page.evaluate((key) => localStorage.getItem(key), storageKey);
  expect(JSON.parse(stored ?? "null")).toEqual(current.envelope);
  expect(Object.keys(JSON.parse(stored ?? "null"))).toEqual(["seed", "history"]);

  let reloadedHuman = false;
  let reloadedScoring = false;
  while (current.game.status !== "game_complete") {
    if (current.game.phase.kind === "human_turn") {
      if (!reloadedHuman) {
        const expected = current;
        const resume = page.waitForResponse((candidate) =>
          new URL(candidate.url()).pathname === "/Dracula/api/games/resume",
        );
        await page.reload();
        current = await (await resume).json() as GameResponse;
        expect(current).toEqual(expected);
        reloadedHuman = true;
      }
      current = await playFirstLegal(page, current);
      continue;
    }
    if (current.game.status === "round_complete") {
      if (!reloadedScoring) {
        const expected = current;
        const resume = page.waitForResponse((candidate) =>
          new URL(candidate.url()).pathname === "/Dracula/api/games/resume",
        );
        await page.reload();
        current = await (await resume).json() as GameResponse;
        expect(current).toEqual(expected);
        reloadedScoring = true;
      }
      if (current.game.round_number === 6) {
        const completed = page.waitForResponse((candidate) =>
          new URL(candidate.url()).pathname === "/Dracula/api/games/command",
        );
        current = await (await completed).json() as GameResponse;
        continue;
      }
      await expect(page.getByRole("button", { name: "Deal Next Round" })).toBeVisible();
      await expectDialogue(page, "Round transition cue");
      if (current.game.round_number === 2) {
        const typed = page.locator(".commentary-typed");
        for (const width of [390, 320, 560, 390]) {
          await page.setViewportSize({ width, height: 844 });
          await expect(typed).toContainText(closingText);
          await expect.poll(() => page.evaluate(() => {
            const text = document.querySelector(".commentary-typed")!.getBoundingClientRect();
            const score = document.querySelector(".round-totals")!.getBoundingClientRect();
            return score.top - text.bottom;
          })).toBeGreaterThanOrEqual(8);
          await expect(page.locator(".portrait-placeholder")).toHaveCSS("height", "90px");
        }
        await page.setViewportSize({ width: 1440, height: 900 });
        await expect(typed).toContainText(closingText);
        await expect(page.locator(".commentary-measure")).toBeHidden();
      }
      current = await advance(page);
      await expect(page.locator(".commentary-stream")).toBeEmpty();
      continue;
    }
    throw new Error(`unexpected game phase ${current.game.phase.kind}`);
  }

  expect(current.game.completed_rounds).toHaveLength(6);
  await expectDialogue(page, "Final result cue");
  await expect(page.getByRole("button", { name: "Play Again" })).toBeVisible();
  const before = current;
  const resume = page.waitForResponse((candidate) =>
    new URL(candidate.url()).pathname === "/Dracula/api/games/resume",
  );
  await page.reload();
  current = await (await resume).json() as GameResponse;
  expect(current).toEqual(before);
  await expect(page.getByRole("button", { name: "Play Again" })).toBeVisible();
});

test("narration failure cannot block gameplay or change the mobile and desktop layouts", async ({ page }) => {
  await page.route("**/api/narration", async (route) => {
    const request = route.request().postDataJSON() as { cue_type: string };
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ cue_type: request.cue_type, status: "unavailable", text: null }),
    });
  });
  await page.setViewportSize({ width: 390, height: 640 });
  let current = await start(page, "King");
  await expect(page.locator(".commentary-stream")).toBeEmpty();
  const move = current.game.legal_moves[0];
  if (move === undefined) throw new Error("human turn has no legal move");
  await page.evaluate(() => window.scrollTo(0, Math.min(120, document.documentElement.scrollHeight)));
  const scrollBefore = await page.evaluate(() => window.scrollY);
  const response = page.waitForResponse((candidate) =>
    new URL(candidate.url()).pathname === "/Dracula/api/games/command",
  );
  await page.locator(`[data-hand-slot="${move.hand_slot}"]`).evaluate((element: HTMLElement) => element.click());
  await page.locator(`button.legal-target[data-position="${move.position}"]`)
    .evaluate((element: HTMLElement) => element.click());
  current = await (await response).json() as GameResponse;
  expect(await page.evaluate(() => window.scrollY)).toBe(scrollBefore);
  expect(current.game.status).toBe("playing");

  for (const viewport of [{ width: 390, height: 844 }, { width: 1440, height: 900 }]) {
    await page.setViewportSize(viewport);
    const geometry = await page.evaluate(() => ({
      width: document.documentElement.scrollWidth,
      viewport: window.innerWidth,
      coffin: document.querySelector(".coffin")?.getBoundingClientRect().width ?? 0,
      hand: document.querySelector(".hand")?.getBoundingClientRect().width ?? 0,
    }));
    expect(geometry.width).toBeLessThanOrEqual(geometry.viewport + 1);
    expect(geometry.coffin).toBeGreaterThan(0);
    expect(geometry.hand).toBeGreaterThan(0);
  }
});

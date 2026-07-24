import { expect, test, type APIRequestContext, type Page, type Response } from "@playwright/test";
import { execFileSync } from "node:child_process";
import path from "node:path";

import type { EventsResponse, HumanGameView, Player } from "../src/contracts";

const API = "http://127.0.0.1:8011";
const opponentMode = process.env.DRACULA_E2E_OPPONENT ?? "archive";
const candidatePath = opponentMode === "guided"
  ? (process.env.DRACULA_POLICY_VALUE_ARTIFACT ?? "runs/search-warmstart-smoke-001/warm-start-smoke.pt")
  : "runs/training-004/archives/policy-2-policy-2-v20.pt";
const privateKeys = new Set([
  "seed",
  "stock",
  "hands",
  "opponent_hand",
  "hidden_state",
  "policy_hidden_state",
  "raw_logits",
  "engine_state",
  "move_secret",
  "state_dict",
  "archive_path",
  "model_path",
]);

interface NetworkAudit {
  createBodies: unknown[];
  payloads: unknown[];
  pending: Promise<void>[];
  opponentAccepted: number;
  opponentClaimed: number;
}

function uuid(): string {
  return crypto.randomUUID();
}

function collectKeys(value: unknown, keys = new Set<string>()): Set<string> {
  if (Array.isArray(value)) {
    for (const item of value) collectKeys(item, keys);
  } else if (value !== null && typeof value === "object") {
    for (const [key, item] of Object.entries(value)) {
      keys.add(key);
      collectKeys(item, keys);
    }
  }
  return keys;
}

function auditNetwork(page: Page): NetworkAudit {
  const audit: NetworkAudit = {
    createBodies: [],
    payloads: [],
    pending: [],
    opponentAccepted: 0,
    opponentClaimed: 0,
  };
  page.on("request", (request) => {
    if (request.method() === "POST" && new URL(request.url()).pathname === "/api/games") {
      audit.createBodies.push(request.postDataJSON());
    }
  });
  page.on("response", (response) => {
    const url = new URL(response.url());
    if (!url.pathname.startsWith("/api/")) return;
    if (url.pathname.endsWith("/opponent-turn")) {
      if (response.status() === 202) audit.opponentClaimed += 1;
      if (response.status() === 200) audit.opponentAccepted += 1;
    }
    const task = response
      .json()
      .then((payload: unknown) => { audit.payloads.push(payload); })
      .catch(() => undefined);
    audit.pending.push(task);
  });
  return audit;
}

async function assertPublicAudit(page: Page, audit: NetworkAudit): Promise<void> {
  await Promise.all(audit.pending);
  expect(audit.createBodies.length).toBeGreaterThan(0);
  for (const body of audit.createBodies) {
    expect(body).not.toHaveProperty("seed");
  }
  for (const payload of audit.payloads) {
    const observedPrivate = [...collectKeys(payload)].filter((key) => privateKeys.has(key));
    expect(observedPrivate).toEqual([]);
    const serialized = JSON.stringify(payload);
    expect(serialized).not.toContain("dracula-browser-e2e-v1");
    expect(serialized).not.toContain(candidatePath);
  }
  const browserState = await page.evaluate(() => ({
    local: Object.keys(localStorage),
    session: Object.keys(sessionStorage),
    body: document.body.textContent ?? "",
    resources: performance.getEntriesByType("resource").map((entry) => entry.name),
  }));
  expect(browserState.local).toEqual([]);
  expect(browserState.session).toEqual([]);
  expect(browserState.body).not.toContain("dracula-browser-e2e-v1");
  expect(browserState.body).not.toContain(candidatePath);
  expect(browserState.resources.join("\n")).not.toContain(candidatePath);
}

async function apiView(request: APIRequestContext, gameId: string): Promise<HumanGameView> {
  const response = await request.get(`${API}/games/${gameId}`);
  expect(response.status()).toBe(200);
  return await response.json() as HumanGameView;
}

async function apiEvents(request: APIRequestContext, gameId: string): Promise<EventsResponse> {
  const response = await request.get(`${API}/games/${gameId}/events`);
  expect(response.status()).toBe(200);
  return await response.json() as EventsResponse;
}

async function startFromUi(page: Page, role: Player): Promise<HumanGameView> {
  await page.goto("/");
  const responsePromise = page.waitForResponse((response) =>
    response.request().method() === "POST" &&
    new URL(response.url()).pathname === "/api/games" &&
    response.status() === 201,
  );
  await page.getByRole("button", { name: `Start as ${role === "queen" ? "Queen" : "King"}` }).click();
  const response = await responsePromise;
  const view = await response.json() as HumanGameView;
  await expect(page).toHaveURL(new RegExp(`/games/${view.game_id}$`));
  return view;
}

async function authoritativeView(request: APIRequestContext, gameId: string): Promise<HumanGameView> {
  await expect.poll(async () => (await apiView(request, gameId)).phase.kind, { timeout: 20_000 }).not.toBe(
    "opponent_turn",
  );
  return await apiView(request, gameId);
}

async function assertPageMatchesApi(
  page: Page,
  request: APIRequestContext,
  gameId: string,
): Promise<HumanGameView> {
  const view = await apiView(request, gameId);
  await expect(page.getByText(`Round ${view.round_number} of 6`)).toBeVisible();
  await expect(page.getByLabel(`Your score: ${view.total_scores.human}`)).toBeVisible();
  await expect(page.getByLabel(`Dracula's score: ${view.total_scores.opponent}`)).toBeVisible();

  if (view.phase.kind === "human_turn") {
    const coffin = await page.locator(".coffin > [data-position]").evaluateAll((items) =>
      items.map((item) => ({
        position: Number((item as HTMLElement).dataset.position),
        card: (item as HTMLElement).dataset.cardId ?? null,
      })).sort((left, right) => left.position - right.position),
    );
    expect(coffin.map((item) => item.card)).toEqual(view.coffin);
    const hand = await page.locator(".hand > [data-hand-slot]").evaluateAll((items) =>
      items.map((item) => ({
        slot: Number((item as HTMLElement).dataset.handSlot),
        card: (item as HTMLElement).dataset.cardId ?? null,
      })).sort((left, right) => left.slot - right.slot),
    );
    expect(hand.map((item) => item.card)).toEqual(view.human_hand);

    const selectedSlot = view.legal_moves[0]?.hand_slot;
    if (selectedSlot !== undefined) {
      await page.locator(`[data-hand-slot="${selectedSlot}"]`).click();
      const legalPositions = await page.locator("button.legal-target").evaluateAll((items) =>
        items.map((item) => Number((item as HTMLElement).dataset.position)).sort((a, b) => a - b),
      );
      expect(legalPositions).toEqual(
        view.legal_moves
          .filter((move) => move.hand_slot === selectedSlot)
          .map((move) => move.position)
          .sort((a, b) => a - b),
      );
    }
  }

  const events = await apiEvents(request, gameId);
  expect(events.events).toEqual(view.events);
  expect(events.latest_sequence).toBe(view.latest_event_sequence);
  return view;
}

async function playHumanMove(page: Page, view: HumanGameView): Promise<Response> {
  const move = view.legal_moves[0];
  if (move === undefined) throw new Error("human turn has no legal move");
  const card = page.locator(`[data-hand-slot="${move.hand_slot}"]`);
  if (await card.getAttribute("aria-pressed") !== "true") await card.click();
  const responsePromise = page.waitForResponse((response) =>
    response.request().method() === "POST" &&
    new URL(response.url()).pathname.endsWith("/moves"),
  );
  await page.locator(`button.legal-target[data-position="${move.position}"]`).click();
  const response = await responsePromise;
  expect(response.status()).toBe(200);
  const accepted = await response.json() as HumanGameView;
  expect(accepted.version).toBe(view.version + 1);
  expect(accepted.current_round_moves.at(-1)).toMatchObject({
    card_id: move.card_id,
    hand_slot: move.hand_slot,
    position: move.position,
    player: view.human_role,
  });
  return response;
}

async function waitThroughScoring(
  page: Page,
  view: HumanGameView,
  reload: boolean,
): Promise<void> {
  const record = view.pending_round_result;
  if (record === null) throw new Error("scoring view has no round result");
  const firstPlayer = record.scoring_sequence.find((step) => step.kind === "score_line")?.player;
  const firstHeading = firstPlayer === view.human_role ? "Your Score" : "My Score";
  const secondHeading = firstHeading === "Your Score" ? "My Score" : "Your Score";
  const firstDirections = record.scoring_sequence
    .filter((step) => step.kind === "score_line")
    .slice(0, 3)
    .map((step) => step.line?.direction);
  expect(firstDirections).toEqual(Array(3).fill(record.dealer === "queen" ? "row" : "column"));

  await expect(page.locator(".score-heading")).toHaveText(firstHeading);
  if (reload) {
    await page.reload();
    await expect(page.getByLabel(`Round ${view.round_number} scoring presentation`)).toBeVisible();
    await expect(page.locator(".score-heading")).toHaveText(firstHeading);
  }
  await expect(page.locator(".score-heading")).toHaveText(secondHeading);
  if (view.round_number < 6) {
    await page.getByRole("button", { name: "Deal Next Round" }).click();
  } else {
    await expect(page.getByRole("button", { name: "Play Again" })).toBeVisible({ timeout: 30_000 });
  }
}

async function completeGame(
  page: Page,
  request: APIRequestContext,
  role: Player,
  viewport: { width: number; height: number },
): Promise<{ audit: NetworkAudit; gameId: string }> {
  await page.setViewportSize(viewport);
  const audit = auditNetwork(page);
  const created = await startFromUi(page, role);
  const gameId = created.game_id;
  let reloadedHuman = false;
  let reloadedScoring = false;
  const dealerOrders = new Set<string>();

  while (true) {
    const current = await apiView(request, gameId);
    if (current.phase.kind === "opponent_turn") {
      await expect.poll(async () => (await apiView(request, gameId)).phase.kind).not.toBe("opponent_turn");
      continue;
    }
    if (current.phase.kind === "human_turn") {
      await expect(page.getByLabel("Current turn: Your turn")).toBeVisible();
      const matched = await assertPageMatchesApi(page, request, gameId);
      if (!reloadedHuman) {
        await page.reload();
        const restored = await assertPageMatchesApi(page, request, gameId);
        expect(restored.version).toBe(matched.version);
        reloadedHuman = true;
      }
      await playHumanMove(page, matched);
      continue;
    }
    if (current.phase.kind === "scoring") {
      const record = current.pending_round_result;
      if (record === null) throw new Error("missing round result");
      dealerOrders.add(record.dealer === "queen" ? "row-column" : "column-row");
      await waitThroughScoring(page, current, !reloadedScoring);
      reloadedScoring = true;
      if (current.round_number < 6) {
        await expect.poll(async () => (await apiView(request, gameId)).round_number).toBe(
          current.round_number + 1,
        );
      }
      continue;
    }
    if (current.phase.kind === "game_complete") {
      await expect(page.getByRole("button", { name: "Play Again" })).toBeVisible();
      expect(current.completed_rounds).toHaveLength(6);
      expect(current.events.filter((event) => event.event_type === "move_accepted")).toHaveLength(48);
      expect(current.events.at(-1)?.event_type).toBe("game_completed");
      expect(dealerOrders).toEqual(new Set(["row-column", "column-row"]));
      break;
    }
    throw new Error(`unexpected phase ${current.phase.kind}`);
  }

  expect(audit.opponentClaimed).toBe(24);
  expect(audit.opponentAccepted).toBe(24);
  await assertPublicAudit(page, audit);
  return { audit, gameId };
}

test("health, startup, rules tab, and configured candidate identity", async ({ page, request, context }) => {
  const health = await request.get(`${API}/health`);
  expect(health.status()).toBe(200);
  expect(await health.json()).toEqual({
    schema_version: "dracula-health-v1",
    api_version: "dracula-api-v1",
    status: "ok",
    narration_enabled: false,
  });

  await page.goto("/");
  const rulesPagePromise = context.waitForEvent("page");
  await page.getByRole("link", { name: "Read the rules" }).click();
  const rulesPage = await rulesPagePromise;
  await expect(rulesPage).toHaveURL(/\/rules$/);
  await expect(rulesPage.getByRole("heading", { name: "How to play" })).toBeVisible();
  await rulesPage.close();

  const view = await startFromUi(page, "queen");
  const created = view.events.find((event) => event.event_type === "game_created");
  expect(created?.payload).toMatchObject(
    opponentMode === "guided"
      ? {
          policy_id: "guided-information-set-search",
          policy_version: "dracula-guided-information-search-v1",
          inference_profile: "max-visits-v1",
          narration_enabled: false,
        }
      : {
          policy_id: "policy-2",
          policy_version: "policy-2-v20",
          inference_profile: "argmax-v1",
          narration_enabled: false,
        },
  );
  await expect(page.getByText("Narration disabled")).toBeVisible();
});

test("completes all six rounds as Queen on desktop", async ({ page, request }) => {
  const { gameId } = await completeGame(page, request, "queen", { width: 1440, height: 900 });
  await page.screenshot({ path: "test-results/queen-six-round-desktop.png", fullPage: true });
  await page.getByRole("button", { name: "Play Again" }).click();
  await expect(page).toHaveURL(/\/$/);
  await expect(page.getByRole("button", { name: "Start as Queen" })).toBeVisible();

  const database = path.resolve(
    process.env.DRACULA_E2E_DATABASE ?? "../.local/dracula-e2e.sqlite3",
  );
  const evidence = JSON.parse(execFileSync(
    "../.venv/bin/python",
    ["-c", [
      "import json, sqlite3, sys, base64",
      "row=sqlite3.connect(sys.argv[1]).execute('select session_json from games where game_id=?',(sys.argv[2],)).fetchone()",
      "data=json.loads(row[0])",
      "hidden=base64.b64decode(data['policy_session']['hidden_state'])",
      "rounds=data['engine_state']['completed_rounds']",
      "opponent='king' if data['human_role']=='queen' else 'queen'",
      "forced=sum(1 for r in rounds if r['dealer']==opponent and r['moves'][-1]['player']==opponent and r['moves'][-1]['turn_number']==8)",
      "print(json.dumps({'bytes':len(hidden),'nonzero':any(hidden),'rounds':len(rounds),'forced':forced,'version':data['policy_session']['policy_version']}))",
    ].join(";"), database, gameId],
    { cwd: process.cwd(), encoding: "utf8" },
  ));
  expect(evidence).toEqual(
    opponentMode === "guided"
      ? {
          bytes: 512,
          nonzero: false,
          rounds: 6,
          forced: 3,
          version: "dracula-guided-information-search-v1",
        }
      : { bytes: 512, nonzero: true, rounds: 6, forced: 3, version: "policy-2-v20" },
  );
});

test("completes all six rounds as King in the narrow layout", async ({ page, request }) => {
  await completeGame(page, request, "king", { width: 390, height: 844 });
  await expect(page.locator(".commentary-panel")).toHaveCSS("height", "160px");
  await page.screenshot({ path: "test-results/king-six-round-mobile.png", fullPage: true });
});

test("reloads a claimed opponent turn and resumes the same persisted job", async ({ page, request }) => {
  const created = await startFromUi(page, "queen");
  const ready = await authoritativeView(request, created.game_id);
  await assertPageMatchesApi(page, request, ready.game_id);

  let release!: () => void;
  const gate = new Promise<void>((resolve) => { release = resolve; });
  let claimed!: () => void;
  const claimObserved = new Promise<void>((resolve) => { claimed = resolve; });
  let held = false;
  await page.route("**/api/games/*/opponent-turn", async (route) => {
    if (held) {
      await route.continue();
      return;
    }
    held = true;
    const response = await route.fetch();
    expect(response.status()).toBe(202);
    claimed();
    await gate;
    await route.fulfill({ response });
  });

  await playHumanMove(page, ready);
  await claimObserved;
  await expect(page.getByLabel("Current turn: Dracula is deciding…")).toBeVisible();
  const reload = page.reload();
  release();
  await reload;
  await page.unroute("**/api/games/*/opponent-turn");
  await expect.poll(async () => (await apiView(request, ready.game_id)).phase.kind).not.toBe("opponent_turn");
  await assertPageMatchesApi(page, request, ready.game_id);
});

test("reconciles a stale browser move from the authoritative response", async ({ page, request }) => {
  const created = await startFromUi(page, "king");
  const view = await authoritativeView(request, created.game_id);
  await assertPageMatchesApi(page, request, view.game_id);
  const move = view.legal_moves[0];
  if (move === undefined) throw new Error("missing stale fixture move");

  const outside = await request.post(`${API}/games/${view.game_id}/moves`, {
    data: { move_id: move.move_id, expected_version: view.version, request_id: uuid() },
  });
  expect(outside.status()).toBe(200);
  const staleResponse = page.waitForResponse((response) =>
    new URL(response.url()).pathname.endsWith("/moves") && response.status() === 409,
  );
  const card = page.locator(`[data-hand-slot="${move.hand_slot}"]`);
  if (await card.getAttribute("aria-pressed") !== "true") await card.click();
  await page.locator(`button.legal-target[data-position="${move.position}"]`).click();
  const stale = await staleResponse;
  const body = await stale.json();
  expect(body.code).toBe("stale_version");
  expect(body.current_game.version).toBe(view.version + 1);
  expect([...collectKeys(body)].filter((key) => privateKeys.has(key))).toEqual([]);
  await expect.poll(async () => (await apiView(request, view.game_id)).phase.kind).not.toBe("opponent_turn");
  await assertPageMatchesApi(page, request, view.game_id);
});

test("keeps cards, controls, and layout reachable across acceptance viewports", async ({ page, request }) => {
  const created = await startFromUi(page, "queen");
  const view = await authoritativeView(request, created.game_id);
  await assertPageMatchesApi(page, request, view.game_id);
  const viewports = [
    { width: 360, height: 640 },
    { width: 390, height: 844 },
    { width: 768, height: 1024 },
    { width: 899, height: 800 },
    { width: 900, height: 800 },
    { width: 1024, height: 768 },
    { width: 1440, height: 900 },
  ];
  for (const viewport of viewports) {
    await page.setViewportSize(viewport);
    const geometry = await page.evaluate(() => {
      const rect = (selector: string) => {
        const value = document.querySelector(selector)?.getBoundingClientRect();
        if (value === undefined) throw new Error(`missing ${selector}`);
        return { x: value.x, y: value.y, right: value.right, bottom: value.bottom, width: value.width, height: value.height };
      };
      const cards = [...document.querySelectorAll(".coffin > *, .hand > *")].map((item) => {
        const value = item.getBoundingClientRect();
        return { width: value.width, height: value.height, right: value.right };
      });
      return {
        main: rect(".main-display"),
        commentary: rect(".commentary-panel"),
        coffin: rect(".coffin"),
        hand: rect(".hand"),
        cards,
        pageWidth: document.documentElement.scrollWidth,
        statusFont: Number.parseFloat(getComputedStyle(document.querySelector(".turn-status")!).fontSize),
        secondaryFont: Number.parseFloat(getComputedStyle(document.querySelector(".commentary-status")!).fontSize),
      };
    });
    expect(geometry.cards).toHaveLength(13);
    expect(geometry.cards.every((card) => card.width >= 64 && card.height >= 64)).toBe(true);
    expect(geometry.cards.every((card) => card.right <= geometry.pageWidth + 1)).toBe(true);
    expect(geometry.statusFont).toBeGreaterThanOrEqual(16);
    expect(geometry.secondaryFont).toBeGreaterThanOrEqual(14);
    expect(geometry.coffin.bottom).toBeLessThanOrEqual(geometry.hand.y + 1);
    if (viewport.width < 900) {
      expect(geometry.commentary.bottom).toBeLessThanOrEqual(geometry.main.y + 1);
    } else {
      expect(geometry.main.right).toBeLessThanOrEqual(geometry.commentary.x + 1);
    }
  }
});

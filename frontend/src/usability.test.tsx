// @vitest-environment jsdom

import "@testing-library/jest-dom/vitest";
import { readFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

import axe from "axe-core";
import { cleanup, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";

import gameViewFixture from "../../contracts/v1/human-game-view.json";
import { GameStart } from "./App";
import type { ApiClient } from "./api";
import type { HumanGameView } from "./contracts";
import { GameController } from "./gameStore";
import { GameWindow } from "./gameplay";
import { RulesPage } from "./RulesPage";

afterEach(cleanup);

const requestId = "00000000-0000-4000-8000-000000000099";

function client(overrides: Partial<ApiClient> = {}): ApiClient {
  return {
    health: vi.fn(),
    createGame: vi.fn(),
    getGame: vi.fn(),
    submitMove: vi.fn(),
    opponentTurn: vi.fn(),
    advanceRound: vi.fn(),
    getEvents: vi.fn(),
    ...overrides,
  } as ApiClient;
}

async function humanTurnController(): Promise<GameController> {
  const view = structuredClone(gameViewFixture) as unknown as HumanGameView;
  view.phase = { kind: "human_turn" };
  view.active_player = view.human_role;
  view.legal_moves = [{ move_id: "m", card_id: "2C", hand_slot: 0, position: 8 }];
  const controller = new GameController(client({
    getGame: vi.fn(async () => ({ status: 200 as const, data: view })),
  }), { requestId: () => requestId });
  await controller.loadGame(view.game_id);
  return controller;
}

async function expectNoAxeViolations(container: HTMLElement): Promise<void> {
  // jsdom cannot evaluate painted colors; contrast is verified numerically below.
  const results = await axe.run(container, { rules: { "color-contrast": { enabled: false } } });
  expect(results.violations, results.violations.map((item) => `${item.id}: ${item.help}`).join("\n")).toEqual([]);
}

describe("accessible application surfaces", () => {
  it("passes automated checks on start, game, and rules views", async () => {
    let rendered = render(<GameStart controller={new GameController(client())} onGameCreated={() => undefined} />);
    await expectNoAxeViolations(rendered.container);
    rendered.unmount();

    const controller = await humanTurnController();
    rendered = render(<GameWindow controller={controller} onNewGame={() => undefined} />);
    await expectNoAxeViolations(rendered.container);
    rendered.unmount();

    rendered = render(<RulesPage />);
    await expectNoAxeViolations(rendered.container);
  });

  it("supports the complete move path with keyboard controls", async () => {
    const controller = await humanTurnController();
    render(<GameWindow controller={controller} onNewGame={() => undefined} />);
    const card = screen.getByRole("button", { name: "2 of Clubs (2C), hand slot 1" });
    card.focus();
    await userEvent.keyboard("{Enter}");
    const target = screen.getByRole("button", { name: "Play 2 of Clubs at coffin position 9" });
    await userEvent.tab({ shift: true });
    expect(target).toHaveFocus();
  });

  it("renders the authoritative setup, play, scoring, and winning rules", () => {
    render(<RulesPage />);
    expect(screen.getByRole("heading", { name: "Setup and dealing" })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Playing a round" })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Scoring" })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Ending the game" })).toBeInTheDocument();
    expect(screen.getByRole("table")).toHaveTextContent("Queen");
    expect(screen.getByText(/line containing a Vampire scores zero/i)).toBeInTheDocument();
  });
});

describe("responsive contract", () => {
  it("encodes the documented floors, aspect ratio, breakpoint, container query, and scrolling", async () => {
    const frontendRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
    const css = await readFile(path.join(frontendRoot, "src", "styles.css"), "utf8");
    expect(css).toContain("--minimum-card-size: 64px");
    expect(css).toContain("aspect-ratio: 4 / 3");
    expect(css).toContain("@container game-layout (max-width: 899px)");
    expect(css).toContain("container-type: inline-size");
    expect(css).toContain("min-width: 360px");
    expect(css).toContain("font-size: 14px");
    expect(css).toContain("overflow-y: auto");
    expect(css).toContain("@media (prefers-reduced-motion: reduce)");
  });

  it.each([
    ["body text", "#241f23", "#f0eae0", 4.5],
    ["headings", "#421b24", "#fffdf8", 4.5],
    ["errors", "#7d1525", "#fffdf8", 4.5],
    ["commentary", "#c7b7b7", "#241622", 4.5],
    ["legal targets", "#5d5046", "#e6f2ef", 4.5],
    ["focus", "#14728b", "#f3eadc", 3],
  ] as const)("meets the contrast floor for %s", (_label, foreground, background, minimum) => {
    const luminance = (hex: string): number => {
      const channels = [1, 3, 5].map((offset) => Number.parseInt(hex.slice(offset, offset + 2), 16) / 255);
      const [red, green, blue] = channels.map((value) =>
        value <= 0.04045 ? value / 12.92 : ((value + 0.055) / 1.055) ** 2.4,
      );
      return 0.2126 * red + 0.7152 * green + 0.0722 * blue;
    };
    const first = luminance(foreground);
    const second = luminance(background);
    const contrast = (Math.max(first, second) + 0.05) / (Math.min(first, second) + 0.05);
    expect(contrast).toBeGreaterThanOrEqual(minimum);
  });
});

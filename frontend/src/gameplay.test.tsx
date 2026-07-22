// @vitest-environment jsdom

import "@testing-library/jest-dom/vitest";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";

import gameViewFixture from "../../contracts/v1/human-game-view.json";
import { GameStart } from "./App";
import type { ApiClient } from "./api";
import type { HumanGameView, MoveRequest, ResumablePhase } from "./contracts";
import { GameController } from "./gameStore";
import { GameWindow, Hand } from "./gameplay";
import { scoringRecord } from "./scoringTestFixtures";

afterEach(cleanup);

const requestId = "00000000-0000-4000-8000-000000000077";

function view(
  phase: ResumablePhase,
  changes: Partial<HumanGameView> = {},
): HumanGameView {
  return {
    ...(structuredClone(gameViewFixture) as unknown as HumanGameView),
    phase,
    ...changes,
  };
}

function api(overrides: Partial<ApiClient> = {}): ApiClient {
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

async function loadedController(
  current: HumanGameView,
  overrides: Partial<ApiClient> = {},
): Promise<{ controller: GameController; client: ApiClient }> {
  const client = api({
    getGame: vi.fn(async () => ({ status: 200 as const, data: current })),
    ...overrides,
  });
  const controller = new GameController(client, {
    requestId: () => requestId,
    opponentPollDelay: async () => undefined,
  });
  await controller.loadGame(current.game_id);
  return { controller, client };
}

describe("game start", () => {
  it.each(["queen", "king"] as const)("creates a %s game from the role choice", async (role) => {
    const created = view({ kind: "human_turn" }, { human_role: role, game_id: `game-${role}` });
    const createGame = vi.fn(async () => ({ status: 201 as const, data: created }));
    const controller = new GameController(api({ createGame }), { requestId: () => requestId });
    const onGameCreated = vi.fn();
    render(<GameStart controller={controller} onGameCreated={onGameCreated} />);

    await userEvent.click(screen.getByRole("button", { name: `Start as ${role === "queen" ? "Queen" : "King"}` }));
    expect(createGame).toHaveBeenCalledWith({ human_role: role, request_id: requestId });
    expect(onGameCreated).toHaveBeenCalledWith(`game-${role}`);
    expect(screen.getByRole("link", { name: "Read the rules" })).toHaveAttribute("target", "_blank");
  });
});

describe("card interaction", () => {
  const humanView = () => view(
    { kind: "human_turn" },
    {
      active_player: "queen",
      legal_moves: [{ move_id: "opaque-move", card_id: "2C", hand_slot: 0, position: 8 }],
    },
  );

  async function submitThrough(method: "tap" | "keyboard" | "drag"): Promise<string> {
    const current = humanView();
    const accepted = { ...current, version: current.version + 1 };
    const submitMove = vi.fn(async (gameId: string, request: MoveRequest) => {
      void gameId;
      void request;
      return { status: 200 as const, data: accepted };
    });
    const { controller } = await loadedController(current, { submitMove });
    render(<GameWindow controller={controller} onNewGame={() => undefined} />);
    const card = screen.getByRole("button", { name: "2 of Clubs (2C), hand slot 1" });

    if (method === "drag") {
      const transfer = { effectAllowed: "none", dropEffect: "none", setData: vi.fn() };
      fireEvent.dragStart(card, { dataTransfer: transfer });
      const target = await screen.findByRole("button", { name: "Play 2 of Clubs at coffin position 9" });
      fireEvent.drop(target, { dataTransfer: transfer });
    } else if (method === "keyboard") {
      card.focus();
      await userEvent.keyboard("{Enter}");
      const target = screen.getByRole("button", { name: "Play 2 of Clubs at coffin position 9" });
      target.focus();
      await userEvent.keyboard("{Enter}");
    } else {
      await userEvent.click(card);
      await userEvent.click(screen.getByRole("button", { name: "Play 2 of Clubs at coffin position 9" }));
    }

    await waitFor(() => expect(submitMove).toHaveBeenCalledTimes(1));
    const call = submitMove.mock.calls[0];
    if (call === undefined) throw new Error("move was not submitted");
    return call[1].move_id;
  }

  it("submits the same server move ID through drag, tap, and keyboard paths", async () => {
    expect(await submitThrough("drag")).toBe("opaque-move");
    cleanup();
    expect(await submitThrough("tap")).toBe("opaque-move");
    cleanup();
    expect(await submitThrough("keyboard")).toBe("opaque-move");
  });

  it("does not accept an unlisted drop or grid activation", async () => {
    const current = humanView();
    const submitMove = vi.fn();
    const { controller } = await loadedController(current, { submitMove });
    render(<GameWindow controller={controller} onNewGame={() => undefined} />);
    await userEvent.click(screen.getByRole("button", { name: "2 of Clubs (2C), hand slot 1" }));
    const illegal = screen.getByLabelText("Empty coffin position 1");
    fireEvent.drop(illegal, { dataTransfer: { dropEffect: "none" } });
    fireEvent.click(illegal);

    expect(submitMove).not.toHaveBeenCalled();
    expect(screen.getAllByText("Legal")).toHaveLength(1);
  });

  it("disables every hand input while a mutation is pending or Dracula is active", async () => {
    const current = humanView();
    let resolveMove: ((value: { status: 200; data: HumanGameView }) => void) | undefined;
    const submitMove = vi.fn(() => new Promise<{ status: 200; data: HumanGameView }>((resolve) => {
      resolveMove = resolve;
    }));
    const { controller } = await loadedController(current, { submitMove });
    render(<GameWindow controller={controller} onNewGame={() => undefined} />);
    await userEvent.click(screen.getByRole("button", { name: "2 of Clubs (2C), hand slot 1" }));
    await userEvent.click(screen.getByRole("button", { name: "Play 2 of Clubs at coffin position 9" }));
    screen.getAllByRole("button", { name: /hand slot/ }).forEach((button) => expect(button).toBeDisabled());
    resolveMove?.({ status: 200, data: { ...current, version: 1 } });
    cleanup();

    const opponent = view({ kind: "opponent_turn", status: "ready", job_id: null, retryable: true });
    const loaded = await loadedController(opponent);
    render(<Hand controller={loaded.controller} />);
    screen.getAllByRole("button", { name: /hand slot/ }).forEach((button) => expect(button).toBeDisabled());
  });
});

describe("resumable phases", () => {
  it("renders human, scoring, and round-advance views directly from reload data", async () => {
    const human = view({ kind: "human_turn" }, {
      active_player: "queen",
      legal_moves: [{ move_id: "m", card_id: "2C", hand_slot: 0, position: 0 }],
      total_scores: { human: 731, opponent: 509 },
    });
    let loaded = await loadedController(human);
    const rendered = render(<GameWindow controller={loaded.controller} onNewGame={() => undefined} />);
    expect(screen.getByRole("button", { name: "2 of Clubs (2C), hand slot 1" })).toBeEnabled();
    expect(screen.getByLabelText("Score tally")).toHaveTextContent("731");
    rendered.unmount();

    const result = scoringRecord();
    const scoring = view(
      { kind: "scoring", round_number: 1, next_step_index: 0, pending_narration_id: null },
      { status: "round_complete", active_player: null, pending_round_result: result },
    );
    loaded = await loadedController(scoring, {
      advanceRound: vi.fn(async () => ({ status: 200 as const, data: human })),
    });
    render(<GameWindow controller={loaded.controller} onNewGame={() => undefined} />);
    expect(screen.getByLabelText("Round 1 scoring presentation")).toHaveTextContent("Preparing");
    cleanup();

    const advancing = { ...scoring, phase: { kind: "round_advance", round_number: 1 } as const };
    loaded = await loadedController(advancing, {
      advanceRound: vi.fn(async () => ({ status: 200 as const, data: human })),
    });
    render(<GameWindow controller={loaded.controller} onNewGame={() => undefined} />);
    expect(screen.getByLabelText("Round 1 scoring presentation")).toHaveTextContent("Preparing");
  });

  it("automatically resumes one pending opponent job after reload", async () => {
    const pending = view({ kind: "opponent_turn", status: "pending", job_id: requestId, retryable: true });
    const completed = view({ kind: "human_turn" }, { version: 1, active_player: "queen" });
    const opponentTurn = vi.fn(async () => ({ status: 200 as const, data: completed }));
    const { controller } = await loadedController(pending, { opponentTurn });
    render(<GameWindow controller={controller} onNewGame={() => undefined} />);

    await waitFor(() => expect(opponentTurn).toHaveBeenCalledTimes(1));
    expect(opponentTurn).toHaveBeenCalledWith(pending.game_id, {
      expected_version: pending.version,
      request_id: requestId,
    });
  });

  it("uses only server values for legality and scoring presentation", async () => {
    const arbitrary = view(
      { kind: "human_turn" },
      {
        active_player: "queen",
        legal_moves: [{ move_id: "server-only", card_id: "2C", hand_slot: 0, position: 8 }],
        total_scores: { human: 12345, opponent: 6789 },
      },
    );
    const { controller } = await loadedController(arbitrary);
    render(<GameWindow controller={controller} onNewGame={() => undefined} />);
    await userEvent.click(screen.getByRole("button", { name: "2 of Clubs (2C), hand slot 1" }));

    expect(screen.getByRole("button", { name: "Play 2 of Clubs at coffin position 9" })).toBeEnabled();
    expect(screen.getAllByText("Legal")).toHaveLength(1);
    expect(screen.getByLabelText("Score tally")).toHaveTextContent("12345");
    expect(screen.queryByText("server-only")).not.toBeInTheDocument();
  });
});

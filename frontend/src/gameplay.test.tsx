// @vitest-environment jsdom

/**
 * Verifies board, hand, status, and scoring composition from controller state.
 * The suite uses a test controller to focus on interaction and presentation
 * while transport and stateless sequencing remain covered separately.
 */

import "@testing-library/jest-dom/vitest";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";

import { GameStart } from "./App";
import type { LegalMove, PresentationPhase } from "./contractPrimitives";
import type { HumanGameView } from "./gameView";
import { GameWindow, Hand } from "./gameplay";
import { scoringRecord } from "./scoringTestFixtures";
import {
  TestGameController,
  testGameView,
  type TestControllerActions,
} from "./testGameController";

afterEach(cleanup);

function view(
  phase: PresentationPhase,
  changes: Partial<HumanGameView> = {},
): HumanGameView {
  return testGameView(phase, changes);
}

function loadedController(
  current: HumanGameView,
  actions: TestControllerActions = {},
): TestGameController {
  return new TestGameController(current, actions);
}

describe("game start", () => {
  it("shows the starting box and prevents repeat role clicks until the game arrives", async () => {
    let finish!: (value: HumanGameView) => void;
    const createGame = vi.fn(() => new Promise<HumanGameView>((resolve) => { finish = resolve; }));
    const controller = new TestGameController(null, { createGame });
    const onGameCreated = vi.fn();
    render(<GameStart controller={controller} onGameCreated={onGameCreated} />);
    const queen = screen.getByRole("button", { name: "Start as Queen" });
    const king = screen.getByRole("button", { name: "Start as King" });

    await userEvent.click(queen);
    expect(screen.getByRole("status")).toHaveTextContent("Starting game");
    expect(screen.getByRole("status").closest(".start-overlay")?.parentElement)
      .toBe(queen.closest(".start-card"));
    expect(queen).toBeDisabled();
    expect(king).toBeDisabled();
    await userEvent.click(king);
    expect(createGame).toHaveBeenCalledTimes(1);
    expect(onGameCreated).not.toHaveBeenCalled();

    finish(view({ kind: "human_turn" }));
    await waitFor(() => expect(onGameCreated).toHaveBeenCalledTimes(1));
    expect(screen.queryByRole("status")).not.toBeInTheDocument();
  });

  it.each(["queen", "king"] as const)("creates a %s game from the role choice", async (role) => {
    const created = view({ kind: "human_turn" }, { human_role: role });
    const createGame = vi.fn(async () => created);
    const controller = new TestGameController(null, { createGame });
    const onGameCreated = vi.fn();
    render(<GameStart controller={controller} onGameCreated={onGameCreated} />);

    await userEvent.click(screen.getByRole("button", { name: `Start as ${role === "queen" ? "Queen" : "King"}` }));
    expect(createGame).toHaveBeenCalledWith(role);
    expect(onGameCreated).toHaveBeenCalledWith();
    expect(screen.getByRole("link", { name: "Read the rules" })).toHaveAttribute("target", "_blank");
  });
});

describe("card interaction", () => {
  const humanView = () => view(
    { kind: "human_turn" },
    {
      active_player: "queen",
      legal_moves: [{ card_id: "2C", hand_slot: 0, position: 8 }],
    },
  );

  async function submitThrough(method: "tap" | "keyboard" | "drag"): Promise<LegalMove> {
    const current = humanView();
    const submitted: LegalMove[] = [];
    const submitMove = vi.fn(async (move: LegalMove) => {
      submitted.push(move);
      return current;
    });
    const controller = loadedController(current, { playMove: submitMove });
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
    const move = submitted[0];
    if (move === undefined) throw new Error("move was not submitted");
    return move;
  }

  it("submits the same legal placement through drag, tap, and keyboard paths", async () => {
    const expected = { card_id: "2C", hand_slot: 0, position: 8 };
    expect(await submitThrough("drag")).toEqual(expected);
    cleanup();
    expect(await submitThrough("tap")).toEqual(expected);
    cleanup();
    expect(await submitThrough("keyboard")).toEqual(expected);
  });

  it("does not accept an unlisted drop or grid activation", async () => {
    const current = humanView();
    const submitMove = vi.fn();
    const controller = loadedController(current, { playMove: submitMove });
    render(<GameWindow controller={controller} onNewGame={() => undefined} />);
    await userEvent.click(screen.getByRole("button", { name: "2 of Clubs (2C), hand slot 1" }));
    const illegal = screen.getByLabelText("Empty coffin position 1");
    fireEvent.drop(illegal, { dataTransfer: { dropEffect: "none" } });
    fireEvent.click(illegal);

    expect(submitMove).not.toHaveBeenCalled();
    expect(screen.queryByText("Legal")).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Play 2 of Clubs at coffin position 9" })).toHaveTextContent("+");
  });

  it("disables every hand input while a mutation is pending or Dracula is active", async () => {
    const current = humanView();
    let resolveMove: ((value: HumanGameView) => void) | undefined;
    const submitMove = vi.fn(() => new Promise<HumanGameView>((resolve) => {
      resolveMove = resolve;
    }));
    const controller = loadedController(current, { playMove: submitMove });
    render(<GameWindow controller={controller} onNewGame={() => undefined} />);
    await userEvent.click(screen.getByRole("button", { name: "2 of Clubs (2C), hand slot 1" }));
    await userEvent.click(screen.getByRole("button", { name: "Play 2 of Clubs at coffin position 9" }));
    screen.getAllByRole("button", { name: /hand slot/ }).forEach((button) => expect(button).toBeDisabled());
    resolveMove?.(current);
    cleanup();

    const opponent = view({ kind: "opponent_turn" });
    const opponentController = loadedController(opponent);
    render(<Hand
      controller={opponentController}
      view={opponent}
      presentation={opponentController.getSnapshot().presentation}
    />);
    screen.getAllByRole("button", { name: /hand slot/ }).forEach((button) => expect(button).toBeDisabled());
  });
});

describe("game phases", () => {
  it("renders human and scoring views directly from replay data", async () => {
    const human = view({ kind: "human_turn" }, {
      active_player: "queen",
      legal_moves: [{ card_id: "2C", hand_slot: 0, position: 0 }],
      total_scores: { human: 731, opponent: 509 },
    });
    let controller = loadedController(human);
    const rendered = render(<GameWindow controller={controller} onNewGame={() => undefined} />);
    expect(screen.getByRole("button", { name: "2 of Clubs (2C), hand slot 1" })).toBeEnabled();
    expect(screen.getByLabelText("Score tally")).toHaveTextContent("731");
    rendered.unmount();

    const result = scoringRecord();
    const scoring = view(
      { kind: "scoring", round_number: 1 },
      { status: "round_complete", active_player: null, pending_round_result: result },
    );
    controller = loadedController(scoring, { advanceRound: vi.fn(async () => human) });
    render(<GameWindow controller={controller} onNewGame={() => undefined} />);
    expect(screen.getByLabelText("Round 1 scoring presentation")).toHaveTextContent("Preparing");
  });

  it("uses only server values for legality and scoring presentation", async () => {
    const arbitrary = view(
      { kind: "human_turn" },
      {
        active_player: "queen",
        legal_moves: [{ card_id: "2C", hand_slot: 0, position: 8 }],
        total_scores: { human: 12345, opponent: 6789 },
      },
    );
    const controller = loadedController(arbitrary);
    render(<GameWindow controller={controller} onNewGame={() => undefined} />);
    await userEvent.click(screen.getByRole("button", { name: "2 of Clubs (2C), hand slot 1" }));

    expect(screen.getByRole("button", { name: "Play 2 of Clubs at coffin position 9" })).toBeEnabled();
    expect(screen.queryByText("Legal")).not.toBeInTheDocument();
    expect(screen.getByLabelText("Score tally")).toHaveTextContent("12345");
  });
});

describe("turn identity", () => {
  it.each([
    ["queen", "Your turn — Queen · Rows"],
    ["king", "Your turn — King · Columns"],
  ] as const)("shows the human's %s orientation", async (humanRole, message) => {
    const current = view(
      { kind: "human_turn" },
      { human_role: humanRole, active_player: humanRole },
    );
    const controller = loadedController(current);
    render(<GameWindow controller={controller} onNewGame={() => undefined} />);

    expect(screen.getByLabelText(`Current turn: ${message}`)).toHaveTextContent(message);
  });
});

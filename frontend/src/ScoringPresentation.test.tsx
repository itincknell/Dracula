// @vitest-environment jsdom

import "@testing-library/jest-dom/vitest";
import { act, cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { FinalRoundPresentation, ScoringPresentation } from "./ScoringPresentation";
import type { ApiClient } from "./api";
import type { HumanGameView } from "./contracts";
import { GameController } from "./gameStore";
import {
  createScoringModel,
  createScoringTimeline,
  type ScoringFrame,
  type ScoringTiming,
} from "./scoringStateMachine";
import { scoringView } from "./scoringTestFixtures";

const requestId = "00000000-0000-4000-8000-000000000055";
const fastTiming: ScoringTiming = {
  entering: 10,
  revealValue: 10,
  collapseSum: 10,
  multiplierLabel: 10,
  multiplierFactor: 10,
  lineTotal: 10,
  orientationRanked: 10,
  narratorWait: 10,
  orientationHandoff: 10,
  compareRank: 10,
  selectRoundScore: 10,
  updateTotals: 10,
  reducedStep: 5,
};

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

async function setup(
  current: HumanGameView,
  overrides: Partial<ApiClient> = {},
  reducedMotion = false,
) {
  const client = api({
    getGame: vi.fn(async () => ({ status: 200 as const, data: current })),
    ...overrides,
  });
  const controller = new GameController(client, { requestId: () => requestId });
  await controller.loadGame(current.game_id);
  const rendered = render(
    <ScoringPresentation
      controller={controller}
      view={current}
      reducedMotion={reducedMotion}
      timing={fastTiming}
    />,
  );
  const timeline = createScoringTimeline(
    createScoringModel(current.pending_round_result!, current.human_role),
    current.narration_enabled,
  );
  return { client, controller, rendered, timeline };
}

async function advanceFrames(count: number, reducedMotion = false): Promise<void> {
  const duration = reducedMotion ? fastTiming.reducedStep : fastTiming.entering;
  for (let index = 0; index < count; index += 1) {
    await act(async () => {
      vi.advanceTimersByTime(duration);
    });
  }
}

function frameIndex(timeline: ScoringFrame[], predicate: (frame: ScoringFrame) => boolean): number {
  const index = timeline.findIndex(predicate);
  if (index < 0) throw new Error("requested scoring frame is absent");
  return index;
}

beforeEach(() => vi.useFakeTimers());
afterEach(() => {
  cleanup();
  vi.clearAllTimers();
  vi.useRealTimers();
});

describe("scoring presentation frames", () => {
  it.each([
    [0, 0, "No Multiplier", []],
    [0, 1, "Vampire", ["V1"]],
    [0, 2, "3× Hearts", ["7H", "8H", "9H"]],
    [1, 0, "2× Hearts", ["4H", "7H"]],
    [1, 1, "3× Red", ["2D", "V1", "8H"]],
  ] as const)(
    "shows the supplied multiplier for orientation %s line %s",
    async (orientationIndex, lineIndex, label, highlighted) => {
      const current = scoringView();
      const { rendered, timeline } = await setup(current);
      const target = frameIndex(
        timeline,
        (frame) =>
          frame.kind === "multiplier_label" &&
          frame.orientationIndex === orientationIndex &&
          frame.lineIndex === lineIndex,
      );
      await advanceFrames(target);

      expect(screen.getByText(label)).toBeInTheDocument();
      const emphasized = rendered.container.querySelectorAll(".scoring-card.multiplier-card");
      expect(emphasized).toHaveLength(highlighted.length);
      highlighted.forEach((cardId) => {
        expect(rendered.container.querySelector(`[data-card-id="${cardId}"]`)).toHaveClass("multiplier-card");
      });
    },
  );

  it("reveals values, the supplied base, numeric factor, and supplied total in order", async () => {
    const current = scoringView();
    const { timeline } = await setup(current);
    const firstReveal = frameIndex(timeline, (frame) => frame.kind === "reveal_value");
    await advanceFrames(firstReveal + 2);
    expect(screen.getByText("11 + 7 + 5")).toBeInTheDocument();
    await advanceFrames(1);
    expect(screen.getByText("47")).toBeInTheDocument();
    await advanceFrames(1);
    expect(screen.getByText("No Multiplier")).toBeInTheDocument();
    await advanceFrames(1);
    expect(screen.getByText("47 × 1")).toBeInTheDocument();
    await advanceFrames(1);
    expect(document.querySelector(".score-expression")).toHaveTextContent("30");
  });

  it("displays zero for the supplied Vampire card while totaling its line", async () => {
    const current = scoringView();
    const { timeline } = await setup(current);
    const vampire = frameIndex(
      timeline,
      (frame) =>
        frame.kind === "reveal_value" &&
        frame.orientationIndex === 0 &&
        frame.lineIndex === 1 &&
        frame.cardIndex === 1,
    );
    await advanceFrames(vampire);
    expect(document.querySelector(".score-expression")).toHaveTextContent("11 + 0");
  });

  it("uses server ranks for the centered tally and hides it before the second orientation", async () => {
    const current = scoringView();
    const { timeline } = await setup(current);
    const ranked = frameIndex(
      timeline,
      (frame) => frame.kind === "orientation_ranked" && frame.orientationIndex === 0,
    );
    await advanceFrames(ranked);
    expect(
      Array.from(document.querySelectorAll(".ranked-line-totals span"), (node) => node.textContent),
    ).toEqual(["90", "30", "0"]);

    const second = frameIndex(
      timeline,
      (frame) => frame.kind === "reveal_value" && frame.orientationIndex === 1,
    );
    await advanceFrames(second - ranked);
    expect(screen.getByText("Dracula's Score")).toBeInTheDocument();
    expect(document.querySelector(".ranked-line-totals")).not.toBeInTheDocument();
  });

  it("retains multiple ties, strikes them, then shows the server-selected third rank", async () => {
    const current = scoringView({ selectedRank: 3, thirdRankTied: true });
    const { timeline } = await setup(current);
    const comparisons = timeline
      .map((frame, index) => ({ frame, index }))
      .filter(({ frame }) => frame.kind === "compare_rank");
    await advanceFrames(comparisons[0].index);
    expect(screen.getAllByText("Tie")).toHaveLength(2);
    await advanceFrames(comparisons[1].index - comparisons[0].index);
    expect(screen.getAllByText("Tie")).toHaveLength(4);
    await advanceFrames(comparisons[2].index - comparisons[1].index);
    expect(screen.getAllByText("Tie")).toHaveLength(6);
    await advanceFrames(1);
    expect(screen.getAllByText("Round Score")).toHaveLength(2);
  });

  it("uses reduced motion while preserving every information stage", async () => {
    const current = scoringView();
    const { rendered, timeline } = await setup(current, {}, true);
    expect(rendered.container.querySelector(".scoring-presentation")).toHaveAttribute(
      "data-reduced-motion",
      "true",
    );
    const multiplier = frameIndex(timeline, (frame) => frame.kind === "multiplier_label");
    await advanceFrames(multiplier, true);
    expect(screen.getByText("No Multiplier")).toBeInTheDocument();
  });

  it.each([
    "reveal_value",
    "collapse_sum",
    "multiplier_label",
    "orientation_ranked",
    "orientation_handoff",
    "compare_rank",
    "update_totals",
    "complete",
  ] as const)("restarts from the dealer's first series after reload during %s", async (stage) => {
    const current = scoringView();
    const first = await setup(current);
    const target = frameIndex(first.timeline, (frame) => frame.kind === stage);
    await advanceFrames(target);
    expect(first.rendered.container.querySelector(".scoring-presentation")).toHaveAttribute("data-stage", stage);
    first.rendered.unmount();
    vi.clearAllTimers();

    const second = await setup(current);
    expect(second.rendered.container.querySelector(".scoring-presentation")).toHaveAttribute(
      "data-stage",
      "entering",
    );
  });
});

describe("round completion controls", () => {
  it("sends exactly one round-advance request even under repeated activation", async () => {
    const current = scoringView();
    const next = {
      ...current,
      version: 9,
      status: "playing" as const,
      turn_number: 0,
      pending_round_result: null,
      phase: { kind: "human_turn" as const },
    };
    let resolveAdvance: ((value: { status: 200; data: HumanGameView }) => void) | undefined;
    const advanceRound = vi.fn(
      () => new Promise<{ status: 200; data: HumanGameView }>((resolve) => {
        resolveAdvance = resolve;
      }),
    );
    const { timeline } = await setup(current, { advanceRound });
    await advanceFrames(timeline.length - 1);
    const button = screen.getByRole("button", { name: "Deal Next Round" });
    fireEvent.click(button);
    fireEvent.click(button);
    expect(advanceRound).toHaveBeenCalledTimes(1);
    resolveAdvance?.({ status: 200, data: next });
  });

  it("finalizes round six once and leaves Play Again for the completed-game view", async () => {
    const current = scoringView({ roundNumber: 6 });
    const advanceRound = vi.fn(
      () => new Promise<{ status: 200; data: HumanGameView }>(() => undefined),
    );
    const { timeline } = await setup(current, { advanceRound });
    await advanceFrames(timeline.length - 1);
    await act(async () => undefined);
    expect(advanceRound).toHaveBeenCalledTimes(1);
    await advanceFrames(3);
    expect(advanceRound).toHaveBeenCalledTimes(1);
    expect(screen.getByText("Finalizing game…")).toBeInTheDocument();
  });

  it("shows Play Again only after round six has entered the completed-game lifecycle", () => {
    const scoring = scoringView({ roundNumber: 6 });
    const record = scoring.pending_round_result!;
    const completed: HumanGameView = {
      ...scoring,
      status: "game_complete",
      pending_round_result: null,
      completed_rounds: [record],
      phase: { kind: "game_complete", outcome: "human" },
    };
    const onNewGame = vi.fn();
    render(<FinalRoundPresentation record={record} view={completed} onNewGame={onNewGame} />);
    fireEvent.click(screen.getByRole("button", { name: "Play Again" }));
    expect(onNewGame).toHaveBeenCalledTimes(1);
  });

  it("renders cumulative totals exactly as supplied", async () => {
    const current = scoringView();
    const { timeline } = await setup(current);
    const totals = frameIndex(timeline, (frame) => frame.kind === "update_totals");
    await advanceFrames(totals);
    const panel = screen.getByLabelText("Round 1 totals");
    expect(panel).toHaveTextContent("101");
    expect(panel).toHaveTextContent("90");
    expect(panel).toHaveTextContent("997");
    expect(panel).toHaveTextContent("203");
    expect(panel).toHaveTextContent("509");
  });
});

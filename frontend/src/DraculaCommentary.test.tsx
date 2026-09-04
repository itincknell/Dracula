// @vitest-environment jsdom

import "@testing-library/jest-dom/vitest";
import { act, cleanup, render } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import gameViewFixture from "../../contracts/v1/human-game-view.json";
import type { HumanGameView } from "./contracts";
import {
  currentDialogueIsVisible,
  DraculaCommentary,
  draculaMood,
  RetroDialogue,
} from "./DraculaCommentary";

afterEach(() => {
  cleanup();
  vi.useRealTimers();
});

function view(changes: Partial<HumanGameView> = {}): HumanGameView {
  return {
    ...(structuredClone(gameViewFixture) as unknown as HumanGameView),
    ...changes,
  };
}

describe("Dracula portrait state", () => {
  it("uses the exact cumulative-score and round priorities", () => {
    expect(draculaMood(view({ round_number: 1, total_scores: { human: 0, opponent: 0 } })))
      .toBe("default");
    expect(draculaMood(view({ round_number: 3, total_scores: { human: 10, opponent: 40 } })))
      .toBe("default");
    expect(draculaMood(view({ round_number: 2, total_scores: { human: 11, opponent: 10 } })))
      .toBe("angrier");
    expect(draculaMood(view({ round_number: 4, total_scores: { human: 80, opponent: 31 } })))
      .toBe("angrier");
    expect(draculaMood(view({ round_number: 4, total_scores: { human: 80, opponent: 30 } })))
      .toBe("angriest");
    expect(draculaMood(view({ round_number: 4, total_scores: { human: 30, opponent: 31 } })))
      .toBe("winning");
    expect(draculaMood(view({
      round_number: 6,
      status: "game_complete",
      total_scores: { human: 101, opponent: 100 },
    }))).toBe("angriest");
  });

  it("reveals score-driven faces only with that round's dialogue", () => {
    const roundFour = view({
      round_number: 4,
      status: "round_complete",
      total_scores: { human: 100, opponent: 40 },
    });
    expect(currentDialogueIsVisible(roundFour, 0)).toBe(false);
    expect(currentDialogueIsVisible(roundFour, 1)).toBe(true);
    expect(currentDialogueIsVisible({ ...roundFour, round_number: 1 }, 1)).toBe(false);
    expect(currentDialogueIsVisible({ ...roundFour, round_number: 1 }, 2)).toBe(true);
    expect(currentDialogueIsVisible(
      { ...roundFour, round_number: 5, status: "playing" },
      5,
    )).toBe(false);
  });

  it("changes to the score-driven face exactly when dialogue starts printing", () => {
    vi.useFakeTimers();
    const rendered = render(
      <DraculaCommentary
        view={view({
          round_number: 4,
          status: "round_complete",
          total_scores: { human: 100, opponent: 40 },
        })}
        narration={{
          enabled: true,
          pending: false,
          messages: ["Opening", "One", "Two", "Three", "Four"],
        }}
      />,
    );
    const portrait = rendered.container.querySelector(".portrait-placeholder");
    expect(portrait).toHaveAttribute("data-mood", "default");
    expect(portrait).not.toHaveAttribute("data-animated");

    act(() => vi.advanceTimersByTime(34));
    expect(portrait).toHaveAttribute("data-mood", "angriest");
    expect(portrait).not.toHaveAttribute("data-animated");
    const sources = Array.from(rendered.container.querySelectorAll(".dracula-avatar"))
      .map((image) => image.getAttribute("src"));
    expect(sources).toEqual([
      "/portraits/dracula-angry-frown.jpg",
      "/portraits/dracula-angrier-frown.jpg",
      "/portraits/dracula-angriest-grimace.jpg",
      "/portraits/dracula-winning-grin.jpg",
    ]);
    expect(sources.join(" ")).not.toContain("dracula.png");
  });

  it("does not change an already-printed face when scores change without new dialogue", () => {
    vi.useFakeTimers();
    const rendered = render(
      <DraculaCommentary
        view={view({ total_scores: { human: 0, opponent: 0 } })}
        narration={{ enabled: true, pending: false, messages: ["Opening"] }}
      />,
    );
    act(() => vi.advanceTimersByTime(34));
    const portrait = rendered.container.querySelector(".portrait-placeholder");
    const defaultImage = rendered.container.querySelector(
      ".dracula-avatar[data-portrait='default']",
    );
    expect(portrait).toHaveAttribute("data-mood", "default");

    rendered.rerender(
      <DraculaCommentary
        view={view({
          round_number: 4,
          status: "round_complete",
          total_scores: { human: 100, opponent: 20 },
        })}
        narration={{ enabled: true, pending: true, messages: ["Opening"] }}
      />,
    );
    expect(portrait).toHaveAttribute("data-mood", "default");
    expect(rendered.container.querySelector(".dracula-avatar[data-portrait='default']"))
      .toBe(defaultImage);

    rendered.rerender(
      <DraculaCommentary
        view={view({
          round_number: 4,
          status: "round_complete",
          total_scores: { human: 100, opponent: 20 },
        })}
        narration={{ enabled: true, pending: false, messages: ["Opening", "New taunt"] }}
      />,
    );
    expect(rendered.container.querySelector(".portrait-placeholder"))
      .toHaveAttribute("data-mood", "default");
    act(() => vi.advanceTimersByTime(34));
    expect(rendered.container.querySelector(".portrait-placeholder"))
      .toHaveAttribute("data-mood", "angriest");
  });

  it("keeps a round-four losing face after its dialogue has printed", () => {
    vi.useFakeTimers();
    const losing = view({
      round_number: 4,
      status: "round_complete",
      total_scores: { human: 104, opponent: 100 },
    });
    const rendered = render(
      <DraculaCommentary
        view={losing}
        narration={{ enabled: true, pending: false, messages: ["A narrow defeat."] }}
      />,
    );

    act(() => vi.advanceTimersByTime(34));
    expect(rendered.container.querySelector(".portrait-placeholder"))
      .toHaveAttribute("data-mood", "angrier");

    rendered.rerender(
      <DraculaCommentary
        view={{ ...losing, status: "playing", round_number: 5 }}
        narration={{ enabled: true, pending: false, messages: ["A narrow defeat."] }}
      />,
    );
    expect(rendered.container.querySelector(".portrait-placeholder"))
      .toHaveAttribute("data-mood", "angrier");
  });

  it("keeps a severe-loss face static after round advance", () => {
    vi.useFakeTimers();
    const losing = view({
      round_number: 4,
      status: "round_complete",
      total_scores: { human: 100, opponent: 40 },
    });
    const rendered = render(
      <DraculaCommentary
        view={losing}
        narration={{ enabled: true, pending: false, messages: ["You will regret this."] }}
      />,
    );

    act(() => vi.advanceTimersByTime(34));
    const portrait = rendered.container.querySelector(".portrait-placeholder");
    expect(portrait).toHaveAttribute("data-mood", "angriest");
    expect(portrait).not.toHaveAttribute("data-animated");

    rendered.rerender(
      <DraculaCommentary
        view={{ ...losing, status: "playing", round_number: 5 }}
        narration={{ enabled: true, pending: false, messages: ["You will regret this."] }}
      />,
    );
    expect(portrait).toHaveAttribute("data-mood", "angriest");
    expect(portrait).not.toHaveAttribute("data-animated");
  });

  it("shows a static angriest face for any final-game loss until a new game", () => {
    vi.useFakeTimers();
    const finalLoss = view({
      round_number: 6,
      status: "game_complete",
      total_scores: { human: 104, opponent: 100 },
    });
    const rendered = render(
      <DraculaCommentary
        view={finalLoss}
        narration={{ enabled: true, pending: false, messages: ["Impossible!"] }}
      />,
    );

    act(() => vi.advanceTimersByTime(34));
    const portrait = rendered.container.querySelector(".portrait-placeholder");
    expect(portrait).toHaveAttribute("data-mood", "angriest");
    expect(portrait).not.toHaveAttribute("data-animated");

    rendered.rerender(
      <DraculaCommentary
        view={{
          ...finalLoss,
          game_id: `${finalLoss.game_id}-next`,
          round_number: 1,
          status: "playing",
          total_scores: { human: 0, opponent: 0 },
        }}
        narration={{ enabled: true, pending: false, messages: ["Welcome."] }}
      />,
    );
    expect(portrait).toHaveAttribute("data-mood", "default");
    expect(portrait).not.toHaveAttribute("data-animated");
  });
});

describe("retro dialogue", () => {
  it("reveals text one character at a time while moving a block cursor", () => {
    vi.useFakeTimers();
    const rendered = render(<RetroDialogue text="Bite!" characterDelayMs={10} />);
    const visible = rendered.container.querySelector<HTMLElement>("[aria-hidden='true']");
    expect(visible).toHaveTextContent("█");
    expect(visible).not.toHaveTextContent("Bite!");

    act(() => vi.advanceTimersByTime(30));
    expect(visible).toHaveTextContent("Bit█");

    act(() => vi.advanceTimersByTime(20));
    expect(visible).toHaveTextContent("Bite!█");
    expect(rendered.container.querySelector(".commentary-dialogue"))
      .toHaveAttribute("data-complete", "true");
  });
});

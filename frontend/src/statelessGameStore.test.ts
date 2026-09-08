// @vitest-environment jsdom

/**
 * Exercises stateless store sequencing through its observable public contract.
 * Tests cover commands, recovery, opponent delay, narration races, and errors
 * using fake transport and browser storage boundaries.
 */

import { cleanup, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { createElement } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { GameStart } from "./App";
import type { ServerGamePhase } from "./contractPrimitives";
import type {
  NarrationResponse,
  RecoveryEnvelope,
  StatelessGameResponse,
} from "./statelessContracts";
import type { StatelessApiClient } from "./statelessApi";
import { StatelessApiError } from "./statelessApi";
import { StatelessGameController } from "./statelessGameStore";
import { RECOVERY_STORAGE_KEY, type BrowserStorage } from "./statelessRecovery";

afterEach(cleanup);

class MemoryStorage implements BrowserStorage {
  readonly entries = new Map<string, string>();
  getItem(key: string) { return this.entries.get(key) ?? null; }
  setItem(key: string, value: string) { this.entries.set(key, value); }
  removeItem(key: string) { this.entries.delete(key); }
}

function envelope(history: RecoveryEnvelope["history"] = [
  { type: "select_role", human_role: "queen" },
]): RecoveryEnvelope {
  return { seed: "stateless-browser-fixture", history };
}

function game(
  phase: ServerGamePhase = { kind: "human_turn" },
  changes: Partial<StatelessGameResponse["game"]> = {},
  recovery = envelope(),
): StatelessGameResponse {
  return {
    envelope: recovery,
    game: {
      status: "playing",
      round_number: 1,
      turn_number: 0,
      dealer: "king",
      active_player: "queen",
      human_role: "queen",
      opponent_role: "king",
      coffin: [null, null, null, null, "4C", null, null, null, null],
      current_round_moves: [],
      pending_round_result: null,
      completed_rounds: [],
      total_scores: { human: 0, opponent: 0 },
      phase,
      human_hand: ["2C", "10D", "QH", "KS"],
      legal_moves: [{ card_id: "2C", hand_slot: 0, position: 1 }],
      ...changes,
    },
  };
}

function api(overrides: Partial<StatelessApiClient> = {}): StatelessApiClient {
  return {
    startGame: vi.fn(),
    applyCommand: vi.fn(),
    resumeGame: vi.fn(),
    narrate: vi.fn(async ({ cue_type }) => (
      { cue_type, status: "unavailable", text: null } as NarrationResponse
    )),
    ...overrides,
  } as StatelessApiClient;
}

describe("stateless browser game controller", () => {
  it("enters the game screen while Dracula's opening delay is still active", async () => {
    const opened = game(
      { kind: "human_turn" },
      {
        dealer: "queen",
        turn_number: 1,
        coffin: [null, "8D", null, null, "4C", null, null, null, null],
        current_round_moves: [{
          player: "king",
          card_id: "8D",
          position: 1,
          turn_number: 1,
        }],
      },
    );
    let releaseDelay: (() => void) | undefined;
    const delay = vi.fn(() => new Promise<void>((resolve) => {
      releaseDelay = resolve;
    }));
    const controller = new StatelessGameController(api({
      startGame: vi.fn(async () => opened),
    }), new MemoryStorage(), delay);
    const onGameCreated = vi.fn();
    render(createElement(GameStart, { controller, onGameCreated }));

    await userEvent.click(screen.getByRole("button", { name: "Start as Queen" }));
    await vi.waitFor(() => expect(delay).toHaveBeenCalledTimes(1));
    expect(onGameCreated).toHaveBeenCalledTimes(1);
    expect(controller.getSnapshot().view).toMatchObject({
      turn_number: 0,
      phase: { kind: "opponent_turn" },
    });

    releaseDelay?.();
    await vi.waitFor(() => expect(controller.getSnapshot().view?.turn_number).toBe(1));
  });

  it("shows the dealt center before revealing Dracula's opening placement", async () => {
    const opened = game(
      { kind: "human_turn" },
      {
        dealer: "queen",
        turn_number: 1,
        coffin: [null, "8D", null, null, "4C", null, null, null, null],
        current_round_moves: [{
          player: "king",
          card_id: "8D",
          position: 1,
          turn_number: 1,
        }],
      },
    );
    let releaseDelay: (() => void) | undefined;
    const delay = vi.fn(() => new Promise<void>((resolve) => {
      releaseDelay = resolve;
    }));
    const controller = new StatelessGameController(api({
      startGame: vi.fn(async () => opened),
    }), new MemoryStorage(), delay);

    const creating = controller.createGame("queen");
    await vi.waitFor(() => expect(delay).toHaveBeenCalledTimes(1));
    expect(controller.getSnapshot()).toMatchObject({
      view: {
        turn_number: 0,
        active_player: "king",
        coffin: [null, null, null, null, "4C", null, null, null, null],
        current_round_moves: [],
        phase: { kind: "opponent_turn" },
      },
      presentation: { pending: "opponent_turn" },
    });

    releaseDelay?.();
    await expect(creating).resolves.toMatchObject({
      turn_number: 1,
      coffin: [null, "8D", null, null, "4C", null, null, null, null],
    });
  });

  it("applies the same Dracula pause to later-round opening placements", async () => {
    const scoring = game(
      { kind: "scoring", round_number: 1 },
      { status: "round_complete", active_player: null, legal_moves: [] },
    );
    const nextEnvelope = envelope([
      ...scoring.envelope.history,
      { type: "advance_round" },
    ]);
    const opened = game(
      { kind: "human_turn" },
      {
        round_number: 2,
        dealer: "queen",
        turn_number: 1,
        coffin: [null, null, null, null, "6H", null, null, "9S", null],
        current_round_moves: [{
          player: "king",
          card_id: "9S",
          position: 7,
          turn_number: 1,
        }],
      },
      nextEnvelope,
    );
    let releaseDelay: (() => void) | undefined;
    const delay = vi.fn(() => new Promise<void>((resolve) => {
      releaseDelay = resolve;
    }));
    const controller = new StatelessGameController(api({
      startGame: vi.fn(async () => scoring),
      applyCommand: vi.fn(async () => opened),
    }), new MemoryStorage(), delay);
    await controller.createGame("queen");

    const advancing = controller.advanceRound();
    await vi.waitFor(() => expect(delay).toHaveBeenCalledTimes(1));
    expect(controller.getSnapshot()).toMatchObject({
      view: {
        round_number: 2,
        turn_number: 0,
        active_player: "king",
        coffin: [null, null, null, null, "6H", null, null, null, null],
        current_round_moves: [],
        phase: { kind: "opponent_turn" },
      },
      presentation: { pending: "opponent_turn" },
    });

    releaseDelay?.();
    await expect(advancing).resolves.toBe(true);
    expect(controller.getSnapshot().view).toMatchObject({
      turn_number: 1,
      coffin: [null, null, null, null, "6H", null, null, "9S", null],
    });
  });

  it("stores only the accepted recovery envelope and replays it after a cold reload", async () => {
    const storage = new MemoryStorage();
    const started = game();
    const startGame = vi.fn(async () => started);
    const first = new StatelessGameController(api({ startGame }), storage);
    expect(await first.createGame("queen")).not.toBeNull();

    const stored = JSON.parse(storage.getItem(RECOVERY_STORAGE_KEY) ?? "null") as Record<string, unknown>;
    expect(stored).toEqual(started.envelope);
    expect(Object.keys(stored).sort()).toEqual(["history", "seed"]);
    expect(JSON.stringify(stored)).not.toMatch(/logit|tensor|engine|search|stock|hand/i);

    const resumeGame = vi.fn(async () => started);
    const cold = new StatelessGameController(api({ resumeGame }), storage);
    const restored = await cold.loadGame();
    expect(restored).toEqual(first.getSnapshot().view);
    expect(resumeGame).toHaveBeenCalledWith({ envelope: started.envelope });
  });

  it("clears malformed local state without sending it to the server", async () => {
    const storage = new MemoryStorage();
    storage.setItem(RECOVERY_STORAGE_KEY, JSON.stringify({ seed: "x", history: [] }));
    const resumeGame = vi.fn();
    const controller = new StatelessGameController(api({ resumeGame }), storage);
    expect(await controller.loadGame()).toBeNull();
    expect(storage.getItem(RECOVERY_STORAGE_KEY)).toBeNull();
    expect(resumeGame).not.toHaveBeenCalled();
    expect(controller.getSnapshot().presentation.error?.message).toContain("cleared");
  });

  it("clears a well-formed envelope rejected as an impossible history", async () => {
    const storage = new MemoryStorage();
    const saved = envelope();
    storage.setItem(RECOVERY_STORAGE_KEY, JSON.stringify(saved));
    const resumeGame = vi.fn(async () => {
      throw new StatelessApiError({
        code: "invalid_history",
        message: "history contains an invalid engine transition",
        retryable: false,
      });
    });
    const controller = new StatelessGameController(api({ resumeGame }), storage);

    expect(await controller.loadGame()).toBeNull();
    expect(resumeGame).toHaveBeenCalledWith({ envelope: saved });
    expect(storage.getItem(RECOVERY_STORAGE_KEY)).toBeNull();
    expect(controller.getSnapshot().presentation.error?.message).toContain("cleared");
  });

  it("persists only a server-accepted placement and uses it for exact retry-safe recovery", async () => {
    const storage = new MemoryStorage();
    const started = game();
    const acceptedEnvelope = envelope([
      ...started.envelope.history,
      { type: "place", hand_slot: 0, position: 1 },
    ]);
    const accepted = game(
      { kind: "human_turn" },
      {
        turn_number: 2,
        coffin: [null, "2C", null, null, "4C", "8D", null, null, null],
        human_hand: [null, "10D", "QH", "KS"],
      },
      acceptedEnvelope,
    );
    const applyCommand = vi.fn(async () => accepted);
    const delay = vi.fn(async () => undefined);
    const controller = new StatelessGameController(api({
      startGame: vi.fn(async () => started),
      applyCommand,
    }), storage, delay);
    await controller.createGame("queen");
    controller.selectCard(0);
    expect(await controller.playPosition(1)).toBe(true);
    expect(applyCommand).toHaveBeenCalledWith({
      envelope: started.envelope,
      command: { type: "place", hand_slot: 0, position: 1 },
    });
    expect(delay).toHaveBeenCalledTimes(1);
    expect(JSON.parse(storage.getItem(RECOVERY_STORAGE_KEY) ?? "null")).toEqual(acceptedEnvelope);
  });

  it("shows the human placement immediately and holds only Dracula's move behind the pause", async () => {
    const storage = new MemoryStorage();
    const started = game();
    const accepted = game(
      { kind: "human_turn" },
      { turn_number: 2 },
      envelope([
        ...started.envelope.history,
        { type: "place", hand_slot: 0, position: 1 },
      ]),
    );
    let releaseDelay: (() => void) | undefined;
    const delay = vi.fn(() => new Promise<void>((resolve) => {
      releaseDelay = resolve;
    }));
    const controller = new StatelessGameController(api({
      startGame: vi.fn(async () => started),
      applyCommand: vi.fn(async () => accepted),
    }), storage, delay);
    await controller.createGame("queen");
    controller.selectCard(0);

    const play = controller.playPosition(1);
    await vi.waitFor(() => expect(delay).toHaveBeenCalledTimes(1));
    expect(controller.getSnapshot().presentation.pending).toBe("opponent_turn");
    expect(controller.getSnapshot().view).toMatchObject({
      turn_number: 1,
      active_player: "king",
      coffin: [null, "2C", null, null, "4C", null, null, null, null],
      human_hand: [null, "10D", "QH", "KS"],
      phase: { kind: "opponent_turn" },
      legal_moves: [],
      current_round_moves: [{
        player: "queen",
        card_id: "2C",
        position: 1,
        turn_number: 1,
      }],
    });

    releaseDelay?.();
    await expect(play).resolves.toBe(true);
    expect(controller.getSnapshot().presentation.pending).toBeNull();
    expect(controller.getSnapshot().view?.turn_number).toBe(2);
  });

  it("rolls back the visible human placement when the server rejects it", async () => {
    const started = game();
    let rejectCommand: ((error: Error) => void) | undefined;
    const command = new Promise<never>((_resolve, reject) => {
      rejectCommand = reject;
    });
    const controller = new StatelessGameController(api({
      startGame: vi.fn(async () => started),
      applyCommand: vi.fn(() => command),
    }), new MemoryStorage(), vi.fn(async () => undefined));
    await controller.createGame("queen");
    controller.selectCard(0);

    const play = controller.playPosition(1);
    expect(controller.getSnapshot().view?.coffin[1]).toBe("2C");
    expect(controller.getSnapshot().presentation.pending).toBe("opponent_turn");

    rejectCommand?.(new Error("placement failed"));
    await expect(play).resolves.toBe(false);
    expect(controller.getSnapshot().view).toEqual(started.game);
    expect(controller.getSnapshot().presentation.error?.message).toBe("placement failed");
  });

  it("fetches a round cue during scoring and reveals it only when scoring completes", async () => {
    const storage = new MemoryStorage();
    const started = game();
    const roundEnvelope = envelope([
      ...started.envelope.history,
      { type: "place", hand_slot: 0, position: 1 },
    ]);
    const round = game(
      { kind: "scoring", round_number: 1 },
      { status: "round_complete", active_player: null, legal_moves: [] },
      roundEnvelope,
    );
    const nextEnvelope = envelope([
      ...roundEnvelope.history,
      { type: "advance_round" },
    ]);
    const nextRound = game(
      { kind: "human_turn" },
      { round_number: 2 },
      nextEnvelope,
    );
    const narrate = vi.fn(async ({ cue_type }) => (
      {
        cue_type,
        status: "ready",
        text: cue_type === "opening" ? "Opening" : "Round one",
      } as NarrationResponse
    ));
    const applyCommand = vi.fn()
      .mockResolvedValueOnce(round)
      .mockResolvedValueOnce(nextRound);
    const controller = new StatelessGameController(api({
      startGame: vi.fn(async () => started),
      applyCommand,
      narrate,
    }), storage, async () => undefined);
    await controller.createGame("queen");
    await vi.waitFor(() => expect(controller.getSnapshot().narration.messages).toEqual(["Opening"]));
    controller.selectCard(0);
    await controller.playPosition(1);
    await vi.waitFor(() => expect(narrate).toHaveBeenLastCalledWith({
      envelope: roundEnvelope,
      cue_type: "round_transition",
    }));
    expect(controller.getSnapshot().narration.messages).toEqual(["Opening"]);
    controller.revealNarration();
    expect(controller.getSnapshot().narration.messages).toEqual(["Opening", "Round one"]);
    await controller.advanceRound();
    expect(controller.getSnapshot().narration.messages).toEqual([]);
  });

  it("requests final narration only after the game-completing command", async () => {
    const storage = new MemoryStorage();
    const scoringEnvelope = envelope([
      { type: "select_role", human_role: "queen" },
      { type: "place", hand_slot: 0, position: 1 },
    ]);
    const scoring = game(
      { kind: "scoring", round_number: 6 },
      { status: "round_complete", round_number: 6, active_player: null, legal_moves: [] },
      scoringEnvelope,
    );
    const completeEnvelope = envelope([...scoringEnvelope.history, { type: "advance_round" }]);
    const complete = game(
      { kind: "game_complete", outcome: "human" },
      { status: "game_complete", round_number: 6, active_player: null, legal_moves: [] },
      completeEnvelope,
    );
    storage.setItem(RECOVERY_STORAGE_KEY, JSON.stringify(scoringEnvelope));
    const narrate = vi.fn(async ({ cue_type }) => (
      { cue_type, status: "unavailable", text: null } as NarrationResponse
    ));
    const controller = new StatelessGameController(api({
      resumeGame: vi.fn(async () => scoring),
      applyCommand: vi.fn(async () => complete),
      narrate,
    }), storage);
    await controller.loadGame();
    expect(narrate).not.toHaveBeenCalled();
    await controller.advanceRound();
    await vi.waitFor(() => expect(narrate).toHaveBeenCalledWith({
      envelope: completeEnvelope,
      cue_type: "final_result",
    }));
  });
});

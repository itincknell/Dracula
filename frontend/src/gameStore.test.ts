import { describe, expect, it, vi } from "vitest";

import gameViewFixture from "../../contracts/v1/human-game-view.json";
import { ApiError, type ApiClient } from "./api";
import type { ApiErrorResponse, HumanGameView, ResumablePhase } from "./contracts";
import { GameController } from "./gameStore";

const requestId = "00000000-0000-4000-8000-000000000099";

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

describe("authoritative game controller", () => {
  it("keeps the confirmed hand and coffin unchanged while a move is pending", async () => {
    let resolveMove: ((value: { status: 200; data: HumanGameView }) => void) | undefined;
    const initial = view(
      { kind: "human_turn" },
      {
        active_player: "queen",
        legal_moves: [{ move_id: "move-a", card_id: "2C", hand_slot: 0, position: 1 }],
      },
    );
    const submitMove = vi.fn(() => new Promise<{ status: 200; data: HumanGameView }>((resolve) => {
      resolveMove = resolve;
    }));
    const client = api({
      getGame: vi.fn(async () => ({ status: 200 as const, data: initial })),
      submitMove,
    });
    const controller = new GameController(client, { requestId: () => requestId });
    await controller.loadGame(initial.game_id);
    controller.selectCard(0);
    const pending = controller.playPosition(1);

    expect(controller.getSnapshot().presentation.pending).toBe("human_move");
    expect(controller.getSnapshot().view).toEqual(initial);
    resolveMove?.({ status: 200, data: { ...initial, version: 1 } });
    await pending;
  });

  it("reconciles a stale response before input becomes available again", async () => {
    const initial = view(
      { kind: "human_turn" },
      {
        active_player: "queen",
        legal_moves: [{ move_id: "expired", card_id: "2C", hand_slot: 0, position: 1 }],
      },
    );
    const authoritative = view(
      { kind: "human_turn" },
      {
        version: 7,
        active_player: "queen",
        coffin: [null, "2C", null, null, "4C", null, null, null, null],
        human_hand: [null, "10D", "QH", "KS"],
        legal_moves: [{ move_id: "fresh", card_id: "10D", hand_slot: 1, position: 0 }],
      },
    );
    const error: ApiErrorResponse = {
      schema_version: "dracula-error-v1",
      code: "stale_version",
      message: "state changed",
      retryable: false,
      current_version: 7,
      current_game: authoritative,
    };
    const controller = new GameController(api({
      getGame: vi.fn(async () => ({ status: 200 as const, data: initial })),
      submitMove: vi.fn(async () => { throw new ApiError(409, error); }),
    }), { requestId: () => requestId });
    await controller.loadGame(initial.game_id);
    controller.selectCard(0);
    await controller.playPosition(1);

    expect(controller.getSnapshot().view).toEqual(authoritative);
    expect(controller.getSnapshot().presentation.pending).toBeNull();
    expect(controller.selectCard(1)).toBe(true);
  });

  it("polls one 202 claim with one request ID even when resumed twice", async () => {
    const ready = view({ kind: "opponent_turn", status: "ready", job_id: null, retryable: true });
    const pending = view({
      kind: "opponent_turn",
      status: "pending",
      job_id: requestId,
      retryable: true,
    });
    const completed = view(
      { kind: "human_turn" },
      { version: 1, active_player: "queen", legal_moves: [] },
    );
    const opponentTurn = vi.fn()
      .mockResolvedValueOnce({ status: 202 as const, data: pending })
      .mockResolvedValueOnce({ status: 200 as const, data: completed });
    const controller = new GameController(api({
      getGame: vi.fn(async () => ({ status: 200 as const, data: ready })),
      opponentTurn,
    }), {
      requestId: () => requestId,
      opponentPollDelay: async () => undefined,
    });
    await controller.loadGame(ready.game_id);
    const first = controller.progressOpponent();
    const second = controller.progressOpponent();
    await Promise.all([first, second]);

    expect(opponentTurn).toHaveBeenCalledTimes(2);
    expect(opponentTurn.mock.calls.map((call) => call[1])).toEqual([
      { expected_version: ready.version, request_id: requestId },
      { expected_version: ready.version, request_id: requestId },
    ]);
    expect(controller.getSnapshot().view).toEqual(completed);
  });

  it("uses the public pending job ID when reload resumes an opponent turn", async () => {
    const pending = view({
      kind: "opponent_turn",
      status: "pending",
      job_id: requestId,
      retryable: true,
    });
    const completed = view({ kind: "human_turn" }, { version: 1, active_player: "queen" });
    const opponentTurn = vi.fn(async () => ({ status: 200 as const, data: completed }));
    const controller = new GameController(api({
      getGame: vi.fn(async () => ({ status: 200 as const, data: pending })),
      opponentTurn,
    }), { requestId: () => "new-id" });

    await controller.loadGame(pending.game_id);
    await controller.progressOpponent();
    expect(opponentTurn).toHaveBeenCalledWith(pending.game_id, {
      expected_version: pending.version,
      request_id: requestId,
    });
  });
});

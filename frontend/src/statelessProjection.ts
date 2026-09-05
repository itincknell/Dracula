/**
 * Projects trusted stateless API responses into presentation-facing game views.
 * It also derives deliberate intermediate views used to show a human placement
 * immediately while preserving the server's authoritative completed response.
 */
import type { LegalMove } from "./contractPrimitives";
import type { HumanGameView } from "./gameView";
import type { StatelessGameResponse } from "./statelessContracts";

export function projectStatelessResponse(response: StatelessGameResponse): HumanGameView {
  return {
    ...response.game,
    legal_moves: response.game.legal_moves.map((move) => ({
      ...move,
      move_id: `${move.hand_slot}:${move.position}`,
    })),
  };
}

export function humanPlacementPreview(view: HumanGameView, move: LegalMove): HumanGameView {
  const coffin = [...view.coffin] as HumanGameView["coffin"];
  coffin[move.position] = move.card_id;
  const humanHand = [...view.human_hand] as HumanGameView["human_hand"];
  humanHand[move.hand_slot] = null;
  return {
    ...view,
    turn_number: view.turn_number + 1,
    active_player: view.opponent_role,
    coffin,
    current_round_moves: [
      ...view.current_round_moves,
      {
        player: view.human_role,
        card_id: move.card_id,
        position: move.position,
        turn_number: view.turn_number + 1,
      },
    ],
    phase: {
      kind: "opponent_turn",
      status: "pending",
      job_id: null,
      retryable: true,
    },
    human_hand: humanHand,
    legal_moves: [],
  };
}

export function opponentOpeningPreview(view: HumanGameView): HumanGameView | null {
  const openingMove = view.current_round_moves.at(-1);
  if (
    view.status !== "playing" ||
    view.turn_number !== 1 ||
    view.current_round_moves.length !== 1 ||
    view.phase.kind !== "human_turn" ||
    openingMove === undefined ||
    openingMove.player !== view.opponent_role
  ) {
    return null;
  }
  const coffin = [...view.coffin] as HumanGameView["coffin"];
  coffin[openingMove.position] = null;
  return {
    ...view,
    turn_number: 0,
    active_player: view.opponent_role,
    coffin,
    current_round_moves: [],
    phase: {
      kind: "opponent_turn",
      status: "pending",
      job_id: null,
      retryable: true,
    },
    legal_moves: [],
  };
}

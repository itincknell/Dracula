/**
 * Projects trusted stateless API responses into presentation-facing game views.
 * It also derives deliberate intermediate views used to show a human placement
 * immediately while preserving the server's authoritative completed response.
 */
import type { LegalMove } from "./contractPrimitives";
import type { HumanGameView } from "./gameView";

/**
 * Show the human's chosen card before the server finishes Dracula's response.
 *
 * This creates a new presentation object; it never changes the validated view
 * in place. The server response later replaces this preview in full. If the
 * request fails, the controller restores the original view it kept separately.
 */
export function humanPlacementPreview(view: HumanGameView, move: LegalMove): HumanGameView {
  // Tuple spreads become general arrays in TypeScript, so the assertions state
  // that replacing one existing slot preserves the fixed coffin and hand sizes.
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
    phase: { kind: "opponent_turn" },
    human_hand: humanHand,
    legal_moves: [],
  };
}

/**
 * Hide an already-computed Dracula opener until the minimum move delay passes.
 *
 * A stateless create/advance response may contain Dracula's opening placement
 * because the server settles automatic play before responding. Only that exact
 * one-move state can be rewound for display; all other views return null.
 */
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

  // Remove only the public effects of the opener. The authoritative response
  // remains held by the controller and is published after the delay.
  const coffin = [...view.coffin] as HumanGameView["coffin"];
  coffin[openingMove.position] = null;
  return {
    ...view,
    turn_number: 0,
    active_player: view.opponent_role,
    coffin,
    current_round_moves: [],
    phase: { kind: "opponent_turn" },
    legal_moves: [],
  };
}

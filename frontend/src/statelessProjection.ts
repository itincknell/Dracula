import type { LegalMove, RoundRecord } from "./contractPrimitives";
import type { HumanGameView } from "./statefulContracts";
import type {
  StatelessGameResponse,
  StatelessRoundRecord,
  VisiblePlayedMove,
} from "./statelessContracts";

function playedMove(move: VisiblePlayedMove) {
  return { ...move, hand_slot: -1 };
}

function roundRecord(record: StatelessRoundRecord): RoundRecord {
  return {
    ...record,
    moves: record.moves.map(playedMove),
  };
}

export function projectStatelessResponse(response: StatelessGameResponse): HumanGameView {
  const { game, envelope } = response;
  return {
    schema_version: "dracula-human-game-view-v1",
    game_id: "active",
    version: envelope.history.length - 1,
    status: game.status,
    round_number: game.round_number,
    turn_number: game.turn_number,
    dealer: game.dealer,
    active_player: game.active_player,
    human_role: game.human_role,
    opponent_role: game.opponent_role,
    coffin: game.coffin,
    current_round_moves: game.current_round_moves.map(playedMove),
    pending_round_result: game.pending_round_result === null
      ? null
      : roundRecord(game.pending_round_result),
    completed_rounds: game.completed_rounds.map(roundRecord),
    total_scores: game.total_scores,
    phase: game.phase,
    latest_event_sequence: 0,
    // Stateless narration is coordinated separately from the legacy phase lifecycle.
    narration_enabled: false,
    human_hand: game.human_hand,
    legal_moves: game.legal_moves.map((move) => ({
      ...move,
      move_id: `${move.hand_slot}:${move.position}`,
    })),
    events: [],
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
        hand_slot: -1,
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

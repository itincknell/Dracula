import gameViewFixture from "../../contracts/v1/human-game-view.json";
import type {
  HumanGameView,
  LineScore,
  Player,
  RoundRecord,
  ScoringStep,
} from "./contracts";

const coffin: RoundRecord["coffin"] = [
  "AC", "2D", "3S",
  "4H", "V1", "6C",
  "7H", "8H", "9H",
];

function line(
  direction: "row" | "column",
  index: number,
  cardIds: [string, string, string],
  multiplier: LineScore["multiplier"],
  multiplierReason: LineScore["multiplier_reason"],
  multiplierLabel: string,
  highlightedCardIds: string[],
  total: number,
): LineScore {
  return {
    direction,
    index,
    card_ids: cardIds,
    base_value: 47 + index,
    multiplier,
    multiplier_reason: multiplierReason,
    multiplier_label: multiplierLabel,
    highlighted_card_ids: highlightedCardIds,
    total,
  };
}

const queenLines: [LineScore, LineScore, LineScore] = [
  line("row", 0, ["AC", "2D", "3S"], 1, "none", "No Multiplier", [], 30),
  line("row", 1, ["4H", "V1", "6C"], 0, "vampire", "Vampire", ["V1"], 0),
  line("row", 2, ["7H", "8H", "9H"], 5, "same_suit", "3× Hearts", ["7H", "8H", "9H"], 90),
];

const kingLines: [LineScore, LineScore, LineScore] = [
  line("column", 0, ["AC", "4H", "7H"], 2, "suit_pair", "2× Hearts", ["4H", "7H"], 80),
  line("column", 1, ["2D", "V1", "8H"], 3, "same_color", "3× Red", ["2D", "V1", "8H"], 50),
  line("column", 2, ["3S", "6C", "9H"], 1, "none", "No Multiplier", [], 20),
];

const values: [number, number, number] = [11, 7, 5];

function scoreLineStep(player: Player, value: LineScore, rank: 1 | 2 | 3): ScoringStep {
  return {
    kind: "score_line",
    player,
    line: structuredClone(value),
    details: {
      value_1: values[0],
      value_2: values[1],
      value_3: values[2],
      base_value: value.base_value,
      multiplier: value.multiplier,
      total: value.total,
      rank,
    },
  };
}

function rankedSteps(player: Player, rankedTotals: [number, number, number]): ScoringStep[] {
  const lines = player === "queen" ? queenLines : kingLines;
  const ranks: [1 | 2 | 3, 1 | 2 | 3, 1 | 2 | 3] =
    player === "queen" ? [2, 3, 1] : [1, 2, 3];
  return lines.map((value, index) => {
    const lineRank = ranks[index];
    return scoreLineStep(player, { ...value, total: rankedTotals[lineRank - 1] }, lineRank);
  });
}

export interface ScoringFixtureOptions {
  dealer?: Player;
  humanRole?: Player;
  selectedRank?: 1 | 2 | 3;
  thirdRankTied?: boolean;
  roundNumber?: number;
  narrationEnabled?: boolean;
}

export function scoringRecord(options: ScoringFixtureOptions = {}): RoundRecord {
  const dealer = options.dealer ?? "queen";
  const humanRole = options.humanRole ?? "queen";
  const selectedRank = options.selectedRank ?? 1;
  const first = dealer;
  const second = first === "queen" ? "king" : "queen";
  const queenRanked: [number, number, number] = [90, 30, 0];
  const kingRanked: [number, number, number] = [80, 50, 20];
  const humanRanked = structuredClone(humanRole === "queen" ? queenRanked : kingRanked);
  const opponentRanked = structuredClone(humanRole === "queen" ? kingRanked : queenRanked);
  for (let index = 0; index < selectedRank - 1; index += 1) {
    opponentRanked[index] = humanRanked[index];
  }
  if (selectedRank === 3 && options.thirdRankTied === true) {
    opponentRanked[2] = humanRanked[2];
  }
  const comparisons: ScoringStep[] = [];
  for (let rank = 1; rank <= selectedRank; rank += 1) {
    const tied = rank < selectedRank || (rank === 3 && options.thirdRankTied === true);
    comparisons.push({
      kind: "compare_candidates",
      player: null,
      line: null,
      details: {
        rank,
        human_score: humanRanked[rank - 1],
        opponent_score: opponentRanked[rank - 1],
        tied,
      },
    });
  }
  const selection: ScoringStep = {
    kind: "select_round_score",
    player: null,
    line: null,
    details: {
      rank: selectedRank,
      human_score: humanRanked[selectedRank - 1],
      opponent_score: opponentRanked[selectedRank - 1],
    },
  };
  const totals: ScoringStep = {
    kind: "update_total",
    player: null,
    line: null,
    details: {
      human_previous: 101,
      human_round: humanRanked[selectedRank - 1],
      human_total: 997,
      opponent_previous: 203,
      opponent_round: opponentRanked[selectedRank - 1],
      opponent_total: 509,
    },
  };
  const queenPresentation = humanRole === "queen" ? humanRanked : opponentRanked;
  const kingPresentation = humanRole === "king" ? humanRanked : opponentRanked;
  const lineSteps = [
    ...rankedSteps(first, first === "queen" ? queenPresentation : kingPresentation),
    ...rankedSteps(second, second === "queen" ? queenPresentation : kingPresentation),
  ];
  return {
    round_number: options.roundNumber ?? 1,
    dealer,
    coffin: structuredClone(coffin),
    moves: [],
    line_scores: lineSteps.map((step) => structuredClone(step.line as LineScore)),
    scoring_sequence: [...lineSteps, ...comparisons, selection, totals],
    round_scores: {
      human: humanRanked[selectedRank - 1],
      opponent: opponentRanked[selectedRank - 1],
    },
  };
}

export function scoringView(options: ScoringFixtureOptions = {}): HumanGameView {
  const record = scoringRecord(options);
  const humanRole = options.humanRole ?? "queen";
  return {
    ...(structuredClone(gameViewFixture) as unknown as HumanGameView),
    version: 8,
    status: "round_complete",
    round_number: record.round_number,
    turn_number: 8,
    dealer: record.dealer,
    active_player: null,
    human_role: humanRole,
    opponent_role: humanRole === "queen" ? "king" : "queen",
    coffin: structuredClone(record.coffin),
    human_hand: [null, null, null, null],
    legal_moves: [],
    pending_round_result: record,
    total_scores: { human: 997, opponent: 509 },
    phase: {
      kind: "scoring",
      round_number: record.round_number,
      next_step_index: 0,
      pending_narration_id: null,
    },
    narration_enabled: options.narrationEnabled ?? false,
  };
}

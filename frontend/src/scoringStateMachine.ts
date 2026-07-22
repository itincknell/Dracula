import type { LineScore, Player, RoundRecord, ScoringStep } from "./contracts";

export interface PresentedLine {
  player: Player;
  line: LineScore;
  values: [number, number, number];
  rank: 1 | 2 | 3;
}

export interface PresentedOrientation {
  player: Player;
  lines: [PresentedLine, PresentedLine, PresentedLine];
  rankedTotals: [number, number, number];
}

export interface PresentedComparison {
  rank: 1 | 2 | 3;
  humanScore: number;
  opponentScore: number;
  tied: boolean;
}

export interface PresentedSelection {
  rank: 1 | 2 | 3;
  humanScore: number;
  opponentScore: number;
}

export interface PresentedTotals {
  humanPrevious: number;
  humanRound: number;
  humanTotal: number;
  opponentPrevious: number;
  opponentRound: number;
  opponentTotal: number;
}

export interface ScoringPresentationModel {
  record: RoundRecord;
  humanRole: Player;
  orientations: [PresentedOrientation, PresentedOrientation];
  humanRankedTotals: [number, number, number];
  opponentRankedTotals: [number, number, number];
  comparisons: PresentedComparison[];
  selection: PresentedSelection;
  totals: PresentedTotals;
}

export type ScoringFrame =
  | { kind: "entering" }
  | { kind: "reveal_value"; orientationIndex: 0 | 1; lineIndex: 0 | 1 | 2; cardIndex: 0 | 1 | 2 }
  | { kind: "collapse_sum"; orientationIndex: 0 | 1; lineIndex: 0 | 1 | 2 }
  | { kind: "multiplier_label"; orientationIndex: 0 | 1; lineIndex: 0 | 1 | 2 }
  | { kind: "multiplier_factor"; orientationIndex: 0 | 1; lineIndex: 0 | 1 | 2 }
  | { kind: "line_total"; orientationIndex: 0 | 1; lineIndex: 0 | 1 | 2 }
  | { kind: "orientation_ranked"; orientationIndex: 0 | 1 }
  | { kind: "narrator_wait"; orientationIndex: 0 | 1 }
  | { kind: "orientation_handoff" }
  | { kind: "compare_rank"; comparisonIndex: number }
  | { kind: "select_round_score" }
  | { kind: "update_totals" }
  | { kind: "complete" };

export interface ScoringTiming {
  entering: number;
  revealValue: number;
  collapseSum: number;
  multiplierLabel: number;
  multiplierFactor: number;
  lineTotal: number;
  orientationRanked: number;
  narratorWait: number;
  orientationHandoff: number;
  compareRank: number;
  selectRoundScore: number;
  updateTotals: number;
  reducedStep: number;
}

function reducedStepDuration(): number {
  const configured = Number(import.meta.env.VITE_SCORING_REDUCED_STEP_MS);
  return Number.isFinite(configured) && configured >= 20 ? configured : 160;
}

export const SCORING_TIMING: ScoringTiming = {
  entering: 350,
  revealValue: 280,
  collapseSum: 380,
  multiplierLabel: 420,
  multiplierFactor: 360,
  lineTotal: 460,
  orientationRanked: 650,
  narratorWait: 1_500,
  orientationHandoff: 320,
  compareRank: 520,
  selectRoundScore: 620,
  updateTotals: 720,
  reducedStep: reducedStepDuration(),
};

function detailNumber(step: ScoringStep, name: string): number {
  const value = step.details[name];
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new TypeError(`${step.kind}.${name} must be a finite number`);
  }
  return value;
}

function detailBoolean(step: ScoringStep, name: string): boolean {
  const value = step.details[name];
  if (typeof value !== "boolean") throw new TypeError(`${step.kind}.${name} must be a boolean`);
  return value;
}

function rank(step: ScoringStep): 1 | 2 | 3 {
  const value = detailNumber(step, "rank");
  if (value !== 1 && value !== 2 && value !== 3) {
    throw new TypeError(`${step.kind}.rank must be 1, 2, or 3`);
  }
  return value;
}

function presentedLine(step: ScoringStep): PresentedLine {
  if (step.kind !== "score_line" || step.player === null || step.line === null) {
    throw new TypeError("score_line must identify a player and line");
  }
  const lineRank = rank(step);
  if (
    detailNumber(step, "base_value") !== step.line.base_value ||
    detailNumber(step, "multiplier") !== step.line.multiplier ||
    detailNumber(step, "total") !== step.line.total
  ) {
    throw new TypeError("score_line details disagree with LineScore");
  }
  return {
    player: step.player,
    line: step.line,
    values: [
      detailNumber(step, "value_1"),
      detailNumber(step, "value_2"),
      detailNumber(step, "value_3"),
    ],
    rank: lineRank,
  };
}

function rankedTotals(lines: PresentedLine[]): [number, number, number] {
  const totals: Array<number | undefined> = [undefined, undefined, undefined];
  for (const item of lines) {
    if (totals[item.rank - 1] !== undefined) throw new TypeError("line ranks must be unique");
    totals[item.rank - 1] = item.line.total;
  }
  if (totals.some((value) => value === undefined)) throw new TypeError("line ranks must cover 1 through 3");
  return totals as [number, number, number];
}

export function createScoringModel(
  record: RoundRecord,
  humanRole: Player,
): ScoringPresentationModel {
  const lineSteps = record.scoring_sequence.filter((step) => step.kind === "score_line");
  if (lineSteps.length !== 6) throw new TypeError("scoring sequence must contain six line steps");
  const firstLines = lineSteps.slice(0, 3).map(presentedLine);
  const secondLines = lineSteps.slice(3, 6).map(presentedLine);
  if (
    firstLines.some((line) => line.player !== record.dealer) ||
    secondLines.some((line) => line.player === record.dealer) ||
    secondLines.some((line) => line.player !== secondLines[0]?.player)
  ) {
    throw new TypeError("line steps must score dealer then non-dealer");
  }
  const orientations: [PresentedOrientation, PresentedOrientation] = [
    {
      player: firstLines[0].player,
      lines: firstLines as [PresentedLine, PresentedLine, PresentedLine],
      rankedTotals: rankedTotals(firstLines),
    },
    {
      player: secondLines[0].player,
      lines: secondLines as [PresentedLine, PresentedLine, PresentedLine],
      rankedTotals: rankedTotals(secondLines),
    },
  ];

  const comparisonSteps = record.scoring_sequence.filter(
    (step) => step.kind === "compare_candidates",
  );
  if (comparisonSteps.length < 1 || comparisonSteps.length > 3) {
    throw new TypeError("scoring sequence must compare between one and three ranks");
  }
  const comparisons = comparisonSteps.map((step, index): PresentedComparison => {
    const comparisonRank = rank(step);
    if (comparisonRank !== index + 1) throw new TypeError("comparison ranks must be sequential");
    return {
      rank: comparisonRank,
      humanScore: detailNumber(step, "human_score"),
      opponentScore: detailNumber(step, "opponent_score"),
      tied: detailBoolean(step, "tied"),
    };
  });

  const selectionStep = record.scoring_sequence.find((step) => step.kind === "select_round_score");
  const totalsStep = record.scoring_sequence.find((step) => step.kind === "update_total");
  if (selectionStep === undefined || totalsStep === undefined) {
    throw new TypeError("scoring sequence must select a score and update totals");
  }
  const selection: PresentedSelection = {
    rank: rank(selectionStep),
    humanScore: detailNumber(selectionStep, "human_score"),
    opponentScore: detailNumber(selectionStep, "opponent_score"),
  };
  const totals: PresentedTotals = {
    humanPrevious: detailNumber(totalsStep, "human_previous"),
    humanRound: detailNumber(totalsStep, "human_round"),
    humanTotal: detailNumber(totalsStep, "human_total"),
    opponentPrevious: detailNumber(totalsStep, "opponent_previous"),
    opponentRound: detailNumber(totalsStep, "opponent_round"),
    opponentTotal: detailNumber(totalsStep, "opponent_total"),
  };

  const humanOrientation = orientations.find((orientation) => orientation.player === humanRole);
  const opponentOrientation = orientations.find((orientation) => orientation.player !== humanRole);
  if (humanOrientation === undefined || opponentOrientation === undefined) {
    throw new TypeError("scoring sequence must contain both players");
  }
  return {
    record,
    humanRole,
    orientations,
    humanRankedTotals: humanOrientation.rankedTotals,
    opponentRankedTotals: opponentOrientation.rankedTotals,
    comparisons,
    selection,
    totals,
  };
}

export function createScoringTimeline(
  model: ScoringPresentationModel,
  narrationEnabled: boolean,
): ScoringFrame[] {
  const frames: ScoringFrame[] = [{ kind: "entering" }];
  model.orientations.forEach((orientation, orientationIndexValue) => {
    const orientationIndex = orientationIndexValue as 0 | 1;
    orientation.lines.forEach((_line, lineIndexValue) => {
      const lineIndex = lineIndexValue as 0 | 1 | 2;
      for (const cardIndex of [0, 1, 2] as const) {
        frames.push({ kind: "reveal_value", orientationIndex, lineIndex, cardIndex });
      }
      frames.push(
        { kind: "collapse_sum", orientationIndex, lineIndex },
        { kind: "multiplier_label", orientationIndex, lineIndex },
        { kind: "multiplier_factor", orientationIndex, lineIndex },
        { kind: "line_total", orientationIndex, lineIndex },
      );
    });
    frames.push({ kind: "orientation_ranked", orientationIndex });
    if (narrationEnabled) frames.push({ kind: "narrator_wait", orientationIndex });
    if (orientationIndex === 0) frames.push({ kind: "orientation_handoff" });
  });
  model.comparisons.forEach((_comparison, comparisonIndex) => {
    frames.push({ kind: "compare_rank", comparisonIndex });
  });
  frames.push(
    { kind: "select_round_score" },
    { kind: "update_totals" },
    { kind: "complete" },
  );
  return frames;
}

export function scoringFrameDuration(
  frame: ScoringFrame,
  reducedMotion: boolean,
  timing: ScoringTiming = SCORING_TIMING,
): number | null {
  if (frame.kind === "complete") return null;
  if (reducedMotion) return timing.reducedStep;
  const durations: Record<Exclude<ScoringFrame["kind"], "complete">, number> = {
    entering: timing.entering,
    reveal_value: timing.revealValue,
    collapse_sum: timing.collapseSum,
    multiplier_label: timing.multiplierLabel,
    multiplier_factor: timing.multiplierFactor,
    line_total: timing.lineTotal,
    orientation_ranked: timing.orientationRanked,
    narrator_wait: timing.narratorWait,
    orientation_handoff: timing.orientationHandoff,
    compare_rank: timing.compareRank,
    select_round_score: timing.selectRoundScore,
    update_totals: timing.updateTotals,
  };
  return durations[frame.kind];
}

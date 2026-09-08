/**
 * Builds the deterministic presentation timeline for completed-round scoring.
 * Pure functions derive line calculations, transitions, totals, and display
 * durations without reading React state or changing authoritative game data.
 */
import type { LineScore, Player, RoundRecord, ScoringStep } from "./contractPrimitives";

export interface PresentedLine {
  /** Role that owns this row or column. */
  player: Player;
  /** Server-supplied cards, multiplier, and total for the line. */
  line: LineScore;
  /** Individual card values shown before the sum collapses. */
  values: [number, number, number];
  /** Position of this total after the player's three lines are sorted high to low. */
  rank: 1 | 2 | 3;
}

/** All three rows or columns scored by one role. */
export interface PresentedOrientation {
  player: Player;
  lines: [PresentedLine, PresentedLine, PresentedLine];
  /** Line totals in tie-break order, highest first. */
  rankedTotals: [number, number, number];
}

/** One comparison between equally ranked human and Dracula line totals. */
export interface PresentedComparison {
  rank: 1 | 2 | 3;
  humanScore: number;
  opponentScore: number;
  tied: boolean;
}

/** The first non-tied ranked totals selected as the official round scores. */
export interface PresentedSelection {
  rank: 1 | 2 | 3;
  humanScore: number;
  opponentScore: number;
}

/** Arithmetic shown when the round scores are added to the running game totals. */
export interface PresentedTotals {
  humanPrevious: number;
  humanRound: number;
  humanTotal: number;
  opponentPrevious: number;
  opponentRound: number;
  opponentTotal: number;
}

/**
 * Validated, presentation-ready interpretation of a completed round.
 * Components use this model rather than repeatedly looking up loosely typed
 * values in the server's scoring-step `details` objects.
 */
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

/**
 * One stable screen in the scoring animation.
 *
 * Frames that operate on a line identify its player-orientation index and line
 * index. Value revelation additionally identifies one of that line's cards.
 * Later frames compare ranked totals, select the round result, and update the
 * game totals. `complete` has no timer and remains until the user continues.
 */
export type ScoringFrame =
  | { kind: "entering" }
  | {
      kind: "reveal_value";
      orientationIndex: 0 | 1;
      lineIndex: 0 | 1 | 2;
      cardIndex: 0 | 1 | 2;
    }
  | { kind: "collapse_sum"; orientationIndex: 0 | 1; lineIndex: 0 | 1 | 2 }
  | { kind: "multiplier_label"; orientationIndex: 0 | 1; lineIndex: 0 | 1 | 2 }
  | { kind: "multiplier_factor"; orientationIndex: 0 | 1; lineIndex: 0 | 1 | 2 }
  | { kind: "line_total"; orientationIndex: 0 | 1; lineIndex: 0 | 1 | 2 }
  | { kind: "orientation_ranked"; orientationIndex: 0 | 1 }
  | { kind: "orientation_handoff" }
  | { kind: "compare_rank"; comparisonIndex: number }
  | { kind: "select_round_score" }
  | { kind: "update_totals" }
  | { kind: "complete" };

/** Milliseconds assigned to each kind of timed scoring frame. */
export interface ScoringTiming {
  entering: number;
  revealValue: number;
  collapseSum: number;
  multiplierLabel: number;
  multiplierFactor: number;
  lineTotal: number;
  orientationRanked: number;
  orientationHandoff: number;
  compareRank: number;
  selectRoundScore: number;
  updateTotals: number;
  reducedStep: number;
}

/**
 * Resolve the short frame duration used for reduced motion.
 * The environment override exists so browser tests can traverse the complete
 * sequence quickly; ordinary builds use 160 milliseconds.
 */
function reducedStepDuration(): number {
  const configured = Number(import.meta.env.VITE_SCORING_REDUCED_STEP_MS);
  return Number.isFinite(configured) && configured >= 20 ? configured : 160;
}

export const SCORING_PLAYBACK_RATE = 0.75;

/** Convert the original durations to the approved slower 75% playback speed. */
function atPlaybackRate(milliseconds: number): number {
  return Math.round(milliseconds / SCORING_PLAYBACK_RATE);
}

export const SCORING_TIMING: ScoringTiming = {
  entering: atPlaybackRate(1_200),
  revealValue: atPlaybackRate(280),
  collapseSum: atPlaybackRate(380),
  multiplierLabel: atPlaybackRate(420),
  multiplierFactor: atPlaybackRate(360),
  lineTotal: atPlaybackRate(460),
  orientationRanked: atPlaybackRate(650),
  orientationHandoff: atPlaybackRate(1_200),
  compareRank: atPlaybackRate(700),
  selectRoundScore: atPlaybackRate(800),
  updateTotals: atPlaybackRate(720),
  reducedStep: reducedStepDuration(),
};

/** Read one required numeric value from a kind-specific scoring-step detail map. */
function detailNumber(step: ScoringStep, name: string): number {
  const value = step.details[name];
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new TypeError(`${step.kind}.${name} must be a finite number`);
  }
  return value;
}

/** Read one required Boolean value from a kind-specific scoring-step detail map. */
function detailBoolean(step: ScoringStep, name: string): boolean {
  const value = step.details[name];
  if (typeof value !== "boolean") throw new TypeError(`${step.kind}.${name} must be a boolean`);
  return value;
}

/** Validate a server-supplied one-based line rank and narrow its TypeScript type. */
function rank(step: ScoringStep): 1 | 2 | 3 {
  const value = detailNumber(step, "rank");
  if (value !== 1 && value !== 2 && value !== 3) {
    throw new TypeError(`${step.kind}.rank must be 1, 2, or 3`);
  }
  return value;
}

/** Convert a generic `score_line` step into the fields needed by the renderer. */
function presentedLine(step: ScoringStep): PresentedLine {
  if (step.kind !== "score_line" || step.player === null || step.line === null) {
    throw new TypeError("score_line must identify a player and line");
  }
  const lineRank = rank(step);
  // The wire format repeats this arithmetic in `line` and `details` because
  // the latter drives the animation. Reject disagreement once here so the
  // visual sequence cannot show two answers for the same line.
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

/** Place three line totals into their explicit server-supplied rank order. */
function rankedTotals(lines: PresentedLine[]): [number, number, number] {
  const totals: Array<number | undefined> = [undefined, undefined, undefined];
  for (const item of lines) {
    if (totals[item.rank - 1] !== undefined) throw new TypeError("line ranks must be unique");
    totals[item.rank - 1] = item.line.total;
  }
  if (totals.some((value) => value === undefined)) throw new TypeError("line ranks must cover 1 through 3");
  return totals as [number, number, number];
}

/** Require one orientation to contain exactly three lines for the same role. */
function presentedOrientation(
  lines: PresentedLine[],
  expectedPlayer: Player,
): PresentedOrientation {
  if (lines.length !== 3 || lines.some((line) => line.player !== expectedPlayer)) {
    throw new TypeError("each scoring orientation must contain three lines for one player");
  }
  return {
    player: expectedPlayer,
    lines: lines as [PresentedLine, PresentedLine, PresentedLine],
    rankedTotals: rankedTotals(lines),
  };
}

/**
 * Split the six line steps into dealer-first and non-dealer orientations.
 * The ordering is a deliberate presentation choice supplied by the server.
 */
function presentedOrientations(record: RoundRecord): [PresentedOrientation, PresentedOrientation] {
  const lineSteps = record.scoring_sequence.filter((step) => step.kind === "score_line");
  if (lineSteps.length !== 6) throw new TypeError("scoring sequence must contain six line steps");
  const firstLines = lineSteps.slice(0, 3).map(presentedLine);
  const secondLines = lineSteps.slice(3, 6).map(presentedLine);
  const secondPlayer = secondLines[0]?.player;
  if (secondPlayer === undefined || secondPlayer === record.dealer) {
    throw new TypeError("line steps must score dealer then non-dealer");
  }
  return [
    presentedOrientation(firstLines, record.dealer),
    presentedOrientation(secondLines, secondPlayer),
  ];
}

/** Read the consecutive rank comparisons that decide the official round score. */
function presentedComparisons(record: RoundRecord): PresentedComparison[] {
  const comparisonSteps = record.scoring_sequence.filter(
    (step) => step.kind === "compare_candidates",
  );
  if (comparisonSteps.length < 1 || comparisonSteps.length > 3) {
    throw new TypeError("scoring sequence must compare between one and three ranks");
  }
  return comparisonSteps.map((step, index): PresentedComparison => {
    const comparisonRank = rank(step);
    if (comparisonRank !== index + 1) throw new TypeError("comparison ranks must be sequential");
    return {
      rank: comparisonRank,
      humanScore: detailNumber(step, "human_score"),
      opponentScore: detailNumber(step, "opponent_score"),
      tied: detailBoolean(step, "tied"),
    };
  });
}

/** Read the selected round scores and their addition to prior game totals. */
function presentedOutcome(record: RoundRecord): {
  selection: PresentedSelection;
  totals: PresentedTotals;
} {
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
  return { selection, totals };
}

/** Validate one server scoring script and prepare the renderer's lookup model. */
export function createScoringModel(
  record: RoundRecord,
  humanRole: Player,
): ScoringPresentationModel {
  // The server supplies the animation script. These builders verify its
  // cross-step relationships once, then components can render a trusted model.
  const orientations = presentedOrientations(record);
  const comparisons = presentedComparisons(record);
  const { selection, totals } = presentedOutcome(record);

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

/**
 * Expand the scoring model into every screen shown from entry to completion.
 * Each player's three lines reveal three values, then their sum, multiplier,
 * factor, and total. The player totals are ranked before the other role begins.
 */
export function createScoringTimeline(model: ScoringPresentationModel): ScoringFrame[] {
  const frames: ScoringFrame[] = [{ kind: "entering" }];
  model.orientations.forEach((orientation, orientationIndexValue) => {
    // Array iteration produces a general number. These arrays were validated
    // above as exactly two orientations and exactly three lines per role.
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
    // The first player's score lingers before the second player begins.
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

/** Return one frame's display time, or null for the final untimed frame. */
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
    orientation_handoff: timing.orientationHandoff,
    compare_rank: timing.compareRank,
    select_round_score: timing.selectRoundScore,
    update_totals: timing.updateTotals,
  };
  return durations[frame.kind];
}

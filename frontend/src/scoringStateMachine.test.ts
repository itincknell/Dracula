import { describe, expect, it } from "vitest";

import {
  SCORING_TIMING,
  createScoringModel,
  createScoringTimeline,
  scoringFrameDuration,
} from "./scoringStateMachine";
import { scoringRecord } from "./scoringTestFixtures";

describe("deterministic scoring state machine", () => {
  it.each([
    ["queen", "row", "column"],
    ["king", "column", "row"],
  ] as const)("scores a %s dealer before the other orientation", (dealer, firstDirection, secondDirection) => {
    const model = createScoringModel(scoringRecord({ dealer }), "queen");
    expect(model.orientations[0].player).toBe(dealer);
    expect(model.orientations[0].lines.map((item) => item.line.direction)).toEqual([
      firstDirection,
      firstDirection,
      firstDirection,
    ]);
    expect(model.orientations[1].lines.map((item) => item.line.direction)).toEqual([
      secondDirection,
      secondDirection,
      secondDirection,
    ]);
  });

  it("carries every supplied multiplier label and highlight set without interpreting cards", () => {
    const model = createScoringModel(scoringRecord(), "queen");
    const lines = model.orientations.flatMap((orientation) => orientation.lines);
    expect(lines.map((item) => item.line.multiplier)).toEqual([1, 0, 5, 2, 3, 1]);
    expect(lines.map((item) => item.line.multiplier_label)).toEqual([
      "No Multiplier",
      "Vampire",
      "3× Hearts",
      "2× Hearts",
      "3× Red",
      "No Multiplier",
    ]);
    expect(lines[1].line.highlighted_card_ids).toEqual(["V1"]);
    expect(lines[2].line.highlighted_card_ids).toEqual(["7H", "8H", "9H"]);
    expect(lines[3].line.highlighted_card_ids).toEqual(["4H", "7H"]);
  });

  it.each([
    [1, [false]],
    [2, [true, false]],
    [3, [true, true, false]],
    [3, [true, true, true]],
  ] as const)("uses the supplied rank-%s resolution and tie history", (selectedRank, ties) => {
    const model = createScoringModel(
      scoringRecord({ selectedRank, thirdRankTied: ties[2] === true }),
      "queen",
    );
    expect(model.comparisons.map((comparison) => comparison.tied)).toEqual(ties);
    expect(model.selection.rank).toBe(selectedRank);
  });

  it("bypasses both narrator waits when narration is disabled", () => {
    const model = createScoringModel(scoringRecord(), "queen");
    const disabled = createScoringTimeline(model, false);
    const enabled = createScoringTimeline(model, true);
    expect(disabled.some((frame) => frame.kind === "narrator_wait")).toBe(false);
    expect(enabled.filter((frame) => frame.kind === "narrator_wait")).toHaveLength(2);
  });

  it("uses the same ordered information states with shorter reduced-motion timing", () => {
    const timeline = createScoringTimeline(createScoringModel(scoringRecord(), "queen"), false);
    const ordinaryKinds = timeline.map((frame) => frame.kind);
    const reducedKinds = [...timeline].map((frame) => frame.kind);
    expect(reducedKinds).toEqual(ordinaryKinds);
    timeline.slice(0, -1).forEach((frame) => {
      expect(scoringFrameDuration(frame, true)).toBe(SCORING_TIMING.reducedStep);
      expect(scoringFrameDuration(frame, true)).toBeLessThanOrEqual(
        scoringFrameDuration(frame, false) ?? Number.POSITIVE_INFINITY,
      );
    });
  });

  // Deliberately inconsistent arithmetic proves that presentation consumes wire values.
  it("never recalculates line sums, line totals, or cumulative totals", () => {
    const model = createScoringModel(scoringRecord(), "queen");
    const first = model.orientations[0].lines[0];
    expect(first.values).toEqual([11, 7, 5]);
    expect(first.line.base_value).toBe(47);
    expect(first.line.total).toBe(30);
    expect(model.totals.humanPrevious).toBe(101);
    expect(model.totals.humanRound).toBe(90);
    expect(model.totals.humanTotal).toBe(997);
  });
});

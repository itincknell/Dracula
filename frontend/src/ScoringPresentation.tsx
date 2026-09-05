/**
 * Renders and advances the completed-round scoring presentation.
 * It consumes the pure scoring timeline, coordinates animation frames and
 * narration reveal, and exposes round or game continuation controls.
 */
import { useEffect, useMemo, useRef, useState } from "react";

import type { RoundRecord } from "./contractPrimitives";
import type { HumanGameView } from "./gameView";
import { type GameControllerContract, useGameStore } from "./gameControllerContract";
import {
  PresentationWorkspace,
  RoundTotals,
  ScoringCoffin,
} from "./ScoringWorkspaces";
import {
  SCORING_TIMING,
  createScoringModel,
  createScoringTimeline,
  scoringFrameDuration,
  type ScoringTiming,
} from "./scoringStateMachine";

export { RoundTotals } from "./ScoringWorkspaces";

function useReducedMotion(forced: boolean | undefined): boolean {
  const [reduced, setReduced] = useState(() =>
    forced ?? (typeof window.matchMedia === "function" && window.matchMedia("(prefers-reduced-motion: reduce)").matches),
  );
  useEffect(() => {
    if (forced !== undefined || typeof window.matchMedia !== "function") {
      setReduced(forced ?? false);
      return undefined;
    }
    const media = window.matchMedia("(prefers-reduced-motion: reduce)");
    const update = () => setReduced(media.matches);
    media.addEventListener("change", update);
    update();
    return () => media.removeEventListener("change", update);
  }, [forced]);
  return reduced;
}

export interface ScoringPresentationProps {
  controller: GameControllerContract;
  view: HumanGameView;
  reducedMotion?: boolean;
  timing?: ScoringTiming;
}

export function ScoringPresentation({
  controller,
  view,
  reducedMotion: forcedReducedMotion,
  timing = SCORING_TIMING,
}: ScoringPresentationProps) {
  const { presentation } = useGameStore(controller);
  const record = view.pending_round_result;
  if (record === null) throw new TypeError("scoring phase requires pending_round_result");
  const model = useMemo(
    () => createScoringModel(record, view.human_role),
    [record, view.human_role],
  );
  const timeline = useMemo(() => createScoringTimeline(model), [model]);
  const reducedMotion = useReducedMotion(forcedReducedMotion);
  const [frameIndex, setFrameIndex] = useState(0);
  const roundKey = record.round_number;
  const finalizeRequested = useRef<number | null>(null);

  useEffect(() => {
    setFrameIndex(0);
    finalizeRequested.current = null;
  }, [roundKey]);
  const frame = timeline[frameIndex] ?? timeline[timeline.length - 1];

  useEffect(() => {
    const duration = scoringFrameDuration(frame, reducedMotion, timing);
    if (duration === null) return undefined;
    const timer = window.setTimeout(() => {
      setFrameIndex((index) => Math.min(index + 1, timeline.length - 1));
    }, duration);
    return () => window.clearTimeout(timer);
  }, [frame, reducedMotion, timeline.length, timing]);

  useEffect(() => {
    if (frame.kind === "complete") controller.revealNarration();
  }, [controller, frame.kind]);

  useEffect(() => {
    if (
      frame.kind !== "complete" ||
      record.round_number !== 6 ||
      presentation.pending !== null ||
      finalizeRequested.current === roundKey
    ) {
      return;
    }
    finalizeRequested.current = roundKey;
    void controller.advanceRound().then((accepted) => {
      if (!accepted) finalizeRequested.current = null;
    });
  }, [controller, frame.kind, presentation.pending, record.round_number, roundKey]);

  const showCoffin =
    frame.kind !== "compare_rank" &&
    frame.kind !== "select_round_score" &&
    frame.kind !== "update_totals" &&
    frame.kind !== "complete";
  return (
    <section
      className={`scoring-presentation stage-${frame.kind}`}
      data-stage={frame.kind}
      data-reduced-motion={reducedMotion}
      aria-label={`Round ${record.round_number} scoring presentation`}
    >
      {showCoffin ? <ScoringCoffin model={model} frame={frame} reducedMotion={reducedMotion} /> : null}
      <PresentationWorkspace model={model} frame={frame} />
      {frame.kind === "complete" && record.round_number < 6 ? (
        <button
          className="primary-action scoring-advance"
          type="button"
          disabled={presentation.pending !== null}
          onClick={() => void controller.advanceRound()}
        >
          Deal Next Round
        </button>
      ) : null}
      {frame.kind === "complete" && record.round_number === 6 ? (
        <p className="presentation-status">Finalizing game…</p>
      ) : null}
    </section>
  );
}

export function FinalRoundPresentation({
  record,
  view,
  onNewGame,
}: {
  record: RoundRecord;
  view: HumanGameView;
  onNewGame: () => void;
}) {
  const model = createScoringModel(record, view.human_role);
  const outcome = view.phase.kind === "game_complete"
    ? { human: "You won", opponent: "Dracula won", tie: "The game is tied" }[view.phase.outcome]
    : "Game complete";
  return (
    <section className="scoring-presentation final-round" aria-label="Final game result">
      <RoundTotals model={model} />
      <div className="final-outcome">
        <h2>{outcome}</h2>
        <button className="primary-action" type="button" onClick={onNewGame}>Play Again</button>
      </div>
    </section>
  );
}

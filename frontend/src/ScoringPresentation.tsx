/**
 * Renders and advances the completed-round scoring presentation.
 * It consumes the pure scoring timeline, coordinates animation frames and
 * narration reveal, and exposes round or game continuation controls.
 */
import { useEffect, useMemo, useRef, useState } from "react";

import type { RoundRecord } from "./contractPrimitives";
import type { HumanGameView } from "./gameView";
import {
  type GameControllerContract,
  type PendingAction,
  useGameStore,
} from "./gameControllerContract";
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
  type ScoringFrame,
  type ScoringTiming,
} from "./scoringStateMachine";

/** Track the system reduced-motion setting unless a test supplies an override. */
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
    // The listener keeps an in-progress or later scoring sequence aligned with
    // an accessibility preference changed while the application is open.
    const update = () => setReduced(media.matches);
    media.addEventListener("change", update);
    update();
    return () => media.removeEventListener("change", update);
  }, [forced]);
  return reduced;
}

/** Inputs for the timed scoring presentation; timing overrides are test-only. */
interface ScoringPresentationProps {
  controller: GameControllerContract;
  view: HumanGameView;
  reducedMotion?: boolean;
  timing?: ScoringTiming;
}

/** Advance through a round's immutable timeline using one timer per frame. */
function useScoringFrame(
  timeline: ReturnType<typeof createScoringTimeline>,
  roundKey: number,
  reducedMotion: boolean,
  timing: ScoringTiming,
): ScoringFrame {
  const [frameIndex, setFrameIndex] = useState(0);
  // Reloading or entering another round always restarts its visual sequence;
  // frame progress is presentation state and is never persisted.
  useEffect(() => {
    setFrameIndex(0);
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
  return frame;
}

/**
 * Convert round six from scored to game-complete after its animation.
 * Earlier rounds expose a button, but the sixth transition is automatic so
 * the final outcome and final narration can replace the scoring screen.
 */
function useFinalRoundCompletion(
  controller: GameControllerContract,
  frameKind: ScoringFrame["kind"],
  roundNumber: number,
  pending: PendingAction | null,
): void {
  const requestedRound = useRef<number | null>(null);
  useEffect(() => {
    requestedRound.current = null;
  }, [roundNumber]);
  // Round six still needs the authoritative advance command, but has no button.
  // The ref prevents React effects or pending-state changes from sending it twice.
  useEffect(() => {
    if (
      frameKind !== "complete" || roundNumber !== 6 || pending !== null ||
      requestedRound.current === roundNumber
    ) return;
    requestedRound.current = roundNumber;
    void controller.advanceRound().then((accepted) => {
      if (!accepted) requestedRound.current = null;
    });
  }, [controller, frameKind, pending, roundNumber]);
}

/** Hide the card grid once the presentation begins comparing player totals. */
function scoringCoffinIsVisible(frameKind: ScoringFrame["kind"]): boolean {
  return !["compare_rank", "select_round_score", "update_totals", "complete"]
    .includes(frameKind);
}

/** Render and time all stages of the currently pending round result. */
export function ScoringPresentation({
  controller,
  view,
  reducedMotion: forcedReducedMotion,
  timing = SCORING_TIMING,
}: ScoringPresentationProps) {
  const { presentation } = useGameStore(controller);
  const record = view.pending_round_result;
  // The gameplay phase and pending result are produced together by the server.
  // Throwing exposes a broken trusted contract instead of rendering stale data.
  if (record === null) throw new TypeError("scoring phase requires pending_round_result");
  const model = useMemo(
    () => createScoringModel(record, view.human_role),
    [record, view.human_role],
  );
  const timeline = useMemo(() => createScoringTimeline(model), [model]);
  const reducedMotion = useReducedMotion(forcedReducedMotion);
  const frame = useScoringFrame(timeline, record.round_number, reducedMotion, timing);

  useEffect(() => {
    // Round-transition text was fetched before animation started and may be
    // committed only after the complete frame is visible.
    if (frame.kind === "complete") controller.revealNarration();
  }, [controller, frame.kind]);
  useFinalRoundCompletion(
    controller,
    frame.kind,
    record.round_number,
    presentation.pending,
  );
  const showCoffin = scoringCoffinIsVisible(frame.kind);
  // Keep round six's completed totals in place during the final HTTP request.
  // Earlier rounds switch layout to make room for their Deal Next Round button.
  const displayStage = record.round_number === 6 && frame.kind === "complete"
    ? "update_totals"
    : frame.kind;
  return (
    <section
      className={`scoring-presentation stage-${displayStage}`}
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
    </section>
  );
}

/** Render the final round totals and game winner after round-six advancement. */
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
  // A normal call supplies game_complete. The fallback keeps this purely
  // presentational component readable if a test or future caller misuses it.
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

import { useEffect, useMemo, useRef, useState, type CSSProperties, type ReactNode } from "react";

import { CardFace } from "./CardFace";
import type { HumanGameView, Player, RoundRecord } from "./contracts";
import { GameController, useGameStore } from "./gameStore";
import {
  SCORING_TIMING,
  createScoringModel,
  createScoringTimeline,
  scoringFrameDuration,
  type PresentedLine,
  type ScoringFrame,
  type ScoringPresentationModel,
  type ScoringTiming,
} from "./scoringStateMachine";

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

function playerHeading(player: Player, humanRole: Player): string {
  return player === humanRole ? "Your Score" : "Dracula's Score";
}

function activeLine(
  model: ScoringPresentationModel,
  frame: ScoringFrame,
): PresentedLine | null {
  if (!("orientationIndex" in frame) || !("lineIndex" in frame)) return null;
  return model.orientations[frame.orientationIndex].lines[frame.lineIndex];
}

function ScoringCoffin({
  model,
  frame,
  reducedMotion,
}: {
  model: ScoringPresentationModel;
  frame: ScoringFrame;
  reducedMotion: boolean;
}) {
  const line = activeLine(model, frame);
  const popCard = frame.kind === "reveal_value" ? line?.line.card_ids[frame.cardIndex] : null;
  const ripple = frame.kind === "collapse_sum";
  const showEmphasis =
    frame.kind === "multiplier_label" ||
    frame.kind === "multiplier_factor" ||
    frame.kind === "line_total";
  return (
    <section
      className="scoring-coffin"
      aria-label="Completed coffin"
      data-reduced-motion={reducedMotion}
    >
      {model.record.coffin.map((cardId, position) => {
        const inLine = line?.line.card_ids.includes(cardId) ?? false;
        const emphasized = showEmphasis && (line?.line.highlighted_card_ids.includes(cardId) ?? false);
        const popping = popCard === cardId || (frame.kind === "line_total" && inLine);
        return (
          <div
            className={[
              "scoring-card",
              inLine ? "active-series" : "",
              emphasized ? "multiplier-card" : "",
              popping ? "score-pop" : "",
              ripple && inLine ? "score-ripple" : "",
            ].filter(Boolean).join(" ")}
            key={cardId}
            data-card-id={cardId}
            data-position={position}
            style={{ "--ripple-order": line?.line.card_ids.indexOf(cardId) ?? 0 } as CSSProperties}
          >
            <CardFace cardId={cardId} />
          </div>
        );
      })}
    </section>
  );
}

function valueText(item: PresentedLine, cardIndex: number): string {
  const cardId = item.line.card_ids[cardIndex];
  if (
    item.line.multiplier_reason === "vampire" &&
    item.line.highlighted_card_ids.includes(cardId)
  ) {
    return "—";
  }
  return String(item.values[cardIndex]);
}

function LineWorkspace({
  model,
  frame,
}: {
  model: ScoringPresentationModel;
  frame: Extract<
    ScoringFrame,
    { kind: "reveal_value" | "collapse_sum" | "multiplier_label" | "multiplier_factor" | "line_total" }
  >;
}) {
  const orientation = model.orientations[frame.orientationIndex];
  const item = orientation.lines[frame.lineIndex];
  let expression: ReactNode;
  if (frame.kind === "reveal_value") {
    expression = item.values.slice(0, frame.cardIndex + 1).map((_value, index) => valueText(item, index)).join(" + ");
  } else if (frame.kind === "collapse_sum") {
    expression = item.line.base_value;
  } else if (frame.kind === "multiplier_label") {
    expression = <>{item.line.base_value} <span className="multiplier-description">{item.line.multiplier_label}</span></>;
  } else if (frame.kind === "multiplier_factor") {
    expression = <>{item.line.base_value} × {item.line.multiplier}</>;
  } else {
    expression = item.line.total;
  }
  const completed = orientation.lines.slice(0, frame.lineIndex).map((line) => line.line.total);
  if (frame.kind === "line_total") completed.push(item.line.total);
  return (
    <section className="calculation-workspace" aria-live="polite">
      <p className="score-heading">{playerHeading(orientation.player, model.humanRole)}</p>
      <p className="series-label">{item.line.direction === "row" ? "Row" : "Column"} {item.line.index + 1}</p>
      <div className={`score-expression ${frame.kind}`} data-stage={frame.kind}>{expression}</div>
      <div className="completed-line-totals" aria-label="Completed line totals">
        {completed.map((total, index) => <span key={`${index}-${total}`}>{total}</span>)}
      </div>
    </section>
  );
}

function RankedOrientation({
  model,
  orientationIndex,
}: {
  model: ScoringPresentationModel;
  orientationIndex: 0 | 1;
}) {
  const orientation = model.orientations[orientationIndex];
  return (
    <section className="calculation-workspace ranked-orientation" aria-live="polite">
      <p className="score-heading">{playerHeading(orientation.player, model.humanRole)}</p>
      <div className="ranked-line-totals" aria-label="Ranked line totals">
        {orientation.rankedTotals.map((total, index) => (
          <span key={index} data-rank={index + 1}>{total}</span>
        ))}
      </div>
    </section>
  );
}

function ComparisonWorkspace({
  model,
  comparisonIndex,
}: {
  model: ScoringPresentationModel;
  comparisonIndex: number;
}) {
  return (
    <section className="round-comparison" aria-live="polite">
      <div className="comparison-column">
        <h2>Your Score</h2>
        {model.humanRankedTotals.map((total, index) => {
          const comparison = model.comparisons[index];
          const reached = index <= comparisonIndex;
          const tied = reached && comparison?.tied === true;
          return <span className={tied ? "tied-score" : reached ? "active-score" : "pending-score"} key={index}>{total}{tied ? <small>Tie</small> : null}</span>;
        })}
      </div>
      <div className="comparison-column">
        <h2>Dracula's Score</h2>
        {model.opponentRankedTotals.map((total, index) => {
          const comparison = model.comparisons[index];
          const reached = index <= comparisonIndex;
          const tied = reached && comparison?.tied === true;
          return <span className={tied ? "tied-score" : reached ? "active-score" : "pending-score"} key={index}>{total}{tied ? <small>Tie</small> : null}</span>;
        })}
      </div>
    </section>
  );
}

function SelectedScores({ model }: { model: ScoringPresentationModel }) {
  return (
    <section className="selected-round-scores" aria-live="polite">
      <div><span>Your Score</span><strong>{model.selection.humanScore}</strong><small>Round Score</small></div>
      <div><span>Dracula's Score</span><strong>{model.selection.opponentScore}</strong><small>Round Score</small></div>
    </section>
  );
}

export function RoundTotals({ model }: { model: ScoringPresentationModel }) {
  return (
    <section className="round-totals" aria-label={`Round ${model.record.round_number} totals`}>
      <div>
        <span>Your Score</span>
        <strong aria-label={`Your round score: ${model.totals.humanRound}`}>{model.totals.humanRound}</strong>
        <span className="previous-total" aria-label={`Your previous total: ${model.totals.humanPrevious}`}>+ {model.totals.humanPrevious}</span>
        <b aria-label={`Your new total: ${model.totals.humanTotal}`}>{model.totals.humanTotal}</b>
      </div>
      <div>
        <span>Dracula's Score</span>
        <strong aria-label={`Dracula's round score: ${model.totals.opponentRound}`}>{model.totals.opponentRound}</strong>
        <span className="previous-total" aria-label={`Dracula's previous total: ${model.totals.opponentPrevious}`}>+ {model.totals.opponentPrevious}</span>
        <b aria-label={`Dracula's new total: ${model.totals.opponentTotal}`}>{model.totals.opponentTotal}</b>
      </div>
    </section>
  );
}

function PresentationWorkspace({
  model,
  frame,
}: {
  model: ScoringPresentationModel;
  frame: ScoringFrame;
}) {
  if (
    frame.kind === "reveal_value" ||
    frame.kind === "collapse_sum" ||
    frame.kind === "multiplier_label" ||
    frame.kind === "multiplier_factor" ||
    frame.kind === "line_total"
  ) {
    return <LineWorkspace model={model} frame={frame} />;
  }
  if (frame.kind === "orientation_ranked") {
    return <RankedOrientation model={model} orientationIndex={frame.orientationIndex} />;
  }
  if (frame.kind === "compare_rank") {
    return <ComparisonWorkspace model={model} comparisonIndex={frame.comparisonIndex} />;
  }
  if (frame.kind === "select_round_score") return <SelectedScores model={model} />;
  if (frame.kind === "update_totals" || frame.kind === "complete") return <RoundTotals model={model} />;
  const message = frame.kind === "entering"
    ? "Preparing the completed coffin…"
    : frame.kind === "orientation_handoff"
      ? "Now for the other score…"
      : "Waiting for commentary…";
  return <p className="presentation-status" aria-live="polite">{message}</p>;
}

export interface ScoringPresentationProps {
  controller: GameController;
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
  const timeline = useMemo(
    () => createScoringTimeline(model, view.narration_enabled),
    [model, view.narration_enabled],
  );
  const reducedMotion = useReducedMotion(forcedReducedMotion);
  const [frameIndex, setFrameIndex] = useState(0);
  const roundKey = `${view.game_id}:${record.round_number}:${view.version}`;
  const finalizeRequested = useRef<string | null>(null);

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

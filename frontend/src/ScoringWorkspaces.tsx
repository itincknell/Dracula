/**
 * Renders the pure visual workspaces used by the scoring animation.
 * These components translate an already-derived scoring model and timeline
 * frame into cards, arithmetic, comparisons, and final round totals. They do
 * not own timers, narration, or game-controller side effects.
 */
import type { CSSProperties, ReactNode } from "react";

import { CardFace } from "./CardFace";
import type { Player } from "./contractPrimitives";
import type {
  PresentedLine,
  ScoringFrame,
  ScoringPresentationModel,
} from "./scoringStateMachine";

function playerHeading(player: Player, humanRole: Player): string {
  return player === humanRole ? "Your Score" : "Dracula's Score";
}

/** Return the line addressed by frames that operate on one specific line. */
function activeLine(
  model: ScoringPresentationModel,
  frame: ScoringFrame,
): PresentedLine | null {
  if (!("orientationIndex" in frame) || !("lineIndex" in frame)) return null;
  return model.orientations[frame.orientationIndex].lines[frame.lineIndex];
}

/**
 * Render the completed coffin while marking cards used by the current frame.
 * CSS owns motion; this component supplies stable classes for the active line,
 * multiplier highlights, the current value pop, and the sum-collapse ripple.
 */
export function ScoringCoffin({
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
        // Card IDs are unique within a deck, so membership and ripple order can
        // be derived from the active line without referring to grid geometry.
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

/** Display a Vampire card as zero while retaining its engine-supplied value. */
function valueText(item: PresentedLine, cardIndex: number): string {
  const cardId = item.line.card_ids[cardIndex];
  if (
    item.line.multiplier_reason === "vampire" &&
    item.line.highlighted_card_ids.includes(cardId)
  ) {
    return "0";
  }
  return String(item.values[cardIndex]);
}

/** Render every arithmetic stage for one row or column. */
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

  // The same fixed-height workspace changes only its expression as the line
  // moves from card values to sum, multiplier description, factor, and total.
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

  // Totals from earlier lines stay visible. The current total joins them only
  // at its final line_total frame, keeping the tally synchronized with cards.
  const completed = orientation.lines.slice(0, frame.lineIndex).map((line) => line.line.total);
  if (frame.kind === "line_total") completed.push(item.line.total);
  return (
    <section className="calculation-workspace" aria-live="polite">
      <div className="score-workspace-heading">
        <p className="score-heading">{playerHeading(orientation.player, model.humanRole)}</p>
        <p className="series-label">{item.line.direction === "row" ? "Row" : "Col"} {item.line.index + 1}</p>
      </div>
      <div className={`score-expression ${frame.kind}`}>{expression}</div>
      <div className="completed-line-totals" aria-label="Completed line totals">
        {completed.map((total, index) => <span key={`${index}-${total}`}>{total}</span>)}
      </div>
    </section>
  );
}

/** Show one player's completed line totals in strongest-to-weakest order. */
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
      <div className="score-workspace-heading">
        <p className="score-heading">{playerHeading(orientation.player, model.humanRole)}</p>
      </div>
      {/* This invisible-sized placeholder preserves the expression row while
          the ranked totals replace the earlier arithmetic. */}
      <div className="score-expression score-expression-placeholder" aria-hidden="true">0</div>
      <div className="ranked-line-totals" aria-label="Ranked line totals">
        {orientation.rankedTotals.map((total, index) => (
          <span key={index}>{total}</span>
        ))}
      </div>
    </section>
  );
}

function ComparisonColumn({
  heading,
  totals,
  model,
  comparisonIndex,
}: {
  heading: string;
  totals: [number, number, number];
  model: ScoringPresentationModel;
  comparisonIndex: number;
}) {
  return (
    <div className="comparison-column">
      <h2>{heading}</h2>
      {totals.map((total, index) => {
        // Future ranks remain dim. Reached ties are struck and labeled before
        // the animation moves to the next rank; the deciding rank is active.
        const comparison = model.comparisons[index];
        const reached = index <= comparisonIndex;
        const tied = reached && comparison?.tied === true;
        const className = tied ? "tied-score" : reached ? "active-score" : "pending-score";
        return (
          <span className={className} key={index}>
            {total}
            {tied ? <small>Tie</small> : null}
          </span>
        );
      })}
    </div>
  );
}

/** Compare the players' ranked line totals side by side until one differs. */
function ComparisonWorkspace({
  model,
  comparisonIndex,
}: {
  model: ScoringPresentationModel;
  comparisonIndex: number;
}) {
  return (
    <section className="round-comparison" aria-live="polite">
      <ComparisonColumn
        heading="Your Score"
        totals={model.humanRankedTotals}
        model={model}
        comparisonIndex={comparisonIndex}
      />
      <ComparisonColumn
        heading="Dracula's Score"
        totals={model.opponentRankedTotals}
        model={model}
        comparisonIndex={comparisonIndex}
      />
    </section>
  );
}

/** Show the two line totals selected as this round's official scores. */
function SelectedScores({ model }: { model: ScoringPresentationModel }) {
  return (
    <section className="selected-round-scores" aria-live="polite">
      <div><span>Your Score</span><strong>{model.selection.humanScore}</strong><small>Round Score</small></div>
      <div><span>Dracula's Score</span><strong>{model.selection.opponentScore}</strong><small>Round Score</small></div>
    </section>
  );
}

/** Show round additions and resulting cumulative totals in aligned columns. */
export function RoundTotals({ model }: { model: ScoringPresentationModel }) {
  const columns = [
    {
      label: "Your Score",
      owner: "Your",
      round: model.totals.humanRound,
      previous: model.totals.humanPrevious,
      total: model.totals.humanTotal,
    },
    {
      label: "Dracula's Score",
      owner: "Dracula's",
      round: model.totals.opponentRound,
      previous: model.totals.opponentPrevious,
      total: model.totals.opponentTotal,
    },
  ];
  return (
    <section className="round-totals" aria-label={`Round ${model.record.round_number} totals`}>
      {columns.map((column) => (
        <div key={column.label}>
          <span>{column.label}</span>
          <strong aria-label={`${column.owner} round score: ${column.round}`}>
            {column.round}
          </strong>
          <span
            className="previous-total"
            aria-label={`${column.owner} previous total: ${column.previous}`}
          >
            + {column.previous}
          </span>
          <b aria-label={`${column.owner} new total: ${column.total}`}>{column.total}</b>
        </div>
      ))}
    </section>
  );
}

/** Select the workspace matching the current scoring frame. */
export function PresentationWorkspace({
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
  let message = "Waiting for commentary…";
  if (frame.kind === "entering") message = "Preparing the completed coffin…";
  else if (frame.kind === "orientation_handoff") message = "Now for the other score…";
  return <p className="presentation-status" aria-live="polite">{message}</p>;
}

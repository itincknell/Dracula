/**
 * Composes the active game board from controller state and presentation pieces.
 * It renders the coffin, human hand, public status and scores, Dracula bar, and
 * scoring overlay without owning API transport or recovery persistence.
 */
import { useEffect, useRef } from "react";

import { CardFace } from "./CardFace";
import { cardName } from "./cardAssets";
import { DraculaCommentary } from "./DraculaCommentary";
import { FinalRoundPresentation, ScoringPresentation } from "./ScoringPresentation";
import type { HumanGameView } from "./gameView";
import {
  type GameControllerContract,
  type PendingAction,
  type PresentationState,
  useGameStore,
} from "./gameControllerContract";

/** Describe the settled phase or the controller operation currently in progress. */
function TurnStatus({
  view,
  pending,
}: {
  view: HumanGameView;
  pending: PendingAction | null;
}) {
  const humanRole = view.human_role === "queen" ? "Queen" : "King";
  const orientation = view.human_role === "queen" ? "Rows" : "Columns";
  let message = `Your turn — ${humanRole} · ${orientation}`;
  if (pending === "opponent_turn" || view.phase.kind === "opponent_turn") {
    message = "Dracula is deciding…";
  } else if (pending === "round_advance") message = "Dealing the next round…";
  else if (view.phase.kind === "scoring") {
    message = `Round ${view.round_number} complete`;
  } else if (view.phase.kind === "game_complete") {
    message = "Game complete";
  }
  return (
    <p className="turn-status" aria-label={`Current turn: ${message}`} aria-live="polite" tabIndex={-1}>
      {message}
    </p>
  );
}

/** Show the running game totals in human-versus-Dracula terms. */
function Scoreboard({ view }: { view: HumanGameView }) {
  return (
    <section className="scoreboard" aria-label="Score tally">
      <span>You</span><strong aria-label={`Your score: ${view.total_scores.human}`}>{view.total_scores.human}</strong>
      <span>Dracula</span><strong aria-label={`Dracula's score: ${view.total_scores.opponent}`}>{view.total_scores.opponent}</strong>
    </section>
  );
}

interface GameSurfaceProps {
  controller: GameControllerContract;
  view: HumanGameView;
  presentation: PresentationState;
}

function CoffinPosition({
  cardId,
  controller,
  legal,
  position,
  selectedCard,
}: {
  cardId: string | null;
  controller: GameControllerContract;
  legal: boolean;
  position: number;
  selectedCard: string | null;
}) {
  // One component represents all three cell states: occupied card, legal
  // destination button, or inert empty position. This preserves the grid's
  // dimensions while giving only legal destinations interactive semantics.
  const className = [
    "card-slot",
    position === 4 ? "center-card" : "",
    legal ? "legal-target" : "",
  ].filter(Boolean).join(" ");
  if (cardId !== null) {
    return (
      <div
        className={className}
        data-card-id={cardId}
        data-position={position}
        role="group"
        aria-label={`${cardName(cardId)}, coffin position ${position + 1}`}
      >
        <CardFace cardId={cardId} decorative />
      </div>
    );
  }
  if (legal) {
    return (
      <button
        className={className}
        type="button"
        data-position={position}
        aria-label={`Play ${selectedCard === null ? "selected card" : cardName(selectedCard)} at coffin position ${position + 1}`}
        onClick={() => void controller.playPosition(position)}
        onDragOver={(event) => {
          // Preventing the browser default is what permits the subsequent drop.
          event.preventDefault();
          event.dataTransfer.dropEffect = "move";
        }}
        onDrop={(event) => {
          event.preventDefault();
          void controller.playPosition(position);
        }}
      >
        <span aria-hidden="true">+</span>
      </button>
    );
  }
  return (
    <div
      className={className}
      data-position={position}
      role="group"
      aria-label={`Empty coffin position ${position + 1}`}
      aria-disabled="true"
      onDragOver={(event) => {
        event.dataTransfer.dropEffect = "none";
      }}
    />
  );
}

export function CardGrid({ controller, view, presentation }: GameSurfaceProps) {
  // Legality comes from the validated server response. The controller narrows
  // it further to the card currently selected by the human.
  const inputEnabled = presentation.pending === null && view.phase.kind === "human_turn";
  const selectedCard = presentation.selectedHandSlot === null
    ? null
    : view.human_hand[presentation.selectedHandSlot];
  return (
    <section className="coffin" aria-label="Coffin">
      {view.coffin.map((cardId, position) => (
        <CoffinPosition
          cardId={cardId}
          controller={controller}
          key={position}
          legal={inputEnabled && controller.legalMoveAt(position) !== null}
          position={position}
          selectedCard={selectedCard}
        />
      ))}
    </section>
  );
}

function HandPosition({
  cardId,
  controller,
  handSlot,
  selectable,
  selected,
}: {
  cardId: string | null;
  controller: GameControllerContract;
  handSlot: number;
  selectable: boolean;
  selected: boolean;
}) {
  // Played cards leave visible empty slots because engine hand indexes remain
  // stable for the entire round and are submitted in placement commands.
  if (cardId === null) {
    return (
      <div
        className="hand-card played-card"
        data-hand-slot={handSlot}
        role="group"
        aria-label={`Hand slot ${handSlot + 1}, played`}
      />
    );
  }
  return (
    <button
      className={`hand-card${selected ? " selected-card" : ""}`}
      type="button"
      data-hand-slot={handSlot}
      data-card-id={cardId}
      aria-label={`${cardName(cardId)} (${cardId}), hand slot ${handSlot + 1}`}
      aria-pressed={selected}
      disabled={!selectable}
      draggable={selectable}
      onClick={() => controller.selectCard(handSlot)}
      onDragStart={(event) => {
        // Dragging and clicking share controller.selectCard so they cannot
        // disagree about which stable hand slot a destination will submit.
        if (!controller.selectCard(handSlot)) {
          event.preventDefault();
          return;
        }
        event.dataTransfer.effectAllowed = "move";
        event.dataTransfer.setData("text/plain", String(handSlot));
      }}
    >
      <CardFace cardId={cardId} decorative />
    </button>
  );
}

export function Hand({ controller, view, presentation }: GameSurfaceProps) {
  const inputEnabled = presentation.pending === null && view.phase.kind === "human_turn";
  return (
    <section className="hand" aria-label="Your hand">
      {view.human_hand.map((cardId, handSlot) => {
        const selected = presentation.selectedHandSlot === handSlot;
        const selectable =
          inputEnabled &&
          cardId !== null &&
          view.legal_moves.some((move) => move.hand_slot === handSlot);
        return (
          <HandPosition
            cardId={cardId}
            controller={controller}
            handSlot={handSlot}
            key={handSlot}
            selectable={selectable}
            selected={selected}
          />
        );
      })}
    </section>
  );
}

export function MainDisplay({
  controller,
  onNewGame,
  presentation,
  view,
}: {
  controller: GameControllerContract;
  onNewGame: () => void;
  presentation: PresentationState;
  view: HumanGameView;
}) {
  const statusRef = useRef<HTMLDivElement>(null);
  const priorPending = useRef(presentation.pending);

  useEffect(() => {
    // Restore keyboard/screen-reader focus after Dracula's asynchronous turn.
    // preventScroll preserves the approved fixed mobile viewport position.
    if (
      priorPending.current === "opponent_turn" &&
      presentation.pending === null
    ) {
      statusRef.current?.querySelector<HTMLElement>(".turn-status")?.focus({ preventScroll: true });
    }
    priorPending.current = presentation.pending;
  }, [presentation.pending]);

  const showingScores = view.phase.kind === "scoring";
  // The final engine state has no pending result; its last completed record
  // supplies the final round totals displayed beside the game outcome.
  const finalRecord = view.phase.kind === "game_complete" ? view.completed_rounds.at(-1) : undefined;
  return (
    <div
      className={`main-display${showingScores ? " scoring-mode" : ""}${finalRecord === undefined ? "" : " final-mode"}`}
      ref={statusRef}
      aria-busy={presentation.pending !== null}
    >
      <CardGrid controller={controller} view={view} presentation={presentation} />
      <Hand controller={controller} view={view} presentation={presentation} />
      <TurnStatus view={view} pending={presentation.pending} />
      <Scoreboard view={view} />
      {showingScores ? <ScoringPresentation controller={controller} view={view} /> : null}
      {finalRecord === undefined ? null : (
        <FinalRoundPresentation record={finalRecord} view={view} onNewGame={onNewGame} />
      )}
    </div>
  );
}

export function GameWindow({
  controller,
  onNewGame,
}: {
  controller: GameControllerContract;
  onNewGame: () => void;
}) {
  const { view, presentation, narration } = useGameStore(controller);

  // The route can mount while stateless recovery is still rebuilding the view.
  if (view === null) return null;
  return (
    <div className="game-layout-container">
      {presentation.error !== null ? (
        <section className="error-banner" role="alert">
          <span>{presentation.error.message}</span>
          <button type="button" onClick={() => controller.clearError()}>Dismiss</button>
        </section>
      ) : null}
      <section
        className="game-window"
        aria-label="Dracula game"
      >
        <MainDisplay
          controller={controller}
          onNewGame={onNewGame}
          presentation={presentation}
          view={view}
        />
        <DraculaCommentary view={view} narration={narration} />
      </section>
    </div>
  );
}

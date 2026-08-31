import { useEffect, useRef } from "react";

import { CardFace } from "./CardFace";
import { cardName } from "./cardAssets";
import { FinalRoundPresentation, ScoringPresentation } from "./ScoringPresentation";
import type { HumanGameView } from "./contracts";
import { GameController, useGameStore } from "./gameStore";

export function TurnStatus({ view, pending }: { view: HumanGameView; pending: string | null }) {
  const humanRole = view.human_role === "queen" ? "Queen" : "King";
  const orientation = view.human_role === "queen" ? "Rows" : "Columns";
  let message = `Your turn — ${humanRole} · ${orientation}`;
  if (pending === "human_move") message = "Playing your card…";
  else if (pending === "opponent_turn" || view.phase.kind === "opponent_turn") {
    message = "Dracula is deciding…";
  } else if (pending === "round_advance") message = "Dealing the next round…";
  else if (view.phase.kind === "scoring" || view.phase.kind === "round_advance") {
    message = `Round ${view.round_number} complete`;
  } else if (view.phase.kind === "game_complete") {
    message = "Game complete";
  } else if (view.phase.kind === "narration") {
    message = "Preparing the round…";
  }
  return (
    <p className="turn-status" aria-label={`Current turn: ${message}`} aria-live="polite" tabIndex={-1}>
      {message}
    </p>
  );
}

export function Scoreboard({ view }: { view: HumanGameView }) {
  return (
    <section className="scoreboard" aria-label="Score tally">
      <span>You</span><strong aria-label={`Your score: ${view.total_scores.human}`}>{view.total_scores.human}</strong>
      <span>Dracula</span><strong aria-label={`Dracula's score: ${view.total_scores.opponent}`}>{view.total_scores.opponent}</strong>
    </section>
  );
}

export function CardGrid({ controller }: { controller: GameController }) {
  const { view, presentation } = useGameStore(controller);
  if (view === null) return null;
  const inputEnabled = presentation.pending === null && view.phase.kind === "human_turn";
  const selectedCard =
    presentation.selectedHandSlot === null
      ? null
      : view.human_hand[presentation.selectedHandSlot];

  return (
    <section className="coffin" aria-label="Coffin">
      {view.coffin.map((cardId, position) => {
        const legalMove = inputEnabled ? controller.legalMoveAt(position) : null;
        const className = [
          "card-slot",
          position === 4 ? "center-card" : "",
          legalMove === null ? "" : "legal-target",
        ].filter(Boolean).join(" ");
        if (cardId !== null) {
          return (
            <div
              className={className}
              key={position}
              data-card-id={cardId}
              data-position={position}
              role="group"
              aria-label={`${cardName(cardId)}, coffin position ${position + 1}`}
            >
              <CardFace cardId={cardId} decorative />
            </div>
          );
        }
        if (legalMove !== null) {
          return (
            <button
              className={className}
              type="button"
              key={position}
              data-position={position}
              aria-label={`Play ${selectedCard === null ? "selected card" : cardName(selectedCard)} at coffin position ${position + 1}`}
              onClick={() => void controller.playPosition(position)}
              onDragOver={(event) => {
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
            key={position}
            data-position={position}
            role="group"
            aria-label={`Empty coffin position ${position + 1}`}
            aria-disabled="true"
            onDragOver={(event) => {
              event.dataTransfer.dropEffect = "none";
            }}
          />
        );
      })}
    </section>
  );
}

export function Hand({ controller }: { controller: GameController }) {
  const { view, presentation } = useGameStore(controller);
  if (view === null) return null;
  const inputEnabled = presentation.pending === null && view.phase.kind === "human_turn";
  return (
    <section className="hand" aria-label="Your hand">
      {view.human_hand.map((cardId, handSlot) => {
        const selected = presentation.selectedHandSlot === handSlot;
        const selectable =
          inputEnabled &&
          cardId !== null &&
          view.legal_moves.some((move) => move.hand_slot === handSlot);
        return cardId === null ? (
          <div
            className="hand-card played-card"
            key={handSlot}
            data-hand-slot={handSlot}
            role="group"
            aria-label={`Hand slot ${handSlot + 1}, played`}
          />
        ) : (
          <button
            className={`hand-card${selected ? " selected-card" : ""}`}
            type="button"
            key={handSlot}
            data-hand-slot={handSlot}
            data-card-id={cardId}
            aria-label={`${cardName(cardId)} (${cardId}), hand slot ${handSlot + 1}`}
            aria-pressed={selected}
            disabled={!selectable}
            draggable={selectable}
            onClick={() => controller.selectCard(handSlot)}
            onDragStart={(event) => {
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
      })}
    </section>
  );
}

export function MainDisplay({
  controller,
  onNewGame,
}: {
  controller: GameController;
  onNewGame: () => void;
}) {
  const { view, presentation } = useGameStore(controller);
  const statusRef = useRef<HTMLDivElement>(null);
  const priorPending = useRef(presentation.pending);

  useEffect(() => {
    if (priorPending.current === "human_move" && presentation.pending === null) {
      statusRef.current?.querySelector<HTMLElement>(".turn-status")?.focus({ preventScroll: true });
    }
    priorPending.current = presentation.pending;
  }, [presentation.pending]);

  if (view === null) return null;
  const showingScores = view.phase.kind === "scoring" || view.phase.kind === "round_advance";
  const finalRecord = view.phase.kind === "game_complete" ? view.completed_rounds.at(-1) : undefined;
  return (
    <div
      className={`main-display${showingScores ? " scoring-mode" : ""}${finalRecord === undefined ? "" : " final-mode"}`}
      ref={statusRef}
      aria-busy={presentation.pending !== null}
      data-pending={presentation.pending ?? undefined}
    >
      <CardGrid controller={controller} />
      <Hand controller={controller} />
      <TurnStatus view={view} pending={presentation.pending} />
      <Scoreboard view={view} />
      {showingScores ? <ScoringPresentation controller={controller} view={view} /> : null}
      {finalRecord === undefined ? null : (
        <FinalRoundPresentation record={finalRecord} view={view} onNewGame={onNewGame} />
      )}
    </div>
  );
}

export function CommentaryPanel({ narrationEnabled }: { narrationEnabled: boolean }) {
  return (
    <aside
      className="commentary-panel"
      aria-label="Dracula commentary"
      data-narration-enabled={narrationEnabled}
    >
      <div className="portrait-placeholder" aria-hidden="true">
        <img className="dracula-avatar" src="/dracula.png" alt="" draggable={false} />
      </div>
      <div className="commentary-stream" role="status" aria-live="polite" aria-atomic="true" aria-relevant="additions text">
        <span className="commentary-status">
          {narrationEnabled ? "Commentary is not connected." : "Narration disabled"}
        </span>
      </div>
    </aside>
  );
}

export function GameWindow({
  controller,
  onNewGame,
}: {
  controller: GameController;
  onNewGame: () => void;
}) {
  const { view, presentation } = useGameStore(controller);

  useEffect(() => {
    if (
      presentation.pending === null &&
      view?.phase.kind === "opponent_turn" &&
      view.phase.status !== "failed"
    ) {
      void controller.progressOpponent();
    }
  }, [controller, presentation.pending, view]);

  if (view === null) return null;
  return (
    <div className="game-layout-container">
      {presentation.error !== null ? (
        <section className="error-banner" role="alert">
          <span>{presentation.error.message}</span>
          {view.phase.kind === "opponent_turn" && view.phase.status === "failed" ? (
            <button type="button" onClick={() => void controller.progressOpponent({ retryFailed: true })}>
              Retry Dracula turn
            </button>
          ) : (
            <button type="button" onClick={() => controller.clearError()}>Dismiss</button>
          )}
        </section>
      ) : null}
      <section className="game-window" data-game-id={view.game_id} aria-label="Dracula game">
        <MainDisplay controller={controller} onNewGame={onNewGame} />
        <CommentaryPanel narrationEnabled={view.narration_enabled} />
      </section>
    </div>
  );
}

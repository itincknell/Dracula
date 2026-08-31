import { useCallback, useEffect, useMemo, useState } from "react";

import { apiClient } from "./api";
import { cardName } from "./cardAssets";
import type { HumanGameView, Player } from "./contracts";
import { GameController, useGameStore } from "./gameStore";
import { GameWindow } from "./gameplay";
import { RulesPage } from "./RulesPage";
import { siteLinks } from "./siteConfig";

export type AppRoute =
  | { kind: "start" }
  | { kind: "game"; gameId: string }
  | { kind: "rules" }
  | { kind: "not_found" };

export function resolveRoute(pathname: string): AppRoute {
  if (pathname === "/" || pathname === "/start") return { kind: "start" };
  if (pathname === "/rules") return { kind: "rules" };
  const gameMatch = pathname.match(/^\/games\/([^/]+)$/);
  if (gameMatch?.[1]) return { kind: "game", gameId: decodeURIComponent(gameMatch[1]) };
  return { kind: "not_found" };
}

const CARD_RANKS = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"] as const;
const CARD_SUITS = [
  { id: "C", name: "Clubs" },
  { id: "D", name: "Diamonds" },
  { id: "H", name: "Hearts" },
  { id: "S", name: "Spades" },
] as const;

export function seenCardIds(view: HumanGameView): ReadonlySet<string> {
  const cards = new Set<string>();
  for (const round of view.completed_rounds) {
    for (const cardId of round.coffin) cards.add(cardId);
  }
  if (view.pending_round_result !== null) {
    for (const cardId of view.pending_round_result.coffin) cards.add(cardId);
  }
  for (const cardId of view.coffin) {
    if (cardId !== null) cards.add(cardId);
  }
  for (const cardId of view.human_hand) {
    if (cardId !== null) cards.add(cardId);
  }
  return cards;
}

export function SeenCardsExpando({ view }: { view: HumanGameView }) {
  const [expanded, setExpanded] = useState(false);
  const seen = seenCardIds(view);
  const panelId = "seen-card-ledger";

  return (
    <section className="cheat-sheet-window" aria-label="Cheat sheet">
      <button
        className="cheat-sheet-expando"
        type="button"
        aria-controls={panelId}
        aria-expanded={expanded}
        onClick={() => setExpanded((current) => !current)}
      >
        <span>Cheat Sheet</span>
        <span aria-hidden="true">{expanded ? "−" : "+"}</span>
      </button>
      {expanded ? (
        <div className="seen-card-ledger" id={panelId}>
          <div className="seen-card-grid-scroll">
            <table className="seen-card-grid">
              <caption>{seen.size} of 54 cards seen</caption>
              <thead>
                <tr>
                  <th scope="col" aria-label="Suit" />
                  {CARD_RANKS.map((rank) => <th scope="col" key={rank}>{rank}</th>)}
                </tr>
              </thead>
              <tbody>
                {CARD_SUITS.map((suit) => (
                  <tr key={suit.id}>
                    <th scope="row">{suit.name}</th>
                    {CARD_RANKS.map((rank) => {
                      const cardId = `${rank}${suit.id}`;
                      const cardSeen = seen.has(cardId);
                      return (
                        <td key={cardId} data-seen={cardSeen} aria-label={`${cardName(cardId)}: ${cardSeen ? "seen" : "not seen"}`}>
                          <span aria-hidden="true">{cardSeen ? "X" : ""}</span>
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="vampire-ledger" aria-label="Vampires">
            {["V1", "V2"].map((cardId) => {
              const cardSeen = seen.has(cardId);
              return (
                <div className="vampire-ledger-cell" key={cardId} data-seen={cardSeen} aria-label={`${cardName(cardId)}: ${cardSeen ? "seen" : "not seen"}`}>
                  <strong>{cardId}</strong><span aria-hidden="true">{cardSeen ? "X" : ""}</span>
                </div>
              );
            })}
          </div>
        </div>
      ) : null}
    </section>
  );
}

function FooterLinks() {
  return (
    <nav className="footer-links" aria-label="Project links">
      <a href={siteLinks.rules} target="_blank" rel="noreferrer">Rules</a>
      <a href={siteLinks.about}>About</a>
      <a href={siteLinks.contact}>Contact</a>
    </nav>
  );
}

export function GameStart({
  controller,
  onGameCreated,
}: {
  controller: GameController;
  onGameCreated: (gameId: string) => void;
}) {
  const { presentation } = useGameStore(controller);
  const start = async (role: Player) => {
    const view = await controller.createGame(role);
    if (view !== null) onGameCreated(view.game_id);
  };
  return (
    <main className="page-shell start-page">
      <section className="start-card" aria-labelledby="start-title">
        <p className="eyebrow">A game of cards and consequence</p>
        <h1 id="start-title">Dracula</h1>
        <p>Choose the direction you will score for the entire six-round game.</p>
        <div className="role-actions">
          <button
            className="primary-action"
            type="button"
            disabled={presentation.pending !== null}
            onClick={() => void start("queen")}
          >
            Start as Queen
          </button>
          <button
            className="primary-action"
            type="button"
            disabled={presentation.pending !== null}
            onClick={() => void start("king")}
          >
            Start as King
          </button>
        </div>
        {presentation.pending === "create_game" ? <p role="status">Dealing the first round…</p> : null}
        {presentation.error !== null ? (
          <p className="start-error" role="alert">{presentation.error.message}</p>
        ) : null}
        <a className="rules-link" href={siteLinks.rules} target="_blank" rel="noreferrer">
          Read the rules
        </a>
      </section>
      <FooterLinks />
    </main>
  );
}

function GamePage({
  gameId,
  controller,
  navigate,
}: {
  gameId: string;
  controller: GameController;
  navigate: (path: string) => void;
}) {
  const { view, presentation } = useGameStore(controller);

  useEffect(() => {
    if (view?.game_id !== gameId) void controller.loadGame(gameId);
  }, [controller, gameId, view?.game_id]);

  const currentView = view?.game_id === gameId ? view : null;
  return (
    <main className="page-shell">
      <header className="game-heading">
        <h1><a href="/" className="wordmark">Dracula</a></h1>
        <span>{currentView === null ? "Loading game" : `Round ${currentView.round_number} of 6`}</span>
      </header>
      {currentView === null ? (
        <section className="loading-panel" aria-live="polite">
          {presentation.error === null ? (
            <p>Loading game…</p>
          ) : (
            <>
              <p role="alert">{presentation.error.message}</p>
              <button className="primary-action" type="button" onClick={() => void controller.loadGame(gameId)}>
                Retry
              </button>
            </>
          )}
        </section>
      ) : (
        <GameWindow controller={controller} onNewGame={() => navigate("/")} />
      )}
      {currentView === null ? null : <SeenCardsExpando view={currentView} />}
      <FooterLinks />
    </main>
  );
}

function NotFoundPage() {
  return (
    <main className="page-shell start-page">
      <section className="start-card">
        <h1>Page not found</h1>
        <a className="primary-action" href="/">Return to Dracula</a>
      </section>
    </main>
  );
}

export function App({ controller: suppliedController }: { controller?: GameController }) {
  const controller = useMemo(() => suppliedController ?? new GameController(apiClient), [suppliedController]);
  const [route, setRoute] = useState(() => resolveRoute(window.location.pathname));

  useEffect(() => {
    const updateRoute = () => setRoute(resolveRoute(window.location.pathname));
    window.addEventListener("popstate", updateRoute);
    return () => window.removeEventListener("popstate", updateRoute);
  }, []);

  const navigate = useCallback((path: string) => {
    window.history.pushState({}, "", path);
    setRoute(resolveRoute(path));
  }, []);

  switch (route.kind) {
    case "start":
      return <GameStart controller={controller} onGameCreated={(id) => navigate(`/games/${id}`)} />;
    case "game":
      return <GamePage gameId={route.gameId} controller={controller} navigate={navigate} />;
    case "rules":
      return <RulesPage />;
    case "not_found":
      return <NotFoundPage />;
  }
}

/**
 * Assembles the browser application and its small hash-based page flow.
 * The component creates the stateless controller, selects start/game/rules
 * views, and owns only page-level UI such as the cheat sheet and site chrome.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { type GameControllerContract, useGameStore } from "./gameControllerContract";
import { GameWindow } from "./gameplay";
import { RulesPage } from "./RulesPage";
import { SeenCardsExpando } from "./SeenCardsExpando";
import { siteLinks } from "./siteConfig";
import { statelessApiClient } from "./statelessApi";
import { StatelessGameController } from "./statelessGameStore";

/** The four pages recognized by the deliberately small hash router. */
type AppRoute =
  | { kind: "start" }
  | { kind: "game" }
  | { kind: "rules" }
  | { kind: "not_found" };

export function resolveRoute(hash: string): AppRoute {
  if (hash === "" || hash === "#" || hash === "#/" || hash === "#/start") return { kind: "start" };
  if (hash === "#/game") return { kind: "game" };
  if (hash === "#/rules") return { kind: "rules" };
  return { kind: "not_found" };
}

/** Site-level navigation shared by the start and gameplay pages. */
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
  controller: GameControllerContract;
  onGameCreated: () => void;
}) {
  const { view, presentation } = useGameStore(controller);
  const routedToGame = useRef(false);

  useEffect(() => {
    // createGame may already be waiting on the visible Dracula opening delay.
    // Navigate as soon as its preview view exists, and only once per mount.
    if (view !== null && !routedToGame.current) {
      routedToGame.current = true;
      onGameCreated();
    }
  }, [onGameCreated, view]);

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
            onClick={() => void controller.createGame("queen")}
          >
            Start as Queen
          </button>
          <button
            className="primary-action"
            type="button"
            disabled={presentation.pending !== null}
            onClick={() => void controller.createGame("king")}
          >
            Start as King
          </button>
        </div>
        {presentation.pending === "create_game" ? (
          <div className="start-overlay">
            <p className="start-progress" role="status">
              Starting game<span className="start-progress-dots" aria-hidden="true" />
            </p>
          </div>
        ) : null}
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
  controller,
  navigate,
}: {
  controller: GameControllerContract;
  navigate: (path: string) => void;
}) {
  const { view, presentation } = useGameStore(controller);
  const loadAttempted = useRef(false);

  useEffect(() => {
    // React StrictMode runs mount effects twice in development. The local ref
    // prevents duplicate calls here; the controller also shares in-flight
    // recovery among callers as a second boundary-level guarantee.
    if (view === null && !loadAttempted.current) {
      loadAttempted.current = true;
      void controller.loadGame();
    }
  }, [controller, view]);

  const currentView = view;
  return (
    <main className="page-shell">
      <header className="game-heading">
        <h1><a href={siteLinks.home} className="wordmark">Dracula</a></h1>
        <span>{currentView === null ? "Loading game" : `Round ${currentView.round_number} of 6`}</span>
      </header>
      {currentView === null ? (
        <section className="loading-panel" aria-live="polite">
          {presentation.error === null ? (
            <p>Loading game…</p>
          ) : (
            <>
              <p role="alert">{presentation.error.message}</p>
              <button className="primary-action" type="button" onClick={() => void controller.loadGame()}>
                Retry
              </button>
            </>
          )}
        </section>
      ) : (
        <GameWindow controller={controller} onNewGame={() => {
          controller.clearGame();
          navigate("/");
        }} />
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
        <a className="primary-action" href={siteLinks.home}>Return to Dracula</a>
      </section>
    </main>
  );
}

export function App({ controller: suppliedController }: { controller?: GameControllerContract }) {
  // Component tests inject a controller. The real application creates exactly
  // one stateless controller for the lifetime of this App mount.
  const controller = useMemo(
    () => suppliedController ?? new StatelessGameController(statelessApiClient),
    [suppliedController],
  );
  const [route, setRoute] = useState(() => resolveRoute(window.location.hash));

  useEffect(() => {
    // Hash changes cover ordinary link navigation; popstate covers browser
    // back/forward traversal after navigate() uses history.pushState.
    const updateRoute = () => setRoute(resolveRoute(window.location.hash));
    window.addEventListener("popstate", updateRoute);
    window.addEventListener("hashchange", updateRoute);
    return () => {
      window.removeEventListener("popstate", updateRoute);
      window.removeEventListener("hashchange", updateRoute);
    };
  }, []);

  const navigate = useCallback((path: string) => {
    // Hash routing lets every GitHub Pages request resolve the same index.html;
    // no server-side fallback or routing package is required.
    const hash = path === "/" ? "#/" : `#${path}`;
    window.history.pushState({}, "", `${siteLinks.home}${hash}`);
    setRoute(resolveRoute(hash));
  }, []);

  switch (route.kind) {
    case "start":
      return <GameStart controller={controller} onGameCreated={() => navigate("/game")} />;
    case "game":
      return <GamePage controller={controller} navigate={navigate} />;
    case "rules":
      return <RulesPage />;
    case "not_found":
      return <NotFoundPage />;
  }
}

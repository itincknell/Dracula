/**
 * Renders the standalone public rules and reference page.
 * Section navigation remains client-side within the document, while site links
 * use the same environment-aware destinations as the main application.
 */
import type { MouseEvent } from "react";

import { siteLinks } from "./siteConfig";

function sectionLink(event: MouseEvent<HTMLAnchorElement>, id: string): void {
  event.preventDefault();
  document.getElementById(id)?.scrollIntoView();
}

export function RulesPage() {
  return (
    <main className="page-shell rules-page">
      <a href={siteLinks.home} className="wordmark">Dracula</a>
      <article>
        <header>
          <p className="eyebrow">Rules reference</p>
          <h1>How to play</h1>
          <p>
            Dracula is a six-round game for two players using a standard deck and two
            Jokers, called Vampires. Build a shared 3×3 coffin and outscore your opponent.
          </p>
        </header>

        <nav className="rules-contents" aria-label="Rules contents">
          <a href="#/rules" onClick={(event) => sectionLink(event, "setup")}>Setup</a>
          <a href="#/rules" onClick={(event) => sectionLink(event, "playing")}>Playing</a>
          <a href="#/rules" onClick={(event) => sectionLink(event, "scoring")}>Scoring</a>
          <a href="#/rules" onClick={(event) => sectionLink(event, "winning")}>Winning</a>
        </nav>

        <section id="setup">
          <h2>Setup and dealing</h2>
          <p>
            Choose Queen to score horizontal rows or King to score vertical columns.
            Dracula takes the other direction. Directions remain fixed for all six rounds,
            and the dealer alternates each round.
          </p>
          <ol>
            <li>Shuffle all 54 cards, including both Vampires.</li>
            <li>Deal two cards to the non-dealer, then two to the dealer.</li>
            <li>Repeat that pair deal once, giving each player four cards.</li>
            <li>Place the next card face up in the center of the coffin.</li>
          </ol>
          <p>There is no draw during a round. Each hand remains private.</p>
        </section>

        <section id="playing">
          <h2>Playing a round</h2>
          <p>
            The non-dealer plays first. Players alternate until the eight empty coffin
            positions are filled. On each turn, place one card from your hand in an empty
            position that shares an edge with a card already in the coffin. Diagonal contact
            alone is not enough. Vampires follow the same placement rule as every other card.
          </p>
        </section>

        <section id="scoring">
          <h2>Scoring</h2>
          <p>
            Queen scores the three rows. King scores the three columns. Add the three card
            values in each line, then apply the highest applicable multiplier.
          </p>
          <div className="rules-table-scroll" tabIndex={0} role="region" aria-label="Directional card values">
            <table>
              <thead><tr><th>Card</th><th>Row</th><th>Column</th></tr></thead>
              <tbody>
                <tr><th>Ace</th><td>1</td><td>1</td></tr>
                <tr><th>2–10</th><td>Face value</td><td>Face value</td></tr>
                <tr><th>Jack</th><td>0</td><td>0</td></tr>
                <tr><th>Queen</th><td>10</td><td>0</td></tr>
                <tr><th>King</th><td>0</td><td>10</td></tr>
              </tbody>
            </table>
          </div>
          <h3>Multipliers</h3>
          <ul>
            <li>Three cards of one suit: ×5</li>
            <li>Otherwise, three cards of one color: ×3</li>
            <li>Otherwise, at least two cards of one suit: ×2</li>
            <li>Otherwise: ×1</li>
          </ul>
          <p>
            Only the highest applicable multiplier is used. A line containing a Vampire
            scores zero. A zero-valued face card still contributes its suit and color.
          </p>
          <h3>Selecting the round score</h3>
          <p>
            Sort each player&apos;s three line scores from highest to lowest. If the highest
            scores differ, each player records that score. If they tie, compare the second
            scores; if those also tie, record the third scores even when they tie as well.
          </p>
        </section>

        <section id="winning">
          <h2>Ending the game</h2>
          <p>
            After each round, add both selected round scores to the running totals. The
            other player deals the next round from the remaining stock. After six rounds,
            the higher total wins. A tied total is decided by the higher sixth-round score;
            if that also ties, the game is tied.
          </p>
        </section>

        <footer className="rules-sources">
          <h2>Sources</h2>
          <ul>
            <li><a href="https://www.parlettgames.uk/oricards/dracula.html">David Parlett&apos;s rules</a></li>
            <li><a href="https://kevan.org/fdgp/index.php?view=2#Dracula">Freeze-Dried Games Pack summary</a></li>
          </ul>
        </footer>
      </article>
    </main>
  );
}

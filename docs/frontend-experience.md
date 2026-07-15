# Frontend experience

The React, TypeScript, and Vite frontend provides a responsive game window, a
dedicated rules page, direct card interaction, scoring presentations, and
Dracula commentary. The browser renders `HumanGameView` and submits
server-issued legal move IDs. It does not shuffle, score, decide legality, or
receive private agent or deck data.

## Page structure

The game window contains two visual regions:

- The **main display** contains the coffin, human hand, score tally, turn state,
  and scoring presentation.
- The **commentary panel** is visually presented as a chat bar but has no text
  input. It contains Dracula artwork and model-generated commentary.

Rules, About, and Contact links appear below the game window. Rules opens the
dedicated rules page in a new browser tab. About links to the project article;
Contact links to the project owner's homepage.

## Start and game flow

The initial view presents two primary actions: **Start as Queen** and **Start as
King**. It also provides the Rules link. No game exists and no narrator request
is made until the user selects a role.

If the final model configuration exposes multiple public profiles, its
allowlisted selector appears with the role controls. If the MVP has one public
profile, the resolved default is used without showing a redundant control.

The interaction sequence is:

1. The user may open Rules without leaving the start view.
2. The user selects Queen or King, and the browser creates the game with that
   role and the selected or default model profile.
3. The browser displays the dealt round and awaits the required round-opening
   comment before enabling card play.
4. Human and agent turns alternate. Turn status makes clear when the user may
   act and when Dracula is deciding.
5. After the eighth move, the main display runs the row and column scoring
   sequence before advancing.
6. Rounds two through six repeat the opening-comment, play, and scoring flow.
7. The final view retains the completed coffin and scores, displays the outcome
   and closing comment, and offers **Start new game**.

Reloading an active game restores the authoritative state and resumes any
pending agent, narration, scoring, or round-advance phase described by the API.

## Desktop layout

At full desktop width, the main display occupies approximately two-thirds to
three-quarters of the game window. The commentary panel occupies the remaining
width and matches the main display's height.

```text
+------------------------------------------------+------------------+
| MAIN DISPLAY                                   | DRACULA ART       |
|                                                | fixed art region  |
|  +-------------------------------+  +-------+  |------------------|
|  |                               |  | score |  | commentary       |
|  |          3×3 coffin           |  | tally |  | stream           |
|  |                               |  |       |  |                  |
|  +-------------------------------+  +-------+  | scrollable       |
|              human hand                       |                  |
+------------------------------------------------+------------------+
```

The main display uses a fixed design aspect ratio constrained by both available
width and viewport height. The coffin begins near the upper-left of the display
and uses most of its height. The score tally occupies the space between the
coffin and the right boundary. The human hand is centered beneath the coffin.
Cards, spacing, and text resize within the container; the interface is not
scaled as a single bitmap.

The commentary panel reserves a fixed region at the top for the Dracula art
asset. The commentary stream fills the remaining height and scrolls so the
desktop user can review earlier comments. New comments append at the bottom.

Scoring highlights, values, multipliers, comparisons, and totals appear only in
the main display. Its geometry and the commentary panel remain stable during
the animation.

## Narrow and mobile layout

When the window cannot preserve usable card and score sizes, the game switches
to the narrow layout. The breakpoint is determined by the minimum viable main
display width rather than a device category. A narrow desktop window uses this
same arrangement.

```text
+----------------------------------+
| avatar | latest Dracula comment  |
+----------------------------------+
| score tally                      |
+----------------------------------+
|                                  |
|           3×3 coffin             |
|                                  |
+----------------------------------+
|           human hand             |
+----------------------------------+
```

The mobile commentary area appears above the main display. It shows a small
Dracula avatar to the left of only the latest narrator message; it has no
scrolling transcript. The space is sized for the configured maximum commentary
length so ordinary message changes do not shift the coffin.

Within the main display, the score tally appears first, followed by the coffin
and then the human hand. The layout supports the 360-pixel minimum viewport in
the release-ready definition. Short viewports may scroll vertically rather than
making cards or text unreadable.

## Card interaction

The human plays by dragging a card from their hand onto the coffin. Legal drop
targets are derived from legal moves in `HumanGameView`. A valid drop maps to the
corresponding server-issued `move_id`. Invalid positions do not accept the drop,
and the hand is not updated until the server accepts the move.

Touch and keyboard users can perform the same spatial action by selecting a
card and then selecting a highlighted legal coffin position. This is a direct
grid interaction, not a textual list of legal moves. Focus returns to a stable
game control after the server response.

While a move request is pending, further card input is disabled. A rejected or
stale move restores interaction from the returned authoritative view. During an
agent turn, the hand remains visible but inactive and the turn status indicates
that Dracula is deciding.

The frontend does not display a global inventory of seen, hidden, or played
cards. The human's card information is conveyed by their current hand and the
cards currently visible in the coffin.

## Commentary behavior

The commentary panel becomes active only after game creation. Desktop retains
the full commentary stream for the current game. Narrow and mobile layouts show
only the newest comment.

Optional move comments arrive without interrupting card or agent-turn progress.
Required round-opening and scoring comments follow the bounded presentation
waits defined in [architecture](architecture.md#narrator-scheduling). A failed
narrator job ends its wait without inserting substitute Dracula dialogue.

New commentary is announced as a polite live-region update without moving
keyboard focus. A delayed optional comment is shown only if the architecture's
event-order and current-round rules still admit it.

## Round-scoring presentation

At the end of each round, the interface runs an ordered scoring sequence supplied
by the game service. It highlights each Queen row and King column, then displays:

1. Card values and their base total.
2. The applicable suit or color multiplier.
3. The resulting line score.
4. Comparisons between each player's ranked lines.
5. Any move to second- or third-ranked lines caused by ties.
6. The awarded round scores and updated game totals.

The sequence displays deterministic `ScoringStep` data and does not reproduce
scoring logic in the browser. It has two narrator touchpoints:

1. Start row commentary when row animation begins. After the three rows and
   provisional Queen ranking are shown, wait for and append the comment.
2. Start column commentary when column animation begins. After the three
   columns, cross-player tie resolution, awarded scores, and updated totals are
   shown, wait for and append the comment.

Both waits have a configured upper bound; a pending or failed narrator response
cannot prevent round advance. The main display prevents card interaction until
the scoring sequence completes.

## Loading, failure, and recovery

- A game-creation failure leaves the selected role visible and offers retry.
- A pending human move preserves the displayed hand and coffin until accepted.
- A pending agent turn shows its status and resumes the same claimed turn after
  reload or retry.
- A stale-state response replaces the local view with the server response before
  accepting more input.
- Narrator failure does not alter game state or prevent game completion.
- A failed state request preserves the last confirmed display and provides a
  retry action rather than applying an optimistic game change.

## Frontend structure

The principal UI responsibilities are `GameStart`, `GameWindow`, `MainDisplay`,
`CardGrid`, `Hand`, `Scoreboard`, `ScoringPresentation`, `CommentaryPanel`, and
`RulesPage`, supported by an API client and game store. These are component
boundaries, not a requirement that every responsibility occupy a separate
module.

CSS Grid provides the outer desktop/narrow layouts and the coffin. Container
queries, `aspect-ratio`, and bounded fluid sizing support internal scaling. Card
assets will use `@letele/playing-cards` or an equivalent vendored SVG set if
bundling requires it. No drag-and-drop library has been selected; the choice
must support pointer, touch, and keyboard behavior described above.

> **TODO UX-002 — Define usability acceptance.** Choose the desktop aspect ratio,
> functional narrow-layout breakpoint, minimum card and text sizes, supported
> viewports, Dracula and card assets, focus details, contrast, and selection
> feedback. Record the final About and Contact URLs.
>
> **Complete when:** A testable checklist and asset/license decision cover all 54
> cards, supported layouts, direct-grid input methods, commentary variants, and
> external links.

> **TODO UX-004 — Design the scoring sequence.** Specify timing, highlighting,
> transitions, skip/replay behavior, narrator insertion points, and the handoff to
> the next round or final result.
>
> **Complete when:** A reviewed storyboard covers ordinary multipliers, Vampire
> lines, first- and second-level ties, round totals, and final-game scoring.

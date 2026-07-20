# Frontend experience

The React, TypeScript, and Vite frontend provides a responsive game window, a
dedicated rules page, direct card interaction, scoring presentations, and
Dracula commentary. The browser renders `HumanGameView` and submits
server-issued legal move IDs. It does not shuffle, score, decide legality, or
receive private policy, opponent-hand, or deck data.

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

Game creation resolves the production policy and narrator configuration.
The MVP does not expose a model or difficulty selector.

The interaction sequence is:

1. The user may open Rules without leaving the start view.
2. The user selects Queen or King, and the browser creates the game with that
   role.
3. The browser displays the dealt round and awaits the required round-opening
   comment before enabling card play.
4. Human and opponent turns alternate. Turn status makes clear when the user may
   act and when Dracula is deciding.
5. After the eighth move, the main display runs the row and column scoring
   sequence before advancing.
6. Rounds two through six repeat the opening-comment, play, and scoring flow.
7. The final view retains the completed coffin and scores, displays the outcome
   and closing comment, and offers **Start new game**.

Reloading an active game restores the authoritative state and resumes any
pending opponent, narration, scoring, or round-advance phase described by the
API.

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
opponent turn, the hand remains visible but inactive and the turn status
indicates that Dracula is deciding.

The frontend does not display a global inventory of seen, hidden, or played
cards. The human's card information is conveyed by their current hand and the
cards currently visible in the coffin.

## Commentary behavior

The commentary panel becomes active only after game creation. Desktop retains
the full commentary stream for the current game. Narrow and mobile layouts show
only the newest comment.

Optional move comments arrive without interrupting card or opponent-turn
progress.
Required round-opening and scoring comments follow the bounded presentation
waits defined in [architecture](architecture.md#narrator-scheduling). A failed
narrator job ends its wait without inserting substitute Dracula dialogue.

New commentary is announced as a polite live-region update without moving
keyboard focus. A delayed optional comment is shown only if the architecture's
event-order and current-round rules still admit it.

## Round-scoring presentation

The game service supplies the complete ordered `ScoringStep` sequence. The
browser presents those steps and does not recalculate values, multipliers,
rankings, tie resolution, round scores, or cumulative totals.

### Entering scoring

After the eighth move, card interaction is disabled. The score tally and human
hand fade out, and the completed coffin enlarges into the scoring layout. On
desktop, the tall coffin sits beside a white calculation workspace. In the
narrow layout, the same workspace appears below the coffin so the calculation
does not reduce card size.

The dealer is scored first and the non-dealer second. Because either player may
be Queen or King, the sequence supports both row-then-column and
column-then-row ordering. The human calculation is headed **Your Score** and the
opponent calculation is headed **My Score**, regardless of who dealt.

The narrator job for the current orientation starts with its animation. After
the three line scores are sorted, the browser performs the bounded wait and
appends the comment to the unchanged commentary panel. The completed dealer
tally then fades from the calculation workspace while the coffin remains and
the non-dealer calculation begins.

### Calculating one orientation

Queen rows are processed from top to bottom, with cards read left to right. King
columns are processed from left to right, with cards read top to bottom. The
current three-card series receives a bold accent border for the duration of its
calculation.

Within a series, each card briefly enlarges and immediately returns to its
normal size. Its direction-specific value appears in one horizontal expression
at the peak of the motion:

```text
X
X + Y
X + Y + Z
```

When the third value is present, the expression collapses horizontally toward
its center and fades seamlessly into the base sum `W`. A ripple pop passes over
the three cards in reading order during the collapse.

The service-supplied multiplier description then appears immediately to the
right of `W`:

| Multiplier | Description | Card emphasis |
| ---: | --- | --- |
| ×1 | **No Multiplier** | No additional card highlight |
| ×2 | **2× Hearts**, **2× Clubs**, and so on | The two matching-suit cards |
| ×3 | **3× Red** or **3× Black** | All three cards |
| ×5 | **3× Hearts**, **3× Clubs**, and so on | All three cards |
| ×0 | **Vampire** | Every Joker/Vampire in the series |

The emphasized cards receive a distinct inner outline while the bold series
border remains in place. The UI uses `multiplier_label` and
`highlighted_card_ids` from `LineScore`; it does not infer the description or
matching cards.

After a short pause, the description changes to the numeric factor `× V`. After
a second pause, `W × V` collapses horizontally and is replaced by the final
series total `T`. **No Multiplier** therefore becomes `× 1`; **Vampire** becomes
`× 0`. A Vampire is never assigned a directional card value.

After all three series have been calculated, each line retains only `T`. The
three values move to the center of the workspace and then rearrange into
descending order. The service supplies both the sorted order and stable ordering
for equal values.

### Selecting round scores

After both individual tallies are complete, the coffin fades out. **Your Score**
and **My Score** appear side by side, each with its three sorted line values.
Corresponding ranks are compared from highest to lowest:

1. Both values in the current pair pop together.
2. If they differ, **Round Score** appears beside both values and all unselected
   values fade out.
3. If they tie, **Tie** appears and both values receive a strikethrough. The tied
   pair remains visible while the next ranked pair is compared.
4. If the first two pairs tie, the third pair becomes the round score even if
   its values also tie. It receives **Round Score**, not another rejection.

Once the round-score pair is selected, previous tied pairs and all other values
fade away. Under each retained round score, the interface reveals the previous
cumulative total with an addition sign and underline, followed by the new
cumulative total below the line.

The round score, previous total, and new total remain visible. Rounds one through
five end with **Deal Next Round**. Activating it advances the game, deals the
next round, and begins its opening-comment sequence. After round six, the browser
finalizes the game, awaits the closing comment, and shows **Play Again**; the
score calculation remains visible until that action starts a new game.

### Motion and recovery

Card pops have no plateau between expansion and return. Values, result text,
tie labels, and round-score labels appear at the peak of their associated pop.
Exact durations are tuned during implementation; their order and relative
behavior remain fixed.

The MVP has no skip or replay control for scoring. With reduced motion enabled,
scale and movement are replaced by border, opacity, and text-state changes while
every calculation step remains visible. Reloading during scoring restarts the
presentation from the dealer's first series using the stored deterministic
sequence; existing narration jobs are reused and commentary is not duplicated.

Both narrator waits have a configured upper bound. A pending or failed response
cannot prevent the sequence from reaching its completion control.

## Loading, failure, and recovery

- A game-creation failure leaves the selected role visible and offers retry.
- A pending human move preserves the displayed hand and coffin until accepted.
- A pending opponent turn shows its status and resumes the same claimed turn after
  reload or retry.
- A stale-state response replaces the local view with the server response before
  accepting more input.
- Narrator failure does not alter game state or prevent game completion.
- A failed state request preserves the last confirmed display and provides a
  retry action rather than applying an optimistic game change.

## Frontend structure

The principal UI responsibilities are `GameStart`, `GameWindow`, `MainDisplay`,
`CardGrid`, `Hand`, `Scoreboard`, `ScoringPresentation`, `CommentaryPanel`, and
`RulesPage`, supported by an API client and game store.

CSS Grid provides the outer desktop/narrow layouts and the coffin. Container
queries, `aspect-ratio`, and bounded fluid sizing support internal scaling. Card
assets use the Kenney Playing Cards Pack described below. No drag-and-drop
library has been selected; the choice must support pointer, touch, and keyboard
behavior described above.

## Usability acceptance

The initial desktop main-display aspect ratio is `4 / 3`. The desktop layout
uses the existing two-pane proportions until the game window is narrower than
900 CSS pixels, at which point it switches to the narrow layout. The breakpoint
may be adjusted during implementation only when the same acceptance cases show
that cards, scores, or commentary no longer fit at the documented minimum size.

Card image boxes do not render below 64×64 CSS pixels. The visible pixel-art
card remains centered within that box. Body and control text is at least 16 CSS
pixels; secondary labels are at least 14 CSS pixels. Short viewports scroll
instead of reducing either minimum.

The responsive checks cover:

- 360×640 and 390×844 mobile viewports.
- 768×1024 portrait and 1024×768 landscape viewports.
- One pixel below and above the configured narrow-layout breakpoint.
- 1440×900 desktop.

At each size, all four hand cards and the complete coffin remain reachable; card
labels, scores, and turn state remain readable; interactive regions do not
overlap; and the latest mobile commentary is not clipped or given an internal
scrollbar.

Drag, tap-selection, and keyboard-selection paths must submit the same legal
move. A selected card has a persistent outline and raised state. Legal coffin
positions use a second visible treatment that does not depend on color alone.
Pending and inactive controls are visually distinct. Keyboard focus uses a
high-contrast outline at least two CSS pixels wide with visible separation from
the component edge, and focus follows the behavior documented under card
interaction.

Normal text meets a 4.5:1 contrast ratio. Large text, focus indicators, card
selection, and other meaningful interface graphics meet 3:1 against adjacent
colors. Narrator updates remain polite live-region announcements. Reduced-motion
preference disables nonessential movement without skipping scoring information.

## Asset decision

The selected deck is the [Kenney Playing Cards Pack](https://www.kenney.nl/assets/playing-cards-pack),
version 1.0, released under CC0. The 64×64 PNG files in the local
`Cards (large)` directory are the source for the initial implementation. The
`Cards (medium)` directory is retained only as an alternate source and is not
selected automatically at narrower layouts.

Both source directories are ignored by Git and treated as immutable. An asset
preparation step copies only the required files into the frontend asset set and
does not modify the source directories. The copied set contains the 52 suited
cards plus `card_joker_black.png` and
`card_joker_red.png`, which map directly to the two Vampire card IDs. Jokers and
Vampires are the same cards. A later custom Vampire-themed Joker may replace
both images without changing game state or card semantics.

Cards render with nearest-neighbor scaling and do not use interpolation that
softens the pixel art. The asset mapping, Kenney source URL, pack version, and
CC0 license are recorded with the copied frontend assets.

The Dracula portrait and mobile avatar are project-provided artwork. Their final
files do not block layout or interaction implementation: the desktop reserves
its fixed portrait region and mobile uses a square placeholder with the same
eventual dimensions. Before release, the supplied image must remain clear at
both crops and include a recorded source and usage status.

The Rules, About, and Contact destinations are configuration values. All three
must resolve correctly, and the Rules link opens a new tab.

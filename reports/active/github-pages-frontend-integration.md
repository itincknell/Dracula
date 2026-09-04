# GitHub Pages frontend integration

## Result

The user-approved React interface now runs against the production stateless API
without changing its desktop or mobile layout. The production build is rooted
at `/Dracula/`, calls `https://api.ian-tincknell.com`, and contains no local API
URL. No Pages site, DNS record, AWS resource, commit, or Git remote changed.

## Browser state and recovery

The production controller writes one local-storage entry,
`dracula.recovery-envelope`. Its JSON contains exactly the initial seed and the
ordered history returned by accepted API responses. The UI does not persist a
game projection, hand, engine object, model tensor, logit, mask, or search
diagnostic.

Starting a game stores the first accepted envelope. A placement or round
advance replaces it only after the stateless command route accepts the command.
Reload posts the same envelope to `/games/resume`. Malformed local JSON or an
invalid envelope is removed locally before any request. Warm replay and a new
browser controller produce the same projected game; the Stage 3 container test
also established equality after a server-process restart.

The legacy stateful controller and API remain intact for historical/local
tests. The application default is the separately implemented stateless
controller, with no fallback from one mode to the other.

## Narration cadence

Narration is independent of gameplay mutation:

- Opening narration starts after accepted game creation and may appear during
  the first turn.
- One round-transition request starts on receipt of a completed round for
  rounds 1–5. Its ready text remains hidden until the scoring timeline reaches
  its completed frame.
- Round 6 starts no transition request. The final-result request starts only
  after the game-completing advance command returns.

Unavailable, rejected, or delayed narration neither changes the stored envelope
nor blocks a move, scoring, round advance, or completed game. The commentary
surface remains empty rather than inventing fallback dialogue.

## GitHub Pages build

Vite emits all scripts, styles, fonts, cards, and the portrait beneath
`/Dracula/`. The application uses `#/`, `#/game`, and `#/rules` under the one
Pages entry point, so direct refresh needs no client router or duplicate 404
document. Rules retains its new-tab behavior.

`.github/workflows/pages.yml` runs on matching pull requests and manual
dispatch. Both paths install with `npm ci` and run unit tests, TypeScript,
ESLint, and production-build verification. Pull requests cannot deploy. A
manual dispatch publishes only `frontend/dist` when its boolean `publish` input
is true. The workflow has no AWS credential or AWS deployment step.

## Verification

- Frontend unit/component tests: **12 files, 84 tests passed**.
- TypeScript: passed.
- ESLint: passed.
- Production Vite build and post-build verifier: passed.
- Production browser suite: **2 tests passed**.
- A real stateless `pi1` browser game completed all six rounds and restored the
  same API response at opening, human-turn, scoring, and final-result stages.
- Opening, rounds 1–5 transitions, round-6 exclusion, final-result timing, and
  narration-unavailable continuation passed.
- Desktop 1440×900 and mobile 390×844/390×640 geometry remained contained.
- A DOM-driven mobile move retained the exact scroll position.
- Rules opened `/Dracula/#/rules` in a new tab.
- The compiled build contains `/Dracula/` asset URLs and
  `https://api.ian-tincknell.com`; it contains no `localhost`, `127.0.0.1`, or
  IPv6 loopback URL.

Local commands:

```bash
make pages-build
make pages-test
```

Explicit future production publication:

```bash
gh workflow run pages.yml -f publish=true
```

## Later decisions

⭕ USER DECISION LATER: Confirm the final public path capitalization: `/Dracula/` exactly as currently locked.

⭕ USER DECISION LATER: Approve any short public disclosure that client-visible seed/history permits deliberate inspection of hidden cards.

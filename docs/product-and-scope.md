# Product and scope

## North star

Deliver a clear, polished, publicly reachable six-round Dracula card game with
correct deterministic rules, one selected opponent, a dedicated rules page,
and sparse in-character narration.

## Release goals

- Responsive single-player play from 360-pixel mobile through desktop.
- Queen and King starts, six complete rounds, final result, and Play Again.
- Server-owned dealing, legality, transitions, forced moves, and scoring.
- Standalone `pi1` for every non-forced Dracula move at one fixed difficulty.
- Three Bedrock narration cue classes: opening, rounds 1–5 transitions, and
  final result.
- Static React frontend at `https://ian-tincknell.com/Dracula/`.
- Stateless FastAPI and engine execution in AWS Lambda.
- Client recovery from the initial game seed and accepted command history.

## Accepted tradeoffs

The browser-held seed permits an inspecting user to reconstruct Dracula's hand
and future stock. This is acceptable for a noncompetitive single-player game.
There is no production database, authoritative anti-cheat layer, cross-device
recovery, account history, or aggregate result store.

## Non-goals

- Accounts, login, matchmaking, multiplayer, leaderboards, or prizes.
- Server-persistent games or narration.
- SageMaker, gameplay search, hybrid inference, or selectable difficulty.
- LLM move selection or model control of rules and scoring.
- Per-move narration, arbitrary chat, narrator tools, or long-term memory.
- A claim of perfect or game-theoretically optimal play.

## Release-ready definition

The release is ready when:

- The combined FastAPI frontend and API are reachable at `/Dracula/` over
  HTTPS through the Cloudflare-managed domain.
- A first-time user can open Rules, play as either role, finish six rounds, and
  start another game.
- Cache-hit and cache-miss replay produce identical game views and `pi1` moves.
- Reload reconstructs the most recently stored browser envelope.
- Every selected `pi1` move is legal and deterministically resolves symmetry.
- Bedrock receives only public cue data and failure never blocks gameplay.
- Round 6 produces one final-result cue and no round-transition cue.
- Accessibility, responsive layout, engine, API, frontend, privacy, and
  deployment smoke suites pass against the release image.
- Logs, alarms, rate limits, budget alerts, rollback, and teardown are active.

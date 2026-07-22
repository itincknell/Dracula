# Product and scope

## North star

Deliver a clear, polished web game with correct deterministic rules and a
dedicated rules page. The strongest validated information-safe controller
chooses Dracula's moves at one fixed difficulty; a separate narrator supplies
the public voice. Evaluation measures strategic competence, legality,
information boundaries, reliability, latency, and cost.

## MVP goals

- A deployed, responsive, single-player game lasting six rounds.
- A 54-card deck, one shared 3×3 coffin, hands, round scores, and total scores.
- A dedicated rules page and public Dracula commentary at game, move, scoring,
  and completion events.
- Server-owned legal moves, state transitions, and scoring.
- One strongest validated search or search-guided controller for every
  non-forced opponent move.
- Reproducible local search, search-guided training, and absolute controller
  comparison.
- A reusable FastAPI and React application whose deployment topology is chosen
  after opponent latency and memory are measured.

## Non-goals

- Accounts, login, matchmaking, or multiplayer.
- Strands, AgentCore, LLM move selection, or tool-using gameplay agents.
- Hand-authored move rankings or scoring shortcuts outside the rules engine.
- Long-term player memory or arbitrary public model selection.
- A claim of perfect or game-theoretically optimal play.
- Model control of rules, scoring, validation, or persistent state.

## Acceptance

A user can reach the deployed app, consult the rules, and finish a game without
manual state repair. Every non-forced opponent move uses the pinned controller,
the narrator receives no private state, failures cannot corrupt the game, and
gameplay and opponent-development evidence retain resolved versions and seeds.

## Release-ready definition

The MVP is release-ready when:

- It is publicly reachable at a stable HTTPS URL.
- A first-time user can open the rules, start a game as Queen or King, finish all
  six rounds, see the final result, and start another game.
- The interface is usable from a 360-pixel-wide phone through a standard desktop
  display using current Chrome, Firefox, Safari, or Edge.
- The deployed opponent uses the selected search, guided-search, or distilled
  model configuration for every non-forced move, and the separate narrator
  produces visible commentary.
- Rules, scoring, legal-move validation, state visibility, and stale-request
  handling pass automated tests derived from the authoritative rules.
- Policy or narrator failure cannot corrupt state or expose private information.
- The selected controller has reproducible search, training when applicable,
  and absolute comparison evidence. Each game records its game seed, controller
  version and configuration, narrator model, and prompt version.
- A deployment smoke test loads the frontend, passes `/health`, creates a game,
  completes a seeded game through the deployed integrations, and confirms useful
  logs and traces.
- Rate limits and AWS budget alerts are active.
- Deployment and teardown instructions work from a clean environment.

# Product and scope

## North star

Deliver a clear, polished web game with correct deterministic rules and a
dedicated rules page. A recurrent neural policy trained through reproducible
self-play chooses Dracula's moves; a separate Bedrock narrator supplies the
public voice. Evaluation measures policy competence, legality, reliability,
latency, and cost. Platform defaults, customizations, and omissions are
documented with their rationale.

## MVP goals

- A deployed, responsive, single-player game lasting six rounds.
- A 54-card deck, one shared 3×3 coffin, hands, round scores, and total scores.
- A dedicated rules page and public Dracula commentary at game, move, scoring,
  and completion events.
- Server-owned legal moves, state transitions, and scoring.
- A versioned recurrent policy trained by the project and served for every
  opponent move.
- A reproducible self-play training suite with model evaluation and promotion.
- FastAPI on Lambda, DynamoDB, SageMaker Serverless Inference, SageMaker Model
  Registry, Bedrock narration, and observability.

## Non-goals

- Accounts, login, matchmaking, or multiplayer.
- Strands, AgentCore, LLM move selection, or tool-using gameplay agents.
- Hand-authored strategy Oracles or a deterministic live-play substitute for
  the neural policy.
- Long-term player memory or arbitrary public model selection.
- Perfect or game-theoretically optimal play.
- Model control of rules, scoring, validation, or persistent state.

## Acceptance

A user can reach the deployed app, consult the rules, and finish a game without
manual state repair. Every opponent move uses the pinned registered policy, the
narrator receives no private state, failures cannot corrupt the game, and
gameplay and policy-training evidence retain their resolved versions and seeds.

## Release-ready definition

The MVP is release-ready when:

- It is publicly reachable at a stable HTTPS URL.
- A first-time user can open the rules, start a game as Queen or King, finish all
  six rounds, see the final result, and start another game.
- The interface is usable from a 360-pixel-wide phone through a standard desktop
  display using current Chrome, Firefox, Safari, or Edge.
- The deployed opponent uses an approved SageMaker Model Registry version
  through SageMaker Serverless Inference for every move, and the separate
  Bedrock narrator produces visible commentary.
- Rules, scoring, legal-move validation, state visibility, and stale-request
  handling pass automated tests derived from the approved rules.
- Policy or narrator failure cannot corrupt state or expose private information.
- The approved policy has reproducible training and competence evidence. Each
  game records its game seed, policy version, inference configuration, narrator
  model, and prompt version.
- A deployment smoke test loads the frontend, passes `/health`, creates a game,
  completes a seeded game through the deployed integrations, and confirms useful
  logs and traces.
- Rate limits and AWS budget alerts are active.
- Deployment and teardown instructions work from a clean environment.

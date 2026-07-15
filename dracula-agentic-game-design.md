# Dracula Agentic Card Game — Starter Design

> **Status:** Historical starter proposal. The current design is maintained in
> [`docs/`](docs/README.md).

## 1. Goal

Build a deployed single-player web game for **Dracula**, with:

- A human player versus an LLM-controlled opponent.
- Six rounds using a 54-card deck and one shared 3×3 coffin grid.
- Deterministic rules, legal-move validation, scoring, round progression, and cumulative scoring.
- A silent decision agent that uses game tools to choose moves.
- A separate narrator that produces visible, public-safe commentary.
- A dedicated rules page linked from the game.
- Configurable foundation models and a repeatable evaluation harness.

**Important:** Before coding the engine, create `docs/rules.md` from the user's authoritative rules. Do not infer or substitute rules from another game named Dracula.

## 2. MVP scope

### Included

- Complete playable game.
- Responsive 3×3 coffin and card-hand display.
- Running round and total scores.
- Dedicated rules page.
- Public Dracula commentary for game, move, scoring, and completion events.
- Bedrock model selection through named configuration profiles.
- Strands decision agent deployed to AgentCore Runtime.
- DynamoDB game-state persistence.
- FastAPI HTTP API deployed on AWS Lambda.
- CloudWatch and AgentCore traces.
- Headless automated model-evaluation runner.

### Excluded from MVP

- Accounts, social login, matchmaking, or multiplayer.
- MCP, A2A, AgentCore Memory, Gateway, or Knowledge Bases.
- Unrestricted public selection of arbitrary Bedrock model IDs.
- Long-term player memory.
- Perfect or game-theoretically optimal play.

## 3. Architecture

```text
Browser
  React + TypeScript + Vite
  Coffin, scores, narrator log, rules page
        |
        | HTTPS
        v
API Gateway HTTP API
        |
        v
AWS Lambda
  FastAPI via Mangum
  Authoritative game service
  Legal moves, state transitions, scoring
        |
        +---- DynamoDB
        |       Game state and event log
        |
        +---- AgentCore Runtime
                Strands decision agent
                Bedrock decision model
                Deterministic game-analysis tools
        |
        +---- Amazon Bedrock
                Narrator model call using public state only

CloudWatch / AgentCore Observability
  Logs, traces, latency, token usage, failures

S3
  Evaluation JSONL results and aggregate reports
```

### Why Lambda

Lambda runs backend code only when an HTTP request arrives. It avoids maintaining an always-on server. `Mangum` adapts the FastAPI ASGI application to API Gateway/Lambda.

## 4. Separation of responsibilities

### React frontend

- Render public game state only.
- Display the player hand, shared coffin, scores, round, turn, and narrator log.
- Present legal card placements through drag-and-drop and submit the matching
  server-issued move ID.
- Link to the dedicated rules page.
- Never shuffle, score, reveal hidden cards, or decide legality.

### FastAPI game service

- Own the authoritative state machine.
- Shuffle and deal cards.
- Produce human, agent, and public views.
- Generate legal moves.
- Validate and apply moves.
- Calculate round and cumulative scores.
- Use optimistic concurrency to reject stale requests.
- Invoke the decision agent when it is the agent's turn.
- Emit public narrator events after accepted moves and during game opening,
  round scoring, round completion, and game completion.
- Persist state and event records.

### Strands decision agent

- Receive a sanitized private agent view, never the full server state.
- Use deterministic tools to inspect the state and compare legal actions.
- Return one `move_id`.
- Produce no user-visible prose.
- Never write directly to DynamoDB.
- Never apply a move; the FastAPI service validates and applies it.

### Narrator

- Receive only public state, the accepted move, rule effects, and score changes.
- Produce short in-character Dracula commentary for events selected by the
  deterministic cadence gate.
- Never receive either player's hidden cards or the remaining deck.
- Avoid revealing private strategic analysis.

## 5. State and visibility model

Define three projections from one authoritative state:

```python
AuthoritativeGameState  # server only
HumanGameView           # public state + human hand
AgentGameView           # public state + agent hand
PublicGameView          # no private hands or deck order
```

Core state:

```python
class GameState(BaseModel):
    game_id: UUID
    version: int
    seed: str
    status: Literal["playing", "round_complete", "game_complete"]
    round_number: int
    active_player: Literal["human", "agent"]
    dealer: Literal["human", "agent"]

    stock: list[Card]                   # ordered, server only
    human_hand: list[Card]             # hidden from agent model
    agent_hand: list[Card]             # hidden from browser/narrator
    coffin: list[Card | None]           # shared nine-position grid

    completed_rounds: list[RoundRecord]
    total_scores: Score
    narrator_events: list[NarratorEvent]
    created_at: datetime
    updated_at: datetime
```

Resolved model and prompt configuration is persisted in the surrounding game
session record rather than the pure engine state. Card status is derived from
the stock, hands, coffin, and completed rounds.

All random operations must support a seed for reproducible tests and evaluation.

## 6. DynamoDB design

Use one table, on-demand capacity:

```text
PK = GAME#{game_id}
SK = STATE
```

Optional event records:

```text
PK = GAME#{game_id}
SK = EVENT#{sequence:04d}
```

State writes use a conditional expression:

```text
expected version == stored version
```

Increment `version` after every accepted action. Add TTL to abandoned games.

## 7. API

```text
POST /games
  Create a seeded game using an allowlisted model profile.

GET /games/{game_id}
  Return HumanGameView.

POST /games/{game_id}/moves
  Submit a human move with expected_version.

POST /games/{game_id}/agent-turn
  Internal or authenticated endpoint; invoke and apply one agent move.

GET /model-profiles
  Return safe public profile names and descriptions.

GET /health
  Health check.
```

Recommended move request:

```json
{
  "move_id": "hand:QH->grid:4",
  "expected_version": 12
}
```

The server generates move IDs. The browser maps valid card-and-position drops to
these IDs; neither browser nor model constructs arbitrary card/state mutations.

## 8. Deterministic game engine

Implement the engine as a pure Python package with no AWS or LLM dependencies:

```text
game/
  cards.py
  rules.py
  state.py
  visibility.py
  legal_moves.py
  scoring.py
  transitions.py
  simulation.py
```

Principal functions:

```python
create_game(seed, model_profile) -> GameState
get_legal_moves(state, player) -> list[LegalMove]
apply_move(state, player, move_id) -> GameState
score_round(state) -> RoundScore
advance_round(state) -> GameState
to_human_view(state) -> HumanGameView
to_agent_view(state) -> AgentGameView
to_public_view(state) -> PublicGameView
```

The engine must be fully playable against random and heuristic bots without Bedrock.

## 9. Strands tools

The tools operate on an invocation-local `AgentGameView` snapshot:

- `inspect_game_state()`
- `list_legal_moves()`
- `evaluate_move(move_id)`
- `compare_moves(move_ids)`
- `select_move(move_id)`

`select_move` records the proposed move in the invocation result; it does not mutate persistent state.

Tool outputs should be concise and structured. The server still validates the returned move against the current authoritative state.

## 10. Model configuration

Model choice belongs at the **game/experiment profile level**, not inside tool code.

Example `config/model_profiles.yaml`:

```yaml
profiles:
  nova-lite-balanced:
    decision_model_id: us.amazon.nova-2-lite-v1:0
    narrator_model_id: us.amazon.nova-2-lite-v1:0
    temperature: 0.2
    reasoning_effort: low
    max_output_tokens: 300
    prompt_version: decision-v1
    narrator_prompt_version: narrator-v1
    strategy_mode: heuristic-assisted

  nova-lite-raw:
    decision_model_id: us.amazon.nova-2-lite-v1:0
    narrator_model_id: us.amazon.nova-2-lite-v1:0
    temperature: 0.2
    reasoning_effort: medium
    max_output_tokens: 300
    prompt_version: decision-v1
    narrator_prompt_version: narrator-v1
    strategy_mode: model-only
```

Rules:

- Public users select an allowlisted profile name.
- Evaluation jobs may override model ID and inference settings through a separate authenticated CLI/config file.
- Persist the complete resolved model configuration with each game/evaluation.
- Record model ID, prompt version, inference settings, seed, and code version in every result.
- Decision and narrator models are independently configurable.
- The narrator is disabled during bulk strategic evaluations unless narration is the feature being tested.

## 11. Strategy and difficulty

Implement deterministic baselines first:

- `random`: uniform legal move.
- `heuristic`: hand-authored scoring of candidate moves.
- `search`: heuristic plus limited rollout simulation.
- `llm`: model selects using tools.
- `llm_heuristic`: model receives deterministic candidate features/scores.

Difficulty should be a strategy configuration, not merely higher temperature:

```yaml
difficulty:
  easy:
    policy: random_top_k
    top_k: 8
  medium:
    policy: llm_heuristic
    rollout_count: 0
  hard:
    policy: llm_heuristic
    rollout_count: 100
```

## 12. Narrator contract

Input:

```python
class NarratorInput(BaseModel):
    event_id: str
    trigger: Literal[
        "game_started",
        "human_move_accepted",
        "agent_move_accepted",
        "round_scoring",
        "round_completed",
        "game_completed",
    ]
    public_state: PublicGameView
    public_event: PublicGameEvent
    deterministic_rule_effects: list[str]
    recent_commentary: list[str]
```

Output:

```python
class NarratorOutput(BaseModel):
    text: str
```

The deterministic service supplies factual effects. The narrator adds Dracula's
voice rather than recalculating the game. The service does not invoke the
narrator for optional move events rejected by its cadence gate. Every invocation
expects commentary; opening, scoring, round-result, and closing events provide
required commentary opportunities. Narration never blocks state progression.

## 13. Evaluation harness

Provide a CLI:

```bash
python -m eval.run --config experiments/nova-lite-vs-heuristic.yaml
```

Example experiment:

```yaml
name: nova-lite-vs-heuristic-v1
games: 500
seeds:
  start: 1000
  count: 500
agent_profile: nova-lite-balanced
opponent: heuristic
narration: false
parallelism: 5
```

Per-game metrics:

- Win/loss/tie.
- Final score and score differential.
- Invalid proposed moves.
- Tool-call count and sequence.
- Agent retries.
- Decision latency.
- Input/output tokens.
- Estimated model cost.
- Runtime errors and timeouts.
- Seed and model/prompt configuration.

Aggregate metrics:

- Win rate with confidence interval.
- Mean and median score differential.
- Valid-move rate.
- Average cost per game.
- Average latency per move.
- Performance by round and seat/dealer position.
- Pairwise comparison across models/configurations.

Store raw JSONL in S3 and produce a Markdown/CSV summary locally. Use identical seeds when comparing profiles.

## 14. Observability

Attach these attributes to traces and logs:

```text
game_id
round_number
turn_number
model_profile
decision_model_id
prompt_version
seed
move_id
valid_move
input_tokens
output_tokens
latency_ms
estimated_cost
```

Do not log hidden hands, remaining deck order, complete prompts containing private state, or model reasoning text.

## 15. Security and cost controls

- AgentCore Runtime is private; only the Lambda execution role may invoke it.
- DynamoDB is not accessible from the browser.
- Validate every move after the model returns it.
- Maximum one successful decision per turn.
- Maximum tool calls, model retries, output tokens, and wall-clock duration.
- Per-IP/API rate limits before public launch.
- AWS Budget alerts at $25, $50, and $90.
- Short CloudWatch retention during development.
- No arbitrary model IDs from public requests.

## 16. Frontend structure

```text
frontend/
  src/
    components/
      Card.tsx
      CardGrid.tsx
      Hand.tsx
      Scoreboard.tsx
      NarratorLog.tsx
    api/
      client.ts
    state/
      gameStore.ts
```

Use CSS Grid for the coffin. Human card play uses drag-and-drop with legal drop
targets supplied by `HumanGameView`. At round end, an ordered server-supplied
scoring sequence highlights rows and columns, base values, multipliers,
tie-breaking, awarded scores, and total updates. Start with
`@letele/playing-cards` or vendor equivalent SVG card assets if the dependency
causes bundling issues.

## 17. Backend structure

```text
backend/
  app/
    main.py
    api/
      games.py
    game/
      cards.py
      rules.py
      state.py
      visibility.py
      legal_moves.py
      scoring.py
      transitions.py
      simulation.py
    agent/
      runtime.py
      decision_agent.py
      tools.py
      prompts/
        decision-v1.txt
        narrator-v1.txt
    persistence/
      dynamodb.py
    models/
      api.py
      config.py
  tests/
    unit/
    integration/
    golden/
  config/
    model_profiles.yaml
  experiments/
  infra/
```

## 18. Infrastructure

Prefer AWS CDK:

- Amplify Hosting for the static frontend.
- API Gateway HTTP API.
- Lambda for FastAPI.
- DynamoDB on-demand table.
- AgentCore Runtime for the Strands agent.
- IAM roles with least privilege.
- CloudWatch logs and dashboards.
- S3 bucket for evaluation results.
- AWS Budgets alerts.

Deploy all resources in one supported US region when possible.

## 19. Implementation order

1. Write `docs/rules.md`.
2. Implement and unit-test the pure game engine.
3. Add random and heuristic bots.
4. Build the React board, scores, narrator panel, and rules page.
5. Add FastAPI endpoints and local persistence.
6. Add DynamoDB and Lambda deployment.
7. Implement Strands tools and local agent tests.
8. Deploy the agent to AgentCore Runtime.
9. Add narrator calls using public state.
10. Add traces and the evaluation CLI.
11. Run seeded baseline and model comparison batches.

## 20. Acceptance criteria

- A six-round game can be completed without manual state repair.
- All scoring tests derived from `docs/rules.md` pass.
- Browser and narrator never receive the remaining deck or agent hand.
- Decision agent never receives the human hand.
- Every model move is server-validated.
- A failed or invalid model response cannot corrupt game state.
- Model selection and prompt version are configurable and persisted.
- The same seed/configuration is replayable.
- Evaluation output reports quality, legality, latency, tokens, and cost.
- A 500-game evaluation can be stopped and resumed without losing completed results.

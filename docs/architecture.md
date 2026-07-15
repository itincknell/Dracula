# Architecture

## System

```text
React browser
    |
API Gateway -> FastAPI on Lambda
                    |-- DynamoDB state, events, and jobs
                    |-- AgentCore Runtime -> Strands decider -> Bedrock
                    `-- async narrator Lambda -> Bedrock

CloudWatch and AgentCore observability
S3 evaluation results
```

The browser renders public state plus the human hand. FastAPI owns the
authoritative state machine, projections, validation, scoring, persistence, and
model orchestration. The deterministic engine is pure Python with no AWS or LLM
dependency.

## State and visibility

One authoritative state produces three views:

| Data | Human view | Agent view | Public/narrator view |
| --- | --- | --- | --- |
| Coffin grid, round, turn, scores | Yes | Yes | Yes |
| Human hand | Yes | No | No |
| Agent hand | No | Yes | No |
| Deck and order | No | No | No |
| Recipient's legal moves | Human only | Agent only | No |

All random operations accept a seed. Accepted actions increment a version;
DynamoDB conditional writes reject stale requests. Logs exclude hands, deck
order, private prompts, and model reasoning.

### Authoritative schema

```python
class Card:
    # Stable identity used in state, moves, logs, and tests.
    # Examples: "QH", "7S", "V1", and "V2".
    id: str

    # Ace through King, or Vampire.
    rank: Rank

    # Vampires have no suit.
    suit: Suit | None


class GameState:
    # Stable persistence and API identifier.
    game_id: UUID

    # Incremented after every accepted state change.
    version: int

    # Controls the initial shuffle for reproducible games.
    seed: str

    # Animation and presentation state remain frontend concerns.
    status: Literal["playing", "round_complete", "game_complete"]

    # Current round and number of player cards placed in it.
    round_number: int       # 1 through 6
    turn_number: int        # 0 through 8

    # The non-dealer starts each round; the deal alternates by round.
    dealer: Player
    active_player: Player | None  # None while scoring or after game completion.

    # The human chooses a role; the agent always receives the other role.
    human_role: Literal["queen", "king"]
    agent_role: Literal["queen", "king"]

    # Exact remaining draw order. Server-only.
    stock: list[str]

    # Private hands. Each contains four cards at the start of a round.
    human_hand: list[str]
    agent_hand: list[str]

    # Shared 3×3 coffin. None represents an unoccupied position.
    # Grid indexes are 0 1 2 / 3 4 5 / 6 7 8; index 4 is the center.
    coffin: list[str | None]

    # Accepted player moves in the current round, in play order.
    current_round_moves: list[PlayedMove]

    # Set after the eighth move while scoring is being presented. This is a
    # derived result over the current coffin, not another card location.
    pending_round_result: RoundRecord | None

    # Immutable public history for scoring, replay, and card inference.
    completed_rounds: list[RoundRecord]

    # Cumulative scores through the most recently scored round.
    total_scores: PlayerScore


class RoundRecord:
    # Round identity and dealer.
    round_number: int
    dealer: Player

    # Final nine-card coffin and the eight player moves in play order.
    coffin: list[str]
    moves: list[PlayedMove]

    # Deterministic scoring details and ordered presentation sequence.
    line_scores: list[LineScore]
    scoring_sequence: list[ScoringStep]
    round_scores: PlayerScore


class PlayedMove:
    player: Player
    card_id: str
    position: int           # 0 through 8
    turn_number: int        # 1 through 8


class LineScore:
    direction: Literal["row", "column"]
    index: int              # 0 through 2
    card_ids: list[str]
    base_value: int
    multiplier: int         # 0, 1, 2, 3, or 5
    total: int


class ScoringStep:
    # Drives the scoring presentation without recalculating rules in the UI.
    kind: Literal[
        "score_line",
        "compare_candidates",
        "select_round_score",
        "update_total",
    ]
    player: Player | None
    line: LineScore | None
    details: dict[str, int | str | bool]


class PlayerScore:
    human: int
    agent: int


class LegalMove:
    # Opaque server-generated identifier submitted by browser or agent.
    move_id: str

    # Card and destination used to enable valid drag-and-drop targets or tools.
    card_id: str
    position: int
```

Card status is derived from `stock`, both hands, the current coffin, and the
coffins in completed rounds rather than stored as a second mutable ledger. Every
one of the 54 card IDs must exist in exactly one of those ownership locations.
`pending_round_result` and move records contain historical references and are
not additional card locations.

`HumanGameView` contains public state, the human hand, current human legal moves,
and a resumable phase describing a pending agent, narration, scoring, or advance
step. `AgentGameView` contains public state, the agent hand, and current agent
legal moves. `PublicGameView` contains neither hand, the stock, nor private legal
moves. A card-ledger tool derives the agent's known cards and unknown pool from
`AgentGameView`; it does not read authoritative hidden locations.

Model and prompt configuration belongs to the surrounding persisted game
session, not the pure engine state.

### State invariants

- The 54 card IDs are unique and each has exactly one location.
- The center position is occupied before the first player turn.
- Every played position is empty and orthogonally adjacent to an occupied cell.
- During play, the active player alternates, beginning with the non-dealer.
- There is no active player while a round result is pending or the game is complete.
- Queen and King roles are opposite and fixed for the game.
- The dealer alternates after each completed round.
- Line, round, and total scores are reproducible from recorded cards and moves.
- A completed round contains nine coffin cards and eight ordered player moves.
- Only accepted actions increment `version`.

## Game lifecycle

Creating a game selects the first dealer from the seed, deals round one, places
the center card, and emits `game_created` and `round_started`. The
`round_started` narrator projection contains only the center card and first
player. The browser may animate the deal, but play is not enabled until the
required opening comment is available or its bounded wait fails.

During `playing`, an accepted move removes one card from the active hand, places
it in the coffin, appends `current_round_moves`, advances the turn, and emits one
`move_accepted` event. After a human move, the browser separately requests the
agent turn. This keeps human move acknowledgement fast and makes agent retries
independent of move submission.

The eighth move deterministically computes every line and the final round
scores. The service sets `pending_round_result`, updates totals, changes the
status to `round_complete`, and emits `row_score_ready`, `column_score_ready`,
and `round_completed` in that order. The state cannot advance while the scoring
presentation is running.

After presentation, the browser calls the round-advance endpoint. The service
moves `pending_round_result` to `completed_rounds`. For rounds one through five
it clears the current round, alternates the dealer, deals the next round, and
emits `round_started`. After round six it changes the status to `game_complete`
and emits `game_completed`.

Game transitions are independent of presentation state. Reloading during an
agent turn, required narrator wait, or scoring presentation returns enough phase
and event information for the browser to resume the pending step.

## Turn orchestration

### Human turn

1. Drag-and-drop maps a card and destination to a server-issued `move_id`.
2. The browser submits the move with `expected_version` and `request_id`.
3. The service validates and transactionally records the new state and public
   event, then returns the updated `HumanGameView`.
4. If the deterministic narrator cadence admits a move comment, the service
   creates an asynchronous narration job. The comment never delays the accepted
   move or the next turn.

### Agent turn

1. When the returned view names the agent as active, the browser calls
   `agent-turn` with the current version and a new `request_id`.
2. The service conditionally claims that exact turn. Duplicate requests return
   the existing job or result and do not start another model invocation.
3. AgentCore receives an immutable `AgentGameView`. The Strands decider may use
   its tools and returns one server-issued `move_id`.
4. The service revalidates that ID against the current version, applies it once,
   and transactionally records the state and event.
5. The response contains the updated human view. Any eligible comment on the
   accepted agent move is generated asynchronously.

An agent timeout or invalid result leaves the same agent turn active. Retrying
`agent-turn` uses the failure and retry limits defined by the selected profile;
it never substitutes a service-chosen move. Only one successfully validated move
can be applied for a turn.

## Narrator scheduling

Narration jobs are keyed by immutable, server-issued public events. A browser
may start or retrieve a job for an eligible event, but cannot submit an arbitrary
prompt or state projection. The narrator never receives hands, stock order,
decider reasoning, or an unaccepted move, and its output never increments the
game version.

| Trigger | Cadence | Input boundary | Presentation behavior |
| --- | --- | --- | --- |
| `round_started` | Every round | Center card and first player only | Start during the deal animation; await before enabling play |
| `human_move_accepted` | Deterministically selected | Accepted move and concise public effects | Non-blocking banter |
| `agent_move_accepted` | Deterministically selected, lower priority | Accepted move and concise public effects | Non-blocking banter |
| `row_score_ready` | Every round | Three row calculations and provisional Queen ranking | Generate during row animation; await afterward |
| `column_score_ready` | Every round | Three column calculations, tie resolution, awarded scores, totals | Generate during column animation; await afterward |
| `game_completed` | Once | Final totals, sixth-round scores, outcome | Await as part of the final presentation |

The two scoring touchpoints divide the deterministic sequence as follows:

1. The browser starts the `row_score_ready` narration job and immediately runs
   the row highlights, values, multipliers, totals, and provisional ranking.
2. When that animation ends, it issues a blocking long poll for the job, appends
   the comment to the visible log, and begins the column phase.
3. It starts `column_score_ready` narration and runs the column calculations,
   cross-player ranking comparison, tie rules, round award, and total update.
4. It then waits for and appends the column comment before allowing round
   advance.

Required waits are bounded. If a narration job is still pending or fails after
the configured wait, the API returns its explicit status and the browser may
continue the presentation without fabricated commentary. A late result may be
shown only while its round and presentation point remain current.

Optional move banter is limited to two comments per round. The deterministic
gate favors accepted human moves after at least five coffin positions are filled
and suppresses move comments on the final placement because scoring follows
immediately. An agent-move comment is eligible only for a salient public event,
such as playing a Vampire or completing a multiplied line, and counts against
the same limit. Events rejected by the cadence gate do not invoke the narrator.
Every invocation must return commentary text. Each request includes at most the
six most recent comments to reduce repetition.

Late-round taunts may describe the visible result theatrically, but cannot claim
knowledge of the decider's private plan. Completed optional comments are
appended in event order; optional comments that become stale when scoring starts
are retained for replay but not inserted into the live log.

```python
class NarratorInput:
    event_id: str
    event_sequence: int
    trigger: Literal[
        "round_started",
        "human_move_accepted",
        "agent_move_accepted",
        "row_score_ready",
        "column_score_ready",
        "game_completed",
    ]
    public_event: PublicNarratorEvent
    recent_commentary: list[str]


class NarratorOutput:
    text: str
```

## API contract

| Endpoint | Purpose |
| --- | --- |
| `POST /games` | Create a game with a human role and allowlisted profile |
| `GET /games/{game_id}` | Return the current human view and resumable phase |
| `POST /games/{game_id}/moves` | Apply one server-issued human move |
| `POST /games/{game_id}/agent-turn` | Invoke and apply the currently authorized agent turn |
| `POST /games/{game_id}/rounds/{round}/advance` | Advance after scoring presentation |
| `POST /games/{game_id}/events/{event_id}/narration` | Idempotently start an eligible narration job |
| `GET /games/{game_id}/narrations/{narration_id}` | Read or bounded-wait for narration status |
| `GET /games/{game_id}/events?after_sequence={n}` | Poll ordered public events for recovery and commentary |
| `GET /model-profiles` | List safe profile names and descriptions |
| `GET /health` | Health check |

`POST /games` accepts `human_role`, `model_profile`, and a client-generated
`request_id`; the service generates the seed unless a trusted evaluation caller
supplies one. Mutations of an existing game include a new `request_id` and the
last observed `expected_version`. Repeating a request ID returns the original
outcome. Reusing it with a different body is rejected. Narration start is
idempotent by its server-issued event ID and does not require a game version.

A legal `move_id` is bound to its game, state version, player, card, and coffin
position. It expires as soon as any of those inputs no longer describes the
authoritative turn.

Move and completed agent-turn requests return `200` with `HumanGameView`. An
agent turn still running when the response window ends returns `202` with its
job identifier and current resumable phase; the browser polls events or repeats
the same request ID. Game creation returns `201`. Starting or polling unfinished
narration returns `202` with its status and identifier. Completed and failed
narration are terminal results returned with `200`. The narration read
accepts `wait_seconds` from zero through the configured upper bound.

Errors use a stable code and current game version where available:

| Status | Meaning |
| --- | --- |
| `404` | Unknown game, event, or narration |
| `409` | Stale version, wrong turn, wrong phase, or already-advanced round |
| `422` | Malformed request or invalid/expired move ID |
| `429` | Game, caller, or model-profile rate limit reached |
| `503` | Agent or narrator dependency unavailable; response states whether retry is safe |

A `409` includes the current `HumanGameView` so the browser can reconcile
without guessing. A reload calls `GET /games/{game_id}`, renders the authoritative
view, polls events after its last displayed sequence, and resumes the active
agent, scoring, narration, or round-advance step. HTTP polling is sufficient for
the small sequential event stream; the MVP does not require a persistent push
connection.

## Events and transactions

Event records are required. They provide recovery, replay, narrator inputs,
scoring presentation anchors, and an audit of accepted transitions. Each event
contains:

- Game and event identifiers, monotonically increasing sequence, event type,
  timestamp, prior and resulting state versions, and request ID when applicable.
- A public payload suitable for the browser or narrator.
- A private replay payload only when deterministic reconstruction requires it.
- The resolved rules, shuffle, engine, model-profile, and prompt versions on the
  game-created event.

State changes use one transaction containing the conditional state update, the
immutable event, and the idempotency result. Narrator completion is a separate
conditional write and cannot change game state. The agent-turn claim is also
separate from state; applying its validated result uses the normal state/event
transaction.

## DynamoDB item model

The production adapter uses one on-demand table partitioned by game:

| Sort key | Contents |
| --- | --- |
| `STATE` | Authoritative state, resolved configuration, latest event sequence, timestamps, expiry |
| `EVENT#{sequence}` | Immutable public event and any private replay payload |
| `NARRATION#{event_id}` | Pending/completed/failed status, input hash, output, model and prompt metadata, timing |
| `AGENT_TURN#{round}#{turn}` | Conditional claim, lease, attempt count, invocation ID, status, result |
| `REQUEST#{request_id}` | Request hash and original response for idempotency |

All items use `PK = GAME#{game_id}`. Conditional expressions enforce state
version, one agent claim per turn, and one terminal narration result per event.
Application logs contain identifiers and public metrics, not hands, deck order,
private prompts, or model reasoning.

Game creation uses a short-lived `PK = CREATE_REQUEST#{request_id}` result item
so a retried create request returns the same game. It has no game data and uses
the same 24-hour idempotency retention.

## Replay and recovery

Private deterministic replay begins with the stored initial deck order and
resolved version metadata, then reapplies accepted move and round-advance events
through the engine. It must reproduce every stored state version, round result,
and final state hash. The initial deck order remains server-only.

Public replay uses public event projections and stored scoring steps; it never
derives or exposes cards before they became public. Stored narrator outputs are
replayed as recorded and are never regenerated. The current `STATE` item is the
serving snapshot; the event stream is the reconstruction and verification
record.

An incomplete agent job can be resumed while its lease is valid or retried after
lease expiry. An incomplete narration job can be polled or restarted
idempotently. Neither job type is evidence that a gameplay transition occurred;
only the corresponding immutable event is.

## Retention

- All state, event, agent-job, and narration items expire 30 days after game
  creation.
- Idempotency request items expire after 24 hours, except terminal request data
  needed by an active job remains embedded in that job.
- CloudWatch application logs and traces use 14-day retention.
- Raw evaluation runs in S3 use a 90-day lifecycle; compact aggregate reports
  and the configuration required to reproduce them are retained with the
  project artifacts.

These values are configuration, not engine behavior, and are represented in the
deployment stack.

## Persistence adapters

The application depends on interfaces for loading and conditionally updating
game state, appending and listing events, claiming agent turns, storing narrator
jobs, and resolving idempotent requests. No route or engine code imports an AWS
client.

```python
class GameStore(Protocol):
    def create(request_id, state, events) -> CreateResult: ...
    def load(game_id) -> StoredGame | None: ...
    def commit(game_id, expected_version, request, next_state, events) \
        -> CommitResult: ...
    def list_public_events(game_id, after_sequence) -> list[PublicEvent]: ...


class AgentTurnStore(Protocol):
    def claim(game_id, round_number, turn_number, request_id, lease) \
        -> ClaimResult: ...
    def get(job_id) -> AgentTurnJob | None: ...
    def finish(job_id, expected_status, result) -> AgentTurnJob: ...


class NarrationStore(Protocol):
    def start(game_id, event_id, input_hash) -> NarrationJob: ...
    def get(job_id) -> NarrationJob | None: ...
    def finish(job_id, expected_status, output) -> NarrationJob: ...
```

`GameStore.create` and `commit` atomically persist their state, events, and
idempotency result. A conditional conflict returns the current stored state and
never partially appends events. Job `start`, `claim`, and `finish` operations
return an existing compatible result when repeated and reject conflicting
inputs.

The deployed implementation uses DynamoDB transactions and conditional writes.
Local development uses SQLite with transactions and unique constraints that
preserve the same version, event-order, claim, and idempotency semantics. Unit
tests use an in-memory implementation of the same interfaces. Adapter contract
tests run the same transition and conflict cases against all implementations.

# Architecture

## System

```text
React browser
    |
API Gateway -> FastAPI on Lambda
                    |-- DynamoDB state, events, and jobs
                    |-- async policy Lambda -> SageMaker Serverless Inference
                    `-- async narrator Lambda -> Bedrock

SageMaker Model Registry and ECR
CloudWatch observability
```

The browser renders public state plus the human hand. FastAPI owns the
authoritative state machine, projections, validation, scoring, persistence, and
policy-turn and narration orchestration. The deterministic engine is pure Python
with no AWS, neural-network, or LLM dependency.

The pure engine and its policy bridge are specified in
[engine–model contract](engine-model-contract.md). The service adds persistence,
idempotency, and deployment concerns around those deterministic transitions.

## State and visibility

One authoritative state produces three views:

| Data | Human view | Policy view | Public/narrator view |
| --- | --- | --- | --- |
| Coffin grid, round, turn, scores | Yes | Yes | Yes |
| Human hand | Yes | No | No |
| Opponent hand | No | Yes | No |
| Deck and order | No | No | No |
| Recipient's legal moves | Human only | Opponent only | No |

All random operations accept a seed. Accepted actions increment a version;
DynamoDB conditional writes reject stale requests. Logs exclude hands, deck
order, recurrent hidden state, private prompts, and model tensors.

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

    # Per-game seed controlling the initial shuffle for reproducible games.
    seed: str

    # Animation and presentation state remain frontend concerns.
    status: Literal["playing", "round_complete", "game_complete"]

    # Current round and number of player cards placed in it.
    round_number: int       # 1 through 6
    turn_number: int        # 0 through 8

    # The non-dealer starts each round; the deal alternates by round.
    dealer: Player
    active_player: Player | None  # None while scoring or after game completion.

    # The human chooses a role; the opponent receives the other role.
    human_role: Literal["queen", "king"]
    opponent_role: Literal["queen", "king"]

    # Exact remaining draw order. Server-only.
    stock: list[str]

    # Private hands use four stable round slots. A played slot becomes None so
    # policy actions and UI moves retain the same slot mapping for the round.
    human_hand: list[str | None]
    opponent_hand: list[str | None]

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
    hand_slot: int          # 0 through 3
    position: int           # 0 through 8
    turn_number: int        # 1 through 8


class LineScore:
    direction: Literal["row", "column"]
    index: int              # 0 through 2
    card_ids: list[str]
    base_value: int
    multiplier: int         # 0, 1, 2, 3, or 5
    multiplier_reason: Literal[
        "none",
        "suit_pair",
        "same_color",
        "same_suit",
        "vampire",
    ]
    multiplier_label: str   # "No Multiplier", "2× Hearts", or "Vampire".
    highlighted_card_ids: list[str]
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
    opponent: int


class LegalMove:
    # Opaque server-generated identifier submitted by the browser or produced
    # by mapping a policy action.
    move_id: str

    # Card and destination used to enable valid drag-and-drop targets and map
    # policy actions.
    card_id: str
    hand_slot: int
    position: int


class PolicySession:
    # Selected registered model and all schemas needed to interpret its I/O.
    model_package_arn: str
    model_package_version: str
    artifact_id: str
    artifact_sha256: str
    observation_schema_version: str
    action_schema_version: str
    hidden_state_schema_version: str

    # One resolved serving behavior for the lifetime of the game.
    inference_profile_version: str
    action_selection: Literal["argmax", "sample"]
    temperature: float

    # Little-endian float32[128]. It advances with each accepted opponent move,
    # including a forced final placement.
    hidden_state: bytes


class NarratorSession:
    model_id: str
    prompt_version: str
    inference_profile_version: str
```

Card status is derived from `stock`, both hands, the current coffin, and the
coffins in completed rounds rather than stored as a second mutable ledger. Every
one of the 54 card IDs must exist in exactly one of those ownership locations.
`pending_round_result` and move records contain historical references and are
not additional card locations.

`HumanGameView` contains public state, the human hand, current human legal moves,
and a resumable phase describing pending opponent inference, narration, scoring,
or advance. `PolicyGameView` is the information-safe projection encoded for
policy inference; it contains the opponent hand, coffin, card-status classes,
round, own-decision progress, dealer status, and legal-action mask defined in
[neural model](neural-model.md). Its coffin and action positions are normalized
to the policy player's scoring orientation. The deterministic construction of
that view and its action mapping are defined in the
[engine–model contract](engine-model-contract.md). `PublicGameView` contains
neither hand, the stock, nor private legal moves.

`PolicySession` and `NarratorSession` belong to the surrounding persisted game
session, not the pure engine state. A policy model version and hidden-state
schema are immutable for the lifetime of a game.

### State invariants

- The 54 card IDs are unique and each has exactly one location.
- Each hand has exactly four stable slots; occupied slots contain unique cards.
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
the center card, initializes the pinned policy's hidden state, and emits
`game_created` and `round_started`. The
`round_started` narrator projection contains only the center card and first
player. The browser may animate the deal, but play is not enabled until the
required opening comment is available or its bounded wait fails.

During `playing`, an accepted move clears one card from its stable hand slot,
places it in the coffin, appends `current_round_moves`, advances the turn, and
emits one `move_accepted` event. After a human move, the browser separately
requests the opponent turn. This keeps human move acknowledgement fast and makes
policy inference retries independent of move submission.

The eighth move has exactly one legal card-position action. When the active
player is the policy, the service invokes inference to advance the game-scoped
recurrent state and applies that unique move through the engine. A human final
move continues through the normal human-turn contract. After accepting the
eighth move, the service deterministically computes every line and the final
round scores. It sets `pending_round_result`, updates totals, changes the status
to `round_complete`, and emits the `move_accepted`, two orientation events, and
`round_completed`. If the dealer is Queen, their order is `row_score_ready` then
`column_score_ready`; if the dealer is King, their order is reversed. The state
cannot advance while the scoring presentation is running.

After presentation, the browser calls the round-advance endpoint. The service
moves `pending_round_result` to `completed_rounds`. For rounds one through five
it clears the current round, retains the policy hidden state, alternates the
dealer, deals the next round, and emits `round_started`. After round six it
changes the status to `game_complete` and emits `game_completed`.

Game transitions are independent of presentation state. Reloading during an
opponent turn, required narrator wait, or scoring presentation returns enough
phase and event information for the browser to resume the pending step.

## Turn orchestration

### Human turn

1. Drag-and-drop maps a card and destination to a server-issued `move_id`.
2. The browser submits the move with `expected_version` and `request_id`.
3. The service validates and transactionally records the new state and public
   event, then returns the updated `HumanGameView`.
4. If the deterministic narrator cadence admits a move comment, the service
   creates an asynchronous narration job. The comment never delays the accepted
   move or the next turn.

### Opponent turn

1. When the returned view names the opponent as active, the browser calls
   `opponent-turn` with the current version and a new `request_id`.
2. The service conditionally creates or reads the job for that exact game
   version and turn. A new, requeued, or dispatchable expired job invokes the
   policy Lambda asynchronously and returns `202`; a completed job returns its
   existing result.
3. One worker conditionally acquires the job's execution lease. It reloads the
   claimed state, verifies that the turn is still active, constructs the
   immutable `PolicyGameView` and engine-derived legal mask, and sends the
   inference request defined in [neural model](neural-model.md). The request
   carries the game's prior hidden state and pinned artifact ID.
4. The SageMaker container runs deterministic CPU inference and returns raw
   logits and the next hidden state. It retains no game session.
5. The worker validates response versions, shapes, encoding, and finite values.
   It applies the authoritative legal mask. The fixed production profile then
   selects masked argmax or samples the masked temperature-one distribution. A
   sampled profile uses a seed derived from the game ID, round, turn, package
   digest, and sampling-seed version, so every attempt resolves the same action.
   For a one-legal-action turn, the engine supplies that action and the returned
   hidden state remains the recurrent successor.
6. The service maps the selected action through the request's action table and
   revalidates the move against the claimed authoritative state.
7. One DynamoDB transaction conditionally applies the move, commits the next
   hidden state, appends the public event, stores the idempotent response, and
   completes the turn job. The hidden state cannot advance independently of the
   accepted move.
8. The browser retrieves the completed human view by repeating the same request
   or polling events. Any eligible comment on the accepted opponent move starts
   asynchronously.

Worker delivery and SageMaker invocation may repeat. Every attempt reconstructs
the same input from the unchanged claimed state and uses the same action seed.
A timeout, endpoint error, malformed response, or conditional conflict commits
neither the move nor hidden state. The bounded retry profile records each
attempt; after it is exhausted the job reports a retryable dependency failure
while the opponent turn remains active. A later opponent-turn request may
conditionally requeue that same job only while its game version and turn remain
unchanged; the input hash and action seed remain fixed.

## Narrator scheduling

Narration jobs are keyed by immutable, server-issued public events. A browser
may start or retrieve a job for an eligible event, but cannot submit an arbitrary
prompt or state projection. The narrator never receives hands, stock order,
policy hidden state, private inference data, or an unaccepted move, and its
output never increments the game version.

| Trigger | Cadence | Input boundary | Presentation behavior |
| --- | --- | --- | --- |
| `round_started` | Every round | Center card and first player only | Start during the deal animation; await before enabling play |
| `human_move_accepted` | Deterministically selected | Accepted move and concise public effects | Non-blocking banter |
| `opponent_move_accepted` | Deterministically selected, lower priority | Accepted move and concise public effects | Non-blocking banter |
| `row_score_ready` | Every round | Three row calculations and sorted Queen ranking | Generate during row animation; await afterward |
| `column_score_ready` | Every round | Three column calculations and sorted King ranking | Generate during column animation; await afterward |
| `game_completed` | Once | Final totals, sixth-round scores, outcome | Await as part of the final presentation |

The two scoring touchpoints divide the deterministic sequence as follows:

1. The browser selects the dealer's orientation event, starts its narration job,
   and immediately runs that orientation's line calculations and sorted tally.
2. When the animation ends, it issues a blocking long poll, appends the comment,
   and hides the completed tally.
3. It starts the non-dealer orientation's narration job and runs the equivalent
   calculation and ranking sequence.
4. It waits for and appends the second comment, then presents the deterministic
   cross-player comparison, tie resolution, round awards, and cumulative totals
   without a third narrator call.

Required waits are bounded. If a narration job is still pending or fails after
the configured wait, the API returns its explicit status and the browser may
continue the presentation without fabricated commentary. A late result may be
shown only while its round and presentation point remain current.

Optional move banter is limited to two comments per round. The deterministic
gate favors accepted human moves after at least five coffin positions are filled
and suppresses move comments on the final placement because scoring follows
immediately. An opponent-move comment is eligible only for a salient public event,
such as playing a Vampire or completing a multiplied line, and counts against
the same limit. Events rejected by the cadence gate do not invoke the narrator.
Every invocation must return commentary text. Each request includes at most the
six most recent comments to reduce repetition.

Late-round taunts may describe the visible result theatrically, but cannot claim
knowledge of the policy's hidden state or intended strategy. Completed optional
comments are appended in event order; optional comments that become stale when
scoring starts are retained for replay but not inserted into the live log.

```python
class NarratorInput:
    event_id: str
    event_sequence: int
    trigger: Literal[
        "round_started",
        "human_move_accepted",
        "opponent_move_accepted",
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
| `POST /games` | Create a game with a human role and the production policy/narrator configuration |
| `GET /games/{game_id}` | Return the current human view and resumable phase |
| `POST /games/{game_id}/moves` | Apply one server-issued human move |
| `POST /games/{game_id}/opponent-turn` | Invoke and apply the currently authorized opponent turn |
| `POST /games/{game_id}/rounds/{round}/advance` | Advance after scoring presentation |
| `POST /games/{game_id}/events/{event_id}/narration` | Idempotently start an eligible narration job |
| `GET /games/{game_id}/narrations/{narration_id}` | Read or bounded-wait for narration status |
| `GET /games/{game_id}/events?after_sequence={n}` | Poll ordered public events for recovery and commentary |
| `GET /health` | Health check |

`POST /games` accepts `human_role` and a client-generated `request_id`. The
service resolves the single production policy package and narrator configuration
and generates the seed unless a trusted evaluation caller supplies one.
Mutations of an existing game include a new `request_id` and the last observed
`expected_version`. Repeating a completed mutation returns the original outcome.
Repeating an opponent-turn request returns the current state of its one claimed
job, progressing from `202` to its terminal result. Reusing a request ID with a
different body is rejected. Narration start is idempotent by its server-issued
event ID and does not require a game version.

A legal `move_id` is bound to its game, state version, player, card, and coffin
position. It expires as soon as any of those inputs no longer describes the
authoritative turn.

Move requests and completed opponent-turn jobs return `200` with
`HumanGameView`. A newly claimed or running opponent turn returns `202` with its
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
| `429` | Game, caller, or profile rate limit reached |
| `503` | Policy inference or narrator dependency unavailable; response states whether retry is safe |

A `409` includes the current `HumanGameView` so the browser can reconcile
without guessing. A reload calls `GET /games/{game_id}`, renders the authoritative
view, polls events after its last displayed sequence, and resumes the active
opponent, scoring, narration, or round-advance step. HTTP polling is sufficient
for the small sequential event stream; the MVP does not require a persistent
push connection.

## Events and transactions

Event records are required. They provide recovery, replay, narrator inputs,
scoring presentation anchors, and an audit of accepted transitions. Each event
contains:

- Game and event identifiers, monotonically increasing sequence, event type,
  timestamp, prior and resulting state versions, and request ID when applicable.
- A public payload suitable for the browser or narrator.
- A private replay payload only when deterministic reconstruction requires it.
- The resolved rules, shuffle, engine, policy package, tensor schemas, inference
  profile, narrator model, and prompt versions on the game-created event.

State changes use one transaction containing the conditional state update, the
immutable event, and the idempotency result. Narrator completion is a separate
conditional write and cannot change game state. The policy-turn claim is also
separate from state; its completion joins the state, event, hidden-state, and
idempotency writes in one transaction.

## DynamoDB item model

The production adapter uses one on-demand table partitioned by game:

| Sort key | Contents |
| --- | --- |
| `STATE` | Authoritative state, resolved configuration, latest event sequence, timestamps, expiry |
| `EVENT#{sequence}` | Immutable public event and any private replay payload |
| `NARRATION#{event_id}` | Pending/completed/failed status, input hash, output, model and prompt metadata, timing |
| `POLICY_TURN#{round}#{turn}` | Conditional claim, lease, request hash, package version, attempt records, status, action, result hash, next-hidden-state hash |
| `REQUEST#{request_id}` | Request hash and original response for idempotency |

All items use `PK = GAME#{game_id}`. Conditional expressions enforce state
version, one policy claim per turn, and one terminal narration result per event.
Application logs contain identifiers and public metrics, not hands, deck order,
recurrent hidden state, private prompts, or model internals.

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

An incomplete policy job can be resumed while its lease is valid or retried after
lease expiry. An incomplete narration job can be polled or restarted
idempotently. Neither job type is evidence that a gameplay transition occurred;
only the corresponding immutable event is.

## Retention

- All state, event, policy-job, and narration items expire 30 days after game
  creation.
- Idempotency request items expire after 24 hours, except terminal request data
  needed by an active job remains embedded in that job.
- CloudWatch application logs and traces use 14-day retention.
These values are configuration, not engine behavior, and are represented in the
deployment stack.

## Persistence adapters

The application depends on interfaces for loading and conditionally updating
game state and policy hidden state, appending and listing events, claiming policy
turns, invoking policy inference, storing narrator jobs, and resolving idempotent
requests. No route or engine code imports an AWS client.

```python
class GameStore(Protocol):
    def create(request_id, state, events) -> CreateResult: ...
    def load(game_id) -> StoredGame | None: ...
    def commit(game_id, expected_version, request, next_state, events) \
        -> CommitResult: ...
    def commit_policy_turn(job_id, expected_job_status, expected_version,
                           next_state, events, response) -> CommitResult: ...
    def list_public_events(game_id, after_sequence) -> list[PublicEvent]: ...


class PolicyTurnStore(Protocol):
    def claim(game_id, round_number, turn_number, request_id) \
        -> ClaimResult: ...
    def get(job_id) -> PolicyTurnJob | None: ...
    def acquire(job_id, expected_status, lease) -> PolicyTurnJob: ...
    def record_attempt(job_id, expected_status, attempt) -> PolicyTurnJob: ...
    def fail(job_id, expected_status, failure) -> PolicyTurnJob: ...
    def retry(job_id, expected_status, expected_version) -> PolicyTurnJob: ...


class PolicyInference(Protocol):
    def invoke(request: PolicyInferenceRequest) -> PolicyInferenceResult: ...


class NarrationStore(Protocol):
    def start(game_id, event_id, input_hash) -> NarrationJob: ...
    def get(job_id) -> NarrationJob | None: ...
    def finish(job_id, expected_status, output) -> NarrationJob: ...
```

`GameStore.create`, `commit`, and `commit_policy_turn` atomically persist their
state, events, and idempotency result. `commit_policy_turn` also conditionally
completes its claimed job. A conflict returns the current stored state and never
partially appends events or advances hidden state. Repeated job operations return
an existing compatible result and reject conflicting inputs.

The deployed implementation uses DynamoDB transactions and conditional writes.
Local development uses SQLite with transactions and unique constraints that
preserve the same version, event-order, claim, and idempotency semantics. Unit
tests use an in-memory implementation of the same interfaces. Adapter contract
tests run the same transition and conflict cases against all implementations.

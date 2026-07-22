# Application architecture

## System boundary

```text
React browser -> FastAPI service -> deterministic engine
                              |-- SQLite or in-memory persistence
                              |-- opponent decision interface
                              `-- optional narrator interface
```

The browser renders the human view. FastAPI owns projections, validation,
idempotency, lifecycle orchestration, and persistence around the pure engine.
The engine owns game truth. The opponent interface initially runs
information-set search locally; the existing archived PPO adapter remains a
comparison control, not the target opponent.

Cloud topology is deferred until search latency and strength are measured. No
route or engine module depends on an AWS SDK or a specific inference product.

## State and visibility

One authoritative state produces separate views:

| Data | Human view | Opponent information state | Narrator view |
| --- | --- | --- | --- |
| Coffin, public moves, round, dealer, scores | Yes | Yes | Yes |
| Human hand | Yes | Only when opponent is the human-side simulation actor | No |
| Dracula hand | No | Yes when Dracula acts | No |
| Recipient legal actions | Human only | Acting opponent only | No |
| Opponent hand-slot indexes | No | Own slots only | No |
| Stock order, seed, search samples | No | No | No |

Accepted mutations increment a session version. Stale writes fail. Normal logs
exclude hands, hidden-card assignments, stock order, seeds, search trees, model
tensors, and narrator prompts.

### Authoritative session

```python
class GameSession:
    game_id: UUID
    version: int
    human_role: Literal["queen", "king"]
    engine_state: EngineState
    opponent_session: OpponentSession
    phase: ResumablePhase
    events: tuple[PublicEvent, ...]
    idempotency_records: tuple[IdempotencyRecord, ...]
    opponent_turn_claim: OpponentTurnClaim | None


class OpponentSession:
    controller_id: str
    controller_version: str
    configuration_digest: str
    decision_schema_version: str
    seed_schema_version: str


class PublicPlayedMove:
    player: Literal["queen", "king"]
    card_id: str
    position: int
    turn_number: int


class RoundRecord:
    round_number: int
    dealer: Literal["queen", "king"]
    coffin: tuple[str, ...]
    moves: tuple[PublicPlayedMove, ...]
    line_scores: tuple[LineScore, ...]
    scoring_sequence: tuple[ScoringStep, ...]
    round_scores: PlayerScore


class LegalMove:
    move_id: str
    card_id: str
    hand_slot: int
    position: int
```

`EngineState` is defined in the
[engine–opponent contract](engine-model-contract.md#engine-state-and-operations).
The application session adds identity, persistence version, human role,
opponent configuration, events, and resumable work.

The engine's private `EnginePlayedMove` retains the acting hand slot. Public
projections remove it. `LegalMove.hand_slot` is present only for the human's
currently visible hand and is bound into an opaque move ID.

`OpponentSession` is immutable for a game. It records the exact search or model
controller and deterministic decision configuration. It has no recurrent hidden
state. A future model artifact may add immutable schema and digest fields without
changing engine state.

### Derived records

Card status is derived from stock, both hands, the current coffin, and completed
round coffins. Every card occupies exactly one authoritative location. Public
move and scoring records are references, not additional locations.

`HumanGameView` contains public state, the human hand, current human legal
moves, scoring data, ordered public events, and one resumable phase.
`SearchInformationState` contains only the acting player's information as
defined by the [engine–opponent contract](engine-model-contract.md#player-relative-information-state).
The narrator receives an event-specific public projection with neither hand.

### Invariants

- All 54 card IDs are unique and occupy exactly one authoritative location.
- Hands have four stable private slots; occupied cards remain canonically
  ordered.
- The center is occupied before play and every move is orthogonally adjacent.
- The non-dealer begins and players alternate for exactly eight accepted moves.
- Queen and King roles remain opposite; the dealer alternates after each round.
- No player is active while scoring is pending or the game is complete.
- Line, round, and total scores reproduce from engine records.
- Only an accepted mutation increments the session version.
- Public projections never expose an opponent hand slot or hidden assignment.

## Game lifecycle

Creating a game resolves one opponent configuration, generates or accepts a
trusted evaluation seed, creates the engine state, and emits `game_created` and
`round_started`. The browser may animate the deal; narration-disabled local
play begins immediately.

During `playing`, an accepted move clears one private hand slot, places the card,
records the engine move, advances the active player, and emits one public
`move_accepted` event without a hand slot. After a human move, the browser
requests the opponent turn separately.

If an opponent turn has one legal action, the service applies it directly. A
non-forced opponent turn calls the configured opponent interface with an
information-safe view and deterministic request seed. The engine revalidates the
returned action before it can commit.

The eighth move computes all line scores, round-score tie resolution, and new
totals. The state becomes `round_complete` and emits the move, the two
dealer-ordered orientation events, and `round_completed`. Scoring presentation
cannot advance the engine.

After presentation, the browser calls the round-advance endpoint. Through round
five, the engine archives the result, alternates the dealer, deals the next
round, and emits `round_started`. After round six it emits `game_completed`.

Reloading returns enough state and events to resume a human turn, opponent turn,
scoring presentation, or round advancement without client-side rule inference.

## Opponent decision contract

```python
class OpponentDecisionRequest:
    game_id: UUID
    expected_version: int
    request_id: UUID
    turn_number: int
    controller: OpponentDescriptor
    information_state: SearchInformationState
    action_table_digest: str
    decision_seed: bytes


class OpponentDecisionResult:
    controller: OpponentDescriptor
    information_state_digest: str
    action_table_digest: str
    action_index: int
    private_diagnostics_digest: str


class OpponentEngine(Protocol):
    def decide(request: OpponentDecisionRequest) -> OpponentDecisionResult: ...
```

The service constructs the information state and action table once from the
claimed engine version. The controller receives no authoritative state. It
returns an action index and evidence digests, not a mutated state. The service
verifies controller identity, input digests, action range, legality, and finite
diagnostics before resolving the action table and calling `apply_move`.

Search configuration and request identity derive the decision seed. A retry of
the same claimed turn recreates the same information state, samples, and action.
A timeout, malformed result, or conflict commits no move.

Private search diagnostics are stored outside public events and normal logs.
They may be retained for evaluation but are not required to reconstruct game
truth; accepted moves and engine versions remain authoritative.

## Turn orchestration

### Human turn

1. The browser maps a spatial interaction to a server-issued `move_id`.
2. It submits the move with `expected_version` and `request_id`.
3. The service resolves the opaque ID, applies the engine move, and atomically
   stores state, event, and idempotent response.
4. Eligible narrator banter starts independently.

### Opponent turn

1. The browser posts `opponent-turn` for the current version.
2. The service creates or reads one conditional claim for that version and turn.
3. A worker reloads the claimed state and verifies that the turn remains active.
4. A forced action is applied directly; otherwise the worker projects the
   opponent information state and calls `OpponentEngine`.
5. The worker validates and resolves the returned action against the claimed
   action table, then revalidates through the engine.
6. One transaction commits the move, public event, idempotent response, and
   completed claim.
7. The browser repeats the same request or polls events until the completed human
   view is available.

Worker delivery may repeat. Exactly one accepted move can complete a claim.
Exhausted controller failures return a retryable dependency error while the
same opponent turn remains active.

The local adapter may execute inline behind this boundary. Worker technology,
concurrency, and hosting are deployment decisions made after search benchmarks.

## Narrator scheduling

Narration is independent of opponent selection and game-state transactions. It
receives immutable public events and cannot submit a move.

| Trigger | Cadence | Presentation |
| --- | --- | --- |
| `round_started` | Every round | Generate during deal; bounded wait before play when enabled |
| `human_move_accepted` | Deterministically selected | Non-blocking banter |
| `opponent_move_accepted` | Deterministically selected, lower priority | Non-blocking banter |
| `row_score_ready` | Every round | Generate during row animation; bounded wait afterward |
| `column_score_ready` | Every round | Generate during column animation; bounded wait afterward |
| `game_completed` | Once | Bounded final-presentation wait |

Required waits are bounded. Failure or disabled narration never fabricates
dialogue and never blocks completion. Narrator inputs exclude hands, stock,
search state, determinizations, diagnostics, and intended strategy.

## API contract

| Endpoint | Purpose |
| --- | --- |
| `POST /games` | Create a game with a human role and resolved opponent configuration |
| `GET /games/{game_id}` | Return the current human view and resumable phase |
| `POST /games/{game_id}/moves` | Apply one server-issued human move |
| `POST /games/{game_id}/opponent-turn` | Run and apply the authorized opponent turn |
| `POST /games/{game_id}/rounds/{round}/advance` | Advance after scoring presentation |
| `POST /games/{game_id}/events/{event_id}/narration` | Start eligible narration idempotently |
| `GET /games/{game_id}/narrations/{narration_id}` | Read or bounded-wait for narration |
| `GET /games/{game_id}/events?after_sequence={n}` | Poll ordered public events |
| `GET /health` | Return service and integration status |

Mutations include a new `request_id` and last observed `expected_version`.
Repeating an accepted mutation returns the original result. Reusing a request ID
with a different body fails.

A `move_id` binds game, version, player, card, private human slot, and position.
It expires after any accepted state change. Move and completed opponent requests
return `200`; a claimed or running opponent turn returns `202`; creation returns
`201`.

Errors use stable codes:

| Status | Meaning |
| --- | --- |
| `404` | Unknown game, event, or narration |
| `409` | Stale version, wrong turn, wrong phase, or advanced round |
| `422` | Malformed request or invalid move ID |
| `429` | Configured rate limit reached |
| `503` | Opponent or narrator dependency unavailable; retry safety is explicit |

A `409` includes the current human view. Reload always reconciles from the
authoritative response before enabling input. HTTP polling is sufficient for
the sequential MVP event stream.

## Events and transactions

Each event contains game and event IDs, monotonically increasing sequence,
event type, timestamp, prior and resulting versions, request ID when applicable,
and a public payload. `game_created` records rules, engine, opponent controller,
information-state, action, search, and narrator versions.

Private replay payloads contain only data required to reconstruct accepted
engine transitions. Search diagnostics are referenced by digest and kept in a
separate private evaluation store.

A state mutation transaction contains the conditional state update, immutable
events, and idempotency result. Opponent completion also conditionally completes
its turn claim. Narration completion is separate and cannot change game state.

## Replay and recovery

Private deterministic replay begins with the stored initial engine state and
reapplies accepted moves and round advances. It must reproduce every engine
fingerprint, round result, and final outcome. Search is not rerun to replay an
accepted game.

Public replay uses public moves and scoring steps. It never derives cards before
they became public. Stored narrator outputs replay as recorded.

An incomplete opponent claim can retry only while its game version and turn are
unchanged. Its information-state digest, action table, controller configuration,
and decision seed remain fixed. The claim alone is not evidence of a move; only
the accepted event is.

## Persistence adapters

```python
class GameStore(Protocol):
    def create(request_id, state, events) -> CreateResult: ...
    def load(game_id) -> StoredGame | None: ...
    def commit(game_id, expected_version, request, next_state, events) \
        -> CommitResult: ...
    def commit_opponent_turn(claim_id, expected_version, next_state, events,
                             response) -> CommitResult: ...
    def list_public_events(game_id, after_sequence) -> list[PublicEvent]: ...


class OpponentTurnStore(Protocol):
    def claim(game_id, round_number, turn_number, request_id) -> ClaimResult: ...
    def get(claim_id) -> OpponentTurnClaim | None: ...
    def finish(claim_id, expected_status, decision_digest) -> ClaimResult: ...
    def fail(claim_id, expected_status, failure) -> ClaimResult: ...
```

SQLite supplies local transactions and uniqueness constraints. In-memory
storage supplies the same contract for unit tests. Production persistence,
retention, job transport, and hosting are selected with deployment after the
opponent execution profile is measured.

# Engine–model contract

This document defines the deterministic boundary between the game engine, the
learned policy, and the training critic. It applies to self-play collection,
local inference, and the deployed game service.

## Contract versions

Every training run and policy artifact records this version tuple:

```text
rules_version
engine_version
observation_schema_version
action_schema_version
policy_architecture_version
hidden_state_schema_version
```

The engine and bridge derive a canonical state fingerprint for every valid
`EngineState`, including created and accepted-transition states. It is the
SHA-256 digest of the UTF-8 canonical JSON serialization of the versioned
`EngineState`, with fixed field order and card IDs in their stored order.
Fixtures use the fingerprint to verify deterministic replay and cross-runtime
agreement.

## Engine state and operations

`EnginePlayer` identifies a rules role and has one of two values: `queen` or
`king`. The engine keys dealer, active player, hands, move ownership, and round
scores by `EnginePlayer`. The application maps its human and opponent roles to
these two engine players.

`EngineState` contains the deterministic game fields carried by `GameState` in
[architecture](architecture.md): seed, round, dealer, active player, stock,
hands, coffin, current-round moves, completed rounds, pending round result, and
cumulative scores. Its player-keyed fields use `EnginePlayer`. The application
session adds identifiers, persistence versions, request metadata, presentation
state, policy sessions, and narrator sessions around that state.

`EngineState` stores `EnginePlayedMove` and `EngineRoundResult` values. The
application maps those role-keyed results to its human and opponent presentation
records when it persists a game or serves the frontend.

The engine functions are pure. Each accepts an `EngineState`, returns a new
state or result, and validates the supplied state before calculating an outcome.

```python
EnginePlayer = Literal["queen", "king"]


class EngineState:
    seed: str
    status: Literal["playing", "round_complete", "game_complete"]
    round_number: int                  # 1 through 6
    dealer: EnginePlayer
    active_player: EnginePlayer | None
    stock: tuple[str, ...]
    hands: dict[EnginePlayer, tuple[str | None, str | None, str | None, str | None]]
    coffin: tuple[str | None, ...]     # length 9, row-major Queen orientation
    current_round_moves: tuple[EnginePlayedMove, ...]
    pending_round_result: EngineRoundResult | None
    completed_rounds: tuple[EngineRoundResult, ...]
    total_scores: dict[EnginePlayer, int]


class EngineMove:
    player: EnginePlayer
    hand_slot: int            # 0 through 3
    global_grid_index: int    # 0 through 8, row-major Queen orientation


class EnginePlayedMove:
    player: EnginePlayer
    card_id: str
    hand_slot: int
    global_grid_index: int
    turn_number: int          # 1 through 8


class EngineRoundResult:
    round_number: int
    dealer: EnginePlayer
    coffin: tuple[str, ...]
    moves: tuple[EnginePlayedMove, ...]
    line_scores: dict[EnginePlayer, tuple[LineScore, LineScore, LineScore]]
    round_scores: dict[EnginePlayer, int]


class EngineTransition:
    previous_state: EngineState
    state: EngineState
    move: EngineMove
    played_move: EnginePlayedMove
    round_result: EngineRoundResult | None
    state_fingerprint: str


def create_game(seed: str) -> EngineState: ...
def legal_moves(state: EngineState, player: EnginePlayer) -> tuple[EngineMove, ...]: ...
def apply_move(state: EngineState, move: EngineMove) -> EngineTransition: ...
def advance_after_round(state: EngineState) -> EngineState: ...
```

`create_game` shuffles the 54-card deck from the supplied seed, selects the
initial dealer, deals the first round, places the center card, and sets the
non-dealer active. The seeded shuffle and card-ID order are versioned engine
behavior. A game seed identifies one complete six-round game: repeated calls
with the same seed produce the same game. Fixture scheduling derives a distinct
game seed for each game before calling this function.

`legal_moves` resolves card identity from the player's canonical hand slot and
returns every currently legal placement in global grid coordinates. Its ordered
result uses ascending `hand_slot`, then ascending `global_grid_index`.

`apply_move` validates active player, occupied hand slot, empty destination,
orthogonal adjacency, and the current round phase. It clears the hand slot,
places the card, records `PlayedMove`, and advances the active player. The
eighth accepted placement produces the `EngineRoundResult`, applies rules-defined
scoring and cumulative totals, and moves the state to round completion.

`advance_after_round` archives the completed round. Through round five, it
alternates the dealer, deals the next round from the existing stock, places the
center card, resets the current-round move list, and sets the new non-dealer
active. After round six, it produces the terminal game state.

The engine reports typed rule violations for a malformed state, wrong active
player, unavailable hand slot, invalid grid index, occupied destination,
non-adjacent destination, or invalid lifecycle transition.

## Player-relative policy context

The bridge derives a `PolicyTurnContext` from an active engine state and one
player. It contains the policy input and the complete action mapping for that
turn.

```python
class PolicyInput:
    observation: BoolTensor[875]
    legal_mask: BoolTensor[4, 8]


class PolicyTurnContext:
    state_fingerprint: str
    player: EnginePlayer
    round_number: int
    own_decision_index: int   # 0 through 3
    kind: Literal["learned", "forced_recurrent_transition"]
    input: PolicyInput
    action_table: tuple[EngineMove | None, ...]  # length 32
    forced_move: EngineMove | None
```

```python
def build_policy_turn_context(
    state: EngineState,
    player: EnginePlayer,
) -> PolicyTurnContext: ...
```

The policy context uses the observation and action schemas in
[neural model](neural-model.md). Queen uses the authoritative coffin orientation.
King uses the transposed coffin orientation. The same orientation transform
applies to candidate destinations before action indexes are assigned.

`own_decision_index` equals the number of accepted `EnginePlayedMove` values
owned by `player` in the current round. It ranges from zero through three and
selects the matching four-position one-hot field in `PolicyInput`.

The bridge calls `legal_moves(state, player)` once and builds both
`legal_mask` and `action_table` from that result. For policy action index
`i`:

```text
hand_slot = i // 8
policy_position_index = i % 8
policy_grid_index = [0, 1, 2, 3, 5, 6, 7, 8][policy_position_index]
```

`action_table[i]` contains the corresponding global `EngineMove` when the
hand-slot and player-relative destination are legal; otherwise it contains
`None`. The Boolean legal mask and action table therefore describe the same 32
action indexes.

An action table with one move has that move in `forced_move` and has kind
`forced_recurrent_transition`. The bridge constructs a policy context for that
turn, so the policy advances its hidden state from the unique legal action. A
context with two or more moves has kind `learned`.

## Policy action application

The policy invocation and engine transition use an immutable pre-action context:

```python
class PolicyOutput:
    raw_logits: Float32Tensor[4, 8]
    next_hidden_state: bytes


class SelectedPolicyAction:
    action_index: int
    log_probability: float
```

```text
context = build_policy_turn_context(state, active_player)
output = policy(context.input, policy_hidden_state)
if context.kind == "forced_recurrent_transition":
    move = context.forced_move
else:
    selected = select_masked_action(output.raw_logits, context.input.legal_mask)
    move = context.action_table[selected.action_index]
transition = apply_move(state, move)
```

`select_masked_action` applies the contract's global 32-way masked softmax.
The training collector samples from that distribution on learned turns and
records the selected action's masked log probability. On a forced recurrent
transition, the engine applies `forced_move`; the policy output advances hidden
state without an action sample. Serving selection configuration belongs to the
inference and service contracts.

A selected action index resolves to an `EngineMove` in the action table before
the engine transition begins.

The next policy hidden state commits with the accepted `EngineTransition`. The
state fingerprint, player, round, own-decision index, kind, and legal mask
identify the context that authorized the action.

## Critic and trajectory boundary

The critic evaluates the pre-action observation from `PolicyInput`; it does not
receive the policy action or recurrent hidden state. The trajectory records each
policy state update:

```python
class PolicyTransition:
    fixture_id: str
    learner_id: str
    learner_policy_version: str
    opponent_id: str
    opponent_policy_version: str
    player: EnginePlayer
    state_fingerprint: str
    round_number: int
    own_decision_index: int
    recurrent_step_index: int  # 0 through 23 within the player's game
    kind: Literal["learned", "forced_recurrent_transition"]
    policy_input: PolicyInput
    policy_hidden_in: bytes
    action_index: int
    action_log_probability: float | None
    policy_hidden_out: bytes
    critic_value: float | None
    round_return: float | None
    actor_loss_mask: bool
```

Each player produces four `PolicyTransition` values in every round. The first
round occupies recurrent indexes zero through three, and the sixth occupies
indexes 20 through 23. The forced recurrent transition records its unique action
index and has `actor_loss_mask = false`.

For a learned decision at time `t`, collection records the critic estimate
`V_old(observation_t)`. The engine emits an `EngineRoundResult`, from which
training derives one round-local return `R_round` for each player and the fixed
actor advantage `A_t = R_round - V_old(observation_t)`. The return definition
and critic target are specified in [model training](model-training.md).

## Round execution

Self-play and deployed policy turns follow the same transition sequence:

```text
1. Read the active player from EngineState.
2. Build PolicyTurnContext from the active state.
3. Run the policy to produce the next hidden state. Apply `forced_move`, or
   select an action through action_table.
4. Apply the resulting EngineMove through the pure engine.
5. Record a PolicyTransition and commit the next hidden state with the accepted
   engine transition.
6. Attach the round-local return to learned transitions after the engine
   completes the round.
```

Each player maintains its own recurrent hidden state. It starts as the all-zero
model state when the game is created, advances with every accepted move owned by
that player, and remains in place when the next round is dealt. A player receives
only views constructed for that player's turns. Every six-round game therefore
contains 24 ordered recurrent state updates per player: 21 learned decisions and
three forced recurrent transitions.

## Contract fixtures

The contract fixture set covers these cases:

| Fixture | Evidence |
| --- | --- |
| Seeded creation | Repeated creation with one game seed yields equal state fingerprints, dealer, stock, hands, and center card |
| Canonical hands | Equivalent dealt-card sets populate identical fixed hand slots regardless of deal order |
| Queen orientation | Global coffin and action positions map directly to policy positions |
| King orientation | Transposed policy coffin and actions invert to the original global positions |
| Action table | Every legal engine move occupies one legal action index; every masked index maps to `None` |
| Legal-mask agreement | Engine legal moves, action table, and `bool[4,8]` mask agree for initial, middle, and late-round states |
| Information boundary | Policy inputs comprise the active player's hand, public coffin and history, and hidden-card class |
| Recurrent sequence | Each player records four ordered policy transitions per round; the dealer's fourth transition uses the unique legal action and has a false actor-loss mask |
| Round result | The eighth move produces reproducible line calculations, selected round scores, totals, and round-return attachment points |
| Replay | Recorded action indexes and transition kinds reproduce engine states, policy contexts, engine moves, and round-result fingerprints |

# Engine–opponent contract

This document defines the deterministic boundary between the game engine,
information-set search, and a later search-guided model. Application
persistence and HTTP behavior belong in [architecture](architecture.md).

## Contract versions

Every fixture, search report, dataset, and model artifact records:

```text
rules_version
card_schema_version
engine_version
randomness_schema_version
information_state_schema_version
action_schema_version
search_schema_version
model_schema_version when applicable
value_schema_version when applicable
```

The engine fingerprint is the SHA-256 digest of the UTF-8 canonical JSON
serialization of a versioned `EngineState`. Field order and stored card order
are fixed. Search also hashes its player-visible information state separately;
the information-state digest must be identical for authoritative states that
differ only in fields hidden from that player.

## Card identity and order

The engine uses these 54 card IDs in this exact order:

```text
AC, 2C, 3C, 4C, 5C, 6C, 7C, 8C, 9C, 10C, JC, QC, KC,
AD, 2D, 3D, 4D, 5D, 6D, 7D, 8D, 9D, 10D, JD, QD, KD,
AH, 2H, 3H, 4H, 5H, 6H, 7H, 8H, 9H, 10H, JH, QH, KH,
AS, 2S, 3S, 4S, 5S, 6S, 7S, 8S, 9S, 10S, JS, QS, KS,
V1, V2
```

The zero-based card index is its position in the list. `V1` and `V2` are
distinct physical cards with identical Vampire behavior. Hands, card-indexed
representations, serialized state, and fixtures use this order.

## Deterministic seeds and shuffle

All deterministic streams use:

```text
derive_seed(namespace, component_1, ..., component_n) =
    SHA-256(UTF8(namespace) || NUL || UTF8(component_1) || ...
            || NUL || UTF8(component_n))
```

The versioned namespace is the first component. Arguments are Unicode strings
encoded as UTF-8 and separated by one `0x00` byte. Integer components use
unsigned base-10 without leading zeroes; digest components use lowercase
hexadecimal. NUL is forbidden in every argument. The result is the full 32-byte
digest; its integer form is unsigned big-endian.

The engine derives independent values from a supplied game seed:

```text
shuffle_seed = derive_seed("dracula-engine-shuffle-v1", game_seed)
dealer_seed  = derive_seed("dracula-engine-initial-dealer-v1", game_seed)
```

The initial dealer is Queen for an even dealer-seed integer and King for an odd
one. Dealer selection does not consume the shuffle stream.

The shuffle uses a SHA-256 counter stream. Starting at counter zero, each block
is:

```text
derive_seed(
    "dracula-sha256-counter-v1",
    lowercase_hex(shuffle_seed),
    decimal(counter),
)
```

For `randbelow(n)`, interpret the block as a 256-bit unsigned integer `x`, set
`limit = 2^256 - (2^256 mod n)`, reject `x >= limit`, and otherwise return
`x mod n`. The engine applies Fisher-Yates from index 53 through 1 and draws
from the front of the resulting tuple. It does not use a language-runtime
shuffle.

## Engine state and operations

`EnginePlayer` has values `queen` and `king`. The pure engine owns dealing,
legality, transitions, scoring, lifecycle validation, and final outcome.

```python
class EngineState:
    seed: str
    status: Literal["playing", "round_complete", "game_complete"]
    round_number: int
    dealer: EnginePlayer
    active_player: EnginePlayer | None
    stock: tuple[str, ...]
    hands: PlayerValues[tuple[str | None, str | None, str | None, str | None]]
    coffin: tuple[str | None, ...]
    current_round_moves: tuple[EnginePlayedMove, ...]
    pending_round_result: EngineRoundResult | None
    completed_rounds: tuple[EngineRoundResult, ...]
    total_scores: PlayerValues[int]


class EngineMove:
    player: EnginePlayer
    hand_slot: int
    global_grid_index: int


class EnginePlayedMove:
    player: EnginePlayer
    card_id: str
    hand_slot: int
    global_grid_index: int
    turn_number: int


class EngineRoundResult:
    round_number: int
    dealer: EnginePlayer
    coffin: tuple[str, ...]
    moves: tuple[EnginePlayedMove, ...]
    line_scores: PlayerValues[tuple[LineScore, LineScore, LineScore]]
    round_scores: PlayerValues[int]


def create_game(seed: str) -> EngineState: ...
def legal_moves(state: EngineState, player: EnginePlayer) -> tuple[EngineMove, ...]: ...
def apply_move(state: EngineState, move: EngineMove) -> EngineTransition: ...
def advance_after_round(state: EngineState) -> EngineState: ...
```

`create_game` shuffles the canonical deck, selects the dealer independently,
deals two cards to the non-dealer, two to the dealer, two to the non-dealer,
and two to the dealer, then places the next card in the center. Each hand sorts
by canonical card index into four stable slots. The remaining stock retains its
order and the non-dealer becomes active.

`legal_moves` resolves a card from the active player's hand slot and returns
legal placements ordered by ascending hand slot and grid index. `apply_move`
validates lifecycle, player, hand slot, vacancy, and orthogonal adjacency. The
eighth accepted placement computes line scores, tie resolution, round scores,
and cumulative totals. `advance_after_round` alternates the dealer and consumes
the next nine cards through the same pair deal, or completes the game after
round six.

Every operation validates its input and returns a new immutable state. Typed
violations cover malformed state, wrong player, unavailable slot, invalid or
occupied position, non-adjacent placement, and invalid lifecycle transition.

## Public and private move records

The private engine record retains `hand_slot` for deterministic reconstruction.
The public/search record does not expose the former slot of an opponent card:

```python
class PublicPlayedMove:
    player: EnginePlayer
    card_id: str
    global_grid_index: int
    turn_number: int
```

The acting player's current hand uses stable slots because those slots are
private information already known to that player. Public API responses and
search histories use `PublicPlayedMove`. A public opponent slot would reveal
ordering information about cards still hidden in the canonically sorted hand.

## Player-relative information state

Search receives a complete information state for one player, not an
authoritative `EngineState`:

```python
class SearchInformationState:
    schema_version: str
    player: EnginePlayer
    round_number: int
    dealer: EnginePlayer
    active_player: EnginePlayer
    turn_number: int
    total_scores: PlayerValues[int]
    completed_rounds: tuple[PublicRoundRecord, ...]
    own_hand: tuple[str | None, str | None, str | None, str | None]
    coffin: tuple[str | None, ...]                 # player-relative
    current_round_moves: tuple[PublicPlayedMove, ...]
    played_card_ids: tuple[str, ...]               # canonical membership
    unseen_card_ids: tuple[str, ...]               # canonical order
    opponent_remaining_count: int
    stock_count: int
    legal_mask: BoolTensor[4, 8]
```

`completed_rounds` contains public coffins, public move records, and scoring.
`unseen_card_ids` contains exactly the cards in the opponent's remaining hand
and stock. It identifies a belief support, not a location for any card.

Own hand, coffin, unseen-card membership, dealer and decision progress, and
legal actions form the Markov-sufficient core for the round-local planners.
Public move history additionally supports validated engine-state
reconstruction and privacy audit; the initial uniform belief does not infer
hidden cards from its order.

Queen retains global grid coordinates. King transposes them with
`index -> 3 * (index % 3) + index // 3`; the transform is self-inverse. The
coffin, public destinations, and legal actions use the same transform, so the
player's three scoring lines are always rows.

The information state validates these properties:

- Its public projection agrees with the authoritative state.
- Own occupied hand slots are canonical and match own legal actions.
- Public, own-hand, and unseen card IDs partition all 54 cards.
- Current moves have contiguous turn numbers and alternate from the non-dealer.
- No opponent slot, opponent remaining card, stock order, seed, or private
  fingerprint is present.

## Neural observation

The shared policy/value model uses the existing player-relative Boolean
projection and a separate legal mask:

```text
observation[0:216]     = own hand, bool[4, 54]
observation[216:702]   = player-relative coffin, bool[9, 54]
observation[702:756]   = played cards, bool[54]
observation[756:810]   = own remaining hand membership, bool[54]
observation[810:864]   = unseen cards, bool[54]
observation[864:870]   = round index, one-hot bool[6]
observation[870:874]   = own decision index, one-hot bool[4]
observation[874]       = acting player is dealer, bool
legal_mask             = bool[4, 8]
```

The observation has exactly 875 elements. The three status vectors XOR to 54
true values. Player identity, public move order, completed-round layouts,
cumulative scores, and authoritative state are absent. Search retains the
public history required for determinization separately.

## Action mapping

The fixed action space is four private hand slots by eight non-center positions.
For action index `i`:

```text
hand_slot = i // 8
policy_position_index = i % 8
policy_grid_index = [0, 1, 2, 3, 5, 6, 7, 8][policy_position_index]
```

The context contains a 32-entry action table. A legal entry contains the global
`EngineMove`; a masked entry is `None`. The mask and table are derived from one
call to `legal_moves` and must agree exactly. Player-relative action conversion
round-trips for Queen and King.

When exactly one action is legal, the engine supplies it without running
search. The action remains in deterministic replay and search datasets as a
forced placement, but it supplies no policy target.

## Determinization boundary

The determinization builder accepts only `SearchInformationState`, public
engine records, a simulation seed, and the public lifecycle shape. It creates a
valid simulation-only `EngineState` as specified in
[information-set search](search.md#root-belief-and-determinization).

Opponent cards already played in the round and sampled remaining cards form a
complete four-card hand. Canonical sorting determines simulated slots, and the
private slot fields of opponent move records are rebuilt accordingly. The root
player's original slots are reconstructed from their current hand and public
cards played by that player, then canonically sorted. Remaining hidden cards
receive a sampled stock order solely to satisfy engine conservation and
validation.

The authoritative opponent hand, stock, engine seed, and private move slots are
not arguments. Tests construct pairs of authoritative states with equal player
views and different hidden assignments; both must yield identical information
states and identical search results for the same search seed.

The engine represents a determinization as a typed simulation state carrying
the sampled 54-card deal order solely to validate dealing, slots, history, and
conservation. It supports normal moves through completion of the current round
and cannot advance into another round. Its deck provenance never enters a
player information state, public response, or model input.

During a rollout, every acting player receives a newly projected
`SearchInformationState` from the sampled world. The other player's remaining
hand is hidden from that projection. Version 1 selects simulated opponent
actions with its uniform legal policy. Teacher v2 passes the projection to the
shallow greedy response evaluator.

For each response completion index, Teacher v2 samples one world from the actor
projection and evaluates every legal candidate action from that same world.
Each candidate is applied to an immutable copy and completed with uniform legal
play. The response evaluator selects the maximum mean exact round
differential. Response worlds are discarded; only the selected action is
applied to the unchanged outer simulation state.

The Teacher v2 response seed and cache key are functions of the actor
information-state digest and response configuration, never the enclosing
sampled world. Equal actor views therefore produce equal response samples,
mean values, and actions. This invariance is required even when the enclosing
root hand and stock differ.

## Search and neural samples

Search produces one decision record at each non-forced real turn:

```python
class SearchDecision:
    information_state_digest: str
    action_table: tuple[EngineMove | None, ...]
    selected_action_index: int
    visits: tuple[int, ...]            # length 32
    mean_returns: tuple[float | None, ...]  # length 32
    simulation_count: int                  # outer root simulations
    search_config_digest: str


class SearchTrainingSample:
    observation: BoolTensor[875]
    legal_mask: BoolTensor[4, 8]
    search_policy: FloatTensor[32]
    selected_action_index: int
    round_return: float
    information_state_digest: str
    search_config_digest: str
    fixture_id: str
    player: EnginePlayer
    dealer: EnginePlayer
    round_number: int
    placement_number: int
```

`search_policy` is derived only from legal root visit counts. `round_return` is
attached after the engine completes that round. Contract versions accompany the
serialized row. Forced placements have no `SearchTrainingSample`. The full
information state, public history, search diagnostics, and determinizations are
not serialized into model datasets; application events retain only the accepted
public move and resolved opponent configuration.

Teacher v2 retains this boundary unchanged. Response samples, action values,
cache entries, and terminal evaluations are private search diagnostics; they
do not become model rows. Its `search_config_digest` commits to the outer
budget, response completions per action, selection profiles, and seed
namespaces.

## Contract fixtures

Fixtures cover:

| Area | Required evidence |
| --- | --- |
| Cards and shuffle | Canonical indexes, seed vectors, dealer, deck digest, pair deal, and fingerprints remain unchanged |
| Projection | Queen/King transpose is self-inverse and equal information states ignore hidden assignments |
| Public history | Actor, card, destination, and order are retained; opponent hand slots are absent |
| Action table | Engine legality, mask, table, and inverse mapping form one exact bijection |
| Determinization | Every sample conserves 54 cards, preserves public facts and root hand, and varies only hidden assignments |
| Opponent view | Root-private substitutions do not alter opponent inputs or seeded opponent decisions |
| Shallow response | Equal actor views in different outer worlds reproduce shared candidate samples, mean values, and actions |
| Candidate fairness | Every legal response action uses the same determinization at one completion index |
| World isolation | Response sampled cards never replace cards in the outer trajectory; both worlds remain independently engine-valid |
| Terminal result | Search payoff equals the engine round-score differential divided by 150 and changes sign by player |
| Replay | Search seeds, samples, actions, terminal scores, and report digests reproduce exactly on CPU |

The seeded creation fixture remains `engine-contract-fixture-1` with initial
dealer King, shuffled-deck digest
`f4d82c3e12ab9d6cf37d777400e69a702e751ced8d5215d1716248a2aa66583f`,
Queen hand `4D, 6H, AS, 10S`, King hand `9C, 3H, 8H, 10H`, center `7H`,
and remaining stock beginning `2S` and ending `10C`.

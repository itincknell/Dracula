# Stateless gameplay API

This contract defines production gameplay over the deterministic engine. The
browser retains a plain recovery envelope; the service reconstructs the game
without a database or server session. Local development uses this same API.

## Runtime selection

Production loads `dracula.api.production:app`. Local development loads
`dracula.api.development:app`, and dummy-dialogue previews load
`dracula.api.preview:app`. All three compose the same stateless routes.

Production configuration is centralized in `dracula.api.config`:
`DRACULA_POLICY_ARTIFACT`, `DRACULA_NARRATION_ENABLED`, and
`DRACULA_REPLAY_CACHE_ENTRIES`. The production entry point passes the selected
policy directly to the stateless service. There is no gameplay-mode switch or
repository configuration.

## Recovery envelope

The complete browser-owned state is:

```json
{
  "seed": "plain game seed",
  "history": [
    {"type": "select_role", "human_role": "queen"},
    {"type": "place", "hand_slot": 0, "position": 1},
    {"type": "advance_round"}
  ]
}
```

The first command selects the human role exactly once. Later commands are
human placements or explicit round advances. Dracula placements are not
trusted browser commands: replay regenerates them through the selected `pi1`
policy at the same deterministic decision boundary.

The envelope contains no lifecycle counter, request ID, application or model
version, server token, signature, or encrypted state. The seed is nonempty,
contains no NUL byte, and is at most 256 characters. A complete game has at
most 31 commands: one role selection, 24 human placements, and six round
advances.

The round-six advance occurs after the final scoring presentation and converts
the engine's sixth `round_complete` state into `game_complete`. It does not
deal another round.

## Routes

The paths below are relative to `/Dracula/api` in the combined production app.
Local Vite development still proxies its `/api` prefix to these unprefixed
service routes. Static files are mounted separately at `/Dracula/`; API routes
are registered first and never fall back to frontend HTML.

### `GET /health`

Stateless health returns only:

```json
{
  "status": "ok",
  "opponent_configured": true,
  "narration_enabled": false
}
```

It exposes neither an execution-environment identity nor model metadata.

### `POST /games`

The request contains `human_role` and an optional seed. When the seed is
omitted the service creates a random 256-bit hexadecimal seed. The response is
HTTP 201 and contains the new envelope plus the authoritative human view. If
Dracula is the non-dealer, the required opening move is completed before the
view is returned.

### `POST /games/command`

The request contains the current envelope and exactly one `place` or
`advance_round` command. The service reconstructs the submitted envelope,
validates and applies the new command, completes the required Dracula turn,
and returns the extended envelope and authoritative view.

The same exact request body always returns the same body. There is no mutable
session in which a retry could apply the command twice.

### `POST /games/resume`

The request contains an envelope and no mutation. The service reconstructs and
returns its authoritative view. This route is used after browser reload and is
identical on a warm cache hit or cold replay.

### `POST /narration`

The request contains the same recovery envelope and exactly one cue type:
`opening`, `round_transition`, or `final_result`. It does not contain cue facts,
a prompt, or a model selector. The service replays the envelope, verifies cue
eligibility, derives public facts, and invokes the configured narrator
separately from gameplay.

Opening is eligible only immediately after game creation. Round transition is
eligible only for a `round_complete` state in rounds 1–5. Final result is
eligible only after the sixth `advance_round` command has produced
`game_complete`. An ineligible cue returns HTTP 409 and never invokes Bedrock.

Ready narration returns text. Disabled narration, timeout, provider failure, or
malformed output returns HTTP 200 with `status: unavailable` and `text: null`.
This response never changes the submitted envelope or re-applies a command.

## Replay rules

On a cache miss the service:

1. Creates the seeded engine state.
2. Uses the human role and game seed directly when a selected strategic group
   needs a deterministic paired-destination coin flip.
3. Runs a required Dracula opening turn.
4. Replays each human command through normal engine legality.
5. Runs each required non-forced Dracula move from Dracula's
   `SearchInformationState` and representative action mask.
6. Applies forced Dracula placements directly through the bridge.

Any invalid or masked policy action fails the request without extending the
envelope or caching a result.

Commands with impossible order, occupied destinations, unavailable cards,
premature advances, repeated role selection, or invalid lifecycle transitions
are rejected. An older valid envelope remains a valid branch point, matching
the accepted single-player stateless design. Only invalid divergence is
rejected.

## Replay cache

Each Lambda environment owns a bounded least-recently-used cache keyed directly
by the immutable recovery envelope. Entries contain immutable reconstructed
engine state, human role, and the plain role-and-seed key used for deterministic
paired-destination choices.

`DRACULA_REPLAY_CACHE_ENTRIES` controls the bound and defaults to 256. Zero
disables storage. Cache clearing, eviction, and process replacement change
latency only; they never change replay results.

## Public response boundary

Successful gameplay responses contain exactly:

```text
envelope
game
```

The game projection contains public coffin and scoring information, the human
hand, legal human card-slot/destination choices, totals, and the derived
presentation phase. Public move records omit former hand slots, including
Dracula's. The projection contains no internal policy key, stock, opponent
hand, engine object, state fingerprint, policy identity, artifact digest,
observation, legal mask, logits, tensors, search diagnostics, or
Lambda instance identity.

The plain seed/history envelope remains intentionally inspectable and is not
an anti-cheat boundary.

## Error behavior

Requests reject unknown JSON fields. Stateless errors use a compact code,
message, and retryable flag:

- `validation_error`: malformed wire request.
- `invalid_history`: a submitted envelope cannot be replayed.
- `invalid_command`: the new placement is not legal.
- `wrong_turn` or `wrong_phase`: the new command does not fit the reconstructed
  lifecycle.
- `ineligible_cue`: narration does not match the reconstructed lifecycle.
- `dependency_unavailable`: policy inference failed before any returned
  envelope was extended; retrying the original body is safe.

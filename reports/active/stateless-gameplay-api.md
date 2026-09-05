# Stateless gameplay API implementation

Date: 2026-08-30

## Result

The production gameplay boundary is implemented as deterministic
seed-and-command-history replay. It requires no repository, server session,
sticky Lambda environment, request signature, or hidden browser token.

The explicit production ASGI entry point is:

```text
dracula.api.production:app
```

It always constructs stateless mode and cannot fall back to the local
repository-backed service. `dracula.api.app:app` remains the explicit local
entry point and preserves the existing in-memory/SQLite API.

## Public contract

The production routes are:

| Method and route | Result |
| --- | --- |
| `GET /health` | Stateless mode and whether an opponent is configured |
| `POST /games` | Seed, role command, required opening `pi1` move, and initial view |
| `POST /games/command` | Replay, one human placement or round advance, required `pi1` move, and extended envelope |
| `POST /games/resume` | Replay without mutation |

The browser envelope contains exactly `seed` and `history`. History begins with
one role selection and then contains human placements and round advances. A
complete six-round game contains 31 commands: one role, 24 human placements,
and six advances. Dracula moves are regenerated from `pi1`; they are not
trusted client commands.

Successful responses contain exactly `envelope` and `game`. Public move records
omit former hand slots. The response omits the internal deterministic game ID,
stock, opponent hand, engine state, state fingerprint, policy identity,
artifact metadata, observations, masks, logits, tensors, and
search diagnostics.

The seed intentionally remains public and therefore remains an accepted
cheating surface.

## Replay and transactions

Every request reconstructs an immutable engine state from a valid envelope.
Human placement commands use direct card-slot and destination coordinates and
are revalidated against exact engine legality. They do not use the local API's
per-session HMAC move tokens.

The service synchronously completes a required Dracula turn before returning.
Non-forced turns receive only Dracula's existing `SearchInformationState`,
policy observation, legal action table, and zero recurrent state. Stateless
`pi1` must return no recurrent state. Forced placements bypass model inference
and pass through the bridge and engine directly.

There is no mutation before a response: retrying an identical envelope and
command reproduces the same result. An older valid envelope intentionally
creates an independent deterministic branch. Impossible ordering, duplicated
commands, illegal destinations, unavailable slots, premature advances, and
malformed histories fail without returning an extended envelope.

The round-six advance converts the sixth scored round into `game_complete`; it
does not deal another round.

## Cache

`ReplayCache` is a process-local, thread-safe, bounded least-recently-used
cache. Its key is the SHA-256 digest of canonical envelope JSON. Its value is
an immutable reconstructed engine state, human role, and internal deterministic
game identity.

The default bound is 256 entries through
`DRACULA_REPLAY_CACHE_ENTRIES`; zero disables storage. Cache statistics remain
private. Tests prove identical responses after hits, clearing, eviction,
zero-size operation, and construction of a new application process.

Request JSON is capped at 64 KiB before parsing. The recovery envelope is
additionally bounded to a 256-character seed and 31 commands, with unknown
fields rejected.

## Verification

The final validation used the latest source:

- Complete Python suite: **651 passed** in 692.13 seconds.
- Stateless API contract cases: **13 passed** in the full suite.
- Frontend unit/component tests: **74 passed**.
- TypeScript: passed.
- ESLint: passed.
- Verified production frontend build: passed with no hard-coded local API URL.
- Python bytecode compilation and dependency check: passed.
- Diff formatting: passed.
- Markdown links: 57 files checked with zero broken local links.
- Tracker IDs: confined to `docs/design-tracker.md`.

Two additional complete games used the exact selected `pi1` artifact with
SHA-256
`70c76f2eb64600eab2297640278a6c94d4336ab8e73bf941a6d96237f69f5b5c`.
Queen and King each completed six rounds with 31-command envelopes, 24 human
placements, retry-identical command responses, and an identical final view
after clearing the replay cache.

The later minimal artifact contains the identical tensors under SHA-256
`d35196cf4513589def0ffb3c4c7c268e78652a46dab8ea41001ff2c648265203`;
the current complete Python and browser suites repeat this behavior check.

The tests also cover both initial dealer identities, cold reconstruction at
every returned lifecycle state, canonical cache keys, eviction, malformed and
reordered histories, valid branching, dependency failure without commitment,
forced-policy bypass, lightweight production imports, and recursive public
response privacy inspection.

## Remaining deployment dependencies

The subsequent
[Bedrock narration implementation](bedrock-narration-implementation.md) added
the fifth production route without changing the gameplay replay contract.

At the end of this earlier stage, it had not implemented or begun:

- Browser envelope persistence and the new frontend API client.
- Bedrock narration or prompt design.
- Lambda container packaging or Lambda Web Adapter.
- API Gateway, IAM, ACM, CloudWatch, ECR, budgets, or Cloudflare configuration.
- Staging or production deployment.

The narration item is now complete; the remaining items continue in later
deployment stages.

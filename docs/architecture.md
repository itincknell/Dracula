# Application architecture

## Final deployment shape

```text
ian-tincknell.com/Dracula/          api.ian-tincknell.com
GitHub Pages React frontend  --->  API Gateway HTTP API
                                          |
                                          v
                                  AWS Lambda container
                                  FastAPI + engine + pi1
                                          |
                                          `-- Amazon Bedrock narration
```

Cloudflare remains the DNS provider. The existing apex-domain records continue
to direct `ian-tincknell.com` to GitHub Pages. A separate `api` CNAME directs
`api.ian-tincknell.com` to the Regional API Gateway custom-domain target.
GitHub Pages hosts only static frontend files. API Gateway invokes the Lambda
container through the AWS Lambda Web Adapter.

There is no SageMaker endpoint, production database, shared server session, or
search process. The selected opponent is the standalone `pi1` policy model.
FastAPI, the deterministic engine, and the exact selected model artifact ship
in one Lambda image.

The deployment details and ordered release work are authoritative in
[deployment](deployment.md).

## Stateless game state

The browser owns the recoverable game envelope:

```text
initial game seed
ordered accepted command history
```

The history contains the selected human role, card placements, and explicit
round-advance commands. It contains no redundant lifecycle counter, application
version, model version, schema version, signature, or encrypted server token.
The deterministic engine derives the deal, current round, dealer, active actor,
coffin, hands, scores, and completion state by replay.

Each game request submits the current envelope and one requested command.
FastAPI:

1. Parses bounded JSON and rejects unknown fields or excessive history.
2. Looks up the envelope digest in the current Lambda environment's local
   reconstruction cache.
3. On a miss, creates the seeded game and replays every command through normal
   engine validation.
4. Applies the requested command through the same engine boundary.
5. Runs `pi1` for each required non-forced Dracula move.
6. Returns the updated envelope, the human view, and any scoring or narration
   cue produced by the transition.

Lambda memory is an optimization only. API Gateway does not provide execution-
environment affinity, so every request remains correct on a cache miss. The
same envelope and command reproduce the same resulting branch. Submitting an
older envelope intentionally creates an independent branch; accounts,
authoritative anti-cheat, shared sessions, and competitive result integrity are
outside scope.

The seed allows a user inspecting browser data to recover Dracula's hand and
stock. That tradeoff is explicitly accepted for this single-player release.
Model inputs remain player-relative: `pi1` receives only Dracula's established
visible observation and representative-action mask even though the Lambda
engine reconstructs the complete state.

## Game lifecycle

Game creation selects Queen or King, creates a random seed, stores the initial
envelope in browser storage, returns the dealt human view, and emits the opening
narration cue. Human and Dracula turns alternate through eight placements.
Rules, legality, scoring, forced placements, and the six-round deck lifecycle
remain owned by the deterministic engine.

The eighth placement returns the complete server-calculated scoring sequence.
The scoring animation changes presentation state only. For rounds one through
five, **Deal Next Round** appends an advance command and deals the next round.
After round six the engine returns the final game result and **Play Again**.

Reload reads the last confirmed envelope from browser storage and asks the API
to reconstruct its human view. Losing browser storage loses the unfinished
game. Cross-device recovery is not part of the release.

## Opponent decision

The selected `pi1` controller performs exactly one standalone policy inference:

1. Project Dracula's player-visible 659-bit observation.
2. Build the engine-legal and representative-only action masks.
3. Run the 754,601-parameter feed-forward policy.
4. Select the highest legal representative logit with canonical tie-breaking.
5. Resolve a paired symmetric destination with the established deterministic
   fair coin.
6. Revalidate and apply the concrete move through the engine.

The selected artifact is
`runs/bgc-policy-pi1-001/artifacts/unaccepted-candidate.pt` in the local evidence
store, with SHA-256 digest
`70c76f2eb64600eab2297640278a6c94d4336ab8e73bf941a6d96237f69f5b5c`.
The word `unaccepted` is historical training nomenclature; the user has now
selected this exact artifact for production. The release build copies and
verifies it inside the Lambda image.

Nested Sam, BGC controllers, `pi0`, PPO, response rankers, and hybrid search are
explicit historical or diagnostic controls. They are not production fallbacks.

## Narrator scheduling

Narration is direct bounded Bedrock inference and never participates in rules
or opponent selection. There are exactly three cue classes:

| Cue | Timing |
| --- | --- |
| Game opening | Requested after game creation and presented when ready |
| Round transition | Rounds 1–5 only; requested after the eighth placement, generated during scoring, and revealed after the animation |
| Final game result | Round 6 only; requested from the completed-game result instead of a round-transition cue |

There are no per-move, per-player-score, or round-six transition calls. The
browser coordinates animation completion and narration readiness. The
narration request may reach a different Lambda environment because it carries
its complete public cue. Failure or delay never changes the game envelope.

Bedrock receives only public round or game facts and character instructions.
It receives neither hand, stock, seed, policy input, logits, nor intended move
strategy.

## API surface

The final route names may adapt the existing FastAPI application, but their
responsibilities are fixed:

| Route | Responsibility |
| --- | --- |
| `POST /games` | Create the seed-and-history envelope and opening human view |
| `POST /games/command` | Replay an envelope, apply one human or advance command, run required `pi1` moves, and return the next envelope/view |
| `POST /narration` | Generate text from one eligible public cue |
| `GET /health` | Report application, model-load, and Bedrock configuration health |

The frontend calls `https://api.ian-tincknell.com` explicitly. API Gateway CORS
allows the deployed GitHub Pages origin. No route accepts a caller-selected
model, prompt, controller, or engine operation.

## Local architecture

SQLite and the existing session-oriented API remain useful local development
and historical test infrastructure. They are not the production persistence
design. In-memory repositories remain unit-test fixtures. The final sprint adds
the stateless routes and replay cache without deleting the validated local
engine, transaction, or recovery tests.

## Required invariants

- Replaying one envelope always reconstructs the same engine state.
- Every replayed command is validated in order; malformed histories fail.
- Cache hit and cache miss return identical public state and actions.
- A Lambda environment change is invisible to gameplay.
- `pi1` never selects an illegal or masked action.
- Scoring comes only from the deterministic engine.
- Round 6 emits only the final-result narration cue.
- Bedrock failure cannot block scoring, round advancement, or game completion.
- Public API responses never contain model logits, masks, tensors, or prompts.

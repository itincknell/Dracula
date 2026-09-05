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

The production image and CloudFormation definitions are implemented and
locally validated. The image embeds the digest-verified `pi1`, runs the
stateless FastAPI entry point through Lambda Web Adapter, and requires no
persistent filesystem. Staging has not been created because the available AWS
CLI session is not authenticated; see the
[packaging report](../reports/active/lambda-packaging-and-infrastructure.md).

## Module ownership

The refactored Python runtime has explicit, cycle-free owners:

- `dracula.engine` is the stable gameplay facade and owns legal moves,
  transitions, and lifecycle orchestration. Domain values, dealing, scoring,
  validation, and canonical serialization live in `engine_types`,
  `engine_dealing`, `scoring`, `engine_validation`, and
  `engine_serialization` respectively.
- `search.information` owns actor-visible state and fingerprints;
  `search.symmetry` owns the exhaustive early-turn destination table;
  `search.determinization` reconstructs private sampled worlds; and
  `search.bgc` owns the retained round-local UCT. `search.belief_greedy` and
  `search.policy_continuation` provide the original and phase-two response
  policies. These search controllers support provenance and comparison but are
  not invoked by production gameplay.
- `strategic_actions` owns strategic groups and deterministic concrete-member
  choice. `action_contract` owns representative masks, proxy mapping, and
  standalone output resolution.
- `bgc_policy_model` owns the compact policy network, `bgc_policy` owns its
  artifact contract, and `active_policy` is the selected `pi1` runtime. The
  retained training path separates converted-corpus loading, metrics,
  optimization, checkpoint persistence, contracts, immutable configuration,
  command-line handling, and run coordination into the corresponding
  `bgc_policy_*` modules.
- `api.stateless_service` owns replay and command application;
  `api.stateless_projection` owns the public view; `api.stateless_contracts`
  owns production wire models; and `api.stateless_routes` plus
  `api.stateless_app` own HTTP assembly. `api.production` is the Lambda entry
  point. Repository-backed sessions remain isolated in the explicit local
  application: `api.session` owns persisted values, `api.local_projection`
  owns browser-safe views and move tokens, `api.local_session_events` owns
  event construction, and `api.service` owns transactions.
- `api.narration_cues` owns grounded cue construction, `api.narration` owns
  provider-independent orchestration and failure isolation, and `api.bedrock`
  owns provider transport.

Maintained code imports these owner modules directly. The retained model
compatibility code is the verified reader for the sealed 659-bit `pi1`
training corpus; restored BGC code uses the current information, symmetry,
observation, and artifact contracts directly.

The frontend follows the same separation:

- `contractPrimitives` owns shared public values and validators,
  `statelessContracts` owns the production wire contract, and `gameView` owns
  the smaller view consumed by presentation components.
- `statelessApi`, `statelessRecovery`, `statelessProjection`, and
  `statelessNarration` own transport, browser persistence, pure response
  projection, and narration coordination. `statelessGameStore` owns gameplay
  sequencing. The frontend supports only the stateless gameplay transport;
  local session gameplay remains a Python API and operational-record surface.
- `gameplay`, `DraculaCommentary`, `scoringStateMachine`,
  `ScoringWorkspaces`, and `ScoringPresentation` own presentation;
  `SeenCardsExpando` owns the public card ledger. Ordered `styles/base.css`,
  `gameplay.css`, `scoring.css`, `rules.css`, and `responsive.css` preserve the
  approved cascade through `styles.css`.

## Stateless game state

The browser owns the recoverable game envelope:

```text
initial game seed
ordered accepted command history
```

The history contains the selected human role, human card placements, and
explicit round-advance commands. Dracula moves are regenerated by the selected
deterministic policy. It contains no redundant lifecycle counter, application
version, model version, schema version, signature, or encrypted server token.
The deterministic engine derives the deal, current round, dealer, active actor,
coffin, hands, scores, and completion state by replay.

Each mutation submits the current envelope and one requested command.
FastAPI:

1. Parses bounded JSON and rejects unknown fields or excessive history.
2. Looks up the envelope digest in the current Lambda environment's local
   reconstruction cache.
3. On a miss, creates the seeded game and replays every command through normal
   engine validation.
4. Applies the requested command through the same engine boundary.
5. Runs `pi1` for each required non-forced Dracula move.
6. Returns the updated envelope and authoritative human view, including any
   scoring result produced by the transition.

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
envelope in browser storage, returns the dealt human view, and makes the opening
narration request separately. If Dracula is the non-dealer, its opening move is
completed synchronously before the view returns. Human and Dracula turns
alternate through eight placements.
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
`runs/bgc-policy-pi1-001/artifacts/pi1-policy.pt` in the local evidence
store, with SHA-256 digest
`d35196cf4513589def0ffb3c4c7c268e78652a46dab8ea41001ff2c648265203`.
It contains the selected weights and one format marker. The release build
copies and verifies it inside the Lambda image.

The repository retains the BGC-128 search lineage needed to review and reproduce
the training process: the original eight-completion belief-greedy continuation
and the phase-two `pi0` continuation. These controllers use the current 659-bit
policy boundary and symmetry contract, but they are not production gameplay
fallbacks. Sam, shallow search, PPO, response rankers, and hybrid-search
implementations remain historical evidence only.

## Narrator scheduling

Narration is direct bounded Bedrock inference and never participates in rules
or opponent selection. There are exactly three cue classes:

| Cue | Timing |
| --- | --- |
| Game opening | Requested after game creation and presented when ready |
| Round transition | Rounds 1–5 only; requested after the eighth placement, generated during scoring, and revealed after the animation |
| Final game result | Round 6 only; requested from the completed-game result instead of a round-transition cue |

There are no per-move, per-player-score, or round-six transition calls. The
browser coordinates animation completion and narration readiness. A request
may reach a different Lambda environment because it carries the complete
recovery envelope and cue type. The server replays the envelope and derives the
public cue; the browser does not submit facts. Failure or delay never changes
the game envelope.

Bedrock receives only public round or game facts and character instructions.
It receives neither hand, stock, seed, policy input, logits, nor intended move
strategy.

## API surface

The production FastAPI route surface is:

| Route | Responsibility |
| --- | --- |
| `POST /games` | Create the seed-and-history envelope and opening human view |
| `POST /games/command` | Replay an envelope, apply one human or advance command, run required `pi1` moves, and return the next envelope/view |
| `POST /games/resume` | Reconstruct one envelope without applying a command |
| `POST /narration` | Replay an envelope, derive one eligible public cue, and generate optional text |
| `GET /health` | Report application, model-load, and Bedrock configuration health |

The exact implemented gameplay schema is in
[stateless gameplay API](stateless-api.md). The frontend calls
`https://api.ian-tincknell.com` explicitly. API Gateway CORS
allows the deployed GitHub Pages origin. No route accepts a caller-selected
model, prompt, controller, or engine operation.

## Local architecture

SQLite and the session-oriented API remain narrowly isolated local-development
and historical-record infrastructure. The normal local preview uses the same
stateless routes and replay cache as production.

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

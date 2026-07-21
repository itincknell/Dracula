# Design decisions

## Deterministic service owns game truth

Pure Python code owns dealing, legal moves, validation, transitions, scoring,
and factual effects. The policy may return an action and the narrator may
paraphrase supplied facts, but neither can mutate or recalculate game truth.

The canonical deck order is `AC` through `KC`, `AD` through `KD`, `AH` through
`KH`, `AS` through `KS`, then `V1` and `V2`. Each round deals pairs to the
non-dealer, dealer, non-dealer, and dealer before taking the next card as the
center; completed hands sort by canonical card index. Deterministic randomness
uses versioned SHA-256 seed namespaces, a SHA-256 counter stream, unbiased
`randbelow`, and Fisher-Yates rather than a language-runtime shuffle. Initial
dealer selection uses an independent derived seed.

## A trained recurrent policy plays Dracula

Dracula's moves are selected by a recurrent neural policy trained through
self-play. The policy receives only its legitimate player view and a
server-issued legal-action mask. LLM move selection, hand-authored strategy
Oracles, and live deterministic substitutes are not part of the design.

Five active policies provide opponent diversity during training. One strongest
checkpoint is selected manually for deployment. The application exposes that
policy as its single opponent and single difficulty mode.

Model training is a separate build with its own architecture, system
requirements, checkpoints, and result reporting. It shares the
authoritative game engine and versioned tensor contract with the web
application, not application persistence or request orchestration.

## Training is local and serving uses SageMaker

Training runs locally with PyTorch on the project M3 MacBook Air. The engine,
fixture scheduler, trajectory storage, and action sampling remain on CPU.
Collection and optimization devices are configured independently; the initial
configuration uses CPU collection and MPS optimization. The implementation uses
float32, CPU trajectory buffers, and conservative minibatches for the machine's
8 GB unified memory.

The default held-out pass uses twelve lane roots and one generation, producing
120 games and 5,040 learned critic-validation rows. Critic burn-in requires a
configurable five-percent MSE improvement over zero prediction for three
consecutive windows after at least 10,000 collected training rounds.

The manually selected policy artifact is versioned in SageMaker Model Registry
and served through one SageMaker Serverless Inference endpoint. An IAM-restricted
application worker invokes the endpoint and the game service revalidates every
returned action. SageMaker Training is not used; AWS model infrastructure begins
with the selected artifact.

## Policy version and recurrence are pinned per game

Each game records one selected model version, tensor schema, inference
configuration, and recurrent hidden-state format. These values cannot change
during the game. The hidden state begins at game creation and persists through
all six rounds. Its next value is committed transactionally with an accepted
opponent move; a failed or retried inference cannot advance it alone.

The SageMaker container is stateless. The application supplies the prior hidden
state on every invocation and receives the next hidden state. The container
returns raw logits; the game service applies the authoritative legal mask and
the single resolved action-selection profile.

Training transitions retain the behavior policy's hidden input and output as
replay evidence. PPO begins the current policy at zero and recomputes all 24
hidden states for backpropagation; stored behavior states are not supplied to
updated weights.

## Forced final placements advance recurrent state

The eighth placement of every round has one legal action. On a policy turn, the
policy processes the legal mask and current view to advance its hidden state;
the deterministic engine applies the unique action. This transition is retained
for recurrence and excluded from the actor, entropy, critic, and illegal-action
losses. Human interaction for the same placement remains governed by the
frontend contract.

## The policy and critic are separate models

The recurrent policy selects moves without consulting a critic. A separate,
player-agnostic critic is used only during training to estimate expected round
return from a perspective-normalized observation. Critic estimates support the
round-local advantage calculation for policy updates; the critic is neither
exported nor invoked by the game application. It is an independently
parameterized feed-forward value network trained by mean squared error against
the normalized round-score difference.

## Policy observations are player-relative

The policy receives a canonical view in which its scoring lines are horizontal.
King views transpose the authoritative coffin and action positions; Queen views
retain them. Hands use fixed canonical card-index slots and never compact during
a round. This removes role and deal-order permutations from the learned policy
input.

## Policy architecture uses structured pair scoring

Policy version 1 uses shared learned card embeddings, separate hand-slot and
coffin-position encoders, explicit card-status inputs, a 128-unit GRU, and a
shared card-destination pair head. The head produces all 32 action logits under
one masked softmax. The critic remains a separate training model.

## Narration is separate from gameplay

The narrator receives immutable public events and returns no move. A direct
Bedrock invocation is sufficient because narration requires neither tools nor an
agent loop. Deterministic cadence selects optional move-comment events before
invocation. Required comments occur at each round opening, both dealer-ordered
orientation scoring touchpoints, and game completion.

## Opponent turns are separate idempotent requests

The browser submits a human move and receives the accepted state before it asks
the service to run the opponent turn. A conditional turn claim prevents
duplicate inference or application. Reload and timeout recovery resume the same
authorized turn.

## Narration uses event-backed asynchronous jobs

Narrator calls never participate in a game-state transaction. Move banter is
non-blocking. Required round-opening, orientation-scoring, and game-closing
comments use bounded presentation waits while game state remains independent.

## Events are required and snapshots remain authoritative

The current state serves gameplay; immutable events support replay, recovery,
narrator inputs, and scoring presentation. DynamoDB is the deployed adapter,
SQLite supplies matching local transaction semantics, and in-memory storage is
reserved for unit tests.

## A rules page replaces the in-game tutorial

The MVP starts with Queen and King game actions plus a Rules link that opens the
dedicated rules page in a new tab. It does not include an animated or staged
tutorial. Game creation and narration begin only after role selection.

## Card knowledge stays in the game surfaces

The frontend shows the human hand, public coffin, and completed public play. It
does not add a global card-status inventory. The policy's card-status tensors
are a private inference representation, not a user-interface feature.

## Kenney supplies the playing-card artwork

The MVP uses the CC0 Kenney Playing Cards Pack. The 64×64 large PNG set is the
selected source; the medium set is retained locally but does not replace it at
narrow sizes. Both source directories remain untracked and immutable. The
frontend asset set contains the 52 suited cards and the two Jokers, which are
the game's two Vampires.

## Scoring presentation follows the dealer

The completed coffin expands into a scripted scoring presentation. The dealer's
row or column tally runs first, followed by the non-dealer's. Individual values,
multipliers, sorted line scores, tie rejection, round scores, and cumulative
totals are driven by server-supplied steps. The MVP does not add skip or replay
controls.

## Agent platforms are omitted

The application does not use Strands, AgentCore, MCP, A2A, agent memory,
gateways, or knowledge bases. The opponent is a trained policy, and the narrator
is a bounded text-generation call. Adding an agent framework would introduce an
unused model-driven tool loop rather than satisfy a system requirement.

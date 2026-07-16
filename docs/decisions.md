# Design decisions

## Deterministic service owns game truth

Pure Python code owns dealing, legal moves, validation, transitions, scoring,
and factual effects. The policy may return an action and the narrator may
paraphrase supplied facts, but neither can mutate or recalculate game truth.

## A trained recurrent policy plays Dracula

Dracula's moves are selected by a recurrent neural policy trained through
self-play. The policy receives only its legitimate player view and a
server-issued legal-action mask. LLM move selection, hand-authored strategy
Oracles, and live deterministic substitutes are not part of the design.

Model training is a separate build with its own architecture, system
requirements, evaluation, artifacts, and cost controls. It shares the
authoritative game engine and versioned tensor contract with the web
application, not application persistence or request orchestration.

## SageMaker serves and governs the policy

Approved policy artifacts are versioned in SageMaker Model Registry and served
through SageMaker Serverless Inference. The application invokes the endpoint
privately and revalidates every returned action. Serverless inference matches
the expected intermittent traffic; a real-time endpoint is considered only if
measured cold-start or latency results fail release requirements.

The training framework and use of SageMaker Training remain evidence-driven.
Local Apple Silicon training is preferred when it meets throughput and
portability requirements; SageMaker Training supplies bounded cloud capacity and
Managed Spot execution when justified.

## Policy version and recurrence are pinned per game

Each game records one approved model version, tensor schema, inference
configuration, and recurrent hidden-state format. These values cannot change
during the game. The next hidden state is committed transactionally with its
accepted opponent move; a failed or retried inference cannot advance it alone.

## Forced final placements bypass policy inference

The eighth placement of every round has one legal action. On a policy turn, the
deterministic engine applies it directly and includes it in the terminal round
calculation. The recurrent policy runs for the seven non-forced decisions only;
the dealer's hidden state does not advance for its forced placement and resets
after the round. Human interaction for the same placement remains governed by
the frontend contract.

## The policy and critic are separate models

The recurrent policy selects moves without consulting a critic. A separate,
player-agnostic critic is used only during training to estimate expected round
return from perspective-normalized states. Critic estimates support advantage
calculation for policy updates; the critic is neither exported nor invoked by
the game application.

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

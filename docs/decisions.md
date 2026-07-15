# Design decisions

## Deterministic service owns game truth

Pure Python code owns dealing, legal moves, validation, transitions, scoring, and
factual effects. Models may propose a move or paraphrase supplied facts, but they
cannot mutate or recalculate game truth. This makes behavior testable, replayable,
and safe when inference fails.

## Decider and narrator are separate

The decider receives the agent's private view and returns no prose. The narrator
receives public event data and returns no move. Together they create the opponent
experience without leaking strategy or giving commentary gameplay authority.
Deterministic cadence selects optional move-comment events before invocation;
every narrator invocation returns commentary. Dracula's voice is always invoked
at each round opening, the row and column scoring touchpoints, and game
completion.

## Strands runs in private AgentCore Runtime

Strands provides bounded analysis and selection tools over invocation-local
state. Only the authoritative service invokes AgentCore, and it revalidates the
returned move. This preserves a clear boundary between model choice and game
authority.

## Agent turns are separate idempotent requests

The browser submits a human move and receives the accepted state before it asks
the service to run the agent turn. A conditional turn claim prevents duplicate
model invocation or duplicate application. Reload and timeout recovery resume
the same authorized turn.

## Narration uses event-backed asynchronous jobs

Narrator calls use immutable public event projections and never participate in a
game-state transaction. Move banter is non-blocking. Required round-opening,
row-scoring, column-scoring, and game-closing comments use bounded presentation
waits while game state remains independent.

## Events are required and snapshots remain authoritative

The current state serves gameplay; immutable events support replay, recovery,
narrator inputs, and scoring presentation. DynamoDB is the deployed adapter,
SQLite supplies matching local transaction semantics, and in-memory storage is
reserved for unit tests.

## A rules page replaces the in-game tutorial

The MVP starts with Queen and King game actions plus a Rules link that opens the
dedicated rules page in a new tab. It does not include an animated or staged
tutorial. Game creation and narrator activity begin only after role selection.

## Card knowledge stays in the game surfaces

The frontend shows the human hand, public coffin, and completed public play. It
does not add a global card-status inventory. The agent's private card-ledger tool
remains an analysis capability and is not a user-interface feature.

## Unneeded agent features are omitted

The MVP does not use MCP, A2A, AgentCore Memory, Gateway, or Knowledge Bases.
The game needs neither external tool discovery, cross-agent communication,
retrieval, nor cross-session memory. These exclusions keep the architecture
focused and may be reconsidered only if a later requirement needs them.

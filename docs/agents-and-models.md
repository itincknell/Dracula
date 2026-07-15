# Agents and models

## Opponent roles

The opponent is composed of two separate LLM roles:

- The **decider** uses Strands tools in AgentCore Runtime and returns one move ID.
- The **narrator** receives selected accepted public events and writes a brief
  visible comment.

The decider receives `AgentGameView`, never the human hand or deck order. It does
not write DynamoDB, apply moves, or produce visible prose. The narrator receives
no private state and does not recalculate game effects.

## Narrator behavior

The narrator speaks in character as Dracula. Its purpose is personality and
liveness; its commentary does not need to explain strategy. It receives only
server-issued public event projections and cannot delay or alter game state.

Required comments occur at the start of every round, after the row-scoring
animation, after the column-scoring and round-award animation, and at game
completion. The round-start input contains only the center card and first player.
Scoring generation runs concurrently with its animation and is awaited at the
end of that animation. These presentation waits are bounded so narrator failure
cannot stall the game indefinitely.

Move comments are optional, non-blocking banter. A deterministic cadence gate
allows at most two per round, favors human moves late in the round, suppresses
the final placement, and gives lower priority to comments on the agent's own
moves. Events that do not pass the gate cause no narrator invocation. Every
invocation expects commentary text. Taunts may dramatize visible events but
cannot assert knowledge of the decider's private plan.

The complete trigger matrix, ordering rules, and delayed-response behavior are
defined in [architecture](architecture.md#narrator-scheduling).

## Strands tools

- `inspect_game_state()`
- `list_legal_moves()`
- `evaluate_move(move_id)`
- `compare_moves(move_ids)`
- `select_move(move_id)`

Tools operate on an invocation-local snapshot and return concise structured data.
`select_move` records a proposal; the service still validates and applies it.

## Configuration

Named, allowlisted profiles independently select decision and narrator models,
prompt versions, inference settings, and strategy mode. Each game and evaluation
stores its complete resolved profile, seed, and code version. Public users cannot
submit arbitrary model IDs; authenticated evaluations may override them.

The Nova Lite profiles in the starter design are examples, not chosen defaults.

## Selected platform capabilities

| Capability | MVP choice | Reason |
| --- | --- | --- |
| Strands tools | Use | Bounded, inspectable decision capabilities |
| AgentCore Runtime | Use | Private deployment for the decision agent |
| Bedrock | Use | Decision and narrator model inference |
| AgentCore observability | Use | Tool, latency, token, and failure evidence |

> **TODO AGENT-001 — Specify tool contracts.** Define exact inputs, outputs,
> semantics, errors, limits, and determinism after the rules and state schemas are
> approved.
>
> **Complete when:** Contract tests run every tool on frozen state without AWS or
> persistent mutation.

> **TODO AGENT-002 — Choose and justify configuration.** Select MVP model profiles,
> prompts, inference settings, strategy/difficulty surface, and relevant
> AgentCore/Strands defaults or custom values.
>
> **Complete when:** Focused evaluation supports each chosen value and the
> resolved configuration is fully persisted and replayable.

> **TODO AGENT-003 — Define bounded failures.** Choose tool-call, retry, output,
> and wall-clock limits and invalid-output behavior.
>
> **Complete when:** Integration tests cover every failure path and prove no turn
> applies more than one move.

> **TODO NARRATOR-002 — Define the narrator persona and prompt.** Specify Dracula's
> voice, factuality, brevity, repetition limits, prohibited disclosure, and
> graceful failure text.
>
> **Complete when:** A versioned prompt passes representative grounding, privacy,
> cadence, and tone cases using public input only.

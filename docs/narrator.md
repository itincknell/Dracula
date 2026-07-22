# Narrator

## Responsibility

The narrator speaks in character as Dracula and supplies the opponent's visible
personality. It receives only server-issued public event projections and cannot
choose moves, inspect private state, recalculate effects, or mutate the game.
The application invokes the configured Bedrock model directly; no agent loop or
tool layer is involved.

## Cadence and presentation

Required comments occur at the start of every round, after each player's
orientation tally, and at game completion. The dealer's row or column tally runs
first. Each scoring input contains only that orientation's calculations and
sorted scores. Generation begins with the corresponding animation and is
awaited at its end through a bounded presentation wait.

Move comments are optional, non-blocking banter. A deterministic cadence gate
allows at most two per round, favors human moves late in the round, suppresses
the final placement, and gives lower priority to Dracula's own moves. Events
that do not pass the gate cause no model invocation. Every invocation expects
commentary text.

Taunts may dramatize visible events but cannot claim knowledge of search state,
hidden cards, model tensors, or intended strategy. The complete
trigger matrix and delayed-response behavior are defined in
[architecture](architecture.md#narrator-scheduling).

## Configuration

The narrator configuration resolves to an allowlisted Bedrock model, prompt
version, inference parameters, output limit, timeout, and retry limit. Games
persist the resolved values. Public callers cannot supply arbitrary model IDs or
prompts.

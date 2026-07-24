# Design decisions

## Deterministic engine owns game truth

Pure Python code owns dealing, legal moves, validation, transitions, scoring,
and final outcomes. Opponent controllers return actions only; narration returns
text only.

The canonical deck order is `AC` through `KC`, `AD` through `KD`, `AH` through
`KH`, `AS` through `KS`, then `V1` and `V2`. Each round deals pairs to the
non-dealer, dealer, non-dealer, and dealer before taking the center card. Hands
sort by canonical card index. Versioned SHA-256 streams provide shuffle, dealer,
search, training, and evaluation randomness.

## Opponent development is search-first

The next opponent is information-set Monte Carlo search through the end of the
current round. It samples hidden assignments consistent with the acting
player's information and uses exact engine round-score differential as its
terminal payoff.

Search must demonstrate constructive and defensive play and beat uniform random
legal play and the archived `policy-2-v20` PPO candidate on fixed role-balanced
fixtures before neural training continues.

## Version 1 search uses POMCP-style root sampling

The first implementation treats the other player as an explicit stochastic
environment policy. Root-player choices use a history tree and UCT; simulated
opponent choices receive their own sampled hand, public history, and legal
actions only. The opponent rollout policy is uniform legal in the first gate.

This is a testable best-response planner, not an equilibrium solver. SO-ISMCTS
does not model the opponent's information directly, multi-observer search adds
opponent-tree leakage and complexity, and ReBeL's public-belief subgame solving
is deferred until evidence requires it.

The validated 500-simulation implementation is frozen as a permanent baseline.
Recorded human play showed that its uniformly random simulated opponent can
still allow strong opposing constructions despite its fixture and control
results.

## Teacher v2 uses shallow greedy responses

Teacher v2 retains version 1 root sampling and outer UCT. At each non-forced
continuation decision, it compares every legal action from the acting player's
information state. Each candidate uses the same indexed hidden-card samples,
completes the round with uniform legal play, and receives the engine's exact
actor-relative round differential. The highest mean action wins with a
canonical tie-break.

The response policy has no tree, exploration term, recursion, or incomplete
board score. Response hidden assignments are sampled independently from the
actor's unseen-card pool and discarded after evaluation. Only the selected
legal action is applied to the unchanged outer world.

The earlier nested actor-local UCT prototype was information-safe but improved
the defensive fixture pass rate only from 78.3% to 83.3% while increasing
fixture p95 latency from 19.4 to 88.1 seconds in its 32-outer/32-response
validation profile. It is retained only as historical evidence. Shallow
response counts of one, two, and four completions per action are evaluated
before one configuration is selected.

Multi-observer ISMCTS remains rejected because opponent statistics can absorb
the root's fixed private hand. Re-determinizing one shared trajectory remains
rejected because it can create incompatible hidden-card histories.

Version 1 remains the default local controller and permanent comparison
control. The user approved the 32×4 Teacher v2 controller after direct browser
play. Its shallow response policy may supply neural teacher data after the
outer visit budget is shown to produce informative distillation targets.

## Search payoff is round-local score differential

The return is `(own_round_score - opponent_round_score) / 150`. All current hand
cards are consumed during the round, so placement cannot change later stock.
Round-local search captures the immediate constructive, blocking, multiplier,
and Vampire decisions needed for the first competence gate. Optimizing final
win probability from a game-score lead is deferred.

## Information state includes public move history

The acting player receives their remaining hand, the current coffin, actor/card/
destination/order for public moves, unseen-card membership, dealer state, and
legal actions. King views transpose the coffin and actions; Queen views retain
the authoritative orientation.

Opponent hand-slot indexes are private. The current 875-bit observation retains
the Markov-sufficient core for the round-local planners. Public move history
remains a typed search and replay field because it is required to reconstruct
engine-valid determinizations and audit the information boundary.

## One feed-forward policy/value model guides later search

The approved Teacher v2 search supplies visit distributions and exact
normalized round results to one 339,978-parameter feed-forward policy/value
network. Its policy and value heads share the structured 875-bit encoder.
Guided search uses
the learned policy as a PUCT prior while retaining full-round exact scoring.
There is no separate critic, recurrent state, PPO update, policy population, or
critic burn-in.

Recurrence is omitted because the explicit current-round information state is
sufficient for the chosen objective. Cross-round opponent adaptation is not
part of the first learned model.

## Absolute controls determine strength

Uniform random legal play, the archived PPO candidate, and the frozen version 1
search configuration are permanent controls. Approved Teacher v2 becomes the
additional search reference for neural candidates. Comparisons reuse fixed
decks, balance Queen and King, report paired uncertainty, and retain role and
dealer splits. Relative self-play rank and training loss cannot establish
competence.

## Training is local; deployment is deferred

Search and model training target the M3 MacBook Air with 8 GB unified memory.
Engine simulation and search run on CPU. PyTorch optimization selects CPU or MPS
from sustained measurements. Search budget, tree memory, model size, and
standalone-model strength determine the eventual serving design.

No cloud inference product, concurrency tier, or stateful model session is
selected before those measurements. AWS remains limited to components required
by the final serving profile.

## Opponent turns are idempotent requests

The browser receives an accepted human move before requesting the opponent
turn. A conditional claim binds one game version, turn, information-state
digest, action table, controller configuration, and decision seed. Retries
cannot apply a second move.

Forced final placements bypass search. The engine supplies the unique action.

## Narration is separate from gameplay

The narrator receives immutable public events and returns no move. A direct
Bedrock invocation is sufficient. Deterministic cadence selects optional banter;
required comments occur at round opening, both scoring orientations, and game
completion. Narration failure never changes game state or fabricates dialogue.

## Events support recovery; snapshots serve gameplay

The current state serves the game. Immutable events support replay, recovery,
narration, and scoring presentation. SQLite supplies local transaction
semantics and in-memory storage supplies unit-test semantics. Production
persistence is selected with deployment.

## A rules page replaces an in-game tutorial

The game starts with Queen and King choices and a Rules link that opens a
dedicated page. There is no staged tutorial.

## Card knowledge stays in game surfaces

The frontend shows the human hand, public coffin, and public play. It does not
show a global card-status inventory. Search information and hidden-card samples
are server-private.

## Kenney supplies the playing-card artwork

The MVP uses the CC0 Kenney Playing Cards Pack. The 64×64 large PNG set is the
source. The medium pack remains quarantined. Frontend assets contain the 52
suited cards and two Jokers used as Vampires.

## Scoring presentation follows the dealer

The completed coffin expands into a scripted scoring sequence. The dealer's
orientation runs first. Server-supplied line values, multipliers, sorted scores,
tie rejection, round scores, and cumulative totals drive the presentation.

## Agent platforms are omitted

The application does not use Strands, AgentCore, MCP, A2A, agent memory,
gateways, or knowledge bases. Search or a trained policy selects moves; the
narrator is a bounded text-generation call.

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

## Initial search uses POMCP-style root sampling

The first implementation treats the other player as an explicit stochastic
environment policy. Root-player choices use a history tree and UCT; simulated
opponent choices receive their own sampled hand, public history, and legal
actions only. The opponent rollout policy is uniform legal in the first gate.

This is a testable best-response planner, not an equilibrium solver. SO-ISMCTS
does not model the opponent's information directly, multi-observer search adds
opponent-tree leakage and complexity, and ReBeL's public-belief subgame solving
is deferred until evidence requires it.

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
the Markov-sufficient core for version 1 round-local decisions. Public move
history remains a typed search and replay field because it is required to
reconstruct engine-valid determinizations and audit the information boundary.

## One feed-forward policy/value model may follow search

After search passes its gates, one shared feed-forward network will learn policy
targets from search visit distributions and value targets from exact normalized
round results. The policy and value heads share an encoder. There is no separate
critic, recurrent state, PPO update, policy population, or critic burn-in.

Recurrence is omitted because the explicit current-round information state is
sufficient for the chosen objective. Cross-round opponent adaptation is not
part of the first learned model.

## Absolute controls determine strength

Uniform random legal play, the archived PPO candidate, and the first validated
search configuration are permanent controls. Comparisons reuse fixed decks,
balance Queen and King, report paired uncertainty, and retain role and dealer
splits. Relative self-play rank and training loss cannot establish competence.

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

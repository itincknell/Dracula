# Information-set search

This document defines the first competent-opponent milestone: deterministic,
information-safe search through the end of the current round. Neural guidance
is admitted only after this search passes the gates in
[model training](model-training.md).

## Search objective

For root player `p`, a completed simulation returns:

```text
z = (round_score[p] - round_score[other]) / 150
```

The engine's round scores already include line ranking, multipliers, Vampires,
and tie resolution. The maximum round score is 150, so `z` is in `[-1, 1]` and
the other player's return is `-z`. Division by 150 changes scale, not action
ordering.

This is the correct initial payoff because all eight hand cards are consumed in
the current round regardless of placement. A move cannot preserve a card for a
later round or alter the remaining stock. Maximizing expected round-score
differential is therefore aligned with the additive game score. Round-local
search does not account for risk preferences created by the current game total;
that limitation is acceptable for the tactical-competence gate and must be
measured before full-game search is considered.

## Information model

At a decision, a player knows:

- Their scoring role, dealer status, and remaining cards in stable hand slots.
- The center card and every accepted current-round move, including actor, card,
  destination, and turn order.
- Completed public rounds and scores.
- The number of cards remaining in each hand and the stock.
- Their legal card-destination actions.

They do not know the other player's remaining cards, the order of the stock,
the game seed, or opponent hand-slot indexes. A public move identifies the card
and destination but not its former private slot.

The search information state is player-relative. Queen uses the authoritative
coffin orientation; King uses its transpose. The acting player's scoring lines
are therefore horizontal in both views. The exact projection and 32-action map
are defined in the [engine–opponent contract](engine-model-contract.md).

The current 875-bit neural observation and separate legal mask contain the
Markov-sufficient decision core for the version 1 uniform belief and opponent
model: own hand, coffin, unseen-card membership, dealer and decision progress,
and legal actions. Move ownership and order do not affect round legality,
scoring, or that uniform belief once the current state is known.

The tensor is not the complete search contract. Public move history is also
required to reconstruct an engine-valid determinization, audit information
boundaries, and support any future behavioral opponent model. Search therefore
consumes the typed information state directly. A later neural model may reuse
the current tensor projection unless measured evidence justifies adding public
history.

## Root belief and determinization

Version 1 uses the uniform posterior over card assignments consistent with the
root information state. It does not infer hidden cards from the opponent's
earlier choices.

For each simulation:

1. Fix every public card and the root player's remaining hand.
2. Form the hidden pool from the canonical deck minus those cards.
3. Uniformly sample the known number of opponent remaining cards without
   replacement.
4. Combine the sampled cards with the opponent cards already played this round,
   sort the reconstructed four-card hand canonically, and derive its stable
   slots.
5. Assign the remaining hidden cards to a deterministically shuffled simulated
   stock. The current-round search never reads that stock, but the assignment
   keeps the simulated engine state complete and valid.
6. Rebuild private move-slot records from the sampled hands while retaining the
   public actor, card, destination, and turn sequence.

Sampling uses a seed derived from the search request, root state fingerprint,
simulation index, and search schema. It never reads the authoritative opponent
hand, stock order, shuffle seed, or private move slots. Repeating a request with
the same configuration produces the same samples and decision.

Uniform sampling is exact before accounting for information conveyed by prior
opponent choices. Behavioral inference is deferred until a validated opponent
model exists; pretending to perform it without such a model would create
unsupported confidence.

## Search algorithm

The first implementation uses POMCP-style root sampling and a history tree from
one acting player's perspective:

1. Sample one root determinization.
2. At a root-player decision node, select a legal action with UCT. Node keys are
   the root player's action-observation history, not the sampled world state.
3. At an opponent turn, construct the opponent's information state from the
   sampled hand and public history. The version 1 opponent model samples
   uniformly from that view's legal actions using its own derived stream.
4. Apply every selected action through the deterministic engine.
5. Expand at most one tree node, then continue the rollout to round completion.
6. Back up the exact normalized terminal return to every visited root-player
   action.

There is no explicit chance node in version 1. Root determinization and
opponent-action sampling are the chance draws; their streams are derived from
the request, simulation, and rollout indexes. Engine legality filters every
action before transition, and the terminal engine result is backed up with
opposite sign for the two player perspectives.

The opponent model is part of the stochastic environment, as in a POMCP
planner. It receives its sampled hand and public state only. The root player's
private hand is neither an input nor a tree key for an opponent decision. Tests
must show that changing only root-private cards cannot change an opponent-model
choice under the same opponent view and seed.

Every root action is visited once before UCT selection. UCT uses mean root
return and an exploration constant supplied by configuration. Actions that are
illegal in a sampled history are unavailable for that visit. The selected real
move is the legal root action with the highest visit count; mean value and
canonical action index break ties. A one-legal-action turn bypasses search and
uses the engine action directly.

The uniform opponent model is intentionally limited. The search computes a
best response to that model, not a Nash equilibrium and not a worst-case
strategy. Search is rerun from the actual acting player's information state on
every real decision, so both sides receive the same planning capability in
search-only self-play. Failure to show defensive play is evidence against this
opponent model and stops the neural phase.

## Algorithm selection

The selected design is smaller than the alternatives while keeping the hidden
information boundary testable:

- SO-ISMCTS represents only the root observer and does not give opponent choices
  their own information-set nodes.
- Multi-observer ISMCTS adds a tree per observer, but classic determinization can
  still leak root knowledge into opponent statistics. Re-determinization avoids
  that leak at the cost of incompatible simulated worlds and additional search
  machinery.
- POMCP-style root sampling cleanly supports a black-box simulator and a belief
  over hidden states when the other player is represented by an explicit,
  information-safe environment policy.
- ReBeL addresses equilibrium search in imperfect-information zero-sum games by
  operating over public belief states. Its belief representation and subgame
  solving are disproportionate before basic tactical competence is established.

The version 1 choice makes no equilibrium claim. A more complex method requires
evidence that the fixed opponent model, rather than search depth or rollout
quality, blocks the search gates.

## Diagnostics

Each decision records private diagnostic evidence:

```text
search schema and configuration digest
root information-state digest
simulation count and elapsed time
per-action visits, mean return, and return variance
chosen action and deterministic tie-break inputs
sample and rollout seed digests
principal sampled continuation for each root action
peak tree node count and process memory
```

Diagnostics may contain root-private card IDs and remain outside public events,
API responses, normal application logs, and narrator inputs.

## Feasibility on the target laptop

A direct benchmark of the current validated Python engine on the project M3
MacBook Air completed about 268 random eight-placement rounds per second and
2,147 placements per second before tree bookkeeping. Observed mean legal-action
counts by placement were approximately:

```text
16, 20, 14.4, 13.1, 7.4, 5.8, 2, 1
```

The tree depth is at most eight placements and decreases as the round fills.
Budgets of 100, 500, and 2,000 simulations per decision are the initial
benchmark points. The expected range is subsecond to several seconds at the low
and middle budgets and potentially tens of seconds at 2,000 once Python tree
overhead is included. Measured latency, not this estimate, selects the search
gate budget.

One simulation expands at most one node. A 2,000-simulation search therefore
needs at most 2,000 tree nodes. Compact fixed-size action statistics should stay
well below 64 MB per active search; the benchmark must record actual resident
memory and reject unbounded retention between decisions.

## Search-only gates

Search-guided neural training remains closed until all gates pass:

1. Tactical fixtures and recorded continuations show constructive multiplier
   completion, blocking of visible opponent constructions, preservation of
   flexible placements, exploitation of opponent cards, and avoidance of
   Vampire-destroyed lines.
2. Information-boundary tests prove that determinization and simulated decisions
   use only the relevant player's information.
3. On fixed, role-balanced deck fixtures, the selected search budget beats both
   uniform random legal play and the archived `policy-2-v20` PPO candidate. The
   paired 95% bootstrap lower bound for the victory-percentage difference must
   be above zero against each control, with Queen, King, dealer, and non-dealer
   splits reported separately.
4. Search-guided neural work cannot begin until gates 1 through 3 pass.
5. Deployment design remains deferred until strength, latency, and memory are
   measured at 100, 500, and 2,000 simulations per move on the target laptop.

Each control comparison uses at least 12 independent deck seeds and both role
assignments: 24 games per control. Resampling uses the deck seed as the paired
block so the two role assignments are not treated as independent decks. The
sample is fixed before play and has no interim stopping. Search configuration,
controls, fixtures, and report format do not change in response to results.

## Validation result

The search-only gate passed on July 22, 2026 with 500 simulations per decision:

- All 33 strategic fixture decisions passed across three search seeds, and all
  21 decisions with exhaustive continuation evidence matched the exact action.
- Search defeated random legal play 24–0, winning 92.4% of rounds with a mean
  round-score differential of `+31.32`.
- Search defeated `policy-2-v20` 24–0, winning 86.8% of rounds with a mean
  round-score differential of `+30.35`.
- Search defeated the 100-simulation planner 18–6. The 100-simulation planner
  also failed four strategic decisions; 2,000 simulations added no fixture
  passes over 500.
- Under four concurrent validation workers, the 500-simulation fixture latency
  was 15.7 seconds at p50 and 26.5 seconds at p95. Peak process memory was about
  203 MiB, including the Python and PyTorch runtime.

The complete local evidence is generated at
`runs/search-validation-001/search-validation.md` with its JSON counterpart.
This result opens search-guided neural work. It does not select a deployment
latency profile.

## Research basis

- [AlphaZero](https://discovery.ucl.ac.uk/id/eprint/10069050/1/alphazero_preprint.pdf)
  demonstrates policy/value learning from search visit distributions and exact
  terminal outcomes in perfect-information games.
- [Information Set Monte Carlo Tree Search](https://eprints.whiterose.ac.uk/75048/1/CowlingPowleyWhitehouse2012.pdf)
  defines information-set trees, determinization sampling, and multi-observer
  variants for adversarial hidden-information games.
- [POMCP](https://proceedings.neurips.cc/paper_files/paper/2010/file/edfbe1afcf9246bb0d40eb4d8027d90f-Paper.pdf)
  establishes root belief sampling with a black-box generative model for online
  planning under partial observation.
- [ReBeL](https://papers.nips.cc/paper_files/paper/2020/file/c61f571dbd2fb949d3fe5ae1608dd48b-Paper.pdf)
  explains why ordinary perfect-information RL plus search is not
  equilibrium-sound in imperfect-information games and introduces public belief
  states for a principled alternative.

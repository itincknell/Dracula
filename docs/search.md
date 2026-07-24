# Information-set search

This document defines deterministic, information-safe search through the end of
the current round. The validated 500-simulation planner is frozen as the
version 1 baseline. Teacher v2 adds strategic simulated responses without
changing the information boundary, payoff, engine, or search-result contract.
The approved Teacher v2 profile uses 32 outer simulations and four response
completions per legal action.

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

Version 1, the Teacher v2 outer search, and each Teacher v2 response evaluation
use the uniform posterior over card assignments consistent with the relevant
player's information state. They do not infer hidden cards from earlier
choices.

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

## Version 1 baseline

Version 1 uses POMCP-style root sampling and a history tree from one acting
player's perspective:

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

The uniform opponent model is intentionally limited. The search computes a best
response to that model, not a Nash equilibrium and not a worst-case strategy.
Search is rerun from the actual acting player's information state on every real
decision, so both sides receive the same planning capability in search-only
self-play. The implementation, 500-simulation configuration, fixtures, and
validation evidence are retained unchanged as the version 1 baseline.

## Teacher v2 algorithm

Teacher v2 retains the version 1 outer root-sampled UCT tree and replaces its
uniform continuation policy with a **shallow greedy Monte Carlo response
policy**. Every non-forced opponent action, and every root-player rollout
action after the outer tree expands, is selected by comparing all legal actions
from the acting player's information state. The comparison has no tree,
exploration term, or recursive planning.

The approved collection configuration is:

```text
outer root simulations:       32
outer UCT exploration:        sqrt(2)
response completions/action:  4
terminal evaluation:          exact completed-round differential
real move selection:          maximum visits, then mean value, then action index
response selection:           maximum mean value, then action index
response continuation:        uniform legal play to round completion
```

### Authoritative early-turn symmetry

The following position reductions are definitive. `C` marks an occupied coffin
position. Card identities do not affect which destination positions are
paired. Search considers only the listed representative positions in these
states.

On the non-dealer's first turn:

```text
1 2 3
4 C 6
7 8 9
```

Consider positions `2` and `4`:

```text
representative 2 -> pair (2, 8)
representative 4 -> pair (4, 6)
```

On the dealer's first turn:

```text
1 C 3
4 C 6
7 8 9
```

Consider positions `1`, `4`, and `8`:

```text
representative 1 -> pair (1, 3)
representative 4 -> pair (4, 6)
representative 8 -> unpaired
```

```text
1 2 3
4 C 6
7 C 9
```

Consider positions `2`, `4`, and `9`:

```text
representative 2 -> unpaired
representative 4 -> pair (4, 6)
representative 9 -> pair (7, 9)
```

```text
1 2 3
C C 6
7 8 9
```

Consider positions `1`, `2`, and `6`:

```text
representative 1 -> pair (1, 7)
representative 2 -> pair (2, 8)
representative 6 -> unpaired
```

```text
1 2 3
4 C C
7 8 9
```

Consider positions `2`, `3`, and `4`:

```text
representative 2 -> pair (2, 8)
representative 3 -> pair (3, 9)
representative 4 -> unpaired
```

The dealer's second-turn reduction applies only when the opponent makes a
line:

```text
1 2 3
C C C
7 8 9
```

Consider positions `1`, `2`, and `3`:

```text
representative 1 -> pair (1, 7)
representative 2 -> pair (2, 8)
representative 3 -> pair (3, 9)
```

```text
1 C 3
4 C 6
7 C 9
```

Consider positions `1`, `4`, and `7`:

```text
representative 1 -> pair (1, 3)
representative 4 -> pair (4, 6)
representative 7 -> pair (7, 9)
```

After search selects a paired representative, a derived fair coin selects one
of its two concrete positions. An unpaired representative maps directly to its
listed position. This reduction does not authorize additional inferred
symmetries outside the listed states.

Legal actions are partitioned first by hand slot and then by the listed
destination groups. Different cards remain different strategic actions. A
group’s representative action owns its UCT visits and accumulated terminal
returns; its concrete destinations do not create additional tree actions.
Mean values are reported for every member from the one pooled group value.
At the teacher boundary, the unchanged pooled visit count is placed on the
group's coin-selected concrete action in the existing 32-action vector.

One outer simulation proceeds as follows:

1. Sample one complete engine-valid world from the real decision maker's
   information state.
2. At a root-player decision before expansion, select a representative action
   group with version 1 UCT. Expand at most one previously unvisited group.
   Resolve a paired group to one legal concrete destination with the derived
   fair coin.
3. At every non-forced opponent decision, and at every root-player decision
   after expansion, project a fresh
   `SearchInformationState` for the acting player from the outer state. Run the
   shallow response evaluation below, select its maximum-mean group, and
   resolve its concrete destination.
4. Apply only that selected action to the unchanged outer sampled world.
5. Continue until the engine completes the round, then back up the exact
   normalized return through the outer root-player path.

For one response evaluation with `K` completions per strategic action:

1. Partition the legal actions by hand slot and the exact destination table.
   Enumerate the representative action indexes in canonical order.
2. For each completion index `k` in `[0, K)`, sample one determinization from
   the actor's information state. This same sampled world is the starting point
   for every candidate action at index `k`.
3. For each representative action, apply its representative destination to an
   independent immutable copy of that shared world. Complete the round with
   uniformly sampled legal actions. Each later actor choice is made from that
   actor's projected information state.
4. Score the completed round as the exact normalized differential from the
   response actor's perspective.
5. Pool the `K` terminal values at the representative group. Select the group
   with the highest mean, resolving exact equality to the lowest canonical
   representative action index. Resolve a paired group to one concrete legal
   destination only after this strategic selection.

Sharing a determinization across candidate actions controls hidden-card
sampling variance without exposing its assignment to the response policy.
Uniform continuation after the candidate action makes the policy one-ply and
scoring-aware. It does not build a response tree or apply a hand-authored
estimate to an incomplete coffin.

### Actor information and hidden cards

The response evaluator receives only the acting player's immutable
`SearchInformationState`. This fixes that player's remaining hand, the public
coffin and move history, played cards, roles, dealer, turn, scores, legal
actions, and unseen-card pool.

Response determinizations independently assign the unseen pool between the
other player's remaining hand and stock. They do not read or preserve the
outer world's hidden allocation. Each response world conserves all 54 cards,
reproduces public history, and is accepted by the deterministic engine. Every
candidate is legal in the outer world because the actor's hand and public
coffin are identical in the response information state.

This boundary prevents three failure modes:

- The response policy cannot condition on the root player's actual private
  hand; that hand appears only as part of the actor's undifferentiated unseen
  pool.
- A single action is selected after aggregating every candidate across the same
  indexed response samples. No sample-specific action is exposed to the outer
  simulation.
- Response hidden assignments never replace outer assignments, so cards cannot
  move between hands or stock during one outer continuation.

All hidden worlds with the same actor information state share one vector of
mean action values and one selected action. The outer planner cannot ask which
action the actor would choose in its particular hidden world. Teacher v2 is a
shallow opponent model, not an equilibrium claim.

Equal actor information states and response configurations produce the same
samples, action values, and action even when their enclosing outer worlds
differ only in facts hidden from that actor.

### Nodes, values, and forced moves

Outer nodes are keyed by the root player's information-state fingerprint.
Response evaluations have no nodes. Their cache key is the acting player's
information-state fingerprint plus the response-configuration digest. A
sampled-world identity is never a tree key or cache key.

Outer UCT stores returns from the real decision maker's perspective and
continues to use:

```text
Q(s, a) + sqrt(2) * sqrt(log(N(s)) / N(s, a))
```

The engine terminal value is always:

```text
(round_score[observer] - round_score[other]) / 150
```

Changing observer negates the value. No intermediate score, defensive bonus,
or hand-authored move value is introduced.

A one-legal-action placement is applied directly. It consumes neither an outer
decision visit nor a response evaluation. It remains present in the principal
continuation and engine replay.

### Determinism and budget accounting

Teacher v2 introduces these namespaces:

| Namespace | Components after namespace |
| --- | --- |
| `dracula-strategic-search-request-v2` | Fixture ID, root information-state digest, player, round, turn, v2 configuration digest |
| `dracula-strategic-search-determinization-v2` | Request digest, outer simulation index |
| `dracula-strategic-search-expansion-v2` | Request digest, outer simulation index, root node digest |
| `dracula-strategic-destination-choice-v1` | Request digest, choice scope, actor information-state digest, representative action, choice index |
| `dracula-greedy-response-request-v1` | Actor information-state digest, response-configuration digest |
| `dracula-greedy-response-determinization-v1` | Response-request digest, completion index |
| `dracula-greedy-response-rollout-v1` | Response-request digest, completion index, candidate action, rollout ply, acting information-state digest |

The Teacher v2 search schema is
`dracula-strategic-information-search-v2`. The response schema is
`dracula-shallow-greedy-response-v1`; its continuation profile is
`uniform-legal-to-round-end-v1` and its selection profile is
`max-mean-action-index-v1`.
Destination grouping uses `dracula-early-destination-symmetry-v1`; concrete
selection uses `derived-fair-coin-after-group-selection-v1`.

The v2 configuration digest is SHA-256 over canonical JSON containing the v2
search schema, outer simulation budget, outer exploration constant, response
schema, completions per action, uniform-continuation profile, exact terminal
value, both selection profiles, destination-symmetry schema, and concrete
selection profile.

A response request seed contains no outer request digest, sampled-world
identity, root hand, stock order, or engine seed. The determinization seed omits
the candidate action so candidates at the same completion index share one
hidden assignment. Candidate-specific rollout seeds are derived only after
that shared sample is created.

An exact response result may be cached for one outer request by actor
information-state digest and response-configuration digest. The cache stores
legal actions, mean terminal values, selected action, and counts; it never
stores a determinization or engine state.

### Response-distillation observer

The bounded response-distillation experiment instruments only a cache miss at
the completed shallow-response boundary. The optional observer receives one
immutable `dracula-response-ranking-example-v1` record per unique response
cache identity. Enabling the observer does not alter configuration digests,
seeds, visits, values, selected actions, or diagnostics.

The record contains:

```text
model-visible observation: bool[875]
legal mask:               bool[4, 8]
strategic groups and their concrete action indexes
four-completion mean terminal differential for every group
selected representative and coin-selected concrete action
placement, actor role, actor-is-dealer
search and response configuration digests
information-state fingerprint plus response digest as the cache identity
```

It contains no determinization, authoritative opponent hand, stock order,
engine seed, outer simulation state, search tree, policy hidden state, logits,
or model data. Cache hits produce no duplicate record.

`simulation_count` and root action visits continue to mean outer simulations
and sum to the configured outer budget. Private diagnostics additionally
report:

```text
response requests
unique response evaluations
response-cache hits
candidate actions evaluated
response terminal evaluations = completions/action * candidate actions evaluated
total terminal evaluations = outer simulations + response terminal evaluations
```

Forced placements contribute zero to every count. Interruption discards the
uncommitted outer result and all request-local response cache entries.

### Selection rationale

Multi-observer ISMCTS was rejected because opponent statistics collected under
the root's fixed private hand can encode information the opponent does not
have. Re-determinizing one shared trajectory was rejected because changing a
hidden allocation between turns can produce incompatible card histories.
The nested actor-local UCT prototype was rejected after it improved the
defensive fixture pass rate only from 78.3% to 83.3% while increasing fixture
p95 latency from 19.4 seconds to 88.1 seconds in its 32-outer/32-response
validation profile. Shallow greedy responses retain information isolation and
exact scoring while removing the inner tree and its exploration overhead.

## Diagnostics

Each decision records private diagnostic evidence:

```text
search schema and configuration digest
root information-state digest
simulation count and elapsed time
group membership, representative action, visits, and pooled mean return
selected representative and selected concrete destination
chosen group and deterministic tie-break inputs
sample and rollout seed digests
principal sampled continuation for each root action
peak tree node count and process memory
response requests, cache hits, candidate counts, and terminal evaluations
mean candidate values for failed fixture decisions
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

Teacher v2 adds shallow response completions, so its work is not described by
the outer budget alone. The implementation reports response decisions,
candidate actions, total terminal evaluations, and cache behavior. Measured
latency and resident memory determine whether the candidate can reach human
testing; neither the outer budget nor completions per action is a proxy for
cost.

## Version 1 gates

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
This result justified building the neural and teacher infrastructure. Recorded
human play later exposed the weak uniform-opponent assumption, so full teacher
collection was gated on Teacher v2 approval. The result remains valid baseline
evidence and does not select a deployment latency profile.

## Teacher v2 gates

The automated gate evaluated one, two, and four completions per action. No
configuration passed every automated requirement; that result remains in the
[shallow Teacher v2 validation report](../reports/teacher-v2-shallow-validation.md)
as historical evidence.

The user subsequently compared version 1 and the 32×4 profile through completed
browser games and approved 32×4 as the neural teacher on July 23, 2026. That
acceptance supersedes the failed automated human-test gate for teacher
selection only. It does not rewrite the automated result or claim that every
statistical threshold below passed. Version 1 remains a permanent control.

### Strategic behavior

The candidate must retain every constructive and tactical pass in the existing
33-case fixture suite across its three fixed search seeds. A separate versioned
defensive suite covers:

- Avoiding a placement that enables an opponent's visible same-suit or
  same-color multiplier.
- Blocking a visible multiplier when its expected denial exceeds the available
  constructive continuation.
- Choosing a defensive Vampire placement without wasting it on a line already
  worth zero.
- Resolving conflicts between immediate construction and opponent denial.
- Producing the same response for identical actor information embedded in
  different hidden outer worlds.

Every accepted action set is fixed before validation and justified by exhaustive
continuation analysis or the existing diagnostic control. Teacher v2 must pass
every defensive fixture across five fixed request seeds and must strictly
exceed version 1's defensive-fixture pass rate.

### Privacy and correctness

Tests must prove that a response evaluation cannot receive the enclosing root's
hand, stock order, engine seed, private move slots, sampled-world digest, or
model data. Hidden substitutions that preserve the actor's information must
reproduce response samples, action values, and action exactly. Every response
and outer trajectory must conserve 54 unique cards, reproduce public history,
use engine legality, and terminate at the engine's exact round result.

### Absolute performance

On fixed decks and both role assignments, Teacher v2 must retain a paired 95%
bootstrap lower bound above zero in victory-percentage difference against
random legal play and `policy-2-v20`, using at least 12 independent decks per
control. Against frozen version 1, it uses 30 independent decks and both role
assignments. The paired lower bound must be at least `-0.05`; defensive
improvement, rather than an absolute head-to-head win, is the purpose of the
response policy. Queen, King, dealer, and non-dealer results are reported
separately. Deck fixtures and sample sizes are sealed before play.

### Local feasibility and human approval

Single-worker benchmarks cover early, middle, and late decisions for one, two,
and four completions per action. They report outer simulations, response
requests, candidate actions, cache hits, terminal evaluations, p50, p95,
maximum latency, evaluations per second, and peak resident memory. The selected
candidate may enter human testing only when p95 latency is no more than 1.5
times paired version 1 latency, no decision exceeds 30 seconds, peak resident
memory stays below 1 GiB, and no request retains a response cache or sampled
world.

The manual comparison found the 32×4 profile notably stronger and sufficiently
challenging, including improved defensive play. This authorizes v2 teacher
collection while preserving the automated failure for later evaluation.

## Teacher v2 implementation sequence

1. Freeze version 1 source behavior with golden configuration, visit, action,
   fixture, and report-digest tests.
2. Replace the historical nested-response prototype with the versioned shallow
   response configuration, seeds, evaluator, and private diagnostics.
3. Implement shared response determinizations, per-action uniform completions,
   mean-value selection, and deterministic tie-breaking.
4. Connect the response evaluator to the outer planner with request-scoped
   caching, exact counters, forced-action bypass, and interruption cleanup.
5. Add actor-view invariance, shared-sample, hidden-world isolation,
   conservation, legality, sign, determinism, retry, and result-compatibility
   tests.
6. Run the constructive, defensive, privacy, absolute-performance, latency,
   and memory gates for one, two, and four completions per action; retain the
   failed result.
7. Expose 32×4 through the local v2 adapter and record the user's completed-game
   comparison and explicit approval.
8. Bind the 32×4 configuration, search and response schemas, report digest, and
   approval identity into teacher artifacts before collection.

## Bounded response-distillation experiment

This experiment replaces only Teacher v2's expensive inner response
enumeration. The 32-simulation outer search, UCT selection, information-safe
sampling, fair-coin destination choice, and exact round-terminal scoring remain
unchanged. Root visit distributions are not training targets.

The existing feed-forward model supplies its 32 raw policy logits. For each
strategic action group, inference uses the arithmetic mean of its concrete
member logits; a paired destination therefore receives no advantage from
having two members. The highest group score wins, with the canonical
representative action index resolving exact ties. The existing derived fair
coin selects a concrete member after the group is chosen.

Training compares every pair of groups. For group estimates `q_i`, `q_j` and
model group logits `l_i`, `l_j`, the contribution is:

```text
abs(q_i - q_j) * softplus(-sign(q_i - q_j) * (l_i - l_j))
```

The loss is the sum of contributions divided by their total weight. Exact
value ties contribute zero. There is no softmax temperature, root-visit target,
or value-head input. The value head remains outside this experiment.

The experiment passes only through a direct measured comparison: use the model
ranker for simulated response decisions, retain exact terminal scoring, and
show lower decision cost without a material loss of playing strength against
the approved 32×4 controller.

### Experimental hybrid controllers

Pure symmetry-aware 32×4 Teacher v2 remains unchanged and is the control.
Explicit experimental modes replace only its inner response selection:

- `student-direct` ranks every legal strategic group with the response ranker
  and selects the highest score. It performs no shallow terminal completions.
- `student-top-2` retains the two highest-ranked groups and applies the existing
  four-completion shallow evaluation only to those groups. Each completion
  samples one actor-valid determinization shared by both candidates.
- `student-top-3` was added after the paired fixture comparison showed that
  top-2 omitted two defensive actions preserved by pure Teacher v2. It applies
  the same four shared completions to the three highest-ranked groups.

Both modes retain the 32-simulation outer UCT search, exact engine transitions,
exact completed-round scoring, destination symmetry, and the derived fair coin.
The value head is unused, and the network supplies neither a root prior nor a
terminal cutoff.

Each model call receives only the acting player's
`SearchInformationState` projection and its legal strategic groups. Hybrid
response cache identity contains the actor information-state fingerprint,
response-ranker artifact SHA-256, shortlist size, and shallow-response
configuration digest. Forced placements bypass model inference and response
evaluation.

Local gameplay selects the modes explicitly:

```text
DRACULA_OPPONENT_MODE=search-v2-student-direct
DRACULA_OPPONENT_MODE=search-v2-student-top-2
DRACULA_OPPONENT_MODE=search-v2-student-top-3
DRACULA_RESPONSE_RANKER_ARTIFACT=<response-ranker artifact path>
```

These modes are experimental controllers, not accepted opponents. Their
latency savings do not establish playing strength.

## Neural-guided search

The frozen 500-simulation version 1 planner remains a permanent control. The
manually approved 32-outer, four-completion Teacher v2 profile supplies the
response-ranking experiment above. Any later root-policy dataset requires a
separately validated target. Validation and gameplay select the maximum-visit
action. The v2 search schema, response schema, configuration digests,
validation-report digest, and approval identity identify Teacher v2 evidence.

After the policy/value model is warm-started from that teacher, guided search
uses PUCT at root-player information nodes. When a node is created, the model
evaluates its player-relative 875-bit observation. External legal masking
produces `P_model(a)`. The stored prior is:

```text
P(a) = 0.95 * P_model(a) + 0.05 / legal_action_count
```

The uniform floor prevents a legal action from disappearing because of an
untrained logit. There is no Dirichlet root noise. Every legal action receives
one initial visit in canonical order. Later visits maximize:

```text
Q_root(s, a) + 1.5 * P(a) * sqrt(N(s)) / (1 + N(s, a))
```

`Q_root` is the mean terminal return from the current real decision maker's
perspective. Exact score equality uses the existing derived tree-selection seed
to break ties. Node keys remain root-player information-state digests rather
than sampled-world fingerprints.

The simulated opponent is still an explicit environment policy rather than a
second adversarial tree. It receives its own information state from the sampled
world, applies the shared model and its own legal mask, and samples from the
temperature-one policy with a derived opponent-choice seed. It cannot observe
the root player's private hand. Non-tree rollout decisions by the root player
use the same temperature-one policy from the root player's sampled view. Forced
placements bypass both model inference and the decision budget.

The initial guided profile runs **100 simulations per learned move** and reaches
the end of every round. The engine's exact normalized round differential is the
only backed-up value. The model value head is trained but does not terminate a
simulation in this profile.

During self-play, the accepted real move is sampled from normalized visit counts
at placements one through four of a round. Placements five through seven use
maximum visits. Evaluation and gameplay always use maximum visits. The eighth
placement is forced. Every sample uses the named deterministic action stream;
temperature does not make a run irreproducible.

### Value-cutoff gate

Network leaf evaluation is an experimental configuration, disabled by default.
It may be benchmarked only after held-out value MSE beats the zero predictor.
At a cutoff, the actor-relative prediction is negated when the actor is not the
root player before backup.

The cutoff profile may become active only when it:

- Passes every strategic fixture at the selected budget.
- Retains a paired 95% lower confidence bound of at least `-0.05` in victory
  percentage against full-round guided search over 60 independent decks with
  both role assignments.
- Continues to beat random legal play and `policy-2-v20` under the permanent
  control rule.
- Reduces p95 decision latency by at least 25%.

Failing this gate leaves full-round exact scoring active. The value head remains
useful as a training diagnostic and does not receive deployment authority from
loss improvement alone.

## Research basis

- [AlphaZero](https://discovery.ucl.ac.uk/id/eprint/10069050/1/alphazero_preprint.pdf)
  demonstrates policy/value learning from search visit distributions and exact
  terminal outcomes in perfect-information games.
- [Information Set Monte Carlo Tree Search](https://eprints.whiterose.ac.uk/75048/1/CowlingPowleyWhitehouse2012.pdf)
  defines information-set trees, determinization sampling, and multi-observer
  variants for adversarial hidden-information games.
- [Re-determinizing Information Set Monte Carlo Tree Search](https://arxiv.org/abs/1902.06075)
  documents opponent-model information leakage in ISMCTS and the incompatible
  hidden worlds that re-determinization can introduce.
- [POMCP](https://proceedings.neurips.cc/paper_files/paper/2010/file/edfbe1afcf9246bb0d40eb4d8027d90f-Paper.pdf)
  establishes root belief sampling with a black-box generative model for online
  planning under partial observation.
- [ReBeL](https://papers.nips.cc/paper_files/paper/2020/file/c61f571dbd2fb949d3fe5ae1608dd48b-Paper.pdf)
  explains why ordinary perfect-information RL plus search is not
  equilibrium-sound in imperfect-information games and introduces public belief
  states for a principled alternative.

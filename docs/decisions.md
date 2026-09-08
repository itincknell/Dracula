# Design decisions

## Deterministic engine owns game truth

Pure Python owns dealing, legality, transitions, scoring, and final outcomes.
Opponent controllers return actions only; Bedrock returns presentation text
only.

## Standalone pi1 is the release opponent

The user selected the 754,601-parameter feed-forward `pi1` classifier after
local human testing and fixed evaluation. It receives the compact 659-bit
player-relative observation, selects one externally masked representative
action, and resolves paired destinations with the established fair coin.

`pi1` runs once per non-forced Dracula move inside the FastAPI Lambda. There is
no deployed search, hybrid controller, value head, recurrence, SageMaker
endpoint, or silent fallback. Retired controller implementations are available
from Git history rather than the installed application.

## Production gameplay is stateless

The browser exchanges an initial game seed and ordered accepted command history
with FastAPI. Lambda reconstructs the exact engine state by deterministic replay
and may cache reconstructed states in local execution-environment memory.
Correctness never depends on cache reuse or Lambda affinity.

The envelope has no redundant lifecycle or version fields, signature, or
encryption. Users may inspect or alter their single-player game data. Strict
bounded parsing and engine replay reject malformed or impossible histories.
There is no production database.

## FastAPI serves the complete game

One Lambda image contains the React distribution, FastAPI, engine, and `pi1`.
Cloudflare forwards `/Dracula/` and `/Dracula/api` to API Gateway's generated
HTTPS endpoint. Other personal-site paths remain on GitHub Pages. No separate
Dracula Pages publication, API custom domain, or ACM certificate is required.

## Bedrock narration is sparse

Direct Bedrock calls occur at game opening, after rounds 1–5, and after the
final game result. A round-transition call begins after the eighth placement
and is revealed after scoring. Round 6 emits only the final-result cue.
There is no move banter or per-score narration.

The browser controls presentation timing. Bedrock receives only a public cue
and cannot block or mutate gameplay.

## Training remains separate from runtime

Maintained information-set UCT and policy-training code support future model
work, but neither runs inside production gameplay. The deployed service loads
only the selected standalone artifact.

## The frontend shows game-relevant knowledge

The interface exposes the human hand, coffin, public play, server-calculated
scores, and the seen-card cheat sheet. The seed in browser recovery data can be
inspected to reconstruct hidden cards; that accepted deployment tradeoff is not
surfaced as a product feature.

## Card assets are explicit

The 52 suited cards come from the CC0 Kenney large card pack. Project-owned
`V1.jpg` and `V2.jpg` artwork supplies the Vampire cards. The source pack
remains immutable.

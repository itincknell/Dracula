# Design tracker

This is the sole index for tracker IDs.

## Current item

Implement and deploy the locked stateless Lambda release with standalone
`pi1` and sparse Bedrock narration.

## Final sprint

```text
locked product and deployment contracts
    -> stateless seed/history replay API and Lambda cache
    -> browser-envelope recovery and three narration cues
    -> Lambda container with embedded pi1
    -> GitHub Pages /Dracula production build
    -> API Gateway + ACM + Cloudflare api CNAME
    -> staging validation
    -> production release
```

| ID | Status | Work | Complete when |
| --- | --- | --- | --- |
| `SERVE-002` | Complete | Select the production controller. | User selection pins standalone `pi1`, artifact digest, one-inference action contract, and no fallback |
| `ARCH-003` | Complete | Lock the final hosting and state topology. | GitHub Pages, Cloudflare DNS, API Gateway, stateless FastAPI/Lambda, local replay cache, embedded `pi1`, and no database are authoritative |
| `APP-003` | Complete | Implement stateless seed-and-history gameplay. | Cache hit and miss replay identically, all six rounds complete, reload works, and malformed histories fail safely |
| `NARRATOR-002` | Complete | Implement direct Bedrock narration at the locked cadence. | Opening, rounds 1–5 transitions, and final-result cues pass timing, grounding, timeout, and privacy tests |
| `RELEASE-001` | In progress | Package, provision, validate, and deploy the public application. | A clean commit produces the Lambda image and Pages build; staging and production smoke, rollback, alarms, budgets, and teardown pass |

The reproducible local release candidate, alarm resources, cutover order, and
rollback plan are complete. The personal AWS account, `us-east-1`, Amazon Nova
Lite, Dracula voice, `/Dracula/`, proxied Cloudflare API record, and no public
seed/history disclosure are selected. Hosted staging awaits a renewed AWS
session. Post-staging resource/notification settings and final go-live
authorization remain in
[remaining user decisions](../reports/active/remaining-user-decisions.md).

The final-sprint hygiene pass removes generated build contexts and caches,
keeps the selected model only in ignored training evidence, and names the
active standalone artifact setting independently from the historical `pi0`
continuation artifact.

## Completed foundations

| ID | Result |
| --- | --- |
| `ENGINE-001` | The immutable engine implements deterministic dealing, legality, scoring, lifecycle, serialization, and invariance tests. |
| `APP-001` | FastAPI, SQLite local development, React, responsive scoring, reload recovery, and complete local games are implemented. |
| `SEARCH-001` | Player-relative information states and deterministic hidden-card sampling mechanically isolate controller inputs. |
| `SEARCH-002` | The original round-local information-set planner established the search and scoring foundation. |
| `SEARCH-004` | Nested actor-local search produced the strong Sam comparison controller. |
| `MINER-007` | The balanced D0 BGC-128 corpus sealed 500,010 visit-target rows. |
| `TRAIN-010` | D0 migration and `pi0` visit-distillation completed. |
| `SEARCH-006` | `pi0` continuation search and D1 collection completed. |
| `TRAIN-011` | D1 trained the selected 754,601-parameter `pi1` policy; fixed evaluation and 54-game self-play are sealed locally. |
| `UI-003` | Mobile and desktop layouts, scoring presentation, card art, Dracula portrait, cheat sheet, and interaction flow received final user approval. |
| `MAINT-001` | Historical reports and artifacts were indexed or archived and generated output excluded from source control. |

## Historical experiments

PPO, recurrent policy, separate critic, the original 500-simulation planner,
shallow Teacher v2, response rankers, policy/value expert iteration, hybrid
search, Sam-128, Sam-32 mining, D0, and D1 remain reproducible evidence under
[reports](../reports/README.md) or ignored sealed runs. They are not active
deployment alternatives.

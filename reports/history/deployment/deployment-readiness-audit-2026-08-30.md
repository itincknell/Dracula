# Deployment readiness audit

Date: 2026-08-30

> Historical pre-selection audit. Superseded by
> [the locked final deployment sprint](../../active/final-deployment-sprint-2026-08-30.md).

## Direct assessment

The game is a functioning local application, but it is not yet a deployable
release.

- Application core: largely built.
- Production opponent: not finally selected.
- Documentation: behind the actual `pi1` work.
- Release source: not committed or clean.
- Production infrastructure: not designed or implemented.
- Deployment automation: absent.

The active documents intentionally stop at a logical deployment boundary. They
do not currently define an AWS topology or an exact deployment command.

## Documented final application shape

```text
HTTPS endpoint
|-- React static frontend
|   `-- Calls same-origin /api
|
`-- FastAPI application
    |-- Deterministic engine: rules, legality, scoring
    |-- Game lifecycle, idempotency, and public projections
    |-- Production persistence -- undecided
    `-- One pinned opponent controller
        `-- Search or standalone policy -- undecided
```

The browser never directly accesses persistence, opponent state, hidden cards,
or search/model diagnostics.

The intended release has:

- One public single-player game.
- Queen and King starts.
- Six-round completion and replay.
- One fixed opponent difficulty.
- No accounts or multiplayer.
- Narration disabled unless separately completed.
- Server-owned rules, legal moves, scoring, and state.
- A responsive React frontend.
- FastAPI behind HTTPS.
- Durable game persistence.
- Rate limiting, logs, budget alerts, rollback, and teardown.

See [application architecture](../../../docs/architecture.md),
[product and scope](../../../docs/product-and-scope.md), and
[evaluation and operations](../../../docs/evaluation-and-operations.md).

## Current repository state

### Implemented

- Deterministic engine and scoring.
- Information-safe search.
- FastAPI API and lifecycle handling.
- SQLite transaction and idempotency support.
- React gameplay, scoring, recovery, rules, accessibility, and responsive UI.
- Nested Sam 32x32.
- BGC-128 and policy-continuation controllers.
- Standalone `pi0` and `pi1` policy artifacts.
- Local privacy and transaction tests.
- Production frontend build.
- Local SQLite game capture.

### Out of date

The tracker still says `pi0` training is active. In reality:

- D0 completed.
- `pi0` trained.
- D1 collected 896,448 rows and stopped.
- `pi1` trained.
- `pi1` evaluation and 54-game self-play were run.
- `pi1` remains an unaccepted candidate.
- The current local preview uses standalone `pi1`.

The `pi1` results and D1 state are not represented in the active tracker or
reports. The active documents also still say production evaluation is waiting
for D0 and `pi0`, which is no longer true.

### Missing

- Final opponent decision.
- Final `pi1`/BGC comparison evidence.
- Production persistence decision.
- Container or other deployable backend package.
- Infrastructure as code.
- CI/CD deployment workflow.
- Production rate limiter.
- Readiness check covering database and opponent.
- Structured production metrics/tracing.
- Traffic and concurrency limits.
- Budget alerts.
- Backup, rollback, and teardown implementation.
- Stable domain and TLS configuration.
- Production browser smoke test.

### Source-control condition

The repository is not currently a release candidate:

- Last commit: 2026-07-24, before most current work.
- 41 modified tracked files.
- 30 tracked deletions.
- Approximately 95 untracked files.
- Most current search, BGC, training, evaluation, and maintenance work is
  uncommitted.

No generated corpus is staged, but the working tree must be reviewed,
organized, committed, and tagged before deployment.

## A. Decisions -- required order

1. **Reconcile the current state.**

   Update the tracker, reports, and active contracts through `pi1` and D1.
   Decide which historical implementations remain packaged versus archived.

2. **Select the production opponent.**

   Choose among the actual viable candidates:

   - Nested Sam 32x32.
   - BGC-128 with `pi1` continuations.
   - Standalone `pi1`.

   Record strength, human-testing results, p50/p95/maximum latency, memory, and
   per-game compute. The current documents only formally contemplate nested Sam
   versus standalone inference, so selecting `pi1`-BGC also requires a contract
   update.

3. **Set the request-execution target.**

   Define maximum opponent-turn latency, timeout, expected concurrency, memory
   ceiling, and acceptable cost per game.

4. **Set narration scope.**

   Either release with narration disabled, which matches the current
   implementation, or complete the narrator contract. This decision affects
   almost nothing else if narration remains disabled.

5. **Choose production persistence.**

   Decide between:

   - Single-instance SQLite with a durable volume and constrained concurrency.
   - A production database adapter, most naturally PostgreSQL.

   Current SQLite is suitable locally but is not documented or validated as
   the final multi-instance store.

6. **Choose the physical hosting topology.**

   Based on the opponent and database decisions, select static frontend
   hosting, FastAPI compute, inline or separate opponent compute, database,
   same-origin `/api` routing, region, and availability requirements.

7. **Set operational policy.**

   Define retention, backups, traffic limits, budget ceiling, diagnostic
   retention, domain, and rollback window.

## B. Finish the build -- required order

1. Update documents and reports through `pi1`/D1.
2. Review and commit the complete working tree.
3. Pin the chosen controller and artifact digest as the sole production
   default.
4. Remove local absolute paths from release configuration and artifact metadata
   where applicable.
5. Implement the selected production persistence adapter and migrations.
6. Package FastAPI and the opponent runtime reproducibly.
7. Add opponent timeouts, concurrency bounds, and clean failure behavior for
   the selected profile.
8. Add production configuration and secret validation.
9. Add useful liveness and readiness endpoints.
10. Add structured logs, metrics, request IDs, and latency/error counters.
11. Implement rate limiting and request-size limits.
12. Configure same-origin frontend/API routing or explicitly implement CORS.
13. Add infrastructure definitions and CI/CD.
14. Run the complete Python suite from the final clean commit.
15. Run frontend unit, accessibility, TypeScript, lint, and production-build
    checks.
16. Run browser tests across the required viewports and browsers.
17. Run selected-controller load, timeout, retry, privacy, and recovery tests.
18. Seal and tag the release artifact set.

The frontend suite passed 73 tests, TypeScript, lint, and production build
during this audit period. The interrupted full-Python attempt is not release
evidence; the final committed source needs an uninterrupted run.

## C. Deploy -- required order

1. Provision a staging environment from the infrastructure definitions.
2. Provision the database, migrations, encryption, backups, and retention.
3. Upload and verify the pinned opponent artifact if the selected controller
   needs one.
4. Deploy FastAPI and opponent execution.
5. Deploy the static frontend.
6. Configure same-origin `/api` routing, domain, HTTPS, and security headers.
7. Enable logs, metrics, rate limits, health alarms, and budget alerts.
8. Run the staging deployment smoke:
   - `/health`.
   - Start as Queen.
   - Start as King.
   - Complete six rounds.
   - Reload during human, opponent, and scoring phases.
   - Verify result persistence.
   - Verify public-response privacy.
9. Run the selected-controller load and failure tests against staging.
10. Exercise backup restoration, rollback, and teardown.
11. Deploy the identical release to production.
12. Run the public smoke test.
13. Enable traffic gradually and watch latency, failures, database health, and
    cost.
14. Retain the prior release until the rollback window closes.

## Bottom line

The shortest route to deployment is:

1. Select the opponent.
2. Release without narration.
3. Update and commit the repository.
4. Choose persistence and hosting from the opponent's measured execution
   profile.
5. Implement production packaging, operations, and infrastructure.
6. Validate staging.
7. Deploy.

The game itself is substantially built. The remaining work is primarily final
selection, source consolidation, production persistence, operational
hardening, and infrastructure rather than game-rule or frontend architecture
work.

# Determinism simplification

## Pre-cutover baseline

The namespace-based implementation was captured before its intentional
replacement. The selected policy artifact was
`70c76f2eb64600eab2297640278a6c94d4336ab8e73bf941a6d96237f69f5b5c`, and
its state dictionary digest was
`2e981eea0880efb3a842258ce559aaf34288ee575fde15b4e080d09d32c33427`.

Policy logits and strategic choices were recorded for two deterministic games
at placements 1–7. The expected representative actions are:

| Fixture | Placements 1–7 |
| --- | --- |
| `namespace-cutover-a` | 25, 12, 10, 12, 29, 31, 30 |
| `namespace-cutover-b` | 9, 22, 13, 11, 28, 29, 28 |

The corresponding raw-logit SHA-256 digests are:

| Fixture | Placements 1–7 |
| --- | --- |
| `namespace-cutover-a` | `25e27a2`, `dcbd14ce`, `3afaab03`, `6fd7d7e5`, `0ba67066`, `44f5fd46`, `663531be` |
| `namespace-cutover-b` | `1c2ce7e0`, `78ce3a3a`, `192c52a1`, `05dfd74d`, `9bd58e1d`, `7eb1b8b4`, `27bdd9a4` |

The pre-cutover focused foundation, model, artifact, and BGC suite passed 41
tests. Exact historical deals, concrete symmetry coins, and BGC visit arrays are
not compatibility requirements after the cutover. Game rules, policy logits,
policy strategic choices, privacy, legal action selection, and same-release
determinism remain required.

## Completed remediation

The pre-cutover implementation is preserved in Git commit `3b1f1da`; the
simplified implementation is sealed by commit **Simplify deterministic
identities and policy artifacts**. The
active implementation now uses ordinary local `random.Random` generators. A
single `stable_seed` helper converts ordinary strings and integers to a
process-independent integer, allowing search, fair coins, training shuffles,
and replay tests to remain reproducible without named random streams, scopes,
or a custom counter generator.

Removed from the maintained path:

- every random namespace and destination scope;
- the SHA counter stream and custom rejection-sampling utilities;
- internal engine, information-state, symmetry, action, observation, model,
  initialization, loss, optimizer, response, and search schema registries;
- source-revision and source-tree identity machinery;
- nested configuration, state-dictionary, and optimizer digest registries;
- recurrent hidden-state fields left by deleted policies;
- artifact readers whose only purpose was accepting retired model formats.

Five format markers remain because they identify bytes that are actually read
from disk: the selected policy, converted corpus, training configuration,
training checkpoint, and local SQLite session. SHA-256 remains only for file or
content identity and for stable conversion of text to an integer seed.

## Selected policy cutover

The release policy is now
`runs/bgc-policy-pi1-001/artifacts/pi1-policy.pt`. It contains only:

```text
format: pi1-policy-v1
state_dict: 754,601 float32 parameters
```

Its file SHA-256 is
`d35196cf4513589def0ffb3c4c7c268e78652a46dab8ea41001ff2c648265203`.
The tensor digest remains
`2e981eea0880efb3a842258ce559aaf34288ee575fde15b4e080d09d32c33427`,
exactly matching the pre-cutover artifact. Frozen π1 logit and strategic-choice
tests pass against the new artifact.

## Intentional compatibility boundary

The cutover deliberately changes deterministic deals, concrete mirrored
destination coins, and exact BGC visit arrays. The current release remains
deterministic when repeated with the same inputs. Game rules, legal-action
ordering, scoring, information privacy, π1 tensors, π1 logits, and π1 strategic
group choices are unchanged.

Old serialized runtime sessions and old rich policy envelopes are not accepted
by the active readers. The prior implementation and its evidence remain at
`3b1f1da`; no compatibility branches remain in ordinary inference.

## Resulting source shape

The cutover commit removed 2,309 lines and added 752 lines of direct
replacements, tests, and documentation.
The active Python package is now 60 modules and 11,940 lines. The randomness
module fell from 124 to 40 lines, the policy artifact boundary from 439 to 128,
and the five retained BGC modules from 1,678 to 1,522.

## Validation

- Complete Python suite: 215 passed.
- Frontend: 85 unit/component tests passed; TypeScript, ESLint, and the
  `/Dracula/` production build passed.
- Browser: both full production-stateless Playwright cases passed.
- Packaging: the Lambda context contains 41 files, embeds only the new policy
  artifact, and excludes training, dataset, local-session, and BGC search code.
- Python wheel build and import compilation passed.
- Markdown internal links and Git diff formatting passed.

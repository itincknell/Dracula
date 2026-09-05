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

## Intended remediation

The active implementation will retain local deterministic random generators,
one minimal policy artifact format, and hashes used for content identity. It
will remove versioned random namespaces, random scopes, the custom SHA counter
stream, internal schema registries, nested artifact provenance checks, and
active readers whose only purpose is historical compatibility.

Historical evidence remains available from the pre-cutover Git commit rather
than through the active runtime.

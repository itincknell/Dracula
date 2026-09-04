# Bedrock narration implementation

Date: 2026-08-31

## Result

The stateless production API now exposes the three locked optional narration
cues through `POST /narration`:

1. Opening immediately after game creation.
2. Round transition from the completed state of rounds 1–5.
3. Final result after round 6 has advanced to `game_complete`.

Round 6 mechanically rejects `round_transition`. There are no move, per-score,
orientation, or additional round-six narration calls.

## Authoritative boundary

The request contains only the recovery envelope and cue type. It cannot contain
cue facts, a prompt, or model selection. `NarrationService` uses the existing
`StatelessGameplayService` to replay the envelope, validates the cue against
the reconstructed engine lifecycle, and derives the public facts.

Opening includes role, dealer, round, and first actor. Completed-round cues may
include the public coffin placements, round and cumulative scores, outcome,
selected score rank, and final engine tie-break. The provider boundary receives
no seed, command history, hand, remaining stock, policy input, mask, logits,
model state, or engine object.

The narration route never calls a gameplay mutation. Identical retry requests
may invoke narration again but cannot reapply, alter, or roll back a move.

## Bedrock adapter

`BedrockRuntimeAdapter` implements the current AWS Bedrock Runtime `Converse`
shape: fixed `modelId`, one system text block, one user text block, and common
`maxTokens` and `temperature` inference parameters. It accepts exactly one
bounded assistant text block from `output.message.content` and records token
counts from `usage` when present.

The implementation follows the official AWS
[`Converse` API](https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html),
[conversation guide](https://docs.aws.amazon.com/bedrock/latest/userguide/conversation-inference.html),
and [Botocore client configuration](https://docs.aws.amazon.com/botocore/latest/reference/config.html).

Defaults are:

- 96 output tokens.
- 400 parsed characters.
- 8-second read timeout and at most 2-second connect timeout.
- Temperature zero.
- One total SDK attempt, so the optional request has no automatic retries.

The AWS SDK import is lazy and occurs only when enabled narration constructs a
real adapter. The following Lambda packaging stage must install and pin Boto3
and Botocore. Mechanical integration uses `FakeNarrationAdapter` and does not
select or invoke a billable model.

## Failure and logging behavior

Disabled narration, timeout, provider failure, empty output, malformed output,
non-text output, control characters, or oversized output returns:

```json
{"cue_type":"round_transition","status":"unavailable","text":null}
```

No canned line is substituted. Invalid histories and lifecycle-ineligible cues
remain request errors and never call the provider.

Server logs contain only cue type, success/failure category, elapsed
milliseconds, and token counts. They omit model ID, prompt, cue facts, seed,
history, cards, hands, stock, and policy data.

## Verification

Focused Python verification passed **160 tests** across engine, bridge,
information-state, selected policy, stateful and stateless API, transaction,
privacy, and narration boundaries. The narration cases cover:

- Canonical Bedrock request construction and normalized response parsing.
- Opening, rounds 1–5 transitions, and final result over a complete game.
- Explicit rejection of the round-six transition cue.
- Engine-grounded score, outcome, tie-break, and public-placement facts.
- Timeout and repeated-request isolation from gameplay.
- Disabled narration and non-fabricated empty responses.
- Malformed, empty, control-character, multi-block, and oversized output.
- Rejection of browser facts, prompts, and model IDs.
- Seed, history, stock, hand, policy, model, and search privacy.
- Lazy API imports with no Boto3, Botocore, SageMaker, or PyTorch import.

The complete frontend check passed **74 tests**, TypeScript, ESLint, and the
verified production build. The frontend was not migrated to the stateless API
in this stage; its later integration owns request timing and reveals a completed
round cue only after the scoring animation.

## Deferred user selections

The exact deferred items are maintained in
[`docs/deployment.md`](../../docs/deployment.md#later-user-decisions). Neither
blocks the completed boundary or fake-adapter tests.

## Remaining dependency

No Lambda image, AWS resource, frontend hosting change, real model invocation,
commit, or push was performed. The next stage packages the stateless API,
selected `pi1`, Bedrock adapter, and pinned runtime dependencies in the Lambda
container.

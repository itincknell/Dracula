# Narrator

## Responsibility

Amazon Bedrock supplies optional Dracula presentation text. It never chooses
moves, scores cards, validates actions, or changes the seed-and-history game
envelope. The stateless FastAPI application invokes one configured Bedrock
text model directly; no agent, tool loop, knowledge base, or SageMaker service
is involved.

The implementation uses the Bedrock Runtime
[`Converse`](https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html)
operation. AWS documents `Converse` as the normalized messages interface for
models that support it, with `system`, `messages`, and `inferenceConfig`
request fields and assistant text under `output.message.content`. Calling it
requires `bedrock:InvokeModel` permission. See the
[Bedrock conversation guide](https://docs.aws.amazon.com/bedrock/latest/userguide/conversation-inference.html).

## Exact cadence

Only these cue classes invoke Bedrock:

1. **Opening** — after game creation.
2. **Round transition** — after the eighth placement in rounds 1–5. The browser
   requests it immediately, runs the scoring animation, and reveals the text
   after the animation ends.
3. **Final result** — after the completed-game result following round 6.

There is no per-move banter, round-opening narration after round 1,
orientation-specific narration, separate response for each player's score, or
round-six transition narration.

The browser owns presentation timing. Narration is a separate request and does
not participate in the accepted gameplay mutation. Delay, timeout, malformed
output, or provider failure returns an unavailable state and cannot block,
retry, alter, or roll back a move.

## Endpoint and grounding

`POST /narration` accepts exactly:

```json
{
  "envelope": {
    "seed": "...",
    "history": [{"type": "select_role", "human_role": "queen"}]
  },
  "cue_type": "opening"
}
```

The caller cannot submit scores, placements, prompts, or a model ID. FastAPI
replays the envelope through the authoritative gameplay service, verifies that
the requested cue matches the reconstructed lifecycle, and derives the public
facts itself.

Opening facts are the two roles and first actor. Each round-transition contains
only the round number, winner or tie, one mechanically worded description of
the winner's best combination, and one mechanically derived lead-change or
catch-up statement. For example: `three Spades for a 5x multiplier`. A tied
round omits the winning combination. Final-result facts contain only the game
winner or tie and the engine tie-break reason. Bedrock receives no raw score,
coffin placement, losing-player combination, hand, remaining stock, seed,
command history, policy input, mask, logits, model data, search state, or Python
engine object.

Success is:

```json
{"cue_type":"opening","status":"ready","text":"..."}
```

Disabled narration or any provider/output failure is the non-fabricated state:

```json
{"cue_type":"opening","status":"unavailable","text":null}
```

An ineligible cue is rejected before provider invocation. Repeating an
eligible request may generate again but never mutates the game envelope.

## Bedrock request boundary

The request constructor emits the approved arcade-villain system prompt and one
canonical JSON user message containing only the derived cue. Dracula opens with
a taunt, gloats when winning, and reacts angrily when losing, without profanity,
memes, or invented facts. Temperature is zero. `maxTokens` defaults to 96, and
parsed text is additionally limited to 400 characters. Only one nonempty assistant text block is accepted; tool,
reasoning, binary, empty, control-character, multi-block, and oversized output
is rejected.

`make preview-dialogue` uses the same grounding and frontend timing with a
deterministic local adapter. It does not call Bedrock.

`DRACULA_NARRATION_TIMEOUT_SECONDS` defaults to 8 seconds. The SDK client uses
bounded connect and read timeouts. Automatic SDK retries are disabled for this
optional presentation request (`total_max_attempts = 1`); the
[Botocore configuration contract](https://docs.aws.amazon.com/botocore/latest/reference/config.html)
defines these timeout and retry fields.

The adapter is created only when narration is enabled, so importing the API in
local narrator-disabled mode does not import AWS libraries. Tests use the
deterministic fake adapter. The Lambda packaging stage installs and pins the
AWS SDK.

## Configuration and logs

Production configuration fixes:

- `DRACULA_NARRATION_ENABLED`
- `DRACULA_BEDROCK_REGION`
- `DRACULA_BEDROCK_MODEL_ID`
- `DRACULA_NARRATION_TIMEOUT_SECONDS`
- `DRACULA_NARRATION_MAX_TOKENS`

Diagnostics contain cue type, success or failure category, latency, and input
and output token counts. They omit seed, command history, cue facts, prompt,
cards, hands, stock, policy data, and model ID. Narration is not
server-persistent. The browser may retain successful display text locally.

## Selected model and voice

Hosted staging uses Amazon Nova Lite (`amazon.nova-lite-v1:0`) in `us-east-1`.
The selected voice is a gloating, egotistical cartoon villain: short, readable,
arcade-style taunts; anger when losing; no profanity, modern slang, personal
abuse, or facts not supplied by the game. A winning combination is mechanically
phrased in player-facing language such as `three Spades for a 5x multiplier`.
The frontend presents the response with its approved retro typing cursor.

See [architecture](architecture.md#narrator-scheduling),
[stateless API](stateless-api.md#post-narration), and
[deployment](deployment.md#selected-release-inputs).

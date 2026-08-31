# Narrator

## Responsibility

Amazon Bedrock supplies Dracula's presentation text. It never chooses moves,
scores cards, validates actions, or changes the seed-and-history game envelope.
The Lambda application invokes one allowlisted Bedrock text model directly; no
agent, tool loop, knowledge base, or SageMaker service is involved.

## Exact cadence

Only these cue classes invoke Bedrock:

1. **Opening** — once after game creation.
2. **Round transition** — after the eighth placement in rounds 1–5. The browser
   requests it immediately, runs the scoring animation, and reveals the text
   after the animation ends.
3. **Final game result** — once after round 6 completes. It replaces the round-
   transition cue; there is no separate final-round response.

There is no per-move banter, round-opening narration after round 1,
orientation-specific narration, or separate response for each player's score.

The browser owns presentation timing. A narration request contains its complete
public cue and may reach any Lambda environment. A delayed response appears
only after its corresponding animation; a failed response produces no invented
fallback and cannot block the game.

## Input boundary

The public cue may contain:

- Human role and Dracula role.
- Round number.
- Public coffin and played cards.
- Engine-calculated round scores and cumulative scores.
- Round winner or final game result.

It excludes both hands, stock order, game seed, model observation, masks,
logits, training data, search diagnostics, and intended strategy. The client
game envelope contains the seed for stateless replay, but that envelope is not
forwarded to Bedrock.

## Configuration

Deployment configuration fixes the Bedrock region, allowlisted model ID,
character prompt, inference parameters, output bound, and timeout. Public
callers supply only an eligible structured cue. They cannot select a model or
provide prompt instructions.

Narration is not server-persistent. The browser may retain successful text with
its local game display state. See [architecture](architecture.md#narrator-scheduling)
and [deployment](deployment.md#bedrock-behavior).

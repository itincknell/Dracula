# Deployment

## Locked topology

The production release uses:

- `https://ian-tincknell.com/Dracula/` for the Vite/React frontend on GitHub
  Pages.
- Cloudflare for DNS.
- `https://api.ian-tincknell.com/` as a Regional API Gateway HTTP API custom
  domain.
- One AWS Lambda container running the Lambda Web Adapter, FastAPI,
  deterministic engine, and selected standalone `pi1` policy.
- Direct Amazon Bedrock inference for the three narration cue classes.
- No SageMaker, DynamoDB, RDS, EFS, or other production persistence.

The existing `ian-tincknell.com` GitHub Pages records remain unchanged.
Cloudflare receives one additional `api` CNAME targeting the hostname returned
by API Gateway. The API Gateway custom domain uses an ACM certificate in the
same AWS region as the API.

## Frontend deployment

The production frontend build uses `/Dracula/` as its Vite base and resolves
all cards, fonts, portraits, rules, and configured destinations from that base.
Its API origin is a build-time value:

```text
https://api.ian-tincknell.com
```

Client-side game recovery uses browser storage containing only the initial seed
and ordered command history. A direct reload of the application must return the
GitHub Pages entry point and reconstruct the game through the API. Clean URL
handling may use a Pages-compatible 404 entry point or a single `/Dracula/`
application route; no AWS frontend hosting is required.

## Lambda package

The image contains:

- The pinned Python runtime and dependencies.
- FastAPI and the Lambda Web Adapter.
- Deterministic engine and stateless replay service.
- The exact `pi1` state dictionary and model code.
- Bedrock client integration.
- No training, miner, search, historical-controller, or SQLite runtime data.

The model loads during Lambda initialization and remains available to warm
invocations. A bounded in-memory cache maps a local digest of seed plus history
to reconstructed engine state. Cache eviction affects performance only.

## State and request behavior

The client envelope is plain bounded JSON with an initial seed and ordered
history. It is neither encrypted nor signed. The service strictly validates
the JSON and replays every command. The release explicitly permits a user to
inspect or alter their single-player game data.

API requests and responses stay well below API Gateway payload limits. The
service sets request-body and history-length bounds independently of those
provider limits.

## Bedrock behavior

The application invokes one allowlisted Bedrock text model with fixed prompt
instructions and bounded output. Eligible calls are:

1. Opening.
2. Round transition after rounds 1–5.
3. Final game result after round 6.

For round transitions, the browser starts the Bedrock request after receiving
the final move and reveals the response only when the scoring animation is
complete. Round 6 substitutes the final-result cue and has no additional round
response. The browser may retain the last successful text for reload display;
narration is not server-persistent.

## Cloudflare and TLS

The first release uses a DNS-only Cloudflare CNAME:

```text
type: CNAME
name: api
target: <API Gateway regional domain>
proxy: DNS only
```

API Gateway terminates TLS with its ACM certificate. Cloudflare proxying is not
required for release. If enabled later, it is an operational change rather
than an application architecture change.

## Configuration and secrets

Release configuration supplies:

- Allowed frontend origin.
- Bedrock region and allowlisted model ID.
- Narration timeout and output bound.
- Replay-cache size.
- Application log level.

AWS IAM permits only the required Bedrock invocation and logging actions.
There is no database credential. The game seed is application data rather than
a secret.

## Release validation

Staging and production smoke tests cover:

- GitHub Pages assets and `/Dracula/` base paths.
- API custom-domain TLS and CORS.
- Cold and warm Lambda initialization.
- Cache-hit/cache-miss replay equivalence.
- Queen and King starts.
- Six complete rounds and Play Again.
- Reload during play and scoring.
- Legal standalone `pi1` moves.
- Opening, rounds 1–5, and final Bedrock cues.
- No round-six transition cue.
- Bedrock timeout without gameplay failure.
- Absence of model data and prompts from public responses.

## Remaining phases

1. Implement the stateless seed-and-history request service and cache.
2. Adapt the frontend to store and submit the envelope.
3. Replace the narration cadence with the three locked cue classes and connect
   Bedrock.
4. Package the Lambda container with the exact selected `pi1` artifact.
5. Add infrastructure definitions for ECR, Lambda, API Gateway, ACM, IAM,
   CloudWatch, and budget alarms.
6. Make the Vite build `/Dracula/`-aware and add the production API origin.
7. Configure the Cloudflare `api` CNAME after API Gateway returns its target.
8. Validate a complete staging deployment, then deploy the identical release
   to production.

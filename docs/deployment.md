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
handling uses `#/`, `#/game`, and `#/rules` beneath the single `/Dracula/`
entry point; no router, copied 404 page, or AWS frontend hosting is required.

The tracked [Pages workflow](../.github/workflows/pages.yml) installs from the
frontend lockfile, runs unit tests, TypeScript, lint, and production build
verification, then uploads only `frontend/dist`. Pull requests validate without
publishing. Deployment occurs only through a manual workflow dispatch whose
`publish` input is explicitly enabled; it requires no AWS credential.

Local verification is:

```bash
make pages-build
make pages-test
```

The first command verifies the `/Dracula/` asset paths, configured production
API origin, and absence of local URLs. The second completes a real stateless
six-round browser game with `pi1`, reloads opening/play/scoring/final states,
checks narration timing and failure isolation, and covers desktop and mobile
geometry without publishing.

## Lambda package

The arm64 image contains:

- The pinned Python runtime and dependencies.
- FastAPI and the Lambda Web Adapter.
- Deterministic engine and stateless replay service.
- The exact `pi1` state dictionary and model code.
- Bedrock client integration.
- No run directories, datasets, checkpoints other than the selected `pi1`,
  SQLite files, frontend output, or historical artifacts. The project Python
  source remains one restrained package; inactive utilities are not invoked by
  the production entry point.

The model loads during Lambda initialization and remains available to warm
invocations. A bounded in-memory cache maps a local digest of seed plus history
to reconstructed engine state. Cache eviction affects performance only.
The production entry point reads its active settings in one place:
`DRACULA_POLICY_ARTIFACT`, `DRACULA_NARRATION_ENABLED`, and
`DRACULA_REPLAY_CACHE_ENTRIES`. Local comparison-controller selectors are not
part of the Lambda configuration.

The release context is always generated under ignored `build/` output. The
builder requires
`runs/bgc-policy-pi1-001/artifacts/pi1-policy.pt`, verifies SHA-256
`d35196cf4513589def0ffb3c4c7c268e78652a46dab8ea41001ff2c648265203`,
and copies only that model artifact into the context. The binary is never
tracked by Git. Python, Lambda Web Adapter 1.0.1, CPU PyTorch, and every runtime
dependency are pinned; the Python and adapter base images are digest-pinned.

The container uses a read-only root filesystem. Only Lambda's `/tmp` is
writable. The production ASGI entry point is
`dracula.api.production:app`, and its Uvicorn/application logs are single-line
JSON.

## Infrastructure definition

[`infrastructure/application.yaml`](../infrastructure/application.yaml) is the
single application definition. It creates:

- One arm64 image Lambda with parameterized memory, timeout, reserved
  concurrency, and replay-cache bound.
- One API Gateway HTTP API using payload format 2.0 and explicit `GET /health`,
  `POST /games`, `POST /games/command`, `POST /games/resume`, and
  `POST /narration` routes.
- API Gateway-managed CORS for one configured frontend origin.
- JSON Lambda and API access logs with retention controls.
- Free API/stage metrics without separately billed route-level detailed metrics.
- Lambda-error and API-5xx alarms, with an optional email-subscribed SNS topic.
- A Lambda execution role that can write only its application log group and,
  when narration is enabled, invoke only the configured Bedrock model ARN.
- An optional TLS 1.2 Regional custom domain and API mapping. Its ACM
  certificate must be in the API's AWS region.
- An optional monthly AWS cost budget with an 80% forecast notification. It is
  disabled until the production amount and recipient are approved.

[`infrastructure/ecr.yaml`](../infrastructure/ecr.yaml) separately creates the
immutable, scan-on-push ECR repository because a Lambda stack can only refer to
an image after that image exists. Both templates remain plain CloudFormation.

Staging and production defaults live in
[`infrastructure/parameters/`](../infrastructure/parameters/). The release is
assigned to the user's personal AWS account in `us-east-1`, with Amazon Nova
Lite (`amazon.nova-lite-v1:0`) selected for narration. The generic tracked
parameter files keep narration and the custom domain disabled so account-bound
ARNs and certificate values are rendered only for an authenticated deployment.

## Build and local validation commands

```bash
.venv/bin/python tools/build_lambda_context.py
docker build --platform linux/arm64 -t dracula-api:local build/lambda-context
.venv/bin/python tools/validate_lambda_container.py \
  --image dracula-api:local --port 18080
```

Equivalent Make targets are `make lambda-build`, `make lambda-run`, and
`make lambda-validate`. Local validation runs a complete six-round game on a
read-only container, verifies the embedded artifact, exercises warm and
cold-process replay, and writes ignored measurements to
`build/lambda-validation.json`.

## Package and staging deployment commands

These commands require a currently authenticated AWS CLI in the user's
personal AWS account.

```bash
export AWS_REGION=us-east-1
export RELEASE_TAG="$(git rev-parse --short=12 HEAD)-$(date -u +%Y%m%d%H%M%S)"

aws cloudformation deploy \
  --region "$AWS_REGION" \
  --template-file infrastructure/ecr.yaml \
  --stack-name dracula-api-registry \
  --parameter-overrides RepositoryName=dracula-api

export ECR_URI="$(aws cloudformation describe-stacks \
  --region "$AWS_REGION" --stack-name dracula-api-registry \
  --query 'Stacks[0].Outputs[?OutputKey==`RepositoryUri`].OutputValue' \
  --output text)"
export AWS_ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
aws ecr get-login-password --region "$AWS_REGION" | \
  docker login --username AWS --password-stdin \
  "$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com"

.venv/bin/python tools/build_lambda_context.py
docker buildx build --platform linux/arm64 --provenance=false \
  --tag "$ECR_URI:$RELEASE_TAG" --push build/lambda-context
export IMAGE_DIGEST="$(aws ecr describe-images \
  --region "$AWS_REGION" --repository-name dracula-api \
  --image-ids imageTag="$RELEASE_TAG" \
  --query 'imageDetails[0].imageDigest' --output text)"
export IMAGE_URI="$ECR_URI@$IMAGE_DIGEST"

jq --arg image "$IMAGE_URI" \
  '. + [{"ParameterKey":"ImageUri","ParameterValue":$image}]' \
  infrastructure/parameters/staging.json \
  > build/lambda-staging-parameters.json
aws cloudformation deploy \
  --region "$AWS_REGION" \
  --template-file infrastructure/application.yaml \
  --stack-name dracula-api-staging \
  --parameter-overrides file://build/lambda-staging-parameters.json \
  --capabilities CAPABILITY_IAM
```

`docker buildx ... --push` is the package operation: ECR stores the image, and
CloudFormation receives its immutable digest URI rather than a mutable tag.

## Update, rollback, and delete

An update repeats the package commands with a new tag/digest and then repeats
the same `cloudformation deploy`. To roll back intentionally, rebuild the
parameter file with the prior recorded immutable `IMAGE_URI` and redeploy:

```bash
export IMAGE_URI=<prior-ecr-uri-at-sha256-digest>
jq --arg image "$IMAGE_URI" \
  '. + [{"ParameterKey":"ImageUri","ParameterValue":$image}]' \
  infrastructure/parameters/staging.json \
  > build/lambda-staging-parameters.json
aws cloudformation deploy --region "$AWS_REGION" \
  --template-file infrastructure/application.yaml \
  --stack-name dracula-api-staging \
  --parameter-overrides file://build/lambda-staging-parameters.json \
  --capabilities CAPABILITY_IAM
```

Delete the staging application without deleting retained ECR images:

```bash
aws cloudformation delete-stack \
  --region "$AWS_REGION" --stack-name dracula-api-staging
aws cloudformation wait stack-delete-complete \
  --region "$AWS_REGION" --stack-name dracula-api-staging
```

The registry stack intentionally refuses deletion while it contains images.
Removing retained releases is a separate explicit operation.

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
instructions and bounded output through Bedrock Runtime `Converse`. The
narration request contains an envelope plus cue type; FastAPI derives all
public facts by replay. Eligible calls are:

1. Opening.
2. Round transition after rounds 1–5.
3. Final game result after round 6.

For round transitions, the browser starts the Bedrock request after receiving
the final move and reveals the response only when the scoring animation is
complete. Round 6 substitutes the final-result cue and has no additional round
response. The browser may retain the last successful text for reload display;
narration is not server-persistent.

The default output bound is 96 tokens and 400 parsed characters. The default
read timeout is eight seconds, and automatic SDK retries are disabled so an
optional cue cannot extend unpredictably. Provider and parsing failures return
an unavailable empty state rather than fabricated dialogue.

## Cloudflare and TLS

The final Cloudflare proxy mode is a release decision. In either mode the
production record is a CNAME:

```text
type: CNAME
name: api
target: <API Gateway regional domain>
proxy: <approved DNS-only or proxied setting>
```

API Gateway terminates TLS with its ACM certificate. Cloudflare proxying is not
required by the application, but the production choice is not made by this
repository. ACM DNS validation also requires its separate CNAME to remain
DNS-only while the certificate is issued and renewed.

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

The complete local staging-equivalent pass is recorded in
[staging deployment validation](../reports/active/staging-deployment-validation.md).
The exact image, four complete role/dealer games, process-replacement replay,
frontend lifecycle, narration failure boundary, privacy, cost, and teardown
passed locally. Hosted staging remains undeployed because the configured AWS
session is expired. Region, model, public path, narration direction, and DNS
proxy mode are selected.

## Production cutover order

The exact command plan, immutable local candidate identity, rollback, and
decision checklist are recorded in
[production release readiness](../reports/active/production-release-readiness.md).
Production proceeds in this order only after every release input is approved:

1. Publish the verified arm64 image to ECR and record its registry digest.
2. Deploy the production Lambda and HTTP API by immutable image digest.
3. Request and DNS-validate the same-region ACM certificate.
4. Add the Regional API Gateway domain and mapping through a reviewed
   CloudFormation change set.
5. Add the approved Cloudflare `api` CNAME and validate HTTPS/CORS.
6. Manually publish the verified `/Dracula/` Pages artifact.
7. Complete production gameplay, narration, privacy, alarm, and rollback
   smoke tests.

No production change set, DNS record, Pages release, or AWS resource is
created by the preparation step.

## Selected release inputs

- AWS owner: the user's existing personal account.
- Region: `us-east-1`.
- Bedrock model: Amazon Nova Lite, `amazon.nova-lite-v1:0`.
- Narration: the approved egotistical cartoon-villain voice in
  [narrator](narrator.md).
- Frontend path: `/Dracula/`.
- Cloudflare API record: proxied; ACM validation records remain DNS-only.
- Seed/history disclosure: no user-facing disclosure.

## Remaining user decisions

⭕ USER DECISION LATER: Approve final Lambda memory, timeout, reserved
concurrency, notification recipients, and budget after hosted staging
measurements.

⭕ USER DECISION LATER: Authorize the production go-live after hosted staging.

## Remaining phases

1. Renew the personal-account AWS session and deploy the already validated
   image/templates to `us-east-1` staging with Amazon Nova Lite.
2. Measure hosted Lambda and narration behavior, then resolve the remaining
   resource and notification settings.
3. Configure the proxied Cloudflare `api` CNAME after API Gateway returns its
   Regional target.
4. Validate the complete staging release, obtain explicit go-live approval,
   manually publish the verified Pages artifact, then promote the identical
   immutable API image to production.

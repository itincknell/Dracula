# Deployment

## One application, one image

FastAPI serves the built React frontend at `/Dracula/` and the stateless JSON
API at `/Dracula/api`. The arm64 Lambda image contains the frontend, engine,
selected standalone pi1, and Bedrock client. It needs no Node runtime, database,
SageMaker endpoint, or persistent filesystem.

The browser uses the relative API prefix `/Dracula/api`. Hash navigation
(`#/game`, `#/rules`) needs no server-side page router or HTML catch-all.
Missing files and unknown API routes return errors, not the home page.

Cloudflare will forward only the Dracula path to API Gateway's generated HTTPS
endpoint. Other paths stay on the personal GitHub Pages website. This does not
require renaming that repository, publishing Dracula to Pages, or creating an
API custom domain and ACM certificate. The public Worker route is **not enabled**
until the user authorizes cutover.

## Build and verify

```bash
make test
make test-e2e
node --test deployment/cloudflare/worker.test.mjs
make web-build
make lambda-build
make lambda-validate LAMBDA_PORT=18080
```

The browser tests temporarily build with faster reduced-motion scoring for
testing. The subsequent normal build resets that test setting before packaging.
The [verification workflow](../.github/workflows/frontend.yml) validates only;
it never publishes a Pages site.

The context builder verifies the selected artifact at
`runs/bgc-policy-pi1-001/artifacts/pi1-policy.pt` against SHA-256
`d35196cf4513589def0ffb3c4c7c268e78652a46dab8ea41001ff2c648265203`.
It copies that artifact, the production Python import closure, and only
`frontend/dist` into ignored `build/lambda-context`. The static mount exposes
only the frontend directory, never source or weights.

Python, Lambda Web Adapter, and runtime dependencies are pinned. The model
loads once during initialization. A bounded cache of immutable reconstructed
games speeds replay but is not needed for correctness. Only `/tmp` is writable.
Structured application logs omit seeds, histories, hands, and model tensors.

The AWS configuration enables the Web Adapter's `AWS_LWA_ASYNC_INIT` setting
so a slow model import is not restarted at the initial ten-second boundary.
The first request still waits for readiness and remains subject to the normal
Lambda/API timeout. This is not provisioned concurrency or a cold-start guarantee.
See [AWS's adapter configuration](https://github.com/aws/aws-lambda-web-adapter#configurations).

The validator runs four complete role/dealer games, compares cached replay with
a replacement process having its cache disabled, verifies served binary files
byte-for-byte, and records timings/memory in `build/lambda-validation.json`.

## Local combined preview

```bash
make preview-app NARRATION_ENABLED=true
```

Open <http://127.0.0.1:4173/Dracula/> in Chrome. Boto3 uses the existing AWS
credential chain; no application-specific profile is required. Omit
`NARRATION_ENABLED=true` to play without Bedrock. Stop with Ctrl-C.

`make dev` retains Vite hot reload and its local API proxy.
`make preview-dialogue` retains deterministic test dialogue.

To serve only the frontend locally while gameplay and narration run on staging:

```bash
make preview-staging STAGING_API_ORIGIN=https://gk6bbk03vd.execute-api.us-east-1.amazonaws.com
```

Open <http://127.0.0.1:4173/Dracula/> in Chrome. Vite forwards `/api` requests to
the hosted `/Dracula/api`; there is no local engine and no AWS credential in the
browser. This separate build lives in `build/staging-frontend`, leaving the
production distribution intact. Stop with Ctrl-C.

## AWS staging

Use the personal account in `us-east-1`. The application template creates one
arm64 image Lambda, HTTP API, least-privilege execution role, retained log
groups, and error alarms. Optional notification emails and budgets remain
disabled until configured. The role permits only its log writes and the selected
Bedrock model. No browser or container receives the operator's AWS credentials.

The selected limits are 2048 MB, 20 seconds, and 256 replay cache entries.
Staging and production use `ReservedConcurrency=-1` (no reservation), sharing
the account's current ten concurrent executions. There is no two-execution
per-function cap and no quota increase is needed. Production's approved monthly
budget is $10 in actual AWS account-wide charges, with an email after the
threshold is exceeded. It is not a spending cap. Supply the user's notification
address at deployment; it is deliberately not stored in tracked configuration.

API POSTs are throttled at 5 requests/second with a burst of 10. Static GETs
allow 20/second and a burst of 50 to accommodate page assets. These are API
Gateway best-effort throttles, not hard spending limits.

Build before running these publication commands:

```bash
export AWS_REGION=us-east-1
export RELEASE_TAG="combined-$(date -u +%Y%m%d%H%M%S)"
aws cloudformation deploy --region "$AWS_REGION" \
  --template-file infrastructure/ecr.yaml --stack-name dracula-api-registry \
  --parameter-overrides RepositoryName=dracula-api
export ECR_URI="$(aws cloudformation describe-stacks --region "$AWS_REGION" \
  --stack-name dracula-api-registry \
  --query 'Stacks[0].Outputs[?OutputKey==`RepositoryUri`].OutputValue' --output text)"
aws ecr get-login-password --region "$AWS_REGION" | \
  docker login --username AWS --password-stdin "${ECR_URI%/*}"
docker tag dracula-api:local "$ECR_URI:$RELEASE_TAG"
docker push "$ECR_URI:$RELEASE_TAG"
export IMAGE_DIGEST="$(aws ecr describe-images --region "$AWS_REGION" \
  --repository-name dracula-api --image-ids imageTag="$RELEASE_TAG" \
  --query 'imageDetails[0].imageDigest' --output text)"
export IMAGE_URI="$ECR_URI@$IMAGE_DIGEST"

aws cloudformation deploy --region "$AWS_REGION" \
  --template-file infrastructure/application.yaml --stack-name dracula-api-staging \
  --capabilities CAPABILITY_IAM --parameter-overrides \
  EnvironmentName=staging ImageUri="$IMAGE_URI" ReservedConcurrency=-1 \
  NarrationEnabled=true BedrockModelId=amazon.nova-lite-v1:0 \
  BedrockModelArn=arn:aws:bedrock:us-east-1::foundation-model/amazon.nova-lite-v1:0
export API_ORIGIN="$(aws cloudformation describe-stacks --region "$AWS_REGION" \
  --stack-name dracula-api-staging \
  --query 'Stacks[0].Outputs[?OutputKey==`DefaultApiUrl`].OutputValue' --output text)"
curl --fail "$API_ORIGIN/Dracula/api/health"
```

Open `$API_ORIGIN/Dracula/` directly for real Lambda/Bedrock testing before
Cloudflare changes. The immutable image URI identifies the exact deployed
release. Docker's `--network=host` build option can be used locally if Docker
Desktop dependency downloads stall; it does not change runtime networking.

## Narration and state

Narration uses Amazon Nova Lite `amazon.nova-lite-v1:0`, with 96 output tokens,
an eight-second provider timeout, and no SDK retries. The browser requests
opening, round transitions 1–5, and the final result separately from gameplay.
Transition text appears after scoring finishes. Failure returns an unavailable
state and never changes a move.

The browser stores the plain seed and accepted command history. Replay,
validation, and public projection remain unchanged. See the
[API contract](stateless-api.md) and [narration contract](narrator.md).

## Public deployment

Production is deployed at
<https://3ylzpjexng.execute-api.us-east-1.amazonaws.com/Dracula/> with shared
concurrency and the $10 monthly email budget alert. Functional checks passed
after two startup timeouts.
The public game is live at <https://ian-tincknell.com/Dracula/>. Cloudflare's
narrow Worker route forwards to this AWS origin; the personal website remains
on GitHub Pages. The latest release includes the approved loading overlay,
final-result transition, and mobile dialogue-spacing fixes.

For future releases or rebuilding this arrangement:

1. Promote the verified image digest to a separately named production stack.
   Confirm concurrency, notification recipients, and budget settings.
2. Set the prepared [Worker](../deployment/cloudflare/README.md) origin to that
   stack's generated HTTPS endpoint.
3. Coordinate with the website agent: preserve existing GitHub origin records,
   enable proxying on the apex record, and verify the existing site's TLS,
   home, projects, and www behavior.
4. Enable only `ian-tincknell.com/Dracula*`. The Worker explicitly passes
   non-Dracula paths through unchanged.
5. Test the public game, binary assets, Rules tab, reload, and Bedrock.
6. Publish the personal website's Dracula link only after the public game works.

API and narration responses are never cached. The Worker caches successful
public files: hashed assets for one year, HTML and named artwork for one hour.
Browsers revalidate HTML/artwork with the edge rather than necessarily reaching
Lambda. After each release or rollback, run the scoped
[Dracula cache purge](../deployment/cloudflare/README.md#release-and-rollback-cache-invalidation)
and check its success before announcing completion. No unrelated website cache
is purged. Role selection displays “Starting game…” during the initial request;
edge misses can still cause a cold start before that screen.

## Monitor, rollback, and teardown

```bash
aws logs tail /aws/lambda/dracula-api-staging --region us-east-1 --follow
aws cloudformation describe-stack-events --region us-east-1 \
  --stack-name dracula-api-staging
```

To roll back code, repeat the deployment command with the previous recorded
immutable `IMAGE_URI`, preserving the other deployed parameters. Do not rebuild
an old tag and assume it produces identical bytes.

To withdraw the public route, disable its Cloudflare Worker route. The personal
site stays available; Dracula returns the site's previous behavior (currently
404). To reverse a Worker-only change, restore its prior published version and
origin setting.

To delete staging without removing release images:

```bash
aws cloudformation delete-stack --region us-east-1 --stack-name dracula-api-staging
aws cloudformation wait stack-delete-complete --region us-east-1 \
  --stack-name dracula-api-staging
```

The ECR registry intentionally rejects deletion while it contains images.
Deleting release images is a separate action.

# Production release readiness

## Bottom line

A reproducible **local production release candidate** is sealed and the
production cutover is scripted. It is not a hosted or approved production
release. No AWS resource, Cloudflare record, GitHub Pages deployment, change
set, commit, or push was created in this stage.

Hosted staging did not run because the configured AWS credentials return
`ExpiredToken`; the AWS region and Bedrock model are also unresolved. The
release therefore remains **not authorized for production cutover**. The exact
remaining user decisions appear below.

## Sealed local candidate

The ignored machine-readable manifest is:

```text
build/release-candidate/manifest.json
```

Its deterministic identifiers are:

| Component | Identity |
| --- | --- |
| Candidate content | `fa9f31c2aadfa9f103e65b94844af86a95314e94a3aebdab06f3cff1087d6a29` |
| Manifest file | `4ca9db52eba4be1ec50bdf897b7b3a2964b453c51278eaa9d8b52af7187c446d` |
| Git HEAD | `81b2504d4cc679e40e36a1fd63160037e6dfbb0d` |
| Tracked working-tree diff | `1e4f4cde3288818b1a34ba35e3a30d16f912a51fd3512556bd20a132742436ad` |
| Release source tree | `5a788bf3f9fe447fdf48acf276c226059a8bc45d921a3cce69010fdcdd8ddda3` |
| Selected `pi1` | `70c76f2eb64600eab2297640278a6c94d4336ab8e73bf941a6d96237f69f5b5c` |
| Local arm64 image | `sha256:41af66013b874447c120c04fa0cdb9ac8fd4e67868567ed23614af23966a0626` |
| Local image size | 996,549,884 bytes (950.4 MiB) |
| Lambda build context | `9a60772fa8e8a35167c4d283095460d40857769dfd98e65d87ee3554d4027df7` |
| Infrastructure tree | `a467a6335f97ecc556b52ec1fe94d43c34ee31cf835cae28ed1e1163f881e58c` |
| Frontend build | `e69836fb4a4b0ab42eec4c830ca8d91f3cf7431af2c838de75460da03cd80e1f` |
| Dependency inputs | `5b783ad8d7667e491294bb8ef74b3f968f44b9ec1cf61d7bff999ff4a05fcd05` |
| Stages 1–5 evidence | `f911d5cdad0207831eb53beed387f09b98bbb513a195b58c7f1c76ce269d5269` |

The local image ID is the exact Lambda container digest available before
publication. The ECR/OCI registry manifest digest is deliberately `null` in
the manifest and must be filled from ECR after the immutable push. A local
image ID is not represented as an ECR digest.

The repository is intentionally dirty because stages 1–6 are uncommitted. The
candidate records HEAD and the tracked diff digest, inventories every untracked
release input by file digest, and independently seals runtime source, frontend,
infrastructure, dependencies, model, image, and staging evidence. After the
later hygiene/review commit, rebuild the manifest and require runtime, model,
and dependency digests to remain unchanged; the Git identity will legitimately
become the clean release commit.

Reproduce the local candidate with:

```bash
make release-candidate LAMBDA_IMAGE=dracula-api:release-candidate
```

Two consecutive production frontend builds and candidate seals produced a
byte-identical manifest and the same candidate digest.

The exact dependency inputs are `pyproject.toml`, `frontend/package-lock.json`,
and the three pinned container requirement files under
`deployment/container/`. The release model remains ignored run data; the
context builder refuses an artifact whose digest differs from the value above.

## Staging evidence carried into release

The candidate binds the five implementation reports and their aggregate
digest. The complete local staging-equivalent pass established:

- CloudFormation lint and package tests passed.
- The arm64 image contains the exact `pi1` artifact and requires no database.
- Four complete games covered both player roles and both initial dealers.
- Cache hits, misses, zero cache, and process replacement replay exactly.
- The browser completed a six-round game and reload/narration lifecycle checks.
- Public responses and logs contained no hidden hand, stock, seed, logits,
  model tensors, or engine state.
- Local warm gameplay averaged 8.770 ms; local `pi1` inference averaged
  1.695 ms; peak container memory was 201.9 MiB.

This is not evidence of managed Lambda cold start, API Gateway latency/CORS,
CloudWatch delivery, real Bedrock behavior, or hosted rollback. Those require
fresh AWS credentials and approved production inputs.

Stage 6 reran the release surface after adding alarms and the candidate sealer:

- 46 focused Python API, narration, stateless, packaging, and privacy tests
  passed.
- 84 frontend unit/component tests, TypeScript, ESLint, and the production
  build passed.
- Two production-shaped Playwright tests passed, including a complete game.
- The exact candidate container again completed four games; warm gameplay
  averaged 8.646 ms, `pi1` inference averaged 1.720 ms, and peak memory was
  201.7 MiB.
- Both CloudFormation templates passed `cfn-lint`; parameter JSON, Markdown
  links, tracker placement, CLI parsing, and diff formatting passed.
- Two consecutive production frontend builds and manifest seals were byte
  identical.

## Release values to record before executing anything

After the circled decisions are resolved, set these in a controlled shell and
save the non-secret outputs with the release record:

```bash
export AWS_REGION=<approved-region>
export BEDROCK_MODEL_ID=<approved-model-id>
export BEDROCK_MODEL_ARN=<approved-model-arn-in-region>
export LAMBDA_MEMORY_MB=<approved-memory>
export LAMBDA_TIMEOUT_SECONDS=<approved-timeout>
export RESERVED_CONCURRENCY=<approved-concurrency>
export MONTHLY_BUDGET_USD=<approved-budget>
export NOTIFICATION_EMAIL=<approved-email>
export CLOUDFLARE_PROXY=<approved-true-or-false>
export RELEASE_COMMIT="$(git rev-parse HEAD)"
test -z "$(git status --porcelain)"
export RELEASE_TAG=<pushed-release-tag>
```

The Cloudflare token and zone ID are shell secrets, never tracked values:

```bash
test -n "$CLOUDFLARE_API_TOKEN"
test -n "$CLOUDFLARE_ZONE_ID"
```

## Exact production sequence

### 1. Build and publish the Lambda image

Authenticate and create the retained immutable ECR repository:

```bash
aws sts get-caller-identity --region "$AWS_REGION"
aws cloudformation deploy \
  --region "$AWS_REGION" \
  --template-file infrastructure/ecr.yaml \
  --stack-name dracula-api-registry \
  --parameter-overrides RepositoryName=dracula-api

export ECR_URI="$(aws cloudformation describe-stacks \
  --region "$AWS_REGION" --stack-name dracula-api-registry \
  --query 'Stacks[0].Outputs[?OutputKey==`RepositoryUri`].OutputValue' \
  --output text)"
export AWS_ACCOUNT_ID="$(aws sts get-caller-identity \
  --query Account --output text)"
aws ecr get-login-password --region "$AWS_REGION" | \
  docker login --username AWS --password-stdin \
  "$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com"

make pages-build
.venv/bin/python tools/build_lambda_context.py --output build/lambda-context
docker buildx build --platform linux/arm64 --provenance=false \
  --tag "$ECR_URI:$RELEASE_TAG" --push build/lambda-context
export IMAGE_DIGEST="$(aws ecr describe-images \
  --region "$AWS_REGION" --repository-name dracula-api \
  --image-ids imageTag="$RELEASE_TAG" \
  --query 'imageDetails[0].imageDigest' --output text)"
export IMAGE_URI="$ECR_URI@$IMAGE_DIGEST"
test "${IMAGE_DIGEST#sha256:}" != "$IMAGE_DIGEST"
```

Record `IMAGE_URI` and `IMAGE_DIGEST`. Lambda is deployed by that digest,
never by the tag.

### 2. Deploy the production API stack

Render approved parameters without modifying tracked defaults:

```bash
jq \
  --arg image "$IMAGE_URI" \
  --arg memory "$LAMBDA_MEMORY_MB" \
  --arg timeout "$LAMBDA_TIMEOUT_SECONDS" \
  --arg concurrency "$RESERVED_CONCURRENCY" \
  --arg model_id "$BEDROCK_MODEL_ID" \
  --arg model_arn "$BEDROCK_MODEL_ARN" \
  --arg budget "$MONTHLY_BUDGET_USD" \
  --arg email "$NOTIFICATION_EMAIL" '
    map(
      if .ParameterKey == "ImageUri" then .ParameterValue = $image
      elif .ParameterKey == "LambdaMemoryMb" then .ParameterValue = $memory
      elif .ParameterKey == "LambdaTimeoutSeconds" then .ParameterValue = $timeout
      elif .ParameterKey == "ReservedConcurrency" then .ParameterValue = $concurrency
      elif .ParameterKey == "NarrationEnabled" then .ParameterValue = "true"
      elif .ParameterKey == "BedrockModelId" then .ParameterValue = $model_id
      elif .ParameterKey == "BedrockModelArn" then .ParameterValue = $model_arn
      elif .ParameterKey == "MonthlyBudgetUsd" then .ParameterValue = $budget
      elif .ParameterKey == "BudgetNotificationEmail" then .ParameterValue = $email
      elif .ParameterKey == "AlarmNotificationEmail" then .ParameterValue = $email
      else . end
    )' infrastructure/parameters/production.json \
  > build/lambda-production-parameters.json

aws cloudformation deploy \
  --region "$AWS_REGION" \
  --template-file infrastructure/application.yaml \
  --stack-name dracula-api-production \
  --parameter-overrides file://build/lambda-production-parameters.json \
  --capabilities CAPABILITY_IAM \
  --no-fail-on-empty-changeset
```

This initial stack exposes the execute-api URL but does not create the custom
domain. Confirm the SNS alarm subscription email when AWS sends it.

### 3. Request and validate the ACM certificate

The certificate must be in the API's `AWS_REGION`:

```bash
export CERTIFICATE_ARN="$(aws acm request-certificate \
  --region "$AWS_REGION" \
  --domain-name api.ian-tincknell.com \
  --validation-method DNS \
  --idempotency-token draculaapi \
  --query CertificateArn --output text)"
read -r ACM_NAME ACM_TYPE ACM_VALUE <<EOF
$(aws acm describe-certificate --region "$AWS_REGION" \
  --certificate-arn "$CERTIFICATE_ARN" \
  --query 'Certificate.DomainValidationOptions[0].ResourceRecord.[Name,Type,Value]' \
  --output text)
EOF
test "$ACM_TYPE" = CNAME
```

Create the validation record in Cloudflare as DNS-only and retain its ID:

```bash
ACM_RECORD_RESPONSE="$(curl --fail-with-body --silent --show-error \
  --request POST \
  "https://api.cloudflare.com/client/v4/zones/$CLOUDFLARE_ZONE_ID/dns_records" \
  --header "Authorization: Bearer $CLOUDFLARE_API_TOKEN" \
  --header 'Content-Type: application/json' \
  --data "$(jq -n --arg name "$ACM_NAME" --arg value "$ACM_VALUE" \
    '{type:"CNAME",name:$name,content:$value,ttl:3600,proxied:false}')")"
export ACM_RECORD_ID="$(jq -er '.result.id' <<<"$ACM_RECORD_RESPONSE")"
aws acm wait certificate-validated --region "$AWS_REGION" \
  --certificate-arn "$CERTIFICATE_ARN"
```

### 4. Add the Regional API Gateway custom domain

Create a reviewed change set; execute only after final go-live authorization:

```bash
jq --arg cert "$CERTIFICATE_ARN" '
  map(
    if .ParameterKey == "CustomDomainName"
      then .ParameterValue = "api.ian-tincknell.com"
    elif .ParameterKey == "CustomDomainCertificateArn"
      then .ParameterValue = $cert
    else . end
  )' build/lambda-production-parameters.json \
  > build/lambda-production-domain-parameters.json

export CHANGE_SET="production-domain-${RELEASE_TAG}"
aws cloudformation create-change-set \
  --region "$AWS_REGION" \
  --stack-name dracula-api-production \
  --change-set-name "$CHANGE_SET" \
  --change-set-type UPDATE \
  --template-body file://infrastructure/application.yaml \
  --parameters file://build/lambda-production-domain-parameters.json \
  --capabilities CAPABILITY_IAM
aws cloudformation wait change-set-create-complete \
  --region "$AWS_REGION" --stack-name dracula-api-production \
  --change-set-name "$CHANGE_SET"
aws cloudformation describe-change-set \
  --region "$AWS_REGION" --stack-name dracula-api-production \
  --change-set-name "$CHANGE_SET" > build/production-change-set.json
jq '.Changes' build/production-change-set.json
```

After explicit authorization:

```bash
aws cloudformation execute-change-set \
  --region "$AWS_REGION" --stack-name dracula-api-production \
  --change-set-name "$CHANGE_SET"
aws cloudformation wait stack-update-complete \
  --region "$AWS_REGION" --stack-name dracula-api-production
```

No change set was created during preparation because values remain unresolved
and hosted staging has not passed.

The domain/certificate order follows AWS's current documentation for
[Regional HTTP API custom domains](https://docs.aws.amazon.com/apigateway/latest/developerguide/http-api-custom-domain-names.html)
and [ACM DNS validation](https://docs.aws.amazon.com/acm/latest/userguide/dns-validation.html).

### 5. Obtain the API Gateway target hostname

```bash
export API_TARGET="$(aws cloudformation describe-stacks \
  --region "$AWS_REGION" --stack-name dracula-api-production \
  --query 'Stacks[0].Outputs[?OutputKey==`RegionalCustomDomainTarget`].OutputValue' \
  --output text)"
test -n "$API_TARGET"
```

### 6. Add the Cloudflare API CNAME

Use the approved proxy value and retain the record ID:

```bash
API_RECORD_RESPONSE="$(curl --fail-with-body --silent --show-error \
  --request POST \
  "https://api.cloudflare.com/client/v4/zones/$CLOUDFLARE_ZONE_ID/dns_records" \
  --header "Authorization: Bearer $CLOUDFLARE_API_TOKEN" \
  --header 'Content-Type: application/json' \
  --data "$(jq -n --arg target "$API_TARGET" \
    --argjson proxied "$CLOUDFLARE_PROXY" \
    '{type:"CNAME",name:"api",content:$target,ttl:1,proxied:$proxied}')")"
export API_RECORD_ID="$(jq -er '.result.id' <<<"$API_RECORD_RESPONSE")"
```

### 7. Validate HTTPS and CORS

```bash
curl --fail --silent --show-error https://api.ian-tincknell.com/health | \
  jq -e '.status == "ok"'
curl --fail --silent --show-error --dump-header build/cors-headers.txt \
  --output /dev/null --request OPTIONS \
  --header 'Origin: https://ian-tincknell.com' \
  --header 'Access-Control-Request-Method: POST' \
  --header 'Access-Control-Request-Headers: content-type' \
  https://api.ian-tincknell.com/games
grep -i '^access-control-allow-origin: https://ian-tincknell.com' \
  build/cors-headers.txt
```

Verify an unapproved origin does not receive an allow-origin header.

### 8. Publish the GitHub Pages frontend

The workflow rebuilds the reviewed pushed tag and publishes only
`frontend/dist`:

```bash
gh workflow run pages.yml --ref "$RELEASE_TAG" -f publish=true
export PAGES_RUN_ID="$(gh run list --workflow pages.yml \
  --event workflow_dispatch --limit 1 --json databaseId \
  --jq '.[0].databaseId')"
gh run watch "$PAGES_RUN_ID" --exit-status
curl --fail --silent --show-error \
  https://ian-tincknell.com/Dracula/ >/dev/null
```

### 9. Run production smoke tests

Run the full release checklist against public URLs:

- Health and exact CORS origin.
- Complete Queen and King games through six rounds.
- Reload after opening, normal play, scoring, transition, and completion.
- Repeated command equality and malformed-history rejection.
- Legal `pi1` actions and no hidden/model state in public responses.
- Opening, rounds 1–5, and final narration, plus failure isolation.
- Desktop and mobile `/Dracula/` layout and rules link.

Before cutover, rerun the automated local release checks:

```bash
.venv/bin/python tools/validate_lambda_container.py \
  --image dracula-api:release-candidate --port 18080
npm --prefix frontend run check
make pages-test
```

The public smoke is intentionally unexecuted until the API and Pages site
exist. Record responses, CloudWatch request IDs, and Pages run ID.

### 10. Verify alarms and budget notifications

The stack creates Lambda `Errors` and API Gateway `5xx` alarms. An approved
notification email creates their SNS action; the approved budget/email creates
the 80%-forecast monthly AWS Budget notification.

```bash
aws cloudwatch describe-alarms --region "$AWS_REGION" \
  --alarm-name-prefix dracula-api-production
export ALARM_TOPIC_ARN="$(aws cloudformation describe-stacks \
  --region "$AWS_REGION" --stack-name dracula-api-production \
  --query 'Stacks[0].Outputs[?OutputKey==`OperationalAlarmTopicArn`].OutputValue' \
  --output text)"
aws sns list-subscriptions-by-topic --region "$AWS_REGION" \
  --topic-arn "$ALARM_TOPIC_ARN"
aws budgets describe-budgets --account-id "$AWS_ACCOUNT_ID" \
  --query 'Budgets[?BudgetName==`dracula-api-production-monthly`]'
```

Confirm the SNS subscription and budget notification before release completion.
The alarm resources use the documented
[`AWS::CloudWatch::Alarm`](https://docs.aws.amazon.com/AWSCloudFormation/latest/TemplateReference/aws-resource-cloudwatch-alarm.html)
contract.

### 11. Record and rehearse rollback

Record the prior immutable image, release tag, Cloudflare record IDs and prior
values, production stack ID, rendered parameters, and Pages deployment ID.

API rollback redeploys the prior immutable image:

```bash
export PRIOR_IMAGE_URI=<recorded-ecr-uri-at-digest>
jq --arg image "$PRIOR_IMAGE_URI" '
  map(if .ParameterKey == "ImageUri" then .ParameterValue = $image else . end)
' build/lambda-production-domain-parameters.json \
  > build/lambda-production-rollback-parameters.json
aws cloudformation deploy \
  --region "$AWS_REGION" \
  --template-file infrastructure/application.yaml \
  --stack-name dracula-api-production \
  --parameter-overrides file://build/lambda-production-rollback-parameters.json \
  --capabilities CAPABILITY_IAM
```

Pages rollback republishes a recorded prior release tag:

```bash
gh workflow run pages.yml --ref <prior-release-tag> -f publish=true
```

DNS rollback restores the prior record or removes the new one:

```bash
curl --fail-with-body --silent --show-error --request DELETE \
  "https://api.cloudflare.com/client/v4/zones/$CLOUDFLARE_ZONE_ID/dns_records/$API_RECORD_ID" \
  --header "Authorization: Bearer $CLOUDFLARE_API_TOKEN"
```

## Teardown

For an abandoned release, remove the public API CNAME first, then:

```bash
aws cloudformation delete-stack --region "$AWS_REGION" \
  --stack-name dracula-api-production
aws cloudformation wait stack-delete-complete --region "$AWS_REGION" \
  --stack-name dracula-api-production
aws acm delete-certificate --region "$AWS_REGION" \
  --certificate-arn "$CERTIFICATE_ARN"
curl --fail-with-body --silent --show-error --request DELETE \
  "https://api.cloudflare.com/client/v4/zones/$CLOUDFLARE_ZONE_ID/dns_records/$ACM_RECORD_ID" \
  --header "Authorization: Bearer $CLOUDFLARE_API_TOKEN"
```

The registry stack remains for rollback images. Deleting it requires a separate
explicit operation and removal of retained ECR images.

## Release checklist

### Mechanically complete

- [x] Stateless seed/history API with bounded replay cache.
- [x] Standalone embedded `pi1`; exact artifact digest enforced.
- [x] Three narration cue classes with failure isolation.
- [x] Reproducible arm64 Lambda Web Adapter image.
- [x] Regional HTTP API, CORS, least-privilege Bedrock/logging role.
- [x] Lambda error and API 5xx alarms; optional SNS action.
- [x] Optional forecast budget resource.
- [x] `/Dracula/` production frontend and manual Pages workflow.
- [x] Local staging-equivalent engine, API, browser, privacy, and rollback
  validation.
- [x] Reproducible candidate manifest and cutover/teardown commands.
- [ ] Clean reviewed release commit and tag (later hygiene/review stage).
- [ ] Hosted staging and managed Lambda/API Gateway/Bedrock measurements.
- [ ] Immutable ECR registry digest recorded.
- [ ] ACM certificate and Regional custom domain validated.
- [ ] Public API and Pages production smoke completed.
- [ ] Alarm recipients and budget notifications confirmed.

### User decisions required before cutover

⭕ USER DECISION LATER: Final Dracula narration style prompt and examples.

⭕ USER DECISION LATER: Exact Bedrock model ID.

⭕ USER DECISION LATER: AWS region.

⭕ USER DECISION LATER: Lambda memory, timeout, reserved concurrency, and budget alarms.

⭕ USER DECISION LATER: Cloudflare DNS proxy mode and final DNS cutover.

⭕ USER DECISION LATER: Final production go-live authorization.

⭕ USER DECISION LATER: Public disclosure, if any, about inspectable seed/history state.

## Release status

**Mechanical release-candidate preparation is complete. Production is not
ready to execute.** Outstanding work is hosted validation and the seven user
decisions, not an unimplemented application path. Because those values are
unresolved and hosted staging did not pass, no CloudFormation change-set
preview was created.

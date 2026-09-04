# Staging deployment validation

## Result

The production release was validated through the complete local
staging-equivalent path. No AWS resource was created because the configured AWS
session is expired:

```text
ExpiredToken: The security token included in the request is expired
```

There is no alternate profile, explicit staging account, selected AWS region,
or final Bedrock model identity in the repository or environment. Guessing any
of those values would make the deployment ambiguous. Resource identifiers,
hosted log groups, and an AWS teardown command therefore do not exist yet.

No production DNS, GitHub Pages publication, database, commit, or push was
performed.

## Exact release package

The generated arm64 container embedded the selected `pi1` artifact with
SHA-256:

```text
70c76f2eb64600eab2297640278a6c94d4336ab8e73bf941a6d96237f69f5b5c
```

The release-context manifest contains 79 digest-bound files and Git revision
`81b2504d4cc679e40e36a1fd63160037e6dfbb0d`. The built image is
`dracula-api:staging-local`, architecture `arm64`, image ID
`sha256:2eeeba90ac85d7c58331eec91512f6508316698e179efb3223959f355352eddc`,
and size 996,554,909 bytes (950.4 MiB).

The root filesystem rejected writes, `/tmp` accepted its bounded probe, the
model digest matched inside the image, `DRACULA_DATABASE_PATH` was absent, and
neither `/var/task` nor `/opt/dracula` contained a SQLite database.

## Infrastructure validation and cost controls

Both CloudFormation templates pass `cfn-lint 1.56.0`. Online
`aws cloudformation validate-template` reaches AWS but fails with the same
expired token before validation; it did not create or update anything.

The staging parameters retain:

- 2,048 MB Lambda memory.
- 20-second Lambda timeout.
- Reserved concurrency of 2.
- API throttling at 5 requests/second with a burst of 10.
- Fourteen-day Lambda and API access-log retention.
- Narration and the custom domain disabled.
- No database, VPC, EFS, SageMaker, or DynamoDB resource.

One cost defect was repaired during this pass. Route-level API Gateway detailed
metrics had been enabled even though AWS charges separately for them. They are
now disabled; free API/stage metrics, structured Lambda logs, and structured
API access logs remain. AWS documents that route-level detailed HTTP API
metrics incur additional charges in its
[HTTP API metrics guide](https://docs.aws.amazon.com/apigateway/latest/developerguide/http-api-metrics.html).

The monthly budget remains disabled because neither its amount nor its
notification recipient has been approved. Reserved concurrency, throttling,
timeouts, and log retention remain active cost bounds.

## Container and API results

The measured validator ran four complete six-round games: Queen and King with
both Queen and King as initial dealer. Each game completed with 31 accepted
commands, for 124 gameplay requests total. All opponent actions were legal and
all final games contained six scored rounds.

The same run established:

- Identical responses for repeated seed/history/command requests.
- Rejection of a duplicated command embedded in history with HTTP 422 and
  `invalid_history`.
- Exact response equality on cache hits, cache misses, zero-sized cache, and
  process replacement.
- Exact cache-disabled reconstruction at every history length from 1 through
  31.
- No model digest, logits, tensors, masks, search data, hidden hand, stock,
  engine object, or seed in public responses or application logs.
- Narration-disabled opening requests returned the non-fabricated unavailable
  response and did not alter gameplay.
- Every one of 648 captured container log lines was one JSON object.

A valid older envelope is not a mutable-session conflict in this architecture:
it deterministically reconstructs its own branch. The browser retains only the
latest accepted envelope. There is also no client-visible `opponent_turn`
lifecycle state—the stateless command synchronously computes Dracula's reply
before returning—so reload during an in-progress opponent mutation is not an
observable browser operation. Retry before or after that response is covered
by the identical-request and process-replacement tests.

## Browser, narration, CORS, and privacy

The production-shaped Playwright suite passed both tests in 16.5 seconds. It
completed a real six-round `pi1` game and covered:

- Opening, human-turn, scoring/round-transition, and final-result reload.
- Opening narration.
- Rounds 1–5 narration requested during scoring and revealed afterward.
- No round-six transition cue and one final-result cue.
- Narration-unavailable play without mutation or interruption.
- Desktop 1440×900 and mobile 390×844/390×640 geometry.
- No move-triggered scrolling.
- Rules opened under `/Dracula/#/rules`.
- Browser storage containing exactly `seed` and accepted `history`.

The fake narration provider completed the exact seven-cue game cadence in a
mean 0.471 ms per API request, maximum 0.708 ms. This measures only local
grounding, replay, and response handling, not Bedrock network or model latency.
The focused suite separately passed configured timeout, provider failure,
malformed output, disabled narration, retry isolation, and public-grounding
tests. A real Bedrock latency or cost figure is unavailable until the user
selects a model.

API Gateway owns production CORS. The validated template permits only
`https://ian-tincknell.com`, methods GET/POST, and the `content-type` header.
The direct local container deliberately does not emulate managed API Gateway
CORS response generation.

## Measurements

These are local arm64 Docker measurements, not managed Lambda or API Gateway
measurements.

| Measurement | Result |
| --- | ---: |
| Container initialization, clean starts | 1.883 s; 1.608 s |
| HTTP readiness, clean starts | 1.448 s; 1.294 s |
| Warm gameplay request mean / p50 / p95 | 8.770 / 9.431 / 12.788 ms |
| Warm replay mean / p95 | 3.531 / 4.524 ms |
| Cache-disabled replay, history 1 | 13.309 ms |
| Cache-disabled replay, history 8 | 30.138 ms |
| Cache-disabled replay, history 15 | 66.319 ms |
| Cache-disabled replay, history 22 | 103.129 ms |
| Cache-disabled replay, history 31 | 155.531 ms |
| `pi1` inference mean / p95 | 1.695 / 2.242 ms |
| Peak container memory | 211,707,494 bytes (201.9 MiB) |
| Primary-run application logs | 56,104 bytes / 252 lines |
| Conservative application-log upper bound | 14,026 bytes per game |
| One measured game request + response payload | 454,197 bytes |
| Image | 996,554,909 bytes (950.4 MiB) |

The complete raw measurement and logs are ignored local artifacts:

```text
build/staging-lambda-validation.json
build/staging-lambda-validation.log
build/staging-cloudformation-validate.err
```

## Illustrative cost

Because no region is selected, this is an explicit US East pricing illustration,
not a deployment quote. It excludes free-tier credits and Bedrock:

- Lambda arm duration: approximately $0.0000073 per warm game using 31 local
  requests, 8.770 ms measured mean, 2 GB memory, and $0.0000133334/GB-second.
- Lambda requests: approximately $0.0000062 per game at $0.20/million.
- HTTP API requests: approximately $0.000031 per game at $1.00/million.
- CloudWatch ingestion: conservative upper estimate $0.000010 per game from
  measured application logs plus estimated API access logs at $0.50/GB.

The sum is approximately **$0.000055 per warm game**, or **0.0055 cents**,
before Bedrock and ordinary data-transfer variation. One fully cold local
initialization adds roughly $0.00005 of Lambda compute at the same 2 GB rate,
bringing the illustrative cold-game total near $0.00010. Low-volume actual
charges may be lower because this calculation ignores free tiers.

The 950.4 MiB private ECR image is approximately $0.095/month at $0.10/GB-month;
same-region ECR-to-Lambda transfer is free. Sources are the current official
[Lambda pricing](https://aws.amazon.com/lambda/pricing/),
[API Gateway pricing](https://aws.amazon.com/api-gateway/pricing/),
[CloudWatch pricing](https://aws.amazon.com/cloudwatch/pricing/), and
[ECR pricing](https://aws.amazon.com/ecr/pricing/). A seven-cue Bedrock amount
cannot be added until the exact model is selected and measured.

## Commands executed

```bash
aws configure list
aws sts get-caller-identity --output json
aws cloudformation validate-template \
  --template-body file://infrastructure/application.yaml
.venv/bin/python tools/build_lambda_context.py --output build/lambda-context
.venv/bin/cfn-lint infrastructure/ecr.yaml infrastructure/application.yaml
docker build --platform linux/arm64 \
  --tag dracula-api:staging-local build/lambda-context
.venv/bin/python tools/validate_lambda_container.py \
  --image dracula-api:staging-local --port 18081 \
  --output build/staging-lambda-validation.json
.venv/bin/python -m pytest -q \
  tests/test_lambda_packaging.py \
  tests/test_stateless_gameplay_api.py \
  tests/test_bedrock_narration.py tests/test_api.py
npm --prefix frontend run check
make pages-test
```

Results:

- Focused Python: 46 passed.
- Frontend unit/component: 12 files, 84 passed.
- TypeScript, ESLint, and production `/Dracula/` build verification: passed.
- Production-shaped browser: 2 passed.
- CloudFormation lint: zero findings.
- Diff formatting: passed.

## Rollback and teardown

No hosted rollback is possible or required because no resource was deployed.
Local validation can be removed with:

```bash
docker rm --force dracula-lambda-validation 2>/dev/null || true
docker image rm dracula-api:staging-local
rm -rf build/lambda-context \
  build/staging-lambda-validation.json \
  build/staging-lambda-validation.log \
  build/staging-cloudformation-validate.json \
  build/staging-cloudformation-validate.err
```

After valid credentials and an explicit region exist, the exact staging deploy,
immutable-image rollback, and stack deletion commands remain in
[`docs/deployment.md`](../../docs/deployment.md). Narration remains disabled
unless a model ID and matching ARN are both explicitly supplied.

## Remaining external staging evidence

The implementation is locally staging-ready. The following evidence cannot be
produced without external configuration:

- ECR resource and immutable pushed-image digest.
- CloudFormation stack, Lambda ARN, API ID, and staging URL.
- Managed Lambda cold-start, API Gateway overhead, CORS response, and CloudWatch
  ingestion measurements.
- Real Bedrock latency, output, token use, and per-game cost.
- Hosted rollback and teardown execution.

The concrete blocker is the expired AWS token plus absent region/model
selections—not an application or template failure.

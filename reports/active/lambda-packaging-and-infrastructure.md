# Lambda packaging and infrastructure

## Result

The production API is packaged as one digest-bound arm64 container and the AWS
resources are defined in plain CloudFormation. Local validation is complete.
No AWS resource, DNS record, or frontend deployment was changed during this
stage.

The selected model remains local ignored evidence at
`runs/bgc-policy-pi1-001/artifacts/unaccepted-candidate.pt`. The release builder
verified and embedded only SHA-256
`70c76f2eb64600eab2297640278a6c94d4336ab8e73bf941a6d96237f69f5b5c`.
The generated `build/lambda-context/` is ignored and contains no other run,
database, test, frontend, or historical artifact.

## AWS basis checked

- AWS requires Linux, a single target architecture, a read-only-compatible
  root, and permits only `/tmp` for writes; container images may be up to 10 GB:
  [Lambda container images](https://docs.aws.amazon.com/lambda/latest/dg/images-create.html)
  and [Lambda quotas](https://docs.aws.amazon.com/lambda/latest/dg/gettingstarted-limits.html).
- The AWS Lambda Web Adapter project documents OCI-image use, local execution,
  HTTP API support, the readiness settings used here, and the pinned 1.0.1
  adapter copy:
  [AWS Lambda Web Adapter](https://github.com/aws/aws-lambda-web-adapter).
- API Gateway HTTP API CORS generates preflight responses and applies the
  configured CORS headers:
  [HTTP API CORS](https://docs.aws.amazon.com/apigateway/latest/developerguide/http-api-cors.html).
- A Regional API custom domain requires an ACM certificate in the same region
  and a DNS record targeting the returned Regional hostname:
  [Regional custom domains](https://docs.aws.amazon.com/apigateway/latest/developerguide/apigateway-regional-api-custom-domain-create.html).
- Bedrock Runtime `Converse` requires `bedrock:InvokeModel`; streaming
  permission is unnecessary for this adapter:
  [Bedrock Converse](https://docs.aws.amazon.com/bedrock/latest/userguide/conversation-inference.html).
- CloudFormation supports image Lambdas, reserved concurrency, and structured
  logging configuration:
  [AWS::Lambda::Function](https://docs.aws.amazon.com/AWSCloudFormation/latest/TemplateReference/aws-resource-lambda-function.html).

## Package

The tracked package inputs are under `deployment/container/`:

- Digest-pinned Python 3.12.9 slim base.
- Digest-pinned Lambda Web Adapter 1.0.1.
- CPU-only PyTorch 2.13.0 and fully constrained runtime dependencies.
- `dracula.api.production:app` with stateless gameplay, standalone `pi1`,
  optional Bedrock Runtime, and bounded replay cache.
- Single-line JSON Uvicorn and application logs.

`tools/build_lambda_context.py` verifies the model before copying, records a
content manifest and Git revision, and atomically replaces only ignored build
output. A wrong or absent model fails closed. `tools/validate_lambda_container.py`
runs the built image with a read-only root and a 512 MB `/tmp` tmpfs.

## Infrastructure

`infrastructure/ecr.yaml` creates one immutable scan-on-push ECR repository.
`infrastructure/application.yaml` creates:

- One arm64 image Lambda with parameterized memory, timeout, reserved
  concurrency, `/tmp`, replay cache, and log retention.
- A conditional model-ARN-scoped `bedrock:InvokeModel` policy and exact
  application-log permissions; no SageMaker or database permissions.
- One Regional API Gateway HTTP API using payload format 2.0.
- Exact health, game, command, resume, and narration routes.
- Explicit single-origin CORS and bounded API throttling.
- Structured Lambda and API access log groups.
- Lambda-error and API-5xx alarms, with an optional email notification topic
  added during release-candidate preparation.
- An optional same-region ACM Regional custom domain and mapping.
- An optional monthly AWS Budget with an 80% forecast notification.

Staging and production parameter files contain no account ID, credential,
secret, absolute workstation path, final model ID, certificate ARN, or DNS
target. Narration, the custom domain, and the monthly budget remain disabled
until their external inputs are approved.

CloudFormation lint completed with zero findings for both templates. An online
`aws cloudformation validate-template` could not run because the installed AWS
session returns `ExpiredToken`; no deployment was attempted.

## Local measurements

Measurements came from the final `dracula-api:local` image on the arm64 M3
MacBook Air. They measure local process/container behavior, not ECR transfer or
an AWS Lambda cold start.

| Measurement | Result |
| --- | ---: |
| Generated context | 4.9 MB |
| Image size | 996,554,909 bytes (950.4 MiB) |
| Process initialization, two clean starts | 2.260 s; 1.463 s |
| Web readiness, two clean starts | 1.400 s; 1.171 s |
| Warm health mean / p95 | 2.663 ms / 3.967 ms |
| Warm cache replay mean / p95 | 4.668 ms / 8.595 ms |
| Cold-process complete-game replay | 165.647 ms |
| `pi1` inference mean / p95 | 3.063 ms / 7.783 ms |
| Peak container memory | 209,820,058 bytes (200.1 MiB) |

The validator completed a six-round game in 31 client requests, then reproduced
the final response after a process restart. Combined original and cold replay
produced 42 deterministic `pi1` inference log entries. The embedded model hash
matched, the root filesystem rejected writes, `/tmp` accepted its probe, all
114 sampled application/access log lines were JSON, and responses exposed no
model digest, logits, tensors, masks, search state, hidden hand, or stock.

The 950.4 MiB image is safely below Lambda's image limit but is mostly CPU
PyTorch. Its actual ECR pull and managed Lambda initialization time remain a
staging measurement.

## Verification

- Full Python suite: **677 passed** in 643.33 seconds.
- Focused stateless, narration, and packaging suite: **39 passed**.
- Frontend: **10 files / 74 tests passed**, TypeScript passed, ESLint passed,
  production build passed, and the build contains no local API URL.
- Complete read-only container game, cache equivalence, cold replay, model
  load, and response privacy: passed.
- Bedrock fake adapter, cue eligibility, failure isolation, and privacy: passed
  in the focused/full Python suites.
- `cfn-lint` over both CloudFormation templates: passed with zero findings.
- `git diff --check`, Make command expansion, package imports, pinned runtime
  version inspection, and model SHA-256: passed.

Ignored raw measurements are in `build/lambda-validation.json` and
`build/lambda-validation.log`.

## Commands

Exact build, local-run, validate, package/push, deploy-staging, update,
immutable-image rollback, and delete commands are maintained in
[`docs/deployment.md`](../../docs/deployment.md). The short local surface is:

```bash
make lambda-build
make lambda-validate LAMBDA_PORT=18080
make lambda-run
```

## External conditions and later decisions

Staging was not deployed because `aws sts get-caller-identity` returns an
expired-token error and no final region/staging Bedrock configuration exists.
That is the only reason AWS-hosted latency and Bedrock cost are absent.

⭕ USER DECISION LATER: Approve the final AWS region after reviewing Bedrock availability, latency, and domain-certificate constraints.

⭕ USER DECISION LATER: Approve production Lambda memory, reserved concurrency, timeout, and monthly budget-alarm amounts after staging measurements.

The previously registered decisions also remain:

- Approve Dracula's final narration voice and examples.
- Select the exact Bedrock model ID and approve measured per-game cost.

# AWS infrastructure

CloudFormation defines the combined frontend/API infrastructure on AWS.

- `ecr.yaml` creates the shared immutable image repository.
- `application.yaml` creates one stateless Lambda container, its least-privilege
  role, one Regional HTTP API forwarding `/Dracula` and its children, structured
  log groups, Lambda/API error alarms, and optional notifications and budget.
  The browser and API share one origin; there is no CORS or custom-domain setup.
- `parameters/staging.json` and `parameters/production.json` hold non-secret
  environment defaults. The immutable image URI is appended in an ignored
  generated parameter file at deployment time.

The deployment uses the user's personal AWS account in `us-east-1`. The
environment parameter files record the selected Amazon Nova Lite model
(`amazon.nova-lite-v1:0`) but contain no account credentials, model ARNs, ACM
certificates, or DNS records. The release process supplies the model ARN and
immutable image URI explicitly. Low-quota staging may use concurrency `-1`
(the existing account pool); production requires a per-function cap. The optional monthly
AWS Budget remains disabled until its amount and notification address are
approved after hosted staging. See
[deployment](../docs/deployment.md) for the complete command order.

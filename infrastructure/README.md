# AWS infrastructure

CloudFormation is the only infrastructure mechanism for the production API.

- `ecr.yaml` creates the shared immutable image repository.
- `application.yaml` creates one stateless Lambda container, its least-privilege
  role, one Regional HTTP API, explicit routes and CORS, structured log groups,
  Lambda/API error alarms, an optional notification topic and budget, and an
  optional Regional custom domain mapping.
- `parameters/staging.json` and `parameters/production.json` hold non-secret
  environment defaults. The immutable image URI is appended in an ignored
  generated parameter file at deployment time.

The deployment uses the user's personal AWS account in `us-east-1` and Amazon
Nova Lite (`amazon.nova-lite-v1:0`). The generic parameter files do not contain
account credentials, model ARNs, ACM certificates, or DNS records; the release
process renders those account-bound values outside Git. The optional monthly
AWS Budget remains disabled until its amount and notification address are
approved after hosted staging. See
[deployment](../docs/deployment.md) for the complete command order.

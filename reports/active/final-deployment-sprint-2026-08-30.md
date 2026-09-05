# Final deployment sprint

Date: 2026-08-30

## Locked decisions

- The vetted React experience is the release UI.
- Standalone `pi1` is the production opponent.
- The frontend is published at `https://ian-tincknell.com/Dracula/` through the
  existing GitHub Pages site.
- Cloudflare continues to manage DNS.
- `api.ian-tincknell.com` maps through a Cloudflare CNAME to a Regional API
  Gateway custom domain.
- API Gateway invokes a Lambda container running the Lambda Web Adapter,
  FastAPI, deterministic engine, and embedded `pi1` artifact.
- There is no SageMaker or production database.
- The browser and Lambda exchange a plain initial seed and ordered accepted
  command history. Lambda replays on a cache miss and caches reconstructed
  states opportunistically in local memory.
- The envelope has no redundant lifecycle, version, signature, or encryption.
- Bedrock runs only for opening, transitions after rounds 1–5, and the final
  game result. Round 6 has no separate transition response.

## Selected model evidence

The selected bytes are currently stored locally at:

```text
runs/bgc-policy-pi1-001/artifacts/pi1-policy.pt
```

SHA-256:

```text
d35196cf4513589def0ffb3c4c7c268e78652a46dab8ea41001ff2c648265203
```

The D1 training snapshot contained 806,610 training rows and 89,670 validation
rows. Epoch 12 supplied the lowest validation cross-entropy, `1.99895417`.
Validation agreement with the most-visited D1 action was `0.4879` at top one,
`0.7139` at top two, and `0.8179` at top three. The model has 754,601 trainable
parameters and performed no illegal action in the retained fixed matchup data.

The deployment artifact contains the selected state dictionary in the minimal
runtime format. Release packaging preserves and verifies those exact bytes.

## Remaining implementation phases

1. Stateless service: add bounded seed/history parsing, deterministic replay,
   command application, and cache equivalence tests.
2. Frontend state: persist the last confirmed envelope, recover on reload, and
   use the stateless command route.
3. Narration: implement the three Bedrock cue classes and scoring-animation
   synchronization.
4. Runtime package: produce a minimal Lambda container containing only the web
   runtime, engine, `pi1`, and narration integration.
5. Infrastructure: define ECR, Lambda, API Gateway, ACM, IAM, CloudWatch,
   throttling, and budget alarms.
6. Pages release: set the Vite `/Dracula/` base, production API origin, assets,
   and direct-reload behavior.
7. Domain: add the Cloudflare DNS-only `api` CNAME after AWS creates the custom
   domain target.
8. Staging: validate cold/warm inference, forced cache misses, both roles, six
   rounds, reloads, narration timing/failure, CORS, logs, rollback, and teardown.
9. Production: deploy the same image and static build, run the public smoke,
   and retain the prior image through the rollback window.

## Superseded assumptions

The prior deployment audit left the opponent, persistence, narration, and
hosting undecided. Those questions are now closed. Nested Sam remains a local
comparison, SQLite remains local-only, and historical training/miner plans are
not release dependencies.

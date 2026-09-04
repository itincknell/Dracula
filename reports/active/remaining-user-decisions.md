# Remaining user decisions

Date: 2026-09-02

The product owner has selected the personal AWS account, `us-east-1`, Amazon
Nova Lite (`amazon.nova-lite-v1:0`), the gloating cartoon-villain narration
voice, `/Dracula/`, a proxied Cloudflare API CNAME, and no public disclosure of
the accepted seed/history inspection tradeoff. Those choices are closed.

Refreshing the personal-account AWS CLI session is an external operating step,
not a design decision. It is required before hosted staging can begin.

## Production resource and notification settings

⭕ USER DECISION LATER

- **Question:** After hosted staging measurements, what production Lambda
  memory, timeout, reserved concurrency, notification recipients, and monthly
  budget are approved?
- **Why measurements cannot decide it:** Hosted measurements establish cost
  and latency, but the acceptable spend, throttling, and notification ownership
  belong to the user.
- **Current staging values:** 2,048 MB, 20 seconds, concurrency 2, a 256-entry
  replay cache, budget disabled, and no notification recipient.
- **Choices and effects:** Retain the measured staging values or adjust them
  after reviewing cold-start, warm-request, Bedrock, and concurrency evidence.
  Enabling notifications requires user-owned email addresses.
- **Changed after approval:**
  `infrastructure/parameters/production.json` and the ignored rendered
  deployment parameters.
- **Blocks:** Production cutover, not hosted staging.

## Production go-live authorization

⭕ USER DECISION LATER

- **Question:** After hosted staging passes, authorize the production AWS
  stack, Regional custom domain, proxied Cloudflare record, and GitHub Pages
  publication?
- **Why code cannot decide it:** These steps create billable infrastructure and
  modify public AWS, DNS, and Pages state.
- **Choices and effects:** Authorize the scripted cutover, request another
  staging rehearsal, or defer. Only authorization publishes the application.
- **Current state:** No hosted staging or production cutover has occurred.
- **Changed after approval:** External AWS, Cloudflare, and GitHub Pages state;
  ignored release evidence; deployment status documentation.
- **Blocks:** Production only.

## Review order

1. Refresh the personal-account AWS session and deploy staging in `us-east-1`.
2. Exercise Amazon Nova Lite narration and collect managed latency, memory,
   concurrency, and cost evidence.
3. Approve production resource and notification settings.
4. Review the immutable release record and explicitly authorize or defer
   production go-live.

# Dracula path forwarding

This Worker serves the live game at <https://ian-tincknell.com/Dracula/>.
It forwards only `/Dracula`
and `/Dracula/…` to the combined application's generated API Gateway endpoint.
The existing personal website remains the origin for all other paths.

`API_ORIGIN` selects the deployed production stack:
`https://3ylzpjexng.execute-api.us-east-1.amazonaws.com`.

When changing the AWS origin, set `API_ORIGIN` in `wrangler.toml` to the production
stack's `DefaultApiUrl` output. It must be the HTTPS `execute-api.us-east-1`
hostname, without an application path. Never point it at the public website.

Validate locally:

```bash
node --test deployment/cloudflare/worker.test.mjs
```

Deploy the checked-in Worker and its narrow route after Wrangler login:

```bash
npx wrangler deploy --config deployment/cloudflare/wrangler.toml
```

The route is `ian-tincknell.com/Dracula*`. The four existing GitHub Pages A
records are proxied; their target addresses were not changed.
The code guard leaves similarly named paths such as `/Dracula-other` untouched.

The checked-in Wrangler configuration disables workers.dev. Deployment is
explicit; there is no automatic publication workflow.

Forwarding preserves method, query, body, and response, including relative
redirects. Successful public GET/HEAD responses are cached at the edge: one
year for content-hashed assets, one hour for HTML and named artwork. API reads,
all writes, redirects, and errors bypass caching. The Worker explicitly restores
`Cache-Control: no-cache` for HTML/artwork after the edge fetch, overriding the
zone's browser TTL. Browsers revalidate with Cloudflare, which can answer without
calling Lambda. A cache miss can still cold-start Lambda on page load.

## Release and rollback cache invalidation

After deploying and checking the new AWS image, purge the Dracula files before
announcing the release. Do the same after restoring an older image. Use a token
with **Cache Purge** permission for the personal website's zone; keep it in the
shell environment, not the Worker or repository. Set `API_ORIGIN` to the same
AWS origin configured on the Worker, and `CLOUDFLARE_ZONE_ID` to the website zone.

```bash
: "${API_ORIGIN:?Set API_ORIGIN to the Worker's HTTPS AWS origin}"
curl --fail-with-body --request POST \
  "https://api.cloudflare.com/client/v4/zones/${CLOUDFLARE_ZONE_ID:?}/purge_cache" \
  --header "Authorization: Bearer ${CLOUDFLARE_API_TOKEN:?}" \
  --json "{\"prefixes\":[\"${API_ORIGIN#https://}/Dracula/\"]}"
```

Confirm the response contains `"success": true`; a failed purge is not a
completed release. Then reload the public `/Dracula/` page and check the game.
The prefix deliberately uses the **AWS subrequest hostname**, not the browser
hostname: that is the URL `fetch()` caches. It includes old/deleted assets and
query variants without purging the personal website. If changing `API_ORIGIN`,
purge both the old and new origin prefixes before completing the switch.

Cloudflare documents [subrequest cache identity](https://developers.cloudflare.com/workers/reference/how-the-cache-works/)
and [prefix purging](https://developers.cloudflare.com/cache/how-to/purge-cache/purge_by_prefix/),
which is available on the free plan. Live HTML, artwork, and hashed-asset cache
hits were verified at cutover; API health remained uncached. Wrangler's current
OAuth login does not grant Cache Purge permission. For future releases, use the
Cloudflare dashboard's scoped purge or a Cache Purge token before announcing
completion. No purge was needed for the first publication of these assets.

Rollback: disable the route to restore the original site's handling, or restore
the previous Worker version and `API_ORIGIN`. Do not change unrelated DNS or
cache rules. See [deployment](../../docs/deployment.md) for the AWS sequence.

import assert from "node:assert/strict";
import test from "node:test";
import worker from "./worker.mjs";

const env = { API_ORIGIN: "https://example123.execute-api.us-east-1.amazonaws.com" };

test("forwards game bodies and query strings without caching or following redirects", async (t) => {
  let forwarded;
  t.mock.method(globalThis, "fetch", async (request, options) => {
    forwarded = { url: request.url, method: request.method, body: await request.text(), options };
    return new Response('{"ok":true}', { headers: { "Cache-Control": "no-store" } });
  });
  const response = await worker.fetch(new Request("https://ian-tincknell.com/Dracula/api/games?test=1", {
    method: "POST", body: '{"human_role":"queen"}', headers: { "Content-Type": "application/json" },
  }), env);
  assert.equal(forwarded.url, `${env.API_ORIGIN}/Dracula/api/games?test=1`);
  assert.equal(forwarded.method, "POST");
  assert.equal(forwarded.body, '{"human_role":"queen"}');
  assert.equal(forwarded.options.redirect, "manual");
  assert.deepEqual(forwarded.options.cf.cacheTtlByStatus, { "100-599": -1 });
  assert.equal(response.headers.get("Cache-Control"), "no-store");
});

test("caches successful public files but excludes API reads and non-read methods", async (t) => {
  const options = [];
  t.mock.method(globalThis, "fetch", async (_request, init) => { options.push(init); return new Response("ok"); });
  for (const path of ["/Dracula/assets/index-AbCd1234.js", "/Dracula/", "/Dracula/cards/2C.png", "/Dracula/api/health", "/Dracula/%61pi/health"]) {
    await worker.fetch(new Request(`https://ian-tincknell.com${path}`), env);
  }
  assert.deepEqual(options[0].cf.cacheTtlByStatus, { "200-299": 31536000, "300-599": -1 });
  for (const option of options.slice(1, 3)) {
    assert.equal(option.cf.cacheEverything, true);
    assert.deepEqual(option.cf.cacheTtlByStatus, { "200-299": 3600, "300-599": -1 });
  }
  for (const option of options.slice(3)) {
    assert.equal(option.cf.cacheEverything, false);
    assert.deepEqual(option.cf.cacheTtlByStatus, { "100-599": -1 });
  }
  await worker.fetch(new Request("https://ian-tincknell.com/Dracula/", { method: "POST" }), env);
  assert.deepEqual(options.at(-1).cf.cacheTtlByStatus, { "100-599": -1 });
  await worker.fetch(new Request("https://ian-tincknell.com/Dracula/", { method: "HEAD" }), env);
  assert.deepEqual(options.at(-1).cf.cacheTtlByStatus, { "200-299": 3600, "300-599": -1 });
});

test("unrelated website paths pass through untouched", async (t) => {
  t.mock.method(globalThis, "fetch", async (request, options) => {
    assert.equal(new URL(request.url).hostname, "ian-tincknell.com");
    assert.equal(options, undefined);
    return new Response("website");
  });
  for (const path of ["/", "/projects/", "/Dracula-other"]) {
    assert.equal(await (await worker.fetch(new Request(`https://ian-tincknell.com${path}`), env)).text(), "website");
  }
});

test("HTML revalidates in browsers even when the edge supplies a browser TTL", async (t) => {
  t.mock.method(globalThis, "fetch", async () => new Response("file", {
    headers: { "Cache-Control": "max-age=14400", "CF-Cache-Status": "HIT" },
  }));
  for (const path of ["/Dracula/", "/Dracula/cards/2C.png"]) {
    const response = await worker.fetch(new Request(`https://ian-tincknell.com${path}`), env);
    assert.equal(response.headers.get("Cache-Control"), "no-cache");
    assert.equal(response.headers.get("CF-Cache-Status"), "HIT");
    assert.equal(await response.text(), "file");
  }
});

test("redirect stays relative and cannot silently change origin", async (t) => {
  t.mock.method(globalThis, "fetch", async () => new Response(null, { status: 308, headers: { Location: "/Dracula/" } }));
  const response = await worker.fetch(new Request("https://ian-tincknell.com/Dracula"), env);
  assert.equal(response.headers.get("Location"), "/Dracula/");
  for (const origin of ["https://ian-tincknell.com", "http://example123.execute-api.us-east-1.amazonaws.com", `${env.API_ORIGIN}/wrong`]) {
    await assert.rejects(worker.fetch(new Request("https://ian-tincknell.com/Dracula/"), { API_ORIGIN: origin }));
  }
});

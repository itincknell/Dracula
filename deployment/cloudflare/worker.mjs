/**
 * Forward only the Dracula application to its fixed AWS origin.
 * The browser stays on the personal website's hostname. Other website paths
 * continue to GitHub Pages. Public files are cached; gameplay never is.
 */
export default {
  async fetch(request, env) {
    const incoming = new URL(request.url);
    if (incoming.pathname !== "/Dracula" && !incoming.pathname.startsWith("/Dracula/")) {
      return fetch(request);
    }
    const origin = new URL(env.API_ORIGIN);
    if (origin.protocol !== "https:" || !/^[a-z0-9]+\.execute-api\.us-east-1\.amazonaws\.com$/.test(origin.hostname)
        || origin.pathname !== "/" || origin.search || origin.hash || origin.username || origin.password || origin.port) {
      throw new Error("API_ORIGIN must be the generated us-east-1 API Gateway HTTPS origin");
    }
    const target = new URL(incoming.pathname + incoming.search, origin);
    // Copy the method, body, and headers. Do not send the personal-site Host
    // header to AWS, or follow an origin redirect away from the public URL.
    const outgoing = new Request(target, request);
    outgoing.headers.delete("Host");
    // Decode before excluding API paths, so encoded spellings cannot put an
    // API response in the public-file cache. Only reads may use that cache.
    const publicFile = ["GET", "HEAD"].includes(request.method)
      && !decodeURIComponent(incoming.pathname).startsWith("/Dracula/api");
    const hashedAsset = /^\/Dracula\/assets\/[^/]+-[A-Za-z0-9_-]{8,}\.[^/]+$/.test(incoming.pathname);
    const response = await fetch(outgoing, {
      redirect: "manual",
      cf: {
        cacheEverything: publicFile,
        // These edge TTLs override browser revalidation headers. Releases and
        // rollbacks purge /Dracula/ at the AWS subrequest hostname; the hour
        // limit also bounds stale HTML/artwork if that purge is missed.
        // Redirects and errors stay uncached, including a missing old asset.
        cacheTtlByStatus: publicFile
          ? { "200-299": hashedAsset ? 31536000 : 3600, "300-599": -1 }
          : { "100-599": -1 },
      },
    });
    // Cloudflare's zone browser TTL can replace the origin's no-cache header.
    // Keep HTML and named artwork revalidating in the browser while the fetch
    // above continues to use the edge cache. Hashed assets remain immutable.
    if (publicFile && !hashedAsset && response.ok) {
      const browserResponse = new Response(response.body, response);
      browserResponse.headers.set("Cache-Control", "no-cache");
      return browserResponse;
    }
    return response;
  },
};

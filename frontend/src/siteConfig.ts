/**
 * Resolve the four links displayed in the application shell.
 *
 * Vite supplies `BASE_URL` as `/` during local development and `/Dracula/` in
 * the production build. Home and Rules belong to this application and must
 * remain under that base path. About and Contact belong to the surrounding
 * personal site, so their root-relative paths deliberately do not include the
 * application base.
 */
export interface SiteLinks {
  home: string;
  rules: string;
  about: string;
  contact: string;
}

/** Build navigation links from Vite's application base path. */
export function resolveSiteLinks(baseUrl?: string): SiteLinks {
  const base = baseUrl?.trim() || "/";
  // A trailing slash lets both `/` and `/Dracula/` accept the same appended
  // hash route without producing malformed paths.
  const normalizedBase = base.endsWith("/") ? base : `${base}/`;
  return {
    home: normalizedBase,
    rules: `${normalizedBase}#/rules`,
    about: "/about",
    contact: "/contact",
  };
}

/** The single link set used by the running application. */
export const siteLinks = resolveSiteLinks(import.meta.env.BASE_URL);

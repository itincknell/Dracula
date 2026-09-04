export interface SiteLinks {
  home: string;
  rules: string;
  about: string;
  contact: string;
}

export function resolveSiteLinks(environment: {
  BASE_URL?: string;
  VITE_RULES_URL?: string;
  VITE_ABOUT_URL?: string;
  VITE_CONTACT_URL?: string;
}): SiteLinks {
  const base = environment.BASE_URL?.trim() || "/";
  const normalizedBase = base.endsWith("/") ? base : `${base}/`;
  return {
    home: normalizedBase,
    rules: environment.VITE_RULES_URL?.trim() || `${normalizedBase}#/rules`,
    about: environment.VITE_ABOUT_URL?.trim() || "/about",
    contact: environment.VITE_CONTACT_URL?.trim() || "/contact",
  };
}

export const siteLinks = resolveSiteLinks(import.meta.env);

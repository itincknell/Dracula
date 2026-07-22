export interface SiteLinks {
  rules: string;
  about: string;
  contact: string;
}

export function resolveSiteLinks(environment: {
  VITE_RULES_URL?: string;
  VITE_ABOUT_URL?: string;
  VITE_CONTACT_URL?: string;
}): SiteLinks {
  return {
    rules: environment.VITE_RULES_URL?.trim() || "/rules",
    about: environment.VITE_ABOUT_URL?.trim() || "/about",
    contact: environment.VITE_CONTACT_URL?.trim() || "/contact",
  };
}

export const siteLinks = resolveSiteLinks(import.meta.env);

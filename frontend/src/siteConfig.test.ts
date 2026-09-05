/**
 * Verifies local defaults and build-time overrides for public navigation links.
 * Tests keep the GitHub Pages subpath and personal-site destinations explicit
 * without involving routing or network requests.
 */
import { describe, expect, it } from "vitest";

import { resolveSiteLinks } from "./siteConfig";

describe("site links", () => {
  it("uses local destinations when no deployment values are supplied", () => {
    expect(resolveSiteLinks({})).toEqual({
      home: "/",
      rules: "/#/rules",
      about: "/about",
      contact: "/contact",
    });
  });

  it("resolves all three destinations from deployment configuration", () => {
    expect(resolveSiteLinks({
      VITE_RULES_URL: "https://example.test/rules",
      VITE_ABOUT_URL: "https://example.test/about",
      VITE_CONTACT_URL: "https://example.test/contact",
    })).toEqual({
      home: "/",
      rules: "https://example.test/rules",
      about: "https://example.test/about",
      contact: "https://example.test/contact",
    });
  });

  it("keeps internal routes under the GitHub Pages project base", () => {
    expect(resolveSiteLinks({ BASE_URL: "/Dracula/" })).toMatchObject({
      home: "/Dracula/",
      rules: "/Dracula/#/rules",
    });
  });
});

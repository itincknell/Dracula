import { describe, expect, it } from "vitest";

import { resolveSiteLinks } from "./siteConfig";

describe("site links", () => {
  it("uses local destinations when no deployment values are supplied", () => {
    expect(resolveSiteLinks({})).toEqual({
      rules: "/rules",
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
      rules: "https://example.test/rules",
      about: "https://example.test/about",
      contact: "https://example.test/contact",
    });
  });
});

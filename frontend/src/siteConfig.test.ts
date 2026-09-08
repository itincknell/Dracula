/**
 * Verifies local defaults and build-time overrides for public navigation links.
 * Tests keep the GitHub Pages subpath and personal-site destinations explicit
 * without involving routing or network requests.
 */
import { describe, expect, it } from "vitest";

import { resolveSiteLinks } from "./siteConfig";

describe("site links", () => {
  it("uses local destinations when no deployment values are supplied", () => {
    expect(resolveSiteLinks()).toEqual({
      home: "/",
      rules: "/#/rules",
      about: "/about",
      contact: "/contact",
    });
  });

  it("keeps internal routes under the GitHub Pages project base", () => {
    expect(resolveSiteLinks("/Dracula/")).toMatchObject({
      home: "/Dracula/",
      rules: "/Dracula/#/rules",
    });
  });

  it("normalizes a base path without a trailing slash", () => {
    expect(resolveSiteLinks("/Dracula").home).toBe("/Dracula/");
  });
});

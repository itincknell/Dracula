/**
 * Applies the maintained JavaScript and TypeScript lint rules to the frontend.
 * Generated build output and browser-test reports are excluded, while browser
 * and Node globals are available to their respective source and tooling files.
 */
import eslint from "@eslint/js";
import globals from "globals";
import typescriptEslint from "typescript-eslint";

export default typescriptEslint.config(
  { ignores: ["dist", "node_modules", "playwright-report", "test-results"] },
  eslint.configs.recommended,
  ...typescriptEslint.configs.recommended,
  {
    files: ["**/*.{js,mjs,ts,tsx}"],
    languageOptions: {
      globals: {
        ...globals.browser,
        ...globals.node,
      },
    },
  },
);

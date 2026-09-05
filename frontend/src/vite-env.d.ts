/// <reference types="vite/client" />

/**
 * Declares build-time environment values consumed by the Vite frontend.
 * All fields are optional because local development provides stable defaults;
 * production builds supply the API and public-site destinations explicitly.
 */

interface ImportMetaEnv {
  readonly BASE_URL: string;
  readonly VITE_API_ORIGIN?: string;
  readonly VITE_RULES_URL?: string;
  readonly VITE_ABOUT_URL?: string;
  readonly VITE_CONTACT_URL?: string;
  readonly VITE_SCORING_REDUCED_STEP_MS?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}

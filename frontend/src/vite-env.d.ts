/// <reference types="vite/client" />

/**
 * Declares build-time environment values consumed by the Vite frontend.
 * All fields are optional because local development provides stable defaults;
 * production builds supply the API origin and application base explicitly.
 */

interface ImportMetaEnv {
  /** Vite-normalized application base, `/` locally and `/Dracula/` in production. */
  readonly BASE_URL: string;
  /** Absolute production API origin; omission selects the local `/api` proxy. */
  readonly VITE_API_ORIGIN?: string;
  /** Browser-test override that shortens every reduced-motion scoring frame. */
  readonly VITE_SCORING_REDUCED_STEP_MS?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}

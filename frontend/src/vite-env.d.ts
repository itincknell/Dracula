/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_API_BASE_URL?: string;
  readonly VITE_RULES_URL?: string;
  readonly VITE_ABOUT_URL?: string;
  readonly VITE_CONTACT_URL?: string;
  readonly VITE_SCORING_REDUCED_STEP_MS?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}

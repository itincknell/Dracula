/**
 * Configures local Vite development and the GitHub Pages production build.
 * It owns the public base path, compile-time API origin, local API proxy, and
 * deterministic asset handling shared by validation and release workflows.
 */
import react from "@vitejs/plugin-react";
import { defineConfig, loadEnv } from "vite";

export default defineConfig(({ mode }) => {
  const environment = { ...loadEnv(mode, process.cwd(), ""), ...process.env };
  const production = mode === "production";
  const base = production
    ? (environment.VITE_BASE_PATH?.trim() || "/Dracula/")
    : "/";
  const apiOrigin = production
    ? (environment.VITE_API_ORIGIN?.trim() || "https://api.ian-tincknell.com")
    : environment.VITE_API_ORIGIN?.trim();
  const apiTarget = environment.DRACULA_API_PROXY_TARGET ?? "http://127.0.0.1:8000";
  const proxy = {
    "/api": {
      target: apiTarget,
      changeOrigin: true,
      rewrite: (path: string) => path.replace(/^\/api/, ""),
    },
  };

  return {
    base,
    define: {
      "import.meta.env.VITE_API_ORIGIN": JSON.stringify(apiOrigin),
    },
    plugins: [react()],
    server: {
      host: "127.0.0.1",
      port: 5173,
      proxy,
    },
    preview: {
      host: "127.0.0.1",
      port: 4173,
      proxy,
    },
  };
});

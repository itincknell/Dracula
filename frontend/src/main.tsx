import { StrictMode } from "react";
import { createRoot } from "react-dom/client";

import "@fontsource/grenze-gotisch/latin-600.css";
import "@fontsource/grenze-gotisch/latin-700.css";
import "@fontsource/ibm-plex-mono/latin-400.css";
import "@fontsource/ibm-plex-mono/latin-600.css";
import "@fontsource/silkscreen/latin-400.css";
import "@fontsource/silkscreen/latin-700.css";

import { App } from "./App";
import "./styles.css";

const root = document.getElementById("root");
if (root === null) throw new Error("Application root is missing");

createRoot(root).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
